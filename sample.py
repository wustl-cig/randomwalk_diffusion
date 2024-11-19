from functools import partial
import os
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_vp_model
from data.dataloader import get_dataset, get_dataloader
from util.tweedie_utility import tween_noisy_training_sample, get_memory_free_MiB, mkdir_exp_recording_folder,clear_color, mask_generator
from util.logger import get_logger
from score_sde_inverse.score_inverse.models.utils import create_ve_model
from vp_langevin import vp_langevin
from ve_langevin import ve_langevin

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion_config', type=str)
    args = parser.parse_args()

    # ------------
    # (Prep step 1) Obtain necessary variable from config for experiment
    # ------------
    diffusion_config = load_yaml(args.diffusion_config)
    logger = get_logger()
    gpu = diffusion_config['machinesetup']['gpu_idx']
    save_dir = diffusion_config['machinesetup']['save_dir']
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    sample_conditionally = False if diffusion_config['measurement']['operator']['name'] == "uncondition" else True
    num_iters = diffusion_config["langevin_hyperparam"]["num_iters"]
    temperature = diffusion_config["langevin_hyperparam"]["temperature"]
    scaling_constant_of_step_size = diffusion_config["langevin_hyperparam"]["scaling_constant_of_step_size"]
    measurement_noise_sigma = torch.tensor([diffusion_config['measurement']['noise']['sigma']], device=device)
    schedule_name = diffusion_config['langevin_hyperparam']["schedule"] if diffusion_config['model']['noise_perturbation_type'] == "vp" else None
    
    # ------------
    # (Prep step 2) Device setting to properly assign to certain gpu
    # ------------
    device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  

    # ------------
    # Load pretrained score function
    # ------------
    if diffusion_config['model']['noise_perturbation_type'] == "vp":
        model = create_vp_model(**diffusion_config['model'])
        model = model.to(device)
        model.eval()
    elif diffusion_config['model']['noise_perturbation_type'] == "ve":
        ckpt_path = diffusion_config['model']['pretrained_check_point']
        loaded_state = torch.load(ckpt_path, map_location=device)
        model = create_ve_model(diffusion_config, map_location=device)
        model.load_state_dict(loaded_state["model"], strict=False)
        model = model.to(device)
    else:
        raise ValueError("Given noise perturbation type is not existing.")

    # ------------
    # (Prep step 3) Initialize necessary forward operator in the case of conditional sampling.
    # ------------
    if sample_conditionally == True:
        measure_config = diffusion_config['measurement']
        operator = get_operator(device=device, **measure_config['operator'])
        extra_measurement_params = {'sigma': diffusion_config['measurement']['noise']['sigma']}
        measurement_noise_config = measure_config['noise']
        combined_measurement_config = {**measurement_noise_config, **extra_measurement_params}
        noiser = get_noise(**combined_measurement_config)
        logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
        cond_method = get_conditioning_method('ps', operator, noiser)
        measurement_cond_fn = cond_method.conditioning
    else:
        operator = None
        measurement_cond_fn = None
        y_n = None
        measure_config = None
    
    # ------------
    # (Prep step 4) Make experiment saving directory
    # ------------
    save_dir, result_csv_file = mkdir_exp_recording_folder(save_dir = save_dir, measurement_operator_name = diffusion_config['measurement']['operator']['name'])
    os.makedirs(save_dir, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress']:
        os.makedirs(os.path.join(save_dir, img_dir), exist_ok=True)
        
    # ------------
    # (Prep step 5) Define dataloader
    # ------------
    data_config = diffusion_config['data']
    if diffusion_config['model']['noise_perturbation_type'] == "ve":
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor()])
    elif diffusion_config['model']['noise_perturbation_type'] == "vp":
        transform = transforms.Compose([transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        raise ValueError("Another types of noise perturbation type is given")
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    
    # ------------
    # (Test stage)
    # ------------
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        if sample_conditionally == True:
            # ------------
            # (Test stage for conditional sampling) Get the forward operator for solving inverse problem
            # ------------
            if diffusion_config['measurement']['operator'] ['name'] == 'inpainting':
                mask_gen = mask_generator(**diffusion_config['measurement']['mask_opt'])
                mask = mask_gen(ref_img)
                mask = mask[:, 0, :, :].unsqueeze(dim=0)
                
                measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)

            else: 
                if diffusion_config['model']['noise_perturbation_type'] in ["vp", "ve"]:
                    measurement_cond_fn = partial(cond_method.conditioning)
                else:
                    raise ValueError(f"Check the 'noise_perturbation type in diffusion_config")

                mask = None
                y = operator.forward(ref_img)
                y_n = noiser(y)
                 
        # ------------
        # (Test stage) Langevin sampling using VP diffusion model
        # ------------ 
        if diffusion_config['model']['noise_perturbation_type'] == "vp":
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            input_ref_images = [y_n, ref_img]
            
            vp_langevin(model = model, sample_conditionally = sample_conditionally, x_start = x_start,  
                        scaling_constant_of_step_size = scaling_constant_of_step_size, temperature = temperature, num_iters = num_iters,
                        measurement = y_n, measurement_cond_fn = measurement_cond_fn, measurement_noise_sigma = measurement_noise_sigma, 
                        diffusion_config = diffusion_config, schedule_name = schedule_name,
                        input_ref_images = input_ref_images, save_root = save_dir, img_file_index = i, gpu = gpu,
                        )
        
        # ------------
        # (Test stage) Langevin sampling using VE diffusion model
        # ------------ 
        elif diffusion_config['model']['noise_perturbation_type'] == "ve":
            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            input_ref_images = [y_n, ref_img]

            ve_langevin(model = model, sample_conditionally = sample_conditionally, x_start = x_start,  
                        scaling_constant_of_step_size = scaling_constant_of_step_size, temperature = temperature, num_iters = num_iters,
                        measurement = y_n, measurement_cond_fn = measurement_cond_fn, measurement_noise_sigma = measurement_noise_sigma, 
                        diffusion_config = diffusion_config, schedule_name = schedule_name,
                        input_ref_images = input_ref_images, save_root = save_dir, img_file_index = i, gpu = gpu,
                        )

        else:
            raise ValueError(f"Check the noise_perturbation_type in diffusion_config.yaml")
        
    return

if __name__ == '__main__':
    main()
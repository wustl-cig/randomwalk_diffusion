import abc
from score_sde_inverse.score_inverse.sde import VESDE
import functools
from typing import Callable
import torch
from torch.types import _size
from score_sde_inverse.score_inverse.sde import SDE
from score_sde_inverse.score_inverse.datasets.scalers import get_data_inverse_scaler, get_data_scaler
from score_sde_inverse.score_inverse.sde import get_sde
from score_sde_inverse.score_inverse.datasets.scalers import get_data_inverse_scaler, get_data_scaler
from score_sde_inverse.score_inverse.models import utils as mutils
from util.tweedie_utility import compute_metrics, clear_color, normalize_np
from util.img_utils import clear_color,mask_generator
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
from functools import partial
from torch.nn import functional as F

def ve_langevin(model,
                x_start,
                measurement,
                measurement_cond_fn,
                measurement_noise_sigma,
                save_root,
                scaling_constant_of_step_size,
                input_ref_images,
                img_file_index,
                temperature,
                sample_conditionally,
                gpu,
                diffusion_config,
                num_iters,
                schedule_name):
    img = x_start
    device = x_start.device
    input_img, ref_img = input_ref_images
    current_step = 0
    
    save_image = diffusion_config['langevin_hyperparam']['save_image']
    continuous=diffusion_config['training']['continuous']
    
    scaler = get_data_scaler(diffusion_config)
    inverse_scaler = get_data_inverse_scaler(diffusion_config)
    sde, sampling_eps = get_sde(diffusion_config)
    
    sampling_shape = (
        diffusion_config['eval']['batch_size'],
        diffusion_config['imagedetail']['num_channels'],
        diffusion_config['imagedetail']['image_size'],
        diffusion_config['imagedetail']['image_size'],
    )
    gif_count = int(sde.N/25)
    img_list = []
    nan_count = 0
    
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    
    # ------------
    # 
    # ------------
    timesteps = torch.linspace(sde.T, sampling_eps, sde.N)
    
    pbar = tqdm(range(sde.N))
    
    
    # ------------
    # Linearly interpolate the temperature from intial temperature to the 1 for sampling from targeted distribution.
    # ------------
    extended_temperature_list = (np.linspace(temperature, 1., len(pbar), dtype=np.float64)).tolist()
    
    
    # ------------
    # Main code part of VE-langevin
    # ------------
    for loop_index, _ in enumerate(pbar):
        t = timesteps[loop_index]
        t = torch.full((sampling_shape[0],), t.item(), device=device)
        time = (t * (sde.N - 1) / sde.T).long()
        if time.device != sde.discrete_sigmas.device:
            sde.discrete_sigmas = sde.discrete_sigmas.to(device)
        img = img.requires_grad_()
        noise_sigma = sde.discrete_sigmas[time]
        adjacent_noise_sigma = torch.where(
            time == 0,
            torch.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[time - 1],
        )
        assert noise_sigma >= adjacent_noise_sigma

        noise_sigma_square = torch.square(noise_sigma)
        adjacent_noise_sigma_square = torch.square(adjacent_noise_sigma)

        with torch.no_grad():
            score = score_fn(img, t)

        if sample_conditionally == True:
            langevin_step_size = noise_sigma_square * scaling_constant_of_step_size
            if langevin_step_size > torch.square(measurement_noise_sigma):
                langevin_step_size = torch.square(measurement_noise_sigma)
        else:
            langevin_step_size = noise_sigma_square * scaling_constant_of_step_size

        lgv_score_x_coefficient = 1
        lgv_score_x_hat_coefficient = langevin_step_size
        # lgv_score_noise_coefficient = torch.sqrt(2*langevin_step_size*(temperature))
        lgv_score_noise_coefficient = torch.sqrt(2*langevin_step_size*(extended_temperature_list[loop_index]))
        noise_N = torch.randn_like(img)
        
        # ------------
        # Doing unconditional Langevin update
        # ------------
        if (loop_index != num_iters - 1):
            img_score = lgv_score_x_coefficient * img + lgv_score_x_hat_coefficient * score + lgv_score_noise_coefficient * noise_N                        
        else:
            img_score = lgv_score_x_coefficient * img + lgv_score_x_hat_coefficient * score

        # ------------
        # Doing conditional Langevin update if solving inverse problems
        # ------------
        if sample_conditionally == True:
            norm_grad, distance, _ = measurement_cond_fn(x_t= img,
                    measurement=measurement,
                    noisy_measurement=measurement,
                    x_prev=img,
                    x_0_hat=img)
            measurement_noise_sigma_square = torch.square(measurement_noise_sigma)
            lgv_likelihood_coefficient = -1. * langevin_step_size * (1/(measurement_noise_sigma_square))
            img_cond = img_score + lgv_likelihood_coefficient * norm_grad
            img = img_cond
        else:
            img = img_score
            
        if sample_conditionally == True:
            recon_psnr_value, recon_snr_value, recon_mse_value = compute_metrics(img, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            pbar.set_postfix({'psnr': recon_psnr_value, 'step_size': langevin_step_size.item()}, refresh=False)
        else:
            pbar.set_postfix({'step_size': langevin_step_size.item()}, refresh=False)

        if save_image == True:
            if loop_index % gif_count == 0 or loop_index == num_iters - 1:
                img_list.append(img)
                
    # ------------
    # When solving inverse problem, this part measures necessary metric to evaluate the performance.
    # ------------
    if sample_conditionally == True:
        if measurement.shape[2] != ref_img.shape[2]:
                up_sample = partial(F.interpolate, scale_factor=4) # I assume that super-resolution scale factor is 4.
                measurement_for_metric = up_sample(measurement)
                input_psnr_value, input_snr_value, input_mse_value = compute_metrics(measurement_for_metric, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
        else:
            input_psnr_value, input_snr_value, input_mse_value = compute_metrics(measurement, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            
        recon_psnr_value, recon_snr_value, recon_mse_value = compute_metrics(img, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")

    # ------------
    # To visualize the output
    # ------------
    formatted_scaling_constant_of_step_size = f"{scaling_constant_of_step_size:.5f}"
    formatted_temperature = f"{temperature:.2f}"
    if save_image == True:
        images = []
        for j in range(len(img_list)):
            processed_image = clear_color(img_list[j])
            processed_image = (processed_image * 255).astype(np.uint8) if processed_image.dtype != np.uint8 else processed_image
            images.append(processed_image)
        
        if sample_conditionally == True:
            sigma_value_for_file = measurement_noise_sigma.item()
            formatted_sigma = f"{sigma_value_for_file:.3f}".zfill(4)

            formatted_input_psnr_value = f"{input_psnr_value:.3f}"
            formatted_recon_psnr_value = f"{recon_psnr_value:.3f}"
            formatted_recon_snr_value = f"{recon_snr_value:.3f}"
            formatted_recon_mse_value = f"{recon_mse_value:.5f}"

            gif_path = os.path.join(save_root, f"progress/{str(img_file_index)}_mnoise_{formatted_sigma}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{sde.N}_epsilonOftau_{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}_psnr{str(formatted_recon_psnr_value).zfill(3)}_snr{str(formatted_recon_snr_value).zfill(3)}_mse{str(formatted_recon_mse_value).zfill(3)}.gif")
            file_path = os.path.join(save_root, f"recon/{str(img_file_index)}_mnoise_{formatted_sigma}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{sde.N}_epsilonOftau_{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}_psnr{str(formatted_recon_psnr_value).zfill(3)}_snr{str(formatted_recon_snr_value).zfill(3)}_mse{str(formatted_recon_mse_value).zfill(3)}.png")
            input_file_path = os.path.join(save_root, f"input/{str(img_file_index)}_mnoise_{formatted_sigma}_psnr{str(formatted_input_psnr_value).zfill(3)}.png")
            gt_file_path = os.path.join(save_root, f"input/gt.png")
            print(f"# ------------")
            print(f"# {diffusion_config['model']['noise_perturbation_type'].upper()}-Langevin configuration: (num_iters: {num_iters} / epsilon in stepsize: {formatted_scaling_constant_of_step_size} / temperature: {formatted_temperature})")
            print(f"# Inverse problem: {diffusion_config['measurement']['operator']['name']}")
            print(f"# Input PSNR: {formatted_input_psnr_value}")
            print(f"# Recon PSNR: {formatted_recon_psnr_value}")
            print(f"# Check out experiment at {save_root}")
            print(f"# ------------")
            
        else:
            gif_path = os.path.join(save_root, f"progress/{str(img_file_index)}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{sde.N}_epsilonOftau_{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}.gif")
            file_path = os.path.join(save_root, f"recon/{str(img_file_index)}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{sde.N}__epsilonOftau__{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}.png")

            print(f"# ------------")
            print(f"# {diffusion_config['model']['noise_perturbation_type'].upper()}-Langevin configuration: (num_iters: {num_iters} / epsilon in stepsize: {formatted_scaling_constant_of_step_size} / temperature: {formatted_temperature})")
            print(f"# Check out experiment at {save_root}")
            print(f"# ------------")

        imageio.mimsave(gif_path, images, duration=0.5)
        plt.imsave(file_path, clear_color(img))
        if sample_conditionally == True and not os.path.exists(input_file_path):
            plt.imsave(input_file_path, clear_color(measurement))
            plt.imsave(gt_file_path, clear_color(ref_img))
            
    return 

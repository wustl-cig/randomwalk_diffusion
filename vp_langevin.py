import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.nn import functional as F
from datetime import datetime
from util.tweedie_utility import get_tween_sampleidx, tween_noisy_training_sample, get_memory_free_MiB, extract_and_expand_value,get_noiselevel_alphas_timestep, clear_color,mask_generator,compute_metrics
import imageio

def vp_langevin(model,
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

        # ------------
        # Obtain time input of score function corresponding to the arbitrary noise level.
        # ------------
        extended_denoiser_noise_sigma_array, extended_alphas_array, time_array, time_idx_array =  get_noiselevel_alphas_timestep(beta_at_clean = diffusion_config['vp_diffusion']['beta_at_clean'], schedule_name = schedule_name, num_diffusion_timesteps = diffusion_config['vp_diffusion']['steps'], last_time_step = 0, num_iters = num_iters, save_plot=True, save_root=save_root)
        
        reverse_time_array = time_array[::-1]
        reverse_time_idx_array = time_idx_array[::-1]
        reverse_alphas_array = extended_alphas_array[::-1]
        reverse_noise_sigma_array = extended_denoiser_noise_sigma_array[::-1]
        
        total_number_of_step = len(time_array)
        scaling_constant_of_step_size_scale_list = [scaling_constant_of_step_size for _ in time_array]
        gif_count = int(total_number_of_step/25)
        
        assert len(time_array) == len(time_idx_array) == len(extended_denoiser_noise_sigma_array) == len(extended_alphas_array)
        
        time_alpha_noisesigma_step_list = list(zip(reverse_time_idx_array, reverse_time_array, reverse_alphas_array, reverse_noise_sigma_array, scaling_constant_of_step_size_scale_list))

        save_image = diffusion_config['langevin_hyperparam']['save_image']
        initial_t = time_alpha_noisesigma_step_list[1][0]
        time = torch.tensor([initial_t] * img.shape[0], device=device)#.clone().detach() # * e.g) time: tensor([999]) at the first index


        if sample_conditionally == True:
            if diffusion_config['measurement']['operator']['name'] == 'super_resolution':
                up_sample = partial(F.interpolate, scale_factor=diffusion_config['measurement']['operator']['scale_factor'])
                measurement_for_metric = up_sample(measurement)
                input_psnr_value, input_snr_value, input_mse_value = compute_metrics(measurement_for_metric, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            else:
                input_psnr_value, input_snr_value, input_mse_value = compute_metrics(measurement, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
                
        img_list = []
            
        pbar = tqdm(time_alpha_noisesigma_step_list)
        
        # ------------
        # Linearly interpolate the temperature from intial temperature to the 1 for sampling from targeted distribution.
        # ------------
        extended_temperature_list = (np.linspace(temperature, 1., len(pbar), dtype=np.float64)).tolist()


        # ------------
        # Main code part of VE-langevin
        # ------------
        for loop_index, (index_reverse_time_str, indexed_reverse_time_str, indexed_reverse_alphas, indexed_reverse_noise_sigma, indexed_scaling_constant_of_step_size_scale) in enumerate(pbar):
            t = indexed_reverse_time_str
            time = torch.tensor([t] * img.shape[0], device=device)#.clone().detach() # * e.g) time: tensor([999]) at the first index
            alphas_coef = extract_and_expand_value(indexed_reverse_alphas, t, img).to(device)
            scaling_constant_of_step_size = torch.tensor(indexed_scaling_constant_of_step_size_scale)

            img = img.requires_grad_()
            model_output = model(torch.sqrt(alphas_coef) * img, time)
            if model_output.shape[1] == 2 * ref_img.shape[1]:
                model_output, model_var_values = torch.split(model_output, ref_img.shape[1], dim=1)

            score = - torch.sqrt((1)/(1-alphas_coef)) * model_output * torch.sqrt(alphas_coef)
            
            noise_sigma_square = torch.square(torch.tensor(indexed_reverse_noise_sigma))

            if sample_conditionally == True:
                langevin_step_size = noise_sigma_square * scaling_constant_of_step_size
                if langevin_step_size > torch.square(measurement_noise_sigma):
                    langevin_step_size = torch.square(measurement_noise_sigma)
            else:
                langevin_step_size = noise_sigma_square * scaling_constant_of_step_size
            
            lgv_score_x_coefficient = 1
            lgv_score_x_hat_coefficient = langevin_step_size
            lgv_score_noise_coefficient = torch.sqrt(2*langevin_step_size*(extended_temperature_list[loop_index]))
            # lgv_score_noise_coefficient = torch.sqrt(2*langevin_step_size*(temperature))
            
            noise_N = torch.randn_like(model_output)

            # ------------
            # Doing unconditional Langevin update
            # ------------
            if (loop_index != total_number_of_step - 1):
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
            
            # ! Compute PSNR
            img = img.detach_()
            img_score = img_score.detach_()
            
            if sample_conditionally == True:
                recon_psnr_value, recon_snr_value, recon_mse_value = compute_metrics(img, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
                pbar.set_postfix({'psnr': recon_psnr_value, 'step_size': langevin_step_size.item()}, refresh=False)
            else:
                pbar.set_postfix({'step_size': langevin_step_size.item()}, refresh=False)
            

            # * 3. Save the image at the intersect. Per step size, let save three or four images
            if save_image == True:
                if loop_index % gif_count == 0 or loop_index == total_number_of_step - 1:
                    img_list.append(img)

        # ------------
        # Below is only saving images related code. You can ignore
        # ------------
        formatted_scaling_constant_of_step_size = f"{indexed_scaling_constant_of_step_size_scale:.5f}"
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
                formatted_recon_mse_value = f"{recon_mse_value:.3f}"
                gif_path = os.path.join(save_root, f"progress/{str(img_file_index)}_mnoise_{formatted_sigma}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{len(reverse_noise_sigma_array)}_epsilonOftau_{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}_psnr{str(formatted_recon_psnr_value).zfill(3)}_snr{str(formatted_recon_snr_value).zfill(3)}_mse{str(formatted_recon_mse_value).zfill(3)}.gif")
                file_path = os.path.join(save_root, f"recon/{str(img_file_index)}_mnoise_{formatted_sigma}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{len(reverse_noise_sigma_array)}_epsilonOftau_{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}_psnr{str(formatted_recon_psnr_value).zfill(3)}_snr{str(formatted_recon_snr_value).zfill(3)}_mse{str(formatted_recon_mse_value).zfill(3)}.png")
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
                gif_path = os.path.join(save_root, f"progress/{str(img_file_index)}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{len(reverse_noise_sigma_array)}_epsilonOftau_{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}.gif")
                file_path = os.path.join(save_root, f"recon/{str(img_file_index)}_{diffusion_config['model']['noise_perturbation_type']}_{schedule_name}_iters_{len(reverse_noise_sigma_array)}__epsilonOftau__{formatted_scaling_constant_of_step_size}_Temperature_{formatted_temperature}.png")
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
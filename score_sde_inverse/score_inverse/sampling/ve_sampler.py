import abc
from score_sde_inverse.score_inverse.sde import VESDE#, VPSDE
import functools
from typing import Callable
import torch
from torch.types import _size
# from score_inverse.tasks import InverseTask
from score_sde_inverse.score_inverse.sde import SDE
from score_sde_inverse.score_inverse.datasets.scalers import get_data_inverse_scaler, get_data_scaler
from score_sde_inverse.score_inverse.sde import get_sde
from score_sde_inverse.score_inverse.datasets.scalers import get_data_inverse_scaler, get_data_scaler
# from score_sde_inverse.score_inverse.sampling import get_predictor
# from score_sde_inverse.score_inverse.sampling.ve_sampler import get_ve_sampler
# from score_sde_inverse.score_inverse.sampling import (
#     shared_predictor_update_fn,
# )
from score_sde_inverse.score_inverse.models import utils as mutils
from util.tweedie_utility import compute_metrics

from util.img_utils import clear_color,mask_generator
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio
from functools import partial
from torch.nn import functional as F

def ve_langevin_sampling(
                model,
                x_start,
                measurement,
                measurement_cond_fn,
                measurement_noise_sigma,
                save_root,
                scaling_constant_of_step_size,
                input_ref_images,
                img_file_index,
                inv_temperature,
                sample_conditionally,
                gpu,
                num_iters,
                diffusion_config):
    
    input_img, ref_img = input_ref_images
    img = x_start
    device = x_start.device

    scaler = get_data_scaler(diffusion_config)
    inverse_scaler = get_data_inverse_scaler(diffusion_config)
    sde, sampling_eps = get_sde(diffusion_config)
    
    sampling_shape = (
        diffusion_config['eval']['batch_size'],
        diffusion_config['imagedetail']['num_channels'],
        diffusion_config['imagedetail']['image_size'],
        diffusion_config['imagedetail']['image_size'],
    )
    
    predictor = get_predictor(diffusion_config['sampling']['predictor'].lower())

    sampling_fn = get_ve_sampler(
        sde=sde,
        shape=sampling_shape,
        predictor=predictor,
        inverse_scaler=inverse_scaler,
        snr=diffusion_config['sampling']['snr'],
        n_steps=diffusion_config['sampling']['n_steps_each'],
        probability_flow=diffusion_config['sampling']['probability_flow'],
        continuous=diffusion_config['training']['continuous'],
        denoise=diffusion_config['sampling']['noise_removal'],
        eps=sampling_eps,
        device=device,
        scaling_constant_of_step_size = scaling_constant_of_step_size,
        sample_conditionally = sample_conditionally,
        measurement = measurement,
        measurement_noise_sigma = measurement_noise_sigma,
        measurement_cond_fn = measurement_cond_fn,
        ref_img = ref_img,
        save_root = save_root,
        gpu = gpu,
        inv_temperature = inv_temperature,
        img_file_index = img_file_index,
    )

    model = model.to(device)
    img, n = sampling_fn(model)
    
    return

class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x

def extract_and_expand(array, target):
    # array = torch.from_numpy(array).to(target.device).float()
    array = array.float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)

class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, scaling_constant_of_step_size, inv_temperature = None, measurement = None, measurement_noise_sigma = None, measurement_cond_fn = None, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        # if not isinstance(sde, VPSDE) and not isinstance(sde, VESDE):
        #     raise NotImplementedError(
        #         f"SDE class {sde.__class__.__name__} not yet supported."
        #     )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"
        self.scaling_constant_of_step_size = scaling_constant_of_step_size
        self.measurement = measurement
        self.measurement_noise_sigma = measurement_noise_sigma
        self.measurement_cond_fn = measurement_cond_fn
        self.inv_temperature = inv_temperature
       
    def vesde_update_fn(self, x, t):
        sde = self.sde
        sample_conditionally = True if self.measurement != None else False
        timestep = (t * (sde.N - 1) / sde.T).long()
        timestep_device = timestep.device
        discrete_sigmas_device = sde.discrete_sigmas.device
        if timestep_device != discrete_sigmas_device:
            sde.discrete_sigmas = sde.discrete_sigmas.to(timestep_device)
        x = x.requires_grad_()
        sigma = sde.discrete_sigmas[timestep]

        adjacent_sigma = torch.where(
            timestep == 0,
            torch.zeros_like(t),
            sde.discrete_sigmas.to(t.device)[timestep - 1],
        )
        assert sigma >= adjacent_sigma
        test_noise_sigma = sigma
        noise_sigma_square = torch.square(test_noise_sigma)
        adjacent_noise_sigma_square = torch.square(adjacent_sigma)
        with torch.no_grad():
            score = self.score_fn(x, t)
        langevin_step_size = noise_sigma_square * self.scaling_constant_of_step_size
        if sample_conditionally == True:
            if langevin_step_size > torch.square(self.measurement_noise_sigma):
                langevin_step_size = torch.square(self.measurement_noise_sigma)
        lgv_score_x_coefficient = 1
        lgv_score_x_hat_coefficient = langevin_step_size
        lgv_score_noise_coefficient = torch.sqrt(langevin_step_size*2*(1/self.inv_temperature))
        noise_N = torch.randn_like(x)

        if sample_conditionally == True:
            # ------------
            # Compute the log gradient of likelihood
            # ------------
            norm_grad, distance, _ = self.measurement_cond_fn(x_t= x,
                    measurement=self.measurement,
                    noisy_measurement=self.measurement,
                    x_prev=x,
                    x_0_hat=x)
            measurement_noise_sigma_square = torch.square(self.measurement_noise_sigma)

            lgv_likelihood_coefficient = -1. * langevin_step_size * (1/(measurement_noise_sigma_square))

        x_mean = lgv_score_x_coefficient * x + lgv_score_x_hat_coefficient * score
        if (timestep != 0):
            x = x_mean + lgv_score_noise_coefficient * noise_N
        else:
            x = x_mean
        
        if sample_conditionally == True:
            x = x + lgv_likelihood_coefficient * norm_grad# * 0 #! HERE
            x_mean = x
        
        import os
        import matplotlib.pyplot as plt
        
        return x, x_mean, langevin_step_size

    def update_fn(self, x, t):
        if isinstance(self.sde, VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, VPSDE):
            return self.vpsde_update_fn(x, t)

def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

def shared_predictor_update_fn(
    x, t, sde, model, predictor, probability_flow, continuous, scaling_constant_of_step_size, measurement, measurement_noise_sigma, measurement_cond_fn, inv_temperature, 
):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        # print(f"[init] scaling_constant_of_step_size: {scaling_constant_of_step_size}")
        predictor_obj = predictor(sde, score_fn, measurement = measurement, measurement_noise_sigma = measurement_noise_sigma, measurement_cond_fn=measurement_cond_fn, probability_flow = probability_flow, scaling_constant_of_step_size = scaling_constant_of_step_size, inv_temperature = inv_temperature)
    return predictor_obj.update_fn(x, t)

def get_predictor(name):
    _PREDICTORS = {
        "ancestral_sampling": AncestralSamplingPredictor,
    }
    return _PREDICTORS[name]

def get_ve_sampler(
    sde: SDE,
    measurement:torch.Tensor,
    measurement_noise_sigma:torch.Tensor,
    ref_img:torch.Tensor,
    measurement_cond_fn,
    img_file_index:int,
    sample_conditionally:bool,
    save_root:str,
    device:str,
    gpu:int,
    shape: _size,
    predictor: Predictor,
    inverse_scaler: Callable,
    scaling_constant_of_step_size: float,
    inv_temperature: float,
    snr: float,
    lambda_: float = 1,
    n_steps: int = 1,
    probability_flow: bool = False,
    continuous: bool = False,
    denoise: bool = True,
    eps: float = 1e-5,
):
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )

    def pc_solver(model):
        gif_count = int(sde.N/25)
        img_list = []
        idx_list = []
        psnr_list = []
        pbar = tqdm(range(sde.N))
        nan_count = 0
        
        if sample_conditionally == False:
            # with torch.no_grad():
            # ------------
            # Initialize image with zero mean and one standard deviation.
            # ------------
            x = torch.randn(shape, device=device).requires_grad_()
            timesteps = torch.linspace(sde.T, eps, sde.N)

            for i, _ in enumerate(pbar):
                t = timesteps[i]
                vec_t = torch.full((shape[0],), t.item(), device=device)
                x, x_mean, langevin_step_size = predictor_update_fn(x, vec_t, model = model, scaling_constant_of_step_size = scaling_constant_of_step_size, measurement = measurement, measurement_noise_sigma = measurement_noise_sigma, measurement_cond_fn = measurement_cond_fn, inv_temperature = inv_temperature)
                
                pbar.set_postfix({'step_size': langevin_step_size.item()}, refresh=False)
                if i % gif_count == 0:
                    img_list.append(x)
                    idx_list.append(i)

            # ! TODO
            formatted_scaling_constant_of_step_size = f"{scaling_constant_of_step_size:.5f}"
            formatted_inv_temperature = f"{inv_temperature:.2f}"
            # recon_psnr_value, recon_snr_value, recon_mse_value = new_tween_compute_metrics(x, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            gif_path = os.path.join(save_root, f"progress/VE_iters_{sde.N}_epsilonIntau_{formatted_scaling_constant_of_step_size}_invTemperature_{formatted_inv_temperature}.gif")
            file_path = os.path.join(save_root, f"recon/VE_iters_{sde.N}_epsilonIntau_{formatted_scaling_constant_of_step_size}_invTemperature_{formatted_inv_temperature}.png")
            
            images = []
            for j in range(len(img_list)):
                processed_image = clear_color(img_list[j])
                processed_image = (processed_image * 255).astype(np.uint8) if processed_image.dtype != np.uint8 else processed_image
                images.append(processed_image)
            imageio.mimsave(gif_path, images, duration=0.5)  # Adjust the duration as needed
            plt.imsave(file_path, clear_color(x))

            print(f"#------------")
            print(f"# Image is generated. Check the saving directory: {save_root}")
            print(f"#------------")

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)  # type: ignore
            # return x
        else:
            x = torch.randn(shape, device=device).requires_grad_()

            timesteps = torch.linspace(sde.T, eps, sde.N)
            
            for i, _ in enumerate(pbar):
                t = timesteps[i]
                vec_t = torch.full((shape[0],), t.item(), device=device)
                x, x_mean, langevin_step_size = predictor_update_fn(x, vec_t, model = model, scaling_constant_of_step_size = scaling_constant_of_step_size, measurement = measurement, measurement_noise_sigma = measurement_noise_sigma, measurement_cond_fn = measurement_cond_fn, inv_temperature = inv_temperature)
                
                recon_psnr_value, recon_snr_value, recon_mse_value = compute_metrics(x, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
                pbar.set_postfix({'psnr': recon_psnr_value, 'step_size': langevin_step_size.item()}, refresh=False)

                if i % gif_count == 0:
                    img_list.append(x)
                    idx_list.append(i)
                    psnr_list.append(recon_psnr_value)

            formatted_sigma = f"{measurement_noise_sigma.item():.3f}".zfill(4)
            formatted_scaling_constant_of_step_size = f"{scaling_constant_of_step_size:.5f}"
            formatted_inv_temperature = f"{inv_temperature:.2f}"
            recon_psnr_value, recon_snr_value, recon_mse_value = compute_metrics(x, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            formatted_recon_psnr_value = f"{recon_psnr_value:.3f}"#.zfill(4)
            formatted_recon_snr_value = f"{recon_snr_value:.3f}"#.zfill(4)
            formatted_recon_mse_value = f"{recon_mse_value:.5f}"#.zfill(4)
            gif_path = os.path.join(save_root, f"progress/VE_mnoise_{formatted_sigma}_iters_{sde.N}_epsilonIntau_{formatted_scaling_constant_of_step_size}_invTemperature_{formatted_inv_temperature}_psnr{str(formatted_recon_psnr_value).zfill(3)}_snr{str(formatted_recon_snr_value).zfill(3)}_mse{str(formatted_recon_mse_value).zfill(3)}.gif")
            file_path = os.path.join(save_root, f"recon/VE_mnoise_{formatted_sigma}_iters_{sde.N}_epsilonIntau_{formatted_scaling_constant_of_step_size}_invTemperature_{formatted_inv_temperature}_psnr{str(formatted_recon_psnr_value).zfill(3)}_snr{str(formatted_recon_snr_value).zfill(3)}_mse{str(formatted_recon_mse_value).zfill(3)}.png")
            gt_file_path = os.path.join(save_root, f"input/gt.png")
            
            if measurement.shape[2] != ref_img.shape[2]:
                up_sample = partial(F.interpolate, scale_factor=4) # I assume that super-resolution scale factor is 4.
                measurement_for_metric = up_sample(measurement)
                input_psnr_value, input_snr_value, input_mse_value = compute_metrics(measurement_for_metric, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            else:
                input_psnr_value, input_snr_value, input_mse_value = compute_metrics(measurement, ref_img, loss_fn=None, gpu=gpu, mode = "tau_tuning")
            formatted_input_psnr_value = f"{input_psnr_value:.3f}"#.zfill(4)
            formatted_input_snr_value = f"{input_snr_value:.3f}"#.zfill(4)
            formatted_input_mse_value = f"{input_mse_value:.5f}"#.zfill(4)
            input_file_path = os.path.join(save_root, f"input/mnoise_{formatted_sigma}_psnr{str(formatted_input_psnr_value).zfill(3)}_snr{str(formatted_input_snr_value).zfill(3)}_mse{str(formatted_input_mse_value).zfill(3)}.png")
            plt.imsave(input_file_path, clear_color(measurement))
            plt.imsave(gt_file_path, clear_color(ref_img))
            
            images = []
            for j in range(len(img_list)):
                processed_image = clear_color(img_list[j])
                processed_image = (processed_image * 255).astype(np.uint8) if processed_image.dtype != np.uint8 else processed_image
                images.append(processed_image)
            imageio.mimsave(gif_path, images, duration=0.5)  # Adjust the duration as needed
            plt.imsave(file_path, clear_color(x))
            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)  # type: ignore

    return pc_solver
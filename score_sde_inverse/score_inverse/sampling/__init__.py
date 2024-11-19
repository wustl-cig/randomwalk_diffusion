import functools

import torch
from scipy import integrate

from score_sde_inverse.score_inverse.models import utils as mutils
from score_sde_inverse.score_inverse.models.utils import (
    from_flattened_numpy,
    get_score_fn,
    to_flattened_numpy,
)
from .ve_sampler import (
    Predictor,
    AncestralSamplingPredictor,
)

def get_predictor(name):
    _PREDICTORS = {
        "ancestral_sampling": AncestralSamplingPredictor,
    }
    return _PREDICTORS[name]

# def get_sampling_fn1(config, sde, shape, inverse_scaler, eps):
#     """Create a sampling function.

#     Args:
#       config: A `ml_collections.ConfigDict` object that contains all configuration information.
#       sde: A `sde_lib.SDE` object that represents the forward SDE.
#       shape: A sequence of integers representing the expected shape of a single sample.
#       inverse_scaler: The inverse data normalizer function.
#       eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

#     Returns:
#       A function that takes random states and a replicated training state and outputs samples with the
#         trailing dimensions matching `shape`.
#     """

#     sampler_name = config.sampling.method
#     # Probability flow ODE sampling with black-box ODE solvers
#     if sampler_name.lower() == "ode":
#         sampling_fn = get_ode_sampler(
#             sde=sde,
#             shape=shape,
#             inverse_scaler=inverse_scaler,
#             denoise=config.sampling.noise_removal,
#             eps=eps,
#             device=config.device,
#         )
#     # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
#     elif sampler_name.lower() == "pc":
#         predictor = get_predictor(config.sampling.predictor.lower())
#         corrector = get_corrector(config.sampling.corrector.lower())
#         sampling_fn = get_pc_sampler(
#             sde=sde,
#             shape=shape,
#             predictor=predictor,
#             corrector=corrector,
#             inverse_scaler=inverse_scaler,
#             snr=config.sampling.snr,
#             n_steps=config.sampling.n_steps_each,
#             probability_flow=config.sampling.probability_flow,
#             continuous=config.training.continuous,
#             denoise=config.sampling.noise_removal,
#             eps=eps,
#             device=config.device,
#         )
#     else:
#         raise ValueError(f"Sampler name {sampler_name} unknown.")

#     return sampling_fn





# def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
#     """A wrapper tha configures and returns the update function of correctors."""
#     score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
#     if corrector is None:
#         # Predictor-only sampler
#         corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
#     else:
#         corrector_obj = corrector(sde, score_fn, snr, n_steps)
#     return corrector_obj.update_fn(x, t)


# def get_pc_sampler(
#     sde,
#     shape,
#     predictor,
#     corrector,
#     inverse_scaler,
#     snr,
#     n_steps=1,
#     probability_flow=False,
#     continuous=False,
#     denoise=True,
#     eps=1e-3,
#     device="cuda",
# ):
#     """Create a Predictor-Corrector (PC) sampler.

#     Args:
#       sde: An `sde_lib.SDE` object representing the forward SDE.
#       shape: A sequence of integers. The expected shape of a single sample.
#       predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
#       corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
#       inverse_scaler: The inverse data normalizer.
#       snr: A `float` number. The signal-to-noise ratio for configuring correctors.
#       n_steps: An integer. The number of corrector steps per predictor update.
#       probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
#       continuous: `True` indicates that the score model was continuously trained.
#       denoise: If `True`, add one-step denoising to the final samples.
#       eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
#       device: PyTorch device.

#     Returns:
#       A sampling function that returns samples and the number of function evaluations during sampling.
#     """
#     # Create predictor & corrector update functions
#     predictor_update_fn = functools.partial(
#         shared_predictor_update_fn,
#         sde=sde,
#         predictor=predictor,
#         probability_flow=probability_flow,
#         continuous=continuous,
#     )
#     corrector_update_fn = functools.partial(
#         shared_corrector_update_fn,
#         sde=sde,
#         corrector=corrector,
#         continuous=continuous,
#         snr=snr,
#         n_steps=n_steps,
#     )

#     def pc_sampler(model):
#         """The PC sampler funciton.

#         Args:
#           model: A score model.
#         Returns:
#           Samples, number of function evaluations.
#         """
#         with torch.no_grad():
#             # Initial sample
#             x = sde.prior_sampling(shape).to(device)
#             timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

#             for i in range(sde.N):
#                 t = timesteps[i]
#                 vec_t = torch.ones(shape[0], device=t.device) * t
#                 x, x_mean = corrector_update_fn(x, vec_t, model=model)
#                 x, x_mean = predictor_update_fn(x, vec_t, model=model)

#             return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)  # type: ignore

#     return pc_sampler


# def get_ode_sampler(
#     sde,
#     shape,
#     inverse_scaler,
#     denoise=False,
#     rtol=1e-5,
#     atol=1e-5,
#     method="RK45",
#     eps=1e-3,
#     device="cuda",
# ):
#     """Probability flow ODE sampler with the black-box ODE solver.

#     Args:
#       sde: An `sde_lib.SDE` object that represents the forward SDE.
#       shape: A sequence of integers. The expected shape of a single sample.
#       inverse_scaler: The inverse data normalizer.
#       denoise: If `True`, add one-step denoising to final samples.
#       rtol: A `float` number. The relative tolerance level of the ODE solver.
#       atol: A `float` number. The absolute tolerance level of the ODE solver.
#       method: A `str`. The algorithm used for the black-box ODE solver.
#         See the documentation of `scipy.integrate.solve_ivp`.
#       eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
#       device: PyTorch device.

#     Returns:
#       A sampling function that returns samples and the number of function evaluations during sampling.
#     """

#     def denoise_update_fn(model, x):
#         score_fn = get_score_fn(sde, model, train=False, continuous=True)
#         # Reverse diffusion predictor for denoising
#         predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
#         vec_eps = torch.ones(x.shape[0], device=x.device) * eps
#         _, x = predictor_obj.update_fn(x, vec_eps)
#         return x

#     def drift_fn(model, x, t):
#         """Get the drift function of the reverse-time SDE."""
#         score_fn = get_score_fn(sde, model, train=False, continuous=True)
#         rsde = sde.reverse(score_fn, probability_flow=True)
#         return rsde.sde(x, t)[0]

#     def ode_sampler(model, z=None):
#         """The probability flow ODE sampler with black-box ODE solver.

#         Args:
#           model: A score model.
#           z: If present, generate samples from latent code `z`.
#         Returns:
#           samples, number of function evaluations.
#         """
#         with torch.no_grad():
#             # Initial sample
#             if z is None:
#                 # If not represent, sample the latent code from the prior distibution of the SDE.
#                 x = sde.prior_sampling(shape).to(device)
#             else:
#                 x = z

#             def ode_func(t, x):
#                 x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
#                 vec_t = torch.ones(shape[0], device=x.device) * t
#                 drift = drift_fn(model, x, vec_t)
#                 return to_flattened_numpy(drift)

#             # Black-box ODE solver for the probability flow ODE
#             solution = integrate.solve_ivp(
#                 ode_func,
#                 (sde.T, eps),
#                 to_flattened_numpy(x),
#                 rtol=rtol,
#                 atol=atol,
#                 method=method,
#             )
#             nfe = solution.nfev
#             x = (
#                 torch.tensor(solution.y[:, -1])
#                 .reshape(shape)
#                 .to(device)
#                 .type(torch.float32)
#             )

#             # Denoising is equivalent to running one predictor step without adding noise
#             if denoise:
#                 x = denoise_update_fn(model, x)

#             x = inverse_scaler(x)
#             return x, nfe

#     return ode_sampler

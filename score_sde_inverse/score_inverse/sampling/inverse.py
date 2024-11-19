import functools
from typing import Callable

import torch
from torch.types import _size

from score_sde_inverse.score_inverse.tasks import InverseTask
from score_sde_inverse.score_inverse.sde import SDE
from score_sde_inverse.score_inverse.sampling import (
    Predictor,
    Corrector,
    shared_corrector_update_fn,
    shared_predictor_update_fn,
)


def get_pc_inverse_solver(
    sde: SDE,
    shape: _size,
    predictor: Predictor,
    corrector: Corrector,
    inverse_scaler: Callable,
    snr: float,
    inverse_task: InverseTask,
    lambda_: float = 1,
    n_steps: int = 1,
    probability_flow: bool = False,
    continuous: bool = False,
    denoise: bool = True,
    eps: float = 1e-5,
    device="cuda",
):
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn,
        sde=sde,
        corrector=corrector,
        continuous=continuous,
        snr=snr,
        n_steps=n_steps,
    )

    def get_inverse_update_fn(update_fn):
        """Core logic of https://arxiv.org/pdf/2111.08005.pdf

        Modify update functions of predictor & corrector to incorporate information of y.
        """

        def inverse_update_fn(model, y, x, vec_t):
            # Section 3.2
            y_mean, std = sde.marginal_prob(y, vec_t)
            z = torch.randn_like(x)
            y_hat_t = y_mean + std[:, None, None, None] * inverse_task.A(z)

            # Eq. 9
            x = inverse_task.transform_inv(
                lambda_ * inverse_task.mask(inverse_task.drop_inv(y_hat_t))
                + (1 - lambda_) * inverse_task.mask(inverse_task.transform(x))
                + inverse_task.mask_inv(inverse_task.transform(x))
            )

            # Eq. 7
            return update_fn(x, vec_t, model=model)

        return inverse_update_fn

    predictor_colorize_update_fn = get_inverse_update_fn(predictor_update_fn)
    corrector_colorize_update_fn = get_inverse_update_fn(corrector_update_fn)

    def pc_inverse_solver(model, y):
        with torch.no_grad():
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.full((shape[0],), t.item(), device=device)
                x, x_mean = corrector_colorize_update_fn(model, y, x, vec_t)
                x, x_mean = predictor_colorize_update_fn(model, y, x, vec_t)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)  # type: ignore

    return pc_inverse_solver

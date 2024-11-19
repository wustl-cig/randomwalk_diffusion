from .base import SDE
from .vesde import VESDE

def get_sde(config) -> tuple[SDE, float]:
    if config['training']['sde'].lower() == "vesde":
        sde = VESDE(
            sigma_min=config['model']['sigma_min'],
            sigma_max=config['model']['sigma_max'],
            N=config['langevin_hyperparam']['num_iters'],
        )
        sampling_eps = 1e-5
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    return sde, sampling_eps

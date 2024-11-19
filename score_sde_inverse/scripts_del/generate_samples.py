import os
import sys
from pathlib import Path

import numpy as np
import torch
from absl import app, flags, logging
from PIL import Image

sys.path.append(os.getcwd())  # add the current directory to the path to import modules

from configs.ve.celebahq_256_ncsnpp_continuous import get_config as get_celeba_config
from configs.ve.cifar10_ncsnpp_deep_continuous import get_config as get_cifar10_config
from score_inverse.datasets import CIFAR10, CelebA
from score_inverse.datasets.scalers import get_data_inverse_scaler
from score_inverse.models.ema import ExponentialMovingAverage
from score_inverse.models.utils import create_model
from score_inverse.sampling import get_corrector, get_predictor
from score_inverse.sampling.inverse import get_pc_inverse_solver
from score_inverse.sde import get_sde
from score_inverse.tasks import (
    ColorizationTask,
    CombinedTask,
    DeblurTask,
    DenoiseTask,
    SuperResolutionTask,
)

FLAGS = flags.FLAGS
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "celeba"], "Dataset to use.")
flags.DEFINE_bool(
    "train", False, "Generate from the train set instead of the train set."
)
flags.DEFINE_integer("num_scales", 50, "Number of discretisation steps")
flags.DEFINE_integer("batch_size", 10, "Batch size")
flags.DEFINE_integer("num_batches", 1, "Number of samples to generate")
flags.DEFINE_integer("samples_per_image", 1, "No. of reconstructed samples per image")
flags.DEFINE_string("save_dir", "./logs/samples", "Directory to save samples")
flags.DEFINE_float("lambda_", 0.1, "Lambda parameter for inverse task")
flags.DEFINE_enum(
    "task",
    "deblur_gaussian",
    [
        "deblur_gaussian",
        "sr_4x",
        "sr_16x",
        "sr_4x_noisy",
        "sr_16x_noisy",
        "denoise",
        "deblur_colorise",
        "denoise_colorise",
        "sr_4x_colorise",
        "sr_4x_deblur",
        "denoise_deblur",
    ],
    "Inverse task to use",
)
flags.DEFINE_enum("sampling_method", "pc", ["pc", "ode"], "Sampling method to use")
flags.DEFINE_enum(
    "noise_type",
    "normal",
    ["normal", "gaussian", "poisson", "shot", "salt_and_pepper", "impulse"],
    "Type of noise to apply to denoising tasks",
)
flags.DEFINE_integer(
    "noise_severity",
    1,
    "Noise severity from 1-5 based on https://arxiv.org/abs/1903.12261",
)
flags.DEFINE_integer(
    "data_index", 0, "Starting index of the dataset to generate samples from"
)


def main(_):
    if FLAGS.dataset == "cifar10":
        config = get_cifar10_config()
        ckpt_path = "checkpoints/ve/cifar10_ncsnpp_deep_continuous/checkpoint_12.pth"
        dataset = CIFAR10(train=FLAGS.train)
    elif FLAGS.dataset == "celeba":
        config = get_celeba_config()
        ckpt_path = "checkpoints/ve/celebahq_256_ncsnpp_continuous/checkpoint_48.pth"
        dataset = CelebA(train=FLAGS.train)
    else:
        raise ValueError(f"Unknown dataset {FLAGS.dataset}")

    config.sampling.method = FLAGS.sampling_method
    config.model.num_scales = FLAGS.num_scales
    config.eval.batch_size = FLAGS.batch_size

    score_model = load_checkpoint(config, ckpt_path)
    inverse_task = get_inverse_task(
        config, dataset, FLAGS.task, FLAGS.noise_type, FLAGS.noise_severity
    )
    data_loader = get_dataloader(
        dataset, FLAGS.samples_per_image, FLAGS.batch_size, FLAGS.data_index
    )
    sampling_fn = get_sampling_fn(config, dataset, inverse_task, lambda_=FLAGS.lambda_)

    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (ds_indices, s_indices, x) in enumerate(data_loader):
        if i == FLAGS.num_batches:
            break

        logging.info("Sampling batch %d...", i)

        x = x.to(device=config.device)
        y = inverse_task.forward(x)
        x_hat, _ = sampling_fn(score_model, y)

        for j in range(FLAGS.batch_size):
            source = tensor_to_image(y[j])
            target = tensor_to_image(x[j])
            reconstructed = tensor_to_image(x_hat[j])

            save_dir_sample = save_dir / str(ds_indices[j]) / str(s_indices[j])
            save_dir_sample.mkdir(parents=True, exist_ok=True)
            source.save(save_dir_sample / f"source.png", "PNG")
            target.save(save_dir_sample / f"target.png", "PNG")
            reconstructed.save(save_dir_sample / f"reconstructed.png", "PNG")


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = np.clip(x.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = x[:, :, 0]
    return Image.fromarray(x)


def get_dataloader(dataset, samples_per_image: int, batch_size: int, start_idx=0):
    dataset_idx, sample_idx, batch = [], [], []
    for i in range(len(dataset)):
        if i < start_idx:
            continue
        for j in range(samples_per_image):
            dataset_idx.append(i)
            sample_idx.append(j)
            batch.append(dataset[i])

            if len(batch) == batch_size:
                yield dataset_idx, sample_idx, torch.stack(batch)
                dataset_idx, sample_idx, batch = [], [], []


def get_inverse_task(
    config, dataset, task_name: str, noise_type: str, noise_severity: int
):
    if task_name == "deblur_gaussian":
        return DeblurTask(dataset.img_size, kernel_type="gaussian", kernel_size=5).to(
            device=config.device
        )
    elif task_name == "sr_4x":
        return SuperResolutionTask(dataset.img_size, scale_factor=4).to(
            device=config.device
        )
    elif task_name == "sr_4x_noisy":
        sr = SuperResolutionTask(dataset.img_size, scale_factor=4)
        denoise = DenoiseTask(
            sr.output_shape, noise_type=noise_type, severity=noise_severity
        )
        return CombinedTask([sr, denoise]).to(device=config.device)
    elif task_name == "sr_16x":
        return SuperResolutionTask(dataset.img_size, scale_factor=16).to(
            device=config.device
        )
    elif task_name == "sr_16x_noisy":
        sr = SuperResolutionTask(dataset.img_size, scale_factor=16)
        denoise = DenoiseTask(
            dataset.img_size, noise_type=noise_type, severity=noise_severity
        )
        return CombinedTask([sr, denoise]).to(device=config.device)
    elif task_name == "deblur_colorise":
        colorise = ColorizationTask(dataset.img_size)
        deblur = DeblurTask(
            colorise.output_shape, kernel_type="gaussian", kernel_size=5
        ).to(device=config.device)
        return CombinedTask([colorise, deblur]).to(device=config.device)
    elif task_name == "denoise_colorise":
        colorise = ColorizationTask(dataset.img_size)
        denoise = DenoiseTask(
            colorise.output_shape, noise_type=noise_type, severity=noise_severity
        )
        return CombinedTask([colorise, denoise]).to(device=config.device)
    elif task_name == "sr_4x_colorise":
        colorise = ColorizationTask(dataset.img_size)
        sr = SuperResolutionTask(colorise.output_shape, scale_factor=4)
        return CombinedTask([colorise, sr]).to(device=config.device)
    elif task_name == "sr_4x_deblur":
        deblur = DeblurTask(dataset.img_size, kernel_type="gaussian", kernel_size=5).to(
            device=config.device
        )
        sr = SuperResolutionTask(deblur.output_shape, scale_factor=4)
        return CombinedTask([deblur, sr]).to(device=config.device)
    elif task_name == "denoise":
        return DenoiseTask(
            dataset.img_size, noise_type=noise_type, severity=noise_severity
        ).to(device=config.device)
    elif task_name == "denoise_deblur":
        deblur = DeblurTask(dataset.img_size, kernel_type="gaussian", kernel_size=5).to(
            device=config.device
        )
        denoise = DenoiseTask(
            deblur.output_shape, noise_type=noise_type, severity=noise_severity
        ).to(device=config.device)
        return CombinedTask([deblur, denoise]).to(device=config.device)

    else:
        raise ValueError(f"Unknown inverse task {task_name}")


def load_checkpoint(config, ckpt_path):
    logging.info("Loading checkpoint from %s", ckpt_path)
    loaded_state = torch.load(ckpt_path, map_location=config.device)
    score_model = create_model(config)
    # Still need to load the base model state since non-trainable params aren't covered by EMA
    score_model.load_state_dict(loaded_state["model"], strict=False)

    # Replace trainable model params with EMA params
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    ema.load_state_dict(loaded_state["ema"])
    ema.copy_to(score_model.parameters())

    return score_model


def get_sampling_fn(config, dataset, inverse_task, lambda_: float = 1):
    inverse_scaler = get_data_inverse_scaler(config)
    sde, sampling_eps = get_sde(config)

    sampling_shape = (config.eval.batch_size, *dataset.img_size)
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_inverse_solver(
        sde=sde,
        shape=sampling_shape,
        predictor=predictor,
        corrector=corrector,
        inverse_scaler=inverse_scaler,
        snr=config.sampling.snr,
        n_steps=config.sampling.n_steps_each,
        probability_flow=config.sampling.probability_flow,
        continuous=config.training.continuous,
        denoise=config.sampling.noise_removal,
        eps=sampling_eps,
        device=config.device,
        inverse_task=inverse_task,
        lambda_=lambda_,
    )
    return sampling_fn


if __name__ == "__main__":
    app.run(main)

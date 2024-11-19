import numpy as np
import torch
from absl import logging
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio

from configs.ve.celebahq_256_ncsnpp_continuous import get_config as get_celeba_config
from configs.ve.cifar10_ncsnpp_deep_continuous import get_config as get_cifar10_config
from score_inverse.datasets import CIFAR10, CelebA
from score_inverse.datasets.scalers import get_data_inverse_scaler
from score_inverse.models.ema import ExponentialMovingAverage
from score_inverse.models.utils import create_model
from score_inverse.sampling import get_corrector, get_predictor
from score_inverse.sampling.inverse import get_pc_inverse_solver
from score_inverse.sde import get_sde
from score_inverse.tasks.deblur import DeblurTask
from score_inverse.tasks.denoise import DenoiseTask


def tensor_to_image(x: torch.Tensor) -> Image.Image:
    x = np.clip(x.permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    if x.shape[-1] == 1:
        x = x[:, :, 0]
    return Image.fromarray(x)


def compute_psnr(source, reconstructed):
    psnr = PeakSignalNoiseRatio(data_range=(0, 1)).to(source.device)
    return psnr(reconstructed, source).cpu().numpy()


class SharedUtils:
    def __init__(self, FLAGS, train=True):
        if FLAGS.dataset == "cifar10":
            self.config = get_cifar10_config()
            self.ckpt_path = (
                "checkpoints/ve/cifar10_ncsnpp_deep_continuous/checkpoint_12.pth"
            )
            self.dataset = CIFAR10(train=train)
        elif FLAGS.dataset == "celeba":
            self.config = get_celeba_config()
            self.ckpt_path = (
                "checkpoints/ve/celebahq_256_ncsnpp_continuous/checkpoint_48.pth"
            )
            self.dataset = CelebA(train=train)
        else:
            raise ValueError(f"Unknown dataset {FLAGS.dataset}")

        self.config.model.num_scales = FLAGS.num_scales
        self.config.eval.batch_size = FLAGS.batch_size
        self.data_loader = self.get_dataloader(
            FLAGS.samples_per_image, FLAGS.batch_size
        )
        self.score_model = self.load_checkpoint()
        self.inverse_task = self.get_inverse_task(FLAGS.task)

    def get_dataloader(self, samples_per_image: int, batch_size: int):
        dataset_idx, sample_idx, batch = [], [], []
        for i in range(len(self.dataset)):
            for j in range(samples_per_image):
                dataset_idx.append(i)
                sample_idx.append(j)
                batch.append(self.dataset[i])

                if len(batch) == batch_size:
                    yield dataset_idx, sample_idx, torch.stack(batch)
                    dataset_idx, sample_idx, batch = [], [], []

    def get_inverse_task(self, task_name: str):
        if task_name == "deblur_gaussian":
            return DeblurTask(
                self.dataset.img_size, kernel_type="gaussian", kernel_size=5
            ).to(device=self.config.device)
        elif task_name == "denoise":
            return DenoiseTask(self.dataset.img_size).to(device=self.config.device)
        else:
            raise ValueError(f"Unknown inverse task {task_name}")

    def load_checkpoint(self):
        logging.info("Loading checkpoint from %s", self.ckpt_path)
        loaded_state = torch.load(self.ckpt_path, map_location=self.config.device)
        score_model = create_model(self.config)
        # Still need to load the base model state since non-trainable params aren't covered by EMA
        score_model.load_state_dict(loaded_state["model"], strict=False)

        # Replace trainable model params with EMA params
        ema = ExponentialMovingAverage(
            score_model.parameters(), decay=self.config.model.ema_rate
        )
        ema.load_state_dict(loaded_state["ema"])
        ema.copy_to(score_model.parameters())

        return score_model

    def get_sampling_fn(self, lambda_):
        inverse_scaler = get_data_inverse_scaler(self.config)
        sde, sampling_eps = get_sde(self.config)

        sampling_shape = (self.config.eval.batch_size, *self.dataset.img_size)
        predictor = get_predictor(self.config.sampling.predictor.lower())
        corrector = get_corrector(self.config.sampling.corrector.lower())
        sampling_fn = get_pc_inverse_solver(
            sde=sde,
            shape=sampling_shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=self.config.sampling.snr,
            n_steps=self.config.sampling.n_steps_each,
            probability_flow=self.config.sampling.probability_flow,
            continuous=self.config.training.continuous,
            denoise=self.config.sampling.noise_removal,
            eps=sampling_eps,
            device=self.config.device,
            inverse_task=self.inverse_task,
            lambda_=lambda_,
        )
        return sampling_fn

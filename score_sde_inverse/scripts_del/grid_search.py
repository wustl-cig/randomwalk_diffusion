import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from absl import app, flags
from sklearn.model_selection import GridSearchCV
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from sklearn.metrics import make_scorer
from pathlib import Path
import dill

from score_inverse.sde import get_sde
from score_inverse.datasets.scalers import get_data_inverse_scaler
from score_inverse.sampling import get_corrector, get_predictor
from score_inverse.sampling.inverse import get_pc_inverse_solver
from score_inverse.tasks import DenoiseTask

from scripts.shared_utils import SharedUtils

FLAGS = flags.FLAGS
flags.DEFINE_enum("dataset", "cifar10", ["cifar10", "celeba"], "Dataset to use.")
flags.DEFINE_integer("num_scales", 100, "Number of discretisation steps")
flags.DEFINE_integer("batch_size", 10, "Batch size")
flags.DEFINE_integer("num_batches", 5, "Number of samples to generate")
flags.DEFINE_integer(
    "samples_per_image", 1, "Number of reconstructed samples per image"
)
flags.DEFINE_integer(
    "iterations", 10, "Number of Bayesian optimization iterations to perform"
)
flags.DEFINE_enum(
    "task", "denoise", ["deblur_gaussian", "denoise"], "Inverse task to use"
)
flags.DEFINE_string("save_dir", './logs/gridsearch', "Directory to save samples")
flags.DEFINE_integer("cv", 5, "Number of cross-validation folds")
flags.DEFINE_integer("n_jobs", 1, "Number of parallel gridsearch jobs to run simultaneously.")


class InverseSolverSampler:
    def __init__(self, score_model, inverse_task, config, lambda_, num_batches, batch_size):
        self.score_model = score_model
        self.inverse_task = inverse_task
        self.config = config
        self.lambda_ = lambda_
        self.num_batches = num_batches
        self.batch_size = batch_size

        inverse_scaler = get_data_inverse_scaler(self.config)
        sde, sampling_eps = get_sde(self.config)

        sampling_shape = (
            self.batch_size,
            self.config.data.num_channels,
            self.config.data.image_size,
            self.config.data.image_size,
        )

        predictor = get_predictor(self.config.sampling.predictor.lower())
        corrector = get_corrector(self.config.sampling.corrector.lower())

        self.sampling_fn = get_pc_inverse_solver(
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
            lambda_=self.lambda_,
        )


    def fit(self, *_):
        return self

    def predict(self, X):
        samples = []

        for i in range(self.num_batches):
            x = X[i*self.batch_size : (i+1)*self.batch_size].to(self.config.device)
            y = self.inverse_task.forward(x)
            x_hat, _ = self.sampling_fn(self.score_model, y)
            samples.append(x_hat.detach().cpu())

        samples = torch.cat(samples)
        torch.cuda.empty_cache()

        return samples

    def get_params(self, deep=False):
        return dict(score_model = self.score_model,
                    inverse_task = self.inverse_task,
                    config = self.config,
                    lambda_ = self.lambda_,
                    num_batches = self.num_batches,
                    batch_size = self.batch_size)

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self



def main(_):
    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    utils = SharedUtils(FLAGS, train=True)

    ssim = StructuralSimilarityIndexMeasure(data_range=(0,1))
    ssim.__name__ = 'StructuralSimilarityIndexMeasure'
    psnr = PeakSignalNoiseRatio(data_range=(0,1))
    psnr.__name__ = 'PeakSignalNoiseRatio'

    lambda_range = np.arange(0.01, 0.11, 0.01)

    dataset = torch.stack([utils.dataset[ind] for ind in range(FLAGS.batch_size*FLAGS.num_batches*FLAGS.cv)])

    results = {}

    for noise_type in ['gaussian', 'shot']:
        results[noise_type] = {}
        for severity in range(1,6):
            # ! Only applied to denoise task
            inverse_task = DenoiseTask(utils.dataset.img_size, noise_type=noise_type, severity=severity).to(device=utils.config.device)
            sampler = InverseSolverSampler(utils.score_model, inverse_task, utils.config, lambda_=0.05, num_batches=FLAGS.num_batches, batch_size=FLAGS.batch_size)

            gscv = GridSearchCV(sampler, dict(lambda_=lambda_range), scoring={'ssim': make_scorer(ssim), 'psnr': make_scorer(psnr)}, error_score='raise', verbose=4, cv=FLAGS.cv, refit=False, n_jobs=FLAGS.n_jobs)
            results[noise_type][severity] = gscv.fit(X=dataset, y=dataset).cv_results_

            torch.cuda.empty_cache()


    dill.dump(results, open(f'{FLAGS.save_dir}/gridsearch_results.pkl', 'wb'))

    for noise_type, severity_results in results.items():
        print(noise_type, 'noise')
        for severity, gscv_results in severity_results.items():
            print(f'ssim: value = {gscv_results["mean_test_ssim"][0]:.3f}, severity = {severity}, best lambda = {gscv_results["param_lambda_"][gscv_results["mean_test_ssim"].argmax()]}')
            print(f'psnr: value = {gscv_results["mean_test_psnr"][0]:.3f}, severity = {severity}, best lambda = {gscv_results["param_lambda_"][gscv_results["mean_test_psnr"].argmax()]}\n')
        print()


if __name__ == "__main__":
    app.run(main)

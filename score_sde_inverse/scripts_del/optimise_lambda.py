import math
import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())

import pandas as pd
import torch
from absl import app, flags
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import RBF

from scripts.shared_utils import SharedUtils, compute_psnr

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
flags.DEFINE_integer(
    "init_points", 2, "Number of random Bayesian optimization initialization points"
)
flags.DEFINE_enum(
    "task", "denoise", ["deblur_gaussian", "denoise"], "Inverse task to use"
)
flags.DEFINE_string("save_dir", './logs/bayes_opt', "Directory to save samples")

iteration_counter = 0
results = []


def main(_):
    global iteration_counter
    save_dir = Path(FLAGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    utils = SharedUtils(FLAGS, train=True)

    optimizer = BayesianOptimization(
        f=lambda log_lambda: objective_function(utils, log_lambda),
        pbounds={"log_lambda": (-2, 0)},
        random_state=42,
        # allow_duplicate_points=True,
    )
    optimizer.set_gp_params(kernel=RBF(length_scale=2), normalize_y=True)
    optimizer.maximize(
        init_points=FLAGS.init_points,
        n_iter=FLAGS.iterations,
        acquisition_function=UtilityFunction("ei"),
    )

    best_log_lambda = optimizer.max["params"]["log_lambda"]
    best_lambda = math.pow(10, best_log_lambda)
    print(f"Best lambda value: {best_lambda}")
    print(f"Total iterations: {iteration_counter}")

    # Convert results to a DataFrame
    df = pd.DataFrame(results)
    df.to_csv(save_dir / "optimization_results.csv", index=False)


def objective_function(utils, log_lambda):
    global iteration_counter
    lambda_ = math.pow(10, log_lambda)
    iteration_counter += 1
    sampling_fn = utils.get_sampling_fn(lambda_)
    dataset = torch.stack([utils.dataset[i] for i in range(FLAGS.batch_size*FLAGS.num_batches)])
    samples = []

    for i in range(FLAGS.num_batches):
        x = dataset[i*FLAGS.batch_size : (i+1)*FLAGS.batch_size].to(utils.config.device)
        y = utils.inverse_task.forward(x)
        x_hat, _ = sampling_fn(utils.score_model, y)
        samples.append(x_hat.detach().cpu())

    samples = torch.cat(samples)
    psnr = compute_psnr(dataset, samples)
    torch.cuda.empty_cache()

    results.append({"iteration": iteration_counter, "lambda": lambda_, "psnr": psnr})
    return psnr


if __name__ == "__main__":
    app.run(main)

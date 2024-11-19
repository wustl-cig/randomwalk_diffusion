from pathlib import Path
import numpy as np
import pandas as pd
from absl import app, flags
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_tensor
import lpips
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("samples_dir", "./logs/cifar", "Directory to save samples")
flags.DEFINE_string("output_file", "metrics_output.csv", "File to save results to")


def main(_):
    samples_dir = Path(FLAGS.samples_dir)
    output_file = FLAGS.output_file
    metrics = {
        "ssim": StructuralSimilarityIndexMeasure(data_range=(0, 1)),
        "psnr": PeakSignalNoiseRatio(data_range=(0, 1)),
        "lpips": lpips.LPIPS(net="alex"),
    }

    task_metrics = {task: {m: [] for m in metrics} for task in os.listdir(samples_dir)}
    individual_metrics = []

    for task_dir in samples_dir.iterdir():
        if not task_dir.is_dir():
            continue

        for sample_dir in task_dir.iterdir():
            if not sample_dir.is_dir():
                continue
            inner_sample_dir = next(sample_dir.iterdir())
            target = Image.open(inner_sample_dir / "target.png")
            reconstructed = Image.open(inner_sample_dir / "reconstructed.png")
            target = to_tensor(target).unsqueeze(0)
            reconstructed = to_tensor(reconstructed).unsqueeze(0)

            metric_values = {'task_name': task_dir.name, 'sample_dir': sample_dir.name}
            for name, metric in metrics.items():
                if name == 'lpips':
                    value = metric(reconstructed, target).detach().numpy().item()
                else:
                    value = metric(reconstructed, target).cpu().numpy().item()
                metric_values[name] = value
                task_metrics[task_dir.name][name].append(value)

            individual_metrics.append(metric_values)

    # Compute aggregated metrics and confidence intervals
    aggregated_metrics = []
    for task, task_data in task_metrics.items():
        for metric_name, values in task_data.items():
            # Bootstrap and statistical calculations
            bootstrap_values = bootstrap(values, func=np.mean)
            boot_mean = np.mean(bootstrap_values)
            boot_std = np.std(bootstrap_values)
            boot_p025, boot_p975 = np.quantile(bootstrap_values, [0.025, 0.975])

            mean = np.mean(values)
            std = np.std(values) / np.sqrt(len(values))
            p025, p975 = mean - 1.96 * std, mean + 1.96 * std

            agg_metric = {
                'task_name': task,
                'metric': metric_name,
                'mean': mean,
                'std': std,
                'bootstrap_mean': boot_mean,
                'bootstrap_std': boot_std,
                'bootstrap_CI_lower': boot_p025,
                'bootstrap_CI_upper': boot_p975,
                'statistical_CI_lower': p025,
                'statistical_CI_upper': p975
            }
            aggregated_metrics.append(agg_metric)

    # Convert to DataFrame and save as CSV
    individual_df = pd.DataFrame(individual_metrics)
    aggregated_df = pd.DataFrame(aggregated_metrics)
    individual_df.to_csv(output_file.replace('.csv', '_individual.csv'), index=False)
    aggregated_df.to_csv(output_file.replace('.csv', '_aggregated.csv'), index=False)
    print(f"Saved individual metrics to {output_file.replace('.csv', '_individual.csv')}")
    print(f"Saved aggregated metrics to {output_file.replace('.csv', '_aggregated.csv')}")


def bootstrap(data, n=10000, func=np.mean):
    """Bootstrap estimate of distribution of statistics."""
    data = np.array(data)
    samples = np.random.choice(data, size=(n, len(data)))
    return [func(s) for s in samples]


if __name__ == "__main__":
    app.run(main)

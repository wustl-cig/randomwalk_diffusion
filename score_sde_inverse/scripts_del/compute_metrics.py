from pathlib import Path

import numpy as np
from absl import app, flags
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.transforms.functional import to_tensor
import lpips

FLAGS = flags.FLAGS
flags.DEFINE_string("samples_dir", "./logs/samples", "Directory to save samples")
flags.DEFINE_float("confidence", 0.95, "Confidence of computed metrics")


def main(_):
    samples_dir = Path(FLAGS.samples_dir)

    metrics = {
        "ssim": StructuralSimilarityIndexMeasure(data_range=(0, 1)),
        "psnr": PeakSignalNoiseRatio(data_range=(0, 1)),
        "lpips": lpips.LPIPS(net="alex"),
    }
    metric_values = {name: [] for name in metrics.keys()}

    for dataset_item_dir in samples_dir.iterdir():
        if not dataset_item_dir.is_dir():
            continue

        for sample_dir in dataset_item_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            target = Image.open(sample_dir / "target.png")
            reconstructed = Image.open(sample_dir / "reconstructed.png")
            target = to_tensor(target).unsqueeze(0)
            reconstructed = to_tensor(reconstructed).unsqueeze(0)

            for name, metric in metrics.items():
                value = metric(reconstructed, target)
                metric_values[name].append(value.detach().cpu().numpy().item())

    for name, values in metric_values.items():
        # Compute confidence interval for the metric
        # We do it in two ways: Bootstrapping / statistical.
        # They should give similar results.
        means = bootstrap(values, func=np.mean)
        mean = np.mean(means)
        std = np.std(means)
        p025, p975 = np.quantile(means, [0.025, 0.975])
        print(
            f"{name}: {mean:.4f} +/- {std:.4f} 95% CI: ({p025:.4f} - {p975:.4f}) (Bootstrap)"
        )

        mean = np.mean(values)
        std = np.std(values) / np.sqrt(len(values))
        p025, p975 = mean - 1.96 * std, mean + 1.96 * std
        print(
            f"{name}: {mean:.4f} +/- {std:.4f} 95% CI ({p025:.4f} - {p975:.4f}) (Statistical)"
        )


def bootstrap(data, n=10000, func=np.mean):
    """Bootstrap estimate of distribution of statistics."""
    data = np.array(data)

    samples = np.random.choice(data, size=(n, len(data)))
    stats = [func(s) for s in samples]

    return stats


if __name__ == "__main__":
    app.run(main)
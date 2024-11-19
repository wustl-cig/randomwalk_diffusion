from pathlib import Path

import gdown
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "checkpoint",
    None,
    "See README of https://github.com/yang-song/score_sde_pytorch. E.g. ve/cifar10_ncsnpp",
)
flags.mark_flag_as_required("checkpoint")

# To add other checkpoints, copy the checkpoint's share link from Google Drive and paste it here.
ALL_CHECKPOINTS = {
    "ve/cifar10_ncsnpp_deep_continuous": "https://drive.google.com/file/d/1yS8QZb_6tCeZkY7DK4_RI-Crc6LQLILN/view?usp=sharing",
    "subvp/cifar10_ddpmpp_deep_continuous": "https://drive.google.com/file/d/1bXoPbY28nReVIaNGutZfLzgS2uDK4k0I/view?usp=sharing",
    "ve/celebahq_256_ncsnpp_continuous": "https://drive.google.com/file/d/1ocvHVzAeYtwIRFPgqG1CPPHdXUzY85UG/view?usp=sharing",
}


def main(_):
    checkpoint = FLAGS.checkpoint

    url = ALL_CHECKPOINTS[checkpoint]
    output = Path("checkpoints") / checkpoint

    output.mkdir(parents=True, exist_ok=True)

    gdown.download(url=url, output=f"{output}/", fuzzy=True)


if __name__ == "__main__":
    app.run(main)

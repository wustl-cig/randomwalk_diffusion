import torch
from torch.utils.data import Dataset
from torchvision.datasets import CelebA as _CelebA
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor


class CelebA(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()

        self.dataset = _CelebA(
            "~/.cache/torchvision",
            download=True,
            transform=Compose(
                [
                    CenterCrop(140),
                    Resize(256, antialias=True),
                    ToTensor(),
                ]
            ),
            split='train' if train else 'test',
        )
        self.img_size = (3, 256, 256)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index) -> torch.Tensor:
        x, y = self.dataset[index]
        return x

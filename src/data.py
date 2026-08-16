"""MNIST data pipeline shared by every training script."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Maps [0,1] pixel range to [-1,1], matching the UNet's expected input range
# (all 5 methods define x1 ~ N(0,I) as "full noise", so data must be centered).
_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)


def get_mnist_loader(
    root: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    download: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=root, train=train, download=download, transform=_TRANSFORM
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
        pin_memory=torch.cuda.is_available(),
    )

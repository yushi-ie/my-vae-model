from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(
    data_dir: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int = 0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and test data loaders for the MNIST dataset.
    """
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )
    train_size = int(len(train_dataset) * 0.8)
    test_size = len(train_dataset) - train_size
    train_data, test_data = random_split(train_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader

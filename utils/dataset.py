from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class MusicImageDataset(Dataset):
    """
    Dataset for flat directories that store spectrogram (or other) image files.
    It returns a dummy label (0) because the current VAE training loop expects
    each sample to be a tuple of (image, label).
    """

    _SUPPORTED_EXTS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

    def __init__(
        self,
        root: Union[Path, str],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Music dataset path not found: {self.root}")

        self.transform = transform
        self.images = self._gather_images()
        if not self.images:
            raise RuntimeError(f"No supported image files were found in {self.root}.")

    def _gather_images(self) -> List[Path]:
        images: List[Path] = []
        for ext in self._SUPPORTED_EXTS:
            images.extend(self.root.glob(f"*{ext}"))
        images.sort()
        return images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


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


def get_ImageNet_loaders(
    data_dir: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int = 4,
    download: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and validation loaders for spectrogram images stored under `data/music`.
    The `download` flag is kept for compatibility but ignored because the data must
    already exist locally.
    """

    music_dir = Path(data_dir) / "music"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    dataset = MusicImageDataset(music_dir, transform=transform)
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader

import argparse
from pathlib import Path
from typing import Optional

import torch

from config import Config
from model.vae_cnn import VAE
from utils.dataset import get_mnist_loaders
from utils.dataset import get_ImageNet_loaders
from utils.dataset import MusicImageDataset
from utils.train_cnn import train
from utils.visualize import save_history_curves, save_recon_grid, save_samples



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dual-encoder VAE on MNIST.")
    parser.add_argument("--epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--test-batch-size", type=int, help="Validation batch size.")
    parser.add_argument("--z-dim", type=int, help="Latent space dimensionality.")
    parser.add_argument("--output-dir", type=Path, help="Directory for model outputs.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        help="Computation device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=0,
        help="Digit label used for average latent decoding.",
    )
    return parser.parse_args()


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.test_batch_size is not None:
        config.test_batch_size = args.test_batch_size
    if args.z_dim is not None:
        config.z_dim = args.z_dim
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    return config


def select_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

features = {}

def save_hook(name):
    def hook(module, input, output):
        features[name] = output.detach()
    return hook



def main() -> None:
    args = parse_args()
    config = apply_overrides(Config(), args)
    device = select_device(args.device)

    train_loader, test_loader = get_ImageNet_loaders(
        data_dir=str(config.data_dir),
        batch_size=config.batch_size,
        test_batch_size=config.test_batch_size,
        num_workers=config.num_workers,
    )

    model = VAE(config.z_dim)
    history = train(
        model,
        train_loader,
        test_loader,
        device=device,
        epochs=config.epochs,
        lr=config.learning_rate,
        step_size=config.step_size,
        gamma=config.gamma,
        output_dir=str(config.output_dir),
    )

    config.output_dir = Path(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    recon_path = config.output_dir / config.recon_filename
    save_recon_grid(model, test_loader, device, str(recon_path))

    sample_path = config.output_dir / config.sample_filename
    save_samples(model, history, device, str(sample_path), label=args.label)

    history_path = config.output_dir / "history.png"
    save_history_curves(history, str(history_path))

    model_path = config.output_dir / "vae.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model checkpoint saved to {model_path}")

    save_history_curves(
    history,
    str(config.output_dir / "metrics.png"),
    metrics=("train_loss", "val_loss", "train_acc", "val_acc"),
)


if __name__ == "__main__":
    main()

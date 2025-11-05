import math
import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import torch


HistoryLike = Dict[str, Sequence[torch.Tensor]]


def save_recon_grid(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: str,
    num_images: int = 8,
) -> None:
    """
    Save a grid that shows original inputs (top row) and reconstructions (bottom row).
    """
    model.eval()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    images, _ = next(iter(data_loader))
    images = images[:num_images]
    inputs = images.view(images.size(0), -1).to(device, dtype=torch.float32)

    with torch.no_grad():
        reconstructions, *_ = model(inputs)

    reconstructions = reconstructions.view(-1, 1, 28, 28).cpu()
    originals = images.cpu()

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.5, 3))
    for idx in range(num_images):
        axes[0, idx].imshow(originals[idx, 0], cmap="gray")
        axes[0, idx].axis("off")
        axes[1, idx].imshow(reconstructions[idx, 0], cmap="gray")
        axes[1, idx].axis("off")

    fig.suptitle("Top: original digits, Bottom: reconstructions", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_samples(
    model: torch.nn.Module,
    history: HistoryLike,
    device: torch.device,
    out_path: str,
    label: int = 0,
) -> None:
    """
    Decode the average latent vector for a given label and save it as an image.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    mu1 = torch.cat(history["mu1"], dim=0)
    mu2 = torch.cat(history["mu2"], dim=0)
    labels = torch.cat(history["labels"], dim=0)

    mask = labels == label
    if mask.sum() == 0:
        raise ValueError(f"No samples with label {label} found in history.")

    avg_mu = (mu1[mask].mean(dim=0) + mu2[mask].mean(dim=0)) / 2

    model.eval()
    with torch.no_grad():
        decoded = model.decoder(avg_mu.to(device))

    image = decoded.view(28, 28).cpu().numpy()

    fig = plt.figure(figsize=(2, 2))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    fig.suptitle(f"Average latent sample for label {label}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

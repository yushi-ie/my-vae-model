import math
import os
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import torch


HistoryLike = Dict[str, Sequence[torch.Tensor]]


def _prepare_inputs(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Prepare an input batch for the given model.

    Linear VAEs expect flattened vectors while CNN VAEs typically receive 4D tensors.
    """
    if images.ndim == 4 and getattr(model, "in_channels", None) is not None:
        return images
    if images.ndim == 4:
        return images.view(images.size(0), -1)
    return images


def _prepare_recons(reconstructions: torch.Tensor) -> torch.Tensor:
    """
    Ensure reconstructions are returned as (N, C, H, W).
    """
    if reconstructions.ndim == 4:
        return reconstructions
    if reconstructions.ndim == 2:
        side = int(math.sqrt(reconstructions.size(1)))
        if side * side != reconstructions.size(1):
            raise ValueError("Reconstruction vector cannot be reshaped into a square image.")
        return reconstructions.view(-1, 1, side, side)
    raise ValueError("Unexpected reconstruction tensor shape.")


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _to_series(values: Sequence[torch.Tensor | float]) -> list[float]:
    series: list[float] = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                series.append(float(value.item()))
            else:
                series.append(float(value.mean().item()))
        else:
            series.append(float(value))
    return series


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
    _ensure_dir(out_path)

    images, _ = next(iter(data_loader))
    images = images[:num_images]
    inputs = _prepare_inputs(model, images).to(device, dtype=torch.float32)

    with torch.no_grad():
        reconstructions, *_ = model(inputs)

    reconstructions = _prepare_recons(reconstructions).cpu()
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
    _ensure_dir(out_path)

    labels = torch.cat(history["labels"], dim=0)
    mask = labels == label
    if mask.sum() == 0:
        raise ValueError(f"No samples with label {label} found in history.")

    if "mu1" in history and "mu2" in history:
        mu1 = torch.cat(history["mu1"], dim=0)
        mu2 = torch.cat(history["mu2"], dim=0)
        avg_mu = (mu1[mask].mean(dim=0) + mu2[mask].mean(dim=0)) / 2
    elif "mu" in history:
        mu = torch.cat(history["mu"], dim=0)
        avg_mu = mu[mask].mean(dim=0)
    else:
        raise KeyError("History must contain mu information (mu1/mu2 or mu).")

    model.eval()
    latent = avg_mu.unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        decoded = model.decoder(latent)

    decoded = _prepare_recons(decoded).squeeze(0)
    image = decoded[0].cpu().numpy()

    fig = plt.figure(figsize=(2, 2))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    fig.suptitle(f"Average latent sample for label {label}")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_history_curves(
    history: HistoryLike,
    out_path: str,
    metrics: Sequence[str] = ("train_loss", "val_loss"),
) -> None:
    """
    Plot learning curves for the requested metrics and save to disk.
    """
    _ensure_dir(out_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    epochs = None

    for metric in metrics:
        if metric not in history:
            continue
        values = _to_series(history[metric])
        if not values:
            continue
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", label=metric.replace("_", " ").title())
        plotted = True

    if not plotted:
        plt.close(fig)
        raise ValueError("No valid metrics were found in history to plot.")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Training History")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


History = Dict[str, List[Union[torch.Tensor, float]]]


def criterion(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the VAE loss (reconstruction + KL divergence for both encoders).
    """
    bce_loss = F.binary_cross_entropy(reconstruction, target, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    device: Optional[torch.device] = None,
    epochs: int = 30,
    lr: float = 1e-3,
    step_size: int = 15,
    gamma: float = 0.1,
    output_dir: Optional[str] = None,
) -> History:
    """
    Run the training loop for the VAE model and return training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    history: History = {
        "train_loss": [],
        "val_loss": [],
        "mu": [],
        "logvar": [],
        "labels": [],
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            inputs = images.to(device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            reconstruction, z, mu, logvar = model(inputs)
            loss = criterion(reconstruction, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            history["mu"].append(mu.detach().cpu())
            history["logvar"].append(logvar.detach().cpu())
            history["labels"].append(labels.detach().cpu())

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}, step {batch_idx + 1}: loss={loss.item():.4f}")
                print(z.shape)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                inputs = images.to(device, dtype=torch.float32)
                reconstruction, _, mu, logvar = model(inputs)
                batch_loss = criterion(reconstruction, inputs, mu, logvar)
                val_loss += batch_loss.item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)
        print(f"Epoch {epoch + 1}: val_loss={epoch_val_loss:.4f}")

        scheduler.step()

    return history

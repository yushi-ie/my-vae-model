import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc_mu = nn.Linear(100, z_dim)
        self.fc_logvar = nn.Linear(100, z_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.fc1(x))
        hidden = self.activation(self.fc2(hidden))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z, mu, logvar


class Encoder2(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 150)
        self.fc_mu = nn.Linear(150, z_dim)
        self.fc_logvar = nn.Linear(150, z_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.dropout(self.activation(self.fc1(x)))
        hidden = self.activation(self.fc2(hidden))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(z_dim, 100)
        self.fc2 = nn.Linear(100, 300)
        self.fc3 = nn.Linear(300, 28 * 28)
        self.activation = nn.ReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.fc1(z))
        hidden = self.activation(self.fc2(hidden))
        logits = self.fc3(hidden)
        return torch.sigmoid(logits)


class VAE(nn.Module):
    def __init__(self, z_dim: int) -> None:
        super().__init__()
        self.encoder1 = Encoder(z_dim)
        self.encoder2 = Encoder2(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1, mu1, logvar1 = self.encoder1(x)
        z2, mu2, logvar2 = self.encoder2(x)
        z = 0.5 * (z1 + z2)
        reconstruction = self.decoder(z)
        return reconstruction, z1, z2, mu1, mu2, logvar1, logvar2

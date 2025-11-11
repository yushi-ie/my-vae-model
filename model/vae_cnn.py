import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, base_width: int, z_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_width*2, base_width*4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_width*4, base_width*8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_width*8, base_width*16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_width*16, base_width*32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        hidden_dim = base_width * 32 * 4 * 4
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.features(x).view(x.size(0), -1)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(0.5 * logvar) * eps
        return z, mu, logvar


class FeatureDecoderExtended(nn.Module):
    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()
        
        # --- Stage 1: 4x4 -> 8x8 (C: 1024 -> 512) ---
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            # 入力を base_width * 32 (1024) に修正
            nn.Conv2d(base_width * 32, base_width * 16, kernel_size=3, padding=1), 
            nn.InstanceNorm2d(base_width * 16),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 16, base_width * 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 16, base_width * 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 16),
            nn.ReLU(inplace=True),
        )
        
        # --- Stage 2: 8x8 -> 16x16 (C: 512 -> 256) ---
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 16, base_width * 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        # --- Stage 3: 16x16 -> 32x32 (C: 256 -> 128) ---
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        
        # --- Stage 4: 32x32 -> 64x64 (C: 128 -> 64) ---
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        # --- Stage 5: 64x64 -> 128x128 (C: 64 -> 32) ---
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db5 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        
        # --- Stage 6: 128x128 -> 256x256 (C: 32 -> 16) ---
        # 最終段階ではチャネル数を base_width/2 に減らします
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
        )
        self.db6 = nn.Sequential(
            nn.Conv2d(base_width // 2, base_width // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width // 2, base_width // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width // 2),
            nn.ReLU(inplace=True),
        )
        
        # 最終出力層
        self.fin_out = nn.Conv2d(base_width // 2, out_channels, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Stage 1: 4x4 -> 8x8
        x = self.up1(features)
        x = self.db1(x)
        
        # Stage 2: 8x8 -> 16x16
        x = self.up2(x)
        x = self.db2(x)
        
        # Stage 3: 16x16 -> 32x32
        x = self.up3(x)
        x = self.db3(x)

        # Stage 4: 32x32 -> 64x64
        x = self.up4(x)
        x = self.db4(x)

        # Stage 5: 64x64 -> 128x128
        x = self.up5(x)
        x = self.db5(x)

        # Stage 6: 128x128 -> 256x256
        x = self.up6(x)
        x = self.db6(x)
        
        # Final Output
        return self.fin_out(x)


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int, base_width: int, z_dim: int) -> None:
        super().__init__()
        
        # FC層の出力次元を 4x4 特徴マップ (C=base_width*32) に修正
        final_channels = base_width * 32  # 1024
        final_map_size = 4
        hidden_dim = final_channels * final_map_size * final_map_size  # 16384
        
        self.fc = nn.Linear(z_dim, hidden_dim)
        
        # 拡張したデコーダを組み込む
        self.feature_decoder = FeatureDecoderExtended(base_width, out_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 1. FC層で拡大
        hidden = self.fc(z)
        
        # 2. 4x4の特徴マップ形状に整形
        final_map_size = 4
        # out_features // (4 * 4) でチャネル数を計算する
        final_channels = self.fc.out_features // (final_map_size * final_map_size) 
        x = hidden.view(z.size(0), final_channels, final_map_size, final_map_size) 
        
        # 3. アップサンプリング
        logits = self.feature_decoder(x)
        
        # 4. 最終出力
        return torch.sigmoid(logits)


class VAE(nn.Module):
    def __init__(
        self, z_dim: int, in_channels: int = 1, base_width: int = 32
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.encoder = ConvEncoder(in_channels, base_width, z_dim)
        self.decoder = ConvDecoder(in_channels, base_width, z_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            side = int(x.size(1) ** 0.5)
            x = x.view(x.size(0), self.in_channels, side, side)
        z, mu, logvar = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z, mu, logvar

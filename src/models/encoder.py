"""
Latent Satellite Embedding Network (SatelliteEncoder) for OceanEmbed.
Combines CoordConv, Multi-Scale Residual CNN, Squeeze-and-Excitation Channel Attention,
and Spatial Window Self-Attention (Swin/ViT-inspired) to transform multi-modal 2D surface
observations into rich, compact latent ocean embeddings.
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Module):
    """
    CoordConv: Injects explicit 2D spatial coordinate channels (Latitude & Longitude)
    into the input tensor.

    Oceanographic Rationale:
    Planetary fluid dynamics strictly depend on latitude due to the Coriolis parameter
    f = 2 * Omega * sin(lat). Standard translation-invariant convolutions cannot inherently
    distinguish near-equatorial dynamics (e.g., 5°N) from mid-latitude/subtropical regimes (e.g., 25°N).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B, C, H, W)
        Returns:
            torch.Tensor: Shape (B, C + 2, H, W)
        """
        b, _, h, w = x.shape
        device = x.device
        dtype = x.dtype

        # Coordinate grids in [-1, 1]
        lat_coords = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        lon_coords = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)

        return torch.cat([x, lat_coords, lon_coords], dim=1)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for adaptive multi-modal channel recalibration.
    Allows the model to dynamically weight multi-modal features (e.g. prioritize SSS in the
    salinity-stratified Bay of Bengal, or SSH during mesoscale eddy passages).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_ch = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, reduced_ch, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced_ch, channels, kernel_size=1)
        self.act = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.act(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w


class ResidualBlock2d(nn.Module):
    """
    2D Residual Convolutional Block with GroupNorm, GELU, and Squeeze-and-Excitation.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.GELU()

        self.se = SqueezeExcitation(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.se(out)
        out = self.act2(out + res)
        return out


class WindowSelfAttention2d(nn.Module):
    """
    Swin/ViT-inspired Spatial Self-Attention for planetary wave teleconnections.
    Enables long-range dependencies across the basin (e.g. Rossby & Kelvin wave propagation).
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(min(8, dim), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2] # (b, heads, head_dim, N)

        # Scaled dot-product attention
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale # (b, heads, N, N)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(v, attn.transpose(-2, -1)) # (b, heads, head_dim, N)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)

        return residual + out


class SatelliteEncoder(nn.Module):
    """
    Multi-Scale Latent Satellite Embedding Network.
    
    Transforms 7 multi-modal surface observations:
    [SST, SSS, SSH, U_curr, V_curr, U_wind, V_wind] (+ 2 CoordConv coords)
    into a structured 256-channel latent embedding space.
    """

    def __init__(
        self,
        in_channels: int = 7,
        use_coordconv: bool = True,
        base_channels: int = 64,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_coordconv = use_coordconv
        actual_in_channels = in_channels + (2 if use_coordconv else 0)

        # Stage 0: Stem Projection
        self.stem = nn.Sequential(
            nn.Conv2d(actual_in_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            ResidualBlock2d(base_channels, base_channels),
        )

        # Stage 1: Multi-scale Block 1 (Full resolution H, W)
        self.stage1 = ResidualBlock2d(base_channels, base_channels)

        # Stage 2: Downsampled Block 2 (H/2, W/2)
        ch2 = base_channels * 2 # 128
        self.stage2 = nn.Sequential(
            ResidualBlock2d(base_channels, ch2, stride=2),
            ResidualBlock2d(ch2, ch2),
        )

        # Stage 3: Downsampled Block 3 (H/4, W/4) with Spatial Self-Attention
        ch3 = base_channels * 4 # 256
        self.stage3 = nn.Sequential(
            ResidualBlock2d(ch2, ch3, stride=2),
            ResidualBlock2d(ch3, ch3),
            WindowSelfAttention2d(ch3, num_heads=4),
        )

        # Bottleneck: Satellite Embedding Projection
        self.embedding_conv = nn.Sequential(
            nn.Conv2d(ch3, embedding_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, embedding_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
        )

        # Global basin state pooling (e.g. basin-wide heat content / IOD index)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        if use_coordconv:
            self.coord_layer = CoordConv2d()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x (torch.Tensor): Shape (B, C_in, H, W) where C_in is 7 (or 9 if already appended)
        Returns:
            embedding (torch.Tensor): Shape (B, embedding_dim, H/4, W/4)
            global_vec (torch.Tensor): Shape (B, embedding_dim)
            skip_features (List[torch.Tensor]): [f1: (B, 64, H, W), f2: (B, 128, H/2, W/2)]
        """
        # If coordconv is configured and not already added in x
        if self.use_coordconv and x.shape[1] == 7:
            x = self.coord_layer(x)

        f1 = self.stem(x)         # (B, 64, H, W)
        f1 = self.stage1(f1)      # (B, 64, H, W)

        f2 = self.stage2(f1)      # (B, 128, H/2, W/2)

        f3 = self.stage3(f2)      # (B, 256, H/4, W/4)

        # Compute compact 256-channel embedding
        embedding = self.embedding_conv(f3) # (B, 256, H/4, W/4)

        # Global pooled vector
        b, c, _, _ = embedding.shape
        pooled = self.global_pool(embedding).view(b, c)
        global_vec = self.global_fc(pooled) # (B, 256)

        return embedding, global_vec, [f1, f2]

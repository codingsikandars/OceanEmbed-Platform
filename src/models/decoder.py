"""
3D Depth Profile Reconstruction Decoder (SubsurfaceDecoder) for OceanEmbed.
Hierarchical multi-scale decoder with skip-connections and Vertical Depth-Wise Attention
to reconstruct 3D subsurface temperature fields across standard vertical depth levels.
"""

from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ResidualBlock2d


class DepthAttentionBlock(nn.Module):
    """
    Vertical Depth-Wise Attention Mechanism.
    
    Models non-local vertical interactions across the 15 ocean depth levels.
    In the ocean, vertical stratification creates physical coupling:
    - Upper depths (0-50m) represent the turbulent Mixed Layer
    - Middle depths (50-200m) represent the sharp Thermocline (D20 isotherm)
    - Lower depths (200-1000m) represent intermediate and deep water masses
    """

    def __init__(self, num_depths: int = 15, channel_dim: int = 64):
        super().__init__()
        self.num_depths = num_depths
        self.channel_dim = channel_dim

        # Depth-level learnable positional embeddings (encodes 0m to 1000m depth positions)
        self.depth_embedding = nn.Parameter(torch.randn(1, num_depths, channel_dim) * 0.02)

        # Cross-depth self-attention
        self.query = nn.Linear(channel_dim, channel_dim)
        self.key = nn.Linear(channel_dim, channel_dim)
        self.value = nn.Linear(channel_dim, channel_dim)
        self.scale = 1.0 / math.sqrt(channel_dim)

        self.norm = nn.LayerNorm(channel_dim)
        self.proj = nn.Linear(channel_dim, channel_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Shape (B, num_depths, channel_dim, H, W)
        Returns:
            torch.Tensor: Shape (B, num_depths, channel_dim, H, W)
        """
        b, d, c, h, w = x.shape
        # Permute to (B, H*W, D, C) to apply vertical depth attention at each spatial location
        x_perm = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, d, c)
        
        # Add vertical depth coordinate embedding
        x_emb = x_perm + self.depth_embedding
        x_norm = self.norm(x_emb)

        q = self.query(x_norm) # (B*H*W, D, C)
        k = self.key(x_norm)   # (B*H*W, D, C)
        v = self.value(x_norm) # (B*H*W, D, C)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale # (B*H*W, D, D)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attended = torch.bmm(attn_weights, v) # (B*H*W, D, C)
        attended = self.proj(attended)

        out = (x_perm + attended).reshape(b, h, w, d, c).permute(0, 3, 4, 1, 2).contiguous()
        return out


class SubsurfaceDecoder(nn.Module):
    """
    Hierarchical 3D Subsurface Temperature Reconstruction Decoder.
    
    Decodes the 256-channel satellite embedding Z into a 3D subsurface temperature field
    (B, 15, H, W) across the 15 standard depth levels.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        base_channels: int = 64,
        num_depth_levels: int = 15,
        use_depth_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_depth_levels = num_depth_levels
        self.use_depth_attention = use_depth_attention

        # Stage 1: Upsample from H/4, W/4 to H/2, W/2 and fuse skip connection f2 (128 channels)
        ch_mid = base_channels * 2 # 128
        self.upconv1 = nn.Conv2d(embedding_dim, ch_mid, kernel_size=3, padding=1)
        self.fuse1 = ResidualBlock2d(ch_mid + ch_mid, ch_mid)

        # Stage 2: Upsample from H/2, W/2 to H, W and fuse skip connection f1 (64 channels)
        self.upconv2 = nn.Conv2d(ch_mid, base_channels, kernel_size=3, padding=1)
        self.fuse2 = ResidualBlock2d(base_channels + base_channels, base_channels)

        # Stage 3: Subsurface 3D Projection
        # Generates a depth-feature volume: (B, num_depth_levels * 16, H, W)
        self.depth_feature_dim = 16
        self.pre_depth_conv = nn.Sequential(
            ResidualBlock2d(base_channels, base_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(base_channels, num_depth_levels * self.depth_feature_dim, kernel_size=3, padding=1),
        )

        # Stage 4: Depth-Wise Attention Block
        if use_depth_attention:
            self.depth_attention = DepthAttentionBlock(
                num_depths=num_depth_levels,
                channel_dim=self.depth_feature_dim
            )

        # Stage 5: Final Temperature Prediction Head
        # Projects each depth feature slice to a scalar temperature field
        self.final_head = nn.Sequential(
            nn.Conv2d(self.depth_feature_dim, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )

    def forward(
        self,
        embedding: torch.Tensor,
        skip_features: List[torch.Tensor],
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        Args:
            embedding (torch.Tensor): Latent embedding (B, embedding_dim, H/4, W/4)
            skip_features (List[torch.Tensor]): [f1: (B, 64, H, W), f2: (B, 128, H/2, W/2)]
            target_size (Tuple[int, int]): Final (H, W) resolution (default: from f1)
        Returns:
            temp_3d (torch.Tensor): Reconstructed temperature profiles (B, 15, H, W)
        """
        f1, f2 = skip_features
        if target_size is None:
            target_size = (f1.shape[2], f1.shape[3])

        # Step 1: Upsample to f2 resolution (H/2, W/2)
        x_up1 = F.interpolate(embedding, size=(f2.shape[2], f2.shape[3]), mode="bilinear", align_corners=False)
        x_up1 = self.upconv1(x_up1)
        x_fuse1 = self.fuse1(torch.cat([x_up1, f2], dim=1)) # (B, 128, H/2, W/2)

        # Step 2: Upsample to f1 resolution (H, W)
        x_up2 = F.interpolate(x_fuse1, size=target_size, mode="bilinear", align_corners=False)
        x_up2 = self.upconv2(x_up2)
        x_fuse2 = self.fuse2(torch.cat([x_up2, f1], dim=1)) # (B, 64, H, W)

        # Step 3: Expand into depth-channel tensor
        b, _, h, w = x_fuse2.shape
        depth_vol = self.pre_depth_conv(x_fuse2) # (B, 15 * 16, H, W)
        depth_vol = depth_vol.view(b, self.num_depth_levels, self.depth_feature_dim, h, w)

        # Step 4: Cross-depth vertical attention
        if self.use_depth_attention:
            depth_vol = self.depth_attention(depth_vol)

        # Step 5: Final temperature estimation per depth
        # Reshape to batch across depths for efficient 2D convolution
        depth_vol_flat = depth_vol.contiguous().reshape(b * self.num_depth_levels, self.depth_feature_dim, h, w)
        temp_flat = self.final_head(depth_vol_flat) # (B * 15, 1, H, W)
        temp_3d = temp_flat.reshape(b, self.num_depth_levels, h, w) # (B, 15, H, W)

        return temp_3d

"""
OceanEmbedNet: Complete End-to-End Model Wrapper for Subsurface Ocean Temperature Reconstruction.
Integrates SatelliteEncoder (Swin/ViT-inspired multi-scale CNN) and SubsurfaceDecoder (with Depth-Wise Attention).
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn

from .encoder import SatelliteEncoder
from .decoder import SubsurfaceDecoder


class OceanEmbedNet(nn.Module):
    """
    End-to-End Deep Learning Architecture for SIH Problem Statement ID 26066.
    
    Inputs:
        2D Multi-Modal Daily Surface Satellite Observations:
        - SST (OSTIA)
        - SSS (SMAP/SMOS)
        - SSH/SLA (DUACS)
        - OSCAR Currents (U, V)
        - ASCAT/CCMP Winds (U, V)
        (+ CoordConv 2D positional coordinates)

    Outputs:
        - 3D Reconstructed Subsurface Temperature across 15 standard depths:
          [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] meters.
        - Compact 256-channel Latent Satellite Embedding.
    """

    def __init__(
        self,
        in_channels: int = 7,
        num_depth_levels: int = 15,
        embedding_dim: int = 256,
        base_channels: int = 64,
        use_coordconv: bool = True,
        use_depth_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_depth_levels = num_depth_levels
        self.embedding_dim = embedding_dim
        self.use_coordconv = use_coordconv

        # 1. Latent Satellite Embedding Network
        self.encoder = SatelliteEncoder(
            in_channels=in_channels,
            use_coordconv=use_coordconv,
            base_channels=base_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

        # 2. 3D Subsurface Profile Reconstruction Decoder
        self.decoder = SubsurfaceDecoder(
            embedding_dim=embedding_dim,
            base_channels=base_channels,
            num_depth_levels=num_depth_levels,
            use_depth_attention=use_depth_attention,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training & inference.

        Args:
            x (torch.Tensor): Surface satellite observations (B, C_in, H, W)
        Returns:
            temp_3d (torch.Tensor): Predicted subsurface temperature profiles (B, 15, H, W)
        """
        # Encode to latent satellite embedding and multiscale skips
        embedding, _, skips = self.encoder(x)
        # Decode to 3D subsurface temperature field
        temp_3d = self.decoder(embedding, skips, target_size=(x.shape[2], x.shape[3]))
        return temp_3d

    def forward_with_embeddings(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns predictions along with the compact latent satellite embedding.

        Returns:
            temp_3d: (B, 15, H, W)
            embedding: (B, 256, H/4, W/4)
            global_vec: (B, 256)
        """
        embedding, global_vec, skips = self.encoder(x)
        temp_3d = self.decoder(embedding, skips, target_size=(x.shape[2], x.shape[3]))
        return temp_3d, embedding, global_vec

    def extract_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts only the latent satellite embeddings without running the decoder.
        Useful for downstream oceanographic foundation model tasks (e.g. marine heatwaves,
        cyclone track prediction, eddy classification).

        Returns:
            embedding: (B, 256, H/4, W/4)
            global_vec: (B, 256)
        """
        with torch.no_grad():
            embedding, global_vec, _ = self.encoder(x)
        return embedding, global_vec

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_oceanembed_model(config: Dict[str, Any]) -> OceanEmbedNet:
    """
    Factory function to construct an OceanEmbedNet instance from a YAML config dictionary.
    """
    model_cfg = config.get("model", {})
    depth_cfg = config.get("depth", {})
    features_cfg = config.get("features", {})

    base_feature_channels = features_cfg.get("in_channels", 7)
    time_window = int(config.get("temporal", {}).get("time_window", 1))
    in_channels = base_feature_channels * time_window
    num_depths = depth_cfg.get("num_levels", 15)
    embedding_dim = model_cfg.get("embedding_dim", 256)
    base_channels = model_cfg.get("base_channels", 64)
    use_coordconv = model_cfg.get("use_coordconv", True)
    use_depth_attention = model_cfg.get("depth_attention", True)
    dropout = model_cfg.get("dropout", 0.1)

    model = OceanEmbedNet(
        in_channels=in_channels,
        num_depth_levels=num_depths,
        embedding_dim=embedding_dim,
        base_channels=base_channels,
        use_coordconv=use_coordconv,
        use_depth_attention=use_depth_attention,
        dropout=dropout,
    )
    return model

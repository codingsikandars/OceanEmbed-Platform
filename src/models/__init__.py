"""
Deep Learning model architectures for OceanEmbed:
- SatelliteEncoder: Latent multi-modal satellite embedding network
- SubsurfaceDecoder: 3D Depth profile reconstruction decoder
- OceanEmbedNet: Complete end-to-end framework
"""

from .encoder import SatelliteEncoder, CoordConv2d
from .decoder import SubsurfaceDecoder, DepthAttentionBlock
from .oceanembed_net import OceanEmbedNet, build_oceanembed_model

__all__ = [
    "SatelliteEncoder",
    "CoordConv2d",
    "SubsurfaceDecoder",
    "DepthAttentionBlock",
    "OceanEmbedNet",
    "build_oceanembed_model",
]

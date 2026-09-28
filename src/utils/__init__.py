"""
Loss functions, physics-informed regularizers, and oceanographic evaluation metrics for OceanEmbed.
"""

from .losses import PhysicsInformedOceanLoss, MaskedMSELoss, VerticalGradientLoss
from .metrics import (
    compute_depth_metrics,
    compute_vertical_gradient_error,
    compute_thermocline_depth_error,
    calculate_correlation,
)

__all__ = [
    "PhysicsInformedOceanLoss",
    "MaskedMSELoss",
    "VerticalGradientLoss",
    "compute_depth_metrics",
    "compute_vertical_gradient_error",
    "compute_thermocline_depth_error",
    "calculate_correlation",
]

"""
Physics-Informed Loss Functions for Subsurface Ocean Temperature Reconstruction.
Supports SIH 26066: OceanEmbed.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.preprocess import STANDARD_DEPTHS


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss computed exclusively over valid ocean cells.
    Land pixels (mask == 0) are strictly excluded from gradient updates.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, D, H, W)
            target: (B, D, H, W)
            mask: (B, H, W) or (H, W) where 1.0 = ocean, 0.0 = land
        """
        if mask.dim() == 2:
            # (H, W) -> (1, 1, H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # (B, H, W) -> (B, 1, H, W)
            mask = mask.unsqueeze(1)

        diff = (pred - target) * mask
        num_ocean_pixels = torch.clamp(mask.sum() * pred.shape[1], min=1.0)
        loss = torch.sum(diff ** 2) / num_ocean_pixels
        return loss


class VerticalGradientLoss(nn.Module):
    """
    Vertical Thermal Gradient Loss (dT/dz).
    
    Penalizes errors in the vertical derivative of temperature across standard depth levels.
    Preserves thermocline sharpness, prevents vertical over-smoothing, and ensures accurate
    reconstruction of the 20°C isotherm depth (D20) and Mixed Layer Depth (MLD).
    """

    def __init__(self, depths: Optional[np.ndarray] = None):
        super().__init__()
        if depths is None:
            depths = STANDARD_DEPTHS
        
        self.depths = torch.from_numpy(depths.astype(np.float32))
        # Compute delta z between consecutive depth levels: dz_k = z_{k+1} - z_k
        dz = self.depths[1:] - self.depths[:-1]
        self.register_buffer("dz", dz.view(1, -1, 1, 1)) # (1, D-1, 1, 1)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, D, H, W)
            target: (B, D, H, W)
            mask: (B, H, W) or (H, W)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Finite difference dT/dz along depth dimension (dim=1)
        # grad_T shape: (B, D-1, H, W)
        grad_pred = (pred[:, 1:] - pred[:, :-1]) / self.dz
        grad_target = (target[:, 1:] - target[:, :-1]) / self.dz

        diff = (grad_pred - grad_target) * mask
        num_elements = torch.clamp(mask.sum() * (pred.shape[1] - 1), min=1.0)
        return torch.sum(diff ** 2) / num_elements


class SurfaceConsistencyLoss(nn.Module):
    """
    Surface Dirichlet Consistency Loss.
    
    Enforces that the reconstructed top vertical layer (z = 0m) matches the observed
    satellite Sea Surface Temperature (SST).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_top: torch.Tensor,
        sst_input: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_top: (B, H, W) Predicted temperature at depth index 0 (0m)
            sst_input: (B, H, W) Observed surface SST from satellite
            mask: (B, H, W) or (H, W)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        diff = (pred_top - sst_input) * mask
        num_ocean_pixels = torch.clamp(mask.sum(), min=1.0)
        return torch.sum(diff ** 2) / num_ocean_pixels


class StratificationPenaltyLoss(nn.Module):
    """
    Monotonic Stratification Stability Loss.
    
    In open ocean physics, temperature strictly decreases with depth (gravitational stability),
    with the exception of small barrier layer inversions (<0.4°C) in the Bay of Bengal.
    This loss penalizes unphysical vertical temperature inversions where T(z_{k+1}) > T(z_k).
    """

    def __init__(self, inversion_threshold: float = 0.4):
        super().__init__()
        self.threshold = inversion_threshold

    def forward(self, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, D, H, W)
            mask: (B, H, W) or (H, W)
        """
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # dT = T(k+1) - T(k). Stably stratified ocean should have dT <= 0
        dt = pred[:, 1:] - pred[:, :-1]
        
        # Penalize positive dT exceeding inversion threshold
        inversions = F.relu(dt - self.threshold) * mask
        num_elements = torch.clamp(mask.sum() * (pred.shape[1] - 1), min=1.0)
        return torch.sum(inversions ** 2) / num_elements


class PhysicsInformedOceanLoss(nn.Module):
    """
    Comprehensive Physics-Informed Multi-Objective Loss:
    
    L_total = lambda_mse * L_mse 
            + lambda_grad * L_grad 
            + lambda_surf * L_surf 
            + lambda_strat * L_strat
    """

    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_grad: float = 0.5,
        lambda_surf: float = 0.2,
        lambda_strat: float = 0.05,
        depths: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.lambda_surf = lambda_surf
        self.lambda_strat = lambda_strat

        self.mse_loss = MaskedMSELoss()
        self.grad_loss = VerticalGradientLoss(depths=depths)
        self.surf_loss = SurfaceConsistencyLoss()
        self.strat_loss = StratificationPenaltyLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sst_input: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: Predicted temperature (B, 15, H, W)
            target: Ground truth temperature (B, 15, H, W)
            mask: Land/Ocean mask (B, H, W) or (H, W)
            sst_input: Optional surface SST (B, H, W)
        Returns:
            total_loss (torch.Tensor): Scalar loss for backpropagation
            loss_components (Dict[str, float]): Breakdown of individual loss terms
        """
        l_mse = self.mse_loss(pred, target, mask)
        l_grad = self.grad_loss(pred, target, mask)
        l_strat = self.strat_loss(pred, mask)

        total_loss = self.lambda_mse * l_mse + self.lambda_grad * l_grad + self.lambda_strat * l_strat

        loss_dict = {
            "loss_mse": l_mse.item(),
            "loss_grad": l_grad.item(),
            "loss_strat": l_strat.item(),
        }

        if sst_input is not None and self.lambda_surf > 0:
            l_surf = self.surf_loss(pred[:, 0], sst_input, mask)
            total_loss = total_loss + self.lambda_surf * l_surf
            loss_dict["loss_surf"] = l_surf.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict

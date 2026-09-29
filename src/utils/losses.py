"""
Physics-informed losses for OceanEmbed.

The reconstruction objective is computed in normalized model space for the
main MSE, while physically interpretable constraints are evaluated in Celsius.
This avoids applying a 0.4°C inversion threshold to z-scored temperatures.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.preprocess import STANDARD_DEPTHS


def _mask4(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 2:
        return mask[None, None]
    if mask.dim() == 3:
        return mask[:, None]
    return mask


class MaskedMSELoss(nn.Module):
    def forward(self, pred, target, mask):
        m = _mask4(mask)
        diff = (pred - target) * m
        denom = torch.clamp(m.sum() * pred.shape[1], min=1.0)
        return torch.sum(diff.square()) / denom


class VerticalGradientLoss(nn.Module):
    """MSE of dT/dz, scaled to degrees C per 100 m for numerically useful gradients."""

    def __init__(self, depths: Optional[np.ndarray] = None, scale_per_m: float = 100.0):
        super().__init__()
        depths = STANDARD_DEPTHS if depths is None else depths
        dz = np.diff(depths).astype(np.float32)
        self.register_buffer("dz", torch.from_numpy(dz).view(1, -1, 1, 1))
        self.scale_per_m = float(scale_per_m)

    def forward(self, pred_c, target_c, mask):
        m = _mask4(mask)
        grad_p = (pred_c[:, 1:] - pred_c[:, :-1]) / self.dz * self.scale_per_m
        grad_t = (target_c[:, 1:] - target_c[:, :-1]) / self.dz * self.scale_per_m
        diff = (grad_p - grad_t) * m
        denom = torch.clamp(m.sum() * (pred_c.shape[1] - 1), min=1.0)
        return torch.sum(diff.square()) / denom


class SurfaceConsistencyLoss(nn.Module):
    def forward(self, pred_top_c, sst_c, mask):
        m = _mask4(mask).squeeze(1)
        diff = (pred_top_c - sst_c) * m
        return torch.sum(diff.square()) / torch.clamp(m.sum(), min=1.0)


class StratificationPenaltyLoss(nn.Module):
    """
    Penalizes strong positive dT/dz inversions in Celsius.

    A small inversion tolerance is retained because barrier-layer and
    measurement/reanalysis effects can produce weak inversions.
    """

    def __init__(self, inversion_threshold_c: float = 0.4):
        super().__init__()
        self.threshold = float(inversion_threshold_c)

    def forward(self, pred_c, mask):
        m = _mask4(mask)
        dt = pred_c[:, 1:] - pred_c[:, :-1]
        penalty = F.relu(dt - self.threshold) * m
        denom = torch.clamp(m.sum() * (pred_c.shape[1] - 1), min=1.0)
        return torch.sum(penalty.square()) / denom


class PhysicsInformedOceanLoss(nn.Module):
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_grad: float = 0.5,
        lambda_surf: float = 0.2,
        lambda_strat: float = 0.05,
        depths: Optional[np.ndarray] = None,
        target_mean: Optional[np.ndarray] = None,
        target_std: Optional[np.ndarray] = None,
        sst_mean: Optional[float] = None,
        sst_std: Optional[float] = None,
        inversion_threshold_c: float = 0.4,
        gradient_scale_c_per_100m: float = 100.0,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_grad = lambda_grad
        self.lambda_surf = lambda_surf
        self.lambda_strat = lambda_strat

        self.mse_loss = MaskedMSELoss()
        self.grad_loss = VerticalGradientLoss(depths=depths, scale_per_m=gradient_scale_c_per_100m)
        self.surf_loss = SurfaceConsistencyLoss()
        self.strat_loss = StratificationPenaltyLoss(inversion_threshold_c)

        if target_mean is not None and target_std is not None:
            self.register_buffer("target_mean", torch.as_tensor(target_mean, dtype=torch.float32))
            self.register_buffer("target_std", torch.as_tensor(target_std, dtype=torch.float32))
        else:
            self.target_mean = None
            self.target_std = None

        if sst_mean is not None and sst_std is not None:
            self.register_buffer("sst_mean", torch.as_tensor(float(sst_mean), dtype=torch.float32))
            self.register_buffer("sst_std", torch.as_tensor(float(sst_std), dtype=torch.float32))
        else:
            self.sst_mean = None
            self.sst_std = None

    def _to_celsius(self, x):
        if self.target_mean is None:
            return x
        return x * self.target_std + self.target_mean

    def _sst_to_celsius(self, x):
        if self.sst_mean is None:
            return x
        return x * self.sst_std + self.sst_mean

    def forward(
        self,
        pred,
        target,
        mask,
        sst_input=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_mse = self.mse_loss(pred, target, mask)
        pred_c = self._to_celsius(pred)
        target_c = self._to_celsius(target)

        l_grad = self.grad_loss(pred_c, target_c, mask)
        l_strat = self.strat_loss(pred_c, mask)
        total = (
            self.lambda_mse * l_mse
            + self.lambda_grad * l_grad
            + self.lambda_strat * l_strat
        )
        out = {
            "loss_mse": float(l_mse.detach()),
            "loss_grad": float(l_grad.detach()),
            "loss_strat": float(l_strat.detach()),
        }

        if sst_input is not None and self.lambda_surf > 0:
            sst_c = self._sst_to_celsius(sst_input)
            l_surf = self.surf_loss(pred_c[:, 0], sst_c, mask)
            total = total + self.lambda_surf * l_surf
            out["loss_surf"] = float(l_surf.detach())

        out["loss_total"] = float(total.detach())
        return total, out

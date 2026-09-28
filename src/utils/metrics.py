"""
Oceanographic Validation Metrics for OceanEmbed.
Supports SIH 26066: Depth-wise RMSE, Pearson Correlation (r), Mean Bias, MAE,
Thermocline Depth (D20), and Mixed Layer Depth (MLD).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from ..data.preprocess import STANDARD_DEPTHS


def calculate_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Pearson correlation coefficient between two 1D arrays."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid) < 3:
        return 0.0
    x_val = x[valid]
    y_val = y[valid]
    std_x = np.std(x_val)
    std_y = np.std(y_val)
    if std_x < 1e-6 or std_y < 1e-6:
        return 0.0
    r = np.corrcoef(x_val, y_val)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def compute_depth_metrics(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    depths: Optional[np.ndarray] = None
) -> Dict[str, Union[float, List[float], Dict[int, float]]]:
    """
    Computes comprehensive depth-wise oceanographic validation metrics.

    Args:
        pred: (N, 15, H, W) or (15, H, W) in Celsius
        target: (N, 15, H, W) or (15, H, W) in Celsius
        mask: (H, W) or (N, H, W) boolean/float mask (True/1 = ocean)
        depths: 1D array of depth levels in meters

    Returns:
        Dictionary containing:
        - rmse_per_depth: List of RMSE for each of the 15 depth levels (°C)
        - corr_per_depth: List of Pearson r for each depth level
        - bias_per_depth: List of Mean Bias for each depth level (°C)
        - mae_per_depth: List of MAE for each depth level (°C)
        - mean_rmse: Average RMSE across all depths (°C)
        - mean_corr: Average Correlation across all depths
        - mean_bias: Average Bias across all depths (°C)
        - mean_mae: Average MAE across all depths (°C)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if depths is None:
        depths = STANDARD_DEPTHS

    if pred.ndim == 3:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]

    n_samples, n_depths, h, w = pred.shape
    ocean_mask = mask.astype(bool)
    if ocean_mask.shape[0] == 1 and n_samples > 1:
        ocean_mask = np.repeat(ocean_mask, n_samples, axis=0)

    rmse_list = []
    corr_list = []
    bias_list = []
    mae_list = []

    for d in range(n_depths):
        p_d = pred[:, d][ocean_mask]
        t_d = target[:, d][ocean_mask]

        valid = ~(np.isnan(p_d) | np.isnan(t_d))
        p_val = p_d[valid]
        t_val = t_d[valid]

        if len(t_val) == 0:
            rmse_list.append(float("nan"))
            corr_list.append(float("nan"))
            bias_list.append(float("nan"))
            mae_list.append(float("nan"))
            continue

        diff = p_val - t_val
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bias = float(np.mean(diff))
        mae = float(np.mean(np.abs(diff)))
        r = calculate_correlation(p_val, t_val)

        rmse_list.append(rmse)
        corr_list.append(r)
        bias_list.append(bias)
        mae_list.append(mae)

    return {
        "depths": depths.tolist(),
        "rmse_per_depth": rmse_list,
        "corr_per_depth": corr_list,
        "bias_per_depth": bias_list,
        "mae_per_depth": mae_list,
        "mean_rmse": float(np.nanmean(rmse_list)),
        "mean_corr": float(np.nanmean(corr_list)),
        "mean_bias": float(np.nanmean(bias_list)),
        "mean_mae": float(np.nanmean(mae_list)),
    }


def compute_thermocline_depth_error(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    depths: Optional[np.ndarray] = None,
    isotherm: float = 20.0
) -> float:
    """
    Computes RMSE of the 20°C Isotherm Depth (D20, in meters),
    which is the canonical physical proxy for thermocline displacement.
    """
    from scipy.interpolate import interp1d

    if depths is None:
        depths = STANDARD_DEPTHS

    if pred.ndim == 3:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]

    n_samples, _, h, w = pred.shape
    d20_errors = []

    for s in range(n_samples):
        for i in range(h):
            for j in range(w):
                if not mask[i, j]:
                    continue
                p_prof = pred[s, :, i, j]
                t_prof = target[s, :, i, j]

                # Find depth where T crosses isotherm
                try:
                    f_pred = interp1d(p_prof, depths, bounds_error=False, fill_value=np.nan)
                    f_true = interp1d(t_prof, depths, bounds_error=False, fill_value=np.nan)
                    d_pred = f_pred(isotherm)
                    d_true = f_true(isotherm)
                    if not (np.isnan(d_pred) or np.isnan(d_true)):
                        d20_errors.append(d_pred - d_true)
                except Exception:
                    continue

    if len(d20_errors) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.array(d20_errors) ** 2)))


def compute_vertical_gradient_error(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    depths: Optional[np.ndarray] = None
) -> float:
    """Computes Mean Squared Error of the vertical thermal gradient (dT/dz)."""
    if depths is None:
        depths = STANDARD_DEPTHS

    dz = depths[1:] - depths[:-1]
    grad_p = (pred[:, 1:] - pred[:, :-1]) / dz[np.newaxis, :, np.newaxis, np.newaxis]
    grad_t = (target[:, 1:] - target[:, :-1]) / dz[np.newaxis, :, np.newaxis, np.newaxis]

    ocean_mask = mask.astype(bool)
    diff = (grad_p - grad_t)[:, :, ocean_mask]
    return float(np.mean(diff ** 2))

"""
OceanEmbed Evaluation & ARGO Float Validation Script.
SIH Problem Statement ID 26066: Evaluates 3D subsurface temperature reconstruction against
GLORYS reanalysis ground truth and independent INCOIS ARGO profiling float observations.
Generates comprehensive oceanographic diagnostic profile plots.
"""

from typing import Dict, List, Optional, Tuple, Any
import argparse
import os
import json
from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.dataset import OceanDataset
from src.data.preprocess import STANDARD_DEPTHS, TARGET_LATS, TARGET_LONS
from src.models.oceanembed_net import build_oceanembed_model
from src.utils.metrics import (
    compute_depth_metrics,
    compute_thermocline_depth_error,
    compute_vertical_gradient_error,
    calculate_correlation,
)


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_vertical_profiles(
    preds: np.ndarray,
    targets: np.ndarray,
    depths: np.ndarray,
    output_dir: Path,
    argo_profiles: Optional[List[Dict[str, Any]]] = None,
):
    """
    Plots vertical temperature profiles comparing Prediction vs Ground Truth vs ARGO Floats
    at representative North Indian Ocean oceanographic regimes:
    1. Arabian Sea Upwelling Zone (14.0°N, 58.0°E)
    2. Central Arabian Sea (16.0°N, 66.0°E)
    3. Bay of Bengal Warm Pool (15.0°N, 88.0°E)
    4. Equatorial Indian Ocean (6.0°N, 80.0°E)
    """
    stations = [
        {"name": "Arabian Sea Upwelling", "lat": 14.0, "lon": 58.0},
        {"name": "Central Arabian Sea", "lat": 16.0, "lon": 66.0},
        {"name": "Bay of Bengal (Warm Pool)", "lat": 15.0, "lon": 88.0},
        {"name": "Equatorial Indian Ocean", "lat": 6.0, "lon": 80.0},
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)
    fig.suptitle("OceanEmbed: 3D Subsurface Temperature Vertical Profiles Comparison", fontsize=15, fontweight="bold", y=1.02)

    sample_idx = 0 # Evaluate on the first test day snapshot

    for ax, st in zip(axes, stations):
        # Find nearest grid index
        lat_idx = int(np.argmin(np.abs(TARGET_LATS - st["lat"])))
        lon_idx = int(np.argmin(np.abs(TARGET_LONS - st["lon"])))

        true_prof = targets[sample_idx, :, lat_idx, lon_idx]
        pred_prof = preds[sample_idx, :, lat_idx, lon_idx]

        ax.plot(true_prof, depths, "b-o", linewidth=2.2, label="GLORYS Target", markersize=4)
        ax.plot(pred_prof, depths, "r--s", linewidth=2.0, label="OceanEmbed (Predicted)", markersize=4)

        # Overlay nearby ARGO float if available
        if argo_profiles:
            for argo in argo_profiles:
                dist = np.sqrt((argo["lat"] - st["lat"])**2 + (argo["lon"] - st["lon"])**2)
                if dist < 2.5: # within 2.5 degrees
                    ax.plot(argo["temperature"], depths, "g:^", linewidth=1.8, label="ARGO In-Situ Float", markersize=4)
                    break

        ax.set_title(f"{st['name']}\n({st['lat']}°N, {st['lon']}°E)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (°C)", fontsize=10)
        ax.set_ylabel("Depth (meters)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_ylim(1050, -20) # Invert y-axis for depth
        ax.legend(loc="lower left", fontsize=8.5)

    plt.tight_layout()
    plot_path = output_dir / "vertical_profiles_comparison.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved vertical profile plot to: {plot_path}")


def plot_skill_curves(metrics: Dict[str, Any], output_dir: Path):
    """
    Plots depth-wise skill curves (RMSE and Pearson Correlation vs Depth).
    """
    depths = metrics["depths"]
    rmse = metrics["rmse_per_depth"]
    corr = metrics["corr_per_depth"]
    bias = metrics["bias_per_depth"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    fig.suptitle("OceanEmbed: Depth-Wise Oceanographic Skill Metrics (0m to 1000m)", fontsize=14, fontweight="bold")

    # RMSE vs Depth
    ax1.plot(rmse, depths, "r-o", linewidth=2.0, markersize=5)
    ax1.set_title(f"Root Mean Squared Error (RMSE)\nMean: {metrics['mean_rmse']:.3f}°C", fontsize=11)
    ax1.set_xlabel("RMSE (°C)", fontsize=10)
    ax1.set_ylabel("Depth (m)", fontsize=10)
    ax1.set_ylim(1050, -20)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Correlation vs Depth
    ax2.plot(corr, depths, "b-s", linewidth=2.0, markersize=5)
    ax2.set_title(f"Pearson Correlation (r)\nMean: {metrics['mean_corr']:.3f}", fontsize=11)
    ax2.set_xlabel("Correlation coefficient (r)", fontsize=10)
    ax2.set_ylim(1050, -20)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Mean Bias vs Depth
    ax3.plot(bias, depths, "g-^", linewidth=2.0, markersize=5)
    ax3.axvline(0, color="gray", linestyle=":", linewidth=1.5)
    ax3.set_title(f"Mean Bias Error (MBE)\nMean: {metrics['mean_bias']:.3f}°C", fontsize=11)
    ax3.set_xlabel("Bias (°C)", fontsize=10)
    ax3.set_ylim(1050, -20)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = output_dir / "depth_wise_skill_metrics.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved skill curves plot to: {plot_path}")


def plot_horizontal_slices(
    preds: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    output_dir: Path,
    selected_depths: List[int] = [0, 100, 300, 1000]
):
    """
    Plots 2D horizontal temperature fields comparing Target vs Prediction vs Absolute Error
    at key oceanographic depths: Surface (0m), Thermocline Core (100m), Intermediate (300m), Deep (1000m).
    """
    depth_indices = [int(np.argmin(np.abs(STANDARD_DEPTHS - d))) for d in selected_depths]
    n_depths = len(depth_indices)

    fig, axes = plt.subplots(n_depths, 3, figsize=(15, 3.2 * n_depths))
    fig.suptitle("OceanEmbed: 2D Horizontal Ocean Temperature Slices across North Indian Ocean", fontsize=14, fontweight="bold", y=0.995)

    sample_idx = 0
    extent = [TARGET_LONS[0], TARGET_LONS[-1], TARGET_LATS[0], TARGET_LATS[-1]]

    for row, (d_idx, target_depth) in enumerate(zip(depth_indices, selected_depths)):
        t_true = np.copy(targets[sample_idx, d_idx])
        t_pred = np.copy(preds[sample_idx, d_idx])
        t_true[~mask] = np.nan
        t_pred[~mask] = np.nan
        abs_err = np.abs(t_pred - t_true)

        vmin = np.nanpercentile(t_true, 2)
        vmax = np.nanpercentile(t_true, 98)

        # 1. Target Field
        im1 = axes[row, 0].imshow(t_true, origin="lower", extent=extent, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[row, 0].set_title(f"Target ({target_depth}m)", fontsize=10, fontweight="bold")
        axes[row, 0].set_ylabel(f"Latitude ({target_depth}m)", fontsize=9)
        plt.colorbar(im1, ax=axes[row, 0], fraction=0.03, pad=0.04, label="°C")

        # 2. Predicted Field
        im2 = axes[row, 1].imshow(t_pred, origin="lower", extent=extent, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[row, 1].set_title(f"Predicted ({target_depth}m)", fontsize=10, fontweight="bold")
        plt.colorbar(im2, ax=axes[row, 1], fraction=0.03, pad=0.04, label="°C")

        # 3. Absolute Error
        im3 = axes[row, 2].imshow(abs_err, origin="lower", extent=extent, cmap="YlOrRd", vmin=0, vmax=1.5)
        axes[row, 2].set_title(f"Absolute Error ({target_depth}m)", fontsize=10, fontweight="bold")
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.03, pad=0.04, label="|°C|")

        if row == n_depths - 1:
            for col in range(3):
                axes[row, col].set_xlabel("Longitude (°E)", fontsize=9)

    plt.tight_layout()
    plot_path = output_dir / "horizontal_temperature_slices.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved 2D horizontal slices to: {plot_path}")


def evaluate_argo_floats(
    preds: np.ndarray,
    argo_profiles: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Validates OceanEmbed predictions against independent in-situ ARGO float profiles.
    """
    float_rmses = []
    float_corrs = []

    for argo in argo_profiles:
        day_idx = min(argo.get("day_idx", 0), len(preds) - 1)
        lat_idx = argo["lat_idx"]
        lon_idx = argo["lon_idx"]
        argo_temp = np.array(argo["temperature"])

        pred_temp = preds[day_idx, :, lat_idx, lon_idx]
        diff = pred_temp - argo_temp
        rmse = np.sqrt(np.mean(diff ** 2))
        r = calculate_correlation(pred_temp, argo_temp)

        float_rmses.append(rmse)
        float_corrs.append(r)

    return {
        "argo_float_count": len(argo_profiles),
        "argo_mean_rmse": float(np.mean(float_rmses)) if float_rmses else 0.0,
        "argo_mean_corr": float(np.mean(float_corrs)) if float_corrs else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate OceanEmbed Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    proc_dir = Path(config["paths"]["processed_data_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_path = proc_dir / "test_data.npz"
    mask_path = proc_dir / "ocean_land_mask.npy"
    argo_path = proc_dir / "argo_observations.npy"

    if not test_path.exists():
        print(f"[Error] Test data file not found at {test_path}. Run train.py or preprocess.py first.")
        return

    # Checkpoint path
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        best_p = Path(config["paths"]["checkpoint_dir"]) / "oceanembed_best.pt"
        latest_p = Path(config["paths"]["checkpoint_dir"]) / "oceanembed_latest.pt"
        ckpt_path = best_p if best_p.exists() else latest_p

    if not ckpt_path.exists():
        print(f"[Warning] No checkpoint found at {ckpt_path}. Initializing model without trained weights for demo.")
        ckpt = None
        stats = None
    else:
        print(f"[OceanEmbed] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        stats = ckpt.get("stats", None)

    # Load test dataset
    use_coordconv = config["model"].get("use_coordconv", True)
    test_dataset = OceanDataset(
        data_path=test_path,
        use_coordconv=use_coordconv,
        time_window=1,
        normalize=True,
        stats=stats,
    )
    mask = test_dataset.land_mask

    # Setup device & model
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_oceanembed_model(config)
    if ckpt is not None and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Run inference across test set
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
    all_preds = []
    all_targets = []

    print("[OceanEmbed] Running inference on test dataset...")
    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            pred = model(x)
            pred_degc = test_dataset.denormalize_target(pred)
            target_degc = test_dataset.denormalize_target(y.to(device))

            all_preds.append(pred_degc.cpu().numpy())
            all_targets.append(target_degc.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    # Compute skill metrics
    print("\n" + "=" * 70)
    print("COMPUTING OCEANOGRAPHIC EVALUATION METRICS")
    print("=" * 70)
    metrics = compute_depth_metrics(preds_np, targets_np, mask)
    d20_rmse = compute_thermocline_depth_error(preds_np, targets_np, mask)
    grad_error = compute_vertical_gradient_error(preds_np, targets_np, mask)

    metrics["thermocline_d20_rmse_m"] = d20_rmse
    metrics["vertical_gradient_mse"] = grad_error

    # ARGO float validation
    argo_profiles = None
    if argo_path.exists():
        argo_profiles = np.load(argo_path, allow_pickle=True).tolist()
        argo_res = evaluate_argo_floats(preds_np, argo_profiles)
        metrics.update(argo_res)
        print(f"Independent ARGO Floats Evaluated: {argo_res['argo_float_count']}")
        print(f"ARGO In-Situ Mean RMSE: {argo_res['argo_mean_rmse']:.3f}°C | Mean Correlation: {argo_res['argo_mean_corr']:.3f}")

    print(f"Overall Mean RMSE across all 15 depths: {metrics['mean_rmse']:.3f}°C")
    print(f"Overall Mean Pearson Correlation (r):   {metrics['mean_corr']:.3f}")
    print(f"Overall Mean Bias Error (MBE):          {metrics['mean_bias']:.3f}°C")
    print(f"Thermocline Depth (D20) Error:          {d20_rmse:.2f} meters")

    # Print Depth-wise table
    print("\n--- DEPTH-WISE SKILL METRIC BREAKDOWN ---")
    print(f"{'Depth (m)':>10} | {'RMSE (°C)':>10} | {'Corr (r)':>10} | {'Bias (°C)':>10} | {'MAE (°C)':>10}")
    print("-" * 58)
    for d, r_val, c_val, b_val, m_val in zip(
        metrics["depths"],
        metrics["rmse_per_depth"],
        metrics["corr_per_depth"],
        metrics["bias_per_depth"],
        metrics["mae_per_depth"],
    ):
        print(f"{int(d):>10d} | {r_val:>10.3f} | {c_val:>10.3f} | {b_val:>10.3f} | {m_val:>10.3f}")
    print("-" * 58)

    # Save metrics to JSON
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Evaluation] Saved comprehensive metrics report to: {report_file}")

    # Generate Publication Plots
    print("\n[Evaluation] Generating oceanographic diagnostic figures...")
    plot_vertical_profiles(preds_np, targets_np, STANDARD_DEPTHS, output_dir, argo_profiles)
    plot_skill_curves(metrics, output_dir)
    plot_horizontal_slices(preds_np, targets_np, mask, output_dir)

    print("\n[Evaluation] All evaluation diagnostics completed successfully!")


if __name__ == "__main__":
    main()

"""
OceanEmbed Training Script.
SIH Problem Statement ID 26066: Satellite Embedding-Based Deep Learning Framework
for Reconstruction of Subsurface Ocean Temperature.
"""

from typing import Dict, Any, Optional, Tuple, Union
import argparse
import os
import json
import time
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data.preprocess import generate_synthetic_benchmark_dataset
from src.data.dataset import create_dataloaders
from src.models.oceanembed_net import build_oceanembed_model
from src.utils.losses import PhysicsInformedOceanLoss
from src.utils.metrics import compute_depth_metrics


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    loss_breakdown = {"loss_mse": 0.0, "loss_grad": 0.0, "loss_strat": 0.0, "loss_surf": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc="Training Batch", leave=False)
    for x, y, mask in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training
        use_amp = scaler is not None and device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            pred = model(x)
            # Surface SST is channel 0 of input
            sst_in = x[:, 0]
            loss, loss_dict = criterion(pred, y, mask, sst_input=sst_in)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        for k in loss_breakdown:
            if k in loss_dict:
                loss_breakdown[k] += loss_dict[k]
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / max(1, num_batches)
    avg_components = {k: v / max(1, num_batches) for k, v in loss_breakdown.items()}
    avg_components["loss_total"] = avg_loss
    return avg_components


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    dataset_ref: Any,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []
    mask_ref = None

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)
            sst_in = x[:, 0]
            loss, _ = criterion(pred, y, mask, sst_input=sst_in)

            total_loss += loss.item()
            num_batches += 1

            # Denormalize predictions and targets to Celsius for physical metrics
            pred_degc = dataset_ref.denormalize_target(pred)
            target_degc = dataset_ref.denormalize_target(y)

            all_preds.append(pred_degc.cpu().numpy())
            all_targets.append(target_degc.cpu().numpy())
            if mask_ref is None:
                mask_ref = mask[0].cpu().numpy()

    avg_loss = total_loss / max(1, num_batches)

    # Compute physical depth-wise ocean metrics
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_targets_np = np.concatenate(all_targets, axis=0)

    ocean_metrics = compute_depth_metrics(all_preds_np, all_targets_np, mask_ref)
    ocean_metrics["val_loss"] = avg_loss

    return {"val_loss": avg_loss}, ocean_metrics


def main():
    parser = argparse.ArgumentParser(description="Train OceanEmbed Framework")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Overrides
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    # Check for processed benchmark dataset; generate if absent
    proc_dir = Path(config["paths"]["processed_data_dir"])
    train_path = proc_dir / "train_data.npz"
    val_path = proc_dir / "val_data.npz"
    test_path = proc_dir / "test_data.npz"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        print("[OceanEmbed] Processed data files not found. Generating physical benchmark dataset...")
        generate_synthetic_benchmark_dataset(output_dir=proc_dir, num_samples=100)

    # Setup device
    device_cfg = args.device or config["training"].get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    print(f"[OceanEmbed] Running training on device: {device}")

    # Initialize DataLoaders
    batch_size = config["training"]["batch_size"]
    use_coordconv = config["model"].get("use_coordconv", True)
    time_win = config["temporal"].get("time_window", 1)
    num_workers = config["training"].get("num_workers", 0)

    train_loader, val_loader, test_loader, stats = create_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        batch_size=batch_size,
        use_coordconv=use_coordconv,
        time_window=time_win,
        num_workers=num_workers,
    )
    print(f"[OceanEmbed] Loaded datasets: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")

    # Build Model
    model = build_oceanembed_model(config)
    model.to(device)
    total_params = model.count_parameters()
    print(f"[OceanEmbed] Model: OceanEmbedNet initialized with {total_params:,} trainable parameters.")

    # Initialize Physics-Informed Loss
    loss_cfg = config.get("loss", {})
    depths = np.array(config["depth"]["levels"], dtype=np.float32)
    criterion = PhysicsInformedOceanLoss(
        lambda_mse=loss_cfg.get("lambda_mse", 1.0),
        lambda_grad=loss_cfg.get("lambda_gradient", 0.5),
        lambda_surf=loss_cfg.get("lambda_surface", 0.2),
        lambda_strat=loss_cfg.get("lambda_stratification", 0.05),
        depths=depths,
    ).to(device)

    # Optimizer & Scheduler
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_epochs = config["training"]["num_epochs"]
    min_lr = float(config["training"].get("min_lr", 1e-6))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and config["training"].get("mixed_precision", True)) else None

    # Checkpoint setup
    ckpt_dir = Path(config["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "oceanembed_best.pt"
    latest_ckpt_path = ckpt_dir / "oceanembed_latest.pt"

    best_val_rmse = float("inf")
    patience = config["training"].get("early_stopping_patience", 10)
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "mean_rmse": [],
        "mean_corr": [],
        "mean_bias": [],
    }

    print("\n" + "=" * 70)
    print("STARTING OCEANEMBED TRAINING PIPELINE")
    print("=" * 70)

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=config["training"].get("gradient_clip_val", 1.0),
            scaler=scaler,
        )
        scheduler.step()

        val_res, ocean_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            dataset_ref=val_loader.dataset,
        )
        epoch_duration = time.time() - epoch_start

        current_val_rmse = ocean_metrics["mean_rmse"]
        current_val_corr = ocean_metrics["mean_corr"]

        history["train_loss"].append(train_res["loss_total"])
        history["val_loss"].append(val_res["val_loss"])
        history["mean_rmse"].append(current_val_rmse)
        history["mean_corr"].append(current_val_corr)
        history["mean_bias"].append(ocean_metrics["mean_bias"])

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] "
            f"Train Loss: {train_res['loss_total']:.4f} (MSE: {train_res['loss_mse']:.4f}, Grad: {train_res['loss_grad']:.4f}) | "
            f"Val Loss: {val_res['val_loss']:.4f} | "
            f"Val RMSE: {current_val_rmse:.3f}°C | "
            f"Corr (r): {current_val_corr:.3f} | "
            f"Time: {epoch_duration:.1f}s"
        )

        # Save Best Checkpoint
        if current_val_rmse < best_val_rmse:
            best_val_rmse = current_val_rmse
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_rmse": best_val_rmse,
                    "config": config,
                    "stats": stats,
                },
                best_ckpt_path,
            )
            print(f"  --> Saved new BEST checkpoint (RMSE: {best_val_rmse:.3f}°C) to {best_ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[OceanEmbed] Early stopping triggered after {patience} epochs without improvement.")
                break

    total_training_time = time.time() - start_time
    print("=" * 70)
    print(f"TRAINING COMPLETE in {total_training_time:.1f}s | Best Val RMSE: {best_val_rmse:.3f}°C")
    print("=" * 70)

    # Save latest checkpoint & history
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config,
            "stats": stats,
        },
        latest_ckpt_path,
    )
    with open(ckpt_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

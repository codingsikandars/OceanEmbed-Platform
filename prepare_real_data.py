"""
Prepare real OceanEmbed training arrays from local NetCDF/Zarr products.

Example:
    python prepare_real_data.py --manifest configs/real_data.yaml

The script expects the data to have already been downloaded. See
docs/DATA_ACCESS.md for Copernicus Marine Toolbox and PO.DAAC commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np

from src.data.real_data import SourceSpec, build_harmonized_arrays, save_harmonized_dataset


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_temporally(
    x: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    train_fraction: float,
    val_fraction: float,
):
    n = len(dates)
    i_train = max(1, int(n * train_fraction))
    i_val = max(i_train + 1, int(n * (train_fraction + val_fraction)))
    i_val = min(i_val, n - 1)
    return (
        (x[:i_train], y[:i_train], dates[:i_train]),
        (x[i_train:i_val], y[i_train:i_val], dates[i_train:i_val]),
        (x[i_val:], y[i_val:], dates[i_val:]),
    )


def save_split(path: Path, x, y, mask, dates):
    save_harmonized_dataset(path, x, y, mask, dates)
    print(f"[OceanEmbed] wrote {path} | X={x.shape}, y={y.shape}, dates={len(dates)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/real_data.yaml")
    args = parser.parse_args()

    cfg = load_manifest(args.manifest)
    surfaces = [SourceSpec(**item) for item in cfg["surface_sources"]]
    target = SourceSpec(**cfg["target_source"])
    x, y, mask, dates = build_harmonized_arrays(
        surfaces,
        target,
        start=cfg.get("start"),
        end=cfg.get("end"),
        require_complete_days=cfg.get("require_complete_days", False),
    )

    train, val, test = split_temporally(
        x, y, dates,
        cfg["split"]["train_fraction"],
        cfg["split"]["val_fraction"],
    )

    out = Path(cfg.get("output_dir", "data/processed/real"))
    out.mkdir(parents=True, exist_ok=True)
    save_split(out / "train_data.npz", *train[:2], mask, train[2])
    save_split(out / "val_data.npz", *val[:2], mask, val[2])
    save_split(out / "test_data.npz", *test[:2], mask, test[2])
    np.save(out / "ocean_land_mask.npy", mask)
    np.save(out / "dates.npy", dates)

    print("[OceanEmbed] real-data preparation complete.")
    print(f"Domain samples: {len(dates)} | ocean cells: {int(mask.sum())}/{mask.size}")


if __name__ == "__main__":
    main()

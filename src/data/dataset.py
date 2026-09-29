"""
PyTorch dataset for OceanEmbed.

The dataset supports:
- temporal sliding windows;
- shared training-only normalization statistics;
- optional CoordConv channels;
- real-data NaN fallback handling;
- explicit ocean masks and date metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocess import (
    TARGET_LATS,
    TARGET_LONS,
    create_north_indian_ocean_land_mask,
)


class OceanDataset(Dataset):
    """Dataset returning one sample as (channels, lat, lon), target and ocean mask."""

    def __init__(
        self,
        data_path: Union[str, Path],
        land_mask: Optional[np.ndarray] = None,
        use_coordconv: bool = True,
        time_window: int = 1,
        normalize: bool = True,
        stats: Optional[Dict[str, np.ndarray]] = None,
        fill_missing: bool = True,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.use_coordconv = use_coordconv
        self.time_window = max(1, int(time_window))
        self.normalize = normalize
        self.fill_missing = fill_missing

        data = np.load(self.data_path, allow_pickle=True)
        self.inputs = data["inputs"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)

        if self.inputs.ndim != 4 or self.targets.ndim != 4:
            raise ValueError(
                f"Expected inputs (T,C,H,W) and targets (T,D,H,W), got "
                f"{self.inputs.shape} and {self.targets.shape}"
            )
        if self.inputs.shape[1] != 7:
            raise ValueError("OceanEmbed expects exactly 7 surface input channels.")

        self.num_samples = len(self.inputs)
        self.n_lat = self.inputs.shape[2]
        self.n_lon = self.inputs.shape[3]
        self.num_depths = self.targets.shape[1]

        if "ocean_mask" in data:
            file_mask = data["ocean_mask"].astype(bool)
        else:
            file_mask = create_north_indian_ocean_land_mask(TARGET_LATS, TARGET_LONS)

        if land_mask is None:
            self.land_mask = file_mask
        else:
            self.land_mask = land_mask.astype(bool)

        if self.land_mask.shape != (self.n_lat, self.n_lon):
            raise ValueError(
                f"Mask shape {self.land_mask.shape} does not match data "
                f"{(self.n_lat, self.n_lon)}"
            )

        if "dates" in data:
            self.dates = data["dates"].astype("datetime64[D]")
        else:
            self.dates = np.arange(self.num_samples).astype("timedelta64[D]")

        lat_norm = np.linspace(-1.0, 1.0, self.n_lat, dtype=np.float32)
        lon_norm = np.linspace(-1.0, 1.0, self.n_lon, dtype=np.float32)
        lon_mesh, lat_mesh = np.meshgrid(lon_norm, lat_norm)
        self.coord_grid = np.stack([lat_mesh, lon_mesh], axis=0)

        # Resolve missing values before statistics are calculated. This is only a
        # fallback; provider-native quality flags should be used where available.
        if self.fill_missing:
            self._fill_missing_with_channel_mean()

        self.stats = stats if stats is not None else self._compute_statistics()

        if self.normalize:
            self._apply_normalization()

        self.mask_tensor = torch.from_numpy(self.land_mask.astype(np.float32))

    @property
    def input_channels(self) -> int:
        return 7 * self.time_window + (2 if self.use_coordconv else 0)

    @property
    def sst_channel_index(self) -> int:
        """Channel containing SST for the target/final day of a temporal window."""
        return 7 * (self.time_window - 1)

    def _fill_missing_with_channel_mean(self) -> None:
        """Fill remaining NaNs using training-file channel means inside ocean cells."""
        ocean = self.land_mask
        for c in range(self.inputs.shape[1]):
            vals = self.inputs[:, c][:, ocean]
            mean = np.nanmean(vals)
            if not np.isfinite(mean):
                mean = 0.0
            field = self.inputs[:, c]
            bad = ~np.isfinite(field)
            field[bad & ocean[None, ...]] = np.float32(mean)
            field[:, ~ocean] = 0.0

        for d in range(self.targets.shape[1]):
            vals = self.targets[:, d][:, ocean]
            mean = np.nanmean(vals)
            if not np.isfinite(mean):
                mean = 0.0
            field = self.targets[:, d]
            bad = ~np.isfinite(field)
            field[bad & ocean[None, ...]] = np.float32(mean)
            field[:, ~ocean] = 0.0

    def _compute_statistics(self) -> Dict[str, np.ndarray]:
        ocean_indices = np.where(self.land_mask)
        n_channels = self.inputs.shape[1]

        input_mean = np.zeros((n_channels, 1, 1), dtype=np.float32)
        input_std = np.zeros((n_channels, 1, 1), dtype=np.float32)
        for c in range(n_channels):
            vals = self.inputs[:, c, ocean_indices[0], ocean_indices[1]]
            input_mean[c, 0, 0] = np.nanmean(vals)
            input_std[c, 0, 0] = max(float(np.nanstd(vals)), 1e-4)

        target_mean = np.zeros((self.num_depths, 1, 1), dtype=np.float32)
        target_std = np.zeros((self.num_depths, 1, 1), dtype=np.float32)
        for d in range(self.num_depths):
            vals = self.targets[:, d, ocean_indices[0], ocean_indices[1]]
            target_mean[d, 0, 0] = np.nanmean(vals)
            target_std[d, 0, 0] = max(float(np.nanstd(vals)), 1e-4)

        return {
            "input_mean": input_mean,
            "input_std": input_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }

    def _apply_normalization(self):
        self.inputs = (self.inputs - self.stats["input_mean"]) / self.stats["input_std"]
        self.targets = (self.targets - self.stats["target_mean"]) / self.stats["target_std"]
        self.inputs[:, :, ~self.land_mask] = 0.0
        self.targets[:, :, ~self.land_mask] = 0.0

    def denormalize_target(self, target_tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.stats["target_mean"], device=target_tensor.device)
        std = torch.as_tensor(self.stats["target_std"], device=target_tensor.device)
        return target_tensor * std + mean

    def __len__(self) -> int:
        return max(0, self.num_samples - self.time_window + 1)

    def __getitem__(self, idx: int):
        end_idx = idx + self.time_window
        sample_inputs = self.inputs[idx:end_idx]  # T,C,H,W
        sample_target = self.targets[end_idx - 1]

        # Flatten temporal context into channels for a standard 2-D encoder.
        x_arr = sample_inputs.reshape(self.time_window * 7, self.n_lat, self.n_lon)
        if self.use_coordconv:
            x_arr = np.concatenate([x_arr, self.coord_grid], axis=0)

        return (
            torch.from_numpy(x_arr.copy()),
            torch.from_numpy(sample_target.copy()),
            self.mask_tensor,
        )


def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Union[str, Path],
    batch_size: int = 8,
    use_coordconv: bool = True,
    time_window: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    fill_missing: bool = True,
):
    train_dataset = OceanDataset(
        train_path, use_coordconv=use_coordconv, time_window=time_window,
        normalize=True, fill_missing=fill_missing
    )
    stats = train_dataset.stats

    val_dataset = OceanDataset(
        val_path, use_coordconv=use_coordconv, time_window=time_window,
        normalize=True, stats=stats, fill_missing=fill_missing
    )
    test_dataset = OceanDataset(
        test_path, use_coordconv=use_coordconv, time_window=time_window,
        normalize=True, stats=stats, fill_missing=fill_missing
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=len(train_dataset) > batch_size, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, stats

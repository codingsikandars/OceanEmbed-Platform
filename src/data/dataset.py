"""
PyTorch Dataset and DataLoader implementation for OceanEmbed.
Supports sliding windows, spatial coordinate injection (CoordConv),
and multi-channel normalization.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocess import (
    TARGET_LATS,
    TARGET_LONS,
    STANDARD_DEPTHS,
    create_north_indian_ocean_land_mask,
)


class OceanDataset(Dataset):
    """
    PyTorch Dataset for multi-modal surface satellite observations and 3D subsurface temperature.

    Inputs:
        7 multi-modal surface channels:
        [0: SST, 1: SSS, 2: SSH, 3: U_curr, 4: V_curr, 5: U_wind, 6: V_wind]
        Optional: + 2 coordinate channels (normalized Lat, Lon) -> 9 channels total.

    Targets:
        15 subsurface depth levels in meters:
        [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        land_mask: Optional[np.ndarray] = None,
        use_coordconv: bool = True,
        time_window: int = 1,
        normalize: bool = True,
        stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            data_path: Path to .npz file containing 'inputs' and 'targets'
            land_mask: 2D boolean array (n_lat, n_lon). If None, generated automatically.
            use_coordconv: If True, appends 2 normalized spatial coordinate channels (lat, lon).
            time_window: Number of consecutive daily observation steps (default: 1).
            normalize: If True, normalizes inputs and targets using mean and std.
            stats: Dictionary containing 'input_mean', 'input_std', 'target_mean', 'target_std'.
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.use_coordconv = use_coordconv
        self.time_window = max(1, time_window)
        self.normalize = normalize

        # Load numpy data
        data = np.load(self.data_path)
        self.inputs = data["inputs"].astype(np.float32)   # Shape: (T, 7, H, W)
        self.targets = data["targets"].astype(np.float32) # Shape: (T, 15, H, W)
        
        self.num_samples = len(self.inputs)
        self.n_lat = self.inputs.shape[2]
        self.n_lon = self.inputs.shape[3]
        self.num_depths = self.targets.shape[1]

        # Land mask (True = Ocean, False = Land)
        if land_mask is None:
            self.land_mask = create_north_indian_ocean_land_mask(TARGET_LATS, TARGET_LONS)
        else:
            self.land_mask = land_mask.astype(bool)

        # Precompute normalized spatial coordinate grids: range [-1.0, 1.0]
        lat_norm = np.linspace(-1.0, 1.0, self.n_lat, dtype=np.float32)
        lon_norm = np.linspace(-1.0, 1.0, self.n_lon, dtype=np.float32)
        lon_mesh, lat_mesh = np.meshgrid(lon_norm, lat_norm)
        self.coord_grid = np.stack([lat_mesh, lon_mesh], axis=0) # Shape: (2, H, W)

        # Compute or apply normalization stats
        if stats is not None:
            self.stats = stats
        else:
            self.stats = self._compute_statistics()

        if self.normalize:
            self._apply_normalization()

        # Land mask as float tensor
        self.mask_tensor = torch.from_numpy(self.land_mask.astype(np.float32))

    def _compute_statistics(self) -> Dict[str, np.ndarray]:
        """
        Computes channel-wise mean and std across valid ocean grid cells only.
        """
        # Shape: (C, 1, 1)
        ocean_indices = np.where(self.land_mask)
        
        # Inputs stats (7 channels)
        n_channels = self.inputs.shape[1]
        input_mean = np.zeros((n_channels, 1, 1), dtype=np.float32)
        input_std = np.zeros((n_channels, 1, 1), dtype=np.float32)
        
        for c in range(n_channels):
            vals = self.inputs[:, c, ocean_indices[0], ocean_indices[1]]
            m = np.nanmean(vals)
            s = np.nanstd(vals)
            input_mean[c, 0, 0] = m
            input_std[c, 0, 0] = max(s, 1e-4)

        # Target stats (15 depths)
        target_mean = np.zeros((self.num_depths, 1, 1), dtype=np.float32)
        target_std = np.zeros((self.num_depths, 1, 1), dtype=np.float32)
        
        for d in range(self.num_depths):
            vals = self.targets[:, d, ocean_indices[0], ocean_indices[1]]
            m = np.nanmean(vals)
            s = np.nanstd(vals)
            target_mean[d, 0, 0] = m
            target_std[d, 0, 0] = max(s, 1e-4)

        return {
            "input_mean": input_mean,
            "input_std": input_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }

    def _apply_normalization(self):
        """Standardizes data in-place and zeros out land cells."""
        # Normalize inputs
        self.inputs = (self.inputs - self.stats["input_mean"]) / self.stats["input_std"]
        # Normalize targets
        self.targets = (self.targets - self.stats["target_mean"]) / self.stats["target_std"]

        # Ensure land cells are zeroed
        for c in range(self.inputs.shape[1]):
            self.inputs[:, c, ~self.land_mask] = 0.0
        for d in range(self.targets.shape[1]):
            self.targets[:, d, ~self.land_mask] = 0.0

    def denormalize_target(self, target_tensor: torch.Tensor) -> torch.Tensor:
        """Denormalizes a predicted or target temperature tensor back to Celsius."""
        mean = torch.from_numpy(self.stats["target_mean"]).to(target_tensor.device)
        std = torch.from_numpy(self.stats["target_std"]).to(target_tensor.device)
        return target_tensor * std + mean

    def __len__(self) -> int:
        return self.num_samples - self.time_window + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x (torch.Tensor): Shape (C_in, H, W) or (T_win, C_in, H, W)
                              C_in is 7 (or 9 if CoordConv is enabled)
            y (torch.Tensor): Target 3D temperature shape (15, H, W)
            mask (torch.Tensor): Land/Ocean binary mask shape (H, W)
        """
        end_idx = idx + self.time_window
        sample_inputs = self.inputs[idx:end_idx] # (T_win, 7, H, W)
        sample_target = self.targets[end_idx - 1] # Target is the current/final day (15, H, W)

        if self.time_window == 1:
            x_arr = sample_inputs[0] # (7, H, W)
            if self.use_coordconv:
                # Concatenate 2 coordinate channels -> (9, H, W)
                x_arr = np.concatenate([x_arr, self.coord_grid], axis=0)
        else:
            # Multi-step window
            if self.use_coordconv:
                coords_expanded = np.repeat(self.coord_grid[np.newaxis, ...], self.time_window, axis=0)
                x_arr = np.concatenate([sample_inputs, coords_expanded], axis=1) # (T_win, 9, H, W)
            else:
                x_arr = sample_inputs

        x = torch.from_numpy(x_arr)
        y = torch.from_numpy(sample_target)
        mask = self.mask_tensor

        return x, y, mask


def create_dataloaders(
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Union[str, Path],
    batch_size: int = 8,
    use_coordconv: bool = True,
    time_window: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Factory function to initialize Train, Validation, and Test DataLoaders
    with shared normalization statistics computed exclusively from the training set.
    """
    train_dataset = OceanDataset(
        data_path=train_path,
        use_coordconv=use_coordconv,
        time_window=time_window,
        normalize=True,
    )
    
    # Use training statistics for val and test to prevent data leakage
    stats = train_dataset.stats

    val_dataset = OceanDataset(
        data_path=val_path,
        use_coordconv=use_coordconv,
        time_window=time_window,
        normalize=True,
        stats=stats,
    )

    test_dataset = OceanDataset(
        data_path=test_path,
        use_coordconv=use_coordconv,
        time_window=time_window,
        normalize=True,
        stats=stats,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_dataset) > batch_size else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, stats

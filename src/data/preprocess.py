"""
Data Preprocessing, Harmonization, Interpolation, and Land Masking Pipeline.
Supports SIH 26066: OceanEmbed.
"""

from typing import Dict, List, Optional, Tuple, Union
import os
import numpy as np
from pathlib import Path


# Standard Spatial Grid Configuration
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RESOLUTION = 0.25

# Standard Depth Levels (in meters)
STANDARD_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)

# Grid dimensions: 101 x 241
TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + 1e-5, RESOLUTION, dtype=np.float32)
TARGET_LONS = np.arange(LON_MIN, LON_MAX + 1e-5, RESOLUTION, dtype=np.float32)


def create_north_indian_ocean_land_mask(
    lats: np.ndarray = TARGET_LATS,
    lons: np.ndarray = TARGET_LONS
) -> np.ndarray:
    """
    Constructs a high-fidelity land/sea binary mask for the North Indian Ocean domain
    (5°N to 30°N, 45°E to 105°E).

    Returns:
        mask (np.ndarray): 2D boolean array of shape (len(lats), len(lons))
                          True = Ocean (valid water cell)
                          False = Land (masked cell)
    """
    n_lat = len(lats)
    n_lon = len(lons)
    
    # 2D coordinate mesh
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Start with all ocean (True)
    ocean_mask = np.ones((n_lat, n_lon), dtype=bool)

    # 1. Arabian Peninsula & Middle East
    # West of 60E and north of ~13N
    cond_arabia = (lon_grid < 55.0) & (lat_grid > 13.0)
    cond_arabia_northeast = (lon_grid >= 55.0) & (lon_grid < 60.0) & (lat_grid > (22.0 + (lon_grid - 55.0) * 0.4))
    ocean_mask[cond_arabia | cond_arabia_northeast] = False

    # 2. Iran / Pakistan / Makran Coast
    cond_iran_pak = (lon_grid >= 57.0) & (lon_grid < 68.5) & (lat_grid > 24.5)
    ocean_mask[cond_iran_pak] = False

    # 3. Indian Subcontinent Landmass
    # Gujarat peninsula and northwest India
    cond_gujarat_nw = (lon_grid >= 68.5) & (lon_grid < 74.0) & (lat_grid > (20.5 - (lon_grid - 68.5) * 0.2))
    ocean_mask[cond_gujarat_nw] = False

    # Western Ghats and West Coast of India down to Kanyakumari (~8.1°N, 77.5°E)
    # Southern tip taper
    cond_peninsula_west = (
        (lat_grid >= 8.1) & (lat_grid <= 22.0) &
        (lon_grid >= (77.5 - (lat_grid - 8.1) * 0.4)) &
        (lon_grid <= 88.5)
    )
    # Bay of Bengal eastern boundary of Indian landmass (Odisha, West Bengal)
    cond_peninsula_east = (
        (lat_grid >= 11.5) & (lat_grid <= 22.5) &
        (lon_grid >= 77.5) &
        (lon_grid <= (80.0 + (lat_grid - 11.5) * 0.85))
    )
    # North India & Ganges Basin (north of 22°N)
    cond_north_india = (lat_grid > 21.5) & (lon_grid >= 68.5) & (lon_grid <= 90.0)
    ocean_mask[cond_peninsula_west & cond_peninsula_east] = False
    ocean_mask[cond_north_india] = False

    # 4. Sri Lanka (Island: ~5.9°N to 9.8°N, 79.7°E to 81.9°E)
    cond_sri_lanka = (
        (lat_grid >= 5.9) & (lat_grid <= 9.8) &
        (lon_grid >= 79.7) & (lon_grid <= 81.9)
    )
    ocean_mask[cond_sri_lanka] = False

    # 5. Myanmar, Bangladesh, and Indochina / Malay Peninsula
    # Bangladesh head of Bay of Bengal
    cond_bangladesh = (lat_grid >= 21.8) & (lon_grid >= 89.0) & (lon_grid <= 93.0)
    ocean_mask[cond_bangladesh] = False

    # Myanmar & Southeast Asia Landmass
    cond_myanmar_north = (lat_grid >= 16.0) & (lon_grid >= 92.5)
    ocean_mask[cond_myanmar_north] = False

    # Myanmar south coast & Thailand / Malay peninsula
    cond_malay_peninsula = (
        (lat_grid >= 5.0) & (lat_grid < 16.0) &
        (lon_grid >= (98.5 + (lat_grid - 5.0) * 0.2))
    )
    ocean_mask[cond_malay_peninsula] = False

    return ocean_mask


class DataHarmonizer:
    """
    Harmonizes multi-source satellite and reanalysis observations into a standard
    0.25° x 0.25° grid and standardized depth levels for the North Indian Ocean.
    """

    def __init__(
        self,
        lats: np.ndarray = TARGET_LATS,
        lons: np.ndarray = TARGET_LONS,
        depths: np.ndarray = STANDARD_DEPTHS
    ):
        self.lats = lats
        self.lons = lons
        self.depths = depths
        self.land_mask = create_north_indian_ocean_land_mask(lats, lons)

    def interpolate_2d_surface(
        self,
        source_data: np.ndarray,
        source_lats: np.ndarray,
        source_lons: np.ndarray,
        method: str = "bilinear"
    ) -> np.ndarray:
        """
        Interpolates a 2D surface field (e.g. OSTIA SST 0.05° or SMAP SSS 0.125°)
        to the target 0.25° grid (101 x 241).
        """
        from scipy.interpolate import RegularGridInterpolator

        # Ensure source lats are monotonically increasing
        if source_lats[1] < source_lats[0]:
            source_lats = np.flip(source_lats)
            source_data = np.flip(source_data, axis=0)
            
        interp = RegularGridInterpolator(
            (source_lats, source_lons),
            source_data,
            method="linear" if method == "bilinear" else "nearest",
            bounds_error=False,
            fill_value=np.nan
        )
        
        lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
        points = np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=-1)
        regridded = interp(points).reshape(len(self.lats), len(self.lons))
        
        # Apply land mask
        regridded[~self.land_mask] = np.nan
        return regridded.astype(np.float32)

    def interpolate_vertical_profile(
        self,
        source_profiles: np.ndarray,
        source_depths: np.ndarray
    ) -> np.ndarray:
        """
        Interpolates 3D vertical temperature profiles to the 15 standard depth levels.
        
        Args:
            source_profiles (np.ndarray): Shape (n_src_depths, n_lat, n_lon)
            source_depths (np.ndarray): 1D array of source depths in meters
            
        Returns:
            interpolated (np.ndarray): Shape (15, n_lat, n_lon)
        """
        from scipy.interpolate import interp1d

        n_levels = len(self.depths)
        n_lat = source_profiles.shape[1]
        n_lon = source_profiles.shape[2]
        
        out = np.zeros((n_levels, n_lat, n_lon), dtype=np.float32)
        
        for i in range(n_lat):
            for j in range(n_lon):
                prof = source_profiles[:, i, j]
                if np.all(np.isnan(prof)):
                    out[:, i, j] = np.nan
                    continue
                # Fill missing or extrapolate using linear/nearest
                valid_idx = ~np.isnan(prof)
                if np.sum(valid_idx) < 2:
                    out[:, i, j] = np.nan
                    continue
                
                f = interp1d(
                    source_depths[valid_idx],
                    prof[valid_idx],
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                out[:, i, j] = f(self.depths)
                
        return out


def generate_synthetic_benchmark_dataset(
    output_dir: Union[str, Path] = "data/processed",
    num_samples: int = 120,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Generates a physically consistent synthetic benchmark dataset for the North Indian Ocean
    to enable complete out-of-the-box training, validation, and evaluation without requiring
    immediate multi-gigabyte downloads from external servers.

    Physics encoded:
    - Sea Surface Temperature (SST): Warm pool in BoB (~29-30°C), cooler Western Arabian Sea upwelling (~24-26°C).
    - Sea Surface Salinity (SSS): High salinity Arabian Sea (~36 PSU), low salinity BoB river plume (~31-33 PSU).
    - Sea Surface Height (SSH): Cyclonic and anticyclonic mesoscale eddy anomalies (±0.15m).
    - Surface Currents (U, V): Geostrophic balance derived from SSH gradients plus Ekman component.
    - Surface Winds (U, V): Summer Southwest monsoon regime (strong Findlater Jet across Arabian Sea).
    - Subsurface Temperature (15 depths):
        * Mixed layer depth: D_mld ~ 30-50m where dT/dz ~ 0
        * Thermocline (50-200m): Steep thermal gradient modulated by SSH (thermocline displacement)
        * Deep water (200-1000m): Asymptotically approaches ~5.0°C
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(random_seed)
    
    n_lat = len(TARGET_LATS)
    n_lon = len(TARGET_LONS)
    n_depths = len(STANDARD_DEPTHS)
    
    land_mask = create_north_indian_ocean_land_mask(TARGET_LATS, TARGET_LONS)
    lon_grid, lat_grid = np.meshgrid(TARGET_LONS, TARGET_LATS)
    
    # Pre-allocate arrays
    # 7 input channels: [sst, sss, ssh, u_curr, v_curr, u_wind, v_wind]
    inputs = np.zeros((num_samples, 7, n_lat, n_lon), dtype=np.float32)
    # Target 3D temperature across 15 depths
    targets = np.zeros((num_samples, n_depths, n_lat, n_lon), dtype=np.float32)
    
    for t in range(num_samples):
        # Time-varying phase for seasonal / planetary wave propagation
        day_phase = 2.0 * np.pi * (t / 365.0)
        eddy_phase = 2.0 * np.pi * (t / 45.0)
        
        # 1. SST: Latitudinal gradient + Arabian Sea upwelling + BoB warm pool
        sst_base = 27.5 - 0.12 * (lat_grid - 5.0) # Cooler to north
        sst_bob_warm_pool = 1.8 * np.exp(-((lon_grid - 88.0)**2 / 120.0 + (lat_grid - 12.0)**2 / 30.0))
        sst_somali_upwelling = -2.5 * np.exp(-((lon_grid - 52.0)**2 / 40.0 + (lat_grid - 12.0)**2 / 40.0))
        sst_seasonal = 1.2 * np.sin(day_phase)
        sst = sst_base + sst_bob_warm_pool + sst_somali_upwelling + sst_seasonal + 0.15 * np.random.randn(n_lat, n_lon)
        
        # 2. SSS: Arabian Sea high salinity vs BoB low salinity
        sss_base = 35.8 - 0.05 * (lon_grid - 45.0) # Fresher towards east
        sss_bob_plume = -3.8 * np.exp(-((lon_grid - 90.0)**2 / 90.0 + (lat_grid - 18.0)**2 / 25.0))
        sss = sss_base + sss_bob_plume + 0.1 * np.random.randn(n_lat, n_lon)
        
        # 3. SSH / SLA: Mesoscale eddies + Rossby waves
        ssh_eddy1 = 0.14 * np.sin(0.18 * lon_grid + eddy_phase) * np.cos(0.22 * lat_grid)
        ssh_eddy2 = -0.10 * np.cos(0.35 * lon_grid - eddy_phase) * np.sin(0.3 * lat_grid)
        ssh = ssh_eddy1 + ssh_eddy2 + 0.02 * np.random.randn(n_lat, n_lon)
        
        # 4 & 5. Surface Currents (U, V): Geostrophic approximation from SSH gradients
        # u_g = -(g/f) * d(ssh)/dy, v_g = (g/f) * d(ssh)/dx
        d_ssh_dy, d_ssh_dx = np.gradient(ssh, RESOLUTION * 111e3, RESOLUTION * 111e3)
        # Coriolis f = 2*omega*sin(phi) (clipped near 5N to prevent division by zero)
        f_coriolis = 2.0 * 7.2921e-5 * np.sin(np.maximum(lat_grid, 5.0) * np.pi / 180.0)
        u_curr = - (9.81 / f_coriolis) * d_ssh_dy * 0.08
        v_curr = (9.81 / f_coriolis) * d_ssh_dx * 0.08
        # Clamp currents to realistic ocean range [-1.5, 1.5] m/s
        u_curr = np.clip(u_curr + 0.05 * np.random.randn(n_lat, n_lon), -1.5, 1.5)
        v_curr = np.clip(v_curr + 0.05 * np.random.randn(n_lat, n_lon), -1.5, 1.5)
        
        # 6 & 7. Surface 10m Winds (U, V): Southwest Monsoon Findlater Jet
        wind_jet = 9.0 * np.exp(-((lon_grid - 62.0)**2 / 180.0 + (lat_grid - 14.0)**2 / 50.0))
        u_wind = 2.0 + wind_jet + 1.2 * np.cos(day_phase) + 0.4 * np.random.randn(n_lat, n_lon)
        v_wind = 1.0 + 0.6 * wind_jet + 0.8 * np.sin(day_phase) + 0.4 * np.random.randn(n_lat, n_lon)
        
        # Store surface inputs
        inputs[t, 0] = sst
        inputs[t, 1] = sss
        inputs[t, 2] = ssh
        inputs[t, 3] = u_curr
        inputs[t, 4] = v_curr
        inputs[t, 5] = u_wind
        inputs[t, 6] = v_wind
        
        # Mask inputs with land mask (fill land with 0.0 for clean tensor handling)
        for c in range(7):
            inputs[t, c][~land_mask] = 0.0
            
        # 3D Subsurface Temperature Profile (15 standard depths)
        # Thermocline depth D20 (20°C isotherm depth) dynamically modulated by SSH:
        # High SSH -> downwelling / deeper thermocline (e.g. 110m)
        # Low SSH -> upwelling / shallower thermocline (e.g. 60m)
        d20_depth = 95.0 + 180.0 * ssh # meters
        d20_depth = np.clip(d20_depth, 45.0, 160.0)
        
        for k, depth_m in enumerate(STANDARD_DEPTHS):
            if depth_m <= 30.0:
                # Mixed Layer: nearly isothermal with SST
                temp_k = sst - 0.015 * depth_m
            else:
                # Thermocline and deep ocean: sigmoid / hyperbolic profile
                # Asymptotic deep ocean temperature ~ 4.8°C at 1000m
                thermocline_factor = 1.0 / (1.0 + np.exp((depth_m - d20_depth) / 38.0))
                temp_deep = 4.8 + 2.2 * np.exp(- (depth_m - 200.0) / 400.0)
                temp_k = temp_deep + (sst - temp_deep) * thermocline_factor
            
            temp_k[~land_mask] = 0.0
            targets[t, k] = temp_k

    # Split into Train (70%), Val (15%), Test (15%)
    n_train = int(num_samples * 0.70)
    n_val = int(num_samples * 0.15)
    
    train_inputs, train_targets = inputs[:n_train], targets[:n_train]
    val_inputs, val_targets = inputs[n_train:n_train + n_val], targets[n_train:n_train + n_val]
    test_inputs, test_targets = inputs[n_train + n_val:], targets[n_train + n_val:]
    
    # Also generate independent synthetic ARGO float observations for validation
    # Real Argo floats drift freely: we sample 25 random profile locations per test day
    n_test_days = len(test_inputs)
    argo_profiles = []
    
    # Valid ocean indices
    ocean_y, ocean_x = np.where(land_mask)
    for day_idx in range(n_test_days):
        # Sample 20 random ocean points
        sampled_pts = np.random.choice(len(ocean_y), size=20, replace=False)
        for pt in sampled_pts:
            lat_idx = ocean_y[pt]
            lon_idx = ocean_x[pt]
            true_prof = test_targets[day_idx, :, lat_idx, lon_idx]
            # Add small sensor measurement noise (+/- 0.05°C typical for CTD sensors)
            argo_temp = true_prof + np.random.normal(0, 0.05, size=n_depths)
            
            argo_profiles.append({
                "day_idx": day_idx,
                "lat": float(TARGET_LATS[lat_idx]),
                "lon": float(TARGET_LONS[lon_idx]),
                "lat_idx": int(lat_idx),
                "lon_idx": int(lon_idx),
                "depths": STANDARD_DEPTHS.tolist(),
                "temperature": argo_temp.astype(np.float32).tolist()
            })
            
    # Save datasets
    train_file = output_path / "train_data.npz"
    val_file = output_path / "val_data.npz"
    test_file = output_path / "test_data.npz"
    argo_file = output_path / "argo_observations.npy"
    mask_file = output_path / "ocean_land_mask.npy"
    grid_file = output_path / "grid_metadata.npz"
    
    np.savez_compressed(train_file, inputs=train_inputs, targets=train_targets)
    np.savez_compressed(val_file, inputs=val_inputs, targets=val_targets)
    np.savez_compressed(test_file, inputs=test_inputs, targets=test_targets)
    np.save(argo_file, argo_profiles)
    np.save(mask_file, land_mask)
    np.savez_compressed(grid_file, lats=TARGET_LATS, lons=TARGET_LONS, depths=STANDARD_DEPTHS)
    
    print(f"[OceanEmbed Preprocessing] Successfully generated benchmark dataset at {output_path}")
    print(f"  - Train samples: {len(train_inputs)}")
    print(f"  - Val samples:   {len(val_inputs)}")
    print(f"  - Test samples:  {len(test_inputs)}")
    print(f"  - ARGO profiles: {len(argo_profiles)}")
    print(f"  - Grid: {n_lat} lats x {n_lon} lons, 15 depths, 7 input channels")
    
    return {
        "train": str(train_file),
        "val": str(val_file),
        "test": str(test_file),
        "argo": str(argo_file),
        "mask": str(mask_file),
        "grid": str(grid_file)
    }


if __name__ == "__main__":
    generate_synthetic_benchmark_dataset()

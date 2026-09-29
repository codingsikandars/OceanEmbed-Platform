"""
Real-data harmonization utilities for OceanEmbed (SIH PS 26066).

The pipeline is intentionally provider-agnostic at the NetCDF layer:
- Copernicus Marine products are accessed/downloaded with the official
  Copernicus Marine Toolbox, then opened here with xarray.
- NASA/PO.DAAC products can be downloaded with podaac-data-subscriber and
  then opened here with xarray.
- Every source is clipped to the North Indian Ocean, converted to daily
  snapshots, and interpolated to the canonical 0.25-degree grid.

Expected output:
    inputs  -> (time, 7, lat, lon)
    targets -> (time, 15, lat, lon)
    ocean_mask -> (lat, lon)
    dates -> datetime64[D]

This module does not silently download data. That separation makes experiments
reproducible and prevents accidental multi-terabyte downloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from .preprocess import (
    STANDARD_DEPTHS,
    TARGET_LATS,
    TARGET_LONS,
)


CANONICAL_INPUTS = (
    "sst", "sss", "ssh", "u_curr", "v_curr", "u_wind", "v_wind"
)


@dataclass(frozen=True)
class SourceSpec:
    """Description of one local NetCDF/Zarr source."""

    name: str
    path: str
    variable: str
    kind: str = "surface"  # surface | target
    time_aggregation: str = "mean"  # mean | nearest
    lon_convention: str = "auto"


COORD_ALIASES = {
    "lat": ("latitude", "lat", "nav_lat", "y"),
    "lon": ("longitude", "lon", "nav_lon", "x"),
    "time": ("time", "TIME", "datetime", "date"),
    "depth": ("depth", "deptht", "lev", "level", "z", "Z"),
}


def _find_coord(ds: xr.Dataset, kind: str, required: bool = True) -> str | None:
    """Find a coordinate/dimension using common ocean-data naming conventions."""
    aliases = COORD_ALIASES[kind]
    for candidate in aliases:
        if candidate in ds.coords or candidate in ds.dims:
            return candidate
    if required:
        raise ValueError(
            f"Could not find {kind!r} coordinate. Available coordinates: "
            f"{list(ds.coords)}; dimensions: {list(ds.dims)}"
        )
    return None


def _normalise_longitude(da: xr.DataArray, lon_name: str) -> xr.DataArray:
    """Convert longitudes to [0, 360) and sort them."""
    lon = da[lon_name]
    if float(lon.min()) < 0:
        new_lon = (lon % 360)
        da = da.assign_coords({lon_name: new_lon})
    return da.sortby(lon_name)


def open_variable(
    path: str | Path,
    variable: str,
    chunks: Mapping[str, int] | None = None,
) -> xr.DataArray:
    """Open a variable from a NetCDF/Zarr dataset and normalise coordinate names."""
    path = str(path)
    if path.endswith(".zarr"):
        ds = xr.open_zarr(path, chunks=chunks)
    else:
        ds = xr.open_dataset(path, chunks=chunks)

    if variable not in ds:
        raise KeyError(f"Variable {variable!r} not found in {path}. Found: {list(ds.data_vars)}")

    da = ds[variable]
    rename = {}
    for kind in ("lat", "lon", "time", "depth"):
        src = _find_coord(ds, kind, required=(kind != "depth"))
        if src and src != kind:
            rename[src] = kind
    da = da.rename(rename)
    if "lon" in da.coords:
        da = _normalise_longitude(da, "lon")
    return da


def subset_domain(da: xr.DataArray) -> xr.DataArray:
    """Clip a field to 5–30 N, 45–105 E, handling ascending/descending axes."""
    da = _normalise_longitude(da, "lon")
    lat_slice = slice(float(TARGET_LATS[0]), float(TARGET_LATS[-1]))
    lon_slice = slice(float(TARGET_LONS[0]), float(TARGET_LONS[-1]))
    if float(da.lat[0]) > float(da.lat[-1]):
        lat_slice = slice(float(TARGET_LATS[-1]), float(TARGET_LATS[0]))
    return da.sel(lat=lat_slice, lon=lon_slice)


def to_daily(
    da: xr.DataArray,
    start: str | None = None,
    end: str | None = None,
    method: str = "mean",
) -> xr.DataArray:
    """Convert sub-daily observations to daily values."""
    if start or end:
        da = da.sel(time=slice(start, end))
    da = da.sortby("time")
    if method == "nearest":
        # Resample to midnight and choose nearest observation within one day.
        return da.resample(time="1D").nearest(tolerance=np.timedelta64(1, "D"))
    if method == "mean":
        return da.resample(time="1D").mean(skipna=True)
    raise ValueError(f"Unsupported daily aggregation: {method}")


def regrid_surface(da: xr.DataArray) -> xr.DataArray:
    """Interpolate a surface field to the canonical 0.25-degree grid."""
    da = subset_domain(da)
    # xarray's linear interpolation handles regular lat/lon source grids.
    out = da.interp(
        lat=xr.DataArray(TARGET_LATS, dims="lat"),
        lon=xr.DataArray(TARGET_LONS, dims="lon"),
        method="linear",
    )
    return out.transpose("time", "lat", "lon")


def regrid_temperature(
    da: xr.DataArray,
    extrapolate_deep: bool = False,
) -> xr.DataArray:
    """Regrid GLORYS potential temperature horizontally and vertically."""
    if "depth" not in da.dims:
        raise ValueError("Target temperature must contain a depth dimension.")

    da = subset_domain(da)
    # Ocean products sometimes expose positive-down depth; enforce ascending.
    if float(da.depth[0]) > float(da.depth[-1]):
        da = da.sortby("depth")

    # First horizontally; then interpolate to the 15 benchmark depths.
    out = da.interp(
        lat=xr.DataArray(TARGET_LATS, dims="lat"),
        lon=xr.DataArray(TARGET_LONS, dims="lon"),
        method="linear",
    )
    kwargs = {"method": "linear"}
    if extrapolate_deep:
        kwargs["kwargs"] = {"fill_value": "extrapolate"}
    out = out.interp(depth=xr.DataArray(STANDARD_DEPTHS, dims="depth"), **kwargs)
    return out.transpose("time", "depth", "lat", "lon")


def _load_and_harmonize_surface(
    spec: SourceSpec,
    start: str | None,
    end: str | None,
) -> xr.DataArray:
    da = open_variable(spec.path, spec.variable)
    da = to_daily(da, start=start, end=end, method=spec.time_aggregation)
    return regrid_surface(da)


def build_harmonized_arrays(
    surface_sources: Sequence[SourceSpec],
    target_source: SourceSpec,
    start: str | None = None,
    end: str | None = None,
    require_complete_days: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build aligned NumPy arrays from seven surface sources and one target source.

    Missing values are retained as NaN during validation. Training should either
    use a valid-data mask or a deliberate gap-filling strategy; this function
    never silently fills missing observations with zeros.
    """
    if {s.name for s in surface_sources} != set(CANONICAL_INPUTS):
        raise ValueError(
            f"Surface sources must cover exactly {CANONICAL_INPUTS}; "
            f"received {[s.name for s in surface_sources]}"
        )
    if target_source.kind != "target":
        raise ValueError("target_source.kind must be 'target'.")

    fields: dict[str, xr.DataArray] = {}
    for spec in surface_sources:
        fields[spec.name] = _load_and_harmonize_surface(spec, start, end)

    target = open_variable(target_source.path, target_source.variable)
    target = to_daily(target, start=start, end=end, method=target_source.time_aggregation)
    target = regrid_temperature(target)

    # Inner join prevents a source with a different time coverage from creating
    # misaligned samples.
    arrays = list(fields.values()) + [target]
    aligned = xr.align(*arrays, join="inner")
    fields_aligned = dict(zip(fields.keys(), aligned[:-1]))
    target_aligned = aligned[-1]

    if target_aligned.sizes.get("time", 0) == 0:
        raise RuntimeError("No overlapping dates remain after temporal alignment.")

    # Build a common finite-data mask. Land is naturally represented by NaNs in
    # ocean products; retaining this mask avoids hand-written coastline polygons.
    surface_stack = xr.concat(
        [fields_aligned[name] for name in CANONICAL_INPUTS],
        dim="feature",
    )
    valid = np.isfinite(surface_stack).all("feature") & np.isfinite(target_aligned).all("depth")

    if require_complete_days:
        day_valid = valid.all(dim=("lat", "lon"))
        keep = day_valid.values
        if not np.any(keep):
            # A strict complete-domain requirement is often too restrictive for
            # satellite products. Fall back to days with useful ocean coverage.
            coverage = valid.mean(dim=("lat", "lon"))
            keep = (coverage.values >= 0.50)
        surface_stack = surface_stack.isel(time=keep)
        target_aligned = target_aligned.isel(time=keep)
        valid = valid.isel(time=keep)

    x = surface_stack.transpose("time", "feature", "lat", "lon").values.astype(np.float32)
    y = target_aligned.transpose("time", "depth", "lat", "lon").values.astype(np.float32)
    dates = target_aligned.time.values.astype("datetime64[D]")

    # The mask must describe valid ocean cells, not valid observations on one day.
    ocean_mask = np.isfinite(y).any(axis=(0, 1))
    x[:, :, ~ocean_mask] = np.nan
    y[:, :, ~ocean_mask] = np.nan

    return x, y, ocean_mask.astype(bool), dates


def save_harmonized_dataset(
    output_path: str | Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    ocean_mask: np.ndarray,
    dates: np.ndarray,
) -> None:
    """Persist the harmonized arrays in the same format consumed by OceanDataset."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs,
        targets=targets,
        ocean_mask=ocean_mask,
        dates=dates,
        depths=STANDARD_DEPTHS,
        lats=TARGET_LATS,
        lons=TARGET_LONS,
    )

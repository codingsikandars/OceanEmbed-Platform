"""Print variables, dimensions and coordinates from a NetCDF/Zarr file."""
from __future__ import annotations
import argparse
import xarray as xr

p = argparse.ArgumentParser()
p.add_argument("path")
args = p.parse_args()

ds = xr.open_zarr(args.path) if args.path.endswith(".zarr") else xr.open_dataset(args.path)
print(ds)
print("\nVariables:")
for name, da in ds.data_vars.items():
    print(f"  {name}: dims={da.dims}, shape={da.shape}, dtype={da.dtype}")

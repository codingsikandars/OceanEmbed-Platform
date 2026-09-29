"""Basic environment diagnostic for OceanEmbed."""
import importlib.util
import sys

packages = ["torch", "numpy", "scipy", "xarray", "netCDF4", "yaml", "matplotlib"]
print("Python:", sys.version)
for name in packages:
    print(f"{name:12s}", "OK" if importlib.util.find_spec(name) else "MISSING")

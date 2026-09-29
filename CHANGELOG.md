# Changelog

## OceanEmbed continuation — SIH 26066

### Added
- Real NetCDF/Zarr harmonization pipeline in `src/data/real_data.py`.
- `configs/real_data.yaml` manifest for seven surface inputs + GLORYS target.
- `prepare_real_data.py` for temporal splitting and dataset export.
- Official data-access/provenance documentation.
- Model card and scientific validation guidance.
- Optional data-access and development requirements.
- Environment and NetCDF inspection utilities.

### Improved
- Temporal sliding windows are flattened into model channels and wired through
  the model factory.
- Decoder channel dimensions now correctly respect `base_channels`.
- Real-data masks can be loaded from processed files.
- Remaining NaNs use an explicit fallback rather than silently entering the model.
- Physics constraints are evaluated in physical Celsius space.
- D20 calculation now uses robust sign-change crossing detection.
- Training seeds Python/NumPy/PyTorch for reproducibility.
- Evaluation supports configured temporal windows and explicit checkpoint loading.

### Scientific notes
- GLORYS12V1 is treated as a data-assimilative training target, not truth.
- ARGO validation is flagged as potentially non-independent when observations
  overlap the GLORYS assimilation system.
- CCMP V3.1 is treated as 6-hourly and aggregated to daily means.
- Product versions and access paths are documented rather than bundled.

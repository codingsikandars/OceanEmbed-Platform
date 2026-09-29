# OceanEmbed data access and provenance

OceanEmbed is designed around **real observational/reanalysis products**, but the
repository does not bundle third-party ocean data. This keeps the repository
small, respects provider access conditions, and makes the experiment dates
explicit.

## 1. Recommended training/evaluation stack

| Variable | Recommended source | Native resolution / cadence | Role |
|---|---|---|---|
| SST | Copernicus Marine OSTIA reprocessed | 0.05°, daily | surface input |
| SSS | Copernicus Marine multi-observation SSS | 1/8°, daily | surface input |
| SSH/SLA | Copernicus Marine DUACS delayed-time | 0.25°, daily | surface input |
| U/V current | NASA PO.DAAC OSCAR L4 final V2 | 0.25°, daily | surface input |
| U/V wind | NASA PO.DAAC CCMP V3.1 | 0.25°, 6-hourly | daily-mean surface input |
| Temperature | Copernicus Marine GLORYS12V1 | 0.083°, daily, 50 levels | training target |
| In-situ temperature | INCOIS ARGO / LAS or ERDDAP | product-dependent | independent validation |

The problem statement's source list is therefore a **logical source specification**,
not a promise that every cited product has exactly 0.25°/daily native resolution.

## 2. Official sources

- GLORYS12V1: https://doi.org/10.48670/moi-00021
- OSTIA: https://doi.org/10.48670/moi-00168
- Multi-observation SSS: https://doi.org/10.48670/moi-00051
- DUACS: https://doi.org/10.48670/moi-00145
- OSCAR: https://doi.org/10.5067/OSCAR-25F20
- CCMP V3.1: https://doi.org/10.5067/CCMP-6HW10M-L4V31
- INCOIS data holdings: https://incois.gov.in/site/dataholdings.jsp
- INCOIS ERDDAP ARGO example: https://erddap.incois.gov.in/erddap/griddap/incois_argo_mnt_VAM.html

## 3. Copernicus Marine access

Install the official toolbox:

```bash
python -m pip install copernicusmarine
copernicusmarine login
```

Use the catalogue before downloading:

```bash
copernicusmarine describe --contains GLORYS
copernicusmarine describe --product-id GLOBAL_MULTIYEAR_PHY_001_030
```

Then subset only the North Indian Ocean and requested dates/variables. The
current GLORYS catalogue exposes the daily dataset as
`cmems_mod_glo_phy_my_0.083deg_P1D-m`.

Do not download the global archive if a regional subset is sufficient.

## 4. PO.DAAC access

For OSCAR/CCMP, use Earthdata/PO.DAAC tools and subset by the bounding box:

```text
45E, 5N, 105E, 30N
```

For CCMP, retain its native 6-hourly observations initially and let
`src.data.real_data.to_daily()` compute the daily mean. This is preferable to
pretending that the native product is daily.

## 5. Inspect every product before creating the manifest

```bash
python scripts/inspect_netcdf.py data/raw/ostia.nc
python scripts/inspect_netcdf.py data/raw/glorys12v1.nc
```

Then update `configs/real_data.yaml` with the exact variable names present in
the downloaded files.

## 6. Build the harmonized dataset

```bash
python prepare_real_data.py --manifest configs/real_data.yaml
```

The pipeline:

1. clips each source to 5–30°N, 45–105°E;
2. normalizes longitude convention;
3. aggregates sub-daily fields to daily;
4. horizontally interpolates all surface inputs to the 0.25° grid;
5. interpolates GLORYS temperature vertically to the 15 requested levels;
6. aligns all variables by date;
7. derives an ocean-valid mask from the target;
8. creates a chronological train/validation/test split.

## 7. Important scientific leakage issue

GLORYS is a data-assimilative reanalysis. Its documentation states that
satellite SST, sea level, and in-situ temperature/salinity profiles are
assimilated. Therefore, an ARGO profile that contributed to GLORYS is not
strictly independent of the training target.

For a defensible competition result:

- train on a historical period;
- evaluate on a later held-out period;
- where possible, use withheld ARGO profiles/floats or a float/time subset
  excluded from the GLORYS-derived training target;
- report the validation protocol explicitly.

Do not call an ARGO-vs-GLORYS comparison "independent" merely because the ARGO
file is stored separately.

## 8. Missing data

The default PyTorch dataset has a conservative channel-mean fallback for
remaining NaNs after provider-native quality control. For publication-grade
experiments, replace this with a provider-specific valid-data mask or
physically justified gap-filling method and report it.

## 9. Reproducibility checklist

Record:

- exact provider product ID/version;
- download date;
- requested spatial/time subset;
- variable names;
- quality-control flags;
- regridding method;
- temporal aggregation;
- train/validation/test date ranges;
- normalization statistics;
- random seed;
- git commit;
- model configuration.

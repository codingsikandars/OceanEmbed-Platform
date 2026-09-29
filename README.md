# OceanEmbed — SIH Problem Statement 26066

**OceanEmbed** is an end-to-end PyTorch research framework for reconstructing
daily 3-D subsurface ocean temperature from multi-modal surface observations
over the North Indian Ocean (NIO).

> **SIH 2026 Problem Statement:** 26066  
> **Title:** OceanEmbed - Satellite Embedding-Based Deep Learning Framework for
> Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations

---

## 1. What the problem is actually asking

The inverse problem is:

```text
7 surface observations
        ↓
satellite/ocean-state embedding
        ↓
nonlinear surface → subsurface mapping
        ↓
15-depth temperature profile
```

The model receives only surface information but is trained against a 3-D
temperature field. This is possible because surface variables contain indirect
signatures of subsurface structure through thermocline displacement, eddies,
upwelling/downwelling, horizontal advection, freshwater stratification,
wind-driven mixing and air-sea coupling.

### Domain

- Latitude: **5°N to 30°N**
- Longitude: **45°E to 105°E**
- Target grid: **0.25° × 0.25°**
- Target temporal cadence: **daily**
- Target vertical levels:

```text
[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m
```

The resulting grid has **101 × 241 = 24,341 grid points**.

---

# 2. Data architecture

## Surface input channels

| # | Channel | Recommended product | Native product | Harmonization |
|---|---|---|---|---|
| 1 | SST | OSTIA | 0.05°, daily | → 0.25° |
| 2 | SSS | CMEMS multi-observation SSS | 1/8°, daily | → 0.25° |
| 3 | SSH/SLA | DUACS | 0.25°, daily | → target grid |
| 4 | Current U | OSCAR L4 | 0.25°, daily | → target grid |
| 5 | Current V | OSCAR L4 | 0.25°, daily | → target grid |
| 6 | Wind U | CCMP V3.1 | 0.25°, 6-hourly | daily mean |
| 7 | Wind V | CCMP V3.1 | 0.25°, 6-hourly | daily mean |

### Training target

**GLORYS12V1 / Global Ocean Physics Reanalysis**

- Product ID: `GLOBAL_MULTIYEAR_PHY_001_030`
- DOI: `10.48670/moi-00021`
- Current catalogue resolution: ~0.083° × 0.083°
- 50 vertical levels
- Daily and monthly datasets
- Target variable: sea-water potential temperature (`thetao`)
- Regridded horizontally to 0.25°
- Interpolated vertically to the 15 competition levels

GLORYS is a **data-assimilative reanalysis**, not direct ground truth. Its
documentation states that satellite SST, sea level and in-situ temperature/
salinity observations are assimilated. This matters when designing ARGO
validation.

### Validation

The intended independent observation source is **INCOIS ARGO/LAS**. For a
strictly independent scientific experiment, use held-out ARGO profiles/floats
or an evaluation period that is not represented in the training target
construction.

---

# 3. Official data references

| Dataset | Official reference |
|---|---|
| GLORYS12V1 | https://doi.org/10.48670/moi-00021 |
| OSTIA | https://doi.org/10.48670/moi-00168 |
| Multi-observation SSS | https://doi.org/10.48670/moi-00051 |
| DUACS | https://doi.org/10.48670/moi-00145 |
| OSCAR L4 Final V2 | https://doi.org/10.5067/OSCAR-25F20 |
| CCMP 6-hourly V3.1 | https://doi.org/10.5067/CCMP-6HW10M-L4V31 |
| INCOIS holdings | https://incois.gov.in/site/dataholdings.jsp |
| INCOIS ARGO ERDDAP example | https://erddap.incois.gov.in/erddap/griddap/incois_argo_mnt_VAM.html |

**Important:** the PS's source table should be treated as a recommended source
stack. Product versions and native resolution can change. Record the exact
product/version used in every experiment.

---

# 4. Repository

```text
OceanEmbed-Platform/
├── configs/
│   ├── default.yaml
│   └── real_data.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── DATA_ACCESS.md
│   └── MODEL_CARD.md
├── scripts/
│   ├── check_environment.py
│   └── inspect_netcdf.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── preprocess.py
│   │   └── real_data.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── decoder.py
│   │   └── oceanembed_net.py
│   └── utils/
│       ├── losses.py
│       └── metrics.py
├── checkpoints/
├── outputs/
├── prepare_real_data.py
├── train.py
├── evaluate.py
├── requirements.txt
├── requirements-data.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

---

# 5. Model architecture

```text
7 surface channels × T daily context
             │
             ├── optional latitude/longitude CoordConv
             ↓
     Multi-modal CNN encoder
             │
      Residual + SE blocks
             │
       Multi-scale features
             ↓
     Spatial self-attention
             │
             ↓
   256-channel satellite embedding
             │
        ┌────┴────┐
        │ skip f2 │
        │ skip f1 │
        └────┬────┘
             ↓
       Hierarchical decoder
             │
       Depth feature volume
             ↓
       Cross-depth attention
             ↓
     15 temperature layers
```

## Why an embedding?

The central idea is not just to predict temperature directly.

The encoder learns a latent representation:

```text
Z = Encoder(SST, SSS, SLA, Ucurr, Vcurr, Uwind, Vwind, coordinates)
```

The spatial embedding can later be reused for:

- subsurface temperature reconstruction;
- marine heatwave characterization;
- eddy classification;
- ocean-regime clustering;
- downstream ocean prediction tasks.

---

# 6. Why each input is useful

### SST

Constrains the upper-ocean thermal state and carries signatures of mixed-layer
heat exchange, fronts, upwelling and atmospheric forcing.

### SSS

Especially important in the Bay of Bengal, where freshwater input produces strong
salinity stratification and shallow barrier layers. This can weaken the direct
relationship between SST and deeper thermocline structure.

### SSH / SLA

Provides information about dynamic height and mesoscale eddies. Positive and
negative sea-level anomalies are strongly associated with thermocline
displacement.

### Surface currents

Represent horizontal advection and boundary-current/eddy dynamics.

### Winds

Provide information about Ekman transport, mixing, upwelling/downwelling and
air-sea momentum exchange.

The seven channels therefore encode complementary physical information rather
than seven redundant measurements.

---

# 7. Physics-informed loss

The implemented objective is:

```text
L =
  λ_mse  L_masked_MSE
+ λ_grad L_vertical_gradient
+ λ_surf L_surface_consistency
+ λ_strat L_stratification
```

## Masked MSE

Only ocean cells contribute to the reconstruction loss.

## Vertical gradient

The loss compares:

```text
dT/dz
```

between prediction and target. It is evaluated after converting predictions
back to Celsius, so the depth spacing and inversion tolerance retain physical
meaning.

## Surface consistency

The reconstructed 0 m temperature is encouraged to remain consistent with
the observed SST.

## Stratification

A tolerance is used rather than forcing every profile to be perfectly
monotonic. This is important because real upper-ocean profiles can exhibit
weak inversions/barrier-layer effects.

---

# 8. Temporal windows

`configs/default.yaml` contains:

```yaml
temporal:
  time_window: 1
```

For `time_window: 3`, the dataset produces:

```text
day t-2:
  SST SSS SLA Uc Vc Uw Vw
day t-1:
  SST SSS SLA Uc Vc Uw Vw
day t:
  SST SSS SLA Uc Vc Uw Vw
```

These are flattened into 21 feature channels plus two coordinate channels.

This lets the model learn temporal evolution while retaining a standard 2-D
convolutional backbone.

---

# 9. Real-data workflow

## Step 1 — install

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then:

```bash
pip install -r requirements.txt
```

Optional official Copernicus data tooling:

```bash
pip install -r requirements-data.txt
```

The Copernicus Marine Toolbox currently supports catalogue discovery, spatial/
temporal subsetting, original-file download and lazy remote xarray access.

---

## Step 2 — authenticate with Copernicus Marine

```bash
copernicusmarine login
```

Inspect the catalogue:

```bash
copernicusmarine describe --product-id GLOBAL_MULTIYEAR_PHY_001_030
```

Subset only the region and dates required by the experiment.

---

## Step 3 — download/subset the sources

Place local files under:

```text
data/raw/
```

Example:

```text
data/raw/
├── ostia.nc
├── sss.nc
├── duacs.nc
├── oscar.nc
├── ccmp.nc
└── glorys12v1.nc
```

Do **not** commit these files to Git.

---

## Step 4 — inspect variables

```bash
python scripts/inspect_netcdf.py data/raw/ostia.nc
python scripts/inspect_netcdf.py data/raw/glorys12v1.nc
```

Update `configs/real_data.yaml` if the provider uses a different variable name.

---

## Step 5 — harmonize

```bash
python prepare_real_data.py --manifest configs/real_data.yaml
```

The pipeline performs:

```text
raw NetCDF
   ↓
coordinate normalization
   ↓
NIO spatial subset
   ↓
daily temporal aggregation
   ↓
0.25° horizontal interpolation
   ↓
GLORYS vertical interpolation
   ↓
date alignment
   ↓
ocean mask
   ↓
chronological train / val / test split
   ↓
NPZ arrays
```

---

# 10. Synthetic benchmark

The repository contains a synthetic benchmark so the complete ML pipeline can
be demonstrated without downloading large scientific archives.

Generate it with:

```bash
python -m src.data.preprocess
```

Then train:

```bash
python train.py --config configs/default.yaml --epochs 2 --batch_size 2
```

The synthetic benchmark is for:

- code testing;
- architecture debugging;
- presentation demonstrations;
- smoke tests.

It must **not** be reported as real oceanographic skill.

---

# 11. Training

Basic:

```bash
python train.py
```

Override epochs:

```bash
python train.py --epochs 50
```

Override batch size:

```bash
python train.py --batch_size 4
```

Force GPU:

```bash
python train.py --device cuda
```

The trainer uses:

- AdamW;
- cosine learning-rate schedule;
- optional CUDA AMP;
- gradient clipping;
- validation RMSE;
- early stopping;
- best/latest checkpoints;
- training-history JSON.

---

# 12. Evaluation

```bash
python evaluate.py
```

The evaluation framework reports:

- depth-wise RMSE;
- Pearson correlation;
- mean bias;
- MAE;
- vertical gradient error;
- D20 error where valid;
- horizontal temperature slices;
- depth-wise skill plots;
- ARGO profile comparison where supplied.

---

# 13. Metrics that should appear in the SIH presentation

Do not present only one average RMSE.

Recommended dashboard:

```text
                 Surface → Deep

Depth       RMSE      Pearson r      Bias
------------------------------------------------
0 m         ...          ...          ...
5 m         ...          ...          ...
...
1000 m      ...          ...          ...
```

Also report:

### Regional

- Arabian Sea
- Bay of Bengal

### Seasonal

- pre-monsoon
- southwest monsoon
- post-monsoon
- winter

### Physical

- D20 RMSE
- mixed-layer error if a robust MLD definition is implemented
- vertical-gradient error

### Validation

- held-out ARGO profiles
- held-out time period
- number of valid profiles
- number of valid grid cells

---

# 14. Critical scientific validation issue

A major point for the final SIH explanation:

**GLORYS is not an observation-only truth field.**

The product assimilates observations, including in-situ temperature/salinity
profiles. Therefore:

```text
ARGO → GLORYS assimilation
ARGO → "independent" validation
```

can create statistical dependence.

A stronger evaluation protocol is:

```text
Historical surface data ───────┐
                               ↓
                         OceanEmbed training
                               ↓
                     Held-out future period
                               ↓
                         ARGO validation
```

and, where feasible, exclude the validation float/time subset from the target
construction/training experiment.

---

# 15. What “production-grade” should mean for this project

For an SIH PoC, this repository provides the engineering foundation. A true
scientific production system should additionally implement:

1. provider-specific quality flags;
2. per-variable observation masks;
3. uncertainty channels;
4. robust coastal/ocean masks;
5. data version manifests;
6. experiment tracking;
7. reproducible temporal holdouts;
8. regional/seasonal stratified validation;
9. uncertainty estimation;
10. ensemble or probabilistic prediction;
11. D20/MLD diagnostics;
12. bias correction;
13. automated data freshness checks;
14. model monitoring;
15. containerized deployment;
16. automated tests and CI.

---

# 16. Recommended SIH architecture story

For a PPT/demo, explain the system as four layers:

### Layer 1 — Observation fusion

```text
OSTIA
SMAP/SMOS
DUACS
OSCAR
CCMP
  ↓
harmonization
```

### Layer 2 — Ocean embedding

```text
7 modalities
  ↓
CoordConv
  ↓
Residual CNN + SE
  ↓
attention
  ↓
256-D satellite embedding
```

### Layer 3 — 3-D reconstruction

```text
embedding
  ↓
multi-scale decoder
  ↓
vertical attention
  ↓
15-depth temperature profile
```

### Layer 4 — scientific validation

```text
GLORYS training target
        +
held-out ARGO observations
        ↓
RMSE / r / Bias / D20
```

---

# 17. Existing demo artifacts

The original uploaded project already contained:

- trained checkpoints;
- synthetic train/validation/test arrays;
- training history;
- evaluation scripts.

Those artifacts are retained as a demonstration baseline. Large third-party
scientific datasets are intentionally not included.

---

# 18. Citation / attribution

When presenting results, cite the exact dataset versions used and include the
official DOI/product identifier.

At minimum:

- Copernicus Marine GLORYS12V1;
- Copernicus Marine OSTIA;
- Copernicus Marine multi-observation SSS;
- Copernicus Marine DUACS;
- NASA PO.DAAC OSCAR;
- NASA PO.DAAC CCMP;
- INCOIS ARGO.

---

## License

The repository code can be distributed under the project's chosen open-source
license, but third-party datasets remain subject to their own provider
licenses and citation requirements.

# OceanEmbed model card

## Intended use

OceanEmbed is a research/PoC framework for estimating daily subsurface
temperature profiles from multi-modal surface observations over the North
Indian Ocean.

It is intended for:

- SIH 26066 demonstration;
- methodological experiments;
- oceanographic representation learning;
- profile reconstruction research.

It is **not** a replacement for operational ocean analyses, safety-critical
forecasting, navigation, or data assimilation systems without independent
scientific validation.

## Inputs

Seven surface channels:

1. SST
2. SSS
3. SSH/SLA
4. current U
5. current V
6. wind U
7. wind V

Optional temporal context stacks consecutive daily snapshots into channels.

## Output

Temperature at:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

at the canonical 0.25° grid.

## Architecture

- CoordConv positional injection;
- residual CNN multi-scale encoder;
- squeeze-and-excitation channel attention;
- basin-level self-attention bottleneck;
- 256-dimensional spatial latent embedding;
- hierarchical decoder with skip connections;
- cross-depth attention;
- physics-informed reconstruction loss.

## Known limitations

1. Surface-to-deep inversion is ill-posed; multiple subsurface states can share
   similar surface observations.
2. GLORYS is a reanalysis target, not direct truth.
3. GLORYS assimilates observations, so naive ARGO validation can have leakage.
4. A CNN prior can oversmooth sharp thermoclines.
5. The 1000 m level has weaker direct surface observability than the mixed layer.
6. Regional coastline masks and missing-data treatment materially affect scores.
7. The included synthetic benchmark is only a pipeline test and must not be
   presented as real oceanographic skill.

## Recommended reporting

Always report:

- depth-wise RMSE;
- depth-wise Pearson r;
- depth-wise bias/MAE;
- D20 error where a valid 20°C crossing exists;
- skill by Arabian Sea vs Bay of Bengal;
- skill by season/monsoon phase;
- number of valid ocean cells/profiles;
- the exact temporal holdout.

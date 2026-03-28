# SNN Simulated-Noise Eval Snapshot (2026-03-28)

- Snapshot time: `2026-03-28 00:58:19 UTC`
- Sequence: `zurich_city_14_c`
- Queue split: host A (`snow/shot/gaussian/impulse + clean`) and host B (`glass/motion/fog/frost`)
- Completed: `28` / `41`
- Pending: `13`

## Artifacts

- Curve plot: `results/analysis/snn_sim_best14c_status_20260328/map50_severity_curves_snapshot.png`
- Flat table CSV: `results/analysis/snn_sim_best14c_status_20260328/noise_map50_snapshot.csv`

## mAP@0.5 By Condition (snapshot)

- `clean_s0`: `0.001308`

| Noise | s1 | s2 | s3 | s4 | s5 |
|---|---:|---:|---:|---:|---:|
| snow | 0.000654 | 0.000725 | 0.000224 | 0.000599 | 0.000172 |
| shot_noise | 0.001873 | 0.001185 | 0.000457 | 0.000427 | 0.000231 |
| gaussian_noise | 0.001890 | 0.000451 | 0.000846 | 0.000400 | 0.010174 |
| impulse_noise | 0.001098 | PENDING | 0.000808 | PENDING | 0.010090 |
| glass_blur | 0.000364 | 0.000606 | 0.000358 | 0.000381 | 0.000220 |
| motion_blur | 0.000509 | 0.000292 | 0.000194 | PENDING | 0.000212 |
| fog | PENDING | PENDING | PENDING | PENDING | PENDING |
| frost | PENDING | PENDING | PENDING | PENDING | PENDING |

## Missing / Pending Conditions

- `impulse_noise_s2`
- `impulse_noise_s4`
- `motion_blur_s4`
- `fog_s1`
- `fog_s2`
- `fog_s3`
- `fog_s4`
- `fog_s5`
- `frost_s1`
- `frost_s2`
- `frost_s3`
- `frost_s4`
- `frost_s5`

## Quick Read

- Most conditions are very low (`~1e-4` to `~2e-3`).
- Two strong outliers spike high at severity 5:
  - `gaussian_noise_s5 = 0.010174`
  - `impulse_noise_s5 = 0.010090`
- This non-monotonic behavior suggests evaluation is sensitive to how severe corruption shifts confidence/NMS behavior and false-positive density, not just "harder noise => lower mAP".
- Confirm after all pending conditions finish before drawing final robustness conclusions.

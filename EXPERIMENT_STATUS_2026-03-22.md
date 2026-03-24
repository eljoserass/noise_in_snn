# Neuromorph vs Noise - Experiment Status (2026-03-22)

## 1) Scope and Current Goal
- Main goal: compare ANN vs SNN robustness under image/event noise on DSEC.
- Practical constraint: finish reliable experiments fast under compute/storage pressure.
- Current priority: get stable SNN training/eval running, then compare against ANN on a focused test protocol.

## 2) Dataset Setup We Are Using
- Training subset (targeted):
  - `thun_00_a`, `interlaken_00_c`, `interlaken_00_e`, `interlaken_00_g`,
  - `zurich_city_00_b`, `zurich_city_02_c`, `zurich_city_05_b`, `zurich_city_10_b`
- Validation subset:
  - `zurich_city_16_a`, `zurich_city_17_a`, `zurich_city_18_a`, `zurich_city_19_a`, `zurich_city_20_a`, `zurich_city_21_a`
- Test strategy now: depth-first on one sequence (`zurich_city_14_c`) to get complete condition coverage faster.

## 3) Models and Training Pipeline
- ANN: SSD-style detector (VGG11-based baseline in this repo), trained on RGB.
- SNN: spiking SSD variant (same detection task), trained on event representations.
- Recent SNN iteration changes:
  - move to reliable H5 event inputs (`events.h5`) instead of fragile text event dumps,
  - timestep accumulation style training enabled (`timesteps_per_frame > 1`),
  - sequence stride/length tuning for memory/runtime control,
  - balanced subset retained for comparability with ANN subset experiments.

## 4) What We Already Ran (ANN)
- Full-data ANN reference run:
  - W&B: `joserass/dsec_baselines_v1/08oyau6n`
  - best val loss: `1.031839` (epoch 172)
  - final train/val loss: `0.874266 / 1.032333`
- Subset ANN (first attempt):
  - W&B: `joserass/dsec_subset_ann/npj0rzel`
  - best val loss: `4.607113` (epoch 10)
- Subset ANN (current improved subset setup):
  - W&B: `joserass/dsec_subset_ann/wsvx4fr0`
  - best val loss: `2.813453` (epoch 17)
  - final train/val loss: `0.961461 / 3.099035`
- Noise-eval comparison (previous subset vs current subset):
  - summary file: `results/analysis/ann_prev_vs_current_20260317/summary.txt`
  - clean mAP@0.5 improved: `0.090909 -> 0.149403` (`+0.058494`)
  - mean delta over non-clean conditions: `+0.004551`

## 5) SNN History and Current State
- Earlier SNN runs on simulated TXT events repeatedly failed or underperformed because many `dvs_events.txt` files were header-only/empty (loader skipped sequences, causing near-zero detections).
- That failure mode explains multiple runs with 0 predictions / 0 mAP behavior.
- Current SNN direction:
  - use native raw events (`events/left/events.h5`) for training/eval,
  - keep subset protocol aligned with ANN subset,
  - then evaluate clean first, then targeted noise conditions.

## 6) Event Generation + Data Engineering Changes
- We shifted event generation to H5 outputs (or native H5 usage) because TXT files were too brittle and huge.
- We removed large stale TXT event artifacts to reclaim storage.
- Generation order now (test):
  1. clean,
  2. native noise variants,
  3. noise-from-RGB variants (clean v2e over corrupted RGB),
  4. final upload/sync.
- Reason for depth-first on one test sequence:
  - lower risk of partially-complete matrix,
  - faster path to publishable ANN vs SNN comparison,
  - avoids storage/network bottlenecks from broad half-finished generation.

## 7) Noise Protocol (Operational)
- Native event noise (event-domain): shot/leak/threshold-jitter/refractory/bandwidth-limit/photoreceptor style variants, typically across severities.
- RGB corruption -> event simulation path: apply image corruption first, then generate events from corrupted RGB.
- For reporting, we keep condition naming explicit by noise type + severity.

## 8) Why We Focused a Single Test Sequence for Now
- Full-grid over all test sequences became too expensive and slow (CPU bottlenecks, storage near limits, stalled workers).
- One well-chosen sequence allows:
  - complete clean + noise depth,
  - quick ANN/SNN apples-to-apples plots,
  - decision making before scaling again.

## 9) Files With Current Analysis Outputs
- ANN previous vs current noise comparison:
  - `results/analysis/ann_prev_vs_current_20260317/summary.txt`
  - `results/analysis/ann_prev_vs_current_20260317/noise_metrics_comparison.csv`
  - `results/analysis/ann_prev_vs_current_20260317/map50_prev_vs_current.png`
  - `results/analysis/ann_prev_vs_current_20260317/map50_delta.png`
- ANN train/loss comparison artifacts:
  - `results/analysis/ann_train_compare_20260317_141557/summary.txt`
  - `results/analysis/ann_train_compare_20260317_141557/train_val_loss_compare.png`
  - `results/analysis/ann_train_compare_20260317_141557/val_cls_loc_loss_compare.png`

## 10) Current Iteration (What We Are Trying Right Now)
- Finish robust SNN training on H5/native events (subset protocol).
- Run clean test first (quick gate).
- Then run prioritized noise tests in controlled order.
- Keep logging consistent in W&B so per-condition and aggregate ANN vs SNN can be plotted cleanly.

## 11) Live Update (2026-03-24)
- ANN (subset, 8-train + class-balance, grayscale) is running on host `5de37e9747f2`.
  - Run tag: `ann_subset8_balanced_gray_5de37e9747f2_20260324_222952`
  - W&B: `joserass/dsec_subset_ann/runs/2le75b2j`
  - After training, eval auto-runs on `zurich_city_14_c` with priority:
    1) `clean_s0`
    2) `snow`
    3) `shot_noise`
    4) `gaussian_noise`
    5) `impulse_noise`
    6) `glass_blur`
    7) `fog`
    8) `frost`
    9) `motion_blur`
  - Severity order per noise family: `s3`, `s1`, `s5`, `s2`, `s4`.
- SNN raw-events training is active on host `7f8d62967447` (`events/left/events.h5`).
- SNN simulated-events training is active on host `4701f6917ed0` (`v2e_output_distorted_gray_clean/dvs_events.h5`).
- New orchestration script added for reproducible train->eval chain:
  - `scripts/night_runs/run_ann_subset8_balanced_then_eval14c.sh`
  - Includes host guard, venv auto-selection, isolated checkpoint/results folders, and forced W&B logging.

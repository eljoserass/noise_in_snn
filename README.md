# Neuromorph vs Noise

This repository now has two tracks:

- `DSEC` (current, active): main training/evaluation path.
- `TUMTraf` (legacy): kept for backward compatibility.

The main goal of this README is reproducible **training**.

## Current Status (DSEC)

Use these scripts as the default:

- `scripts/train_ann_dsec.py`
- `scripts/train_snn_dsec.py`
- `scripts/evaluate_dsec.py`
- `scripts/night_runs/machine_b_train.sh`

Top-level runners:

- `./run_dsec.sh` (active)
- `./run_tumtraf.sh` (legacy)
- `./run.sh` (legacy alias to `run_tumtraf.sh`)

Deterministic view/config used right now:

- RGB input view: `images/left/distorted`
- RGB labels: `object_detections/left/tracks.npy`
- Event stream: `events/left/events.h5`
- ANN color mode: grayscale by default (`--rgb-color` not set)
- SNN timing: no timestep accumulation/repetition; one forward pass per event frame
- Validation split note: DSEC often ships as `train/` + `test/` folders only; validation is a sequence list sampled from `train`.

## Environment Setup

```bash
cd noise_in_snn
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install wandb
```

For W&B:

```bash
export WANDB_API_KEY='YOUR_KEY'
export WANDB_PROJECT='dsec_baselines_v1'
```

## Reproduce DSEC Baselines

### 1) ANN baseline (grayscale RGB)

```bash
python scripts/train_ann_dsec.py \
  --dsec-root ../data/dsec \
  --train-split train \
  --val-split val \
  --image-relpath images/left/distorted \
  --tracks-relpath object_detections/left/tracks.npy \
  --class-ids 0,1,2,3,4,5,6,7 \
  --time-mode nearest \
  --max-time-delta-us 50000 \
  --input-height 480 \
  --input-width 640 \
  --batch-size 16 \
  --num-workers 4 \
  --wandb \
  --wandb-project dsec_baselines_v1
```

### 2) SNN baseline (real events)

```bash
python scripts/train_snn_dsec.py \
  --dsec-root ../data/dsec \
  --train-split train \
  --val-split val \
  --image-relpath images/left/distorted \
  --tracks-relpath object_detections/left/tracks.npy \
  --event-source real \
  --event-relpath events/left/events.h5 \
  --sequence-length 8 \
  --sequence-stride 8 \
  --event-window-mode between_frames \
  --input-height 480 \
  --input-width 640 \
  --batch-size 4 \
  --num-workers 4 \
  --wandb \
  --wandb-project dsec_baselines_v1
```

### 3) Evaluate checkpoints on test split

```bash
python scripts/evaluate_dsec.py \
  --model-path checkpoints/vgg11_ssd_ann_dsec_best.pth \
  --model-type ann \
  --dsec-root ../data/dsec \
  --test-splits test \
  --image-relpath images/left/distorted \
  --tracks-relpath object_detections/left/tracks.npy

python scripts/evaluate_dsec.py \
  --model-path checkpoints/vgg11_ssd_snn_dsec_best.pth \
  --model-type snn \
  --dsec-root ../data/dsec \
  --test-splits test \
  --image-relpath images/left/distorted \
  --tracks-relpath object_detections/left/tracks.npy \
  --event-source real \
  --event-relpath events/left/events.h5 \
  --sequence-length 8 \
  --sequence-stride 1
```

### 4) Overnight resume-safe loop (Machine B)

```bash
export DSEC_ROOT=../data/dsec
export USE_WANDB=1
export WANDB_PROJECT=dsec_baselines_v1
export DEVICE=cuda
export VAL_SPLIT=val

bash scripts/night_runs/machine_b_train.sh
```

`machine_b_train.sh` and direct `train_*_dsec.py` runs now auto-fall back to `VAL_SPLIT=train` and read val subsequences from `config/dsec_train_val_test_split.yaml` when no physical `data/dsec/val` folder exists.

## Quick Sanity Check (what model sees)

```bash
python scripts/preview_dsec_training_view.py \
  --dsec-root ../data/dsec \
  --split train \
  --sequences zurich_city_02_b \
  --event-source real \
  --event-relpath events/left/events.h5 \
  --out-dir outputs/dsec_training_view_check
```

## Legacy TUMTraf (Separated)

Legacy scripts are explicitly separated and should not be mixed with DSEC runs:

- `scripts/train_ann_tumtraf.py`
- `scripts/train_snn_tumtraf.py`
- `scripts/evaluate_tumtraf.py`
- `scripts/night_runs/machine_b_train_tumtraf.sh`

Run legacy overnight loop only if needed:

```bash
export DATA_PATH=data/preprocessed
export WANDB_PROJECT=tumtraf_legacy_baselines
bash scripts/night_runs/machine_b_train_tumtraf.sh
```

`run.sh` and the old generic script names remain for compatibility, but are legacy/TUMTraf-oriented.

## Appendix A: Data Download / Processing (DSEC)

Training-focused details above are the source of truth. Data movement and conversion utilities are here:

- R2 sync: `scripts/r2_sync.py`
- DSEC conversion batch: `scripts/dsec_batch_pipeline.py`
- RGB->event conversion per sequence: `scripts/dsec_rgb_event_pipeline.py`
- Night scripts for multi-machine:
  - `scripts/night_runs/machine_a_download.sh`
  - `scripts/night_runs/machine_c_convert_loop.sh`

By default, `machine_a_download.sh` pulls `train,test` (not `val`) because validation is typically defined as a subsequence list inside `train`.
`machine_c_convert_loop.sh` also supports periodic background uploads via `UPLOAD_INTERVAL_SECS` so partial results are backed up during long runs.
`machine_c_convert_loop.sh` runs split-specific processing by default: clean-only event simulation on `train,val`, and full corruption/noise + event simulation on `test`.
Set `VALIDATE_RECTIFICATION=1` only when needed; default `0` avoids expensive repeated validation during long loops.

For operational variables and loop behavior, see:

- `scripts/night_runs/README.md`

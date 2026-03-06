#!/usr/bin/env bash
set -euo pipefail

# Machine B (legacy): TUMTraf baseline training loop (resume-safe).
# This script is intentionally separate from the active DSEC loop.

cd "$(dirname "$0")/../.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

DATA_PATH="${DATA_PATH:-data/preprocessed}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TEST_SPLITS="${TEST_SPLITS:-test/day test/night_with_light_off test/night_with_light_on}"
SAVE_DIR="${SAVE_DIR:-checkpoints_tumtraf}"
EPOCHS="${EPOCHS:-200}"
POLL_SECS="${POLL_SECS:-900}"
USE_WANDB="${USE_WANDB:-1}"
DEVICE="${DEVICE:-cuda}"
RUN_EVAL="${RUN_EVAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ANN_BATCH_SIZE="${ANN_BATCH_SIZE:-16}"
SNN_BATCH_SIZE="${SNN_BATCH_SIZE:-16}"
NUM_CLASSES="${NUM_CLASSES:-6}"
TIMESTEPS_PER_FRAME="${TIMESTEPS_PER_FRAME:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-tumtraf_legacy_baselines}"
WANDB_RUN_NAME_ANN="${WANDB_RUN_NAME_ANN:-}"
WANDB_RUN_NAME_SNN="${WANDB_RUN_NAME_SNN:-}"
WANDB_RUN_NAME_EVAL_ANN="${WANDB_RUN_NAME_EVAL_ANN:-}"
WANDB_RUN_NAME_EVAL_SNN="${WANDB_RUN_NAME_EVAL_SNN:-}"

mkdir -p "$SAVE_DIR"

wandb_flag=""
if [ "$USE_WANDB" = "1" ]; then
  wandb_flag="--wandb"
fi

ann_run_name_args=()
snn_run_name_args=()
eval_ann_run_name_args=()
eval_snn_run_name_args=()
if [ -n "$WANDB_RUN_NAME_ANN" ]; then
  ann_run_name_args=(--wandb-run-name "$WANDB_RUN_NAME_ANN")
fi
if [ -n "$WANDB_RUN_NAME_SNN" ]; then
  snn_run_name_args=(--wandb-run-name "$WANDB_RUN_NAME_SNN")
fi
if [ -n "$WANDB_RUN_NAME_EVAL_ANN" ]; then
  eval_ann_run_name_args=(--wandb-run-name "$WANDB_RUN_NAME_EVAL_ANN")
fi
if [ -n "$WANDB_RUN_NAME_EVAL_SNN" ]; then
  eval_snn_run_name_args=(--wandb-run-name "$WANDB_RUN_NAME_EVAL_SNN")
fi

read -r -a test_split_arr <<< "$TEST_SPLITS"

echo "[B:TUMTRAF] training loop started"
echo "[B:TUMTRAF] data path: $DATA_PATH"
echo "[B:TUMTRAF] train split: $TRAIN_SPLIT | val split: $VAL_SPLIT"
echo "[B:TUMTRAF] test splits: ${TEST_SPLITS}"
echo "[B:TUMTRAF] timesteps/frame: ${TIMESTEPS_PER_FRAME}"
echo "[B:TUMTRAF] dataloader workers: ${NUM_WORKERS} | batch ann/snn: ${ANN_BATCH_SIZE}/${SNN_BATCH_SIZE}"

while true; do
  ann_ckpt="${SAVE_DIR}/vgg11_ssd_ann_best.pth"
  snn_ckpt="${SAVE_DIR}/vgg11_ssd_snn_best.pth"

  ann_resume=()
  snn_resume=()
  if [ -f "$ann_ckpt" ]; then
    ann_resume=(--resume "$ann_ckpt")
  fi
  if [ -f "$snn_ckpt" ]; then
    snn_resume=(--resume "$snn_ckpt")
  fi

  echo "[B:TUMTRAF] ANN train pass"
  python scripts/train_ann_tumtraf.py \
    --data-path "$DATA_PATH" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --epochs "$EPOCHS" \
    --batch-size "$ANN_BATCH_SIZE" \
    --num-classes "$NUM_CLASSES" \
    --num-workers "$NUM_WORKERS" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --wandb-project "$WANDB_PROJECT" \
    "${ann_run_name_args[@]}" \
    $wandb_flag \
    "${ann_resume[@]}"

  echo "[B:TUMTRAF] SNN train pass"
  python scripts/train_snn_tumtraf.py \
    --data-path "$DATA_PATH" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --epochs "$EPOCHS" \
    --batch-size "$SNN_BATCH_SIZE" \
    --num-classes "$NUM_CLASSES" \
    --num-workers "$NUM_WORKERS" \
    --timesteps-per-frame "$TIMESTEPS_PER_FRAME" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --wandb-project "$WANDB_PROJECT" \
    "${snn_run_name_args[@]}" \
    $wandb_flag \
    "${snn_resume[@]}"

  if [ "$RUN_EVAL" = "1" ]; then
    if [ -f "$ann_ckpt" ]; then
      echo "[B:TUMTRAF] ANN eval pass"
      python scripts/evaluate_tumtraf.py \
        --model-path "$ann_ckpt" \
        --model-type ann \
        --data-path "$DATA_PATH" \
        --test-split "${test_split_arr[@]}" \
        --num-classes "$NUM_CLASSES" \
        --num-workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --output-dir "${SAVE_DIR}/eval_ann_tumtraf" \
        --wandb-project "$WANDB_PROJECT" \
        "${eval_ann_run_name_args[@]}" \
        $wandb_flag
    fi

    if [ -f "$snn_ckpt" ]; then
      echo "[B:TUMTRAF] SNN eval pass"
      python scripts/evaluate_tumtraf.py \
        --model-path "$snn_ckpt" \
        --model-type snn \
        --data-path "$DATA_PATH" \
        --test-split "${test_split_arr[@]}" \
        --num-classes "$NUM_CLASSES" \
        --timesteps-per-frame "$TIMESTEPS_PER_FRAME" \
        --num-workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --output-dir "${SAVE_DIR}/eval_snn_tumtraf" \
        --wandb-project "$WANDB_PROJECT" \
        "${eval_snn_run_name_args[@]}" \
        $wandb_flag
    fi
  fi

  echo "[B:TUMTRAF] sleeping ${POLL_SECS}s before next resume pass"
  sleep "$POLL_SECS"
done

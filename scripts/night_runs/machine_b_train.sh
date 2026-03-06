#!/usr/bin/env bash
set -euo pipefail

# Machine B: DSEC baseline training loop (resume-safe).
# Runs ANN (RGB) + SNN (event) training and optional test evaluation.

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

DSEC_ROOT="${DSEC_ROOT:-data/dsec}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-}"
VAL_SEQUENCES="${VAL_SEQUENCES:-}"
SPLIT_CONFIG_PATH="${SPLIT_CONFIG_PATH:-../dsec_data_managing/dsec-det/config/train_val_test_split.yaml}"
TEST_SPLITS="${TEST_SPLITS:-test}"
SAVE_DIR="${SAVE_DIR:-checkpoints}"
EPOCHS="${EPOCHS:-200}"
POLL_SECS="${POLL_SECS:-900}"
USE_WANDB="${USE_WANDB:-1}"
DEVICE="${DEVICE:-cuda}"
RUN_EVAL="${RUN_EVAL:-1}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ANN_BATCH_SIZE="${ANN_BATCH_SIZE:-16}"
SNN_BATCH_SIZE="${SNN_BATCH_SIZE:-4}"
ANN_IMAGE_RELPATH="${ANN_IMAGE_RELPATH:-images/left/distorted}"
ANN_TRACKS_RELPATH="${ANN_TRACKS_RELPATH:-object_detections/left/tracks.npy}"
ANN_CROP_TOP_PX="${ANN_CROP_TOP_PX:-0}"
ANN_CROP_BOTTOM_PX="${ANN_CROP_BOTTOM_PX:-0}"
ANN_CROP_LEFT_PX="${ANN_CROP_LEFT_PX:-0}"
ANN_CROP_RIGHT_PX="${ANN_CROP_RIGHT_PX:-0}"
SNN_IMAGE_RELPATH="${SNN_IMAGE_RELPATH:-images/left/distorted}"
SNN_TRACKS_RELPATH="${SNN_TRACKS_RELPATH:-object_detections/left/tracks.npy}"
SNN_SEQUENCE_LENGTH="${SNN_SEQUENCE_LENGTH:-8}"
SNN_SEQUENCE_STRIDE="${SNN_SEQUENCE_STRIDE:-8}"
EVENT_SOURCE="${EVENT_SOURCE:-auto}"
EVENT_RELPATH="${EVENT_RELPATH:-events/left/events.h5}"
WANDB_PROJECT="${WANDB_PROJECT:-dsec_clean_baselines}"
WANDB_RUN_NAME_ANN="${WANDB_RUN_NAME_ANN:-}"
WANDB_RUN_NAME_SNN="${WANDB_RUN_NAME_SNN:-}"
WANDB_RUN_NAME_EVAL_ANN="${WANDB_RUN_NAME_EVAL_ANN:-}"
WANDB_RUN_NAME_EVAL_SNN="${WANDB_RUN_NAME_EVAL_SNN:-}"

# DSEC is often distributed as train/test directories only; val is a subsequence list inside train.
if [ "$VAL_SPLIT" = "val" ] && [ ! -d "${DSEC_ROOT}/val" ]; then
  echo "[B] no physical val directory found at ${DSEC_ROOT}/val"
  if [ -z "$VAL_SEQUENCES" ] && [ -f "$SPLIT_CONFIG_PATH" ]; then
    parsed_val_sequences="$(
      awk '
        $1=="val:" {in_val=1; next}
        in_val && $1=="test:" {in_val=0}
        in_val && $1=="-" {print $2}
      ' "$SPLIT_CONFIG_PATH" | paste -sd "," -
    )"
    if [ -n "$parsed_val_sequences" ]; then
      VAL_SPLIT="train"
      VAL_SEQUENCES="$parsed_val_sequences"
      echo "[B] using val subsequences from split config: ${SPLIT_CONFIG_PATH}"
    else
      VAL_SPLIT="train"
      echo "[B] split config found but val list was empty; fallback VAL_SPLIT=train"
    fi
  else
    VAL_SPLIT="train"
    echo "[B] fallback VAL_SPLIT=train (set VAL_SEQUENCES explicitly for strict val)"
  fi
fi

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

echo "[B] training loop started"
echo "[B] dsec root: $DSEC_ROOT"
echo "[B] train split: $TRAIN_SPLIT | val split: $VAL_SPLIT"
echo "[B] train sequences: ${TRAIN_SEQUENCES:-<all>}"
echo "[B] val sequences: ${VAL_SEQUENCES:-<all>}"
echo "[B] class ids: $CLASS_IDS"
echo "[B] snn sequence length/stride: ${SNN_SEQUENCE_LENGTH}/${SNN_SEQUENCE_STRIDE}"
echo "[B] ann image/tracks: ${ANN_IMAGE_RELPATH} | ${ANN_TRACKS_RELPATH}"
echo "[B] snn image/tracks/events: ${SNN_IMAGE_RELPATH} | ${SNN_TRACKS_RELPATH} | ${EVENT_RELPATH}"
echo "[B] dataloader workers: ${NUM_WORKERS} | batch ann/snn: ${ANN_BATCH_SIZE}/${SNN_BATCH_SIZE}"

while true; do
  ann_ckpt="${SAVE_DIR}/vgg11_ssd_ann_dsec_best.pth"
  snn_ckpt="${SAVE_DIR}/vgg11_ssd_snn_dsec_best.pth"

  ann_resume=()
  snn_resume=()
  if [ -f "$ann_ckpt" ]; then
    ann_resume=(--resume "$ann_ckpt")
  fi
  if [ -f "$snn_ckpt" ]; then
    snn_resume=(--resume "$snn_ckpt")
  fi

  echo "[B] ANN train pass"
  python scripts/train_ann_dsec.py \
    --dsec-root "$DSEC_ROOT" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --train-sequences "$TRAIN_SEQUENCES" \
    --val-sequences "$VAL_SEQUENCES" \
    --image-relpath "$ANN_IMAGE_RELPATH" \
    --tracks-relpath "$ANN_TRACKS_RELPATH" \
    --class-ids "$CLASS_IDS" \
    --crop-top-px "$ANN_CROP_TOP_PX" \
    --crop-bottom-px "$ANN_CROP_BOTTOM_PX" \
    --crop-left-px "$ANN_CROP_LEFT_PX" \
    --crop-right-px "$ANN_CROP_RIGHT_PX" \
    --batch-size "$ANN_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --input-height "$INPUT_HEIGHT" \
    --input-width "$INPUT_WIDTH" \
    --epochs "$EPOCHS" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --wandb-project "$WANDB_PROJECT" \
    "${ann_run_name_args[@]}" \
    $wandb_flag \
    "${ann_resume[@]}"

  echo "[B] SNN train pass"
  python scripts/train_snn_dsec.py \
    --dsec-root "$DSEC_ROOT" \
    --train-split "$TRAIN_SPLIT" \
    --val-split "$VAL_SPLIT" \
    --train-sequences "$TRAIN_SEQUENCES" \
    --val-sequences "$VAL_SEQUENCES" \
    --image-relpath "$SNN_IMAGE_RELPATH" \
    --tracks-relpath "$SNN_TRACKS_RELPATH" \
    --class-ids "$CLASS_IDS" \
    --event-source "$EVENT_SOURCE" \
    --event-relpath "$EVENT_RELPATH" \
    --sequence-length "$SNN_SEQUENCE_LENGTH" \
    --sequence-stride "$SNN_SEQUENCE_STRIDE" \
    --batch-size "$SNN_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --input-height "$INPUT_HEIGHT" \
    --input-width "$INPUT_WIDTH" \
    --epochs "$EPOCHS" \
    --save-dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --wandb-project "$WANDB_PROJECT" \
    "${snn_run_name_args[@]}" \
    $wandb_flag \
    "${snn_resume[@]}"

  if [ "$RUN_EVAL" = "1" ]; then
    if [ -f "$ann_ckpt" ]; then
      echo "[B] ANN eval pass"
      python scripts/evaluate_dsec.py \
        --model-path "$ann_ckpt" \
        --model-type ann \
        --dsec-root "$DSEC_ROOT" \
        --test-splits "$TEST_SPLITS" \
        --image-relpath "$ANN_IMAGE_RELPATH" \
        --tracks-relpath "$ANN_TRACKS_RELPATH" \
        --class-ids "$CLASS_IDS" \
        --crop-top-px "$ANN_CROP_TOP_PX" \
        --crop-bottom-px "$ANN_CROP_BOTTOM_PX" \
        --crop-left-px "$ANN_CROP_LEFT_PX" \
        --crop-right-px "$ANN_CROP_RIGHT_PX" \
        --num-workers "$NUM_WORKERS" \
        --input-height "$INPUT_HEIGHT" \
        --input-width "$INPUT_WIDTH" \
        --device "$DEVICE" \
        --output-dir "${SAVE_DIR}/eval_ann_dsec" \
        --wandb-project "$WANDB_PROJECT" \
        "${eval_ann_run_name_args[@]}" \
        $wandb_flag
    fi

    if [ -f "$snn_ckpt" ]; then
      echo "[B] SNN eval pass"
      python scripts/evaluate_dsec.py \
        --model-path "$snn_ckpt" \
        --model-type snn \
        --dsec-root "$DSEC_ROOT" \
        --test-splits "$TEST_SPLITS" \
        --image-relpath "$SNN_IMAGE_RELPATH" \
        --tracks-relpath "$SNN_TRACKS_RELPATH" \
        --class-ids "$CLASS_IDS" \
        --event-source "$EVENT_SOURCE" \
        --event-relpath "$EVENT_RELPATH" \
        --sequence-length "$SNN_SEQUENCE_LENGTH" \
        --sequence-stride 1 \
        --num-workers "$NUM_WORKERS" \
        --input-height "$INPUT_HEIGHT" \
        --input-width "$INPUT_WIDTH" \
        --device "$DEVICE" \
        --output-dir "${SAVE_DIR}/eval_snn_dsec" \
        --wandb-project "$WANDB_PROJECT" \
        "${eval_snn_run_name_args[@]}" \
        $wandb_flag
    fi
  fi

  echo "[B] sleeping ${POLL_SECS}s before next resume pass"
  sleep "$POLL_SECS"
done

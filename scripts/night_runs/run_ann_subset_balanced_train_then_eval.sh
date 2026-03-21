#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

HOST_TAG="$(hostname | cut -d'.' -f1)"
VENV_DIR="${VENV_DIR:-.venv_train_${HOST_TAG}}"
DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"

RUN_TAG="${RUN_TAG:-ann_subset_balanced_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/${RUN_TAG}_train.log}"
POST_LOG="${POST_LOG:-${LOG_DIR}/${RUN_TAG}_post.log}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-train}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b,zurich_city_07_a,zurich_city_09_d,zurich_city_09_e,zurich_city_04_d}"
VAL_SEQUENCES="${VAL_SEQUENCES:-zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a}"

CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"
IMAGE_RELPATH="${IMAGE_RELPATH:-images/left/distorted_gray}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
MOMENTUM="${MOMENTUM:-0.9}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"

CLASS_BALANCE_BG_WEIGHT="${CLASS_BALANCE_BG_WEIGHT:-0.25}"
CLASS_BALANCE_POWER="${CLASS_BALANCE_POWER:-0.5}"
CLASS_BALANCE_MIN="${CLASS_BALANCE_MIN:-0.25}"
CLASS_BALANCE_MAX="${CLASS_BALANCE_MAX:-8.0}"

SAVE_DIR="${SAVE_DIR:-checkpoints/${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"

ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT_TRAIN="${WANDB_PROJECT_TRAIN:-dsec_subset_ann}"
WANDB_PROJECT_EVAL="${WANDB_PROJECT_EVAL:-dsec_noise_benchmark}"

CONF_THRESHOLD="${CONF_THRESHOLD:-0.2}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.5}"
SEVERITIES="${SEVERITIES:-1,3,5}"
CORRUPTIONS="${CORRUPTIONS:-snow,fog,frost,motion_blur,defocus_blur,gaussian_noise,shot_noise}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: missing venv ${VENV_DIR}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
mkdir -p "${SAVE_DIR}"

echo "[pipeline] run_tag=${RUN_TAG}" | tee -a "${POST_LOG}"
echo "[pipeline] save_dir=${SAVE_DIR}" | tee -a "${POST_LOG}"
echo "[pipeline] train_log=${TRAIN_LOG}" | tee -a "${POST_LOG}"
echo "[pipeline] post_log=${POST_LOG}" | tee -a "${POST_LOG}"

train_cmd=(
  python scripts/train_ann_dsec.py
  --dsec-root "${DSEC_ROOT}"
  --train-split "${TRAIN_SPLIT}"
  --val-split "${VAL_SPLIT}"
  --train-sequences "${TRAIN_SEQUENCES}"
  --val-sequences "${VAL_SEQUENCES}"
  --image-relpath "${IMAGE_RELPATH}"
  --class-ids "${CLASS_IDS}"
  --input-height "${INPUT_HEIGHT}"
  --input-width "${INPUT_WIDTH}"
  --num-workers "${NUM_WORKERS}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --momentum "${MOMENTUM}"
  --early-stop-patience "${EARLY_STOP_PATIENCE}"
  --class-balance
  --class-balance-bg-weight "${CLASS_BALANCE_BG_WEIGHT}"
  --class-balance-power "${CLASS_BALANCE_POWER}"
  --class-balance-min "${CLASS_BALANCE_MIN}"
  --class-balance-max "${CLASS_BALANCE_MAX}"
  --save-dir "${SAVE_DIR}"
  --device "${DEVICE}"
)

if [ "${ENABLE_WANDB}" = "1" ]; then
  train_cmd+=(
    --wandb
    --wandb-project "${WANDB_PROJECT_TRAIN}"
    --wandb-run-name "${RUN_TAG}__train"
  )
fi

echo "[pipeline] starting ANN train..." | tee -a "${POST_LOG}"
"${train_cmd[@]}" > "${TRAIN_LOG}" 2>&1
echo "[pipeline] training finished" | tee -a "${POST_LOG}"

ANN_CKPT="${SAVE_DIR}/vgg11_ssd_ann_dsec_best.pth"
if [ ! -f "${ANN_CKPT}" ]; then
  echo "ERROR: best checkpoint not found at ${ANN_CKPT}" | tee -a "${POST_LOG}"
  exit 1
fi

echo "[pipeline] starting clean baseline + noise sweep..." | tee -a "${POST_LOG}"
env \
  VENV_DIR="${VENV_DIR}" \
  DSEC_ROOT="${DSEC_ROOT}" \
  ANN_CKPT="${ANN_CKPT}" \
  CLASS_IDS="${CLASS_IDS}" \
  INPUT_HEIGHT="${INPUT_HEIGHT}" \
  INPUT_WIDTH="${INPUT_WIDTH}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  DEVICE="${DEVICE}" \
  CONF_THRESHOLD="${CONF_THRESHOLD}" \
  NMS_THRESHOLD="${NMS_THRESHOLD}" \
  SEVERITIES="${SEVERITIES}" \
  CORRUPTIONS="${CORRUPTIONS}" \
  WANDB_PROJECT="${WANDB_PROJECT_EVAL}" \
  RUN_TAG="${RUN_TAG}__noise_eval" \
  LOG_FILE="${POST_LOG}" \
  RESULTS_ROOT="results/noise_benchmark/${RUN_TAG}__noise_eval" \
  bash scripts/night_runs/run_ann_noise_eval_sweep.sh

echo "[pipeline] all done" | tee -a "${POST_LOG}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

HOST_TAG="$(hostname | cut -d'.' -f1)"
VENV_DIR="${VENV_DIR:-.venv_train_${HOST_TAG}}"
RECREATE_VENV="${RECREATE_VENV:-1}"          # 1 = flush venv each run
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"            # 1 = fail if CUDA not available

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision torchaudio}"

DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-train}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b}"
VAL_SEQUENCES="${VAL_SEQUENCES:-zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"

ANN_GPU="${ANN_GPU:-0}"
SNN_GPU="${SNN_GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ANN_BATCH_SIZE="${ANN_BATCH_SIZE:-16}"
SNN_BATCH_SIZE="${SNN_BATCH_SIZE:-4}"
ANN_EPOCHS="${ANN_EPOCHS:-200}"
SNN_EPOCHS="${SNN_EPOCHS:-200}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"

SIM_EVENT_RELPATH="${SIM_EVENT_RELPATH:-v2e_output_distorted_gray_clean/dvs_events.txt}"
SIM_FPS="${SIM_FPS:-20}"
SNN_SEQUENCE_LENGTH="${SNN_SEQUENCE_LENGTH:-8}"
SNN_SEQUENCE_STRIDE="${SNN_SEQUENCE_STRIDE:-8}"

ANN_SAVE_DIR="${ANN_SAVE_DIR:-checkpoints/dsec_subset_ann}"
SNN_SAVE_DIR="${SNN_SAVE_DIR:-checkpoints/dsec_subset_snn_sim}"
ANN_LOG_FILE="${ANN_LOG_FILE:-${LOG_DIR}/train_ann_subset.log}"
SNN_LOG_FILE="${SNN_LOG_FILE:-${LOG_DIR}/train_snn_subset_sim.log}"
ENV_LOG_FILE="${ENV_LOG_FILE:-${LOG_DIR}/train_env_setup.log}"

AUTO_RESUME="${AUTO_RESUME:-1}"

ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT_ANN="${WANDB_PROJECT_ANN:-dsec_subset_ann}"
WANDB_PROJECT_SNN="${WANDB_PROJECT_SNN:-dsec_subset_snn_sim}"
WANDB_RUN_NAME_ANN="${WANDB_RUN_NAME_ANN:-}"
WANDB_RUN_NAME_SNN="${WANDB_RUN_NAME_SNN:-}"

if [ "${RECREATE_VENV}" = "1" ] && [ -d "${VENV_DIR}" ]; then
  echo "[setup] removing existing venv: ${VENV_DIR}" | tee -a "${ENV_LOG_FILE}"
  rm -rf "${VENV_DIR}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] creating venv: ${VENV_DIR}" | tee -a "${ENV_LOG_FILE}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

{
  echo "[setup] python: $(python --version 2>&1)"
  echo "[setup] pip: $(pip --version 2>&1)"
  echo "[setup] upgrading pip/setuptools/wheel"
} | tee -a "${ENV_LOG_FILE}"

pip install --upgrade pip setuptools wheel >>"${ENV_LOG_FILE}" 2>&1

if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then
  echo "[setup] installing requirements.txt" | tee -a "${ENV_LOG_FILE}"
  pip install -r requirements.txt >>"${ENV_LOG_FILE}" 2>&1
fi

echo "[setup] reinstalling torch packages from ${TORCH_INDEX_URL}" | tee -a "${ENV_LOG_FILE}"
# shellcheck disable=SC2086
pip install --force-reinstall --no-cache-dir --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} >>"${ENV_LOG_FILE}" 2>&1

python - "${REQUIRE_CUDA}" <<'PY' | tee -a "${ENV_LOG_FILE}"
import sys
import torch

require_cuda = sys.argv[1] == "1"
print(f"[setup] torch={torch.__version__}")
print(f"[setup] cuda_available={torch.cuda.is_available()}")
print(f"[setup] cuda_device_count={torch.cuda.device_count()}")

if require_cuda and not torch.cuda.is_available():
    raise SystemExit("[setup] ERROR: CUDA required but not available.")
PY

mkdir -p "${ANN_SAVE_DIR}" "${SNN_SAVE_DIR}"

ann_cmd=(
  python scripts/train_ann_dsec.py
  --dsec-root "${DSEC_ROOT}"
  --train-split "${TRAIN_SPLIT}"
  --val-split "${VAL_SPLIT}"
  --train-sequences "${TRAIN_SEQUENCES}"
  --val-sequences "${VAL_SEQUENCES}"
  --class-ids "${CLASS_IDS}"
  --epochs "${ANN_EPOCHS}"
  --batch-size "${ANN_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --input-height "${INPUT_HEIGHT}"
  --input-width "${INPUT_WIDTH}"
  --save-dir "${ANN_SAVE_DIR}"
  --device cuda
)

snn_cmd=(
  python scripts/train_snn_dsec.py
  --dsec-root "${DSEC_ROOT}"
  --train-split "${TRAIN_SPLIT}"
  --val-split "${VAL_SPLIT}"
  --train-sequences "${TRAIN_SEQUENCES}"
  --val-sequences "${VAL_SEQUENCES}"
  --class-ids "${CLASS_IDS}"
  --event-source simulated
  --event-relpath "${SIM_EVENT_RELPATH}"
  --simulated-fps "${SIM_FPS}"
  --sequence-length "${SNN_SEQUENCE_LENGTH}"
  --sequence-stride "${SNN_SEQUENCE_STRIDE}"
  --epochs "${SNN_EPOCHS}"
  --batch-size "${SNN_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --input-height "${INPUT_HEIGHT}"
  --input-width "${INPUT_WIDTH}"
  --save-dir "${SNN_SAVE_DIR}"
  --device cuda
)

if [ "${AUTO_RESUME}" = "1" ]; then
  ann_ckpt="${ANN_SAVE_DIR}/vgg11_ssd_ann_dsec_best.pth"
  snn_ckpt="${SNN_SAVE_DIR}/vgg11_ssd_snn_dsec_best.pth"
  if [ -f "${ann_ckpt}" ]; then
    ann_cmd+=(--resume "${ann_ckpt}")
  fi
  if [ -f "${snn_ckpt}" ]; then
    snn_cmd+=(--resume "${snn_ckpt}")
  fi
fi

if [ "${ENABLE_WANDB}" = "1" ]; then
  ann_cmd+=(--wandb --wandb-project "${WANDB_PROJECT_ANN}")
  snn_cmd+=(--wandb --wandb-project "${WANDB_PROJECT_SNN}")
  if [ -n "${WANDB_RUN_NAME_ANN}" ]; then
    ann_cmd+=(--wandb-run-name "${WANDB_RUN_NAME_ANN}")
  fi
  if [ -n "${WANDB_RUN_NAME_SNN}" ]; then
    snn_cmd+=(--wandb-run-name "${WANDB_RUN_NAME_SNN}")
  fi
fi

echo "[launch] ANN on GPU ${ANN_GPU} -> ${ANN_LOG_FILE}"
nohup env CUDA_VISIBLE_DEVICES="${ANN_GPU}" "${ann_cmd[@]}" >"${ANN_LOG_FILE}" 2>&1 &
ann_pid=$!

echo "[launch] SNN(sim) on GPU ${SNN_GPU} -> ${SNN_LOG_FILE}"
nohup env CUDA_VISIBLE_DEVICES="${SNN_GPU}" "${snn_cmd[@]}" >"${SNN_LOG_FILE}" 2>&1 &
snn_pid=$!

echo "ann_pid=${ann_pid}"
echo "snn_pid=${snn_pid}"
echo "env_log=${ENV_LOG_FILE}"
echo "ann_log=${ANN_LOG_FILE}"
echo "snn_log=${SNN_LOG_FILE}"
echo "tail -f ${ANN_LOG_FILE}"
echo "tail -f ${SNN_LOG_FILE}"

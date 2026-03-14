#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

HOST_TAG="$(hostname | cut -d'.' -f1)"
VENV_BASENAME_DEFAULT=".venv_train_${HOST_TAG}"
VENV_DIR="${VENV_DIR:-${VENV_BASENAME_DEFAULT}}"
CREATE_NEW_VENV="${CREATE_NEW_VENV:-1}"      # 1 = always create a fresh venv without deleting old ones
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"            # 1 = fail if CUDA not available
REINSTALL_TORCH="${REINSTALL_TORCH:-1}"      # 1 = force reinstall torch each launch

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
CHECK_SIM_EVENTS="${CHECK_SIM_EVENTS:-1}"
MIN_EVENT_BYTES="${MIN_EVENT_BYTES:-32}"
MIN_EVENT_LINES="${MIN_EVENT_LINES:-1}"

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
RUN_ANN="${RUN_ANN:-1}"
RUN_SNN="${RUN_SNN:-1}"

parse_csv() {
  local v="$1"
  v="${v// /}"
  if [ -z "$v" ]; then
    return 0
  fi
  tr ',' '\n' <<< "$v" | sed '/^$/d'
}

count_non_comment_lines() {
  local f="$1"
  if [ ! -f "${f}" ]; then
    echo 0
    return 0
  fi
  awk '!/^[[:space:]]*#/{c++} END{print c+0}' "${f}" 2>/dev/null || echo 0
}

if [ "${RUN_ANN}" != "1" ] && [ "${RUN_SNN}" != "1" ]; then
  echo "ERROR: both RUN_ANN and RUN_SNN are disabled." | tee -a "${ENV_LOG_FILE}"
  exit 1
fi

if [ "${CREATE_NEW_VENV}" = "1" ] && [ -d "${VENV_DIR}" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  VENV_DIR="${VENV_DIR}_${stamp}"
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] creating venv: ${VENV_DIR}" | tee -a "${ENV_LOG_FILE}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

{
  echo "[setup] create_new_venv=${CREATE_NEW_VENV}"
  echo "[setup] venv_dir=${VENV_DIR}"
  echo "[setup] python: $(python --version 2>&1)"
  echo "[setup] pip: $(pip --version 2>&1)"
  echo "[setup] upgrading pip/setuptools/wheel"
} | tee -a "${ENV_LOG_FILE}"

pip install --upgrade pip setuptools wheel >>"${ENV_LOG_FILE}" 2>&1

if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then
  echo "[setup] installing requirements.txt" | tee -a "${ENV_LOG_FILE}"
  pip install -r requirements.txt >>"${ENV_LOG_FILE}" 2>&1
fi

if [ "${REINSTALL_TORCH}" = "1" ]; then
  echo "[setup] reinstalling torch packages from ${TORCH_INDEX_URL}" | tee -a "${ENV_LOG_FILE}"
  # shellcheck disable=SC2086
  pip install --force-reinstall --no-cache-dir --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES} >>"${ENV_LOG_FILE}" 2>&1
else
  echo "[setup] skipping torch reinstall (REINSTALL_TORCH=0)" | tee -a "${ENV_LOG_FILE}"
fi

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

if [ "${RUN_SNN}" = "1" ] && [ "${CHECK_SIM_EVENTS}" = "1" ]; then
  missing=0
  checked=0
  while IFS= read -r seq; do
    [ -n "${seq}" ] || continue
    p="${DSEC_ROOT}/${TRAIN_SPLIT}/${seq}/${SIM_EVENT_RELPATH}"
    checked=$((checked + 1))
    if [ ! -f "${p}" ]; then
      echo "[preflight] missing simulated events file: ${p}" | tee -a "${ENV_LOG_FILE}"
      missing=$((missing + 1))
      continue
    fi
    size="$(stat -c%s "${p}" 2>/dev/null || echo 0)"
    if [ "${size}" -lt "${MIN_EVENT_BYTES}" ]; then
      echo "[preflight] simulated events too small (${size} B): ${p}" | tee -a "${ENV_LOG_FILE}"
      missing=$((missing + 1))
      continue
    fi
    lines="$(count_non_comment_lines "${p}")"
    if [ "${lines}" -lt "${MIN_EVENT_LINES}" ]; then
      echo "[preflight] simulated events has no data rows (${lines}): ${p}" | tee -a "${ENV_LOG_FILE}"
      missing=$((missing + 1))
    fi
  done < <(parse_csv "${TRAIN_SEQUENCES}")

  while IFS= read -r seq; do
    [ -n "${seq}" ] || continue
    p="${DSEC_ROOT}/${VAL_SPLIT}/${seq}/${SIM_EVENT_RELPATH}"
    checked=$((checked + 1))
    if [ ! -f "${p}" ]; then
      echo "[preflight] missing simulated events file: ${p}" | tee -a "${ENV_LOG_FILE}"
      missing=$((missing + 1))
      continue
    fi
    size="$(stat -c%s "${p}" 2>/dev/null || echo 0)"
    if [ "${size}" -lt "${MIN_EVENT_BYTES}" ]; then
      echo "[preflight] simulated events too small (${size} B): ${p}" | tee -a "${ENV_LOG_FILE}"
      missing=$((missing + 1))
      continue
    fi
    lines="$(count_non_comment_lines "${p}")"
    if [ "${lines}" -lt "${MIN_EVENT_LINES}" ]; then
      echo "[preflight] simulated events has no data rows (${lines}): ${p}" | tee -a "${ENV_LOG_FILE}"
      missing=$((missing + 1))
    fi
  done < <(parse_csv "${VAL_SEQUENCES}")

  echo "[preflight] checked_sim_event_files=${checked} missing_or_small=${missing}" | tee -a "${ENV_LOG_FILE}"
  if [ "${missing}" -gt 0 ]; then
    echo "[preflight] ERROR: simulated event files are missing on this machine." | tee -a "${ENV_LOG_FILE}"
    echo "[preflight] sync/download them first, then re-run." | tee -a "${ENV_LOG_FILE}"
    exit 1
  fi
fi

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

ann_pid=""
if [ "${RUN_ANN}" = "1" ]; then
  echo "[launch] ANN on GPU ${ANN_GPU} -> ${ANN_LOG_FILE}"
  nohup env CUDA_VISIBLE_DEVICES="${ANN_GPU}" "${ann_cmd[@]}" >"${ANN_LOG_FILE}" 2>&1 &
  ann_pid=$!
fi

snn_pid=""
if [ "${RUN_SNN}" = "1" ]; then
  echo "[launch] SNN(sim) on GPU ${SNN_GPU} -> ${SNN_LOG_FILE}"
  nohup env CUDA_VISIBLE_DEVICES="${SNN_GPU}" "${snn_cmd[@]}" >"${SNN_LOG_FILE}" 2>&1 &
  snn_pid=$!
fi

if [ -n "${ann_pid}" ]; then
  echo "ann_pid=${ann_pid}"
fi
if [ -n "${snn_pid}" ]; then
  echo "snn_pid=${snn_pid}"
fi
echo "venv_dir=${VENV_DIR}"
echo "env_log=${ENV_LOG_FILE}"
if [ "${RUN_ANN}" = "1" ]; then
  echo "ann_log=${ANN_LOG_FILE}"
  echo "tail -f ${ANN_LOG_FILE}"
fi
if [ "${RUN_SNN}" = "1" ]; then
  echo "snn_log=${SNN_LOG_FILE}"
  echo "tail -f ${SNN_LOG_FILE}"
fi

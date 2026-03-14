#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/watch_snn_when_ready.log}"

POLL_SECS="${POLL_SECS:-300}"
MIN_EVENT_LINES="${MIN_EVENT_LINES:-1}"

DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-train}"
TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b}"
VAL_SEQUENCES="${VAL_SEQUENCES:-zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a}"
SIM_EVENT_RELPATH="${SIM_EVENT_RELPATH:-v2e_output_distorted_gray_clean/dvs_events.txt}"
SNN_SAVE_DIR="${SNN_SAVE_DIR:-checkpoints/dsec_subset_snn_sim}"

# Launch settings forwarded to training launcher
ANN_GPU="${ANN_GPU:-0}"
SNN_GPU="${SNN_GPU:-1}"
CREATE_NEW_VENV="${CREATE_NEW_VENV:-0}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
REINSTALL_TORCH="${REINSTALL_TORCH:-0}"
CHECK_SIM_EVENTS="${CHECK_SIM_EVENTS:-1}"
ENABLE_WANDB="${ENABLE_WANDB:-0}"
WANDB_PROJECT_SNN="${WANDB_PROJECT_SNN:-dsec_subset_snn_sim}"
WANDB_RUN_NAME_SNN="${WANDB_RUN_NAME_SNN:-}"

parse_csv() {
  local v="$1"
  v="${v// /}"
  if [ -z "$v" ]; then
    return 0
  fi
  tr ',' '\n' <<< "$v" | sed '/^$/d'
}

event_rows() {
  local f="$1"
  if [ ! -f "$f" ]; then
    echo 0
    return 0
  fi
  awk '!/^[[:space:]]*#/{c++} END{print c+0}' "$f" 2>/dev/null || echo 0
}

is_snn_running() {
  pgrep -fa "python scripts/train_snn_dsec.py" | grep -q -- "--save-dir ${SNN_SAVE_DIR}"
}

check_ready() {
  local missing=0
  local checked=0

  while IFS= read -r seq; do
    [ -n "$seq" ] || continue
    f="${DSEC_ROOT}/${TRAIN_SPLIT}/${seq}/${SIM_EVENT_RELPATH}"
    rows="$(event_rows "$f")"
    checked=$((checked + 1))
    if [ "$rows" -lt "$MIN_EVENT_LINES" ]; then
      echo "[watch] not-ready train seq=${seq} rows=${rows} file=${f}" | tee -a "${LOG_FILE}"
      missing=$((missing + 1))
    fi
  done < <(parse_csv "${TRAIN_SEQUENCES}")

  while IFS= read -r seq; do
    [ -n "$seq" ] || continue
    f="${DSEC_ROOT}/${VAL_SPLIT}/${seq}/${SIM_EVENT_RELPATH}"
    rows="$(event_rows "$f")"
    checked=$((checked + 1))
    if [ "$rows" -lt "$MIN_EVENT_LINES" ]; then
      echo "[watch] not-ready val seq=${seq} rows=${rows} file=${f}" | tee -a "${LOG_FILE}"
      missing=$((missing + 1))
    fi
  done < <(parse_csv "${VAL_SEQUENCES}")

  echo "[watch] checked=${checked} missing_or_empty=${missing}" | tee -a "${LOG_FILE}"
  [ "$missing" -eq 0 ]
}

echo "[watch] started at $(date -Iseconds)" | tee -a "${LOG_FILE}"
echo "[watch] poll_secs=${POLL_SECS} dsec_root=${DSEC_ROOT}" | tee -a "${LOG_FILE}"

while true; do
  if is_snn_running; then
    echo "[watch] SNN already running for save_dir=${SNN_SAVE_DIR}; exiting." | tee -a "${LOG_FILE}"
    exit 0
  fi

  if check_ready; then
    echo "[watch] data ready; launching SNN at $(date -Iseconds)" | tee -a "${LOG_FILE}"
    env \
      RUN_ANN=0 \
      RUN_SNN=1 \
      DSEC_ROOT="${DSEC_ROOT}" \
      TRAIN_SPLIT="${TRAIN_SPLIT}" \
      VAL_SPLIT="${VAL_SPLIT}" \
      TRAIN_SEQUENCES="${TRAIN_SEQUENCES}" \
      VAL_SEQUENCES="${VAL_SEQUENCES}" \
      SIM_EVENT_RELPATH="${SIM_EVENT_RELPATH}" \
      SNN_SAVE_DIR="${SNN_SAVE_DIR}" \
      ANN_GPU="${ANN_GPU}" \
      SNN_GPU="${SNN_GPU}" \
      CREATE_NEW_VENV="${CREATE_NEW_VENV}" \
      INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS}" \
      REINSTALL_TORCH="${REINSTALL_TORCH}" \
      CHECK_SIM_EVENTS="${CHECK_SIM_EVENTS}" \
      ENABLE_WANDB="${ENABLE_WANDB}" \
      WANDB_PROJECT_SNN="${WANDB_PROJECT_SNN}" \
      WANDB_RUN_NAME_SNN="${WANDB_RUN_NAME_SNN}" \
      bash scripts/night_runs/start_machine_b_train_subset_parallel.sh | tee -a "${LOG_FILE}"

    echo "[watch] launch command returned; exiting watcher." | tee -a "${LOG_FILE}"
    exit 0
  fi

  echo "[watch] sleeping ${POLL_SECS}s" | tee -a "${LOG_FILE}"
  sleep "${POLL_SECS}"
done

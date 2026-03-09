#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/machine_c_targeted.log}"
mkdir -p "${LOG_DIR}"

# 8-sequence simulated-train subset chosen to cover all available location buckets
# and spread Zurich across early/mid/late sequence ids.
TRAIN_SIM_SEQUENCES_DEFAULT="thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b"

# Keep the official validation sequences intact for simulated-event validation.
VAL_SIM_SEQUENCES_DEFAULT="zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a"

TRAIN_SIM_SEQUENCES="${TRAIN_SIM_SEQUENCES:-${TRAIN_SIM_SEQUENCES_DEFAULT}}"
VAL_SIM_SEQUENCES="${VAL_SIM_SEQUENCES:-${VAL_SIM_SEQUENCES_DEFAULT}}"

export DSEC_ROOT="${DSEC_ROOT:-../data/dsec}"
export TRAINVAL_SPLITS="${TRAINVAL_SPLITS:-train}"
export TEST_SPLITS="${TEST_SPLITS:-test}"
export TRAINVAL_SEQUENCES="${TRAINVAL_SEQUENCES:-${TRAIN_SIM_SEQUENCES},${VAL_SIM_SEQUENCES}}"
export TEST_SEQUENCES="${TEST_SEQUENCES:-}"
export TRAINVAL_V2E_MODES="${TRAINVAL_V2E_MODES:-clean}"
export TRAINVAL_MANUAL_V2E_NOISES="${TRAINVAL_MANUAL_V2E_NOISES:-none}"
export TEST_V2E_MODES="${TEST_V2E_MODES:-clean,noisy}"
export TEST_MANUAL_V2E_NOISES="${TEST_MANUAL_V2E_NOISES:-shot_noise,leak_noise,threshold_jitter,bandwidth_limit,refractory,photoreceptor_noise}"
export TEST_SEVERITIES="${TEST_SEVERITIES:-1,2,3,4,5}"
export TEST_CORRUPTION_SUBSET="${TEST_CORRUPTION_SUBSET:-all}"
export INPUT_FPS="${INPUT_FPS:-20}"
export V2E_EXPOSURE_DURATION="${V2E_EXPOSURE_DURATION:-0.005}"
export JOBS="${JOBS:-2}"
export POLL_SECS="${POLL_SECS:-900}"
export VALIDATE_RECTIFICATION="${VALIDATE_RECTIFICATION:-0}"

if [ "${FOREGROUND:-0}" = "1" ]; then
  echo "[start_machine_c_targeted] foreground=1 log=${LOG_FILE}"
  bash scripts/night_runs/machine_c_convert_loop.sh 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

nohup bash scripts/night_runs/machine_c_convert_loop.sh >"${LOG_FILE}" 2>&1 &
pid=$!
echo "pid=${pid}"
echo "log=${LOG_FILE}"
echo "tail -f ${LOG_FILE}"

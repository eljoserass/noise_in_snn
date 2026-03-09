#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
EVENT_SOURCE="${EVENT_SOURCE:-real}"   # real | simulated
RUN_TAG="${RUN_TAG:-${EVENT_SOURCE}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/machine_b_train_${RUN_TAG}.log}"
mkdir -p "${LOG_DIR}"

TRAIN_SEQUENCES_DEFAULT="thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b"
VAL_SEQUENCES_DEFAULT="zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a"

export DSEC_ROOT="${DSEC_ROOT:-../data/dsec}"
export TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
export VAL_SPLIT="${VAL_SPLIT:-train}"
export TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-${TRAIN_SEQUENCES_DEFAULT}}"
export VAL_SEQUENCES="${VAL_SEQUENCES:-${VAL_SEQUENCES_DEFAULT}}"
export EVENT_SOURCE="${EVENT_SOURCE}"
export SAVE_DIR="${SAVE_DIR:-checkpoints/dsec_subset_${RUN_TAG}}"
export WANDB_PROJECT="${WANDB_PROJECT:-dsec_subset_${RUN_TAG}}"

if [ "${FOREGROUND:-0}" = "1" ]; then
  echo "[start_machine_b_train_subset] foreground=1 log=${LOG_FILE}"
  bash scripts/night_runs/machine_b_train.sh 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

nohup bash scripts/night_runs/machine_b_train.sh >"${LOG_FILE}" 2>&1 &
pid=$!
echo "pid=${pid}"
echo "log=${LOG_FILE}"
echo "tail -f ${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

# Disposable launcher for DSEC training on Machine B.
# Usage:
#   bash scripts/night_runs/run_dsec_train_now.sh
# Optional env overrides before running:
#   export DSEC_ROOT=../data/dsec
#   export WANDB_PROJECT=dsec_baselines_v1
#   export DEVICE=cuda
#   export USE_WANDB=1

cd "$(dirname "$0")/../.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

DSEC_ROOT="${DSEC_ROOT:-../data/dsec}"
WANDB_PROJECT="${WANDB_PROJECT:-dsec_baselines_v1}"
USE_WANDB="${USE_WANDB:-1}"
DEVICE="${DEVICE:-cuda}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-train}"
SPLIT_CONFIG_PATH="${SPLIT_CONFIG_PATH:-config/dsec_train_val_test_split.yaml}"

# If VAL_SEQUENCES is not provided, try to parse from dsec-det split config.
if [ -z "${VAL_SEQUENCES:-}" ] && [ -f "$SPLIT_CONFIG_PATH" ]; then
  VAL_SEQUENCES="$(awk '$1=="val:"{f=1;next} f&&$1=="test:"{f=0} f&&$1=="-"{print $2}' "$SPLIT_CONFIG_PATH" | paste -sd "," -)"
fi

# Fallback known val list if config is missing/unavailable.
if [ -z "${VAL_SEQUENCES:-}" ]; then
  VAL_SEQUENCES="zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a"
fi

echo "[run_dsec_train_now] DSEC_ROOT=$DSEC_ROOT"
echo "[run_dsec_train_now] TRAIN_SPLIT=$TRAIN_SPLIT"
echo "[run_dsec_train_now] VAL_SPLIT=$VAL_SPLIT"
echo "[run_dsec_train_now] VAL_SEQUENCES=$VAL_SEQUENCES"
echo "[run_dsec_train_now] WANDB_PROJECT=$WANDB_PROJECT"

DSEC_ROOT="$DSEC_ROOT" \
TRAIN_SPLIT="$TRAIN_SPLIT" \
VAL_SPLIT="$VAL_SPLIT" \
VAL_SEQUENCES="$VAL_SEQUENCES" \
USE_WANDB="$USE_WANDB" \
WANDB_PROJECT="$WANDB_PROJECT" \
DEVICE="$DEVICE" \
bash scripts/night_runs/machine_b_train.sh

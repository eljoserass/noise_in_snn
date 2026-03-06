#!/usr/bin/env bash
set -euo pipefail

show_usage() {
  cat <<'USAGE'
Usage: ./run_dsec.sh <stage> [args...]

Stages:
  train-ann       Run DSEC ANN training (scripts/train_ann_dsec.py)
  train-snn       Run DSEC SNN training (scripts/train_snn_dsec.py)
  eval            Run DSEC evaluation (scripts/evaluate_dsec.py)
  preview         Render DSEC train-view sanity images (scripts/preview_dsec_training_view.py)
  night-train     Start resume-safe DSEC night loop (scripts/night_runs/machine_b_train.sh)
  night-download  Start DSEC R2 download loop (scripts/night_runs/machine_a_download.sh)
  night-convert   Start DSEC conversion loop (scripts/night_runs/machine_c_convert_loop.sh)

Examples:
  ./run_dsec.sh train-ann --dsec-root ../data/dsec --train-split train --val-split val
  ./run_dsec.sh train-snn --dsec-root ../data/dsec --event-source real
  ./run_dsec.sh eval --model-path checkpoints/vgg11_ssd_ann_dsec_best.pth --model-type ann
  ./run_dsec.sh night-train
USAGE
}

if [ $# -eq 0 ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  show_usage
  exit 0
fi

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

stage="$1"
shift || true

case "$stage" in
  train-ann)
    python scripts/train_ann_dsec.py "$@"
    ;;
  train-snn)
    python scripts/train_snn_dsec.py "$@"
    ;;
  eval)
    python scripts/evaluate_dsec.py "$@"
    ;;
  preview)
    python scripts/preview_dsec_training_view.py "$@"
    ;;
  night-train)
    bash scripts/night_runs/machine_b_train.sh "$@"
    ;;
  night-download)
    bash scripts/night_runs/machine_a_download.sh "$@"
    ;;
  night-convert)
    bash scripts/night_runs/machine_c_convert_loop.sh "$@"
    ;;
  *)
    echo "Unknown stage: $stage"
    show_usage
    exit 1
    ;;
esac

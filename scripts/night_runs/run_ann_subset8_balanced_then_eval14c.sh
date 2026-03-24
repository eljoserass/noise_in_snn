#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

EXPECTED_HOSTNAME="${EXPECTED_HOSTNAME:-5de37e9747f2}"
ACTUAL_HOSTNAME="$(hostname)"
if [ "$ACTUAL_HOSTNAME" != "$EXPECTED_HOSTNAME" ]; then
  echo "[abort] wrong host: expected=$EXPECTED_HOSTNAME actual=$ACTUAL_HOSTNAME"
  exit 2
fi

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
if [ -n "${WANBD_API_KEY:-}" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY="$WANBD_API_KEY"
fi

pick_working_venv() {
  for cand in ${VENV_DIR:-} .venv_machine_c .venv_train_* .venv_machine_* .venv; do
    [ -n "$cand" ] || continue
    [ -x "$cand/bin/python" ] || continue
    if "$cand/bin/python" - <<"PY" >/dev/null 2>&1
import importlib.util, sys
ok = importlib.util.find_spec("numpy") is not None and importlib.util.find_spec("torch") is not None
sys.exit(0 if ok else 1)
PY
    then
      echo "$cand"
      return 0
    fi
  done
  return 1
}

if VENV_PICKED="$(pick_working_venv)"; then
  # shellcheck disable=SC1090
  source "$VENV_PICKED/bin/activate"
else
  VENV_PICKED=".venv_ann_eval_${ACTUAL_HOSTNAME}"
  python3 -m venv "$VENV_PICKED"
  # shellcheck disable=SC1090
  source "$VENV_PICKED/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="ann_subset8_balanced_gray_${ACTUAL_HOSTNAME}_${RUN_TS}"
SAVE_DIR="checkpoints/${RUN_TAG}"
RESULT_ROOT="results/noise_benchmark/${RUN_TAG}__noise_eval_seqagg"
PIPE_LOG="logs/${RUN_TAG}.pipeline.log"
TRAIN_LOG="logs/${RUN_TAG}.train.log"

mkdir -p logs "$SAVE_DIR" "$RESULT_ROOT"

TRAIN_SEQS="thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b"
VAL_SEQS="zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a"
CLASS_IDS="0,1,2,3,4,5,6,7"

{
  echo "[start] $(date -Is)"
  echo "[host] $ACTUAL_HOSTNAME"
  echo "[venv] $VENV_PICKED"
  echo "[run_tag] $RUN_TAG"
  echo "[save_dir] $SAVE_DIR"
  echo "[result_root] $RESULT_ROOT"
  echo "[train_sequences] $TRAIN_SEQS"
  echo "[val_sequences] $VAL_SEQS"
} | tee -a "$PIPE_LOG"

TRAIN_CMD=(
  python scripts/train_ann_dsec.py
  --dsec-root /workspace/data/dsec
  --train-split train
  --val-split train
  --train-sequences "$TRAIN_SEQS"
  --val-sequences "$VAL_SEQS"
  --image-relpath images/left/distorted_gray
  --class-ids "$CLASS_IDS"
  --epochs 200
  --batch-size 16
  --num-workers 8
  --input-height 480
  --input-width 640
  --save-dir "$SAVE_DIR"
  --device cuda
  --class-balance
  --class-balance-bg-weight 0.25
  --class-balance-power 0.5
  --class-balance-min 0.25
  --class-balance-max 8.0
  --wandb
  --wandb-project dsec_subset_ann
  --wandb-run-name "${RUN_TAG}__train"
)

echo "[train] launching" | tee -a "$PIPE_LOG"
"${TRAIN_CMD[@]}" 2>&1 | tee "$TRAIN_LOG"

BEST_CKPT="$SAVE_DIR/vgg11_ssd_ann_dsec_best.pth"
if [ ! -f "$BEST_CKPT" ]; then
  echo "[error] missing checkpoint: $BEST_CKPT" | tee -a "$PIPE_LOG"
  exit 3
fi

echo "[train] done; checkpoint=$BEST_CKPT" | tee -a "$PIPE_LOG"

noise_order=(snow shot_noise gaussian_noise impulse_noise glass_blur fog frost motion_blur)
sev_order=(3 1 5 2 4)

conditions=(clean_s0)
for noise in "${noise_order[@]}"; do
  for s in "${sev_order[@]}"; do
    conditions+=("${noise}_s${s}")
  done
done

eval_one() {
  local cond="$1"
  local relpath
  if [ "$cond" = "clean_s0" ]; then
    relpath="images/left/distorted_gray"
  else
    local noise="${cond%_s*}"
    local sev="${cond##*_s}"
    relpath="images/left/distorted_gray_${noise}_s${sev}"
  fi

  local out_dir="$RESULT_ROOT/ann_rgb_${cond}/seq_zurich_city_14_c"
  mkdir -p "$out_dir"

  echo "[eval] start cond=${cond} relpath=${relpath}" | tee -a "$PIPE_LOG"
  if python scripts/evaluate_dsec.py \
    --model-path "$BEST_CKPT" \
    --model-type ann \
    --dsec-root /workspace/data/dsec \
    --test-splits test \
    --sequences zurich_city_14_c \
    --image-relpath "$relpath" \
    --class-ids "$CLASS_IDS" \
    --input-height 480 \
    --input-width 640 \
    --num-workers 8 \
    --device cuda \
    --conf-threshold 0.2 \
    --nms-threshold 0.5 \
    --output-dir "$out_dir" \
    --wandb \
    --wandb-project dsec_noise_benchmark \
    --wandb-run-name "${RUN_TAG}__ann_rgb__${cond}__zurich_city_14_c"; then
    echo "[eval] ok ${cond}" | tee -a "$PIPE_LOG"
  else
    echo "[eval] fail ${cond}" | tee -a "$PIPE_LOG"
  fi
}

for cond in "${conditions[@]}"; do
  eval_one "$cond"
done

echo "[done] $(date -Is)" | tee -a "$PIPE_LOG"

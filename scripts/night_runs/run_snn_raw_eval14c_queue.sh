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
  # shellcheck disable=SC1091
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
ok = (
    importlib.util.find_spec("numpy") is not None
    and importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("h5py") is not None
)
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
  VENV_PICKED=".venv_snn_eval_${ACTUAL_HOSTNAME}"
  python3 -m venv "$VENV_PICKED"
  # shellcheck disable=SC1090
  source "$VENV_PICKED/bin/activate"
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="snn_raw_eval14c_${ACTUAL_HOSTNAME}_${RUN_TS}"

SNN_CKPT="${SNN_CKPT:-checkpoints/snn_raw_train_eval_20260321_170306/vgg11_ssd_snn_dsec_best.pth}"
RESULT_ROOT="results/noise_benchmark/${RUN_TAG}__noise_eval_seqagg"
PIPE_LOG="logs/${RUN_TAG}.pipeline.log"

SEQ="${SEQ:-zurich_city_14_c}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"

SEQ_LEN="${SEQ_LEN:-8}"
SEQ_STRIDE="${SEQ_STRIDE:-8}"
TIMESTEPS="${TIMESTEPS:-5}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-4}"

mkdir -p logs "$RESULT_ROOT"

if [ ! -f "$SNN_CKPT" ]; then
  echo "[error] checkpoint not found: $SNN_CKPT"
  exit 3
fi

{
  echo "[start] $(date -Is)"
  echo "[host] $ACTUAL_HOSTNAME"
  echo "[venv] $VENV_PICKED"
  echo "[run_tag] $RUN_TAG"
  echo "[checkpoint] $SNN_CKPT"
  echo "[result_root] $RESULT_ROOT"
  echo "[sequence] $SEQ"
  echo "[params] seq_len=$SEQ_LEN seq_stride=$SEQ_STRIDE timesteps=$TIMESTEPS conf=$CONF_THRESHOLD workers=$NUM_WORKERS"
} | tee -a "$PIPE_LOG"

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

  local out_dir="$RESULT_ROOT/snn_raw_${cond}/seq_${SEQ}"
  mkdir -p "$out_dir"

  echo "[eval] start cond=${cond} relpath=${relpath}" | tee -a "$PIPE_LOG"
  if python scripts/evaluate_dsec.py \
    --model-path "$SNN_CKPT" \
    --model-type snn \
    --dsec-root /workspace/data/dsec \
    --test-splits test \
    --sequences "$SEQ" \
    --image-relpath "$relpath" \
    --class-ids "$CLASS_IDS" \
    --event-source real \
    --event-relpath events/left/events.h5 \
    --sequence-length "$SEQ_LEN" \
    --sequence-stride "$SEQ_STRIDE" \
    --timesteps-per-frame "$TIMESTEPS" \
    --input-height 480 \
    --input-width 640 \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --conf-threshold "$CONF_THRESHOLD" \
    --nms-threshold 0.5 \
    --output-dir "$out_dir" \
    --wandb \
    --wandb-project dsec_noise_benchmark \
    --wandb-run-name "${RUN_TAG}__snn_raw__${cond}__${SEQ}"; then
    echo "[eval] ok ${cond}" | tee -a "$PIPE_LOG"
  else
    echo "[eval] fail ${cond}" | tee -a "$PIPE_LOG"
  fi
}

for cond in "${conditions[@]}"; do
  eval_one "$cond"
done

echo "[done] $(date -Is)" | tee -a "$PIPE_LOG"


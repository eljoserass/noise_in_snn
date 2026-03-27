#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

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
mods = ("numpy", "torch", "h5py", "torchvision")
ok = all(importlib.util.find_spec(m) is not None for m in mods)
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
  echo "[error] no working venv found" >&2
  exit 2
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
HOST="$(hostname)"
RUN_TAG="${RUN_TAG:-snn_sim_best_eval14c_${HOST}_${RUN_TS}}"
DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
SEQ="${SEQ:-zurich_city_14_c}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"
SNN_CKPT="${SNN_CKPT:-checkpoints/snn_sim_tpf5_host14929_20260322_214922/vgg11_ssd_snn_dsec_best.pth}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEQ_LEN="${SEQ_LEN:-8}"
SEQ_STRIDE="${SEQ_STRIDE:-8}"
TIMESTEPS="${TIMESTEPS:-5}"
BETA="${BETA:-0.9}"
THRESHOLD="${THRESHOLD:-1.0}"
SLOPE="${SLOPE:-25.0}"
INCLUDE_CLEAN="${INCLUDE_CLEAN:-1}"
NOISE_TYPES="${NOISE_TYPES:-snow,shot_noise,gaussian_noise,impulse_noise,glass_blur,motion_blur,fog,frost}"
SEVERITIES="${SEVERITIES:-3,1,5,2,4}"
WANDB_PROJECT="${WANDB_PROJECT:-dsec_noise_benchmark}"

RESULT_ROOT="${RESULT_ROOT:-results/noise_benchmark/${RUN_TAG}__noise_eval_seqagg}"
LOG_FILE="${LOG_FILE:-logs/${RUN_TAG}.log}"
mkdir -p "$(dirname "$LOG_FILE")" "$RESULT_ROOT"

if [ ! -f "$SNN_CKPT" ]; then
  echo "[error] checkpoint not found: $SNN_CKPT" | tee -a "$LOG_FILE"
  exit 3
fi

echo "[start] $(date -Is)" | tee -a "$LOG_FILE"
echo "[host] $HOST" | tee -a "$LOG_FILE"
echo "[venv] $VENV_PICKED" | tee -a "$LOG_FILE"
echo "[checkpoint] $SNN_CKPT" | tee -a "$LOG_FILE"
echo "[sequence] $SEQ" | tee -a "$LOG_FILE"
echo "[noise_types] $NOISE_TYPES" | tee -a "$LOG_FILE"
echo "[severities] $SEVERITIES" | tee -a "$LOG_FILE"
echo "[params] seq_len=$SEQ_LEN seq_stride=$SEQ_STRIDE timesteps=$TIMESTEPS conf=$CONF_THRESHOLD workers=$NUM_WORKERS" | tee -a "$LOG_FILE"

run_one() {
  local cond="$1"
  local image_rel event_rel out_dir
  if [ "$cond" = "clean_s0" ]; then
    image_rel="images/left/distorted_gray"
    event_rel="v2e_output_distorted_gray_clean/dvs_events.h5"
    out_dir="$RESULT_ROOT/snn_sim_clean_s0/seq_${SEQ}"
  else
    local noise="${cond%_s*}"
    local sev="${cond##*_s}"
    image_rel="images/left/distorted_gray_${noise}_s${sev}"
    event_rel="v2e_output_distorted_gray_${noise}_s${sev}_clean/dvs_events.h5"
    out_dir="$RESULT_ROOT/snn_sim_${noise}_s${sev}/seq_${SEQ}"
  fi

  local event_path="$DSEC_ROOT/test/$SEQ/$event_rel"
  local metrics_json="$out_dir/eval_dsec_snn_test.json"
  local run_name="${RUN_TAG}__snn_sim_best__${cond}__${SEQ}"

  if [ ! -f "$event_path" ]; then
    echo "[skip] ${cond} missing_event=${event_path}" | tee -a "$LOG_FILE"
    return 0
  fi
  if [ -s "$metrics_json" ]; then
    echo "[skip] ${cond} already_done=${metrics_json}" | tee -a "$LOG_FILE"
    return 0
  fi

  mkdir -p "$out_dir"
  echo "[run] ${cond}" | tee -a "$LOG_FILE"

  if python scripts/evaluate_dsec.py \
    --model-path "$SNN_CKPT" \
    --model-type snn \
    --dsec-root "$DSEC_ROOT" \
    --test-splits test \
    --sequences "$SEQ" \
    --class-ids "$CLASS_IDS" \
    --image-relpath "$image_rel" \
    --event-source simulated \
    --event-relpath "$event_rel" \
    --sequence-length "$SEQ_LEN" \
    --sequence-stride "$SEQ_STRIDE" \
    --timesteps-per-frame "$TIMESTEPS" \
    --beta "$BETA" \
    --threshold "$THRESHOLD" \
    --surrogate-slope "$SLOPE" \
    --input-height 480 \
    --input-width 640 \
    --batch-size 1 \
    --num-workers "$NUM_WORKERS" \
    --device cuda \
    --conf-threshold "$CONF_THRESHOLD" \
    --nms-threshold 0.5 \
    --output-dir "$out_dir" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "$run_name" >> "$LOG_FILE" 2>&1; then
    if [ -s "$metrics_json" ]; then
      local map_line
      map_line="$(python - <<PY
import json
p = "$metrics_json"
d = json.load(open(p))
print(f"mAP@0.5={d.get('mAP@0.5', 0.0):.6f} preds={d.get('num_predictions', -1)}")
PY
)"
      echo "[ok] ${cond} ${map_line}" | tee -a "$LOG_FILE"
    else
      echo "[ok] ${cond} (no metrics file?)" | tee -a "$LOG_FILE"
    fi
  else
    echo "[fail] ${cond}" | tee -a "$LOG_FILE"
    return 1
  fi
}

if [ "$INCLUDE_CLEAN" = "1" ]; then
  run_one "clean_s0"
fi

IFS=, read -r -a noise_arr <<< "$NOISE_TYPES"
IFS=, read -r -a sev_arr <<< "$SEVERITIES"
for noise in "${noise_arr[@]}"; do
  noise="${noise// /}"
  [ -n "$noise" ] || continue
  for sev in "${sev_arr[@]}"; do
    sev="${sev// /}"
    [ -n "$sev" ] || continue
    run_one "${noise}_s${sev}"
  done
done

echo "[done] $(date -Is)" | tee -a "$LOG_FILE"

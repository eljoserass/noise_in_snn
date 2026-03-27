#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CMD="${1:-status}"
shift || true

VENV_DIR="${VENV_DIR:-.venv_train_532b884b9814}"
DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
TEST_SPLIT="${TEST_SPLIT:-test}"
SEQ="${SEQ:-zurich_city_14_c}"

ANN_CKPT="${ANN_CKPT:-checkpoints/dsec_subset_ann/vgg11_ssd_ann_dsec_best.pth}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
CONF_THRESHOLD="${CONF_THRESHOLD:-0.2}"
NMS_THRESHOLD="${NMS_THRESHOLD:-0.5}"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dsec_noise_benchmark}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-ann_subset_8seq_14c_queue}"

# Accept typo alias and auto-load .env for wandb auth.
if [ -z "${WANDB_API_KEY:-}" ] && [ -n "${WANBD_API_KEY:-}" ]; then
  export WANDB_API_KEY="${WANBD_API_KEY}"
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env || true
  set +a
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ -n "${WANBD_API_KEY:-}" ]; then
  export WANDB_API_KEY="${WANBD_API_KEY}"
fi

RESULTS_ROOT="${RESULTS_ROOT:-results/noise_benchmark/ann_subset_balanced_queue_20260316_153448__noise_eval_seqagg}"
EXISTING_ROOT="${EXISTING_ROOT:-${RESULTS_ROOT}}"

QUEUE_FILE="${QUEUE_FILE:-logs/ann_eval_14c.queue}"
LOCK_FILE="${LOCK_FILE:-logs/ann_eval_14c.queue.lock}"
CURRENT_FILE="${CURRENT_FILE:-logs/ann_eval_14c.current}"
QUEUE_LOG="${QUEUE_LOG:-logs/ann_eval_14c.queue.log}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"
ALLOW_HIGH_PARALLEL="${ALLOW_HIGH_PARALLEL:-0}"

# User-requested priority:
# snow -> shot_noise -> gaussian_noise -> glass_blur -> rest
ORDER_CORRUPTIONS="${ORDER_CORRUPTIONS:-fog,frost,snow,shot_noise,gaussian_noise,impulse_noise,glass_blur,motion_blur,defocus_blur,zoom_blur,jpeg_compression,pixelate,brightness,contrast,elastic_transform}"
SEVERITIES="${SEVERITIES:-1,2,3,4,5}"
INCLUDE_CLEAN="${INCLUDE_CLEAN:-0}"

mkdir -p logs
touch "${QUEUE_FILE}" "${LOCK_FILE}" "${QUEUE_LOG}"

parse_csv() {
  local v="${1// /}"
  [ -z "${v}" ] && return 0
  local item
  IFS=, read -r -a items <<< "${v}"
  for item in "${items[@]}"; do
    [ -n "${item}" ] && echo "${item}"
  done
}

condition_to_relpath() {
  local cond="$1"
  if [ "${cond}" = "clean_s0" ]; then
    echo "images/left/distorted_gray"
    return 0
  fi
  local noise="${cond%_s*}"
  local sev="${cond##*_s}"
  echo "images/left/distorted_gray_${noise}_s${sev}"
}

condition_to_outdir() {
  local cond="$1"
  local noise="${cond%_s*}"
  local sev="${cond##*_s}"
  echo "${RESULTS_ROOT}/ann_rgb_${noise}_s${sev}/seq_${SEQ}"
}

condition_eval_exists() {
  local cond="$1"
  local noise="${cond%_s*}"
  local sev="${cond##*_s}"
  local p="${EXISTING_ROOT}/ann_rgb_${noise}_s${sev}/seq_${SEQ}/eval_dsec_ann_test.json"
  [ -s "${p}" ]
}

available_corruptions() {
  local root="${DSEC_ROOT}/${TEST_SPLIT}/${SEQ}/images/left"
  [ -d "${root}" ] || return 0
  ls -1 "${root}" 2>/dev/null \
    | sed -n 's/^distorted_gray_\(.*\)_s[1-5]$/\1/p' \
    | sort -u
}

all_conditions_in_order() {
  local c s
  if [ "${INCLUDE_CLEAN}" = "1" ]; then
    echo "clean_s0"
  fi

  while IFS= read -r c; do
    [ -n "${c}" ] || continue
    while IFS= read -r s; do
      [ -n "${s}" ] || continue
      echo "${c}_s${s}"
    done < <(parse_csv "${SEVERITIES}")
  done < <(parse_csv "${ORDER_CORRUPTIONS}")

  # Append any extra generated corruptions not listed in ORDER_CORRUPTIONS.
  local listed extra
  listed="$(parse_csv "${ORDER_CORRUPTIONS}" | tr '\n' ' ')"
  while IFS= read -r extra; do
    [ -n "${extra}" ] || continue
    case " ${listed} " in
      *" ${extra} "*) continue ;;
    esac
    while IFS= read -r s; do
      [ -n "${s}" ] || continue
      echo "${extra}_s${s}"
    done < <(parse_csv "${SEVERITIES}")
  done < <(available_corruptions)
}

queue_build() {
  local tmp
  tmp="$(mktemp)"
  > "${tmp}"
  local cond rel
  while IFS= read -r cond; do
    [ -n "${cond}" ] || continue
    rel="$(condition_to_relpath "${cond}")"
    [ -d "${DSEC_ROOT}/${TEST_SPLIT}/${SEQ}/${rel}" ] || continue
    condition_eval_exists "${cond}" && continue
    echo "${cond}" >> "${tmp}"
  done < <(all_conditions_in_order)

  {
    flock -x 9
    cp "${tmp}" "${QUEUE_FILE}"
  } 9>"${LOCK_FILE}"
  rm -f "${tmp}"
  echo "[build] queue rebuilt at ${QUEUE_FILE}" | tee -a "${QUEUE_LOG}"
  queue_status
}

queue_status() {
  local n
  n="$(grep -Ec '^[^#[:space:]].*$' "${QUEUE_FILE}" || true)"
  echo "[status] queue_items=${n}" | tee -a "${QUEUE_LOG}"
  if [ -s "${CURRENT_FILE}" ]; then
    echo "[status] running=$(cat "${CURRENT_FILE}")" | tee -a "${QUEUE_LOG}"
  else
    echo "[status] running=<none>" | tee -a "${QUEUE_LOG}"
  fi
  echo "[status] next10:" | tee -a "${QUEUE_LOG}"
  grep -E '^[^#[:space:]].*$' "${QUEUE_FILE}" | head -n 10 | tee -a "${QUEUE_LOG}" || true
}

queue_remove() {
  local cond="${1:-}"
  [ -n "${cond}" ] || { echo "Usage: $0 remove <condition>"; exit 1; }
  local tmp
  tmp="$(mktemp)"
  {
    flock -x 9
    awk -v c="${cond}" '($0!=c){print $0}' "${QUEUE_FILE}" > "${tmp}"
    cp "${tmp}" "${QUEUE_FILE}"
  } 9>"${LOCK_FILE}"
  rm -f "${tmp}"
  echo "[remove] removed=${cond}" | tee -a "${QUEUE_LOG}"
  queue_status
}

queue_clear() {
  : > "${QUEUE_FILE}"
  : > "${CURRENT_FILE}"
  echo "[clear] queue cleared" | tee -a "${QUEUE_LOG}"
}

pop_next_condition() {
  local out_var="$1"
  local tmp first
  tmp="$(mktemp)"
  first="$(
    awk '
      BEGIN{printed=0}
      {
        if(!printed && $0 ~ /^[^#[:space:]].*$/){
          print $0
          printed=1
          exit
        }
      }' "${QUEUE_FILE}" || true
  )"
  if [ -z "${first}" ]; then
    rm -f "${tmp}"
    return 1
  fi
  awk '
    BEGIN{removed=0}
    {
      if(!removed && $0 ~ /^[^#[:space:]].*$/){
        removed=1
        next
      }
      print $0
    }' "${QUEUE_FILE}" > "${tmp}"
  cp "${tmp}" "${QUEUE_FILE}"
  rm -f "${tmp}"
  printf -v "${out_var}" '%s' "${first}"
  return 0
}

run_one_condition() {
  local cond="$1"
  local rel out_dir noise sev run_name
  rel="$(condition_to_relpath "${cond}")"
  out_dir="$(condition_to_outdir "${cond}")"
  noise="${cond%_s*}"
  sev="${cond##*_s}"

  if [ ! -d "${DSEC_ROOT}/${TEST_SPLIT}/${SEQ}/${rel}" ]; then
    echo "[skip] ${cond} missing relpath=${rel}" | tee -a "${QUEUE_LOG}"
    return 0
  fi
  if [ -s "${out_dir}/eval_dsec_ann_test.json" ]; then
    echo "[skip] ${cond} already done at ${out_dir}" | tee -a "${QUEUE_LOG}"
    return 0
  fi

  run_name="${RUN_NAME_PREFIX}__ann_rgb__${noise}__s${sev}__${SEQ}"
  mkdir -p "${out_dir}"

  local cmd=(
    python scripts/evaluate_dsec.py
    --model-path "${ANN_CKPT}"
    --model-type ann
    --dsec-root "${DSEC_ROOT}"
    --test-splits "${TEST_SPLIT}"
    --sequences "${SEQ}"
    --image-relpath "${rel}"
    --class-ids "${CLASS_IDS}"
    --input-height "${INPUT_HEIGHT}"
    --input-width "${INPUT_WIDTH}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --conf-threshold "${CONF_THRESHOLD}"
    --nms-threshold "${NMS_THRESHOLD}"
    --output-dir "${out_dir}"
  )
  if [ "${USE_WANDB}" = "1" ]; then
    cmd+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-run-name "${run_name}")
  fi

  echo "[run] ${cond}" | tee -a "${QUEUE_LOG}"
  if "${cmd[@]}" >> "${QUEUE_LOG}" 2>&1; then
    echo "[ok] ${cond}" | tee -a "${QUEUE_LOG}"
  else
    echo "[fail] ${cond}" | tee -a "${QUEUE_LOG}"
    return 1
  fi
}

run_worker() {
  local wid="$1"
  local cond
  while true; do
    cond=""
    {
      flock -x 9
      pop_next_condition cond || true
      if [ -n "${cond}" ]; then
        echo "${cond}" > "${CURRENT_FILE}"
      fi
    } 9>"${LOCK_FILE}"

    if [ -z "${cond}" ]; then
      echo "[worker:${wid}] queue empty, done" | tee -a "${QUEUE_LOG}"
      break
    fi
    echo "[worker:${wid}] picked ${cond}" | tee -a "${QUEUE_LOG}"
    run_one_condition "${cond}" || true
  done
}

run_queue() {
  if [ ! -d "${VENV_DIR}" ]; then
    echo "ERROR: missing venv ${VENV_DIR}"
    exit 1
  fi
  if [ ! -f "${ANN_CKPT}" ]; then
    echo "ERROR: missing ANN checkpoint ${ANN_CKPT}"
    exit 1
  fi
  source "${VENV_DIR}/bin/activate"

  # Safety guard: inside containers, host RAM can look large while cgroup memory is much lower.
  # If memory limit is <= 64 GiB, force sequential eval unless explicitly overridden.
  local mem_max_bytes=""
  mem_max_bytes="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"
  if [ -n "${mem_max_bytes}" ] && [ "${mem_max_bytes}" != "max" ]; then
    if [ "${mem_max_bytes}" -le 68719476736 ] && [ "${PARALLEL_WORKERS}" -gt 1 ] && [ "${ALLOW_HIGH_PARALLEL}" != "1" ]; then
      echo "[guard] cgroup memory.max=${mem_max_bytes} (<=64GiB): forcing PARALLEL_WORKERS=1" | tee -a "${QUEUE_LOG}"
      PARALLEL_WORKERS=1
    fi
  fi

  echo "[run] parallel_workers=${PARALLEL_WORKERS} device=${DEVICE} seq=${SEQ}" | tee -a "${QUEUE_LOG}"

  local pids=()
  local i
  for i in $(seq 1 "${PARALLEL_WORKERS}"); do
    run_worker "${i}" &
    pids+=("$!")
  done

  local rc=0
  for p in "${pids[@]}"; do
    wait "${p}" || rc=1
  done
  : > "${CURRENT_FILE}"
  queue_status
  return "${rc}"
}

case "${CMD}" in
  build) queue_build ;;
  run) run_queue ;;
  status) queue_status ;;
  remove) queue_remove "${1:-}" ;;
  clear) queue_clear ;;
  *)
    cat <<'USAGE'
Usage:
  bash scripts/night_runs/ann_eval_queue_manager.sh build
  bash scripts/night_runs/ann_eval_queue_manager.sh run
  bash scripts/night_runs/ann_eval_queue_manager.sh status
  bash scripts/night_runs/ann_eval_queue_manager.sh remove <condition>
  bash scripts/night_runs/ann_eval_queue_manager.sh clear
USAGE
    exit 1
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

VENV_DIR="${VENV_DIR:-.venv_train_$(hostname | cut -d'.' -f1)}"
DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
TEST_SPLITS="${TEST_SPLITS:-test}"
TEST_SEQUENCES="${TEST_SEQUENCES:-}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"

ANN_CKPT="${ANN_CKPT:-checkpoints/dsec_subset_ann/vgg11_ssd_ann_dsec_best.pth}"
SNN_CKPT="${SNN_CKPT:-checkpoints/dsec_subset_snn_sim/vgg11_ssd_snn_dsec_best.pth}"
SNN_SEQUENCE_LENGTH="${SNN_SEQUENCE_LENGTH:-8}"
SNN_SEQUENCE_STRIDE="${SNN_SEQUENCE_STRIDE:-1}"
SNN_TIMESTEPS_PER_FRAME="${SNN_TIMESTEPS_PER_FRAME:-5}"
SIM_FPS="${SIM_FPS:-20}"

SEVERITIES="${SEVERITIES:-1,3,5}"
CORRUPTIONS="${CORRUPTIONS:-snow,fog,frost,motion_blur,defocus_blur,gaussian_noise,shot_noise}"
NATIVE_NOISES="${NATIVE_NOISES:-shot_noise,leak_noise,threshold_jitter,refractory}"

RUN_ANN="${RUN_ANN:-1}"
RUN_SNN_SIM_CLEAN="${RUN_SNN_SIM_CLEAN:-1}"
RUN_SNN_SIM_NOISY="${RUN_SNN_SIM_NOISY:-1}"
RUN_SNN_NATIVE="${RUN_SNN_NATIVE:-1}"

ENABLE_WANDB="${ENABLE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-dsec_noise_benchmark}"
WANDB_GROUP="${WANDB_GROUP:-noise_bench_$(date +%Y%m%d_%H%M%S)}"
WANDB_TAGS="${WANDB_TAGS:-noise-benchmark,dsec,subset}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-bench_}"

RESULTS_ROOT="${RESULTS_ROOT:-results/noise_benchmark/${WANDB_GROUP}}"
LOG_FILE="${LOG_FILE:-logs/eval_noise_benchmark_${WANDB_GROUP}.log}"

parse_csv() {
  local v="$1"
  v="${v// /}"
  if [ -z "$v" ]; then
    return 0
  fi
  tr ',' '\n' <<< "$v" | sed '/^$/d'
}

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: venv not found: ${VENV_DIR}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
mkdir -p "${RESULTS_ROOT}" "$(dirname "${LOG_FILE}")"

if [ ! -f "${ANN_CKPT}" ] && [ "${RUN_ANN}" = "1" ]; then
  echo "ERROR: ANN checkpoint missing: ${ANN_CKPT}"
  exit 1
fi
if [ ! -f "${SNN_CKPT}" ] && { [ "${RUN_SNN_SIM_CLEAN}" = "1" ] || [ "${RUN_SNN_SIM_NOISY}" = "1" ] || [ "${RUN_SNN_NATIVE}" = "1" ]; }; then
  echo "ERROR: SNN checkpoint missing: ${SNN_CKPT}"
  exit 1
fi

mapfile -t SPLIT_LIST < <(parse_csv "${TEST_SPLITS}")
if [ "${#SPLIT_LIST[@]}" -eq 0 ]; then
  echo "ERROR: TEST_SPLITS is empty"
  exit 1
fi

build_sequence_list() {
  if [ -n "${TEST_SEQUENCES}" ]; then
    parse_csv "${TEST_SEQUENCES}"
  else
    local split="${SPLIT_LIST[0]}"
    find "${DSEC_ROOT}/${split}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
  fi
}

mapfile -t SEQ_LIST < <(build_sequence_list)
if [ "${#SEQ_LIST[@]}" -eq 0 ]; then
  echo "ERROR: no test sequences discovered"
  exit 1
fi

has_ann_relpath() {
  local rel="$1"
  local split seq
  for split in "${SPLIT_LIST[@]}"; do
    for seq in "${SEQ_LIST[@]}"; do
      if [ -d "${DSEC_ROOT}/${split}/${seq}/${rel}" ]; then
        return 0
      fi
    done
  done
  return 1
}

has_snn_relpath() {
  local rel="$1"
  local split seq
  for split in "${SPLIT_LIST[@]}"; do
    for seq in "${SEQ_LIST[@]}"; do
      if [ -f "${DSEC_ROOT}/${split}/${seq}/${rel}" ]; then
        return 0
      fi
    done
  done
  return 1
}

run_eval() {
  local model_type="$1"
  local noise_type="$2"
  local severity="$3"
  local variant="$4"
  local image_relpath="$5"
  local event_source="$6"
  local event_relpath="$7"

  local slug="${model_type}_${variant}_${noise_type}_s${severity}"
  local out_dir="${RESULTS_ROOT}/${slug}"
  local run_name="${RUN_NAME_PREFIX}${slug}"
  mkdir -p "${out_dir}"

  local -a cmd=(
    python scripts/evaluate_dsec.py
    --model-type "${model_type}"
    --dsec-root "${DSEC_ROOT}"
    --test-splits "${TEST_SPLITS}"
    --class-ids "${CLASS_IDS}"
    --image-relpath "${image_relpath}"
    --input-height "${INPUT_HEIGHT}"
    --input-width "${INPUT_WIDTH}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --output-dir "${out_dir}"
    --noise-type "${noise_type}"
    --noise-severity "${severity}"
    --eval-variant "${variant}"
  )

  if [ -n "${TEST_SEQUENCES}" ]; then
    cmd+=(--sequences "${TEST_SEQUENCES}")
  fi

  if [ "${model_type}" = "ann" ]; then
    cmd+=(--model-path "${ANN_CKPT}")
  else
    cmd+=(
      --model-path "${SNN_CKPT}"
      --event-source "${event_source}"
      --event-relpath "${event_relpath}"
      --sequence-length "${SNN_SEQUENCE_LENGTH}"
      --sequence-stride "${SNN_SEQUENCE_STRIDE}"
      --timesteps-per-frame "${SNN_TIMESTEPS_PER_FRAME}"
      --simulated-fps "${SIM_FPS}"
    )
  fi

  if [ "${ENABLE_WANDB}" = "1" ]; then
    cmd+=(
      --wandb
      --wandb-project "${WANDB_PROJECT}"
      --wandb-group "${WANDB_GROUP}"
      --wandb-tags "${WANDB_TAGS}"
      --wandb-run-name "${run_name}"
    )
  fi

  echo "[run] ${slug}" | tee -a "${LOG_FILE}"
  "${cmd[@]}" >> "${LOG_FILE}" 2>&1
}

echo "[cfg] dsec_root=${DSEC_ROOT}" | tee -a "${LOG_FILE}"
echo "[cfg] test_splits=${TEST_SPLITS}" | tee -a "${LOG_FILE}"
echo "[cfg] sequences=${TEST_SEQUENCES:-<all in ${SPLIT_LIST[0]}>}" | tee -a "${LOG_FILE}"
echo "[cfg] ann_ckpt=${ANN_CKPT}" | tee -a "${LOG_FILE}"
echo "[cfg] snn_ckpt=${SNN_CKPT}" | tee -a "${LOG_FILE}"
echo "[cfg] wandb_project=${WANDB_PROJECT}" | tee -a "${LOG_FILE}"
echo "[cfg] wandb_group=${WANDB_GROUP}" | tee -a "${LOG_FILE}"

# Clean baselines
if [ "${RUN_ANN}" = "1" ] && has_ann_relpath "images/left/distorted_gray"; then
  run_eval "ann" "clean" "0" "rgb" "images/left/distorted_gray" "auto" ""
fi
if [ "${RUN_SNN_SIM_CLEAN}" = "1" ] && has_snn_relpath "v2e_output_distorted_gray_clean/dvs_events.txt"; then
  run_eval "snn" "clean" "0" "sim_clean" "images/left/distorted_gray" "simulated" "v2e_output_distorted_gray_clean/dvs_events.txt"
fi
if [ "${RUN_SNN_SIM_NOISY}" = "1" ] && has_snn_relpath "v2e_output_distorted_gray_noisy/dvs_events.txt"; then
  run_eval "snn" "clean" "0" "sim_noisy" "images/left/distorted_gray" "simulated" "v2e_output_distorted_gray_noisy/dvs_events.txt"
fi

while IFS= read -r sev; do
  [ -n "${sev}" ] || continue
  while IFS= read -r corr; do
    [ -n "${corr}" ] || continue
    img_rel="images/left/distorted_gray_${corr}_s${sev}"
    snn_clean_rel="v2e_output_distorted_gray_${corr}_s${sev}_clean/dvs_events.txt"
    snn_noisy_rel="v2e_output_distorted_gray_${corr}_s${sev}_noisy/dvs_events.txt"

    if [ "${RUN_ANN}" = "1" ] && has_ann_relpath "${img_rel}"; then
      run_eval "ann" "${corr}" "${sev}" "rgb" "${img_rel}" "auto" ""
    else
      echo "[skip] ann ${corr} s${sev} missing ${img_rel}" | tee -a "${LOG_FILE}"
    fi
    if [ "${RUN_SNN_SIM_CLEAN}" = "1" ] && has_snn_relpath "${snn_clean_rel}"; then
      run_eval "snn" "${corr}" "${sev}" "sim_clean" "${img_rel}" "simulated" "${snn_clean_rel}"
    else
      echo "[skip] snn sim_clean ${corr} s${sev} missing ${snn_clean_rel}" | tee -a "${LOG_FILE}"
    fi
    if [ "${RUN_SNN_SIM_NOISY}" = "1" ] && has_snn_relpath "${snn_noisy_rel}"; then
      run_eval "snn" "${corr}" "${sev}" "sim_noisy" "${img_rel}" "simulated" "${snn_noisy_rel}"
    else
      echo "[skip] snn sim_noisy ${corr} s${sev} missing ${snn_noisy_rel}" | tee -a "${LOG_FILE}"
    fi
  done < <(parse_csv "${CORRUPTIONS}")
done < <(parse_csv "${SEVERITIES}")

# Native-noise SNN on clean grayscale input.
if [ "${RUN_SNN_NATIVE}" = "1" ]; then
  while IFS= read -r sev; do
    [ -n "${sev}" ] || continue
    while IFS= read -r native; do
      [ -n "${native}" ] || continue
      native_rel="v2e_output_distorted_gray_${native}_s${sev}/dvs_events.txt"
      if has_snn_relpath "${native_rel}"; then
        run_eval "snn" "${native}" "${sev}" "native" "images/left/distorted_gray" "simulated" "${native_rel}"
      else
        echo "[skip] snn native ${native} s${sev} missing ${native_rel}" | tee -a "${LOG_FILE}"
      fi
    done < <(parse_csv "${NATIVE_NOISES}")
  done < <(parse_csv "${SEVERITIES}")
fi

echo "[done] results_root=${RESULTS_ROOT}" | tee -a "${LOG_FILE}"
echo "[done] log_file=${LOG_FILE}" | tee -a "${LOG_FILE}"

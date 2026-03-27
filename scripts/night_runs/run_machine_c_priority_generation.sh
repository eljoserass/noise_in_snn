#!/usr/bin/env bash
set -euo pipefail

# Priority, resume-safe test generation runner.
# - Sequential phases so high-priority outputs finish first.
# - Safe to re-run: all phases use --skip-existing.
# - Supports sequence batching via SEQ_OFFSET / SEQ_LIMIT.

cd "$(dirname "$0")/../.."

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}"

RUN_TAG="${RUN_TAG:-priority_gen_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_DIR}/${RUN_TAG}}"
mkdir -p "${RUN_LOG_DIR}"

DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
SPLIT="${SPLIT:-test}"
JOBS="${JOBS:-4}"
V2E_SCRIPT="${V2E_SCRIPT:-tools/v2e/v2e.py}"
INPUT_FPS="${INPUT_FPS:-20}"
V2E_EXPOSURE_DURATION="${V2E_EXPOSURE_DURATION:-0.005}"
V2E_EVENTS_FORMAT="${V2E_EVENTS_FORMAT:-h5}"   # h5 | text | both
OVERWRITE_V2E="${OVERWRITE_V2E:-1}"

# Optional env bootstrap for machine C style runs.
BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-1}"
VENV_DIR="${VENV_DIR:-.venv_machine_c}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-1}"
REPAIR_V2E_NUMPY_ALIASES="${REPAIR_V2E_NUMPY_ALIASES:-1}"

# Ordered priority list. Example override:
# PHASES="clean,native_core,weather,blur,digital"
PHASES="${PHASES:-clean,native_core,weather,blur}"

# Optional sequence filter:
# TEST_SEQUENCES="interlaken_00_a,interlaken_00_b"
TEST_SEQUENCES="${TEST_SEQUENCES:-}"

# Batch controls over selected sequences:
# - SEQ_OFFSET: skip first N selected sequences
# - SEQ_LIMIT: process only next N sequences (0 => all remaining)
SEQ_OFFSET="${SEQ_OFFSET:-0}"
SEQ_LIMIT="${SEQ_LIMIT:-0}"

# Severity / noise presets
SEVERITY_CORE="${SEVERITY_CORE:-1,3,5}"
SEVERITY_FULL="${SEVERITY_FULL:-1,2,3,4,5}"
NATIVE_CORE_NOISES="${NATIVE_CORE_NOISES:-shot_noise,leak_noise,threshold_jitter,refractory}"
NATIVE_EXTRA_NOISES="${NATIVE_EXTRA_NOISES:-bandwidth_limit,photoreceptor_noise}"

# Set to 0 if you want best-effort and continue after a failed phase.
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"

# R2 backup upload (read from ../dsec_data_managing/.env by default)
ENABLE_UPLOAD="${ENABLE_UPLOAD:-1}"
UPLOAD_AFTER_EACH_PHASE="${UPLOAD_AFTER_EACH_PHASE:-0}"
UPLOAD_GRANULARITY="${UPLOAD_GRANULARITY:-sequence}" # sequence | phase
ENV_FILE="${ENV_FILE:-../dsec_data_managing/.env}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-32}"
UPLOAD_REMOTE_PREFIX="${UPLOAD_REMOTE_PREFIX:-dsec_processed}"
UPLOAD_SCOPE="${UPLOAD_SCOPE:-all_processed}"  # events_only | all_processed

# If set, remove header-only/empty event files before processing each sequence
# so --skip-existing does not lock in broken outputs.
PRUNE_INVALID_EVENTS="${PRUNE_INVALID_EVENTS:-1}"
MIN_EVENT_ROWS="${MIN_EVENT_ROWS:-1}"

parse_csv() {
  local v="$1"
  v="${v// /}"
  if [ -z "$v" ]; then
    return 0
  fi
  tr ',' '\n' <<< "$v" | sed '/^$/d'
}

count_non_comment_lines() {
  local f="$1"
  if [ ! -f "${f}" ]; then
    echo 0
    return 0
  fi
  awk '!/^[[:space:]]*#/{c++} END{print c+0}' "${f}" 2>/dev/null || echo 0
}

join_by_comma() {
  local out=""
  local first=1
  for x in "$@"; do
    if [ $first -eq 1 ]; then
      out="$x"
      first=0
    else
      out="${out},${x}"
    fi
  done
  printf "%s" "$out"
}

bootstrap_env() {
  if [ "${BOOTSTRAP_ENV}" != "1" ]; then
    return 0
  fi

  if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup] creating venv: ${VENV_DIR}" | tee -a "${RUN_LOG_DIR}/summary.log"
    python3 -m venv "${VENV_DIR}"
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  if [ "${INSTALL_REQUIREMENTS}" = "1" ]; then
    echo "[setup] installing requirements into ${VENV_DIR}" | tee -a "${RUN_LOG_DIR}/summary.log"
    pip install --upgrade pip >/dev/null 2>&1 || true
    pip install -r requirements.txt >>"${RUN_LOG_DIR}/setup.log" 2>&1 || true
  fi

  if ! python3 - <<'PY' >/dev/null 2>&1
import cv2  # noqa: F401
import imagecorruptions  # noqa: F401
PY
  then
    echo "[setup] installing missing runtime deps (opencv/imagecorruptions)" | tee -a "${RUN_LOG_DIR}/summary.log"
    pip install opencv-python imagecorruptions >>"${RUN_LOG_DIR}/setup.log" 2>&1
  fi
}

repair_v2e_numpy_aliases() {
  if [ "${REPAIR_V2E_NUMPY_ALIASES}" != "1" ]; then
    return 0
  fi
  if [ ! -f "${V2E_SCRIPT}" ]; then
    return 0
  fi

  python3 - "${V2E_SCRIPT}" <<'PY' >>"${RUN_LOG_DIR}/setup.log" 2>&1
from pathlib import Path
import re
import sys

v2e_script = Path(sys.argv[1]).resolve()
root = v2e_script.parent
patched = []
for path in root.rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    new = text
    new = re.sub(r'dtype\s*=\s*float(16|32|64)\b', r'dtype=np.float\1', new)
    new = re.sub(r'astype\(\s*float(16|32|64)\s*\)', r'astype(np.float\1)', new)
    new = re.sub(r'\bnp\.float\b', 'float', new)
    if new != text:
        path.write_text(new, encoding="utf-8")
        patched.append(str(path))

print(f"[setup] v2e numpy alias repair patched_files={len(patched)}")
for p in patched[:30]:
    print(f"[setup] patched: {p}")
if len(patched) > 30:
    print(f"[setup] ... and {len(patched)-30} more")
PY
}

prune_invalid_events_for_sequence() {
  local seq="$1"
  local seq_root="${DSEC_ROOT}/${SPLIT}/${seq}"
  [ -d "${seq_root}" ] || return 0

  local removed=0
  local min_event_bytes="${MIN_EVENT_BYTES:-32}"
  while IFS= read -r evf; do
    [ -f "${evf}" ] || continue
    case "${evf}" in
      *.txt)
        rows="$(count_non_comment_lines "${evf}")"
        if [ "${rows}" -lt "${MIN_EVENT_ROWS}" ]; then
          rm -f "${evf}"
          removed=$((removed + 1))
        fi
        ;;
      *.h5)
        size="$(stat -c%s "${evf}" 2>/dev/null || echo 0)"
        if [ "${size}" -lt "${min_event_bytes}" ]; then
          rm -f "${evf}"
          removed=$((removed + 1))
        fi
        ;;
      *)
        ;;
    esac
  done < <(
    find "${seq_root}" -type f \
      \( -path '*/v2e_output*/dvs_events.txt' -o -path '*/v2e_output*/dvs_events.h5' \) \
      2>/dev/null
  )

  if [ "${removed}" -gt 0 ]; then
    echo "[pre] seq=${seq} removed_invalid_event_files=${removed}" | tee -a "${RUN_LOG_DIR}/summary.log"
  fi
}

run_upload_once() {
  local upload_log="$1"
  local seq_name="${2:-}"
  local split_prefix="${SPLIT}"
  local -a upload_args=(
    "--env-file" "${ENV_FILE}"
    "--workers" "${UPLOAD_WORKERS}"
  )

  if [ "${UPLOAD_SCOPE}" = "events_only" ]; then
    if [ -n "${seq_name}" ]; then
      upload_globs=(
        "${split_prefix}/${seq_name}/v2e_output*/dvs_events.h5"
        "${split_prefix}/${seq_name}/v2e_output*/dvs_events.txt"
        "${split_prefix}/${seq_name}/v2e_output*/v2e-args.txt"
        "${split_prefix}/${seq_name}/object_detections/left/tracks_rectified*.npy"
      )
    else
      upload_globs=(
        "${split_prefix}/*/v2e_output*/dvs_events.h5"
        "${split_prefix}/*/v2e_output*/dvs_events.txt"
        "${split_prefix}/*/v2e_output*/v2e-args.txt"
        "${split_prefix}/*/object_detections/left/tracks_rectified*.npy"
      )
    fi
  else
    if [ -n "${seq_name}" ]; then
      upload_globs=(
        "${split_prefix}/${seq_name}/v2e_output*/dvs_events.h5"
        "${split_prefix}/${seq_name}/v2e_output*/dvs_events.txt"
        "${split_prefix}/${seq_name}/v2e_output*/v2e-args.txt"
        "${split_prefix}/${seq_name}/images/left/distorted_gray/*.png"
        "${split_prefix}/${seq_name}/images/left/distorted_*_s*/*.png"
        "${split_prefix}/${seq_name}/object_detections/left/tracks_rectified*.npy"
        "${split_prefix}/${seq_name}/object_detections/left/rectified_validation*/*.png"
      )
    else
      upload_globs=(
        "${split_prefix}/*/v2e_output*/dvs_events.h5"
        "${split_prefix}/*/v2e_output*/dvs_events.txt"
        "${split_prefix}/*/v2e_output*/v2e-args.txt"
        "${split_prefix}/*/images/left/distorted_gray/*.png"
        "${split_prefix}/*/images/left/distorted_*_s*/*.png"
        "${split_prefix}/*/object_detections/left/tracks_rectified*.npy"
        "${split_prefix}/*/object_detections/left/rectified_validation*/*.png"
      )
    fi
  fi

  for g in "${upload_globs[@]}"; do
    upload_args+=("--include-glob" "${g}")
  done

  python3 scripts/r2_sync.py \
    "${upload_args[@]}" \
    upload \
    --local-dir "${DSEC_ROOT}" \
    --remote-prefix "${UPLOAD_REMOTE_PREFIX}" >>"${upload_log}" 2>&1
}

discover_sequences() {
  local root="$1"
  local split="$2"
  find "${root}/${split}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
}

apply_seq_batch() {
  local -n _arr_ref=$1
  local offset="$2"
  local limit="$3"
  local n="${#_arr_ref[@]}"

  if [ "$offset" -ge "$n" ]; then
    _arr_ref=()
    return 0
  fi

  local sliced=("${_arr_ref[@]:offset}")
  if [ "$limit" -gt 0 ] && [ "$limit" -lt "${#sliced[@]}" ]; then
    sliced=("${sliced[@]:0:limit}")
  fi
  _arr_ref=("${sliced[@]}")
}

run_phase_for_sequence() {
  local phase="$1"
  local extra_args="$2"
  local seq="$3"
  local phase_log="${RUN_LOG_DIR}/${phase}.log"

  if [ "${PRUNE_INVALID_EVENTS}" = "1" ]; then
    prune_invalid_events_for_sequence "${seq}"
  fi

  echo "[phase:${phase}] seq=${seq} start $(date -Iseconds)" | tee -a "${RUN_LOG_DIR}/summary.log"

  local -a cmd=(
    python3 scripts/dsec_batch_pipeline.py
    --dsec-root "${DSEC_ROOT}"
    --splits "${SPLIT}"
    --jobs 1
    --sequences "${seq}"
    --extra-args "${extra_args}"
  )

  set +e
  "${cmd[@]}" >>"${phase_log}" 2>&1
  local rc=$?
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "[phase:${phase}] seq=${seq} FAIL rc=${rc} (see ${phase_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
    if [ "${STOP_ON_ERROR}" = "1" ]; then
      exit "$rc"
    fi
    return 0
  fi

  echo "[phase:${phase}] seq=${seq} OK" | tee -a "${RUN_LOG_DIR}/summary.log"

  if [ "${ENABLE_UPLOAD}" = "1" ] && [ "${UPLOAD_GRANULARITY}" = "sequence" ]; then
    local upload_log="${RUN_LOG_DIR}/upload_${phase}.log"
    echo "[upload:${phase}] seq=${seq} start $(date -Iseconds)" | tee -a "${RUN_LOG_DIR}/summary.log"
    set +e
    run_upload_once "${upload_log}" "${seq}"
    local urc=$?
    set -e
    if [ "${urc}" -ne 0 ]; then
      echo "[upload:${phase}] seq=${seq} FAIL rc=${urc} (see ${upload_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
      if [ "${STOP_ON_ERROR}" = "1" ]; then
        exit "${urc}"
      fi
    else
      echo "[upload:${phase}] seq=${seq} OK (see ${upload_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
    fi
  fi
}

run_phase_bulk() {
  local phase="$1"
  local extra_args="$2"
  local phase_log="${RUN_LOG_DIR}/${phase}.log"

  echo "[phase:${phase}] start $(date -Iseconds)" | tee -a "${RUN_LOG_DIR}/summary.log"
  echo "[phase:${phase}] args: ${extra_args}" | tee -a "${RUN_LOG_DIR}/summary.log"

  local -a cmd=(
    python3 scripts/dsec_batch_pipeline.py
    --dsec-root "${DSEC_ROOT}"
    --splits "${SPLIT}"
    --jobs "${JOBS}"
    --extra-args "${extra_args}"
  )
  if [ -n "${RUN_SEQUENCES_CSV}" ]; then
    cmd+=(--sequences "${RUN_SEQUENCES_CSV}")
  fi

  set +e
  "${cmd[@]}" >"${phase_log}" 2>&1
  local rc=$?
  set -e

  if [ "$rc" -ne 0 ]; then
    echo "[phase:${phase}] FAIL rc=${rc} (see ${phase_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
    if [ "${STOP_ON_ERROR}" = "1" ]; then
      exit "$rc"
    fi
  else
    echo "[phase:${phase}] OK (see ${phase_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
    if [ "${ENABLE_UPLOAD}" = "1" ] && [ "${UPLOAD_AFTER_EACH_PHASE}" = "1" ]; then
      upload_log="${RUN_LOG_DIR}/upload_${phase}.log"
      echo "[upload:${phase}] start $(date -Iseconds)" | tee -a "${RUN_LOG_DIR}/summary.log"
      set +e
      run_upload_once "${upload_log}"
      local urc=$?
      set -e
      if [ "${urc}" -ne 0 ]; then
        echo "[upload:${phase}] FAIL rc=${urc} (see ${upload_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
        if [ "${STOP_ON_ERROR}" = "1" ]; then
          exit "${urc}"
        fi
      else
        echo "[upload:${phase}] OK (see ${upload_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
      fi
    fi
  fi
}

run_phase() {
  local phase="$1"
  local extra_args="$2"

  if [ "${UPLOAD_GRANULARITY}" = "sequence" ]; then
    local phase_log="${RUN_LOG_DIR}/${phase}.log"
    : > "${phase_log}"
    for seq in "${selected_sequences[@]}"; do
      run_phase_for_sequence "${phase}" "${extra_args}" "${seq}"
    done
  else
    run_phase_bulk "${phase}" "${extra_args}"
  fi
}

if [ ! -d "${DSEC_ROOT}/${SPLIT}" ]; then
  echo "ERROR: split dir not found: ${DSEC_ROOT}/${SPLIT}"
  exit 1
fi

bootstrap_env
repair_v2e_numpy_aliases

mapfile -t all_sequences < <(discover_sequences "${DSEC_ROOT}" "${SPLIT}")
if [ "${#all_sequences[@]}" -eq 0 ]; then
  echo "ERROR: no sequences found in ${DSEC_ROOT}/${SPLIT}"
  exit 1
fi

selected_sequences=()
if [ -n "${TEST_SEQUENCES}" ]; then
  mapfile -t requested_sequences < <(parse_csv "${TEST_SEQUENCES}")
  declare -A have=()
  for s in "${all_sequences[@]}"; do
    have["$s"]=1
  done
  for s in "${requested_sequences[@]}"; do
    if [ "${have[$s]+x}" = "x" ]; then
      selected_sequences+=("$s")
    else
      echo "WARN: requested sequence not found under ${SPLIT}: ${s}" | tee -a "${RUN_LOG_DIR}/summary.log"
    fi
  done
else
  selected_sequences=("${all_sequences[@]}")
fi

apply_seq_batch selected_sequences "${SEQ_OFFSET}" "${SEQ_LIMIT}"

if [ "${#selected_sequences[@]}" -eq 0 ]; then
  echo "ERROR: no sequences selected after TEST_SEQUENCES/SEQ_OFFSET/SEQ_LIMIT filtering."
  exit 1
fi

RUN_SEQUENCES_CSV="$(join_by_comma "${selected_sequences[@]}")"

{
  echo "== priority generation run =="
  echo "run_tag=${RUN_TAG}"
  echo "run_log_dir=${RUN_LOG_DIR}"
  echo "dsec_root=${DSEC_ROOT}"
  echo "split=${SPLIT}"
  echo "jobs=${JOBS}"
  echo "v2e_script=${V2E_SCRIPT}"
  echo "v2e_events_format=${V2E_EVENTS_FORMAT}"
  echo "overwrite_v2e=${OVERWRITE_V2E}"
  echo "phases=${PHASES}"
  echo "sequences_total=${#all_sequences[@]}"
  echo "sequences_selected=${#selected_sequences[@]}"
  echo "sequences_csv=${RUN_SEQUENCES_CSV}"
  echo "seq_offset=${SEQ_OFFSET}"
  echo "seq_limit=${SEQ_LIMIT}"
  echo "severity_core=${SEVERITY_CORE}"
  echo "severity_full=${SEVERITY_FULL}"
  echo "native_core_noises=${NATIVE_CORE_NOISES}"
  echo "native_extra_noises=${NATIVE_EXTRA_NOISES}"
  echo "stop_on_error=${STOP_ON_ERROR}"
  echo "enable_upload=${ENABLE_UPLOAD}"
  echo "upload_after_each_phase=${UPLOAD_AFTER_EACH_PHASE}"
  echo "upload_granularity=${UPLOAD_GRANULARITY}"
  echo "upload_scope=${UPLOAD_SCOPE}"
  echo "upload_remote_prefix=${UPLOAD_REMOTE_PREFIX}"
  echo "upload_workers=${UPLOAD_WORKERS}"
  echo "bootstrap_env=${BOOTSTRAP_ENV}"
  echo "venv_dir=${VENV_DIR}"
  echo "repair_v2e_numpy_aliases=${REPAIR_V2E_NUMPY_ALIASES}"
  echo "prune_invalid_events=${PRUNE_INVALID_EVENTS}"
  echo "min_event_rows=${MIN_EVENT_ROWS}"
} | tee -a "${RUN_LOG_DIR}/summary.log"

overwrite_arg=""
if [ "${OVERWRITE_V2E}" = "1" ]; then
  overwrite_arg=" --overwrite-v2e"
fi

mapfile -t phase_list < <(parse_csv "${PHASES}")
for phase in "${phase_list[@]}"; do
  case "${phase}" in
    clean)
      run_phase "${phase}" \
        "--run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-modes clean --manual-v2e-noises none --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    native_core)
      run_phase "${phase}" \
        "--run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-modes none --manual-v2e-noises ${NATIVE_CORE_NOISES} --severity-levels ${SEVERITY_CORE} --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    native_extra)
      run_phase "${phase}" \
        "--run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-modes none --manual-v2e-noises ${NATIVE_EXTRA_NOISES} --severity-levels ${SEVERITY_CORE} --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    weather)
      run_phase "${phase}" \
        "--run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-on-corruptions --corruption-subset weather --severity-levels ${SEVERITY_CORE} --v2e-modes clean,noisy --manual-v2e-noises none --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    blur)
      run_phase "${phase}" \
        "--run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-on-corruptions --corruption-subset blur --severity-levels ${SEVERITY_CORE} --v2e-modes clean,noisy --manual-v2e-noises none --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    noise)
      run_phase "${phase}" \
        "--run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-on-corruptions --corruption-subset noise --severity-levels ${SEVERITY_CORE} --v2e-modes clean,noisy --manual-v2e-noises none --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    digital)
      run_phase "${phase}" \
        "--run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-on-corruptions --corruption-subset digital --severity-levels ${SEVERITY_CORE} --v2e-modes clean,noisy --manual-v2e-noises none --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    full_weather)
      run_phase "${phase}" \
        "--run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-events-format ${V2E_EVENTS_FORMAT} --v2e-on-corruptions --corruption-subset weather --severity-levels ${SEVERITY_FULL} --v2e-modes clean,noisy --manual-v2e-noises none --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing${overwrite_arg}"
      ;;
    *)
      echo "ERROR: unknown phase '${phase}'"
      echo "Valid phases: clean,native_core,native_extra,weather,blur,noise,digital,full_weather"
      exit 1
      ;;
  esac
done

if [ "${ENABLE_UPLOAD}" = "1" ] && [ "${UPLOAD_GRANULARITY}" != "sequence" ] && [ "${UPLOAD_AFTER_EACH_PHASE}" != "1" ]; then
  final_upload_log="${RUN_LOG_DIR}/upload_final.log"
  echo "[upload:final] start $(date -Iseconds)" | tee -a "${RUN_LOG_DIR}/summary.log"
  set +e
  run_upload_once "${final_upload_log}"
  urc=$?
  set -e
  if [ "${urc}" -ne 0 ]; then
    echo "[upload:final] FAIL rc=${urc} (see ${final_upload_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
    if [ "${STOP_ON_ERROR}" = "1" ]; then
      exit "${urc}"
    fi
  else
    echo "[upload:final] OK (see ${final_upload_log})" | tee -a "${RUN_LOG_DIR}/summary.log"
  fi
fi

echo "== done ==" | tee -a "${RUN_LOG_DIR}/summary.log"
echo "logs: ${RUN_LOG_DIR}"

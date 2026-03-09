#!/usr/bin/env bash
set -euo pipefail

# Machine C: DSEC conversion/noise loop.
# Re-runs batch pipeline periodically with --skip-existing, so new sequences
# downloaded by Machine A get picked up automatically.

cd "$(dirname "$0")/../.."

# Use a dedicated, persistent env for machine C to avoid cross-machine
# corruption when multiple pods share the same workspace.
VENV_DIR="${VENV_DIR:-.venv_machine_c}"

if [ ! -d "${VENV_DIR}" ]; then
  python3 -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

# imagecorruptions currently imports pkg_resources; ensure compatible setuptools.
USE_IMAGECORRUPTIONS_SRC="${USE_IMAGECORRUPTIONS_SRC:-1}"
IMAGECORRUPTIONS_SRC="${IMAGECORRUPTIONS_SRC:-../imagecorruptions}"
if ! python - <<'PY' >/dev/null 2>&1
import pkg_resources  # noqa: F401
import imagecorruptions  # noqa: F401
PY
then
  echo "[C] fixing python deps for imagecorruptions/pkg_resources compatibility"
  pip install "setuptools<81"
  if [ "$USE_IMAGECORRUPTIONS_SRC" = "1" ] && [ -f "${IMAGECORRUPTIONS_SRC}/setup.py" ]; then
    echo "[C] installing imagecorruptions from source: ${IMAGECORRUPTIONS_SRC}"
    pip install -e "${IMAGECORRUPTIONS_SRC}"
  else
    pip install imagecorruptions
  fi
fi

# Ensure we're on source imagecorruptions if provided (PyPI can be outdated).
if [ "$USE_IMAGECORRUPTIONS_SRC" = "1" ] && [ -f "${IMAGECORRUPTIONS_SRC}/setup.py" ]; then
  echo "[C] ensuring imagecorruptions source install: ${IMAGECORRUPTIONS_SRC}"
  pip install -e "${IMAGECORRUPTIONS_SRC}"
fi

# v2e bootstrap settings (for fresh machines)
BOOTSTRAP_V2E="${BOOTSTRAP_V2E:-1}"
V2E_REPO_URL="${V2E_REPO_URL:-https://github.com/SensorsINI/v2e.git}"
V2E_DIR="${V2E_DIR:-tools/v2e}"
V2E_SCRIPT="${V2E_SCRIPT:-${V2E_DIR}/v2e.py}"
V2E_REF="${V2E_REF:-}"      # optional git ref/commit/tag
PULL_V2E="${PULL_V2E:-0}"    # set 1 to pull latest if repo already exists
INSTALL_V2E_DEPS="${INSTALL_V2E_DEPS:-0}"  # 0 recommended on py3.12+/linux
V2E_DEPS_MODE="${V2E_DEPS_MODE:-minimal}"  # minimal | full
ENSURE_V2E_RUNTIME_DEPS="${ENSURE_V2E_RUNTIME_DEPS:-1}"
ENSURE_TORCH="${ENSURE_TORCH:-1}"
TORCH_INSTALL_MODE="${TORCH_INSTALL_MODE:-cpu}"   # cpu | default
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision torchaudio}"

DSEC_ROOT="${DSEC_ROOT:-data/dsec}"
# Split-specific processing policy:
# - train (includes val subset sequences in DSEC split config): grayscale + clean v2e only
# - test: grayscale + imagecorruptions + v2e on corruptions + v2e manual noise sweeps
TRAINVAL_SPLITS="${TRAINVAL_SPLITS:-train}"
TEST_SPLITS="${TEST_SPLITS:-test}"
TRAINVAL_SEQUENCES="${TRAINVAL_SEQUENCES:-}"
TEST_SEQUENCES="${TEST_SEQUENCES:-}"
JOBS="${JOBS:-2}"
POLL_SECS="${POLL_SECS:-900}"

# v2e / corruption settings
TRAINVAL_V2E_MODES="${TRAINVAL_V2E_MODES:-clean}"
TRAINVAL_MANUAL_V2E_NOISES="${TRAINVAL_MANUAL_V2E_NOISES:-none}"
TEST_SEVERITIES="${TEST_SEVERITIES:-${SEVERITIES:-1,2,3,4,5}}"
TEST_CORRUPTION_SUBSET="${TEST_CORRUPTION_SUBSET:-${CORRUPTION_SUBSET:-all}}"
TEST_V2E_MODES="${TEST_V2E_MODES:-${V2E_MODES:-clean,noisy}}"
TEST_MANUAL_V2E_NOISES="${TEST_MANUAL_V2E_NOISES:-${MANUAL_V2E_NOISES:-shot_noise,leak_noise,threshold_jitter,bandwidth_limit,refractory,photoreceptor_noise}}"
INPUT_FPS="${INPUT_FPS:-20}"
V2E_EXPOSURE_DURATION="${V2E_EXPOSURE_DURATION:-0.005}"
VALIDATE_RECTIFICATION="${VALIDATE_RECTIFICATION:-0}"

# Optional upload of generated artifacts
ENABLE_UPLOAD="${ENABLE_UPLOAD:-1}"
ENV_FILE="${ENV_FILE:-../dsec_data_managing/.env}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-32}"
UPLOAD_REMOTE_PREFIX="${UPLOAD_REMOTE_PREFIX:-dsec_processed}"
UPLOAD_SCOPE="${UPLOAD_SCOPE:-all_processed}"  # events_only | all_processed
UPLOAD_INTERVAL_SECS="${UPLOAD_INTERVAL_SECS:-1800}"  # periodic background upload cadence

# Auto-detect common DSEC root locations when default path is wrong.
if [ ! -d "${DSEC_ROOT}" ]; then
  for candidate in "../data/dsec" "/workspace/data/dsec"; do
    if [ -d "${candidate}" ]; then
      echo "[C] DSEC root not found at ${DSEC_ROOT}; using ${candidate}"
      DSEC_ROOT="${candidate}"
      break
    fi
  done
fi

if [ ! -d "${DSEC_ROOT}" ]; then
  echo "[C] ERROR: DSEC root not found: ${DSEC_ROOT}"
  echo "[C] Set DSEC_ROOT explicitly (e.g. DSEC_ROOT=../data/dsec)"
  exit 1
fi

if [ "$BOOTSTRAP_V2E" = "1" ]; then
  if [ ! -f "${V2E_SCRIPT}" ]; then
    echo "[C] v2e not found at ${V2E_SCRIPT}; cloning ${V2E_REPO_URL} -> ${V2E_DIR}"
    rm -rf "${V2E_DIR}"
    git clone --depth 1 "${V2E_REPO_URL}" "${V2E_DIR}"
  elif [ "${PULL_V2E}" = "1" ] && [ -d "${V2E_DIR}/.git" ]; then
    echo "[C] pulling latest v2e in ${V2E_DIR}"
    git -C "${V2E_DIR}" pull --ff-only || true
  fi

  if [ -n "${V2E_REF}" ] && [ -d "${V2E_DIR}/.git" ]; then
    echo "[C] checking out v2e ref ${V2E_REF}"
    git -C "${V2E_DIR}" fetch --depth 1 origin "${V2E_REF}" || true
    git -C "${V2E_DIR}" checkout "${V2E_REF}" || git -C "${V2E_DIR}" checkout FETCH_HEAD
  fi

  if [ "$INSTALL_V2E_DEPS" = "1" ] && [ -f "${V2E_DIR}/requirements.txt" ]; then
    if [ "${V2E_DEPS_MODE}" = "full" ]; then
      echo "[C] installing v2e requirements (full mode, linux-filtered)"
      tmp_req="$(mktemp)"
      grep -Ev '^(pywin32|pywinpty|wincertstore|pyreadline|wxPython)\b' "${V2E_DIR}/requirements.txt" > "${tmp_req}"
      if ! pip install -r "${tmp_req}"; then
        echo "[C] warning: full v2e requirements install failed; continuing."
      fi
      rm -f "${tmp_req}"
    else
      echo "[C] installing minimal v2e runtime deps"
      pip install "numpy<2.0" scipy opencv-python tqdm h5py || true
    fi
  fi
fi

if [ ! -f "${V2E_SCRIPT}" ]; then
  echo "[C] ERROR: v2e script not found at ${V2E_SCRIPT}"
  exit 1
fi

ensure_headless_easygui_stub() {
  local stub_path="${V2E_DIR}/easygui.py"
  if [ -f "${stub_path}" ]; then
    return 0
  fi
  echo "[C] creating headless easygui stub at ${stub_path}"
  mkdir -p "$(dirname "${stub_path}")"
  cat > "${stub_path}" <<'PY'
"""Headless stub for machine-C server runs."""

def fileopenbox(*args, **kwargs):
    raise RuntimeError(
        "easygui file dialogs are disabled in headless mode. "
        "Run v2e with --input <video_or_folder>."
    )
PY
}

ensure_headless_easygui_stub

# Ensure lightweight v2e runtime deps needed for folder-input conversion.
if [ "${ENSURE_V2E_RUNTIME_DEPS}" = "1" ]; then
  if ! python - <<'PY' >/dev/null 2>&1
import argcomplete  # noqa: F401
from engineering_notation import EngNumber  # noqa: F401
import screeninfo  # noqa: F401
PY
  then
    echo "[C] installing missing lightweight v2e runtime deps"
    pip install argcomplete engineering-notation screeninfo || true
  fi
fi

if [ "${ENSURE_TORCH}" = "1" ]; then
  if ! python - <<'PY' >/dev/null 2>&1
import torch  # noqa: F401
PY
  then
    echo "[C] torch import is broken; reinstalling (${TORCH_INSTALL_MODE} mode)"
    if [ "${TORCH_INSTALL_MODE}" = "cpu" ]; then
      # CPU wheels are enough for v2e conversion and avoid CUDA coupling.
      pip install --force-reinstall --no-cache-dir --index-url https://download.pytorch.org/whl/cpu ${TORCH_PACKAGES}
    else
      pip install --force-reinstall --no-cache-dir ${TORCH_PACKAGES}
    fi
  fi
fi

run_upload_once() {
  upload_args=(
    "--env-file" "${ENV_FILE}"
    "--workers" "${UPLOAD_WORKERS}"
  )

  if [ "${UPLOAD_SCOPE}" = "events_only" ]; then
    upload_globs=(
      "*/*/v2e_output*/dvs_events.txt"
      "*/*/v2e_output*/v2e-args.txt"
      "*/*/object_detections/left/tracks_rectified*.npy"
    )
  else
    upload_globs=(
      "*/*/v2e_output*/dvs_events.txt"
      "*/*/v2e_output*/v2e-args.txt"
      "*/*/images/left/distorted_gray/*.png"
      "*/*/images/left/distorted_*_s*/*.png"
      "*/*/object_detections/left/tracks_rectified*.npy"
      "*/*/object_detections/left/rectified_validation*/*.png"
    )
  fi

  for g in "${upload_globs[@]}"; do
    upload_args+=("--include-glob" "${g}")
  done

  python scripts/r2_sync.py \
    "${upload_args[@]}" \
    upload \
    --local-dir "${DSEC_ROOT}" \
    --remote-prefix "${UPLOAD_REMOTE_PREFIX}"
}

uploader_pid=""
stop_uploader() {
  if [ -n "${uploader_pid}" ] && kill -0 "${uploader_pid}" >/dev/null 2>&1; then
    kill "${uploader_pid}" >/dev/null 2>&1 || true
  fi
}
trap stop_uploader EXIT INT TERM

if [ "$ENABLE_UPLOAD" = "1" ] && [ "${UPLOAD_INTERVAL_SECS}" -gt 0 ]; then
  (
    while true; do
      echo "[C] periodic upload run: $(date -Iseconds)"
      if ! run_upload_once; then
        echo "[C] periodic upload warning: upload failed"
      fi
      sleep "${UPLOAD_INTERVAL_SECS}"
    done
  ) &
  uploader_pid="$!"
  echo "[C] periodic uploader started (pid=${uploader_pid}, interval=${UPLOAD_INTERVAL_SECS}s)"
fi

echo "[C] conversion loop started"
echo "[C] dsec root: ${DSEC_ROOT}"
echo "[C] train splits: ${TRAINVAL_SPLITS} (clean v2e only; includes val subset sequences)"
echo "[C] train sequences: ${TRAINVAL_SEQUENCES:-<all in selected splits>}"
echo "[C] test splits: ${TEST_SPLITS} (full corruption + noisy v2e sweeps)"
echo "[C] test sequences: ${TEST_SEQUENCES:-<all in selected splits>}"
echo "[C] test severities: ${TEST_SEVERITIES} | corruption subset: ${TEST_CORRUPTION_SUBSET}"
echo "[C] train/val v2e modes: ${TRAINVAL_V2E_MODES} | manual noises: ${TRAINVAL_MANUAL_V2E_NOISES}"
echo "[C] test v2e modes: ${TEST_V2E_MODES} | manual noises: ${TEST_MANUAL_V2E_NOISES}"
echo "[C] jobs: ${JOBS}"
echo "[C] validate rectification: ${VALIDATE_RECTIFICATION}"
echo "[C] v2e script: ${V2E_SCRIPT}"

run_batch() {
  local label="$1"
  local splits="$2"
  local sequences="$3"
  local extra_args="$4"
  if [ -z "${splits}" ]; then
    echo "[C] ${label}: no splits configured, skipping"
    return 0
  fi
  local -a cmd=(
    python scripts/dsec_batch_pipeline.py
    --dsec-root "${DSEC_ROOT}"
    --splits "${splits}"
    --jobs "${JOBS}"
  )
  if [ -n "${sequences}" ]; then
    cmd+=(--sequences "${sequences}")
  fi
  cmd+=(--extra-args "${extra_args}")
  echo "[C] ${label}: splits=${splits} sequences=${sequences:-<all>}"
  "${cmd[@]}"
}

while true; do
  echo "[C] batch run: $(date -Iseconds)"
  rectify_arg=""
  if [ "${VALIDATE_RECTIFICATION}" = "1" ]; then
    rectify_arg="--validate-rectification"
  fi
  trainval_extra_args="${rectify_arg} --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-modes ${TRAINVAL_V2E_MODES} --manual-v2e-noises ${TRAINVAL_MANUAL_V2E_NOISES} --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing"
  test_extra_args="${rectify_arg} --run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-on-corruptions --severity-levels ${TEST_SEVERITIES} --corruption-subset ${TEST_CORRUPTION_SUBSET} --v2e-modes ${TEST_V2E_MODES} --manual-v2e-noises ${TEST_MANUAL_V2E_NOISES} --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing"

  if ! run_batch "train/val pass" "${TRAINVAL_SPLITS}" "${TRAINVAL_SEQUENCES}" "${trainval_extra_args}"; then
    echo "[C] batch warning: train/val pass failed for one or more sequences; continuing"
  fi
  if ! run_batch "test pass" "${TEST_SPLITS}" "${TEST_SEQUENCES}" "${test_extra_args}"; then
    echo "[C] batch warning: test pass failed for one or more sequences; continuing"
  fi

  if [ "$ENABLE_UPLOAD" = "1" ]; then
    echo "[C] end-of-batch upload run: $(date -Iseconds)"
    if ! run_upload_once; then
      echo "[C] end-of-batch upload warning: upload failed"
    fi
  fi

  echo "[C] sleeping ${POLL_SECS}s"
  sleep "${POLL_SECS}"
done

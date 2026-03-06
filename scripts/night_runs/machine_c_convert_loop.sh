#!/usr/bin/env bash
set -euo pipefail

# Machine C: DSEC conversion/noise loop.
# Re-runs batch pipeline periodically with --skip-existing, so new sequences
# downloaded by Machine A get picked up automatically.

cd "$(dirname "$0")/../.."

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

DSEC_ROOT="${DSEC_ROOT:-data/dsec}"
SPLITS="${SPLITS:-train,val,test}"
JOBS="${JOBS:-2}"
POLL_SECS="${POLL_SECS:-900}"

# v2e / corruption settings
SEVERITIES="${SEVERITIES:-1,2,3,4,5}"
CORRUPTION_SUBSET="${CORRUPTION_SUBSET:-all}"
V2E_MODES="${V2E_MODES:-clean}"
MANUAL_V2E_NOISES="${MANUAL_V2E_NOISES:-shot_noise,leak_noise,threshold_jitter,bandwidth_limit,refractory,photoreceptor_noise}"
INPUT_FPS="${INPUT_FPS:-20}"
V2E_EXPOSURE_DURATION="${V2E_EXPOSURE_DURATION:-0.005}"

# Optional upload of generated artifacts
ENABLE_UPLOAD="${ENABLE_UPLOAD:-1}"
ENV_FILE="${ENV_FILE:-../dsec_data_managing/.env}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-32}"
UPLOAD_REMOTE_PREFIX="${UPLOAD_REMOTE_PREFIX:-dsec_processed}"
UPLOAD_SCOPE="${UPLOAD_SCOPE:-all_processed}"  # events_only | all_processed

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

echo "[C] conversion loop started"
echo "[C] dsec root: ${DSEC_ROOT}"
echo "[C] splits: ${SPLITS} | jobs=${JOBS}"
echo "[C] severities: ${SEVERITIES} | corruption subset: ${CORRUPTION_SUBSET}"
echo "[C] v2e script: ${V2E_SCRIPT}"

while true; do
  echo "[C] batch run: $(date -Iseconds)"
  python scripts/dsec_batch_pipeline.py \
    --dsec-root "${DSEC_ROOT}" \
    --splits "${SPLITS}" \
    --jobs "${JOBS}" \
    --extra-args "--validate-rectification --run-imagecorruptions --run-v2e --v2e-script ${V2E_SCRIPT} --v2e-on-corruptions --severity-levels ${SEVERITIES} --corruption-subset ${CORRUPTION_SUBSET} --v2e-modes ${V2E_MODES} --manual-v2e-noises ${MANUAL_V2E_NOISES} --input-frame-rate ${INPUT_FPS} --v2e-exposure-duration ${V2E_EXPOSURE_DURATION} --skip-existing"

  if [ "$ENABLE_UPLOAD" = "1" ]; then
    echo "[C] upload run: $(date -Iseconds)"
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
  fi

  echo "[C] sleeping ${POLL_SECS}s"
  sleep "${POLL_SECS}"
done

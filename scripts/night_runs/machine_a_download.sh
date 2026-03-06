#!/usr/bin/env bash
set -euo pipefail

# Machine A: R2 -> local DSEC download

cd "$(dirname "$0")/../.."

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

ENV_FILE="${ENV_FILE:-../dsec_data_managing/.env}"
REMOTE_ROOT="${REMOTE_ROOT:-dsec}"
LOCAL_ROOT="${LOCAL_ROOT:-data/dsec}"
# DSEC-Detection commonly uses physical train/test folders and defines val as a sequence subset of train.
SPLITS="${SPLITS:-train,test}"
WORKERS="${WORKERS:-64}"

IFS=',' read -r -a SPLIT_ARR <<< "$SPLITS"

echo "[A] starting downloads: splits=${SPLITS} workers=${WORKERS}"
echo "[A] env file: ${ENV_FILE}"
echo "[A] remote root: ${REMOTE_ROOT}"
echo "[A] local root: ${LOCAL_ROOT}"

pids=()
for split in "${SPLIT_ARR[@]}"; do
  split_trimmed="$(echo "$split" | xargs)"
  if [ -z "$split_trimmed" ]; then
    continue
  fi
  echo "[A] launch split=${split_trimmed}"
  python scripts/r2_sync.py \
    --env-file "$ENV_FILE" \
    --workers "$WORKERS" \
    download \
    --remote-prefix "${REMOTE_ROOT}/${split_trimmed}" \
    --local-dir "${LOCAL_ROOT}/${split_trimmed}" &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "[A] one or more split downloads failed"
  exit 1
fi

echo "[A] all downloads completed"

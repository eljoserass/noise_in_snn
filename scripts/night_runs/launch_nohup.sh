#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/night_runs/launch_nohup.sh scripts/night_runs/machine_a_download.sh logs/machine_a.log

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <script_path> <log_file>"
  exit 1
fi

SCRIPT_PATH="$1"
LOG_FILE="$2"

mkdir -p "$(dirname "$LOG_FILE")"
nohup bash "$SCRIPT_PATH" >"$LOG_FILE" 2>&1 &
PID=$!
echo "pid=${PID}"
echo "log=${LOG_FILE}"
echo "tail -f ${LOG_FILE}"


#!/usr/bin/env bash
set -euo pipefail

echo "[legacy] run.sh is TUMTraf-oriented."
echo "[legacy] Use ./run_dsec.sh for the active DSEC pipeline."

exec ./run_tumtraf.sh "$@"

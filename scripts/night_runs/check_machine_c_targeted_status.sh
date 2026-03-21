#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DSEC_ROOT="${DSEC_ROOT:-../data/dsec}"
LOG_FILE="${LOG_FILE:-logs/machine_c_targeted.log}"

TRAIN_SIM_SEQUENCES_DEFAULT="thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b"
VAL_SIM_SEQUENCES_DEFAULT="zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a"
TRAIN_SIM_SEQUENCES="${TRAIN_SIM_SEQUENCES:-${TRAIN_SIM_SEQUENCES_DEFAULT}}"
VAL_SIM_SEQUENCES="${VAL_SIM_SEQUENCES:-${VAL_SIM_SEQUENCES_DEFAULT}}"
MIN_EVENT_ROWS="${MIN_EVENT_ROWS:-1}"

if [ ! -d "${DSEC_ROOT}" ]; then
  for candidate in "data/dsec" "../data/dsec" "/workspace/data/dsec"; do
    if [ -d "${candidate}" ]; then
      DSEC_ROOT="${candidate}"
      break
    fi
  done
fi

if [ ! -d "${DSEC_ROOT}" ]; then
  echo "ERROR: DSEC root not found: ${DSEC_ROOT}"
  echo "Set DSEC_ROOT explicitly, for example:"
  echo "  DSEC_ROOT=/path/to/dsec bash scripts/night_runs/check_machine_c_targeted_status.sh"
  exit 1
fi

train_subset_count=$(awk -F',' 'BEGIN{n=0} {for(i=1;i<=NF;i++) if(length($i)) n++} END{print n}' <<< "${TRAIN_SIM_SEQUENCES}")
val_subset_count=$(awk -F',' 'BEGIN{n=0} {for(i=1;i<=NF;i++) if(length($i)) n++} END{print n}' <<< "${VAL_SIM_SEQUENCES}")
if [ -d "${DSEC_ROOT}/test" ]; then
  test_total=$(find "${DSEC_ROOT}/test" -mindepth 1 -maxdepth 1 -type d | wc -l)
else
  test_total=0
fi

count_non_comment_lines() {
  local f="$1"
  if [ ! -f "${f}" ]; then
    echo 0
    return 0
  fi
  local out
  out="$(awk '!/^[[:space:]]*#/ {c++} END{print c+0}' "${f}" 2>/dev/null || true)"
  out="$(printf '%s\n' "${out}" | tail -n 1)"
  if [[ "${out}" =~ ^[0-9]+$ ]]; then
    echo "${out}"
  else
    echo 0
  fi
}

sequence_clean_status() {
  local seq="$1"
  local out_dir="${DSEC_ROOT}/train/${seq}/v2e_output_distorted_gray_clean"
  local out_file="${out_dir}/dvs_events.txt"
  local rows=0
  rows="$(count_non_comment_lines "${out_file}")"
  if [ "${rows}" -ge "${MIN_EVENT_ROWS}" ]; then
    echo "done:${rows}:${out_file}"
    return 0
  fi

  local alt
  while IFS= read -r alt; do
    [ -f "${alt}" ] || continue
    rows="$(count_non_comment_lines "${alt}")"
    if [ "${rows}" -ge "${MIN_EVENT_ROWS}" ]; then
      echo "needs_fix:${rows}:${alt}"
      return 0
    fi
  done < <(find "${DSEC_ROOT}/train/${seq}" -maxdepth 2 -type f -path '*/v2e_output_distorted_gray_clean*/dvs_events.txt' 2>/dev/null | sort)

  echo "pending:0:${out_file}"
}

train_clean_done=0
for seq in $(tr ',' ' ' <<< "${TRAIN_SIM_SEQUENCES},${VAL_SIM_SEQUENCES}"); do
  s="$(sequence_clean_status "${seq}")"
  status="${s%%:*}"
  if [ "${status}" = "done" ]; then
    train_clean_done=$((train_clean_done + 1))
  fi
done

if [ -d "${DSEC_ROOT}/test" ]; then
  test_event_done=$(find "${DSEC_ROOT}/test" -type f -path '*/v2e_output*/dvs_events.txt' | wc -l)
else
  test_event_done=0
fi
test_event_expected=$((test_total * 182))
train_clean_expected=$((train_subset_count + val_subset_count))
target_subset_complete=0
if [ "${train_clean_done}" -eq "${train_clean_expected}" ]; then
  target_subset_complete=1
fi

echo "== machine C targeted status =="
du -sh "${DSEC_ROOT}" 2>/dev/null || true
echo "dsec_root=${DSEC_ROOT}"
echo "log=${LOG_FILE}"
echo
echo "selected train targets (${train_subset_count}): ${TRAIN_SIM_SEQUENCES}"
echo "selected val targets (${val_subset_count}): ${VAL_SIM_SEQUENCES}"
echo
awk -v a="${train_clean_done}" -v b="${train_clean_expected}" 'BEGIN{printf "train+val clean v2e: %d/%d (%.1f%%)\n",a,b,(b?100*a/b:0)}'
if [ "${test_total}" -gt 0 ]; then
  awk -v a="${test_event_done}" -v b="${test_event_expected}" 'BEGIN{printf "test event outputs: %d/%d (%.1f%%)\n",a,b,(b?100*a/b:0)}'
else
  echo "test event outputs: n/a (no ${DSEC_ROOT}/test directory)"
fi
echo
if [ "${target_subset_complete}" -eq 1 ]; then
  echo "targeted subset status: COMPLETE (selected train+val targets are done)"
  echo "stop recommendation: SAFE TO STOP if your goal is only the targeted train/val subset."
else
  echo "targeted subset status: INCOMPLETE"
  echo "stop recommendation: KEEP RUNNING until all selected train/val targets are done."
fi

echo
echo "== per-train/val clean v2e =="
for seq in $(tr ',' ' ' <<< "${TRAIN_SIM_SEQUENCES}"); do
  out_dir="${DSEC_ROOT}/train/${seq}/v2e_output_distorted_gray_clean"
  s="$(sequence_clean_status "${seq}")"
  status="${s%%:*}"
  rest="${s#*:}"
  rows="${rest%%:*}"
  selected_file="${rest#*:}"
  if [ -d "${out_dir}" ]; then
    files="$(find "${out_dir}" -maxdepth 1 -type f -printf "%f " | sed 's/[[:space:]]*$//')"
    files="${files:-<no files>}"
  else
    files="<missing dir>"
  fi
  printf "%-24s %-10s rows=%-8s selected=%s | %s\n" "${seq}" "${status}" "${rows}" "${selected_file}" "${files}"
done
for seq in $(tr ',' ' ' <<< "${VAL_SIM_SEQUENCES}"); do
  out_dir="${DSEC_ROOT}/train/${seq}/v2e_output_distorted_gray_clean"
  s="$(sequence_clean_status "${seq}")"
  status="${s%%:*}"
  rest="${s#*:}"
  rows="${rest%%:*}"
  selected_file="${rest#*:}"
  if [ -d "${out_dir}" ]; then
    files="$(find "${out_dir}" -maxdepth 1 -type f -printf "%f " | sed 's/[[:space:]]*$//')"
    files="${files:-<no files>}"
  else
    files="<missing dir>"
  fi
  printf "%-24s %-10s rows=%-8s selected=%s | %s\n" "${seq}" "${status}" "${rows}" "${selected_file}" "${files}"
done

echo
echo "== per-test sequence event outputs =="
if [ -d "${DSEC_ROOT}/test" ]; then
  for s in "${DSEC_ROOT}"/test/*; do
    [ -d "${s}" ] || continue
    n=$(find "${s}" -type f -path '*/v2e_output*/dvs_events.txt' | wc -l)
    awk -v name="$(basename "${s}")" -v n="${n}" 'BEGIN{printf "%-24s %3d/182 (%.1f%%)\n",name,n,100*n/182}'
  done | sort
else
  echo "no test split found under ${DSEC_ROOT}/test"
fi

echo
echo "== active r2_sync processes =="
if pgrep -fa "scripts/r2_sync.py" >/dev/null 2>&1; then
  pgrep -fa "scripts/r2_sync.py"
else
  echo "none"
fi

echo
echo "== recent log tail =="
tail -n 40 "${LOG_FILE}" 2>/dev/null || echo "log not found"

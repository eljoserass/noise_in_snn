#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DSEC_ROOT="${DSEC_ROOT:-../data/dsec}"
LOG_FILE="${LOG_FILE:-logs/machine_c_targeted.log}"

TRAIN_SIM_SEQUENCES_DEFAULT="thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b"
VAL_SIM_SEQUENCES_DEFAULT="zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a"
TRAIN_SIM_SEQUENCES="${TRAIN_SIM_SEQUENCES:-${TRAIN_SIM_SEQUENCES_DEFAULT}}"
VAL_SIM_SEQUENCES="${VAL_SIM_SEQUENCES:-${VAL_SIM_SEQUENCES_DEFAULT}}"

train_subset_count=$(awk -F',' 'BEGIN{n=0} {for(i=1;i<=NF;i++) if(length($i)) n++} END{print n}' <<< "${TRAIN_SIM_SEQUENCES}")
val_subset_count=$(awk -F',' 'BEGIN{n=0} {for(i=1;i<=NF;i++) if(length($i)) n++} END{print n}' <<< "${VAL_SIM_SEQUENCES}")
test_total=$(find "${DSEC_ROOT}/test" -mindepth 1 -maxdepth 1 -type d | wc -l)

train_clean_done=0
for seq in $(tr ',' ' ' <<< "${TRAIN_SIM_SEQUENCES},${VAL_SIM_SEQUENCES}"); do
  if [ -f "${DSEC_ROOT}/train/${seq}/v2e_output_distorted_gray_clean/dvs_events.txt" ]; then
    train_clean_done=$((train_clean_done + 1))
  fi
done

test_event_done=$(find "${DSEC_ROOT}/test" -type f -path '*/v2e_output*/dvs_events.txt' | wc -l)
test_event_expected=$((test_total * 182))
train_clean_expected=$((train_subset_count + val_subset_count))

echo "== machine C targeted status =="
du -sh "${DSEC_ROOT}"
echo "log=${LOG_FILE}"
echo
awk -v a="${train_clean_done}" -v b="${train_clean_expected}" 'BEGIN{printf "train+val clean v2e: %d/%d (%.1f%%)\n",a,b,(b?100*a/b:0)}'
awk -v a="${test_event_done}" -v b="${test_event_expected}" 'BEGIN{printf "test event outputs: %d/%d (%.1f%%)\n",a,b,(b?100*a/b:0)}'

echo
echo "== per-train/val clean v2e =="
for seq in $(tr ',' ' ' <<< "${TRAIN_SIM_SEQUENCES}"); do
  if [ -f "${DSEC_ROOT}/train/${seq}/v2e_output_distorted_gray_clean/dvs_events.txt" ]; then
    status="done"
  else
    status="pending"
  fi
  printf "%-24s %s\n" "${seq}" "${status}"
done
for seq in $(tr ',' ' ' <<< "${VAL_SIM_SEQUENCES}"); do
  if [ -f "${DSEC_ROOT}/train/${seq}/v2e_output_distorted_gray_clean/dvs_events.txt" ]; then
    status="done"
  else
    status="pending"
  fi
  printf "%-24s %s\n" "${seq}" "${status}"
done

echo
echo "== per-test sequence event outputs =="
for s in "${DSEC_ROOT}"/test/*; do
  [ -d "${s}" ] || continue
  n=$(find "${s}" -type f -path '*/v2e_output*/dvs_events.txt' | wc -l)
  awk -v name="$(basename "${s}")" -v n="${n}" 'BEGIN{printf "%-24s %3d/182 (%.1f%%)\n",name,n,100*n/182}'
done | sort

echo
echo "== recent log tail =="
tail -n 40 "${LOG_FILE}" 2>/dev/null || echo "log not found"

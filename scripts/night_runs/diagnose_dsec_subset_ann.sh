#!/usr/bin/env bash
set -euo pipefail

# Diagnose ANN subset quality:
# 1) data coverage/class stats for selected train/val sequences and full test split
# 2) mAP sensitivity to confidence threshold
#
# Usage:
#   DSEC_ROOT=/workspace/data/dsec \
#   ANN_CKPT=checkpoints/dsec_subset_ann/vgg11_ssd_ann_dsec_best.pth \
#   bash scripts/night_runs/diagnose_dsec_subset_ann.sh

cd "$(dirname "$0")/../.."

VENV_DIR="${VENV_DIR:-.venv_train_$(hostname | cut -d'.' -f1)}"
DSEC_ROOT="${DSEC_ROOT:-/workspace/data/dsec}"
ANN_CKPT="${ANN_CKPT:-checkpoints/dsec_subset_ann/vgg11_ssd_ann_dsec_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-results/diagnostics/ann_subset_$(date +%Y%m%d_%H%M%S)}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-train}"
TEST_SPLIT="${TEST_SPLIT:-test}"

TRAIN_SEQUENCES="${TRAIN_SEQUENCES:-thun_00_a,interlaken_00_c,interlaken_00_e,interlaken_00_g,zurich_city_00_b,zurich_city_02_c,zurich_city_05_b,zurich_city_10_b}"
VAL_SEQUENCES="${VAL_SEQUENCES:-zurich_city_16_a,zurich_city_17_a,zurich_city_18_a,zurich_city_19_a,zurich_city_20_a,zurich_city_21_a}"

IMAGE_RELPATH="${IMAGE_RELPATH:-images/left/distorted_gray}"
TRACKS_RELPATH="${TRACKS_RELPATH:-object_detections/left/tracks.npy}"
TIMESTAMPS_RELPATH="${TIMESTAMPS_RELPATH:-images/timestamps.txt}"
CLASS_IDS="${CLASS_IDS:-0,1,2,3,4,5,6,7}"
INPUT_HEIGHT="${INPUT_HEIGHT:-480}"
INPUT_WIDTH="${INPUT_WIDTH:-640}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
CONF_THRESHOLDS="${CONF_THRESHOLDS:-0.5,0.3,0.1,0.05,0.01}"

mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${VENV_DIR}" ]; then
  echo "ERROR: missing venv ${VENV_DIR}"
  exit 1
fi
if [ ! -f "${ANN_CKPT}" ]; then
  echo "ERROR: missing checkpoint ${ANN_CKPT}"
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "== diagnose_dsec_subset_ann =="
echo "dsec_root=${DSEC_ROOT}"
echo "ann_ckpt=${ANN_CKPT}"
echo "output_dir=${OUTPUT_DIR}"
echo "image_relpath=${IMAGE_RELPATH}"
echo "class_ids=${CLASS_IDS}"

python scripts/diagnose_dsec_subset.py \
  --dsec-root "${DSEC_ROOT}" \
  --train-split "${TRAIN_SPLIT}" \
  --val-split "${VAL_SPLIT}" \
  --test-split "${TEST_SPLIT}" \
  --train-sequences "${TRAIN_SEQUENCES}" \
  --val-sequences "${VAL_SEQUENCES}" \
  --image-relpath "${IMAGE_RELPATH}" \
  --tracks-relpath "${TRACKS_RELPATH}" \
  --timestamps-relpath "${TIMESTAMPS_RELPATH}" \
  --class-ids "${CLASS_IDS}" \
  --output-json "${OUTPUT_DIR}/data_coverage.json"

IFS=',' read -r -a THRS <<< "${CONF_THRESHOLDS}"
for thr in "${THRS[@]}"; do
  thr="$(echo "${thr}" | xargs)"
  [ -n "${thr}" ] || continue
  out_dir="${OUTPUT_DIR}/eval_conf_${thr}"
  mkdir -p "${out_dir}"
  echo "[eval] conf_threshold=${thr}"
  python scripts/evaluate_dsec.py \
    --model-path "${ANN_CKPT}" \
    --model-type ann \
    --dsec-root "${DSEC_ROOT}" \
    --test-splits "${TEST_SPLIT}" \
    --image-relpath "${IMAGE_RELPATH}" \
    --tracks-relpath "${TRACKS_RELPATH}" \
    --class-ids "${CLASS_IDS}" \
    --input-height "${INPUT_HEIGHT}" \
    --input-width "${INPUT_WIDTH}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --conf-threshold "${thr}" \
    --output-dir "${out_dir}"
done

python - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for d in sorted(root.glob("eval_conf_*")):
    p = d / "eval_dsec_ann_test.json"
    if not p.exists():
        continue
    obj = json.loads(p.read_text())
    try:
        conf = float(d.name.replace("eval_conf_", ""))
    except Exception:
        conf = float("nan")
    rows.append(
        {
            "conf": conf,
            "map50": float(obj.get("mAP@0.5", 0.0)),
            "map75": float(obj.get("mAP@0.75", 0.0)),
            "map_avg": float(obj.get("mAP_avg", 0.0)),
            "num_preds": int(obj.get("num_predictions", 0)),
            "num_gts": int(obj.get("num_ground_truths", 0)),
            "num_images": int(obj.get("num_images", 0)),
        }
    )

rows.sort(key=lambda r: r["conf"], reverse=True)
print("\n== threshold sweep summary ==")
for r in rows:
    print(
        f"conf={r['conf']:.3f} "
        f"mAP@0.5={r['map50']:.4f} mAP@0.75={r['map75']:.4f} mAP_avg={r['map_avg']:.4f} "
        f"preds={r['num_preds']} gts={r['num_gts']} images={r['num_images']}"
    )

out_csv = root / "threshold_sweep_summary.csv"
with out_csv.open("w", encoding="utf-8") as f:
    f.write("conf,mAP@0.5,mAP@0.75,mAP_avg,num_predictions,num_ground_truths,num_images\n")
    for r in rows:
        f.write(
            f"{r['conf']},{r['map50']},{r['map75']},{r['map_avg']},"
            f"{r['num_preds']},{r['num_gts']},{r['num_images']}\n"
        )
print(f"saved: {out_csv}")
PY

echo
echo "done: ${OUTPUT_DIR}"

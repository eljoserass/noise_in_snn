#!/usr/bin/env python3
"""
Diagnose DSEC subset/data coverage for ANN training and clean-test evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DSEC_CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "train",
}


def parse_class_ids(class_ids: str | list[int] | None) -> list[int]:
    if class_ids is None:
        return sorted(DSEC_CLASS_NAMES.keys())
    if isinstance(class_ids, str):
        parsed = [int(x.strip()) for x in class_ids.split(",") if x.strip()]
    else:
        parsed = [int(x) for x in class_ids]
    unique = sorted(set(parsed))
    if not unique:
        raise ValueError("class_ids resolved to empty list")
    return unique


def parse_list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def sequence_dirs(dsec_root: Path, split: str, sequences_csv: str) -> list[Path]:
    split_dir = dsec_root / split
    if not split_dir.exists():
        return []
    names = parse_list(sequences_csv)
    if names:
        return [split_dir / n for n in names if (split_dir / n).is_dir()]
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def load_ts(ts_path: Path, n_frames: int) -> np.ndarray:
    if not ts_path.exists():
        return np.zeros((0,), dtype=np.int64)
    ts = np.loadtxt(ts_path, dtype=np.int64)
    if ts.ndim == 0:
        ts = np.asarray([np.int64(ts)], dtype=np.int64)
    return ts[:n_frames]


def per_seq_stats(
    seq_dir: Path,
    image_relpath: str,
    tracks_relpath: str,
    timestamps_relpath: str,
    class_ids: list[int],
) -> dict[str, Any]:
    image_dir = seq_dir / image_relpath
    frames = sorted(list(image_dir.glob("*.png")))
    if not frames:
        frames = sorted(list(image_dir.glob("*.jpg")))
    n_frames = len(frames)
    frame_ids = []
    for fp in frames:
        try:
            frame_ids.append(int(fp.stem))
        except ValueError:
            pass
    frame_min = min(frame_ids) if frame_ids else -1
    frame_max = max(frame_ids) if frame_ids else -1
    if frame_ids and frame_max >= frame_min:
        expected = frame_max - frame_min + 1
        missing_frame_ids = max(0, expected - len(set(frame_ids)))
    else:
        missing_frame_ids = 0

    ts = load_ts(seq_dir / timestamps_relpath, n_frames)
    tracks_path = seq_dir / tracks_relpath
    if not tracks_path.exists():
        return {
            "sequence": seq_dir.name,
            "frames": n_frames,
            "frame_id_min": int(frame_min),
            "frame_id_max": int(frame_max),
            "missing_frame_ids_in_range": int(missing_frame_ids),
            "usable_frames_with_gt": 0,
            "boxes_total": 0,
            "boxes_per_class": {},
            "missing_tracks": True,
        }

    tracks = np.load(tracks_path)
    if tracks.dtype.names is None:
        raise RuntimeError(f"tracks file not structured: {tracks_path}")
    needed = {"t", "class_id"}
    if not needed.issubset(set(tracks.dtype.names)):
        raise RuntimeError(f"tracks missing fields {needed}: {tracks_path}")

    class_mask = np.isin(tracks["class_id"].astype(np.int64), np.asarray(class_ids, dtype=np.int64))
    tracks = tracks[class_mask]
    track_ts = tracks["t"].astype(np.int64)
    class_vals = tracks["class_id"].astype(np.int64)

    # Approximate frame->track assignment in nearest mode (training/eval defaults).
    usable = 0
    boxes_total = 0
    per_class = {str(cid): 0 for cid in class_ids}
    if ts.size > 0 and track_ts.size > 0:
        uniq, first_idx, counts = np.unique(track_ts, return_index=True, return_counts=True)
        for t in ts:
            pos = int(np.searchsorted(uniq, int(t)))
            cands = []
            if pos < len(uniq):
                cands.append(pos)
            if pos > 0:
                cands.append(pos - 1)
            if not cands:
                continue
            best = min(cands, key=lambda i: abs(int(uniq[i]) - int(t)))
            if abs(int(uniq[best]) - int(t)) > 50_000:
                continue
            s = int(first_idx[best])
            e = s + int(counts[best])
            if e <= s:
                continue
            usable += 1
            boxes_total += int(e - s)
            cls_slice = class_vals[s:e]
            for cid in class_ids:
                per_class[str(cid)] += int((cls_slice == cid).sum())

    return {
        "sequence": seq_dir.name,
        "frames": n_frames,
        "frame_id_min": int(frame_min),
        "frame_id_max": int(frame_max),
        "missing_frame_ids_in_range": int(missing_frame_ids),
        "usable_frames_with_gt": int(usable),
        "boxes_total": int(boxes_total),
        "boxes_per_class": per_class,
        "missing_tracks": False,
    }


def aggregate(split_name: str, stats: list[dict[str, Any]], class_ids: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "split": split_name,
        "num_sequences": len(stats),
        "frames_total": int(sum(s["frames"] for s in stats)),
        "missing_frame_ids_in_range_total": int(sum(s.get("missing_frame_ids_in_range", 0) for s in stats)),
        "usable_frames_with_gt_total": int(sum(s["usable_frames_with_gt"] for s in stats)),
        "boxes_total": int(sum(s["boxes_total"] for s in stats)),
        "boxes_per_class": {str(cid): 0 for cid in class_ids},
        "sequences": stats,
    }
    for s in stats:
        for cid in class_ids:
            out["boxes_per_class"][str(cid)] += int(s["boxes_per_class"].get(str(cid), 0))
    return out


def class_name_map(class_ids: list[int]) -> dict[str, str]:
    return {str(cid): DSEC_CLASS_NAMES.get(cid, f"class_{cid}") for cid in class_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose DSEC subset coverage and class distribution.")
    ap.add_argument("--dsec-root", type=Path, required=True)
    ap.add_argument("--train-split", type=str, default="train")
    ap.add_argument("--val-split", type=str, default="train")
    ap.add_argument("--test-split", type=str, default="test")
    ap.add_argument("--train-sequences", type=str, required=True)
    ap.add_argument("--val-sequences", type=str, required=True)
    ap.add_argument("--image-relpath", type=str, default="images/left/distorted_gray")
    ap.add_argument("--tracks-relpath", type=str, default="object_detections/left/tracks.npy")
    ap.add_argument("--timestamps-relpath", type=str, default="images/timestamps.txt")
    ap.add_argument("--class-ids", type=str, default="0,1,2,3,4,5,6,7")
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args()

    class_ids = parse_class_ids(args.class_ids)

    tr_dirs = sequence_dirs(args.dsec_root, args.train_split, args.train_sequences)
    va_dirs = sequence_dirs(args.dsec_root, args.val_split, args.val_sequences)
    te_dirs = sequence_dirs(args.dsec_root, args.test_split, "")

    tr_stats = [
        per_seq_stats(p, args.image_relpath, args.tracks_relpath, args.timestamps_relpath, class_ids) for p in tr_dirs
    ]
    va_stats = [
        per_seq_stats(p, args.image_relpath, args.tracks_relpath, args.timestamps_relpath, class_ids) for p in va_dirs
    ]
    te_stats = [
        per_seq_stats(p, args.image_relpath, args.tracks_relpath, args.timestamps_relpath, class_ids) for p in te_dirs
    ]

    payload = {
        "class_names": class_name_map(class_ids),
        "train_subset": aggregate("train_subset", tr_stats, class_ids),
        "val_subset": aggregate("val_subset", va_stats, class_ids),
        "test_full": aggregate("test_full", te_stats, class_ids),
    }

    print("== subset coverage summary ==")
    for key in ["train_subset", "val_subset", "test_full"]:
        x = payload[key]
        print(
            f"{key}: seq={x['num_sequences']} frames={x['frames_total']} "
            f"missing_frame_ids={x['missing_frame_ids_in_range_total']} "
            f"usable_frames_with_gt={x['usable_frames_with_gt_total']} boxes={x['boxes_total']}"
        )
        class_counts = ", ".join(
            f"{payload['class_names'][cid]}={x['boxes_per_class'][cid]}" for cid in sorted(x["boxes_per_class"])
        )
        print(f"  classes: {class_counts}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()

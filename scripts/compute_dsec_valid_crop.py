#!/usr/bin/env python3
"""
Compute deterministic valid-image crop for DSEC left/distorted view from calibration mappings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def compute_remapping(sequence_dir: Path) -> np.ndarray:
    # Reuse official dsec-det implementation for consistency.
    dsec_det_src = Path(__file__).resolve().parents[2] / "dsec_data_managing" / "dsec-det" / "src"
    if not dsec_det_src.exists():
        dsec_det_src = Path("/home/jose/Documents/projects/neuromorph_vs_noise/dsec_data_managing/dsec-det/src")
    if not dsec_det_src.exists():
        raise FileNotFoundError(f"dsec-det src not found at {dsec_det_src}")

    sys.path.insert(0, str(dsec_det_src))
    from dsec_det.io import h5_file_to_dict, yaml_file_to_dict  # type: ignore
    from dsec_det.remapping import compute_remapping as dsec_compute_remapping  # type: ignore

    calibration = yaml_file_to_dict(sequence_dir / "calibration" / "cam_to_cam.yaml")
    rectify_map = h5_file_to_dict(sequence_dir / "events" / "left" / "rectify_map.h5")
    return dsec_compute_remapping(calibration, rectify_map)


def valid_crop_from_map(sequence_dir: Path, remap: np.ndarray) -> dict[str, int]:
    rectified_img = cv2.imread(str(sequence_dir / "images" / "left" / "rectified" / "000000.png"))
    if rectified_img is None:
        raise FileNotFoundError(f"Missing rectified image for size inference: {sequence_dir}")
    rect_h, rect_w = rectified_img.shape[:2]

    valid = (
        (remap[..., 0] >= 0)
        & (remap[..., 0] < rect_w)
        & (remap[..., 1] >= 0)
        & (remap[..., 1] < rect_h)
        & np.isfinite(remap[..., 0])
        & np.isfinite(remap[..., 1])
    )
    ys, xs = np.where(valid)
    if ys.size == 0 or xs.size == 0:
        raise RuntimeError(f"No valid mapping pixels for sequence {sequence_dir.name}")

    h, w = remap.shape[:2]
    return {
        "crop_top_px": int(ys.min()),
        "crop_bottom_px": int(h - ys.max() - 1),
        "crop_left_px": int(xs.min()),
        "crop_right_px": int(w - xs.max() - 1),
        "valid_ratio": float(valid.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DSEC deterministic valid crop from calibration.")
    parser.add_argument("--dsec-root", type=Path, required=True, help="Root with split dirs (train/val/test).")
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--sequences", type=str, default="", help="Optional comma list, applied to all splits.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/dsec_valid_crop.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    splits = parse_list(args.splits)
    explicit_sequences = parse_list(args.sequences) if args.sequences.strip() else None

    result: dict[str, dict[str, dict[str, int | float]]] = {}
    global_crop = {"crop_top_px": 0, "crop_bottom_px": 0, "crop_left_px": 0, "crop_right_px": 0}

    for split in splits:
        split_dir = args.dsec_root / split
        if not split_dir.exists():
            continue
        seq_dirs = (
            [split_dir / s for s in explicit_sequences if (split_dir / s).is_dir()]
            if explicit_sequences
            else sorted([p for p in split_dir.iterdir() if p.is_dir()])
        )
        if not seq_dirs:
            continue
        result[split] = {}
        for seq in seq_dirs:
            remap = compute_remapping(seq)
            crop = valid_crop_from_map(seq, remap)
            result[split][seq.name] = crop
            for k in global_crop:
                global_crop[k] = max(global_crop[k], int(crop[k]))
            print(
                f"{split}/{seq.name}: "
                f"top={crop['crop_top_px']} bottom={crop['crop_bottom_px']} "
                f"left={crop['crop_left_px']} right={crop['crop_right_px']} "
                f"valid={crop['valid_ratio']:.4f}"
            )

    payload = {
        "dsec_root": str(args.dsec_root),
        "splits": splits,
        "global_crop_px_max_over_sequences": global_crop,
        "per_sequence": result,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {args.output_json}")


if __name__ == "__main__":
    main()

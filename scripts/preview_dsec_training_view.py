#!/usr/bin/env python3
"""
Render what ANN/SNN training actually sees for DSEC (with GT boxes).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.data.dsec_dataset import DSEC_CLASS_NAMES, DSECSSD_ANN, DSECSSD_SNN, parse_class_ids


def parse_list(value: str) -> list[str] | None:
    if not value.strip():
        return None
    out = [x.strip() for x in value.split(",") if x.strip()]
    return out or None


def class_name_from_label(label: int, class_ids: list[int]) -> str:
    # labels are 1..N in dataset
    if label <= 0 or label > len(class_ids):
        return str(label)
    class_id = class_ids[label - 1]
    return DSEC_CLASS_NAMES.get(class_id, f"class_{class_id}")


def draw_boxes(
    image_bgr: np.ndarray,
    boxes_norm: torch.Tensor,
    labels: torch.Tensor,
    class_ids: list[int],
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    for box, label in zip(boxes_norm, labels):
        cx, cy, bw, bh = [float(v) for v in box]
        x1 = int((cx - 0.5 * bw) * w)
        y1 = int((cy - 0.5 * bh) * h)
        x2 = int((cx + 0.5 * bw) * w)
        y2 = int((cy + 0.5 * bh) * h)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cname = class_name_from_label(int(label), class_ids)
        cv2.putText(
            out,
            cname,
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def ann_image_to_bgr(img_chw: torch.Tensor) -> np.ndarray:
    arr = img_chw.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)


def snn_frame_to_bgr(event_2chw: torch.Tensor) -> np.ndarray:
    arr = event_2chw.detach().cpu().numpy()
    pos = arr[1] if arr.shape[0] > 1 else arr[0]
    neg = arr[0]
    pos = np.clip(pos, 0.0, 1.0)
    neg = np.clip(neg, 0.0, 1.0)
    ch_r = (pos * 255.0).astype(np.uint8)
    ch_b = (neg * 255.0).astype(np.uint8)
    ch_g = np.clip((pos + neg) * 0.5, 0.0, 1.0)
    ch_g = (ch_g * 255.0).astype(np.uint8)
    return cv2.merge([ch_b, ch_g, ch_r])


def render_ann(args, out_dir: Path, class_ids: list[int]) -> list[Path]:
    ds = DSECSSD_ANN(
        dsec_root=args.dsec_root,
        split=args.split,
        image_relpath=args.ann_image_relpath,
        timestamps_relpath=args.timestamps_relpath,
        tracks_relpath=args.ann_tracks_relpath,
        class_ids=class_ids,
        sequences=parse_list(args.sequences),
        force_grayscale=not args.ann_rgb_color,
        crop_top_px=args.ann_crop_top_px,
        crop_bottom_px=args.ann_crop_bottom_px,
        crop_left_px=args.ann_crop_left_px,
        crop_right_px=args.ann_crop_right_px,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )

    paths = []
    n = min(args.num_ann_samples, len(ds))
    step = max(1, len(ds) // n)
    for i in range(0, len(ds), step):
        if len(paths) >= n:
            break
        img, target = ds[i]
        bgr = ann_image_to_bgr(img)
        painted = draw_boxes(bgr, target["boxes"], target["labels"], class_ids)
        p = out_dir / f"ann_view_{len(paths):02d}.png"
        cv2.imwrite(str(p), painted)
        paths.append(p)
    return paths


def render_snn(args, out_dir: Path, class_ids: list[int]) -> list[Path]:
    ds = DSECSSD_SNN(
        dsec_root=args.dsec_root,
        split=args.split,
        image_relpath=args.snn_image_relpath,
        timestamps_relpath=args.timestamps_relpath,
        tracks_relpath=args.snn_tracks_relpath,
        class_ids=class_ids,
        sequences=parse_list(args.sequences),
        event_source=args.event_source,
        event_relpath=args.event_relpath,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        max_frames_per_sequence=args.max_frames_per_sequence,
    )
    seq, targets = ds[0]

    paths = []
    frames_to_save = min(args.num_snn_frames, seq.shape[0])
    for t in range(frames_to_save):
        frame = seq[t]
        if tuple(frame.shape[-2:]) != (args.input_height, args.input_width):
            frame = torch.nn.functional.interpolate(
                frame.unsqueeze(0),
                size=(args.input_height, args.input_width),
                mode="bilinear",
                align_corners=False,
            )[0]
        bgr = snn_frame_to_bgr(frame)
        painted = draw_boxes(bgr, targets[t]["boxes"], targets[t]["labels"], class_ids)
        p = out_dir / f"snn_view_t{t:02d}.png"
        cv2.imwrite(str(p), painted)
        paths.append(p)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview DSEC training views with labels.")
    p.add_argument("--dsec-root", type=Path, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--sequences", type=str, default="")
    p.add_argument("--class-ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--timestamps-relpath", type=str, default="images/timestamps.txt")
    p.add_argument("--max-frames-per-sequence", type=int, default=64)

    p.add_argument("--ann-image-relpath", type=str, default="images/left/distorted")
    p.add_argument("--ann-tracks-relpath", type=str, default="object_detections/left/tracks.npy")
    p.add_argument("--ann-rgb-color", action="store_true")
    p.add_argument("--ann-crop-top-px", type=int, default=0)
    p.add_argument("--ann-crop-bottom-px", type=int, default=0)
    p.add_argument("--ann-crop-left-px", type=int, default=0)
    p.add_argument("--ann-crop-right-px", type=int, default=0)
    p.add_argument("--num-ann-samples", type=int, default=6)

    p.add_argument("--snn-image-relpath", type=str, default="images/left/distorted")
    p.add_argument("--snn-tracks-relpath", type=str, default="object_detections/left/tracks.npy")
    p.add_argument("--event-source", type=str, default="real", choices=["auto", "real", "simulated"])
    p.add_argument("--event-relpath", type=str, default="events/left/events.h5")
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--sequence-stride", type=int, default=8)
    p.add_argument("--num-snn-frames", type=int, default=6)
    p.add_argument("--input-height", type=int, default=480)
    p.add_argument("--input-width", type=int, default=640)

    p.add_argument("--out-dir", type=Path, default=Path("outputs/dsec_training_view"))
    return p.parse_args()


def main():
    args = parse_args()
    class_ids = parse_class_ids(args.class_ids)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_paths = render_ann(args, out_dir, class_ids)
    snn_paths = render_snn(args, out_dir, class_ids)

    print("ANN preview images:")
    for p in ann_paths:
        print(f"  {p}")
    print("SNN preview images:")
    for p in snn_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()

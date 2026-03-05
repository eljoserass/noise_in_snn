#!/usr/bin/env python3
"""
Probe v2e timestamp behavior for folder input.

It runs v2e with multiple --input_frame_rate values on the same frame folder and
summarizes how event timestamps scale.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def parse_fps_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one fps value is required.")
    return vals


def sorted_pngs(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.png"), key=lambda p: int(p.stem))


def run_v2e_once(
    python_exe: str,
    v2e_script: Path,
    frames_dir: Path,
    out_dir: Path,
    fps: float,
    width: int,
    height: int,
) -> Path:
    cmd = [
        python_exe,
        str(v2e_script),
        "--input",
        str(frames_dir),
        "--input_frame_rate",
        str(fps),
        "--output_folder",
        str(out_dir),
        "--overwrite",
        "--output_width",
        str(width),
        "--output_height",
        str(height),
        "--disable_slomo",
        "--dvs_params",
        "clean",
        "--dvs_text",
        "dvs_events.txt",
        "--skip_video_output",
        "--no_preview",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("--- v2e stderr tail ---")
        print(res.stderr[-2000:])
        raise RuntimeError(f"v2e failed for fps={fps}")
    events = out_dir / "dvs_events.txt"
    if not events.exists():
        raise RuntimeError(f"Expected output missing: {events}")
    return events


def summarize_events(events_file: Path) -> tuple[int, float, float, float, int]:
    raw = np.loadtxt(events_file, comments="#")
    if raw.ndim == 1:
        raw = raw[None, :]
    t = raw[:, 0]
    dt = np.diff(t)
    positive_dt = dt[dt > 0]
    median_dt = float(np.median(positive_dt)) if positive_dt.size else 0.0
    unique_count = int(np.unique(np.round(positive_dt, 9)).size) if positive_dt.size else 0
    return len(t), float(t.min()), float(t.max()), median_dt, unique_count


def infer_hw(frames: list[Path]) -> tuple[int, int]:
    import cv2

    img = cv2.imread(str(frames[0]))
    if img is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
    h, w = img.shape[:2]
    return w, h


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe v2e timestamp scaling with folder input.")
    parser.add_argument("--frames-dir", type=Path, required=True, help="Input PNG folder.")
    parser.add_argument("--v2e-script", type=Path, default=Path("tools/v2e/v2e.py"), help="Path to v2e.py.")
    parser.add_argument("--fps-list", type=str, default="2,10,20", help="Comma-separated fps values.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/v2e_timestamp_probe"),
        help="Where to write probe outputs and summary.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    v2e_script = (repo_root / args.v2e_script).resolve() if not args.v2e_script.is_absolute() else args.v2e_script
    frames_dir = args.frames_dir.resolve()
    output_root = (repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if not v2e_script.exists():
        raise FileNotFoundError(f"v2e script not found: {v2e_script}")
    frames = sorted_pngs(frames_dir)
    if not frames:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")
    fps_values = parse_fps_list(args.fps_list)
    width, height = infer_hw(frames)

    rows: list[tuple[str, int, float, float, float, int]] = []
    for fps in fps_values:
        tag = f"fps{str(fps).replace('.', 'p')}"
        out_dir = output_root / tag
        events = run_v2e_once(
            python_exe=sys.executable,
            v2e_script=v2e_script,
            frames_dir=frames_dir,
            out_dir=out_dir,
            fps=fps,
            width=width,
            height=height,
        )
        stats = summarize_events(events)
        rows.append((tag, *stats))
        print(f"[ok] {tag}: events={stats[0]} t=[{stats[1]:.6f},{stats[2]:.6f}] median_dt={stats[3]:.9f}")

    summary_file = output_root / "summary.txt"
    with summary_file.open("w") as f:
        f.write("v2e timestamp probe summary\n")
        f.write("tag events t_min_s t_max_s median_positive_dt_s unique_positive_dt_count\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]:.9f} {r[3]:.9f} {r[4]:.9f} {r[5]}\n")

    print(f"[done] summary: {summary_file}")


if __name__ == "__main__":
    main()


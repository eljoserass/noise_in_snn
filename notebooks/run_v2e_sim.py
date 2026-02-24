"""
run_v2e_sim.py
==============
Standalone script equivalent to the 5 new cells planned for Section 6
("Processing the data") of:
  noise_in_snn/notebooks/video2events_desec.ipynb

What it does
------------
1. Clones v2e from GitHub (if not already present) and pip-installs it.
2. Runs v2e on the left/distorted 640x480 PNG frames that were downloaded
   in earlier notebook sections.
3. Loads the simulated event text file produced by v2e.
4. Displays the first 6 simulated event frames with DSEC ground-truth
   bounding boxes overlaid — identical visual style to Section 4 of the
   notebook (black bg, red=ON, blue=OFF, coloured labelled boxes).

Assumptions / paths
-------------------
- Working directory when running: repo root  OR  any path, because all
  paths below are resolved relative to this file's location.
- Data lives at:
    <repo>/noise_in_snn/data/dsec_sample/zurich_city_02_b/
- v2e will be cloned to:
    <repo>/tools/v2e/
- img_timestamps.txt and tracks.npy must already exist (downloaded in
  earlier notebook sections).

Run
---
    python noise_in_snn/scripts/run_v2e_sim.py

Dependencies (beyond the notebook's existing env):
    v2e is installed automatically by this script.
    Everything else (numpy, matplotlib, cv2, pathlib) is already present.
"""

import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths — all relative to this script's location so it works from anywhere
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # noise_in_snn/scripts -> repo root
DATA_DIR = REPO_ROOT / "noise_in_snn" / "data" / "dsec_sample" / "zurich_city_02_b"
V2E_DIR = REPO_ROOT / "tools" / "v2e"

# ---------------------------------------------------------------------------
# Class metadata (mirrors cell `aone5lz8t9v` in the notebook)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "pedestrian",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bicycle",
    6: "motorcycle",
    7: "train",
}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (255, 165, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (128, 0, 128),
    5: (255, 255, 0),
    6: (0, 255, 255),
    7: (128, 128, 0),
}


# ===========================================================================
# CELL 2 — Install v2e
# ===========================================================================
def install_v2e():
    V2E_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not (V2E_DIR / "v2e.py").exists():
        print(f"Cloning v2e into {V2E_DIR} ...")
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/SensorsINI/v2e",
                str(V2E_DIR),
            ]
        )
        print("Clone done.")
    else:
        print(f"v2e already present at {V2E_DIR}")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(V2E_DIR),
            "-q",
            "--no-warn-script-location",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("pip stderr:", result.stderr[-500:])
        raise RuntimeError("pip install of v2e failed — see stderr above")
    print("v2e ready")


# ===========================================================================
# CELL 3 — Run v2e
# ===========================================================================
def run_v2e():
    v2e_in_dir = DATA_DIR / "images" / "left" / "distorted"
    v2e_out_dir = DATA_DIR / "v2e_output"
    v2e_out_dir.mkdir(parents=True, exist_ok=True)

    frames_in = sorted(v2e_in_dir.glob("*.png"), key=lambda p: int(p.stem))
    print(f"Input : {len(frames_in)} frames at 640x480  ({v2e_in_dir})")
    print(f"Output: {v2e_out_dir}")

    cmd = [
        sys.executable,
        str(V2E_DIR / "v2e.py"),
        "--input",
        str(v2e_in_dir),
        "--input_frame_rate",
        "20",
        "--output_folder",
        str(v2e_out_dir),
        "--overwrite",
        "--output_width",
        "640",
        "--output_height",
        "480",
        "--disable_slomo",
        "--dvs_params",
        "clean",
        "--dvs_text",
        "dvs_events.txt",
        "--skip_video_output",
        "--no_preview",
    ]

    print("\nRunning v2e ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout
    print(out[-3000:] if len(out) > 3000 else out)

    if result.returncode != 0:
        print("--- STDERR ---")
        print(result.stderr[-1500:])
        raise RuntimeError("v2e exited with non-zero return code — see output above")

    dvs_txt = v2e_out_dir / "dvs_events.txt"
    if dvs_txt.exists():
        print(f"\nEvents file: {dvs_txt}  ({dvs_txt.stat().st_size / 1e3:.1f} KB)")
    else:
        print("WARNING: dvs_events.txt not found — check v2e output above.")

    return v2e_out_dir


# ===========================================================================
# CELL 4 — Load simulated events
# ===========================================================================
def load_sim_events(v2e_out_dir: Path):
    dvs_txt = v2e_out_dir / "dvs_events.txt"
    print(f"Loading {dvs_txt} ...")

    sim_raw = np.loadtxt(str(dvs_txt), comments="#")
    if sim_raw.ndim == 1:
        sim_raw = sim_raw[None, :]

    t_sim = sim_raw[:, 0]
    x_sim = sim_raw[:, 1].astype(np.int32)
    y_sim = sim_raw[:, 2].astype(np.int32)
    p_sim = sim_raw[:, 3].astype(np.int8)

    print(f"Simulated events : {len(t_sim):,}")
    print(f"  time span      : {t_sim[0]:.4f}s -> {t_sim[-1]:.4f}s")
    print(f"  ON             : {(p_sim == 1).sum():,}   OFF: {(p_sim == 0).sum():,}")
    print(
        f"  x              : {x_sim.min()} - {x_sim.max()}   "
        f"y: {y_sim.min()} - {y_sim.max()}"
    )

    return t_sim, x_sim, y_sim, p_sim


# ===========================================================================
# CELL 5 — Display simulated events with bounding boxes
# ===========================================================================
def display_sim_events(t_sim, x_sim, y_sim, p_sim):
    # --- load ground-truth data produced by earlier notebook sections ---
    timestamps_txt = DATA_DIR / "images" / "timestamps.txt"
    tracks_npy = DATA_DIR / "object_detections" / "left" / "tracks.npy"

    img_timestamps = np.loadtxt(str(timestamps_txt), dtype=np.int64)  # µs
    tracks = np.load(str(tracks_npy))

    # --- sorted input frames ---
    distorted_dir = DATA_DIR / "images" / "left" / "distorted"
    available_frames = sorted(distorted_dir.glob("*.png"), key=lambda p: int(p.stem))
    frame_indices = [int(p.stem) for p in available_frames]

    # absolute seconds (DSEC) and v2e-normalised seconds (start at 0)
    ts_s_abs = img_timestamps[frame_indices] / 1e6
    ts_s_norm = ts_s_abs - ts_s_abs[0]

    half_win_s = 0.025  # ±25 ms around each frame
    half_win_us = 25_000  # same in µs for DSEC bbox lookup

    n_show = min(6, len(frame_indices))
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    for ax_idx in range(n_show):
        row, col = divmod(ax_idx, 3)
        frame_idx = frame_indices[ax_idx]
        frame_ts_v2e = ts_s_norm[ax_idx]
        frame_ts_dsec = img_timestamps[frame_idx]

        # --- simulated events in ±25 ms window ---
        mask = (t_sim >= frame_ts_v2e - half_win_s) & (
            t_sim < frame_ts_v2e + half_win_s
        )
        xv, yv, pv = x_sim[mask], y_sim[mask], p_sim[mask]

        # black canvas
        ev_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        valid = (xv >= 0) & (xv < 640) & (yv >= 0) & (yv < 480)
        pos = valid & (pv == 1)
        neg = valid & (pv == 0)
        ev_frame[yv[pos], xv[pos]] = [255, 50, 50]  # red  = ON
        ev_frame[yv[neg], xv[neg]] = [50, 50, 255]  # blue = OFF

        # --- DSEC ground-truth bboxes ---
        frame_tracks = tracks[
            (tracks["t"] >= frame_ts_dsec - half_win_us)
            & (tracks["t"] < frame_ts_dsec + half_win_us)
        ]
        for tr in frame_tracks:
            bx = max(0, int(tr["x"]))
            by = max(0, int(tr["y"]))
            bx2 = min(bx + int(tr["w"]), 640)
            by2 = min(by + int(tr["h"]), 480)
            cls = int(tr["class_id"])
            color = CLASS_COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(ev_frame, (bx, by), (bx2, by2), color, 2)
            cv2.putText(
                ev_frame,
                CLASS_NAMES.get(cls, str(cls)),
                (bx, max(by - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        axes[row, col].imshow(ev_frame)
        axes[row, col].set_title(
            f"Frame {frame_idx}  |  {mask.sum():,} events  |  {len(frame_tracks)} objs",
            fontsize=9,
        )
        axes[row, col].axis("off")

    for ax_idx in range(n_show, 6):
        row, col = divmod(ax_idx, 3)
        axes[row, col].axis("off")

    plt.suptitle(
        "Simulated event frames — v2e on left/distorted 640x480 (clean, disable_slomo)\n"
        "red=ON, blue=OFF  |  bboxes from DSEC ground truth",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    install_v2e()
    v2e_out_dir = run_v2e()
    t_sim, x_sim, y_sim, p_sim = load_sim_events(v2e_out_dir)
    display_sim_events(t_sim, x_sim, y_sim, p_sim)

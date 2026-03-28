#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

NOISE_TYPES = [
    "gaussian_noise",
    "snow",
    "fog",
    "frost",
]
SEVERITIES = [1, 2, 3, 4, 5]
DISPLAY_NAMES = {
    "gaussian_noise": "Gaussian noise",
    "snow": "Snow",
    "fog": "Fog",
    "frost": "Frost",
}


def pick_representative_frame(seq_dir: Path) -> tuple[int, tuple[int, int]]:
    ts = np.loadtxt(seq_dir / "images/timestamps.txt", dtype=np.int64)
    tracks = np.load(seq_dir / "object_detections/left/tracks.npy", allow_pickle=False)
    preferred = {0, 2}

    best_idx = None
    best_score = (-1, -1)
    track_t = tracks["t"].astype(np.int64)
    track_cls = tracks["class_id"].astype(int)
    for i, t in enumerate(ts):
        m = track_t == int(t)
        if not np.any(m):
            continue
        cls = track_cls[m]
        score = (int(np.isin(cls, list(preferred)).sum()), int(m.sum()))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        best_idx = len(ts) // 2
    return int(best_idx), best_score


def h5_searchsorted(ds, target: int, side: str = "left") -> int:
    lo, hi = 0, ds.shape[0]
    tgt = int(target)
    right = side == "right"
    while lo < hi:
        mid = (lo + hi) // 2
        t_mid = int(ds[mid, 0])
        if t_mid < tgt or (right and t_mid == tgt):
            lo = mid + 1
        else:
            hi = mid
    return lo


def event_rgb_from_window(h5_path: Path, start_us: int, end_us: int, h: int, w: int, clip_v: float = 5.0) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        ds = f["events"]
        lo = h5_searchsorted(ds, start_us, side="left")
        hi = h5_searchsorted(ds, end_us, side="left")
        pos = np.zeros((h, w), dtype=np.float32)
        neg = np.zeros((h, w), dtype=np.float32)
        if hi > lo:
            block = ds[lo:hi]
            x = block[:, 1].astype(np.int64, copy=False)
            y = block[:, 2].astype(np.int64, copy=False)
            p = block[:, 3].astype(np.uint8, copy=False)
            valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x = x[valid]
            y = y[valid]
            p = p[valid] > 0
            if x.size:
                if np.any(p):
                    np.add.at(pos, (y[p], x[p]), 1.0)
                if np.any(~p):
                    np.add.at(neg, (y[~p], x[~p]), 1.0)

    pos = np.log1p(pos)
    neg = np.log1p(neg)
    pos = np.clip(pos, 0.0, clip_v) / clip_v
    neg = np.clip(neg, 0.0, clip_v) / clip_v
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = pos  # positive = red
    rgb[..., 2] = neg  # negative = blue
    return np.clip(rgb, 0.0, 1.0)


def rgb_frame(seq_dir: Path, noise: str, sev: int, frame_name: str) -> np.ndarray:
    p = seq_dir / f"images/left/distorted_gray_{noise}_s{sev}" / frame_name
    img = plt.imread(p)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def sim_h5(seq_dir: Path, noise: str, sev: int) -> Path:
    return seq_dir / f"v2e_output_distorted_gray_{noise}_s{sev}_clean" / "dvs_events.h5"


def get_time_window(seq_dir: Path, frame_idx: int) -> tuple[int, int]:
    ts_abs = np.loadtxt(seq_dir / "images/timestamps.txt", dtype=np.int64)
    ts_rel = ts_abs - int(ts_abs[0])
    start_ts = int(ts_rel[frame_idx - 1]) if frame_idx > 0 else int(ts_rel[frame_idx] - 50_000)
    end_ts = int(ts_rel[frame_idx])
    return start_ts, end_ts


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix(".png"), dpi=300)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300)
    plt.close(fig)


def make_pruned_grid(seq_dir: Path, out_dir: Path, frame_idx: int) -> None:
    frame_name = f"{frame_idx:06d}.png"
    s_ts, e_ts = get_time_window(seq_dir, frame_idx)

    sample = plt.imread(seq_dir / "images/left/distorted_gray" / frame_name)
    if sample.ndim == 3:
        sample = sample[..., 0]
    h, w = sample.shape[:2]

    rows = len(NOISE_TYPES)
    cols = len(SEVERITIES)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(6.70, 3.85),
        dpi=300,
        gridspec_kw={"hspace": 0.035, "wspace": 0.012},
    )

    if rows == 1:
        axs = np.expand_dims(axs, 0)

    for c, sev in enumerate(SEVERITIES):
        axs[0, c].set_title(f"S{sev}", fontsize=7, pad=2)

    for r, noise in enumerate(NOISE_TYPES):
        for c, sev in enumerate(SEVERITIES):
            ax = axs[r, c]
            ax.set_axis_off()
            rgb = rgb_frame(seq_dir, noise, sev, frame_name)
            ev = event_rgb_from_window(sim_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)

            top = ax.inset_axes([0.0, 0.5, 1.0, 0.5])
            bot = ax.inset_axes([0.0, 0.0, 1.0, 0.5])
            top.imshow(rgb, cmap="gray", vmin=0, vmax=255 if rgb.dtype != np.float32 else None)
            bot.imshow(ev)
            top.set_axis_off()
            bot.set_axis_off()

        bb = axs[r, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        fig.text(0.0135, y, DISPLAY_NAMES.get(noise, noise), ha="left", va="center", fontsize=6.3, weight="bold")

    # Minimal, clean global tags for split cells.
    fig.text(0.205, 0.982, "top: RGB", ha="left", va="top", fontsize=5.6)
    fig.text(0.285, 0.982, "bottom: Events (sim)", ha="left", va="top", fontsize=5.6)

    fig.subplots_adjust(left=0.175, right=0.996, top=0.952, bottom=0.02)
    out_base = out_dir / "compact_variant_D_pruned_no_native_clean_tags"
    save_fig(fig, out_base)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-dir", type=Path, default=Path("/workspace/data/dsec/test/zurich_city_14_c"))
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/noise_in_snn_analysis_push_compact/results/analysis/noise_corruption_grid_appendix/compact_variants"),
    )
    p.add_argument("--frame-idx", type=int, default=-1)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_idx >= 0:
        frame_idx = args.frame_idx
        score = (-1, -1)
    else:
        frame_idx, score = pick_representative_frame(args.seq_dir)

    make_pruned_grid(args.seq_dir, args.out_dir, frame_idx)

    meta = args.out_dir / "compact_variant_D_pruned_no_native_clean_tags.meta.txt"
    meta.write_text(
        f"frame_idx={frame_idx}\n"
        f"score={score}\n"
        "noise_types=gaussian_noise,snow,fog,frost\n"
        "excluded=shot_noise,impulse_noise,glass_blur,motion_blur,native_event_noise\n"
        "rendering=red_blue_dark_background_snn_like_log1p_clip5\n"
    )

    print(f"frame_idx={frame_idx} score={score}")
    print(f"out={args.out_dir / 'compact_variant_D_pruned_no_native_clean_tags'}")


if __name__ == "__main__":
    main()

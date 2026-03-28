#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

NOISE_TYPES = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'glass_blur',
    'motion_blur',
    'snow',
    'fog',
    'frost',
]
SEVERITIES = [1, 2, 3, 4, 5]
NATIVE_ROWS = [
    ('shot_noise', 3),
    ('leak_noise', 3),
    ('threshold_jitter', 3),
    ('bandwidth_limit', 3),
]


def pick_representative_frame(seq_dir: Path) -> tuple[int, tuple[int, int]]:
    ts = np.loadtxt(seq_dir / 'images/timestamps.txt', dtype=np.int64)
    tracks = np.load(seq_dir / 'object_detections/left/tracks.npy', allow_pickle=False)
    preferred = {0, 2}

    best_idx = None
    best_score = (-1, -1)
    track_t = tracks['t'].astype(np.int64)
    track_cls = tracks['class_id'].astype(int)
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


def h5_searchsorted(ds, target: int, side: str = 'left') -> int:
    lo, hi = 0, ds.shape[0]
    tgt = int(target)
    right = side == 'right'
    while lo < hi:
        mid = (lo + hi) // 2
        t_mid = int(ds[mid, 0])
        if t_mid < tgt or (right and t_mid == tgt):
            lo = mid + 1
        else:
            hi = mid
    return lo


def event_rgb_from_window(h5_path: Path, start_us: int, end_us: int, h: int, w: int, clip_v: float = 5.0) -> np.ndarray:
    with h5py.File(h5_path, 'r') as f:
        ds = f['events']
        lo = h5_searchsorted(ds, start_us, side='left')
        hi = h5_searchsorted(ds, end_us, side='left')
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
    rgb[..., 0] = pos  # positive red
    rgb[..., 2] = neg  # negative blue
    return np.clip(rgb, 0.0, 1.0)


def rgb_frame(seq_dir: Path, noise: str, sev: int, frame_name: str) -> np.ndarray:
    p = seq_dir / f'images/left/distorted_gray_{noise}_s{sev}' / frame_name
    img = plt.imread(p)
    if img.ndim == 3:
        img = img[..., 0]
    return img


def sim_h5(seq_dir: Path, noise: str, sev: int) -> Path:
    return seq_dir / f'v2e_output_distorted_gray_{noise}_s{sev}_clean' / 'dvs_events.h5'


def native_h5(seq_dir: Path, noise: str, sev: int) -> Path:
    return seq_dir / f'v2e_output_distorted_gray_{noise}_s{sev}' / 'dvs_events.h5'


def get_time_window(seq_dir: Path, frame_idx: int) -> tuple[int, int]:
    ts_abs = np.loadtxt(seq_dir / 'images/timestamps.txt', dtype=np.int64)
    ts_rel = ts_abs - int(ts_abs[0])
    start_ts = int(ts_rel[frame_idx - 1]) if frame_idx > 0 else int(ts_rel[frame_idx] - 50_000)
    end_ts = int(ts_rel[frame_idx])
    return start_ts, end_ts


def save_fig(fig: plt.Figure, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix('.png'), dpi=300)
    fig.savefig(out_base.with_suffix('.pdf'), dpi=300)
    plt.close(fig)


def make_variant_a(seq_dir: Path, out_dir: Path, frame_idx: int) -> None:
    """A: 1 row per noise, each cell split top=RGB bottom=Events(sim). Native rows below."""
    frame_name = f'{frame_idx:06d}.png'
    s_ts, e_ts = get_time_window(seq_dir, frame_idx)

    sample = plt.imread(seq_dir / 'images/left/distorted_gray' / frame_name)
    if sample.ndim == 3:
        sample = sample[..., 0]
    h, w = sample.shape[:2]

    rows = len(NOISE_TYPES) + len(NATIVE_ROWS)
    fig, axs = plt.subplots(rows, 5, figsize=(6.70, 8.8), dpi=300, gridspec_kw={'hspace': 0.03, 'wspace': 0.02})

    for c, sev in enumerate(SEVERITIES):
        axs[0, c].set_title(f'S{sev}', fontsize=7, pad=2)

    # Main rows
    for r, noise in enumerate(NOISE_TYPES):
        for c, sev in enumerate(SEVERITIES):
            ax = axs[r, c]
            ax.set_axis_off()
            rgb = rgb_frame(seq_dir, noise, sev, frame_name)
            ev = event_rgb_from_window(sim_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)

            # Split cell into upper/lower halves with inset axes
            top = ax.inset_axes([0.0, 0.5, 1.0, 0.5])
            bot = ax.inset_axes([0.0, 0.0, 1.0, 0.5])
            top.imshow(rgb, cmap='gray', vmin=0, vmax=255 if rgb.dtype != np.float32 else None)
            bot.imshow(ev)
            top.set_axis_off(); bot.set_axis_off()

        bb = axs[r, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        fig.text(0.012, y, noise, ha='left', va='center', fontsize=6.0, weight='bold')

    # Native rows: center column only
    base = len(NOISE_TYPES)
    for i, (noise, sev) in enumerate(NATIVE_ROWS):
        rr = base + i
        for c in range(5):
            axs[rr, c].set_axis_off()
        ev = event_rgb_from_window(native_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)
        axs[rr, 2].imshow(ev)
        axs[rr, 2].set_axis_off()
        bb = axs[rr, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        if i == 0:
            fig.text(0.012, y + 0.018, 'Native event noise', ha='left', va='center', fontsize=6.1, weight='bold')
        fig.text(0.028, y, f'{noise} (s{sev})', ha='left', va='center', fontsize=5.6)

    fig.text(0.12, 0.985, 'top: RGB  |  bottom: Events (sim)', ha='left', va='top', fontsize=5.6)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.975, bottom=0.015)
    save_fig(fig, out_dir / 'compact_variant_A_stacked_cell')


def make_variant_b(seq_dir: Path, out_dir: Path, frame_idx: int) -> None:
    """B: Interleaved columns per severity: RGB|EV for each severity (10 columns)."""
    frame_name = f'{frame_idx:06d}.png'
    s_ts, e_ts = get_time_window(seq_dir, frame_idx)

    sample = plt.imread(seq_dir / 'images/left/distorted_gray' / frame_name)
    if sample.ndim == 3:
        sample = sample[..., 0]
    h, w = sample.shape[:2]

    rows = len(NOISE_TYPES) + len(NATIVE_ROWS)
    cols = 10
    fig, axs = plt.subplots(rows, cols, figsize=(6.70, 7.6), dpi=300, gridspec_kw={'hspace': 0.03, 'wspace': 0.01})

    for s_i, sev in enumerate(SEVERITIES):
        c0 = 2 * s_i
        c1 = c0 + 1
        axs[0, c0].set_title(f'S{sev} RGB', fontsize=5.6, pad=2)
        axs[0, c1].set_title(f'S{sev} EV', fontsize=5.6, pad=2)

    for r, noise in enumerate(NOISE_TYPES):
        for s_i, sev in enumerate(SEVERITIES):
            rgb = rgb_frame(seq_dir, noise, sev, frame_name)
            ev = event_rgb_from_window(sim_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)
            axs[r, 2 * s_i].imshow(rgb, cmap='gray', vmin=0, vmax=255 if rgb.dtype != np.float32 else None)
            axs[r, 2 * s_i + 1].imshow(ev)
        for c in range(cols):
            axs[r, c].set_axis_off()
        bb = axs[r, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        fig.text(0.012, y, noise, ha='left', va='center', fontsize=5.8, weight='bold')

    base = len(NOISE_TYPES)
    for i, (noise, sev) in enumerate(NATIVE_ROWS):
        rr = base + i
        for c in range(cols):
            axs[rr, c].set_axis_off()
        ev = event_rgb_from_window(native_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)
        # place in middle pair
        axs[rr, 4].imshow(ev)
        axs[rr, 4].set_axis_off()
        bb = axs[rr, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        if i == 0:
            fig.text(0.012, y + 0.016, 'Native event noise', ha='left', va='center', fontsize=6.0, weight='bold')
        fig.text(0.028, y, f'{noise} (s{sev})', ha='left', va='center', fontsize=5.4)

    fig.subplots_adjust(left=0.16, right=0.995, top=0.975, bottom=0.015)
    save_fig(fig, out_dir / 'compact_variant_B_interleaved_cols')


def make_variant_c(seq_dir: Path, out_dir: Path, frame_idx: int) -> None:
    """C: Main 8x5 stacked-cell grid + compact native strip of 4 cells at bottom."""
    frame_name = f'{frame_idx:06d}.png'
    s_ts, e_ts = get_time_window(seq_dir, frame_idx)

    sample = plt.imread(seq_dir / 'images/left/distorted_gray' / frame_name)
    if sample.ndim == 3:
        sample = sample[..., 0]
    h, w = sample.shape[:2]

    rows = len(NOISE_TYPES) + 1
    fig, axs = plt.subplots(rows, 5, figsize=(6.70, 7.9), dpi=300, gridspec_kw={'hspace': 0.035, 'wspace': 0.02, 'height_ratios':[1]*len(NOISE_TYPES)+[0.85]})

    for c, sev in enumerate(SEVERITIES):
        axs[0, c].set_title(f'S{sev}', fontsize=7, pad=2)

    for r, noise in enumerate(NOISE_TYPES):
        for c, sev in enumerate(SEVERITIES):
            ax = axs[r, c]
            ax.set_axis_off()
            rgb = rgb_frame(seq_dir, noise, sev, frame_name)
            ev = event_rgb_from_window(sim_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)
            top = ax.inset_axes([0.0, 0.5, 1.0, 0.5])
            bot = ax.inset_axes([0.0, 0.0, 1.0, 0.5])
            top.imshow(rgb, cmap='gray', vmin=0, vmax=255 if rgb.dtype != np.float32 else None)
            bot.imshow(ev)
            top.set_axis_off(); bot.set_axis_off()
        bb = axs[r, 0].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        fig.text(0.012, y, noise, ha='left', va='center', fontsize=6.0, weight='bold')

    # Native strip
    rr = len(NOISE_TYPES)
    for c in range(5):
        axs[rr, c].set_axis_off()
    fig.text(0.012, 0.5*(axs[rr,0].get_position().y0+axs[rr,0].get_position().y1), 'Native event noise', ha='left', va='center', fontsize=6.0, weight='bold')

    for i, (noise, sev) in enumerate(NATIVE_ROWS):
        x0 = 0.22 + i * 0.19
        y0 = axs[rr, 0].get_position().y0 + 0.005
        w0 = 0.15
        h0 = axs[rr, 0].get_position().height - 0.012
        ax = fig.add_axes([x0, y0, w0, h0])
        ev = event_rgb_from_window(native_h5(seq_dir, noise, sev), s_ts, e_ts, h, w)
        ax.imshow(ev)
        ax.set_axis_off()
        fig.text(x0 + w0/2, y0 - 0.004, f'{noise} (s{sev})', ha='center', va='top', fontsize=5.3)

    fig.text(0.12, 0.985, 'top: RGB  |  bottom: Events (sim)', ha='left', va='top', fontsize=5.6)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.975, bottom=0.02)
    save_fig(fig, out_dir / 'compact_variant_C_native_strip')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--seq-dir', type=Path, default=Path('/workspace/data/dsec/test/zurich_city_14_c'))
    p.add_argument('--out-dir', type=Path, default=Path('/workspace/noise_in_snn/results/analysis/noise_corruption_grid_appendix/compact_variants'))
    p.add_argument('--frame-idx', type=int, default=-1)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.frame_idx >= 0:
        frame_idx = args.frame_idx
        score = (-1, -1)
    else:
        frame_idx, score = pick_representative_frame(args.seq_dir)

    make_variant_a(args.seq_dir, args.out_dir, frame_idx)
    make_variant_b(args.seq_dir, args.out_dir, frame_idx)
    make_variant_c(args.seq_dir, args.out_dir, frame_idx)

    meta = args.out_dir / 'compact_variants.meta.txt'
    meta.write_text(
        f'frame_idx={frame_idx}\n'
        f'score={score}\n'
        'rendering=red_blue_dark_background_snn_like_log1p_clip5\n'
        'variants=compact_variant_A_stacked_cell,compact_variant_B_interleaved_cols,compact_variant_C_native_strip\n'
    )

    print(f'frame_idx={frame_idx} score={score}')
    print(f'out_dir={args.out_dir}')


if __name__ == '__main__':
    main()

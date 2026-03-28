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
    preferred = {0, 2}  # ped/car

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


def event_rgb_from_window(
    h5_path: Path,
    start_us: int,
    end_us: int,
    h: int,
    w: int,
    count_clip_value: float = 5.0,
) -> np.ndarray:
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

    # Match SNN preprocessing behavior: log1p + clip/normalize.
    pos = np.log1p(pos)
    neg = np.log1p(neg)
    if count_clip_value > 0:
        pos = np.clip(pos, 0.0, count_clip_value) / count_clip_value
        neg = np.clip(neg, 0.0, count_clip_value) / count_clip_value

    # Dark-off palette: background black, pos red, neg blue.
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[..., 0] = pos
    rgb[..., 2] = neg
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


def make_appendix(seq_dir: Path, out_dir: Path, frame_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_name = f'{frame_idx:06d}.png'

    ts_abs = np.loadtxt(seq_dir / 'images/timestamps.txt', dtype=np.int64)
    ts_rel = ts_abs - int(ts_abs[0])
    start_ts = int(ts_rel[frame_idx - 1]) if frame_idx > 0 else int(ts_rel[frame_idx] - 50_000)
    end_ts = int(ts_rel[frame_idx])

    sample = plt.imread(seq_dir / 'images/left/distorted_gray' / frame_name)
    if sample.ndim == 3:
        sample = sample[..., 0]
    h, w = sample.shape[:2]

    rows_top = len(NOISE_TYPES) * 2
    rows_bottom = len(NATIVE_ROWS)
    rows_total = rows_top + 1 + rows_bottom
    height_ratios = [1.0] * rows_top + [0.22] + [1.0] * rows_bottom

    fig, axs = plt.subplots(
        rows_total,
        5,
        figsize=(6.70, 11.8),
        dpi=300,
        gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.03, 'wspace': 0.02},
    )

    for c, sev in enumerate(SEVERITIES):
        axs[0, c].set_title(f'Severity {sev}', fontsize=7, pad=3)

    for g, noise in enumerate(NOISE_TYPES):
        r_rgb = g * 2
        r_ev = r_rgb + 1
        for c, sev in enumerate(SEVERITIES):
            ax_rgb = axs[r_rgb, c]
            ax_ev = axs[r_ev, c]
            rgb = rgb_frame(seq_dir, noise, sev, frame_name)
            ev_rgb = event_rgb_from_window(sim_h5(seq_dir, noise, sev), start_ts, end_ts, h, w)

            ax_rgb.imshow(rgb, cmap='gray', vmin=0, vmax=255 if rgb.dtype != np.float32 else None)
            ax_ev.imshow(ev_rgb)
            ax_rgb.set_axis_off()
            ax_ev.set_axis_off()

    gap_row = rows_top
    for c in range(5):
        axs[gap_row, c].set_axis_off()

    native_start = rows_top + 1
    for i, (noise, sev) in enumerate(NATIVE_ROWS):
        r = native_start + i
        for c in range(5):
            axs[r, c].set_axis_off()
        c = 2
        ev_rgb = event_rgb_from_window(native_h5(seq_dir, noise, sev), start_ts, end_ts, h, w)
        axs[r, c].imshow(ev_rgb)
        axs[r, c].text(
            0.02,
            0.95,
            f's{sev}',
            transform=axs[r, c].transAxes,
            ha='left',
            va='top',
            fontsize=6,
            color='white',
            bbox=dict(facecolor='black', alpha=0.35, edgecolor='none', pad=1.2),
        )

    # Labels on left.
    for g, noise in enumerate(NOISE_TYPES):
        r_rgb = g * 2
        r_ev = r_rgb + 1
        bb_rgb = axs[r_rgb, 0].get_position()
        bb_ev = axs[r_ev, 0].get_position()
        y_group = 0.5 * (bb_rgb.y0 + bb_ev.y1)
        y_rgb = 0.5 * (bb_rgb.y0 + bb_rgb.y1)
        y_ev = 0.5 * (bb_ev.y0 + bb_ev.y1)
        fig.text(0.012, y_group, noise, ha='left', va='center', fontsize=6.2, weight='bold')
        fig.text(0.118, y_rgb, 'RGB', ha='right', va='center', fontsize=5.8)
        fig.text(0.118, y_ev, 'Events (sim)', ha='right', va='center', fontsize=5.8)

    native_bb_top = axs[native_start, 2].get_position()
    native_bb_bot = axs[native_start + rows_bottom - 1, 2].get_position()
    y_native = 0.5 * (native_bb_top.y1 + native_bb_bot.y0)
    fig.text(0.012, y_native, 'Native event noise', ha='left', va='center', fontsize=6.5, weight='bold')
    for i, (noise, sev) in enumerate(NATIVE_ROWS):
        bb = axs[native_start + i, 2].get_position()
        y = 0.5 * (bb.y0 + bb.y1)
        fig.text(0.028, y, f'{noise} (s{sev})', ha='left', va='center', fontsize=5.8)

    fig.subplots_adjust(left=0.16, right=0.995, top=0.985, bottom=0.015)

    png_path = out_dir / 'noise_corruption_grid_appendix.png'
    pdf_path = out_dir / 'noise_corruption_grid_appendix.pdf'
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    (out_dir / 'noise_corruption_grid_appendix.meta.txt').write_text(
        f'frame_idx={frame_idx}\n'
        f'frame_file={frame_name}\n'
        f'start_ts_us={start_ts}\n'
        f'end_ts_us={end_ts}\n'
        'rendering=snn_like_log1p_clip5_dark_background\n'
    )


def make_v4(seq_dir: Path, out_dir: Path, frame_idx: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_name = f'{frame_idx:06d}.png'

    ts_abs = np.loadtxt(seq_dir / 'images/timestamps.txt', dtype=np.int64)
    ts_rel = ts_abs - int(ts_abs[0])
    start_ts = int(ts_rel[frame_idx - 1]) if frame_idx > 0 else int(ts_rel[frame_idx] - 50_000)
    end_ts = int(ts_rel[frame_idx])

    sample = plt.imread(seq_dir / 'images/left/distorted_gray' / frame_name)
    if sample.ndim == 3:
        sample = sample[..., 0]
    h, w = sample.shape[:2]

    rows = len(NOISE_TYPES) * 2
    fig, axs = plt.subplots(
        rows,
        5,
        figsize=(6.70, 9.8),
        dpi=300,
        gridspec_kw={'hspace': 0.03, 'wspace': 0.02},
    )

    for c, sev in enumerate(SEVERITIES):
        axs[0, c].set_title(f'Severity {sev}', fontsize=7, pad=3)

    for g, noise in enumerate(NOISE_TYPES):
        r_rgb = g * 2
        r_ev = r_rgb + 1
        for c, sev in enumerate(SEVERITIES):
            rgb = rgb_frame(seq_dir, noise, sev, frame_name)
            ev_rgb = event_rgb_from_window(sim_h5(seq_dir, noise, sev), start_ts, end_ts, h, w)
            axs[r_rgb, c].imshow(rgb, cmap='gray', vmin=0, vmax=255 if rgb.dtype != np.float32 else None)
            axs[r_ev, c].imshow(ev_rgb)
            axs[r_rgb, c].set_axis_off()
            axs[r_ev, c].set_axis_off()

    for g, noise in enumerate(NOISE_TYPES):
        r_rgb = g * 2
        r_ev = r_rgb + 1
        bb_rgb = axs[r_rgb, 0].get_position()
        bb_ev = axs[r_ev, 0].get_position()
        y_group = 0.5 * (bb_rgb.y0 + bb_ev.y1)
        y_rgb = 0.5 * (bb_rgb.y0 + bb_rgb.y1)
        y_ev = 0.5 * (bb_ev.y0 + bb_ev.y1)
        fig.text(0.012, y_group, noise, ha='left', va='center', fontsize=6.2, weight='bold')
        fig.text(0.118, y_rgb, 'RGB', ha='right', va='center', fontsize=5.8)
        fig.text(0.118, y_ev, 'Events (sim)', ha='right', va='center', fontsize=5.8)

    fig.subplots_adjust(left=0.16, right=0.995, top=0.985, bottom=0.02)
    fig.savefig(out_dir / 'noise_severity_grid_v4.png', dpi=300)
    fig.savefig(out_dir / 'noise_severity_grid_v4.pdf', dpi=300)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--seq-dir', type=Path, default=Path('/workspace/data/dsec/test/zurich_city_14_c'))
    p.add_argument('--appendix-out', type=Path, default=Path('/workspace/noise_in_snn/results/analysis/noise_corruption_grid_appendix'))
    p.add_argument('--v4-out-dir', type=Path, default=Path('/workspace/noise_in_snn/figures'))
    p.add_argument('--frame-idx', type=int, default=-1, help='-1 auto-select object-rich frame')
    args = p.parse_args()

    if args.frame_idx >= 0:
        frame_idx = args.frame_idx
        score = (-1, -1)
    else:
        frame_idx, score = pick_representative_frame(args.seq_dir)

    make_appendix(args.seq_dir, args.appendix_out, frame_idx)
    make_v4(args.seq_dir, args.v4_out_dir, frame_idx)

    print(f'frame_idx={frame_idx} score={score}')
    print(f'appendix={args.appendix_out}')
    print(f'v4={args.v4_out_dir}')


if __name__ == '__main__':
    main()

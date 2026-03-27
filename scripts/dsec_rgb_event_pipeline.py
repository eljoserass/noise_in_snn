#!/usr/bin/env python3
"""
DSEC RGB -> event generation pipeline.

Pipeline stages:
1) Optional label rectification validation (left distorted -> left rectified).
2) Grayscale conversion for fair RGB/event comparison.
3) ImageCorruptions sweep (all types x selected severities).
4) v2e conversion:
   - clean/noisy on clean + corrupted RGB folders
   - manual v2e noise sweeps on clean grayscale folder
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from imagecorruptions import corrupt, get_corruption_names
except ImportError as exc:
    raise ImportError(
        "imagecorruptions is required for this script. "
        "Install with: pip install imagecorruptions"
    ) from exc


MANUAL_V2E_NOISE_LEVELS = {
    # severity 1..5
    "shot_noise": [0.0005, 0.001, 0.005, 0.01, 0.05],
    "leak_noise": [0.001, 0.01, 0.03, 0.1, 0.3],
    "threshold_jitter": [0.01, 0.02, 0.03, 0.05, 0.08],
    # lower cutoff means heavier low-pass blur / slower response
    "bandwidth_limit": [500.0, 300.0, 200.0, 100.0, 50.0],
    "refractory": [0.0001, 0.0005, 0.001, 0.002, 0.005],
    "photoreceptor_noise": [0.001, 0.003, 0.005, 0.01, 0.02],
}


def parse_levels(levels: str) -> list[int]:
    parsed = [int(x.strip()) for x in levels.split(",") if x.strip()]
    for s in parsed:
        if s < 1 or s > 5:
            raise ValueError(f"Severity levels must be in [1,5], got: {s}")
    return sorted(set(parsed))


def parse_list(values: str) -> list[str]:
    parsed = [x.strip() for x in values.split(",") if x.strip()]
    return [x for x in parsed if x.lower() not in {"none", "off", "null"}]


def sorted_pngs(folder: Path, max_frames: int | None = None) -> list[Path]:
    frames = sorted(folder.glob("*.png"), key=lambda p: int(p.stem))
    if max_frames is not None:
        return frames[:max_frames]
    return frames


def convert_to_grayscale(src_frames: list[Path], out_dir: Path, skip_existing: bool) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = set(p.name for p in out_dir.iterdir()) if skip_existing else set()
    written = 0
    for src in src_frames:
        dst = out_dir / src.name
        if existing and src.name in existing:
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read frame: {src}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok = cv2.imwrite(str(dst), gray_3ch)
        if not ok:
            raise RuntimeError(f"Could not write frame: {dst}")
        written += 1
    return written


def apply_imagecorruptions(
    src_frames: list[Path],
    out_parent: Path,
    src_tag: str,
    corruption_names: list[str],
    severities: list[int],
    skip_existing: bool,
) -> dict[tuple[str, int], Path]:
    out_dirs: dict[tuple[str, int], Path] = {}
    for cname in corruption_names:
        for sev in severities:
            out_dir = out_parent / f"{src_tag}_{cname}_s{sev}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_dirs[(cname, sev)] = out_dir
            existing = set(p.name for p in out_dir.iterdir()) if skip_existing else set()
            written = 0
            failed = False
            error_msg = ""
            for src in src_frames:
                dst = out_dir / src.name
                if existing and src.name in existing:
                    continue
                img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    failed = True
                    error_msg = f"Could not read frame: {src}"
                    break
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                try:
                    corr_rgb = corrupt(img_rgb, corruption_name=cname, severity=sev)
                except Exception as exc:
                    failed = True
                    error_msg = str(exc)
                    break
                corr_bgr = cv2.cvtColor(corr_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(str(dst), corr_bgr)
                if not ok:
                    failed = True
                    error_msg = f"Could not write frame: {dst}"
                    break
                written += 1
            if failed:
                print(f"[corrupt][warn] {cname:<18} s{sev} failed, skipping this corruption: {error_msg}")
                continue
            total = len(list(out_dir.glob("*.png")))
            print(f"[corrupt] {cname:<18} s{sev} -> {out_dir.name} (total={total}, new={written})")
    return out_dirs


def run_v2e(
    python_exe: str,
    v2e_script: Path,
    input_dir: Path,
    output_dir: Path,
    fps: float,
    width: int,
    height: int,
    overwrite: bool,
    dvs_mode: str | None,
    exposure_duration_s: float,
    events_format: str,
    events_h5_name: str,
    events_txt_name: str,
    manual_args: list[str] | None = None,
) -> Path:
    cmd = [
        python_exe,
        str(v2e_script),
        "--input",
        str(input_dir),
        "--input_frame_rate",
        str(fps),
        "--output_folder",
        str(output_dir),
        "--output_width",
        str(width),
        "--output_height",
        str(height),
        "--disable_slomo",
        "--dvs_exposure",
        "duration",
        str(exposure_duration_s),
        "--unique_output_folder",
        "False",
        "--skip_video_output",
        "--no_preview",
    ]
    if events_format in ("h5", "both"):
        cmd += ["--dvs_h5", events_h5_name]
    if events_format in ("text", "both"):
        cmd += ["--dvs_text", events_txt_name]
    if overwrite:
        cmd.append("--overwrite")

    if dvs_mode in ("clean", "noisy"):
        cmd += ["--dvs_params", dvs_mode]
    else:
        cmd += ["--dvs_params", "None"]
        if manual_args:
            cmd += manual_args

    print(f"[v2e] input={input_dir.name} -> out={output_dir.name} mode={dvs_mode or 'manual'}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("--- v2e stderr tail ---")
        print(res.stderr[-2000:])
        raise RuntimeError(f"v2e failed for input: {input_dir}")

    expected: list[Path] = []
    if events_format in ("h5", "both"):
        expected.append(output_dir / events_h5_name)
    if events_format in ("text", "both"):
        expected.append(output_dir / events_txt_name)

    existing = [p for p in expected if p.exists()]
    if existing:
        return existing[0]

    if not existing:
        # Some v2e builds can still materialize into suffixed folders if output_dir is non-empty.
        parent = output_dir.parent
        stem = output_dir.name
        candidates: list[Path] = []
        for out in parent.glob(f"{stem}*"):
            if not out.is_dir():
                continue
            for p in (
                out / events_h5_name,
                out / events_txt_name,
            ):
                if p.exists():
                    candidates.append(p)
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            resolved = candidates[0]
            print(f"[v2e] output resolved from sibling folder: {resolved}")
            return resolved
        print("--- v2e stdout tail ---")
        print(res.stdout[-2000:])
        print("--- v2e stderr tail ---")
        print(res.stderr[-2000:])
        raise RuntimeError(f"v2e output missing in: {output_dir}")
    return existing[0]


def manual_v2e_args(noise_type: str, severity: int) -> list[str]:
    idx = severity - 1
    if noise_type == "shot_noise":
        return ["--shot_noise_rate_hz", str(MANUAL_V2E_NOISE_LEVELS[noise_type][idx])]
    if noise_type == "leak_noise":
        return ["--leak_rate_hz", str(MANUAL_V2E_NOISE_LEVELS[noise_type][idx])]
    if noise_type == "threshold_jitter":
        return ["--sigma_thres", str(MANUAL_V2E_NOISE_LEVELS[noise_type][idx])]
    if noise_type == "bandwidth_limit":
        return ["--cutoff_hz", str(MANUAL_V2E_NOISE_LEVELS[noise_type][idx])]
    if noise_type == "refractory":
        return ["--refractory_period", str(MANUAL_V2E_NOISE_LEVELS[noise_type][idx])]
    if noise_type == "photoreceptor_noise":
        rate = MANUAL_V2E_NOISE_LEVELS[noise_type][idx]
        return ["--photoreceptor_noise", "--shot_noise_rate_hz", str(rate)]
    raise ValueError(f"Unsupported manual noise type: {noise_type}")


def infer_hw(frames: list[Path]) -> tuple[int, int]:
    if not frames:
        raise RuntimeError("No input frames found.")
    img = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")
    h, w = img.shape[:2]
    return w, h


def maybe_validate_rectification(
    sequence_dir: Path,
    python_exe: str,
    run: bool,
    max_frames: int,
) -> None:
    if not run:
        return
    validate_dir = sequence_dir / "object_detections" / "left" / "rectified_validation_remap_check"
    out_tracks = sequence_dir / "object_detections" / "left" / "tracks_rectified_remap_check.npy"
    cmd = [
        python_exe,
        str(Path(__file__).resolve().parent / "rectify_dsec_labels.py"),
        "--sequence-dir",
        str(sequence_dir),
        "--transform-method",
        "auto",
        "--time-mode",
        "nearest",
        "--max-frames",
        str(max_frames),
        "--output-tracks",
        str(out_tracks),
        "--validation-dir",
        str(validate_dir),
    ]
    print(f"[rectify] running validation on {sequence_dir}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("--- rectify stderr ---")
        print(res.stderr[-2000:])
        raise RuntimeError("Rectification validation failed.")
    print(res.stdout.strip())
    print(f"[rectify] validation images: {validate_dir}")
    print(f"[rectify] rectified tracks: {out_tracks}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DSEC RGB/event generation pipeline.")
    p.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="Path to one DSEC sequence folder (contains images/, events/, calibration/).",
    )
    p.add_argument(
        "--source-subdir",
        type=str,
        default="images/left/distorted",
        help="Source RGB frame folder relative to sequence dir.",
    )
    p.add_argument(
        "--grayscale-subdir",
        type=str,
        default="images/left/distorted_gray",
        help="Output grayscale folder relative to sequence dir.",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames for quick runs.")
    p.add_argument(
        "--severity-levels",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated severity levels for imagecorruptions/manual v2e noise.",
    )
    p.add_argument(
        "--corruption-subset",
        type=str,
        default="all",
        choices=["all", "common", "noise", "blur", "weather", "digital"],
        help="Subset passed to imagecorruptions get_corruption_names.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing images/events if outputs already exist.",
    )
    p.add_argument(
        "--validate-rectification",
        action="store_true",
        help="Run label rectification validation before generating data.",
    )
    p.add_argument(
        "--rectification-validation-frames",
        type=int,
        default=6,
        help="Frames for rectification montage if --validate-rectification is set.",
    )

    p.add_argument("--run-imagecorruptions", action="store_true", help="Generate imagecorruption sweeps.")
    p.add_argument("--run-v2e", action="store_true", help="Run v2e conversion.")
    p.add_argument(
        "--v2e-script",
        type=Path,
        default=Path("tools/v2e/v2e.py"),
        help="Path to v2e.py (relative to repo root or absolute).",
    )
    p.add_argument("--input-frame-rate", type=float, default=20.0, help="v2e input frame rate for folder input.")
    p.add_argument(
        "--v2e-exposure-duration",
        type=float,
        default=0.01,
        help="v2e DVS exposure duration in seconds (e.g., 0.005 for 5ms).",
    )
    p.add_argument("--v2e-width", type=int, default=None, help="Output width for v2e.")
    p.add_argument("--v2e-height", type=int, default=None, help="Output height for v2e.")
    p.add_argument(
        "--v2e-modes",
        type=str,
        default="clean,noisy",
        help="Comma-separated built-in v2e modes to run on clean+corrupted inputs. Use empty string to disable.",
    )
    p.add_argument(
        "--manual-v2e-noises",
        type=str,
        default="shot_noise,leak_noise,threshold_jitter,bandwidth_limit,refractory,photoreceptor_noise",
        help="Comma-separated manual v2e noise types to run on CLEAN grayscale input only.",
    )
    p.add_argument(
        "--v2e-on-corruptions",
        action="store_true",
        help="If set, run built-in v2e modes on each corruption folder too.",
    )
    p.add_argument(
        "--overwrite-v2e",
        action="store_true",
        help="Pass --overwrite to v2e runs.",
    )
    p.add_argument(
        "--v2e-events-format",
        type=str,
        default="h5",
        choices=["h5", "text", "both"],
        help="Which v2e event file format to write.",
    )
    p.add_argument(
        "--v2e-events-h5-name",
        type=str,
        default="dvs_events.h5",
        help="Output H5 event filename used when --v2e-events-format includes h5.",
    )
    p.add_argument(
        "--v2e-events-txt-name",
        type=str,
        default="dvs_events.txt",
        help="Output text event filename used when --v2e-events-format includes text.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    seq_dir: Path = args.sequence_dir.resolve()
    source_dir = (seq_dir / args.source_subdir).resolve()
    grayscale_dir = (seq_dir / args.grayscale_subdir).resolve()
    v2e_script = (repo_root / args.v2e_script).resolve() if not args.v2e_script.is_absolute() else args.v2e_script

    if not source_dir.exists():
        raise FileNotFoundError(f"Source frames dir not found: {source_dir}")
    if args.run_v2e and not v2e_script.exists():
        raise FileNotFoundError(f"v2e script not found: {v2e_script}")

    severities = parse_levels(args.severity_levels)
    corruption_names = get_corruption_names() if args.corruption_subset == "all" else get_corruption_names(args.corruption_subset)
    manual_noise_types = parse_list(args.manual_v2e_noises)
    built_in_modes = [m for m in parse_list(args.v2e_modes) if m in ("clean", "noisy")]
    skip_existing = args.skip_existing

    print(f"[cfg] sequence: {seq_dir}")
    print(f"[cfg] source frames: {source_dir}")
    print(f"[cfg] grayscale out: {grayscale_dir}")
    print(f"[cfg] severities: {severities}")
    print(
        f"[cfg] v2e events format: {args.v2e_events_format} "
        f"(h5={args.v2e_events_h5_name}, txt={args.v2e_events_txt_name})"
    )

    maybe_validate_rectification(
        sequence_dir=seq_dir,
        python_exe=sys.executable,
        run=args.validate_rectification,
        max_frames=args.rectification_validation_frames,
    )

    src_frames = sorted_pngs(source_dir, max_frames=args.max_frames)
    if not src_frames:
        raise RuntimeError(f"No PNG frames found in {source_dir}")
    print(f"[frames] found {len(src_frames)} source frames")

    n_gray_new = convert_to_grayscale(src_frames, grayscale_dir, skip_existing=skip_existing)
    print(f"[gray] ready: {grayscale_dir} (new={n_gray_new}, total={len(list(grayscale_dir.glob('*.png')))})")

    corruption_dirs: dict[tuple[str, int], Path] = {}
    if args.run_imagecorruptions:
        gray_frames = sorted_pngs(grayscale_dir, max_frames=args.max_frames)
        src_tag = grayscale_dir.name
        corruption_dirs = apply_imagecorruptions(
            src_frames=gray_frames,
            out_parent=grayscale_dir.parent,
            src_tag=src_tag,
            corruption_names=corruption_names,
            severities=severities,
            skip_existing=skip_existing,
        )
        print(f"[corrupt] total folders: {len(corruption_dirs)}")

    if not args.run_v2e:
        print("[done] image generation only (v2e skipped).")
        return

    gray_frames_for_v2e = sorted_pngs(grayscale_dir, max_frames=args.max_frames)
    if len(gray_frames_for_v2e) < 2:
        raise RuntimeError(
            "v2e folder input requires at least 2 frames. "
            f"Found {len(gray_frames_for_v2e)} in {grayscale_dir}."
        )

    if args.v2e_width is None or args.v2e_height is None:
        width, height = infer_hw(gray_frames_for_v2e[:1])
    else:
        width, height = args.v2e_width, args.v2e_height
    print(f"[v2e] output size: {width}x{height}")

    # 1) Built-in v2e clean/noisy on clean grayscale and optionally on corruption folders.
    base_inputs = [grayscale_dir]
    if args.v2e_on_corruptions:
        base_inputs += [corruption_dirs[k] for k in sorted(corruption_dirs.keys())]

    event_files: list[Path] = []
    for mode in built_in_modes:
        for inp in base_inputs:
            out_dir = seq_dir / f"v2e_output_{inp.name}_{mode}"
            preferred = (
                out_dir / args.v2e_events_h5_name
                if args.v2e_events_format in ("h5", "both")
                else out_dir / args.v2e_events_txt_name
            )
            if skip_existing and preferred.exists():
                print(f"[v2e-skip] {preferred}")
                event_files.append(preferred)
                continue
            event_files.append(
                run_v2e(
                    python_exe=sys.executable,
                    v2e_script=v2e_script,
                    input_dir=inp,
                    output_dir=out_dir,
                    fps=args.input_frame_rate,
                    width=width,
                    height=height,
                    overwrite=args.overwrite_v2e,
                    dvs_mode=mode,
                    exposure_duration_s=args.v2e_exposure_duration,
                    events_format=args.v2e_events_format,
                    events_h5_name=args.v2e_events_h5_name,
                    events_txt_name=args.v2e_events_txt_name,
                    manual_args=None,
                )
            )

    # 2) Manual v2e noise sweeps on clean grayscale input only.
    for noise_type in manual_noise_types:
        if noise_type not in MANUAL_V2E_NOISE_LEVELS:
            raise ValueError(
                f"Unknown manual noise type: {noise_type}. "
                f"Available: {list(MANUAL_V2E_NOISE_LEVELS.keys())}"
            )
        for sev in severities:
            manual_args = manual_v2e_args(noise_type, sev)
            out_dir = seq_dir / f"v2e_output_{grayscale_dir.name}_{noise_type}_s{sev}"
            preferred = (
                out_dir / args.v2e_events_h5_name
                if args.v2e_events_format in ("h5", "both")
                else out_dir / args.v2e_events_txt_name
            )
            if skip_existing and preferred.exists():
                print(f"[v2e-skip] {preferred}")
                event_files.append(preferred)
                continue
            event_files.append(
                run_v2e(
                    python_exe=sys.executable,
                    v2e_script=v2e_script,
                    input_dir=grayscale_dir,
                    output_dir=out_dir,
                    fps=args.input_frame_rate,
                    width=width,
                    height=height,
                    overwrite=args.overwrite_v2e,
                    dvs_mode=None,
                    exposure_duration_s=args.v2e_exposure_duration,
                    events_format=args.v2e_events_format,
                    events_h5_name=args.v2e_events_h5_name,
                    events_txt_name=args.v2e_events_txt_name,
                    manual_args=manual_args,
                )
            )

    print(f"[done] event files ready: {len(event_files)}")
    for p in event_files[:20]:
        print(f"  {p}")
    if len(event_files) > 20:
        print(f"  ... ({len(event_files) - 20} more)")


if __name__ == "__main__":
    main()

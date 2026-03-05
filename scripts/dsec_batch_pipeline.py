#!/usr/bin/env python3
"""
Batch runner for dsec_rgb_event_pipeline.py across splits/sequences.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def discover_sequences(root: Path, split: str, explicit: list[str] | None) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    if explicit:
        return [split_dir / name for name in explicit if (split_dir / name).exists()]
    return sorted([p for p in split_dir.iterdir() if p.is_dir()])


def run_one(
    pipeline_script: Path,
    sequence_dir: Path,
    extra_args: list[str],
) -> tuple[Path, int, str]:
    cmd = [sys.executable, str(pipeline_script), "--sequence-dir", str(sequence_dir)] + extra_args
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tail = proc.stdout[-1200:] if proc.stdout else ""
    if proc.returncode != 0 and proc.stderr:
        tail = (tail + "\n--- stderr ---\n" + proc.stderr[-1600:]).strip()
    return sequence_dir, proc.returncode, tail


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch DSEC RGB/event processing.")
    p.add_argument(
        "--dsec-root",
        type=Path,
        required=True,
        help="Root dir that contains split subdirs (train/val/test).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split names to process.",
    )
    p.add_argument(
        "--sequences",
        type=str,
        default="",
        help="Optional comma-separated sequence names (applies to all selected splits).",
    )
    p.add_argument("--jobs", type=int, default=1, help="Parallel sequence workers.")
    p.add_argument(
        "--pipeline-script",
        type=Path,
        default=Path("scripts/dsec_rgb_event_pipeline.py"),
        help="Path to sequence pipeline script.",
    )
    p.add_argument(
        "--extra-args",
        type=str,
        default="",
        help=(
            "Extra args forwarded to dsec_rgb_event_pipeline.py as raw string. "
            "Example: \"--run-imagecorruptions --run-v2e --v2e-on-corruptions --skip-existing\""
        ),
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    pipeline_script = (repo_root / args.pipeline_script).resolve() if not args.pipeline_script.is_absolute() else args.pipeline_script
    dsec_root = args.dsec_root.resolve()

    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")
    if not dsec_root.exists():
        raise FileNotFoundError(f"DSEC root not found: {dsec_root}")

    splits = parse_list(args.splits)
    explicit_sequences = parse_list(args.sequences) if args.sequences.strip() else None
    extra_args = args.extra_args.split() if args.extra_args.strip() else []

    all_sequences: list[Path] = []
    for split in splits:
        seqs = discover_sequences(dsec_root, split, explicit_sequences)
        print(f"[discover] split={split} sequences={len(seqs)}")
        all_sequences.extend(seqs)

    if not all_sequences:
        raise RuntimeError("No sequences discovered. Check --dsec-root/--splits/--sequences.")

    print(f"[run] total sequences: {len(all_sequences)} | jobs={args.jobs}")
    failures: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, pipeline_script, seq, extra_args) for seq in all_sequences]
        for fut in as_completed(futures):
            seq, code, tail = fut.result()
            status = "OK" if code == 0 else "FAIL"
            print(f"[{status}] {seq}")
            if tail:
                print(tail)
            if code != 0:
                failures.append(seq)

    print(f"[done] total={len(all_sequences)} failures={len(failures)}")
    if failures:
        for seq in failures:
            print(f"  failed: {seq}")
        sys.exit(1)


if __name__ == "__main__":
    main()


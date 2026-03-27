#!/usr/bin/env python3
"""
Backfill existing ANN eval JSON outputs into Weights & Biases runs.

Designed for outputs like:
  .../ann_rgb_<noise>_s<severity>/seq_<sequence>/eval_dsec_ann_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def normalize_wandb_env() -> None:
    # Accept common typo used in some shells.
    if not os.getenv("WANDB_API_KEY") and os.getenv("WANBD_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["WANBD_API_KEY"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill ANN eval JSONs to wandb")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results/noise_benchmark/ann_subset_balanced_queue_20260316_153448__noise_eval_seqagg"),
        help="Root directory containing ann_rgb_* result folders",
    )
    p.add_argument(
        "--glob",
        type=str,
        default="ann_rgb_*/seq_*/eval_dsec_ann_test.json",
        help="Relative glob pattern under --root",
    )
    p.add_argument("--project", type=str, default="dsec_noise_benchmark")
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--run-name-prefix", type=str, default="backfill_ann_eval")
    p.add_argument(
        "--done-file",
        type=Path,
        default=Path("logs/wandb_backfill_ann_eval.done"),
        help="Track already-backfilled files to avoid duplicates",
    )
    p.add_argument("--force", action="store_true", help="Ignore done-file and backfill all discovered files")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_condition(json_path: Path) -> tuple[str, str, str]:
    """
    Returns (sequence, noise, severity)
    """
    cond_dir = json_path.parents[1].name  # ann_rgb_<noise>_s<sev>
    seq_dir = json_path.parent.name  # seq_<sequence>

    m = re.match(r"^ann_rgb_(.+)_s([0-9]+)$", cond_dir)
    if not m:
        raise ValueError(f"Unexpected condition folder format: {cond_dir}")
    noise, severity = m.group(1), m.group(2)

    if not seq_dir.startswith("seq_"):
        raise ValueError(f"Unexpected sequence folder format: {seq_dir}")
    sequence = seq_dir[len("seq_") :]
    return sequence, noise, severity


def load_done(done_file: Path) -> set[str]:
    if not done_file.exists():
        return set()
    return {line.strip() for line in done_file.read_text().splitlines() if line.strip()}


def append_done(done_file: Path, item: str) -> None:
    done_file.parent.mkdir(parents=True, exist_ok=True)
    with done_file.open("a", encoding="utf-8") as f:
        f.write(item + "\n")


def main() -> int:
    args = parse_args()
    normalize_wandb_env()

    json_paths = sorted((args.root / "").glob(args.glob))
    if not json_paths:
        print(f"No files found under {args.root} with pattern {args.glob}")
        return 1

    done = set() if args.force else load_done(args.done_file)
    pending = [p for p in json_paths if str(p) not in done]

    print(f"discovered={len(json_paths)} pending={len(pending)} done_file={args.done_file}")
    if args.dry_run:
        for p in pending[:20]:
            print(p)
        if len(pending) > 20:
            print(f"... ({len(pending) - 20} more)")
        return 0

    try:
        import wandb
    except Exception as e:
        print(f"wandb import failed: {e}")
        return 2

    ok = 0
    fail = 0
    for idx, p in enumerate(pending, start=1):
        try:
            with p.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            sequence, noise, severity = parse_condition(p)
            split = p.stem.replace("eval_dsec_ann_", "")
            run_name = f"{args.run_name_prefix}__{sequence}__{noise}_s{severity}"

            config = payload.get("config", {})
            config.update(
                {
                    "backfill_source_json": str(p),
                    "backfill_sequence": sequence,
                    "backfill_noise": noise,
                    "backfill_severity": int(severity),
                    "backfill_split": split,
                }
            )

            run = wandb.init(
                project=args.project,
                entity=args.entity,
                name=run_name,
                config=config,
                tags=["backfill", "evaluation", "dsec", "ann", sequence, noise, f"s{severity}", split],
                reinit=True,
            )

            log_dict = {
                "eval/mAP_avg": payload.get("mAP_avg", 0.0),
                "eval/mAP@0.5": payload.get("mAP@0.5", 0.0),
                "eval/mAP@0.75": payload.get("mAP@0.75", 0.0),
                "eval/num_images": payload.get("num_images", 0),
                "eval/num_predictions": payload.get("num_predictions", 0),
                "eval/num_ground_truths": payload.get("num_ground_truths", 0),
                "meta/severity": int(severity),
            }

            latency = payload.get("latency", {})
            for key in ("mean_ms", "std_ms", "fps"):
                if key in latency:
                    log_dict[f"eval/latency_{key}"] = latency[key]

            per_class_50 = payload.get("per_class_AP@0.5", {})
            for cls_name, ap in per_class_50.items():
                log_dict[f"eval/per_class_ap50/{cls_name}"] = ap

            wandb.log(log_dict)

            artifact = wandb.Artifact(name=f"eval_dsec_ann_{split}_{sequence}_{noise}_s{severity}", type="evaluation")
            artifact.add_file(str(p))
            wandb.log_artifact(artifact)
            wandb.finish()

            append_done(args.done_file, str(p))
            ok += 1
            print(f"[{idx}/{len(pending)}] ok {p}")
        except Exception as e:
            fail += 1
            print(f"[{idx}/{len(pending)}] FAIL {p}: {e}")
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

    print(f"completed ok={ok} fail={fail} total={len(pending)}")
    return 0 if fail == 0 else 3


if __name__ == "__main__":
    sys.exit(main())


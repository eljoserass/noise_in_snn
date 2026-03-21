#!/usr/bin/env python3
"""
Build ANN previous-vs-current comparison plots directly from W&B runs.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import wandb


@dataclass
class EvalPoint:
    noise_type: str
    severity: int
    map50: float
    run_id: str
    run_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two ANN eval sweeps from W&B.")
    p.add_argument("--entity", type=str, default="joserass")
    p.add_argument("--eval-project", type=str, default="dsec_noise_benchmark")
    p.add_argument("--train-project", type=str, default="dsec_subset_ann")

    p.add_argument("--prev-key", type=str, default="ann_noise_20260315_163523")
    p.add_argument("--curr-key", type=str, default="ann_subset_balanced_queue_20260316_153448")

    p.add_argument("--prev-eval-run-id", type=str, default="")
    p.add_argument("--curr-eval-run-id", type=str, default="")
    p.add_argument("--prev-train-run-id", type=str, default="")
    p.add_argument("--curr-train-run-id", type=str, default="")

    p.add_argument("--model-type", type=str, default="ann")
    p.add_argument("--variant", type=str, default="rgb")
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis") / f"ann_prev_vs_current_wandb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    p.add_argument("--max-runs-scan", type=int, default=3000)
    return p.parse_args()


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _to_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _contains_key(run: wandb.apis.public.Run, key: str) -> bool:
    if not key:
        return False
    key = key.strip()
    if not key:
        return False
    hay = [
        str(getattr(run, "name", "") or ""),
        str(getattr(run, "id", "") or ""),
        str(getattr(run, "group", "") or ""),
        str(getattr(run, "display_name", "") or ""),
        str((run.config or {}).get("wandb_run_name", "") or ""),
    ]
    return any(key in s for s in hay)


def _iter_runs_limited(runs: Iterable[wandb.apis.public.Run], limit: int) -> list[wandb.apis.public.Run]:
    out: list[wandb.apis.public.Run] = []
    for run in runs:
        out.append(run)
        if len(out) >= limit:
            break
    return out


def _resolve_run(
    api: wandb.Api,
    entity: str,
    project: str,
    run_id: str,
    key: str,
    required_model: str,
    required_variant: str | None = None,
) -> wandb.apis.public.Run:
    if run_id.strip():
        return api.run(f"{entity}/{project}/{run_id.strip()}")

    runs = _iter_runs_limited(
        api.runs(
            f"{entity}/{project}",
            filters={"config.model_type": required_model},
            order="-created_at",
        ),
        limit=4000,
    )
    matches: list[wandb.apis.public.Run] = []
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config or {}
        if required_variant is not None and str(cfg.get("eval_variant", "") or "") != required_variant:
            continue
        if key and not _contains_key(run, key):
            continue
        matches.append(run)

    if not matches:
        # fallback: do not require variant in case older runs used different config naming
        for run in runs:
            if run.state != "finished":
                continue
            if key and not _contains_key(run, key):
                continue
            matches.append(run)

    if not matches:
        raise RuntimeError(f"No run matched key='{key}' in {entity}/{project}")
    return matches[0]


def _eval_point_from_run(run: wandb.apis.public.Run) -> EvalPoint | None:
    cfg = run.config or {}
    summ = run.summary or {}

    metric = _to_float(summ.get("eval/mAP@0.5"))
    if not math.isfinite(metric):
        return None

    noise_type = str(summ.get("meta/noise_type", cfg.get("noise_type", "clean")) or "clean")
    severity = _to_int(summ.get("meta/noise_severity", cfg.get("noise_severity", 0)), default=0)
    if noise_type == "clean" and severity < 0:
        severity = 0

    return EvalPoint(
        noise_type=noise_type,
        severity=severity,
        map50=metric,
        run_id=str(run.id),
        run_name=str(run.name),
    )


def _collect_eval_points(
    api: wandb.Api,
    entity: str,
    project: str,
    key: str,
    run_id: str,
    model_type: str,
    variant: str,
    split: str,
    max_scan: int,
) -> tuple[wandb.apis.public.Run, dict[tuple[str, int], EvalPoint]]:
    anchor = _resolve_run(
        api=api,
        entity=entity,
        project=project,
        run_id=run_id,
        key=key,
        required_model=model_type,
        required_variant=variant,
    )

    anchor_key = run_id.strip() or key
    runs = _iter_runs_limited(
        api.runs(
            f"{entity}/{project}",
            filters={"config.model_type": model_type},
            order="-created_at",
        ),
        limit=max_scan,
    )

    points: dict[tuple[str, int], EvalPoint] = {}
    for run in runs:
        if run.state != "finished":
            continue
        cfg = run.config or {}
        if str(cfg.get("model_type", "")) != model_type:
            continue
        if str(cfg.get("eval_variant", "") or "") != variant:
            continue

        split_cfg = str(cfg.get("test_splits", "") or "")
        if split and split not in split_cfg:
            continue

        if anchor_key and not _contains_key(run, anchor_key):
            continue

        pt = _eval_point_from_run(run)
        if pt is None:
            continue
        points[(pt.noise_type, pt.severity)] = pt

    if not points:
        raise RuntimeError(f"No eval points found for key='{anchor_key}' in {entity}/{project}")
    return anchor, points


def _write_eval_outputs(prev_pts: dict[tuple[str, int], EvalPoint], curr_pts: dict[tuple[str, int], EvalPoint], out_dir: Path) -> pd.DataFrame:
    keys = sorted(set(prev_pts.keys()) | set(curr_pts.keys()), key=lambda x: (x[0] != "clean", x[0], x[1]))
    rows: list[dict[str, Any]] = []
    for noise_type, sev in keys:
        prev = prev_pts.get((noise_type, sev))
        curr = curr_pts.get((noise_type, sev))
        prev_val = prev.map50 if prev else float("nan")
        curr_val = curr.map50 if curr else float("nan")
        rows.append(
            {
                "noise_type": noise_type,
                "severity": sev,
                "condition": f"{noise_type}_s{sev}",
                "prev_map50": prev_val,
                "curr_map50": curr_val,
                "delta": curr_val - prev_val if math.isfinite(prev_val) and math.isfinite(curr_val) else float("nan"),
                "prev_run_id": prev.run_id if prev else "",
                "curr_run_id": curr.run_id if curr else "",
            }
        )

    df = pd.DataFrame(rows)
    out_csv = out_dir / "noise_metrics_comparison.csv"
    df.to_csv(out_csv, index=False)

    # plot 1: previous vs current map50
    fig, ax = plt.subplots(figsize=(14, 6))
    x = list(range(len(df)))
    width = 0.42
    ax.bar([i - width / 2 for i in x], df["prev_map50"], width=width, label="previous")
    ax.bar([i + width / 2 for i in x], df["curr_map50"], width=width, label="current")
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=70, ha="right")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("ANN mAP@0.5: previous vs current subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "map50_prev_vs_current.png", dpi=160)
    plt.close(fig)

    # plot 2: delta
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2e7d32" if (isinstance(v, float) and v >= 0) else "#c62828" for v in df["delta"].tolist()]
    ax.bar(df["condition"], df["delta"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticklabels(df["condition"], rotation=70, ha="right")
    ax.set_ylabel("delta mAP@0.5 (current - previous)")
    ax.set_title("ANN mAP@0.5 delta by condition")
    fig.tight_layout()
    fig.savefig(out_dir / "map50_delta.png", dpi=160)
    plt.close(fig)
    return df


def _collect_train_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in run.scan_history(keys=["epoch", "train/loss", "val/loss"]):
        rows.append(
            {
                "step": _to_int(rec.get("_step"), default=-1),
                "epoch": _to_int(rec.get("epoch"), default=-1),
                "train_loss": _to_float(rec.get("train/loss")),
                "val_loss": _to_float(rec.get("val/loss")),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["epoch", "step"]).drop_duplicates(subset=["epoch"], keep="last")
    return df


def _write_loss_outputs(prev_hist: pd.DataFrame, curr_hist: pd.DataFrame, out_dir: Path) -> None:
    prev_hist = prev_hist.copy()
    curr_hist = curr_hist.copy()
    prev_hist["run"] = "previous"
    curr_hist["run"] = "current"
    loss_df = pd.concat([prev_hist, curr_hist], ignore_index=True)
    loss_df.to_csv(out_dir / "loss_curves.csv", index=False)

    if loss_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
    for run_name, run_df in loss_df.groupby("run"):
        axes[0].plot(run_df["epoch"], run_df["train_loss"], label=run_name)
        axes[1].plot(run_df["epoch"], run_df["val_loss"], label=run_name)
    axes[0].set_title("Train loss")
    axes[1].set_title("Val loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves_prev_vs_current.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=45)

    prev_anchor, prev_eval = _collect_eval_points(
        api=api,
        entity=args.entity,
        project=args.eval_project,
        key=args.prev_key,
        run_id=args.prev_eval_run_id,
        model_type=args.model_type,
        variant=args.variant,
        split=args.split,
        max_scan=args.max_runs_scan,
    )
    curr_anchor, curr_eval = _collect_eval_points(
        api=api,
        entity=args.entity,
        project=args.eval_project,
        key=args.curr_key,
        run_id=args.curr_eval_run_id,
        model_type=args.model_type,
        variant=args.variant,
        split=args.split,
        max_scan=args.max_runs_scan,
    )

    df_eval = _write_eval_outputs(prev_eval, curr_eval, args.out_dir)

    prev_train = _resolve_run(
        api=api,
        entity=args.entity,
        project=args.train_project,
        run_id=args.prev_train_run_id,
        key=args.prev_key,
        required_model=args.model_type,
        required_variant=None,
    )
    curr_train = _resolve_run(
        api=api,
        entity=args.entity,
        project=args.train_project,
        run_id=args.curr_train_run_id,
        key=args.curr_key,
        required_model=args.model_type,
        required_variant=None,
    )
    prev_hist = _collect_train_history(prev_train)
    curr_hist = _collect_train_history(curr_train)
    _write_loss_outputs(prev_hist, curr_hist, args.out_dir)

    clean_prev = df_eval.loc[df_eval["condition"] == "clean_s0", "prev_map50"]
    clean_curr = df_eval.loc[df_eval["condition"] == "clean_s0", "curr_map50"]
    clean_prev_val = float(clean_prev.iloc[0]) if not clean_prev.empty else float("nan")
    clean_curr_val = float(clean_curr.iloc[0]) if not clean_curr.empty else float("nan")
    nonclean = df_eval[df_eval["noise_type"] != "clean"]
    mean_delta_nonclean = float(nonclean["delta"].mean()) if not nonclean.empty else float("nan")

    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"prev_anchor={args.entity}/{args.eval_project}/{prev_anchor.id} name={prev_anchor.name}",
                f"curr_anchor={args.entity}/{args.eval_project}/{curr_anchor.id} name={curr_anchor.name}",
                f"prev_train={args.entity}/{args.train_project}/{prev_train.id} name={prev_train.name}",
                f"curr_train={args.entity}/{args.train_project}/{curr_train.id} name={curr_train.name}",
                f"conditions={len(df_eval)}",
                f"clean_prev_map50={clean_prev_val:.6f}",
                f"clean_curr_map50={clean_curr_val:.6f}",
                f"clean_delta={(clean_curr_val - clean_prev_val):.6f}",
                f"mean_delta_nonclean={mean_delta_nonclean:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved: {args.out_dir}")
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

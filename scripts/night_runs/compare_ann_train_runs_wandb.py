#!/usr/bin/env python3
"""
Compare ANN training runs from W&B and export local plots/csv/summary.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare ANN W&B train runs.")
    p.add_argument("--entity", type=str, default="joserass")
    p.add_argument("--run-a", type=str, required=True, help="Format: project/run_id")
    p.add_argument("--run-b", type=str, required=True, help="Format: project/run_id")
    p.add_argument("--run-c", type=str, default="", help="Optional third run: project/run_id")
    p.add_argument("--label-a", type=str, default="run_a")
    p.add_argument("--label-b", type=str, default="run_b")
    p.add_argument("--label-c", type=str, default="run_c")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/analysis") / f"ann_train_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
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


def _fetch_history(run: wandb.apis.public.Run) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    wanted = [
        "_step",
        "epoch",
        "train/loss",
        "val/loss",
        "train/cls_loss",
        "val/cls_loss",
        "train/loc_loss",
        "val/loc_loss",
        "learning_rate",
    ]
    for rec in run.scan_history(keys=wanted):
        rows.append(
            {
                "step": _to_int(rec.get("_step"), default=-1),
                "epoch": _to_int(rec.get("epoch"), default=-1),
                "train_loss": _to_float(rec.get("train/loss")),
                "val_loss": _to_float(rec.get("val/loss")),
                "train_cls_loss": _to_float(rec.get("train/cls_loss")),
                "val_cls_loss": _to_float(rec.get("val/cls_loss")),
                "train_loc_loss": _to_float(rec.get("train/loc_loss")),
                "val_loc_loss": _to_float(rec.get("val/loc_loss")),
                "lr": _to_float(rec.get("learning_rate")),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Keep last record per epoch for clean curves.
    df = df.sort_values(["epoch", "step"]).drop_duplicates(subset=["epoch"], keep="last")
    return df


def _run_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "epochs_logged": 0,
            "best_val_loss": float("nan"),
            "best_val_epoch": -1,
            "final_train_loss": float("nan"),
            "final_val_loss": float("nan"),
        }
    val_idx = df["val_loss"].idxmin()
    return {
        "epochs_logged": int(df["epoch"].max()) + 1,
        "best_val_loss": float(df.loc[val_idx, "val_loss"]),
        "best_val_epoch": int(df.loc[val_idx, "epoch"]),
        "final_train_loss": float(df.iloc[-1]["train_loss"]),
        "final_val_loss": float(df.iloc[-1]["val_loss"]),
    }


def _save_plots(df_all: pd.DataFrame, out_dir: Path) -> None:
    # Train/val loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, g in df_all.groupby("label"):
        axes[0].plot(g["epoch"], g["train_loss"], label=label)
        axes[1].plot(g["epoch"], g["val_loss"], label=label)
    axes[0].set_title("Train loss")
    axes[1].set_title("Val loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_val_loss_compare.png", dpi=160)
    plt.close(fig)

    # cls/loc decomposition on val
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, g in df_all.groupby("label"):
        axes[0].plot(g["epoch"], g["val_cls_loss"], label=label)
        axes[1].plot(g["epoch"], g["val_loc_loss"], label=label)
    axes[0].set_title("Val cls loss")
    axes[1].set_title("Val loc loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "val_cls_loc_loss_compare.png", dpi=160)
    plt.close(fig)


def _resolve_run(api: wandb.Api, entity: str, project_and_id: str) -> wandb.apis.public.Run:
    if "/" not in project_and_id:
        raise ValueError(f"Invalid run reference '{project_and_id}'. Expected project/run_id.")
    project, run_id = project_and_id.split("/", 1)
    return api.run(f"{entity}/{project}/{run_id}")


def main() -> None:
    args = parse_args()
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError("WANDB_API_KEY is not set.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)

    run_a = _resolve_run(api, args.entity, args.run_a)
    run_b = _resolve_run(api, args.entity, args.run_b)
    run_c = _resolve_run(api, args.entity, args.run_c) if args.run_c.strip() else None

    df_a = _fetch_history(run_a)
    df_b = _fetch_history(run_b)
    df_a["label"] = args.label_a
    df_b["label"] = args.label_b
    frames = [df_a, df_b]
    if run_c is not None:
        df_c = _fetch_history(run_c)
        df_c["label"] = args.label_c
        frames.append(df_c)
    else:
        df_c = pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    df_all.to_csv(args.out_dir / "loss_curves.csv", index=False)

    _save_plots(df_all, args.out_dir)

    stats_a = _run_stats(df_a)
    stats_b = _run_stats(df_b)
    stats_c = _run_stats(df_c) if run_c is not None else None

    lines = [
        f"a={args.entity}/{args.run_a} label={args.label_a} name={run_a.name}",
        f"b={args.entity}/{args.run_b} label={args.label_b} name={run_b.name}",
        f"a_epochs={stats_a['epochs_logged']}",
        f"b_epochs={stats_b['epochs_logged']}",
        f"a_best_val_loss={stats_a['best_val_loss']:.6f} @epoch={stats_a['best_val_epoch']}",
        f"b_best_val_loss={stats_b['best_val_loss']:.6f} @epoch={stats_b['best_val_epoch']}",
        f"a_final_train_loss={stats_a['final_train_loss']:.6f}",
        f"b_final_train_loss={stats_b['final_train_loss']:.6f}",
        f"a_final_val_loss={stats_a['final_val_loss']:.6f}",
        f"b_final_val_loss={stats_b['final_val_loss']:.6f}",
    ]
    if run_c is not None and stats_c is not None:
        lines.extend(
            [
                f"c={args.entity}/{args.run_c} label={args.label_c} name={run_c.name}",
                f"c_epochs={stats_c['epochs_logged']}",
                f"c_best_val_loss={stats_c['best_val_loss']:.6f} @epoch={stats_c['best_val_epoch']}",
                f"c_final_train_loss={stats_c['final_train_loss']:.6f}",
                f"c_final_val_loss={stats_c['final_val_loss']:.6f}",
            ]
        )
    summary = "\n".join(lines)
    (args.out_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"saved: {args.out_dir}")
    print(summary)


if __name__ == "__main__":
    main()

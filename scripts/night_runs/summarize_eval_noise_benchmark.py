#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize DSEC noise benchmark eval JSON files.")
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="dsec_noise_benchmark")
    p.add_argument("--wandb-run-name", type=str, default="dsec_noise_benchmark_summary")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="noise-benchmark,summary")
    return p.parse_args()


def as_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def as_int(v: Any, default: int = -1) -> int:
    try:
        return int(v)
    except Exception:
        return default


def gather_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_root.rglob("eval_dsec_*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue

        cfg = payload.get("config", {})
        model_type = cfg.get("model_type", "unknown")
        noise_type = cfg.get("noise_type", "") or "clean"
        severity = as_int(cfg.get("noise_severity", -1), -1)
        variant = cfg.get("eval_variant", "") or "default"
        image_relpath = cfg.get("image_relpath", "")
        event_relpath = cfg.get("event_relpath", "")
        event_source = cfg.get("event_source", "")
        split = path.stem.replace(f"eval_dsec_{model_type}_", "")

        row = {
            "json_path": str(path),
            "split": split,
            "model_type": model_type,
            "variant": variant,
            "noise_type": noise_type,
            "severity": severity,
            "mAP@0.5": as_float(payload.get("mAP@0.5", float("nan"))),
            "mAP@0.75": as_float(payload.get("mAP@0.75", float("nan"))),
            "mAP_avg": as_float(payload.get("mAP_avg", float("nan"))),
            "num_images": as_int(payload.get("num_images", -1), -1),
            "image_relpath": image_relpath,
            "event_source": event_source,
            "event_relpath": event_relpath,
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "split",
        "model_type",
        "variant",
        "noise_type",
        "severity",
        "mAP@0.5",
        "mAP@0.75",
        "mAP_avg",
        "num_images",
        "image_relpath",
        "event_source",
        "event_relpath",
        "json_path",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def log_wandb(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed; skipping W&B summary logging.")
        return

    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=tags,
        config={"results_root": str(args.results_root)},
        reinit=True,
    )

    columns = [
        "split",
        "model_type",
        "variant",
        "noise_type",
        "severity",
        "mAP@0.5",
        "mAP@0.75",
        "mAP_avg",
        "num_images",
    ]
    table = wandb.Table(columns=columns)
    for r in rows:
        table.add_data(
            r["split"],
            r["model_type"],
            r["variant"],
            r["noise_type"],
            r["severity"],
            r["mAP@0.5"],
            r["mAP@0.75"],
            r["mAP_avg"],
            r["num_images"],
        )
    wandb.log({"benchmark/table": table})

    # One line chart per noise type (mAP@0.5 vs severity), separate lines by model+variant.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r["split"], r["noise_type"])
        grouped.setdefault(key, []).append(r)

    for (split, noise_type), items in grouped.items():
        sev_values = sorted({int(x["severity"]) for x in items if int(x["severity"]) >= 0})
        if not sev_values:
            continue
        by_series: dict[str, dict[int, float]] = {}
        for r in items:
            series_name = f'{r["model_type"]}:{r["variant"]}'
            by_series.setdefault(series_name, {})[int(r["severity"])] = float(r["mAP@0.5"])

        keys = sorted(by_series.keys())
        ys = []
        for k in keys:
            series = []
            for sev in sev_values:
                v = by_series[k].get(sev, float("nan"))
                series.append(v if math.isfinite(v) else float("nan"))
            ys.append(series)

        chart_key = f"benchmark/map50_vs_severity/{split}/{noise_type}"
        wandb.log(
            {
                chart_key: wandb.plot.line_series(
                    xs=sev_values,
                    ys=ys,
                    keys=keys,
                    title=f"{split} | {noise_type} | mAP@0.5",
                    xname="severity",
                )
            }
        )

    run.finish()


def main() -> None:
    args = parse_args()
    rows = gather_rows(args.results_root)
    if not rows:
        print(f"No eval JSON files found under {args.results_root}")
        return

    rows.sort(key=lambda r: (r["split"], r["noise_type"], r["severity"], r["model_type"], r["variant"]))
    out_csv = args.output_csv or (args.results_root / "summary.csv")
    write_csv(rows, out_csv)
    print(f"rows={len(rows)} csv={out_csv}")
    print("preview:")
    for r in rows[:12]:
        print(
            f'  split={r["split"]} model={r["model_type"]}:{r["variant"]} '
            f'noise={r["noise_type"]} s={r["severity"]} mAP@0.5={r["mAP@0.5"]:.4f}'
        )

    if args.wandb:
        log_wandb(rows, args)


if __name__ == "__main__":
    main()


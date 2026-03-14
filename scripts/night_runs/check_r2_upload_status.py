#!/usr/bin/env python3
"""
Read-only R2 upload status checker for DSEC processed outputs.

Default behavior checks:
1) Focused train/val subset clean-event artifacts (dvs_events.txt + v2e-args.txt).
2) Per-test-sequence event upload breakdown:
   - clean v2e
   - noisy v2e
   - corruption + clean mode
   - corruption + noisy mode
   - manual/native v2e noise sweeps (no _clean/_noisy suffix)
3) Presence of RGB corruption folders in test (`images/left/distorted_gray_*_s*`).
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_TRAIN_SIM_SEQUENCES = [
    "thun_00_a",
    "interlaken_00_c",
    "interlaken_00_e",
    "interlaken_00_g",
    "zurich_city_00_b",
    "zurich_city_02_c",
    "zurich_city_05_b",
    "zurich_city_10_b",
]

DEFAULT_VAL_SIM_SEQUENCES = [
    "zurich_city_16_a",
    "zurich_city_17_a",
    "zurich_city_18_a",
    "zurich_city_19_a",
    "zurich_city_20_a",
    "zurich_city_21_a",
]

ENV_KEYS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET")


@dataclass
class TestSeqStats:
    seq: str
    clean: int = 0
    noisy: int = 0
    corr_clean: int = 0
    corr_noisy: int = 0
    native_manual: int = 0
    total_event_files: int = 0
    rgb_corruption_dirs: int = 0


def parse_csv(values: str) -> list[str]:
    return [x.strip() for x in values.split(",") if x.strip()]


def load_env_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Env file not found: {path}")
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_env_file(cli_env_file: Path | None) -> Path:
    if cli_env_file is not None:
        return cli_env_file
    candidates = [
        Path("../dsec_data_managing/.env"),
        Path("/workspace/dsec_data_managing/.env"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find .env file. Pass --env-file explicitly "
        "(e.g. --env-file ../dsec_data_managing/.env)."
    )


def require_env() -> dict[str, str]:
    missing = [k for k in ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")
    return {k: os.environ[k] for k in ENV_KEYS}


def build_client(env: dict[str, str]):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{env['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=env["R2_ACCESS_KEY"],
        aws_secret_access_key=env["R2_SECRET_KEY"],
        region_name="auto",
    )


def normalize_prefix(prefix: str) -> str:
    return prefix.strip("/").rstrip("/")


def iter_objects(s3, bucket: str, prefix: str) -> Iterable[str]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            yield key


def list_child_prefix_names(s3, bucket: str, prefix: str) -> list[str]:
    names: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for cp in page.get("CommonPrefixes", []):
            child = cp["Prefix"]
            name = child[len(prefix) :].strip("/")
            if name:
                names.append(name)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return sorted(set(names))


def check_targeted_train_val(s3, bucket: str, remote_prefix: str, targets: list[str]) -> tuple[int, int, list[str]]:
    expected = len(targets) * 2  # dvs_events.txt + v2e-args.txt
    present = 0
    missing: list[str] = []
    for seq in targets:
        for leaf in (
            "v2e_output_distorted_gray_clean/dvs_events.txt",
            "v2e_output_distorted_gray_clean/v2e-args.txt",
        ):
            key = f"{remote_prefix}/train/{seq}/{leaf}"
            try:
                s3.head_object(Bucket=bucket, Key=key)
                present += 1
            except ClientError:
                missing.append(key)
    return present, expected, missing


def classify_test_events(keys: Iterable[str]) -> TestSeqStats:
    stats = TestSeqStats(seq="")
    for key in keys:
        if "/v2e_output" not in key or not key.endswith("/dvs_events.txt"):
            continue
        stats.total_event_files += 1
        folder = key.rsplit("/", 1)[0].rsplit("/", 1)[-1]
        if folder == "v2e_output_distorted_gray_clean":
            stats.clean += 1
        elif folder == "v2e_output_distorted_gray_noisy":
            stats.noisy += 1
        elif folder.endswith("_clean"):
            stats.corr_clean += 1
        elif folder.endswith("_noisy"):
            stats.corr_noisy += 1
        else:
            stats.native_manual += 1
    return stats


def check_test_split(s3, bucket: str, remote_prefix: str) -> list[TestSeqStats]:
    test_root = f"{remote_prefix}/test/"
    seqs = list_child_prefix_names(s3, bucket, test_root)
    all_stats: list[TestSeqStats] = []
    for seq in seqs:
        seq_prefix = f"{test_root}{seq}/"
        events = list(iter_objects(s3, bucket, seq_prefix))
        stats = classify_test_events(events)
        stats.seq = seq

        # Check RGB corruption folders quickly via one-level listing.
        left_prefix = f"{seq_prefix}images/left/"
        child_dirs = list_child_prefix_names(s3, bucket, left_prefix)
        stats.rgb_corruption_dirs = sum(
            1 for name in child_dirs if fnmatch.fnmatch(name, "distorted_gray_*_s*")
        )
        all_stats.append(stats)
    return sorted(all_stats, key=lambda s: s.seq)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check R2 upload status for DSEC processed outputs.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env with R2 credentials. Default: auto-detect ../dsec_data_managing/.env",
    )
    parser.add_argument(
        "--remote-prefix",
        type=str,
        default="dsec_processed",
        help="Remote prefix in bucket (default: dsec_processed).",
    )
    parser.add_argument(
        "--train-sequences",
        type=str,
        default=",".join(DEFAULT_TRAIN_SIM_SEQUENCES),
        help="CSV list of focused train sequences.",
    )
    parser.add_argument(
        "--val-sequences",
        type=str,
        default=",".join(DEFAULT_VAL_SIM_SEQUENCES),
        help="CSV list of focused val sequences.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_file = resolve_env_file(args.env_file)
    load_env_file(env_file)
    env = require_env()
    s3 = build_client(env)
    bucket = env["R2_BUCKET"]
    remote_prefix = normalize_prefix(args.remote_prefix)
    targets = parse_csv(args.train_sequences) + parse_csv(args.val_sequences)

    try:
        s3.head_bucket(Bucket=bucket)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Could not access R2 bucket '{bucket}': {exc}") from exc

    print("== R2 upload status ==")
    print(f"bucket={bucket}")
    print(f"prefix={remote_prefix}")
    print(f"env_file={env_file}")
    print()

    present, expected, missing = check_targeted_train_val(s3, bucket, remote_prefix, targets)
    pct = (100.0 * present / expected) if expected else 0.0
    print(f"focused train+val files: {present}/{expected} ({pct:.1f}%)")
    if present == expected:
        print("focused subset status: COMPLETE")
    else:
        print("focused subset status: INCOMPLETE")
        for key in missing:
            print(f"  missing: {key}")
    print()

    per_seq = check_test_split(s3, bucket, remote_prefix)
    if not per_seq:
        print("No test sequences found under remote prefix.")
        return

    print("== test sequence breakdown ==")
    print("sequence                 rgb_corr_dirs clean noisy corr_clean corr_noisy native total_events")
    total_events = 0
    for s in per_seq:
        total_events += s.total_event_files
        print(
            f"{s.seq:<24} {s.rgb_corruption_dirs:>12} {s.clean:>5} {s.noisy:>5} "
            f"{s.corr_clean:>10} {s.corr_noisy:>10} {s.native_manual:>6} {s.total_event_files:>12}"
        )
    print()
    print(f"test sequences: {len(per_seq)}")
    print(f"total uploaded test event files: {total_events}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Sync data between Cloudflare R2 (S3-compatible) and local filesystem.

Examples
--------
Download a full split:
    python scripts/r2_sync.py download \
      --remote-prefix dsec/train \
      --local-dir data/dsec/train

Upload processed outputs:
    python scripts/r2_sync.py upload \
      --local-dir data/dsec_processed/test \
      --remote-prefix dsec_processed/test
"""

from __future__ import annotations

import argparse
import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import boto3
from botocore.exceptions import ClientError


ENV_KEYS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_BUCKET")


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


def require_env() -> dict[str, str]:
    missing = [k for k in ENV_KEYS if k not in os.environ or not os.environ[k]]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"Missing required env vars: {joined}. "
            "Set them in environment or pass --env-file."
        )
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


def iter_remote_objects(s3, bucket: str, prefix: str):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            yield obj


def list_local_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def match_any(path_str: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path_str, pat) for pat in patterns)


def remote_exists_with_size(s3, bucket: str, key: str, size: int) -> bool:
    try:
        meta = s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return False
    return int(meta.get("ContentLength", -1)) == size


def download_one(
    s3,
    bucket: str,
    key: str,
    local_path: Path,
    expected_size: int,
    skip_existing: bool,
) -> str:
    if skip_existing and local_path.exists() and local_path.stat().st_size == expected_size:
        return f"skip {key}"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))
    return f"done {key}"


def upload_one(
    s3,
    bucket: str,
    local_path: Path,
    key: str,
    skip_existing: bool,
) -> str:
    size = local_path.stat().st_size
    if skip_existing and remote_exists_with_size(s3, bucket, key, size):
        return f"skip {key}"
    s3.upload_file(str(local_path), bucket, key)
    return f"done {key}"


def run_download(
    s3,
    bucket: str,
    remote_prefix: str,
    local_dir: Path,
    include_globs: list[str],
    workers: int,
    skip_existing: bool,
) -> None:
    remote_prefix = normalize_prefix(remote_prefix)
    objs = list(iter_remote_objects(s3, bucket, remote_prefix))
    manifest: list[tuple[str, Path, int]] = []
    base = remote_prefix + "/"

    for obj in objs:
        key = obj["Key"]
        if not key.startswith(base):
            continue
        rel = key[len(base) :]
        if not rel:
            continue
        if not match_any(rel, include_globs):
            continue
        manifest.append((key, local_dir / rel, int(obj["Size"])))

    print(f"download manifest: {len(manifest)} files")
    done = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                download_one,
                s3,
                bucket,
                key,
                local_path,
                size,
                skip_existing,
            )
            for key, local_path, size in manifest
        ]
        for fut in as_completed(futures):
            msg = fut.result()
            if msg.startswith("skip"):
                skipped += 1
            else:
                done += 1
            print(msg)
    print(f"download complete: done={done}, skipped={skipped}")


def run_upload(
    s3,
    bucket: str,
    local_dir: Path,
    remote_prefix: str,
    include_globs: list[str],
    workers: int,
    skip_existing: bool,
) -> None:
    remote_prefix = normalize_prefix(remote_prefix)
    files = list_local_files(local_dir)
    manifest: list[tuple[Path, str]] = []
    for path in files:
        rel = str(path.relative_to(local_dir)).replace(os.sep, "/")
        if not match_any(rel, include_globs):
            continue
        key = f"{remote_prefix}/{rel}" if remote_prefix else rel
        manifest.append((path, key))

    print(f"upload manifest: {len(manifest)} files")

    # Pre-fetch remote listing so we can skip without per-file HEAD requests.
    remote_sizes: dict[str, int] = {}
    if skip_existing:
        print("listing remote objects for skip check...")
        for obj in iter_remote_objects(s3, bucket, remote_prefix):
            remote_sizes[obj["Key"]] = int(obj["Size"])
        print(f"remote listing: {len(remote_sizes)} objects")

    done = 0
    skipped = 0
    upload_manifest: list[tuple[Path, str]] = []
    for path, key in manifest:
        if skip_existing and key in remote_sizes and path.stat().st_size == remote_sizes[key]:
            skipped += 1
            print(f"skip {key}")
            continue
        upload_manifest.append((path, key))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(upload_one, s3, bucket, path, key, False)
            for path, key in upload_manifest
        ]
        for fut in as_completed(futures):
            msg = fut.result()
            done += 1
            print(msg)
    print(f"upload complete: done={done}, skipped={skipped}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync data between R2 and local filesystem.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file for R2 credentials.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers.")
    parser.add_argument(
        "--include-glob",
        action="append",
        default=[],
        help=(
            "Optional include glob on relative path. Can be repeated. "
            "Example: --include-glob 'images/left/distorted/*.png'"
        ),
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Do not skip existing files with matching size.",
    )

    sub = parser.add_subparsers(dest="mode", required=True)

    p_down = sub.add_parser("download", help="Download R2 prefix to local directory.")
    p_down.add_argument("--remote-prefix", required=True, help="Remote key prefix in bucket.")
    p_down.add_argument("--local-dir", type=Path, required=True, help="Local destination directory.")

    p_up = sub.add_parser("upload", help="Upload local directory to R2 prefix.")
    p_up.add_argument("--local-dir", type=Path, required=True, help="Local source directory.")
    p_up.add_argument("--remote-prefix", required=True, help="Remote key prefix in bucket.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)
    env = require_env()
    s3 = build_client(env)
    bucket = env["R2_BUCKET"]
    skip_existing = not args.no_skip_existing

    s3.head_bucket(Bucket=bucket)
    print(f"connected to bucket: {bucket}")

    if args.mode == "download":
        run_download(
            s3=s3,
            bucket=bucket,
            remote_prefix=args.remote_prefix,
            local_dir=args.local_dir,
            include_globs=args.include_glob,
            workers=args.workers,
            skip_existing=skip_existing,
        )
    elif args.mode == "upload":
        run_upload(
            s3=s3,
            bucket=bucket,
            local_dir=args.local_dir,
            remote_prefix=args.remote_prefix,
            include_globs=args.include_glob,
            workers=args.workers,
            skip_existing=skip_existing,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()


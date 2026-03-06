Night Run Scripts
=================

This folder contains long-running scripts for multi-machine overnight execution.

Active DSEC Pipeline
--------------------

Machine A: Download from R2

```bash
bash scripts/night_runs/machine_a_download.sh
```

Default split list is `train,test` (not `val`) because DSEC validation is often a sequence subset of `train`.

Machine B: Train DSEC baselines (resume-safe)

```bash
bash scripts/night_runs/machine_b_train.sh
```

If `data/dsec/val` is missing, this script automatically uses `train` as the split root and loads validation subsequences from `config/dsec_train_val_test_split.yaml` (if available).

Machine C: DSEC conversion/noise generation/upload loop

```bash
bash scripts/night_runs/machine_c_convert_loop.sh
```

Legacy TUMTraf Pipeline
-----------------------

Use only for old experiments:

```bash
bash scripts/night_runs/machine_b_train_tumtraf.sh
```

Notes
-----

- All scripts assume execution from `noise_in_snn/` repo root.
- Override defaults via env vars.

Example:

```bash
WORKERS=48 DSEC_ROOT=../data/dsec bash scripts/night_runs/machine_a_download.sh
```

Machine C upload behavior:

- `ENABLE_UPLOAD=1` enables upload to R2.
- `UPLOAD_INTERVAL_SECS` controls periodic background uploads during long processing runs (default `1800`).
- `UPLOAD_SCOPE=events_only` uploads event artifacts + rectified tracks.
- `UPLOAD_SCOPE=all_processed` uploads event artifacts + noisy/grayscale frames + validation images + rectified tracks.

Machine C v2e bootstrap/deps:

- `BOOTSTRAP_V2E=1` clones/updates v2e if missing.
- `INSTALL_V2E_DEPS=0` is default (recommended on Linux/Python 3.12+).
- Optional: `INSTALL_V2E_DEPS=1 V2E_DEPS_MODE=minimal`.

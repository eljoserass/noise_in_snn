Night Run Scripts
=================

These scripts are designed for 3-machine overnight execution.

Machine A: Download from R2
---------------------------
Run:

```bash
bash scripts/night_runs/machine_a_download.sh
```

Machine B: Train DSEC baselines (resume-safe)
---------------------------------------------
Run:

```bash
bash scripts/night_runs/machine_b_train.sh
```

Machine C: DSEC conversion/noise generation loop
------------------------------------------------
Run:

```bash
bash scripts/night_runs/machine_c_convert_loop.sh
```

Notes
-----
- All scripts assume execution from `noise_in_snn/` repo root.
- Override defaults with environment variables, e.g.:

```bash
WORKERS=48 DSEC_ROOT=data/dsec bash scripts/night_runs/machine_a_download.sh
```

- Machine C upload behavior:
  - Set `ENABLE_UPLOAD=1` to enable upload to R2.
  - `UPLOAD_SCOPE=events_only` uploads only event artifacts + rectified tracks.
  - `UPLOAD_SCOPE=all_processed` uploads event artifacts + noisy/grayscale frames + rectification validation images + rectified tracks.
- Machine C v2e bootstrap/deps:
  - `BOOTSTRAP_V2E=1` clones/updates v2e if missing.
  - `INSTALL_V2E_DEPS=0` is the default (recommended on Linux/Python 3.12+).
  - If needed: `INSTALL_V2E_DEPS=1 V2E_DEPS_MODE=minimal` for a lightweight install.

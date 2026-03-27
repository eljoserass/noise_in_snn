# ANN Eval / WandB Status (2026-03-24)

## Current Priority (zurich_city_14_c)
- Eval JSON status: `76 / 76` complete (no missing eval outputs for this sequence).
- WandB backfill status: in progress on eval machine.
- Running command:
  - `python scripts/night_runs/backfill_ann_eval_to_wandb.py --project dsec_noise_benchmark --run-name-prefix backfill_ann_eval_14c --glob "ann_rgb_*/seq_zurich_city_14_c/eval_dsec_ann_test.json"`
- Logs:
  - `logs/wandb_backfill_ann_eval_14c.out`
  - `logs/wandb_backfill_ann_eval.done` (resume-safe done tracker)

## Why Large Counts Appeared
- Across test data folders currently present, total possible ANN eval conditions are `988`.
- Already evaluated JSON outputs present: `361`.
- Missing eval JSON outputs (not run yet): `627`.
- This is why counts looked like “~700 missing”: `988 - 361 = 627`.

## Sequence Breakdown (expected / evaluated / missing)
- `interlaken_00_a`: `76 / 22 / 54`
- `interlaken_00_b`: `76 / 22 / 54`
- `interlaken_01_a`: `76 / 22 / 54`
- `thun_01_a`: `76 / 22 / 54`
- `thun_01_b`: `76 / 22 / 54`
- `thun_02_a`: `76 / 21 / 55`
- `zurich_city_12_a`: `76 / 22 / 54`
- `zurich_city_13_a`: `76 / 22 / 54`
- `zurich_city_13_b`: `76 / 22 / 54`
- `zurich_city_14_a`: `76 / 22 / 54`
- `zurich_city_14_b`: `76 / 22 / 54`
- `zurich_city_14_c`: `76 / 76 / 0`
- `zurich_city_15_a`: `76 / 22 / 54`

## Later (cheaper machine) TODO
- Finish WandB logging for any already-evaluated JSONs not yet uploaded:
  - `python scripts/night_runs/backfill_ann_eval_to_wandb.py --project dsec_noise_benchmark --run-name-prefix backfill_ann_eval`
- If needed, run remaining missing evals (the `627` missing JSON outputs) with the queue/eval scripts after deciding sequence/noise priority.

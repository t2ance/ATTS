# `analysis/archive/` — One-off Scripts & Event-specific Dumps

This directory holds files that:
- (a) were written for **one specific incident** or **one-shot sanity check**, and
- (b) we want to keep around because either the **data is not regenerable**
      or the script **encodes how to reproduce a past investigation**.

Nothing in here is on an active code path. Nothing in here should grow.
Append a new entry below when you drop a new file in.

## Contents (chronological)

### `judge_regression_records.jsonl` — 2026-04-13

60-row JSONL of judge regression cases captured during the judge-stochasticity
investigation. Each row carries `{cls, qid, gold, pred, expected, got,
extracted, reasoning, ...}` and is the raw evidence that the LLM judge was
returning inconsistent grades on identical inputs. Cannot be regenerated —
the underlying API calls were already paid for and not deterministic.

### `check_explores.py` — 2026-04-13 (approx)

One-shot count of how many rows in
`core_code/training/training_data/grpo/train.parquet` carry at least one
correct cached explore. Used once to validate the GRPO data pipeline
(`prepare_data_hle.py`) was producing a non-degenerate training set. No
imports from elsewhere; run with `python check_explores.py`.

### `sft_sample_row.md` — 2026-04-15 (approx)

Human-readable dump of the first row of
`training/training_data/sft_hle_q101_300_thinking.jsonl` (qid
`66fc45034293a9638d7e0f47`, 9 messages). Written so we could eyeball the
SFT format end-to-end without writing JSON parsing each time. Re-generate
with `head -1 sft_hle_q101_300_thinking.jsonl | python -m json.tool` if
the format changes.

### `plot_precrash_metrics.py` — 2026-04-15

Plots training metrics (training reward EMA, val accuracy EMA) for steps
0–47 of W&B run `kudzrfba` (`atts-grpo / 8b-sft-2gpu`), the run where the
Judge crashed at 11:48 UTC on 2026-04-15 and val accuracy zeroed from step
48 onward. Raw metric values are hardcoded inside the script — pulling
them down again from W&B is more expensive than re-running this. Output:
`precrash_metrics.{pdf,png}` in CWD.

### `judge_swap_monitor.sh` — 2026-04-17

Tail-and-trigger script that watched
`core_code/tmp/grpo_8b_sft_2gpu_bs96.log` for the line
`local_global_step_folder.*global_step_20` and, on match, called
`core_code/training/scripts/serve_judge_1gpu.sh` with
`NEW_UTIL=0.5` to swap the judge GPU memory budget mid-training. Specific
to that one bs96 run. Kept as a template for any future "automatically
react when training reaches step N" wiring.

## When to add an entry here

Add a new file + a paragraph above when:
- You wrote a script for a single incident and it produced a non-trivial
  finding (so the next investigator can re-run it).
- You captured raw data that **cannot** be regenerated (judge outputs,
  W&B metric snapshots that have since rolled off, etc.).
- You eyeballed something complex and wrote it down (markdown dumps,
  config diffs).

Do **not** add files that are still on an active code path. Those go in
`plots/`, `smoke/`, or `audit/` instead.

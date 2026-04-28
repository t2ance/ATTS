# `analysis/` — Top-level Map

This directory holds everything **downstream** of an experimental run:
the cached candidates, the run outputs, the analysis tools, the paper
plots, the smoke tests, and the historical archives. Think of it as the
opposite of `core_code/` (which is the code that *produces* runs).

Last full sweep: 2026-04-28 (Peijia).

## Subdirectory map

| Path | Purpose | Update cadence | Owner README |
|---|---|---|---|
| `cache/` | Pre-generated explore candidates per `(benchmark, model)` — the shared substrate the methods read from. Never edit by hand; produced by `core_code/scripts/<bench>/<model>/run_precache.sh`. | Append-only as new precache runs land. | (none — directory layout matches `<bench>/<model>/<qid>/explore_<n>/`) |
| `run/` | All evaluation runs — one directory per `(benchmark, method)`, with timestamped run subdirs. The primary output of `core_code/eval.py`. | Add a row to `run/README.md` §4.x **in the same commit** as a new run. | [`run/README.md`](run/README.md) — 7 per-benchmark tables, naming legend, maintenance contract |
| `audit/` | The 7-atom analysis framework (`audit.py`). Reads `run_config.json` + `results.jsonl` + `trajectories/` from `run/`. Self-contained tool. | Code edits when atoms change. No data files here. | (none — `audit.py` docstring is the spec) |
| `plots/` | Scripts that generate **paper figures**. These outputs are referenced from `Publication/paper/main.tex`. Treat as A-class assets — never delete without updating the paper. | When a figure changes for the paper. | (none — each script's docstring states which figure it produces) |
| `smoke/` | Reusable smoke tests / discriminating diagnostics for the orchestrator + StructuredOutput pipeline. Written for one investigation but generalisable; keep around for the next regression. | Add when a non-trivial diagnostic is built. | (none — each script's docstring is the spec) |
| `archive/` | One-off scripts, raw debugging dumps, and event-specific data. Not running code. Not deleted because (a) data is not regenerable, or (b) the script encodes how to reproduce a past investigation. | Append-only; entries dated by event. | [`archive/README.md`](archive/README.md) — per-file event log |

## Where to put a new file

| You produced... | Goes in... |
|---|---|
| A new evaluation run (eval.py output) | `run/<bench>/<method>/run_*/` + row in `run/README.md` §4.x |
| New cached explores from a precache script | `cache/<bench>/<model>/` (auto by `run_precache.sh`) |
| A new paper figure script | `plots/`, with docstring naming the `main.tex` figure id |
| A new smoke test that's likely to be re-run | `smoke/` |
| A one-off debug script / data dump for an incident | `archive/` + new entry in `archive/README.md` |
| A new audit atom | edit `audit/audit.py` |

## Cross-references

- `run/README.md` §5 lists the **downstream consumers** of every run
  directory (training data sources, paper figures, audit framework). Read
  that section before you delete or rename anything under `run/`.
- `core_code/training/sft/build_sft_hle_q101_300_thinking.py` hardcodes
  one specific run directory as its SFT data source — see
  `run/README.md` §4.5 for the "DO NOT DELETE" annotation.

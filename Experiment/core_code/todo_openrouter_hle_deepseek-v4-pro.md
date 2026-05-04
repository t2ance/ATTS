# TODO: `deepseek/deepseek-v4-pro` — HLE-100 ATTS via OpenRouter (paid)

## What this is

Run ATTS evaluation on the HLE-Verified gold-100 subset using
`deepseek/deepseek-v4-pro` via OpenRouter. Paid tier (NOT `:free`):
context_length 1048576, max_completion_tokens 384000, $0.435/M input,
$0.870/M output, reasoning=True, tools=True (verified via OpenRouter
`/api/v1/models` 2026-05-04). Standard paper config: `num_explores=8`,
`num=100`, `num_workers=4`, `seed=42`.

## Output target

- Per-explore cache: `analysis/cache/hle/openrouter_deepseek-v4-pro/gold/<qid>/explore_<n>/result.json`
- Per-question results: `analysis/run/hle/openrouter_deepseek-v4-pro/run_<TS>/results.jsonl`
- Paper integration: `tab:backbone-ablation` at `Publication/paper/main.tex` line 402-420. Add 1 row for DeepSeek-V4-Flash.

## Discipline

Every event below has explicit Gates with checkboxes. An event flips from `☐`
to `✓` only after all Gates pass AND each Gate's `Evidence ·` line is filled
with the actual measurement (qid counts, OpenRouter usage delta, log line
ref, results.jsonl line count — not "looks fine"). No silent skipping. No
narrative-only claims. No marking done before Evidence is recorded. A failed
Gate stops the run; do not advance until the failing Gate is either resolved
or escalated to the user.

## Cost / model anchors (DeepSeek-V4-Flash on OpenRouter, paid)

| Field | Value | Source |
|---|---|---|
| context_length | 1048576 (1M) | OpenRouter `/api/v1/models` 2026-05-04 |
| max_completion_tokens | 384000 (384K) | same |
| input price | $0.14 / M tokens | same |
| output price | $0.28 / M tokens | same |
| reasoning support | True | same |
| tools support | True | same |
| 429 handling | SDK auto-retries up to 8x with exponential backoff (`backends/openrouter.py:76`) | shared infra |
| Estimated precache cost | $2-4 (800 explores × ~10K output tokens × $0.870/M) | linear from output price |
| Estimated eval cost (orchestrator turns + Haiku judge) | $4-7 ($1-3 orchestrator + ~$3-4 Haiku at effort=low) | extrapolated from gpt-oss-20b eval |
| Estimated TOTAL | **<$15** (hard ceiling) | sum |

## API-key freshness pre-flight (NON-NEGOTIABLE per CLAUDE.md)

Before launching any phase below: curl OpenRouter to verify the active key returns HTTP 200, NOT just check that `$OPENROUTER_API_KEY` is non-empty.

```bash
curl -sS -o /dev/null -w "HTTP=%{http_code}\n" \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/auth/key
```

If HTTP=401 with `{"error":{"message":"User not found."}}` body, the env carries a stale key. Long-lived bash sessions (the parent of this Claude Code process) fixate env at startup; `~/.bashrc` edits do NOT propagate. The naive fix `bash -c 'source ~/.bashrc; ...'` does NOT work because stock `~/.bashrc` opens with `case $- in *i*) ;; *) return;; esac` early-return guard. Use this instead:

```bash
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"
```

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Precache process killed mid-run | Re-launch with same yaml; cache discipline auto-skips any qid+explore_idx that already has `result.json` | Log line `Tasks: K to run, J already cached` with J > 0 |
| Eval process killed mid-run | Add/refresh `resume: <run_dir>` in eval yaml pointing to RUN_DIR with the largest `results.jsonl` (newest mtime is NOT always largest — pick by `wc -l`) | Log line `Resuming ...: N rollouts already completed` with N > 0 |
| 429 burst exceeds SDK retries | Drop `num_workers` from 4 to 2 in yaml and restart | Log shows reduced concurrent in-flight |
| `output_tokens=384000` clusters in cache | Switch `effort` from `medium` to `low`; if still >5%, escalate — likely indicates prompt+model intrinsically over-budget | Cache audit shows >5% explores at exactly 384000 |
| Haiku judge `rate_limit · resets <UTC time>` | Wait until reset, then resume the eval RUN_DIR; cached grades persist as `<RUN_DIR>/grading/<qid>/judges/claude__claude-haiku-4-5-20251001/grade.json` | Resume banner shows `N already completed` matches results.jsonl `wc -l` |

## Risk register (known failure modes from 2026-05-04 gpt-oss-20b runs)

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | Free-tier `max_completion_tokens` cap silently clips reasoning + tool_call | gpt-oss-20b:free has 8192; chosen model here is PAID `deepseek/deepseek-v4-pro` with 384000 max_out — risk does NOT apply | Phase 2 G6 audits p95 against cap as belt-and-suspenders |
| R2 | LaTeX `\X` in tool_call.arguments → `json.JSONDecodeError` → `invalid_json_in_tool_args` timed_out | OpenAI parser does not unescape LaTeX backslashes | `backends/openrouter.py` already returns `timed_out` cache record (no crash); Phase 2 G3 measures rate; if >5% escalate code fix |
| R3 | Model returns reasoning-only with no tool_call | Model occasionally ignores forced `tool_choice` | `backends/openrouter.py` returns `timed_out` (no crash); Phase 2 G3 cap rolls into 10% threshold |
| R4 | 429 burst exceeds SDK retries (8) | OpenRouter per-minute cap | `backends/openrouter.py:76` `max_retries=8`; if Phase 2 G5 throughput drops, restart with `num_workers=2` |
| R5 | OpenRouter HTTP 200 with `choices=None` → orchestrator `RuntimeError` halts entire gather | sustained provider failure on one qid kills batch (per 2026-05-04 design directive) | Phase 3 On-fail spec: re-launch with resume — only that qid retries, rest of batch already in results.jsonl |
| R6 | OpenRouter returns non-JSON HTTP body | Provider HTML error page on overload | `backends/openrouter.py` `except (... + json.JSONDecodeError)` (incident 2026-05-03 fix) |
| R7 | Stale `OPENROUTER_API_KEY` in long-lived shell env → 401 "User not found." | Parent bash fixated at .bashrc-pre-edit state | API-key freshness pre-flight section above; curl-200 gate before each launch |
| R8 | Anthropic Haiku `rate_limit` mid-eval | Account daily/weekly quota | Resume mechanism reuses cached grades (judge_label-keyed); zero re-judge cost |

## Co-monitor — log paths for parallel watching

Working dir: `/data3/peijia/dr-claw/Explain/Experiment/core_code`. Absolute paths:

| Phase | Run log | Cache count |
|---|---|---|
| Phase 1 smoke | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_deepseek-v4-pro_smoke_<DATE>.log` | `find /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_deepseek-v4-pro_smoke/gold -name 'result.json' \| wc -l` |
| Phase 2 precache | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_deepseek-v4-pro_precache_<DATE>.log` | `find /data3/peijia/dr-claw/Explain/Experiment/analysis/cache/hle/openrouter_deepseek-v4-pro/gold -name 'result.json' \| wc -l` |
| Phase 3 eval | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/openrouter_hle_deepseek-v4-pro_eval_<DATE>.log` | `wc -l /data3/peijia/dr-claw/Explain/Experiment/analysis/run/hle/openrouter_deepseek-v4-pro/run_*/results.jsonl` |

---

## Phase 1 — Setup [0 done / 2]

01 ☐ Create yamls + launcher scripts
   ├ G1 ☐ Gate · `scripts/hle/openrouter/hle_deepseek-v4-pro_precache.yaml` exists with `explore_model: deepseek/deepseek-v4-pro`, `effort: medium`, `num_explores: 8`, `num: 100`, `num_workers: 4`, `cache_dir: ../analysis/cache/hle/openrouter_deepseek-v4-pro/gold`, `seed: 42`, `explore_timeout: 600.0`, judge `claude-haiku-4-5-20251001` (`effort: low` default per `benchmarks/specs.py:50`), `subset: gold`, `text_only: true`, all non-default values commented per config-override discipline
   │      Evidence ·
   ├ G2 ☐ Gate · `scripts/hle/openrouter/hle_deepseek-v4-pro_eval.yaml` exists; identical except `method.name: tts-agent`, `no_integrate: true`, points to same `cache_dir`, `log_dir: ../analysis/run/hle/openrouter_deepseek-v4-pro`
   │      Evidence ·
   ├ G3 ☐ Gate · `scripts/hle/openrouter/hle_deepseek-v4-pro_smoke.yaml` exists with `num: 2`, `num_explores: 2`, `effort: medium` (matches production), separate `cache_dir: ../analysis/cache/hle/openrouter_deepseek-v4-pro_smoke/gold` (do NOT pollute production cache)
   │      Evidence ·
   ├ G4 ☐ Gate · launcher `scripts/hle/openrouter/run_deepseek-v4-pro_precache.sh` and `_eval.sh` and `_smoke.sh` exist; each uses `PYTHONUNBUFFERED=1`, `conda run -n explain --no-capture-output`, NO inline API key (resolved via `eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"`), no shell args, writes log under `tmp/`
   │      Evidence ·
   └ How · Mirror `scripts/hle/openrouter/hle_gpt-oss-20b_free_low_precache.yaml` (and its `eval` and `smoke` siblings); diff: `explore_model`, `cache_dir` slug (no `_free`), absolute paths in launchers.

02 ☐ Pre-flight smoke (num=2, n_explore=2 → 4 explores total)
   ├ G1 ☐ Gate · API-key freshness curl returns HTTP=200 BEFORE launch (run pre-flight curl from "API-key freshness pre-flight" section above)
   │      Evidence ·
   ├ G2 ☐ Gate · Smoke completes 4/4 explores within 5 min wall-clock without uncaught crash (`grep -c Traceback <log> == 0` and `find <smoke_cache> -name result.json | wc -l == 4`)
   │      Evidence ·
   ├ G3 ☐ Gate · ≥3/4 result.json have non-null `answer` field (≤1/4 timed_out — proves end-to-end ATTS works)
   │      Evidence ·
   ├ G4 ☐ Gate · `output_tokens` distribution across 4 explores: max sample < 384000 (proves the 384K cap is NOT systematically hit at effort=medium)
   │      Evidence ·
   ├ G5 ☐ Gate · `finish_reason='length'` rate < 25% across the 4 explores (paid tier has 384K headroom; high length rate would indicate runaway reasoning)
   │      Evidence ·
   ├ G6 ☐ Gate · trajectory text in ≥1 result.json contains BOTH a non-empty reasoning excerpt AND a parsed `StructuredOutput` JSON block (proves Anthropic-Skin → tool-call path is working end-to-end)
   │      Evidence ·
   └ How · `bash scripts/hle/openrouter/run_deepseek-v4-pro_smoke.sh` ; tail log; after exit, audit smoke cache with one-shot Python (`for f in result.json files: print finish_reason, output_tokens, reason if timed_out`)

---

## Phase 2 — Precache [0 done / 1]

03 ☐ Full precache 800/800 explores (100 qids × 8 explores each)
   ├ G1 ☐ Gate · API-key freshness curl returns HTTP=200 BEFORE launch
   │      Evidence ·
   ├ G2 ☐ Gate · `find <cache_dir> -name "result.json" | wc -l` returns exactly 800
   │      Evidence ·
   ├ G3 ☐ Gate · `find <cache_dir> -mindepth 1 -maxdepth 1 -type d | wc -l` returns exactly 100; every qid dir contains exactly 8 sub-dirs
   │      Evidence ·
   ├ G4 ☐ Gate · cumulative `timed_out` rate ≤ **5%** at `effort: medium` (calibration: gpt-oss-20b:free LOW achieved 8.8% under a 8192 cap; deepseek-v4-pro with 384K cap should comfortably beat that)
   │      On-fail · IF measured > 5% AND ≤ 10%: log breakdown by reason; advance to Phase 3 with footnote in paper. IF > 10%: STOP, audit failure modes, decide whether `effort: low` fallback or model swap.
   │      Evidence ·
   ├ G5 ☐ Gate · zero `Traceback` in run log; the 5 known soft-failure modes (no_tool_call / invalid_json / transient / empty_choices / json_decode_body) all return cached `timed_out` records
   │      Evidence ·
   ├ G6 ☐ Gate · throughput ≥ 1 explore/min over a rolling 10-min window in steady state (paid tier has no daily-budget throttle; sub-1/min indicates routing issue)
   │      Evidence ·
   ├ G7 ☐ Gate · `output_tokens` p95 < 200000 (well under the 384K cap; if p95 >= 200K, prompt is exhausting reasoning budget — investigate before eval)
   │      Evidence ·
   ├ G8 ☐ Gate · OpenRouter cumulative cost for this precache < $5 (curl `/auth/key` after run; `usage` delta from start)
   │      Evidence ·
   ├ G9 ☐ Soft-Gate · sample 5 random non-`timed_out` result.json (`find ... | shuf -n 5`); for each verify:
   │      (a) trajectory text non-empty AND contains both reasoning_details and StructuredOutput JSON
   │      (b) `answer` field populated and matches the schema
   │      (c) `confidence` ∈ [0, 1]
   │      (d) reasoning text shows actual problem-solving (not "I don't know" / not garbled)
   │      (e) no LaTeX repair regression (problem with LaTeX → trajectory references math symbols correctly)
   │      Justification (required) · Write 1-2 sentences per sub-check (a-e) citing the qid, the result.json path, and the concrete evidence (e.g. "qid `66e8...` line `<n>` of trajectory contains `<excerpt>`"). Do NOT say "looks fine" or "verified manually" — both are auto-rejected.
   │      Evidence ·
   └ How · `bash scripts/hle/openrouter/run_deepseek-v4-pro_precache.sh` ; capture PID and absolute log path immediately and surface to user; periodic check via `find <cache_dir> -name "result.json" | wc -l`. ETA ~3-5 hours at num_workers=4 paid-tier rate (no free-tier throttle).

---

## Phase 3 — Eval [0 done / 1]

04 ☐ Eval n=100 against precache → Pass@1 + oracle Pass@8
   ├ G1 ☐ Gate · API-key freshness curl returns HTTP=200 BEFORE launch
   │      Evidence ·
   ├ G2 ☐ Gate · `wc -l results.jsonl` == 100
   │      Evidence ·
   ├ G3 ☐ Gate · `progress.json` reports `total_correct`, `accuracy_pct`, `judge_cost_usd` numerically (not null) AND `status: completed`
   │      Evidence ·
   ├ G4 ☐ Gate · Pass@1 ∈ [10%, 70%] sanity envelope. Lower bound: deepseek-v4-pro is paid-tier reasoning model, expected to beat gpt-oss-20b:free's 3% by a wide margin. Upper bound: HLE-Verified gold is hard; >70% suggests data leak. STOP and escalate either way.
   │      Evidence ·
   ├ G5 ☐ Gate · oracle@8 ≥ Pass@1 by at least 5pp (otherwise ATTS adds no value over single explore — model can't use parallel exploration)
   │      Evidence ·
   ├ G6 ☐ Gate · Haiku judge cost < $5 (calibration: gpt-oss-20b:free LOW eval cost ~$6.5 across 100 questions with multi-resume overhead)
   │      Evidence ·
   ├ G7 ☐ Gate · zero `Traceback` (or all Traceback are caught soft-fails: empty_choices RuntimeError counts as a recoverable halt with resume — see On-fail)
   │      On-fail (R5 trigger) · If `RuntimeError [openrouter run_tool_conversation] sustained empty_choices` halts the gather mid-run, just re-launch the SAME yaml — `resume:` in yaml makes only un-completed questions retry; the 99 already-graded persist via `<RUN_DIR>/grading/<qid>/judges/claude__claude-haiku-4-5-20251001/grade.json` cache.
   │      Evidence ·
   └ How · `bash scripts/hle/openrouter/run_deepseek-v4-pro_eval.sh` ; cache discipline ensures explores are not re-generated (`cache_only` mode); only Haiku judge calls cost money on first pass; resumes are free for already-graded questions.

---

## Phase 4 — Paper integration [0 done / 1]

05 ☐ Add 1 row to `tab:backbone-ablation` in `Publication/paper/main.tex`
   ├ G1 ☐ Gate · row inserted between line 402 and line 420 with: model name `deepseek-v4-pro`, n_explore=8, Pass@1, oracle Pass@8 numbers from Phase 3 G3
   │      Evidence ·
   ├ G2 ☐ Gate · footnote attached to the row reading exactly "OpenRouter paid; effort=medium; ctx=1M, max_out=384K"
   │      Evidence ·
   ├ G3 ☐ Gate · `cd ../../Publication/paper && bash compile.sh` exits with status 0 (no LaTeX errors)
   │      Evidence ·
   ├ G4 ☐ Gate · `Publication/paper/build/main.pdf` mtime newer than `main.tex`; PDF visually inspected — table renders without `Overfull \hbox` warning beyond existing baseline; no margin overflow
   │      Evidence ·
   └ How · edit main.tex, add row + footnote; recompile; visually verify rendered PDF page containing tab:backbone-ablation.

# TODO: GPT-OSS-20B Variant B (`_sonnet_orch`) — paper appendix `tab:qwen36-baseline`

## What this is

Variant B of the GPT-OSS-20B experiment plan: **Sonnet-cached explorer + GPT-OSS-20B orchestrator**, mirroring the existing appendix table `tab:qwen36-baseline` already in the paper for Qwen3.6-35B-A3B-FP8 (paper line 824-839 area).

- **Explorer:** Claude Sonnet 4.6 — but we DON'T actually call the API. We read the pre-existing cache at `analysis/cache/<bench>/sonnet/...` (or `sonnet/gold` for HLE) populated by the prior Sonnet runs that produced `tab:main-results`. `cache_only=true` enforces this: any cache miss → STOP, never silently call the API.
- **Orchestrator:** `gptoss-20b` (alias of `openai/gpt-oss-20b`) served by local vLLM single-card (or DP=N if multiple cards free).
- **Synthesizer / final answer:** GPT-OSS orchestrator (`no_integrate=true`).
- **Benchmarks:** HLE-Verified text-only / GPQA-Diamond / LCB. **BabyVision is OUT-OF-SCOPE** — gpt-oss-20b is text-only (no vision tower); sending image-bearing prompts to a text-only model and reporting whatever degraded number falls out would contaminate the appendix table.
- **Sampling block on the orchestrator:** OpenAI gpt-oss harmony defaults — T=1.0, top_p=1.0, max_tokens=32768 for orchestrator turns (matches Variant-A orchestrator sampling so the only knob that differs is the explorer source). Reasoning effort pinned to `medium` via `extra_body.reasoning={"effort":"medium"}` (verify exact key during item 02 G3).

This experiment isolates the **orchestrator's contribution** from explorer quality. The companion Variant A (`_exp_orch`, separate todo file) measures "what GPT-OSS-20B can do as a full ATTS stack on its own explores"; this Variant measures "what GPT-OSS-20B can do as a pure aggregator of high-quality (Sonnet-grade) candidates". Both numbers are needed for the paper's GRPO uplift story — the latter is the untrained-base baseline against which post-GRPO orchestrator improvement will be measured.

## Output target

3 GPT-OSS rows added to paper appendix table `tab:qwen36-baseline` at `Publication/paper/main.tex` (currently around line 824-839): HLE / GPQA / LCB. New `Backbone` value `GPT-OSS-20B (untrained)` and `Effort = thinking` (or `medium` reflecting reasoning-effort knob). The existing Qwen rows + any Gemma rows already in the table stay intact; GPT-OSS rows go below as a separate sub-block. Compiled `build/main.pdf` shows the table with all included backbones' untrained-base data side-by-side. Caption updated with "GPT-OSS-20B is text-only; BabyVision N/A" footnote so absence of BV row is explicit.

The numbers establish what an untrained ~21B (3.6B-active) MoE orchestrator can do as a pure aggregator of fixed Sonnet-quality explores — directly comparable to the Qwen-thinking row (which measures the same thing for ~35B FP8 thinking model) and to the Sonnet-self orchestrator (main results table).

## Discipline

Every event has Gates with checkboxes. An event flips `☐` → `✓` only after **all** its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement (e.g. "27/800 = 3.4% timed_out"). Soft-Gates require Justification with concrete qid + line-number evidence — never narrative claims like "looks fine". No silent skipping, no marking done before evidence is recorded. The prior Gemma run (2026-05-02 morning) violated this discipline and produced contaminated paper numbers (8.00 written from 100% empty predictions + judge hallucination); this file enforces the discipline structurally.

## GPU availability is a SOFT constraint

Target topology: single-card serve (gpt-oss-20b ~16GB VRAM under MXFP4 quant fits in any A100/A6000/L40 80GB). Optional DP=2/4 if multiple cards free. Pre-flight measures `N_avail`; serve script and downstream throughput expectations scale with N_avail. Quality gates (Pass@1 deterministic check, judge integrity, cache-hit rate) are NOT affected by N_avail — they are pipeline-correctness checks. Only N_avail=0 stops the run; 1/2/4 all proceed.

Allocation rule: do NOT collide with running Gemma serve or running GPT-OSS Variant-A serve. If either alive, wait for completion OR use a non-overlapping card subset.

To start the vLLM serve: `bash scripts/gpqa/grpo/serve_gptoss20b_dp{N_avail}.sh` (clone from `serve_gemma4_26b_a4b_dp2.sh` and adapt — see Phase 1 item 05 for full flag list). Verify health: `curl http://localhost:8000/v1/models` returns `gptoss-20b`; tail log for `Maximum concurrency for X tokens per request` line; `nvidia-smi` shows N_avail × ~18 GiB allocated.

## Leaderboard anchor (Variant-B-specific, deterministic)

For Variant B, `Pass@1 = first cached Sonnet explore's correctness rate`. Cache reads are deterministic — Pass@1 must EXACTLY match Sonnet's published Pass@1 from the paper:

| Bench | Sonnet Pass@1 | Tolerance | Source line in `main.tex` |
|---|---:|---:|---|
| HLE | 48.00 | ±0.5pp (FP noise) | `tab:backbone-ablation` line 414 |
| GPQA | 74.24 | ±0.5pp | `tab:backbone-ablation` line 415 |
| LCB | TBD (look up `tab:main-results` Sonnet row before running Phase 2) | ±0.5pp | `tab:main-results` |

Deviation > 0.5pp = cache key mismatch / wrong cache_dir / corrupted cache file / different rollout idx 0. STOP and debug; do NOT proceed to write paper. The `±0.5pp` tolerance is for floating-point noise only — not run-to-run variance, since cache is deterministic.

For ATTS Acc (orchestrator output), the leaderboard anchor is the existing **Qwen36-baseline appendix Gain values** (paper line 836-839 area, "thinking" rows): HLE +8.0, GPQA +8.58, LCB −0.57. GPT-OSS's Variant B Gain should fall within ±10pp of those (similar open-weights MoE orchestrator on identical Sonnet cache, but smaller active-param count so larger tolerance). Outside that band → flag for manual review (not auto-stop, since this is cross-model-family comparison and gpt-oss-20b active params are ~half of Qwen-A3B).

## Resume / restart procedure

If a run dies mid-way (OOM / SIGTERM / network blip / vllm crash), the cache + results.jsonl are the resume substrate. NEVER re-run from scratch — that burns wall time. Cache is read-only in this Variant (we never modify Sonnet cache files), so cache integrity = file integrity (immutable during run).

| Failure point | Recover by | Banner verification (mandatory after restart) |
|---|---|---|
| Mid eval (e.g. Q50/100) | Add `resume: <RUN_DIR>` to the eval YAML pointing to the dying run's `analysis/run/<bench>/<...>/run_<timestamp>/`. Eval skips already-graded `(qid, rollout_idx)` rows via `results.jsonl`. | Both lines must appear: (a) `Resuming ...: N rollouts already completed` with N>0; (b) `Questions to run: M (N already completed, M+N total)`. |
| vLLM serve crash | Restart serve via `bash scripts/gpqa/grpo/serve_gptoss20b_dp{N_avail}.sh`. Eval clients retry on next call; no data loss. | `curl :8000/v1/models` returns 200 + `gptoss-20b` alias. |
| Pick which `RUN_DIR` to resume from | Pick by **largest** `wc -l results.jsonl`, NOT mtime (newer dirs may have crashed earlier) | n/a |

## Risk register (known failure modes)

These are the bugs the gates below are designed to catch — do NOT remove the corresponding gate just because a run looks fine.

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | Orchestrator emits tool calls in harmony channels but vllm OpenAI compat layer doesn't surface as `message.tool_calls[]` → 100/100 empty predictions | gpt-oss harmony format requires correct `--tool-call-parser openai`; without it assistant_text channel leaks as plain `content` | Phase 1 item 05 smoke G4 (verify ≥1 structured `tool_calls` per question + cache hits visible in trajectory) |
| R2 | Judge marks empty `predicted_answer` as correct → contaminated paper number | Judge ignored `_JUDGE_BASE` rule 5 ("no extractable answer → correct=false") | Phase 2 each eval G2/G3 (post-run filter on empty + refusal regex; `is_correct=true` AND empty must be 0 rows) |
| R3 | LCB `metadata_list` IndexError on subprocess SIGKILL/segfault | Upstream `lcb_runner` bug (already patched in `compute_code_generation_metrics.py:53`) | Phase 2 item 08 G2 (subprocess survival check: no IndexError raised) |
| R4 | vLLM version too old to support gpt-oss harmony format | gpt-oss requires vllm with `+gptoss` extras OR vllm ≥ 0.20 mainstream | Pre-flight env version check (vllm ≥ 0.20 OR `pip show vllm | grep gptoss`) before serve |
| R5 | `served-model-name` mismatch — yaml references `gptoss-20b` but serve aliases `openai/gpt-oss-20b` (full HF id) → 404 on every chat completion | inconsistent alias naming between YAML and serve script | Phase 1 item 05 G1 (`curl :8000/v1/models` returns alias matching what YAMLs reference) |
| RB1 | `cache_only=true` flag silently bypassed → Anthropic API call charged | yaml parsing bug or wrong field name | Phase 2 each eval `Cost budget` Gate (`sum(cost_usd) == 0` mandatory) |
| RB2 | Wrong `cache_dir` path → cache miss for every (qid, explore_idx); cache_only blocks the API call → eval crashes early with "cache miss" | typo in YAML | Phase 1 item 01 (Sonnet cache integrity verification BEFORE eval) |
| RB3 | Sonnet cache files modified between Sonnet run and this Variant B run | accidental rerun / cache pollution / wrong git branch | Phase 1 item 01 G6 (mtime check: no file modified after Sonnet runs completed) |
| RB4 | Pass@1 doesn't match published Sonnet Pass@1 → wrong cache or cache key changed | cache key includes (model, sampling, prompt); one of these drifted | Phase 2 each eval `Pass@1 deterministic check` Gate (±0.5pp hard, STOP on miss) |
| RB5 | Gain compared against wrong reference (e.g. some unrelated GPT-OSS experiment's number rather than Qwen36-baseline appendix) → mis-interpreted result | confusing this Variant's reference anchor in paper writeup | Phase 3 G_review explicit comparison anchor: Qwen36-baseline appendix Gain (HLE +8.0, GPQA +8.58, LCB −0.57) |
| RB6 | Sonnet cache directory layout differs from expected (e.g. extra `gold/` subdir for HLE only) → wrong path | path convention drift between benchmarks | Phase 1 item 01 G1-G3 (per-benchmark explicit path check: HLE uses `sonnet/gold/`, others use `sonnet/`) |

## Co-monitor — log paths for parallel watching

| Phase | Run log (stdout/stderr) | Power log |
|---|---|---|
| 05 smoke (HLE, num=2) | `tmp/eval_hle_gptoss_sonnet_smoke.log` | `tmp/power_hle_gptoss_sonnet_smoke.log` |
| 06 HLE eval (100Q) | `tmp/eval_hle_gptoss_sonnet_orch.log` | `tmp/power_hle_gptoss_sonnet_orch.log` |
| 07 GPQA eval (198Q) | `tmp/eval_gpqa_gptoss_sonnet_orch.log` | `tmp/power_gpqa_gptoss_sonnet_orch.log` |
| 08 LCB eval (175Q) | `tmp/eval_lcb_gptoss_sonnet_orch.log` | `tmp/power_lcb_gptoss_sonnet_orch.log` |

User can `tail -f /data3/peijia/dr-claw/Explain/Experiment/core_code/<path>` for any of these. All paths are absolute-resolvable from `core_code/` working dir.

## Phase 1 — Config & cache verification + serve [0/5]

01 ☐ Verify Sonnet cache is intact for the 3 benchmarks we will use (read-only integrity check before any eval)
   ├ G1 ☐ Gate · `cache/hle/sonnet/gold/` exists and contains ≥100 qid subdirs (matches num=100; each qid is a Mongo-style 24-char hex string)
   │      Evidence · 
   ├ G2 ☐ Gate · `cache/gpqa/sonnet/` exists and contains ≥198 qid subdirs (GPQA-Diamond size)
   │      Evidence · 
   ├ G3 ☐ Gate · `cache/lcb/sonnet/` exists with ≥175 qid subdirs (LCB v6 size)
   │      Evidence · 
   ├ G4 ☐ Gate · For each of the 3 benchmarks, sample 5 random qids — each has 8/8 `explore_<n>/result.json` non-empty AND `output.md` non-empty AND `result.json` lacks `timed_out=true` or `parse_failed=true`
   │      Evidence · 
   ├ G5 ☐ Gate · No file in any sonnet cache subtree was modified after the Sonnet runs completed (file mtimes ≤ 2026-04-30); `find cache/<bench>/sonnet -newer cache/<bench>/sonnet/.timestamp_marker` returns empty (or use the most-recent-Sonnet-run-mtime as marker)
   │      Evidence · 
   └ How  · `for b in hle/gold gpqa lcb; do echo "$b:"; ls cache/$b/sonnet 2>/dev/null | wc -l; done` + sample-5 file integrity check + `find cache/<b>/sonnet -newer <marker>`

02 ☐ Create `hle_gptoss20b_sonnet_orch.yaml`
   ├ G1 ☐ Gate · YAML parses via `load_config(path, schema=EvalConfig)` without ValidationError; written to `scripts/hle/grpo/`
   │      Evidence · 
   ├ G2 ☐ Gate · Required fields verified by reading YAML: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gptoss-20b`; `cache_dir: ../analysis/cache/hle/sonnet/gold`; `cache_only: true`; `no_integrate: true`; `num_explores: 8`; `num: 100`; `log_dir: ../analysis/run/hle/gptoss20b_sonnet_orch`
   │      Evidence · 
   ├ G3 ☐ Gate · sampling block matches the canonical GPT-OSS orchestrator block defined in companion Variant-A YAML at `scripts/hle/grpo/hle_gptoss20b_exp_orch.yaml` (T=1.0, top_p=1.0, max_tokens=32768, reasoning effort `medium`) — orchestrator behavior identical between Variants
   │      Evidence · 
   ├ G4 ☐ Gate · `benchmark.judge` block matches canonical GPT-OSS judge block at `scripts/hle/grpo/hle_gptoss20b_exp_orch.yaml` (vllm gptoss-20b, max_tokens=4096) — judge identity must match for apples-to-apples comparison and judge cache reuse. If Variant-A YAML doesn't exist yet, this gate blocks until it does (cross-Variant judge consistency is non-negotiable; either define both YAMLs together with shared judge block, or copy Variant-A's exact judge block once it's authored).
   │      Evidence · 
   └ How  · clone `scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml` as starting template, swap `explore_model` → `claude-sonnet-4-6`, swap `orchestrator_model` → `gptoss-20b`, swap `cache_dir` → `../analysis/cache/hle/sonnet/gold`, set `cache_only: true`, change `log_dir` suffix to `_sonnet_orch`, swap sampling + judge blocks to GPT-OSS defaults, save

03 ☐ Create `gpqa_gptoss20b_sonnet_orch.yaml` and `lcb_gptoss20b_sonnet_orch.yaml`
   ├ G1 ☐ Gate · both YAMLs parse without ValidationError; written to `scripts/gpqa/grpo/` and `scripts/lcb/grpo/` respectively
   │      Evidence · 
   ├ G2 ☐ Gate · GPQA YAML required fields: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gptoss-20b`; `cache_dir: ../analysis/cache/gpqa/sonnet`; `cache_only: true`; `num: 198`; `log_dir: ../analysis/run/gpqa/gptoss20b_sonnet_orch`. NO `judge` block (GPQA uses string-match A-E, GPQASpec rejects judge field).
   │      Evidence · 
   ├ G3 ☐ Gate · LCB YAML required fields: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gptoss-20b`; `cache_dir: ../analysis/cache/lcb/sonnet`; `cache_only: true`; `num: 175`; `log_dir: ../analysis/run/lcb/gptoss20b_sonnet_orch`. Sampling `max_tokens=20000` for orchestrator (LCB uses 20000 to leave headroom for accumulated multi-turn explore tool_response inputs). NO `judge` block (LCB uses code execution).
   │      Evidence · 
   ├ G4 ☐ Gate · BOTH YAMLs reference the same canonical GPT-OSS orchestrator sampling block as item 02 (T=1.0, top_p=1.0, reasoning effort `medium`) so orchestrator behavior is identical across all 3 benchmarks; only `max_tokens` differs (LCB 20000 vs HLE/GPQA 32768)
   │      Evidence · 
   └ How  · clone `scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_exp_orch.yaml` and `scripts/lcb/grpo/lcb_gemma4_26b_a4b_exp_orch.yaml`, swap fields per G2/G3, save

04 ☐ Write `serve_gptoss20b_dp{N_avail}.sh` under `scripts/gpqa/grpo/`
   ├ G1 ☐ Gate · script written with these flags: `--served-model-name gptoss-20b`, `--tensor-parallel-size 1`, `--data-parallel-size {N_avail}`, `--gpu-memory-utilization 0.85`, `--max-model-len 131072` (gpt-oss native context per recipe), `--max-num-batched-tokens 8192`, `--enable-auto-tool-choice`, `--tool-call-parser openai`, `--trust-remote-code`, `--port 8000`, log to `tmp/vllm_serve_gptoss20b_dp{N_avail}.log`. Inline comments explain each non-default override (per `comment_on_config_overrides` rule)
   │      Evidence · 
   ├ G2 ☐ Gate · script does NOT include `--reasoning-parser` (gpt-oss harmony parser is built-in via `--tool-call-parser openai` per vllm docs as of 2026-05); script does NOT include `--structured-outputs-config` xgrammar/outlines flags (gpt-oss should NOT have Gemma's vllm#40080 JSON-loop bug per harmony format design — verify in smoke item 05)
   │      Evidence · 
   ├ G3 ☐ Gate · CUDA_VISIBLE_DEVICES set to N_avail card ids that do NOT collide with active Gemma serve or active GPT-OSS Variant-A serve
   │      Evidence · 
   └ How  · clone `scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp2.sh` as template; replace model id / alias / TP-DP / context-len / parsers; remove Gemma-specific comments; add gpt-oss-specific comments

05 ☐ Variant B serve health + smoke test (HLE, num=2 = 16 cache reads, GPT-OSS orch synthesizes)
   ├ G1 ☐ Gate · vLLM serve up: `bash scripts/gpqa/grpo/serve_gptoss20b_dp{N_avail}.sh`; `curl http://localhost:8000/v1/models` returns 200 with `"id":"gptoss-20b"` (defends R5); serve log shows `Maximum concurrency for X tokens per request` and zero `Traceback` / `RuntimeError` / `OOM`
   │      Evidence · 
   ├ G2 ☐ Gate · 2 questions × 8 explores = 16 cache reads, all hit (zero cache misses; if any miss → cache_dir wrong or cache key mismatch → cache_only path raises and crashes)
   │      Evidence · 
   ├ G3 ☐ Gate · `sum(cost_usd) == 0` across 2 results.jsonl rows (cache_only=true forbids API call)
   │      Evidence · 
   ├ G4 ☐ Gate · 2 final `predicted_answer` non-empty (orchestrator successfully synthesized) AND orchestrator emitted ≥1 structured `tool_calls` per question (verify by `grep -c "tool_calls" trajectories/<qid>/trajectory.md` ≥ 1) AND zero token-repetition loops in orchestrator output (defends R1 + harmony format sanity)
   │      Evidence · 
   ├ G5 ☐ Gate · `first_candidate_correct` for these 2 questions matches the Sonnet first-candidate ground truth (cross-check by reading `cache/hle/sonnet/gold/<qid>/explore_1/result.json` and the Sonnet run's results.jsonl for same qid)
   │      Evidence · 
   └ How  · start serve via the new sh; temp YAML clone of `hle_gptoss20b_sonnet_orch.yaml` with `num: 2`; run `eval.py`; verify via `results.jsonl` + `trajectories/<qid>/trajectory.md`

## Phase 2 — Eval [0/3]

06 ☐ HLE eval (Sonnet cache, GPT-OSS orch, 100Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true` (judge integrity, defends R2)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with refusal phrase (regex `(?i)i (don'?t|do not) know|cannot (determine|answer)|unable to|insufficient`) AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` rate ∈ [47.5%, 48.5%] (Sonnet HLE Pass@1 = 48.00 ± 0.5pp). Outside range → STOP, cache key or cache_dir wrong; do NOT write to appendix.
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0` (cache_only=true forbids Anthropic API call); non-zero = cache_only flag silently bypassed (RB1)
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥150W × ≥80% wall time (calibrated for gpt-oss-20b 3.6B-active MoE on A100 80GB PCIe; lower than Gemma's 200W threshold because smaller active params dispatch lower expert load)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%); only decrease if vllm log shows `waiting>0` or KV pool pressure
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_hle_gptoss_sonnet_orch.log`
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids (seed=42); for each verify:
   │      (a) judge `verdict_reasoning` cites gold value verbatim, ≤2 sentences, no re-derivation keywords
   │      (b) orchestrator visibly READ Sonnet cache content (final answer references explore reasoning) AND emitted ≥1 `tool_calls`; trajectory shows `tool_result` content from cache (not "cache miss" or empty)
   │      (c) `predicted_answer` length distribution across 100 rows: median ≥50 chars AND IQR > 30 chars
   │      (d) Gain plausible vs Qwen36-baseline appendix HLE thinking Gain (+8.0); |GPT-OSS Variant B Gain − +8.0| < 10pp expected (smaller backbone, larger tolerance); outside band = flag, not auto-stop
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + `trajectories/<qid>/trajectory.md` line numbers + `results.jsonl` row index + cache_dir hits as concrete evidence; do NOT just say "looks fine". Failing this gate (a-c hard, d soft flag) = STOP, do NOT write to appendix.
   │      Evidence · 
   └ How  · `eval.py --config scripts/hle/grpo/hle_gptoss20b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_hle_gptoss_sonnet_orch.log`

07 ☐ GPQA eval (Sonnet cache, GPT-OSS orch, 198Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows where `predicted_answer` lacks A-E letter AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` ∈ [73.74%, 74.74%] (Sonnet GPQA Pass@1 = 74.24 ± 0.5pp from `tab:backbone-ablation`)
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0`
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥150W × ≥80% wall time (tail 5% exempt)
   │      On-fail · tune `num_workers` Δ=+50% first
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_gpqa_gptoss_sonnet_orch.log`
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) extracted MC letter matches orchestrator's committed final letter (no stray-letter extraction)
   │      (b) orchestrator visibly aggregated 8 Sonnet explore letters (e.g. majority vote, principled selection); NOT first-explore copy AND `cost_usd==0` confirms cache_only
   │      (c) extracted letter distribution across 198 rows: A/B/C/D within ±15% of uniform (24.3% ± 3.6pp)
   │      (d) Gain plausible vs Qwen36-baseline appendix GPQA thinking Gain (+8.58); |GPT-OSS Variant B Gain − +8.58| < 10pp expected
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + trajectory line + cache hit trace. Failing (a-c hard) = STOP.
   │      Evidence · 
   └ How  · `eval.py --config scripts/gpqa/grpo/gpqa_gptoss20b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_gpqa_gptoss_sonnet_orch.log`

08 ☐ LCB eval (Sonnet cache, GPT-OSS orch, 175Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · `lcb_runner` returns no `metadata_list` IndexError (R3 defense)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with empty `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · 0 rows where extracted code block empty AND `is_correct=true`
   │      Evidence · 
   ├ G5 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` ∈ [Sonnet LCB Pass@1 ± 0.5pp]; LCB Sonnet Pass@1 looked up from `tab:main-results` Sonnet row BEFORE running this eval (record value here in Evidence)
   │      Evidence · 
   ├ G6 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0`
   │      Evidence · 
   ├ G7 ☐ Gate · per-GPU power ≥150W × ≥80% wall time
   │      On-fail · tune `num_workers` Δ=+50% first
   │      Evidence · 
   ├ G8 ☐ Gate · zero `Traceback` in `tmp/eval_lcb_gptoss_sonnet_orch.log`
   │      Evidence · 
   ├ G9 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) extracted code is valid Python (`ast.parse`); lcb_runner pass/fail flags consistent with `is_correct`
   │      (b) orchestrator visibly synthesized from Sonnet cache (final code structurally similar to ≥1 cached explore code, not byte-identical); `cost_usd==0` confirms cache_only
   │      (c) submitted code length distribution: median ≥30 lines across 175 rows
   │      (d) Gain plausible vs Qwen36-baseline appendix LCB thinking Gain (−0.57); |GPT-OSS Variant B Gain − (−0.57)| < 12pp expected (LCB is the hard case for orchestrator-only ablation; smaller backbone may amplify the negative gain)
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + trajectory + lcb_runner metadata. Failing (a-c hard) = STOP.
   │      Evidence · 
   └ How  · `eval.py --config scripts/lcb/grpo/lcb_gptoss20b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_lcb_gptoss_sonnet_orch.log`

## Phase 3 — Paper integration [0/3]

09 ☐ Parse 3 `results.jsonl` (Variant B `_sonnet_orch`) → Pass@1 / Acc / Gain / $/q
   ├ G1 ☐ Gate · 3 metric rows printed for HLE / GPQA / LCB
   │      Evidence · 
   ├ G2 ☐ Gate · sum check `Acc - Pass@1 == Gain` for each row (within 0.1pp rounding)
   │      Evidence · 
   ├ G3 ☐ Gate · all 3 Pass@1 values match Sonnet's published Pass@1 (deterministic): HLE 48.00 / GPQA 74.24 / LCB <looked-up>; ±0.5pp
   │      Evidence · 
   ├ G4 ☐ Gate · `sum(cost_usd) for all 3 results.jsonl ≤ $0.10` — should be exactly $0.00 (cache_only mode); any non-zero indicates RB1 (cache_only bypass) and invalidates all rows
   │      Evidence · 
   └ How  · python script: read each `run/<bench>/gptoss20b_sonnet_orch/run_*/results.jsonl`, count `first_candidate_correct` + `is_correct`, sum `cost_usd`

10 ☐ Append 3 GPT-OSS rows to paper appendix `tab:qwen36-baseline`
   ├ G1 ☐ Gate · table at `Publication/paper/main.tex` (lines 824-839 area) has 3 NEW rows below the existing Qwen-thinking block (and Gemma block if present), with `Backbone="GPT-OSS-20B (untrained)"`, `Effort="thinking"` (or `"medium"` reflecting reasoning-effort knob), per-bench Pass@1 / Acc / Gain / $/q
   │      Evidence · 
   ├ G2 ☐ Gate · row schema matches existing rows in same table (columns: Backbone & Effort & Bench & N & Pass@1 & Acc & Gain & $/q)
   │      Evidence · 
   ├ G3 ☐ Gate · `\midrule` separator inserted between previous backbone block and new GPT-OSS block; existing rows UNTOUCHED
   │      Evidence · 
   ├ G4 ☐ Gate · table caption (`\caption{...}`) updated to mention GPT-OSS-20B baseline rows AND note "GPT-OSS-20B is text-only; BabyVision N/A"; `\label{tab:qwen36-baseline}` retained but caption text expanded
   │      Evidence · 
   └ How  · Edit tool, insert under existing rows; update caption text

11 ☐ Recompile paper + verify the GPT-OSS table-block renders correctly
   ├ G1 ☐ Gate · `compile.sh` exits 0
   │      Evidence · 
   ├ G2 ☐ Gate · `Publication/paper/build/main.pdf` mtime updated to current run
   │      Evidence · 
   ├ G3 ☐ Gate · `tab:qwen36-baseline` (appendix) shows 3 NEW GPT-OSS rows below existing rows; total table rows = previous count + 3
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Overfull \hbox` warnings on the table page in compile log
   │      Evidence · 
   ├ G5 ☐ Gate · cross-Variant sanity: this Variant's Pass@1 (Sonnet first-candidate) ≫ the full-stack GPT-OSS Pass@1 from `tab:backbone-ablation` on every benchmark — explorer quality difference reflects in Pass@1 column directly
   │      Evidence · 
   └ How  · `cd Publication/paper && bash compile.sh`; visual inspection of the appendix table page

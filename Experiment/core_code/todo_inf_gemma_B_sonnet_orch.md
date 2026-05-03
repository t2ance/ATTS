# TODO: Gemma-4-26B-A4B-it Variant B (`_sonnet_orch`) — paper appendix `tab:qwen36-baseline`

## What this is

Variant B of the Gemma-4-26B-A4B-it experiment plan: **Sonnet-cached explorer + Gemma orchestrator**, mirroring the existing appendix table `tab:qwen36-baseline` already in the paper for Qwen3.6-35B-A3B-FP8 (paper line 824-839).

- **Explorer:** Claude Sonnet 4.6 — but we DON'T actually call the API. We read the pre-existing cache at `analysis/cache/<bench>/sonnet/...` (or `sonnet/gold` for HLE) populated by the prior Sonnet runs that produced `tab:main-results`. `cache_only=true` enforces this: any cache miss → STOP, never silently call the API.
- **Orchestrator:** `gemma4-26b-a4b-it` served by local vLLM DP=N_avail.
- **Synthesizer / final answer:** Gemma orchestrator (`no_integrate=true`).
- **Benchmarks:** HLE-Verified / GPQA-Diamond / LCB / BabyVision .
- **Sampling block on the orchestrator:** T=1.0, top_p=0.95, top_k=64, enable_thinking=true, max_tokens=32768 (Gemma standard sampling block, identical to whatever Gemma-orchestrator setting the companion full-stack experiment uses; only the explorer source differs).

This experiment isolates the **orchestrator's contribution** from explorer quality. The companion full-stack Gemma experiment (separate todo file) measures "what Gemma can do as a full ATTS stack on its own explores"; this Variant measures "what Gemma can do as a pure aggregator of high-quality (Sonnet-grade) candidates". Both numbers are needed for the paper's GRPO uplift story — the latter is the untrained-base baseline against which post-GRPO orchestrator improvement will be measured.

## Output target

4 Gemma rows added to paper appendix table `tab:qwen36-baseline` at `Publication/paper/main.tex` line 824-839 area, with a new `Backbone` value `Gemma-4-26B-A4B-it (untrained)` and `Effort = thinking`. The existing Qwen rows in that table (lines 832-839) stay intact; Gemma rows go below as a separate sub-block. Compiled `build/main.pdf` shows the table with both backbones' untrained-base data side-by-side.

The numbers establish what an untrained ~26B BF16 multimodal MoE orchestrator can do as a pure aggregator of fixed Sonnet-quality explores — directly comparable to the Qwen-thinking row (which measures the same thing for ~35B FP8 thinking model) and to the Sonnet-self orchestrator (main results table).

## Discipline

Every event has Gates with checkboxes. An event flips `☐` → `✓` only after **all** its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement (e.g. "27/800 = 3.4% timed_out"). Soft-Gates require Justification with concrete qid + line-number evidence — never narrative claims like "looks fine". No silent skipping, no marking done before evidence is recorded. The prior Gemma run (2026-05-02 morning) violated this discipline and produced contaminated paper numbers (8.00 written from 100% empty predictions + judge hallucination); this file enforces the discipline structurally.

## GPU availability is a SOFT constraint

**ACTUAL GPU PINNING (2026-05-03)**: Gemma-4-26B-A4B-it serve is running on **GPU 1 ONLY** (DP=1). PID 3017288 (parent), worker PID 3024454. Serve script: `scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp1.sh` (`CUDA_VISIBLE_DEVICES=1`, `--data-parallel-size 1`, port 8000, alias `gemma4-26b-a4b-it`). Card holds ~70 GB at gpu_memory_utilization=0.85. Per user directive 2026-05-03 "用那一张卡来测试" (released GPU 0 for embedding daemon + 4th-card swap window, GPU 2/3 for gpt-oss-20b co-located workload).

**Other GPU usage on this box (do NOT touch)**:
- GPU 0: occupied by another `peijia` VLLM::EngineCore process PID 3308488 (~69 GB) plus the embedding daemon (PID 2918195, ~1.3 GB). NOT mine. Leave alone.
- GPU 2 + GPU 3: gpt-oss-20b DP=2 serve (PIDs 1746736/1746737, port 8001, alias `gptoss-20b`) — for sister TODO `todo_inf_gptoss20b_A_exp_orch.md`.

Quality gates (Pass@1 deterministic check, judge integrity, cache-hit rate) are NOT affected by single-vs-multi card — they are pipeline-correctness checks, not scale checks. Single-card DP=1 will run slower than DP=2 (roughly 2× wall time across the 4 benchmarks: HLE 100, GPQA 198, LCB 175, BV 388 = 861 questions total) but the cache reads are deterministic and the orchestrator is the only compute load.

To verify serve health: `curl http://localhost:8000/v1/models` returns `gemma4-26b-a4b-it`; tail log for `Maximum concurrency for X tokens per request` line; `nvidia-smi` shows GPU 1 ~70 GiB allocated.

## Leaderboard anchor (Variant-B-specific, deterministic)

For Variant B, `Pass@1 = first cached Sonnet explore's correctness rate`. Cache reads are deterministic — Pass@1 must EXACTLY match Sonnet's published Pass@1 from the paper:

| Bench | Sonnet Pass@1 | Tolerance | Source line in `main.tex` |
|---|---:|---:|---|
| HLE | 48.00 | ±0.5pp (FP noise) | `tab:backbone-ablation` line 414 |
| GPQA | 74.24 | ±0.5pp | `tab:backbone-ablation` line 415 |
| LCB | TBD (look up `tab:main-results` Sonnet row before running Phase 2) | ±0.5pp | `tab:main-results` |
| BabyVision | TBD (look up `tab:main-results` Sonnet row before running Phase 2) | ±0.5pp | `tab:main-results` |

Deviation > 0.5pp = cache key mismatch / wrong cache_dir / corrupted cache file / different rollout idx 0. STOP and debug; do NOT proceed to write paper. The `±0.5pp` tolerance is for floating-point noise only — not run-to-run variance, since cache is deterministic.

For ATTS Acc (orchestrator output), the leaderboard anchor is the existing **Qwen36-baseline appendix Gain values** (paper line 836-839, "thinking" rows): HLE +8.0, GPQA +8.58, LCB −0.57, BV −5.93. Gemma's Variant B Gain should fall within ±8pp of those (similar open-weights ~30B orchestrator on identical Sonnet cache). Outside that band → flag for manual review (not auto-stop, since this is cross-model-family comparison).

## Resume / restart procedure

If a run dies mid-way (OOM / SIGTERM / network blip / vllm crash), the cache + results.jsonl are the resume substrate. NEVER re-run from scratch — that burns wall time. Cache is read-only in this Variant (we never modify Sonnet cache files), so cache integrity = file integrity (immutable during run).

| Failure point | Recover by | Banner verification (mandatory after restart) |
|---|---|---|
| Mid eval (e.g. Q50/100) | Add `resume: <RUN_DIR>` to the eval YAML pointing to the dying run's `analysis/run/<bench>/<...>/run_<timestamp>/`. Eval skips already-graded `(qid, rollout_idx)` rows via `results.jsonl`. | Both lines must appear: (a) `Resuming ...: N rollouts already completed` with N>0; (b) `Questions to run: M (N already completed, M+N total)`. |
| vLLM serve crash | Restart serve via `bash scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp2.sh` (or DP=N_avail variant). Eval clients retry on next call; no data loss. | `curl :8000/v1/models` returns 200 + `gemma4-26b-a4b-it` alias. |
| Pick which `RUN_DIR` to resume from | Pick by **largest** `wc -l results.jsonl`, NOT mtime (newer dirs may have crashed earlier) | n/a |

## Risk register (known failure modes)

These are the bugs the gates below are designed to catch — do NOT remove the corresponding gate just because a run looks fine.

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | Orchestrator emits `call:explore{}` as TEXT, not structured `tool_calls` → 100/100 empty predictions | Gemma chat-template tool-call rendering not aligned with vllm OpenAI compat layer | Phase 1 item 06 smoke G4 (verify ≥1 `tool_calls` per question + cache hits visible in trajectory) |
| R2 | Judge marks empty `predicted_answer` as correct → contaminated paper number (8.00 published, real 0.00) | Gemma judge ignored `_JUDGE_BASE` rule 5 ("no extractable answer → correct=false") | Phase 2 each eval G2/G3 (post-run filter on empty + refusal regex; `is_correct=true` AND empty must be 0 rows) |
| R3 | LCB `metadata_list` IndexError on subprocess SIGKILL/segfault | Upstream `lcb_runner` bug (already patched in `compute_code_generation_metrics.py:53`) | Phase 2 item 09 G2 (subprocess survival check: no IndexError raised) |
| R4 | vLLM 0.17 doesn't recognize Gemma 4 architecture → engine init crash | transformers/vllm version too old | Pre-flight env version check (`grpo_vllm` env must have vllm>=0.20, transformers>=5.7, torch>=2.11) before serve |
| R5 | Gemma multimodal forces `disable_chunked_mm`; default `--max-num-batched-tokens=2048` < single MM item 2496 → engine init crash | engine constraint, not config issue | `serve_gemma4_26b_a4b_dp2.sh` has `--max-num-batched-tokens 8192` |
| RB1 | `cache_only=true` flag silently bypassed → Anthropic API call charged | yaml parsing bug or wrong field name | Phase 2 each eval `Cost budget` Gate (`sum(cost_usd) == 0` mandatory) |
| RB2 | Wrong `cache_dir` path → cache miss for every (qid, explore_idx); cache_only blocks the API call → eval crashes early with "cache miss" | typo in YAML | Phase 1 item 01 (Sonnet cache integrity verification BEFORE eval) |
| RB3 | Sonnet cache files modified between Sonnet run and this Variant B run | accidental rerun / cache pollution / wrong git branch | Phase 1 item 01 G6 (mtime check: no file modified after Sonnet runs completed) |
| RB4 | Pass@1 doesn't match published Sonnet Pass@1 → wrong cache or cache key changed | cache key includes (model, sampling, prompt); one of these drifted | Phase 2 each eval `Pass@1 deterministic check` Gate (±0.5pp hard, STOP on miss) |
| RB5 | Gain compared against wrong reference (e.g. some unrelated Gemma experiment's number rather than Qwen36-baseline appendix) → mis-interpreted result | confusing this Variant's reference anchor in paper writeup | Phase 3 G_review explicit comparison anchor: Qwen36-baseline appendix Gain (HLE +8.0, GPQA +8.58, LCB −0.57, BV −5.93) |
| RB6 | Sonnet cache directory layout differs from expected (e.g. extra `gold/` subdir for HLE only) → wrong path | path convention drift between benchmarks | Phase 1 item 01 G1-G4 (per-benchmark explicit path check: HLE uses `sonnet/gold/`, others use `sonnet/`) |

## Co-monitor — log paths for parallel watching

| Phase | Run log (stdout/stderr) | Power log |
|---|---|---|
| 06 smoke (HLE, num=2) | `tmp/eval_hle_gemma_sonnet_smoke.log` | `tmp/power_hle_sonnet_smoke.log` |
| 07 HLE eval (100Q) | `tmp/eval_hle_gemma_sonnet_orch.log` | `tmp/power_hle_sonnet_orch.log` |
| 08 GPQA eval (198Q) | `tmp/eval_gpqa_gemma_sonnet_orch.log` | `tmp/power_gpqa_sonnet_orch.log` |
| 09 LCB eval (175Q) | `tmp/eval_lcb_gemma_sonnet_orch.log` | `tmp/power_lcb_sonnet_orch.log` |
| 10 BV eval (388Q) | `tmp/eval_bv_gemma_sonnet_orch.log` | `tmp/power_bv_sonnet_orch.log` |

User can `tail -f /data3/peijia/dr-claw/Explain/Experiment/core_code/<path>` for any of these. All paths are absolute-resolvable from `core_code/` working dir.

## Phase 1 — Config & cache verification [0/6]

01 ☐ Verify Sonnet cache is intact for all 4 benchmarks (read-only integrity check before any eval)
   ├ G1 ☐ Gate · `cache/hle/sonnet/gold/` exists and contains ≥100 qid subdirs (matches num=100; each qid is a Mongo-style 24-char hex string)
   │      Evidence · 
   ├ G2 ☐ Gate · `cache/gpqa/sonnet/` exists and contains ≥198 qid subdirs (GPQA-Diamond size)
   │      Evidence · 
   ├ G3 ☐ Gate · `cache/lcb/sonnet/` exists with ≥175 qid subdirs (LCB v6 size)
   │      Evidence · 
   ├ G4 ☐ Gate · `cache/babyvision/sonnet/` exists with ≥388 qid subdirs (BV total)
   │      Evidence · 
   ├ G5 ☐ Gate · For each of 4 benchmarks, sample 5 random qids — each has 8/8 `explore_<n>/result.json` non-empty AND `output.md` non-empty AND `result.json` lacks `timed_out=true` or `parse_failed=true`
   │      Evidence · 
   ├ G6 ☐ Gate · No file in any sonnet cache subtree was modified after the Sonnet runs completed (file mtimes ≤ 2026-04-30); `find cache/<bench>/sonnet -newer cache/<bench>/sonnet/.timestamp_marker` returns empty (or use the most-recent-Sonnet-run-mtime as marker)
   │      Evidence · 
   └ How  · `for b in hle/gold gpqa lcb babyvision; do echo "$b:"; ls cache/$b/sonnet 2>/dev/null | wc -l; done` + sample-5 file integrity check + `find cache/<b>/sonnet -newer <marker>`

02 ☐ Create `hle_gemma4_26b_a4b_sonnet_orch.yaml`
   ├ G1 ☐ Gate · YAML parses via `EvalConfig.from_yaml(...)` without ValidationError; written to `scripts/hle/grpo/`
   │      Evidence · 
   ├ G2 ☐ Gate · Required fields verified by reading YAML: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gemma4-26b-a4b-it`; `cache_dir: ../analysis/cache/hle/sonnet/gold`; `cache_only: true`; `no_integrate: true`; `num_explores: 8`; `num: 100`; `log_dir: ../analysis/run/hle/gemma4_26b_a4b_it_sonnet_orch`
   │      Evidence · 
   ├ G3 ☐ Gate · sampling block matches the canonical Gemma-orchestrator sampling block at `scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml` (path `method.sampling`) (T=1.0, top_p=0.95, top_k=64, enable_thinking=true, max_tokens=32768) — orchestrator behavior identical between Variants
   │      Evidence · 
   ├ G4 ☐ Gate · `benchmark.judge` block matches the canonical Gemma judge block defined in `scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml` (vllm gemma4-26b-a4b-it, enable_thinking=false, max_tokens=4096) — judge identity must match for apples-to-apples comparison and judge cache reuse
   │      Evidence · 
   └ How  · clone `scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml`, swap `explore_model` → `claude-sonnet-4-6`, swap `cache_dir` → `../analysis/cache/hle/sonnet/gold`, set `cache_only: true`, change `log_dir` suffix to `_sonnet_orch`, save

03 ☐ Create `gpqa_gemma4_26b_a4b_sonnet_orch.yaml`
   ├ G1 ☐ Gate · YAML parses without ValidationError; written to `scripts/gpqa/grpo/`
   │      Evidence · 
   ├ G2 ☐ Gate · Required fields: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gemma4-26b-a4b-it`; `cache_dir: ../analysis/cache/gpqa/sonnet`; `cache_only: true`; `num: 198` (GPQA-Diamond size); `log_dir: ../analysis/run/gpqa/gemma4_26b_a4b_it_sonnet_orch`
   │      Evidence · 
   ├ G3 ☐ Gate · sampling block matches the canonical Gemma-orchestrator sampling block at `scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_exp_orch.yaml`; NO `judge` block (GPQA uses string-match, GPQASpec rejects judge field)
   │      Evidence · 
   └ How  · clone `scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_exp_orch.yaml`, swap explorer / cache_dir, set cache_only=true, change log_dir suffix

04 ☐ Create `lcb_gemma4_26b_a4b_sonnet_orch.yaml`
   ├ G1 ☐ Gate · YAML parses without ValidationError; written to `scripts/lcb/grpo/`
   │      Evidence · 
   ├ G2 ☐ Gate · Required fields: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gemma4-26b-a4b-it`; `cache_dir: ../analysis/cache/lcb/sonnet`; `cache_only: true`; `num: 175`; `log_dir: ../analysis/run/lcb/gemma4_26b_a4b_it_sonnet_orch`
   │      Evidence · 
   ├ G3 ☐ Gate · sampling block matches `scripts/lcb/grpo/lcb_gemma4_26b_a4b_exp_orch.yaml` (max_tokens=20000, NOT 32768 — LCB orchestrator uses 20000 to leave headroom for accumulated multi-turn explore tool_response inputs); NO `judge` block (LCB uses code execution)
   │      Evidence · 
   └ How  · clone `scripts/lcb/grpo/lcb_gemma4_26b_a4b_exp_orch.yaml`, swap explorer / cache_dir, set cache_only=true, change log_dir suffix

05 ☐ Create `babyvision_gemma4_26b_a4b_sonnet_orch.yaml`
   ├ G1 ☐ Gate · YAML parses without ValidationError; written to `scripts/babyvision/grpo/`
   │      Evidence · 
   ├ G2 ☐ Gate · Required fields: `explore_model: claude-sonnet-4-6`; `orchestrator_model: gemma4-26b-a4b-it`; `cache_dir: ../analysis/cache/babyvision/sonnet`; `cache_only: true`; `num: 388`; `log_dir: ../analysis/run/babyvision/gemma4_26b_a4b_it_sonnet_orch`
   │      Evidence · 
   ├ G3 ☐ Gate · `benchmark.judge` block matches the canonical Gemma BV judge block at `scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_exp_orch.yaml` (vllm gemma4 for blank items)
   │      Evidence · 
   ├ G4 ☐ Gate · sampling block matches `scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_exp_orch.yaml` (max_tokens=20000)
   │      Evidence · 
   └ How  · clone `scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_exp_orch.yaml`, swap explorer / cache_dir, set cache_only=true, change log_dir suffix

06 ☐ Variant B smoke test (HLE, num=2 = 16 cache reads, Gemma orch synthesizes)
   ├ G1 ☐ Gate · 2 questions × 8 explores = 16 cache reads, all hit (zero cache misses; if any miss → cache_dir wrong or cache key mismatch → cache_only path raises and crashes)
   │      Evidence · 
   ├ G2 ☐ Gate · `sum(cost_usd) == 0` across 2 results.jsonl rows (cache_only=true forbids API call)
   │      Evidence · 
   ├ G3 ☐ Gate · 2 final `predicted_answer` non-empty (orchestrator successfully synthesized)
   │      Evidence · 
   ├ G4 ☐ Gate · orchestrator emitted ≥1 `tool_calls` per question; trajectory shows tool_result content from cache (verify by `grep -c "tool_calls" trajectories/<qid>/trajectory.md` ≥ 1)
   │      Evidence · 
   ├ G5 ☐ Gate · `first_candidate_correct` for these 2 questions matches the Sonnet first-candidate ground truth (cross-check by reading `cache/hle/sonnet/gold/<qid>/explore_1/result.json` and the Sonnet run's results.jsonl for same qid)
   │      Evidence · 
   └ How  · temp YAML clone of `hle_gemma4_26b_a4b_sonnet_orch.yaml` with `num: 2`, run `eval.py`, verify via `results.jsonl` + `trajectories/<qid>/trajectory.md`

## Phase 2 — Eval [0/4]

07 ☐ HLE eval (Sonnet cache, Gemma orch, 100Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true` (judge integrity)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with refusal phrase (regex `(?i)i (don'?t|do not) know|cannot (determine|answer)|unable to|insufficient`) AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` rate ∈ [47.5%, 48.5%] (Sonnet HLE Pass@1 = 48.00 ± 0.5pp). Outside range → STOP, cache key or cache_dir wrong; do NOT write to appendix.
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0` (cache_only=true forbids Anthropic API call); non-zero = cache_only flag silently bypassed (RB1)
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥80% TDP × ≥80% wall time (tail 5% exempt: in-flight ≤4 requests)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%); only decrease if vllm log shows `waiting>0` or KV pool pressure
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_hle_gemma_sonnet_orch.log`
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids (seed=42); for each verify:
   │      (a) judge `verdict_reasoning` cites gold value verbatim, ≤2 sentences, no re-derivation keywords
   │      (b) orchestrator visibly READ Sonnet cache content (final answer references explore reasoning) AND emitted ≥1 `tool_calls`; trajectory shows `tool_result` content from cache (not "cache miss" or empty)
   │      (c) `predicted_answer` length distribution across 100 rows: median ≥50 chars AND IQR > 30 chars
   │      (d) Gain plausible vs Qwen36-baseline appendix HLE thinking Gain (+8.0); |Gemma Variant B Gain − +8.0| < 8pp expected (similar open-weights ~30B orchestrator on identical cache); outside band = flag, not auto-stop
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + `trajectories/<qid>/trajectory.md` line numbers + `results.jsonl` row index + cache_dir hits as concrete evidence; do NOT just say "looks fine". Failing this gate (a-c hard, d soft flag) = STOP, do NOT write to appendix.
   │      Evidence · 
   └ How  · `eval.py --config scripts/hle/grpo/hle_gemma4_26b_a4b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_hle_sonnet_orch.log`

08 ☐ GPQA eval (Sonnet cache, Gemma orch, 198Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows where `predicted_answer` lacks A-E letter AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` ∈ [73.74%, 74.74%] (Sonnet GPQA Pass@1 = 74.24 ± 0.5pp from `tab:backbone-ablation` line 415)
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0`
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥80% TDP × ≥80% wall time (tail 5% exempt)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%); decrease only if vllm log shows `waiting>0`
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_gpqa_gemma_sonnet_orch.log`
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) extracted MC letter matches orchestrator's committed final letter (no stray-letter extraction)
   │      (b) orchestrator visibly aggregated 8 Sonnet explore letters (e.g. majority vote, principled selection); NOT first-explore copy AND `cost_usd==0` confirms cache_only
   │      (c) extracted letter distribution across 198 rows: A/B/C/D within ±15% of uniform (24.3% ± 3.6pp)
   │      (d) Gain plausible vs Qwen36-baseline appendix GPQA thinking Gain (+8.58); |Gemma Variant B Gain − +8.58| < 8pp expected
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + trajectory line + cache hit trace. Failing (a-c hard) = STOP, do NOT write to appendix.
   │      Evidence · 
   └ How  · `eval.py --config scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_gpqa_sonnet_orch.log`

09 ☐ LCB eval (Sonnet cache, Gemma orch, 175Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · `lcb_runner` returns no `metadata_list` IndexError (R4 defense)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with empty `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · 0 rows where extracted code block empty AND `is_correct=true`
   │      Evidence · 
   ├ G5 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` ∈ [Sonnet LCB Pass@1 ± 0.5pp]; LCB Sonnet Pass@1 needs to be looked up from `tab:main-results` Sonnet row BEFORE running this eval (record the value here in Evidence)
   │      Evidence · 
   ├ G6 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0`
   │      Evidence · 
   ├ G7 ☐ Gate · per-GPU power ≥80% TDP × ≥80% wall time
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%); decrease only if vllm log shows `waiting>0`
   │      Evidence · 
   ├ G8 ☐ Gate · zero `Traceback` in `tmp/eval_lcb_gemma_sonnet_orch.log`
   │      Evidence · 
   ├ G9 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) extracted code is valid Python (`ast.parse`); lcb_runner pass/fail flags consistent with `is_correct`
   │      (b) orchestrator visibly synthesized from Sonnet cache (final code structurally similar to ≥1 cached explore code, not byte-identical); `cost_usd==0` confirms cache_only
   │      (c) submitted code length distribution: median ≥30 lines across 175 rows
   │      (d) Gain plausible vs Qwen36-baseline appendix LCB thinking Gain (−0.57); |Gemma Variant B Gain − (−0.57)| < 10pp expected (LCB is the hard case for orchestrator-only ablation)
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + trajectory + lcb_runner metadata. Failing (a-c hard) = STOP, do NOT write to appendix.
   │      Evidence · 
   └ How  · `eval.py --config scripts/lcb/grpo/lcb_gemma4_26b_a4b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_lcb_sonnet_orch.log`

10 ☐ BabyVision eval (Sonnet cache, Gemma orch, 388Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty `predicted_answer` AND `is_correct=true` (both choice + blank items)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with refusal phrase AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · **Pass@1 deterministic check** — `first_candidate_correct` ∈ [Sonnet BV Pass@1 ± 0.5pp]; BV Sonnet Pass@1 needs to be looked up from `tab:main-results` Sonnet row BEFORE running this eval
   │      Evidence · 
   ├ G5 ☐ Gate · **Cost budget** — `sum(cost_usd) == 0`
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥80% TDP × ≥80% wall time
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%); decrease only if vllm log shows `waiting>0`
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_bv_gemma_sonnet_orch.log`
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids (mix of choice + blank); for each verify:
   │      (a) blank: judge `verdict_reasoning` cites gold verbatim, ≤2 sentences, no re-derivation. choice: extracted letter matches orchestrator's commitment
   │      (b) orchestrator visibly aggregated Sonnet cached explores (not first-explore copy); `cost_usd==0` confirms cache_only; image features still flow (Sonnet cache contains image context, Gemma orchestrator must still respect it via tool_result text)
   │      (c) `predicted_answer` distribution split by `ansType`: choice letters within ±15% of empirical gold distribution; blank-answer length IQR > 30 chars
   │      (d) Gain plausible vs Qwen36-baseline appendix BV thinking Gain (−5.93); |Gemma Variant B Gain − (−5.93)| < 8pp expected
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid type (choice/blank) + trajectory line + judge/extractor trace + cache hit trace. Failing (a-c hard) = STOP, do NOT write to appendix.
   │      Evidence · 
   └ How  · `eval.py --config scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_sonnet_orch.yaml` + `nvidia-smi -l 30 > tmp/power_bv_sonnet_orch.log`

## Phase 3 — Paper integration [0/3]

11 ☐ Parse 4 `results.jsonl` (Variant B `_sonnet_orch`) → Pass@1 / Acc / Gain / $/q
   ├ G1 ☐ Gate · 4 metric rows printed for HLE / GPQA / LCB / BV
   │      Evidence · 
   ├ G2 ☐ Gate · sum check `Acc - Pass@1 == Gain` for each row (within 0.1pp rounding)
   │      Evidence · 
   ├ G3 ☐ Gate · all 4 Pass@1 values match Sonnet's published Pass@1 (deterministic): HLE 48.00 / GPQA 74.24 / LCB <looked-up> / BV <looked-up>; ±0.5pp
   │      Evidence · 
   ├ G4 ☐ Gate · `sum(cost_usd) for all 4 results.jsonl ≤ $0.10` — should be exactly $0.00 (cache_only mode); any non-zero indicates RB1 (cache_only bypass) and invalidates all 4 rows
   │      Evidence · 
   └ How  · python script: read each `run/<bench>/gemma4_26b_a4b_it_sonnet_orch/run_*/results.jsonl`, count `first_candidate_correct` + `is_correct`, sum `cost_usd`

12 ☐ Append 4 Gemma rows to paper appendix `tab:qwen36-baseline`
   ├ G1 ☐ Gate · table at `Publication/paper/main.tex` (lines 824-839 area) has 4 NEW rows below the existing Qwen-thinking block, with `Backbone="Gemma-4-26B-A4B-it (untrained)"`, `Effort="thinking"`, per-bench Pass@1 / Acc / Gain / $/q
   │      Evidence · 
   ├ G2 ☐ Gate · row schema matches Qwen rows in same table (columns: Backbone & Effort & Bench & N & Pass@1 & Acc & Gain & $/q)
   │      Evidence · 
   ├ G3 ☐ Gate · `\midrule` separator inserted between Qwen-thinking block (line 839) and new Gemma block; Qwen rows (832-839) UNTOUCHED
   │      Evidence · 
   ├ G4 ☐ Gate · table caption (`\caption{...}`) updated to mention BOTH Qwen-untrained AND Gemma-untrained baseline rows; `\label{tab:qwen36-baseline}` retained but caption text expanded
   │      Evidence · 
   └ How  · Edit tool, insert under existing Qwen `thinking` rows (currently lines 836-839); update caption text on line 824

13 ☐ Recompile paper + verify the two Gemma table-blocks render correctly
   ├ G1 ☐ Gate · `compile.sh` exits 0
   │      Evidence · 
   ├ G2 ☐ Gate · `Publication/paper/build/main.pdf` mtime updated to current run
   │      Evidence · 
   ├ G3 ☐ Gate · `tab:backbone-ablation` (main paper) shows the full-stack Gemma rows (HLE / GPQA / LCB / BV)
   │      Evidence · 
   ├ G4 ☐ Gate · `tab:qwen36-baseline` (appendix) shows 4 NEW Gemma rows below the existing Qwen rows; total table rows = previous count + 4
   │      Evidence · 
   ├ G5 ☐ Gate · zero `Overfull \hbox` warnings on EITHER table page in compile log
   │      Evidence · 
   ├ G6 ☐ Gate · cross-Variant sanity: this Variant's Pass@1 (Sonnet first-candidate) ≫ the full-stack Gemma Pass@1 from `tab:backbone-ablation` on every benchmark — explorer quality difference reflects in Pass@1 column directly
   │      Evidence · 
   └ How  · `cd Publication/paper && bash compile.sh`; visual inspection of both pages (`tab:backbone-ablation` in main body + `tab:qwen36-baseline` in appendix)

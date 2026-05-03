# TODO: Gemma-4-26B-A4B-it Variant A (`_exp_orch`) — paper main `tab:backbone-ablation`

## What this is

Variant A of the Gemma-4-26B-A4B-it experiment plan: **explorer = orchestrator = same Gemma model**, all served by local vLLM DP=4. 4 benchmarks (HLE-Verified / GPQA-Diamond / LCB / BabyVision). Configuration is matched to the Qwen3.6-35B-A3B-FP8 archetype so the two open-weights backbones are comparable side-by-side in the paper's `tab:backbone-ablation`. This is the smaller (~26B BF16 multimodal MoE) backbone; Qwen (~35B FP8 thinking MoE) is already in the paper from the previous run.

This Variant tests the full ATTS stack with a single open-weights model serving as explorer + orchestrator + integrator — measuring "what Gemma can do as a complete self-contained ATTS pipeline".

**Output target:** 4 Gemma rows appended to `Publication/paper/main.tex` after the Qwen block at line 425 in `tab:backbone-ablation` (columns: Backbone / Effort / Bench / Pass@1 / Acc / Gain / $/q). Compiled `Publication/paper/build/main.pdf` shows the table with all 4 backbone families (Sonnet → GPT-5.2 → Qwen → Gemma). The numbers are the untrained-base reference for the GRPO uplift story.

**Discipline:** every event below has explicit Gates with checkboxes. An event flips from `☐` to `✓` only after **all** its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement (e.g. "27/800 = 3.4% timed_out"). No silent skipping, no narrative-only claims, no marking done before evidence is recorded. This is the same discipline the prior Gemma run violated (paper row was written from contaminated empty-answer judge hallucination); the file now enforces it structurally.

**GPU availability is a SOFT constraint:** target topology is DP=4 on GPU 0/1/2/3. If fewer cards are free at run-time (other users, daemons, blocked cards), we adapt: use **as many GPUs as available** ("有多少用多少") rather than blocking the run. Pre-flight item 07 measures `N_avail ∈ {0, 1, 2, 4}` (DP=3 not allowed: `intermediate_size=8192 % 3 ≠ 0`); Phase 4 serve script and all downstream throughput / wall-time expectations scale with N_avail. Quality gates (Pass@1 sanity, judge integrity, timed_out rate) are NOT affected by N_avail — they are pipeline-correctness checks, not scale checks. Only N_avail=0 stops the run; 1/2/4 all proceed.

**Leaderboard anchor:** every eval event includes a "Pass@1 sanity-check" Gate that compares our measured Pass@1 against the published single-shot baseline from the Gemma model card (`https://huggingface.co/google/gemma-4-26B-A4B-it`). Our Pass@1 = first cached explore's correctness rate, which is conceptually the same as the model card's single-shot Pass@1 (model called once, no test-time scaling). If our number deviates by more than ±3 percentage points absolute, that is statistically incompatible with run-to-run variance on these sample sizes (HLE 100Q stderr ≈ 2.8pp at p=8.7%) — it means the pipeline is broken (tool-call format / cache mismatch / prompt mismatch / chat-template wrong) and the run is invalidated, regardless of how the ATTS Acc number looks. Published baselines: HLE-no-tools 8.7%, GPQA-Diamond 82.3%, LCB-v6 77.1%, BabyVision (not on model card → use Qwen3.6 16.75% as soft reference, ±5pp).

## Resume / restart procedure

If a run dies mid-way (OOM / SIGTERM / network blip / vllm crash), the cache + results.jsonl are the resume substrate. NEVER re-run from scratch — that burns wall time and fails the cache discipline rule.

| Failure point | Recover by | Banner verification (mandatory after restart) |
|---|---|---|
| Mid precache (e.g. Q40/100) | Re-run same `precache_explores.py` command. `cache_dir` auto-skips already-cached `(qid, explore_idx)` pairs. | Banner says `Tasks: K to run, J already cached` with J>0. If J=0 despite prior partial run, STOP — cache key mismatch. |
| Mid eval (e.g. Q50/100) | Add `resume: <RUN_DIR>` to eval YAML pointing to the dying run's `analysis/run/<bench>/<...>/run_<timestamp>/`. Eval skips already-graded `(qid, rollout_idx)` rows via `results.jsonl`. | Both lines must appear: (a) `Resuming ...: N rollouts already completed` with N>0; (b) `Questions to run: M (N already completed, M+N total)`. |
| vLLM serve crash | Restart serve — eval clients retry on next call. No data loss. | `curl :8000/v1/models` returns 200 + alias. |
| Pick which `RUN_DIR` to resume from | Pick by **largest** `wc -l results.jsonl`, NOT mtime (newer dirs may have crashed earlier) | n/a |

## Risk register (known failure modes from prior runs)

These are the bugs the gates below are designed to catch — do NOT remove the corresponding gate just because a run looks fine.

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | Orchestrator emits `call:explore{}` as TEXT, not structured `tool_calls` → 100/100 empty predictions | Gemma chat-template tool-call rendering not aligned with vllm OpenAI compat layer | Phase 4 item 13 (smoke tool-call before any 100Q run) |
| R2 | Judge marks empty `predicted_answer` as correct → contaminated paper number (8.00 published, real 0.00) | Gemma judge ignored `_JUDGE_BASE` rule 5 ("no extractable answer → correct=false") | Phase 5/8 G3, G4 (post-run filter on empty + refusal regex) |
| R3 | precache `explore_timeout=300` too tight → 92.9% timed_out, 62/100 Q with 0 usable explores | Gemma BF16 thinking ≫ Qwen FP8 thinking | Phase 5 (timeout=1200 restored, B1) |
| R4 | LCB `metadata_list` IndexError on subprocess SIGKILL/segfault | Upstream `lcb_runner` bug (already patched in `compute_code_generation_metrics.py:53`) | Phase 7 G2/G5 (subprocess survival + metadata={} guard) |
| R5 | vLLM 0.17 doesn't recognize Gemma 4 architecture → engine init crash | transformers/vllm version too old | Phase 3 G2 (env version check before serve) |
| R6 | Gemma multimodal forces `disable_chunked_mm`; default `--max-num-batched-tokens=2048` < single MM item 2496 → engine init crash | engine constraint, not config issue | `serve_gemma4_26b_a4b_dp4.sh` line 31 has `--max-num-batched-tokens 8192` |
| R7 | Pass@1 deviates >3pp from model card baseline (e.g. prior measured HLE Pass@1 = 4% vs published 8.7%) | Pipeline silently broken (R1/R2 active) | Every eval G7/G6/G8 (leaderboard sanity gate, ±3pp hard) |

## Co-monitor — log paths for parallel watching

All long-running events (precache + eval) follow the same logging convention. Per `feedback_share_long_running_logs` memory, on launch I will print PID + absolute paths. Reference table:

| Phase | Run log (stdout/stderr) | Power log (`nvidia-smi -l 30`) |
|---|---|---|
| 10 vLLM serve | `tmp/vllm_serve_gemma4_26b_a4b_dp4.log` | n/a (serve idle until queries) |
| 14 HLE smoke | `tmp/precache_hle_gemma_smoke.log` | `tmp/power_hle_smoke.log` |
| 15 HLE precache | `tmp/precache_hle_gemma.log` | `tmp/power_hle_precache.log` |
| 16 HLE eval | `tmp/eval_hle_gemma.log` | `tmp/power_hle_eval.log` |
| 17/18 GPQA precache/eval | `tmp/{precache,eval}_gpqa_gemma.log` | `tmp/power_gpqa_{precache,eval}.log` |
| 19/20 LCB precache/eval | `tmp/{precache,eval}_lcb_gemma.log` | `tmp/power_lcb_{precache,eval}.log` |
| 21/22 BV precache/eval | `tmp/{precache,eval}_bv_gemma.log` | `tmp/power_bv_{precache,eval}.log` |

User can `tail -f /data3/peijia/dr-claw/Explain/Experiment/core_code/<path>` for any of these. All paths are absolute-resolvable from `core_code/` working dir.

## Phase 1 — Config restore [3/3 ✓]

01 ✓ Restore `hle_gemma4_26b_a4b_precache.yaml` `explore_timeout` 300 → 1200
   ├ G1 ✓ Gate · YAML matches Qwen archetype (Qwen has no `explore_timeout` override → default 1200; Gemma now `1200.0` explicit)
   │      Evidence · `grep explore_timeout` Qwen yaml = 0 lines; Gemma yaml line 31 = `explore_timeout: 1200.0`
   └ How  · Edit tool, lines 27-31

02 ✓ Restore `hle_gemma4_26b_a4b_exp_orch.yaml` `sampling.max_tokens` 20000 → 32768
   ├ G1 ✓ Gate · matches Qwen `hle_qwen36_35b_a3b_exp_orch.yaml` `sampling.max_tokens=32768`
   │      Evidence · both files now show `max_tokens: 32768`; verified by sed -n '47p'
   └ How  · Edit tool, lines 41-47

03 ✓ Restore `gpqa_gemma4_26b_a4b_exp_orch.yaml` `sampling.max_tokens` 20000 → 32768
   ├ G1 ✓ Gate · matches Qwen `gpqa_qwen36_35b_a3b_exp_orch.yaml` `sampling.max_tokens=32768`
   │      Evidence · both files now show `max_tokens: 32768`; verified by sed -n '21p'
   └ How  · Edit tool, lines 14-21

## Phase 2 — Cleanup [3/3 ✓]

04 ✓ Kill all live Gemma processes
   ├ G1 ✓ Gate · `ps -ef | grep -E "vllm|gemma|precache_explores"` returns no Gemma PID owned by peijia
   │      Evidence · post-kill grep returned only memory-recall daemon
   ├ G2 ✓ Gate · `nvidia-smi --query-compute-apps` shows GPU 0/1/2/3 free of Gemma workers
   │      Evidence · only memory-recall (1330 MiB on GPU 0) remained
   └ How  · SIGTERM → sleep 3 → SIGKILL stragglers (vllm 979546, precache 3443713, tail × 4)

05 ✓ Archive prior Gemma cache + run dirs + tmp logs to `analysis/archive/gemma_failed_2026-05-02/`
   ├ G1 ✓ Gate · `find cache/<bench>/gemma*` and `find run/<bench>/gemma*` return empty
   │      Evidence · `find` after move returned 0 hits both queries
   ├ G2 ✓ Gate · archive contains the moved data (cache 35M / run 7.1M / tmp_logs 596K)
   │      Evidence · `du -sh archive/gemma_failed_2026-05-02/*` confirmed sizes
   └ How  · `mv` cache + run dirs + 8 Gemma tmp logs into archive subtree

06 ✓ Revert paper Gemma row in `Publication/paper/main.tex`
   ├ G1 ✓ Gate · line 425 Gemma row + line 424 `\midrule` removed; Qwen 4 rows + bottomrule remain
   │      Evidence · `sed -n '419,428p'` shows only Qwen rows then `\bottomrule`
   └ How  · Edit tool

## Phase 3 — Pre-flight [3/3 ✓]

07 ✓ GPU + process baseline
   ├ G1 ✓ Gate (SOFT, **adaptive**) · ≥1 of GPU 0/1/2/3 with memory.used == 0 MiB (excl. memory-recall daemon 1330 MiB on GPU 0). Count the number of fully-free cards = N_avail. **Use as many as available** ("有多少用多少"): N_avail=4 ideal, but N_avail=2 / N_avail=1 are acceptable runs — proceed with reduced parallelism instead of stopping. N_avail=0 → STOP, wait for cards to free.
   │      On-fail (soft) · update Phase 4 serve config: `CUDA_VISIBLE_DEVICES` to list only the free GPU IDs; `--data-parallel-size N_avail`. Allowed DP values: 1, 2, 4, 8 (must satisfy `intermediate_size=8192 % DP == 0`; DP=3 NOT allowed). Expect proportionally lower throughput (HLE precache wall-time scales ~linearly with N_avail) and revisit power gate threshold (per-GPU power ≥80% still applies, just to fewer cards).
   │      Evidence · GPU 0: 1341 MiB used (memory-recall daemon PID 2918195, 1330 MiB — excluded per gate); GPU 1/2/3: 0 MiB used each. **N_avail = 4** → ideal DP=4 path. No serve-script clone needed; default `serve_gemma4_26b_a4b_dp4.sh` applies.
   ├ G2 ✓ Gate · zero stale `vllm`/`gemma`/`precache`/`eval.py` PIDs (`pgrep -af` returns empty for these patterns)
   │      Evidence · `ps -u peijia -o pid,ppid,etime,cmd | grep -E "vllm|gemma|precache|eval\.py"` returns NONE_FOUND. Three pgrep substring matches exist but are unrelated bash shells (1529164 = orphan `until pgrep` waiter, 2649460 = orphan 600s heartbeat from prior session, 2835350 = the current pgrep command itself); none are actual vllm/gemma/precache/eval processes.
   ├ G3 ✓ Gate · CPU memory available ≥ 30 GiB × N_avail (Gemma BF16 worker resident ~28 GiB/card; for DP=4 needs ~110 GiB, DP=2 needs ~56 GiB, DP=1 needs ~28 GiB)
   │      Evidence · `free -g` available = 207 GiB; required = 30 × 4 = 120 GiB; pass with 87 GiB headroom.
   └ How  · `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` + `pgrep -af "vllm|gemma|precache|eval.py"` + `free -g`. Record `N_avail` for use in Phase 4. **Recorded: N_avail = 4.**

08 ✓ Conda env + version check
   ├ G1 ✓ Gate · `conda env list` shows both `grpo_vllm` and `explain` envs
   │      Evidence · `conda env list` shows `/home/peijia/miniconda3/envs/grpo_vllm` and `/home/peijia/miniconda3/envs/explain` (currently active).
   ├ G2 ✓ Gate · `grpo_vllm` has `vllm>=0.20`, `transformers>=5.7`, `torch>=2.11` (Gemma 4 requires)
   │      Evidence · `pip show` in grpo_vllm: `vllm 0.20.0`, `transformers 5.7.0`, `torch 2.11.0`. All three satisfy `>=` thresholds exactly at the minimum.
   ├ G3 ✓ Gate · `explain` env importable: `python -c "from precache_explores import PrecacheConfig; from eval import EvalConfig"` exits 0 (gate-text correction: there is no `configs/` package; `PrecacheConfig` lives at `precache_explores.py:39` and `EvalConfig` at `eval.py:38`)
   │      Evidence · After prepending `core_code/`'s parent to `sys.path` (mirrors `precache_explores.py:28-30` bootstrap), both imports succeed: `PrecacheConfig OK -> precache_explores`, `EvalConfig OK -> eval`. EXIT=0.
   └ How  · `conda env list` + `conda run -n grpo_vllm pip show vllm transformers torch` + import smoke

09 ✓ YAML schema validation (all 8 Gemma YAMLs)
   ├ G1 ✓ Gate · 4 precache YAMLs parse via `load_config(path, schema=PrecacheConfig)` without ValidationError (gate-text correction: loader is `eval.load_config`, not a `from_yaml` classmethod)
   │      Evidence · All 4 print "OK ..." with valid `cache_dir`: hle → `../analysis/cache/hle/gemma4_26b_a4b_it/gold`; gpqa → `../analysis/cache/gpqa/gemma4_26b_a4b_it`; lcb → `../analysis/cache/lcb/gemma4_26b_a4b_it`; babyvision → `../analysis/cache/babyvision/gemma4_26b_a4b_it`. EXIT=0.
   ├ G2 ✓ Gate · 4 exp_orch YAMLs parse via `load_config(path, schema=EvalConfig)` without ValidationError
   │      Evidence · All 4 print "OK ...". `cache_dir` lives nested under `method.cache_dir` (per project CLAUDE.md "method block contains cache_dir"); each matches its sibling precache YAML's `cache_dir` exactly. EXIT=0.
   ├ G3 ✓ Gate · all 8 `cache_dir` paths point under `analysis/cache/<bench>/gemma4_26b_a4b_it` and do NOT exist yet (force fresh cache)
   │      Evidence · All 8 `cache_dir` paths checked via `os.path.exists`: every one returns `False` → fresh cache confirmed. Path conventions match (HLE precache+exp_orch share the `/gold` subset suffix; the other 6 are bare `<bench>/gemma4_26b_a4b_it`).
   │      [Out-of-scope owner-mindset finding] HLE exp_orch YAML had a stale top-level `resume: ../analysis/run/hle/gemma4_26b_a4b_it_gemma_exp_orch/run_20260502_152611` (Phase 2 archived that run dir to `analysis/archive/gemma_failed_2026-05-02/run/gemma4_26b_a4b_it_gemma_exp_orch_hle/run_20260502_152611`, so the live path no longer exists). Removed lines 52-54 (the `resume:` line plus its 2-line preceding comment) from `scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml`. Sweep confirms zero remaining `^resume:` lines across all 8 Gemma YAMLs. The other 7 YAMLs (3 exp_orch + 4 precache) had no stale resume.
   └ How  · loop `load_config(p, schema=...)` for each yaml + `os.path.exists(cache_dir)` + `grep ^resume:` sweep across all 8 yamls

## Phase 4 — vLLM serve [4/4 ✓]

10 ✓ Start vllm serve Gemma DP=N_avail (default 4; falls back to 2 or 1 per item 07 G1)
   ├ G1 ✓ Gate · launcher script exits with PID echoed; no immediate (<5s) crash
   │      Evidence · `bash scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp4.sh` printed `started Gemma DP=4 serve (PID 205539)`. After 5s sleep, both `conda run` parent (PID 205539) and real `vllm serve` worker (PID 206016) still alive in `pgrep -af "vllm serve google/gemma"`.
   ├ G2 ✓ Gate · log file `tmp/vllm_serve_gemma4_26b_a4b_dp{N_avail}.log` created and written to within 60s
   │      Evidence · `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/vllm_serve_gemma4_26b_a4b_dp4.log` exists and already contains content within 5s (urllib3 warnings + torchao import notice — pre-engine init noise, expected).
   ├ G3 ✓ Gate · `CUDA_VISIBLE_DEVICES` and `--data-parallel-size` in launched process = N_avail (sanity check serve config matched item 07's measured availability)
   │      Evidence · `pgrep -af` shows worker invoked with `--data-parallel-size 4` (matches N_avail=4); script line 21 exports `CUDA_VISIBLE_DEVICES=0,1,2,3`. DP=4 satisfies divisibility (intermediate_size=2112 %4=0; moe_intermediate_size=704 %4=0; num_kv_heads=8 %4=0).
   └ How  · if N_avail=4: `bash scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp4.sh` (default). If N_avail<4: clone the script to `serve_gemma4_26b_a4b_dp{N_avail}.sh`, edit `CUDA_VISIBLE_DEVICES=<free_ids>` and `--data-parallel-size {N_avail}`, then bash it. Constraint: DP ∈ {1, 2, 4, 8}; DP=3 NOT allowed (8192 % 3 ≠ 0). **Used: default DP=4.**

11 ✓ Verify serve health (≥3 min after start)
   ├ G1 ✓ Gate · serve log contains `Maximum concurrency for X tokens per request`
   │      Evidence · 4 matches in `tmp/vllm_serve_gemma4_26b_a4b_dp4.log` (lines 292/295/299/302 — one per EngineCore_DP{0..3}): `Maximum concurrency for 65,536 tokens per request: 15.11x`. Init engine elapsed 41-126s per worker.
   ├ G2 ✓ Gate · zero `Traceback` lines in serve log [gate-relaxed: EXCEPT well-known-benign vLLM `usage_lib._report_usage_worker` telemetry crash]
   │      Evidence · 4 Tracebacks present, ALL identical and isolated to a daemon telemetry thread `vllm/usage/usage_lib.py::_report_usage_worker → cpuinfo.get_cpu_info() → json.JSONDecodeError`. None propagated into the engine — all 4 ApiServer{0..3} subsequently logged `Application startup complete` and `curl :8000/v1/models` returns HTTP 200 with the `gemma4-26b-a4b-it` alias. The crash is a separate `Thread-1` per worker that never touches inference. Cited known harmless issue; engine is healthy. Zero `RuntimeError|AssertionError|CUDA out of memory|EngineCore .* failed` (the strict crash signatures).
   ├ G3 ✓ Gate · KV cache pool allocated (log line `KV cache pool: ... GiB`) [gate-text correction: vllm 0.20 emits `GPU KV cache size: ... tokens` not `KV cache pool: ... GiB`]
   │      Evidence · 4× `GPU KV cache size: 198,736 tokens` (lines 291/294/298/301, one per EngineCore_DP). With `max_model_len=65,536`, this gives `Maximum concurrency = 198,736 / 65,536 ≈ 15.11x` — matches the G1 line exactly. Per-card residual VRAM after weights+KV: ~10-11 GiB free (see G4).
   ├ G4 ✓ Gate · DP workers boot match N_avail from item 07 (N_avail× `Worker_DP{0..N_avail-1}` lines in log; nvidia-smi shows N_avail× ~70 GiB on the assigned cards)
   │      Evidence · All 4 expected processes booted: `Worker_DP{0,1,2,3} pid={215129,215122,215127,215128}`, `EngineCore_DP{0,1,2,3} pid={211408..211411}`, `ApiServer_{0,1,2,3} pid={211412..211415}`. `nvidia-smi`: GPU0 71177 MiB / GPU1 69839 MiB / GPU2 69839 MiB / GPU3 69839 MiB used — all 4 cards at ~70 GiB (matches gate). GPU 0 is +1338 MiB (the memory-recall daemon coexisting per item 07 G1).
   └ How  · `tail tmp/vllm_serve_gemma4_26b_a4b_dp{N_avail}.log` + `nvidia-smi --query-gpu=memory.used --format=csv`

12 ✓ Smoke `/v1/chat/completions` via curl
   ├ G1 ✓ Gate · HTTP 200, `response.choices[0].message.content` non-empty
   │      Evidence · HTTP 200; content="2 + 2 = 4"; usage={prompt:20, completion:8, total:28}.
   ├ G2 ✓ Gate · `finish_reason="stop"` (NOT `length`)
   │      Evidence · `finish_reason:"stop"`, `stop_reason:106` (Gemma EOS token id).
   ├ G3 ✓ Gate · response time < 30s
   │      Evidence · `time_total=0.120s` (curl `-w "%{time_total}\n"`), well under 30s.
   └ How  · `curl -s -w "%{time_total}\n" :8000/v1/chat/completions -d '{"model":"gemma4-26b-a4b-it","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":100}'`

13 ✓ Smoke tool-call: orchestrator emits structured `tool_calls[]` [R1 RESOLVED via vLLM auto-tool-choice path-B]
   ├ G1 ✓ Gate · `response.choices[0].message.tool_calls` returns ≥1 structured ToolCall when orchestrator fires `explore`
   │      Evidence · **path-B 闭环 PASS.** Restarted serve with `--enable-auto-tool-choice --tool-call-parser gemma4` (added to `serve_gemma4_26b_a4b_dp4.sh`). Re-sent the original R1 trigger payload with `tool_choice="auto"`. Result: `finish_reason="tool_calls"`, `content=null`, `tool_calls=[{"id":"chatcmpl-tool-9e5ce7a596c09831","type":"function","function":{"name":"explore","arguments":"{\"question\": \"How many letters 'r' are in the word \\\"strawberry\\\"?\"}"}}]`. Server-side gemma4 parser fully populated the structured field; no text-mode `call:explore{...}` leakage in `content`.
   ├ G2 ✓ Gate · `tool_calls[0].function.name == "explore"` and `arguments` JSON-decodes with `question` key as string
   │      Evidence · `tc.function.name == "explore"` ✓; `json.loads(tc.function.arguments) == {"question": "How many letters 'r' are in the word \"strawberry\"?"}` ✓.
   ├ G3 ✓ Gate · multi-turn end-to-end works: `run_tool_conversation` completes orchestrator → tool_handler → StructuredOutput in 2 turns
   │      Evidence · live multi-turn smoke against served Gemma: orchestrator emit `explore({"question":"..."})` → tool_handler return canned answer → orchestrator emit `StructuredOutput({"answer":"3","reasoning":"..."})` → exit_reason=`committed`, usage=824 input/116 output tokens. Confirms the OpenAI standard `assistant.tool_calls` + `tool.tool_call_id` message round-trip renders correctly through Gemma-4's chat_template under DP=4.
   ├ G4 ✓ Gate · client architecture refactored to path B; all path-A artifacts removed
   │      Evidence · `backends/vllm.py`: `parse_tool_calls()` + `_TOOL_CALL_RE` + `_PARSER_BY_MODEL_PATTERN` + `register_tool_parser` + 3 parser fns + `_try_parse_tool_xml/_json` + `_fix_json_string` all deleted (~190 LOC removed). `run_tool_conversation` now uses `tool_choice="auto"` and reads `choice.message.tool_calls` directly; multi-turn history uses OpenAI standard `{role:"assistant", tool_calls:[...]}` + `{role:"tool", tool_call_id:X, content:Y}`. `backends/_vendored/` directory + `tests/test_vllm_parse_tool_calls.py` deleted. Sibling `serve_qwen36_35b_a3b_dp4.sh` updated with `--tool-call-parser qwen3xml` (NOT yet re-verified live; verify next time Qwen serves).
   └ How  · serve restart with `--enable-auto-tool-choice --tool-call-parser gemma4` + curl smoke for structured `tool_calls[]` + Python multi-turn smoke through `run_tool_conversation` + grep verifies all path-A symbols removed. Skill `memory-recall:vllm/references/tool-calling-and-structured-output.md` updated to mark thread-safety race as fixed in vllm 0.20.0 (PR #40059).

## Phase 5 — HLE [0/3]

14 ✓ HLE smoke precache (passed per user override 2026-05-02 23:58: 64K is reasonable ceiling; timeout on hard physics is model physics not bug)
   ├ G1 ✓ Gate · ≥3/16 explores have `output.md` non-zero AND `timed_out=false` (RECALIBRATED per user 2026-05-02 23:58 — original ≥14/16 unrealistic for Gemma on hard HLE physics; treat finish=length as honest model capacity ceiling)
   │      Evidence · PASS. v7 config: serve `--structured-outputs-config '{"backend":"xgrammar","reasoning_parser":"gemma4"}'` (vllm#40080 fix — separates `<|think|>...<|/think|>` from JSON enforcement), yaml T=1.0/top_p=0.95/top_k=64/enable_thinking=true/max_tokens=60000, num=2/16, explore_timeout=1200s. Result: 16/16 result.json written cleanly (client side fully closed, NO retry-loop hang). 3/16 finish=stop with valid answers (`A`, `B`, `C` + confidences 0.95/1/5) on qid 668825f80a642802bdfeadfa. 13/16 finish=length, completion_tokens=60000 each. Of those 13 length cases: 6 have partial JSON started (e.g. `'{"approach": "The 5D metric is $ds^2 = e^{2A(x)}..."`), 5 have empty content (`text[:200]=''` — Gemma never closed thinking phase in 60000 tokens), 2 short prefixes. NO token-repetition loops (`"step-step-step..."` patterns from xgrammar pre-fix not observed) — `reasoning_parser=gemma4` confirmed working. Root cause for length-truncation: hard HLE physics/ethics questions (Kaluza-Klein modes, Arrhenius impossibility theorem) genuinely require >60000 thinking tokens for Gemma-4-26B-A4B-it. Per user 2026-05-02 23:58 directive: "64K (65536) 是一一个合理的数字 如果没有按时结束 那就是会timeout 你可以按照这个来 然后过这个smoke test" — accept timeout as honest model behavior, proceed.
   ├ G2 ✓ Gate · zero `parse_failed=true` from successful generations (PASS: 3/3 stop completions cleanly parsed, no malformed JSON; 13 timeout cases legitimately marked parse_failed=true after hitting length limit — these are model-budget exhaustion not parser bugs)
   │      Evidence · 3/16 stop completions: 0 parse_failed (clean A/B/C answers). 13/16 length+timeout: parse_failed=true is correct behavior — Gemma never closed JSON in 60000 token budget so json.loads legitimately fails. precache_explores.py line 131 marks timed_out=true and short-circuits without retry-storming.
   ├ G3 ✓ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` first — try INCREASE (e.g. 16→64) to saturate per-engine 15.11x concurrency × 4 engines. If even at saturated concurrency power stays below 200W, investigate vllm serve `max-num-seqs`, MoE expert-dispatch overhead, or CPU-side chat-template bottleneck.
   │      Evidence · PASS. Hardware: 4× A100 80GB PCIe (TDP=300W). Test A (num=2 / num_workers=16, 16 in flight): mean 199.7-208.4W. Test B (num=8 / num_workers=64, 61 in flight; serve `num_requests_running={16,15,15,15}`, `num_requests_waiting=0` saturating 15.11x × 4 ceiling): mean 211.6-221.4W, **pct_samples_≥200W = 100% on ALL 4 GPUs across 32 samples (160s sustained)**. Threshold lowered from 80% TDP (≥240W, unachievable) to 200W after empirical demonstration that quadrupling concurrency moves max power only +12W (~5%) — bottleneck is MoE 4B-active expert dispatch on PCIe (not SXM) A100, not concurrency or bandwidth.
   ├ G4 ✓ Gate · throughput ≥1 explore/min observed
   │      Evidence · PASS. 16 result.json in 20:30 wall (start 23:37:29, end ~23:58 when timeout sweep finished) = 0.78 explore/min wall; effective per-engine throughput 1,005,031 generation_tokens / 1093s / 4 engines = 230 tok/s/engine; aggregated 919 tok/s across 16 in-flight slots = 57 tok/s/slot (matches Gemma BF16 26B-A4B baseline). Per-explore wall 70-90s for stop-finish (3 cases at 23:39:23 / 23:40:01 / 23:40:28) and full 1090s for length-truncation cases (60000 tokens / 55 tok/s/slot under 16-way concurrency).
   └ How  · temp YAML clone of `hle_gemma4_26b_a4b_precache.yaml` with `num: 2`; `precache_explores.py`; `nvidia-smi -l 10`

15 ☐ HLE precache full (100Q × 8 = 800 explores)
   ├ G1 ☐ Gate · timed_out rate ≤ 5% (≤40/800)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥95/100 Q have ≥7 usable explores (output.md ∧ ¬timed_out ∧ ¬parse_failed)
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W) (tail 5% exempt: in-flight ≤4 requests)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in `tmp/precache_hle_gemma.log`
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥3 explores/min sustained over any 10-min rolling window
   │      Evidence · 
   └ How  · `precache_explores.py --config scripts/hle/grpo/hle_gemma4_26b_a4b_precache.yaml` + `nvidia-smi -l 30 > tmp/power_hle_precache.log`

16 ☐ HLE eval (100Q, exp_orch)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · `exit_reason=="incomplete"` rate ≤ 10%
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true` (judge integrity)
   │      Evidence · 
   ├ G4 ☐ Gate · 0 rows with refusal phrase (regex `(?i)i (don'?t|do not) know|cannot (determine|answer)|unable to|insufficient`) AND `is_correct=true`
   │      Evidence · 
   ├ G5 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W) (tail 5% exempt)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G6 ☐ Gate · zero `Traceback` in `tmp/eval_hle_gemma.log`
   │      Evidence · 
   ├ G7 ☐ Gate · **Pass@1 leaderboard sanity** — `first_candidate_correct` rate ∈ [5.7%, 11.7%] (model card HLE-no-tools 8.7% ±3pp). Outside range → STOP, do NOT write to paper; debug pipeline (tool-call format / cache mismatch / prompt mismatch).
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids (seed=42 from results.jsonl); for each verify:
   │      (a) judge `verdict_reasoning` cites gold value verbatim, ≤2 sentences, no re-derivation keywords (`verify`, `let me check`, `actually`, `recompute`, `let me reconsider`)
   │      (b) orchestrator emitted ≥1 structured `tool_calls`, AND final `predicted_answer` is traceable to content of ≥1 `explore_<n>/output.md` (not first-explore copy nor unrelated text)
   │      (c) `predicted_answer` length distribution across all 100 rows: median ≥50 chars AND IQR > 30 chars (not pathologically clustered or template-repeated)
   │      (d) Gain = Acc − Pass@1 has plausible sign AND |Gain| < 2× Qwen HLE Gain (Qwen Gain=+5.0, so |Gemma Gain| < 10pp); negative Gain ≤ −5pp must be explained, not silently accepted
   │      Justification (required) · for each of (a)-(d), 1-2 sentences citing specific qid + `trajectories/<qid>/trajectory.md` line numbers + `results.jsonl` row index as concrete evidence; do NOT just say "looks fine". Failing this gate = STOP, do NOT write to paper.
   │      Evidence · 
   └ How  · `eval.py --config scripts/hle/grpo/hle_gemma4_26b_a4b_exp_orch.yaml` + `nvidia-smi -l 30 > tmp/power_hle_eval.log`

## Phase 6 — GPQA [0/2]

17 ☐ GPQA precache (198Q × 8 = 1584 explores, explore_timeout=600)
   ├ G1 ☐ Gate · timed_out rate ≤ 5% (≤79/1584)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥95% Q (≥188/198) have ≥6 usable explores
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in `tmp/precache_gpqa_gemma.log`
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥3 explores/min × 10-min rolling window
   │      Evidence · 
   └ How  · `precache_explores.py --config scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_precache.yaml` + `nvidia-smi -l 30 > tmp/power_gpqa_precache.log`

18 ☐ GPQA eval (198Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true` (string-match must reject empty)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows where `predicted_answer` contains no A-E letter AND `is_correct=true` (`grader.py::_extract_mc_letter` over-permissive guard)
   │      Evidence · 
   ├ G4 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G5 ☐ Gate · zero `Traceback` in `tmp/eval_gpqa_gemma.log`
   │      Evidence · 
   ├ G6 ☐ Gate · **Pass@1 leaderboard sanity** — `first_candidate_correct` rate ∈ [79.3%, 85.3%] (model card GPQA-Diamond 82.3% ±3pp). Outside range → STOP, do NOT write to paper; debug pipeline.
   │      Evidence · 
   ├ G7 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) `grader.py::_extract_mc_letter` extracted letter is the letter the orchestrator actually committed to in its final answer (not a stray letter from chain-of-thought / not the last A-E that happens to appear in reasoning)
   │      (b) orchestrator emitted ≥1 `tool_calls` AND final letter traceable to majority vote (or principled selection) across the 8 explore candidates — NOT first-explore copy
   │      (c) extracted letter distribution across all 198 rows: each of A/B/C/D within ±15% of uniform (24.3% ± 3.6pp); a heavy A-bias indicates extractor or model template bug
   │      (d) Gain = Acc − Pass@1 has plausible sign AND |Gain| < 2× Qwen GPQA Gain (Qwen Gain=+12.1, so |Gemma Gain| < 24pp); negative Gain ≤ −5pp must be explained
   │      Justification (required) · for each of (a)-(d), 1-2 sentences with qid + trajectory line numbers + extracted-letter trace as concrete evidence. Failing this gate = STOP, do NOT write to paper.
   │      Evidence · 
   └ How  · `eval.py --config scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_exp_orch.yaml` + `nvidia-smi -l 30 > tmp/power_gpqa_eval.log`

## Phase 7 — LCB [0/2]

19 ☐ LCB precache (175Q × 8 = 1400 explores, explore_timeout=1200 default)
   ├ G1 ☐ Gate · timed_out rate ≤ 5% (≤70/1400)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥95% Q (≥166/175) have ≥7 usable explores
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in `tmp/precache_lcb_gemma.log`
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥3 explores/min × 10-min window
   │      Evidence · 
   └ How  · `precache_explores.py --config scripts/lcb/grpo/lcb_gemma4_26b_a4b_precache.yaml` + `nvidia-smi -l 30 > tmp/power_lcb_precache.log`

20 ☐ LCB eval (175Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · `lcb_runner` returns no `metadata_list` IndexError (subprocess survival path)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · 0 rows where extracted code block empty AND `is_correct=true`
   │      Evidence · 
   ├ G5 ☐ Gate · 0 rows where `metadata_list[0]=={}` (subprocess SIGKILL fallback) AND `is_correct=true`
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_lcb_gemma.log`
   │      Evidence · 
   ├ G8 ☐ Gate · **Pass@1 leaderboard sanity** — `first_candidate_correct` rate ∈ [74.1%, 80.1%] (model card LCB-v6 77.1% ±3pp). Outside range → STOP, do NOT write to paper; debug pipeline.
   │      Evidence · 
   ├ G9 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) extracted code block from `predicted_answer` is syntactically valid Python (parses via `ast.parse`); lcb_runner verdict pass/fail flags are consistent with `is_correct` field
   │      (b) orchestrator emitted ≥1 `tool_calls` AND final code is structurally similar to (but NOT byte-identical to) ≥1 explore code candidate — confirms real synthesis, not first-explore copy
   │      (c) submitted code length distribution: median ≥30 lines, IQR > 10 lines (LCB problems require non-trivial code; <30-line median across 175 problems suggests model is producing stubs)
   │      (d) Gain = Acc − Pass@1 has plausible sign AND |Gain| < 2× Qwen LCB Gain (Qwen Gain=+12.0, so |Gemma Gain| < 24pp); negative Gain ≤ −5pp must be explained
   │      Justification (required) · for each of (a)-(d), 1-2 sentences with qid + trajectory line + lcb_runner metadata trace as concrete evidence. Failing this gate = STOP, do NOT write to paper.
   │      Evidence · 
   └ How  · `eval.py --config scripts/lcb/grpo/lcb_gemma4_26b_a4b_exp_orch.yaml` + `nvidia-smi -l 30 > tmp/power_lcb_eval.log`

## Phase 8 — BabyVision [0/2]

21 ☐ BabyVision precache (388Q × 8 = 3104 explores, multimodal)
   ├ G1 ☐ Gate · timed_out rate ≤ 5% (≤155/3104)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥95% Q (≥369/388) have ≥7 usable explores
   │      Evidence · 
   ├ G3 ☐ Gate · vit/mm-encoder no OOM (zero `CUDA out of memory` in log)
   │      Evidence · 
   ├ G4 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G5 ☐ Gate · zero `Traceback` in `tmp/precache_bv_gemma.log`
   │      Evidence · 
   ├ G6 ☐ Gate · throughput ≥2 explores/min × 10-min (multimodal slower → relaxed from 3)
   │      Evidence · 
   └ How  · `precache_explores.py --config scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_precache.yaml` + `nvidia-smi -l 30 > tmp/power_bv_precache.log`

22 ☐ BabyVision eval (388Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true` (applies to both choice and blank items)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with refusal phrase AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · per-GPU power ≥200W × ≥80% wall time (calibrated for Gemma-4-26B-A4B MoE on A100 80GB PCIe; 4B-active expert dispatch caps power well below 80% TDP — A/B verified at 16 vs 64 in-flight, max never crosses 232W)
   │      On-fail · tune YAML `num_workers` — try INCREASE first (Δ=+50%; e.g. 128→192) to drive more in-flight load; only decrease if vllm serve log shows `waiting>0` (KV-cache saturated) or 5xx errors
   │      Evidence · 
   ├ G5 ☐ Gate · zero `Traceback` in `tmp/eval_bv_gemma.log`
   │      Evidence · 
   ├ G6 ☐ Gate · **Pass@1 sanity (soft)** — no published Gemma BV baseline; use Qwen3.6-35B-A3B-FP8 BV `first_candidate_correct=16.75%` as cross-model soft reference, tolerance ±5pp → expect Pass@1 ∈ [11.75%, 21.75%]. Outside this range is suspicious but NOT auto-stop (different model family, broader prior); flag for manual review before paper write.
   │      Evidence · 
   ├ G7 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids (mix of choice + blank); for each verify:
   │      (a) for blank items: Gemma judge `verdict_reasoning` cites gold value verbatim, ≤2 sentences, no re-derivation keywords. For choice items: extracted letter matches what orchestrator committed to (no stray-letter extraction).
   │      (b) orchestrator emitted ≥1 `tool_calls`; vit/mm-encoder produced image features (check `usage.input_tokens` includes image-token contribution >> text-only baseline; if image_tokens=0 the multimodal pipeline silently fell back to text-only)
   │      (c) `predicted_answer` distribution split by `ansType`: choice letters within ±15% of empirical gold distribution; blank-answer length IQR > 30 chars
   │      (d) Gain has plausible sign AND |Gain| < 2× Qwen BV Gain (Qwen Gain=+6.2, so |Gemma Gain| < 12pp); negative Gain ≤ −5pp must be explained
   │      Justification (required) · for each of (a)-(d), 1-2 sentences with qid type (choice/blank) + trajectory line + judge or extractor trace. Failing this gate = STOP, do NOT write to paper.
   │      Evidence · 
   └ How  · `eval.py --config scripts/babyvision/grpo/babyvision_gemma4_26b_a4b_exp_orch.yaml` + `nvidia-smi -l 30 > tmp/power_bv_eval.log`

## Phase 9 — Paper integration [0/4]

23 ☐ Parse 4 `results.jsonl` (Variant A `_exp_orch`) → Pass@1 / Acc / Gain / $/q metrics
   ├ G1 ☐ Gate · 4 metric rows printed for HLE / GPQA / LCB / BV
   │      Evidence · 
   ├ G2 ☐ Gate · sum check `Acc - Pass@1 == Gain` for each row (within rounding)
   │      Evidence · 
   ├ G3 ☐ Gate · all 4 rows have Acc ∈ [0, 100], Pass@1 ∈ [0, 100], $/q ≥ 0
   │      Evidence · 
   ├ G4 ☐ Gate · **Cost budget** — `sum(cost_usd) for all 4 results.jsonl ≤ $0.10` (vllm calls = $0; only Anthropic API would cost; non-zero indicates a backend leaked into Anthropic)
   │      Evidence · 
   └ How  · python script: read each `run/<bench>/gemma4_26b_a4b_it_gemma_exp_orch/run_*/results.jsonl`, count `first_candidate_correct` + `is_correct`, sum `cost_usd`

24 ☐ Append 4 Gemma rows to `Publication/paper/main.tex`
   ├ G1 ☐ Gate · lines 425-428 contain 4 rows (HLE / GPQA / LCB / BV in that order)
   │      Evidence · 
   ├ G2 ☐ Gate · `\midrule` separator inserted between Qwen block and Gemma block
   │      Evidence · 
   ├ G3 ☐ Gate · row schema matches Qwen rows (Backbone & Effort & Bench & Pass@1 & Acc & Gain & $/q)
   │      Evidence · 
   └ How  · Edit tool, insert after Qwen block, before `\bottomrule`

25 ☐ Compile paper
   ├ G1 ☐ Gate · `compile.sh` exits 0
   │      Evidence · 
   ├ G2 ☐ Gate · `Publication/paper/build/main.pdf` mtime updated to current run
   │      Evidence · 
   ├ G3 ☐ Gate · zero `Overfull \hbox` warnings on the table page in compile log
   │      Evidence · 
   └ How  · `cd Publication/paper && bash compile.sh`

26 ☐ Verify table renders
   ├ G1 ☐ Gate · `tab:backbone-ablation` page shows 4 Gemma rows visible
   │      Evidence · 
   ├ G2 ☐ Gate · all column values aligned with headers; no margin overflow visually
   │      Evidence · 
   ├ G3 ☐ Gate · row order: Sonnet → GPT-5.2 → Qwen → Gemma (Gemma at bottom)
   │      Evidence · 
   └ How  · open `build/main.pdf`, navigate to `tab:backbone-ablation` page


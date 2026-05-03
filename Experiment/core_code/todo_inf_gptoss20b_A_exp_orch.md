# TODO: GPT-OSS-20B Variant A (`_exp_orch`) — paper main `tab:backbone-ablation`

## What this is

Variant A of the GPT-OSS-20B experiment plan: **explorer = orchestrator = same gpt-oss-20b model**, served by local vLLM. 3 benchmarks (HLE-Verified text-only / GPQA-Diamond / LCB). BabyVision is OUT-OF-SCOPE for this Variant — gpt-oss-20b is text-only (no vision tower); sending image-bearing prompts to a text-only model and reporting whatever degraded number falls out would contaminate the table. Configuration is matched to the Qwen3.6-35B-A3B-FP8 and Gemma-4-26B-A4B-it archetypes so the three open-weights backbones are comparable side-by-side in `tab:backbone-ablation`. This is the smallest backbone in the lineup (~21B total / 3.6B active MoE, MXFP4 quantization, ~16GB VRAM single-card resident).

This Variant tests the full ATTS stack with a single open-weights model serving as explorer + orchestrator + integrator — measuring "what GPT-OSS-20B can do as a complete self-contained ATTS pipeline".

## Output target

3 GPT-OSS rows appended to `Publication/paper/main.tex` `tab:backbone-ablation` (columns: Backbone / Effort / Bench / Pass@1 / Acc / Gain / $/q): HLE / GPQA / LCB. Placed below the existing Sonnet / GPT-5.2 / Qwen / Gemma backbone blocks. Compiled `Publication/paper/build/main.pdf` shows the table with all backbone families. Table caption notes "GPT-OSS-20B is text-only; BabyVision N/A". The numbers are the small-model untrained-base reference for the GRPO uplift story; together with Qwen-FP8 (~35B) and Gemma-BF16 (~26B) they bracket the ~10-40B open-weights regime.

## Discipline

Every event below has explicit Gates with checkboxes. An event flips from `☐` to `✓` only after **all** its Gates pass AND each Gate's `Evidence ·` line is filled with the actual measurement (e.g. "27/800 = 3.4% timed_out"). Soft-Gates require Justification with concrete qid + line-number evidence — never narrative claims like "looks fine". No silent skipping, no marking done before evidence is recorded. The prior Gemma run (2026-05-02 morning) violated this discipline and produced contaminated paper numbers (8.00 written from 100% empty predictions + judge hallucination); this file enforces the discipline structurally.

## GPU availability is a SOFT constraint

**ACTUAL GPU PINNING (2026-05-03)**: gpt-oss-20b serve is running on **GPU 2 + GPU 3** (DP=2). PIDs 1746736 (GPU 3) + 1746737 (GPU 2). Serve script: `scripts/gpqa/grpo/serve_gptoss20b_dp2.sh` (`CUDA_VISIBLE_DEVICES=2,3`, `--data-parallel-size 2`, port 8001, alias `gptoss-20b`). Each card holds ~70 GB at gpu_memory_utilization=0.85.

**Other GPU usage on this box (do NOT touch)**:
- GPU 0: occupied by another `peijia` process — VLLM::EngineCore PID 3308488 (~69 GB) plus the embedding daemon (PID 2918195, ~1.3 GB). NOT mine. Leave alone.
- GPU 1: my Gemma DP=1 serve (PID 3017288, port 8000, alias `gemma4-26b-a4b-it`) — for Variant B (sister TODO `todo_inf_gemma_B_sonnet_orch.md`).

Target topology: single-card serve (gpt-oss-20b ~16GB VRAM under MXFP4 quant fits in any A100/A6000/L40 80GB). Optional DP=2/4 if multiple cards free for higher concurrent throughput. Pre-flight item 03 measures `N_avail`; serve script and downstream throughput expectations scale with N_avail. Quality gates (Pass@1 sanity, judge integrity, timed_out rate) are NOT affected by N_avail — they are pipeline-correctness checks. Only N_avail=0 stops the run; 1/2/4 all proceed.

Allocation rule: do NOT collide with running Gemma serve. Currently Gemma is on GPU 1 only (DP=1) — gpt-oss on GPU 2/3 is non-overlapping and safe. Pre-flight item 03 G1 enforces this.

## Leaderboard anchor

Every eval event includes a "Pass@1 sanity-check" Gate that compares our measured Pass@1 (= first cached explore's correctness rate) against published single-shot baselines from the OpenAI gpt-oss model card and `gpt-oss-20b` arXiv companion paper. Look up the exact published numbers BEFORE starting Phase 5 and record them in the corresponding eval-item Evidence header. Tolerance ±3pp (HLE 100Q stderr ≈ 3pp at p≈10%).

Published baseline values to look up and record in evidence:
- HLE-no-tools Pass@1 (gpt-oss-20b model card or technical report)
- GPQA-Diamond Pass@1 (model card)
- LCB-v6 Pass@1 (model card or matching subset)

Deviation > ±3pp absolute = pipeline silently broken (tool-call format / prompt mismatch / chat-template wrong). Run is invalidated regardless of Acc.

## Resume / restart procedure

If a run dies mid-way (OOM / SIGTERM / network blip / vllm crash), the cache + results.jsonl are the resume substrate. NEVER re-run from scratch — that burns wall time and fails the cache discipline rule.

| Failure point | Recover by | Banner verification (mandatory after restart) |
|---|---|---|
| Mid precache (e.g. Q40/100) | Re-run same `precache_explores.py` command. `cache_dir` auto-skips already-cached `(qid, explore_idx)` pairs. | Banner says `Tasks: K to run, J already cached` with J>0. If J=0 despite prior partial run, STOP — cache key mismatch. |
| Mid eval (e.g. Q50/100) | Add `resume: <RUN_DIR>` to eval YAML pointing to the dying run's `analysis/run/<bench>/<...>/run_<timestamp>/`. Eval skips already-graded `(qid, rollout_idx)` rows via `results.jsonl`. | Both lines must appear: (a) `Resuming ...: N rollouts already completed` with N>0; (b) `Questions to run: M (N already completed, M+N total)`. |
| vLLM serve crash | Restart serve via `bash scripts/gpqa/grpo/serve_gptoss20b_dp{N_avail}.sh`. Eval clients retry on next call; no data loss. | `curl :8000/v1/models` returns 200 + `gptoss-20b` alias. |
| Pick which `RUN_DIR` to resume from | Pick by **largest** `wc -l results.jsonl`, NOT mtime (newer dirs may have crashed earlier) | n/a |

## Risk register (known failure modes)

These are the bugs the gates below are designed to catch — do NOT remove the corresponding gate just because a run looks fine.

| # | Failure | Root cause | Defense in this todo |
|---|---|---|---|
| R1 | Orchestrator emits tool calls in harmony channels but vllm OpenAI compat layer doesn't surface as `message.tool_calls[]` → 100/100 empty predictions | gpt-oss harmony format requires correct `--tool-call-parser openai` flag; without it the assistant_text channel leaks through as plain `content` | Phase 4 item 12 smoke G1-G4 (verify ≥1 structured `tool_calls` per question end-to-end) |
| R2 | Judge marks empty `predicted_answer` as correct → contaminated paper number | Judge ignored `_JUDGE_BASE` rule 5 ("no extractable answer → correct=false") | Phase 5/6/7 G2/G3 (post-run filter on empty + refusal regex; `is_correct=true` AND empty must be 0 rows) |
| R3 | precache `explore_timeout` too tight for harmony-format reasoning rollouts → high timed_out rate | gpt-oss high-effort reasoning budget can exceed naive 300s | Phase 4 item 13 smoke + Phase 5 item 14 G1 calibrate explore_timeout against measured per-explore wall time |
| R4 | LCB `metadata_list` IndexError on subprocess SIGKILL/segfault | Upstream `lcb_runner` bug (already patched in `compute_code_generation_metrics.py:53`) | Phase 7 item 19 G2 (subprocess survival check: no IndexError raised) |
| R5 | vLLM version too old to support gpt-oss harmony format | gpt-oss requires vllm with `+gptoss` extras OR vllm ≥ 0.20 mainstream | Phase 2 item 06 G2 (env version check before serve; `vllm -V` ≥ 0.20 OR `pip show vllm | grep gptoss`) |
| R6 | `served-model-name` mismatch — yaml references `gptoss-20b` but serve aliases `openai/gpt-oss-20b` (full HF id) → 404 on every chat completion | inconsistent alias naming between YAML and serve script | Phase 4 item 11 G1 (`curl :8000/v1/models` returns alias matching what YAMLs reference) |
| R7 | Pass@1 deviates >3pp from model card baseline | Pipeline silently broken (R1/R2/R6 active) | Every eval G_LB (leaderboard sanity, ±3pp hard) |
| R8 | Reasoning effort level mismatch — harmony format supports low/medium/high reasoning effort; using "high" by default may blow up wall time AND token budget unnecessarily | OpenAI harmony default unclear | Phase 1 item 01 G3 (sampling block explicitly pins `reasoning.effort` to a chosen level + comment explaining why; default = `medium` unless benchmark-specific evidence supports `high` or `low`) |

## Co-monitor — log paths for parallel watching

All long-running events follow the same logging convention. Per `feedback_share_long_running_logs` memory, on launch print PID + absolute paths. Reference table:

| Phase | Run log (stdout/stderr) | Power log (`nvidia-smi -l 30`) |
|---|---|---|
| 09 vLLM serve | `tmp/vllm_serve_gptoss20b_dp{N}.log` | n/a (serve idle until queries) |
| 12 HLE smoke | `tmp/precache_hle_gptoss_smoke.log` | `tmp/power_hle_gptoss_smoke.log` |
| 13 HLE precache | `tmp/precache_hle_gptoss.log` | `tmp/power_hle_gptoss_precache.log` |
| 14 HLE eval | `tmp/eval_hle_gptoss.log` | `tmp/power_hle_gptoss_eval.log` |
| 15/16 GPQA precache/eval | `tmp/{precache,eval}_gpqa_gptoss.log` | `tmp/power_gpqa_gptoss_{precache,eval}.log` |
| 17/18 LCB precache/eval | `tmp/{precache,eval}_lcb_gptoss.log` | `tmp/power_lcb_gptoss_{precache,eval}.log` |

User can `tail -f /data3/peijia/dr-claw/Explain/Experiment/core_code/<path>` for any of these. All paths are absolute-resolvable from `core_code/` working dir.

## Phase 1 — Config creation [3/3 ✓]

01 ✓ Create `hle_gptoss20b_precache.yaml` and `hle_gptoss20b_exp_orch.yaml` under `scripts/hle/grpo/`
   ├ G1 ✓ Gate · both YAMLs parse via `load_config(path, schema=PrecacheConfig)` and `EvalConfig` respectively without ValidationError
   │      Evidence · `OK scripts/hle/grpo/hle_gptoss20b_precache.yaml` and `OK scripts/hle/grpo/hle_gptoss20b_exp_orch.yaml` from python smoke 2026-05-03 03:53.
   ├ G2 ✓ Gate · `cache_dir: ../analysis/cache/hle/gptoss20b/gold` in BOTH (precache writes, eval reads same dir); `explore_model: gptoss-20b`; precache `num: 100`, `num_workers: 64`, `num_explores: 8`, `explore_timeout: 1200.0`, `text_only: true`; eval `orchestrator_model: gptoss-20b`, `explore_model: gptoss-20b`, `no_integrate: true`
   │      Evidence · scripts/hle/grpo/hle_gptoss20b_precache.yaml lines 24/27/30/3-4 carry the required values; hle_gptoss20b_exp_orch.yaml lines 14-19/27-30 carry orchestrator_model + explore_model + no_integrate. Both share cache_dir `../analysis/cache/hle/gptoss20b/gold`.
   ├ G3 ✓ Gate · sampling block matches OpenAI gpt-oss harmony defaults: T=1.0, top_p=1.0, max_tokens=32768 for orchestrator (eval YAML); precache uses larger max_tokens=60000. Reasoning effort default (medium) — vllm 0.20.0 auto-injects `reasoning_parser='openai_gptoss'` per server log line `structured_outputs_config=StructuredOutputsConfig(... reasoning_parser='openai_gptoss', ...)`; per-request reasoning effort knob deferred to Phase 4 smoke item 12 (only override if smoke shows pathological behavior — KISS).
   │      Evidence · precache.yaml line 47-49 (T=1.0, top_p=1.0, max_tokens=60000); exp_orch.yaml line 23-27 (T=1.0, top_p=1.0, max_tokens=32768). NO top_k (gpt-oss training did not use top_k).
   ├ G4 ✓ Gate · `benchmark.judge` block in BOTH yamls: `vllm gptoss-20b T=1.0 top_p=1.0 max_tokens=4096` (judge is same model — single-model self-consistency)
   │      Evidence · precache.yaml lines 5-15 + exp_orch.yaml lines 5-11: judge `name=vllm model=gptoss-20b T=1.0 top_p=1.0 max_tokens=4096`. Same-model judge intentional; if length-truncation observed post-launch, override extra_body.reasoning={"effort":"low"} per Phase 4 smoke.
   └ How  · clone `scripts/hle/grpo/hle_gemma4_26b_a4b_precache.yaml` and `..._exp_orch.yaml`, swap explore_model + orchestrator_model + cache_dir + sampling block, save with new names

02 ✓ Create `gpqa_gptoss20b_precache.yaml` and `gpqa_gptoss20b_exp_orch.yaml` under `scripts/gpqa/grpo/`
   ├ G1 ✓ Gate · both YAMLs parse without ValidationError
   │      Evidence · `OK scripts/gpqa/grpo/gpqa_gptoss20b_precache.yaml` and `OK scripts/gpqa/grpo/gpqa_gptoss20b_exp_orch.yaml` from python smoke 2026-05-03 03:53.
   ├ G2 ✓ Gate · `cache_dir: ../analysis/cache/gpqa/gptoss20b`; `explore_model: gptoss-20b`; precache `num: 198` (full Diamond) — DEVIATION: yaml does NOT pin `num`, defaulting to GPQASpec's full set; downstream eval YAMLs that need a 198 cap can add `num: 198` at top level. `num_workers: 64`, `num_explores: 8`, `explore_timeout: 600`; NO `judge` block (GPQA uses string-match A-E, GPQASpec rejects judge field)
   │      Evidence · scripts/gpqa/grpo/gpqa_gptoss20b_precache.yaml lines 4/5/6/9-12; no judge block (validated by parse PASS). Eval yaml omits judge for the same reason.
   ├ G3 ✓ Gate · sampling block matches HLE precache YAML except `max_tokens=20000` for orchestrator — DEVIATION: kept at 32768 (matches Qwen archetype for paper-row comparability per Gemma exp_orch precedent at line 23 of gpqa_gemma4_26b_a4b_exp_orch.yaml). reasoning effort default same as HLE.
   │      Evidence · precache.yaml lines 16-19 (T=1.0, top_p=1.0, max_tokens=60000); exp_orch.yaml lines 13-17 (T=1.0, top_p=1.0, max_tokens=32768). max_tokens=32768 (not 20000) deliberately, matches Gemma + Qwen GPQA archetype for cross-model comparability in tab:backbone-ablation.
   └ How  · clone `scripts/gpqa/grpo/gpqa_gemma4_26b_a4b_*.yaml`, swap fields, save with new names

03 ✓ Create `lcb_gptoss20b_precache.yaml` and `lcb_gptoss20b_exp_orch.yaml` under `scripts/lcb/grpo/`
   ├ G1 ✓ Gate · both YAMLs parse without ValidationError
   │      Evidence · `OK scripts/lcb/grpo/lcb_gptoss20b_precache.yaml` and `OK scripts/lcb/grpo/lcb_gptoss20b_exp_orch.yaml` from python smoke 2026-05-03 03:53.
   ├ G2 ✓ Gate · `cache_dir: ../analysis/cache/lcb/gptoss20b`; `explore_model: gptoss-20b`; precache `num: 175` — DEVIATION: yaml does not pin `num`, defaulting to LCBSpec's full set. `num_workers: 64`, `num_explores: 8`, `explore_timeout: 1200`; NO `judge` block (LCB uses code execution)
   │      Evidence · scripts/lcb/grpo/lcb_gptoss20b_precache.yaml lines 4/5/8-13; eval yaml lines 4-19. No judge block (parse PASS).
   ├ G3 ✓ Gate · sampling block: precache `max_tokens=60000`, eval orchestrator `max_tokens=20000` (shorter to leave headroom for accumulated multi-turn explore tool_response inputs)
   │      Evidence · precache.yaml lines 14-17 (max_tokens=60000); exp_orch.yaml lines 13-21 (max_tokens=20000).
   └ How  · clone `scripts/lcb/grpo/lcb_gemma4_26b_a4b_*.yaml`, swap fields

## Phase 2 — Pre-flight [3/3 ✓]

04 ✓ GPU + process baseline
   ├ G1 ✓ Gate (SOFT, **adaptive**) · pre-launch GPU 2 + GPU 3 = 0 MiB used (Gemma DP=2 occupies 0+1; gpt-oss takes 2+3 per user directive 2026-05-03 "如果两个空出来了 你可以用那两个跑gpt oss"). N_avail=2 → DP=2 selected.
   │      Evidence · pre-launch nvidia-smi snapshot (before serve_gptoss20b_dp2.sh) showed GPU 2=0 MiB, GPU 3=0 MiB; GPU 0+1 occupied by Gemma DP=2 PID 3623749 (alias gemma4-26b-a4b-it on port 8000). Gemma serve intentionally NOT killed — co-located.
   ├ G2 ✓ Gate · zero stale `vllm`/`gptoss`/`precache`/`eval.py` PIDs from prior gpt-oss attempts
   │      Evidence · `pgrep -af 'gptoss|vllm.*gpt-oss'` pre-launch was empty. Post-launch the only matches are the new gpt-oss serve PID 1732164 (parent) + 1732688 (child python) — both intended.
   ├ G3 ✓ Gate · CPU memory available ≥ 20 GiB × N_avail = 40 GiB
   │      Evidence · `free -g`: total=251, available=177 GiB ≫ 40 GiB threshold (4.4× margin). DP=2 worker resident ~16 GiB/card → ~32 GiB MXFP4 footprint, well within budget.
   └ How  · `nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv` ; `free -g` ; `pgrep -af 'vllm|gptoss|gpt-oss|precache|eval\.py'`

05 ✓ Conda env + version check
   ├ G1 ✓ Gate · `grpo_vllm` env exists and used for serve
   │      Evidence · serve PID 1732164 cmdline: `/home/peijia/miniconda3/condabin/conda run --no-capture-output -n grpo_vllm vllm serve openai/gpt-oss-20b ...` confirms env binding.
   ├ G2 ✓ Gate · `grpo_vllm` env has vllm 0.20.0 with native gpt-oss support
   │      Evidence · `conda run -n grpo_vllm pip show vllm | grep Version` → `Version: 0.20.0`. Serve log line `Resolved architecture: GptOssForCausalLM` + `Using 'MARLIN' Mxfp4 MoE backend` confirms native support — no separate gpt-oss build tag needed.
   ├ G3 ✓ Gate · `explain` env importable
   │      Evidence · python smoke 2026-05-03 03:53 imported `from eval import load_config, EvalConfig` and `from precache_explores import PrecacheConfig` and `from backends.vllm import _get_client, MODEL_TO_BASE_URL` all without error.
   └ How  · the three commands above

06 ✓ YAML schema validation (all 6 gpt-oss YAMLs)
   ├ G1 ✓ Gate · 3 precache YAMLs parse via `load_config(path, schema=PrecacheConfig)` without ValidationError
   │      Evidence · `OK scripts/{hle,gpqa,lcb}/grpo/*_gptoss20b_precache.yaml` × 3 from python smoke 2026-05-03 03:53.
   ├ G2 ✓ Gate · 3 exp_orch YAMLs parse via `load_config(path, schema=EvalConfig)` without ValidationError
   │      Evidence · `OK scripts/{hle,gpqa,lcb}/grpo/*_gptoss20b_exp_orch.yaml` × 3 from same smoke.
   ├ G3 ✓ Gate · all 6 `cache_dir` paths point under `analysis/cache/<bench>/gptoss20b` and do NOT exist yet (force fresh cache)
   │      Evidence · `ls ../analysis/cache/{hle,gpqa,lcb}/gptoss20b/ 2>&1` → all 3 dirs do not exist (will be created by precache_explores.py on first run).
   └ How  · python script that imports `eval.load_config`, iterates yaml paths, asserts no exception, asserts cache_dir absent

## Phase 3 — vLLM serve [3/3 ✓]

07 ✓ Write `serve_gptoss20b_dp2.sh` under `scripts/gpqa/grpo/`
   ├ G1 ✓ Gate · script written with the required flags + inline comments (per `comment_on_config_overrides` memory)
   │      Evidence · `scripts/gpqa/grpo/serve_gptoss20b_dp2.sh` lines 32-44 carry: `--served-model-name gptoss-20b`, `--tensor-parallel-size 1`, `--data-parallel-size 2`, `--gpu-memory-utilization 0.85`, `--max-model-len 131072`, `--max-num-batched-tokens 8192`, `--enable-auto-tool-choice`, `--tool-call-parser openai`, `--trust-remote-code`, `--port 8001` (NOT 8000 — Gemma occupies 8000). Lines 4-22 carry per-override rationale comments. log → `tmp/vllm_serve_gptoss20b_dp2.log`.
   ├ G2 ✓ Gate · script does NOT include `--reasoning-parser` or `--structured-outputs-config`
   │      Evidence · grep on the script: no `--reasoning-parser` flag; no `--structured-outputs-config` flag. NOTE: vllm 0.20.0 nonetheless auto-injects `reasoning_parser='openai_gptoss'` server-side (log line `structured_outputs_config=StructuredOutputsConfig(... reasoning_parser='openai_gptoss', enable_in_reasoning=False)`); this is harmless — harmony channel parsing happens transparently, no client config needed.
   ├ G3 ✓ Gate · CUDA_VISIBLE_DEVICES set to N_avail=2 card ids (GPU 2,3) matching item 04 measurement
   │      Evidence · script line 32: `CUDA_VISIBLE_DEVICES=2,3`. Process cmdline (PID 1732164) confirms.
   └ How  · clone `scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp2.sh` as template; replace model id / alias / TP-DP / context-len / parsers; remove Gemma-specific comments; add gpt-oss-specific comments

08 ✓ Start vllm serve gpt-oss-20b
   ├ G1 ✓ Gate · launcher exits with PID echoed; no immediate (<5s) crash
   │      Evidence · `bash scripts/gpqa/grpo/serve_gptoss20b_dp2.sh` printed `started gpt-oss-20b DP=2 serve (PID 1732164)`. After 5s + 20s log peek, PID still alive (etime=01:01 → 05:27 over the bring-up window).
   ├ G2 ✓ Gate · log file `tmp/vllm_serve_gptoss20b_dp2.log` created and written to within 60s
   │      Evidence · log file populated within 9s of launch (first content line `INFO 05-03 03:41:48 [nixl_utils.py:20]`); `Resolved architecture: GptOssForCausalLM` at 03:41:58 (10s after launch).
   ├ G3 ✓ Gate · `CUDA_VISIBLE_DEVICES=2,3` and `--data-parallel-size 2` in launched process
   │      Evidence · `pgrep -af gptoss` shows PID 1732164/1732688 with `--data-parallel-size 2 ... --port 8001`. nvidia-smi confirms only GPU 2/3 allocated by this serve.
   └ How  · `bash scripts/gpqa/grpo/serve_gptoss20b_dp2.sh` ; `pgrep -af gptoss` ; `ls -la tmp/vllm_serve_gptoss20b_dp2.log`

09 ✓ Verify serve health (≥3 min after start)
   ├ G1 ✓ Gate · serve log contains `Maximum concurrency for X tokens per request`
   │      Evidence · log line at 03:44:59 (3min13s after launch): `(EngineCore_DP0) GPU KV cache size: 1,211,392 tokens` + `Maximum concurrency for 131,072 tokens per request: 17.38x`. Same line for DP1.
   ├ G2 ✓ Gate · zero `Traceback`/`RuntimeError`/`OOM` lines in serve log; `EngineDeadError` absent
   │      Evidence · `grep -cE "Traceback.*Worker|RuntimeError|EngineDeadError|CUDA out of memory" tmp/vllm_serve_gptoss20b_dp2.log` → 0. (vllm telemetry `_report_usage_worker → cpuinfo.get_cpu_info → JSONDecodeError` traceback IS present but harmless — same noise pattern as Gemma serve, does not affect serving.)
   ├ G3 ✓ Gate · KV cache pool allocated
   │      Evidence · `GPU KV cache size: 1,211,392 tokens` × 2 (DP0 + DP1) = 2.4M tokens total pool.
   ├ G4 ✓ Gate · DP=2 workers boot; nvidia-smi shows ~70 GiB per card (vs todo prediction 18 GiB)
   │      Evidence · GPU 2 = 70129 MiB, GPU 3 = 70127 MiB used. **Discrepancy from todo's 18 GiB prediction**: gpu_memory_utilization=0.85 on 80 GiB A100 → ~68 GiB target including weights (~12 GiB MXFP4) + cuda graphs + KV pool (1.21M tokens × ~50 KB/token ≈ 60 GiB). Actual matches `0.85 × 80 ≈ 68 GiB` formula precisely. todo's 18 GiB estimate was wrong (assumed weights-only fill).
   └ How  · `tail tmp/vllm_serve_gptoss20b_dp2.log` ; `nvidia-smi --query-gpu=memory.used,index --format=csv -i 2,3`

## Phase 4 — Smoke + connectivity [0/3]

10 ✓ Smoke `/v1/models` and basic `/v1/chat/completions` via curl (port REMAPPED to 8001)
   ├ G1 ✓ Gate · `curl http://localhost:8001/v1/models` returns 200; JSON contains `"id":"gptoss-20b"`
   │      Evidence · `curl -s http://localhost:8001/v1/models` → HTTP 200, JSON `data:[{id:"gptoss-20b", ...}]`. Port REMAPPED 8000→8001 because Gemma DP=2 occupies 8000; `backends/vllm.py` MODEL_TO_BASE_URL routes `gptoss-20b` → `http://localhost:8001/v1` automatically.
   ├ G2 ✓ Gate · chat completion returns 200, content non-empty, finish_reason=stop
   │      Evidence · `{"model":"gptoss-20b","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":256}` → HTTP 200, finish_reason=stop, completion_tokens=47, content="2 + 2 = 4." (with U+202F narrow no-break spaces). reasoning_content empty (harmony channel skipped on simple prompt — same as Gemma simple-prompt skip).
   ├ G3 ✓ Gate · response time < 30s for simple prompt
   │      Evidence · curl -w "%{time_total}" → 0.255s on second invocation (post warmup). First invocation similar order. ≪ 30s threshold by 117× margin.
   └ How  · `curl :8001/v1/models` ; `curl :8001/v1/chat/completions -d '{"model":"gptoss-20b","messages":[{"role":"user","content":"What is 2+2?"}]}'`

11 ☐ Smoke tool-call: orchestrator emits structured `tool_calls[]` (R1 defense)
   ├ G1 ☐ Gate · single chat completion request with `tools=[explore_tool_def]`, `tool_choice="auto"` returns `response.choices[0].message.tool_calls` ≥ 1 structured ToolCall (NOT plain text in `message.content`)
   │      Evidence · 
   ├ G2 ☐ Gate · `tool_calls[0].function.name == "explore"` and `arguments` JSON-decodes with `question` key as string
   │      Evidence · 
   ├ G3 ☐ Gate · multi-turn end-to-end via `run_tool_conversation` in `backends/vllm.py`: orchestrator → tool_handler → StructuredOutput in 2 turns without crash
   │      Evidence · 
   ├ G4 ☐ Gate · zero `parse_tool_calls` exceptions in client log (server-side parser populated correctly per `--tool-call-parser openai`)
   │      Evidence · 
   └ How  · python script invoking `backends.vllm.run_tool_conversation` with a fake explore tool def and minimal HLE-style prompt; assert structured tool_calls received

12 ☐ HLE smoke precache (num=2 = 16 explores, verify max_tokens budget + reasoning effort don't trigger pathological behavior)
   ├ G1 ☐ Gate · ≥10/16 explores have `output.md` non-zero AND `timed_out=false` (lower bar than full run since smoke samples just 2 questions; full run G1 in item 13 sets the production bar)
   │      Evidence · 
   ├ G2 ☐ Gate · zero token-repetition loops in any output.md (grep for runs of ≥10 identical adjacent tokens like `"step-step-step..."`); harmony format should not have Gemma's xgrammar repetition bug but verify
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥150W × ≥80% wall time on each active card (lowered vs Gemma's 200W threshold because gpt-oss-20b is 3.6B-active vs Gemma's 4B-active — slightly lower expert dispatch load; recalibrate after measurement, this is a soft starting threshold)
   │      On-fail · tune YAML `num_workers` first (try INCREASE 16→64); only investigate vllm `max-num-seqs` if power stays below threshold at saturated concurrency
   │      Evidence · 
   ├ G4 ☐ Gate · throughput ≥ 1 explore/min observed (per-explore avg wall ≤ 60s at concurrency); if much slower, recalibrate explore_timeout in item 13
   │      Evidence · 
   └ How  · clone `hle_gptoss20b_precache.yaml` with `num: 2`, `num_workers: 16`, separate `cache_dir` to `../analysis/cache/hle/gptoss20b_smoke/gold`; run `precache_explores.py`; nvidia-smi monitoring

## Phase 5 — HLE [0/2]

13 ☐ HLE precache full (100Q × 8 = 800 explores)
   ├ G1 ☐ Gate · timed_out rate ≤ 10% (≤80/800; calibrated against Gemma's observed model-physics ceiling on hard physics questions; gpt-oss may be similar or better, recalibrate based on smoke item 12 actual rate)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥90/100 Q have ≥6 usable explores (output.md ∧ ¬timed_out ∧ ¬parse_failed)
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥150W × ≥80% wall time (tail 5% exempt: in-flight ≤4 requests)
   │      On-fail · tune `num_workers` Δ=+50% first; only decrease if vllm log shows `waiting>0`
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in `tmp/precache_hle_gptoss.log`
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥ 3 explores/min sustained over any 10-min rolling window
   │      Evidence · 
   └ How  · `conda run -n explain python precache_explores.py --config scripts/hle/grpo/hle_gptoss20b_precache.yaml > tmp/precache_hle_gptoss.log 2>&1 &` ; `nvidia-smi -l 30 > tmp/power_hle_gptoss_precache.log`

13B ✓ TEMPORARY: client-side patch for chat/completions tool-name leak (vllm#32587)
   ├ G1 ✓ Gate · `backends/vllm.py` carries `_HARMONY_LEAK_RE` regex stripping `(<\|...\|>)?(commentary|analysis|final|json)$` suffix from `tc.function.name`
   │      Evidence · backends/vllm.py:57-65 defines the constant; backends/vllm.py:518-548 applies it in the tool-call parsing loop; assistant-message history append at backends/vllm.py:563 uses the cleaned tool_calls list so polluted name never feeds back to the model
   ├ G2 ✓ Gate · unit test 10/10 cases pass (incl. `explorecommentary` no-separator variant)
   │      Evidence · 2026-05-03 05:39 — 10 cases run via `re.sub`, all PASS: `explore<|channel|>{commentary,json,analysis}` → `explore`; `explore{commentary,analysis,final,json}` → `explore`; `StructuredOutput`, `synthesize` unchanged
   ├ G3 ✓ Gate · KNOWN INSUFFICIENT — masks the FORWARD leak (server → client) but does NOT fix the REVERSE leak (client → server HTTP 500 "unexpected tokens remaining in message header"); item 14B is the real fix
   │      Evidence · 2026-05-03 05:39 — gpt-oss HLE eval re-launched with regex patch; strip warnings fired correctly on 4 turns then crashed at vllm.py:485 with `openai.InternalServerError 500 'unexpected tokens remaining in message header: Some("...have three candidates: #1 L, #2 L...<|end|><|start|>assistant<|channel|>commentary")'`. Patch is defensive only; KEEP it (no-op for non-leaked names) but do not rely on it as the production fix.
   └ How  · regex defined as `_HARMONY_LEAK_RE = re.compile(r"(<\|[^|]*\|>)?(commentary|analysis|final|json)$")`; applied to `tc.function.name` in the tool-call parsing loop

14B ☐ REAL FIX: write `/v1/responses` adapter for gpt-oss tool calling (vllm#22578 wontfix workaround)
   ├ G1 ☐ Gate · `backends/vllm.py` carries new function `run_tool_conversation_responses(...)` that wraps `client.responses.create()` (Responses API) and emits the same `(cost, usage, exit_reason)` tuple as the chat/completions version
   │      Evidence · 
   ├ G2 ☐ Gate · dispatch path: `run_tool_conversation()` checks if model alias is `gptoss-20b` (or any registered "harmony" model) and routes to `run_tool_conversation_responses()`; all other models continue through chat/completions. Single registry constant `HARMONY_MODELS: set[str] = {"gptoss-20b"}` at module top. No silent fallback.
   │      Evidence · 
   ├ G3 ☐ Gate · responses-API tool-call shape correctly maps to our `tool_handler(name, args)` callback. Each `output[].type == "function_call"` item provides clean `name` field (no harmony channel leak — that's the whole point of using this endpoint).
   │      Evidence · 
   ├ G4 ☐ Gate · multi-turn history maintained via `previous_response_id` chaining (Responses API native) OR via `input` message-history rebuild — pick whichever matches OpenAI SDK's documented Responses pattern; cite the Cookbook section used.
   │      Evidence · 
   ├ G5 ☐ Gate · `cache_only` enforcement preserved: any cache miss during tool dispatch raises (no Anthropic API charge possible). Cost accounting matches chat/completions path (P5 fair-cost rule).
   │      Evidence · 
   ├ G6 ☐ Gate · standalone smoke test: 2 questions × 8 explores via responses adapter, zero crashes, results.jsonl emitted with non-empty `predicted_answer` and `tool_calls` evidence in trajectory
   │      Evidence · 
   ├ G7 ☐ Gate · `_HARMONY_LEAK_RE` regex from item 13B kept as defensive layer (no-op for clean names from responses adapter; only fires if a non-harmony model unexpectedly leaks)
   │      Evidence · 
   └ How  · references: vLLM Recipes GPT-OSS section (`/v1/responses` recommended), OpenAI Cookbook gpt-oss/run-vllm tool-calling example, openai-python SDK `client.responses.create()` API. Estimated ~80-100 LoC. Confidence ~65% on first iteration; budget 2-3 debug rounds.

14 ☐ HLE eval (100Q, exp_orch)  [NOTE: depends on item 14B adapter being live]
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · `exit_reason=="incomplete"` rate ≤ 10%
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true` (judge integrity, defends R2)
   │      Evidence · 
   ├ G4 ☐ Gate · 0 rows with refusal phrase (regex `(?i)i (don'?t|do not) know|cannot (determine|answer)|unable to|insufficient`) AND `is_correct=true`
   │      Evidence · 
   ├ G5 ☐ Gate · per-GPU power ≥150W × ≥80% wall time (tail 5% exempt)
   │      On-fail · tune `num_workers` Δ=+50% first; decrease only if waiting>0
   │      Evidence · 
   ├ G6 ☐ Gate · zero `Traceback` in `tmp/eval_hle_gptoss.log`
   │      Evidence · 
   ├ G7 ☐ Gate · **Pass@1 leaderboard sanity** — `first_candidate_correct` rate within ±3pp of gpt-oss-20b model card HLE-no-tools Pass@1 (look up exact value before run; record here in Evidence). Outside range → STOP, do NOT write to paper; debug pipeline.
   │      Evidence · 
   ├ G8 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids (seed=42 from results.jsonl); for each verify:
   │      (a) judge `verdict_reasoning` cites gold value verbatim, ≤ 2 sentences, no re-derivation keywords
   │      (b) orchestrator visibly READ explore cache content (final answer references explore reasoning) AND emitted ≥ 1 structured `tool_calls`
   │      (c) `predicted_answer` length distribution across 100 rows: median ≥ 50 chars AND IQR > 30 chars
   │      (d) ATTS Acc Gain plausible vs Pass@1 baseline (cross-bench rule of thumb: gain ≥ 0pp; large negative gain = pipeline regression)
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + `trajectories/<qid>/trajectory.md` line numbers + `results.jsonl` row index as concrete evidence; do NOT just say "looks fine". Failing this gate (a-c hard, d soft flag) = STOP, do NOT write to paper.
   │      Evidence · 
   └ How  · `conda run -n explain python eval.py --config scripts/hle/grpo/hle_gptoss20b_exp_orch.yaml > tmp/eval_hle_gptoss.log 2>&1 &` ; `nvidia-smi -l 30 > tmp/power_hle_gptoss_eval.log`

## Phase 6 — GPQA [0/2]

15 ☐ GPQA precache (198Q × 8 = 1584 explores, explore_timeout=600)
   ├ G1 ☐ Gate · timed_out rate ≤ 10% (≤158/1584)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥90% Q (≥178/198) have ≥6 usable explores
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥150W × ≥80% wall time
   │      On-fail · tune `num_workers` Δ=+50% first
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in `tmp/precache_gpqa_gptoss.log`
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥ 3 explores/min × 10-min rolling window
   │      Evidence · 
   └ How  · `precache_explores.py --config scripts/gpqa/grpo/gpqa_gptoss20b_precache.yaml`

16 ☐ GPQA eval (198Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows where `predicted_answer` contains no A-E letter AND `is_correct=true` (`grader.py::_extract_mc_letter` over-permissive guard)
   │      Evidence · 
   ├ G4 ☐ Gate · per-GPU power ≥150W × ≥80% wall time
   │      On-fail · tune `num_workers` Δ=+50% first
   │      Evidence · 
   ├ G5 ☐ Gate · zero `Traceback` in `tmp/eval_gpqa_gptoss.log`
   │      Evidence · 
   ├ G6 ☐ Gate · **Pass@1 leaderboard sanity** — `first_candidate_correct` rate within ±3pp of gpt-oss-20b model card GPQA-Diamond Pass@1 (record value here). Outside → STOP, do NOT write to paper.
   │      Evidence · 
   ├ G7 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; for each verify:
   │      (a) extracted MC letter matches orchestrator's committed final letter (no stray-letter extraction)
   │      (b) orchestrator visibly aggregated 8 explore letters (e.g. majority vote, principled selection); NOT first-explore copy
   │      (c) extracted letter distribution across 198 rows: A/B/C/D within ±15% of uniform
   │      (d) ATTS Acc Gain ≥ 0pp vs Pass@1
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + trajectory line + row index. Failing (a-c hard) = STOP.
   │      Evidence · 
   └ How  · `eval.py --config scripts/gpqa/grpo/gpqa_gptoss20b_exp_orch.yaml`

## Phase 7 — LCB [0/2]

17 ☐ LCB precache (175Q × 8 = 1400 explores, explore_timeout=1200)
   ├ G1 ☐ Gate · timed_out rate ≤ 10% (≤140/1400)
   │      Evidence · 
   ├ G2 ☐ Gate · ≥90% Q (≥158/175) have ≥6 usable explores
   │      Evidence · 
   ├ G3 ☐ Gate · per-GPU power ≥150W × ≥80% wall time
   │      Evidence · 
   ├ G4 ☐ Gate · zero `Traceback` in `tmp/precache_lcb_gptoss.log`
   │      Evidence · 
   ├ G5 ☐ Gate · throughput ≥ 3 explores/min × 10-min window
   │      Evidence · 
   └ How  · `precache_explores.py --config scripts/lcb/grpo/lcb_gptoss20b_precache.yaml`

18 ☐ LCB eval (175Q)
   ├ G1 ☐ Gate · non-empty `predicted_answer` rate ≥ 90%
   │      Evidence · 
   ├ G2 ☐ Gate · `lcb_runner` returns no `metadata_list` IndexError (R4 defense; subprocess survival)
   │      Evidence · 
   ├ G3 ☐ Gate · 0 rows with empty/whitespace `predicted_answer` AND `is_correct=true`
   │      Evidence · 
   ├ G4 ☐ Gate · 0 rows where extracted code block empty AND `is_correct=true`
   │      Evidence · 
   ├ G5 ☐ Gate · 0 rows where `metadata_list[0]=={}` (subprocess SIGKILL fallback) AND `is_correct=true`
   │      Evidence · 
   ├ G6 ☐ Gate · per-GPU power ≥150W × ≥80% wall time
   │      Evidence · 
   ├ G7 ☐ Gate · zero `Traceback` in `tmp/eval_lcb_gptoss.log`
   │      Evidence · 
   ├ G8 ☐ Gate · **Pass@1 leaderboard sanity** — `first_candidate_correct` rate within ±3pp of gpt-oss-20b model card LCB-v6 Pass@1 (record value). Outside → STOP, do NOT write to paper.
   │      Evidence · 
   ├ G9 ☐ Soft-Gate · **Post-eval sanity review (Claude self-conducted)** — sample 5 random qids; verify:
   │      (a) extracted code is valid Python (`ast.parse`); lcb_runner pass/fail flags consistent with `is_correct`
   │      (b) orchestrator visibly synthesized from explore cache (final code structurally similar to ≥1 cached explore code, not byte-identical)
   │      (c) submitted code length: median ≥ 30 lines across 175 rows
   │      (d) ATTS Acc Gain ≥ 0pp vs Pass@1
   │      Justification (required) · 1-2 sentences per (a)-(d) with qid + trajectory + lcb_runner metadata. Failing (a-c hard) = STOP.
   │      Evidence · 
   └ How  · `eval.py --config scripts/lcb/grpo/lcb_gptoss20b_exp_orch.yaml`

## Phase 8 — Paper integration [0/1]

19 ☐ Append GPT-OSS rows to `tab:backbone-ablation` and recompile
   ├ G1 ☐ Gate · 3 rows added to `Publication/paper/main.tex` `tab:backbone-ablation` (HLE / GPQA / LCB) with `Backbone="GPT-OSS-20B (untrained)"`, `Effort="thinking"` (or `"medium"` reflecting reasoning-effort setting), per-bench Pass@1 / Acc / Gain / $/q populated from results.jsonl
   │      Evidence · 
   ├ G2 ☐ Gate · row schema matches existing Sonnet / GPT-5.2 / Qwen / Gemma rows in same table (same column count + ordering)
   │      Evidence · 
   ├ G3 ☐ Gate · `\midrule` separator inserted between previous backbone block and new GPT-OSS block; existing rows UNTOUCHED
   │      Evidence · 
   ├ G4 ☐ Gate · table caption (or footnote) updated with "GPT-OSS-20B is text-only; BabyVision N/A" so absence of BV row is explicit, not implicit
   │      Evidence · 
   ├ G5 ☐ Gate · `compile.sh` exits 0; `Publication/paper/build/main.pdf` mtime updated
   │      Evidence · 
   ├ G6 ☐ Gate · zero `Overfull \hbox` warnings on the table page in compile log
   │      Evidence · 
   ├ G7 ☐ Gate · cross-Variant sanity: this Variant's Pass@1 ≪ this Variant's Acc when ATTS adds value (Gain > 0); negative Gain on ≥2/3 benchmarks = flag for manual review (small backbone may genuinely under-orchestrate, but document the finding)
   │      Evidence · 
   └ How  · python script reads each `run/<bench>/gptoss20b_exp_orch/run_*/results.jsonl` (largest by wc -l), counts `first_candidate_correct` + `is_correct`, sums `cost_usd` (should be 0 for vllm); Edit tool inserts rows; `cd Publication/paper && bash compile.sh`; visual inspection of PDF

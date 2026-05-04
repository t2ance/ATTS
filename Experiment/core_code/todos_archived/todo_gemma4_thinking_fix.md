# TODO: Gemma-4 thinking double-bug fix end-to-end

## What this is

`google/gemma-4-26B-A4B-it` served on vLLM 0.20.x produces empty `reasoning_content` and the cached `output.md` contains only the final JSON (no thinking trace). Diagnosed today as two stacked bugs: (a) HF `chat_template.jinja` `enable_thinking=true` branch only injects `<|think|>` into the system turn but does NOT prefill `<|channel>thought\n` at the model-turn end, so the IT-tuned weights never open the thinking channel; (b) vLLM 0.20.x detokenizes with `skip_special_tokens=True` by default, stripping `<|channel>` / `<channel|>` markers before `Gemma4ReasoningParser` runs string-matching, so even when thinking is emitted the parser returns `reasoning_content=None`.

This run patches both layers and validates end-to-end that a precached HLE explore on Gemma-4 lands `output.md` with a non-empty `<think>...</think>` block followed by the schema-constrained JSON answer.

## Output target

- Patched files (committed to current branch):
  - `scripts/gpqa/grpo/tool_chat_template_gemma4_fixed.jinja` (new)
  - `scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp4.sh` (modified: `--chat-template` flag + comment)
  - `backends/vllm.py` (modified: `skip_special_tokens=false` + `reasoning_content` plumbing in both `call_sub_model` and `run_tool_conversation`)
- Reference artifact: one HLE explore at `../analysis/cache/hle/gemma4_26b_a4b_it/gold/<qid>/explore_<i>/output.md` containing thinking trace + JSON.
- Validation evidence: 20-row smoke batch with median `reasoning_content` length ≥ 300 chars, zero `<channel>` marker leakage into final `content`.

## Discipline

Every event has Gates with checkboxes; flips ☐→✓ only after all Gates pass AND each Gate's `Evidence ·` line is filled with a concrete measurement (numeric value, qid, line number, log line ref); no silent skipping; no narrative-only claims like "looks fine"; on-fail remediation is executed before retrying the Gate.

## Local-box GPU topology (HARD constraint)

- GPU 0: claude embedding daemon (~1.37 GB resident); do not co-locate.
- GPU 1, 2: free; can be used for vLLM serve at TP/DP scaling.
- GPU 3: blocked by another user.

The current Gemma-4 serve uses `CUDA_VISIBLE_DEVICES=0,1,2,3 --data-parallel-size 4`. The restart in Phase 2 must respect "GPU 0 has the daemon" — keep DP=4 only if the daemon's 1.37 GB does not push GPU 0 past the 0.95 gpu-memory-utilization ceiling. If it does, fall back to `CUDA_VISIBLE_DEVICES=1,2 --data-parallel-size 2` and document the change.

## Cache discipline

The Phase-4 single-question test MUST run on a NEW qid (one not currently in `../analysis/cache/hle/gemma4_26b_a4b_it/gold/`) OR explicitly delete one explore's cache directory before launching. Do not pollute the existing cache with mixed-protocol entries — old entries lack `<think>` blocks while new entries have them, and downstream `load_cached_candidates` cannot distinguish.

After Phase 5 succeeds, the existing 800-row HLE cache (`gemma4_26b_a4b_it/gold/`) is now stale-shaped (no thinking traces). Decision deferred to user: either rerun Phase 6 to repopulate or accept mixed-shape cache as historical.

## Resume / restart procedure

| Failure point | Recover by | Banner verification |
|---|---|---|
| Phase 2 serve fails to come up | `pgrep -af 'VLLM::EngineCore'` to find orphans, kill, retry | `curl http://localhost:8000/v1/models` returns 200 with `gemma4-26b-a4b-it` |
| Phase 3 client patch breaks existing tests | `git diff backends/vllm.py` review, isolate the minimal change | `python -c "from backends.vllm import call_sub_model"` imports clean |
| Phase 4 single-question test produces empty `<think>` | Re-run the diagnostic curl from `troubleshooting.md` (manual prefill via `/v1/completions`); compare against chat-completions response to localize which layer regressed | Diagnostic curl shows `<channel\|>` count = 1; chat-completions response shows `reasoning_content` non-empty |
| Phase 5 batch shows >5% timeout rate | Inspect timeout-row token counts; if thinking traces are pushing past `max_model_len=65536`, raise `max_tokens` in the YAML or shorten `--max-model-len` headroom | log shows median `output_tokens < 0.7 × max_tokens` |

## Risk register

| # | Failure | Root cause | Defense |
|---|---|---|---|
| R1 | Fixed jinja prefills `<|channel>thought\n` but vLLM rejects the template at startup | Jinja syntax error or missing macro | Phase 1 G2: `/tokenize` returns 200 with expected token sequence before serve restart |
| R2 | Serve comes up but `--chat-template` silently ignored | vLLM precedence bug or path typo | Phase 2 G3: serve log contains `chat_template` line referencing the new jinja path |
| R3 | `skip_special_tokens=false` propagated to client but parser still returns None | vLLM 0.20.x parser internal strip | Phase 4 G1: `<channel\|>` literally appears in raw `message.content` when `skip_special_tokens=false` is set; `reasoning_content` is non-empty |
| R4 | Thinking traces explode token usage, timeout rate jumps | Long-tail thinking exceeds `max_tokens=32000` | Phase 5 G4: timeout rate ≤ 5% (current baseline); if higher, raise `max_tokens` per-yaml |
| R5 | `reasoning_content` non-empty but `<channel>` markers leak into `content` (parser bug) | Layer-2 fix incomplete | Phase 5 G5: zero rows where `'<\|channel>' in content` or `'<channel\|>' in content` after fix |
| R6 | Pass@1 on first 20 rows drifts > ±10pp from existing 800-row cache | Thinking-mode answers change the model's distribution | Phase 5 G6: same-qid spot-check (5 qids both in old cache and new run), Pass@1 within ±2 per 5 rows |
| R7 | Wall-clock for 20 rows > 4× expected | DP=4 GPU saturation issue or per-card OOM trigger | Phase 5 G3: per-card power ≥ 80% TDP × ≥ 80% wall-time |

## Co-monitor — log paths for parallel watching

| Phase | Run log | Power log |
|---|---|---|
| Phase 2 | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/vllm_serve_gemma4_26b_a4b_dp4.log` | `nvidia-smi --query-gpu=index,power.draw --format=csv -l 5` |
| Phase 4 | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/precache_gemma4_thinking_smoke_n1.log` | (single GPU spike, watch GPU 0/1/2/3) |
| Phase 5 | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/precache_gemma4_thinking_smoke_n20.log` | `nvidia-smi --query-gpu=index,power.draw --format=csv -l 5 > tmp/power_phase5.log` |
| Phase 6 | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/precache_gemma4_thinking_full_hle.log` | `tmp/power_phase6.log` |

---

## Phase 1 — Build & validate fixed chat_template [3 done / 3]

01 ✓ Author the fixed `tool_chat_template_gemma4_fixed.jinja`
   ├ G1 ✓ Gate · File written at `/data3/peijia/dr-claw/Explain/Experiment/core_code/scripts/gpqa/grpo/tool_chat_template_gemma4_fixed.jinja`. Inspect HF default `chat_template.jinja` (line 178-185 system-turn block, 348-352 model-turn block); modify so `enable_thinking=true` also emits `<|channel>thought\n` at the model-turn end (mirroring the `enable_thinking=false` empty-block prefill but leaving the channel OPEN).
   │      Evidence · 354-line file written; line 350-355 patched: HF default's single `if not enable_thinking` branch replaced with explicit `if enable_thinking → '<|channel>thought\n'` else `'<|channel>thought\n<channel|>'`. Diff = 4 lines net inserted.
   ├ G2 ✓ Gate · Inline comment in the jinja file states: (a) what was changed vs HF default, (b) why (link to vllm#39130 / model card / today's diagnostic), (c) the file's coupling — "if HF chat_template upstream changes, re-diff against this fork". Per `comment_on_config_overrides.md` rule.
   │      Evidence · Comment block at line 350: states all three (what changed: only-prefilled-on-false → both-branches-prefill; why: IT-tuned weights' first-token logit at <|turn>model\n prefers text token over <|channel>; coupling: re-diff on every Gemma-4 model upgrade). Reference vllm#39130 + troubleshooting.md included.
   └ How  · Read `/data1/peijia/hf_cache/models--google--gemma-4-26B-A4B-it/snapshots/4c55b528bdc40b4e79ed7fd4e2f8e46fa5aaed5a/chat_template.jinja` first; copy + minimal patch.

02 ✓ Validate fixed jinja via local jinja2 render (serve still on OLD template; per-request `chat_template` override only on chat-completions, not /tokenize, so local render is the cleanest verification)
   ├ G1 ✓ Gate · Render with `enable_thinking=True` ends in `<|turn>model\n<|channel>thought\n` (channel OPEN, no closing token). Verified via `repr(out[-50:])`.
   │      Evidence · last 50 chars repr: `'7 times 23?<turn|>\n<|turn>model\n<|channel>thought\n'` — matches expected token sequence `[<|turn>, model, \n, <|channel>, thought, \n]`. NO trailing `<channel|>`.
   ├ G2 ✓ Gate · Render with `enable_thinking=False` ends in `<|turn>model\n<|channel>thought\n<channel|>` (channel CLOSED — HF default behavior preserved). Same with no kwarg passed (default false).
   │      Evidence · enable_thinking=False render last 50: `'?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>'`. No-kwarg render last 50: identical. Default-false branch unchanged.
   └ How  · `conda run -n explain python` jinja2 render at the local file; output captured to `tmp/phase1_tokenize_evidence.txt`.

03 ✓ Reference render evidence captured to file
   ├ G1 ✓ Gate · `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/phase1_tokenize_evidence.txt` exists, contains both renders' full repr() output + last-50-char tails for both branches.
   │      Evidence · File written at /data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/phase1_tokenize_evidence.txt; contains 3 renders (enable_thinking=True, False, missing-kwarg-default) with full text + tails labeled.
   └ How  · Tee'd from the python render block above.

## Phase 2 — Restart vLLM serve with fixed template [0 done / 3]

04 ☐ Patch `serve_gemma4_26b_a4b_dp4.sh` to add `--chat-template <fixed-jinja-path>`
   ├ G1 ☐ Gate · `git diff scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp4.sh` shows exactly one inserted `--chat-template` line + one comment block above it explaining (a) which file, (b) why (Layer 1 fix for vllm#39130-class bug), (c) coupling — "must be re-diffed against HF default chat_template.jinja on every Gemma-4 model upgrade". Per `comment_on_config_overrides.md`.
   │      Evidence · 
   └ How  · Edit; commit-pending.

05 ☐ Bring down old serve, bring up new serve
   ├ G1 ☐ Gate · `pgrep -af 'VLLM::EngineCore'` returns empty AFTER kill (5-sec window post-`kill`); wait until process tree fully drained before launching new serve.
   │      Evidence · 
   ├ G2 ☐ Gate · New serve PID + log path printed; `tail -f` shows engine-core ready within 90s (no flash-infer JIT compile delay since weight cache exists).
   │      Evidence · 
   ├ G3 ☐ Gate · Serve startup log contains a line referencing the fixed jinja path — search `grep -i "chat.template\|tool_chat_template" tmp/vllm_serve_gemma4_26b_a4b_dp4.log`. If absent, `--chat-template` was silently ignored (R2).
   │      Evidence · 
   └ How  · `pkill -f 'vllm serve google/gemma-4'` (case-sensitive — uppercase "VLLM" only in title); after drain, `bash scripts/gpqa/grpo/serve_gemma4_26b_a4b_dp4.sh`.

06 ☐ Smoke-test serve responsiveness post-restart
   ├ G1 ☐ Gate · `curl http://localhost:8000/v1/models` returns 200 with `id: gemma4-26b-a4b-it`.
   │      Evidence · 
   ├ G2 ☐ Gate · One-shot `curl /v1/chat/completions` with `enable_thinking=true` + `skip_special_tokens=false` + simple prompt returns `reasoning_content` (non-empty string OR null — null only acceptable if also `<channel|>` count in `content` ≥ 1, indicating chat-completions path drops parsed reasoning differently).
   │      Evidence · 
   └ How  · Same curl as today's TEST 17 in `troubleshooting.md` "Diagnostic procedure".

## Phase 3 — Patch backends/vllm.py to plumb reasoning_content [0 done / 3]

07 ☐ Modify `backends/vllm.py:call_sub_model` to:
   (a) add `extra_body["skip_special_tokens"] = False` (in `_split_sampling_kwargs` returned `extra` dict OR directly at the request site)
   (b) read `getattr(msg, "reasoning_content", None)` after response
   (c) prepend `<think>{reasoning}</think>\n\n` to trajectory when reasoning is non-empty; trajectory still ends in the JSON-fenced answer
   ├ G1 ☐ Gate · `git diff backends/vllm.py` shows ≤ 15 added lines, no defensive `try/except`, no fallback values per `CLAUDE.md` rules. Each non-default override has an inline comment explaining WHY.
   │      Evidence · 
   ├ G2 ☐ Gate · Imports clean: `python -c "from backends.vllm import call_sub_model; print('ok')"` returns "ok" with no warnings.
   │      Evidence · 
   └ How  · Edit; cross-check against today's TEST 17 logic.

08 ☐ Mirror the same plumbing in `backends/vllm.py:run_tool_conversation` (chat-completions multi-turn path used by orchestrator)
   ├ G1 ☐ Gate · Both code paths now read `reasoning_content` and pass it to `writer.write_text(...)` — search `git grep "reasoning_content" backends/vllm.py` returns ≥ 2 hits.
   │      Evidence · 
   └ How  · Same pattern as item 07; do NOT copy-paste — extract the helper if it exceeds 5 lines.

09 ☐ Minimal smoke import + 1-shot call test
   ├ G1 ☐ Gate · A throwaway `tmp/phase3_smoke.py` script invokes `call_sub_model(...)` once on the live serve with a simple HLE-shaped schema; returns `(result_dict, trajectory_str, ...)` where `trajectory_str` contains `<think>` substring AND parses cleanly when split on `</think>`.
   │      Evidence · 
   ├ G2 ☐ Gate · Script runs in < 60s, no traceback in stderr.
   │      Evidence · 
   └ How  · `conda run -n explain --no-capture-output python tmp/phase3_smoke.py 2>&1 | tee tmp/phase3_smoke.log`. Per CLAUDE.md `explain` env discipline.

## Phase 4 — Single-question end-to-end via precache_explores.py [0 done / 2]

10 ☐ Pick one fresh HLE qid that does NOT exist in current cache. Pre-clean its directory.
   ├ G1 ☐ Gate · `ls ../analysis/cache/hle/gemma4_26b_a4b_it/gold/<qid>/` either returns "No such file" OR is explicitly `rm -rf`'d before launch. Record qid.
   │      Evidence · 
   └ How  · `ls ../analysis/cache/hle/gemma4_26b_a4b_it/gold/ | head -10` — pick one NOT yet cached, OR pick a cached one and `rm -rf` its dir.

11 ☐ Run `precache_explores.py` for `num=1` on that qid
   ├ G1 ☐ Gate · Result completeness — `result.json` exists for `explore_0` (or whichever idx), no `timed_out: true`.
   │      Evidence · 
   ├ G2 ☐ Gate · Output integrity — `output.md` for that explore contains a non-empty `<think>...</think>` block (regex `<think>([\s\S]+?)</think>` matches, captured group length ≥ 200 chars).
   │      Evidence · 
   ├ G3 ☐ Gate · Schema validity — JSON answer block is parseable, contains all required keys (`approach`, `reasoning`, `answer`, `confidence`).
   │      Evidence · 
   ├ G4 ☐ Gate · No leakage — `output.md` contains zero literal `<|channel>` or `<channel|>` strings (parser stripped them before write).
   │      Evidence · 
   ├ G5 ☐ Gate · Trajectory ordering — `<think>` block appears BEFORE the JSON answer in `output.md` (semantic ordering, not just both present).
   │      Evidence · 
   └ How  · Construct a minimal precache yaml at `scripts/hle/grpo/hle_gemma4_26b_a4b_thinking_smoke_n1.yaml` (set `num: 1`, point cache_dir at a fresh path like `../analysis/cache/hle/gemma4_26b_a4b_it_thinking_smoke/gold`); run `PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py --config <yaml> > tmp/precache_gemma4_thinking_smoke_n1.log 2>&1 &`. Share PID + abs log path immediately per `feedback_share_long_running_logs`.

## Phase 5 — 20-row smoke batch (sanity + throughput regression check) [0 done / 1]

12 ☐ Run `precache_explores.py` for `num=20` HLE rows on the new cache dir
   ├ G1 ☐ Gate · Result completeness — `wc -l results.jsonl` returns 20 rows; timed_out rows ≤ 1/20 (5%).
   │      Evidence · 
   ├ G2 ☐ Gate · Per-row thinking — 19/20 rows have non-empty `<think>` block (allow 1 row miss for variance); median thinking length across rows ≥ 300 chars.
   │      Evidence · 
   ├ G3 ☐ Gate · Resource utilization — per-card power ≥ 80% TDP across ≥ 80% of wall-clock; `nvidia-smi -l 5` log shows GPU 0/1/2/3 each averaging ≥ 280 W (assuming 350 W TDP).
   │      Evidence · 
   ├ G4 ☐ Gate · Throughput — wall-clock for 20 rows ≤ 4× the 1-row Phase-4 wall-clock (linear scaling baseline, allowing batch overhead).
   │      Evidence · 
   ├ G5 ☐ Gate · No marker leakage — across all 20 `output.md` files, zero contain literal `<|channel>` OR `<channel|>`.
   │      Evidence · 
   ├ G6 ☐ Soft-Gate · Pass@1 sanity vs old cache — pick 5 qids that exist in BOTH old `gemma4_26b_a4b_it/gold` cache (no thinking) AND new thinking cache; compute is_correct delta.
   │      (a) For each of the 5 qids: report old answer, new answer, gold answer, old `is_correct`, new `is_correct`.
   │      (b) Compute |sum(new) − sum(old)| across the 5; expect ≤ 2 (threshold rationale: 5 qids, std dev for binary outcomes is ~√(5 × 0.5 × 0.5) ≈ 1.1, so ≤ 2 is within 2σ).
   │      Justification (required) · 1-2 sentences citing each qid's old/new is_correct + the count delta. Do NOT say "looks consistent."
   │      Evidence · 
   └ How  · Yaml at `scripts/hle/grpo/hle_gemma4_26b_a4b_thinking_smoke_n20.yaml` (`num: 20`, same fresh cache_dir, `num_workers: 8`); run with same command pattern as item 11. Power log to `tmp/power_phase5.log` in parallel.

## Phase 6 — Decision gate: full repopulation or accept mixed cache [0 done / 1]

13 ☐ Report Phase 5 results to user; await decision on whether to wipe and rerun the 800-row HLE cache for Gemma-4 (or accept the existing cache as historical-shape and proceed)
   ├ G1 ☐ Gate · A summary written to `tmp/phase6_decision_brief.md` covering: (a) Phase 5 evidence in 5 bullets, (b) cost estimate of full 800-row repopulation in wall-clock minutes (extrapolated from Phase 5 wall-clock × 40), (c) cost estimate in API/GPU-hours, (d) recommendation with rationale.
   │      Evidence · 
   └ How  · Compose the brief from Phase 4-5 evidence; present to user; do NOT auto-launch full repopulation without explicit user approval.

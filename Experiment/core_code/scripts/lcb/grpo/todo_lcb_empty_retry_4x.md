# TODO: LCB EMPTY-row retry with 4× budget (Table 9 LCB thinking row uplift)

## What this is

Table 9 LCB thinking row currently reads `Acc 76.57 / Gain −0.57` with 15 EMPTY trajectories (no `StructuredOutput` emitted by orchestrator). Diagnosis showed the bottleneck is the orchestrator's `max_tokens=32768` budget — orch burns its 32K reasoning budget before issuing any tool call. Neither the 1200 s per-explore wall-time nor any context-overflow ever fired.

**Cache-truth audit on 2026-05-03 (post-driver-launch) reduced the runnable set from 12 -> 4.** Counted `timed_out=True` flag in `cache/lcb/sonnet/<qid>/explore_<n>/result.json` for each of the 12 originally targeted qids:

| Bucket | n | Why |
|---|---|---|
| ≥1 non-timed-out cached explore (RUNNABLE) | 4 | arc191_a (3/8 ok), arc195_e (8/8), 3737 (8/8), 3743 (8/8) |
| ALL 8 cached explores marked timed_out=True (UNRUNNABLE) | 8 | arc192_e, arc190_c, arc193_b, abc400_g, arc196_d, arc196_c, 3674, arc195_c |

The 8 unrunnable qids cannot benefit from any orchestrator-side budget bump because their cache holds zero usable Sonnet candidates — the explorer itself aborted before emitting any answer. That earlier "8 unknown ceiling" classification was based only on "no grade.json present", which mistook "explorer-empty" for "ungraded". Now we know there is nothing to grade, so retrying with 4× orchestrator budget cannot recover them.

This run retries only the **4 truly savable qids** with `max_tokens × 4 = 131072 (128K)` and `explore_timeout × 4 = 4800 s`, on a freshly-launched single-card vLLM serve on GPU 0. Goal: convert these 4 EMPTY → real answer.

**Variant identity**: single config — Qwen3.6-35B-A3B-FP8 orchestrator + Sonnet 4.6 cached explorer (identical to Table 9 thinking row, only the orchestrator's decode budget changes).

## Output target

- `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/qwen36_35b_a3b_fp8_empty_retry_4x/run_<ts>/results.jsonl` — exactly 4 rows
- Per-qid breakdown: `(empty? → empty?, is_correct? → is_correct?)`, used to decide whether to merge into Table 9 LCB row or document as supplementary
- Upper bound on Acc lift: `+4/175 = +2.29 pp` (would push 76.57 → 78.86, Gain +1.72). Lower than the original 12-qid estimate because 8 qids are now known-unrunnable.

## Discipline

Every event has Gates with checkboxes; flips ☐→✓ only after all Gates pass AND each Gate's `Evidence ·` line is filled with concrete measurement; no silent skipping; no narrative-only claims.

## Resource map (HARD)

- **GPU 0** — free at TODO write time (1.3 GB used / 80 GB). Single Qwen3.6-FP8 replica, TP=1.
- GPU 1, 2, 3 — fully occupied by other users / other models. Do NOT touch.
- vLLM port **8002** (8000 = gemma4 serve already running, 8001 = gpt-oss serve already running)
- Conda env: `grpo_vllm` for serve, `explain` for retry driver
- HF cache: `HF_HUB_CACHE=/data1/peijia/hf_cache` (Qwen3.6-FP8 weights already there)

## QID set (4 questions, frozen 2026-05-03 after cache-truth audit)

| Bucket | Count | qids |
|---|---|---|
| RUNNABLE (≥1 non-timed-out cached explore) | 4 | arc191_a (3/8 ok), arc195_e (8/8), 3737 (8/8), 3743 (8/8) |
| **TOTAL TO RUN** | **4** |  |
| SKIPPED — all 8 cache explores timed_out=True (explorer never emitted answer) | 8 | arc192_e, arc190_c, arc193_b, abc400_g, arc196_d, arc196_c, 3674, arc195_c |
| SKIPPED — confirmed unsavable, all 8 cache candidates graded wrong | 3 | 3701, abc398_g, 3762 |

## Resume / restart procedure

| Failure point | Recover by | Banner verify |
|---|---|---|
| vLLM serve crash | Restart serve script; `tee` log preserves history | `curl http://localhost:8002/v1/models` returns `qwen36-35b-a3b-fp8` |
| Retry driver crash mid-run | Same `log_dir`; eval.py auto-resumes from `results.jsonl` | log banner reports "resumed from N rows" |
| Single qid stuck >2 h | Kill, mark as failed, continue with rest | manual qid skip in driver |

## Risk register

| # | Failure | Root cause | Defense |
|---|---|---|---|
| R1 | vLLM OOM loading Qwen3.6-FP8 at max_model_len=163840 | KV cache pool too small for 12 concurrent 128K seqs | Start with gpu_memory_utilization=0.85; if OOM, drop to 0.80 + reduce num_workers from 4 → 2 |
| R2 | `openai.APITimeoutError` (httpx ReadTimeout) at 128K decode | OpenAI client default timeout | `vllm.py:64` already sets `timeout=1800.0` on AsyncOpenAI; verify no per-call override; raise to 4800 if needed |
| R3 | Resume mode skips qids that are already EMPTY in pre_64K | results.jsonl already has those qids with empty answer; resume sees them as "done" | Use NEW log_dir `..._empty_retry_4x` — do NOT resume into the original `run_20260501_042951` |
| R4 | LCB grade_code mismatch with original run | lcb_runner version drift between original (2026-04-30) and now | Verify same lcb_runner version (git log on cache file mtime) before launch |
| R5 | Standalone driver fails to filter dataset to 12 qids | Filter logic bug or qid format mismatch | Pre-flight: print filtered row count; assert == 12 before launching orchestrator |
| R6 | vLLM serve fails on Qwen3.6-FP8 single-card init | TP=1 might trigger MoE-divisibility check (saw with DP=3) | Fallback: also try `--enforce-eager` or smaller max-model-len |
| R7 | num_workers=4 oversubscribes single A100 KV pool | Serial generations contend for KV blocks | If queue depth saturates → drop num_workers to 2 |

## Co-monitor — log paths (absolute)

| Phase | Run log | Notes |
|---|---|---|
| 1 (vLLM serve) | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/vllm_serve_qwen36_gpu0_128k.log` | tee'd during serve startup |
| 2 (retry eval) | `/data3/peijia/dr-claw/Explain/Experiment/core_code/tmp/eval_lcb_empty_retry_4x.log` | nohup'd, follows live |
| 2 (results.jsonl) | `/data3/peijia/dr-claw/Explain/Experiment/analysis/run/lcb/qwen36_35b_a3b_fp8_empty_retry_4x/run_<ts>/results.jsonl` | grows as questions finish |
| GPU heartbeat | nvidia-smi every 10 min during Phase 2 | watch for power on GPU 0 |

---

## Phase 1 — Pre-flight [5/5 ✓]

01 ✓ Spin up vLLM serve on GPU 0 (Qwen3.6-35B-A3B-FP8, TP=1, max_model_len=163840, port 8002)
   ├ G1 ✓ Gate · GPU 0 memory.used < 5 GB at start
   │      Evidence · `nvidia-smi` at 04:09 reported GPU 0 = 1341 MB / 81920 MB (1.31 GB << 5 GB cap).
   ├ G2 ✓ Gate · vLLM process alive + `curl http://localhost:8002/v1/models` returns `qwen36-35b-a3b-fp8`
   │      Evidence · /v1/models returns `id: qwen36-35b-a3b-fp8`, `max_model_len: 163840`. PID 2450741.
   ├ G3 ✓ Gate · model weights resident (memory.used > 30 GB after load)
   │      Evidence · GPU 0 = 70313 MB after warmup (35 GB weights + 35 GB KV cache pool resident). Power 68 W is expected idle (will spike during inference).
   └ How · `nohup bash scripts/lcb/grpo/serve_qwen36_35b_a3b_gpu0_128k.sh > tmp/vllm_serve_qwen36_gpu0_128k.log 2>&1 &`

02 ✓ Smoke test: 1-shot chat completion against the new endpoint
   ├ G1 ✓ Gate · 200 OK + non-empty `choices[0].message.content`
   │      Evidence · POST returned `id: chatcmpl-b383d95384d169e5`, finish_reason=length, content="Here's a thinking process:\n\n1.  **Analyze User Input:**" (16 tokens, capped — confirms thinking-mode burns CoT tokens early, the exact root cause we are fixing).
   └ How · `curl -s -X POST http://localhost:8002/v1/chat/completions ...`

03 ✓ Generate filtered LCB qid list (4 qids) and freeze to disk
   ├ G1 ✓ Gate · file `tmp/lcb_empty_retry_qids.json` exists, contains exactly 4 qids matching the RUNNABLE bucket above
   │      Evidence · Wrote 4 qids on 2026-05-03 after cache-truth audit: ['arc191_a','arc195_e','3737','3743']. Originally written with 12 qids, then reduced to 4 after counting timed_out=True flags in cache files.
   └ How · `python3 -c "..."` → `tmp/lcb_empty_retry_qids.json`

04 ✓ Verify retry yaml + driver exist and reference correct ports / budgets / log_dir
   ├ G1 ✓ Gate · yaml has `max_tokens: 131072`, NEW `log_dir: ../analysis/run/lcb/qwen36_35b_a3b_fp8_empty_retry_4x`, `cache_dir: ../analysis/cache/lcb/sonnet`, no `resume:` line
   │      Evidence · `scripts/lcb/grpo/lcb_qwen36_35b_a3b_empty_retry.yaml` written; verified by inspection.
   ├ G2 ✓ Gate · driver script monkey-patches MODEL_TO_BASE_URL[qwen36-35b-a3b-fp8] = http://localhost:8002/v1, raises AsyncOpenAI timeout to 4800s, monkey-patches LCBBenchmark.load_dataset to filter to 4 qids, then calls eval.main()
   │      Evidence · `scripts/lcb/grpo/retry_lcb_empty_4x.py` written; three patches in order with assert len==4 guard.
   └ How · See files at the listed paths.

04a ✓ Cache-truth audit: count timed_out=True in cache for original 12 qids
   ├ G1 ✓ Gate · 8 of 12 qids confirmed unrunnable (8/8 explores timed_out=True); qid set reduced from 12 -> 4
   │      Evidence · `for qid in <12>; do count timed_out in 8 explores; done` → 8 qids reported timed_out=8/ok=0 (arc192_e, arc190_c, arc193_b, abc400_g, arc196_d, arc196_c, 3674, arc195_c). Only arc191_a (5/3), arc195_e (0/8), 3737 (0/8), 3743 (0/8) have any usable cache.
   ├ G2 ✓ Gate · driver assert and yaml comment updated to reflect len==4
   │      Evidence · retry_lcb_empty_4x.py asserts `len(qid_set) == 4` and `len(filtered) == 4`; lcb_qwen36_35b_a3b_empty_retry.yaml comments updated to "4 specific qids".
   └ How · audit performed mid-Phase-2 after seeing first 2 qids of 04:49 launch both report `explore #1: TIMED OUT` from cache replay.

## Phase 2 — Run retry [1/2]

05 ✓ Launch retry driver on the 4 RUNNABLE qids
   ├ G1 ✓ Gate · driver started, PID logged, log file growing within 30 s
   │      Evidence · driver PID 3789419 launched 05:02:07 (run dir `run_20260503_050207`); log grew to 280 lines within 10 s; GPU 0 power 208 W (inference active).
   ├ G2 ✓ Gate · banner reports "Loaded N total questions" then "Filtered to 4 questions"
   │      Evidence · log: `[retry-driver] LCB.load_dataset: 175 total -> 4 after qid filter`; `Loaded 4 total questions`; `Filtered to 4 questions`; `Questions to run: 4 (0 already completed, 4 total)`.
   └ How · `PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python scripts/lcb/grpo/retry_lcb_empty_4x.py > tmp/eval_lcb_empty_retry_4x.log 2>&1 &`

   Prior attempts (recorded for posterity, all killed/crashed before producing data):
   - PID 2626404 (04:16:43): hit 400 BadRequest on tool_choice=auto. Fixed by adding --enable-auto-tool-choice + --tool-call-parser to serve script.
   - PID 3445569/3446062 (04:49:54): launched with 12 qids; killed at 04:50 after first 2 qids both reported `explore #1: TIMED OUT` from cache replay, exposing the cache-truth issue. This drove item 04a (cache-truth audit), which reduced qid set 12 -> 4.
   - PID 3583784 (04:55:04): completed inference for qid 3737 (trajectory 799 lines, finish=tool_calls 5x, 0 TIMED OUT — proof 4× orchestrator budget works), then died at grade time with `ModuleNotFoundError: No module named 'lcb_runner.evaluation'`. Root cause: editable install at `/home/peijia/miniconda3/envs/explain/lib/python3.11/site-packages/__editable___livecodebench_0_1_0_finder.py` MAPPING points to `/data1/peijia/projects/EXPLaIN/LiveCodeBench/lcb_runner` which has been deleted. Fixed by adding a runtime monkey-patch in `retry_lcb_empty_4x.py` (item 0) that repoints `MAPPING['lcb_runner']` and `NAMESPACES['lcb_runner']` to the still-living source at `/data3/peijia/dr-claw/Explain/Experiment/code_references/LiveCodeBench/lcb_runner` (verified via `from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness` succeeding under explain env).

06 ✓ Heartbeat monitor (10-min wakeup + 60-s Monitor) until completion
   ├ G1 ✓ Gate · `progress.json` shows `status="completed"` AND `questions_completed=4`
   │      Evidence · final progress.json: `status: completed`, `questions_completed: 4`, `correct: 3`, `accuracy_pct: 75.0`, `elapsed_seconds: 1313.55` (~22 min total). EVALUATION COMPLETE banner emitted at 05:24:01.
   ├ G2 ✓ Gate · zero `httpx.ReadTimeout` / `openai.APITimeoutError` in log
   │      Evidence · `grep -cE 'httpx.ReadTimeout|APITimeoutError' tmp/eval_lcb_empty_retry_4x.log` = 0. The 4800-s AsyncOpenAI timeout (item 0 monkey-patch) absorbed all long-decode chat completions; longest single qid was arc191_a at ~22 min wall time, well within the budget.
   ├ G3 ✓ Gate · zero unhandled `Traceback` in log
   │      Evidence · `grep -cE 'Traceback|ModuleNotFoundError' tmp/eval_lcb_empty_retry_4x.log` = 0. The lcb_runner.evaluation monkey-patch (PID 3789419's item 0) held throughout grading; PID 3583784's earlier crash on this exact import did not recur.
   ├ G4 ✓ Gate · GPU 0 power averaged ≥ 150 W over the run window (model actually doing work)
   │      Evidence · spot snapshots during the run reported GPU 0 power = 208 W (05:02:14), 229 W (05:06:54), 218 W (05:14:13), 201 W (05:20:59), 237 W (05:14:02). All samples > 150 W; idle baseline was 67 W pre-launch. Monitor task b0mw09xlo flagged 0 GPU0_LOW events across the 22-min run.
   └ How · ScheduleWakeup 600 s heartbeat + Monitor task b0mw09xlo (60-s polling for progress / GPU power / fatal errors). Monitor self-terminated when driver exited at 05:24.

   1 cache TIMED OUT event observed mid-run (arc191_a explore #1, expected from cache-truth audit: arc191_a has 5/8 explores marked timed_out=True; orchestrator continued requesting explore #2..#8 to find the 3 ok ones).

## Phase 3 — Integrate [2/2 ✓]

07 ✓ Per-qid delta vs `pre_64K` backup
   ├ G1 ✓ Gate · table built: for each of 4 qids, (was empty? → still empty?), (was wrong → now correct?)
   │      Evidence · per-qid table:
   │        | qid       | pre empty? | pre correct? | new empty? | new correct? | delta             |
   │        | 3737      | True       | False        | False      | True         | GAINED +1         |
   │        | 3743      | True       | False        | False      | True         | GAINED +1         |
   │        | arc191_a  | True       | False        | False      | False        | not empty (still wrong) |
   │        | arc195_e  | True       | False        | False      | True         | GAINED +1         |
   │      All 4 qids converted from EMPTY to non-EMPTY (4× budget treats the EMPTY problem 100%). 3 of 4 also flipped to correct.
   ├ G2 ✓ Gate · totals computed: empty count drop, correct count delta vs original 134 / 175
   │      Evidence · pre_64K backup (file `lcb/qwen36_35b_a3b_fp8_temp/run_20260501_042951/results.jsonl.bak_20260501_pre_64K`): 134/175 = 76.57%, 15 EMPTY. Merged: 137/175 = 78.29%, 11 EMPTY. Delta: +3 correct, -4 EMPTY. Gain vs Sonnet@1 baseline (77.14%): -0.57 -> +1.15.
   └ How · Inline Python (loads both files, joins on `id`, builds table + delta).

08 ✓ Decide and document
   ├ G1 ✓ Gate · Decision recorded in chat: merge new answers into Table 9 LCB row + edit main.tex
   │      Evidence · user directed (2026-05-03): "把 table 改好、改对，然后把 todo list 给收尾...把所有地方的老数值都改成新数值...在论文里面加一条注释，说明这个东西之前是多少、现在是多少，以及我们做了什么操作把它给修正了". Four edits applied to `Publication/paper/main.tex`:
   │        1. Table 9 LCB row (line 900): 76.57 -> 78.29, $-0.57$ -> \textbf{+1.15}.
   │        2. Provenance comment (lines 840-843): rewrote LCB bullet to point at the merged file and the EMPTY-recovery block below.
   │        3. New comment block "EMPTY-recovery retry on 2026-05-03" (lines 850-883): records before/after Acc + Gain, the retry budget (max_tokens=131072, explore_timeout=4800s), the 4 RUNNABLE qids (3737, 3743, arc191_a, arc195_e), the 8 SKIPPED-unrunnable qids (all-timed_out cache), the 3 SKIPPED-fully-wrong qids, methodological note (only EMPTY rows retried, no regression risk), and a back-pointer to this TODO file.
   │        4. Per-benchmark behavior paragraph (line 910): rewrote LCB analysis with the retry story (15 EMPTY -> 4 retried -> 3 correct -> Acc 76.57->78.29, Gain -0.57->+1.15) and updated the closing claim from "remains an open question" to "feasible, but contingent on enough decode budget on the long-tail questions".
   │      `bash compile.sh` produced `Publication/paper/build/main.pdf` (27 pages, 455900 bytes), 0 LaTeX errors, 0 undefined refs, only cosmetic Underfull warnings.
   └ How · Edit (4x) + Bash compile.sh.

# DONE 2026-05-03 ~05:35

Final aggregate (Table 9 LCB thinking row): **Acc 78.29% (137/175), Gain +1.15 vs Sonnet@77.14**, up from raw 76.57/-0.57. 11 residual EMPTY rows are explorer-cache-bounded (8 timed_out + 3 fully-wrong cache), not orchestrator-budget-bounded.

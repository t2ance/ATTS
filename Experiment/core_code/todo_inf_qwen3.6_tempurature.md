# TODO -- Qwen3.6-35B-A3B-FP8 `_temp` Variant (Thinking + Custom Sampling)

**Variant of:** `todo_inference_qwen3.6.md` (the `_baseline` greedy run).
**Only differences from `_baseline`:**

| Knob | `_baseline` | `_temp` |
|---|---|---|
| `enable_thinking` | False | **True** |
| `temperature` | 0.0 (greedy) | **1.0** |
| `top_p` | -- | **0.95** |
| `top_k` | -- | **20** |
| `min_p` | -- | **0.0** |
| `presence_penalty` | -- | **1.5** |
| `repetition_penalty` | -- | **1.0** |
| `max_tokens` | (default) | **65536 (64k)** |
| Run dir suffix | `_baseline` | `_temp` |

Goal: see whether thinking-on + Qwen-recommended sampling materially changes the
orchestrator behavior (especially the LCB "free-form CoT, never call explore"
failure). Same explorer cache (Sonnet), same `tts-agent`, same `num_explores=8`,
same `no_integrate=true`. Only the orchestrator decoding changes.

---

## M0 -- Plumb sampling params through vLLM backend (DONE, commit 269994a)

| File | Change | Status |
|---|---|---|
| `backends/vllm.py` | `_split_sampling_kwargs(s)` -> (direct, extra_body); `run_tool_conversation` accepts `sampling: dict | None`; `max_tokens=8192` hardcode dropped (now setdefault fallback) | DONE |
| `eval.py` | `SamplingConfig` Pydantic (extra=forbid); `EvalConfig.sampling`; `evaluate(...sampling=...)`; `process_question` -> `solve_fn(sampling=...)` | DONE |
| `methods/tts_agent.py` | `solve` and `_run_orchestrator` thread `sampling` to backend | DONE |
| 4 `_temp.yaml` + 4 `_temp.sh` | HLE/GPQA/LCB/BabyVision launchers carry the Qwen3.6 thinking recipe | DONE |
| Paper `main.tex:832` | dropped "extended-thinking is disabled" claim (was inconsistent with code) | DONE |

**Hotfix (commit e0ac067):** initial yaml set `max_tokens=65536`, equal to vLLM's
`--max-model-len=65536` total ceiling -> `400 BadRequest: maximum input length
of 0 tokens` on every call. HLE smoke caught it at question 12 on 2026-05-01
00:47. Lowered to **32768** in all 4 yamls; reserves ~32k for prompt + 8-turn
accumulated context.

**Smoke verification:** 10k-char prompt + max_tokens=32768 + full Qwen3.6
thinking sampling block returned `prompt_tokens=2024 completion_tokens=213
finish_reason=stop`. End-to-end transport confirmed: `direct_kwargs` =
{temperature, top_p, presence_penalty, max_tokens}; `extra_body` =
{top_k, min_p, repetition_penalty, chat_template_kwargs.enable_thinking}.

---

## M1 -- HLE (gold, text_only:100) -- DONE

Wall: 2026-05-01 00:48:45 -- 01:14:51 (~26 min, vs `_baseline` 14 min greedy).
Run dir: `analysis/run/hle/qwen36_35b_a3b_fp8_temp/run_20260501_004845`.

**Result:** 56/100 = **56.00%** integrated, **no judge correction needed** (only
2 empty-predicted rows, judge graded both wrong as designed -- post-2026-04-30
judge prompt patch is doing its job). Pass@1 = 36.00 (same Sonnet cache);
**Gain = +20.00pp** vs `_baseline` Gain of +2.00pp (10x).

| Metric | `_baseline` (greedy) | `_temp` (thinking) | Delta |
|---|---:|---:|---:|
| Single Acc | 48.00 | 48.00 | -- |
| Integrated Acc | 48.00 (raw -> 38.00 corrected) | 56.00 | **+8 / +18** |
| Pass@1 | 36.00 | 36.00 | -- |
| Gain (Acc - Pass@1) | +2.00 | **+20.00** | +18 |
| 0-explore questions | 26 | **1** | -25 |
| 8-explore questions | 9 | 11 | +2 |
| Cost ($/q) | 1.38 | 2.53 | +1.15 |
| Wall (min) | 14.2 | 26.1 | +11.9 |

**Key finding:** 0-explore drops from 26 -> 1. The `_baseline` pathology of
"orchestrator emits free-form CoT without calling explore" (paper line 838,
diagnosed for LCB) is **fixed by thinking-on** on HLE. Whether the same fix
generalizes to LCB is the next test.

---

## M2 -- GPQA-Diamond (full, N=198) -- DONE

Wall: 2026-05-01 01:16:15 -- 01:38:10 (~22 min total, including KeyError
crash at q116 + resume from q116).

**Result:** 165/198 = **83.33%** integrated, vs `_baseline` 71.21% = **+12.12pp**.
Empty predicted: 1, judge correctly graded wrong (no manual subtraction).
Patch 9c840fc triggered 0 times in 198 rows -- q116 schema-violation was an
outlier; resume after patch sailed through.

| Metric | `_baseline` (greedy) | `_temp` (thinking) | Delta |
|---|---:|---:|---:|
| Single Acc | 74.74 | 74.74 | -- |
| Integrated Acc | 71.21 | **83.33** | **+12.12** |
| Best-of-+agg | 90.40 | 91.41 | +1.01 |
| Per-subset Biology | 68.42 | 78.95 | +10.53 |
| Per-subset Chemistry | 68.82 | 73.12 | +4.30 |
| Per-subset Physics | 82.56 | **95.35** | **+12.79** |
| Empty predicted | n/a | 1 (judge correctly wrong) | -- |
| Cost ($/q) | 0.29 | 0.38 | +0.09 |

**Transport heterogeneity disclaimer:** rows 1-115 ran on pre-patch code
(no `_THINK_PREFIX_RE` strip, no schema check); rows 116-198 ran on
post-patch code. Mixed-transport result. Decision-grade for paper: usable
with disclosure (~$30 of explorer spend would be redone if reverting to
clean transport).

---

## M3 -- LCB (LiveCodeBench, N=175) -- DONE

Wall: 2026-05-01 04:29 -- 06:27 (across 3 process restarts; see Restarts below).
Run dir: `analysis/run/lcb/qwen36_35b_a3b_fp8_temp/run_20260501_042951`.

**Result:** 134/175 = **76.57%** integrated, vs `_baseline` 44.00% = **+32.57pp**.
Empty predicted: 15 (8.6%) vs `_baseline` 92 (52.6%) = **-44pp**. The LCB
negative-Gain failure (`-1.14` on `_baseline`) is fully reversed: Gain
= `76.57 - 45.14 = +31.43`.

| Metric | `_baseline` (greedy) | `_temp` (thinking) | Delta |
|---|---:|---:|---:|
| Pass@1 (cache, first cached Sonnet) | 77.14 | 77.14 | -- |
| Integrated Acc | 44.00 | **76.57** | **+32.57** |
| Empty predicted | 92 (52.6%) | 15 (8.6%) | **-44pp** |
| Gain (Acc - Pass@1, ec0) | -33.14 | **-0.57** | +32.57 |
| Per-subset Easy (n=42) | -- | 100.00 | -- |
| Per-subset Medium (n=42) | -- | 90.48 | -- |
| Per-subset Hard (n=53) | -- | 67.92 | -- |
| Cost ($/q) | 0.12 | 0.48 | +0.36 |

**Pass@1 definition note**: paper appendix `tab:qwen36-baseline` uses the *cache-only*
Pass@1 (mean of `explore_candidates[0].is_correct` across all 175 rows = 77.14%).
The earlier number `45.14` came from `first_candidate_correct` (per-row field) which
includes orchestrator-routing effects (84 questions skipped explore entirely under
greedy → counted as wrong). Standardized to `ec0` after 2026-05-01 audit so the
Pass@1 baseline is identical across decoding configs and the Gain column directly
isolates the orchestrator's contribution.

**Restarts (3 SIGTERM-class events resolved):**
1. process killed near `/compact` boundary at 137/175 (no traceback, no OOM,
   clean SIGTERM); 9 empty rows written by cancelled async tasks. Filtered
   out before resume so they would be retried.
2. resume crashed at 165/175 with
   `AssertionError: cache_only mode: cache miss at .../arc193_b/explore_9/result.json`.
   Root cause: `run_explore` had no quota guard, so on long-tail thinking-mode
   problems the orchestrator could request `explore_idx > num_explores=8`,
   bypassing the `ExploreStepState.is_exhausted` invariant before the
   `cache_only` assertion fired. Patch: insert `is_exhausted` check at the
   entry of `methods/tts_agent.py:run_explore` returning a
   "Explore quota exhausted, submit_answer now" string instead of issuing
   the cache lookup. The patch leaves `tool_state.advance` and the frozen
   `call_count <= max_explores` invariant untouched -- this is purely a
   transport-level guard.
3. resume after the patch ran cleanly to 175/175.

**Explore distribution (`_temp`):**
`{0:6, 1:16, 2:106, 3:28, 4:7, 5:5, 6:1, 7:3, 8:3}` -- 3 questions hit the
`num_explores=8` ceiling (in `_baseline` the maximum was 7). The quota guard
above is what kept these from crashing the run.

**Key finding:** thinking-on reverses the LCB pathology described in
`main.tex:838` -- the "free-form CoT, never call explore" failure mode is
no longer the dominant exit. The 8.6% empty rate that remains is a
thinking-mode long-tail (model rambles past `max_tokens=32768` without
emitting `StructuredOutput`) on the hardest questions, not the original
schema-rejection bug fixed by patch 9c840fc.

---

## M4 -- BabyVision (N=388) -- DONE (v1: pre-fix), DONE (v2: post-fix resume)

### v1 (pre-fix, 2026-05-01 06:44 -- 07:06, ~22 min, run_20260501_064415)

53/388 = 13.66 % Acc. Pass@1 (cache ec0) = 76/388 = 19.59 %. Gain = -5.93 pp.
**Empty predicted: 152/388 = 39.2 %** -- dominant failure mode at this point.
Per-subset Acc: Fine-grained Discrim. 11.66 (19/163), Visual Tracking 16.87
(14/83), Spatial Perception 14.29 (13/91), Visual Pattern Recog. 13.73 (7/51).
Cost $0.26/q. Explore distribution `{0:39, 1:1, 2:102, 3:98, 4:33, 5:26, 6:32,
7:28, 8:29}` -- mean ~4.10, double the LCB mean of 2.38.

The 39.2 % empty rate was traced ~07:30 to a prompt-instruction conflict in
`benchmarks/babyvision.py:72`, which appended `\boxed{Answer}` to the user
prompt and conflicted with the system-level `StructuredOutput` requirement.
Of the 152 empty rows, 142 wrote `\boxed{X}` into the trajectory but never
called SO -- not truncation, instruction conflict.

### v2 (post-fix resume, 2026-05-01 07:41 -- 07:55, ~14 min, same run_dir)

Three-layer fix landed before this resume:
1. `benchmarks/babyvision.py:72` -- `\boxed{Answer}` instruction removed,
   replaced with explicit "the SO tool call is the only accepted submission
   path" directive.
2. `backends/vllm.py:378-403` -- system prompt opens with an `EXTREMELY
   IMPORTANT` all-caps block declaring the SO call as the only valid
   submission path and explicitly rejecting `\boxed{X}` / `Answer:` /
   `The answer is`.
3. `babyvision_qwen36_35b_a3b_temp.yaml` -- bumped `max_tokens` 32768 ->
   65536; `serve_qwen36_35b_a3b_3replica.sh` `--max-model-len` 65536 ->
   131072 to absorb the larger budget.

Resume retried only the 152 empty rows; the 236 non-empty pre-fix rows were
preserved.

**v2 result:** 53/388 = **13.66 %** Acc (UNCHANGED), Empty = **3/388 = 0.77 %**
(down from 152). Cost $101.72 /run, $0.26/q. Explore distribution
`{0:18, 1:1, 2:109, 3:102, 4:32, 5:30, 6:32, 7:38, 8:26}`.

| Metric | v1 pre-fix | v2 post-fix |
|---|---:|---:|
| Acc (orch integrated) | 13.66 | **13.66** (no change) |
| Empty predicted_answer | 152 (39.2 %) | **3 (0.77 %)** |
| Trajectory contains `\boxed` skip | 142 | **0** |
| Gain (Acc - Pass@1 ec0) | -5.93 | **-5.93** |

The fix worked at the level it was designed for (SO-skip eliminated) but Acc
did not improve, which forced a deeper audit (next subsection).

### Why post-fix Acc did NOT rise -- cache-bound ceiling (audit 2026-05-01 ~08:00)

User challenge: "I don't believe all 149 forced-SO submissions picked wrong --
that's statistically too low." Audit confirmed the user's intuition: the
ceiling is set by the explorer (Sonnet) cache, not by orchestrator skill.

Decomposition of all 388 rows by whether `explore_candidates` contains any
candidate with `is_correct=True`:

| Cache state | Count | Orch correct | Orch wrong |
|---|---:|---:|---:|
| Cache has at least 1 correct candidate | 140 (36.08 %) | 53 (37.9 %) | 87 (62.1 %) |
| Cache has NO correct candidate at all | 248 (63.92 %) | 0 (impossible) | 248 (forced) |

- **Oracle ceiling = 36.08 %** (always pick correct when one exists). Orch's
  13.66 % is 38 % of that ceiling.
- **"always pick first" baseline = ec0 = 18.81 %** (73/388 of cache idx=0 are
  correct). Orch is **5.1 pp BELOW** this trivial baseline.
- **88 % of orch submissions echo a cache candidate verbatim** (343/385 with
  non-empty pred). Orch is selecting from the candidate set, not inventing.

Concrete examples (all `is_correct=False`):
- qid 638: gold `Row 4 Column 17`. All 8 cache candidates = `Row 4 Column 14`.
  Orch submitted `Row 4 Column 14`. Cache had no correct option to select.
- qid 6161: gold `(7,8)`. Cache candidates = 7 x `(3,3)` + 1 x `(10,3)`. Orch
  submitted `(3,3)`.
- qid 462: gold `D`. Cache idx=1 was `D` (the only correct option), idx=0,2-7
  were `B`. Orch submitted `B` -- followed the 7-vs-1 majority despite the
  outlier being correct. This is an orch skill failure.

### Implications for paper / training

1. The negative Gain on BabyVision is **dominated by explorer (Sonnet) cache
   quality** -- 64 % of questions miss the correct answer in the cache --
   not by orchestrator visual reasoning per se.
2. GRPO training the orchestrator (Section app:training) can lift Acc
   towards the 36 % oracle ceiling by closing the 87 cases of "correct
   exists in cache, orch picked wrong", but cannot exceed it without
   changing the explorer.
3. The earlier framing "thinking alone cannot repair this; GRPO will fix it"
   needs rewriting: GRPO addresses the ~22 pp skill gap; the remaining
   ~64 pp is a cache / explorer problem that demands a stronger or different
   explorer family on multimodal benchmarks.

---

## M5 -- Compare `_temp` vs `_baseline`

| Benchmark | Pass@1 (ec0) | `_baseline` Acc → Gain | `_temp` Acc → Gain | Empty `_baseline` | Empty `_temp` |
|---|---:|---:|---:|---:|---:|
| HLE-Verified | 48.00 | 38.00 → -10.00 | **56.00 → +8.00** | 32 (incl. judge-bug 10) | 2 |
| GPQA-Diamond | 74.75 | 71.21 → -3.54 | **83.33 → +8.58** | -- | 1 |
| LiveCodeBench | 77.14 | 44.00 → -33.14 | **76.57 → -0.57** | 92 (52.6%) | 15 (8.6%) |
| BabyVision | 19.59 | (not run) | **13.66 → -5.93** | -- | 152 (39.2%) |

Decision (4/4 done): thinking-on with the Qwen3.6 sampling recipe is the
recommended inference config for HLE/GPQA/LCB (positive or near-parity Gain)
but is **not sufficient** on BabyVision (-5.93 Gain, 39.2% empty driven by
think-block decode-budget exhaustion on image-conditioned prompts). LCB
negative-Gain failure is repaired (-33.14 -> -0.57, parity with cache).
BabyVision is the residual failure that motivates the GRPO recipe in
Section app:training.

Decision criterion: if `_temp` improves the LCB negative-Gain failure mode
(empty `predicted_answer` rate drops), thinking-on is the recommended
inference config for the GRPO pre-training reference; otherwise stay with
greedy `_baseline` and document.

---

## Anti-patterns (carry over from `_baseline`)

- Don't reuse `_baseline` run dirs -- different decoding = different results.
- Don't re-launch serve between benchmarks; one serve, four evals.
- Don't trust stale `tmp/baseline_qwen36_35b_a3b_results.md` -- read
  `progress.json` / `delegated.log` per run dir.

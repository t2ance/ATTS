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

## M2 -- GPQA-Diamond (full, N=198)

| Step | Status |
|---|---|
| Write `scripts/gpqa/grpo/gpqa_qwen36_35b_a3b_temp.yaml` + `.sh` | pending |
| Reuse serve | pending |
| Launch eval; cache hits against `cache/gpqa/sonnet` | pending |
| Capture Acc / Pass@1 / Gain | pending |

**Run dir:** `analysis/run/gpqa/qwen36_35b_a3b_fp8_temp/run_<ts>/`

---

## M3 -- LCB (LiveCodeBench, N=175)

| Step | Status |
|---|---|
| Write `scripts/lcb/grpo/lcb_qwen36_35b_a3b_temp.yaml` + `.sh` (carry the PYTHONPATH shim from `_baseline`) | pending |
| Reuse serve | pending |
| Launch eval; cache hits against `cache/lcb/sonnet` | pending |
| Capture pass@1 + per-difficulty | pending |

**Run dir:** `analysis/run/lcb/qwen36_35b_a3b_fp8_temp/run_<ts>/`

**Hypothesis to check:** does thinking-on reduce the 48% "exit_reason: incomplete"
rate seen in `_baseline`? Compare per-question `exit_reason` distribution against
`_baseline`.

---

## M4 -- BabyVision (N=388)

| Step | Status |
|---|---|
| Write `scripts/babyvision/grpo/babyvision_qwen36_35b_a3b_temp.yaml` + `.sh` | pending |
| Reuse serve | pending |
| Launch eval; cache hits against `cache/babyvision/sonnet` | pending |
| Capture Acc | pending |

**Run dir:** `analysis/run/babyvision/qwen36_35b_a3b_fp8_temp/run_<ts>/`

---

## M5 -- Compare `_temp` vs `_baseline`

| Benchmark | `_baseline` Acc | `_temp` Acc | Delta |
|---|---:|---:|---:|
| HLE-Verified | 38.00 (judge-corrected) | __ | __ |
| GPQA-Diamond | 71.21 | __ | __ |
| LiveCodeBench | 44.00 | __ | __ |
| BabyVision | (deferred in `_baseline`) | __ | __ |

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

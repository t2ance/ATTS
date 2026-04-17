# Step 2: Evidence Collection

## W&B Run State
- **State**: `crashed`
- **Run ID**: rilxxd8r, displayName: `8b-sft-2gpu-bs96`
- **Created**: 2026-04-15T20:15:56Z
- **Last heartbeat**: 2026-04-15T23:32:41Z (~3h17m runtime)
- **Last logged step**: `_step: 4`, `training/global_step: 4`, `training/epoch: 3`

## Summary Metrics at Crash (step 4)
- `critic/rewards/mean`: 0.3257
- `critic/score/mean`: 0.3257
- `val-core/atts_hle/acc/mean@4`: 0.2596
- `val-core/atts_hle/acc/best@4/mean`: 0.4254
- `actor/grad_norm`: 1.97
- `perf/time_per_step`: 2161s (~36 min/step)
- `timing_s/gen`: 318s, `timing_s/ref`: 412s, `timing_s/update_actor`: 1140s

## Process State
- No `main_ppo` / verl training process alive (confirmed via ps)
- Judge vllm server (`vllm serve Qwen/Qwen3-8B`, pid=322076) still running since Apr15 on GPU 2

## Checkpoint State
- Only `global_step_3` exists
- `latest_checkpointed_iteration.txt` = "3"
- Resume would start from step 3

## GPU State (snapshot, point-in-time)
- GPU 0: 0 MiB used, 0% (freed after crash)
- GPU 1: 0 MiB used, 0% (freed after crash)
- GPU 2: 74073 MiB used, 0% SM (Judge vllm server, still loaded)
- GPU 3: 0 MiB used, 0%

## Crash Root Cause (from log)
AssertionError in `reward_fn.py:106`:
```
assert finish_reason == "stop", (
AssertionError: judge truncated (finish=length, max_tokens=4096).
reasoning field likely too long for budget.
```
Call stack: `reward_fn._judge_remote` → `RewardLoopWorker.compute_score` → `AgentLoopWorker.generate_sequences` → Ray RayTaskError → trainer crash.

The Judge LLM (vllm) returned `finish_reason=length` (hit max_tokens=4096 token limit) while generating its grading JSON. The `_judge_remote` function asserts `finish_reason == "stop"` with no fallback.

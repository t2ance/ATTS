# Plan: Training a Local ATTS Orchestrator

## Goal

Train a local model to serve as the ATTS orchestrator, replacing Claude Sonnet 4.6. The trained model makes explore/stop decisions and synthesizes final answers from candidate pools. Target: paper-ready numbers on all 4 benchmarks (GPQA, LCB, HLE, BabyVision).

## Key Decisions (2026-03-31)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Success standard | Paper-ready comparison | Add a row "ATTS (Qwen3-8B)" to the main results table across all 4 benchmarks |
| SFT method | Both distillation and rejection sampling; start with rejection sampling | Distillation = teacher (Sonnet/Haiku) trajectories. Rejection sampling = Qwen3-8B self-generated, keep correct only |
| Data source | Single script, data source as argument | build_sft_data.py takes trajectory directory as input. Same harness for validation and training data |
| Thinking mode | Keep `template: qwen3` (enable_thinking=True) | E2E consistency: both LLaMA-Factory and vLLM chat_template add `<think>` tags. Cost is a few empty tokens |
| Haiku/Sonnet explore shift | Deferred | Known distribution shift (training: Haiku explores, eval: Sonnet explores). Revisit if eval results are poor. Rejection sampling eliminates this shift |
| Cross-domain generalization | Needs experiment | No empirical evidence for single-domain (HLE) training generalizing to GPQA/LCB. If poor, expand to multi-domain data |

## Infrastructure: Structured Output / Tool Calling

**Problem**: Claude SDK guarantees valid JSON via native structured output. vLLM Qwen3-8B uses free-form text, sometimes generates invalid JSON.

**User preference**: Do NOT rely on model capability. Use infrastructure-level guarantees.

**Research findings** (2026-03-31):
- vLLM natively supports Anthropic Messages API (`/v1/messages`, PRs #22627 + #27882)
- Qwen official docs recommend: `vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes`
- `tool_choice=required` triggers constrained decoding since vllm>=0.8.3 (guaranteed JSON)
- Main blocking bug (#19051: tool_choice=required + thinking = 400) was FIXED (PR #19075, June 2025)
- Remaining open issue (#27447) only affects enable_thinking=False; we use True

**Experiment result** (2026-03-31): All 4 tests passed:
1. `tool_choice=auto` + explore call: tool call returned correctly
2. `tool_choice=named(StructuredOutput)` + constrained decoding: guaranteed valid JSON
3. Multi-turn (explore -> tool result -> StructuredOutput): correct decision + valid JSON
4. Anthropic endpoint `/v1/messages`: responds correctly

**Next step**: Switch `backends/vllm.py` to use native tool calling. Long-term: eliminate vllm.py entirely by pointing `backends/claude.py` at vLLM's Anthropic endpoint.

**Action item**: Delete fabricated StructuredOutput fallback in `backends/vllm.py:130-140`. Fabricated data violates P4.

Sources: [vLLM #21313](https://github.com/vllm-project/vllm/issues/21313), [vLLM #19051](https://github.com/vllm-project/vllm/issues/19051), [Qwen vLLM docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html), [vLLM Anthropic API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/anthropic/)

## Design Principle: Harness Engineering

The scaffold stays IDENTICAL. Only the model inside the orchestrator slot changes. Evaluation uses the same cached explores that Sonnet used.

## Compute

- 4x NVIDIA A100 80GB PCIe
- Conda env: `verl`

## Package Versions (verl env)

| Package | Version |
|---------|---------|
| torch | 2.10.0+cu128 |
| transformers | 4.57.6 |
| peft | 0.18.1 |
| vllm | 0.18.0 |
| verl | 0.7.0 |
| llamafactory | 0.9.5.dev0 (from source) |
| sglang | NOT INSTALLED (needed for GRPO) |
| flash_attn | 2.8.3 |

## Model Selection

| Model | Status | Notes |
|-------|--------|-------|
| Qwen3-8B | **Current choice** | SFT + eval working |
| Qwen3.5-9B | SFT verified, vLLM blocked | Needs transformers 5.x; vLLM 0.18 pins 4.57.6 |

---

## Phase 1: Data Pipeline

**Status**: PARTIAL (validation data done, training data in progress)

### Validation Data (pipeline testing only)

Source: existing Sonnet orchestrator trajectories. **CONTAMINATED** -- these overlap with eval sets.

| Benchmark | Episodes | Correct | Source |
|-----------|----------|---------|--------|
| GPQA | 198 | 164 (83%) | `analysis/run/gpqa/sonnet_no_integrate/run_20260317_181859` |
| HLE | 100 | 56 (56%) | `analysis/run/hle/sonnet_no_integrate/run_20260319_003712` |
| **Total** | **298** | **220** | |

### Training Data (for final model)

HLE questions 101-668 (568 questions). NOT overlapping with any eval set.

| Step | Status | Notes |
|------|--------|-------|
| Pre-cache Haiku explores | 262/668 questions (in progress) | ~$470 estimated total |
| Generate Sonnet trajectories | NOT STARTED | Depends on pre-cache |
| Build SFT data (rejection) | NOT STARTED | Only keep correct episodes |

Parser: `training/build_sft_data.py` -- takes trajectory directory as argument.

---

## Phase 2: SFT Training

**Status**: Infrastructure validated. No clean training data results yet.

Uses LLaMA-Factory. Current validation checkpoint trained on contaminated data (298 episodes).

### Configuration

| Parameter | Value |
|-----------|-------|
| Framework | LLaMA-Factory 0.9.5.dev0 |
| Base model | Qwen/Qwen3-8B |
| Method | LoRA (r=64, alpha=128, target=all) |
| Template | qwen3 (ReasoningTemplate, enable_thinking=True) |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Effective batch | 8 (bs=1, grad_accum=8) |
| Max seq length | 16384 |

### Validation Results (CONTAMINATED -- pipeline testing only)

| Benchmark | Sonnet | Qwen3-8B SFT | Notes |
|-----------|--------|--------------|-------|
| GPQA (198) | 80.81% | 74.7% | **CONTAMINATED**: trained on same 198 questions |
| LCB (44/175) | 82.29% | 88.6% | Partial (died at 44), contaminated |
| HLE (100) | 56.00% | TBD | Blocked on judge SDK |

These numbers validate that the pipeline works end-to-end. They do NOT represent clean evaluation.

---

## Phase 3: Evaluation

### vLLM Serving

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n verl vllm serve Qwen/Qwen3-8B \
    --enable-lora --max-lora-rank 64 \
    --lora-modules "atts-orch=checkpoints/sft_qwen3_8b" \
    --max-model-len 32768 --port 8000 --dtype bfloat16 --trust-remote-code
```

After verifying native tool calling, add: `--enable-auto-tool-choice --tool-call-parser hermes`

### Known Issues

1. **StructuredOutput JSON parsing**: Model generates invalid unicode escapes. `_fix_json_string()` handles most cases. Fabricated fallback at vllm.py:130-140 must be deleted.
2. **HLE grading**: Needs `claude_agent_sdk` (not in verl env). Run from `explain` env.
3. **BabyVision**: Qwen3-8B is text-only; BabyVision is multimodal. Not compatible.

---

## Phase 4: GRPO

### Status: Both fixes implemented, ready for testing.

### Environment (grpo conda env)

| Package | Version | Notes |
|---------|---------|-------|
| verl | 0.7.0 | From PyPI |
| sglang | 0.5.6 | Scheduler patched: assertion -> drain (see Fix 1 below) |
| torch | 2.9.1+cu128 | Pinned by verl sglang extra |
| transformers | 4.57.6 | Must be <5 for verl 0.7.0 compat |
| flash_attn | 2.8.3 or 4.0.0b7 | Installed |
| torch-memory-saver | latest | Required for hybrid mode |
| peft | 0.18.1 | For LoRA |

**CRITICAL env vars for launch:**
- `CUDA_VISIBLE_DEVICES=<free GPU>` (check nvidia-smi first)
- `RAY_ADDRESS=local` (avoid connecting to other projects' Ray clusters)
- `RAY_memory_monitor_refresh_ms=0` (disable Ray OOM killer, system has 251GB RAM)
- `LD_LIBRARY_PATH=.../nvidia/cuda_runtime/lib:.../torch/lib` (fix SGLang subprocess CUDA import)

### Completed work (2026-04-07)

1. **Tool implementations**: `training/grpo/explore_tool.py` (cached explores), `training/grpo/answer_tool.py` (StructuredOutput)
2. **Tool config**: `training/grpo/tool_config.yaml` (fixed: `required: []` for explore)
3. **Reward function**: `training/grpo/reward_fn.py` (R = correctness - 0.05 * num_explores)
4. **Data preparation**: `training/grpo/prepare_data.py` (198 train, 22 val parquet, agent_name=atts_agent)
5. **GRPO config**: `training/grpo/grpo_config.yaml` (Hydra, `pkg://verl.trainer.config`)
6. **Launch script**: `scripts/training/train_grpo.sh` (env vars, uses run_grpo.py entrypoint)
7. **SFT merged model**: `checkpoints/sft_qwen3_8b_merged/` (16GB, contaminated data)
8. **Custom agent loop**: `training/grpo/atts_agent_loop.py` (registered as `atts_agent`, direct cached explore lookup)
9. **Entrypoint**: `training/grpo/run_grpo.py` (imports atts_agent_loop, delegates to verl)

### Verified stages

| Stage | Status |
|-------|--------|
| Hydra config loading | PASS |
| Data loading (198 train, 22 val, 24 steps) | PASS |
| Config validation | PASS |
| FSDP actor model init | PASS |
| SGLang server startup | PASS (with LD_LIBRARY_PATH fix) |
| KV cache capture | PASS |
| Tool config loading | PASS (after adding `required: []`) |
| Hybrid mode memory swap (`release_memory_occupation`) | FIXED (see Fix 1) |
| Rollout generation | Not tested |
| Training step | Not tested |

### Fix 1: sglang 0.5.6 scheduler patch (DONE)

**Root cause**: sglang 0.5.6 scheduler has `assert self._is_no_request()` in `release_memory_occupation`. In multi-turn scenarios, requests can still be queued when this is called, causing the assertion to fail and the HTTP endpoint to hang. Fixed upstream in sglang 0.5.10 (PR #20296).

**Applied patch**: Replaced assertion with drain logic in `scheduler_update_weights_mixin.py:111-126`. Drains waiting_queue and clears batch state instead of asserting, with a warning log.

**File**: `/home/peijia/miniconda3/envs/grpo/lib/python3.12/site-packages/sglang/srt/managers/scheduler_update_weights_mixin.py`

**Why we can't just upgrade sglang**: sglang 0.5.10 has breaking API changes (`_launch_subprocesses` moved to `Engine._launch_subprocesses`, different return signature, different required args). A simple shim doesn't work because deeper internal APIs also changed.

### Fix 2: Custom ATTSAgentLoop (DONE)

`training/grpo/atts_agent_loop.py` registered as `atts_agent`. Benefits vs ToolAgentLoop:
- No dependency on verl's tool registry (avoids schema validation issues)
- Tool calling logic matches eval code exactly (train/eval consistency)
- Cached explore lookup is direct Python, no HTTP tool calling overhead
- Tool response format matches `tts_agent.py:102-109`

Entrypoint: `training/grpo/run_grpo.py` imports agent loop, delegates to `verl.trainer.main_ppo`.
Data format: `agent_name: "atts_agent"` in parquet rows.

Deleted: `patch_verl_sglang.py` (abandoned sglang 0.5.10 upgrade shim).

### Key design principle (CRITICAL)

Train/eval harness MUST be identical. ATTSAgentLoop uses the same:
- Tool calling protocol (hermes `<tool_call>` format via verl HermesToolParser)
- Explore result format as `tts_agent.py`
- StructuredOutput schema as `benchmarks/base.py` EXPLORE_SCHEMA
- Tool schemas match `tts_agent.py` EXPLORE_TOOL

### Next: end-to-end test

Run `train_grpo.sh` and verify rollout generation + training step complete. Resource needs:
- 4x A100 80GB (or fewer with param_offload=True)
- CPU RAM: check `free -h` before launch

---

## Phase 5: Paper Integration (future direction)

Add "ATTS (Qwen3-8B)" row to main results table. New subsection on learning the orchestrator policy.

---

## Next Steps (Priority Order)

1. ~~**Verify native tool calling**~~ -- DONE (all 4 tests passed)
2. ~~**Delete fabricated fallback**~~ -- DONE (vllm.py rewritten with native tool calling)
3. **Complete HLE pre-cache** -- 333/668 in progress
4. **Generate training trajectories** -- After pre-cache, run Sonnet orchestrator on 568 training questions
5. **Train clean SFT** -- Rejection sampling on HLE training data only
6. **Clean evaluation** -- GPQA, LCB, HLE (all unseen during training)
7. **Assess cross-domain generalization** -- If poor, expand to multi-domain training data
8. **GRPO** -- Both fixes implemented. Test with contaminated data first, then clean data after SFT
9. **Paper section** -- Write results

## File Layout

```
Experiment/core_code/
  backends/
    claude.py          # Existing
    vllm.py            # Local model backend
  training/
    build_sft_data.py  # Trajectory parser (takes directory as argument)
    dataset_info.json  # LLaMA-Factory dataset descriptor
    sft_config_qwen3_8b.yaml  # Qwen3-8B config
    preference.md      # User decisions log (this content merged into PLAN)
  scripts/training/
    precache_hle_training.sh
    generate_hle_trajectories.sh
    serve_vllm.sh
    train_sft.sh
  checkpoints/          # .gitignore'd
    sft_qwen3_8b/       # Validation checkpoint (contaminated data)
```

# Plan: Training a Local ATTS Orchestrator with VOC-Based Reward

## Goal

Train a local model to serve as the ATTS orchestrator, replacing Claude Sonnet 4.6. The trained model makes explore/stop decisions and synthesizes final answers from candidate pools. The result demonstrates that the orchestrator policy is learnable, with evaluation on ALL four paper benchmarks (clean, no train/test contamination).

## Design Principle: Harness Engineering

The scaffold (orchestrator infrastructure, tool definitions, candidate format, system prompts) stays IDENTICAL. The local model sees exactly what Sonnet sees and outputs exactly what Sonnet outputs. Only the model inside the orchestrator slot changes. No simplification of inputs, no modification to the scaffold.

Evaluation uses the existing `eval.py --backend vllm` with the same cached explores that Sonnet used. The only new code is `backends/vllm.py` (~250 LOC).

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
| sglang | NOT INSTALLED (needed for GRPO multi-turn) |
| flash_attn | 2.8.3 |

## Model Selection

| Model | Status | Notes |
|-------|--------|-------|
| Qwen2.5-7B-Instruct | Tested, works | transformers 4.57.6 supports it |
| Qwen3-8B | **Current choice**, tested, SFT+eval working | transformers 4.57.6 supports it |
| Qwen3.5-9B | SFT verified, vLLM blocked | Needs transformers 5.x; vLLM 0.18 pins 4.57.6 |

Qwen3.5-9B is the preferred model long-term but currently blocked by vLLM's transformers pin. Once vLLM supports transformers 5.x, switch to Qwen3.5-9B (SFT already verified to work via LLaMA-Factory with transformers 5.2.0).

---

## Current Status (2026-03-30)

### What's DONE

| Component | Status | Location |
|-----------|--------|----------|
| Trajectory parser | Done | `training/build_sft_data.py` |
| SFT data (298 episodes) | Done | `training_data/sft_all.jsonl` (GPQA 198 + HLE 100) |
| LLaMA-Factory configs | Done | `training/sft_config*.yaml` + `training/dataset_info.json` |
| SFT training (Qwen3-8B) | Done | `checkpoints/sft_qwen3_8b/` (3 epochs, loss 1.19->0.14) |
| SFT training (Qwen3.5-9B) | Done | `checkpoints/sft_qwen35_test/` (1 epoch, loss 0.66->0.31) |
| vLLM backend | Done | `backends/vllm.py` (text-based tool_call parsing) |
| GPQA eval (SFT) | **Complete** | 148/198 = **74.7%** (Sonnet: 80.81%) |
| LCB eval (SFT) | **Partial** | 39/44 = **88.6%** (Sonnet: 82.29%), died at 44/175 |
| vLLM server | Running | GPU 0, port 8000, models: [Qwen/Qwen3-8B, atts-orch] |
| HLE pre-cache (Haiku) | Paused | 2,877/5,344 explores (259/668 questions complete) |
| wandb integration | Config ready | `report_to: wandb` in all YAML configs |

### What's IN PROGRESS / BLOCKED

| Component | Status | Blocker |
|-----------|--------|---------|
| LCB eval | Died at 44/175 | StructuredOutput parse fail on some questions (invalid JSON escapes); need resume |
| HLE eval | Blocked | Grading needs `claude_agent_sdk` (not in verl env); run in `explain` env instead |
| BabyVision eval | N/A | Qwen3-8B is text-only; BabyVision is multimodal |
| GRPO training | Not started | Needs sglang install; data prep; tool implementation |

### Key Results

| Benchmark | Sonnet Orchestrator | Qwen3-8B SFT-only | Notes |
|-----------|--------------------|--------------------|-------|
| GPQA (198) | 80.81% | **74.7%** | Complete. 92.5% of Sonnet. |
| LCB (44/175) | 82.29% | **88.6%** | Partial (died at 44). Exceeds Sonnet on early questions. |
| HLE (100) | 56.00% | TBD | Blocked on judge SDK in verl env |
| BabyVision (388) | 23.20% | N/A | Text-only model can't handle multimodal |

### Known Issues

1. **StructuredOutput JSON parsing**: Model sometimes generates invalid unicode escapes (e.g. `\u208i` where `i` is not hex). Fixed with `_fix_json_string()` in `backends/vllm.py` but edge cases still occur. Some LCB questions fail.

2. **HLE grading**: HLE uses LLM judge (`claude-haiku-4-5-20251001`). When `--backend vllm`, grading tries to call the judge via vLLM (wrong). Fixed in `benchmarks/base.py` to route grading to Claude backend, but `claude_agent_sdk` is not in verl env. Run HLE eval from `explain` env instead.

3. **Context overflow**: Some long conversations (8+ explores) exceed 16K tokens. vLLM server now runs with `--max-model-len 32768` and backend uses `max_tokens=4096`.

---

## Phase 1: Data Pipeline (DONE)

### SFT Data

Source: existing ATTS trajectories (Sonnet orchestrator).

| Benchmark | Episodes | Correct | Source |
|-----------|----------|---------|--------|
| GPQA | 198 | 164 (83%) | `analysis/run/gpqa/sonnet_no_integrate/run_20260317_181859` |
| HLE | 100 | 56 (56%) | `analysis/run/hle/sonnet_no_integrate/run_20260319_003712` |
| **Total** | **298** | **220** | |

Parser: `training/build_sft_data.py` converts `trajectory.md` + `input.md` into JSONL with Qwen-style `<tool_call>` formatting.

### Train/Test Contamination Strategy

For the paper's main table (Path 2: comparable numbers on full benchmarks):
- Train on HLE remaining questions (101-668, ~568 questions) — NOT the 100 eval questions
- Evaluate cleanly on all 4 paper benchmarks (GPQA, LCB, HLE, BabyVision all unseen during training)

Current SFT uses existing GPQA+HLE eval trajectories for pipeline validation only. Final training will use HLE training set exclusively.

### HLE Pre-Cache Status

Pre-caching Haiku explores for HLE training questions. Paused (rate limit + cost).

| Metric | Value |
|--------|-------|
| Target | 668 questions x 8 explores = 5,344 |
| Completed | 2,877 explores (259 complete questions + 1 partial) |
| Cost so far | ~$173 |
| Estimated total | ~$470 |

---

## Phase 2: SFT Training (DONE)

Uses LLaMA-Factory (no custom training code needed).

### Configuration

| Parameter | Value |
|-----------|-------|
| Framework | LLaMA-Factory 0.9.5.dev0 |
| Base model | Qwen/Qwen3-8B |
| Method | LoRA (r=64, alpha=128, target=all) |
| Template | qwen3 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Effective batch | 8 (bs=1, grad_accum=8) |
| Max seq length | 16384 |
| GPU | Single A100 80GB |
| Time | ~21 min |
| Loss | 1.19 -> 0.14 (train), 0.49 (eval) |

### Commands

```bash
# Build SFT data
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
python -m training.build_sft_data

# Train
CUDA_VISIBLE_DEVICES=0 conda run -n verl llamafactory-cli train training/sft_config_qwen3_8b.yaml
```

### Checkpoints

| Model | Location | Status |
|-------|----------|--------|
| Qwen3-8B SFT | `checkpoints/sft_qwen3_8b/` | 3 epochs, validated |
| Qwen3.5-9B SFT | `checkpoints/sft_qwen35_test/` | 1 epoch, validated (vLLM blocked) |

---

## Phase 3: Evaluation (PARTIAL)

### vLLM Serving

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n verl vllm serve Qwen/Qwen3-8B \
    --enable-lora --max-lora-rank 64 \
    --lora-modules "atts-orch=checkpoints/sft_qwen3_8b" \
    --max-model-len 32768 --port 8000 --dtype bfloat16 --trust-remote-code
```

Note: Do NOT use `--enable-auto-tool-choice --tool-call-parser hermes`. The hermes parser strips `<tool_call>` tags from content inconsistently on complex JSON. Use text-based parsing in `backends/vllm.py` instead.

### Eval Commands

```bash
# GPQA (works)
python eval.py --benchmark gpqa --backend vllm --method tts-agent \
    --num 198 --seed 42 --num-explores 8 --num-workers 4 \
    --log-dir ../analysis/run/gpqa/qwen3_8b_sft \
    --orchestrator-model atts-orch --explore-model claude-sonnet-4-6 \
    --integrate-model atts-orch \
    --cache-dirs ../analysis/cache/gpqa/sonnet --no-integrate

# LCB (works but some questions fail on JSON parsing)
python eval.py --benchmark lcb --backend vllm --method tts-agent \
    --num 175 --seed 42 --num-explores 8 --num-workers 1 \
    --log-dir ../analysis/run/lcb/qwen3_8b_sft \
    --orchestrator-model atts-orch --explore-model claude-sonnet-4-6 \
    --integrate-model atts-orch \
    --cache-dirs ../analysis/cache/lcb/sonnet --no-integrate

# HLE (run from explain env due to claude_agent_sdk dependency)
# Or install claude_agent_sdk in verl env
```

---

## Phase 4: GRPO with VOC Reward (NOT STARTED)

### Architecture (from research)

verl 0.7.0 has built-in multi-turn tool-calling GRPO via `ToolAgentLoop`.

**Required components:**
1. `training/tools/atts_tools.py` — `ATTSExploreTool` and `ATTSStructuredOutputTool` (subclass `verl.tools.base_tool.BaseTool`)
2. `training/tools_config.yaml` — tool registration
3. `training/reward_fn.py` — `compute_score()`: R = correct(answer) - 0.05 * num_explores
4. `training/build_grpo_data.py` — build parquet with `agent_name`, `tools_kwargs`, cached explores
5. `scripts/training/run_grpo.sh` — launch command

### Key Design Decisions

- **Rollout engine**: Must use `sglang` (NOT vllm). verl's `ToolAgentLoop` only works with sglang async rollout.
- **LoRA + SGLang**: Requires `lora.merge=True` (SGLang doesn't support native LoRA adapter loading).
- **Tool response masking**: `response_mask=0` for tool outputs (model doesn't learn to predict explore results, only decisions).
- **Data format**: Parquet with `agent_name="tool_agent"` column (CRITICAL — without it, tools silently never called).
- **Reward**: Episode-level binary correctness + per-step -0.05 explore penalty. No dense shaping (causes reward hacking per literature).
- **Format**: `multi_turn.format=hermes` for Qwen tool calling.

### Blockers

1. **sglang not installed** — `pip install "sglang[all]>=0.4.0"` in verl env
2. **GRPO data not built** — Need `build_grpo_data.py` to convert cached explores to parquet
3. **Tools not implemented** — Need `ATTSExploreTool` and `ATTSStructuredOutputTool`
4. **Untested combination** — LoRA + multi-turn + SGLang hasn't been tested together

### GRPO Training Command (Draft)

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=training/data/grpo_train.parquet \
    data.val_files=training/data/grpo_val.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=checkpoints/sft_qwen3_8b \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=training/tools_config.yaml \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    custom_reward_function.path=training/reward_fn.py \
    custom_reward_function.name=compute_score \
    trainer.n_gpus_per_node=4 \
    trainer.total_epochs=3 \
    trainer.project_name=atts_grpo \
    trainer.experiment_name=qwen3_8b_grpo_v1 \
    trainer.logger=['console','wandb']
```

---

## Phase 5: Paper Integration (NOT STARTED)

If results meet criteria, add to the paper:
1. New row in main results table: "ATTS (Qwen3-8B)" with accuracy on GPQA, LCB, (HLE if available)
2. New subsection "Learning the Orchestrator Policy" in Experiments
3. Comparison: SFT-only vs SFT+GRPO vs Sonnet
4. Stopping distribution analysis: 7B vs Sonnet explore counts

---

## File Layout

```
Experiment/core_code/
  backends/
    claude.py          # Existing
    codex.py           # Existing
    vllm.py            # NEW: local model backend (text-based tool_call parsing)
  training/
    __init__.py
    build_sft_data.py  # NEW: trajectory.md -> SFT JSONL
    dataset_info.json  # NEW: LLaMA-Factory dataset descriptor
    sft_config.yaml    # NEW: LLaMA-Factory SFT config (generic)
    sft_config_qwen3_8b.yaml   # NEW: Qwen3-8B specific
    sft_config_qwen35.yaml     # NEW: Qwen3.5-9B specific
    sft_config_test.yaml       # NEW: test config
    tools/                     # TODO: verl BaseTool subclasses for GRPO
    reward_fn.py               # TODO: GRPO reward function
    build_grpo_data.py         # TODO: build GRPO parquet
  scripts/training/
    build_sft_data.sh
    train_sft.sh
    serve_vllm.sh
    test_eval_vllm.sh
    precache_hle_training.sh
    generate_hle_trajectories.sh
    resume_precache.sh
    test_pipeline.sh
    run_grpo.sh                # TODO
  checkpoints/          # .gitignore'd
    sft_qwen3_8b/       # Qwen3-8B LoRA adapter (3 epochs)
    sft_qwen35_test/    # Qwen3.5-9B LoRA adapter (1 epoch)
  training_data/        # .gitignore'd
    sft_all.jsonl        # 298 episodes (GPQA 198 + HLE 100)
```

---

## Next Steps (Priority Order)

1. **Complete LCB eval** — Resume from 44/175. May need to add error handling for StructuredOutput parse failures (skip question instead of crash).
2. **Run HLE eval** — Either install `claude_agent_sdk` in verl env, or run from `explain` env with vLLM backend.
3. **Install sglang** — `pip install "sglang[all]>=0.4.0"` in verl env. Required for GRPO.
4. **Implement GRPO tools** — `ATTSExploreTool`, `ATTSStructuredOutputTool`, `reward_fn.py`.
5. **Build GRPO data** — Convert cached explores to parquet format with `agent_name`, `tools_kwargs`.
6. **Run GRPO** — Train with VOC reward, evaluate improvement over SFT-only.
7. **Scale data** — Resume HLE pre-cache (2,877/5,344), generate trajectories, retrain with more data.
8. **Write paper section** — Add 7B orchestrator results to paper.

# Plan: Training a 7B ATTS Orchestrator with VOC-Based Reward

## Goal

Train Qwen2.5-7B-Instruct to serve as the ATTS orchestrator, replacing Claude Sonnet 4.6. The trained model makes explore/stop decisions and synthesizes final answers from candidate pools. Success is an existence proof that the orchestrator decision is learnable via RL with a VOC-inspired reward.

## Design Principle: Harness Engineering

The scaffold (orchestrator infrastructure, tool definitions, candidate format, system prompts) stays IDENTICAL. The 7B model sees exactly what Sonnet sees and outputs exactly what Sonnet outputs. Only the model inside the orchestrator slot changes. No simplification of inputs, no modification to the scaffold.

## Compute

- 4x NVIDIA A100 80GB PCIe
- PyTorch 2.10, Transformers 4.57.6
- Base model: `Qwen/Qwen2.5-7B-Instruct` (128K context, native tool calling with `<tool_call>` tags)

## Success Criteria

| Metric | Target |
|--------|--------|
| SFT-only accuracy on GPQA | >40% (>50% of Sonnet's 80.81%) |
| SFT+RL accuracy on GPQA | Measurable improvement over SFT-only |
| Stopping distribution correlation with Sonnet | >0.7 |
| VOC reward vs accuracy-only reward | VOC reward produces better accuracy-cost tradeoff |

---

## Phase 1: Data Pipeline

### 1.1 SFT Data Source

Orchestrator traces from existing ATTS v3 (no-integrate) runs. ALL traces use StructuredOutput (v3), verified with zero v1 traces.

| Benchmark | Source | Episodes |
|-----------|--------|----------|
| GPQA | `analysis/run/gpqa/sonnet_no_integrate{,_run1..5}/run_*/trajectories/` | 198 * 6 = 1188 |
| HLE | `analysis/run/hle/sonnet_no_integrate/run_*/trajectories/` | 100 |
| LCB | `analysis/run/lcb/sonnet_no_integrate/run_*/trajectories/` | 175 |
| BabyVision | `analysis/run/babyvision/sonnet_no_integrate/run_*/trajectories/` | 388 |
| **Total** | | **~1851** |

**Filtering**: Use ALL episodes regardless of correctness. Tag each with metadata `{"correct": bool, "num_explores": int, "benchmark": str}` for downstream analysis. No filtering by default.

### 1.2 Trajectory Parsing (markdown -> structured messages)

Each trajectory is in `trajectory.md` as markdown. There is NO structured JSON representation. The data pipeline must parse markdown into structured multi-turn conversation format.

**Input file**: `<run_dir>/trajectories/<qid>/trajectory.md`
**Companion file**: `<run_dir>/trajectories/<qid>/input.md` (system prompt + user message)

**Parsing rules** (delimiters in `trajectory.md`):

| Markdown pattern | Message role | Content |
|------------------|-------------|---------|
| `<thinking>...</thinking>` | (see below) | Chain-of-thought reasoning |
| Plain text between markers | assistant | Text output |
| `**Tool call**: <name>` | assistant | Tool call (extract name) |
| `<details><summary>Tool result...</summary>...` | tool | Tool result (extract text inside code fence) |
| `<details><summary><b>StructuredOutput</b></summary>...` | assistant | StructuredOutput tool call (extract JSON from code fence) |

**Thinking block handling**: Convert `<thinking>` blocks to regular CoT text PREPENDED to the assistant turn. The 7B model generates CoT as regular text, followed by a `<tool_call>` tag. The `<tool_call>` marker provides the clean boundary between reasoning and action.

**Output format**: Qwen2.5 ChatML with native tool calling:

```json
{
  "messages": [
    {"role": "system", "content": "<orchestrator system prompt>"},
    {"role": "user", "content": "<problem text>"},
    {"role": "assistant", "content": "<CoT reasoning>\n<tool_call>\n{\"name\": \"explore\", \"arguments\": {}}\n</tool_call>"},
    {"role": "tool", "content": "Candidate #1 recorded.\n- Answer: B\n- Confidence: 0.92\n..."},
    {"role": "assistant", "content": "<CoT reasoning>\n<tool_call>\n{\"name\": \"explore\", \"arguments\": {}}\n</tool_call>"},
    {"role": "tool", "content": "Candidate #2 recorded.\n..."},
    {"role": "assistant", "content": "<CoT synthesis reasoning>\n<tool_call>\n{\"name\": \"StructuredOutput\", \"arguments\": {\"answer\": \"B\", \"reasoning\": \"...\", ...}}\n</tool_call>"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "explore", "description": "...", "parameters": {}}},
    {"type": "function", "function": {"name": "StructuredOutput", "description": "Submit final answer", "parameters": {"type": "object", "properties": {"answer": ..., "reasoning": ...}}}}
  ],
  "metadata": {"correct": true, "num_explores": 2, "benchmark": "gpqa", "qid": "recWXwn9v4IG9ZrM6"}
}
```

**Key details**:
- The `tools` array includes `StructuredOutput` as an explicit tool with the benchmark-specific answer schema as its parameter schema (GPQA: answer is A/B/C/D, LCB: answer is code string, BabyVision: answer is free-form text)
- Each episode is self-contained: tool definitions + system prompt + conversation
- The system prompt comes from `prompts.py` (`_ORCHESTRATOR_BASE` + v3 finalize/stop instructions)

### 1.3 Implementation: `scripts/build_sft_data.py`

```
Input:  trajectory directories across all benchmarks
Output: training_data/sft_train.jsonl, training_data/sft_eval.jsonl
Split:  80% train, 20% eval (stratified by benchmark)
```

Steps:
1. For each benchmark, load `BenchmarkConfig` to get the StructuredOutput schema
2. For each run directory, iterate over trajectory directories
3. Parse `input.md` to extract system prompt + user message
4. Parse `trajectory.md` using the delimiter rules above
5. Convert thinking blocks to CoT text
6. Format as Qwen2.5 ChatML with tool definitions
7. Tag with metadata (correctness from `trajectory.md` grading section)
8. Split 80/20 stratified by benchmark
9. Write to JSONL

### 1.4 Train/Eval Split

| Benchmark | Train | Eval |
|-----------|-------|------|
| GPQA | 158 questions (all episodes from these questions) | 40 questions |
| HLE | 80 | 20 |
| LCB | 140 | 35 |
| BabyVision | 310 | 78 |

Split is BY QUESTION (not by episode) to prevent data leakage — all episodes from the same question go to the same split.

---

## Phase 2: SFT Training

### 2.1 Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Method | LoRA |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Learning rate | 2e-4 |
| Epochs | 3 |
| Effective batch size | 32 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 |
| GPUs | 4x A100 80GB |
| DeepSpeed | ZeRO Stage 2 |
| Max sequence length | 16384 tokens |
| Sequence packing | Yes (episodes range from ~3K to ~20K tokens) |
| Framework | TRL `SFTTrainer` + PEFT |
| Training targets | Assistant messages only (CoT + tool calls) |

### 2.2 Implementation: `scripts/train_sft.py`

```python
# Pseudocode
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="checkpoints/sft",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    max_seq_length=16384,
    packing=True,
    deepspeed="configs/ds_zero2.json",
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=load_dataset("json", data_files="training_data/sft_train.jsonl"),
    eval_dataset=load_dataset("json", data_files="training_data/sft_eval.jsonl"),
    peft_config=lora_config,
    args=sft_config,
)
trainer.train()
```

### 2.3 SFT Validation

After SFT, verify the model can:
1. Generate valid `<tool_call>` JSON (not malformed)
2. Produce CoT reasoning before tool calls
3. Call `explore` and `StructuredOutput` tools correctly
4. Stop after a reasonable number of explores (not always 1, not always 8)

Run on 10 eval questions manually to sanity-check before proceeding to RL.

---

## Phase 3: RL with VOC Reward

### 3.1 Environment: Cached Explore Simulator

The RL environment simulates the ATTS loop using cached explores. No API calls needed.

**State**: (system prompt, user message with problem, tool definitions, conversation history so far)

**Actions**: The model generates text. The environment parses it:
- If `<tool_call>{"name": "explore", ...}</tool_call>` → load next cached candidate, append tool result to conversation, continue
- If `<tool_call>{"name": "StructuredOutput", "arguments": {...}}</tool_call>` → episode ends, extract answer for grading
- If no valid tool call detected → episode ends with error (reward = -1)
- If explore count reaches budget T=8 → force stop, grade whatever answer is available (or reward = 0 if none)

**Cached explores location**:
- GPQA: `analysis/cache/gpqa/sonnet/<qid>/explore_<n>/result.json`
- LCB: `analysis/cache/lcb/sonnet/<qid>/explore_<n>/result.json`
- BabyVision: `analysis/cache/babyvision/sonnet/<qid>/explore_<n>/result.json`
- HLE: NOT AVAILABLE (no precached explores)

Each `result.json` contains `{approach, reasoning, answer, confidence}` or `{timed_out: true}`. Timed-out explores are presented to the model as "Candidate #N: TIMED OUT (solver exceeded time limit)".

### 3.2 Reward Function

**Terminal reward (applied at episode end)**:

```
R = correct(final_answer) - lambda * num_explores * C
```

Where:
- `correct(final_answer)` = 1 if grading passes, 0 otherwise
- `lambda` = cost tradeoff hyperparameter (tune: start with lambda=0.05)
- `num_explores` = number of explore() calls in the episode
- `C` = normalized cost per explore = 1.0 (so lambda directly controls the penalty scale)

**Grading per benchmark**:
- GPQA: exact letter match (A/B/C/D)
- LCB: code execution (run test cases) — adds latency per episode
- BabyVision: LLM judge — adds API cost per episode

**Practical simplification**: RL training uses GPQA only (fast, deterministic grading). LCB/BabyVision are evaluation-only.

### 3.3 Curriculum

**Phase 2a — Stopping policy only (GPQA)**:
- Oracle synthesis: majority vote on candidate answer letters seen so far
- The model only decides WHEN to stop (explore vs stop actions)
- At stop: the oracle computes majority vote → grade → reward
- This isolates the stopping policy from synthesis quality
- Only works on GPQA (multiple-choice with categorical answers)

**Phase 2b — End-to-end (GPQA, extend to LCB/BabyVision for eval)**:
- The model's own synthesis (StructuredOutput) is graded
- Full reward: accuracy of model's own answer - lambda * cost
- This trains both stopping AND synthesis

### 3.4 GRPO Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | GRPO (Group Relative Policy Optimization) |
| Rollouts per question (K) | 8 |
| Learning rate | 5e-6 |
| KL penalty (beta) | 0.05 |
| Epochs per RL iteration | 3 |
| Temperature during rollouts | 0.7 |
| RL training set | GPQA train split (158 questions) |
| RL iterations | 5-10 (monitor for convergence) |
| Framework | Custom rollout engine + TRL GRPO loss |

### 3.5 Custom Rollout Engine (~500 LOC)

This is the riskiest component. It must:

1. **Serve the model**: Use vLLM for fast inference (1 GPU dedicated to serving)
2. **Simulate ATTS loop**: For each question, run K=8 rollouts:
   - Format system prompt + tools + problem as Qwen2.5 ChatML
   - Generate model output (with temperature=0.7)
   - Parse output: detect `<tool_call>` tag
   - If explore: load `cache/<qid>/explore_<n>/result.json`, format as tool result, append to conversation
   - If StructuredOutput: extract answer, grade, compute reward
   - If malformed: reward = -1
   - If budget exhausted: reward = 0
3. **Collect trajectories**: Record all tokens generated by the model (for GRPO loss)
4. **Compute group-relative rewards**: For K rollouts of the same question, normalize rewards
5. **Feed to GRPO loss**: Use TRL's loss computation on the collected trajectories

**Key implementation detail**: vLLM serves the model for fast inference during rollout collection. After collecting a batch of rollouts, the LoRA weights are updated via TRL's GRPO loss on a separate GPU. Then vLLM reloads the updated weights. This alternation (collect → update → reload) repeats for each RL iteration.

### 3.6 Implementation: `scripts/train_rl.py` + `rl/rollout_engine.py` + `rl/environment.py`

```
rl/
  environment.py    # Cached explore simulator (~200 LOC)
  rollout_engine.py # vLLM-based rollout collection (~300 LOC)
  grpo_trainer.py   # GRPO loss + weight update (~200 LOC)
  parsing.py        # Shared <tool_call> parsing (~200 LOC, shared with backends/vllm.py)
scripts/
  train_rl.py       # Main RL training loop
```

---

## Phase 4: Evaluation Backend

### 4.1 `backends/vllm.py` (~300 LOC)

New backend module implementing the same interface as `backends/claude.py`:

```python
async def run_tool_conversation(
    *, system_prompt, user_message, image_data_url, model, tools,
    max_turns, tool_handler, effort=None, output_format=None,
    writer=None, quiet=True, on_structured_output=None,
) -> tuple[float, dict[str, Any]]:
```

This module:
1. Connects to a vLLM server (started separately via `vllm serve`)
2. Formats messages as Qwen2.5 ChatML with tool definitions
3. Generates → parses `<tool_call>` blocks → calls `tool_handler` → injects result → generates again
4. Detects `StructuredOutput` calls and invokes `on_structured_output`
5. Writes events via `writer` (for trajectory recording)
6. Computes cost from token counts (using a configurable $/token rate for reporting)

**Shared with RL engine**: The `<tool_call>` parsing logic (`rl/parsing.py`) is shared between this backend and the RL rollout engine.

### 4.2 Evaluation Runs

Run the trained 7B orchestrator in the full ATTS pipeline:

```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct --lora-modules atts-orch=checkpoints/rl_final

# Run evaluation (same eval.py, different backend)
python eval.py --benchmark gpqa --backend vllm --method tts_agent \
    --orchestrator-model atts-orch --cache-dirs analysis/cache/gpqa/sonnet --cache-only
```

### 4.3 Evaluation Metrics

| Metric | How measured |
|--------|-------------|
| **Accuracy** | Standard per-benchmark grading |
| **Cost per question** | Token count * assumed $/token rate |
| **Explores per question** | Count from trajectory |
| **Stopping quality** | Compare explore count distribution with Sonnet's |
| **Synthesis quality** | At matched stopping points: 7B answer vs Sonnet answer vs majority vote |

### 4.4 Baselines

| Baseline | Description |
|----------|-------------|
| Sonnet orchestrator | Current ATTS (from existing runs) |
| 7B SFT-only | Before RL training |
| 7B SFT+RL (VOC reward) | After RL with `R = correct - lambda * cost` |
| 7B SFT+RL (accuracy-only) | After RL with `R = correct` (no cost penalty) — ablation |
| Random stopping | Stop at random N in [1,8], use majority vote |
| Always-8 | Use all 8 explores, majority vote |

---

## Phase 5: Paper Integration

If results meet success criteria, add to the paper:

1. **Methodology**: Formalize the metalevel MDP (state/action/reward) connecting ATTS to rational metareasoning
2. **Experiments**: New subsection "Learning the Orchestrator Policy" with:
   - SFT vs SFT+RL comparison
   - Stopping quality analysis
   - VOC reward ablation
   - Cost-accuracy frontier (7B vs Sonnet)
3. **Theory connection**: The VOC-based reward is not just an interpretive lens — it's a validated design principle for training orchestrators

---

## Timeline

| Week | Deliverable |
|------|-------------|
| 1 | `build_sft_data.py` (trajectory parser + data pipeline), SFT training started |
| 2 | SFT complete, `backends/vllm.py`, RL environment + rollout engine |
| 3 | GRPO training (Phase 2a stopping-only, Phase 2b end-to-end) |
| 4 | Evaluation + ablations + paper section draft |
| 5 | Buffer / iteration |

---

## File Layout

```
Experiment/core_code/
  backends/
    claude.py          # Existing
    vllm.py            # NEW: local model backend
  rl/
    environment.py     # NEW: cached explore simulator
    rollout_engine.py  # NEW: vLLM-based rollout collection
    grpo_trainer.py    # NEW: GRPO loss computation
    parsing.py         # NEW: shared <tool_call> parsing
  scripts/
    build_sft_data.py  # NEW: trajectory -> SFT JSONL
    train_sft.py       # NEW: SFT training script
    train_rl.py        # NEW: RL training loop
  configs/
    ds_zero2.json      # NEW: DeepSpeed ZeRO-2 config
  training_data/
    sft_train.jsonl    # Generated
    sft_eval.jsonl     # Generated
  checkpoints/
    sft/               # SFT checkpoint
    rl_phase2a/        # RL Phase 2a checkpoint
    rl_final/          # Final RL checkpoint
```

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| 7B can't synthesize well from complex reasoning traces | High | Measure stopping vs synthesis separately; even with poor synthesis, stopping quality is a valid contribution |
| Custom rollout engine bugs | High | Test extensively on 10 questions before full training; share parsing code with eval backend |
| RL doesn't converge | Medium | SFT-only is the fallback result; curriculum (Phase 2a→2b) reduces difficulty |
| 761 RL questions too few → overfitting | Medium | Train/eval split, cross-benchmark training, monitor per-question rewards |
| Context too long for 7B (8-explore = ~20K tokens) | Low | Qwen2.5 supports 128K; max_seq_length=16384 covers >95% of episodes |
| vLLM ↔ training weight sync issues | Medium | Use LoRA adapters loaded by vLLM; reload after each RL iteration |

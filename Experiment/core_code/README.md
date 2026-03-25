# ATTS -- Agentic Test-Time Scaling via Adaptive Exploration

ATTS improves answer quality at inference time by adaptively exploring a problem from different angles, building a pool of diverse candidate answers, and synthesizing them into a final response.

## Workflow

The orchestrator has a single tool:

- **explore** -- Dispatch a fresh, independent solver. Each candidate includes an approach description, step-by-step reasoning, an answer, and a confidence score.

The orchestrator decides when to stop exploring and directly synthesizes the final answer from the accumulated evidence. It is given a fixed exploration budget (e.g., 8 rounds).

## Directory Layout (dr-claw structure)

```
Explain/
  Experiment/
    core_code/              # This directory -- main codebase
      eval.py               # CLI entry point + evaluation loop
      precache_explores.py  # Pre-generate explore results
      prompts.py            # Orchestrator/explorer prompts
      backends/             # Claude, Codex API backends
      benchmarks/           # Benchmark configs (HLE, LCB, GPQA, BabyVision, AIME)
      methods/              # TTS agent, self-refine, budget forcing, reward rerank
      scripts/              # Launch scripts (all constants hardcoded)
    analysis/               # Experiment outputs
      cache/{benchmark}/{model}/  # Pre-cached explore results
      run/{benchmark}/{variant}/  # Run logs, results, trajectories
    code_references/        # Auxiliary repos (claude-relay-service, LiveCodeBench, etc.)
  Publication/
    paper/                  # LaTeX source, figures, compiled PDF
```

## Output Structure (`analysis/`)

```
analysis/
  cache/{benchmark}/{model}[/{subset}]/{question_id}/explore_{n}/
    input.md              # Prompt sent to the explorer
    output.md             # Raw model response
    result.json           # Parsed structured result

  run/{benchmark}/{variant}/run_{timestamp}/
    run_config.json       # Run configuration
    rounds.jsonl          # Per-round log (each explore action with cost)
    results.jsonl         # Per-question results
    progress.json         # Live progress + final summary (accuracy, cost, best-of-n curves)
    trajectories/{question_id}/
      trajectory.md       # Full human-readable trajectory
```

## Usage

### Precache explores

```bash
python precache_explores.py \
  --benchmark hle \
  --backend claude \
  --cache-dir ../analysis/cache/hle/opus/gold \
  --subset gold \
  --num 100 \
  --num-explores 8 \
  --num-workers 10 \
  --seed 42 \
  --text-only \
  --explore-model claude-opus-4-6
```

### Run evaluation

```bash
python eval.py --benchmark hle \
  --backend claude \
  --subset gold \
  --num 100 \
  --seed 42 \
  --num-explores 8 \
  --num-workers 10 \
  --text-only \
  --log-dir ../analysis/run/hle/opus \
  --orchestrator-model claude-opus-4-6 \
  --explore-model claude-opus-4-6 \
  --cache-dir ../analysis/cache/hle/opus/gold
```

### Supported benchmarks

| Benchmark | Dataset | Grading |
|-----------|---------|---------|
| `hle` | [HLE-Verified](https://huggingface.co/datasets/skylenage/HLE-Verified) | LLM judge |
| `lcb` | [LiveCodeBench v6](https://huggingface.co/datasets/livecodebench/code_generation_lite) | Code execution |
| `gpqa` | [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | Exact match |
| `babyvision` | [BabyVision](https://huggingface.co/datasets/UnipatAI/BabyVision) | LLM judge |
| `aime` | AIME 2025/2026 | Exact integer match |

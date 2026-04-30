# CLAUDE.md

# Developing
CRITICAL: NEVER CREATE NEW BRANCH UNTIL THE USER ASK YOU TO DO SO. ESPECIALLY WHEN DEVELOP A NEW FEATURE, DIRECTLY ADD TO THE CURRENT BRANCH.

# Monitoring
WHENEVER YOU RUN MONITROING TOOL, DO NOT LEAVE THE TASK ALONG. THERE MIGHT BE CASES WHERE IT SLIENTLY FAILED WITHOUT TELLING YOU. THEREFORE, YOU SHOULD HAVE AT LEAST ON HEARTBEAT BASED ON TIME. DEFAULT TO 5 MINS, SO THAT YOU CAN CHECK IT REGULARLY EVEN IF YOU DO NOT RECEIVE ANYTHING FROM THE PROCESS.

# Training Performance Optimizations (MUST-TRY for verl GRPO/SFT)

When launching any verl-based training (GRPO, SFT) on this project, the launcher
script MUST attempt every optimization below. These are pure perf wins (no math
change) — skipping them produces a slower, more memory-hungry, lower-throughput
training run. If any one fails to apply, document why in the launcher with a
comment explaining the gating reason.

## Mandatory verl/FSDP knobs

| Knob | Setting | Why |
|---|---|---|
| `actor_rollout_ref.model.use_liger` | `True` | Liger-Kernel: fused RMSNorm / RoPE / SwiGLU / CrossEntropy. Drop-in throughput win. |
| `actor_rollout_ref.actor.use_fused_kernels` | `True` | Activates verl's per-arch fused-kernel monkey patches (Qwen3.5MoE, etc.). Without this, log shows `Skipping monkey patch ... use_fused_kernels is False`. |
| `actor_rollout_ref.actor.use_remove_padding` | `True` (default) | Sequence packing — packed forward instead of padded. Verify in log: `Actor use_remove_padding=True`. |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | `True` | Chunks the entropy/logits computation. Saves actor-update memory peak. |
| `actor_rollout_ref.actor.use_dynamic_bsz` | `True` | Dynamic micro-batch packing by token budget. Combine with `ppo_max_token_len_per_gpu`. |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch` | `True` | Overlap next-layer all-gather with current-layer compute. Apply to both `actor` AND `ref` FSDP configs. |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | `True` | Free unsharded params immediately after forward — drops peak GPU mem. |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True` | Trade compute for memory. |

## Symmetric remaining memory principle (multi-GPU training)

When launching multi-GPU training (FSDP DP, vLLM TP, or hybrid), the training GPUs
**must have identical remaining free memory at init time**. The card with the LEAST
free memory caps `gpu_memory_utilization` for the entire job — vLLM/FSDP allocates
symmetrically across ranks, so an asymmetric setup wastes the larger card's headroom.

**Why (2026-04-30 user directive):** On the local box, GPU 0 carries a 1.37 GB
claude embedding daemon. Co-locating training on GPU 0+1 made cuda:0 have 43.6 GB
free vs cuda:1 at 45 GB — a 1.4 GB asymmetry. Every cuda:1 byte beyond cuda:0's
ceiling was unusable. v12 GRPO 35B-A3B died on cuda:0 with `400 MB free` while
cuda:1 still had ~1.5 GB headroom.

**How to apply:**
1. Before launching multi-GPU training, run `nvidia-smi --query-gpu=index,memory.free`
   and pick a contiguous set of cards with **equal free memory** for the training ranks.
2. Move auxiliary processes (judge vllm serve, embedding daemons, dev shells) onto
   the SAME card to consolidate asymmetry on one card and leave the rest clean.
3. On this local box specifically: GPU 0 = judge + daemon; GPU 1+2 = training
   (symmetric); GPU 3 = blocked by another user.
4. If you cannot achieve symmetry, the asymmetry must be reflected in the chosen
   `gpu_memory_utilization` (use the smaller-free-card fraction, not the average).

## vLLM rollout knobs (sm_80 / A100 / H100)

| Knob | Setting | Why |
|---|---|---|
| Flash Attention | default-on (sm_80+) | vLLM ≥0.18 picks `FLASH_ATTN` automatically; verify in log: `Using FLASH_ATTN attention backend`. |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | tune to free GPU | Run vLLM dummy-init once, read `Free memory on device cuda:N (XX/79.25 GiB)` from any past failure log, set ≤ XX/79.25. Co-resident FSDP shards eat free memory before vLLM init. |
| `+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce` | `true` (A100 PCIe only) | Workaround for `custom_all_reduce` CUDA `invalid argument` on A100 PCIe. Skip on H100/NVLink. |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` if model fits one GPU else `>=2` | TP=1 (replicate) gives more KV-cache headroom than TP=N for small models; for big models that don't fit one GPU, TP must shard. |

## Offload is the LAST resort, not a first knob

`actor.fsdp_config.param_offload=True` and `optimizer_offload=True` move FSDP
weights/optimizer to CPU. **Do NOT enable by default.** They (1) shift pressure
to CPU and inflate `update_weights` (FSDP→vLLM) RSS peaks — observed 226 GiB OOM
kill on 35B-A3B GRPO 2026-04-29 — and (2) slow down weight sync significantly.

Order of memory-reduction knobs to try BEFORE touching offload:

1. `gpu_memory_utilization` — retune to free memory after FSDP shard.
2. `+actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs` — reduce vLLM concurrency (each seq = Mamba state + KV cache).
3. `+actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=true` — skip CUDA graph capture buffers (~3-5 GiB).
4. `actor.ppo_micro_batch_size_per_gpu` — halve actor backward activations.
5. `actor.use_dynamic_bsz=True` + lower `ppo_max_token_len_per_gpu`.
6. `enable_gradient_checkpointing=True` (already on).

Only after all six fail to fit, consider `param_offload=True`.

For LoRA training specifically, offload buys nothing meaningful — base model is
frozen (no optimizer states), only LoRA adapters get optimized (~MB scale).

## Production token budget is non-negotiable

Knobs that affect policy training quality must NOT be reduced for "smoke" runs —
shrinking them produces a worse-trained policy than the production target,
defeating the smoke's purpose:

- `data.max_prompt_length`, `data.max_response_length`, `rollout.max_model_len`
- `rollout.n` (GRPO group size)
- `rollout.multi_turn.max_assistant_turns`
- `format` (e.g. `hermes`)
- KL settings (`use_kl_loss=True`, `kl_loss_type=low_var_kl`, `kl_loss_coef=0.001`)

Knobs that are pure scheduling/budget (OK to reduce for smoke):
`train_batch_size`, `total_training_steps`, `total_epochs`, `save_freq`,
`val_before_train`.

If production token-budget knobs cannot fit on the available topology, the
**topology is wrong, not the knobs** — add GPUs, switch to PP, or move to NRP.

## Comment every override that drifts from default

Every hydra/CLI override that diverges from a library default must carry an
inline comment explaining (a) what the default is, (b) why we override, (c)
known couplings/risks. See memory `comment_on_config_overrides.md`.

# Conda Environment

**MANDATORY: All experiment scripts (eval.py, precache_explores.py, and any other
research code under `Experiment/`) MUST run in the `explain` conda env.** This
is a hard requirement — the base env has different package versions and Python
3.13 vs explain's 3.11; results across envs are NOT comparable.

Invocation pattern for every launcher .sh:

```bash
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python eval.py \
    --config <path> ... \
    >  <log> 2>&1 &
```

- `-n explain` selects the env without `conda activate` (works inside scripts
  where activation hooks may not be sourced).
- `--no-capture-output` (alias `-s` / `--live-stream`) is **mandatory** for
  long-running background jobs. Without it `conda run` buffers stdout/stderr
  until process exit, leaving the log file empty until the run finishes —
  defeating the entire purpose of `tail -f`.
- `PYTHONUNBUFFERED=1` is kept as belt-and-suspenders even though
  `--no-capture-output` already streams.

Never call bare `python` in a launcher; that inherits whatever env was active
in the parent shell (usually `base`), which is silently wrong.

## Per-benchmark grading reference

Each benchmark's grading logic is FIXED — we never swap judge models or grading
strategies in practice. The `judge_model` class attribute and `_JUDGE_MODEL_CODEX`
mapping in `benchmarks/base.py` are flexibility hooks that are not currently
exercised. This table is the source of truth for "how is X graded".

Verified by reading `benchmarks/*.py` and `benchmarks/grader.py` on 2026-04-28.

### Routing logic (`benchmarks/grader.py:135-147`)

`grade_answer(predicted, gold, question, answer_type, judge_model, ...)` decides:

```
if answer_type == "multipleChoice":   return check_answer(...)         # string match A-E
if judge_model is None:               return check_answer(...)         # string match
else:                                 return judge_answer(...)         # LLM judge
```

So the routing depends on TWO things: per-row `answer_type` and per-class
`judge_model`. Below is the materialized truth table per benchmark.

### Per-benchmark table

| Benchmark | grade() override? | answer_type | judge_model | Effective grader |
|---|---|---|---|---|
| LCB | `lcb.py:142` → `grade_code(predicted, row)` | (bypassed) | `None` | Run predicted code against `public_test_cases` + `private_test_cases` via `lcb_runner.evaluation.compute_code_generation_metrics`. is_correct = all tests pass. |
| AIME (2025/2026) | `aime.py:119` → string equality | `exactMatch` | `None` | `_normalize_aime_answer(predicted) == _normalize_aime_answer(gold)`. Strips `\boxed{}`, `$…$`, lowercases. Integer-string match. |
| GPQA | none (uses `base.py:379`) | `multipleChoice` (`gpqa.py:104`) | `None` | grader.py short-circuits at `multipleChoice` → `check_answer` → `_extract_mc_letter` regex extracts A-E from predicted, compares to gold letter. |
| HLE | none (uses `base.py:379`) | per-row from dataset (usually `exactMatch`) | `claude-haiku-4-5-20251001` (`hle.py:123`) | LLM judge via `judge_answer`. Haiku reads (question, predicted, gold) and emits a yes/no semantic-equivalence verdict. Required because HLE answers are LaTeX/free-form text. |
| BabyVision | none (uses `base.py:379`) | hybrid: `multipleChoice` if `ansType=="choice"` else `exactMatch` (`babyvision.py:89`) | `claude-haiku-4-5-20251001` (`babyvision.py:50`) | Hybrid: choice questions → string match A-E; blank questions → LLM judge (Haiku). The judge_model only fires on blank questions. |
| RBenchV | none (uses `base.py:379`) | `exactMatch` (default from `base.py:372`) | `claude-haiku-4-5-20251001` (`rbenchv.py:40`) | LLM judge (Haiku). Visual reasoning answers are free-form, semantic equivalence required. |

### Key facts to remember

- **judge_model is always Haiku or None.** No benchmark currently uses any other
  judge model. The `_JUDGE_MODEL_CODEX` codex-mapping is dormant.
- **Three grading mechanisms exist**: code execution (LCB), string match
  (AIME / GPQA / BabyVision-choice), LLM judge (HLE / BabyVision-blank / RBenchV).
- **`judge_model = None` is correct for LCB / AIME / GPQA**, not a leftover.
  LCB doesn't need a judge (test cases). AIME's integer-string match doesn't need
  one. GPQA's multipleChoice short-circuits before judge_model is consulted.
- **Once-burned: HLE.** `hle.py:123`'s `judge_model` was temporarily set to
  `None` for a 2026-04-11 smoke test and not restored, causing all 100 HLE
  socratic-self-refine grades to fall through to `check_answer` with `gold=""`,
  underestimating accuracy by ~8 pp. Restored 2026-04-28. Lesson: any non-trivial
  override of these class defaults must carry a comment with date + rationale +
  rollback instruction.

### Cache validity

`eval.py:_grade_with_cache` (line 45-73) caches verdicts as `grade.json` and
trusts the cache only if the stored `judge_model` matches the current
`benchmark.judge_model`. This protects against the HLE-class bug (changing the
class attribute auto-invalidates old cache). It does NOT protect against:

- Backend swap (claude → codex): not part of cache key.
- Gold answer changes (dataset upstream updates): stored in cache file but not
  checked on read.
- Question text changes: same as above.
- LCB: `judge_model` is always `None` → cache key always matches → stale grades
  from lcb_runner bugs would persist forever. Mitigation: never modify
  lcb_runner without invalidating the LCB cache directory.

If you change any of the above (backend, dataset version, lcb_runner version,
judge prompt template), bump or wipe the affected cache directory; the
auto-invalidation only catches `judge_model`.

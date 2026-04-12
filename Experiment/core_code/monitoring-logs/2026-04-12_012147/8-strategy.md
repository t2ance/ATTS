# Step 8: Strategy

## Situation in one paragraph
The 8B-base HLE 3-GPU GRPO run (`atts-grpo/8b-base-hle-k16-3gpu`) started 2026-04-11 23:52, survived `val_before_train` (step 0 baseline: HLE acc 0.0, reward -0.2775, has_answer 0.3, num_turns_mean 15.3), then ran for 1h 17min attempting rollout for step 0 before crashing at 01:21:39 with CUDA OOM inside `_compute_ref_log_prob` on GPU 0. Zero training steps completed. Training process is fully gone (no verl/main_ppo in ps, GPUs 0/1/2 at 0% / 0 MiB). GPU 3 is occupied by another user's unrelated job. A secondary crisis is separate: `/data3` is 99% full (184 GB free of 19 TB), and `checkpoints/atts-grpo/8b-sft-3gpu` alone is 458 GB — the first `save_freq=6` would likely fail regardless of the OOM fix.

## Hypothesis ranking

Three falsifiable hypotheses, in order of likelihood given the evidence:

### H1 (confidence high): **Long-sequence ref `lm_head` allocation + vLLM KV reservation on the same GPU is the OOM root cause**
- Prediction if true: a re-run with `max_response_length ≤ 8192` (or ref log_prob via dynamic bsz split, or ref param offload) completes step 0 past ref_log_prob cleanly on the same hardware.
- Prediction if false: the OOM recurs at the same call site with shorter sequences, implying something else (memory fragmentation, hidden alloc) is in play.
- Evidence strength: strong. Reconstructed the byte budget on GPU 0 to within a few GB (~74 GB), and the `[2, 28672, 152K] bf16` lm_head tensor alone is 17.4 GB. A working peer run with max_response=4096 (lm_head tensor 9.9 GB) finished 78 steps on this machine.

### H2 (confidence medium): **The raw-base-model choice compounds H1 and makes GRPO signal-poor even if OOM is patched**
- Prediction if true: even after fixing memory, HLE val acc stays at 0 for many steps (GRPO has no non-zero advantages to propagate), and wall time bleeds into > 90 hours because rollouts max out response length every time.
- Prediction if false: base-init GRPO on HLE shows non-zero acc within the first 6-12 steps, validating it as a research path.
- Evidence strength: medium. The zero-acc baseline is a proven blocker for the *step-0 validation metric*. Whether GRPO can find a signal from that start is a research question I can't answer from the logs alone. Memory suggests "GRPO from base on this task is usually pathological", but that is a prior, not evidence about this run.

### H3 (confidence low-medium): **Disk pressure on /data3 would crash the run at first checkpoint save even after an OOM fix**
- Prediction if true: at step 6 save, Ray / verl throws a disk full error or spills fail. Need to free ~100+ GB before starting.
- Prediction if false: the run streams checkpoints into the 184 GB free space without error for 1-2 saves, then likely runs out.
- Evidence strength: medium. 184 GB free is below the ~130 GB peak a 2-ckpt rolling buffer needs on an 8B model with optimizer state; the margin is razor-thin.

## Options for the user (presented in AskUserQuestion)
- **Option A: Minimal fix, keep research question intact.** Keep raw base model; reduce `max_response_length` to 8192 (or 12288), drop `rollout.gpu_memory_utilization` to 0.25, keep everything else. Re-run. Preserves the "GRPO from base" research intent but accepts a shorter CoT budget.
- **Option B: Rebase on the proven SFT recipe.** Switch `MODEL_PATH` to `sft_qwen3_8b_merged` and copy the `8b-sft-3gpu` knobs (max_response 4096, n=8, max_assistant_turns=5, log_prob_micro_bs=4, gpu_mem_util=0.5). Effectively abandons "base → GRPO" and re-runs the proven path with new data. Very likely to complete end-to-end.
- **Option C: Stop, rethink, don't restart yet.** Use this session to tighten configs, free disk, and have a design conversation about whether the base→GRPO experiment is the right thing to spend 90+ hours of 3-GPU compute on.

Each option has a tied decision about **disk freeing** and **epoch budget (30 is huge)**. Those get their own questions.

## Success criteria I will check on the next session (if the user picks A or B)
- `val-core/atts_hle/acc/mean@1` visibly above 0 by step 6 (at minimum via has_answer climbing past 0.5)
- No OOM anywhere in the log
- Wall time per optimizer step ≤ 2h (or user-agreed budget)
- At least one checkpoint saved without disk error
- `evaluate_after` = 2-3 hours into the new run

## Non-starter alternatives (rejected and why)
- **Shrink micro_bs to 1 only, change nothing else**: insufficient headroom (halves lm_head alloc to 8.7 GB but still bumps against 6-8 GB of free space, very tight — brittle).
- **Increase GPU count**: GPU 3 belongs to another user for the foreseeable future. Not under my control.
- **Reduce n=16 → 8**: changes GRPO group size, halves rollouts per step, helps wall time but does **not** directly fix ref_log_prob OOM (the ref pass runs on actor-sized micro-batches regardless of n).

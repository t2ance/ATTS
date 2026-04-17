# Step 5: Resource Check

## GPU State (snapshot at 2026-04-16T00:16)

| GPU | Model | Used | Total | SM Util |
|-----|-------|------|-------|---------|
| 0 | A100 80GB PCIe | 0 MiB | 81920 MiB | 0% |
| 1 | A100 80GB PCIe | 0 MiB | 81920 MiB | 0% |
| 2 | A100 80GB PCIe | 74073 MiB | 81920 MiB | 0% |
| 3 | A100 80GB PCIe | 0 MiB | 81920 MiB | 0% |

GPU 0/1: freed after training crash. No training process holds memory.
GPU 2: Judge vllm server (`vllm serve Qwen/Qwen3-8B`, pid=322076, started Apr15) still loaded at 74073/81920 MiB (90.4%). SM=0% because no inference is running.
GPU 3: completely free.

## Process State

- No `main_ppo` / verl trainer process alive.
- No Ray worker processes from the GRPO job (they were killed when trainer exited).
- Judge vllm server still alive on GPU 2.
- Stale `tail -f grpo_8b_sft_2gpu.log` process (pid=2148583, from Apr13) — monitoring an older log, harmless.

## Disk / Checkpoint

- Only `global_step_3` saved. `latest_checkpointed_iteration.txt` = 3.
- Checkpoint dir: `/data3/peijia/dr-claw/Explain/Experiment/core_code/checkpoints/atts-grpo/8b-sft-2gpu-bs96/`
- Resume from step 3 is possible (step 4 will be re-run from scratch).

## Resource Assessment

No resource-level issue contributed to the crash:
- No OOM (GPUs 0/1 freed cleanly, no CUDA error in log)
- No disk full condition indicated
- Judge server is alive and was accepting requests (it was the judge's *output length*, not unavailability, that triggered the crash)
- CPU memory at step 4: `perf/cpu_memory_used_gb` = 137 GB (within expected range for 8B FSDP + Ray workers)

The crash is purely a code-level assertion failure, not a hardware or resource exhaustion event.

# Step 5: Resources

## GPU state (full snapshot, all 4 GPUs)
| GPU | Util | Memory | Temp | Power | Owned by |
|-----|------|--------|------|-------|----------|
| 0 | 0 % | 0 MiB / 80 GB | 45 C | 48 W | — (dead training) |
| 1 | 0 % | 0 MiB / 80 GB | 45 C | 50 W | — (dead training) |
| 2 | 0 % | 0 MiB / 80 GB | 44 C | 48 W | — (dead training) |
| 3 | 58 % | 36897 MiB / 80 GB | 58 C | 180 W | **other user `namdo`** (`pid=1887150`, `train_esm.py`, etime 03:29:13) |

No orphan training processes on GPU 0-2. Training footprint is fully reclaimed.

## Disk
- `/data3`: **184 GB free of 19 TB** — **99% full**. Ray logged `file_system_monitor` warnings every ~10s throughout the run. A checkpoint save during training would have competed for the remaining space.
- `/`: 35 GB free of 1.9 TB — **99% full** (less critical but tight).
- Biggest deletable: `checkpoints/atts-grpo/8b-sft-3gpu` = **458 GB** (holds 16 checkpoints from the earlier SFT-initialized GRPO run). If anything on this host needs to be freed to make room for a retry + future checkpoints, this is the obvious candidate (user decision).
- `checkpoints/` total: 509 GB
- `wandb/`: 11 MB (not a concern)
- `/data3/tmp/ray`: 3.2 GB (negligible)

## Host
- RAM: 39/251 GB used, 194 GB free, 16 GB buff/cache, 208 GB available. Healthy.
- Swap: **8/8 GB, fully used**. Not caused by training (training workers are gone and the host still has 194 GB free). Left over from some other workload; not a current blocker.
- Uptime: 71 days, 53 active sessions.
- Load average: `2.05 / 4.29 / 7.41` — dropping consistently with the crash having freed CPU load.
- Processes on host: ~1028.

## Processes
- `ps -ef | grep -E "verl|main_ppo|grpo_vllm"` → empty. No training residue.
- `ps -p 1887150` → `namdo` running `python train_esm.py` on GPU 3 (**unrelated**; this is why the script was configured for 3-GPU only).

## Implications for retry
1. **Disk must be freed before retry**, or a successful run will OOM the filesystem at the first checkpoint save (step 6). Candidate actions (needs user decision): delete old `8b-sft-3gpu` checkpoints you no longer need, move them off `/data3`, or switch `save_freq` higher and `max_actor_ckpt_to_keep` to 1.
2. **GPU 3 will remain unavailable** while `namdo` runs. We stay on 3 GPUs (0/1/2). The 3-GPU script is the correct choice for this machine state.
3. Host RAM is ample; no memory fix needed at the OS level.

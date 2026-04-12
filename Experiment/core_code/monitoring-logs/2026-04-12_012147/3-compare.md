# Step 3: Predictions vs Actuals

| ID | Prediction | Actual | Match | Significance |
|---|---|---|---|---|
| P1 | main Python process alive | **DEAD** — no verl/main_ppo in ps | MISS | CRITICAL |
| P2 | GPU 0/1/2 >70% util, GPU3 idle | **GPU0/1/2 = 0% util, 0 MiB**; GPU3 owned by another user | MISS (both) | CRITICAL |
| P3 | GPU mem ~55-75 GB/GPU | 0 MiB (process dead); at crash GPU0 had 74.09 GiB used | partial — matches my memory-pressure worry | — |
| P4 | val_before_train ran first | YES, completed and logged (line 1301) | HIT | confirms path |
| P5 | current step in [0, ~30], lean < 6 | step 0 (never completed) | HIT (lower bound) | CRITICAL |
| P6 | no atts-grpo/8b-base-hle checkpoint yet | confirmed — only `8b-sft-3gpu` exists | HIT | — |
| P7 | step wall time 15-40 min | step 0 ran 1h 17m 50s before OOM on ref log_prob | partial — rollout alone dominated; training never reached optimizer step | — |
| P8 | val_before_train 5-15 min | approx — val ran between ~23:56 and the "0/60 [00:00" marker; no explicit start/end line, bounded by log evidence | weak hit | — |
| P9 | reward ~0 at start (base model) | **val acc = 0.0, reward = -0.2775, has_answer = 0.3** | HIT (strong) | KEY: confirms the raw-base-model concern |
| P10 | no NaN/Inf, moderate KL | NEVER REACHED optimizer update | N/A | — |
| P11 | response length may trend toward max, truncation=error risk | OOM occurred on long-seq forward (seq up to 28672), consistent with P11's concern | HIT (in spirit) | contributes to root cause |
| A1 | zero/near-zero reward stuck | only 1 val data point, but 0% acc and -0.2775 reward → GRPO signal would have been degenerate | partial HIT | supports "base-init GRPO is pathological here" |
| A2 | truncation=error crash risk | did not trigger; failed by OOM first | miss (not this run's failure) | — |
| A3 | NaN/Inf/NCCL/OOM | **OOM confirmed** during ref log_prob | HIT | CRITICAL |
| A4 | validation eating >20% wall time | val vs total wall can't be split cleanly, but val ran at least once and was followed by 1h 17m of failing rollout — not the problem this round | — | — |
| A5 | silent hang | no — explicit crash with traceback | — | — |
| U1 | fresh run or resume? | **FRESH** (no prior ckpt exists for this experiment name) | resolved | — |
| U2 | running how long? | 1h 29 min from start to crash | resolved | — |
| U3 | why start GRPO from base? | unresolved — user decision, and a separate strategic question | still open | — |

## New surprises (not in predictions)
- **S1**: `/data3 is 99% full` (184 GB free of 18.5 TB). Ray flagged this continuously from line 16 onward. Even if the OOM were fixed, the first `save_freq=6` save would likely fail or fill the disk. An 8B actor ckpt + optimizer state × ckpt_to_keep=2 is on the order of 100-130 GB.
- **S2**: Dataloader printed `Size of train dataloader: 2, Size of val dataloader: 1` — only **2 batches per epoch** (not the 3 I loosely estimated). With `filter_overlong_prompts=True` + batch=63, the effective dataset is ~126 prompts out of the 180 in train.parquet, not all 180. Worth noting but not critical.
- **S3**: `val-aux/num_turns/max = 20` while config sets `max_assistant_turns=10`. The verl counter semantics ≠ config keyword (probably counts role alternations or includes tool responses). Not a crasher; flagged for later.
- **S4**: vLLM engines ran with `enable_sleep_mode: True` → in principle vLLM should release KV when training computes log_prob. Yet GPU 0 had 74 GB used at crash time. Either sleep mode isn't freeing the KV reservation before ref_log_prob, or the wake/offload sequence isn't being called. Plausible fix angle.

## Health assessment
**Status: CRITICAL** — process dead, zero training progress, root cause is a reproducible OOM that will recur on the next run unless configs change. Also separate disk crisis and separate strategic concern about base-model init.

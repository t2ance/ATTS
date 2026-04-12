"""GRPO training entrypoint with NCCL weight sync patch.

Applies monkey patch to replace CUDA IPC weight sync with NCCL broadcast,
then delegates to verl's main_ppo.

Usage: python -m training.grpo.run_grpo [hydra args...]
"""

from __future__ import annotations

import training.grpo.nccl_weight_sync_patch

training.grpo.nccl_weight_sync_patch.apply()

# Now run verl's main_ppo (hydra entry point)
import runpy

runpy.run_module("verl.trainer.main_ppo", run_name="__main__")

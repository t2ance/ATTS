"""Monkey-patch verl's SglangRollout to use NCCL weight sync instead of CUDA IPC.

Problem: verl 0.7.0 uses `update_weights_from_tensor` (CUDA IPC) for weight sync.
CUDA IPC handles accumulate across training steps, causing `resume_memory_occupation`
to hang after 4-5 steps (PyTorch "over 1000 memory blocks" issue).

Fix: Switch to `update_weights_from_distributed` (NCCL broadcast), which creates
zero CUDA IPC handles.

Usage: import this module before training starts.
    import training.grpo.nccl_weight_sync_patch
    training.grpo.nccl_weight_sync_patch.apply()

Ref: https://github.com/volcengine/verl/issues/3377
     https://github.com/pytorch/pytorch/issues/118859
"""

from __future__ import annotations

import logging
import socket
from typing import Generator

import aiohttp
import torch
from torch.distributed.tensor import DTensor

from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter as SglangRollout

logger = logging.getLogger(__name__)

NCCL_GROUP_NAME = "verl_nccl_weight_sync"


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


async def _init_nccl_group(self):
    """Initialize NCCL group between training worker (rank 0) and sglang scheduler (rank 1)."""
    master_port = _get_free_port()
    master_address = "127.0.0.1"

    # sglang scheduler joins as rank 1 (via HTTP -> scheduler subprocess)
    url = f"http://{self._engine.host}:{self._engine.port}/init_weights_update_group"
    payload = {
        "master_address": master_address,
        "master_port": master_port,
        "rank_offset": 1,
        "world_size": 2,
        "group_name": NCCL_GROUP_NAME,
        "backend": "nccl",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
            result = await resp.json()
            assert result.get("success", False), (
                f"sglang init_weights_update_group failed: {result.get('message', 'unknown')}"
            )

    # Training worker joins as rank 0 (using same rendezvous)
    from sglang.srt.utils.common import init_custom_process_group

    self._nccl_weight_pg = init_custom_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=2,
        rank=0,
        group_name=NCCL_GROUP_NAME,
    )
    logger.info(f"NCCL weight sync group ready: {master_address}:{master_port}")


async def _update_weights_nccl(
    self,
    weights: Generator[tuple[str, torch.Tensor], None, None],
    **kwargs,
):
    """Replacement for SglangRollout.update_weights.
    Uses NCCL broadcast instead of CUDA IPC handles."""
    if self.device_mesh["infer_tp"].get_local_rank() == 0:
        await self._init_server_adapter()

    # Materialize weights
    weight_list = []
    for name, tensor in weights:
        t = tensor.detach()
        if isinstance(t, DTensor):
            t = t.full_tensor()
        weight_list.append((name, t))

    # Init NCCL group on first call
    if not getattr(self, "_nccl_group_initialized", False):
        await _init_nccl_group(self)
        self._nccl_group_initialized = True

    # Broadcast each weight via NCCL (training worker is src=0)
    names = []
    dtypes = []
    shapes = []
    for name, tensor in weight_list:
        torch.distributed.broadcast(tensor, src=0, group=self._nccl_weight_pg)
        names.append(name)
        dtypes.append(str(tensor.dtype).replace("torch.", ""))
        shapes.append(list(tensor.shape))

    # Tell sglang scheduler to load the received weights
    if self.device_mesh["infer_tp"].get_local_rank() == 0:
        url = f"http://{self._engine.host}:{self._engine.port}/update_weights_from_distributed"
        payload = {
            "names": names,
            "dtypes": dtypes,
            "shapes": shapes,
            "group_name": NCCL_GROUP_NAME,
            "flush_cache": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                result = await resp.json()
                assert result.get("success", False), (
                    f"update_weights_from_distributed failed: {result.get('message', 'unknown')}"
                )

        await self._engine.flush_cache()


def apply():
    """Apply the monkey patch. Call once before training starts."""
    SglangRollout._original_update_weights = SglangRollout.update_weights
    SglangRollout.update_weights = _update_weights_nccl
    logger.info("Patched SglangRollout.update_weights: CUDA IPC -> NCCL broadcast")

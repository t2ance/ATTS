"""Bootstrap: inject structural_tag into vLLMHttpServer.generate to constrain
<tool_call>...</tool_call> generation to valid JSON at token level.

Why this file exists
--------------------
verl's HermesToolParser (verl/experimental/agent_loop/tool_parser.py:104)
calls json.loads directly on the raw text between <tool_call> and </tool_call>.
When the model emits invalid JSON inside those tags, json.loads fails and the
trajectory loses the final answer submission. Three observed failure modes:

  - Invalid \\escape  (LaTeX backslashes: \\alpha, \\sqrt, \\sum)
  - Expecting ',' delimiter  (model emits literal newline inside JSON string)
  - Extra data  (model emits content after the closing })

Post-parse sanitization (the previous approach) is fragile: it cannot fix
structural errors, and its regex incorrectly breaks \\alpha (valid JSON for
the string \alpha) by doubling the second backslash to \\\alpha.

Root fix: vLLM structural_tag constrains token sampling within
<tool_call>...</tool_call> to match a JSON schema at generation time.
Invalid JSON cannot be emitted -- all three error classes are eliminated.

Why sitecustomize.py
--------------------
Ray worker processes and the verl TaskRunner are non-interactive Python
interpreters. sitecustomize.py is auto-imported by Python's site module at
interpreter startup regardless of interactive mode. The training script sets
PYTHONPATH to include this directory, so every Ray-spawned Python process
loads this file first.

CRITICAL: Guard against FSDP worker processes
---------------------------------------------
sitecustomize.py runs in ALL Ray worker processes, including FSDP actor/ref
workers (WorkerDict). Importing vLLM in those processes triggers CUDA context
initialization that interferes with Ray's GPU rank allocation, causing NCCL
"Duplicate GPU detected" errors. The fix: skip the patch when WG_BACKEND=ray
(set by verl's _create_worker for all FSDP actors). Only apply when running
in the vLLMHttpServer actor (which does NOT have WG_BACKEND set).
"""
from __future__ import annotations

import json
import logging
import os

_logger = logging.getLogger("structural_tag_patch")

# JSON constraint applied to the <tool_call>...</tool_call> region.
#
# Prior schema was just {"type": "object"} -- technically valid but too
# permissive. Observed failure (2026-04-17): greedy decode (temp=0) kept
# emitting `,"arguments": {}` forever and never closed the object, because
# the FSA allowed unlimited additionalProperties + duplicate keys + no
# required fields. Model burned 16384-token response budget per rollout.
#
# Tightening below forces the FSA to accept ONLY:
#   {"name": "explore"|"StructuredOutput", "arguments": {...}}
# Once both keys appear, additionalProperties:false + required leave `}`
# as the only legal next token, breaking the infinite-key-append loop.
#
# arguments schema stays {"type": "object"} on purpose -- real tool
# arguments structure varies per tool, and tightening further risks
# blocking legitimate calls. tool_agent_loop validates argument content
# downstream.
#
# Revisit if: (a) xgrammar CUDA crashes worsen under tighter constraints
# (see #24107), (b) model gets stuck generating minimal {"arguments":{}}
# and never explores meaningfully, (c) new tool names are added and the
# enum must be extended.
_STRUCTURAL_TAG = json.dumps({
    "structures": [{
        "begin": "<tool_call>",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"enum": ["explore", "StructuredOutput"]},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        },
        "end": "</tool_call>",
    }],
    "triggers": ["<tool_call>"],
})


def _install_patch() -> None:
    # WG_BACKEND=ray is set by verl's _create_worker for ALL FSDP actors (actor,
    # ref, critic). Importing vLLM in those processes triggers CUDA init that
    # breaks Ray's GPU rank allocation. Skip entirely if we're an FSDP worker.
    if os.environ.get("WG_BACKEND") == "ray":
        return

    try:
        from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        # verl or vllm not installed -- nothing to patch.
        return

    if getattr(vLLMHttpServer, "_structural_tag_patched", False):
        return

    _orig_generate = vLLMHttpServer.generate

    async def _patched_generate(self, prompt_ids, sampling_params, request_id, **kwargs):
        patched_params = dict(sampling_params)
        patched_params["structured_outputs"] = StructuredOutputsParams(structural_tag=_STRUCTURAL_TAG)
        return await _orig_generate(self, prompt_ids, patched_params, request_id, **kwargs)

    vLLMHttpServer.generate = _patched_generate
    vLLMHttpServer._structural_tag_patched = True
    _logger.info("vLLMHttpServer.generate patched with structural_tag JSON constraint")


def _install_dp_group_patch() -> None:
    """verl PR #5591 backport: default dp_group=WORLD for prepare_dynamic_batch.

    Fixes NCCL deadlock where FSDP dp_actor's prepare_dynamic_batch calls
    silently skip num_micro_batches sync across DP ranks (verl 0.7.1 bug).
    Covers both compute_log_prob (line 468) and update_policy (line 560).

    Valid scope: pure FSDP, HSDP, Ulysses SP, TP=1. Does NOT override
    explicitly-passed dp_group.
    """
    import sys

    import torch.distributed as dist

    try:
        from verl.utils import seqlen_balancing
    except ImportError:
        return

    if getattr(seqlen_balancing, "_dp_group_patched", False):
        return
    _orig = seqlen_balancing.prepare_dynamic_batch

    def _patched(data, max_token_len, dp_group=None, **kw):
        if dp_group is None and dist.is_initialized():
            dp_group = dist.group.WORLD
        return _orig(data, max_token_len, dp_group=dp_group, **kw)

    seqlen_balancing.prepare_dynamic_batch = _patched
    seqlen_balancing._dp_group_patched = True
    for m in ("verl.workers.actor.dp_actor", "verl.workers.critic.dp_critic"):
        mod = sys.modules.get(m)
        if mod is not None and hasattr(mod, "prepare_dynamic_batch"):
            mod.prepare_dynamic_batch = _patched


def _install_hf_push_patch() -> None:
    """Auto-push each verl checkpoint to HF Hub via HfApi.upload_folder(run_as_future=True).

    verl 0.7.1 has no native HF Hub integration (verl#190 open 14mo). We hook
    RayPPOTrainer._save_checkpoint; after the original save completes, we queue
    a background upload of `$CKPT_DIR/global_step_{step}/` to the repo named by
    env var CHECKPOINT_REPO_ID. run_as_future=True returns immediately -- zero
    training stall; uploads drain on a daemon thread.

    Guard: WG_BACKEND=ray (FSDP worker) never reaches _save_checkpoint; patch
    is a no-op in workers. Guard on CHECKPOINT_REPO_ID unset so local runs work
    unchanged.
    """
    if os.environ.get("WG_BACKEND") == "ray":
        return
    repo_id = os.environ.get("CHECKPOINT_REPO_ID")
    if not repo_id:
        return

    try:
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from huggingface_hub import HfApi
    except ImportError:
        return

    if getattr(RayPPOTrainer, "_hf_push_patched", False):
        return

    _api = HfApi(token=os.environ.get("HF_TOKEN"))
    _api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    _orig_save = RayPPOTrainer._save_checkpoint

    def _save_and_push(self):
        _orig_save(self)
        step = self.global_steps
        folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        if not os.path.isdir(folder):
            return
        _api.upload_folder(
            repo_id=repo_id,
            folder_path=folder,
            path_in_repo=f"global_step_{step}",
            commit_message=f"step {step}",
            run_as_future=True,
        )
        _logger.info("queued HF upload for global_step_%d -> %s", step, repo_id)

    RayPPOTrainer._save_checkpoint = _save_and_push
    RayPPOTrainer._hf_push_patched = True
    _logger.info("RayPPOTrainer._save_checkpoint patched for HF push -> %s", repo_id)


def _install_lora_qkv_patch() -> None:
    """vllm#36372 / #36478 backport: bounds-checked slice_lora_b + set_lora,
    plus GQA-aware create_lora_weights override for MergedQKVParallelLinearWithLoRA.

    Bug class: enable_lora=True on Qwen3_5ForConditionalGeneration (and any GQA
    model with Q output dim != K/V output dim) crashes during
    profile_cudagraph_memory -> maybe_dummy_run_with_lora with one of:
      (A) IndexError: list index out of range  (PR#36395, merged 2026-04-17)
      (B) IndexError: too many indices for tensor of dimension 1  (PR#36603, open)
          OR RuntimeError: size of tensor a must match size of tensor b
          at non-singleton dimension 0
    Phase 0.3 dummy load on Qwen3.6-27B (TP=2 + LoRA + enforce_eager=False)
    reproduces (B) at column_parallel_linear.py:259 slice_lora_b.

    Without this patch the only workaround is enforce_eager=True, which costs
    ~50% rollout decode throughput because CUDA graph capture is bypassed.
    With this patch enforce_eager can return to False -> CUDA graphs restored ->
    measured power gate climbs from ~107 W to >150 W (target).

    Refs:
      - vllm#36372: https://github.com/vllm-project/vllm/issues/36372
      - vllm#36478: https://github.com/vllm-project/vllm/issues/36478
      - vllm PR#36395 (merged): https://github.com/vllm-project/vllm/pull/36395
      - vllm PR#36603 (open):  https://github.com/vllm-project/vllm/pull/36603

    Guard: skip in FSDP worker (WG_BACKEND=ray); only patches in vLLMHttpServer
    actor and its Worker_TP children where vLLM model load actually runs.
    """
    if os.environ.get("WG_BACKEND") == "ray":
        return

    try:
        import torch
        from vllm.distributed.utils import divide
        from vllm.lora.layers.column_parallel_linear import (
            MergedColumnParallelLinearWithLoRA,
            MergedQKVParallelLinearWithLoRA,
        )
    except ImportError:
        return

    if getattr(MergedColumnParallelLinearWithLoRA, "_lora_qkv_patched", False):
        return

    # Patch A1: slice_lora_b — guard `lora_b[i]` with `i < len(lora_b)`.
    def _patched_slice_lora_b(self, lora_b):
        sliced_lora_b = [None] * self.n_slices
        for i, (shard_id, shard_size) in enumerate(
            zip(self.output_ids, self.output_slices)
        ):
            if i < len(lora_b) and (lora_b_i := lora_b[i]) is not None:
                sliced_lora_b[i] = lora_b_i[
                    shard_size * shard_id : shard_size * (shard_id + 1), :
                ]
        return sliced_lora_b

    # Patch A2: set_lora — iterate via zip(strict=False) instead of range(n_slices)
    # so that lora_a/lora_b shorter than n_slices doesn't raise IndexError.
    def _patched_set_lora(self, index, lora_a, lora_b):
        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)
        for i, (lora_a_i, lora_b_i) in enumerate(
            zip(lora_a, lora_b, strict=False)
        ):
            if lora_a_i is not None:
                self.lora_a_stacked[i][
                    index, 0, : lora_a_i.shape[0], : lora_a_i.shape[1]
                ].copy_(lora_a_i, non_blocking=True)
            if lora_b_i is not None:
                self.lora_b_stacked[i][
                    index, 0, : lora_b_i.shape[0], : lora_b_i.shape[1]
                ].copy_(lora_b_i, non_blocking=True)

    MergedColumnParallelLinearWithLoRA.slice_lora_b = _patched_slice_lora_b
    MergedColumnParallelLinearWithLoRA.set_lora = _patched_set_lora

    # Patch B: GQA-aware create_lora_weights override on MergedQKVParallelLinearWithLoRA.
    # Allocates lora_b_stacked with per-slice output dims (q_proj_shard_size,
    # kv_proj_shard_size, kv_proj_shard_size) so that set_lora can copy
    # heterogeneous Q/K/V tensors without shape mismatch on GQA models.
    def _patched_create_lora_weights(
        self, max_loras, lora_config, model_config=None
    ):
        self.lora_config = lora_config
        lora_a_output_size_per_partition = (
            lora_config.max_lora_rank
            if not lora_config.fully_sharded_loras
            else divide(lora_config.max_lora_rank, self.tp_size)
        )
        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_output_size_per_partition,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                output_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for output_size in self.output_slices
        )

    MergedQKVParallelLinearWithLoRA.create_lora_weights = _patched_create_lora_weights
    MergedColumnParallelLinearWithLoRA._lora_qkv_patched = True
    _logger.info(
        "vllm LoRA QKV patches applied (issue #36372/#36478, PR#36395+#36603)"
    )


_install_dp_group_patch()
_install_patch()
_install_hf_push_patch()
_install_lora_qkv_patch()

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


_install_dp_group_patch()
_install_patch()

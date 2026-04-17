"""verl Tool: returns pre-cached explore results for ATTS GRPO training.

Stateless across calls. Per-rollout state lives in a class-level dict
keyed by `agent_data.request_id` because:
- verl's `tool_agent_loop._call_tool()` invokes `create()` / `release()`
  around every single `execute()` call (not once per rollout), so
  `self._instances[instance_id]` doesn't persist -- `create()` generates
  a fresh uuid each time.
- `agent_data.extra_fields` cannot be used either: verl's output
  aggregation (`verl.utils.py_functional.list_of_dict_to_dict_of_list`)
  asserts that every rollout's `extra_fields` shares the same keys as
  the first one in the list, and we only populate our key in rollouts
  that actually call explore.
- `agent_data.request_id` is the only stable per-rollout identifier
  (`tool_agent_loop.py:133` sets it via `uuid4().hex` once per rollout
  inside `run()`).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# Make `methods.tool_io`, `methods.tool_state` importable from verl worker.
_CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CORE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_CODE_DIR))

from methods.tool_io import CandidateRecord, FullRenderer  # noqa: E402
from methods.tool_state import ExploreStepState, advance  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_RENDERER = FullRenderer()


class ExploreTool(BaseTool):
    """Returns pre-cached explore results; shuffle seeded by request_id
    for stable per-rollout order, varied across rollouts.
    """

    # Per-rollout state, keyed by agent_data.request_id. See module docstring
    # for why neither self._instances[instance_id] nor
    # agent_data.extra_fields works.
    _rollout_state: dict[str, dict] = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        return (instance_id or str(uuid4())), ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs["agent_data"]
        key = agent_data.request_id
        bucket = self._rollout_state.get(key)
        if bucket is None:
            # verl's `_call_tool` forwards `create_kwargs` only to `create()`,
            # not to `execute()`. The sole persistent per-rollout source is
            # `agent_data.tools_kwargs` (stored once when AgentData is built).
            create_kwargs = agent_data.tools_kwargs.get("explore", {}).get("create_kwargs", {})
            cached = tuple(create_kwargs["cached_explores"])
            max_explores = int(create_kwargs.get("max_explores", len(cached)))
            assert len(cached) >= max_explores, (
                f"cached_explores ({len(cached)}) must have at least "
                f"max_explores ({max_explores}) items"
            )
            # Order fixed by prepare_data_hle.py offline shuffle (per-qid seed).
            # All n rollouts of the same sample consume the same pre-shuffled order,
            # matching reward_fn's positional-index grade lookup.
            bucket = {
                "cached_explores": cached,
                "explore_state": ExploreStepState(max_explores=max_explores),
            }
            self._rollout_state[key] = bucket

        bucket["explore_state"] = advance(bucket["explore_state"])
        step = bucket["explore_state"]
        explore = bucket["cached_explores"][step.used - 1]

        record = CandidateRecord(
            idx=step.used,
            answer=explore["answer"],
            confidence=float(explore["confidence"]),
            approach=explore["approach"],
            reasoning=explore["reasoning"],
            cost_usd=float(explore.get("cost_usd", 0.0)),
            used=step.used,
            max_explores=step.max_explores,
            cache_id=str(explore.get("cache_id", "")),
        )
        response = ToolResponse(text=_RENDERER.render(record))

        # Opportunistic cleanup: drop this rollout's entry once the budget
        # is exhausted (model can't call explore again). Rollouts that
        # early-stop before exhaustion leak a small entry (~few KB); the
        # class-level dict grows bounded by concurrent-rollouts-in-flight,
        # not by total rollout count.
        if step.is_exhausted:
            self._rollout_state.pop(key, None)

        return response, 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        pass

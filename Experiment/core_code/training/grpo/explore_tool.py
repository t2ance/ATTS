"""verl Tool: returns cached explore results for ATTS GRPO training."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

# Make `methods.tool_io` importable from inside verl's worker process.
_CORE_CODE_DIR = Path(__file__).resolve().parent.parent.parent
if str(_CORE_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CORE_CODE_DIR))

from methods.tool_io import CandidateRecord, FullRenderer  # noqa: E402

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_RENDERER = FullRenderer()


class ExploreTool(BaseTool):
    """Returns pre-cached explore results sequentially.

    On create(), receives a list of cached explore result dicts and the
    max_explores budget cap. Each execute() returns the next cached result
    rendered via the canonical `methods.tool_io.FullRenderer`, so the GRPO
    rollout observes the same tool-return string format as the production
    Claude orchestrator and the SFT data builder.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict] = {}

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        # Copy (do not mutate shared parquet-backed list) and shuffle per rollout.
        # Diversity source: ATTS orchestration is order-sensitive (early stop,
        # convergence detection), so the same 8 cached explores in different
        # orders yield different trajectories. Combined with rollout temperature,
        # this multiplies rollout variance at zero extra data cost.
        cached_explores = list(create_kwargs.get("cached_explores", []))
        random.shuffle(cached_explores)
        max_explores = int(create_kwargs.get("max_explores", len(cached_explores)))
        self._instances[instance_id] = {
            "cached_explores": cached_explores,
            "call_count": 0,
            "max_explores": max_explores,
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instances[instance_id]
        idx = state["call_count"]
        state["call_count"] += 1

        if idx >= len(state["cached_explores"]):
            return ToolResponse(text="No more cached explores available."), 0.0, {}

        explore = state["cached_explores"][idx]
        used = idx + 1
        record = CandidateRecord(
            idx=used,
            answer=explore["answer"],
            confidence=float(explore["confidence"]),
            approach=explore["approach"],
            reasoning=explore["reasoning"],
            cost_usd=float(explore.get("cost_usd", 0.0)),
            used=used,
            max_explores=state["max_explores"],
        )
        return ToolResponse(text=_RENDERER.render(record)), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

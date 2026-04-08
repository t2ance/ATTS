"""verl Tool: returns cached explore results for ATTS GRPO training."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ExploreTool(BaseTool):
    """Returns pre-cached explore results sequentially.

    On create(), receives a list of cached explore result dicts.
    Each execute() returns the next result formatted as the orchestrator expects.
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
        cached_explores = create_kwargs.get("cached_explores", [])
        self._instances[instance_id] = {
            "cached_explores": cached_explores,
            "call_count": 0,
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
        candidate_num = idx + 1
        text = (
            f"Candidate #{candidate_num} recorded.\n"
            f"- Answer: {explore['answer']}\n"
            f"- Confidence: {explore['confidence']}\n"
            f"- Approach: {explore['approach']}\n"
            f"- Reasoning: {explore['reasoning']}"
        )
        return ToolResponse(text=text), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

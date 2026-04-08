"""verl Tool: captures the final StructuredOutput answer for ATTS GRPO training."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AnswerTool(BaseTool):
    """Captures the orchestrator's final answer via StructuredOutput.

    Records the answer for grading. Step reward is always 0.0 --
    trajectory-level reward is computed by the custom reward function.
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
        self._instances[instance_id] = {
            "ground_truth": create_kwargs.get("ground_truth", ""),
            "submitted_answer": None,
        }
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        state = self._instances[instance_id]
        answer = parameters.get("answer", "")
        state["submitted_answer"] = answer
        return ToolResponse(text=f"Answer recorded: {answer}"), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

"""verl Tool: captures the final StructuredOutput answer for ATTS GRPO training.

Also monkey-patches ToolAgentLoop to treat StructuredOutput as a terminal tool:
after the tool executes and its response is added to the token sequence, the loop
returns TERMINATED instead of GENERATING. This mirrors inference behavior where
StructuredOutput ends the episode (backends/claude.py output_format, backends/vllm.py
explicit return on StructuredOutput detection).

The patch is applied at module import time. Since veRL loads this module via
tool_config.yaml -> initialize_tools_from_config, it runs before any rollout.
Survives veRL package reinstalls because the patch lives in project code.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# --- Monkey-patch: terminal tool support for ToolAgentLoop ---

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

_TERMINAL_TOOLS = frozenset({"StructuredOutput"})
_original_handle_processing_tools = ToolAgentLoop._handle_processing_tools_state


async def _patched_handle_processing_tools_state(self, agent_data):
    result = await _original_handle_processing_tools(self, agent_data)
    if result == AgentState.GENERATING and any(
        tc.name in _TERMINAL_TOOLS for tc in agent_data.tool_calls
    ):
        return AgentState.TERMINATED
    return result


ToolAgentLoop._handle_processing_tools_state = _patched_handle_processing_tools_state
logger.info("Patched ToolAgentLoop: StructuredOutput is now a terminal tool")


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

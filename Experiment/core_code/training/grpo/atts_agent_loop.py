"""Custom AgentLoop for ATTS GRPO training.

Replaces verl's generic ToolAgentLoop with ATTS-specific logic:
- Explore tool returns cached results directly (no BaseTool registry)
- StructuredOutput tool records the answer and terminates
- Tool response format matches eval code (tts_agent.py) exactly

Registered as "atts_agent" in verl's agent loop registry.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any
from uuid import uuid4

from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AsyncLLMServerManager,
    DictConfigWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Tool schemas -- must match eval code (tts_agent.py EXPLORE_TOOL + benchmarks/base.py EXPLORE_SCHEMA)
EXPLORE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "explore",
        "description": (
            "Dispatch a fresh, independent solver on the original problem. "
            "Returns a structured candidate with answer, reasoning, approach, and confidence."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

STRUCTURED_OUTPUT_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "StructuredOutput",
        "description": "Submit the final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "approach": {
                    "type": "string",
                    "description": "What method/angle you used (one sentence)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Detailed step-by-step reasoning",
                },
                "answer": {
                    "type": "string",
                    "description": "The final answer only -- a short, direct value",
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in this answer (0.0 - 1.0)",
                },
            },
            "required": ["approach", "reasoning", "answer", "confidence"],
        },
    },
}

TOOL_SCHEMAS = [EXPLORE_TOOL_SCHEMA, STRUCTURED_OUTPUT_TOOL_SCHEMA]

MAX_EXPLORES = 8


def _format_explore_result(explore: dict, call_count: int) -> str:
    """Format cached explore result to match eval code (tts_agent.py:102-109)."""
    used = call_count
    remaining = MAX_EXPLORES - used
    return (
        f"Candidate #{used} recorded.\n"
        f"- Answer: {explore['answer']}\n"
        f"- Confidence: {explore.get('confidence', 'N/A')}\n"
        f"- Approach: {explore.get('approach', 'N/A')}\n"
        f"- Reasoning: {explore.get('reasoning', 'N/A')}\n"
        f"- Cost: $0.00\n\n"
        f"Explore budget: {used}/{MAX_EXPLORES} used, {remaining} remaining."
    )


@register("atts_agent")
class ATTSAgentLoop(AgentLoopBase):
    """ATTS-specific agent loop for GRPO training.

    Handles two tools:
    - explore: returns pre-cached explore results sequentially
    - StructuredOutput: records the final answer and terminates
    """

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset] = RLHFDataset,
        dataset_config=None,
        **kwargs,
    ):
        super().__init__(
            trainer_config, server_manager, tokenizer, processor,
            dataset_cls=dataset_cls, dataset_config=dataset_config or {}, **kwargs,
        )
        config = trainer_config.config
        self.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        self.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.tool_parser = ToolParser.get_tool_parser("hermes", self.tokenizer)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        tools_kwargs = kwargs.get("tools_kwargs", {})
        cached_explores = tools_kwargs.get("explore", {}).get("create_kwargs", {}).get("cached_explores", [])

        request_id = uuid4().hex
        explore_count = 0
        assistant_turns = 0
        user_turns = 0

        # Tokenize initial prompt with tool schemas
        prompt_ids = await self.apply_chat_template(messages, tools=TOOL_SCHEMAS)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []

        for _turn in range(self.max_assistant_turns):
            # Generate model response
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

            assistant_turns += 1
            new_ids = output.token_ids
            prompt_ids = prompt_ids + new_ids
            response_ids += new_ids
            response_mask += [1] * len(new_ids)
            if output.log_probs:
                response_logprobs += output.log_probs

            # Check length limit
            if len(response_mask) >= self.response_length:
                break
            if assistant_turns >= self.max_assistant_turns:
                break

            # Parse tool calls from generated tokens
            _, tool_calls = await self.tool_parser.extract_tool_calls(new_ids)
            if not tool_calls:
                break

            # Process first tool call only (ATTS does one tool call per turn)
            tc = tool_calls[0]
            args = json.loads(tc.arguments)

            if tc.name == "StructuredOutput":
                # Answer submitted, terminate
                break

            if tc.name == "explore":
                explore_count += 1
                if explore_count <= len(cached_explores):
                    explore = cached_explores[explore_count - 1]
                    tool_text = _format_explore_result(explore, explore_count)
                else:
                    tool_text = "No more cached explores available."
            else:
                tool_text = f"Unknown tool: {tc.name}"

            # Truncate tool response if needed
            if len(tool_text) > self.max_tool_response_length:
                tool_text = tool_text[: self.max_tool_response_length] + "...(truncated)"

            # Tokenize tool response and append as non-trainable tokens
            tool_msg = [{"role": "tool", "content": tool_text}]
            tool_ids = await self.apply_chat_template(tool_msg, remove_system_prompt=True)

            if len(response_mask) + len(tool_ids) >= self.response_length:
                break

            prompt_ids = prompt_ids + tool_ids
            response_ids += tool_ids
            response_mask += [0] * len(tool_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(tool_ids)
            user_turns += 1

        # Split prompt_ids into prompt and response
        actual_prompt = prompt_ids[: len(prompt_ids) - len(response_mask)]
        actual_response = prompt_ids[len(actual_prompt):]

        return AgentLoopOutput(
            prompt_ids=actual_prompt,
            response_ids=actual_response[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            num_turns=assistant_turns + user_turns,
            metrics={},
            extra_fields={"explore_count": explore_count},
        )

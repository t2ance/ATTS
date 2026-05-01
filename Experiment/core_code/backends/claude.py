"""Claude Agent SDK backend.

Exposes transport primitives only:
  - call_sub_model(...)         -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)  -> (cost_usd, usage, exit_reason)
    where exit_reason ∈ {"committed", "cap_exceeded", "incomplete"}
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

_SDK_INIT_MAX_RETRIES = 3
_SDK_INIT_RETRY_DELAY = 5.0  # seconds

# Usage Policy refusals appear transient on some qids; retry twice before crashing.
# Anthropic docs say "try rephrasing" but same input may succeed on retry (~30s gap).
_POLICY_MAX_RETRIES = 2
_POLICY_RETRY_DELAY = 30.0  # seconds


def _is_sdk_init_error(e: Exception) -> bool:
    """Check if an exception is a transient SDK subprocess initialization error."""
    return "Control request timeout" in str(e) or "ProcessError" in type(e).__name__


def _is_policy_error(e: Exception) -> bool:
    """Check if an exception is a Claude Usage Policy refusal."""
    return isinstance(e, RuntimeError) and "Usage Policy" in str(e)


class MalformedToolCallError(RuntimeError):
    """Model produced a tool call with XML-embedded parameters instead of proper JSON."""
    pass


def _check_structured_output(result: dict) -> None:
    """Raise MalformedToolCallError if the model embedded fields using XML tags."""
    for v in result.values():
        if isinstance(v, str) and '<parameter name="' in v:
            raise MalformedToolCallError(
                f"Model used XML <parameter> tags inside JSON field. Keys present: {list(result.keys())}"
            )

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    UserMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
    tool,
    create_sdk_mcp_server,
)
from claude_agent_sdk.types import StreamEvent
from multimodal_input import build_claude_prompt_events


# All built-in Claude Code tools to disallow in our conversations.
# We only want the model to use our MCP tools (explore, integrate, etc.).
_ALL_BUILTIN_TOOLS = [
    "Task", "AskUserQuestion", "Agent",
    "Bash", "BashOutput", "KillBash",
    "Read", "Write", "Edit",
    "Glob", "Grep",
    "NotebookEdit",
    "WebFetch", "WebSearch",
    "TodoWrite",
    "ExitPlanMode",
    "ListMcpResources", "ReadMcpResource",
]


# ---------------------------------------------------------------------------
# Sub-model call (single structured query)
# ---------------------------------------------------------------------------

async def call_sub_model(
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    output_schema: dict[str, Any],
    writer,
    budget_tokens: int = 32000,
    effort: str | None = None,
) -> tuple[dict[str, Any], str, float, dict[str, Any]]:
    """Call a sub-model via Claude Agent SDK.

    Returns (structured_output, trajectory_text, cost_usd, usage).
    """
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=10,
        model=model,
        output_format={"type": "json_schema", "schema": output_schema},
        thinking={"type": "enabled", "budget_tokens": budget_tokens},
        effort=effort,
        disallowed_tools=_ALL_BUILTIN_TOOLS,
        include_partial_messages=True,
    )

    policy_attempts = 0
    while True:
        trajectory_parts: list[str] = []
        cost_usd = 0.0
        usage: dict[str, Any] = {}
        result: dict[str, Any] | None = None

        try:
            for attempt in range(_SDK_INIT_MAX_RETRIES):
                try:
                    async with ClaudeSDKClient(options=options) as client:
                        await client.query(
                            build_claude_prompt_events(user_message, image_data_url)
                        )
                        async for msg in client.receive_response():
                            if isinstance(msg, StreamEvent):
                                event = msg.event
                                if event.get("type") == "content_block_delta":
                                    delta = event.get("delta", {})
                                    if delta.get("type") == "thinking_delta":
                                        writer.write_chunk(delta["thinking"])
                            elif isinstance(msg, AssistantMessage):
                                if msg.error:
                                    if msg.error == "max_output_tokens":
                                        result = {"timed_out": True}
                                        continue
                                    error_text = " ".join(
                                        block.text for block in msg.content if isinstance(block, TextBlock)
                                    )
                                    raise RuntimeError(f"Claude API error ({msg.error}): {error_text}")
                                for block in msg.content:
                                    if isinstance(block, ThinkingBlock):
                                        trajectory_parts.append(f"{block.thinking}\n\n")
                                        writer.write_chunk("\n\n")
                                    elif isinstance(block, ToolUseBlock) and block.name == "StructuredOutput":
                                        result = block.input
                            elif isinstance(msg, ResultMessage):
                                cost_usd = msg.total_cost_usd
                                usage = msg.usage
                    break  # success
                except Exception as e:
                    if _is_sdk_init_error(e) and attempt < _SDK_INIT_MAX_RETRIES - 1:
                        print(f"  [sub-model] SDK init error (attempt {attempt + 1}/{_SDK_INIT_MAX_RETRIES}), retrying in {_SDK_INIT_RETRY_DELAY}s: {e}")
                        await asyncio.sleep(_SDK_INIT_RETRY_DELAY)
                        continue
                    raise
        except Exception as e:
            if _is_policy_error(e) and policy_attempts < _POLICY_MAX_RETRIES:
                policy_attempts += 1
                print(f"  [sub-model] Usage Policy refusal (attempt {policy_attempts}/{_POLICY_MAX_RETRIES}), retrying in {_POLICY_RETRY_DELAY}s: {e}")
                await asyncio.sleep(_POLICY_RETRY_DELAY)
                continue
            raise
        break  # success

    assert result is not None, "Sub-model did not call StructuredOutput"
    if result.get("timed_out"):
        writer.close()
        return result, "".join(trajectory_parts), cost_usd, usage
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    trajectory_parts.append(f"```json\n{json_str}\n```\n\n")
    writer.write_text(f"```json\n{json_str}\n```")
    writer.close()
    return result, "".join(trajectory_parts), cost_usd, usage


# ---------------------------------------------------------------------------
# Generic tool-calling conversation
# ---------------------------------------------------------------------------

_MCP_PREFIX = "mcp__delegated__"


def _build_mcp_tools(
    tool_defs: list[dict[str, Any]],
    tool_handler: Callable[[str, dict], Awaitable[tuple[str, bool]]],
):
    """Build MCP tool functions from generic tool definitions.

    Returns (mcp_tools, get_should_stop).
    """
    _should_stop = False

    mcp_tools = []
    for td in tool_defs:
        name, desc, params = td["name"], td["description"], td["parameters"]

        async def _handler(args: dict[str, Any], _name: str = name) -> dict[str, Any]:
            nonlocal _should_stop
            text, stop = await tool_handler(_name, args)
            if stop:
                _should_stop = True
            return {"content": [{"type": "text", "text": text}]}

        mcp_tools.append(tool(name, desc, params)(_handler))

    return mcp_tools, lambda: _should_stop


async def run_tool_conversation(
    *,
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    tools: list[dict[str, Any]],
    max_turns: int,
    tool_handler: Callable[[str, dict], Awaitable[tuple[str, bool]]],
    effort: str | None = None,
    output_format: dict[str, Any] | None = None,
    writer=None,
    quiet: bool = True,
    on_structured_output: Callable[[dict], None] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[float, dict[str, Any], bool]:
    """Run a multi-turn tool-calling conversation via Claude Agent SDK.

    tool_handler(name, args) -> (result_text, should_stop)
    writer: if provided, automatically writes all events to the trajectory.
    quiet: if False, prints events to console.
    output_format: if provided, the model can call the built-in StructuredOutput
        tool to emit structured data (using the same mechanism as call_sub_model).
    on_structured_output: callback for StructuredOutput business logic (state updates).
    Returns (cost_usd, usage, exit_reason) where exit_reason ∈
    {"committed", "cap_exceeded", "incomplete"}.
    """
    # Append StructuredOutput schema to system prompt when output_format is provided
    if output_format and output_format.get("schema"):
        schema_json = json.dumps(output_format["schema"], indent=2)
        required = output_format["schema"].get("required", [])
        system_prompt = (
            system_prompt +
            "\n\nWhen you are ready to submit your final answer, call the StructuredOutput tool. "
            "Use strict JSON format for all parameters. Do not use XML tags like <parameter>. "
            f"All of these fields are required and must each be a separate JSON key: {', '.join(required)}.\n"
            f"Schema:\n```json\n{schema_json}\n```\n"
        )

    mcp_tools, get_should_stop = _build_mcp_tools(tools, tool_handler)
    mcp_server = create_sdk_mcp_server(
        name="delegated", version="1.0.0", tools=mcp_tools,
    )

    allowed = [f"{_MCP_PREFIX}{td['name']}" for td in tools]
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        mcp_servers={"delegated": mcp_server},
        allowed_tools=allowed,
        disallowed_tools=_ALL_BUILTIN_TOOLS,
        max_turns=max_turns,
        model=model,
        effort=effort,
        output_format=output_format,
    )

    cost_usd = 0.0
    usage: dict[str, Any] = {}
    _output_tokens = 0
    # Default 'incomplete'; promoted to 'committed' on StructuredOutput
    # tool_use, or 'cap_exceeded' on token cap break.
    _exit_reason = "incomplete"
    _structured_output_emitted = False

    for attempt in range(_SDK_INIT_MAX_RETRIES):
        try:
            async with ClaudeSDKClient(options=options) as client:
                await client.query(build_claude_prompt_events(user_message, image_data_url))
                async for msg in client.receive_response():
                    if isinstance(msg, StreamEvent):
                        event = msg.event
                        event_type = event.get("type")
                        if event_type == "content_block_delta":
                            delta = event.get("delta", {})
                            if delta.get("type") == "thinking_delta" and writer:
                                writer.write_chunk(delta["thinking"])
                            # Note: char counting removed (2026-04-30). Token-based
                            # cap is checked on message_delta events below — that's
                            # Anthropic's natural unit (cumulative usage.output_tokens).
                        elif event_type == "message_delta":
                            # Anthropic streaming protocol: message_delta events
                            # carry cumulative usage in their `usage` field. We use
                            # this to gate orchestrator output by token budget.
                            evt_usage = event.get("usage", {})
                            if "output_tokens" in evt_usage:
                                _output_tokens = evt_usage["output_tokens"]
                                if max_output_tokens is not None and _output_tokens > max_output_tokens:
                                    if not quiet:
                                        print(
                                            f"  [orchestrator] output token cap exceeded "
                                            f"({_output_tokens} > {max_output_tokens}), terminating"
                                        )
                                    _exit_reason = "cap_exceeded"
                                    break

                    elif isinstance(msg, AssistantMessage):
                        if msg.error:
                            error_text = " ".join(
                                block.text for block in msg.content if isinstance(block, TextBlock)
                            )
                            raise RuntimeError(f"Claude API error ({msg.error}): {error_text}")
                        for block in msg.content:
                            if isinstance(block, ThinkingBlock):
                                if writer:
                                    writer.write_text(f"\n\n<thinking>\n{block.thinking}\n</thinking>\n\n")
                            elif isinstance(block, TextBlock):
                                if not quiet:
                                    print(f"[orchestrator] {block.text}")
                                if writer:
                                    writer.write_text(block.text)
                            elif isinstance(block, ToolUseBlock):
                                if block.name == "StructuredOutput":
                                    _check_structured_output(block.input)
                                    if not quiet:
                                        print(f"[structured_output] {block.input}")
                                    if writer:
                                        writer.write_tool_use("StructuredOutput", block.input)
                                    if on_structured_output:
                                        on_structured_output(block.input)
                                    _structured_output_emitted = True
                                elif block.name.startswith(_MCP_PREFIX):
                                    tool_name = block.name.removeprefix(_MCP_PREFIX)
                                    if not quiet:
                                        print(f"[tool_use] {tool_name}")
                                    if writer:
                                        writer.write_tool_use(tool_name, block.input)

                    elif isinstance(msg, UserMessage):
                        if isinstance(msg.content, list):
                            for block in msg.content:
                                if isinstance(block, ToolResultBlock):
                                    text = ""
                                    if isinstance(block.content, str):
                                        text = block.content
                                    elif isinstance(block.content, list):
                                        text = "\n".join(
                                            item["text"] for item in block.content
                                            if isinstance(item, dict) and "text" in item
                                        )
                                    if writer:
                                        writer.write_tool_result(text)
                        if get_should_stop():
                            break

                    elif isinstance(msg, ResultMessage):
                        cost_usd = msg.total_cost_usd
                        usage = msg.usage
            break  # success
        except Exception as e:
            is_retryable = _is_sdk_init_error(e) or isinstance(e, MalformedToolCallError)
            if is_retryable and attempt < _SDK_INIT_MAX_RETRIES - 1:
                print(f"  [orchestrator] retryable error (attempt {attempt + 1}/{_SDK_INIT_MAX_RETRIES}), retrying in {_SDK_INIT_RETRY_DELAY}s: {e}")
                await asyncio.sleep(_SDK_INIT_RETRY_DELAY)
                continue
            raise

    # Final exit_reason: cap_exceeded (set above on break) wins; otherwise,
    # if StructuredOutput emitted then committed; else fall through to default
    # 'incomplete' (loop ended without commit signal).
    if _exit_reason == "cap_exceeded":
        return cost_usd, usage, _exit_reason
    if _structured_output_emitted:
        return cost_usd, usage, "committed"
    return cost_usd, usage, "incomplete"

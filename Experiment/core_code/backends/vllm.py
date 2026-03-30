"""vLLM backend for local model inference.

Exposes the same transport primitives as claude.py:
  - call_sub_model(...)         -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)  -> (cost_usd, usage)

Connects to a vLLM server (started separately via ``vllm serve``).
Uses native OpenAI tool-calling API.  Falls back to ``<tool_call>`` tag
parsing when the model does not emit native tool_calls.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VLLM_BASE_URL = "http://localhost:8000/v1"
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="not-needed")
    return _client


# Cost estimation ($/1M tokens) — configurable for reporting
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "default": (0.0, 0.0),  # local models are "free"
}


def _estimate_cost_usd(usage: dict[str, Any], model: str) -> float:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    input_price, output_price = PRICING_PER_1M.get(model, PRICING_PER_1M["default"])
    return input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000


def _build_user_content(text: str, image_data_url: str | None) -> str | list:
    """Build user message content, with optional image."""
    if not image_data_url:
        return text
    return [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]


# ---------------------------------------------------------------------------
# <tool_call> parsing (fallback for text-mode tool calling)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse all ``<tool_call>`` blocks from generated text.

    Returns list of (tool_name, arguments_dict).
    Skips blocks with malformed JSON.
    """
    results = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            call = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = call.get("name", "")
        args = call.get("arguments", {})
        results.append((name, args))
    return results


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
    """Single structured query. Returns (result_dict, trajectory, cost, usage)."""
    client = _get_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(user_message, image_data_url)},
    ]

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=budget_tokens,
        extra_body={"guided_json": output_schema} if output_schema else {},
    )

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    cost = _estimate_cost_usd(usage, model)

    # Parse structured output — try direct JSON, then <tool_call>, then regex
    result: dict = {}
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        calls = parse_tool_calls(text)
        for name, args in calls:
            if name == "StructuredOutput":
                result = args
                break

    assert result, f"Sub-model returned unparseable output: {text[:200]}"

    trajectory = text
    if writer:
        writer.write_text(text)
        writer.close()

    return result, trajectory, cost, usage


# ---------------------------------------------------------------------------
# Multi-turn tool-calling conversation
# ---------------------------------------------------------------------------

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
) -> tuple[float, dict[str, Any]]:
    """Run a multi-turn tool-calling conversation via vLLM OpenAI-compatible API.

    Uses native OpenAI tool-calling API. Falls back to ``<tool_call>`` tag
    parsing when the model does not emit native tool_calls.
    """
    client = _get_client()

    # Append StructuredOutput schema to system prompt (same as claude.py)
    if output_format and output_format.get("schema"):
        schema_json = json.dumps(output_format["schema"], indent=2)
        required = output_format["schema"].get("required", [])
        system_prompt = (
            system_prompt
            + "\n\nWhen you are ready to submit your final answer, call the StructuredOutput tool. "
            "Use strict JSON format for all parameters. "
            f"All of these fields are required and must each be a separate JSON key: {', '.join(required)}.\n"
            f"Schema:\n```json\n{schema_json}\n```\n"
        )

    # Build OpenAI-style tools array (native tool calling)
    openai_tools = []
    for td in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": td["name"],
                "description": td["description"],
                "parameters": td["parameters"],
            },
        })
    if output_format and output_format.get("schema"):
        openai_tools.append({
            "type": "function",
            "function": {
                "name": "StructuredOutput",
                "description": "Submit your final answer.",
                "parameters": output_format["schema"],
            },
        })

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(user_message, image_data_url)},
    ]

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for turn in range(max_turns):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=8192,
            tools=openai_tools,
        )

        choice = response.choices[0]
        if response.usage:
            total_usage["input_tokens"] += response.usage.prompt_tokens
            total_usage["output_tokens"] += response.usage.completion_tokens

        # --- Path A: Native tool calls (OpenAI-style) ---
        if choice.message.tool_calls:
            # Write assistant reasoning text if present
            assistant_text = choice.message.content or ""
            if assistant_text and writer:
                writer.write_text(assistant_text)

            # Process all tool calls
            tool_results: list[dict] = []
            should_stop = False

            for tc in choice.message.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}

                if name == "StructuredOutput":
                    if not quiet:
                        print(f"[structured_output] {args}")
                    if writer:
                        writer.write_tool_use("StructuredOutput", args)
                    if on_structured_output:
                        on_structured_output(args)
                    return _estimate_cost_usd(total_usage, model), total_usage

                if not quiet:
                    print(f"[tool_use] {name}")
                if writer:
                    writer.write_tool_use(name, args)

                result_text, stop = await tool_handler(name, args)
                if writer:
                    writer.write_tool_result(result_text)
                if stop:
                    should_stop = True

                tool_results.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "content": result_text,
                })

            # Append ONE assistant message with ALL tool_calls
            messages.append({
                "role": "assistant",
                "content": assistant_text,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            })
            # Append one tool result per call
            messages.extend(tool_results)

            if should_stop:
                return _estimate_cost_usd(total_usage, model), total_usage
            continue

        # --- Path B: Text-based <tool_call> tags (fallback) ---
        text = choice.message.content or ""
        if writer:
            writer.write_text(text)

        calls = parse_tool_calls(text)
        if not calls:
            if not quiet:
                print("[orchestrator] no tool call in response, ending")
            break

        # Append assistant message once
        messages.append({"role": "assistant", "content": text})

        for name, args in calls:
            if name == "StructuredOutput":
                if not quiet:
                    print(f"[structured_output] {args}")
                if writer:
                    writer.write_tool_use("StructuredOutput", args)
                if on_structured_output:
                    on_structured_output(args)
                return _estimate_cost_usd(total_usage, model), total_usage

            if not quiet:
                print(f"[tool_use] {name}")
            if writer:
                writer.write_tool_use(name, args)

            result_text, should_stop = await tool_handler(name, args)
            if writer:
                writer.write_tool_result(result_text)

            # Inject tool result as user message (Qwen convention)
            messages.append({
                "role": "user",
                "content": f"<tool_response>\n{result_text}\n</tool_response>",
            })

            if should_stop:
                return _estimate_cost_usd(total_usage, model), total_usage

    return _estimate_cost_usd(total_usage, model), total_usage

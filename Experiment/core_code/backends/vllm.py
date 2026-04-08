"""vLLM backend for local model inference.

Exposes the same transport primitives as claude.py:
  - call_sub_model(...)         -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)  -> (cost_usd, usage)

Connects to a vLLM server (started separately via ``vllm serve``).
Uses vLLM native tool calling API (--enable-auto-tool-choice --tool-call-parser hermes).
"""

from __future__ import annotations

import json
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


# Cost estimation ($/1M tokens) -- configurable for reporting
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

    result = json.loads(text)

    trajectory = text
    if writer:
        writer.write_text(text)
        writer.close()

    return result, trajectory, cost, usage


# ---------------------------------------------------------------------------
# Multi-turn tool-calling conversation (native tool calling)
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
    """Run a multi-turn tool-calling conversation via vLLM native tool calling API."""
    client = _get_client()

    # Convert Claude SDK tool format to OpenAI format if needed
    openai_tools = []
    for t in tools:
        if "type" in t and "function" in t:
            openai_tools.append(t)  # already OpenAI format
        else:
            openai_tools.append({"type": "function", "function": t})

    # Add StructuredOutput tool from output_format schema
    if output_format and output_format.get("schema"):
        schema = output_format["schema"]
        openai_tools.append({
            "type": "function",
            "function": {
                "name": "StructuredOutput",
                "description": "Submit the final answer.",
                "parameters": schema,
            },
        })
        required = schema.get("required", [])
        system_prompt = (
            system_prompt
            + "\n\nWhen you are ready to submit your final answer, call the StructuredOutput tool. "
            f"All of these fields are required: {', '.join(required)}.\n"
        )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(user_message, image_data_url)},
    ]

    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for turn in range(max_turns):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=16384,
        )

        choice = response.choices[0]
        if response.usage:
            total_usage["input_tokens"] += response.usage.prompt_tokens
            total_usage["output_tokens"] += response.usage.completion_tokens

        content = choice.message.content or ""
        native_calls = choice.message.tool_calls or []

        if not quiet:
            print(f"[vllm turn {turn}] finish={choice.finish_reason} tools={len(native_calls)} content_len={len(content)}")

        if writer and content:
            writer.write_text(content)

        if not native_calls:
            if not quiet:
                print("[orchestrator] no tool call in response, ending")
            break

        # Build assistant message with tool_calls for conversation history
        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if content:
            assistant_msg["content"] = content
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in native_calls
        ]
        messages.append(assistant_msg)

        # Process each tool call
        for tc in native_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)

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

            # Native tool response format
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

            if should_stop:
                return _estimate_cost_usd(total_usage, model), total_usage

    return _estimate_cost_usd(total_usage, model), total_usage

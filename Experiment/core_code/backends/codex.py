"""Codex Responses API backend.

Exposes transport primitives only:
  - call_sub_model(...)          -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)   -> (cost_usd, usage)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from trajectory import TrajectoryWriter


# ---------------------------------------------------------------------------
# Codex API client
# ---------------------------------------------------------------------------

CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
AUTH_PATH = Path.home() / ".codex" / "auth.json"

# USD per 1M tokens (input, output)
PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.3-codex": (1.75, 14.00),
    "gpt-5.2": (1.75, 14.00),
    "gpt-5.2-codex": (1.75, 14.00),
    "gpt-5.1": (1.25, 10.00),
    "gpt-5.1-codex-max": (1.25, 10.00),
    "gpt-5.1-codex": (1.25, 10.00),
    "gpt-5": (1.25, 10.00),
    "gpt-5-codex": (1.25, 10.00),
    "gpt-5-codex-mini": (0.25, 2.00),
}


def _estimate_cost_usd(usage: dict[str, Any], model: str) -> float:
    """Estimate cost in USD from token usage and model pricing."""
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    input_price, output_price = PRICING_PER_1M.get(model, (0.0, 0.0))
    return input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000


def _load_codex_auth() -> str:
    """Read access_token from ~/.codex/auth.json."""
    data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    return data["tokens"]["access_token"]


def _build_user_content(text: str, image_data_url: str | None) -> str | list[dict[str, Any]]:
    """Build Codex user message content, with optional image."""
    if image_data_url:
        return [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": image_data_url},
        ]
    return text


async def _codex_request(
    messages: list[dict[str, Any]],
    instructions: str,
    model: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    effort: str | None = None,
) -> tuple[str | None, list[dict[str, Any]], dict[str, Any]]:
    """Send a request to the Codex Responses API (SSE streaming).

    Returns (output_text, tool_calls, usage_dict).
    """
    token = _load_codex_auth()

    body: dict[str, Any] = {
        "model": model,
        "input": messages,
        "instructions": instructions,
        "stream": True,
        "store": False,
    }
    if effort:
        body["reasoning"] = {"effort": effort}
    if tools:
        body["tools"] = tools
        body["parallel_tool_calls"] = False
    if response_format:
        body["text"] = {"format": response_format}

    import asyncio as _asyncio

    output_text: str | None = None
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}

    max_retries = 20
    for _attempt in range(max_retries):
        output_text = None
        tool_calls = []
        usage = {}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
                async with client.stream(
                    "POST",
                    CODEX_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    content=json.dumps(body),
                ) as resp:
                    if (resp.status_code >= 500 or resp.status_code == 429) and _attempt < max_retries - 1:
                        delay = min(60 * (_attempt + 1), 600) if resp.status_code == 429 else 5 * (_attempt + 1)
                        print(f"  [codex] {resp.status_code}, retrying in {delay}s (attempt {_attempt + 1}/{max_retries})")
                        await _asyncio.sleep(delay)
                        continue
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[len("data: "):]
                        if payload == "[DONE]":
                            break
                        event = json.loads(payload)
                        event_type = event.get("type", "")

                        if event_type == "response.output_text.done":
                            output_text = event.get("text")
                        elif event_type == "response.completed":
                            response_obj = event.get("response", {})
                            usage = response_obj.get("usage", {})
                            for item in response_obj.get("output", []):
                                if item.get("type") == "function_call":
                                    tool_calls.append({
                                        "call_id": item["call_id"],
                                        "name": item["name"],
                                        "arguments": item.get("arguments", "{}"),
                                    })
            break  # success
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError) as e:
            if _attempt < max_retries - 1:
                print(f"  [codex] connection error (attempt {_attempt + 1}/3), retrying: {e}")
                await _asyncio.sleep(5 * (_attempt + 1))
                continue
            raise

    return output_text, tool_calls, usage


# ---------------------------------------------------------------------------
# Sub-model call (single structured query)
# ---------------------------------------------------------------------------

async def call_sub_model(
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    output_schema: dict[str, Any],
    writer: "TrajectoryWriter",
    budget_tokens: int = 32000,
    effort: str | None = None,
) -> tuple[dict[str, Any], str, float, dict[str, Any]]:
    """Call a sub-model via Codex Responses API.

    Returns (structured_output, trajectory_text, cost_usd, usage).
    """
    messages = [{"role": "user", "content": _build_user_content(user_message, image_data_url)}]

    response_format = {
        "type": "json_schema",
        "name": "result",
        "schema": output_schema,
        "strict": True,
    }
    output_text, _, usage = await _codex_request(
        messages, instructions=system_prompt, model=model,
        response_format=response_format, effort=effort,
    )
    assert output_text is not None, "Codex returned no output text for structured request"
    result = json.loads(output_text)

    reasoning = result.get("reasoning", "")
    json_str = json.dumps(result, indent=2, ensure_ascii=False)
    trajectory_text = f"{reasoning}\n\n```json\n{json_str}\n```\n\n"
    writer.write_text(reasoning)
    writer.write_text(f"```json\n{json_str}\n```")
    writer.close()

    cost_usd = _estimate_cost_usd(usage, model)
    return result, trajectory_text, cost_usd, usage


# ---------------------------------------------------------------------------
# Generic tool-calling conversation
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
    """Run a multi-turn tool-calling conversation via Codex Responses API.

    tool_handler(name, args) -> (result_text, should_stop)
    writer: if provided, writes events to the trajectory.
    quiet: if False, prints events to console.
    output_format: if provided, constrains the model's final text output to
        match the given JSON schema (via Codex response_format).
    on_structured_output: callback when the model emits structured output.
    Returns (total_cost_usd, merged_usage).
    """
    codex_tools = [
        {
            "type": "function",
            "name": td["name"],
            "description": td["description"],
            "parameters": td["parameters"],
        }
        for td in tools
    ]

    response_format = None
    if output_format and output_format.get("schema"):
        response_format = {
            "type": "json_schema",
            "name": "structured_output",
            "schema": output_format["schema"],
            "strict": True,
        }

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_content(user_message, image_data_url)},
    ]

    total_cost = 0.0
    total_usage: dict[str, int] = {}
    structured_output_emitted = False

    for _turn in range(max_turns):
        output_text, tool_calls, usage = await _codex_request(
            messages, instructions=system_prompt, model=model,
            tools=codex_tools, response_format=response_format, effort=effort,
        )
        total_cost += _estimate_cost_usd(usage, model)
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                total_usage[k] = total_usage.get(k, 0) + v

        # Handle text output (either plain text or structured output)
        if output_text:
            if not quiet:
                print(f"[orchestrator] {output_text[:200]}")
            if writer:
                writer.write_text(output_text)

        if not tool_calls:
            # Conversation ended -- check for structured output
            if output_text and response_format:
                parsed = json.loads(output_text)
                if not quiet:
                    print(f"[structured_output] {parsed}")
                if writer:
                    writer.write_tool_use("StructuredOutput", parsed)
                if on_structured_output:
                    on_structured_output(parsed)
                structured_output_emitted = True
            break

        should_stop = False
        for tc in tool_calls:
            name = tc["name"]
            args = json.loads(tc["arguments"])

            if not quiet:
                print(f"[tool_use] {name}")
            if writer:
                writer.write_tool_use(name, args)

            result_text, stop = await tool_handler(name, args)
            if stop:
                should_stop = True

            if writer:
                writer.write_tool_result(result_text)

            messages.append({
                "type": "function_call",
                "call_id": tc["call_id"],
                "name": name,
                "arguments": tc["arguments"],
            })
            messages.append({
                "type": "function_call_output",
                "call_id": tc["call_id"],
                "output": result_text,
            })

        if should_stop:
            break

    # If structured output was expected but never emitted, force a final
    # request with no tools so the model must produce structured text.
    if response_format and on_structured_output and not structured_output_emitted:
        output_text, _, usage = await _codex_request(
            messages, instructions=system_prompt, model=model,
            response_format=response_format, effort=effort,
        )
        total_cost += _estimate_cost_usd(usage, model)
        for k, v in usage.items():
            if isinstance(v, (int, float)):
                total_usage[k] = total_usage.get(k, 0) + v
        assert output_text is not None, "Final structured output request returned no text"
        parsed = json.loads(output_text)
        if not quiet:
            print(f"[structured_output] (forced) {parsed}")
        if writer:
            writer.write_tool_use("StructuredOutput", parsed)
        on_structured_output(parsed)

    return total_cost, total_usage

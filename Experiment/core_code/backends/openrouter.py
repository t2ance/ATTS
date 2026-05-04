"""OpenRouter backend — direct OpenAI-compatible chat/completions, no claude_agent_sdk.

Replaces a previous thin-alias implementation that re-exported `backends.claude`.
The thin-alias path was retracted 2026-05-03 because every claude_agent_sdk
invocation triggered a hidden Anthropic Haiku 1-token classifier call billed at
~$0.0066 per call regardless of which model the user requested. Suppression
attempts via 11 `CLAUDE_CODE_DISABLE_*` / `DISABLE_*` env vars failed to block
the call (initial $0 deltas were measurement artifacts of OpenRouter's ledger
lag, not real suppression). See docs/superpowers/plans/2026-05-03-openrouter-real-backend.md.

This implementation is a sibling of `backends/vllm.py` — same OpenAI Chat
Completions protocol, same tool-calling pattern (server-side parser populates
`message.tool_calls`), same StructuredOutput-as-fake-function-tool injection.
Differences from vllm.py: single base_url (no per-model routing), no harmony
parser fixups, no context-overflow soft-skip, no `enable_thinking` chat-template
kwarg, no `disable_response_format` Gemma-4 workaround. Costs are read from
OpenRouter's `usage.cost` field (real per-call USD), not fabricated client-side.

Returns the same outward contract as backends/claude.py:
  - call_sub_model(...)         -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)  -> (cost_usd, usage, exit_reason)
    where exit_reason ∈ {"committed", "cap_exceeded", "incomplete"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Awaitable, Callable

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    APIError,
    APITimeoutError,
    RateLimitError,
)

# Transient/upstream API failures we treat as soft-failure (cache as timed_out,
# do not crash the gather). Free-tier OpenRouter routinely emits these:
#   RateLimitError        — 429 (per-minute global cap, or upstream provider RL)
#   APIConnectionError    — TCP reset / DNS hiccup
#   APITimeoutError       — http connect/read timeout
#   APIError (catch-all)  — 500/502/503 from OpenRouter or upstream
# These are operational events, not bugs in our code; the same gather-killing
# class as the no_tool_call / invalid_json patches above.
_TRANSIENT_API_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, APIError)

logger = logging.getLogger(__name__)


# OpenRouter provider routing.
#
# Why per-call (post-modelconfig refactor 2026-05-04): every role's ModelConfig
# carries its own openrouter_provider_order / openrouter_provider_allow_fallbacks.
# Routing is injected into extra_body.provider at the per-call level via the
# kwargs threaded through methods.base.make_sub_model_caller. Module-level
# globals + set_provider() were deleted because they couldn't represent
# different roles wanting different providers within one eval.
#
# Background incident 2026-05-04: OpenRouter `/api/v1/models` returns
# `tools=True` for `deepseek/deepseek-v4-flash`, but the model is routed to 7
# upstream providers and 2-3 of them (DeepSeek's own deepseek-reasoner endpoint,
# AtlasCloud, Novita) reject the forced `tool_choice={"type":"function",...}`
# with HTTP 400 "deepseek-reasoner does not support this tool_choice". Without
# pinning, ~33% of explore calls cache as timed_out — far over the 5% red line.
# Pinning to a tool_choice-compatible provider (e.g. Parasail) eliminates this
# routing-flake mode while leaving the model identity untouched.

assert "OPENROUTER_API_KEY" in os.environ, (
    "backends.openrouter requires OPENROUTER_API_KEY in env; export it before "
    "running eval.py / precache_explores.py with backend=openrouter"
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Single shared client; AsyncOpenAI maintains an internal connection pool.
# timeout=1800.0 (30 min) mirrors backends/vllm.py:126 — long-tail thinking-mode
# decode on hard problems can exceed the openai client's default 600s timeout.
_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=1800.0,
            # Default openai SDK max_retries=2; bumped to 8 to absorb upstream
            # free-tier 429 bursts (gpt-oss-20b:free shares per-minute provider
            # cap with all OpenRouter free-tier users; we observed 3+ 429s in
            # 30s windows). SDK uses exponential backoff with jitter, so 8
            # retries ≈ ~30-60s of wait before bubbling up.
            max_retries=8,
        )
    return _client


def _split_sampling_kwargs(s: dict | None) -> tuple[dict, dict]:
    """Split a sampling dict into (OpenAI-direct kwargs, extra_body kwargs).

    OpenAI-native (passed as kwargs to `client.chat.completions.create`):
      temperature, top_p, presence_penalty, max_tokens, frequency_penalty.
    OpenRouter / provider extensions (must go via `extra_body`):
      top_k, min_p, repetition_penalty.

    None-valued entries are dropped so the upstream library default applies.
    Mirrors the contract of `backends/vllm.py:_split_sampling_kwargs` to keep
    SamplingConfig usage uniform across backends.
    """
    if s is None:
        return {}, {}
    direct: dict = {}
    extra: dict = {}
    for k in ("temperature", "top_p", "presence_penalty", "max_tokens", "frequency_penalty"):
        if s.get(k) is not None:
            direct[k] = s[k]
    for k in ("top_k", "min_p", "repetition_penalty"):
        if s.get(k) is not None:
            extra[k] = s[k]
    return direct, extra


async def call_sub_model(
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    output_schema: dict[str, Any],
    writer,
    budget_tokens: int = 32000,
    effort: str | None = None,
    sampling: dict | None = None,
    provider_order: list[str] | None = None,
    provider_allow_fallbacks: bool = True,
) -> tuple[dict[str, Any], str, float, dict[str, Any]]:
    """Single structured query via OpenRouter chat/completions + forced StructuredOutput tool.

    Implementation note: we DO NOT use `response_format={"type":"json_schema",...}` here
    because empirically (verified 2026-05-03 against `openai/gpt-oss-120b:free`) several
    OpenRouter-routed models hit a deterministic whitespace-padding loop under grammar-
    constrained decoding — output starts with `{\\n  "answer":   \\n\\n  \\n\\n ...` and
    pads until max_tokens is reached, finish_reason=length, no JSON value emitted. This
    is the same vllm#40080 / xgrammar grammar-loop class of bug we mitigate in vllm.py.

    Workaround: register `StructuredOutput` as the ONLY function tool + `tool_choice` set
    to force a function call. The server-side parser then populates `message.tool_calls`
    with structured arguments matching `output_schema` — same mechanism we already verified
    works for run_tool_conversation. Returns the parsed args dict plus a serialized JSON
    trajectory for caching.

    `effort`: passed via `extra_body={"reasoning": {"effort": effort}}`.
    `sampling`: vllm-style — full block accepted (temperature / top_p / etc).
    """
    if image_data_url is not None:
        raise NotImplementedError(
            "backends.openrouter does not currently support image_data_url. "
            "Scope is text-only HLE / LCB / GPQA. Implement multimodal content "
            "blocks before enabling for BabyVision / RBenchV."
        )

    direct_kwargs, extra_body = _split_sampling_kwargs(sampling)
    direct_kwargs.setdefault("temperature", 0.0)
    direct_kwargs.setdefault("max_tokens", budget_tokens)

    if effort:
        extra_body["reasoning"] = {"effort": effort}
    extra_body["usage"] = {"include": True}
    if provider_order:
        extra_body["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": provider_allow_fallbacks,
        }

    # Force the model to call StructuredOutput (function tool) so the response
    # arrives as parsed structured args, not free-form JSON in content.
    tools = [{
        "type": "function",
        "function": {
            "name": "StructuredOutput",
            "description": "Submit the final structured answer.",
            "parameters": output_schema,
        },
    }]
    tool_choice = {"type": "function", "function": {"name": "StructuredOutput"}}

    try:
        response = await _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
            **direct_kwargs,
        )
    except (_TRANSIENT_API_ERRORS + (json.JSONDecodeError,)) as e:
        # SDK already exhausted max_retries=8 with exponential backoff. Treat as
        # operational soft-failure equivalent to wall-clock timeout — same
        # timed_out sentinel contract as no_tool_call / invalid_json above.
        # Failure modes covered:
        #   RateLimitError        — 429 upstream rate-limit
        #   APIConnectionError    — TCP reset / DNS hiccup
        #   APITimeoutError       — http connect/read timeout
        #   APIError              — 500/502/503 from provider
        #   json.JSONDecodeError  — OpenRouter sometimes returns non-JSON body
        #                            (HTML error page, truncated stream); openai SDK
        #                            does NOT classify this as retriable, so it
        #                            bubbles bare from httpx._models.json() call.
        # Crashing the gather over any of these wastes hundreds of cached explores.
        logger.warning(
            f"[openrouter call_sub_model] model {model!r} hit transient API error "
            f"after retries: {type(e).__name__}: {str(e)[:200]}; caching as "
            f"timed_out and continuing"
        )
        return {
            "timed_out": True,
            "reason": "transient_api_error",
            "error_type": type(e).__name__,
            "error_message": str(e)[:500],
        }, "", 0.0, {"input_tokens": 0, "output_tokens": 0}

    # Free-tier providers occasionally return HTTP 200 with `choices=None` or
    # `choices=[]` (no JSON-decode error, no transient API error — payload is
    # syntactically valid but semantically empty). Guard before [0] indexing
    # so it's the same timed_out soft-failure contract as no_tool_call /
    # invalid_json_args / transient_api_error above. Concrete crash 2026-05-03
    # 23:23:11 at PID 2270633 [308/696]: TypeError 'NoneType' is not subscriptable
    # at this exact line — killed entire 696-task gather, lost cache discipline.
    if not response.choices:
        logger.warning(
            f"[openrouter call_sub_model] model {model!r} returned empty choices "
            f"(choices={response.choices!r}); caching as timed_out and continuing"
        )
        usage_dict = {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }
        cost = float(getattr(response.usage, "cost", 0.0) or 0.0) if response.usage else 0.0
        return {
            "timed_out": True,
            "reason": "empty_choices",
            "error_type": "EmptyChoices",
        }, "", cost, usage_dict

    msg = response.choices[0].message
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    # OpenRouter native cost (real USD), not SDK-fabricated.
    cost = float(getattr(response.usage, "cost", 0.0) or 0.0)

    raw_tool_calls = msg.tool_calls or []
    if not raw_tool_calls:
        # Free-tier models (gpt-oss-20b/120b, gemma-4-26b) occasionally ignore
        # forced tool_choice and return reasoning-only with no tool_call. Treat
        # this as a soft failure equivalent to a wall-clock timeout: cache as
        # timed_out so resume skips it, log the diagnostics, and let the rest
        # of the batch continue. base.py:303 uses the same `{"timed_out": True}`
        # sentinel for asyncio.TimeoutError; worker line 132-135 logs and moves
        # on. Crashing the entire 800-task gather on one bad call violates
        # cache discipline (results.jsonl resume).
        logger.warning(
            f"[openrouter call_sub_model] model {model!r} returned no tool_call "
            f"(finish_reason={response.choices[0].finish_reason!r}, "
            f"content_len={len(msg.content or '')}, "
            f"reasoning_len={len(getattr(msg, 'reasoning', '') or '')}); "
            f"caching as timed_out and continuing"
        )
        result = {
            "timed_out": True,
            "reason": "no_tool_call",
            "finish_reason": response.choices[0].finish_reason,
        }
        return result, "", cost, usage
    args_str = raw_tool_calls[0].function.arguments or "{}"
    try:
        result = json.loads(args_str)
    except json.JSONDecodeError as e:
        # Free-tier models occasionally emit invalid JSON in tool_call.arguments
        # (bad escape sequences, truncated strings). Same soft-failure contract as
        # no_tool_call: cache as timed_out + continue, do NOT kill the gather.
        logger.warning(
            f"[openrouter call_sub_model] model {model!r} emitted invalid JSON in "
            f"tool_call.arguments: {e}; args_len={len(args_str)} "
            f"args_head={args_str[:120]!r}; caching as timed_out and continuing"
        )
        return {
            "timed_out": True,
            "reason": "invalid_json_in_tool_args",
            "json_error": str(e),
        }, "", cost, usage

    # Trajectory text: include any reasoning the model emitted, then the JSON args
    # (mirrors backends/claude.py's call_sub_model trajectory_text shape — reasoning
    # block followed by ```json ... ``` snapshot of the structured output).
    reasoning_text = getattr(msg, "reasoning", "") or ""
    json_block = json.dumps(result, indent=2, ensure_ascii=False)
    trajectory = (f"{reasoning_text}\n\n" if reasoning_text else "") + f"```json\n{json_block}\n```\n\n"

    if writer:
        if reasoning_text:
            writer.write_text(reasoning_text)
        writer.write_text(f"```json\n{json_block}\n```")
        writer.close()

    return result, trajectory, cost, usage


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
    on_structured_output: Callable[[dict], None] | None = None,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    sampling: dict | None = None,
    provider_order: list[str] | None = None,
    provider_allow_fallbacks: bool = True,
) -> tuple[float, dict[str, Any], str]:
    """Multi-turn tool-calling via OpenRouter chat/completions + tools=[...] + tool_choice='auto'.

    Mirrors backends/vllm.py:run_tool_conversation. Server-side parser populates
    `message.tool_calls=[...]` (structured) and leaves `message.content=None` on
    tool turns. StructuredOutput is injected as a fake function tool — when the
    model "calls" it, we treat that as the commit signal and exit with
    exit_reason='committed'.

    Writer events match backends/claude.py's NON-streaming form:
    write_text per visible content / per reasoning block, write_tool_use per
    tool call, write_tool_result per tool response.
    """
    if image_data_url is not None:
        raise NotImplementedError(
            "backends.openrouter does not currently support image_data_url. "
            "Text-only scope: HLE / LCB / GPQA."
        )

    direct_kwargs, extra_body = _split_sampling_kwargs(sampling)
    if "temperature" not in direct_kwargs and temperature is not None:
        direct_kwargs["temperature"] = temperature
    direct_kwargs.setdefault("temperature", 0.0)
    direct_kwargs.setdefault("max_tokens", 8192)

    if effort:
        extra_body["reasoning"] = {"effort": effort}
    extra_body["usage"] = {"include": True}
    if provider_order:
        extra_body["provider"] = {
            "order": list(provider_order),
            "allow_fallbacks": provider_allow_fallbacks,
        }

    # Append StructuredOutput protocol block to system prompt — matches the
    # defense-in-depth pattern from backends/vllm.py:485-513. Even though the
    # tool is registered server-side, weaker models occasionally write the
    # answer in free-form text without calling StructuredOutput.
    required: list[str] = []
    if output_format and output_format.get("schema"):
        required = output_format["schema"].get("required", [])
        schema_json = json.dumps(output_format["schema"], indent=2)
        system_prompt = (
            system_prompt
            + "\n\n=== EXTREMELY IMPORTANT --- FINAL ANSWER SUBMISSION PROTOCOL ===\n"
            "WHEN YOUR REASONING IS COMPLETE, YOU MUST SUBMIT YOUR FINAL ANSWER BY CALLING THE `StructuredOutput` TOOL.\n"
            "WRITING THE ANSWER IN FREE-FORM TEXT (e.g. `\\boxed{X}`, `Answer: X`, `The answer is X`) IS NOT A VALID SUBMISSION; IT WILL BE GRADED AS INCORRECT.\n"
            "THE `StructuredOutput` TOOL CALL IS THE ONLY ACCEPTED SUBMISSION PATH.\n"
            "Use strict JSON format for all parameters. "
            f"All of these fields are required and must each be a separate JSON key: {', '.join(required)}.\n"
            f"Schema:\n```json\n{schema_json}\n```\n"
        )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    openai_tools: list[dict[str, Any]] = [
        {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in tools
    ]
    if output_format and output_format.get("schema"):
        openai_tools.append({
            "type": "function",
            "function": {
                "name": "StructuredOutput",
                "description": "Submit the final structured answer.",
                "parameters": output_format["schema"],
            },
        })

    total_usage = {"input_tokens": 0, "output_tokens": 0}
    total_cost = 0.0

    # Retry policy for OpenRouter HTTP 200 + choices=None edge case (free-tier
    # provider sometimes returns syntactically-valid but semantically-empty
    # payload — SDK does NOT auto-retry these because the HTTP status looks
    # successful). Retry up to MAX_EMPTY_CHOICES_RETRIES with linear backoff;
    # if all retries exhausted, RAISE rather than return a degraded result —
    # per user 2026-05-04 directive: "不要 silently 就给用户跳过了". A sustained
    # empty-choices stream indicates a structural provider failure, not a
    # transient hiccup, and the user wants visibility, not a fake empty
    # `predicted=""` row written to results.jsonl downstream.
    MAX_EMPTY_CHOICES_RETRIES = 5
    EMPTY_CHOICES_BACKOFF_SECONDS = 0.5  # linear: 0.5, 1.0, 1.5, 2.0, 2.5

    for turn in range(max_turns):
        response = None
        try:
            for empty_retry in range(MAX_EMPTY_CHOICES_RETRIES):
                response = await _get_client().chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools or None,
                    tool_choice="auto" if openai_tools else None,
                    extra_body=extra_body or None,
                    **direct_kwargs,
                )
                if response.choices:
                    break  # got a real response, proceed to parsing
                logger.warning(
                    f"[openrouter run_tool_conversation turn {turn}] OpenRouter "
                    f"HTTP 200 with choices=None on model {model!r} "
                    f"(empty_retry {empty_retry + 1}/{MAX_EMPTY_CHOICES_RETRIES}); "
                    f"sleeping {EMPTY_CHOICES_BACKOFF_SECONDS * (empty_retry + 1):.1f}s"
                )
                await asyncio.sleep(EMPTY_CHOICES_BACKOFF_SECONDS * (empty_retry + 1))
            else:
                # All retries exhausted — sustained provider failure. RAISE rather
                # than silently soft-fail. Halts the gather so the operator sees
                # the issue at run time. Do NOT write a partial result row.
                raise RuntimeError(
                    f"[openrouter run_tool_conversation] OpenRouter returned "
                    f"HTTP 200 with choices=None for {MAX_EMPTY_CHOICES_RETRIES} "
                    f"consecutive retries on model {model!r} at turn {turn}. "
                    f"This is a sustained provider failure, not a transient "
                    f"hiccup. Halting eval to surface the issue rather than "
                    f"producing a fake empty `predicted` row downstream."
                )
        except (_TRANSIENT_API_ERRORS + (json.JSONDecodeError,)) as e:
            # Same soft-failure contract as call_sub_model — exit the orchestrator
            # cleanly with exit_reason="incomplete" so the caller treats it like
            # any other early termination (caller already handles "incomplete";
            # see backends/claude.py contract). Crashing the gather mid-eval
            # would lose all in-flight orchestrator work.
            logger.warning(
                f"[openrouter run_tool_conversation turn {turn}] transient API error "
                f"after retries: {type(e).__name__}: {str(e)[:200]}; "
                f"exiting with incomplete"
            )
            return total_cost, total_usage, "incomplete"

        # response.choices is guaranteed non-empty here (loop broke on success
        # or RAISE'd on exhaustion).
        choice = response.choices[0]
        if response.usage:
            total_usage["input_tokens"] += response.usage.prompt_tokens
            total_usage["output_tokens"] += response.usage.completion_tokens
            total_cost += float(getattr(response.usage, "cost", 0.0) or 0.0)

        if max_output_tokens is not None and total_usage["output_tokens"] > max_output_tokens:
            logger.info(
                f"[orchestrator] output token cap exceeded "
                f"({total_usage['output_tokens']} > {max_output_tokens}), terminating"
            )
            return total_cost, total_usage, "cap_exceeded"

        text_content = choice.message.content or ""
        raw_tool_calls = choice.message.tool_calls or []

        if writer and text_content:
            writer.write_text(text_content)

        if not raw_tool_calls:
            logger.info("[orchestrator] no tool call in response, ending")
            return total_cost, total_usage, "incomplete"

        # Decode each tool_call.function.arguments. Skip those that fail to decode.
        text_calls: list[tuple[Any, str, dict]] = []
        cleaned_tool_calls: list[dict] = []
        for tc in raw_tool_calls:
            try:
                args_dict = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[openrouter turn {turn}] tool_call arguments JSON decode failed: "
                    f"name={tc.function.name!r} args_raw={tc.function.arguments!r} err={e}"
                )
                continue
            text_calls.append((tc, tc.function.name, args_dict))
            cleaned_tool_calls.append({
                "id": tc.id, "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            })

        logger.info(
            f"[openrouter turn {turn}] finish={choice.finish_reason} "
            f"tools={len(text_calls)} content_len={len(text_content)}"
        )

        # Append assistant message with structured tool_calls (OpenAI standard).
        messages.append({"role": "assistant", "content": None, "tool_calls": cleaned_tool_calls})

        for tc, name, args in text_calls:
            if name == "StructuredOutput":
                missing = [k for k in required if k not in args]
                if missing:
                    logger.info(f"[structured_output_invalid] missing {missing}; got keys={list(args.keys())}")
                    if writer:
                        writer.write_text(
                            f"[StructuredOutput rejected: missing required fields {missing}; "
                            f"got keys={list(args.keys())}]"
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            f"StructuredOutput rejected: missing required fields {missing}. "
                            "Call StructuredOutput again with ALL required fields filled per the schema."
                        ),
                    })
                    break
                logger.info(f"[structured_output] {args}")
                if writer:
                    writer.write_tool_use("StructuredOutput", args)
                if on_structured_output:
                    on_structured_output(args)
                return total_cost, total_usage, "committed"

            logger.info(f"[tool_use] {name}")
            if writer:
                writer.write_tool_use(name, args)

            result_text, should_stop = await tool_handler(name, args)
            if writer:
                writer.write_tool_result(result_text)

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

            if should_stop:
                return total_cost, total_usage, "committed"

    return total_cost, total_usage, "incomplete"

"""vLLM backend for local model inference.

Exposes the same transport primitives as claude.py:
  - call_sub_model(...)         -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)  -> (cost_usd, usage, exit_reason)
    where exit_reason ∈ {"committed", "cap_exceeded", "incomplete"}

Connects to a vLLM server (started separately via ``vllm serve``).
Uses text-based <tool_call> parsing (no --enable-auto-tool-choice needed).
"""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Multi-replica vLLM serve (2026-05-01): 3 independent TP=1 servers on
# GPU 0/1/2 ports 8000/8001/8002. TP=3/DP=3 are blocked because
# intermediate_size=8192 is not divisible by 3 (verified by pydantic
# ValidationError when launching TP=3). Each call to _get_client() returns
# the next client in a round-robin cycle so concurrent ATTS workers
# (num_workers=32) distribute uniformly across all 3 replicas. The cycle is
# safe under asyncio because the event loop is single-threaded -- next() on
# itertools.cycle is atomic w.r.t. coroutine scheduling. If only one replica
# is up, set VLLM_BASE_URLS to a single-entry list.
VLLM_BASE_URLS = [
    "http://localhost:8000/v1",
    "http://localhost:8001/v1",
    "http://localhost:8002/v1",
]
_clients: list[AsyncOpenAI] | None = None
_client_cycle = None


def _get_client() -> AsyncOpenAI:
    global _clients, _client_cycle
    if _clients is None:
        _clients = [AsyncOpenAI(base_url=u, api_key="not-needed") for u in VLLM_BASE_URLS]
        _client_cycle = itertools.cycle(_clients)
    return next(_client_cycle)


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


def _split_sampling_kwargs(s: dict | None) -> tuple[dict, dict]:
    """Split a sampling dict into (OpenAI-direct kwargs, vLLM extra_body).

    OpenAI-native (passed as kwargs to client.chat.completions.create):
      temperature, top_p, presence_penalty, max_tokens.
    vLLM extensions (must go via extra_body):
      top_k, min_p, repetition_penalty, chat_template_kwargs.enable_thinking.

    None-valued entries are dropped so the upstream library default applies.
    """
    if s is None:
        return {}, {}
    direct: dict = {}
    extra: dict = {}
    for k in ("temperature", "top_p", "presence_penalty", "max_tokens"):
        if s.get(k) is not None:
            direct[k] = s[k]
    for k in ("top_k", "min_p", "repetition_penalty"):
        if s.get(k) is not None:
            extra[k] = s[k]
    if s.get("enable_thinking") is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": s["enable_thinking"]}
    return direct, extra


# ---------------------------------------------------------------------------
# <tool_call> parsing (text-mode tool calling)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# History reconstruction policy (additive extraction, 2026-05-01):
# After an assistant turn, only the <tool_call>...</tool_call> blocks are
# kept in the multi-turn history; all surrounding text (reasoning / chatter /
# free-form CoT) is dropped before the next request. Rationale: the prior
# subtractive design (strip text before </think>) silently no-oped on outputs
# without a </think> boundary -- LCB orchestrator emitted free-form CoT with
# zero </think> tags, so prior reasoning leaked back into messages and input
# crossed max_model_len=65536 at run_20260501_022727 q90 (input=32769 vs
# cap=32768). The fix below defines what to KEEP, not what to REMOVE; missing
# boundaries can no longer cause leaks. Trajectory file keeps the full
# unstripped text for offline analysis.
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)

# Qwen3.6+ chat_template emits XML body inside <tool_call>:
#   <function=NAME>
#     <parameter=KEY>VAL</parameter>
#     ...
#   </function>
# (chat_template.jinja line 53; Qwen3-8B uses JSON body instead.)
_TOOL_FUNCTION_RE = re.compile(
    r"<function=([^>\s]+)\s*>(.*?)</function>",
    re.DOTALL,
)
_TOOL_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter>",
    re.DOTALL,
)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Parse all ``<tool_call>`` blocks from generated text.

    Returns list of (tool_name, arguments_dict).
    Tries JSON body first (Qwen3-8B chat_template format), then XML body
    (Qwen3.6+ chat_template format with ``<function=...><parameter=...>``).
    Handles double-escaped JSON from vLLM responses.
    Skips blocks with malformed bodies in both formats.
    """
    results = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        # Path A: JSON body (Qwen3-8B style)
        call = _try_parse_tool_json(raw)
        if call is None:
            # vLLM sometimes double-escapes: \\" -> ", \\u -> \u
            unescaped = raw.replace('\\\\"', '\x00QUOTE\x00').replace('\\"', '"').replace('\x00QUOTE\x00', '\\"')
            call = _try_parse_tool_json(unescaped)
        # Path B: XML body (Qwen3.6+ style)
        if call is None:
            call = _try_parse_tool_xml(raw)
        if call is None:
            continue
        name = call.get("name", "")
        args = call.get("arguments", {})
        results.append((name, args))
    return results


def _try_parse_tool_xml(raw: str) -> dict | None:
    """Try to parse a Qwen3.6-style XML tool-call body.

    Body shape: ``<function=NAME>(<parameter=KEY>VAL</parameter>)*</function>``.
    Each parameter value is JSON-decoded if possible (so ``0.85`` → float, ``"x"``
    → ``x``); otherwise kept as the trimmed raw string. This matches what the
    chat_template.jinja serializer does in reverse (jinja: ``v|tojson if v is not
    string``), so a round-trip through JSON values + string passthrough recovers
    the original Python types.
    """
    fn = _TOOL_FUNCTION_RE.search(raw)
    if fn is None:
        return None
    name = fn.group(1).strip()
    body = fn.group(2)
    args: dict = {}
    for pm in _TOOL_PARAMETER_RE.finditer(body):
        key = pm.group(1).strip()
        val_raw = pm.group(2).strip()
        try:
            args[key] = json.loads(val_raw)
        except (json.JSONDecodeError, ValueError):
            args[key] = val_raw
    return {"name": name, "arguments": args}


_INVALID_UNICODE_RE = re.compile(r"\\u([0-9a-fA-F]{0,3}[^0-9a-fA-F])")


_BAD_ESCAPE_RE = re.compile(r"\\(?![\"\\\/bfnrt]|u[0-9a-fA-F]{4})")


def _fix_json_string(raw: str) -> str:
    """Fix common JSON string issues from LLM output.

    - Invalid \\uXXXX escapes (e.g. \\u208i)
    - Unescaped control characters
    """
    # Remove invalid \uXXXX (non-hex chars after \u)
    fixed = _INVALID_UNICODE_RE.sub(lambda m: "u" + m.group(1), raw)
    # Escape any remaining invalid backslash sequences
    fixed = _BAD_ESCAPE_RE.sub(lambda m: "\\\\" + m.group(0)[1:], fixed)
    return fixed


def _try_parse_tool_json(raw: str) -> dict | None:
    """Try to parse a tool call JSON string, with progressive fixing."""
    for attempt_raw in [raw, _fix_json_string(raw)]:
        try:
            return json.loads(attempt_raw)
        except json.JSONDecodeError:
            continue
    # Last resort: extract name and try to build a minimal result
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', raw)
    args_match = re.search(r'"arguments"\s*:\s*\{(.*)\}\s*\}', raw, re.DOTALL)
    if name_match and args_match:
        # Try parsing just the arguments with fixes
        args_raw = "{" + args_match.group(1) + "}"
        args_fixed = _fix_json_string(args_raw)
        try:
            args = json.loads(args_fixed)
            return {"name": name_match.group(1), "arguments": args}
        except json.JSONDecodeError:
            # Extract individual fields with regex
            answer = re.search(r'"answer"\s*:\s*"([^"]*)"', raw)
            if answer and name_match.group(1) == "StructuredOutput":
                return {
                    "name": "StructuredOutput",
                    "arguments": {
                        "answer": answer.group(1),
                        "reasoning": "parsed from malformed JSON",
                        "confidence": 0.5,
                    },
                }
    return None


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
    sampling: dict | None = None,
) -> tuple[dict[str, Any], str, float, dict[str, Any]]:
    """Single structured query. Returns (result_dict, trajectory, cost, usage).

    Sampling-knob precedence (mirrors run_tool_conversation):
      1. `sampling` dict -- full SamplingConfig pulled from yaml. When present,
         this is the authoritative source for temperature / top_p / top_k /
         min_p / presence_penalty / repetition_penalty / enable_thinking /
         max_tokens. The explorer (precache_explores.py) and the orchestrator
         (eval.py) MUST share the same sampling block so cache distribution
         matches what the orchestrator would produce on a cache miss; this is
         enforced at the yaml level by reusing the same `sampling:` snippet.
      2. defaults (sampling=None): temperature=0.0 (greedy), max_tokens=
         budget_tokens, no top_p/top_k/min_p/presence/repetition tuning,
         enable_thinking falls back to the chat_template default (Qwen3.6:
         True; Qwen3-8B has no thinking branch so the kwarg no-ops).

    The legacy "explorer must run thinking-off to keep strict-JSON closing
    inside max_tokens" assumption was retired 2026-05-01: the right knob to
    grow when thinking-mode long tails truncate is `max_tokens`, not the
    `enable_thinking` flag. Forcing thinking-off here would silently diverge
    explorer-cache distribution from on-the-fly orchestrator-generated explores
    whenever a cache miss occurs, breaking the like-for-like guarantee that
    the cache-only assertion in make_sub_model_caller exists to enforce.
    """
    client = _get_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(user_message, image_data_url)},
    ]

    direct_kwargs, extra_body = _split_sampling_kwargs(sampling)
    direct_kwargs.setdefault("temperature", 0.0)
    direct_kwargs.setdefault("max_tokens", budget_tokens)
    assert 0.0 <= direct_kwargs["temperature"] <= 2.0, (
        f"temperature out of range: {direct_kwargs['temperature']}"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=(
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": output_schema,
                    "strict": True,
                },
            }
            if output_schema
            else None
        ),
        extra_body=extra_body or None,
        **direct_kwargs,
    )

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    cost = _estimate_cost_usd(usage, model)

    # Parse structured output -- try direct JSON, then <tool_call>, then regex
    result: dict = {}
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        calls = parse_tool_calls(text)
        for name, args in calls:
            if name == "StructuredOutput":
                result = args
                break

    # Soft-fail on unparseable output. Prior to 2026-05-01 this branch raised
    # AssertionError, which propagated through `await asyncio.gather(...)` in
    # precache_explores.py:precache and killed the entire 800/1584/1400/3104
    # task batch on a single bad row (one HLE question's truncated thinking-
    # mode JSON wiped 138/800 cached + halted the run at 17% throughput).
    # Returning `{"timed_out": True, "parse_failed": True}` lets the precache
    # worker's existing `if result.get("timed_out")` branch (line 120-122 in
    # precache_explores.py) record the failure as a timeout-equivalent, mark
    # the row as cached-but-skipped, and continue. The eval-time orchestrator
    # treats a `timed_out` cache entry as a missing candidate (load_cached_
    # candidates in methods/base.py:131 filters them out), so the question
    # downgrades from 8 explores to 7 (or fewer) without crashing the eval.
    # The full `text` is returned as the trajectory so post-mortem inspection
    # of why parsing failed (truncation vs malformed escape vs missing keys)
    # remains possible from the cached result.json.
    if not result:
        finish = response.choices[0].finish_reason if response.choices else "?"
        print(
            f"[vllm.call_sub_model] PARSE FAILED finish={finish} "
            f"completion_tokens={usage['output_tokens']} model={model} "
            f"text[:200]={text[:200]!r}"
        )
        return (
            {"timed_out": True, "parse_failed": True, "finish_reason": finish},
            text,
            cost,
            usage,
        )

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
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    sampling: dict | None = None,
    **kwargs,
) -> tuple[float, dict[str, Any], bool]:
    """Run a multi-turn tool-calling conversation via vLLM.

    Uses text-based <tool_call> tag parsing (no hermes parser needed).

    Sampling-knob precedence (per call to client.chat.completions.create):
      1. `sampling` dict (full Pydantic SamplingConfig from yaml).
      2. legacy `temperature` keyword (kept for the rejection-sampling row path
         in eval.py:923 which sets per-row `_temperature`).
      3. defaults: temperature=0.0 (greedy), max_tokens=8192 (legacy hardcode),
         no top_p/top_k/min_p/presence/repetition tuning, chat_template
         decides enable_thinking (Qwen3.6 default = thinking ON; the old
         hardcode at line 374 silently relied on this template default).
    `max_output_tokens` is the cumulative output-token CAP across turns
    (different concept from per-turn `max_tokens`).
    """
    direct_kwargs, extra_body = _split_sampling_kwargs(sampling)
    # Per-row temperature override (rejection sampling path); only honored when
    # `sampling` did not already pin temperature.
    if "temperature" not in direct_kwargs and temperature is not None:
        direct_kwargs["temperature"] = temperature
    direct_kwargs.setdefault("temperature", 0.0)
    direct_kwargs.setdefault("max_tokens", 8192)
    assert 0.0 <= direct_kwargs["temperature"] <= 2.0, (
        f"temperature out of range: {direct_kwargs['temperature']}"
    )
    client = _get_client()

    # Append StructuredOutput schema to system prompt; also expose the
    # benchmark's `required` keys to the StructuredOutput-arg validator below
    # so it does not hardcode a single field name. LCB schema requires
    # ["approach", "reasoning", "code", "confidence"] (no "answer"); HLE/GPQA
    # schemas include "answer". Hardcoding "answer" silently rejected every
    # well-formed LCB submission (run_20260501_022727: 142/146 trajectories
    # carried "StructuredOutput rejected" before this fix).
    required: list[str] = []
    if output_format and output_format.get("schema"):
        schema_json = json.dumps(output_format["schema"], indent=2)
        required = output_format["schema"].get("required", [])
        # 2026-05-01: strengthened SO instruction with all-caps + EXTREMELY
        # IMPORTANT framing. Empirical motivation: in run_20260501_064415,
        # 142/388 BabyVision questions (36.6%) had the orchestrator complete
        # its reasoning, write `\boxed{X}` in free-form text, and stop without
        # calling StructuredOutput -> predicted_answer empty -> graded wrong.
        # This is BabyVision-specific (HLE/GPQA/LCB had 0 SO-skip) and was
        # triggered by the benchmark question's trailing `\boxed{Answer}`
        # instruction conflicting with the system-level SO requirement. Even
        # after removing the `\boxed{}` instruction from babyvision.py, the
        # cache contains Sonnet outputs that use `\boxed{}` format, which the
        # orchestrator may pattern-match. The EXTREMELY IMPORTANT block below
        # is defense-in-depth at the system-prompt layer.
        system_prompt = (
            system_prompt
            + "\n\n=== EXTREMELY IMPORTANT --- FINAL ANSWER SUBMISSION PROTOCOL ===\n"
            "WHEN YOUR REASONING IS COMPLETE, YOU MUST SUBMIT YOUR FINAL ANSWER BY CALLING THE `StructuredOutput` TOOL.\n"
            "WRITING THE ANSWER IN FREE-FORM TEXT (e.g. `\\boxed{X}`, `Answer: X`, `The answer is X`) IS NOT A VALID SUBMISSION; IT WILL BE GRADED AS INCORRECT.\n"
            "THE `StructuredOutput` TOOL CALL IS THE ONLY ACCEPTED SUBMISSION PATH. NO TEXT-ONLY FINAL ANSWER WILL EVER BE READ.\n"
            "Use strict JSON format for all parameters. "
            f"All of these fields are required and must each be a separate JSON key: {', '.join(required)}.\n"
            f"Schema:\n```json\n{schema_json}\n```\n"
        )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(user_message, image_data_url)},
    ]

    # Build OpenAI-compatible tools list so Qwen3 chat template's {%- if tools %}
    # branch triggers and injects the <tools> XML block + <tool_call> format
    # instructions. We pass tool_choice="none" so vllm does NOT require
    # --enable-auto-tool-choice (verified against vllm 0.17.1 source:
    # protocol.py::check_tool_usage + docs/features/tool_calling.md). Output stays
    # in message.content as raw text and is parsed by the existing <tool_call> regex.
    openai_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
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

    for turn in range(max_turns):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="none",
            extra_body=extra_body or None,
            **direct_kwargs,
        )

        choice = response.choices[0]
        if response.usage:
            total_usage["input_tokens"] += response.usage.prompt_tokens
            total_usage["output_tokens"] += response.usage.completion_tokens

        # Cap check: cumulative output_tokens exceeded user-set ceiling. vLLM's
        # natural unit is tokens (response.usage.completion_tokens), so we use
        # tokens directly instead of mirroring claude.py's char-based check.
        if max_output_tokens is not None and total_usage["output_tokens"] > max_output_tokens:
            if not quiet:
                print(
                    f"[orchestrator] output token cap exceeded "
                    f"({total_usage['output_tokens']} > {max_output_tokens}), terminating"
                )
            return _estimate_cost_usd(total_usage, model), total_usage, "cap_exceeded"

        # --- Parse <tool_call> from text content ---
        text_content = choice.message.content or ""
        text_calls = parse_tool_calls(text_content) if text_content else []

        if not quiet:
            print(f"[vllm turn {turn}] finish={choice.finish_reason} tools={len(text_calls)} content_len={len(text_content)}")

        if writer and text_content:
            writer.write_text(text_content)

        if not text_calls:
            if not quiet:
                print("[orchestrator] no tool call in response, ending")
            # Orchestrator emitted text without tool_call; no commit signal.
            return _estimate_cost_usd(total_usage, model), total_usage, "incomplete"

        # Append assistant message once. Reconstruct history additively:
        # extract just the <tool_call>...</tool_call> blocks and drop all
        # surrounding text. See _TOOL_CALL_BLOCK_RE comment for rationale.
        # Reaching this line guarantees text_calls is non-empty (the early
        # return at "no tool call in response" handles the empty case), so
        # at least one block is always retained.
        history_content = "\n".join(_TOOL_CALL_BLOCK_RE.findall(text_content))
        messages.append({"role": "assistant", "content": history_content})

        for name, args in text_calls:
            if name == "StructuredOutput":
                # Schema-aware required-field check: derived from
                # output_format.schema.required (set above). Each benchmark
                # passes its own schema -- LCB requires
                # ["approach","reasoning","code","confidence"], HLE/GPQA
                # require "answer". Pre-fix this validator hardcoded "answer"
                # and so always rejected LCB even when the model produced the
                # right `code` field; see run_20260501_022727 incident note.
                # If `required` is empty (no schema declared), skip validation.
                missing = [k for k in required if k not in args]
                if missing:
                    if not quiet:
                        print(
                            f"[structured_output_invalid] missing {missing}; "
                            f"got keys={list(args.keys())}"
                        )
                    if writer:
                        writer.write_text(
                            f"[StructuredOutput rejected: missing required "
                            f"fields {missing}; got keys={list(args.keys())}]"
                        )
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your last StructuredOutput call was missing required "
                            f"fields: {missing}. Call StructuredOutput again with "
                            f"ALL required fields filled per the schema."
                        ),
                    })
                    break
                if not quiet:
                    print(f"[structured_output] {args}")
                if writer:
                    writer.write_tool_use("StructuredOutput", args)
                if on_structured_output:
                    on_structured_output(args)
                return _estimate_cost_usd(total_usage, model), total_usage, "committed"

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
                # Tool signaled end (e.g. tts_agent budget exhausted after final
                # explore). Treated as committed — orchestrator's flow ended cleanly.
                return _estimate_cost_usd(total_usage, model), total_usage, "committed"

    # for-loop walked all max_turns without commitment.
    return _estimate_cost_usd(total_usage, model), total_usage, "incomplete"

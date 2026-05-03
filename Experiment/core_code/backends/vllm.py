"""vLLM backend for local model inference.

Exposes the same transport primitives as claude.py:
  - call_sub_model(...)         -> (result, trajectory_text, cost_usd, usage)
  - run_tool_conversation(...)  -> (cost_usd, usage, exit_reason)
    where exit_reason ∈ {"committed", "cap_exceeded", "incomplete", "context_overflow"}

Connects to a vLLM server (started separately via ``vllm serve``).
Uses text-based <tool_call> parsing (no --enable-auto-tool-choice needed).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI, BadRequestError

logger = logging.getLogger(__name__)

# vLLM 0.17.x emits VLLMValidationError with parameter="input_tokens" (token-
# count check, vllm/renderers/params.py:344) or "input_text" (char-count pre-
# check, line 263) ONLY when input + max_tokens exceeds max-model-len. The
# HTTP layer translates this to a 400 BadRequest with `param=<name>` in the
# response body. We use this as the structured signature to soft-skip a
# single explore in the same way wall-clock timeouts are soft-skipped.
# Verified by reading vllm 0.17.1 source 2026-05-02.
_CONTEXT_OVERFLOW_PARAMS = ("input_tokens", "input_text")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Per-model serve URL routing (2026-05-03):
# Multiple vLLM serves can run concurrently on different ports — currently
#   - port 8000: gemma4-26b-a4b-it (DP=2 on GPU 0+1)
#   - port 8001: gptoss-20b        (DP=2 on GPU 2+3)
# Earlier single-port DP=4 setup is now retired; each model gets its own
# port to avoid collisions when running multiple model evals in parallel.
# DEFAULT_VLLM_BASE_URL is the fallback when the model alias is not in the
# routing table (preserves backward compat for callers that pass an unknown
# model name; they hit port 8000).
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_TO_BASE_URL: dict[str, str] = {
    "gemma4-26b-a4b-it": "http://localhost:8000/v1",
    "gptoss-20b":        "http://localhost:8001/v1",
    # Add new model aliases here as new serves come online. Default is 8000.
}
# Pre-DP=4 multi-replica scaffolding kept conceptually via the model-routing
# dict — restore independent-replica round-robin only if a single model is
# served by N independent processes on N ports (rare; co-located DP handles
# load-balancing internally).
_clients: dict[str, AsyncOpenAI] = {}


# Regex stripping the harmony channel-token leak that vllm#32587 produces in
# gpt-oss tool_call.function.name. Observed leak shapes (2026-05-03 field log):
#   trailing only:
#     "explore<|channel|>commentary"   - full marker + suffix
#     "explorecommentary"              - marker stripped by parser, suffix remains
#     "StructuredOutput<|channel|>final"
#     "explore<|channel|>json"
#     "explore<|channel|>"             - bare dangling marker, NO suffix
#   leading + trailing:
#     "<|constrain|>StructuredOutput<|channel|>commentary"
#     "<|constrain|>explore<|channel|>commentary"
# Strategy: TWO passes — (1) kill leading `<|...|>` markers; (2) kill from the
# next `<|` onwards (handles trailing channel marker) AND any of the four
# harmony channel keywords as a bare suffix.
_HARMONY_LEAK_LEADING_RE = re.compile(r"^(?:<\|[^|]*\|>)+")
_HARMONY_LEAK_TRAILING_RE = re.compile(r"<\|.*$|(?:commentary|analysis|final|json)$")


def _strip_harmony_leak(raw_name: str) -> str:
    """Strip harmony channel-token leak from a tool-call name string.

    Returns the cleaned name. If the leak ate the entire name (defensive),
    returns the original to surface the AssertionError downstream rather
    than dispatch a phantom tool with empty name.
    """
    s = _HARMONY_LEAK_LEADING_RE.sub("", raw_name)
    s = _HARMONY_LEAK_TRAILING_RE.sub("", s)
    return s if s else raw_name


# Backwards-compat alias for any remaining call sites.
_HARMONY_LEAK_RE = _HARMONY_LEAK_TRAILING_RE

# Models served via vLLM that REQUIRE the `/v1/responses` (Harmony) endpoint
# for tool calling, NOT `/v1/chat/completions`. vLLM maintainers explicitly
# wontfix the chat/completions tool-calling path for these models (vllm#22578
# closed "not planned"); harmony channel tokens leak both forward (into tool
# name field, vllm#32587) and backward (into next-turn HTTP 500 message-header
# parse rejection). The Responses API is OpenAI's officially recommended path
# for harmony-format models and is the only path where vLLM's tool-call
# extraction is maintained.
#
# Add a new model alias here when it serves harmony format (`--reasoning-parser
# openai_gptoss` or similar harmony-channel parser). The dispatch in
# `run_tool_conversation` routes these models to `_run_tool_conversation_responses`.
HARMONY_MODELS: set[str] = {"gptoss-20b"}


def _get_client(model: str | None = None) -> AsyncOpenAI:
    """Return an AsyncOpenAI client whose base_url matches the model's serve.

    Args:
        model: vLLM `--served-model-name` alias (e.g. `gemma4-26b-a4b-it`,
               `gptoss-20b`). When None or not in MODEL_TO_BASE_URL,
               DEFAULT_VLLM_BASE_URL (port 8000) is used.

    Clients are cached per base_url so repeated calls reuse the same
    AsyncOpenAI instance and its connection pool.
    """
    base_url = MODEL_TO_BASE_URL.get(model, DEFAULT_VLLM_BASE_URL) if model else DEFAULT_VLLM_BASE_URL
    if base_url not in _clients:
        # timeout=1800.0 (30 min): openai client default is 600s. LCB eval with
        # max_tokens=65536 + thinking-mode decode hits >10min single-call wall
        # time on long-tail problems, which triggers httpcore.ReadTimeout and
        # crashes the eval (no try/except in eval.py per fail-fast policy).
        # Set 2026-05-02 after LCB eval crashed at [126/175] mid-decode.
        # 1800s covers up to ~36 tok/s × 1800 = 65k token decode comfortably.
        _clients[base_url] = AsyncOpenAI(base_url=base_url, api_key="not-needed", timeout=1800.0)
    return _clients[base_url]


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
    for k in ("top_k", "min_p", "repetition_penalty", "thinking_token_budget"):
        if s.get(k) is not None:
            extra[k] = s[k]
    if s.get("enable_thinking") is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": s["enable_thinking"]}
    # `disable_response_format` is a backend-side flag (NOT sent to vLLM); the
    # caller in `call_sub_model` reads it to decide whether to skip
    # `response_format=json_schema`. See vllm#40080: Gemma-4 + xgrammar guided
    # JSON deterministically loops; for those models we drop the schema
    # constraint and inject the schema as text instructions in user_message,
    # then post-hoc json.loads the response. The flag stays out of `direct`
    # and `extra` so vLLM never receives an unknown sampling parameter.
    return direct, extra


# ---------------------------------------------------------------------------
# Tool-call parsing
#
# Removed 2026-05-02: client-side tool-call parsing (regex over message.content)
# fully replaced by vLLM's server-side `--enable-auto-tool-choice
# --tool-call-parser <X>` path. The OpenAI response now carries structured
# `message.tool_calls=[{id, type, function:{name, arguments}}]` directly;
# `run_tool_conversation` reads that field instead of regex-parsing text.
#
# Why the switch: keeping client-side parsers meant vendoring + dispatching one
# parser per model family (Qwen3-8B JSON, Qwen3.6 XML, Gemma-4 call:NAME{...}).
# Each new model added a parser to maintain. vLLM ships 25+ tool parsers
# upstream (Hermes/qwen3xml/gemma4/llama3_json/...) and now serves them safely
# under DP after the thread-safety race (vllm#34932 "RuntimeError: Already
# borrowed") was fixed in vllm#40059 — parsers use vocab.get() instead of
# tokenizer.encode(), eliminating the HF Rust borrow conflict. Fix shipped in
# vllm 0.20.0 (2026-04-24).
#
# Each `serve_<model>.sh` MUST pass `--enable-auto-tool-choice
# --tool-call-parser <name>`, picking the parser for that model's chat
# template (gemma4 / qwen3xml / qwen3coder / hermes / mistral / ...).
# ---------------------------------------------------------------------------


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
    client = _get_client(model)

    # Gemma-4 + xgrammar guided JSON deterministically loops (vllm#40080,
    # gemma#622, gemma#610, ollama#15502). Workaround: drop response_format
    # and inject the schema as text instructions; rely on the model's own
    # JSON-emission ability + post-hoc json.loads. Caller opts in via
    # `sampling: {disable_response_format: true}` in YAML.
    disable_rf = bool((sampling or {}).get("disable_response_format"))
    user_text = user_message
    if disable_rf and output_schema:
        import json as _json
        user_text = (
            f"{user_message}\n\n"
            "Respond with ONLY a valid JSON object matching this schema. "
            "No prose, no markdown fences, no commentary — just the JSON object.\n"
            f"Schema:\n```json\n{_json.dumps(output_schema, indent=2)}\n```"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_content(user_text, image_data_url)},
    ]

    direct_kwargs, extra_body = _split_sampling_kwargs(sampling)
    direct_kwargs.setdefault("temperature", 0.0)
    direct_kwargs.setdefault("max_tokens", budget_tokens)
    assert 0.0 <= direct_kwargs["temperature"] <= 2.0, (
        f"temperature out of range: {direct_kwargs['temperature']}"
    )

    # Wrap the chat completion in a try/except that softly handles vllm's
    # context-overflow BadRequest (input + max_tokens > max-model-len). Per
    # design (2026-05-02 user-directed), max-model-len is set to 64K to
    # match explore_timeout=1200s (anything decoding past 64K tokens won't
    # finish under that wall-clock budget anyway). When a single explore's
    # accumulated input + max_tokens exceeds the configured ceiling, we
    # treat it the same as a timeout: write a soft sentinel result and
    # let the caller skip this explore without crashing the whole batch.
    # Only the structured `param in _CONTEXT_OVERFLOW_PARAMS` BadRequest is
    # exempted; other 400s (bad sampling args, unknown model, malformed
    # messages) still raise per fail-fast policy.
    try:
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
                if (output_schema and not disable_rf)
                else None
            ),
            extra_body=extra_body or None,
            **direct_kwargs,
        )
    except BadRequestError as e:
        # openai SDK flattens the response body: e.body is the inner error
        # dict directly (with keys: message, type, param, code), NOT the
        # outer {"error": {...}} wrapper from the raw HTTP response. Verified
        # by repro test 2026-05-02.
        body = e.body if isinstance(e.body, dict) else {}
        param = body.get("param") if isinstance(body, dict) else None
        if param in _CONTEXT_OVERFLOW_PARAMS:
            msg = body.get("message", "") if isinstance(body, dict) else ""
            logger.info(
                f"[vllm.call_sub_model] CONTEXT OVERFLOW soft-skip "
                f"param={param} model={model} msg={msg[:200]!r}"
            )
            # Mirror the timeout-equivalent contract used by parse_failed
            # (see vllm.py:355 below) so precache_explores.py:128 and the
            # eval/method layer's `if result.get('timed_out')` branches
            # short-circuit this explore as cached-but-skipped.
            return (
                {
                    "timed_out": True,
                    "context_overflow": True,
                    "param": param,
                    "finish_reason": "context_overflow",
                },
                "",
                0.0,
                {"input_tokens": 0, "output_tokens": 0},
            )
        raise

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    cost = _estimate_cost_usd(usage, model)

    # Parse structured output: rely on `response_format=json_schema` enforced
    # at the server. Output is direct JSON in `message.content`. No tool-call
    # parsing fallback — single-shot calls do not pass `tools=...`, so the
    # server-side auto-tool-choice path is not exercised here.
    result: dict = {}
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        pass

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
        logger.info(
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
    on_structured_output: Callable[[dict], None] | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    sampling: dict | None = None,
    **kwargs,
) -> tuple[float, dict[str, Any], bool]:
    """Run a multi-turn tool-calling conversation via vLLM.

    Uses vLLM's server-side `--enable-auto-tool-choice --tool-call-parser <X>`
    path. The OpenAI response carries structured `message.tool_calls=[...]`
    and `content=null` when the model fires a tool. No client-side text
    parsing. Each `serve_*.sh` MUST pass the matching `--tool-call-parser`
    name (gemma4 / qwen3xml / hermes / ...) for its model family.

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
    # Harmony-format models (gpt-oss family) cannot use chat/completions for
    # tool calling — vllm#22578 wontfix. Route to the Responses API adapter.
    if model in HARMONY_MODELS:
        return await _run_tool_conversation_responses(
            system_prompt=system_prompt,
            user_message=user_message,
            image_data_url=image_data_url,
            model=model,
            tools=tools,
            max_turns=max_turns,
            tool_handler=tool_handler,
            output_format=output_format,
            writer=writer,
            on_structured_output=on_structured_output,
            temperature=temperature,
            sampling=sampling,
        )

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
    client = _get_client(model)

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

    # Build OpenAI-compatible tools list. With `tool_choice="auto"` and the
    # server-side `--enable-auto-tool-choice --tool-call-parser <X>` flags,
    # vLLM populates structured `message.tool_calls=[...]` and leaves
    # `message.content=None` on tool turns. The vLLM 0.x thread-safety race
    # in tool-parser tokenizer.encode (vllm#34932) was fixed in vllm#40059
    # (merged 2026-04-24, included in 0.20.0); parsers now use vocab.get(...).
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
        # Soft-skip context-overflow BadRequest (input_tokens / input_text):
        # mirrors call_sub_model:326-373. The orchestrator multi-turn input
        # accumulates each cached-explore tool_response, and at the boundary
        # turn the (input + max_tokens) sum can exceed max-model-len by a
        # handful of tokens. Crashing the whole eval over a single boundary
        # turn is the wrong granularity; instead terminate this question's
        # orchestrator loop with exit_reason="context_overflow" and let the
        # eval/method layer continue with the next question.
        # 2026-05-02 incident: LCB eval crashed at 156/175 with
        # input_tokens=45537 (limit 45536); patch added.
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                extra_body=extra_body or None,
                **direct_kwargs,
            )
        except BadRequestError as e:
            body = e.body if isinstance(e.body, dict) else {}
            param = body.get("param") if isinstance(body, dict) else None
            if param in _CONTEXT_OVERFLOW_PARAMS:
                msg = body.get("message", "") if isinstance(body, dict) else ""
                logger.info(
                    f"[orchestrator] CONTEXT OVERFLOW soft-skip turn={turn} "
                    f"param={param} model={model} msg={msg[:200]!r}"
                )
                return _estimate_cost_usd(total_usage, model), total_usage, "context_overflow"
            raise

        choice = response.choices[0]
        if response.usage:
            total_usage["input_tokens"] += response.usage.prompt_tokens
            total_usage["output_tokens"] += response.usage.completion_tokens

        # Cap check: cumulative output_tokens exceeded user-set ceiling. vLLM's
        # natural unit is tokens (response.usage.completion_tokens), so we use
        # tokens directly instead of mirroring claude.py's char-based check.
        if max_output_tokens is not None and total_usage["output_tokens"] > max_output_tokens:
            logger.info(
                f"[orchestrator] output token cap exceeded "
                f"({total_usage['output_tokens']} > {max_output_tokens}), terminating"
            )
            return _estimate_cost_usd(total_usage, model), total_usage, "cap_exceeded"

        # --- Read structured tool_calls (server-side parser populated) ---
        text_content = choice.message.content or ""
        raw_tool_calls = choice.message.tool_calls or []
        # Decode each tool_call.function.arguments (JSON string) into a dict.
        # OpenAI/vLLM contract: arguments is always a JSON-encoded string.
        # Skip individual calls that fail to decode (server-side parser bug
        # OR truncated args) so one bad call doesn't poison the whole turn.
        text_calls: list[tuple[str, dict]] = []
        # cleaned_tool_calls mirrors raw_tool_calls but with sanitized
        # function.name. Used for the assistant-message history append below
        # so the polluted name does not feed back into the next turn.
        cleaned_tool_calls: list[dict] = []
        for tc in raw_tool_calls:
            try:
                args_dict = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[vllm turn {turn}] tool_call arguments JSON decode failed: "
                    f"name={tc.function.name!r} args_raw={tc.function.arguments!r} err={e}"
                )
                continue
            # vllm#32587 (OPEN, no upstream fix as of 2026-05-03): gpt-oss
            # openai tool_call_parser fails to terminate the function name at
            # the harmony channel marker. Two leak shapes seen empirically:
            #   1. with separator:    "explore<|channel|>commentary"
            #   2. without separator: "explorecommentary"  (parser dropped
            #      the `<|channel|>` marker but kept the channel keyword)
            # Regex strips both shapes: optional `<|...|>` + harmony channel
            # keyword suffix (commentary | analysis | final | json). Other
            # parsers (Gemma, Qwen) emit clean names so this is a no-op.
            clean_name = _HARMONY_LEAK_RE.sub("", tc.function.name)
            if clean_name != tc.function.name:
                logger.warning(
                    f"[vllm turn {turn}] tool_name special-token leak "
                    f"stripped: {tc.function.name!r} -> {clean_name!r} "
                    f"(vllm#32587)"
                )
            text_calls.append((clean_name, args_dict))
            cleaned_tool_calls.append({
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": clean_name,
                    "arguments": tc.function.arguments,
                },
            })

        logger.info(
            f"[vllm turn {turn}] finish={choice.finish_reason} "
            f"tools={len(text_calls)} content_len={len(text_content)}"
        )

        if writer and text_content:
            writer.write_text(text_content)

        if not text_calls:
            logger.info("[orchestrator] no tool call in response, ending")
            # Orchestrator emitted text without tool_call; no commit signal.
            return _estimate_cost_usd(total_usage, model), total_usage, "incomplete"

        # Append assistant message with structured tool_calls (OpenAI standard
        # multi-turn format). The chat_template's tool-rendering branch
        # serializes message.tool_calls back into the model's native format
        # (Gemma `<|tool_call>call:NAME{...}<tool_call|>`, Qwen `<tool_call>
        # ...</tool_call>`, etc.), so model sees its own prior calls in the
        # native syntax it was trained on. Drop assistant `content` (free-text
        # reasoning) — historically dropped too, this preserves history budget.
        messages.append({
            "role": "assistant",
            "content": None,
            # cleaned_tool_calls (built above) carries sanitized function.name
            # — see vllm#32587 strip block. Using raw_tool_calls here would
            # feed the polluted name back into the model on the next turn,
            # which compounds the bug.
            "tool_calls": cleaned_tool_calls,
        })

        for tc, (name, args) in zip(raw_tool_calls, text_calls):
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
                    logger.info(
                        f"[structured_output_invalid] missing {missing}; "
                        f"got keys={list(args.keys())}"
                    )
                    if writer:
                        writer.write_text(
                            f"[StructuredOutput rejected: missing required "
                            f"fields {missing}; got keys={list(args.keys())}]"
                        )
                    # Inject corrective `tool` reply so chat history stays valid
                    # (every assistant.tool_call MUST have a matching tool reply).
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": (
                            f"StructuredOutput rejected: missing required "
                            f"fields {missing}. Call StructuredOutput again "
                            f"with ALL required fields filled per the schema."
                        ),
                    })
                    break
                logger.info(f"[structured_output] {args}")
                if writer:
                    writer.write_tool_use("StructuredOutput", args)
                if on_structured_output:
                    on_structured_output(args)
                return _estimate_cost_usd(total_usage, model), total_usage, "committed"

            logger.info(f"[tool_use] {name}")
            if writer:
                writer.write_tool_use(name, args)

            result_text, should_stop = await tool_handler(name, args)
            if writer:
                writer.write_tool_result(result_text)

            # Inject tool result as `role="tool"` message keyed by tool_call_id
            # (OpenAI standard). The chat_template's tool_response branch
            # renders this in the model's native format
            # (Gemma `<|tool_response>response:NAME{value:...}<tool_response|>`,
            # Qwen `<tool_response>...</tool_response>`).
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text,
            })

            if should_stop:
                # Tool signaled end (e.g. tts_agent budget exhausted after final
                # explore). Treated as committed — orchestrator's flow ended cleanly.
                return _estimate_cost_usd(total_usage, model), total_usage, "committed"

    # for-loop walked all max_turns without commitment.
    return _estimate_cost_usd(total_usage, model), total_usage, "incomplete"


async def _run_tool_conversation_responses(
    *,
    system_prompt: str,
    user_message: str,
    image_data_url: str | None,
    model: str,
    tools: list[dict[str, Any]],
    max_turns: int,
    tool_handler: Callable[[str, dict], Awaitable[tuple[str, bool]]],
    output_format: dict[str, Any] | None = None,
    writer=None,
    on_structured_output: Callable[[dict], None] | None = None,
    temperature: float | None = None,
    sampling: dict | None = None,
) -> tuple[float, dict[str, Any], str]:
    """Multi-turn tool-calling via vLLM `/v1/responses` (Harmony) endpoint.

    Mirrors the contract of `run_tool_conversation` (same return tuple, same
    tool_handler / on_structured_output callbacks) but uses the OpenAI SDK's
    Responses API (`client.responses.create`) instead of Chat Completions.

    Why a separate adapter:
    - Harmony-format models (gpt-oss-20b, gpt-oss-120b) leak channel tokens
      (`<|channel|>commentary` etc.) into chat/completions tool_call.name
      AND into next-turn message headers (vllm#32587 OPEN, vllm#22578 wontfix).
    - The Responses API is OpenAI's officially recommended path for these
      models and the path vLLM maintainers actually maintain for tool calling.
    - Responses API uses different shapes:
      - tools: `[{type:function, name, description, parameters}]` — NOT
        nested under `function` key like chat/completions
      - input items: messages + function_call + function_call_output
      - response.output: list of items (function_call, message, reasoning)
      - multi-turn via `previous_response_id` (server-stateful) — next call
        sends only the new function_call_output items, not full history

    Limitations of this adapter (raise loudly rather than silently degrade):
    - image_data_url unsupported (HLE-Verified text_only is the smoke target;
      multimodal Responses API tool-calling shape needs separate validation)

    References:
    - vLLM Recipes GPT-OSS: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
    - OpenAI Cookbook: https://developers.openai.com/cookbook/articles/gpt-oss/run-vllm
    - Responses API tool-call shape: Azure docs (mirrors openai-python SDK)
    """
    if image_data_url is not None:
        raise NotImplementedError(
            "Responses-API adapter does not yet handle image inputs. "
            "Implement multimodal tool-calling shape before enabling for "
            "BabyVision/RBenchV. Current scope: text_only HLE/GPQA/LCB."
        )

    # --- Sampling: Responses API takes temperature / top_p / max_output_tokens
    # at top level; vLLM-specific knobs go via `extra_body`. ---
    direct_kwargs, extra_body = _split_sampling_kwargs(sampling)
    if "temperature" not in direct_kwargs and temperature is not None:
        direct_kwargs["temperature"] = temperature
    direct_kwargs.setdefault("temperature", 0.0)
    # Responses API uses `max_output_tokens` (per-turn cap), NOT `max_tokens`.
    if "max_tokens" in direct_kwargs:
        direct_kwargs["max_output_tokens"] = direct_kwargs.pop("max_tokens")
    direct_kwargs.setdefault("max_output_tokens", 8192)
    assert 0.0 <= direct_kwargs["temperature"] <= 2.0, (
        f"temperature out of range: {direct_kwargs['temperature']}"
    )
    client = _get_client(model)

    # --- StructuredOutput tool injection (parallel to chat/completions branch,
    # but with gpt-oss / harmony-format-specific reinforcements). The earlier
    # version mirrored chat/completions' protocol block. Smoke run on 2026-05-03
    # (run_20260503_071121) showed gpt-oss-20b finishing reasoning with "So
    # answer is B" in its `<|channel|>analysis` reasoning channel and exiting
    # without calling StructuredOutput. The harmony format trains the model to
    # treat reasoning as the answer surface; the prompt must explicitly forbid
    # that and require StructuredOutput as the LAST action of every turn.
    required: list[str] = []
    augmented_system = system_prompt
    if output_format and output_format.get("schema"):
        schema_json = json.dumps(output_format["schema"], indent=2)
        required = output_format["schema"].get("required", [])
        augmented_system = (
            system_prompt
            + "\n\n=== EXTREMELY IMPORTANT --- FINAL ANSWER SUBMISSION PROTOCOL ===\n"
            "WHEN YOUR REASONING IS COMPLETE, YOU MUST SUBMIT YOUR FINAL ANSWER BY CALLING THE `StructuredOutput` TOOL.\n"
            "THIS IS THE ONLY ACCEPTED SUBMISSION PATH. NOTHING ELSE COUNTS AS A SUBMISSION.\n"
            "\n"
            "THE FOLLOWING ARE NOT VALID SUBMISSIONS AND WILL BE GRADED AS 0:\n"
            "  - Stating the answer in your private analysis / reasoning / chain-of-thought channel.\n"
            "  - Writing the answer in free-form text or final message (e.g. `\\boxed{X}`, `Answer: X`, `The answer is X`).\n"
            "  - Saying 'So the answer is X' anywhere except inside the `StructuredOutput` tool arguments.\n"
            "  - Ending the conversation without a `StructuredOutput` tool call.\n"
            "\n"
            "MANDATORY FORMAT: Your VERY LAST action in the conversation MUST be a `StructuredOutput` tool call.\n"
            "If you have arrived at an answer in your reasoning, you have NOT submitted yet — you must still emit the tool call.\n"
            "Even if the answer feels obvious, even if you are certain, even if it is a single character — call `StructuredOutput`.\n"
            "\n"
            "Use strict JSON format for the tool arguments. "
            f"All of these fields are required and must each be a separate JSON key: {', '.join(required)}.\n"
            f"Schema:\n```json\n{schema_json}\n```\n"
        )

    # --- Build Responses-API tools list (flat shape, no `function` nesting) ---
    responses_tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["parameters"],
        }
        for t in tools
    ]
    if output_format and output_format.get("schema"):
        responses_tools.append({
            "type": "function",
            "name": "StructuredOutput",
            "description": "Submit the final structured answer.",
            "parameters": output_format["schema"],
        })

    # --- Conversation history kept CLIENT-SIDE (vLLM `/v1/responses` is
    # stateless — `previous_response_id` returns 404 because vLLM does not
    # implement a response store). Each turn we re-send the full input list
    # including prior function_call items and function_call_output items.
    # The model gets identical context each iteration; vLLM rebuilds harmony
    # state from the input items every call. ---
    full_input: list[dict[str, Any]] = [
        {"role": "system", "content": augmented_system},
        {"role": "user", "content": user_message},
    ]
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for turn in range(max_turns):
        try:
            create_kwargs: dict[str, Any] = {
                "model": model,
                "input": full_input,
                "tools": responses_tools,
                **direct_kwargs,
            }
            if extra_body:
                create_kwargs["extra_body"] = extra_body
            response = await client.responses.create(**create_kwargs)
        except BadRequestError as e:
            body = e.body if isinstance(e.body, dict) else {}
            param = body.get("param") if isinstance(body, dict) else None
            if param in _CONTEXT_OVERFLOW_PARAMS:
                msg = body.get("message", "") if isinstance(body, dict) else ""
                logger.info(
                    f"[orchestrator-responses] CONTEXT OVERFLOW soft-skip turn={turn} "
                    f"param={param} model={model} msg={msg[:200]!r}"
                )
                return _estimate_cost_usd(total_usage, model), total_usage, "context_overflow"
            raise

        if getattr(response, "usage", None):
            total_usage["input_tokens"] += getattr(response.usage, "input_tokens", 0)
            total_usage["output_tokens"] += getattr(response.usage, "output_tokens", 0)

        # Walk response.output[] for function_call items. message / reasoning
        # items get written to trajectory so we can debug "why did the model
        # decide not to call any tool" — gpt-oss in particular tends to emit
        # only function_call items and skip the message channel entirely; the
        # reasoning channel is then the only place where its decision logic is
        # visible. Without writing reasoning to trajectory, the trajectory.md
        # for an "incomplete" exit (no commit, no tool calls) is empty and
        # diagnosis is impossible.
        function_calls: list[Any] = []
        text_messages: list[str] = []
        reasoning_messages: list[str] = []
        # ROOT-CAUSE FIX 2026-05-03 (vllm#33089 + vLLM Responses-API harmony
        # spec): the harmony format requires conversation history to include
        # reasoning items AND assistant message items, not just function_calls.
        # vLLM's response_input_to_harmony() expects 4 item kinds in input:
        # messages, reasoning items, function_call, function_call_output.
        # Earlier code dropped reasoning + assistant message items between
        # turns, breaking the model's chain-of-thought continuity. Symptom:
        # gpt-oss in multi-turn ATTS commits StructuredOutput only ~20% of
        # the time at T=1.0 / T=0.7 because each turn it "forgets" why it
        # called explore previously and arbitrarily decides to stop. Keep
        # the raw items so we can roundtrip them back into next-turn input.
        all_items_for_history: list[Any] = []
        for item in response.output:
            item_type = getattr(item, "type", None)
            all_items_for_history.append(item)
            # gpt-oss with `--reasoning-parser openai_gptoss` returns
            # tool calls as `mcp_call` items (server_label="functions") rather
            # than the plain `function_call` items emitted on the chat/completions
            # path. Both carry .name and .arguments and dispatch identically.
            # Verified empirically 2026-05-03: with reasoning-parser flag enabled
            # 100% of explore dispatches arrived as mcp_call; without the flag
            # they arrived as function_call. Accept both.
            if item_type in ("function_call", "mcp_call"):
                function_calls.append(item)
            elif item_type == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text_messages.append(getattr(content, "text", "") or "")
            elif item_type == "reasoning":
                # gpt-oss / harmony-format models emit analysis-channel CoT
                # here. Concatenate any reasoning_text content blocks for
                # trajectory.md AND keep the raw item for next-turn history.
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "reasoning_text":
                        reasoning_messages.append(getattr(content, "text", "") or "")

        text_content = "\n".join(text_messages)
        reasoning_content = "\n".join(reasoning_messages)
        logger.info(
            f"[vllm-responses turn {turn}] tools={len(function_calls)} "
            f"content_len={len(text_content)} reasoning_len={len(reasoning_content)}"
        )
        if writer and reasoning_content:
            writer.write_text(f"[reasoning]\n{reasoning_content}")
        if writer and text_content:
            writer.write_text(text_content)

        # No tool calls — model emitted a final message OR ran out of turns
        if not function_calls:
            # DEBUG 2026-05-03: dump every output item raw shape to diagnose
            # whether the model actually emitted a function_call item that we
            # dropped, or genuinely emitted no function_call.
            try:
                debug_items = [item.model_dump(exclude_none=True) for item in response.output]
                logger.info(
                    f"[vllm-responses INCOMPLETE_DEBUG turn={turn}] "
                    f"output_items_count={len(response.output)} "
                    f"item_types={[getattr(it, 'type', '?') for it in response.output]} "
                    f"raw_dump={json.dumps(debug_items)[:2000]}"
                )
            except Exception as _e:
                logger.info(f"[vllm-responses INCOMPLETE_DEBUG turn={turn}] dump failed: {_e}")
            return _estimate_cost_usd(total_usage, model), total_usage, "incomplete"

        # Dispatch each function_call → collect function_call_output items for next turn
        next_input: list[dict[str, Any]] = []
        for fc in function_calls:
            # vllm#32587 strip applies to Responses API too — empirically the
            # harmony channel-token leak surfaces in `fc.name` ("explore<|channel|>json")
            # on the responses path as well, not just chat/completions. Same
            # regex defensive layer.
            raw_name = fc.name
            name = _strip_harmony_leak(raw_name)
            if name != raw_name:
                logger.warning(
                    f"[vllm-responses turn {turn}] tool_name special-token "
                    f"leak stripped: {raw_name!r} -> {name!r} (vllm#32587)"
                )
                # Mutate fc.name so downstream history append uses clean name
                fc.name = name
            try:
                args = json.loads(fc.arguments) if fc.arguments else {}
            except json.JSONDecodeError as e:
                logger.warning(
                    f"[vllm-responses turn {turn}] function_call arguments "
                    f"JSON decode failed: name={name!r} args_raw={fc.arguments!r} err={e}"
                )
                continue

            if name == "StructuredOutput":
                missing = [k for k in required if k not in args]
                if missing:
                    logger.info(
                        f"[structured_output_invalid] missing {missing}; "
                        f"got keys={list(args.keys())}"
                    )
                    if writer:
                        writer.write_text(
                            f"[StructuredOutput rejected: missing required "
                            f"fields {missing}; got keys={list(args.keys())}]"
                        )
                    next_input.append({
                        "type": "function_call_output",
                        # mcp_call items lack call_id — fall back to id.
                        # function_call items always carry both.
                        "call_id": getattr(fc, "call_id", None) or fc.id,
                        "output": (
                            f"StructuredOutput rejected: missing required "
                            f"fields {missing}. Call StructuredOutput again "
                            f"with ALL required fields filled per the schema."
                        ),
                    })
                    continue
                logger.info(f"[structured_output] {args}")
                if writer:
                    writer.write_tool_use("StructuredOutput", args)
                if on_structured_output:
                    on_structured_output(args)
                return _estimate_cost_usd(total_usage, model), total_usage, "committed"

            logger.info(f"[tool_use] {name}")
            if writer:
                writer.write_tool_use(name, args)

            result_text, should_stop = await tool_handler(name, args)
            if writer:
                writer.write_tool_result(result_text)

            next_input.append({
                "type": "function_call_output",
                "call_id": getattr(fc, "call_id", None) or fc.id,
                "output": result_text,
            })

            if should_stop:
                return _estimate_cost_usd(total_usage, model), total_usage, "committed"

        # Append ALL of this turn's output items (reasoning, message, function_call)
        # plus our function_call_output items to the running history. Per
        # vllm#33089 + harmony spec, multi-turn Responses API requires the
        # full set of item kinds to be roundtripped back as input — dropping
        # reasoning items breaks the model's chain-of-thought continuity and
        # the model behaves as if each turn starts fresh.
        #
        # ASYMMETRY (verified empirically 2026-05-03): vLLM Responses API can
        # OUTPUT `mcp_call` items but its input parser rejects them with
        # `400 'Unknown input type: mcp_call'`. So mcp_call items must be
        # reshaped to `function_call` shape (which the input parser DOES accept)
        # before going into history. Both carry .name/.arguments/.id; the only
        # field difference is mcp_call has `server_label` while function_call
        # has `call_id`. Drop server_label and synthesize call_id = id.
        for item in all_items_for_history:
            d = item.model_dump(exclude_none=True)
            if d.get("type") == "mcp_call":
                d = {
                    "type": "function_call",
                    "name": d["name"],
                    "arguments": d.get("arguments", "{}"),
                    "call_id": d.get("call_id") or d.get("id"),
                    "id": d.get("id"),
                }
            full_input.append(d)
        full_input.extend(next_input)

    return _estimate_cost_usd(total_usage, model), total_usage, "incomplete"

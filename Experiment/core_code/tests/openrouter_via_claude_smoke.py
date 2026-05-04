"""Smoke test: can backends/claude.py be reused for OpenRouter via env-var routing?

Three probes, each takes one (probe, model) pair via CLI:

  probe a — call_sub_model via claude.py (Anthropic Skin path)
            tests "can the model produce structured JSON when wrapped by claude_agent_sdk
            via OpenRouter's Anthropic-compatible endpoint?"
  probe c — discriminator: OpenAI-compatible endpoint + mock tool
            tests "does this model have tool-calling capability AT ALL on OpenRouter?"
            Bypasses claude.py entirely; goes direct to /api/v1/chat/completions.
  probe b — run_tool_conversation via claude.py (Anthropic Skin path)
            tests "can the orchestrator pattern (multi-turn explore + StructuredOutput)
            work via the Anthropic Skin path?"

Discriminator logic (when paired with same-model results):
  a✓ + c✓ + b✓ → VIABLE: claude.py can be reused for this model end-to-end
  a✓ + c✓ + b✗ → H2 (protocol translation issue): model has tool-call capability on
                  OpenRouter but Anthropic Skin's translation layer drops it
  a✓ + c✗ + b✗ → H1 (model capability issue): model itself can't tool-call on OpenRouter,
                  not a Skin issue
  a✗            → model can't even produce JSON via claude.py; investigate before deeper

Run examples:
    python tests/openrouter_via_claude_smoke.py --probe a --model openai/gpt-oss-120b:free
    python tests/openrouter_via_claude_smoke.py --probe c --model openai/gpt-oss-120b:free
    python tests/openrouter_via_claude_smoke.py --probe b --model openai/gpt-oss-120b:free
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Anthropic Skin envs — required for probe a/b (claude.py path). Harmless to set
# when only running probe c (AsyncOpenAI uses its own auth via api_key kwarg).
# MUST be set before any claude_agent_sdk import.
os.environ["ANTHROPIC_BASE_URL"] = "https://openrouter.ai/api"
os.environ["ANTHROPIC_AUTH_TOKEN"] = os.environ["OPENROUTER_API_KEY"]
os.environ["ANTHROPIC_API_KEY"] = ""  # OpenRouter docs: must be explicitly empty
# Required when this script is launched from inside an active Claude Code session
# (CLAUDECODE=1 makes the SDK refuse to spawn a nested CLI). Mirrors the same
# defensive pop in precache_explores.py:25.
os.environ.pop("CLAUDECODE", None)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Optional: when CLAUDE_AGENT_BARE_MODE=1, monkey-patch ClaudeAgentOptions so that
# every options instance includes extra_args={"bare": None}. This makes the
# spawned `claude` CLI pass --bare, which (per its --help) skips: hooks, LSP,
# plugin sync, attribution, auto-memory, background prefetches, keychain reads,
# and CLAUDE.md auto-discovery. Hypothesis: this also suppresses the hidden
# Haiku safety/classifier 1-token call that costs ~$0.0066 per claude.py
# invocation (verified 2026-05-03 against OpenRouter activity log).
if os.environ.get("CLAUDE_AGENT_BARE_MODE") == "1":
    import claude_agent_sdk
    _orig_options_cls = claude_agent_sdk.ClaudeAgentOptions

    def _bare_options(*args, **kwargs):
        extra = kwargs.get("extra_args") or {}
        extra = {**extra, "bare": None}
        kwargs["extra_args"] = extra
        return _orig_options_cls(*args, **kwargs)

    claude_agent_sdk.ClaudeAgentOptions = _bare_options
    # backends/claude.py imports ClaudeAgentOptions directly from claude_agent_sdk,
    # so the patch must run BEFORE that import. The lazy `import backends.claude as cb`
    # inside probe_a/probe_b ensures this ordering as long as no claude_agent_sdk
    # symbol was already imported at module level (guarded above by the if).


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning":  {"type": "string", "description": "Brief reasoning (1-2 sentences)."},
        "answer":     {"type": "string", "description": "Final answer (single integer or short phrase)."},
        "confidence": {"type": "number", "description": "0-1 confidence."},
    },
    "required": ["reasoning", "answer", "confidence"],
    "additionalProperties": False,
}

SYSTEM_PROMPT_A = "You are a math solver. Reason briefly and answer."
USER_MSG_A = "What is 17 * 23? Submit your answer via StructuredOutput."

SYSTEM_PROMPT_B = (
    "You are an orchestrator. You have one tool: `mock_explore` (no params). "
    "Call `mock_explore` exactly once, then submit the final answer via StructuredOutput."
)
USER_MSG_B = "What is 17 * 23?"

SYSTEM_PROMPT_C = (
    "You have one tool: mock_explore (no params). Call it once to get a candidate "
    "answer; then state the final numerical answer in plain text."
)
USER_MSG_C = "What is 17 * 23?"


async def probe_a(model: str) -> dict:
    """Path A: single call_sub_model via claude.py (Anthropic Skin)."""
    import backends.claude as cb
    from trajectory import TrajectoryWriter

    obs = {"path": "A", "model": model}
    t0 = time.time()
    try:
        result, traj, cost, usage = await cb.call_sub_model(
            system_prompt=SYSTEM_PROMPT_A,
            user_message=USER_MSG_A,
            image_data_url=None,
            model=model,
            output_schema=SIMPLE_SCHEMA,
            writer=TrajectoryWriter.noop(),
            budget_tokens=4000,
            effort="low",
            sampling=None,
        )
        obs.update(
            ok=True,
            duration=round(time.time() - t0, 2),
            answer=str(result.get("answer", ""))[:80],
            reasoning_len=len(str(result.get("reasoning", ""))),
            confidence=result.get("confidence"),
            timed_out=bool(result.get("timed_out")),
            trajectory_len=len(traj),
            cost_usd=cost,
            input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
            structured_output_fired=("answer" in result and not result.get("timed_out")),
        )
    except Exception as e:
        obs.update(
            ok=False,
            duration=round(time.time() - t0, 2),
            error_type=type(e).__name__,
            error=str(e)[:500],
        )
    return obs


async def probe_b(model: str) -> dict:
    """Path B: run_tool_conversation with mock explore tool + StructuredOutput, via claude.py."""
    import backends.claude as cb
    from trajectory import TrajectoryWriter

    obs = {"path": "B", "model": model}
    t0 = time.time()

    explore_calls = {"n": 0}
    structured_payload = {"value": None}

    async def tool_handler(name: str, args: dict) -> tuple[str, bool]:
        if name == "mock_explore":
            explore_calls["n"] += 1
            return "Explore returned: candidate answer 391, reasoning: 17*20+17*3=340+51=391.", False
        return f"Unknown tool: {name}", False

    def on_structured_output(payload: dict) -> None:
        structured_payload["value"] = payload

    tools = [{
        "name": "mock_explore",
        "description": "Dispatch a fresh solver to produce a candidate answer. Takes no parameters.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }]

    try:
        cost, usage, exit_reason = await cb.run_tool_conversation(
            system_prompt=SYSTEM_PROMPT_B,
            user_message=USER_MSG_B,
            image_data_url=None,
            model=model,
            tools=tools,
            max_turns=4,
            tool_handler=tool_handler,
            effort="low",
            output_format={"type": "json_schema", "schema": SIMPLE_SCHEMA},
            writer=TrajectoryWriter.noop(),
            on_structured_output=on_structured_output,
        )
        sp = structured_payload["value"] or {}
        obs.update(
            ok=True,
            duration=round(time.time() - t0, 2),
            exit_reason=exit_reason,
            explore_calls=explore_calls["n"],
            structured_output_fired=structured_payload["value"] is not None,
            structured_payload_keys=list(sp.keys()) if sp else [],
            answer=str(sp.get("answer", ""))[:80] if sp else None,
            cost_usd=cost,
            input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
            output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
        )
    except Exception as e:
        obs.update(
            ok=False,
            duration=round(time.time() - t0, 2),
            exit_reason=None,
            explore_calls=explore_calls["n"],
            structured_output_fired=structured_payload["value"] is not None,
            error_type=type(e).__name__,
            error=str(e)[:500],
        )
    return obs


async def probe_c(model: str) -> dict:
    """Discriminator: OpenAI-compatible endpoint + mock tool. Bypasses claude.py entirely.

    Pass = model can call tools on OpenRouter at all (capability confirmed).
    Used together with probe_b: if c passes but b fails, the issue is the
    Anthropic-Skin protocol translation layer (H2), not the model (H1).
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mock_explore",
                "description": "Dispatch a fresh solver. Takes no parameters.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]

    obs = {"path": "C", "model": model}
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_C},
                {"role": "user", "content": USER_MSG_C},
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=2048,
            timeout=180.0,
        )
        choice = resp.choices[0]
        tcs = choice.message.tool_calls or []
        first_tool_name = tcs[0].function.name if tcs else None
        usage = getattr(resp, "usage", None)
        obs.update(
            ok=True,
            duration=round(time.time() - t0, 2),
            finish_reason=choice.finish_reason,
            tool_call_count=len(tcs),
            tool_called=first_tool_name,
            tool_call_correct=(first_tool_name == "mock_explore"),
            content_len=len(choice.message.content or ""),
            input_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            output_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )
    except Exception as e:
        obs.update(
            ok=False,
            duration=round(time.time() - t0, 2),
            error_type=type(e).__name__,
            error=str(e)[:500],
        )
    return obs


async def main(probe: str, model: str) -> int:
    banner = f"PROBE={probe}  MODEL={model}  TS={time.strftime('%Y-%m-%d %H:%M:%S')}"
    print("=" * len(banner))
    print(banner)
    print("=" * len(banner))
    print(f"ANTHROPIC_BASE_URL  = {os.environ.get('ANTHROPIC_BASE_URL')}")
    print(f"ANTHROPIC_AUTH_TOKEN= {os.environ.get('ANTHROPIC_AUTH_TOKEN', '')[:10]}...")
    print(f"OPENROUTER_API_KEY  = {os.environ.get('OPENROUTER_API_KEY', '')[:10]}...")
    print(f"CLAUDECODE          = {os.environ.get('CLAUDECODE', '<unset>')}")
    print()

    if probe == "a":
        obs = await probe_a(model)
    elif probe == "b":
        obs = await probe_b(model)
    elif probe == "c":
        obs = await probe_c(model)
    elif probe in ("a_via_openrouter", "b_via_openrouter"):
        # End-to-end via the real backends/openrouter.py (no claude_agent_sdk).
        # No Haiku tax by construction: this path uses AsyncOpenAI directly.
        from importlib import import_module
        backend_mod = import_module("backends.openrouter")
        from trajectory import TrajectoryWriter
        if probe == "a_via_openrouter":
            obs = {"path": "A_via_openrouter", "model": model}
            t0 = time.time()
            try:
                result, traj, cost, usage = await backend_mod.call_sub_model(
                    system_prompt=SYSTEM_PROMPT_A, user_message=USER_MSG_A,
                    image_data_url=None, model=model, output_schema=SIMPLE_SCHEMA,
                    writer=TrajectoryWriter.noop(), budget_tokens=4000, effort="low", sampling=None,
                )
                obs.update(
                    ok=True, duration=round(time.time() - t0, 2),
                    answer=str(result.get("answer", ""))[:80],
                    timed_out=bool(result.get("timed_out")),
                    trajectory_len=len(traj),
                    cost_usd=cost,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    structured_output_fired=("answer" in result and not result.get("timed_out")),
                )
            except Exception as e:
                obs.update(ok=False, duration=round(time.time() - t0, 2),
                           error_type=type(e).__name__, error=str(e)[:500])
        else:  # b_via_openrouter
            obs = {"path": "B_via_openrouter", "model": model}
            t0 = time.time()
            explore_calls_b = {"n": 0}
            structured_payload_b = {"value": None}

            async def tool_handler_b(name, args):
                if name == "mock_explore":
                    explore_calls_b["n"] += 1
                    return "Explore returned: candidate answer 391, reasoning: 17*20+17*3=391.", False
                return f"Unknown tool: {name}", False

            def on_structured_output_b(payload):
                structured_payload_b["value"] = payload

            tools_b = [{"name": "mock_explore",
                        "description": "Dispatch a fresh solver to produce a candidate answer. Takes no parameters.",
                        "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}]
            try:
                cost, usage, exit_reason = await backend_mod.run_tool_conversation(
                    system_prompt=SYSTEM_PROMPT_B, user_message=USER_MSG_B,
                    image_data_url=None, model=model, tools=tools_b, max_turns=4,
                    tool_handler=tool_handler_b, effort="low",
                    output_format={"type": "json_schema", "schema": SIMPLE_SCHEMA},
                    writer=TrajectoryWriter.noop(),
                    on_structured_output=on_structured_output_b,
                )
                sp = structured_payload_b["value"] or {}
                obs.update(
                    ok=True, duration=round(time.time() - t0, 2),
                    exit_reason=exit_reason, explore_calls=explore_calls_b["n"],
                    structured_output_fired=structured_payload_b["value"] is not None,
                    structured_payload_keys=list(sp.keys()) if sp else [],
                    answer=str(sp.get("answer", ""))[:80] if sp else None,
                    cost_usd=cost,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                )
            except Exception as e:
                obs.update(ok=False, duration=round(time.time() - t0, 2),
                           exit_reason=None, explore_calls=explore_calls_b["n"],
                           structured_output_fired=structured_payload_b["value"] is not None,
                           error_type=type(e).__name__, error=str(e)[:500])
    elif probe == "a_via_dispatcher":
        # End-to-end verification: import via the same dispatcher path used by
        # methods/base.py (`import_module(f"backends.{ctx.backend}")`). Validates
        # that backends/openrouter.py + the in-source DISABLE flags in
        # backends/claude.py work together without any external env-var prep.
        from importlib import import_module
        backend_mod = import_module("backends.openrouter")

        from trajectory import TrajectoryWriter
        obs = {"path": "A_via_dispatcher", "model": model}
        t0 = time.time()
        try:
            result, traj, cost, usage = await backend_mod.call_sub_model(
                system_prompt=SYSTEM_PROMPT_A,
                user_message=USER_MSG_A,
                image_data_url=None,
                model=model,
                output_schema=SIMPLE_SCHEMA,
                writer=TrajectoryWriter.noop(),
                budget_tokens=4000,
                effort="low",
                sampling=None,
            )
            obs.update(
                ok=True,
                duration=round(time.time() - t0, 2),
                answer=str(result.get("answer", ""))[:80],
                timed_out=bool(result.get("timed_out")),
                trajectory_len=len(traj),
                input_tokens=usage.get("input_tokens") if isinstance(usage, dict) else None,
                output_tokens=usage.get("output_tokens") if isinstance(usage, dict) else None,
                structured_output_fired=("answer" in result and not result.get("timed_out")),
            )
        except Exception as e:
            obs.update(ok=False, duration=round(time.time() - t0, 2),
                       error_type=type(e).__name__, error=str(e)[:500])
    else:
        print(f"Unknown probe: {probe!r}", file=sys.stderr)
        return 2

    print("--- OBSERVATIONS ---")
    print(json.dumps(obs, indent=2, default=str))

    passed = obs.get("ok") and (
        obs.get("structured_output_fired") if probe in ("a", "b", "a_via_dispatcher", "a_via_openrouter", "b_via_openrouter") else
        obs.get("tool_call_correct") if probe == "c" else False
    )
    tag = "PASS" if passed else "FAIL"
    print()
    print(f"[{tag}] probe={probe.upper()} model={model}")
    return 0 if passed else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True,
                    choices=["a", "b", "c", "a_via_dispatcher", "a_via_openrouter", "b_via_openrouter"],
                    help="a/b=via claude.py; c=OpenAI-endpoint discriminator; a_via_dispatcher=legacy thin-alias; a_via_openrouter / b_via_openrouter = real backends.openrouter")
    ap.add_argument("--model", required=True, help="OpenRouter model id, e.g. openai/gpt-oss-120b:free")
    args = ap.parse_args()
    try:
        sys.exit(asyncio.run(main(args.probe, args.model)))
    except Exception:
        traceback.print_exc()
        sys.exit(1)

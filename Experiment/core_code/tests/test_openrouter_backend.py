"""Unit tests for backends/openrouter.py contract.

Mocked AsyncOpenAI; no live network calls. Run via:
    cd /data3/peijia/dr-claw/Explain/Experiment/core_code
    conda run -n explain pytest tests/test_openrouter_backend.py -v

Tests are written with `asyncio.run(...)` inline rather than @pytest.mark.asyncio
so pytest-asyncio is NOT required.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_module_imports_and_asserts_api_key(monkeypatch):
    """Module-level assert OPENROUTER_API_KEY in env."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AssertionError, match="OPENROUTER_API_KEY"):
        importlib.import_module("backends.openrouter")


def test_module_imports_with_api_key_set(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-fake-token-do-not-use")
    mod = importlib.import_module("backends.openrouter")
    assert mod.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_split_sampling_none(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    direct, extra = mod._split_sampling_kwargs(None)
    assert direct == {}
    assert extra == {}


def test_split_sampling_drops_none_values(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    s = {"temperature": None, "top_p": 0.9, "max_tokens": None}
    direct, extra = mod._split_sampling_kwargs(s)
    assert direct == {"top_p": 0.9}
    assert extra == {}


def test_split_sampling_routes_vllm_extras(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    s = {"temperature": 0.7, "top_k": 50, "min_p": 0.05}
    direct, extra = mod._split_sampling_kwargs(s)
    assert direct == {"temperature": 0.7}
    assert extra == {"top_k": 50, "min_p": 0.05}


def _make_mock_call_sub_resp(args_json: str, prompt_tokens: int = 100, completion_tokens: int = 20, cost: float = 0.0, reasoning: str = ""):
    """Mock the call_sub_model response shape: tool_calls with StructuredOutput function."""
    fn = MagicMock()
    fn.name = "StructuredOutput"
    fn.arguments = args_json
    tc = MagicMock(id="call_so", type="function", function=fn)
    msg = MagicMock(content=None, tool_calls=[tc], reasoning=reasoning)
    choice = MagicMock(message=msg, finish_reason="tool_calls")
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost)
    return MagicMock(choices=[choice], usage=usage)


def test_call_sub_model_returns_parsed_args_from_tool_call(monkeypatch):
    """Mechanism: forced tool_choice on StructuredOutput; parse args, not content."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    fake_resp = _make_mock_call_sub_resp(
        '{"answer": "42", "reasoning": "obvious", "confidence": 0.9}',
        cost=0.0, reasoning="thinking thinking",
    )
    mock_create = AsyncMock(return_value=fake_resp)
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    from trajectory import TrajectoryWriter
    schema = {"type": "object", "properties": {"answer": {"type": "string"}, "reasoning": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["answer", "reasoning", "confidence"]}
    result, traj, cost, usage = asyncio.run(mod.call_sub_model(
        system_prompt="sys", user_message="What is 6*7?",
        image_data_url=None, model="openai/gpt-oss-120b:free",
        output_schema=schema, writer=TrajectoryWriter.noop(),
        budget_tokens=1024, effort="low", sampling=None,
    ))
    assert result == {"answer": "42", "reasoning": "obvious", "confidence": 0.9}
    assert "42" in traj
    assert "thinking thinking" in traj  # reasoning preserved
    assert cost == 0.0
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20

    # request shape verification: forced tool, not response_format
    call = mock_create.await_args
    assert call.kwargs["model"] == "openai/gpt-oss-120b:free"
    assert "response_format" not in call.kwargs
    tools = call.kwargs["tools"]
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "StructuredOutput"
    assert tools[0]["function"]["parameters"] == schema
    assert call.kwargs["tool_choice"] == {"type": "function", "function": {"name": "StructuredOutput"}}
    assert call.kwargs["extra_body"]["reasoning"] == {"effort": "low"}
    assert call.kwargs["extra_body"]["usage"] == {"include": True}


def test_call_sub_model_returns_timed_out_when_no_tool_call(monkeypatch):
    """Free-tier models occasionally ignore forced tool_choice and return reasoning-only.
    Contract: cache as timed_out + continue; do NOT crash the gather. Matches
    base.py:303 wall-clock TimeoutError sentinel shape so worker line 132-135 fires."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    # response with no tool_calls (reasoning-only output, model ignored forced tool_choice)
    msg = MagicMock(content="", tool_calls=None, reasoning="some reasoning trace")
    choice = MagicMock(message=msg, finish_reason=None)
    usage = MagicMock(prompt_tokens=10, completion_tokens=5, cost=0.0)
    fake_resp = MagicMock(choices=[choice], usage=usage)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    from trajectory import TrajectoryWriter
    result, traj, cost, usage_out = asyncio.run(mod.call_sub_model(
        system_prompt="s", user_message="u", image_data_url=None,
        model="openai/gpt-oss-20b:free", output_schema={"type": "object"},
        writer=TrajectoryWriter.noop(),
    ))
    assert result.get("timed_out") is True
    assert result.get("reason") == "no_tool_call"
    assert traj == ""
    assert cost == 0.0


def test_call_sub_model_returns_timed_out_on_rate_limit_error(monkeypatch):
    """Free-tier 429 RateLimitError after SDK exhausts retries should NOT crash
    the gather. Same soft-failure contract — return timed_out sentinel."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    from openai import RateLimitError

    err = RateLimitError(
        message="429 Too Many Requests",
        response=MagicMock(status_code=429, request=MagicMock()),
        body=None,
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=err)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    from trajectory import TrajectoryWriter
    result, traj, cost, usage_out = asyncio.run(mod.call_sub_model(
        system_prompt="s", user_message="u", image_data_url=None,
        model="openai/gpt-oss-20b:free", output_schema={"type": "object"},
        writer=TrajectoryWriter.noop(),
    ))
    assert result.get("timed_out") is True
    assert result.get("reason") == "transient_api_error"
    assert result.get("error_type") == "RateLimitError"
    assert cost == 0.0


def test_call_sub_model_returns_timed_out_on_api_connection_error(monkeypatch):
    """APIConnectionError (TCP reset / DNS hiccup) after retries → timed_out."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    from openai import APIConnectionError

    err = APIConnectionError(request=MagicMock())
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=err)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    from trajectory import TrajectoryWriter
    result, _, _, _ = asyncio.run(mod.call_sub_model(
        system_prompt="s", user_message="u", image_data_url=None,
        model="openai/gpt-oss-20b:free", output_schema={"type": "object"},
        writer=TrajectoryWriter.noop(),
    ))
    assert result.get("timed_out") is True
    assert result.get("reason") == "transient_api_error"
    assert result.get("error_type") == "APIConnectionError"


def test_call_sub_model_returns_timed_out_on_response_body_json_decode_error(monkeypatch):
    """OpenRouter sometimes returns non-JSON HTTP body (HTML error page, truncated
    stream). openai SDK does NOT classify json.JSONDecodeError as retriable, so it
    bubbles bare from httpx._models.json(). Should be treated as transient API
    failure → return timed_out sentinel."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    import json as _json

    err = _json.JSONDecodeError("Expecting value", "garbage body", 0)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=err)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    from trajectory import TrajectoryWriter
    result, _, _, _ = asyncio.run(mod.call_sub_model(
        system_prompt="s", user_message="u", image_data_url=None,
        model="openai/gpt-oss-20b:free", output_schema={"type": "object"},
        writer=TrajectoryWriter.noop(),
    ))
    assert result.get("timed_out") is True
    assert result.get("reason") == "transient_api_error"
    assert result.get("error_type") == "JSONDecodeError"


def test_call_sub_model_returns_timed_out_on_invalid_json_args(monkeypatch):
    """Free-tier models occasionally emit invalid JSON in tool_call.arguments
    (bad escape, truncation). Contract: same as no_tool_call — return timed_out
    sentinel; do NOT crash the gather."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    bad_call = MagicMock()
    bad_call.function.arguments = '{"answer": "broken \\escape"}'
    msg = MagicMock(content="", tool_calls=[bad_call], reasoning="")
    choice = MagicMock(message=msg, finish_reason="tool_calls")
    usage = MagicMock(prompt_tokens=10, completion_tokens=5, cost=0.0)
    fake_resp = MagicMock(choices=[choice], usage=usage)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    from trajectory import TrajectoryWriter
    result, traj, cost, usage_out = asyncio.run(mod.call_sub_model(
        system_prompt="s", user_message="u", image_data_url=None,
        model="openai/gpt-oss-20b:free", output_schema={"type": "object"},
        writer=TrajectoryWriter.noop(),
    ))
    assert result.get("timed_out") is True
    assert result.get("reason") == "invalid_json_in_tool_args"
    assert "json_error" in result


def test_call_sub_model_image_input_raises(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")
    from trajectory import TrajectoryWriter
    with pytest.raises(NotImplementedError, match="image_data_url"):
        asyncio.run(mod.call_sub_model(
            system_prompt="s", user_message="u",
            image_data_url="data:image/png;base64,iVBOR...", model="x",
            output_schema={"type": "object"}, writer=TrajectoryWriter.noop(),
        ))


def _make_mock_tool_call(tool_name: str, args_json: str, call_id: str = "call_1"):
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = args_json
    tc = MagicMock(id=call_id, type="function", function=fn)
    return tc


def _make_mock_completion_with_tool(tool_calls, prompt_tokens=100, completion_tokens=20, cost=0.0):
    msg = MagicMock(content=None, tool_calls=tool_calls)
    choice = MagicMock(message=msg, finish_reason="tool_calls")
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost)
    return MagicMock(choices=[choice], usage=usage)


def test_run_tool_conversation_committed_via_structured_output(monkeypatch):
    """Single-turn: model immediately calls StructuredOutput → exit_reason='committed'."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    so_call = _make_mock_tool_call(
        "StructuredOutput",
        '{"answer": "5", "reasoning": "trivial", "confidence": 1.0}',
    )
    fake_resp = _make_mock_completion_with_tool([so_call], cost=0.0)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    captured: dict = {}
    async def tool_handler(name, args):
        return f"unhandled {name}", False
    def on_so(payload):
        captured["payload"] = payload

    schema = {"type": "object", "properties": {"answer": {"type": "string"}, "reasoning": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["answer", "reasoning", "confidence"]}

    cost, usage, exit_reason = asyncio.run(mod.run_tool_conversation(
        system_prompt="sys", user_message="u", image_data_url=None,
        model="m", tools=[], max_turns=4, tool_handler=tool_handler,
        effort="low", output_format={"type": "json_schema", "schema": schema},
        writer=None, on_structured_output=on_so,
    ))
    assert exit_reason == "committed"
    assert captured["payload"] == {"answer": "5", "reasoning": "trivial", "confidence": 1.0}
    assert cost == 0.0


def test_run_tool_conversation_explore_then_commit(monkeypatch):
    """Two turns: turn 0 calls mock_explore; turn 1 calls StructuredOutput."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    explore_call = _make_mock_tool_call("mock_explore", "{}", call_id="c1")
    so_call = _make_mock_tool_call("StructuredOutput", '{"answer": "42", "reasoning": "via tool", "confidence": 1.0}', call_id="c2")
    resp1 = _make_mock_completion_with_tool([explore_call], cost=0.0)
    resp2 = _make_mock_completion_with_tool([so_call], cost=0.0)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    explore_calls = {"n": 0}
    async def tool_handler(name, args):
        if name == "mock_explore":
            explore_calls["n"] += 1
            return "candidate=42", False
        return "unhandled", False

    captured: dict = {}
    def on_so(payload):
        captured["payload"] = payload

    schema = {"type": "object", "properties": {"answer": {"type": "string"}, "reasoning": {"type": "string"}, "confidence": {"type": "number"}}, "required": ["answer", "reasoning", "confidence"]}
    tools = [{"name": "mock_explore", "description": "produce candidate", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}]

    cost, usage, exit_reason = asyncio.run(mod.run_tool_conversation(
        system_prompt="sys", user_message="u", image_data_url=None,
        model="m", tools=tools, max_turns=4, tool_handler=tool_handler,
        effort=None, output_format={"type": "json_schema", "schema": schema},
        writer=None, on_structured_output=on_so,
    ))
    assert exit_reason == "committed"
    assert explore_calls["n"] == 1
    assert captured["payload"]["answer"] == "42"


def test_run_tool_conversation_cap_exceeded(monkeypatch):
    """max_output_tokens cap fires after the first turn whose cumulative output exceeds the limit."""
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    explore_call = _make_mock_tool_call("mock_explore", "{}")
    fake_resp = _make_mock_completion_with_tool([explore_call], completion_tokens=200, cost=0.0)
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_resp)
    monkeypatch.setattr(mod, "_get_client", lambda: mock_client)

    async def tool_handler(name, args):
        return "ok", False

    cost, usage, exit_reason = asyncio.run(mod.run_tool_conversation(
        system_prompt="sys", user_message="u", image_data_url=None,
        model="m", tools=[{"name": "mock_explore", "description": "x", "parameters": {"type": "object", "properties": {}}}],
        max_turns=10, tool_handler=tool_handler, output_format=None, writer=None,
        on_structured_output=None, max_output_tokens=50,
    ))
    assert exit_reason == "cap_exceeded"
    assert usage["output_tokens"] == 200

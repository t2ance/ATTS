# OpenRouter Real Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `backends/openrouter.py` (currently a thin re-export of `claude.py`) with a real `AsyncOpenAI`-based implementation that talks to `https://openrouter.ai/api/v1` directly, eliminating the $0.0066-per-call hidden Haiku tax incurred when `claude_agent_sdk` is in the call chain.

**Architecture:** Mirror the OpenAI Chat Completions logic from `backends/vllm.py:213-720` (proven solid for tool-calling + StructuredOutput + max_output_tokens cap), drop vllm-specific paths (harmony format, context-overflow soft-skip, multi-port routing, `enable_thinking` template kwarg), and align trajectory writer events with `backends/claude.py`'s NON-streaming form (per-message `write_text` for content, `write_tool_use` per call, `write_tool_result` per response). Cost is read from OpenRouter's `usage.cost` field (real per-call USD), not fabricated.

**Tech Stack:** Python 3.11 (`explain` conda env), `openai` AsyncOpenAI client (already used by vllm.py), `pytest` + `pytest-asyncio` for unit tests, real OpenRouter free model (`openai/gpt-oss-120b:free`) for integration smoke.

---

## Scope

In scope:
- Single-shot structured query path (`call_sub_model`).
- Multi-turn tool-calling conversation path (`run_tool_conversation`).
- Three benchmarks: HLE, LCB, GPQA (text-only).
- Server-side `tools` + `tool_choice="auto"` + StructuredOutput-as-fake-function-tool.

Out of scope (deferred):
- Multimodal `image_data_url` support (BabyVision / RBenchV) — raise `NotImplementedError` at the boundary.
- Identifying minimum subset of the 11 DISABLE flags previously added to `backends/claude.py` (those don't suppress the Haiku tax in any case; the new backend bypasses claude.py entirely).
- Per-provider unified error handling beyond the basic `httpx`/`openai` retry loop.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `Experiment/core_code/backends/openrouter.py` | REPLACE | Real implementation (~350 lines). Drop the current 51-line thin alias. |
| `Experiment/core_code/tests/test_openrouter_backend.py` | CREATE | Unit tests with mocked `AsyncOpenAI`. Validates contract: signatures, return tuples, `exit_reason` set, writer event sequence, cost reading. |
| `Experiment/core_code/tests/openrouter_via_claude_smoke.py` | MODIFY | Add `--probe a_via_openrouter` and `--probe b_via_openrouter` modes that route through the new `backends.openrouter` (not via claude.py). Reuse the existing `probe_c` discriminator. |
| `Experiment/core_code/backends/claude.py` | MODIFY | REVERT the 11-flag setdefault block (lines 17-50) — proven non-load-bearing on 2026-05-03. Add a retraction comment pointing to this plan. |
| `Experiment/core_code/todo_openrouter_via_claude.md` | APPEND | Phase 4 section with γ retraction notice + α completion link. |
| `Experiment/core_code/tests/openrouter_e2e_precache.yaml` | NO CHANGE | Already references `backend: openrouter`; the new openrouter.py is a drop-in replacement. |

## Pre-flight (before Task 1)

- [ ] Verify `OPENROUTER_API_KEY` is set in shell: `echo "len=${#OPENROUTER_API_KEY}"` should print `len=73` or similar.
- [ ] Verify clean workspace: `cd /data3/peijia/dr-claw/Explain && git status --short` — capture current state; we will commit incrementally.
- [ ] Verify `pytest` + `pytest-asyncio` available in `explain` env: `conda run -n explain python -c "import pytest, pytest_asyncio; print(pytest.__version__, pytest_asyncio.__version__)"` should succeed.

If `pytest-asyncio` is missing, install it via `conda run -n explain pip install pytest-asyncio` (cost: $0, local).

---

### Task 1: openrouter.py Skeleton + Constants + Auth Assertion

**Files:**
- Create: `Experiment/core_code/backends/openrouter.py` (this OVERWRITES the existing 51-line thin alias — do not preserve any of the old content)
- Test: `Experiment/core_code/tests/test_openrouter_backend.py` (new file)

- [ ] **Step 1: Write the failing test** (new file `tests/test_openrouter_backend.py`)

```python
"""Unit tests for backends/openrouter.py contract.

Mocked AsyncOpenAI; no live network calls. Run via:
    cd /data3/peijia/dr-claw/Explain/Experiment/core_code
    conda run -n explain pytest tests/test_openrouter_backend.py -v
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_module_imports_and_asserts_api_key(monkeypatch):
    """Module-level assert OPENROUTER_API_KEY in env."""
    # ensure module is fresh (drop cached import)
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(AssertionError, match="OPENROUTER_API_KEY"):
        importlib.import_module("backends.openrouter")


def test_module_imports_with_api_key_set(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-fake-token-do-not-use")
    mod = importlib.import_module("backends.openrouter")
    assert hasattr(mod, "call_sub_model")
    assert hasattr(mod, "run_tool_conversation")
    assert mod.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py::test_module_imports_and_asserts_api_key -v`
Expected: FAIL — current openrouter.py is the thin alias and may not raise the same assertion message.

- [ ] **Step 3: Write minimal implementation** (overwrite `backends/openrouter.py`)

```python
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

import json
import logging
import os
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

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
        )
    return _client
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: BOTH `test_module_imports_and_asserts_api_key` AND `test_module_imports_with_api_key_set` PASS.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/backends/openrouter.py Experiment/core_code/tests/test_openrouter_backend.py
git commit -m "feat(openrouter): replace thin alias with skeleton + auth assertion"
```

---

### Task 2: _split_sampling_kwargs Helper

**Files:**
- Modify: `Experiment/core_code/backends/openrouter.py` (add helper after `_get_client`)
- Test: `Experiment/core_code/tests/test_openrouter_backend.py` (append test)

- [ ] **Step 1: Append test for _split_sampling_kwargs**

Append to `tests/test_openrouter_backend.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: 3 new tests FAIL with `AttributeError: module 'backends.openrouter' has no attribute '_split_sampling_kwargs'`.

- [ ] **Step 3: Add _split_sampling_kwargs to openrouter.py**

Append to `backends/openrouter.py` (after `_get_client`):

```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/backends/openrouter.py Experiment/core_code/tests/test_openrouter_backend.py
git commit -m "feat(openrouter): add _split_sampling_kwargs helper"
```

---

### Task 3: call_sub_model — Single Structured Query

**Files:**
- Modify: `Experiment/core_code/backends/openrouter.py` (add `call_sub_model` async function)
- Test: `Experiment/core_code/tests/test_openrouter_backend.py` (append mock-based unit test)

- [ ] **Step 1: Append failing unit test (mocked OpenAI)**

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock


def _make_mock_completion(json_content: str, prompt_tokens: int = 100, completion_tokens: int = 20, cost: float = 0.0):
    msg = MagicMock(content=json_content, tool_calls=None)
    choice = MagicMock(message=msg, finish_reason="stop")
    usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, cost=cost)
    return MagicMock(choices=[choice], usage=usage)


def test_call_sub_model_returns_parsed_json(monkeypatch):
    sys.modules.pop("backends.openrouter", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    mod = importlib.import_module("backends.openrouter")

    fake_resp = _make_mock_completion('{"answer": "42", "reasoning": "obvious", "confidence": 0.9}', cost=0.0)
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
    assert cost == 0.0
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 20

    # request shape verification: response_format json_schema, reasoning effort
    call = mock_create.await_args
    assert call.kwargs["model"] == "openai/gpt-oss-120b:free"
    rf = call.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"] == schema
    assert call.kwargs["extra_body"]["reasoning"] == {"effort": "low"}
    assert call.kwargs["extra_body"]["usage"] == {"include": True}


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
```

- [ ] **Step 2: Run tests, expect both new ones FAIL**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: 2 new tests fail with `AttributeError: ... has no attribute 'call_sub_model'`.

- [ ] **Step 3: Implement call_sub_model**

Append to `backends/openrouter.py`:

```python
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
    """Single structured query via OpenRouter chat/completions + json_schema.

    Server-side constrained decoding guarantees `message.content` is valid JSON
    matching `output_schema`. We parse it directly; no post-hoc retry / repair.

    `effort`: passed via `extra_body={"reasoning": {"effort": effort}}` —
    OpenRouter's unified knob that translates to each upstream provider's
    native thinking parameter (Anthropic `thinking.budget_tokens`, OpenAI
    `reasoning.effort`, Google `thinkingConfig`).

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
    # Request OpenRouter to populate `usage.cost` (real billed USD per call).
    # See https://openrouter.ai/docs/use-cases/usage-accounting.
    extra_body["usage"] = {"include": True}

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_output",
            "schema": output_schema,
            "strict": True,
        },
    }

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format=response_format,
        extra_body=extra_body,
        **direct_kwargs,
    )

    text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
    # OpenRouter native cost (real USD), not SDK-fabricated. Falls back to 0.0
    # if the provider didn't surface cost (e.g. older OpenRouter API tier).
    cost = float(getattr(response.usage, "cost", 0.0) or 0.0)

    result = json.loads(text)

    if writer:
        writer.write_text(text)
        writer.close()

    return result, text, cost, usage
```

- [ ] **Step 4: Run tests, expect pass**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: 7 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/backends/openrouter.py Experiment/core_code/tests/test_openrouter_backend.py
git commit -m "feat(openrouter): implement call_sub_model with json_schema response_format"
```

---

### Task 4: run_tool_conversation — Multi-Turn Tool Loop

**Files:**
- Modify: `Experiment/core_code/backends/openrouter.py` (add `run_tool_conversation` async function)
- Test: `Experiment/core_code/tests/test_openrouter_backend.py` (append three mock-based tests)

- [ ] **Step 1: Append three failing tests**

```python
def _make_mock_tool_call(tool_name: str, args_json: str, call_id: str = "call_1"):
    fn = MagicMock(name="function", function=MagicMock(name=tool_name, arguments=args_json))
    fn.function.name = tool_name
    fn.id = call_id
    fn.type = "function"
    return fn


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

    captured = {}
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

    captured = {}
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
```

- [ ] **Step 2: Run tests, expect 3 new failures**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: 3 new failures (`AttributeError: ... has no attribute 'run_tool_conversation'`).

- [ ] **Step 3: Implement run_tool_conversation**

Append to `backends/openrouter.py`:

```python
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

    for turn in range(max_turns):
        response = await _get_client().chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools or None,
            tool_choice="auto" if openai_tools else None,
            extra_body=extra_body or None,
            **direct_kwargs,
        )

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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/test_openrouter_backend.py -v`
Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/backends/openrouter.py Experiment/core_code/tests/test_openrouter_backend.py
git commit -m "feat(openrouter): implement run_tool_conversation with tool_calls + StructuredOutput"
```

---

### Task 5: Live Smoke — Probe A on Real Free Model

**Files:**
- Modify: `Experiment/core_code/tests/openrouter_via_claude_smoke.py` (add `--probe a_via_openrouter`)

**Note:** This task incurs ZERO cost (free model, real backend bypasses claude_agent_sdk so no Haiku tax). However, it requires network. Run only after Tasks 1-4 are committed.

- [ ] **Step 1: Add new probe choice in smoke harness**

In `tests/openrouter_via_claude_smoke.py`, find the `--probe` argparse choices line and the dispatch block in `main()`. Add `a_via_openrouter` to choices and a new branch in dispatch.

Locate `ap.add_argument("--probe", required=True, choices=["a", "b", "c", "a_via_dispatcher"], ...)` and change `choices` to:

```python
ap.add_argument("--probe", required=True,
                choices=["a", "b", "c", "a_via_dispatcher", "a_via_openrouter", "b_via_openrouter"],
                help="a/b/c=via claude.py and OpenAI direct; a_via_dispatcher=legacy thin-alias path; "
                     "a_via_openrouter / b_via_openrouter = real backends.openrouter (no claude_agent_sdk)")
```

In `main()`, before the `else` branch, add:

```python
    elif probe == "a_via_openrouter":
        # End-to-end: real backends.openrouter call_sub_model.
        from importlib import import_module
        backend_mod = import_module("backends.openrouter")
        from trajectory import TrajectoryWriter
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
    elif probe == "b_via_openrouter":
        # End-to-end: real backends.openrouter run_tool_conversation.
        from importlib import import_module
        backend_mod = import_module("backends.openrouter")
        from trajectory import TrajectoryWriter
        obs = {"path": "B_via_openrouter", "model": model}
        t0 = time.time()
        explore_calls = {"n": 0}
        structured_payload = {"value": None}

        async def tool_handler(name, args):
            if name == "mock_explore":
                explore_calls["n"] += 1
                return "Explore returned: candidate answer 391, reasoning: 17*20+17*3=391.", False
            return f"Unknown tool: {name}", False

        def on_structured_output(payload):
            structured_payload["value"] = payload

        tools = [{"name": "mock_explore",
                  "description": "Dispatch a fresh solver to produce a candidate answer. Takes no parameters.",
                  "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}]
        try:
            cost, usage, exit_reason = await backend_mod.run_tool_conversation(
                system_prompt=SYSTEM_PROMPT_B, user_message=USER_MSG_B,
                image_data_url=None, model=model, tools=tools, max_turns=4,
                tool_handler=tool_handler, effort="low",
                output_format={"type": "json_schema", "schema": SIMPLE_SCHEMA},
                writer=TrajectoryWriter.noop(),
                on_structured_output=on_structured_output,
            )
            sp = structured_payload["value"] or {}
            obs.update(
                ok=True, duration=round(time.time() - t0, 2),
                exit_reason=exit_reason, explore_calls=explore_calls["n"],
                structured_output_fired=structured_payload["value"] is not None,
                structured_payload_keys=list(sp.keys()) if sp else [],
                answer=str(sp.get("answer", ""))[:80] if sp else None,
                cost_usd=cost,
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            )
        except Exception as e:
            obs.update(ok=False, duration=round(time.time() - t0, 2),
                       exit_reason=None, explore_calls=explore_calls["n"],
                       structured_output_fired=structured_payload["value"] is not None,
                       error_type=type(e).__name__, error=str(e)[:500])
```

Update the verdict-tag selection at the bottom of `main()`:

```python
    passed = obs.get("ok") and (
        obs.get("structured_output_fired") if probe in ("a", "b", "a_via_dispatcher", "a_via_openrouter", "b_via_openrouter") else
        obs.get("tool_call_correct") if probe == "c" else False
    )
```

- [ ] **Step 2: Sanity-check the harness compiles**

Run: `cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain python -c "import ast; ast.parse(open('tests/openrouter_via_claude_smoke.py').read())"`
Expected: no output (parse OK).

- [ ] **Step 3: Run probe a_via_openrouter against gpt-oss-120b:free (live, free)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
TS=$(date +%Y%m%d_%H%M%S)
LOG=tests/logs/path_a_via_openrouter_${TS}.log
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python tests/openrouter_via_claude_smoke.py --probe a_via_openrouter --model openai/gpt-oss-120b:free > "$LOG" 2>&1 &
echo "PID=$!  LOG=$LOG"
```

Wait until process exits (~5-30s typical). Inspect log:

```bash
cat /data3/peijia/dr-claw/Explain/Experiment/core_code/tests/logs/path_a_via_openrouter_*.log | tail -30
```

Expected: `[PASS] probe=A_via_openrouter` AND `"answer": "391"` AND `"cost_usd": 0` (or very small).

- [ ] **Step 4: Run probe b_via_openrouter (live, free)**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
TS=$(date +%Y%m%d_%H%M%S)
LOG=tests/logs/path_b_via_openrouter_${TS}.log
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python tests/openrouter_via_claude_smoke.py --probe b_via_openrouter --model openai/gpt-oss-120b:free > "$LOG" 2>&1 &
echo "PID=$!  LOG=$LOG"
```

Expected: `[PASS] probe=B_via_openrouter`, `"exit_reason": "committed"`, `"explore_calls": 1`, `"structured_payload_keys": ["answer", "reasoning", "confidence"]`, `"cost_usd": 0`.

- [ ] **Step 5: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/tests/openrouter_via_claude_smoke.py Experiment/core_code/tests/logs/
git commit -m "test(openrouter): add a/b_via_openrouter probes; verified gpt-oss-120b:free passes both"
```

---

### Task 6: Cost Verification — OpenRouter `usage.cost` is the Authoritative Field

**Files:**
- (No source change — this is a verification-only task to confirm cost reporting is correct)

The new `call_sub_model` reads `response.usage.cost`. Need to confirm OpenRouter returns this when we set `extra_body={"usage": {"include": True}}`.

- [ ] **Step 1: Probe and inspect raw response**

Create temporary script `tests/inspect_openrouter_usage.py`:

```python
"""One-off inspector — confirms OpenRouter usage.cost field is populated."""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
    resp = await client.chat.completions.create(
        model="openai/gpt-oss-120b:free",
        messages=[{"role": "user", "content": "Reply: pong"}],
        max_tokens=8,
        extra_body={"usage": {"include": True}},
    )
    print(f"prompt_tokens   = {resp.usage.prompt_tokens}")
    print(f"completion_tokens= {resp.usage.completion_tokens}")
    print(f"cost            = {getattr(resp.usage, 'cost', None)!r}")
    print(f"is_byok         = {getattr(resp.usage, 'is_byok', None)!r}")
    print(f"all attrs       = {[a for a in dir(resp.usage) if not a.startswith('_')]}")


asyncio.run(main())
```

Run:

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain python tests/inspect_openrouter_usage.py
```

Expected: `cost = 0.0` (or `0`) for free model, plus tokens > 0. Note attribute names — if OpenRouter SDK exposes the field differently, adjust `call_sub_model` accordingly.

- [ ] **Step 2: If `cost` attribute is named differently, fix call_sub_model**

If output shows e.g. `cost = None` (attribute exists but unset), the current `or 0.0` fallback is correct.
If attribute is named differently (e.g. `total_cost`), update both `call_sub_model` and `run_tool_conversation` to use the correct name. Re-run unit tests.

- [ ] **Step 3: Delete inspector (do not commit it)**

```bash
rm /data3/peijia/dr-claw/Explain/Experiment/core_code/tests/inspect_openrouter_usage.py
```

(No commit — the inspector is throwaway.)

---

### Task 7: Revert claude.py 11-flag Block

**Files:**
- Modify: `Experiment/core_code/backends/claude.py` (lines 17-50: revert the DISABLE setdefault block)

The 11 flags do NOT suppress the Haiku tax. Adding them to claude.py is dead weight. Revert.

- [ ] **Step 1: Show the current block**

```bash
sed -n '15,55p' /data3/peijia/dr-claw/Explain/Experiment/core_code/backends/claude.py
```

Confirm it includes the `os.environ.setdefault(_flag, "1")` loop and the `for _flag in (...)` block.

- [ ] **Step 2: Apply the revert via Edit tool**

Open `backends/claude.py`. Replace the entire block (from `import os` line through `del _flag` line) with a brief retraction comment:

```python
# (Removed 2026-05-03: an earlier commit added a setdefault block for 11
# CLAUDE_CODE_DISABLE_*/DISABLE_* env vars in an attempt to suppress a
# hidden Anthropic Haiku 1-token classifier call billed at ~$0.0066 per
# claude_agent_sdk invocation. Empirically those flags do NOT suppress the
# call; the original "$0 delta" measurements were ledger-lag artifacts.
# The Haiku tax is intrinsic to the SDK's CLI subprocess and cannot be
# turned off via env. Use backends/openrouter.py for cost-zero free-tier
# routing instead. See docs/superpowers/plans/2026-05-03-openrouter-real-backend.md.)
```

(Keep `import os` — there are other uses elsewhere in the file.)

- [ ] **Step 3: Verify file still parses + test claude.py basic shape**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
conda run -n explain python -c "import ast; ast.parse(open('backends/claude.py').read()); from backends import claude; print(claude.call_sub_model)"
```

Expected: prints `<function call_sub_model at 0x...>` — module loads, function still exported.

- [ ] **Step 4: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/backends/claude.py
git commit -m "revert(claude): remove non-load-bearing DISABLE flag setdefault block"
```

---

### Task 8: precache_explores.py End-to-End Smoke (Real Backend)

**Files:**
- (No source change — uses existing `tests/openrouter_e2e_precache.yaml` unchanged)

- [ ] **Step 1: Wipe old smoke cache (it was generated by thin-alias openrouter.py)**

```bash
rm -rf /data3/peijia/dr-claw/Explain/Experiment/core_code/tests/cache/openrouter_e2e_smoke
```

- [ ] **Step 2: Capture pre-call OpenRouter usage baseline**

```bash
eval "$(grep -E '^[[:space:]]*export[[:space:]]+OPENROUTER_API_KEY=' ~/.bashrc)"  # pull current key from .bashrc
USAGE_BEFORE=$(curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])")
echo "USAGE_BEFORE=$USAGE_BEFORE"
```

- [ ] **Step 3: Run precache smoke with real backend**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
TS=$(date +%Y%m%d_%H%M%S)
LOG=tests/logs/precache_e2e_real_${TS}.log
PYTHONUNBUFFERED=1 nohup conda run -n explain --no-capture-output python precache_explores.py --config tests/openrouter_e2e_precache.yaml > "$LOG" 2>&1 &
echo "PID=$!  LOG=$LOG"
```

Wait for completion (~20-30s).

- [ ] **Step 4: Verify cache files written**

```bash
find /data3/peijia/dr-claw/Explain/Experiment/core_code/tests/cache/openrouter_e2e_smoke -name 'result.json'
```

Expected: one path under `gpqa/<qid>/explore_1/result.json`.

```bash
RESULT=$(find /data3/peijia/dr-claw/Explain/Experiment/core_code/tests/cache/openrouter_e2e_smoke -name 'result.json' | head -1)
python3 -c "import json; d=json.load(open('$RESULT')); print('answer:', d.get('answer')); print('cost_usd:', d.get('cost_usd')); print('model:', d.get('model'))"
```

Expected: `answer` is one of A-D, `cost_usd` is 0 or near-zero, `model` is `openai/gpt-oss-120b:free`.

- [ ] **Step 5: Verify OpenRouter delta — wait 60 seconds for ledger to settle**

The previous γ-failure incident showed OpenRouter `/auth/key.usage` lags by ~30-60s. Sleep, then measure.

```bash
sleep 60
USAGE_AFTER=$(curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['usage'])")
python3 -c "
before = $USAGE_BEFORE
after = $USAGE_AFTER
delta = after - before
print(f'BEFORE: \${before}')
print(f'AFTER : \${after}')
print(f'DELTA : \${delta:.7f}')
print('PASS' if delta < 0.001 else 'FAIL — non-zero delta on free model')
"
```

Expected: PASS — delta < $0.001 (target is $0; non-zero would mean OpenRouter served a paid fallback model OR some other unforeseen tax).

- [ ] **Step 6: Commit (logs only; no source change)**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/tests/logs/precache_e2e_real_*.log
git commit -m "test(openrouter): live precache e2e on gpt-oss-120b:free passes with $0 delta"
```

---

### Task 9: Update todo_openrouter_via_claude.md with α-Path Outcome

**Files:**
- Modify: `Experiment/core_code/todo_openrouter_via_claude.md` (append a Phase 4 section)

- [ ] **Step 1: Append Phase 4 retraction + completion record**

At the END of `todo_openrouter_via_claude.md`, append:

```markdown
## Phase 4 — γ retraction + α completion (2026-05-03)

### γ verdict retraction

Items 03/04 (probe A) and 09 (probe B) on `openai/gpt-oss-120b:free`, plus the
e2e dispatcher and precache smokes earlier in this file, all reported
$0.0000000 OpenRouter usage delta. Subsequent investigation (timeline
reconstruction across 13 known calls in the same session) showed every
delta-zero measurement was a ledger-lag artifact: OpenRouter's `/auth/key.usage`
field updates ~30-60 seconds after a call settles. By taking
`baseline → after` measurements within that lag window, the Haiku tax was
silently deferred to the next baseline measurement, where it appeared as if it
had come from another call. Total reconstructed Haiku tax: ~13 × $0.0066 =
$0.086, which closely matches the observed lifetime delta of $0.0798 over the
session. The 11 `CLAUDE_CODE_DISABLE_*` / `DISABLE_*` env vars added to
backends/claude.py to suppress this tax were therefore non-load-bearing.

The verdict in this file's VIABLE block (which claimed claude.py reuse was
viable for cost-zero free-tier routing) is RETRACTED.

### α path taken instead

`backends/openrouter.py` was rewritten as a real `AsyncOpenAI`-based
implementation that bypasses `claude_agent_sdk` entirely. By construction it
cannot incur the Haiku tax (no `claude` CLI subprocess in the call chain).

Implementation plan: `docs/superpowers/plans/2026-05-03-openrouter-real-backend.md`
Final code: `backends/openrouter.py` (real AsyncOpenAI implementation, ~350 lines)
Validation: 10 unit tests + live probe a_via_openrouter + live probe b_via_openrouter + precache e2e, all $0 delta after a 60-second ledger settle window.

### Updated VIABLE block

```
VERDICT = VIABLE (via real backends/openrouter.py, NOT via claude.py reuse)
WORKING_MODEL = "openai/gpt-oss-120b:free"
ENV (auto-injected by backends/openrouter.py at module load time):
  base_url   = https://openrouter.ai/api/v1
  api_key    = $OPENROUTER_API_KEY    # caller exports this only
  extra_body usage.include = True     # OpenRouter returns real per-call cost

USAGE (yaml):
  backend:
    name: openrouter
  ...

KNOWN COUPLINGS / RISKS:
  - image_data_url not supported (raise NotImplementedError). Text-only scope
    HLE / LCB / GPQA. BabyVision needs a follow-up to add multimodal content
    blocks.
  - Sampling is vllm-style (full block accepted), not claude-style (None-only
    assert). If a future caller relies on the assert behavior, document the
    departure.
```
```

- [ ] **Step 2: Commit**

```bash
cd /data3/peijia/dr-claw/Explain && git add Experiment/core_code/todo_openrouter_via_claude.md
git commit -m "docs(todo): retract gamma verdict; record alpha path completion"
```

---

### Task 10: Final Sanity — All Existing Tests Still Pass

**Files:**
- (No source change)

- [ ] **Step 1: Run the full backend unit test suite**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain pytest tests/ -v --ignore=tests/cache --ignore=tests/logs -x
```

Expected: all collected tests pass; no regressions in pre-existing tests (`test_benchmark_grade.py`, `test_eval_config.py`, etc.).

- [ ] **Step 2: Run static syntax check on the 4 backends modules**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code
for B in claude.py codex.py vllm.py openrouter.py; do
    conda run -n explain python -c "import ast; ast.parse(open('backends/$B').read()); print('$B OK')"
done
```

Expected: 4 lines of `<file> OK`.

- [ ] **Step 3: Confirm no orphan import paths**

```bash
cd /data3/peijia/dr-claw/Explain/Experiment/core_code && conda run -n explain python -c "
import os
os.environ.setdefault('OPENROUTER_API_KEY', 'sk-or-v1-test')
from importlib import import_module
for b in ('claude', 'codex', 'vllm', 'openrouter'):
    m = import_module(f'backends.{b}')
    print(f'{b}.call_sub_model:', m.call_sub_model)
    print(f'{b}.run_tool_conversation:', m.run_tool_conversation)
"
```

Expected: 4 backend modules each show both function objects.

- [ ] **Step 4: No commit needed (read-only verification)**

If all three checks pass, the implementation is complete. If any fails, stop and report.

---

## Self-Review Checklist (run after writing — completed inline)

**1. Spec coverage:**
- ✓ `call_sub_model` (signature, return tuple) — Task 3
- ✓ `run_tool_conversation` (signature, exit_reason) — Task 4
- ✓ Server-side tool_calls (mechanism A) — Task 4
- ✓ StructuredOutput injection — Task 4 (line 537-545 of impl)
- ✓ OpenRouter `usage.cost` reading — Task 3 + Task 6 verification
- ✓ Skip image_data_url — Task 3 (NotImplementedError)
- ✓ IO align with claude.py (write_text per content / write_tool_use / write_tool_result) — Task 4 (no streaming chunks; per-message write_text)
- ✓ Reuse vllm.py logic — Task 4 (mirror :526-720 minus harmony / context_overflow)
- ✓ Cache file structure unchanged — Task 8 (uses methods/base.py:save_sub_model_result, no openrouter-specific fields)
- ✓ Revert claude.py 11-flag block — Task 7
- ✓ TODO retraction — Task 9

**2. Placeholder scan:** No "TBD", "implement later", "similar to Task N", "etc." — every step has concrete code or commands.

**3. Type consistency:**
- `call_sub_model` returns `tuple[dict, str, float, dict]` — used identically in Task 3 test and impl.
- `run_tool_conversation` returns `tuple[float, dict, str]` — used identically in Task 4 tests and impl.
- `exit_reason` set: `"committed" | "cap_exceeded" | "incomplete"` — exactly what Task 4 implements.
- Tool definition shape: `{"name": str, "description": str, "parameters": dict}` — consistent across tests and impl (matches existing vllm.py contract).

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-openrouter-real-backend.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch with checkpoints for review.

Which approach?

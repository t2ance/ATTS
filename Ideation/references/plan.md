# LangGraph TTS Agent Rewrite Plan

Branch: `peijia-langgraph`

## Background

Rewrite `tts-agent/` in-place to use **LangGraph** for control flow, removing direct mode and keeping only delegated mode. Preserve ALL existing behavior: thinking trajectories, conversation history, cost tracking, trajectory.md format, caching, evaluation.

### API Backend Options (Research Findings)

We investigated 4 possible API backends. Summary:

#### Option 1: CRS Claude (OpenAI-compatible proxy)

CRS exposes an OpenAI-compatible endpoint at `http://127.0.0.1:3456/openai/claude/v1` that relays to `api.anthropic.com/v1/messages` using Claude Max OAuth credentials.

Tested and working: text generation, multimodal, temperature/top_p, tool calling, custom system prompt, extended thinking passthrough.

**Problem**: Anthropic banned third-party OAuth usage in Jan 2026. They deployed server-side client fingerprinting -- non-official clients get: "This credential is only authorized for use with Claude Code". CRS currently works (likely spoofing headers) but could be blocked at any time. Anthropic also sent legal letters to projects like OpenCode.

Sources:
- https://awesomeagents.ai/news/claude-code-oauth-policy-third-party-crackdown/
- https://daveswift.com/claude-oauth-update/
- https://news.ycombinator.com/item?id=47069299

#### Option 2: ChatAnthropic (LangChain direct to Anthropic API)

`langchain-anthropic` ChatAnthropic natively supports extended thinking. The `anthropic` Python SDK supports OAuth via `auth_token=` parameter.

**Problem 1**: ChatAnthropic only accepts `api_key`, not `auth_token`. It sends the token as `x-api-key` header, but OAuth tokens need `Authorization: Bearer` header.
**Problem 2**: Same Anthropic OAuth ban applies -- direct API calls with OAuth tokens are blocked.

#### Option 3: Claude Agent SDK (current approach)

`claude-agent-sdk` wraps the official Claude Code CLI binary. Always works with Max subscription (official client, not blocked).

**Problem**: Limited control -- MCP server pattern for tools, structured output via SDK's `output_format`, no easy way to extract thinking content programmatically. This is what we're trying to move away from.

#### Option 4: Gemini CLI OAuth → Cloud Code Assist API (TESTED, WORKING)

Google Gemini CLI stores OAuth tokens in `~/.gemini/oauth_creds.json`. Uses Google personal OAuth (Login with Google), not Gemini API key.

**Endpoint**: `POST https://cloudcode-pa.googleapis.com/v1internal:{generateContent,streamGenerateContent}`
**Auth**: `Authorization: Bearer <access_token>` (standard Google OAuth2 token)
**Format**: Vertex AI / Gemini API format (generateContent with contents/parts)
**Token refresh**: `POST https://oauth2.googleapis.com/token` with client_id, client_secret, refresh_token

OAuth credentials:
- Client ID: `681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com`
- Client Secret: `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`
- Scopes: `cloud-platform`, `userinfo.profile`, `userinfo.email`

Setup requires calling `loadCodeAssist` first to get project ID (returns `cloudaicompanionProject` field).

Tested models:

| Model | Status | Notes |
|-------|--------|-------|
| `gemini-2.5-flash` | OK | Default fast model |
| `gemini-2.5-pro` | OK | Deeper thinking (2144 tokens vs flash 143) |
| `gemini-3.1-pro-preview` | OK | Latest reasoning model |
| `gemini-3-flash-preview` | Not tested | Should work |

User has **Google One AI Pro** tier (`g1-pro-tier`).

**Parameter support (all working)**:

| Parameter | Status |
|---|---|
| `temperature` | OK |
| `maxOutputTokens` | OK |
| `topP` | OK |
| `thinkingConfig.thinkingBudget` | OK |
| `thinkingConfig.includeThoughts` | **OK -- thinking content fully visible** |
| `tools` (function calling) | OK |
| `systemInstruction` | OK |
| Streaming (SSE via `?alt=sse`) | OK |

**Key advantage over Codex**: thinking/reasoning content is **fully visible** with `includeThoughts: true`. This means `output.md` trajectory files can contain the full chain-of-thought, matching the current Claude extended thinking behavior.

**Disadvantages**:
- Rate limiting is aggressive: ~1 request per 50 seconds on flash, resets after quota exhaustion
- `cloudcode-pa.googleapis.com/v1internal` is an internal API, not the standard `generativelanguage.googleapis.com` -- LangChain's `ChatGoogleGenerativeAI` cannot be used directly (needs custom wrapper)
- `gemini-3-pro-preview` is deprecated March 9, 2026; use `gemini-3.1-pro-preview` instead

**Request format**:

``json
{
  "model": "gemini-2.5-flash",
  "project": "<cloudaicompanionProject>",
  "request": {
    "contents": [{"role": "user", "parts": [{"text": "..."}]}],
    "systemInstruction": {"role": "user", "parts": [{"text": "system prompt"}]},
    "tools": [{"functionDeclarations": [...]}],
    "generationConfig": {
      "temperature": 0.7,
      "maxOutputTokens": 4096,
      "thinkingConfig": {
        "thinkingBudget": 8192,
        "includeThoughts": true
      }
    }
  }
}
``

**Response format (non-streaming)**:

``json
{
  "response": {
    "candidates": [{
      "content": {
        "role": "model",
        "parts": [
          {"thought": true, "text": "thinking content..."},
          {"text": "answer content..."}
        ]
      },
      "finishReason": "STOP"
    }],
    "usageMetadata": {
      "promptTokenCount": 13,
      "candidatesTokenCount": 100,
      "totalTokenCount": 500,
      "thoughtsTokenCount": 400
    }
  }
}
``

**Streaming response** (SSE via `?alt=sse`):
Each `data:` line contains a partial response with the same structure. Thought parts have `"thought": true` flag. Tool calls arrive as `functionCall` parts with `thoughtSignature` for verification.

Sources:
- Gemini CLI source: `@google/gemini-cli-core/dist/src/code_assist/server.js` (CODE_ASSIST_ENDPOINT constant)
- Gemini CLI models: `@google/gemini-cli-core/dist/src/config/models.js`
- CLIProxyAPI OAuth scope issue: https://github.com/router-for-me/CLIProxyAPI/issues/637
- Gemini 3.1 Pro announcement: https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/
- Gemini 3 on Gemini CLI: https://geminicli.com/docs/get-started/gemini-3/
- Gemini 3 Pro deprecation notice: https://discuss.ai.google.dev/t/migrate-from-gemini-3-pro-preview-to-gemini-3-1-pro-preview-before-march-9-2026/127062

#### Option 5: Codex OAuth → ChatGPT Backend API (TESTED, WORKING)

OpenAI Codex CLI stores OAuth tokens in `~/.codex/auth.json`. Unlike Anthropic, **OpenAI actively supports third-party usage** -- they helped OpenCode add ChatGPT Plus support after Anthropic's ban.

**Endpoint**: `POST https://chatgpt.com/backend-api/codex/responses`
**Auth**: `Authorization: Bearer <access_token>` + `chatgpt-account-id: <account_id>`
**Format**: OpenAI Responses API (not Chat Completions API)

Tested models (ChatGPT Plus account):

| Model | Status |
|-------|--------|
| `gpt-5.2` | OK |
| `gpt-5` | OK |
| `gpt-5.1-codex` | OK |
| `gpt-5.1-codex-mini` | OK |
| `gpt-5.3-codex` | OK |
| `gpt-4.1`, `gpt-4o`, `o3`, `o4-mini` | BLOCKED |

**Constraints**:
- Must use `stream: true`, `store: false`
- `input` must be a list (not string)
- System prompt goes in `instructions` field (not in messages)
- Only Responses API format, not Chat Completions -- LangChain `ChatOpenAI` cannot be used directly

**Responses API vs Chat Completions API**:
- Chat Completions: `messages` array, `{"role": "system", ...}`, single JSON response
- Responses API: `input` array + `instructions` field, SSE streaming events (`response.output_text.delta`, etc.)
- Same capabilities (text, tool calling), different wire format

Token refresh: `~/.codex/auth.json` contains `refresh_token`. Codex CLI handles refresh automatically. For programmatic use, need ~20 lines of refresh logic (POST to OpenAI token endpoint).

### Codex OAuth Endpoint & Parameter Limitations (Tested)

**Key finding**: Codex OAuth token is restricted to `chatgpt.com/backend-api/codex/responses` endpoint only. The same models (gpt-5.2, gpt-5.2-codex) are available on the standard `api.openai.com/v1/responses` endpoint with an API key, but Codex OAuth tokens return "insufficient permissions" when used against `api.openai.com`.

| Endpoint | Auth | gpt-5.2 | gpt-5.3-codex | temperature | max_output_tokens |
|---|---|---|---|---|---|
| `chatgpt.com/backend-api/codex/responses` | OAuth | OK | OK | BLOCKED | BLOCKED |
| `api.openai.com/v1/responses` | API key | OK | 404 | OK (reasoning=none) | OK |
| `api.openai.com/v1/responses` | OAuth | "insufficient permissions" | - | - | - |

**Parameter support on chatgpt.com backend**:

| Parameter | Status | Notes |
|---|---|---|
| `reasoning.effort` | OK | low/medium/high |
| `tools` (function calling) | OK | single/multi tool, forced tool_choice |
| `text.format` (structured output) | OK | json_schema works |
| base64 images | OK | multimodal input |
| concurrent requests | OK | tested 3 concurrent |
| `reasoning.summary` | Partial | Returns brief label (e.g. "Preparing proof by contradiction"), not full reasoning |
| `temperature` | BLOCKED | "Unsupported parameter" |
| `max_output_tokens` | BLOCKED | "Unsupported parameter" |
| `top_p` | BLOCKED | "Unsupported parameter" |
| `frequency_penalty` | BLOCKED | "Unsupported parameter" |
| `presence_penalty` | BLOCKED | "Unsupported parameter" |
| URL images | BLOCKED | 403 error |

**Reasoning/thinking trajectory**: GPT reasoning content is hidden (OpenAI standard behavior). `reasoning.summary` with `summary: "detailed"` only produces a one-sentence label, not the full chain-of-thought. This means `output.md` trajectory files will be significantly shorter than with Claude extended thinking. This is accepted as a trade-off for free access.

**Standard API (api.openai.com) supports more parameters**: temperature (when reasoning=none), max_output_tokens, top_p are all supported there. The limitations are specific to the chatgpt.com backend, not the models themselves. We chose to accept these limitations to use the free Codex OAuth path.

Sources:
- Codex Auth docs: https://developers.openai.com/codex/auth/
- Codex Models docs: https://developers.openai.com/codex/models/
- GPT-5.2 guide (parameter support): https://developers.openai.com/api/docs/guides/latest-model/
- gpt-5.3-codex 404 with API key (confirms OAuth-only): https://github.com/anomalyco/opencode/issues/12839
- OpenCode Codex Auth plugin (third-party OAuth usage): https://github.com/numman-ali/opencode-openai-codex-auth
- Codex OAuth ToS discussion (forking allowed): https://github.com/openai/codex/discussions/8338

### Backend Decision: Codex OAuth (Option 5)

Chosen: **Codex OAuth + chatgpt.com/backend-api** (free via ChatGPT Plus subscription).

Accepted trade-offs:
- No temperature/max_output_tokens (use `reasoning.effort` instead)
- Thinking/reasoning not visible (output.md will only contain final answers, not chain-of-thought)
- Gemini (Option 4) was also tested and has better parameter support + visible thinking, but aggressive rate limiting (~1 req/50s) makes it impractical for tts-agent workloads

Glue layer: custom `ChatCodex(BaseChatModel)` class (~150 lines) that reads `~/.codex/auth.json`, calls Responses API, returns LangChain `AIMessage` objects. No CRS needed.

---

## Behavioral Consistency Analysis

Issues depend on which backend is chosen. Common issues (backend-independent):

### Issue 1: Orchestrator conversation history

**Problem**: Current orchestrator is a multi-turn Claude SDK session. Each explore result feeds back as a tool result, and the orchestrator sees its own previous reasoning. A stateless per-call approach would lose this context.

**Solution**: Use **tool calling** pattern. Orchestrator LLM has `explore`/`integrate` bound as tools. The `messages` list in LangGraph state accumulates naturally (SystemMessage -> HumanMessage -> AIMessage with tool_calls -> ToolMessage with results -> ...), replicating the current multi-turn session.

### Issue 2: trajectory.md format

**Problem**: Current code uses `TrajectoryWriter` to stream orchestrator text, tool calls, and tool results in real-time to `trajectory.md`.

**Solution**: Graph nodes call `TrajectoryWriter` methods at the same points as the current code:
- Orchestrator text from `AIMessage.content` -> `write_text()`
- Tool call from `AIMessage.tool_calls` -> `write_tool_use()`
- Tool result from `ToolMessage.content` -> `write_tool_result()`
- Sub-model thinking -> `write_chunk()` to `output.md`

### Issue 3: Sub-model thinking trajectory (backend-dependent)

**If CRS Claude**: CRS `_convertClaudeMessage()` drops `type: "thinking"` blocks. Need to patch CRS to preserve them in a `thinking_content` field. Sub-model calls use direct `requests.post()` to get the full response.

**If Codex/GPT**: GPT-5 models support `reasoning` in Responses API (seen in test: `"reasoning":{"effort":"medium"}`). Need to verify if reasoning content is accessible in response events.

**If Claude Agent SDK**: Thinking is already available via `ThinkingBlock` in SDK streaming.

### Issue 4: Cost calculation (backend-dependent)

**If CRS Claude**: Need to patch `_convertUsage()` to include raw Anthropic usage. Cost computed using published per-token pricing.

**If Codex/GPT**: Responses API includes `usage` in completion event. Cost computed using OpenAI per-token pricing.

**If Claude Agent SDK**: `ResultMessage.total_cost_usd` provided directly.

### Not affected (identical behavior regardless of backend)

- `RoundLog` format (round_num, action, tool_input)
- `eval_hle.py` results.jsonl record format
- `progress.json`, `rounds.jsonl` format
- `best_of_n` analysis
- `cost_vs_accuracy.png` generation
- Cache read logic (reads existing cache_haiku/, cache_sonnet/ without issues)

---

## Architecture

```
[START]
  -> orchestrator_node (LLM + bind_tools, maintains full message history)
  -> route:
      - has tool_call "explore" AND iter < max -> explore_node
      - has tool_call "integrate" -> integrate_node
      - no tool_call or iter >= max -> fallback_node
  -> explore_node (sub-model call, returns ToolMessage) -> orchestrator_node (loop)
  -> integrate_node (sub-model call, returns ToolMessage) -> END
  -> fallback_node (pick highest confidence candidate) -> END
```

**Key design**: LangGraph manages control flow. Orchestrator LLM maintains conversation history via messages list. Sub-models run in fresh sessions with thinking enabled. Backend (CRS/Codex/SDK) is pluggable via `llm.py`.

---

## CRS Patches (only needed if using CRS Claude backend)

File: `claude-relay-service/src/services/openaiToClaude.js`

### Already done on this branch

1. `convertRequest()` -- extended thinking passthrough (line 64-67)
2. `convertRequest()` -- custom system prompt passthrough (line 37-47)

### Still needed (if using CRS)

3. `_convertClaudeMessage()` -- preserve `type: "thinking"` content blocks as `thinking_content` field
4. `_convertUsage()` -- include raw `anthropic_usage` for exact cost calculation

## Codex OAuth Details (if using Codex/GPT backend)

### Auth file

`~/.codex/auth.json`:
``json
{
  "tokens": {
    "access_token": "eyJ...",
    "refresh_token": "rt_...",
    "account_id": "8e9f6044-..."
  },
  "last_refresh": "2026-02-27T16:09:49Z"
}
``

### API call example

``bash
curl -s "https://chatgpt.com/backend-api/codex/responses" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "chatgpt-account-id: $ACCOUNT_ID" \
  -H "originator: codex_cli_rs" \
  -d '{
    "model": "gpt-5.2",
    "input": [{"role": "user", "content": "..."}],
    "instructions": "system prompt here",
    "store": false,
    "stream": true
  }'
``

### Response format (SSE events)

``
event: response.created
event: response.output_item.added  (type: "reasoning")
event: response.output_item.done   (reasoning done)
event: response.output_item.added  (type: "message")
event: response.output_text.delta  (delta: "Hello")
event: response.output_text.delta  (delta: " world")
event: response.completed          (includes usage)
``

### Tool calling in Responses API

TBD -- needs testing. Responses API supports `tools` parameter with function definitions. Need to verify tool_call events format for orchestrator use.

---

## File-by-File Implementation Plan

### Files to DELETE

| File | Reason |
|------|--------|
| `tts-agent/orchestrator.py` | Direct mode orchestrator + `run_session()` helper |
| `tts-agent/tools.py` | Direct mode MCP tools (data classes move to state.py) |
| `tts-agent/tools_delegated.py` | Delegated MCP tools (replaced by graph nodes) |
| `tts-agent/prompts.py` | Direct mode prompts |
| `tts-agent/prompts_delegated.py` | Replaced by new prompts.py |
| `tts-agent/orchestrator_delegated.py` | Replaced by new orchestrator.py |

### New files to CREATE

#### `tts-agent/llm.py` -- LLM factory (backend-dependent)

Abstracts LLM calls behind a uniform interface. Two functions:
- `make_orchestrator_llm(model, backend)` -> LLM instance with tool calling support
- `call_sub_model_llm(messages, model, system_prompt, tools, thinking, backend)` -> response dict with text, tool_calls, thinking_content, usage

**If CRS Claude backend**:
- Orchestrator: `ChatOpenAI(base_url=CRS_URL, api_key=CRS_KEY).bind_tools([...])`
- Sub-model: `requests.post(CRS_URL + "/chat/completions", ...)` with thinking parameter

**If Codex/GPT backend**:
- Orchestrator: custom wrapper around Responses API (ChatOpenAI doesn't support it)
- Sub-model: `requests.post("https://chatgpt.com/backend-api/codex/responses", ...)` with Bearer token from `~/.codex/auth.json`

**If Claude Agent SDK backend**:
- Keep current `ClaudeSDKClient` approach for sub-models
- Orchestrator: either SDK or custom wrapper

#### `tts-agent/state.py` -- Graph state and data classes

```python
from pydantic import BaseModel, Field
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from trajectory import CostTracker, TrajectoryWriter

class Candidate(BaseModel):
    answer: str
    reasoning: str
    approach: str
    confidence: float
    cost_usd: float = 0.0

class TTSState(TypedDict):
    problem: str
    image_data_url: str | None
    candidates: list[Candidate]
    current_iteration: int
    max_iterations: int
    final_answer: str | None
    final_reasoning: str | None
    final_analysis: str | None
    messages: Annotated[list[BaseMessage], add_messages]
    cost: CostTracker
    writer: TrajectoryWriter
    # Config (read-only during graph execution)
    explore_model: str
    integrate_model: str
    budget_tokens: int
    effort: str | None
    cache_dir: str | None
    cache_only: bool
    quiet: bool
    traj_dir: str | None
```

#### `tts-agent/graph.py` -- LangGraph StateGraph

Nodes:
- `orchestrator_node(state)`: invoke `ChatOpenAI` with `state["messages"]`, write text/tool_call to trajectory, return `{"messages": [ai_message]}`
- `explore_node(state)`: call sub-model via `call_crs()`, append Candidate, write trajectory, return `{"messages": [tool_message], "candidates": [...], "current_iteration": N}`
- `integrate_node(state)`: call sub-model via `call_crs()`, set final_answer, write trajectory, return `{"messages": [tool_message], "final_answer": "...", ...}`
- `fallback_node(state)`: pick highest confidence candidate, return `{"final_answer": "..."}`

Routing:
- `route_after_orchestrator(state)`: read last message's `tool_calls[0].name`, check iteration count
  - `"explore"` and `current_iteration < max_iterations` → `"explore_node"`
  - `"integrate"` → `"integrate_node"`
  - else → `"fallback_node"`

```python
def build_graph(orchestrator_llm, sub_model_fn, ...) -> CompiledGraph:
    graph = StateGraph(TTSState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("explore", explore_node)
    graph.add_node("integrate", integrate_node)
    graph.add_node("fallback", fallback_node)
    graph.set_entry_point("orchestrator")
    graph.add_conditional_edges("orchestrator", route_after_orchestrator,
        {"explore_node": "explore", "integrate_node": "integrate", "fallback_node": "fallback"})
    graph.add_edge("explore", "orchestrator")
    graph.add_edge("integrate", END)
    graph.add_edge("fallback", END)
    return graph.compile()
```

### Files to REWRITE

#### `tts-agent/sub_model.py`

Replace `ClaudeSDKClient` with `call_crs()` from `llm.py`:

```python
async def call_sub_model(
    system_prompt, user_message, image_data_url, model,
    output_schema,  # Pydantic model class
    writer=TrajectoryWriter.noop(),
    budget_tokens=32000, effort=None,
) -> tuple[dict, str, float, dict]:
    """Returns (result_dict, trajectory_text, cost_usd, usage_dict)."""
    # Build messages in OpenAI format
    # Build tools from output_schema (single function "StructuredOutput")
    # Call call_crs() with thinking enabled
    # Parse response: extract thinking_content, tool_calls arguments
    # Write thinking to writer
    # Return same tuple as before
```

Keep `save_sub_model_result()` and `make_sub_model_caller()` caching layer **unchanged**.

#### `tts-agent/prompts.py` (replaces both old prompts files)

From `prompts_delegated.py`:
- `ORCHESTRATOR_SYSTEM_PROMPT` (the meta-reasoning dispatcher prompt)
- `EXPLORER_SYSTEM_PROMPT`, `INTEGRATOR_SYSTEM_PROMPT`
- `build_user_message()`, `build_explorer_message()`, `build_integrator_message()`

New additions:
- `ExploreResult(BaseModel)` with fields: approach, reasoning, answer, confidence
- `IntegrationResult(BaseModel)` with fields: analysis, final_answer, reasoning
- `JudgeResult(BaseModel)` with field: correct (bool)
- Orchestrator tool definitions in OpenAI function format:
  ```python
  EXPLORE_TOOL = {
      "type": "function",
      "function": {
          "name": "explore",
          "description": "Dispatch a fresh, independent solver...",
          "parameters": {"type": "object", "properties": {}, ...}
      }
  }
  INTEGRATE_TOOL = { ... }
  ```

Keep `EXPLORE_SCHEMA` and `INTEGRATION_SCHEMA` dicts for `call_crs()` tool parameter construction.

#### `tts-agent/orchestrator.py` (replaces orchestrator_delegated.py)

```python
async def solve(
    problem, image_data_url=None,
    orchestrator_model="claude-haiku-4-5-20251001",
    explore_model="claude-haiku-4-5-20251001",
    integrate_model="claude-haiku-4-5-20251001",
    max_iterations=8, quiet=False, logger=None, question_id=None,
    cache_dir=None, cache_only=True, budget_tokens=32000, effort=None,
) -> SolveResult:
    # 1. Create CostTracker, TrajectoryWriter
    # 2. Create sub_model_fn via make_sub_model_caller(cache_dir, cache_only)
    # 3. Create orchestrator LLM via make_orchestrator_llm(orchestrator_model)
    # 4. Build initial state with system + user messages
    # 5. Build and invoke graph
    # 6. Extract final state: answer, cost, rounds
    # 7. Handle fallback if no integrate
    # 8. Return SolveResult
```

Same function signature as current `orchestrator_delegated.solve()`.

#### `tts-agent/answer_matching.py`

Replace `call_sub_model` (Claude SDK) with the new version:
```python
from sub_model import call_sub_model  # now uses call_crs() internally
```

Same `judge_answer()` function, same `JUDGE_SYSTEM_PROMPT` and `JUDGE_SCHEMA`.

#### `tts-agent/precache_explores.py`

Replace import:
```python
from sub_model import call_sub_model, save_sub_model_result  # new version
```

Same `precache()` function, same `worker()` logic.

#### `tts-agent/eval_hle.py`

- Remove `--mode direct/delegated` flag and conditional import
- Always: `from orchestrator import solve`
- Remove `args.mode` references

#### `tts-agent/main.py`

Rewrite CLI:
```python
from orchestrator import solve

# Add flags: --orchestrator-model, --explore-model, --integrate-model
# Call solve() with all parameters
```

### Files to KEEP AS-IS (no changes needed)

| File | Contents |
|------|----------|
| `tts-agent/trajectory.py` | CostTracker, RoundLog, TrajectoryWriter, SolveResult |
| `tts-agent/dataset_hle.py` | HLE-Verified dataset loading and filtering |
| `tts-agent/best_of_n.py` | Oracle/majority best-of-n analysis |
| `tts-agent/logger.py` | RunLogger for real-time logging |
| `multimodal_input.py` | `build_openai_content()` for image handling |

---

## Dependencies

```
langgraph
langchain-openai
langchain-core
pydantic
requests
```

Install: `pip install langgraph langchain-openai langchain-core`

---

## Implementation Order

0. **Decide backend** (CRS Claude vs Codex/GPT vs Claude SDK vs multi-backend)
1. If CRS: patch CRS (thinking response + usage) and restart
2. Create `llm.py` (LLM factory + backend-specific HTTP helpers)
3. Create `state.py` (data classes + graph state)
4. Rewrite `prompts.py` (merge prompts + Pydantic models)
5. Rewrite `sub_model.py` (use llm.py, keep caching)
6. Create `graph.py` (LangGraph StateGraph)
7. Rewrite `orchestrator.py` (solve function)
8. Rewrite `answer_matching.py` (judge)
9. Update `precache_explores.py` (import change)
10. Simplify `eval_hle.py` (remove direct mode)
11. Rewrite `main.py` (CLI)
12. Delete old files
13. Test with cached data (no API calls needed)

---

## Testing Strategy

- **Step 1**: Unit test LLM backend with a simple prompt
- **Step 2**: Run `precache_explores.py` on 1 question to verify sub-model + caching works
- **Step 3**: Run `main.py` on a simple problem to verify full graph execution
- **Step 4**: Run `eval_hle.py --cache-dir cache_haiku/gold --num 5` to verify evaluation pipeline with existing cache

---

## Open Questions

1. ~~**Which backend?**~~ RESOLVED: Codex OAuth + chatgpt.com/backend-api.
2. ~~**Codex tool calling**~~ RESOLVED: Tested and working -- single tool, multi tool, forced tool_choice all OK.
3. **GPT vs Claude for HLE**: The tts-agent was tuned/tested with Claude models. GPT-5.2 may perform differently on HLE questions. Needs evaluation.
4. **Token refresh for Codex**: Codex CLI handles refresh automatically. If we use Codex backend programmatically, do we need to implement refresh ourselves, or can we rely on the CLI refreshing `auth.json`?
5. **Parameter workarounds**: temperature/max_output_tokens blocked on chatgpt.com backend. Can approximate temperature via prompt instructions ("be deterministic" / "be creative"). max_output_tokens has no workaround but may not be critical for our use case.

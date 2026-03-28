# Codex Backend: 使用 Codex CLI 凭证调用 OpenAI GPT 模型

## 原理

Codex CLI（`codex`）登录后会在 `~/.codex/auth.json` 中缓存 OAuth access token。我们的 `backends/codex.py` 直接读取这个 token，通过 ChatGPT 的内部 Responses API 端点发送请求，实现无需 OpenAI API key 即可调用 GPT 模型。

```
Codex CLI login → ~/.codex/auth.json → backends/codex.py → chatgpt.com/backend-api/codex/responses
```

## 前置条件

```bash
# 1. 安装 Codex CLI
npm install -g @anthropic-ai/codex  # 或通过 nvm

# 2. 登录（会打开浏览器 OAuth 流程）
codex login

# 3. 验证凭证
cat ~/.codex/auth.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK' if d['tokens']['access_token'] else 'EMPTY')"
```

## auth.json 结构

```json
{
  "OPENAI_API_KEY": null,
  "tokens": {
    "id_token": "eyJ...",
    "access_token": "eyJ...",    // <- 我们用这个
    "refresh_token": "rt_...",
    "account_id": "8e9f..."
  },
  "last_refresh": "2026-03-24T16:47:03Z"
}
```

## API 端点

```
POST https://chatgpt.com/backend-api/codex/responses
Authorization: Bearer {access_token}
Content-Type: application/json
```

请求体格式（Responses API，非 Chat Completions API）：

```json
{
  "model": "gpt-5.2",
  "input": [{"role": "user", "content": "Hello"}],
  "instructions": "System prompt here",
  "stream": true,
  "store": false,
  "tools": [...],                        // 可选：function calling
  "text": {"format": {"type": "json_schema", ...}},  // 可选：structured output
  "reasoning": {"effort": "high"},       // 可选：reasoning effort
  "parallel_tool_calls": false           // 强制 sequential tool calling
}
```

## 可用模型

| Model ID | Input $/M | Output $/M |
|----------|-----------|------------|
| gpt-5.4 | $2.50 | $15.00 |
| gpt-5.2 | $1.75 | $14.00 |
| gpt-5.1 | $1.25 | $10.00 |
| gpt-5 | $1.25 | $10.00 |
| gpt-5-codex-mini | $0.25 | $2.00 |

## backends/codex.py 关键设计

### 接口与 Claude backend 完全一致

两个 backend 暴露相同的函数签名：

```python
async def call_sub_model(
    system_prompt, user_message, image_data_url, model,
    output_schema, writer, budget_tokens, effort
) -> (structured_output, trajectory_text, cost_usd, usage)

async def run_tool_conversation(
    *, system_prompt, user_message, image_data_url, model,
    tools, max_turns, tool_handler, effort,
    output_format, writer, quiet, on_structured_output
) -> (cost_usd, usage)
```

方法代码（tts_agent.py 等）通过 `import_module(f"backends.{backend}")` 动态加载，无需任何 backend-specific 逻辑。

### Sequential tool calling

GPT 默认支持 parallel function calling（一个 turn 返回多个 tool call）。ATTS 要求 sequential exploration（每次 explore 的结果影响下一步决策）。通过 `parallel_tool_calls: false` 强制每 turn 只返回 1 个 tool call。

### Forced structured output

当 orchestrator 用完 max_turns 但未输出 structured answer 时（`structured_output_emitted = False`），发送一次不带 tools 的 request，强制模型输出 JSON structured answer。

### Retry 策略

| 错误类型 | 重试次数 | 延迟 |
|---------|---------|------|
| HTTP 5xx | 6 次 | 5s * attempt |
| HTTP 429 (rate limit) | 6 次 | 60s * attempt |
| RemoteProtocolError / ReadTimeout | 6 次 | 5s * attempt |

### Judge 模型映射

当 backend=codex 时，LLM judge 自动从 Claude 模型映射到 GPT 模型：

```python
_JUDGE_MODEL_CODEX = {"claude-haiku-4-5-20251001": "gpt-5-codex-mini"}
```

GPT-5-codex-mini 与 Claude Haiku 在 20 个 HLE 判分样本上 100% 一致。

## 运行实验

### 1. Precache explores

```bash
cd Experiment/core_code
python precache_explores.py \
    --benchmark gpqa \
    --backend codex \
    --cache-dirs ../analysis/cache/gpqa/gpt5.2 \
    --num-explores 8 \
    --num-workers 4 \
    --seed 42 \
    --explore-model gpt-5.2 \
    --effort low
```

### 2. Run ATTS eval（从 cache 读 explore，orchestrator 实时调用）

```bash
python eval.py --benchmark gpqa \
    --backend codex \
    --seed 42 \
    --num-explores 8 \
    --num-workers 1 \
    --no-integrate \
    --log-dir ../analysis/run/gpqa/gpt5.2_no_integrate_high \
    --orchestrator-model gpt-5.2 \
    --explore-model gpt-5.2 \
    --integrate-model gpt-5.2 \
    --cache-dirs ../analysis/cache/gpqa/gpt5.2 \
    --effort high
```

注意：`--effort low` 用于 precache（explorer），`--effort high` 用于 eval（orchestrator）。Explores 从 cache 读取，不受 eval 的 effort 参数影响。

### 3. Rate limit 注意事项

- Codex API rate limit 比标准 OpenAI API 严格
- `--num-workers 4` 通常可以跑，但高峰期可能需要降到 1-2
- 429 会自动 retry（最多等 6 分钟/请求）
- 如果频繁 429，降低 workers 或等待一段时间

## Token 过期处理

OAuth token 会过期。如果遇到 401 错误：

```bash
codex login  # 重新登录刷新 token
```

## 与其他方案的对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| **backends/codex.py（当前）** | 直接读 auth.json，无需额外服务 | 用的是 Responses API（非标准 Chat Completions） |
| openai-oauth (npx) | 暴露标准 /v1/chat/completions | 需要额外进程 |
| codex-openai-proxy (Rust) | 更健壮，处理 Cloudflare | 需要编译 Rust |

当前方案最简单，因为不需要任何中间代理——直接从 Python 调用 Codex Responses API。

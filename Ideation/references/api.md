# OpenAI-Compatible API for Claude & Gemini

Two approaches to expose Claude and Gemini models as OpenAI-compatible endpoints (`/v1/chat/completions`), enabling LangChain/LangGraph integration.

## 1. Claude via `claude-code-openai-wrapper`

Uses [RichardAtCT/claude-code-openai-wrapper](https://github.com/RichardAtCT/claude-code-openai-wrapper) + Claude Agent SDK. Leverages local `~/.claude` CLI credentials (Claude Max subscription), no API key needed.

### 1.1 Installation

```bash
pip install claude-code-openai-wrapper
```

### 1.2 Startup

```bash
claude-wrapper --port 8000
```

If running inside a Claude Code session (e.g. from claude code terminal), must unset detection env vars to avoid nested session error:

```bash
env -u CLAUDECODE -u CLAUDE_CODE_SSE_PORT -u CLAUDE_CODE_ENTRYPOINT claude-wrapper --port 8000
```

Startup will prompt "Enable API key protection? (y/N)" -- press N for local use. To run non-interactively:

```bash
echo "N" | env -u CLAUDECODE -u CLAUDE_CODE_SSE_PORT -u CLAUDE_CODE_ENTRYPOINT claude-wrapper --port 8000 > tmp/wrapper_server.log 2>&1 &
```

### 1.3 Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="claude-sonnet-4-6",
    api_key="sk-dummy",           # arbitrary, wrapper uses local CLI credentials
    base_url="http://127.0.0.1:8000/v1",
    temperature=0,
)

# Text
response = llm.invoke("What is 3+4? Answer with just the number.")
# -> "7"

# Multimodal (base64 data URL)
msg = HumanMessage(content=[
    {"type": "text", "text": "What color is this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64_data>"}},
])
response = llm.invoke([msg])
# -> "Red"
```

### 1.4 Supported Models

| Model ID | Text | Multimodal | Status |
|----------|------|------------|--------|
| `claude-haiku-4-5-20251001` | OK | OK | OK |
| `claude-sonnet-4-6` | OK | OK | OK |
| `claude-opus-4-6` | OK | OK | OK (occasional 429 rate limit) |
| `claude-sonnet-4-5-20250929` | OK | OK | OK |

### 1.5 Temperature

Temperature 0 ~ 2.0 all accepted. The wrapper applies temperature via system prompt injection ("best-effort"), not native API parameter -- Claude's API does support temperature natively but the wrapper's conversion is approximate.

Tested: 0, 0.5, 0.7, 1.0, 2.0 -- all work.

### 1.6 Test Results (HLE Multimodal)

Test: sheet music image, gold answer = "Shinzou wo Sasageyo!"

| Model | temp | Response |
|-------|------|----------|
| haiku 4.5 | 0 | Could not identify, asked for more info |
| sonnet 4.6 | 0 | "River Flows in You" (wrong) |
| sonnet 4.6 | 0.7 | "A Cruel Angel's Thesis" (wrong, correct genre) |
| opus 4.6 | 0 | "Unravel" from Tokyo Ghoul (wrong, correct genre) |

### 1.7 Limitations

- **Nested session**: Cannot run inside a Claude Code session without unsetting `CLAUDECODE`, `CLAUDE_CODE_SSE_PORT`, `CLAUDE_CODE_ENTRYPOINT` env vars.
- **Rate limiting**: opus 4.6 occasionally returns 429 (retry after 60s). Lower-tier models are more lenient.
- **Temperature**: Applied via system prompt (best-effort), not as a native API parameter.
- **Model names**: The wrapper warns about model names not in its hardcoded list (e.g. `claude-sonnet-4-6`) but still attempts them and they work.
- **Credentials**: Depends on `~/.claude/` CLI login. If token expires, re-run `claude /login`.

---

## 2. Gemini via `claude-relay-service` (CRS)

Uses [claude-relay-service](https://github.com/your-org/claude-relay-service) as a relay. CRS calls Google's internal `cloudcode-pa.googleapis.com/v1internal` endpoint (same as Gemini CLI) using local Gemini OAuth credentials from `~/.gemini/oauth_creds.json`.

### 2.1 Prerequisites

- Docker (for Redis)
- Node.js (v18+)
- Gemini CLI credentials at `~/.gemini/oauth_creds.json` (run `gemini` CLI and login first)

### 2.2 Startup

**Step 1: Start Redis**

```bash
docker run --rm -d --name redis-crs -p 6379:6379 redis:7-alpine
```

**Step 2: Setup CRS**

```bash
cd claude-relay-service
cp config/config.example.js config/config.js
npm install && npm run setup
```

`.env` (minimum required):

``
PORT=3456
HOST=0.0.0.0
NODE_ENV=development
JWT_SECRET=<32+ char random string>
ENCRYPTION_KEY=<exactly 32 char hex string>
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0
LOG_LEVEL=info
ADMIN_USERNAME=cr_admin
ADMIN_PASSWORD=<your password>
``

**Step 3: Start CRS**

```bash
node src/app.js
# or: npm run dev (with hot reload)
```

**Step 4: Add Gemini account**

```bash
# Login
TOKEN=$(curl -s http://127.0.0.1:3456/web/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"cr_admin","password":"<your password>"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# Read local Gemini OAuth credentials
CREDS=$(cat ~/.gemini/oauth_creds.json)
ACCESS_TOKEN=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
REFRESH_TOKEN=$(echo "$CREDS" | python3 -c "import sys,json; print(json.load(sys.stdin)['refresh_token'])")

# Create account
ACCOUNT=$(curl -s http://127.0.0.1:3456/admin/gemini-accounts \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d "{
    \"name\": \"gemini-local\",
    \"accessToken\": \"$ACCESS_TOKEN\",
    \"refreshToken\": \"$REFRESH_TOKEN\"
  }")
ACCOUNT_ID=$(echo "$ACCOUNT" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['id'])")
echo "Account ID: $ACCOUNT_ID"
```

**Step 5: Discover and set projectId**

CRS calls `v1internal:loadCodeAssist` to discover the `cloudaicompanionProject`. For the OpenAI-compat endpoint, the projectId must be stored explicitly:

```bash
# Trigger project discovery (any request through the v1internal endpoint)
curl -s "http://127.0.0.1:3456/gemini/v1internal:generateContent" \
  -H "Authorization: Bearer <api_key>" \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemini-2.5-flash","request":{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}}'

# Read discovered tempProjectId
PROJECT_ID=$(docker exec redis-crs redis-cli hget "gemini_account:$ACCOUNT_ID" tempProjectId)

# Set it as the permanent projectId (required for OpenAI-compat endpoint)
docker exec redis-crs redis-cli hset "gemini_account:$ACCOUNT_ID" projectId "$PROJECT_ID"
```

**Step 6: Create API key**

```bash
curl -s http://127.0.0.1:3456/admin/api-keys \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"name": "gemini-key", "permissions": ["gemini"]}'
# Returns: {"data": {"key": "cr_xxxxx..."}}
```

### 2.3 Usage

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key="cr_<your_key>",
    base_url="http://127.0.0.1:3456/openai/gemini/v1",
    temperature=0,
)

# Text
response = llm.invoke("What is 3+4? Answer with just the number.")
# -> "7"

# Multimodal (base64 data URL)
msg = HumanMessage(content=[
    {"type": "text", "text": "What color is this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<base64_data>"}},
])
response = llm.invoke([msg])
# -> "Red"
```

### 2.4 Supported Models

| Model ID | Text | Multimodal | Status |
|----------|------|------------|--------|
| `gemini-2.5-flash` | OK | OK | OK |
| `gemini-2.5-pro` | OK | OK | OK |
| `gemini-2.0-flash` | OK | OK | OK |

Not available (404): `gemini-3.0-pro`, `gemini-3.0-flash`, `gemini-2.0-pro`, `gemini-1.5-*`, preview versions, `models/` prefix variants. The `v1internal` endpoint only exposes these 3 current-gen models.

### 2.5 Temperature

Temperature 0 ~ 2.0 all work. Passed directly to Gemini API's `generationConfig.temperature`.

Tested: 0, 0.5, 0.7, 1.0, 1.5, 2.0 -- all work.

### 2.6 Test Results (HLE Multimodal)

Same test: sheet music image, gold answer = "Shinzou wo Sasageyo!"

Gemini 2.5 flash correctly received the image and returned a substantive music analysis response (through CRS OpenAI-compat endpoint).

### 2.7 Rate Limit Handling

Google's `cloudcode-pa.googleapis.com` enforces aggressive rate limiting (~1 request per 15-30 seconds). CRS auto-marks accounts as rate-limited on 429.

To clear rate limit state between tests:

```bash
ACCOUNT_ID="<your_account_id>"
docker exec redis-crs redis-cli hdel "gemini_account:$ACCOUNT_ID" rateLimitStatus rateLimitedAt
docker exec redis-crs redis-cli del "temp_unavailable:gemini:$ACCOUNT_ID" "error_history:gemini:$ACCOUNT_ID"
```

### 2.8 Limitations

- **Aggressive rate limiting**: Google's internal API limits to ~1 request per 15-30 seconds. Must wait 45s+ between sequential requests to avoid 429.
- **Limited model selection**: Only 3 models available (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`). No older models, no preview/experimental variants.
- **projectId setup**: The OpenAI-compat endpoint requires `projectId` to be explicitly set in Redis. The v1internal handler auto-discovers it via `loadCodeAssist`, but the OpenAI-compat handler only reads `account.projectId`.
- **OAuth token expiry**: Credentials from `~/.gemini/oauth_creds.json` expire. CRS auto-refreshes using the `refreshToken`, but if the refresh token itself expires, re-run `gemini` CLI login.
- **Multimodal fix required**: CRS's `openaiGeminiRoutes.js` originally dropped `image_url` content parts (only extracted text). A patch to `convertMessagesToGemini()` was applied to add `contentToGeminiParts()` which converts `image_url` data URLs to Gemini's `{inlineData: {mimeType, data}}` format. This fix is in the local codebase at `claude-relay-service/src/routes/openaiGeminiRoutes.js`.

---

## 3. Quick Comparison

| | Claude (wrapper) | Gemini (CRS) |
|---|---|---|
| Endpoint | `http://127.0.0.1:8000/v1` | `http://127.0.0.1:3456/openai/gemini/v1` |
| Auth | `api_key="sk-dummy"` | `api_key="cr_<key>"` |
| Dependencies | `pip install claude-code-openai-wrapper` | Node.js + Redis (Docker) + CRS |
| Credentials | `~/.claude/` (Claude Max login) | `~/.gemini/oauth_creds.json` (Gemini CLI login) |
| Models | haiku/sonnet/opus 4.5-4.6 | gemini-2.0-flash, 2.5-flash, 2.5-pro |
| Rate limit | Moderate (opus ~1/min) | Aggressive (~1 per 15-30s) |
| Multimodal | Native support | Supported (with patched CRS) |
| Temperature | 0-2.0 (via system prompt, best-effort) | 0-2.0 (native) |
| Setup complexity | Low (pip install + start) | High (Redis + CRS + account + API key + projectId) |

---
sidebar_position: 9
title: "REST / WebSocket API"
description: "Integrate Hermes Agent into any application via HTTP and WebSocket endpoints"
---

# REST / WebSocket API

Hermes Agent exposes a REST and WebSocket API so any application — web frontends, mobile apps, CLI tools, automation pipelines, or third-party services — can interact with the agent programmatically. The API follows the same BasePlatformAdapter pattern as Telegram, Discord, and other messaging platforms, giving API clients the same capabilities: tool use, memory, skills, voice, and image generation.

:::info Optional Dependency
The API adapter requires FastAPI and uvicorn. These are included in the `[api]` and `[all]` install extras. If you installed with `pip install -e ".[all]"`, you already have them.
:::

---

## Prerequisites

- Hermes Agent installed
- FastAPI and uvicorn available (included in `[all]` or `[api]` extras)
- An API key configured in `~/.hermes/.env`

---

## Step 1: Configure Hermes

Add the following to your `~/.hermes/.env` file:

```bash
# Required
API_ENABLED=true
API_KEY=your-secret-api-key          # Used for authenticating requests

# Optional
API_PORT=8766                         # Default: 8765
API_RESPONSE_TIMEOUT=300              # Seconds before a request times out (default: 300)
```

:::warning
Choose a strong, random API key. Anyone with this key has full access to the agent's capabilities, including tool use and terminal access.
:::

---

## Step 2: Start the Gateway

```bash
hermes gateway
```

The API server starts alongside any other configured platforms (Telegram, Discord, etc.). You'll see:

```text
INFO  API server starting on port 8766
```

---

## Endpoints

### Health Check

```
GET /v1/health
```

No authentication required. Returns `{"status": "ok"}` when the server is running.

```bash
curl http://localhost:8766/v1/health
```

### Synchronous Chat

```
POST /v1/chat
```

Send a message and receive the complete response when the agent finishes processing.

**Headers:**

| Header | Value |
|--------|-------|
| `Authorization` | `Bearer <your-api-key>` |
| `Content-Type` | `application/json` |

**Request body:**

```json
{
  "message": "What is the weather like today?",
  "session_id": "optional-session-id"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The user message to send to the agent |
| `session_id` | string | No | Reuse an existing session for conversation continuity. A new UUID is generated if omitted. |

**Response:**

```json
{
  "response": "I don't have access to real-time weather data, but I can help you check...",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "media": []
}
```

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | The agent's text response |
| `session_id` | string | Session ID (reuse this for follow-up messages) |
| `media` | array | Any media the agent produced (images, audio, documents) |

**Example:**

```bash
curl -X POST http://localhost:8766/v1/chat \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, Hermes!"}'
```

### Streaming Chat (WebSocket)

```
WS /v1/chat/stream
```

Real-time streaming over WebSocket. The agent sends response chunks as they're generated.

**Connection flow:**

1. Connect to `ws://localhost:8766/v1/chat/stream`
2. Send an auth message (browsers cannot send custom headers on WebSocket connections)
3. Receive `auth_ok` confirmation
4. Send messages and receive streamed responses

**Step 1 — Authenticate:**

```json
{"type": "auth", "token": "your-secret-api-key", "session_id": "optional-session-id"}
```

**Step 2 — Receive confirmation:**

```json
{"type": "auth_ok"}
```

**Step 3 — Send a message:**

```json
{"message": "Explain quantum computing"}
```

**Step 4 — Receive streamed response chunks:**

```json
{"type": "message", "content": "Quantum computing is..."}
{"type": "message", "content": "Unlike classical bits..."}
{"type": "done", "session_id": "a1b2c3d4-..."}
```

The `done` message signals that the agent has finished processing. You can then send another message on the same connection.

**Media messages:**

If the agent generates images, audio, or documents, you'll receive media messages:

```json
{"type": "image", "url": "https://...", "caption": "Generated diagram"}
{"type": "audio", "path": "/tmp/response.mp3", "caption": null}
{"type": "video", "path": "/tmp/clip.mp4", "caption": "Screen recording"}
{"type": "document", "path": "/tmp/report.pdf", "caption": "Analysis report"}
```

**Error messages:**

```json
{"type": "error", "content": "Response timeout"}
```

### Interrupt

```
POST /v1/chat/interrupt?session_id=<session-id>
```

Interrupt a running agent mid-response. Equivalent to sending `/stop` in a messaging platform.

```bash
curl -X POST "http://localhost:8766/v1/chat/interrupt?session_id=a1b2c3d4-..." \
  -H "Authorization: Bearer your-secret-api-key"
```

**Response:**

```json
{"interrupted": true, "session_id": "a1b2c3d4-..."}
```

Or if no active session:

```json
{"interrupted": false, "session_id": "a1b2c3d4-...", "reason": "no active session"}
```

### List Sessions

```
GET /v1/sessions
```

Returns all currently active sessions.

```bash
curl http://localhost:8766/v1/sessions \
  -H "Authorization: Bearer your-secret-api-key"
```

### Get Session Transcript

```
GET /v1/sessions/{session_id}
```

Returns the conversation transcript for a session.

```bash
curl http://localhost:8766/v1/sessions/a1b2c3d4-... \
  -H "Authorization: Bearer your-secret-api-key"
```

---

## WebSocket Client Examples

### Python

```python
import asyncio
import websockets
import json

async def chat():
    async with websockets.connect("ws://localhost:8766/v1/chat/stream") as ws:
        # Authenticate
        await ws.send(json.dumps({
            "type": "auth",
            "token": "your-secret-api-key"
        }))
        auth_resp = json.loads(await ws.recv())
        assert auth_resp["type"] == "auth_ok"

        # Send a message
        await ws.send(json.dumps({"message": "What can you do?"}))

        # Receive streamed response
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] == "done":
                break
            elif msg["type"] == "message":
                print(msg["content"], end="", flush=True)

asyncio.run(chat())
```

### JavaScript / Node.js

```javascript
const WebSocket = require("ws");

const ws = new WebSocket("ws://localhost:8766/v1/chat/stream");

ws.on("open", () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: "auth",
    token: "your-secret-api-key"
  }));
});

ws.on("message", (data) => {
  const msg = JSON.parse(data);

  if (msg.type === "auth_ok") {
    // Send a message after auth
    ws.send(JSON.stringify({ message: "Hello from Node.js!" }));
  } else if (msg.type === "message") {
    process.stdout.write(msg.content);
  } else if (msg.type === "done") {
    console.log("\n--- Done ---");
    ws.close();
  }
});
```

### cURL (HTTP only)

```bash
# Simple chat request
curl -X POST http://localhost:8766/v1/chat \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "List files in the current directory"}'

# With session continuity
curl -X POST http://localhost:8766/v1/chat \
  -H "Authorization: Bearer your-secret-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "Now show the git log", "session_id": "a1b2c3d4-..."}'
```

---

## Session Management

Each `session_id` maintains its own conversation context, just like separate chats in Telegram or Discord. Sessions follow the same reset policies as other platforms:

| Policy | Default | Description |
|--------|---------|-------------|
| Daily | 4:00 AM | Reset at a specific hour each day |
| Idle | 120 min | Reset after N minutes of inactivity |

You can override these in `~/.hermes/gateway.json`:

```json
{
  "reset_by_platform": {
    "api": { "mode": "idle", "idle_minutes": 240 }
  }
}
```

---

## Security

### Authentication

All endpoints except `/v1/health` require authentication:

- **HTTP endpoints** use the `Authorization: Bearer <key>` header
- **WebSocket** uses first-message authentication (browsers cannot send custom headers on WebSocket connections):
  ```json
  {"type": "auth", "token": "<key>"}
  ```

The API key is validated using constant-time comparison (`secrets.compare_digest`) to prevent timing attacks.

### Access Control

The API adapter bypasses the per-user allowlist system used by messaging platforms. Authentication is handled entirely by the API key — anyone with a valid key has full access.

:::warning
**Protect your API key.** It grants the same level of access as being an allowed user on Telegram or Discord — including tool use, terminal commands, and file access. Do not expose the API server to the public internet without additional security measures (reverse proxy, TLS, IP allowlist).
:::

### Production Deployment

For production use, place the API behind a reverse proxy with TLS:

```nginx
server {
    listen 443 ssl;
    server_name hermes-api.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/hermes-api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/hermes-api.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8766;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 300s;
    }
}
```

For more information on securing your Hermes Agent deployment, see the [Security Guide](../security.md).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **"API_KEY not configured on server"** (500) | Set `API_KEY` in `~/.hermes/.env` and restart the gateway. |
| **"Invalid API key"** (401) | Check that your `Authorization` header matches the `API_KEY` value. Include the `Bearer ` prefix. |
| **Missing Authorization header** (422) | Add `-H "Authorization: Bearer <key>"` to your request. |
| **Connection refused** | Verify the gateway is running (`hermes gateway`) and the port matches `API_PORT`. |
| **WebSocket auth timeout** | Send the auth message within 10 seconds of connecting. |
| **Response timeout** | The agent took too long. Increase `API_RESPONSE_TIMEOUT` or simplify the request. |
| **Port already in use** | Another service is using the port. Change `API_PORT` to a different value. |
| **FastAPI not installed** | Run `pip install -e ".[api]"` or `pip install -e ".[all]"` to install API dependencies. |

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_ENABLED` | Yes | `false` | Enable the REST/WebSocket API adapter |
| `API_KEY` | Yes | — | Secret key for authenticating API requests |
| `API_PORT` | No | `8765` | Port the API server listens on |
| `API_RESPONSE_TIMEOUT` | No | `300` | Seconds before a request times out |

---
sidebar_position: 9
title: "Dashboard / Gateway Protocol Reference"
description: "WebSocket JSON-RPC and REST surface of the Hermes web dashboard server for third-party client authors"
---

# Dashboard / Gateway Protocol Reference

This page documents the HTTP and WebSocket protocol exposed by the Hermes web
dashboard server (`hermes web`, default port 9119) as of **v0.16.0**.  It is
intended for authors of third-party clients — native mobile apps, desktop
companions, IDE plugins — that drive Hermes over the network rather than via
the TUI.

Source references used:
`tui_gateway/server.py`,
`hermes_cli/web_server.py`,
`hermes_cli/dashboard_auth/`,
`web/src/lib/gatewayClient.ts`.

---

## Endpoint

Dashboard gateway: `http(s)://<host>:9119`. WebSocket chat: `ws(s)://<host>:9119/api/ws`.

Default bind is `127.0.0.1`. For remote access over a private network (e.g.
Tailscale) the gateway must run `hermes web --host <tailnet-ip-or-0.0.0.0>`.

---

## Auth

### 1. Probe the server

```
GET /api/status
→ {
    "version": "0.16.0",
    "auth_required": bool,
    "gateway_running": bool,
    "active_sessions": int,
    ...
  }
```

### 2. Loopback / `--insecure` mode (`auth_required == false`)

The session token is embedded in the SPA HTML served at `GET /`:

```
GET /
← HTML containing: window.__HERMES_SESSION_TOKEN__ = "<token>"
```

Fetch the root page and extract the token with a regex, or allow manual entry in settings.

- **REST:** `Authorization: Bearer <token>` on all `/api/` requests.
- **WebSocket:** `ws://host:9119/api/ws?token=<token>`

### 3. OAuth / gated mode (`auth_required == true`)

```
POST /auth/password-login
body {"provider": "...", "username": "...", "password": "...", "next": "/"}
← Sets HttpOnly cookies: hermes_session_at (~15 min), hermes_session_rt (24 h)
  Cookies may carry __Host- or __Secure- prefixes on TLS deployments.
```

Mint a single-use WebSocket ticket (valid 30 s) immediately before each connect:

```
POST /api/auth/ws-ticket   (session cookie required)
→ {"ticket": "<43-char token>", "ttl_seconds": 30}
```

- **WebSocket:** `ws://host:9119/api/ws?ticket=<ticket>`
- **Helper endpoints:** `GET /api/auth/me` → user info; `GET /api/auth/providers` → provider list.

### WebSocket close codes

| Code | Meaning |
|------|---------|
| 4401 | Bad credential (token/ticket invalid or missing) |
| 4403 | Host or Origin mismatch |
| 4408 | Peer not allowed |
| 4404 | Chat disabled |
| 4400 | Bad channel |

---

## WebSocket wire format

JSON-RPC 2.0, one JSON object per text frame.

**Client → server request:**

```json
{
  "jsonrpc": "2.0",
  "id": "w1",
  "method": "prompt.submit",
  "params": {
    "session_id": "20250601_123456_abcd1234",
    "text": "Hello"
  }
}
```

**Server → client RPC reply:**

```json
{"jsonrpc": "2.0", "id": "w1", "result": {"status": "streaming"}}
```

**Server → client event (no `id`):**

```json
{
  "jsonrpc": "2.0",
  "method": "event",
  "params": {
    "type": "message.delta",
    "session_id": "...",
    "payload": {"text": "chunk"}
  }
}
```

On connect the server immediately emits a `gateway.ready` event with `{"skin": ...}` payload.

There is no protocol-level ping; detect drops via transport-layer keepalive (e.g. OkHttp
ping interval) and reconnect with backoff.  The server reaps orphan sessions approximately
20 s after disconnect.  The web client uses a 120 s request timeout.

---

## Session IDs

Two IDs appear throughout the protocol:

- `session_id` — the *live* session key, identifying the active agent run.
- `stored_session_id` — the *storage* key for the underlying session record.

After a context-compression rotation these may diverge; always use `session_id`
for subsequent `prompt.submit` and `session.resume` calls.

---

## RPC methods (client → server)

| Method | Params | Result |
|---|---|---|
| `session.create` | `{cols, messages, title, cwd, profile}` (all optional) | `{session_id, stored_session_id, message_count, messages, info}` |
| `session.resume` | `{session_id}` | Emits `session.info` event |
| `session.list` | none | Array of session objects |
| `prompt.submit` | `{session_id, text}` | `{"status": "streaming"}` then async events |
| `slash.exec` | `{session_id, command}` | Long-running; events follow |
| `session.branch` | `{session_id, ...}` | Long-running |
| `session.compress` | `{session_id, ...}` | Long-running |

---

## Event types (server → client)

**Streaming:**
`message.start`, `message.delta` (`payload.text`), `message.complete`,
`thinking.delta`, `reasoning.delta`, `reasoning.available`.

**Tools:**
`tool.generating` (`payload.name`),
`tool.start` (`{tool_id, name, context, args_text}`),
`tool.progress`,
`tool.complete` (`{tool_id, name, args, result, summary, duration_s, inline_diff, todos}`).

**Lifecycle:**
`status.update`, `session.info`, `gateway.ready`, `skin.changed`, `error`.

**Interactive (require client reply):**
`clarify.request`, `approval.request`, `sudo.request`, `secret.request`.

**Proactive:**
`background.complete` — arrives unprompted on any open WebSocket attached to the session.

**Voice:**
`voice.status`, `voice.transcript`.

---

## REST: sessions

```
GET    /api/sessions?limit=&offset=&archived=exclude|only|include&order=created|recent
       → {"sessions": [...], "total": N, "limit": N, "offset": N}
GET    /api/sessions/search?q=&limit=&offset=
GET    /api/sessions/{id}            → full session object
GET    /api/sessions/{id}/messages   → {"session_id": "...", "messages": [...]}
PATCH  /api/sessions/{id}            body {"title" | "archived" | "profile"}
DELETE /api/sessions/{id}
```

---

## REST: voice

**Speech-to-text:**

```
POST /api/audio/transcribe
body {"data_url": "data:audio/mp4;base64,<b64>", "mime_type": "audio/mp4"}
→    {"ok": true, "transcript": "...", "provider": "whisper"}
```

Any `audio/*` MIME type is accepted. Returns 413 if the payload is too large.

**Text-to-speech:**

```
POST /api/audio/speak
body {"text": "..."}
→    {"ok": true, "data_url": "data:audio/mpeg;base64,<b64>", "mime_type": "audio/mpeg", "provider": "..."}
```

No realtime audio streaming; both endpoints are request/response.

---

## REST: file attachments

Multipart upload for binary payloads:

```
POST /api/files/attachment-upload
Content-Type: multipart/form-data
field: file (binary)
→ {"name": "...", "path": "...", "url": "/files/<name>", "size": <bytes>}
```

Max 100 MB. Stored in `~/.hermes/uploads/` with collision-safe naming.

Authenticated download (supports `Authorization: Bearer <token>` header or `?token=` query
parameter for clients that cannot set custom headers on download requests):

```
GET /files/{name}[?token=<session_token>]
← file bytes with appropriate Content-Type
```

:::note
The `POST /api/files/upload` endpoint (no `/attachment-` prefix) is a
separate API for base64-encoded file management and has different semantics.
:::

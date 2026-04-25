---
sidebar_position: 20
title: "NotionAgent"
description: "Connect the NotionAgent web app to Hermes Gateway with signed HTTP messages"
---

# NotionAgent

The NotionAgent adapter lets the [NotionAgent web app](https://github.com/ravenviersechs/NotionAgent) use Hermes as its agent backend without routing through Telegram or another chat platform.

It is a text-only, bidirectional HTTP adapter:

- NotionAgent sends inbound messages to Hermes with `POST /notionagent/in`
- Hermes sends the final reply back to NotionAgent with a signed callback POST
- Each `session_id` maps to one Hermes conversation, so a memo or task keeps its own context

## Quick Start

1. Generate a long shared secret.
2. Configure the same secret in Hermes and NotionAgent.
3. Set the NotionAgent callback URL in Hermes.
4. Point NotionAgent at the Hermes inbound URL.

Add to `~/.hermes/.env`:

```bash
NOTIONAGENT_SECRET=replace-with-a-long-random-secret
NOTIONAGENT_CALLBACK_URL=https://notionagent.example.com/hermes/callback
NOTIONAGENT_PORT=8645
```

Start or restart the gateway:

```bash
hermes gateway start
```

The inbound URL is:

```text
https://your-hermes-host:8645/notionagent/in
```

If Hermes is behind a reverse proxy, terminate TLS at the proxy and forward to the local gateway port.

## Configuration

You can configure NotionAgent either with environment variables or `config.yaml`.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NOTIONAGENT_SECRET` | Yes | Shared HMAC-SHA256 secret for inbound and outbound requests |
| `NOTIONAGENT_CALLBACK_URL` | Yes | NotionAgent endpoint where Hermes posts replies |
| `NOTIONAGENT_PORT` | No | Inbound HTTP port, default `8645` |
| `NOTIONAGENT_HOST` | No | Bind host, default `0.0.0.0` |
| `NOTIONAGENT_PATH` | No | Inbound path, default `/notionagent/in` |
| `NOTIONAGENT_ALLOWED_USERS` | No | Optional comma-separated allowlist of `session_id` values |
| `NOTIONAGENT_ALLOW_ALL_USERS` | No | Optional explicit allow-all flag; HMAC is still required |
| `NOTIONAGENT_HOME_CHANNEL` | No | Default session ID for cron and `send_message` delivery |

When `NOTIONAGENT_ALLOWED_USERS` is unset, all correctly signed requests are accepted. This differs from user-chat platforms because NotionAgent authorization is handled by the HMAC signature.

### config.yaml

```yaml
platforms:
  notionagent:
    enabled: true
    secret: "replace-with-a-long-random-secret"
    callback_url: "https://notionagent.example.com/hermes/callback"
    host: "0.0.0.0"
    port: 8645
    path: "/notionagent/in"
```

The same keys may also be placed under `extra`:

```yaml
platforms:
  notionagent:
    enabled: true
    extra:
      secret: "replace-with-a-long-random-secret"
      callback_url: "https://notionagent.example.com/hermes/callback"
```

## HMAC Signing

Every request body is signed exactly as sent on the wire:

```text
X-NotionAgent-Signature: sha256=<hex-hmac-sha256>
```

The HMAC input is the raw request body bytes. Do not parse and reserialize JSON before verifying the signature.

Python example:

```python
import hashlib
import hmac

signature = "sha256=" + hmac.new(
    secret.encode("utf-8"),
    raw_body,
    hashlib.sha256,
).hexdigest()
```

## JSON Contract

### Inbound: NotionAgent to Hermes

```http
POST /notionagent/in
Content-Type: application/json
X-NotionAgent-Signature: sha256=<hex>
```

```json
{
  "session_id": "memo-abc123",
  "text": "Summarize this transcript"
}
```

`session_id` is the chat ID in Hermes. Use one stable value per memo, task, or conversation thread.

Successful requests return `202 Accepted`:

```json
{
  "status": "accepted",
  "session_id": "memo-abc123",
  "message_id": "generated-message-id"
}
```

### Outbound: Hermes to NotionAgent

Hermes posts to `NOTIONAGENT_CALLBACK_URL`:

```http
POST /your/callback
Content-Type: application/json
X-NotionAgent-Signature: sha256=<hex>
```

```json
{
  "session_id": "memo-abc123",
  "text": "Here is the summary...",
  "message_id": "hermes-generated-uuid"
}
```

Verify this request with the same HMAC rule and shared secret.

## Sample curl

```bash
export BODY='{"session_id":"memo-abc123","text":"Say hello from Hermes"}'
sig=$(python - <<'PY'
import hashlib, hmac, os
body = os.environ["BODY"].encode("utf-8")
secret = os.environ["NOTIONAGENT_SECRET"].encode("utf-8")
print("sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest())
PY
)

curl -i \
  -X POST http://localhost:8645/notionagent/in \
  -H "Content-Type: application/json" \
  -H "X-NotionAgent-Signature: $sig" \
  --data "$BODY"
```

Run it as:

```bash
BODY='{"session_id":"memo-abc123","text":"Say hello from Hermes"}' \
NOTIONAGENT_SECRET='replace-with-a-long-random-secret' \
bash ./send-test.sh
```

## Embedding in Your Own Web App

You can use this adapter from any web app that can make server-side HTTP requests.

The browser should not hold `NOTIONAGENT_SECRET`. Keep the secret on your backend:

1. Browser sends user text to your app server.
2. Your server creates `{session_id, text}` JSON.
3. Your server signs the raw JSON body.
4. Your server POSTs to Hermes.
5. Hermes POSTs the signed `{session_id, text, message_id}` reply to your callback URL.
6. Your server verifies the signature and streams or stores the reply for the browser UI.

Minimal backend contract:

| Direction | Body |
|-----------|------|
| App to Hermes | `{ "session_id": "<stable-id>", "text": "<user text>" }` |
| Hermes to App | `{ "session_id": "<stable-id>", "text": "<reply>", "message_id": "<uuid>" }` |

## Cron and send_message

Set a home session ID to deliver proactive messages:

```bash
NOTIONAGENT_HOME_CHANNEL=memo-abc123
```

Then use:

```text
send_message target="notionagent:memo-abc123" message="Build finished"
```

Cron jobs can deliver to NotionAgent with:

```text
deliver="notionagent:memo-abc123"
```

## Troubleshooting

### 401 Invalid signature

Check that both sides use the same secret and compute HMAC over the exact raw body bytes. Pretty-printing or changing JSON whitespace changes the signature.

### 400 session_id and text are required

The inbound JSON must include non-empty `session_id` and `text` fields.

### Hermes does not start the adapter

Both `NOTIONAGENT_SECRET` and `NOTIONAGENT_CALLBACK_URL` must be set, and `platforms.notionagent.enabled` must be true when using `config.yaml`.

### Callback delivery fails

Check `~/.hermes/logs/gateway.log` for the HTTP status code from the NotionAgent callback endpoint. Make sure the callback URL is reachable from the Hermes host and verifies `X-NotionAgent-Signature`.

### Port already in use

Set a different port:

```bash
NOTIONAGENT_PORT=8655
```

or:

```yaml
platforms:
  notionagent:
    enabled: true
    port: 8655
```

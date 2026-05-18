---
sidebar_position: 20
title: "NATS"
description: "Expose Hermes Agent over NATS using the NATS Agent Protocol for programmatic, request/reply access with streaming responses"
---

# NATS Setup

Hermes Agent can expose itself over [NATS](https://nats.io/) using the **NATS Agent Protocol v0.2**. Instead of a chat app, callers are programs (or other agents) that publish prompts to a well-known subject and iterate streamed responses back. The gateway appears on NATS as a micro service at `agents.hermes.<owner>.<name>`, with heartbeats, discovery via `$SRV.PING`, and mid-stream approval queries.

Unlike Telegram / Slack / Discord, there's no chat UI and no user allowlist — authorization is delegated to the NATS server layer (accounts / NKey / JWT / TLS), the same pattern used by Webhooks (HMAC) and Home Assistant (HASS_TOKEN).

:::info When to use this
Use the NATS gateway when you want to reach Hermes programmatically from other services, embed the agent into an event-driven pipeline, or have one agent call another agent over NATS. For human-facing chat, pick one of the other messaging platforms.
:::

## Prerequisites

- A running NATS server (local or remote). For local testing: `brew install nats-server` then `nats-server -p 4222 -a 127.0.0.1`.
- The `natsagent` SDK. **Until it ships on PyPI, install from source:**
  ```bash
  source venv/bin/activate
  uv pip install --python venv/bin/python -e ../synadia-agents/client-sdk/python
  ```
  Without the SDK, the gateway logs `NATS: natsagent SDK not installed` at startup and does not register the adapter.
- An LLM provider key in `~/.hermes/.env` (e.g. `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). The `/help` and `/status` commands work without one, but actual prompts need a model.

## Step 1: Configure the Gateway

### Option A: Environment variables (fastest for local testing)

Add to `~/.hermes/.env`:

```bash
NATS_URL=nats://127.0.0.1:4222
HERMES_NATS_OWNER=yourname
HERMES_NATS_NAME=gateway
```

Any NATS env var sets `enabled=true` automatically. `HERMES_NATS_AGENT` defaults to `hermes`; override it if you want a different service family name.

### Option B: `config.yaml`

```yaml
platforms:
  nats:
    enabled: true
    extra:
      # One of `servers` or `context` is required.
      servers: ["nats://127.0.0.1:4222"]
      # OR
      # context: "local-nats"       # reads $NATS_CONFIG_HOME/context/<name>.json

      # Identity on the wire — produces subject agents.hermes.<owner>.<name>
      agent: hermes                  # default "hermes"; rarely changed
      owner: yourname                # required
      name: gateway                  # required

      # Behavior tuning (all optional, defaults shown)
      session_default: "default"     # session fallback
      heartbeat_interval_s: 30
      max_payload: "1MB"
      attachments_ok: true
      ack_keepalive_interval_s: 20
```

### Option C: NATS context

If you already manage NATS credentials via `nats context`, set `extra.context` to the context name and omit `servers`:

```yaml
platforms:
  nats:
    enabled: true
    extra:
      context: "my-synadia-cloud-ctx"
      owner: yourname
      name: gateway
```

## Step 2: Start the Gateway

```bash
hermes gateway run
```

On success, the log shows:

```
NATS: connected to nats://127.0.0.1:4222
NATS: registered as agents.hermes.yourname.gateway (heartbeat=30s, max_payload=1MB)
```

Verify the micro service is live:

```bash
nats req '$SRV.INFO.agents' '' --replies=0 --timeout=2s
```

You should see a JSON response listing the `prompt` endpoint with metadata `max_payload=1MB attachments_ok=true` and your identity `hermes/yourname`.

Subscribe to heartbeats:

```bash
nats sub 'agents.hermes.*.*.heartbeat'
```

One frame should arrive every `heartbeat_interval_s` seconds.

## Step 3: Send a Prompt

The `natsagent` SDK ships runnable examples. From the SDK repo:

```bash
cd ../synadia-agents/client-sdk/python
uv run python examples/02-prompt-text.py \
    --url nats://127.0.0.1:4222 \
    "what is 2+2? answer in one short sentence"
```

You'll see the response stream chunk-by-chunk, terminated by an empty-body frame. Other examples:

| Example | Demonstrates |
|---------|--------------|
| `examples/01-discover.py` | List all live agents registered on the NATS server via `$SRV` |
| `examples/02-prompt-text.py` | Send a plain text prompt and iterate the streamed response |
| `examples/03-prompt-attachment.py` | Send an image or document as a base64 attachment in the envelope |
| `examples/04-query-reply.py` | Handle a mid-stream approval query (tool call needs confirmation) |
| `examples/05-liveness.py` | Monitor heartbeats to detect when an agent goes offline |

## How It Works

### Subject layout

| Subject | Direction | Purpose |
|---------|-----------|---------|
| `agents.hermes.<owner>.<name>` | inbound | Prompt endpoint — publish an `Envelope` with `prompt` + optional `attachments` |
| `agents.hermes.<owner>.<name>.heartbeat` | outbound | Liveness beacon every `heartbeat_interval_s` seconds |
| `$SRV.PING.agents`, `$SRV.INFO.agents[.{id}]` | both | NATS micro service discovery |

### Sessions

The gateway treats each caller-supplied `session` field in the envelope (protocol §5.1) as an independent conversation. Callers that don't set the field share the default session. Sessions are isolated from each other — one caller's history doesn't leak to another. Under the hood, `session=foo` produces a gateway session key of `agent:main:nats:dm:foo`. In the Python SDK, pass the field via `remote.prompt(text, session="foo")`; the examples expose it as `--session NAME`.

### Attachments

Attach files inline as base64 in the envelope's `attachments` array. The gateway:

- Routes images (`.png`, `.jpg`, `.webp`, …) through `vision_analyze` so the agent sees the description automatically.
- Routes audio (`.wav`, `.mp3`, `.ogg`, …) with a path-note so the agent can call the transcription tool.
- Routes documents (`.pdf`, `.txt`, `.md`, …) with a path-note so the agent can call `read_file`.

Max envelope size is `max_payload` (default 1 MB); larger files get rejected with a 400 error.

### Streaming model

Every token chunk is a separate publish on the caller's reply subject, not an edit of one message. Each `chunk.data` is the delta text. Empty-body chunks signal end-of-response. The gateway emits a `status:ack` frame every 20 s while otherwise silent, so the caller's `60 s` inactivity timeout (protocol §6.6) stays well-fed.

### Mid-stream approvals

If a dangerous command needs approval (e.g. `rm -rf`), the gateway sends a `query` chunk instead of dropping the stream. Respond to the `stream.ask(...)` with `once`, `session`, `always`, or `deny`; the agent resumes. See `examples/04-query-reply.py` for the caller side and the `request_interaction` mechanism in the design doc for the adapter side.

### Slash commands

All of Hermes's gateway-eligible slash commands work over NATS as plain text prompts:

```bash
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 "/help"
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 "/status"
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 "/new"
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 "/model"
```

The full list: `/new`, `/reset`, `/model`, `/provider`, `/status`, `/stop`, `/help`, `/compress`, `/resume`, `/usage`, `/insights`, `/reasoning`, `/title`, `/rollback`, `/background`, `/reload-mcp`, and any installed skill command. Output arrives as normal streamed chunks.

## Security Model

NATS authorization is **layered at the NATS server**, not inside Hermes. The gateway does not consult `TELEGRAM_ALLOWED_USERS`-style allowlists for NATS callers — if a client can publish to `agents.hermes.<owner>.<name>`, the gateway treats them as authorized.

In practice this means:

- On a local dev NATS with no auth: anyone who can reach the port can prompt the gateway. Keep it on `127.0.0.1` unless you've configured accounts.
- On Synadia Cloud / a production NATS cluster: restrict publish permissions on the `agents.hermes.<owner>.<name>` subject to the accounts / NKey / JWT principals that should be allowed to call the agent.
- TLS + mutual TLS is the recommended production posture.

Dangerous commands still require mid-stream approval (`/approve`-style flow via `request_interaction`), so the damage a caller can do without interactive consent is bounded by the gateway's own `tools/approval.py` policy.

## Profile Isolation

Hermes profiles are fully isolated — each profile gets its own `HERMES_HOME` and can register its own NATS identity. Running two profiles that try to claim the same `(agent, owner, name)` triple on one machine is a footgun (both would receive load-balanced prompts); the gateway acquires a scoped lock on the identity before calling `natsagent.connect()`, so the second profile fails fast with an actionable error.

Cross-machine collisions are allowed — the NATS protocol explicitly permits multiple instances per identity (§3.3) for high availability.

## Troubleshooting

**Gateway startup: `NATS: natsagent SDK not installed`**
The SDK isn't on PyPI yet. Install from source: `uv pip install --python venv/bin/python -e ../synadia-agents/client-sdk/python`.

**`ModuleNotFoundError: No module named 'natsagent'`**
Same as above. Make sure you installed into the gateway's venv, not a global Python.

**`ValueError: could not parse max_payload 'foo'`**
Set `max_payload` to a value matching the pattern `^\d+(B|KB|MB|GB)$` — e.g. `"1MB"`, `"512KB"`, `"104857600B"`.

**Gateway starts but discovery shows nothing**
Check that `enabled=true` and one of `servers` / `context` is set. The gateway logs `get_connected_platforms()` status at startup; if NATS isn't in the connected list, inspect the config.

**Caller hangs after first chunk; `is_online()` shows False**
The gateway probably crashed or lost its NATS connection. The protocol marks an agent offline after three missed heartbeats (~90 s at the 30 s default). Check the gateway log.

**Agent replies "I don't see an image attached"**
This was a Phase 8 bug fixed in the shipping code — the adapter-owned agent path dropped `media_urls`. If you're running a pre-release snapshot, update to a build that includes `NatsAdapter._enrich_event_with_media` (commit after `febf7ba0`).

**Dangerous command hangs for 5 minutes**
If the caller doesn't handle the `query` chunk, `stream.ask(...)` times out at `gateway_timeout` (default 300 s) and the command is denied. Make sure your caller drains the prompt stream's async iterator and responds to query frames — see `examples/04-query-reply.py`.

## Non-Goals (MVP scope)

The v0.2 adapter does **not** support:

- Cron-based proactive delivery (NATS has no persistent reply address for a cron job to target)
- `send_message` tool routing to NATS (same reason)
- The future `attachments` endpoint for chunked uploads >1 MB (inline base64 only)
- JetStream at-least-once delivery
- End-to-end encryption (delegated to NATS server TLS)
- `/stop` interrupting a running NATS agent (the adapter-owned agent pattern bypasses the gateway's `_active_sessions` tracking; callers can drop their subscription to abandon a run)

Each is a candidate for a future phase, not a bug.

## Reference

- **Protocol spec:** `../nats-agent-sdk-docs/core-protocol.md` (v0.2.0-draft)
- **Agent SDK:** `../synadia-agents/client-sdk/python` (package `natsagent`; lives inside the [`synadia-ai/synadia-agents`](https://github.com/synadia-ai/synadia-agents) monorepo)
- **Hermes adapter:** `gateway/platforms/nats.py`
- **Design doc:** `docs/nats-gateway-design.md` — architectural reference, protocol↔adapter mapping, streaming model, failure modes
- **Lessons learned:** `docs/nats-gateway-design.md` §17 — retrospective on surprises during Phases 1–8

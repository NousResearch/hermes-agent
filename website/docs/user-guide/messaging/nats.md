---
sidebar_position: 20
title: "NATS"
description: "Expose Hermes Agent over NATS using the NATS Agent Protocol for programmatic, request/reply access with streaming responses"
---

# NATS Setup

Hermes Agent can expose itself over [NATS](https://nats.io/) using the **NATS Agent Protocol v0.3**. Instead of a chat app, callers are programs (or other agents) that publish prompts to a well-known subject and iterate streamed responses back. The gateway appears on NATS as a micro service at `agents.prompt.hermes.<owner>.<session_name>`, with heartbeats, a `agents.status` request endpoint, discovery via `$SRV.PING`, and mid-stream approval queries.

Unlike Telegram / Slack / Discord, there's no chat UI and no user allowlist — authorization is delegated to the NATS server layer (accounts / NKey / JWT / TLS), the same pattern used by Webhooks (HMAC) and Home Assistant (HASS_TOKEN).

:::info When to use this
Use the NATS gateway when you want to reach Hermes programmatically from other services, embed the agent into an event-driven pipeline, or have one agent call another agent over NATS. For human-facing chat, pick one of the other messaging platforms.
:::

## Prerequisites

- A running NATS server (local or remote). For local testing: `brew install nats-server` then `nats-server -p 4222 -a 127.0.0.1`.
- The `synadia-ai-agents` and `synadia-ai-agent-service` SDKs (host-side imports `synadia_ai.agents` and `synadia_ai.agent_service`). Both are pulled in automatically by the `[nats]` extra: `pip install 'hermes-agent[nats]'` (or `uv sync --extra nats`). Without them, the gateway logs `NATS: synadia-ai-agents / synadia-ai-agent-service not installed` at startup and does not register the adapter.
- An LLM provider key in `~/.hermes/.env` (e.g. `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`). The `/help` and `/status` commands work without one, but actual prompts need a model.

## Step 1: Configure the Gateway

### Recommended: the interactive wizard

```bash
hermes setup gateway
```

Pick **NATS** from the platform checklist. The wizard offers a 3-way transport menu (public demo server / custom URL / existing NATS CLI context auto-discovered from `~/.config/nats/context/`), prompts for owner and session_name, runs a cross-profile collision check on `(agent, owner, session_name)`, and writes the result to `~/.hermes/.env`:

```bash
NATS_URL=nats://demo.nats.io      # OR NATS_CONTEXT=local-nats
HERMES_NATS_OWNER=yourname
HERMES_NATS_SESSION_NAME=default
```

### Manual: edit `.env` directly

If you'd rather skip the wizard, set the same vars by hand:

```bash
NATS_URL=nats://127.0.0.1:4222
HERMES_NATS_OWNER=yourname
HERMES_NATS_SESSION_NAME=default
```

`HERMES_NATS_AGENT` defaults to `hermes`; set it only if you want a different service family name. Any NATS env var sets `enabled=true` automatically.

### Advanced: structured overrides via `config.yaml`

For knobs the wizard doesn't ask about — multi-URL `servers` lists, custom heartbeat interval, payload limits, ack-keepalive timing — hand-edit `~/.hermes/config.yaml`. Env vars stamp on top per-key, so you can mix and match:

```yaml
platforms:
  nats:
    enabled: true
    extra:
      # Multi-URL is config.yaml-only (NATS_URL is single-URL).
      servers: ["nats://primary:4222", "nats://failover:4222"]

      # Behavior tuning (all optional, defaults shown).
      heartbeat_interval_s: 30
      max_payload: "1MB"
      attachments_ok: true
      ack_keepalive_interval_s: 20
```

If you already manage NATS credentials via `nats context`, set `NATS_CONTEXT` (env) or `extra.context` (yaml) instead of `NATS_URL` / `extra.servers`. The wizard surfaces existing contexts as one of the 3 transport options automatically.

### Multiple sessions

Protocol v0.3 collapsed `name` and `session` into a single `session_name` token: one `AgentService` serves exactly one session. To run multiple sessions on the same machine, use Hermes profiles — one profile per session:

```bash
hermes -p alice profile create
hermes -p alice setup gateway    # pick NATS, set session_name=alice
hermes -p bob profile create
hermes -p bob setup gateway      # pick NATS, set session_name=bob

# Each profile gets its own .env, AgentService, and session_name token.
```

## Step 2: Start the Gateway

```bash
hermes gateway run
```

On success, the log shows:

```
NATS: connected to nats://127.0.0.1:4222
NATS: subscribed at agents.prompt.hermes.yourname.default (heartbeat=30s, max_payload=1MB)
```

Verify the micro service is live:

```bash
nats micro list
```

You should see one `agents` service with two endpoints (`prompt`, `status`).

Subscribe to heartbeats:

```bash
nats sub 'agents.hb.>'
```

One frame should arrive every `heartbeat_interval_s` seconds on `agents.hb.hermes.<owner>.<session_name>`.

Query liveness via the request endpoint:

```bash
nats req agents.status.hermes.yourname.default ''
```

A heartbeat-shaped JSON reply with `metadata.protocol_version: "0.3"` confirms the agent is live.

## Step 3: Send a Prompt

The simplest caller is a few lines of Python on top of the `synadia-ai-agents` client SDK. After `pip install synadia-ai-agents`:

```python
# prompt.py
import asyncio, sys
from synadia_ai.agents import Agents, DiscoverFilter, ResponseChunk
import nats

async def main(text: str) -> None:
    nc = await nats.connect("nats://127.0.0.1:4222")
    agents = Agents(nc=nc)
    try:
        found = await agents.discover(filter=DiscoverFilter(session_name="default"))
        if not found:
            sys.exit("no agent found — is the gateway running?")
        async for msg in found[0].prompt(text):
            if isinstance(msg, ResponseChunk):
                sys.stdout.write(msg.text); sys.stdout.flush()
    finally:
        await agents.close()
        await nc.close()

asyncio.run(main("what is 2+2? answer in one short sentence"))
```

You'll see the response stream chunk-by-chunk, terminated by an empty-body frame.

For a full set of runnable callers — discovery, attachments, mid-stream approval handling, liveness monitoring — clone the SDK repo and run them directly:

```bash
git clone https://github.com/synadia-ai/synadia-agents.git
cd synadia-agents/client-sdk/python
uv run python examples/02-prompt-text.py \
    --url nats://127.0.0.1:4222 \
    --session default \
    "what is 2+2? answer in one short sentence"
```

| Example | Demonstrates |
|---------|--------------|
| `examples/01-discover.py` | List all live agents registered on the NATS server via `$SRV` |
| `examples/02-prompt-text.py` | Send a plain text prompt and iterate the streamed response |
| `examples/03-prompt-attachment.py` | Send an image or document as a base64 attachment in the envelope |
| `examples/04-query-reply.py` | Handle a mid-stream approval query (tool call needs confirmation) |
| `examples/05-liveness.py` | Monitor heartbeats / `agents.status` to detect when an agent goes offline |

## How It Works

### Subject layout (v0.3 verb-first)

| Subject | Direction | Purpose |
|---------|-----------|---------|
| `agents.prompt.hermes.<owner>.<session_name>` | inbound | Prompt endpoint — publish an `Envelope` with `prompt` + optional `attachments` |
| `agents.status.hermes.<owner>.<session_name>` | request/reply | On-demand liveness — request returns the current heartbeat payload |
| `agents.hb.hermes.<owner>.<session_name>` | outbound | Liveness beacon every `heartbeat_interval_s` seconds |
| `$SRV.PING.agents`, `$SRV.INFO.agents[.{id}]` | both | NATS micro service discovery |

Caller-side reply inboxes are pinned to the `_INBOX.agents` prefix (v0.3 PR #25), which simplifies NATS account permissions — grant `_INBOX.agents.>` to caller principals so they can receive responses.

### Sessions

The 5th subject token IS the session — there is no separate `session` field on the envelope. One `AgentService` corresponds to one `session_name`; use multiple Hermes profiles for multiple concurrent sessions on the same host.

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
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 --session default "/help"
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 --session default "/status"
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 --session default "/new"
uv run python examples/02-prompt-text.py --url nats://127.0.0.1:4222 --session default "/model"
```

The full list: `/new`, `/reset`, `/model`, `/provider`, `/status`, `/stop`, `/help`, `/compress`, `/resume`, `/usage`, `/insights`, `/reasoning`, `/title`, `/rollback`, `/background`, `/reload-mcp`, and any installed skill command. Output arrives as normal streamed chunks.

## Security Model

NATS authorization is **layered at the NATS server**, not inside Hermes. The gateway does not consult `TELEGRAM_ALLOWED_USERS`-style allowlists for NATS callers — if a client can publish to `agents.prompt.hermes.<owner>.<session_name>`, the gateway treats them as authorized.

In practice this means:

- On a local dev NATS with no auth: anyone who can reach the port can prompt the gateway. Keep it on `127.0.0.1` unless you've configured accounts.
- On Synadia Cloud / a production NATS cluster: restrict publish permissions on the `agents.prompt.hermes.<owner>.<session_name>` subject to the accounts / NKey / JWT principals that should be allowed to call the agent. Grant `_INBOX.agents.>` for replies.
- TLS + mutual TLS is the recommended production posture.

Dangerous commands still require mid-stream approval (`/approve`-style flow via `request_interaction`), so the damage a caller can do without interactive consent is bounded by the gateway's own `tools/approval.py` policy.

## Profile Isolation

Hermes profiles are fully isolated — each profile gets its own `HERMES_HOME` and can register its own NATS identity. Running two profiles that try to claim the same `(agent, owner, session_name)` triple on one machine is a footgun (both would receive load-balanced prompts); the gateway acquires a scoped lock on the identity before calling `nats.connect()`, so the second profile fails fast with an actionable error.

Cross-machine collisions are allowed — the NATS protocol explicitly permits multiple instances per identity (§3.3) for high availability.

## Troubleshooting

**Gateway startup: `NATS: synadia-ai-agents / synadia-ai-agent-service not installed`**
Install the `[nats]` extra: `pip install 'hermes-agent[nats]'` (or `uv sync --extra nats`).

**`ModuleNotFoundError: No module named 'synadia_ai'`**
Same as above. Make sure you installed into the gateway's venv, not a global Python.

**`ValueError: could not parse max_payload 'foo'`**
Set `max_payload` to a value matching the pattern `^\d+(B|KB|MB|GB)$` — e.g. `"1MB"`, `"512KB"`, `"104857600B"`.

**Gateway starts but discovery shows nothing**
Check that `enabled=true` and one of `servers` / `context` is set. The gateway logs `get_connected_platforms()` status at startup; if NATS isn't in the connected list, inspect the config.

**Caller hangs after first chunk; `is_online()` shows False**
The gateway probably crashed or lost its NATS connection. The protocol marks an agent offline after three missed heartbeats (~90 s at the 30 s default). Check the gateway log, or query `agents.status.hermes.<owner>.<session_name>` directly.

**Dangerous command hangs for 5 minutes**
If the caller doesn't handle the `query` chunk, `stream.ask(...)` times out at `gateway_timeout` (default 300 s) and the command is denied. Make sure your caller drains the prompt stream's async iterator and responds to query frames — see `examples/04-query-reply.py`.

## Non-Goals (MVP scope)

The v0.3 adapter does **not** support:

- Cron-based proactive delivery (NATS has no persistent reply address for a cron job to target)
- `send_message` tool routing to NATS (same reason)
- The future `attachments` endpoint for chunked uploads >1 MB (inline base64 only)
- JetStream at-least-once delivery
- End-to-end encryption (delegated to NATS server TLS)
- `/stop` interrupting a running NATS agent (the adapter-owned agent pattern bypasses the gateway's `_active_sessions` tracking; callers can drop their subscription to abandon a run)
- Multi-session multiplexing within one process (one `AgentService` = one `session_name`; use profiles for multi-session deployments)

Each is a candidate for a future phase, not a bug.

## Reference

- **Protocol spec:** `../nats-agent-sdk-docs/core-protocol.md` (v0.3)
- **Client SDK (caller side):** [`synadia-ai-agents`](https://pypi.org/project/synadia-ai-agents/) on PyPI (import root `synadia_ai.agents`)
- **Agent SDK (gateway side):** [`synadia-ai-agent-service`](https://pypi.org/project/synadia-ai-agent-service/) on PyPI (import root `synadia_ai.agent_service`)
- **SDK source:** [`synadia-ai/synadia-agents`](https://github.com/synadia-ai/synadia-agents) monorepo (`client-sdk/python`, `agent-sdk/python`, plus `examples/`)
- **Hermes adapter:** `gateway/platforms/nats.py`
- **Design doc:** `docs/nats-gateway-design.md` — architectural reference, protocol↔adapter mapping, streaming model, failure modes
- **Lessons learned:** `docs/nats-gateway-design.md` §17 — retrospective on surprises during Phases 1–8

---
sidebar_position: 8
title: "AAMP"
description: "Set up Hermes Agent as an AAMP mailbox agent via aamp-sdk and MeshMail-compatible services"
---

# AAMP Setup

Hermes can run as an [AAMP](https://pypi.org/project/aamp-sdk/) mailbox agent. Other AAMP agents or dispatchers send Hermes a `task.dispatch` email, Hermes acknowledges it with `task.ack`, runs the request, and returns a final `task.result`.

This is useful when you want Hermes to participate in agent-to-agent workflows over mailbox infrastructure such as `meshmail.ai`, without exposing a chat bot token or webhook endpoint.

:::info Python SDK
The Hermes AAMP adapter uses the published PyPI package `aamp-sdk==0.1.0`. If you install Hermes with the normal messaging extras, the SDK is included automatically.
:::

---

## Prerequisites

- **An AAMP service endpoint** — for example `https://meshmail.ai`
- **Hermes installed with messaging support**
- Optional: **a dedicated peer mailbox** for cron results and notifications

If you're installing from source, include the messaging extra:

```bash
uv pip install -e ".[messaging,dev]"
```

---

## Quick Setup

The easiest path is the gateway wizard:

```bash
hermes gateway setup
```

Select **AAMP** from the platform list.

The wizard will:

1. default the service URL to `https://meshmail.ai` if you leave it blank
2. auto-register a Hermes mailbox with the default slug `hermes`
3. optionally configure a coarse sender allowlist with `AAMP_ALLOWED_USERS`
4. optionally configure exact-match sender policies with `AAMP_SENDER_POLICIES`
5. optionally configure `AAMP_HOME_CHANNEL` for cron job and notification delivery

At the end of setup, Hermes prints the registered mailbox address so you know where to send `task.dispatch` requests.

Example:

```text
Hermes AAMP mailbox: hermes-abc12345@meshmail.ai
Send AAMP tasks to: hermes-abc12345@meshmail.ai
```

---

## Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
# Required in the common case
AAMP_BASE_URL=https://meshmail.ai

# Security (recommended)
AAMP_ALLOWED_USERS=dispatch@meshmail.ai,ops@meshmail.ai

# Optional: exact sender + dispatch-context rules
AAMP_SENDER_POLICIES=[{"sender":"dispatch@meshmail.ai","dispatchContextRules":{"tenant":["acme"],"workflow":["prod"]}}]

# Optional: default delivery target for cron jobs / notifications
AAMP_HOME_CHANNEL=ops@meshmail.ai
```

That's enough for most users.

Hermes auto-registers the mailbox on first run and caches the identity locally.

### Advanced: Explicit mailbox credentials

If you already have an AAMP mailbox identity and do not want Hermes to auto-register one, you can provide explicit credentials instead:

```bash
AAMP_BASE_URL=https://meshmail.ai
AAMP_EMAIL=hermes-existing@meshmail.ai
AAMP_PASSWORD=your_smtp_password
```

Or use the mailbox token form:

```bash
AAMP_MAILBOX_TOKEN=base64_email_colon_password
```

---

## Start the Gateway

```bash
hermes gateway              # Run in foreground
hermes gateway install      # Install as a user service
sudo hermes gateway install --system   # Linux only: boot-time system service
```

On startup, Hermes:

1. resolves or auto-registers its AAMP mailbox identity
2. connects through the Python AAMP SDK
3. receives inbound `task.dispatch` messages over JMAP push when available
4. falls back to SDK-managed polling/reconnect behavior when needed

---

## How Hermes Behaves on AAMP

### Inbound flow

When Hermes receives a `task.dispatch`:

1. it normalizes the AAMP task into a gateway message event
2. sends `task.ack`
3. runs the prompt through the normal agent loop
4. returns a single final `task.result`

### No progressive streaming

AAMP is treated as a mailbox transport, not a live chat transport.

Hermes intentionally does **not** send:

- token-by-token streaming updates
- interim commentary messages
- tool progress messages

This keeps each task reply as one final result instead of multiple partial mailbox messages.

### Home channel behavior

`AAMP_HOME_CHANNEL` is the default **peer mailbox** for proactive outbound delivery:

- cron job results
- notifications
- bare `send_message` deliveries targeting `aamp`

It is **not** Hermes' own mailbox identity.

---

## Access Control

Hermes supports two AAMP-specific authorization styles.

### 1. Coarse allowlist

```bash
AAMP_ALLOWED_USERS=dispatch@meshmail.ai,ops@meshmail.ai
```

This allows any inbound AAMP task from those sender mailboxes.

### 2. Fine-grained sender policies

```bash
AAMP_SENDER_POLICIES=[{"sender":"dispatch@meshmail.ai","dispatchContextRules":{"tenant":["acme"],"workflow":["prod"]}}]
```

When `AAMP_SENDER_POLICIES` is set, Hermes matches:

- the sender mailbox
- exact `X-AAMP-Dispatch-Context` key/value pairs

If a sender is not authorized, Hermes returns a rejected `task.result` instead of entering the normal agent flow.

:::warning
For AAMP, prefer `AAMP_ALLOWED_USERS` or `AAMP_SENDER_POLICIES`. Unlike chat-native platforms, the safest setup is explicit mailbox authorization rather than relying on a DM pairing flow.
:::

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **No mailbox shown after setup** | Start the gateway once. Hermes may defer auto-registration until first connection, then cache the identity in `~/.hermes/aamp/mailbox_identity.json`. |
| **Replies are slow** | AAMP uses mailbox delivery, so response time includes agent runtime plus mail transport latency. If JMAP push is unavailable, the SDK may temporarily fall back to polling. |
| **Unauthorized sender gets rejected** | Check `AAMP_ALLOWED_USERS` or `AAMP_SENDER_POLICIES`. Sender policies require exact matches for the sender and each configured dispatch-context key. |
| **TLS / certificate issues on a self-hosted AAMP service** | For development only, set `AAMP_REJECT_UNAUTHORIZED=false` to disable TLS certificate verification in the SDK. |
| **Cron jobs do not deliver anywhere** | Set `AAMP_HOME_CHANNEL` to the peer mailbox that should receive proactive Hermes output. |

---

## Security Notes

- Hermes caches its AAMP mailbox identity in `~/.hermes/aamp/mailbox_identity.json`
- Protect `~/.hermes/.env` and the cached identity file like credentials
- Keep `AAMP_REJECT_UNAUTHORIZED=true` in normal deployments so TLS certificates are verified
- Use sender allowlists or sender policies for any mailbox that can trigger terminal-capable agents

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AAMP_BASE_URL` | Yes | `https://meshmail.ai` | AAMP management-service base URL. `AAMP_HOST` is also supported as an alias. |
| `AAMP_ALLOWED_USERS` | No | — | Comma-separated peer mailbox addresses allowed to send tasks |
| `AAMP_SENDER_POLICIES` | No | — | JSON sender policy rules using sender + exact `X-AAMP-Dispatch-Context` matching |
| `AAMP_HOME_CHANNEL` | No | — | Default peer mailbox for cron results and notifications |
| `AAMP_EMAIL` | No | — | Explicit mailbox email, if you want to skip auto-registration |
| `AAMP_PASSWORD` | No | — | Explicit SMTP password paired with `AAMP_EMAIL` |
| `AAMP_MAILBOX_TOKEN` | No | — | Base64 `email:password` mailbox token |
| `AAMP_CREDENTIALS_FILE` | No | `~/.hermes/aamp/mailbox_identity.json` | Path to the cached mailbox identity JSON |
| `AAMP_POLL_INTERVAL` | No | `10` | SDK reconnect / polling-fallback interval in seconds |
| `AAMP_REJECT_UNAUTHORIZED` | No | `true` | Verify TLS certificates for AAMP SDK HTTP / WebSocket / SMTP connections |
| `AAMP_DESCRIPTION` | No | `Hermes Agent AAMP mailbox` | Mailbox description used during auto-registration |
| `AAMP_SLUG` | No | `hermes` | Advanced override for the auto-registration slug |
| `AAMP_ALLOW_ALL_USERS` | No | `false` | Allow all sender mailboxes (not recommended) |

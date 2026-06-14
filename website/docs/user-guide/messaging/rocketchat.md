---
sidebar_position: 9
title: "Rocket.Chat"
description: "Set up Hermes Agent as a Rocket.Chat bot with PAT auth or browser-assisted OAuth bootstrap"
---

# Rocket.Chat Setup

Hermes Agent can run as a Rocket.Chat bot through a bundled platform plugin. The adapter uses the Rocket.Chat REST API for outbound sends, edits, uploads, and cron delivery, and DDP/WebSocket for inbound messages and live room subscriptions.

> Run `hermes gateway setup` and pick **Rocket.Chat** for a guided setup.

## Overview

| Component | Value |
|-----------|-------|
| **Inbound transport** | DDP WebSocket (`/websocket`) |
| **Outbound transport** | REST API (`/api/v1/...`) |
| **Supported spaces** | DMs, public rooms, private rooms, existing threads |
| **Preferred runtime auth** | Personal Access Token (PAT) + `userId` |
| **Bootstrap auth** | Browser-assisted OAuth/OIDC bootstrap that persists Rocket.Chat runtime credentials |
| **Not supported** | Username/password login, typing indicators, voice, auto-created threads |

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| **DMs** | Hermes responds to every authorized message. |
| **Public/private rooms** | By default, Hermes requires an `@mention` unless the room is in `free_response_rooms`. |
| **Threads** | Existing Rocket.Chat `tmid` values are preserved as Hermes thread IDs. Hermes replies inside the same thread when the incoming message came from one. |
| **Shared rooms** | Hermes keeps the existing per-user isolation model by default, so users do not silently share one transcript unless you disable that globally. |

## Authentication Options

### Option A: PAT or existing runtime token

This is the simplest and most stable option for a homelab deployment.

Set:

```bash
ROCKETCHAT_URL=https://chat.example.com
ROCKETCHAT_USER_ID=your-user-id
ROCKETCHAT_AUTH_TOKEN=your-personal-access-token
```

Use a PAT when possible. If you use a regular Rocket.Chat `authToken` instead, it may expire and require a fresh bootstrap later.

### Option B: Browser-assisted OAuth / OIDC bootstrap

If your workspace already uses OIDC or another OAuth login provider in the browser, Hermes can do a one-time bootstrap and persist Rocket.Chat runtime credentials for headless use later.

Set the bootstrap config:

```bash
ROCKETCHAT_URL=https://chat.example.com
ROCKETCHAT_BOOTSTRAP_ENABLED=true
ROCKETCHAT_BOOTSTRAP_ARTIFACT=~/.hermes/rocketchat_auth.json

# Upstream provider / Rocket.Chat login exchange
ROCKETCHAT_OAUTH_SERVICE_NAME=oidc
ROCKETCHAT_OAUTH_AUTHORIZE_URL=https://id.example.com/authorize
ROCKETCHAT_OAUTH_TOKEN_URL=https://id.example.com/token
ROCKETCHAT_OAUTH_CLIENT_ID=...
ROCKETCHAT_OAUTH_CLIENT_SECRET=...

# Optional
ROCKETCHAT_BOOTSTRAP_PAT_NAME=hermes-agent
```

Then run:

```bash
python -m plugins.platforms.rocketchat.auth
```

What the bootstrap does:

1. Opens a browser for the configured OAuth/OIDC provider.
2. Exchanges the returned provider token with Rocket.Chat login.
3. Validates the resulting Rocket.Chat runtime credentials.
4. Tries to create a PAT and store that in the artifact.
5. Falls back to storing Rocket.Chat `authToken` + `userId` if PAT creation is unavailable.

If the workspace does not allow PAT creation, Hermes still works, but you will need to re-run bootstrap when the stored `authToken` expires.

:::warning
Username/password login is intentionally unsupported. Hermes only supports token-based runtime auth.
:::

## Required and optional env vars

Minimum runtime config:

```bash
ROCKETCHAT_URL=https://chat.example.com
ROCKETCHAT_USER_ID=...
ROCKETCHAT_AUTH_TOKEN=...
ROCKETCHAT_ALLOWED_USERS=user-id-1,user-id-2
```

Optional behavior:

```bash
ROCKETCHAT_ALLOW_ALL_USERS=false
ROCKETCHAT_HOME_CHANNEL=GENERAL
ROCKETCHAT_HOME_CHANNEL_NAME=Ops
ROCKETCHAT_HOME_CHANNEL_THREAD_ID=root-thread-id
ROCKETCHAT_REQUIRE_MENTION=true
ROCKETCHAT_FREE_RESPONSE_ROOMS=GENERAL
ROCKETCHAT_ALLOWED_ROOMS=GENERAL,PRIVATEOPS
ROCKETCHAT_COMMAND_PREFIX=!
```

Non-secret behavior can also live in `config.yaml`:

```yaml
rocketchat:
  require_mention: true
  free_response_rooms:
    - GENERAL
  allowed_rooms:
    - GENERAL
    - PRIVATEOPS
  command_prefix: "!"
  bootstrap_enabled: true
  bootstrap_artifact: ~/.hermes/rocketchat_auth.json
  bootstrap_pat_name: hermes-agent
```

Env vars still win over YAML when both are set.

## Slash commands and the `!` fallback

Rocket.Chat sometimes consumes slash-prefixed input before Hermes ever sees it. Hermes therefore supports two command forms:

- `/command` — works when Rocket.Chat delivers the text to Hermes unchanged.
- `!command` — always accepted by the adapter and rewritten to `/command` before it enters Hermes' existing slash-command pipeline.

Use `!new`, `!reset`, `!model`, `!sethome`, and similar commands if you want the portable form that does not depend on Rocket.Chat client behavior.

## Home room / cron delivery

Set the home room manually:

```bash
ROCKETCHAT_HOME_CHANNEL=GENERAL
ROCKETCHAT_HOME_CHANNEL_NAME=Ops
ROCKETCHAT_HOME_CHANNEL_THREAD_ID=root-thread-id   # optional
```

Or run `/sethome` (or `!sethome`) inside the target room. Hermes stores that room as the default delivery destination for cron jobs, restart notifications, and other proactive messages.

## Start the gateway

```bash
hermes gateway
```

Once connected, Hermes listens across the rooms the runtime user can access, applies your allowlists and mention rules, and replies in place.

## Troubleshooting

### Hermes connects but replies only in DMs

Check `ROCKETCHAT_REQUIRE_MENTION`, `ROCKETCHAT_FREE_RESPONSE_ROOMS`, and `ROCKETCHAT_ALLOWED_ROOMS`. Shared rooms require an `@mention` by default.

### Messages stop after a restart when bootstrap is enabled

If the stored credential is a regular `authToken` instead of a PAT, it may have expired. Re-run:

```bash
python -m plugins.platforms.rocketchat.auth
```

### Streaming falls back to a final message

That is expected when Rocket.Chat message editing is blocked by workspace policy or permissions. Hermes still sends the final response.

### Attachment download failed

Hermes logs a warning and continues with the text part of the message. This is a safe fallback, not a hard failure.

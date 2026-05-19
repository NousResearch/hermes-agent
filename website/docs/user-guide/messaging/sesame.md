---
sidebar_position: 28
title: "Sesame"
description: "Connect Hermes Gateway to Sesame using an agent API key and Sesame realtime WebSocket events."
---

# Sesame

Hermes can run as a native Sesame agent through the messaging gateway. The adapter authenticates with a Sesame API key, receives inbound messages over Sesame realtime WebSocket `/v1/connect`, and sends replies/files through the Sesame REST API.

## Requirements

Install the messaging extra so `aiohttp` and `websockets` are available:

```bash
pip install "hermes-agent[messaging]"
```

## Configuration

Run the setup wizard:

```bash
hermes gateway setup
```

Select **Sesame**, then provide:

- `SESAME_API_KEY` — API key for this Hermes agent's Sesame principal.
- `SESAME_HOME_CHANNEL` — optional channel ID for cron jobs and notifications.
- `SESAME_ALLOWED_USERS` — optional comma-separated Sesame principal IDs allowed to talk to the gateway.
- `SESAME_API_URL` — REST API base URL. Defaults to `https://api.sesame.space`.
- `SESAME_WS_URL` — WebSocket base URL. Defaults to `wss://ws.sesame.space`.

You can also set these in `~/.hermes/.env` manually:

```bash
SESAME_API_KEY=sk_...
SESAME_HOME_CHANNEL=<channel-id>
SESAME_ALLOWED_USERS=<principal-id-1>,<principal-id-2>
SESAME_API_URL=https://api.sesame.space
SESAME_WS_URL=wss://ws.sesame.space
```

Restart the gateway after changing credentials:

```bash
hermes gateway restart
```

## Cron delivery

Set `SESAME_HOME_CHANNEL` to make `deliver="sesame"` and origin fallback work for scheduled jobs:

```bash
hermes cron create "every weekday 9am" --deliver sesame
```

## Notes

- Sesame channel IDs can be passed as either raw channel IDs or `sesame:<channel-id>`.
- Replies preserve thread context when Sesame sends a `threadRootId`.
- The adapter ignores messages from its own Sesame principal to prevent echo loops.
- The WebSocket connection authenticates via `/v1/connect`, sends heartbeat pings, and asks Sesame to replay missed messages from the last persisted per-channel sequence cursor.
- Outbound messages include `clientGeneratedId` idempotency keys. Attachments are uploaded through Sesame Drive and sent as `kind: "attachment"` messages with non-empty fallback captions, matching Sesame validation.

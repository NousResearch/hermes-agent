---
sidebar_position: 12
title: Microsoft Teams
---

# Microsoft Teams

Hermes can run as a Microsoft Teams bot through the Bot Framework webhook. This first integration slice is text-only: it supports normal text messages in DMs, channels, and group chats, plus text replies back to the same Teams conversation.

Not included yet: Microsoft Graph, SharePoint files, Adaptive Cards, images, channel history, or scheduled outbound delivery through `send_message`.

## Requirements

- A Microsoft Teams bot registration with an app/client ID and client secret.
- A public HTTPS URL that Microsoft can POST to.
- Gateway dependencies installed with messaging support (`aiohttp`, `httpx`, and `PyJWT[crypto]` are used by the adapter).

## Configuration

Use `~/.hermes/config.yaml`:

```yaml
platforms:
  msteams:
    enabled: true
    extra:
      app_id: "00000000-0000-0000-0000-000000000000"
      # Put app_password in ~/.hermes/.env as MSTEAMS_APP_PASSWORD.
      tenant_id: "botframework.com"   # optional; default shown
      host: "0.0.0.0"
      port: 3978
      path: "/api/messages"

msteams:
  require_mention: true
  mention_patterns:
    - "^hermes[:, ]"
  free_response_conversations:
    - "19:example@thread.tacv2"
```

Put secrets in `~/.hermes/.env`:

```bash
MSTEAMS_APP_PASSWORD=<client-secret-from-azure>
MSTEAMS_ALLOWED_USERS=azure-ad-object-id-1,azure-ad-object-id-2
```

You can also configure the basics entirely with environment variables:

```bash
MSTEAMS_APP_ID=00000000-0000-0000-0000-000000000000
MSTEAMS_APP_PASSWORD=<client-secret-from-azure>
MSTEAMS_TENANT_ID=botframework.com
MSTEAMS_PORT=3978
MSTEAMS_PATH=/api/messages
MSTEAMS_REQUIRE_MENTION=true
```

## Bot Endpoint

Set the bot messaging endpoint in Azure/Bot Framework to:

```text
https://your-public-host.example.com/api/messages
```

Use the configured `path` if you changed it from `/api/messages`.

## Mention Behavior

DMs are processed directly, subject to the normal Hermes gateway authorization flow.

Channels and group chats require an explicit bot mention by default. You can disable this globally with `msteams.require_mention: false`, add regex wake words with `mention_patterns`, or allow specific conversation/team/channel IDs with `free_response_conversations`.

## Start

```bash
hermes gateway run
```

The adapter listens on the configured host and port, validates Bot Framework JWTs, remembers each inbound conversation's `serviceUrl`, and sends replies to that same Bot Framework conversation endpoint.

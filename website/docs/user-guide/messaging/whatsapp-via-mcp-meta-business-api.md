---
sidebar_position: 22
title: "WhatsApp (Meta Cloud via MCP)"
description: "Set up Hermes Agent as a WhatsApp bot via the official Meta Cloud Business API and an external MCP server"
---

# WhatsApp via MCP (Meta Cloud Business API)

This platform lets Hermes act as a WhatsApp bot using the **official Meta
WhatsApp Cloud Business API** (`graph.facebook.com`). It is an alternative to
the bundled [`whatsapp` adapter](./whatsapp.md), which uses the unofficial
Baileys Node bridge (reverse-engineered WhatsApp Web). Pick this one if your
phone number is already enrolled in Meta Business and you want a TOS-compliant,
maintainable integration.

The platform is **webhook-only**: it does not talk to Meta directly. Instead,
it expects an external WhatsApp MCP server to:

1. Own the Meta Cloud webhook (verify handshake + signature validation).
2. Persist incoming messages and tag senders (the MCP is the system of record).
3. Forward messages of interest to this gateway via a simple POST.

This split keeps Hermes focused on agent reasoning while the MCP handles
Meta-side concerns (webhook ingress, contact management, multi-tenant routing,
business logic for non-agent flows like e-commerce orders).

## Architecture

```
WhatsApp users
      |
      v  Cloud API webhook
[ External WA-MCP server ]   (your own; e.g. an FastAPI app you run)
      |  POST /wa  {message_id, phone, content, ...}
      |  X-Webhook-Secret: <shared secret>
      v
[ Hermes gateway, this platform ]
      |  agent reasoning (SOUL.md, memory, skills, MCP tools, ...)
      v  POST graph.facebook.com/{phone-number-id}/messages
[ Meta Cloud API ] -> WhatsApp users
```

The forward POST is acknowledged with `200 {"ok": true, "queued": true}`
immediately so the MCP's short timeout (Meta requires <5 s for the original
webhook) is not exceeded by LLM latency. The actual reply is sent
asynchronously by the platform's `send()` method via Meta's Graph API.

## Prerequisites

1. A WhatsApp Business account with a phone number enrolled in the Cloud API.
   Get this through [Meta for Developers](https://developers.facebook.com).
   You will need:
   - A System User access token with the `whatsapp_business_messaging`
     permission.
   - The `phone_number_id` value (visible in Meta Business Manager → WhatsApp
     → API Setup).
2. An external WhatsApp MCP server. Hermes does not ship one, but the
   contract is intentionally minimal — see [Forward contract](#forward-contract)
   below.

## Setup

### 1. Configure environment variables

Set these via your environment, `.env`, or interactive `hermes gateway setup`:

| Variable | Required | Description |
|---|---|---|
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_TOKEN` | yes | Meta Cloud API access token |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_PHONE_NUMBER_ID` | yes | The phone-number id for `graph.facebook.com/{id}/messages` |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_WEBHOOK_SECRET` | recommended | Shared secret validated as `X-Webhook-Secret` on every forward |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_HOST` | no (default `0.0.0.0`) | aiohttp bind host |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_PORT` | no (default `8643`) | aiohttp bind port |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_PATH` | no (default `/wa`) | URL path for forward POSTs |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_ALLOWED_USERS` | yes (CSV) | Phones (E.164) allowed to talk to the agent |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_ALLOW_ALL_USERS` | no | `true` to skip the allowlist (NOT recommended) |
| `WHATSAPP_VIA_MCP_META_BUSINESS_API_HOME_CHANNEL` | no | Default phone for cron / scheduled-message delivery |

### 2. Configure your WhatsApp MCP

The MCP must POST inbound messages to `http://<hermes-host>:<PORT><PATH>`
(default `:8643/wa`) with body:

```json
{
  "message_id": "wamid.HBgL...",
  "phone": "+34612345678",
  "type": "text",
  "content": "hola"
}
```

and header `X-Webhook-Secret: <your-secret>`. Anything else is ignored. If the
secret env var is unset, the secret check is skipped — fine for local dev,
not for production.

### 3. Start the gateway

```bash
hermes gateway run
```

You should see in logs:

```
WhatsApp via MCP listening on 0.0.0.0:8643/wa (phone_id=<id>)
```

Send a WhatsApp message to your business phone. The MCP forwards, Hermes
processes, the reply lands in WhatsApp via Meta's Graph API.

## How Hermes Behaves

| Context | Behavior |
|---|---|
| **1-to-1 chats** | Hermes responds to every text message from allowlisted phones. |
| **Group chats** | WhatsApp Cloud API does not deliver group messages by default — the MCP would not see them. If you have group webhooks enabled (Meta beta), the MCP can forward them and Hermes will reply via the Graph API. |
| **Media (image / audio / file)** | The MCP can forward `type` other than `text`; this platform currently skips non-text. Outbound, Hermes can send images by URL via `send_image()`. |

Sessions are keyed by phone number. Memory persists across messages using the
standard Hermes session store.

## Forward Contract

The platform exposes one endpoint and one health probe:

- `POST <PATH>` (default `/wa`) — webhook ingest.
- `GET /health` — returns `{"ok": true, "platform": "...", "phone_number_id": "..."}`.

Forward POST headers:

- `X-Webhook-Secret`: must match `WHATSAPP_VIA_MCP_META_BUSINESS_API_WEBHOOK_SECRET`
  (constant-time comparison). Omitted/wrong → `403`.
- `Content-Type: application/json`.

Forward POST body fields:

| Field | Required | Notes |
|---|---|---|
| `phone` (or `from`) | yes | E.164 sender phone |
| `content` (or `text`) | yes for `type=text` | The message body |
| `type` | no (default `text`) | Currently only `text` is processed |
| `message_id` | no | Echoed in logs and used as `MessageEvent.message_id` |
| `tags`, `raw`, etc. | no | Ignored — pass anything else freely |

Response: `200 {"ok": true, "queued": true}` for accepted text messages.
Errors are JSON with HTTP 400 (bad input) or 403 (bad secret).

## Why a Separate Platform?

The bundled `whatsapp` adapter uses the [Baileys](https://github.com/WhiskeySockets/Baileys)
Node bridge, which:

- violates WhatsApp's Terms of Service (it impersonates WhatsApp Web),
- is fragile on Windows (Node native module loading errors),
- requires QR-code pairing each time the session expires.

Many users run a phone number on Meta's official Cloud API and want a
TOS-compliant integration without giving up the agent. This platform fills
that gap by intentionally being thin: it does not own the Meta credentials
end-to-end (a dedicated MCP server is better at that, and can serve other
clients besides Hermes), but it does own the agent loop.

## Limitations

- Group webhooks require Meta whitelisting on most numbers.
- Only `text` ingress is processed today (image/audio are forwarded as
  `200 {"skipped": "non_text"}`). Outbound supports text + image-by-URL.
- Reactions, status updates, and template messages are not implemented yet.
- No typing indicator (Meta Cloud does not expose one at the message level).

## See Also

- [`whatsapp` adapter (Baileys)](./whatsapp.md) — the original adapter.
- [Generic webhook adapter](./webhooks.md) — if your MCP can deliver into
  a `cross_platform` flow instead.

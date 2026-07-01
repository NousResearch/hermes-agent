---
sidebar_position: 18
---

# Linq iMessage

Connect Hermes to **iMessage** through [Linq][linq], a hosted iMessage
API. Linq runs the Apple line and abuse-prevention layer for you, so
there's no Mac relay and no BlueBubbles server to maintain — and unlike
a relay, the channel is fully Apple-native: real **tapback reactions**,
**typing indicators**, **read receipts**, and blue bubbles.

:::info No Mac required
Linq hosts the iMessage line. You provide a partner API token and a
public URL the gateway can receive webhooks on — there is no local
Apple device or `imessage` daemon in the loop.
:::

## Architecture

Linq is a **webhook + REST** channel, like the BlueBubbles iMessage
channel — **not** a persistent gRPC stream like Photon. There is **no
Node sidecar**; the adapter is pure Python.

- **Inbound** — Linq delivers each iMessage to your gateway as an
  HMAC-SHA256-signed webhook (`message.received`). The adapter runs a
  small **aiohttp** listener (default `:8790/linq/webhook`), verifies the
  signature, dedupes on `message.id`, and dispatches a `MessageEvent` to
  the agent. Because inbound is a webhook, the listener needs a **public
  URL** (a tunnel or reverse proxy) registered in your Linq dashboard.
- **Outbound** — replies go **directly** to the Linq REST API
  (`https://api.linqapp.com/api/partner/v3`) over `httpx` — send, typing,
  read receipts, and reactions. No sidecar, no SDK.

```
Inbound:   iMessage → Linq → signed webhook → aiohttp listener → MessageEvent → agent
Outbound:  agent → Linq REST API (httpx) → iMessage
```

## Prerequisites

- A **Linq partner account** with an API token — see [linqapp.com][linq]
- A **public URL** for the inbound webhook (e.g. a reverse proxy or a
  tunnel like `cloudflared` / `ngrok` pointed at the listener)
- **`aiohttp`** on the Python path. It ships with the
  `hermes-agent[messaging]` extra but is *not* in the core install; if
  `hermes linq status` reports it missing, run `pip install aiohttp`.

## Installation

Linq is a **bundled platform plugin** — it ships in-tree with Hermes under
`plugins/platforms/linq/`, so there is nothing to install. The gateway
discovers it automatically; you only need to configure it (below).

Its one optional dependency is `aiohttp`, used by the inbound webhook
listener. It ships with the `hermes-agent[messaging]` extra rather than the
core install; if `hermes linq status` reports it missing, run:

```bash
pip install aiohttp
```

## First-time setup

Either run the unified gateway wizard and pick **Linq iMessage**:

```bash
hermes gateway setup
```

…or run the Linq setup directly (the wizard calls the same flow):

```bash
hermes linq setup
```

Setup asks for:

1. **Linq API token** — from your [Linq][linq] dashboard (or set
   `LINQ_API_TOKEN` in your environment instead).
2. **From-phone** (optional) — the Linq line this agent sends from, in
   E.164 (`+15551234567`). Only needed to pin a multi-number account to
   one line so the adapter ignores traffic to your other numbers.

The token is written to `~/.hermes/auth.json` under
`credential_pool.linq` — the same place every other channel keeps its
credentials. `LINQ_API_TOKEN` / `LINQ_FROM_PHONE` in the environment
take precedence over the stored values.

### Register the inbound webhook

Linq delivers inbound iMessages as webhooks, so it needs a public URL.
Print the URL to register, then add it in your Linq dashboard:

```bash
hermes linq webhook show --public-url https://your-public-host
# → register  https://your-public-host/linq/webhook  in the Linq dashboard
```

Then export the signing secret the dashboard gives you so deliveries
are verified (the listener accepts unsigned deliveries with a warning if
this is unset, which is fine only for local testing):

```bash
export LINQ_WEBHOOK_SECRET=<secret-from-linq-dashboard>
```

Signatures are verified as
`hex(hmac_sha256(secret, "{timestamp}.{body}"))` against the
`X-Webhook-Timestamp` / `X-Webhook-Signature` headers, with a 5-minute
timestamp-drift window for replay protection.

## Authorizing users

Linq uses the same authorization model as every other Hermes channel.
Choose one approach:

**DM pairing (default).** When an unknown number messages your Linq
line, Hermes replies with a pairing code. Approve it with:

```bash
hermes pairing approve linq <CODE>
```

Use `hermes pairing list` to see pending codes and approved users.

**Pre-authorize specific numbers** (in `~/.hermes/.env`):

```bash
LINQ_ALLOWED_USERS=+15551234567,+15559876543
```

**Open access** (dev only, in `~/.hermes/.env`):

```bash
LINQ_ALLOW_ALL_USERS=true
```

When `LINQ_ALLOWED_USERS` is set, unknown senders are silently ignored
rather than offered a pairing code (the allowlist signals you
deliberately restricted access).

### Require mentions in group chats

By default Hermes responds to every authorized DM and group message.
To make group chats opt-in, enable mention gating (DMs still always
work):

```yaml
gateway:
  platforms:
    linq:
      enabled: true
      require_mention: true
```

With `require_mention: true`, group-chat messages are ignored unless
they match a wake-word pattern, and the leading wake word is stripped
from the ones that do. The defaults match `Hermes` and `@Hermes agent`
variants. For a custom agent name, set regex patterns:

```yaml
gateway:
  platforms:
    linq:
      require_mention: true
      mention_patterns:
        - '(?<![\w@])@?amos\b[,:\-]?'
```

Both keys also accept env vars (`LINQ_REQUIRE_MENTION`,
`LINQ_MENTION_PATTERNS`). This is the same mention-gating model the
Photon and BlueBubbles iMessage channels use.

## Start the gateway

```bash
hermes gateway start
```

You'll see something like:

```
[linq] connected — webhook at 0.0.0.0:8790/linq/webhook (outbound via https://api.linqapp.com/api/partner/v3)
```

Send an iMessage to your Linq line and Hermes will reply.

## Status & troubleshooting

```bash
hermes linq status
```

Prints saved credentials, the auth-file location, and a live
connectivity probe against the Linq API:

```
Linq iMessage credentials:
  api token           : ✓ stored
  from phone          : +15551234567
  auth file           : /Users/you/.hermes/auth.json
  connectivity        : ✓ reachable
```

Common issues:

- **`api token : ✗ missing`** — run `hermes linq setup` (or set
  `LINQ_API_TOKEN`).
- **`aiohttp not installed`** — the webhook listener dependency is
  optional in Hermes core. Run `pip install aiohttp`.
- **`webhook port 8790 unavailable`** — another process holds the port.
  Set `LINQ_WEBHOOK_PORT` to a free port and update your tunnel.
- **No inbound messages** — confirm the public URL in your Linq
  dashboard points at `…/linq/webhook` and that `LINQ_WEBHOOK_SECRET`
  matches the dashboard secret (a mismatch returns `401` and the
  delivery is dropped). `hermes linq webhook show` prints the expected
  URL.

## Limits today

- **Inbound images** are downloaded locally so the vision tools can read
  them. Non-image attachments are noted inline (the agent learns
  something was sent and where), not yet read byte-for-byte.
- **Outbound media is by public URL.** Linq sends media by URL rather
  than multipart upload, so the agent attaches a publicly reachable
  image URL; local-file sends on the cron / standalone path are skipped
  with a logged note.
- **Reactions, typing, and read receipts** are first-class. Inbound
  tapbacks arrive as `reaction.received` events; outbound typing and
  read receipts are sent on inbound (toggle with
  `LINQ_SEND_READ_RECEIPTS`).

## Env vars

| Variable                  | Default                                    | Notes                                                        |
|---------------------------|--------------------------------------------|--------------------------------------------------------------|
| `LINQ_API_TOKEN`          | from `auth.json`                           | **Required.** Linq partner API bearer token; set by setup    |
| `LINQ_FROM_PHONE`         | (unset)                                    | Pin a multi-number account to one Linq line (E.164)          |
| `LINQ_WEBHOOK_SECRET`     | (unset)                                    | HMAC-SHA256 secret for inbound signature verification        |
| `LINQ_WEBHOOK_PORT`       | `8790`                                     | Local webhook listener port                                  |
| `LINQ_WEBHOOK_PATH`       | `/linq/webhook`                            | Local webhook listener path                                  |
| `LINQ_WEBHOOK_BIND`       | `0.0.0.0`                                  | Webhook listener bind address                                |
| `LINQ_API_BASE`           | `https://api.linqapp.com/api/partner/v3`   | Linq REST API base URL                                       |
| `LINQ_SEND_READ_RECEIPTS` | `true`                                     | Send read receipts + typing indicator on inbound             |
| `LINQ_HOME_CHANNEL`       | (unset)                                    | Default chat id for cron / notification delivery             |
| `LINQ_ALLOWED_USERS`      | (unset)                                    | Comma-separated E.164 allowlist                              |
| `LINQ_ALLOW_ALL_USERS`    | `false`                                    | Dev only — accept any sender                                 |
| `LINQ_REQUIRE_MENTION`    | `false`                                    | Require a wake word before responding in groups              |
| `LINQ_MENTION_PATTERNS`   | Hermes wake words                          | JSON list / comma / newline regex patterns for group mentions|

[linq]: https://linqapp.com/

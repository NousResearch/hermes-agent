---
sidebar_position: 19
---

# Blooio iMessage

Connect Hermes to **iMessage** through [Blooio](https://blooio.com), a hosted
iMessage API. You send and receive real iMessages — with automatic SMS/RCS
fallback for non-Apple recipients — through a REST API and inbound webhooks,
without running your own Mac relay.

Blooio is a sibling to the built-in [BlueBubbles](bluebubbles.md) channel and
the [Photon](photon.md) plugin: all three reach iMessage, but Blooio does it
through a webhook + REST integration, so the setup below mirrors the LINE and
BlueBubbles webhook model rather than a persistent connection.

## Architecture

Blooio is a **webhook** channel:

- **Inbound** — the plugin runs a small `aiohttp` webhook server. Blooio POSTs
  each event to `/blooio/webhook`; the adapter verifies the HMAC-SHA256
  signature, dedupes on `message_id`, and dispatches the message to the agent.
- **Outbound** — replies POST to `/chats/{chatId}/messages` on the Blooio v2
  REST API (`https://api.blooio.com/v2/api`) with your API key as a bearer
  token. Typing indicators, read receipts, and reactions each map to a
  dedicated Blooio endpoint.

Because inbound delivery relies on webhooks — and outbound local-file
attachments are fetched by Blooio from a public URL — **Hermes must be
reachable at a public HTTPS hostname.** A laptop-only instance needs a tunnel
(Cloudflare Tunnel, Tailscale Funnel, or ngrok).

## Prerequisites

- A Blooio account and an API key — create one in the [Blooio dashboard](https://app.blooio.com/).
- `aiohttp` and `httpx` installed (`pip install aiohttp httpx`).
- A publicly reachable HTTPS URL that forwards to this Hermes instance.

## First-time setup

Run the interactive wizard:

```bash
hermes setup blooio
```

or set the environment variables directly in `~/.hermes/.env`:

```bash
BLOOIO_API_KEY=sk_live_...
BLOOIO_WEBHOOK_SECRET=whsec_...          # from the webhook you create (below)
BLOOIO_PUBLIC_URL=https://my-tunnel.example.com
BLOOIO_ALLOWED_USERS=+15551234567        # allowlist senders
```

Then register the webhook so Blooio knows where to deliver inbound messages.
Either:

- **In the Blooio dashboard**, add a webhook pointing at
  `https://my-tunnel.example.com/blooio/webhook` with type `all`, and copy the
  signing secret it shows (once) into `BLOOIO_WEBHOOK_SECRET`; or
- **Let Hermes register it** by setting `BLOOIO_AUTO_REGISTER_WEBHOOK=true` and
  `BLOOIO_PUBLIC_URL` — the plugin registers the webhook on connect and captures
  the returned signing secret automatically.

:::warning Verify your webhooks
If `BLOOIO_WEBHOOK_SECRET` is unset, the plugin still runs but **does not verify
inbound signatures** (it logs a warning). Always set the signing secret in
production so forged requests to your public webhook are rejected.
:::

### Expose Hermes with a tunnel

Any HTTPS tunnel to the webhook port (default `8647`) works. With
[Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/):

```bash
cloudflared tunnel --url http://localhost:8647
```

Use the printed `https://<name>.trycloudflare.com` URL as `BLOOIO_PUBLIC_URL`
and in the Blooio webhook (`<url>/blooio/webhook`).

## Authorizing users

Blooio uses the same authorization model as every other Hermes channel.

**Pre-authorize specific senders** (phone numbers or emails):

```bash
BLOOIO_ALLOWED_USERS=+15551234567,+15559876543
```

**Allow specific group chats** (Blooio `grp_…` ids):

```bash
BLOOIO_ALLOWED_GROUPS=grp_abc123,grp_def456
```

**Open access** (dev only):

```bash
BLOOIO_ALLOW_ALL_USERS=true
```

### Require mentions in group chats

By default Hermes responds to every authorized DM and group message. To make
group chats opt-in, enable mention gating (DMs still always work):

```yaml
gateway:
  platforms:
    blooio:
      enabled: true
      require_mention: true
```

With `require_mention: true`, group-chat messages are ignored unless they match
a wake-word pattern. The defaults match `Hermes` and `@Hermes agent` variants —
the same model as the BlueBubbles and Photon iMessage channels. For a custom
agent name, set regex patterns via `BLOOIO_MENTION_PATTERNS` (JSON list, or
comma/newline-separated).

## Start the gateway

```bash
hermes gateway start
```

You'll see something like:

```
[blooio] webhook listening on 0.0.0.0:8647/blooio/webhook (public: https://my-tunnel.example.com)
```

Send an iMessage to one of your Blooio numbers and Hermes will reply.

## Features

- **Text** — long replies are chunked and sent as an array (each chunk becomes
  its own iMessage bubble). iMessage renders plain text, so Markdown is stripped
  before sending; bare URLs get a native rich-link preview.
- **Attachments** — inbound images/audio/video/documents are downloaded and
  handed to the agent (e.g. for vision). Outbound: a remote HTTPS URL is sent
  directly; a local file is served from the webhook server behind
  `BLOOIO_PUBLIC_URL`.
- **Typing indicators & read receipts** — typing is sent while the agent works;
  read receipts are opt-in via `BLOOIO_SEND_READ_RECEIPTS`.
- **Reactions** — the agent can add tapbacks (`love`, `like`, `dislike`,
  `laugh`, `emphasize`, `question`) or any emoji. With `BLOOIO_REACTIONS=true`,
  Hermes also tapbacks 👀 while working and 👍/👎 on completion, and routes
  reactions on its own messages back to the agent.
- **Multi-number pools** — replies go out from whichever of your Blooio numbers
  received the inbound message; set `BLOOIO_FROM_NUMBER` to pin a default.

## Env vars

| Variable                      | Default                          | Notes |
|-------------------------------|----------------------------------|-------|
| `BLOOIO_API_KEY`              | (required)                       | Blooio API key (bearer token) |
| `BLOOIO_WEBHOOK_SECRET`       | (unset)                          | Webhook signing secret (`whsec_…`); verifies `X-Blooio-Signature`. Strongly recommended |
| `BLOOIO_PUBLIC_URL`           | (unset)                          | Public HTTPS base URL Blooio can reach; required for inbound + local-file attachments |
| `BLOOIO_FROM_NUMBER`          | inferred per chat                | E.164 number to send from when it can't be inferred |
| `BLOOIO_HOST`                 | `0.0.0.0`                        | Webhook bind host |
| `BLOOIO_PORT`                 | `8647`                           | Webhook listen port |
| `BLOOIO_AUTO_REGISTER_WEBHOOK`| `false`                          | Register the webhook + capture its secret on connect |
| `BLOOIO_ALLOWED_USERS`        | (unset)                          | Comma-separated sender phones/emails allowlist |
| `BLOOIO_ALLOWED_GROUPS`       | (unset)                          | Comma-separated `grp_…` group allowlist |
| `BLOOIO_ALLOW_ALL_USERS`      | `false`                          | Dev only — accept any sender |
| `BLOOIO_REQUIRE_MENTION`      | `false`                          | Require a wake word before responding in groups |
| `BLOOIO_MENTION_PATTERNS`     | Hermes wake words                | JSON list / comma / newline regex patterns for group mentions |
| `BLOOIO_SEND_READ_RECEIPTS`   | `false`                          | Send iMessage read receipts for processed messages |
| `BLOOIO_REACTIONS`            | `false`                          | Tapback 👀/👍/👎 status + route own-message reactions to the agent |
| `BLOOIO_HOME_CHANNEL`         | (unset)                          | Default target for cron / notification delivery |
| `BLOOIO_HOME_CHANNEL_NAME`    | (unset)                          | Human label for the home channel |
| `BLOOIO_API_BASE_URL`         | `https://api.blooio.com/v2/api`  | Override the Blooio API base URL |

## Troubleshooting

- **No inbound messages** — confirm the Blooio dashboard webhook points at
  `<BLOOIO_PUBLIC_URL>/blooio/webhook` and is active, and that your tunnel is up
  (`curl https://<public-url>/blooio/webhook/health` should return
  `{"status":"ok"}`).
- **`invalid signature` in the logs** — `BLOOIO_WEBHOOK_SECRET` doesn't match
  the webhook's signing secret. Rotate it in the dashboard (or via the API) and
  update the env var.
- **Can't send local-file attachments** — set `BLOOIO_PUBLIC_URL`; Blooio fetches
  attachments from a public HTTPS URL and can't read local paths directly.

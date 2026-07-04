---
sidebar_position: 8
title: "Messenger"
description: "Set up Hermes Agent as a Facebook Messenger Page bot (Meta Graph API)"
---

# Facebook Messenger Setup

Run Hermes Agent as the DM bot behind a Facebook Page via Meta's official Messenger Platform. The adapter lives as a bundled platform plugin under `plugins/platforms/messenger/` — no core edits, just enable it like any other platform.

Messenger shares its Meta app with [Instagram DM](instagram.md): one set of credentials, one webhook callback URL, one Graph API send pipeline. If you enable both platforms they share a single webhook listener, and events are routed to the right adapter automatically.

> Run `hermes gateway setup` and pick **Messenger (Meta)** for a guided walk-through.

## How the bot responds

| Context | Behavior |
|---------|----------|
| **Page DM** (PSIDs) | Responds to every message from an allowed user |
| **Group threads** | Not supported by the Messenger Platform API |

Inbound text, images, audio, video, and file attachments are handled — media is downloaded from Meta's CDN and cached for the vision/file tools. Echo events and delivery/read receipts are filtered so the bot never replies to itself.

---

## Step 1: Create a Meta app

1. Go to [Meta for Developers](https://developers.facebook.com/apps/) and create an app (type **Business**).
2. Add the **Messenger** product.
3. Under **Messenger > Messenger API settings**, connect your Facebook Page and **Generate token** — this is your Page access token.
4. Copy the **App secret** from **App settings > Basic**.
5. Invent a random string to use as your webhook **verify token**.

:::warning App Review
Until your app passes Meta App Review with the `pages_messaging` permission, only app admins/developers/testers can message the Page. That is enough for personal use and development — add your own Facebook account as an app admin. Public customer-facing bots need App Review.
:::

---

## Step 2: Expose the webhook

Meta delivers webhooks over public HTTPS. The adapter listens on `127.0.0.1:8647` at `/meta/webhook` by default — override with `META_WEBHOOK_HOST` / `META_WEBHOOK_PORT` / `META_WEBHOOK_PATH`.

```bash
# Cloudflare Tunnel (recommended for production — fixed hostname)
cloudflared tunnel --url http://localhost:8647

# ngrok (good for dev)
ngrok http 8647
```

If you front it with a reverse proxy, forward the path **unchanged** (do not strip the `/meta/webhook` prefix).

---

## Step 3: Configure Hermes

Add to `~/.hermes/.env`:

```env
# Shared Meta app credentials (same values serve Instagram DM)
META_PAGE_ACCESS_TOKEN=EAAG...
META_APP_SECRET=0123456789abcdef...
META_VERIFY_TOKEN=any-random-string

# Enable the Messenger surface
MESSENGER_ENABLED=true

# Gate access: specific page-scoped user IDs (PSIDs), or allow everyone
MESSENGER_ALLOWED_USERS=24681357902468
# MESSENGER_ALLOW_ALL_USERS=true   # public customer-facing bot

# Optional: default recipient for cron jobs / notifications
# MESSENGER_HOME_CHANNEL=24681357902468
```

`MESSENGER_ENABLED` is required — the `META_*` credentials are shared with the Instagram plugin, so each surface needs its own explicit opt-in.

Every webhook POST must carry a valid `X-Hub-Signature-256` header (HMAC-SHA256 of the raw body keyed by `META_APP_SECRET`); events that fail validation are rejected with `403`.

---

## Step 4: Subscribe the webhook

1. Start the gateway: `hermes gateway`
2. Verify the local listener: `curl "http://127.0.0.1:8647/meta/webhook/health"`
3. In the Meta developer portal under **Messenger > Messenger API settings > Webhooks**, set the callback URL to `https://<public-host>/meta/webhook` and the verify token to your `META_VERIFY_TOKEN`, then click **Verify and save**.
4. Subscribe the Page to the **messages** webhook field.
5. DM your Page from an allowed account — the agent should answer.

---

## Message behavior

- Outbound text is chunked at 1900 characters (Messenger's hard limit is 2000; the headroom carries `(1/3)` chunk indicators).
- The agent is told not to use markdown — Messenger renders it literally.
- Publicly reachable image URLs produced by the agent are sent as native photo attachments; local files can't be uploaded through the Send API, so the agent shares links instead.
- A `typing_on` sender action is sent while the agent is thinking.
- PSIDs are page-scoped: the same person has a different ID on every Page, and Instagram IDs are a separate namespace.

## Cron delivery

`deliver=messenger` uses `MESSENGER_HOME_CHANNEL`; `deliver=messenger:<PSID>` targets a specific user. Both work when cron runs in a separate process from the gateway (the plugin registers a standalone sender).

## Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `META_PAGE_ACCESS_TOKEN` | ✅ | Page access token (shared with Instagram) |
| `META_APP_SECRET` | ✅ | App secret for webhook signature validation |
| `META_VERIFY_TOKEN` | ✅ | Webhook verification handshake token |
| `MESSENGER_ENABLED` | ✅ | Explicit opt-in for the Messenger surface |
| `MESSENGER_ALLOWED_USERS` | — | Comma-separated allowed PSIDs |
| `MESSENGER_ALLOW_ALL_USERS` | — | Allow every user (public bots) |
| `MESSENGER_HOME_CHANNEL` | — | Default PSID for cron/notification delivery |
| `META_WEBHOOK_HOST` | — | Bind host (default `127.0.0.1`) |
| `META_WEBHOOK_PORT` | — | Listen port (default `8647`) |
| `META_WEBHOOK_PATH` | — | Webhook path (default `/meta/webhook`) |
| `META_GRAPH_API_BASE` | — | Graph API base URL override |

## Next Steps

- [Instagram DM Setup](instagram.md) — the second surface of the same Meta app
- [WhatsApp Business Cloud API Setup](whatsapp-cloud.md) — Meta's other Graph-webhook platform

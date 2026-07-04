---
sidebar_position: 9
title: "Instagram DM"
description: "Set up Hermes Agent as an Instagram Direct Messages bot (Meta Graph API)"
---

# Instagram DM Setup

Run Hermes Agent as the DM bot behind an Instagram professional account via Meta's official Instagram Messaging API. The adapter lives as a bundled platform plugin under `plugins/platforms/instagram/`.

Instagram is the second surface of a [Messenger](messenger.md) Meta app: an Instagram professional account is linked to a Facebook Page, credentials are shared (`META_*`), events arrive on the **same webhook callback URL** as Messenger, and sends go through the same Graph API `/me/messages` endpoint. When both platforms are enabled they share one webhook listener; Hermes routes each event by the payload's top-level `object` field (`page` → messenger, `instagram` → instagram), so sessions are recorded under the correct platform.

> Run `hermes gateway setup` and pick **Instagram DM (Meta)** for a guided walk-through.

## Prerequisites

- An **Instagram professional account** (Business or Creator) linked to a Facebook Page.
- A Meta app with the **Instagram** product added (and **Messenger** if you also want Page DMs).
- If you already set up [Messenger](messenger.md), the credentials below are the same — you only add `INSTAGRAM_ENABLED=true` and the Instagram webhook subscription.

:::warning App Review
Until your app passes Meta App Review with the `instagram_manage_messages` permission, only app admins/developers/testers can DM the account through the API. Public customer-facing bots need App Review.
:::

---

## Configuration

Add to `~/.hermes/.env`:

```env
# Shared Meta app credentials (same values serve Messenger)
META_PAGE_ACCESS_TOKEN=EAAG...
META_APP_SECRET=0123456789abcdef...
META_VERIFY_TOKEN=any-random-string

# Enable the Instagram surface
INSTAGRAM_ENABLED=true

# Gate access: specific Instagram-scoped user IDs (IGSIDs), or allow everyone
INSTAGRAM_ALLOWED_USERS=17841400000000000
# INSTAGRAM_ALLOW_ALL_USERS=true   # public customer-facing bot

# Optional: default recipient for cron jobs / notifications
# INSTAGRAM_HOME_CHANNEL=17841400000000000
```

`INSTAGRAM_ENABLED` is required — the `META_*` credentials are shared with the Messenger plugin, so each surface needs its own explicit opt-in.

## Webhook

The webhook listener, exposure steps, and signature validation are identical to [Messenger's](messenger.md#step-2-expose-the-webhook) (default `127.0.0.1:8647/meta/webhook`). In the Meta developer portal, additionally subscribe the app's **Instagram** webhook object to the **messages** field — the callback URL and verify token are the same ones Messenger uses.

## Message behavior

- Outbound text is chunked at 950 characters (Instagram's hard limit is 1000; the headroom carries `(1/3)` chunk indicators).
- The agent is told not to use markdown — Instagram renders it literally.
- Publicly reachable image URLs produced by the agent are sent as native photos; local files can't be uploaded through the Send API, so the agent shares links instead.
- Inbound image/audio/video attachments (including story mentions with media URLs) are downloaded from Meta's CDN and cached for the vision/file tools.
- IGSIDs are Instagram-scoped IDs — a different namespace from Messenger PSIDs, even for the same human.

## Cron delivery

`deliver=instagram` uses `INSTAGRAM_HOME_CHANNEL`; `deliver=instagram:<IGSID>` targets a specific user. Both work when cron runs in a separate process from the gateway (the plugin registers a standalone sender).

## Environment variable reference

| Variable | Required | Description |
|----------|----------|-------------|
| `META_PAGE_ACCESS_TOKEN` | ✅ | Page access token (shared with Messenger) |
| `META_APP_SECRET` | ✅ | App secret for webhook signature validation |
| `META_VERIFY_TOKEN` | ✅ | Webhook verification handshake token |
| `INSTAGRAM_ENABLED` | ✅ | Explicit opt-in for the Instagram surface |
| `INSTAGRAM_ALLOWED_USERS` | — | Comma-separated allowed IGSIDs |
| `INSTAGRAM_ALLOW_ALL_USERS` | — | Allow every user (public bots) |
| `INSTAGRAM_HOME_CHANNEL` | — | Default IGSID for cron/notification delivery |
| `META_WEBHOOK_HOST` | — | Bind host (default `127.0.0.1`, shared) |
| `META_WEBHOOK_PORT` | — | Listen port (default `8647`, shared) |
| `META_WEBHOOK_PATH` | — | Webhook path (default `/meta/webhook`, shared) |
| `META_GRAPH_API_BASE` | — | Graph API base URL override |

## Next Steps

- [Messenger Setup](messenger.md) — full webhook exposure walk-through
- [WhatsApp Business Cloud API Setup](whatsapp-cloud.md) — Meta's other Graph-webhook platform

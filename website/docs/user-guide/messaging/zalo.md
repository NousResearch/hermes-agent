---
sidebar_position: 18
title: "Zalo"
description: "Set up Hermes Agent as a Zalo Bot Platform bot"
---

# Zalo Setup

Run Hermes Agent as a [Zalo](https://zalo.me/) bot via the Zalo Bot Platform API. The adapter lives as a bundled platform plugin under `plugins/platforms/zalo/`, so it can be enabled without editing core gateway code.

> Run `hermes gateway setup` and pick **Zalo** for a guided walk-through.

## How the bot responds

| Context | Behavior |
|---------|----------|
| **1:1 chat** | Responds to each allowed user message |
| **Group/channel chat** | Ignored when `ZALO_DM_ONLY=true`; otherwise routed through normal allowlist checks |
| **Cron / notifications** | Delivered to `ZALO_HOME_CHANNEL` when configured |

Inbound text, image, voice, sticker, and supported link content are normalized into Hermes message events when Zalo exposes their content to the bot. Zalo may send `message.unsupported.received` without the original content for protected or unsupported messages; Hermes reports that limitation instead of guessing what was sent.

## Step 1: Create a Zalo bot

1. Open Zalo Bot Manager.
2. Create a bot and copy the Bot Platform token.
3. Disable any platform-side auto-reply behavior that would compete with Hermes replies.

## Step 2: Choose long polling or webhook mode

For local development, use long polling. Hermes calls `getUpdates` and no public URL is required. The default `ZALO_CONNECTION_MODE=auto` uses webhook mode when a webhook URL and secret are present, and polling otherwise. Set `ZALO_CONNECTION_MODE=polling` or `ZALO_CONNECTION_MODE=webhook` when you want to force one mode.

For production, use webhook mode. Expose the local webhook listener through a stable HTTPS URL such as a reverse proxy, Cloudflare Tunnel, or another tunnel with a fixed hostname.

Zalo treats long polling and webhooks as mutually exclusive. If a webhook was set earlier and you are switching back to long polling, either delete the webhook in Zalo Bot Manager or set `ZALO_DELETE_WEBHOOK_ON_POLLING_START=true` once so Hermes calls `deleteWebhook` before polling.

```bash
# Example local tunnel for development
cloudflared tunnel --url http://localhost:18787
```

Set the public URL in Zalo Bot Manager if you are not using `webhook_auto_register: true`.

## Step 3: Configure Hermes

Add credentials to `~/.hermes/.env`:

```env
ZALO_BOT_TOKEN=YOUR_ZALO_BOT_TOKEN
# Required only for webhook mode.
ZALO_WEBHOOK_SECRET=generate-a-long-random-secret-8-to-256-chars
```

Keep normal operator settings in `~/.hermes/config.yaml`:

```yaml
platforms:
  zalo:
    enabled: true

    # Access control. Prefer an allowlist for real deployments.
    allow_from:
      - USER_ID_1
      - USER_ID_2
    # allow_all_users: true
    dm_only: true

    # Long polling defaults.
    connection_mode: auto
    poll_timeout_seconds: 25
    poll_interval_seconds: 1
    # Optional: clear a stale webhook before polling.
    # delete_webhook_on_polling_start: true

    # Optional webhook mode.
    webhook_url: https://your-public-host.example.com/zalo/webhook
    webhook_host: 127.0.0.1
    webhook_port: 18787
    webhook_path: /zalo/webhook
    # webhook_auto_register: true
    # delete_webhook_on_disconnect: false

    # Optional cron / notification target.
    home_channel:
      platform: zalo
      chat_id: USER_OR_CHAT_ID
      name: Zalo Home

    # Disable gateway shutdown/restart/startup lifecycle pings on Zalo.
    # Defaults to true when omitted.
    gateway_restart_notification: false
```

Restart the gateway:

```bash
hermes gateway restart
```

## Relationship to other Zalo work

This adapter uses the official Zalo Bot Platform API and is packaged as a bundled platform plugin. It is different from Zalo personal-account bridges such as zca-js/HZCA-style automation, which use unofficial personal-account sessions and carry different account-policy risk. It is also intentionally narrower than older core-adapter proposals: the plugin path keeps Zalo self-contained while still getting gateway setup, allowlists, cron delivery, status, config UI entries, and system-prompt hints through the platform registry.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Bot never replies | Confirm `ZALO_BOT_TOKEN`, gateway logs, and whether the user is in `ZALO_ALLOWED_USERS`. |
| Group messages are ignored | Check `ZALO_DM_ONLY`. This is intentional when private-chat-only mode is enabled. |
| Webhook starts but receives nothing | Check the public HTTPS URL, `ZALO_WEBHOOK_PATH`, and Bot Manager webhook settings. |
| Long polling receives nothing after webhook testing | Clear the Bot Platform webhook or set `ZALO_DELETE_WEBHOOK_ON_POLLING_START=true` for the long-polling profile. |
| Zalo reports an unsupported message | Ask the user to resend the content as plain text or as a supported image or voice message. Zalo can intentionally omit content for some unsupported events. |
| Gateway shutdown/restart notices are noisy | Set `platforms.zalo.gateway_restart_notification: false`. |

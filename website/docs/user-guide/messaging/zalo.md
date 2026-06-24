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

Inbound text, image, voice, sticker, document, and link-preview events are normalized into Hermes message events when Zalo exposes their content to the bot. Some Zalo clients convert pasted URLs into rich preview cards that may not include the raw URL in bot updates. For those cases, configure the optional URL intake page so the user can submit the URL through a normal HTTPS form and continue the chat in Zalo.

## Step 1: Create a Zalo bot

1. Open Zalo Bot Manager.
2. Create a bot and copy the Bot Platform token.
3. Disable any platform-side auto-reply behavior that would compete with Hermes replies.

## Step 2: Choose long polling or webhook mode

For local development, use long polling. Hermes calls `getUpdates` and no public URL is required.

For production, use webhook mode. Expose the local webhook listener through a stable HTTPS URL such as a reverse proxy, Cloudflare Tunnel, or another tunnel with a fixed hostname.

Zalo treats long polling and webhooks as mutually exclusive. If a webhook was set earlier and you are switching back to long polling, either delete the webhook in Zalo Bot Manager or set `ZALO_DELETE_WEBHOOK_ON_POLLING_START=true` once so Hermes calls `deleteWebhook` before polling.

```bash
# Example local tunnel for development
cloudflared tunnel --url http://localhost:18787
```

Set the public URL in Zalo Bot Manager if you are not using `ZALO_WEBHOOK_AUTO_REGISTER=true`.

## Step 3: Configure Hermes

Add to `~/.hermes/.env`:

```env
ZALO_BOT_TOKEN=YOUR_ZALO_BOT_TOKEN

# Access control. Prefer an allowlist for real deployments.
ZALO_ALLOWED_USERS=USER_ID_1,USER_ID_2
# ZALO_ALLOW_ALL_USERS=true
ZALO_DM_ONLY=true

# Long polling defaults
ZALO_POLL_TIMEOUT_SECONDS=25
ZALO_POLL_INTERVAL_SECONDS=1
# Optional: clear a stale webhook before polling.
# ZALO_DELETE_WEBHOOK_ON_POLLING_START=true

# Optional webhook mode
ZALO_WEBHOOK_URL=https://your-public-host.example.com/zalo/webhook
ZALO_WEBHOOK_SECRET=generate-a-long-random-secret-8-to-256-chars
ZALO_WEBHOOK_HOST=127.0.0.1
ZALO_WEBHOOK_PORT=18787
ZALO_WEBHOOK_PATH=/zalo/webhook
# ZALO_WEBHOOK_AUTO_REGISTER=true

# Optional cron / notification target
ZALO_HOME_CHANNEL=USER_OR_CHAT_ID
ZALO_HOME_CHANNEL_NAME=Zalo Home

# Optional: show transient compression/retry status notices in chat.
# Default is true, which keeps Zalo conversations clean.
# ZALO_SUPPRESS_NOISY_STATUS=false
```

Then enable the platform in `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    zalo:
      enabled: true
```

Restart the gateway:

```bash
hermes gateway restart
```

## Optional URL intake

If your users often send Google Drive, Google Sheets, or other long links that Zalo converts into unreadable preview events, run a small intake page and point the adapter at its public base URL:

```env
ZALO_URL_INTAKE_PUBLIC_BASE=https://your-intake.example.com
ZALO_URL_INTAKE_PENDING_FILE=/var/lib/hermes/zalo-url-intake/pending.json
```

When Zalo delivers an unsupported/no-content event, the adapter can include an instruction for the agent to send the user to the intake page. The next inbound Zalo message from the same chat includes the submitted URL in the session context.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Bot never replies | Confirm `ZALO_BOT_TOKEN`, gateway logs, and whether the user is in `ZALO_ALLOWED_USERS`. |
| Group messages are ignored | Check `ZALO_DM_ONLY`. This is intentional when private-chat-only mode is enabled. |
| Webhook starts but receives nothing | Check the public HTTPS URL, `ZALO_WEBHOOK_PATH`, and Bot Manager webhook settings. |
| Long polling receives nothing after webhook testing | Clear the Bot Platform webhook or set `ZALO_DELETE_WEBHOOK_ON_POLLING_START=true` for the long-polling profile. |
| Link cards are unreadable | Configure URL intake, or ask the user to send the URL as broken plain text such as `docs . google . com / spreadsheets / d / SHEET_ID`. |
| Compression/compaction notices are hidden | This is the default Zalo behavior. Set `ZALO_SUPPRESS_NOISY_STATUS=false` if you want those transient gateway notices delivered. |

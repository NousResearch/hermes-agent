---
sidebar_position: 18
title: "Zalo"
description: "Set up Hermes Agent as an official Zalo Bot Platform bot"
---

# Zalo Setup

Hermes can run as an official [Zalo Bot Platform](https://docs.zaloplatforms.com/docs/BOT/create_bot) bot. Zalo is a bundled platform plugin, so it uses the same gateway setup, authorization, pairing, cron delivery, and channel configuration as the other Hermes messaging platforms.

> Run `hermes gateway setup` and select **Zalo** for guided credential setup.

## Capabilities

| Capability | Support |
|---|---|
| Private and group chats | Yes (group support is currently beta in Zalo) |
| Text and typing indicators | Yes |
| Inbound images, voice messages, and stickers | Yes |
| Outbound images | Yes, from a public HTTP(S) URL |
| Cron and notification delivery | Yes, through the configured home channel |
| Streaming edits, threads, and reactions | No |

Zalo may emit `message.unsupported.received` without the original content for protected or unsupported messages. Hermes reports that limitation and does not try to infer content Zalo omitted.

## 1. Create the bot

1. Open Zalo and find the **Zalo Bot Manager** Official Account.
2. Select **Create bot** and complete the Bot Creator flow.
3. Copy the bot token Zalo sends to you.

Store the token in `~/.hermes/.env`:

```env
ZALO_BOT_TOKEN=YOUR_ZALO_BOT_TOKEN
```

## 2. Configure access

Keep non-secret behavior in `~/.hermes/config.yaml`. An allowlist is recommended for real deployments:

```yaml
platforms:
  zalo:
    enabled: true
    allow_from:
      - USER_ID_1
      - USER_ID_2

    # Zalo group chat support is beta. Disable it for a DM-only bot.
    group_policy: disabled

    # Local development transport.
    connection_mode: polling

    home_channel:
      platform: zalo
      chat_id: USER_OR_CHAT_ID
      name: Zalo Home

    # Optional: suppress gateway restart lifecycle notices.
    gateway_restart_notification: false
```

You can use `ZALO_ALLOWED_USERS` instead of `allow_from`. An explicit environment value takes precedence over YAML. Open access requires `ZALO_ALLOW_ALL_USERS=true` (or the global gateway equivalent).

## 3. Choose a transport

### Polling for local development

`connection_mode: polling` calls Zalo's `getUpdates` API. Zalo documents polling and webhooks as mutually exclusive, so Hermes calls `deleteWebhook` before polling starts. This makes switching from webhook mode predictable, but it also changes the bot's remote webhook setting.

Optional polling configuration:

```yaml
platforms:
  zalo:
    connection_mode: polling
    poll_timeout_seconds: 30
```

### Webhook for production

Zalo recommends webhooks for production to avoid missed events. Create an 8-256 character secret and put it in `~/.hermes/.env`:

```env
ZALO_WEBHOOK_SECRET=YOUR_RANDOM_WEBHOOK_SECRET
```

Then configure the public HTTPS URL and local listener:

```yaml
platforms:
  zalo:
    connection_mode: webhook
    webhook_url: https://bot.example.com/zalo/webhook
    webhook_host: 127.0.0.1
    webhook_port: 18787
    webhook_path: /zalo/webhook
```

On startup, Hermes validates the HTTPS URL and secret, starts the listener, and calls Zalo's `setWebhook`. Every inbound request must carry the matching `X-Bot-Api-Secret-Token` header.

If `connection_mode` is omitted, `auto` selects webhook mode when either webhook setting is present and polling otherwise. Partial webhook configuration fails closed.

## 4. Start and verify

```bash
hermes gateway restart
hermes gateway status
```

Send the bot a Zalo message. If pairing is enabled and the sender is not already allowed, approve the pairing request through the normal Hermes gateway flow.

## Troubleshooting

| Symptom | Check |
|---|---|
| Bot never replies | Verify the bot token, sender allowlist or pairing status, and gateway logs. |
| Polling startup fails | Confirm Zalo API access; Hermes must successfully clear the remote webhook before polling. |
| Webhook startup fails | Use a public HTTPS URL and an 8-256 character `ZALO_WEBHOOK_SECRET`. |
| Webhook receives nothing | Confirm the reverse proxy routes the public path to the configured local host, port, and path. |
| Group messages are ignored | Check `group_policy`; Zalo group support is currently beta. |
| Message is reported unsupported | Ask the user to resend it as text, an image, or a voice message. |

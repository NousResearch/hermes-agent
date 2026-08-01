---
sidebar_position: 18
title: "Zalo Bot"
description: "Connect Hermes Agent to Zalo Bot Platform with polling or webhooks"
---

# Zalo Bot Setup

Hermes connects to **[Zalo Bot Platform](https://bot.zapps.me/docs/)** through the bundled plugin in `plugins/platforms/zalo/`. This is the **Bot Creator** API, not the Zalo Official Account APIs at `developers.zalo.me`.

> Run `hermes gateway setup` and select **Zalo Bot** for guided configuration.

## Capabilities

- Inbound text, images, and stickers
- Outbound text, HTTPS-hosted images, and stickers
- Typing indicators with `sendChatAction`
- Long polling with automatic backoff, or an HTTPS webhook
- Allowlist/pairing integration
- `send_message` and out-of-process cron delivery through the platform registry

The adapter validates the token with `getMe` before reporting connected and takes a machine-local token lock so two Hermes profiles cannot poll the same bot concurrently.

## Choose a connection mode

| Mode | Best for | Requirements |
|------|----------|--------------|
| **Polling** (default) | Local use; no public URL | Core `httpx` dependency and `ZALO_BOT_TOKEN` |
| **Webhook** | Production with stable HTTPS ingress | `aiohttp`, a public HTTPS URL, and an 8–256 character secret |

:::warning Polling and webhook are mutually exclusive
Zalo's `getUpdates` does not work while a webhook is registered. Use webhook mode or delete the existing webhook before starting polling.
:::

## Step 1: Create a bot

1. In Zalo, find **Zalo Bot Manager** and open the Bot Creator flow.
2. Create a bot (Zalo requires the bot name to start with `Bot`).
3. Copy the **Bot Token** sent to your Zalo account.

## Step 2: Configure Hermes

### Setup wizard

```bash
hermes gateway setup
```

Select **Zalo Bot**. The plugin owns this setup flow and stores the values in `~/.hermes/.env`.

### Environment variables

```env
ZALO_BOT_TOKEN=your-bot-token

# Recommended security boundary
ZALO_ALLOWED_USERS=user-id-1,user-id-2
# Development only:
# ZALO_ALLOW_ALL_USERS=true

# Optional default target for cron and send_message
ZALO_HOME_CHANNEL=chat-id
ZALO_HOME_CHANNEL_NAME=Primary
```

A token-only environment setup auto-enables the bundled plugin. To make enablement explicit in `config.yaml`:

```yaml
gateway:
  platforms:
    zalo:
      enabled: true
```

You can also place credentials and transport settings in the platform block:

```yaml
gateway:
  platforms:
    zalo:
      enabled: true
      token: "${ZALO_BOT_TOKEN}"
      extra:
        connection_mode: polling
        poll_timeout: 30
```

Environment variables take precedence over `config.yaml`.

## Polling mode

Polling is the default. Optional tuning:

```env
ZALO_CONNECTION_MODE=polling
ZALO_POLL_TIMEOUT=30
```

`ZALO_POLL_TIMEOUT` is clamped to 1–50 seconds. Transport and API errors use capped exponential backoff with jitter.

## Webhook mode

Install the optional server dependency:

```bash
pip install "hermes-agent[zalo]"
```

Then configure a public HTTPS endpoint:

```env
ZALO_CONNECTION_MODE=webhook
ZALO_WEBHOOK_PUBLIC_URL=https://example.com/zalo/webhook
ZALO_WEBHOOK_SECRET=replace-with-8-to-256-characters

# Optional local listener overrides
ZALO_WEBHOOK_HOST=0.0.0.0
ZALO_WEBHOOK_PORT=8790
# ZALO_WEBHOOK_PATH=/zalo/webhook
```

Hermes starts the local `aiohttp` server and calls `setWebhook`. Incoming requests must carry a matching `X-Bot-Api-Secret-Token`; comparison is constant-time. On clean shutdown Hermes calls `deleteWebhook`.

If `ZALO_WEBHOOK_PATH` is unset, Hermes uses the path from `ZALO_WEBHOOK_PUBLIC_URL`, falling back to `/zalo/webhook` for a root URL.

## Start and verify

```bash
hermes gateway
# In another terminal:
hermes gateway status
```

The status/setup surfaces are registry-driven: Zalo appears as configured when its token is present, without adding an obsolete `Platform.ZALO` core enum member or a hardcoded gateway branch.

## Sending and cron delivery

Set `ZALO_HOME_CHANNEL`, then use Zalo as a cron target:

```python
cronjob(
    action="create",
    schedule="every 1h",
    deliver="zalo",
    prompt="Summarize new alerts.",
)
```

The plugin registers a standalone sender, so `deliver: zalo` and `send_message` work even when cron runs outside the gateway process.

Zalo `sendMessage` accepts about 2000 characters per call; Hermes splits longer text automatically. `sendPhoto` requires an HTTPS image URL. The standalone cron sender does not upload local files.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| `getMe` rejects the token | Re-copy `ZALO_BOT_TOKEN` from Bot Creator. |
| Bot token already in use | Stop the other local Hermes gateway/profile using the same bot. |
| Polling receives nothing | A webhook may still be registered; delete it or use webhook mode. |
| Webhook mode says `aiohttp` is missing | Install `hermes-agent[zalo]`; Hermes also attempts the pinned lazy dependency. |
| Webhook returns 403 | Ensure the proxy forwards `X-Bot-Api-Secret-Token` unchanged. |
| `setWebhook` fails | The public URL must use HTTPS and the secret must be 8–256 characters. |
| Cron has no target | Set `ZALO_HOME_CHANNEL` or use an explicit Zalo chat ID. |

## Further reading

- [Zalo Bot Platform documentation](https://bot.zapps.me/docs/)
- [`getUpdates`](https://bot.zapps.me/docs/apis/getUpdates/)
- [`setWebhook`](https://bot.zapps.me/docs/apis/setWebhook/)
- [Polling tutorial](https://bot.zaloplatforms.com/docs/build-your-bot/)

---
sidebar_position: 11
title: "Feishu / Lark"
description: "Set up Hermes Agent as a Feishu or Lark bot"
---

# Feishu / Lark Setup

Hermes Agent integrates with Feishu and Lark as a full-featured bot. Once connected, you can chat with the agent in direct messages or group chats, receive cron job results in a home chat, and send text, images, audio, and file attachments through the normal gateway flow.

The integration supports both connection modes:

- `websocket` — recommended; Hermes opens the outbound connection and you do not need a public webhook endpoint
- `webhook` — useful when you want Feishu/Lark to push events into your gateway over HTTP

## How Hermes Behaves

| Context | Behavior |
|---------|----------|
| Direct messages | Hermes responds to every message. |
| Group chats | Hermes responds when the bot is addressed in the chat. |
| Shared group chats | By default, session history is isolated per user inside a shared chat. |

This shared-chat behavior is controlled by `config.yaml`:

```yaml
group_sessions_per_user: true
```

Set it to `false` only if you explicitly want one shared conversation per chat.

## Step 1: Create a Feishu / Lark App

1. Open the Feishu or Lark developer console:
   - Feishu: [https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark: [https://open.larksuite.com/](https://open.larksuite.com/)
2. Create a new app.
3. In **Credentials & Basic Info**, copy the **App ID** and **App Secret**.
4. Enable the **Bot** capability for the app.

:::warning
Keep the App Secret private. Anyone with it can impersonate your app.
:::

## Step 2: Choose a Connection Mode

### Recommended: WebSocket mode

Use WebSocket mode when Hermes runs on your laptop, workstation, or a private server. No public URL is required.

```bash
FEISHU_CONNECTION_MODE=websocket
```

### Optional: Webhook mode

Use webhook mode only when you already run Hermes behind a reachable HTTP endpoint.

```bash
FEISHU_CONNECTION_MODE=webhook
```

In webhook mode, Hermes serves a Feishu endpoint at:

```text
/feishu/webhook
```

## Step 3: Configure Hermes

### Option A: Interactive Setup

```bash
hermes gateway setup
```

Select **Feishu / Lark** and fill in the prompts.

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
FEISHU_APP_ID=cli_xxx
FEISHU_APP_SECRET=secret_xxx
FEISHU_DOMAIN=feishu
FEISHU_CONNECTION_MODE=websocket

# Optional but strongly recommended
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
FEISHU_HOME_CHANNEL=oc_xxx
```

`FEISHU_DOMAIN` accepts:

- `feishu` for Feishu China
- `lark` for Lark international

## Step 4: Start the Gateway

```bash
hermes gateway
```

Then message the bot from Feishu/Lark to confirm that the connection is live.

## Home Chat

Use `/set-home` in a Feishu/Lark chat to mark it as the home channel for cron job results and cross-platform notifications.

You can also preconfigure it:

```bash
FEISHU_HOME_CHANNEL=oc_xxx
```

## Security

For production use, set an allowlist:

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

If you leave the allowlist empty, anyone who can reach the bot may be able to use it.

## Toolset

Feishu / Lark uses the `hermes-feishu` platform preset, which includes the same core tools as Telegram and other gateway-based messaging platforms.

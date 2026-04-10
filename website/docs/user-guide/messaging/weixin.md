---
sidebar_position: 12
title: "Weixin"
description: "Set up Hermes Agent for personal Weixin via the built-in local bridge"
---

# Weixin Setup

Hermes connects to personal **Weixin** through a built-in local bridge that speaks Tencent's iLink bot endpoints. This is an unofficial integration for personal Weixin accounts, not the enterprise [WeCom](./wecom.md) platform.

:::warning Experimental / Unofficial
This integration uses a reverse-engineered personal Weixin flow. It is text-first today, and Tencent can change the protocol at any time.
:::

## What It Does Today

- QR-code pairing from your terminal
- Long-poll inbound message sync
- Text replies into the paired Weixin conversation
- Persistent session storage under `~/.hermes/weixin/session`

Current scope is **DM-first and text-first**. Full media parity is not implemented yet.

## Prerequisites

- **Node.js 18+** and **npm**
- **A personal Weixin account** on your phone
- A terminal that can display the QR code, or access to the fallback QR URL printed by the setup flow

## Step 1: Pair Weixin

```bash
hermes weixin
```

The command will:

1. Enable the `weixin` platform in your Hermes env
2. Install bridge dependencies if needed
3. Launch the local bridge on `127.0.0.1`
4. Print a QR code for pairing
5. Save the resulting session for the gateway to reuse

If you already have a saved session, `hermes weixin` lets you keep it or clear it and re-pair.

## Step 2: Start the Gateway

```bash
hermes gateway
```

Or install it as a service:

```bash
hermes gateway install
hermes gateway start
```

Once the gateway is running, message the paired Weixin chat and Hermes will answer there.

## Environment Variables

```bash
WEIXIN_ENABLED=true
WEIXIN_ALLOWED_USERS=o9cq808BEvFSo6VGJ4c1Xjqqv0FU@im.wechat
WEIXIN_HOME_CHANNEL=o9cq808BEvFSo6VGJ4c1Xjqqv0FU@im.wechat
WEIXIN_BRIDGE_PORT=3010

# Optional advanced overrides
# WEIXIN_SESSION_PATH=~/.hermes/weixin/session
# WEIXIN_BRIDGE_SCRIPT=/path/to/hermes-agent/scripts/weixin-bridge/bridge.js
# WEIXIN_BOT_TYPE=3
# WEIXIN_APP_ID=bot
```

- `WEIXIN_ALLOWED_USERS` is the recommended access-control path for a private account.
- If you leave it empty, Hermes falls back to the gateway's normal pairing / allowlist behavior.
- `WEIXIN_HOME_CHANNEL` is used for cron delivery and notifications.

## Session Files

The bridge persists login and polling state here:

```text
~/.hermes/weixin/session/
```

Important files:

- `credentials.json`
- `context-tokens.json`
- `get-updates-buf.txt`

Treat this directory like a password. Anyone who gets it can reuse the paired Weixin session.

## Current Limitations

- Personal Weixin only. Enterprise WeCom is a separate adapter.
- Text sending is supported. Full image/file/video sending is not.
- Incoming media is normalized conservatively and may appear as placeholders or text only.
- The bridge is a local Node.js sidecar, not a hosted API.

## Troubleshooting

| Problem | What to check |
|---------|---------------|
| QR code will not scan | Try the fallback QR URL printed by `hermes weixin`, or use a terminal with better Unicode rendering |
| Pairing keeps expiring | Re-run `hermes weixin`; the bridge refreshes the QR, but the flow times out after a few expired cycles |
| Gateway starts but no messages arrive | Verify `WEIXIN_ALLOWED_USERS` includes the paired user ID, or remove it and rely on DM pairing |
| Bridge does not start | Confirm `node --version` works in the same shell, then rerun `hermes weixin` |
| Session breaks later | Delete `~/.hermes/weixin/session` or rerun `hermes weixin` and choose re-pair |

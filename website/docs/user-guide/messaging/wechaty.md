---
sidebar_position: 16
title: "WeChat (Wechaty)"
description: "Connect Hermes Agent to personal WeChat via Wechaty (unofficial puppet)"
---

# WeChat (Wechaty)

Connect Hermes to **personal WeChat** through [Wechaty](https://wechaty.js.org/).
A supervised Node sidecar runs the Wechaty bot; inbound messages stream to
the Python gateway over loopback NDJSON and outbound replies POST back to
the same sidecar.

:::info Official alternative
For Tencent's **iLink Bot API** (official personal-WeChat bot identity), see
[Weixin (WeChat)](./weixin.md). For enterprise WeChat, see
[WeCom](./wecom.md).
:::

:::warning Unofficial protocol risk
Wechaty puppets often use **unofficial** WeChat protocols. This can violate
Tencent's Terms of Service and may lead to account restrictions. Use only
when you need a real personal account (especially group `@`-mentions) and
accept the maintenance and compliance risks.
:::

## Weixin vs Wechaty

| | Weixin (built-in) | Wechaty (this plugin) |
|--|-------------------|----------------------|
| API | Tencent iLink (official) | Wechaty puppet (often unofficial) |
| Identity | iLink bot (`@im.bot`) | Real personal WeChat account |
| Group @-mentions | Usually does not work on iLink | Supported (with `require_mention`) |
| Stability / ToS | Lower ban risk | Higher maintenance + account risk |
| When to use | Official path, DMs to bot | Need real account + group chat |

## Architecture

Wechaty is a **persistent-connection** channel — no webhook or public URL.

The TypeScript/Node Wechaty SDK runs in a small **sidecar** process:

- **Inbound** — `GET /inbound` (NDJSON) streams normalized events to
  `WechatyAdapter`, which dedupes and dispatches to the agent.
- **Outbound** — loopback POSTs to `/send`, `/send-file`, `/typing`.

The Python plugin starts, supervises, and shuts down the sidecar automatically.

## Prerequisites

- **Node.js 18.17+** on PATH
- Python `httpx` (included with Hermes messaging dependencies)
- A Wechaty puppet — `wechaty-puppet-wechat4u` works without a paid token
  (trial/dev); production setups often use `wechaty-puppet-service` with a
  PadLocal or WorkPro token

## Setup

### 1. Gateway wizard

```bash
hermes gateway setup
```

Select **WeChat (Wechaty)**. The wizard installs sidecar npm dependencies
(if needed), prompts for puppet settings, and writes secrets to
`~/.hermes/.env`.

### 2. Enable the platform

```yaml
gateway:
  platforms:
    wechaty:
      enabled: true
      extra:
        puppet: wechaty-puppet-wechat4u
        require_mention: true
```

Equivalent environment variables (in `~/.hermes/.env`):

```bash
WECHATY_PUPPET=wechaty-puppet-wechat4u
# For wechaty-puppet-service:
# WECHATY_PUPPET=wechaty-puppet-service
# WECHATY_PUPPET_SERVICE_TOKEN=your-token
```

### 3. Start and scan QR

```bash
hermes gateway run
```

Watch `gateway.log` for a QR URL (`https://wechaty.js.org/qrcode/...`) and
scan with WeChat **扫一扫**.

## Authorizing users

**DM pairing (default).** Unknown senders receive a pairing code:

```bash
hermes pairing approve wechaty <CODE>
```

**Pre-authorize contacts:**

```bash
WECHATY_ALLOWED_USERS=contact:id1,contact:id2
```

**Open access** (dev only):

```bash
WECHATY_ALLOW_ALL_USERS=true
```

## Group @-mentions

By default (`WECHATY_REQUIRE_MENTION=true`), the bot only responds in groups
when `@`-mentioned or when the text matches a wake-word pattern (default:
`@hermes`). DMs are never mention-gated.

```yaml
gateway:
  platforms:
    wechaty:
      extra:
        require_mention: true
        mention_patterns: ["@hermes"]
```

## Cron delivery

```bash
WECHATY_HOME_CHANNEL=contact:wxid_abc123
# or for groups:
WECHATY_HOME_CHANNEL=room:room-id-here
WECHATY_HOME_CHANNEL_NAME=Ops Group
```

Use `deliver=wechaty` or `deliver=wechaty:contact:wxid_abc123` in cron jobs.

## Configuration reference

| Key / env var | Default | Description |
|---------------|---------|-------------|
| `puppet` / `WECHATY_PUPPET` | — | Puppet module name |
| `puppet_token` / `WECHATY_PUPPET_SERVICE_TOKEN` | — | Token for `wechaty-puppet-service` |
| `require_mention` / `WECHATY_REQUIRE_MENTION` | `true` | Gate group messages on @ / wake word |
| `mention_patterns` / `WECHATY_MENTION_PATTERNS` | Hermes defaults | Regex wake words for groups |
| `sidecar_port` / `WECHATY_SIDECAR_PORT` | `8790` | Loopback control port |
| `sidecar_autostart` / `WECHATY_SIDECAR_AUTOSTART` | `true` | Spawn sidecar on connect |
| `allowed_users` / `WECHATY_ALLOWED_USERS` | — | Comma-separated contact ids |
| `home_channel` / `WECHATY_HOME_CHANNEL` | — | `contact:ID` or `room:ID` for cron |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Sidecar deps missing | `cd plugins/platforms/wechaty/sidecar && npm install` |
| Port in use | Change `WECHATY_SIDECAR_PORT` |
| QR not visible | Check logs for `wechaty.js.org/qrcode/` URL |
| Group messages ignored | Ensure `@` mention or disable `require_mention` |
| Login unstable | Try a different puppet; wechat4u is fragile |

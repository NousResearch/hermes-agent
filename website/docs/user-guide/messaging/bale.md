---
sidebar_position: 2
title: "Bale"
description: "Connect Hermes Agent to a Bale bot"
---

# Bale Setup

Hermes can connect to [Bale](https://bale.ai/) through Bale's
Telegram-compatible Bot API. The adapter keeps Bale tokens, access rules,
sessions, proxy settings, and scheduled-delivery targets separate from a
Telegram bot running in the same gateway.

## Create and configure a bot

1. In Bale, open **BotFather** and create a bot.
2. Copy the bot token into `~/.hermes/.env`.
3. Add your numeric Bale user ID to the allowlist.

```bash
BALE_BOT_TOKEN=replace-with-your-bot-token
BALE_ALLOWED_USERS=123456789
```

Keep the token private. If it is exposed, revoke it in BotFather and replace
the value in `.env`.

Start or restart the gateway:

```bash
hermes gateway
```

Then send the bot a direct message. `hermes gateway status` should list Bale as
configured and connected.

## Access control

`BALE_ALLOWED_USERS` accepts a comma-separated list of numeric user IDs. The
adapter denies unlisted users. For an intentionally public development bot,
you can set `BALE_ALLOW_ALL_USERS=true`; do not use that setting for a bot with
access to terminal or private-data tools.

The standard gateway `unauthorized_dm_behavior: pair` option also works with
Bale if you prefer approval-based DM pairing over a static allowlist.

## Scheduled delivery

Set a default chat for cron jobs and notifications:

```bash
BALE_HOME_CHANNEL=123456789
```

Jobs can then use `deliver=bale`. Standalone delivery uses Bale's API directly,
so it also works when the cron process is separate from the gateway process.

## Custom endpoint or proxy

The official API endpoint is selected by default. Compatible private relays
can be configured without changing code:

```bash
BALE_API_BASE_URL=https://tapi.bale.ai
BALE_PROXY=http://127.0.0.1:8080
```

The adapter normalizes the API base to the `/bot` form required by
`python-telegram-bot`. `BALE_PROXY` takes precedence over standard
`HTTPS_PROXY`, `HTTP_PROXY`, and `ALL_PROXY` settings.

## Troubleshooting

| Problem | Check |
|---|---|
| Bot does not start | Confirm `BALE_BOT_TOKEN` is present and has no surrounding quotes or spaces. |
| Messages are ignored | Confirm the sender's numeric ID is in `BALE_ALLOWED_USERS`. |
| Scheduled message is not delivered | Set `BALE_HOME_CHANNEL` to the destination chat ID. |
| Requests time out | Test the endpoint from the gateway host or configure `BALE_PROXY`. |

The Bale adapter intentionally does not log raw updates or message bodies.
Enable ordinary gateway debug logging for transport diagnostics without
copying private chat content into retained logs.

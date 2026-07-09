# VK Teams

[VK Teams](https://teams.vk.com/) is a corporate messenger by VK, widely used in Russia where Telegram access is unreliable. Its [Bot API](https://teams.vk.com/botapi/) is a close relative of the Telegram Bot API (both descend from the ICQ/Mail.ru Agent lineage), so Hermes drives it as a first-class gateway platform.

The adapter works with both the SaaS cloud (`myteam.mail.ru`) and **on-premise** VK Teams installations — the only difference is the API base URL.

> The adapter uses `httpx`, already a Hermes dependency. No SDK, no daemon, no Node.js.

## Prerequisites

1. In VK Teams, open a chat with **Metabot** (search for `Metabot`).
2. Send `/newbot` and follow the prompts to create a bot. Metabot returns a **bot token** (looks like `001.0123456789.0123456789:1000000`).
3. **On-premise only:** send `/start` to Metabot — it prints your installation's unique **API base URL**. SaaS users can skip this (the default is used).
4. To use the bot in a group, add it as a member; to DM it, message the bot first (a VK Teams bot cannot initiate a dialog).

## Configure Hermes

### Via setup wizard

```bash
hermes gateway setup
```

Select **VK Teams** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```
VKTEAMS_BOT_TOKEN=001.0123456789.0123456789:1000000
# On-premise only — SaaS uses the default below:
# VKTEAMS_API_BASE=https://your-onprem-host/bot/v1
VKTEAMS_ALLOWED_USERS=you@corp.example,teammate@corp.example
VKTEAMS_HOME_CHANNEL=685000000@chat.agent
```

| Variable | Required | Description |
|---|---|---|
| `VKTEAMS_BOT_TOKEN` | Yes | Bot token from Metabot (`/newbot`) |
| `VKTEAMS_API_BASE` | Optional | Bot API base URL. Default `https://myteam.mail.ru/bot/v1`. **On-premise servers each have their own** — ask Metabot with `/start`. |
| `VKTEAMS_ALLOWED_USERS` | Recommended | Comma-separated user IDs/emails allowed to talk to the bot. Use `*` to allow everyone (not recommended). |
| `VKTEAMS_ALLOW_ALL_USERS` | Optional | `true` disables the allowlist (dev only). |
| `VKTEAMS_HOME_CHANNEL` | Optional | Default chat ID for cron / notification delivery. Must be a group the bot is in, or a user who already messaged it. |
| `VKTEAMS_HOME_CHANNEL_NAME` | Optional | Human label for the home channel. |
| `VKTEAMS_PARSE_MODE` | Optional | Outgoing formatting: `HTML` (**recommended** — set this), `MarkdownV2` (code default), or `none` for plain text. VK Teams' MarkdownV2 parser is fragile — it rejects valid inline code and lone `_ * ~` with a whole-message "Format error"; HTML only escapes `& < >` and renders reliably. |
| `VKTEAMS_POLL_TIME` | Optional | Long-poll hold time in seconds (default `25`). |

> **Security:** the bot token is sent as a query parameter on every API request (a property of the VK Teams API). The adapter redacts it from all logs and error messages, but be mindful of corporate proxies that log full URLs.

## Capabilities

| Feature | Supported | Notes |
|---|---|---|
| Text messages | ✅ | HTML (recommended) / MarkdownV2 / plain, auto-chunked at 4096 chars |
| Streaming (live edits) | ✅ | `editText`; paced to the group rate limit |
| Typing indicator | ✅ | `chats/sendActions` |
| Files / images / video | ✅ | `sendFile`, up to 50 MB |
| Voice messages | ✅ | `sendVoice` (playable for `.aac` / `.ogg` / `.m4a`) |
| Inline buttons | ✅ | Command approvals, `/clarify`, slash-confirm |
| Replies / quotes | ✅ | `replyMsgId` |
| Message deletion | ✅ | 48-hour window; group messages need admin rights |
| Threads | ❌ | Not exposed by the Bot API |
| Reactions | ❌ | Not exposed by the Bot API |
| Media albums | ❌ | Sent as individual files |

## Rate limits

VK Teams allows ~30 messages/second to private dialogs but only **1 message/second per group chat**. The adapter paces streaming edits accordingly and honors the server's `Retry-After` on rate-limit errors, so streaming into a busy group stays within budget (at the cost of a slightly slower live update in groups).

## Limitations

- **No webhooks** — the adapter uses long polling only (the API does not support webhooks).
- **The bot cannot start a conversation.** For cron delivery to a specific user, that user must have messaged the bot first; otherwise deliver to a group the bot belongs to.
- **Formatting.** Prefer `VKTEAMS_PARSE_MODE=HTML`. If you see a "Format error" fallback to plain text in the logs, you are likely on `MarkdownV2` — switch to `HTML`. As a last resort (very old on-premise builds), `VKTEAMS_PARSE_MODE=none` disables formatting entirely.
- **On-premise variance.** Older on-premise builds may predate some features (e.g. `parseMode` on incoming events, button styles).

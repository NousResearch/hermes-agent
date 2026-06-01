---
title: "Fluxer"
description: "Use Hermes from Fluxer, an open-source Discord-like chat platform with self-hostable deployments."
---

# Fluxer

[Fluxer](https://fluxer.app/) is a Discord-like open-source chat platform with channels, DMs, rich Markdown, reactions, and file uploads. Hermes talks to Fluxer through the bot REST API for outbound messages and the Fluxer Gateway WebSocket for inbound events.

Fluxer is a good fit when you want a self-hostable, chat-first surface for Hermes with Discord-style UX but without depending on Discord.

> Run `hermes gateway setup` and pick **Fluxer** for a guided setup when the plugin is installed.

## What works

| Feature | Status | Notes |
|---|---:|---|
| Text chat | ✅ | DMs and channels |
| Markdown | ✅ | Hermes sends normal Markdown; Fluxer renders the supported subset |
| Images/files | ✅ | Uploads use Fluxer's multipart message endpoint |
| Audio / voice files | ✅ | Works when the Fluxer deployment exposes file/voice-message support |
| Typing indicator | ✅ | Refreshed while Hermes is working |
| Message edits | ✅ | Used for progress/stream-style updates where Hermes edits a sent message |
| Message deletes | ✅ | Gateway cleanup and deleted approval prompts are handled |
| Reactions | ✅ | Still accepted as compatibility/fallback events |
| Buttons/components | Deployment-dependent | Hermes has guarded adapter support for component payloads and `INTERACTION_CREATE` callbacks, but the current hosted Fluxer API schema does **not** accept/persist `components` on messages. Use text fallbacks (`/approve`, `/deny`) unless your deployment explicitly exposes native components. |
| Native slash commands | Deployment-dependent | Hermes can translate Fluxer `APPLICATION_COMMAND` interactions if a deployment emits them, but the current hosted API spec does not expose application-command registration routes. Keep `FLUXER_REGISTER_NATIVE_COMMANDS=false` unless your deployment provides those routes. |
| Message pins | ✅ | Uses Fluxer's `/channels/{id}/messages/pins` routes when the server enables pins |
| Threads | — | Hosted Fluxer currently exposes reply references, but not Discord-style thread routes in the public API spec. Hermes keeps speculative helpers for deployments that add thread endpoints, but the platform-comparison table does not count this as end-to-end thread support. |
| Channel directory | ✅ | Lists guild channels, forums when exposed, and session-discovered DMs/known channels |
| Backlog recovery | ✅ | Recent messages in known channels are scanned after startup/reconnect |

## Prerequisites

- A Fluxer bot token in the form `<applicationId>.<secret>`
- The channel or DM ID where Hermes should live
- Optional but recommended: the Fluxer user IDs allowed to talk to the bot

The hosted API defaults to:

```text
https://api.fluxer.app/v1
```

Self-hosted deployments can override the API base with `FLUXER_BASE_URL`.

## Configure Hermes

### Via setup wizard

```bash
hermes gateway setup
```

Select **Fluxer** and follow the prompts.

### Via environment variables

Add these to `~/.hermes/.env`:

```bash
FLUXER_BOT_TOKEN=app_id.secret
FLUXER_HOME_CHANNEL=123456789012345678
FLUXER_ALLOWED_USERS=123456789012345678
```

Then restart the gateway:

```bash
hermes gateway restart
```

For a self-hosted Fluxer instance:

```bash
FLUXER_BASE_URL=https://fluxer.example.com
```

If you already know the WebSocket gateway URL and want to skip discovery:

```bash
FLUXER_GATEWAY_URL=wss://fluxer.example.com/gateway
```

Normally Hermes discovers this from `GET /gateway/bot`.

## Environment variables

| Variable | Required | Description |
|---|---:|---|
| `FLUXER_BOT_TOKEN` | Yes | Bot token, usually `<applicationId>.<secret>` |
| `FLUXER_BASE_URL` | Optional | REST API base URL. Defaults to `https://api.fluxer.app/v1`. Plain self-hosted origins are normalized to `/api`. |
| `FLUXER_GATEWAY_URL` | Optional | Override Gateway WebSocket URL. Normally discovered via `/gateway/bot`. |
| `FLUXER_HOME_CHANNEL` | Recommended | Default channel/DM for cron delivery and notifications. Also treated as a free-response channel. |
| `FLUXER_HOME_CHANNEL_NAME` | Optional | Human label for the home channel. |
| `FLUXER_ALLOWED_USERS` | Recommended | Comma-separated Fluxer user IDs allowed to talk to Hermes. |
| `FLUXER_ALLOW_ALL_USERS` | Optional | Allow every Fluxer user. Only safe for private/dev deployments. Default: `false`. |
| `FLUXER_ALLOWED_CHANNELS` | Optional | Comma-separated channel IDs where Hermes may respond. Empty means no channel whitelist. DMs are still governed by user authorization. |
| `FLUXER_FREE_RESPONSE_CHANNELS` | Optional | Comma-separated channel IDs where Hermes responds without being mentioned. |
| `FLUXER_REQUIRE_MENTION` | Optional | Require a direct mention/address in non-home channels. Default: `true`. |
| `FLUXER_STRICT_MENTION` | Optional | Require a fresh mention on every channel message instead of remembering mentioned threads. Default: `false`. |
| `FLUXER_MENTION_PATTERNS` | Optional | Comma-separated regexes that count as direct addresses. Defaults already cover `Hermes` at the start of a message. |
| `FLUXER_REGISTER_NATIVE_COMMANDS` | Optional | Register Hermes slash commands with Fluxer on gateway connect. Default: `false` so deployments can opt in deliberately. |
| `FLUXER_APPLICATION_ID` | Optional | Fluxer application ID for native command registration. If omitted, Hermes uses the token prefix before the first `.`. |
| `FLUXER_NATIVE_COMMAND_GUILDS` | Optional | Comma-separated guild IDs for guild-scoped command registration. Empty means global application commands. |
| `FLUXER_BACKLOG_RECOVERY` | Optional | Recover recent messages from known channels after startup/reconnect. Default: `true`. |
| `FLUXER_BACKLOG_LIMIT` | Optional | Max recent messages to scan per known channel. Default: `25`, capped at `100`. |
| `FLUXER_BACKLOG_BOOTSTRAP_SECONDS` | Optional | Startup lookback window when no previous disconnect time exists. Default: `120`, capped at `900`. |
| `FLUXER_DELIVERY_VERIFICATION` | Optional | Read back sent/edited messages to verify visible delivery. Default: `true`. |
| `FLUXER_ALLOW_MENTION_EVERYONE` | Optional | Permit outbound `@everyone` / `@here`. Default: `false`; Hermes neutralizes them. |
| `FLUXER_ALLOW_MENTION_ROLES` | Optional | Permit outbound role mentions like `<@&...>`. Default: `false`; Hermes neutralizes them. |
| `FLUXER_ALLOW_MENTION_USERS` | Optional | Permit outbound user mentions. Default: `true`. |
| `FLUXER_ALLOW_MENTION_REPLIED_USER` | Optional | Permit reply notifications to the replied-to user when Fluxer supports `allowed_mentions`. Default: `true`. |

## Authorization and channel behavior

Fluxer uses the normal Hermes gateway authorization model:

```bash
FLUXER_ALLOWED_USERS=123456789012345678,234567890123456789
```

For development-only private deployments, you can allow everyone:

```bash
FLUXER_ALLOW_ALL_USERS=true
```

Do not use `FLUXER_ALLOW_ALL_USERS=true` on a public or shared server if Hermes has terminal/tool access.

Channel behavior is intentionally quiet by default:

- DMs are processed when the user is authorized.
- `FLUXER_HOME_CHANNEL` is processed without a mention, so your main Hermes room feels natural.
- Other channels require a direct mention/address by default.
- Casual mentions in the middle of a conversation do not wake the bot.

Examples:

```text
Hermes, check this please     # accepted
@Hermes summarize the thread  # accepted
I think Hermes said that      # ignored in channels
```

Set `FLUXER_FREE_RESPONSE_CHANNELS` for additional channels where Hermes should always respond.

## Native interactions

Hermes keeps Fluxer native-interaction support guarded and deployment-aware:

- if a Fluxer deployment accepts `components`, Hermes can send action rows/buttons for dangerous-command approvals and slash confirmations;
- if a deployment emits `INTERACTION_CREATE` with `type=3` / `MESSAGE_COMPONENT`, Hermes resolves the matching pending action;
- if a deployment emits `INTERACTION_CREATE` with `type=2` / `APPLICATION_COMMAND`, Hermes translates it into normal slash text such as `/model gpt-5.5`.

The current hosted Fluxer API spec does **not** list message `components`, interaction callback routes, or application-command registration routes. In that environment, native buttons and registered slash commands should be treated as unavailable and Hermes falls back to text commands.

Native command registration is opt-in and should only be enabled for deployments that expose the required application-command routes:

```bash
FLUXER_REGISTER_NATIVE_COMMANDS=true
FLUXER_APPLICATION_ID=app_id   # optional when FLUXER_BOT_TOKEN is app_id.secret
```

For fast guild-scoped registration during testing on compatible deployments:

```bash
FLUXER_REGISTER_NATIVE_COMMANDS=true
FLUXER_NATIVE_COMMAND_GUILDS=123456789012345678,234567890123456789
```

## Dangerous-command approvals

On deployments with component support, Hermes can show native dangerous-command approval buttons:

| Button | Meaning |
|---|---|
| Allow once | approve once |
| Session | approve for this session |
| Always | always allow this command pattern |
| Deny | deny |

On the hosted Fluxer API today, use the text fallback instead:

```text
/approve
/approve session
/approve always
/deny
```

If the approval prompt is deleted before it is resolved, Hermes fails closed and denies the pending approval.

## Media, voice, and streaming

Hermes can send images, videos, documents, audio files, and voice-message-shaped audio through Fluxer's multipart message API. Inbound attachments are cached and passed to the agent as media URLs/types.

Voice/audio support depends on the Fluxer deployment exposing the relevant file and voice-message fields. If a deployment only supports plain files, audio may appear as a normal file instead of a voice bubble.

Hermes streaming on Fluxer is edit-based: the gateway sends or reuses a progress bubble, then updates it with `PATCH /channels/{channel_id}/messages/{message_id}`. The hosted API accepts message edits, so Fluxer counts as supporting Hermes-style streaming/progress updates even though it is not a token-by-token websocket stream to the client.

## Pins, replies, and channel directory

Hermes includes Fluxer pin helpers for deployments that expose pin routes:

```text
GET    /channels/{channel_id}/messages/pins
PUT    /channels/{channel_id}/messages/pins/{message_id}
DELETE /channels/{channel_id}/messages/pins/{message_id}
```

The hosted Fluxer API has been verified to list pins with `GET /channels/{id}/messages/pins`.

Fluxer hosted API supports Discord-like reply references via `message_reference`; Hermes uses those for reply context and referenced-message lookup.

The adapter also keeps guarded helpers for deployments that add Discord-style thread endpoints, but the current hosted public API spec does not expose those routes. Do not rely on thread creation or active-thread enumeration unless your Fluxer deployment documents the routes.

`send_message(action="list")` includes Fluxer guild channels, forums when exposed, and known/session-discovered DMs/channels when the deployment exposes `/users/@me/guilds` and `/guilds/{guild_id}/channels`. Channel gateway events update the adapter's known-channel set, so reconnect backlog recovery can include newly revealed channels.

## Backlog recovery

The Gateway WebSocket is the primary inbound path. On startup or reconnect, Hermes also scans recent messages from known channels so short outages do not silently drop messages.

Known channels include:

- `FLUXER_HOME_CHANNEL`
- `home_channel` from config
- channels the adapter has sent to or received from during the current process

Backlog recovery is conservative:

- It only scans a bounded number of messages.
- It ignores messages older than the reconnect/startup cutoff.
- It dedupes by message ID.
- It still filters self-messages, unauthorized users, and channel mention-gating.

Disable it if your Fluxer deployment cannot list channel messages:

```bash
FLUXER_BACKLOG_RECOVERY=false
```

## Cron and `send_message`

With `FLUXER_HOME_CHANNEL` set, cron jobs can deliver to Fluxer:

```python
cronjob(
    action="create",
    schedule="0 9 * * *",
    deliver="fluxer",
    prompt="Send a short morning status."
)
```

Or send explicitly:

```python
send_message(target="fluxer:123456789012345678", message="Done.")
```

The plugin registers a standalone sender, so cron delivery also works when the cron runner is not the live gateway process.

## Troubleshooting

**Gateway says Fluxer is not configured** — Check `FLUXER_BOT_TOKEN`. `FLUXER_BASE_URL` is optional; the hosted API is the default.

**Connect fails with gateway URL missing** — Hermes could not fetch a usable URL from `/gateway/bot`. Set `FLUXER_GATEWAY_URL` explicitly and restart the gateway.

**Messages send but don't appear** — Keep `FLUXER_DELIVERY_VERIFICATION=true` and check gateway logs. Hermes reads the sent message back when possible and logs delivery mismatch warnings.

**The bot answers in channels where it should stay quiet** — Check `FLUXER_HOME_CHANNEL`, `FLUXER_FREE_RESPONSE_CHANNELS`, and `FLUXER_REQUIRE_MENTION`. The home channel is intentionally free-response.

**The bot ignores channel messages** — In non-home channels, address it directly (`Hermes, ...` or a real bot mention), add the channel to `FLUXER_FREE_RESPONSE_CHANNELS`, or set `FLUXER_REQUIRE_MENTION=false`.

**Dangerous-command reactions do nothing** — Make sure the reacting user's Fluxer ID is in `FLUXER_ALLOWED_USERS` or `FLUXER_ALLOW_ALL_USERS=true` is set for a private/dev setup. Bot reactions are ignored.

**Reconnects / heartbeat warnings** — The adapter responds to server heartbeat requests and sends regular heartbeats. Fresh `4009 Heartbeat timeout` warnings usually mean the deployment closed the WebSocket before the bot could identify/heartbeat; check network path, `FLUXER_GATEWAY_URL`, and server logs.

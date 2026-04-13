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
| Group chats | Hermes responds only when the bot is @mentioned in the chat. |
| Shared group chats | By default, session history is isolated per user inside a shared chat. |

This shared-chat behavior is controlled by `config.yaml`:

```yaml
group_sessions_per_user: true
```

Set it to `false` only if you explicitly want one shared conversation per chat.

## Step 1: Create a Feishu / Lark App

### Recommended: Scan-to-Create (one command)

```bash
hermes gateway setup
```

Select **Feishu / Lark** and scan the QR code with your Feishu or Lark mobile app. Hermes will automatically create a bot application with the correct permissions and save the credentials.

### Alternative: Manual Setup

If scan-to-create is not available, the wizard falls back to manual input:

1. Open the Feishu or Lark developer console:
   - Feishu: [https://open.feishu.cn/](https://open.feishu.cn/)
   - Lark: [https://open.larksuite.com/](https://open.larksuite.com/)
2. Create a new app.
3. In **Credentials & Basic Info**, copy the **App ID** and **App Secret**.
4. Enable the **Bot** capability for the app.
5. Run `hermes gateway setup`, select **Feishu / Lark**, and enter the credentials when prompted.

:::warning
Keep the App Secret private. Anyone with it can impersonate your app.
:::

## Step 2: Choose a Connection Mode

### Recommended: WebSocket mode

Use WebSocket mode when Hermes runs on your laptop, workstation, or a private server. No public URL is required. The official Lark SDK opens and maintains a persistent outbound WebSocket connection with automatic reconnection.

```bash
FEISHU_CONNECTION_MODE=websocket
```

**Requirements:** The `websockets` Python package must be installed. The SDK handles connection lifecycle, heartbeats, and auto-reconnection internally.

**How it works:** The adapter runs the Lark SDK's WebSocket client in a background executor thread. Inbound events (messages, reactions, card actions) are dispatched to the main asyncio loop. On disconnect, the SDK will attempt to reconnect automatically.

### Optional: Webhook mode

Use webhook mode only when you already run Hermes behind a reachable HTTP endpoint.

```bash
FEISHU_CONNECTION_MODE=webhook
```

In webhook mode, Hermes starts an HTTP server (via `aiohttp`) and serves a Feishu endpoint at:

```text
/feishu/webhook
```

**Requirements:** The `aiohttp` Python package must be installed.

You can customize the webhook server bind address and path:

```bash
FEISHU_WEBHOOK_HOST=127.0.0.1   # default: 127.0.0.1
FEISHU_WEBHOOK_PORT=8765         # default: 8765
FEISHU_WEBHOOK_PATH=/feishu/webhook  # default: /feishu/webhook
```

When Feishu sends a URL verification challenge (`type: url_verification`), the webhook responds automatically so you can complete the subscription setup in the Feishu developer console.

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

### User Allowlist

For production use, set an allowlist of Feishu Open IDs:

```bash
FEISHU_ALLOWED_USERS=ou_xxx,ou_yyy
```

If you leave the allowlist empty, anyone who can reach the bot may be able to use it. In group chats, the allowlist is checked against the sender's open_id before the message is processed.

### Webhook Encryption Key

When running in webhook mode, set an encryption key to enable signature verification of inbound webhook payloads:

```bash
FEISHU_ENCRYPT_KEY=your-encrypt-key
```

This key is found in the **Event Subscriptions** section of your Feishu app configuration. When set, the adapter verifies every webhook request using the signature algorithm:

```
SHA256(timestamp + nonce + encrypt_key + body)
```

The computed hash is compared against the `x-lark-signature` header using timing-safe comparison. Requests with invalid or missing signatures are rejected with HTTP 401.

:::tip
In WebSocket mode, signature verification is handled by the SDK itself, so `FEISHU_ENCRYPT_KEY` is optional. In webhook mode, it is strongly recommended for production.
:::

### Verification Token

An additional layer of authentication that checks the `token` field inside webhook payloads:

```bash
FEISHU_VERIFICATION_TOKEN=your-verification-token
```

This token is also found in the **Event Subscriptions** section of your Feishu app. When set, every inbound webhook payload must contain a matching `token` in its `header` object. Mismatched tokens are rejected with HTTP 401.

Both `FEISHU_ENCRYPT_KEY` and `FEISHU_VERIFICATION_TOKEN` can be used together for defense in depth.

## Group Message Policy

The `FEISHU_GROUP_POLICY` environment variable controls whether and how Hermes responds in group chats:

```bash
FEISHU_GROUP_POLICY=allowlist   # default
```

| Value | Behavior |
|-------|----------|
| `open` | Hermes responds to @mentions from any user in any group. |
| `allowlist` | Hermes only responds to @mentions from users listed in `FEISHU_ALLOWED_USERS`. |
| `disabled` | Hermes ignores all group messages entirely. |

In all modes, the bot must be explicitly @mentioned (or @all) in the group before the message is processed. Direct messages bypass this gate.

### Bot Identity for @Mention Gating

For precise @mention detection in groups, the adapter needs to know the bot's identity. It can be provided explicitly:

```bash
FEISHU_BOT_OPEN_ID=ou_xxx
FEISHU_BOT_USER_ID=xxx
FEISHU_BOT_NAME=MyBot
```

If none of these are set, the adapter will attempt to auto-discover the bot name via the Application Info API on startup. For this to work, grant the `admin:app.info:readonly` or `application:application:self_manage` permission scope.

## Interactive Card Actions

When users click buttons or interact with interactive cards sent by the bot, the adapter routes these as synthetic `/card` command events:

- Button clicks become: `/card button {"key": "value", ...}`
- The action's `value` payload from the card definition is included as JSON.
- Card actions are deduplicated with a 15-minute window to prevent double processing.

Card action events are dispatched with `MessageType.COMMAND`, so they flow through the normal command processing pipeline.

This is also how **command approval** works — when the agent needs to run a dangerous command, it sends an interactive card with Allow Once / Session / Always / Deny buttons. The user clicks a button, and the card action callback delivers the approval decision back to the agent.

### Required Feishu App Configuration

Interactive cards require **three** configuration steps in the Feishu Developer Console. Missing any of them causes error **200340** when users click card buttons.

1. **Subscribe to the card action event:**
   In **Event Subscriptions**, add `card.action.trigger` to your subscribed events.

2. **Enable the Interactive Card capability:**
   In **App Features > Bot**, ensure the **Interactive Card** toggle is enabled. This tells Feishu that your app can receive card action callbacks.

3. **Configure the Card Request URL (webhook mode only):**
   In **App Features > Bot > Message Card Request URL**, set the URL to the same endpoint as your event webhook (e.g. `https://your-server:8765/feishu/webhook`). In WebSocket mode this is handled automatically by the SDK.

:::warning
Without all three steps, Feishu will successfully *send* interactive cards (sending only requires `im:message:send` permission), but clicking any button will return error 200340. The card appears to work — the error only surfaces when a user interacts with it.
:::

## Media Support

### Inbound (receiving)

The adapter receives and caches the following media types from users:

| Type | Extensions | How it's processed |
|------|-----------|-------------------|
| **Images** | .jpg, .jpeg, .png, .gif, .webp, .bmp | Downloaded via Feishu API and cached locally |
| **Audio** | .ogg, .mp3, .wav, .m4a, .aac, .flac, .opus, .webm | Downloaded and cached; small text files are auto-extracted |
| **Video** | .mp4, .mov, .avi, .mkv, .webm, .m4v, .3gp | Downloaded and cached as documents |
| **Files** | .pdf, .doc, .docx, .xls, .xlsx, .ppt, .pptx, and more | Downloaded and cached as documents |

Media from rich-text (post) messages, including inline images and file attachments, is also extracted and cached.

For small text-based documents (.txt, .md), the file content is automatically injected into the message text so the agent can read it directly without needing tools.

### Outbound (sending)

| Method | What it sends |
|--------|--------------|
| `send` | Text or rich post messages (auto-detected based on markdown content) |
| `send_image` / `send_image_file` | Uploads image to Feishu, then sends as native image bubble (with optional caption) |
| `send_document` | Uploads file to Feishu API, then sends as file attachment |
| `send_voice` | Uploads audio file as a Feishu file attachment |
| `send_video` | Uploads video and sends as native media message |
| `send_animation` | GIFs are downgraded to file attachments (Feishu has no native GIF bubble) |

File upload routing is automatic based on extension:

- `.ogg`, `.opus` → uploaded as `opus` audio
- `.mp4`, `.mov`, `.avi`, `.m4v` → uploaded as `mp4` media
- `.pdf`, `.doc(x)`, `.xls(x)`, `.ppt(x)` → uploaded with their document type
- Everything else → uploaded as a generic stream file

## Markdown Rendering and Post Fallback

When outbound text contains markdown formatting (headings, bold, lists, code blocks, links, etc.), the adapter automatically sends it as a Feishu **post** message with an embedded `md` tag rather than as plain text. This enables rich rendering in the Feishu client.

If the Feishu API rejects the post payload (e.g., due to unsupported markdown constructs), the adapter automatically falls back to sending as plain text with markdown stripped. This two-stage fallback ensures messages are always delivered.

Plain text messages (no markdown detected) are sent as the simple `text` message type.

## ACK Emoji Reactions

When the adapter receives an inbound message, it immediately adds an ✅ (OK) emoji reaction to signal that the message was received and is being processed. This provides visual feedback before the agent completes its response.

The reaction is persistent — it remains on the message after the response is sent, serving as a receipt marker.

User reactions on bot messages are also tracked. If a user adds or removes an emoji reaction on a message sent by the bot, it is routed as a synthetic text event (`reaction:added:EMOJI_TYPE` or `reaction:removed:EMOJI_TYPE`) so the agent can respond to feedback.

## Burst Protection and Batching

The adapter includes debouncing for rapid message bursts to avoid overwhelming the agent:

### Text Batching

When a user sends multiple text messages in quick succession, they are merged into a single event before being dispatched:

| Setting | Env Var | Default |
|---------|---------|---------|
| Quiet period | `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | 0.6s |
| Max messages per batch | `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | 8 |
| Max characters per batch | `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | 4000 |

### Media Batching

Multiple media attachments sent in quick succession (e.g., dragging several images) are merged into a single event:

| Setting | Env Var | Default |
|---------|---------|---------|
| Quiet period | `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | 0.8s |

### Per-Chat Serialization

Messages within the same chat are processed serially (one at a time) to maintain conversation coherence. Each chat has its own lock, so messages in different chats are processed concurrently.

## Rate Limiting (Webhook Mode)

In webhook mode, the adapter enforces per-IP rate limiting to protect against abuse:

- **Window:** 60-second sliding window
- **Limit:** 120 requests per window per (app_id, path, IP) triple
- **Tracking cap:** Up to 4096 unique keys tracked (prevents unbounded memory growth)

Requests that exceed the limit receive HTTP 429 (Too Many Requests).

### Webhook Anomaly Tracking

The adapter tracks consecutive error responses per IP address. After 25 consecutive errors from the same IP within a 6-hour window, a warning is logged. This helps detect misconfigured clients or probing attempts.

Additional webhook protections:
- **Body size limit:** 1 MB maximum
- **Body read timeout:** 30 seconds
- **Content-Type enforcement:** Only `application/json` is accepted

## WebSocket Tuning

When using `websocket` mode, you can customize reconnect and ping behavior:

```yaml
platforms:
  feishu:
    extra:
      ws_reconnect_interval: 120   # Seconds between reconnect attempts (default: 120)
      ws_ping_interval: 30         # Seconds between WebSocket pings (optional; SDK default if unset)
```

| Setting | Config key | Default | Description |
|---------|-----------|---------|-------------|
| Reconnect interval | `ws_reconnect_interval` | 120s | How long to wait between reconnection attempts |
| Ping interval | `ws_ping_interval` | _(SDK default)_ | Frequency of WebSocket keepalive pings |

## Per-Group Access Control

Beyond the global `FEISHU_GROUP_POLICY`, you can set fine-grained rules per group chat using `group_rules` in config.yaml:

```yaml
platforms:
  feishu:
    extra:
      default_group_policy: "open"     # Default for groups not in group_rules
      admins:                          # Users who can manage bot settings
        - "ou_admin_open_id"
      group_rules:
        "oc_group_chat_id_1":
          policy: "allowlist"          # open | allowlist | blacklist | admin_only | disabled
          allowlist:
            - "ou_user_open_id_1"
            - "ou_user_open_id_2"
        "oc_group_chat_id_2":
          policy: "admin_only"
        "oc_group_chat_id_3":
          policy: "blacklist"
          blacklist:
            - "ou_blocked_user"
```

| Policy | Description |
|--------|-------------|
| `open` | Anyone in the group can use the bot |
| `allowlist` | Only users in the group's `allowlist` can use the bot |
| `blacklist` | Everyone except users in the group's `blacklist` can use the bot |
| `admin_only` | Only users in the global `admins` list can use the bot in this group |
| `disabled` | Bot ignores all messages in this group |

Groups not listed in `group_rules` fall back to `default_group_policy` (defaults to the value of `FEISHU_GROUP_POLICY`).

## Deduplication

Inbound messages are deduplicated using message IDs with a 24-hour TTL. The dedup state is persisted across restarts to `~/.hermes/feishu_seen_message_ids.json`.

| Setting | Env Var | Default |
|---------|---------|---------|
| Cache size | `HERMES_FEISHU_DEDUP_CACHE_SIZE` | 2048 entries |

## All Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FEISHU_APP_ID` | ✅ | — | Feishu/Lark App ID |
| `FEISHU_APP_SECRET` | ✅ | — | Feishu/Lark App Secret |
| `FEISHU_DOMAIN` | — | `feishu` | `feishu` (China) or `lark` (international) |
| `FEISHU_CONNECTION_MODE` | — | `websocket` | `websocket` or `webhook` |
| `FEISHU_ALLOWED_USERS` | — | _(empty)_ | Comma-separated open_id list for user allowlist |
| `FEISHU_HOME_CHANNEL` | — | — | Chat ID for cron/notification output |
| `FEISHU_ENCRYPT_KEY` | — | _(empty)_ | Encrypt key for webhook signature verification |
| `FEISHU_VERIFICATION_TOKEN` | — | _(empty)_ | Verification token for webhook payload auth |
| `FEISHU_GROUP_POLICY` | — | `allowlist` | Group message policy: `open`, `allowlist`, `disabled` |
| `FEISHU_BOT_OPEN_ID` | — | _(empty)_ | Bot's open_id (for @mention detection) |
| `FEISHU_BOT_USER_ID` | — | _(empty)_ | Bot's user_id (for @mention detection) |
| `FEISHU_BOT_NAME` | — | _(empty)_ | Bot's display name (for @mention detection) |
| `FEISHU_WEBHOOK_HOST` | — | `127.0.0.1` | Webhook server bind address |
| `FEISHU_WEBHOOK_PORT` | — | `8765` | Webhook server port |
| `FEISHU_WEBHOOK_PATH` | — | `/feishu/webhook` | Webhook endpoint path |
| `HERMES_FEISHU_DEDUP_CACHE_SIZE` | — | `2048` | Max deduplicated message IDs to track |
| `HERMES_FEISHU_TEXT_BATCH_DELAY_SECONDS` | — | `0.6` | Text burst debounce quiet period |
| `HERMES_FEISHU_TEXT_BATCH_MAX_MESSAGES` | — | `8` | Max messages merged per text batch |
| `HERMES_FEISHU_TEXT_BATCH_MAX_CHARS` | — | `4000` | Max characters merged per text batch |
| `HERMES_FEISHU_MEDIA_BATCH_DELAY_SECONDS` | — | `0.8` | Media burst debounce quiet period |

WebSocket and per-group ACL settings are configured via `config.yaml` under `platforms.feishu.extra` (see [WebSocket Tuning](#websocket-tuning) and [Per-Group Access Control](#per-group-access-control) above).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `lark-oapi not installed` | Install the SDK: `pip install lark-oapi` |
| `websockets not installed; websocket mode unavailable` | Install websockets: `pip install websockets` |
| `aiohttp not installed; webhook mode unavailable` | Install aiohttp: `pip install aiohttp` |
| `FEISHU_APP_ID or FEISHU_APP_SECRET not set` | Set both env vars or configure via `hermes gateway setup` |
| `Another local Hermes gateway is already using this Feishu app_id` | Only one Hermes instance can use the same app_id at a time. Stop the other gateway first. |
| Bot doesn't respond in groups | Ensure the bot is @mentioned, check `FEISHU_GROUP_POLICY`, and verify the sender is in `FEISHU_ALLOWED_USERS` if policy is `allowlist` |
| `Webhook rejected: invalid verification token` | Ensure `FEISHU_VERIFICATION_TOKEN` matches the token in your Feishu app's Event Subscriptions config |
| `Webhook rejected: invalid signature` | Ensure `FEISHU_ENCRYPT_KEY` matches the encrypt key in your Feishu app config |
| Post messages show as plain text | The Feishu API rejected the post payload; this is normal fallback behavior. Check logs for details. |
| Images/files not received by bot | Grant `im:message` and `im:resource` permission scopes to your Feishu app |
| Bot identity not auto-detected | Grant `admin:app.info:readonly` scope, or set `FEISHU_BOT_OPEN_ID` / `FEISHU_BOT_NAME` manually |
| Error 200340 when clicking approval buttons | Enable **Interactive Card** capability and configure **Card Request URL** in the Feishu Developer Console. See [Required Feishu App Configuration](#required-feishu-app-configuration) above. |
| `Webhook rate limit exceeded` | More than 120 requests/minute from the same IP. This is usually a misconfiguration or loop. |

## Toolset

Feishu / Lark uses the `hermes-feishu` platform preset, which includes the same core tools as Telegram and other gateway-based messaging platforms.

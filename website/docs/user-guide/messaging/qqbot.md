# QQ Bot

Connect Hermes to QQ via the **Official QQ Bot API (v2)** — supporting private (C2C), group @-mentions, guild, and direct messages with voice transcription.

## Overview

The QQ Bot adapter uses the [Official QQ Bot API](https://bot.q.qq.com/wiki/develop/api-v2/) to:

- Receive messages via a persistent **WebSocket** connection to the QQ Gateway
- Send text and markdown replies via the **REST API**
- Download and process images, voice messages, and file attachments
- Transcribe voice messages using Tencent's built-in ASR or a configurable STT provider

## Prerequisites

1. **QQ Bot Application** — Register at [q.qq.com](https://q.qq.com):
   - Create a new application and note your **App ID** and **App Secret**
   - Enable the required intents: C2C messages, Group @-messages, Guild messages
   - Configure your bot in sandbox mode for testing, or publish for production

2. **Dependencies** — The adapter requires `aiohttp` and `httpx`:
   ```bash
   pip install aiohttp httpx
   ```

## Configuration

### Interactive setup

```bash
hermes gateway setup
```

Select **QQ Bot** from the platform list and follow the prompts.

### Manual configuration

Set the required environment variables in `~/.hermes/.env`:

```bash
QQ_APP_ID=your-app-id
QQ_CLIENT_SECRET=your-app-secret
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `QQ_APP_ID` | QQ Bot App ID (required) | — |
| `QQ_CLIENT_SECRET` | QQ Bot App Secret (required) | — |
| `QQBOT_HOME_CHANNEL` | OpenID for cron/notification delivery | — |
| `QQBOT_HOME_CHANNEL_NAME` | Display name for home channel | `Home` |
| `QQ_ALLOWED_USERS` | Comma-separated user OpenIDs for DM access | open (all users) |
| `QQ_GROUP_ALLOWED_USERS` | Comma-separated group OpenIDs for group access | — |
| `QQ_ALLOW_ALL_USERS` | Set to `true` to allow all DMs | `false` |
| `QQ_PORTAL_HOST` | Override the QQ portal host (set to `sandbox.q.qq.com` for sandbox routing) | `q.qq.com` |
| `QQ_STT_API_KEY` | API key for voice-to-text provider | — |
| `QQ_STT_BASE_URL` | (Not read directly — set `platforms.qqbot.extra.stt.baseUrl` in `config.yaml` instead) | n/a |
| `QQ_STT_MODEL` | STT model name | `glm-asr` |

## Advanced Configuration

For fine-grained control, add platform settings to `~/.hermes/config.yaml`:

```yaml
platforms:
  qqbot:
    enabled: true
    extra:
      app_id: "your-app-id"
      client_secret: "your-secret"
      markdown_support: true       # enable QQ markdown (msg_type 2). Config-only; no env-var equivalent.
      dm_policy: "open"          # open | allowlist | disabled
      allow_from:
        - "user_openid_1"
      group_policy: "open"       # open | allowlist | disabled (default: disabled)
      group_allow_from:
        - "group_openid_1"
      # ── Group activation mode (always vs mention) ──
      group_require_mention: true  # true = reply only when @-ed (default);
                                   # false = "always" mode, any group message
                                   # may trigger a reply (needs the "receive all
                                   # group messages" permission on the QQ platform)
      groups:                      # optional per-group override (group > global)
        "group_openid_1":
          require_mention: false   # force always mode for this group only
      # ── Group session sharing ──
      group_sessions_per_user: true  # true (default) = each member gets an
                                     # isolated session; set false so ALL members
                                     # of a group share ONE conversation session
      group_history_limit: 20        # (mention mode) non-@ messages buffered per
                                     # group and injected as CONTEXT ONLY on the
                                     # next @-reply; <=0 disables buffering
      # ── Streaming replies (C2C only) ──
      streaming_enabled: true              # true (default) = use QQ's native
                                           # stream_messages endpoint for C2C
                                           # replies; false forces the legacy
                                           # one-shot send path
      streaming_session_ttl_seconds: 600   # in-memory session TTL; QQ's own
                                           # passive-reply window is ~5 min, so
                                           # the default gives modest headroom
      stt:
        provider: "zai"          # zai (GLM-ASR), openai (Whisper), etc.
        baseUrl: "https://open.bigmodel.cn/api/coding/paas/v4"
        apiKey: "your-stt-key"
        model: "glm-asr"
```

## Streaming Replies

The adapter renders long-form model output as it's generated, matching the
experience users get on Feishu / Telegram / Slack.

- **Private (C2C) chats**: streamed **in place** through QQ's official
  `POST /v2/users/{openid}/stream_messages` endpoint. A single message
  bubble updates progressively as the model produces tokens, and
  transitions out of the "generating…" state on the final chunk
  (`input_state=10`).
- **Group / guild chats**: **not supported by the QQ platform** — the
  `stream_messages` endpoint is C2C-only. The adapter transparently
  falls back to the standard send loop, so users see a small number of
  appended messages during a long response instead of a single
  updating bubble. No configuration is required.

### Constraints

- **Passive-reply window**: QQ requires every streamed chunk to carry
  the inbound `msg_id` as its `event_id` / `msg_id`. That id is only
  valid for ~5 minutes, so responses that take longer to generate may
  see the last edits rejected — the adapter logs a warning and the
  stream consumer will fall back to a fresh send.
- **Content type is fixed to markdown** (`content_type=2`). Plain text
  renders correctly under markdown, and rich formatting works
  end-to-end.
- **Content length**: each chunk is truncated to 4096 characters
  (`stream_messages` accepts less than the regular `/messages`
  endpoint).
- **Ordering**: the stream consumer serialises edits, so out-of-order
  updates should not occur in normal operation. As a defensive
  measure, any edit whose sequence index is not strictly greater than
  the last one sent is silently dropped.
- **Restart-safety**: streaming sessions live in memory. After an
  adapter restart, in-flight edits to a pre-restart message will fail
  gracefully; the consumer sends the remainder as a new message.

### Config knobs

| Key | Default | Effect |
|---|---|---|
| `streaming_enabled` | `true` | Set to `false` to force the legacy send-only path for C2C (useful during incident triage). |
| `streaming_session_ttl_seconds` | `600` | Upper bound on how long an in-flight streaming session is kept in memory. |

## Group Activation Mode

QQ groups support two independent axes:

- **Server push mode** (configured on the QQ platform): mention-only, context, or
  full. Mention-only/context deliver `GROUP_AT_MESSAGE_CREATE`; full delivers
  `GROUP_MESSAGE_CREATE` for every message (with a per-message "is this bot
  @-ed" marker). Full mode requires the **"receive all group messages"**
  permission granted on the QQ platform.
- **Plugin activation mode** (`group_require_mention`):
  - **mention** (default): the bot replies only when explicitly @-ed.
  - **always** (`group_require_mention: false`): any group message may trigger a
    reply — the agent decides whether to respond.

The two axes are orthogonal, so all six combinations work. Whatever events
arrive are run through the same pipeline: the bot detects whether it was
@-mentioned (via the `GROUP_AT` event, a `mentions[].is_you` flag, or an
`<@!app_id>` tag) and then applies the activation gate. `group_require_mention`
can be overridden per group under `groups.{group_openid}.require_mention`.

Group ACL (`group_policy` / `group_allow_from`) is a separate, group-level
concern applied before activation — it decides *which groups* the bot serves,
not which members.

## Voice Messages (STT)

Voice transcription works in two stages:

1. **QQ built-in ASR** (free, always tried first) — QQ provides `asr_refer_text` in voice message attachments, which uses Tencent's own speech recognition
2. **Configured STT provider** (fallback) — If QQ's ASR doesn't return text, the adapter calls an OpenAI-compatible STT API:

   - **Zhipu/GLM (zai)**: Default provider, uses `glm-asr` model
   - **OpenAI Whisper**: Set `QQ_STT_BASE_URL` and `QQ_STT_MODEL`
   - Any OpenAI-compatible STT endpoint

## Troubleshooting

### Bot disconnects immediately (quick disconnect)

This usually means:
- **Invalid App ID / Secret** — Double-check your credentials at q.qq.com
- **Missing permissions** — Ensure the bot has the required intents enabled
- **Sandbox-only bot** — If the bot is in sandbox mode, it can only receive messages from QQ's sandbox test channel

### Voice messages not transcribed

1. Check if QQ's built-in `asr_refer_text` is present in the attachment data
2. If using a custom STT provider, verify `QQ_STT_API_KEY` is set correctly
3. Check gateway logs for STT error messages

### Messages not delivered

- Verify the bot's **intents** are enabled at q.qq.com
- Check `QQ_ALLOWED_USERS` if DM access is restricted
- For group messages, ensure the bot is **@mentioned** (default mention mode), or enable **always mode** via `group_require_mention: false` (group policy may also require allowlisting)
- Check `QQBOT_HOME_CHANNEL` for cron/notification delivery

### Connection errors

- Ensure `aiohttp` and `httpx` are installed: `pip install aiohttp httpx`
- Check network connectivity to `api.sgroup.qq.com` and the WebSocket gateway
- Review gateway logs for detailed error messages and reconnect behavior

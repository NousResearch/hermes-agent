---
sidebar_position: 1
title: "Telegram"
description: "Set up Hermes Agent as a Telegram bot"
---

# Telegram Setup

Hermes Agent integrates with Telegram as a full-featured conversational bot. Once connected, you can chat with your agent from any device, send voice memos that get auto-transcribed, receive scheduled task results, and use the agent in group chats. The integration is built on [python-telegram-bot](https://python-telegram-bot.org/) and supports text, voice, images, and file attachments.

## Step 1: Create a Bot via BotFather

Every Telegram bot requires an API token issued by [@BotFather](https://t.me/BotFather), Telegram's official bot management tool.

1. Open Telegram and search for **@BotFather**, or visit [t.me/BotFather](https://t.me/BotFather)
2. Send `/newbot`
3. Choose a **display name** (e.g., "Hermes Agent") — this can be anything
4. Choose a **username** — this must be unique and end in `bot` (e.g., `my_hermes_bot`)
5. BotFather replies with your **API token**. It looks like this:

```
123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
```

:::warning
Keep your bot token secret. Anyone with this token can control your bot. If it leaks, revoke it immediately via `/revoke` in BotFather.
:::

## Step 2: Customize Your Bot (Optional)

These BotFather commands improve the user experience. Message @BotFather and use:

| Command | Purpose |
|---------|---------|
| `/setdescription` | The "What can this bot do?" text shown before a user starts chatting |
| `/setabouttext` | Short text on the bot's profile page |
| `/setuserpic` | Upload an avatar for your bot |
| `/setcommands` | Define the command menu (the `/` button in chat) |
| `/setprivacy` | Control whether the bot sees all group messages (see Step 3) |

:::tip
For `/setcommands`, a useful starting set:

```
help - Show help information
new - Start a new conversation
sethome - Set this chat as the home channel
```
:::

## Step 3: Privacy Mode (Critical for Groups)

Telegram bots have a **privacy mode** that is **enabled by default**. This is the single most common source of confusion when using bots in groups.

**With privacy mode ON**, your bot can only see:
- Messages that start with a `/` command
- Replies directly to the bot's own messages
- Service messages (member joins/leaves, pinned messages, etc.)
- Messages in channels where the bot is an admin

**With privacy mode OFF**, the bot receives every message in the group.

### How to disable privacy mode

1. Message **@BotFather**
2. Send `/mybots`
3. Select your bot
4. Go to **Bot Settings → Group Privacy → Turn off**

:::warning
**You must remove and re-add the bot to any group** after changing the privacy setting. Telegram caches the privacy state when a bot joins a group, and it will not update until the bot is removed and re-added.
:::

:::tip
An alternative to disabling privacy mode: promote the bot to **group admin**. Admin bots always receive all messages regardless of the privacy setting, and this avoids needing to toggle the global privacy mode.
:::

## Step 4: Find Your User ID

Hermes Agent uses numeric Telegram user IDs to control access. Your user ID is **not** your username — it's a number like `123456789`.

**Method 1 (recommended):** Message [@userinfobot](https://t.me/userinfobot) — it instantly replies with your user ID.

**Method 2:** Message [@get_id_bot](https://t.me/get_id_bot) — another reliable option.

Save this number; you'll need it for the next step.

## Step 5: Configure Hermes

### Option A: Interactive Setup (Recommended)

```bash
hermes gateway setup
```

Select **Telegram** when prompted. The wizard asks for your bot token and allowed user IDs, then writes the configuration for you.

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrSTUvwxYZ
TELEGRAM_ALLOWED_USERS=123456789    # Comma-separated for multiple users
```

### Start the Gateway

```bash
hermes gateway
```

The bot should come online within seconds. Send it a message on Telegram to verify.

## Sending Generated Files from Docker-backed Terminals

If your terminal backend is `docker`, keep in mind that Telegram attachments are
sent by the **gateway process**, not from inside the container. That means the
final `MEDIA:/...` path must be readable on the host where the gateway is
running.

Common pitfall:

- the agent writes a file inside Docker to `/workspace/report.txt`
- the model emits `MEDIA:/workspace/report.txt`
- Telegram delivery fails because `/workspace/report.txt` only exists inside the
  container, not on the host

Recommended pattern:

```yaml
terminal:
  backend: docker
  docker_volumes:
    - "/home/user/.hermes/cache/documents:/output"
```

Then:

- write files inside Docker to `/output/...`
- emit the **host-visible** path in `MEDIA:`, for example:
  `MEDIA:/home/user/.hermes/cache/documents/report.txt`

If you already have a `docker_volumes:` section, add the new mount to the same
list. YAML duplicate keys silently override earlier ones.

### Supported `MEDIA:` file extensions

The gateway extracts `MEDIA:/path/to/file` tags from agent replies and ships the referenced file as a platform-native attachment. Supported extensions across all gateway platforms:

| Category | Extensions |
|---|---|
| Images | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| Audio | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| Video | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **Documents** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **Office** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **Archives** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |
| **Books / packages** | `epub`, `apk`, `ipa` |

Anything on this list delivered as a native attachment on platforms that support it (Telegram, Discord, Signal, Slack, WhatsApp, Feishu, Matrix, etc.); on platforms without native support it falls back to a link or plain-text indicator. The **bold** categories were added in the last few releases — if you were relying on the model saying `here is the file: /path/to/report.docx` instead, swap to `MEDIA:/path/to/report.docx` for native delivery.

## Webhook Mode

By default, Hermes connects to Telegram using **long polling** — the gateway makes outbound requests to Telegram's servers to fetch new updates. This works well for local and always-on deployments.

For **cloud deployments** (Fly.io, Railway, Render, etc.), **webhook mode** is more cost-effective. These platforms can auto-wake suspended machines on inbound HTTP traffic, but not on outbound connections. Since polling is outbound, a polling bot can never sleep. Webhook mode flips the direction — Telegram pushes updates to your bot's HTTPS URL, enabling sleep-when-idle deployments.

| | Polling (default) | Webhook |
|---|---|---|
| Direction | Gateway → Telegram (outbound) | Telegram → Gateway (inbound) |
| Best for | Local, always-on servers | Cloud platforms with auto-wake |
| Setup | No extra config | Set `TELEGRAM_WEBHOOK_URL` |
| Idle cost | Machine must stay running | Machine can sleep between messages |

### Configuration

Add the following to `~/.hermes/.env`:

```bash
TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
# TELEGRAM_WEBHOOK_PORT=8443        # optional, default 8443
# TELEGRAM_WEBHOOK_SECRET=mysecret  # optional, recommended
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | Yes | Public HTTPS URL where Telegram will send updates. The URL path is auto-extracted (e.g., `/telegram` from the example above). |
| `TELEGRAM_WEBHOOK_PORT` | No | Local port the webhook server listens on (default: `8443`). |
| `TELEGRAM_WEBHOOK_SECRET` | No | Secret token for verifying that updates actually come from Telegram. **Strongly recommended** for production deployments. |

When `TELEGRAM_WEBHOOK_URL` is set, the gateway starts an HTTP webhook server instead of polling. When unset, polling mode is used — no behavior change from previous versions.

### Cloud deployment example (Fly.io)

1. Add the env vars to your Fly.io app secrets:

```bash
fly secrets set TELEGRAM_WEBHOOK_URL=https://my-app.fly.dev/telegram
fly secrets set TELEGRAM_WEBHOOK_SECRET=$(openssl rand -hex 32)
```

2. Expose the webhook port in your `fly.toml`:

```toml
[[services]]
  internal_port = 8443
  protocol = "tcp"

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

3. Deploy:

```bash
fly deploy
```

The gateway log should show: `[telegram] Connected to Telegram (webhook mode)`.

## Proxy Support

If Telegram's API is blocked or you need to route traffic through a proxy, set a Telegram-specific proxy URL. This takes priority over the generic `HTTPS_PROXY` / `HTTP_PROXY` env vars.

**Option 1: config.yaml (recommended)**

```yaml
telegram:
  proxy_url: "socks5://127.0.0.1:1080"
```

**Option 2: environment variable**

```bash
TELEGRAM_PROXY=socks5://127.0.0.1:1080
```

Supported schemes: `http://`, `https://`, `socks5://`.

The proxy applies to both the main Telegram connection and the fallback IP transport. If no Telegram-specific proxy is set, the gateway falls back to `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` (or macOS system proxy auto-detection).

## Home Channel

Use the `/sethome` command in any Telegram chat (DM or group) to designate it as the **home channel**. Scheduled tasks (cron jobs) deliver their results to this channel.

You can also set it manually in `~/.hermes/.env`:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
Group chat IDs are negative numbers (e.g., `-1001234567890`). Your personal DM chat ID is the same as your user ID.
:::

## Voice Messages

### Incoming Voice (Speech-to-Text)

Voice messages you send on Telegram are automatically transcribed by Hermes's configured STT provider and injected as text into the conversation.

- `local` uses `faster-whisper` on the machine running Hermes — no API key required
- `groq` uses Groq Whisper and requires `GROQ_API_KEY`
- `openai` uses OpenAI Whisper and requires `VOICE_TOOLS_OPENAI_KEY`

### Outgoing Voice (Text-to-Speech)

When the agent generates audio via TTS, it's delivered as native Telegram **voice bubbles** — the round, inline-playable kind.

- **OpenAI and ElevenLabs** produce Opus natively — no extra setup needed
- **Edge TTS** (the default free provider) outputs MP3 and requires **ffmpeg** to convert to Opus:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

Without ffmpeg, Edge TTS audio is sent as a regular audio file (still playable, but uses the rectangular player instead of a voice bubble).

Configure the TTS provider in your `config.yaml` under the `tts.provider` key.

## Group Chat Usage

Hermes Agent works in Telegram group chats with a few considerations:

- **Privacy mode** determines what messages the bot can see (see [Step 3](#step-3-privacy-mode-critical-for-groups))
- `TELEGRAM_ALLOWED_USERS` still applies — only authorized users can trigger the bot, even in groups
- You can keep the bot from responding to ordinary group chatter with `telegram.require_mention: true`
- With `telegram.require_mention: true`, group messages are accepted when they are:
  - slash commands
  - replies to one of the bot's messages
  - `@botusername` mentions
  - matches for one of your configured regex wake words in `telegram.mention_patterns`
- Use `telegram.ignored_threads` to keep Hermes silent in specific Telegram forum topics, even when the group would otherwise allow free responses or mention-triggered replies
- If `telegram.require_mention` is left unset or false, Hermes keeps the previous open-group behavior and responds to normal group messages it can see

### Example group trigger configuration

Add this to `~/.hermes/config.yaml`:

```yaml
telegram:
  require_mention: true
  mention_patterns:
    - "^\\s*chompy\\b"
  ignored_threads:
    - 31
    - "42"
```

This example allows all the usual direct triggers plus messages that begin with `chompy`, even if they do not use an `@mention`.
Messages in Telegram topics `31` and `42` are always ignored before the mention and free-response checks run.

### Notes on `mention_patterns`

- Patterns use Python regular expressions
- Matching is case-insensitive
- Patterns are checked against both text messages and media captions
- Invalid regex patterns are ignored with a warning in the gateway logs rather than crashing the bot
- If you want a pattern to match only at the start of a message, anchor it with `^`

## Private Chat Topics (Bot API 9.4)

Telegram Bot API 9.4 (February 2026) introduced **Private Chat Topics** — bots can create forum-style topic threads directly in 1-on-1 DM chats, no supergroup needed. This lets you run multiple isolated workspaces within your existing DM with Hermes.

### Use case

If you work on several long-running projects, topics keep their context separate:

- **Topic "Website"** — work on your production web service
- **Topic "Research"** — literature review and paper exploration
- **Topic "General"** — miscellaneous tasks and quick questions

Each topic gets its own conversation session, history, and context — completely isolated from the others.

### Configuration

:::caution Prerequisites
Before adding topics to your config, the user must **enable Topics mode** in the DM chat with the bot:

1. Open your private chat with the Hermes bot in Telegram
2. Tap the bot's name at the top to open chat info
3. Enable **Topics** (the toggle to turn the chat into a forum)

Without this, Hermes will log `The chat is not a forum` on startup and skip topic creation. This is a Telegram client-side setting — the bot cannot enable it programmatically.
:::

Add topics under `platforms.telegram.extra.dm_topics` in `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      dm_topics:
      - chat_id: 123456789        # Your Telegram user ID
        topics:
        - name: General
          icon_color: 7322096
        - name: Website
          icon_color: 9367192
        - name: Research
          icon_color: 16766590
          skill: arxiv              # Auto-load a skill in this topic
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Topic display name |
| `icon_color` | No | Telegram icon color code (integer) |
| `icon_custom_emoji_id` | No | Custom emoji ID for the topic icon |
| `skill` | No | Skill to auto-load on new sessions in this topic |
| `thread_id` | No | Auto-populated after topic creation — don't set manually |

### How it works

1. On gateway startup, Hermes calls `createForumTopic` for each topic that doesn't have a `thread_id` yet
2. The `thread_id` is saved back to `config.yaml` automatically — subsequent restarts skip the API call
3. Each topic maps to an isolated session key: `agent:main:telegram:dm:{chat_id}:{thread_id}`
4. Messages in each topic have their own conversation history, memory flush, and context window

### Skill binding

Topics with a `skill` field automatically load that skill when a new session starts in the topic. This works exactly like typing `/skill-name` at the start of a conversation — the skill content is injected into the first message, and subsequent messages see it in the conversation history.

For example, a topic with `skill: arxiv` will have the arxiv skill pre-loaded whenever its session resets (due to idle timeout, daily reset, or manual `/reset`).

:::tip
Topics created outside of the config (e.g., by manually calling the Telegram API) are discovered automatically when a `forum_topic_created` service message arrives. You can also add topics to the config while the gateway is running — they'll be picked up on the next cache miss.
:::

## Group Forum Topic Skill Binding

Supergroups with **Topics mode** enabled (also called "forum topics") already get session isolation per topic — each `thread_id` maps to its own conversation. But you may want to **auto-load a skill** when messages arrive in a specific group topic, just like DM topic skill binding works.

### Use case

A team supergroup with forum topics for different workstreams:

- **Engineering** topic → auto-loads the `software-development` skill
- **Research** topic → auto-loads the `arxiv` skill
- **General** topic → no skill, general-purpose assistant

### Configuration

Add topic bindings under `platforms.telegram.extra.group_topics` in `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      group_topics:
      - chat_id: -1001234567890       # Supergroup ID
        topics:
        - name: Engineering
          thread_id: 5
          skill: software-development
        - name: Research
          thread_id: 12
          skill: arxiv
        - name: General
          thread_id: 1
          # No skill — general purpose
```

**Fields:**

| Field | Required | Description |
|-------|----------|-------------|
| `chat_id` | Yes | The supergroup's numeric ID (negative number starting with `-100`) |
| `name` | No | Human-readable label for the topic (informational only) |
| `thread_id` | Yes | Telegram forum topic ID — visible in `t.me/c/<group_id>/<thread_id>` links |
| `skill` | No | Skill to auto-load on new sessions in this topic |

### How it works

1. When a message arrives in a mapped group topic, Hermes looks up the `chat_id` and `thread_id` in `group_topics` config
2. If a matching entry has a `skill` field, that skill is auto-loaded for the session — identical to DM topic skill binding
3. Topics without a `skill` key get session isolation only (existing behavior, unchanged)
4. Unmapped `thread_id` values or `chat_id` values fall through silently — no error, no skill

### Differences from DM Topics

| | DM Topics | Group Topics |
|---|---|---|
| Config key | `extra.dm_topics` | `extra.group_topics` |
| Topic creation | Hermes creates topics via API if `thread_id` is missing | Admin creates topics in Telegram UI |
| `thread_id` | Auto-populated after creation | Must be set manually |
| `icon_color` / `icon_custom_emoji_id` | Supported | Not applicable (admin controls appearance) |
| Skill binding | ✓ | ✓ |
| Session isolation | ✓ | ✓ (already built-in for forum topics) |

:::tip
To find a topic's `thread_id`, open the topic in Telegram Web or Desktop and look at the URL: `https://t.me/c/1234567890/5` — the last number (`5`) is the `thread_id`. The `chat_id` for supergroups is the group ID prefixed with `-100` (e.g., group `1234567890` becomes `-1001234567890`).
:::

## Recent Bot API Features

- **Bot API 9.4 (Feb 2026):** Private Chat Topics — bots can create forum topics in 1-on-1 DM chats via `createForumTopic`. See [Private Chat Topics](#private-chat-topics-bot-api-94) above.
- **Privacy policy:** Telegram now requires bots to have a privacy policy. Set one via BotFather with `/setprivacy_policy`, or Telegram may auto-generate a placeholder. This is particularly important if your bot is public-facing.
- **Message streaming:** Bot API 9.x added support for streaming long responses, which can improve perceived latency for lengthy agent replies.

## Rendering: Tables and Link Previews

Telegram's MarkdownV2 has no native table syntax — pipe tables render as backslash-escaped noise if passed through raw. Hermes normalizes markdown tables automatically:

- **Small tables** are flattened into **row-group bullets** — each row becomes a readable bulleted list under the column headings. Good for 2–4 columns and short cells.
- **Larger or wider tables** fall back to a **fenced code block** with aligned columns so nothing collapses. A one-line prompt hint is added so the agent knows to prefer prose follow-ups over more tables on Telegram.

There's nothing to configure — the adapter picks the right fallback per message. If you want the legacy "always code-block" behavior, disable table normalization by setting `telegram.pretty_tables: false` in `config.yaml` (default: `true`).

**Link previews.** Telegram auto-generates link previews for URLs in bot messages. If you'd rather suppress those (long `/tools` output, agent reply that mentions ten links, etc.):

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

When enabled, Hermes attaches Telegram's `LinkPreviewOptions(is_disabled=True)` to every outgoing message and falls back to the legacy `disable_web_page_preview` parameter on older `python-telegram-bot` versions.

## Group Allowlisting by Chat ID

In addition to per-user access control via `TELEGRAM_ALLOWED_USERS`, you can allowlist entire group chats (and forum topics) by their numeric chat ID. Useful for team/support bots where any group member should be able to chat, but only in certain groups or topics.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - -1001234567890          # supergroup — all members allowed
          - -1001234567891/42       # supergroup + forum thread_id 42 only
```

Equivalent env var: `TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890,-1001234567891/42"` (comma-separated; the `/<thread_id>` suffix is optional).

Behavior:

- A chat that appears in `group_allowed_chats` bypasses `TELEGRAM_ALLOWED_USERS` for its members — anyone in the group can interact with the bot.
- Omit the `/<thread_id>` suffix to allow the whole group; include it to allow just that forum topic.
- DMs still require the user ID to be in `TELEGRAM_ALLOWED_USERS`.
- This layers cleanly on top of `group_topics` (for topic-scoped skill binding) and `ignored_threads` (for silencing specific topics).

## Interactive Model Picker

When you send `/model` with no arguments in a Telegram chat, Hermes shows an interactive inline keyboard for switching models:

1. **Provider selection** — buttons showing each available provider with model counts (e.g., "OpenAI (15)", "✓ Anthropic (12)" for the current provider).
2. **Model selection** — paginated model list with **Prev**/**Next** navigation, a **Back** button to return to providers, and **Cancel**.

The current model and provider are displayed at the top. All navigation happens by editing the same message in-place (no chat clutter).

:::tip
If you know the exact model name, type `/model <name>` directly to skip the picker. You can also type `/model <name> --global` to persist the change across sessions.
:::

## DNS-over-HTTPS Fallback IPs

In some restricted networks, `api.telegram.org` may resolve to an IP that is unreachable. The Telegram adapter includes a **fallback IP** mechanism that transparently retries connections against alternative IPs while preserving the correct TLS hostname and SNI.

### How it works

1. If `TELEGRAM_FALLBACK_IPS` is set, those IPs are used directly.
2. Otherwise, the adapter automatically queries **Google DNS** and **Cloudflare DNS** via DNS-over-HTTPS (DoH) to discover alternative IPs for `api.telegram.org`.
3. IPs returned by DoH that differ from the system DNS result are used as fallbacks.
4. If DoH is also blocked, a hardcoded seed IP (`149.154.167.220`) is used as a last resort.
5. Once a fallback IP succeeds, it becomes "sticky" — subsequent requests use it directly without retrying the primary path first.

### Configuration

```bash
# Explicit fallback IPs (comma-separated)
TELEGRAM_FALLBACK_IPS=149.154.167.220,149.154.167.221
```

Or in `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      fallback_ips:
        - "149.154.167.220"
```

:::tip
You usually don't need to configure this manually. The auto-discovery via DoH handles most restricted-network scenarios. The `TELEGRAM_FALLBACK_IPS` env var is only needed if DoH is also blocked on your network.
:::

## Proxy Support

If your network requires an HTTP proxy to reach the internet (common in corporate environments), the Telegram adapter automatically reads standard proxy environment variables and routes all connections through the proxy.

### Supported variables

The adapter checks these environment variables in order, using the first one that is set:

1. `HTTPS_PROXY`
2. `HTTP_PROXY`
3. `ALL_PROXY`
4. `https_proxy` / `http_proxy` / `all_proxy` (lowercase variants)

### Configuration

Set the proxy in your environment before starting the gateway:

```bash
export HTTPS_PROXY=http://proxy.example.com:8080
hermes gateway
```

Or add it to `~/.hermes/.env`:

```bash
HTTPS_PROXY=http://proxy.example.com:8080
```

The proxy applies to both the primary transport and all fallback IP transports. No additional Hermes configuration is needed — if the environment variable is set, it's used automatically.

:::note
This covers the custom fallback transport layer that Hermes uses for Telegram connections. The standard `httpx` client used elsewhere already respects proxy env vars natively.
:::

## Message Reactions

The bot can add emoji reactions to messages as visual processing feedback:

- 👀 when the bot starts processing your message
- ✅ when the response is delivered successfully
- ❌ if an error occurs during processing

Reactions are **disabled by default**. Enable them in `config.yaml`:

```yaml
telegram:
  reactions: true
```

Or via environment variable:

```bash
TELEGRAM_REACTIONS=true
```

:::note
Unlike Discord (where reactions are additive), Telegram's Bot API replaces all bot reactions in a single call. The transition from 👀 to ✅/❌ happens atomically — you won't see both at once.
:::

:::tip
If the bot doesn't have permission to add reactions in a group, the reaction calls fail silently and message processing continues normally.
:::

## Per-Channel Prompts

Assign ephemeral system prompts to specific Telegram groups or forum topics. The prompt is injected at runtime on every turn — never persisted to transcript history — so changes take effect immediately.

```yaml
telegram:
  channel_prompts:
    "-1001234567890": |
      You are a research assistant. Focus on academic sources,
      citations, and concise synthesis.
    "42":  |
      This topic is for creative writing feedback. Be warm and
      constructive.
```

Keys are chat IDs (groups/supergroups) or forum topic IDs. For forum groups, topic-level prompts override the group-level prompt:

- Message in topic `42` inside group `-1001234567890` → uses topic `42`'s prompt
- Message in topic `99` (no explicit entry) → falls back to group `-1001234567890`'s prompt
- Message in a group with no entry → no channel prompt applied

Numeric YAML keys are automatically normalized to strings.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot not responding at all | Verify `TELEGRAM_BOT_TOKEN` is correct. Check `hermes gateway` logs for errors. |
| Bot responds with "unauthorized" | Your user ID is not in `TELEGRAM_ALLOWED_USERS`. Double-check with @userinfobot. |
| Bot ignores group messages | Privacy mode is likely on. Disable it (Step 3) or make the bot a group admin. **Remember to remove and re-add the bot after changing privacy.** |
| Voice messages not transcribed | Verify STT is available: install `faster-whisper` for local transcription, or set `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY` in `~/.hermes/.env`. |
| Voice replies are files, not bubbles | Install `ffmpeg` (needed for Edge TTS Opus conversion). |
| Bot token revoked/invalid | Generate a new token via `/revoke` then `/newbot` or `/token` in BotFather. Update your `.env` file. |
| Webhook not receiving updates | Verify `TELEGRAM_WEBHOOK_URL` is publicly reachable (test with `curl`). Ensure your platform/reverse proxy routes inbound HTTPS traffic from the URL's port to the local listen port configured by `TELEGRAM_WEBHOOK_PORT` (they do not need to be the same number). Ensure SSL/TLS is active — Telegram only sends to HTTPS URLs. Check firewall rules. |

## Exec Approval

When the agent tries to run a potentially dangerous command, it asks you for approval in the chat:

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

Reply "yes"/"y" to approve or "no"/"n" to deny.

## Security

:::warning
Always set `TELEGRAM_ALLOWED_USERS` to restrict who can interact with your bot. Without it, the gateway denies all users by default as a safety measure.
:::

Never share your bot token publicly. If compromised, revoke it immediately via BotFather's `/revoke` command.

For more details, see the [Security documentation](/user-guide/security). You can also use [DM pairing](/user-guide/messaging#dm-pairing-alternative-to-allowlists) for a more dynamic approach to user authorization.

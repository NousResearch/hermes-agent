---
sidebar_position: 1
title: "Telegram"
description: "Set up Hermes Agent as a Telegram bot"
---

# Telegram Setup

Hermes Agent integrates with Telegram as a full-featured conversational bot. Once connected, you can chat with your agent from any device, send voice memos that get auto-transcribed, receive scheduled task results, and use the agent in group chats. The integration is built on [python-telegram-bot](https://python-telegram-bot.org/) and supports text, voice, images, and file attachments.

## Quick setup (dashboard and desktop app)

The **Messaging → Telegram** page in the [dashboard](../features/web-dashboard.md) and the [desktop app](../desktop.md) has a **Create with QR** button. Scan the code (or open the link) in Telegram; Hermes creates the bot for you, detects your Telegram user ID, writes `TELEGRAM_BOT_TOKEN` and `TELEGRAM_ALLOWED_USERS` into your profile's `.env`, and restarts the gateway. If you prefer to create the bot yourself, follow the manual steps below.

## Hermes Cloud with Team Gateway

On a Hermes Cloud instance with Team Gateway on, Telegram (and Discord and Slack) reach your agent through a shared Hermes bot. A bot of your own does not run there: the gateway switches direct connections off at every start so the instance can sleep when idle. The Telegram card shows **Via Team Gateway** instead of **Create with QR**.

When Nous Portal manages who can reach the agent (agents created in the Portal), link your account from the Portal: the card lists the linked accounts and its **Link accounts in Nous Portal** button opens the agent there. Use **Message platforms** on the agent to get the link code.

Otherwise the card links the account itself:

1. Click **Link Telegram**. Hermes shows a one-time message such as `/link AB12CD`, a QR code and an **Open Telegram** button.
2. Scan the QR code with your phone (or click the button) to open the shared bot, and send it that message from the Telegram account you want to use.
3. The card lists the account once the bot sees the message. From then on, message the shared bot to talk to your agent.

The gateway must be running for the button to work, because it holds the Team Gateway connection. To run your own bot beside the shared one instead, set `GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true` in the instance's environment. That keeps a live Telegram connection open, so the instance never sleeps.

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

### Online/Offline status indicator (Optional)

Telegram bots have no real online/offline presence dot — that green dot is a
*user-account* feature, not something the Bot API exposes for bots. The closest
surface is the bot's **short description** (the line shown under its name in the
bot's profile).

Enable `status_indicator` and Hermes sets that short description to **Online**
when the gateway connects and **Offline** on a clean shutdown:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        status_indicator: true
        # Optional custom strings (defaults: "Online" / "Offline"):
        status_online: "🟢 Online"
        status_offline: "🔴 Offline"
```

Notes:

- The short description is **global** to the bot (visible to all users), not
  per-chat. Users see it on the bot's profile page, not as a live badge inside
  an open chat.
- Only a **clean** gateway shutdown (`/stop`, `disconnect`) writes "Offline".
  A hard crash leaves the last-known status — the inherent limitation of a
  profile-text indicator.
- Off by default, since it mutates the bot's global profile.
- Hermes saves the bot's own short description (the one set in @BotFather) before
  it first writes **Online**, and keeps that copy across restarts and crashes.
  Turn `status_indicator` off and the next connect puts your text back, unless
  you have set a new description in @BotFather since (yours is kept). Edit the
  description while the gateway is down and the new text is what Hermes saves.
  If Hermes cannot read and save your text at connect, it writes no status that run.

### Cold-boot pending queue (Optional)

Messages sent while the gateway is off (a desktop shut down overnight, a
restart, an update) wait in Telegram's Bot API queue for up to 24 hours. By
default the adapter keeps that queue on a cold boot and answers the backlog
in order once it starts; updates Hermes already handled before the restart
are skipped by its persisted update receipts, so they are not answered twice.

Set `drop_pending_on_cold_boot: true` to discard that backlog on startup
instead (an always-on server where a restart means "start clean"):

```yaml
platforms:
  telegram:
    extra:
      drop_pending_on_cold_boot: true
```

Notes:

- Default is `false`: the backlog is answered. An explicit value in your
  config is respected either way.
- Watcher reconnects (brief network outages with the process still alive)
  always preserve the queue regardless of this setting.
- Conflict recovery (another poller holding the bot's `getUpdates` session)
  also preserves the queue: messages sent during a 409 conflict are delivered
  once polling resumes.
- After a crash, a preserved queue can redeliver an update the crashed
  instance partially processed. Telegram's offset usually prevents this,
  but time-sensitive commands sent during a long outage will run on boot.

### Concurrent update handling

Updates from different chats are processed concurrently, so one slow turn
(a long provider retry, a large download) no longer stalls every other chat.
Updates from the same chat still run one after another, in arrival order.
The cross-chat pool defaults to 32; an invalid value logs a warning and uses
the default:

```yaml
platforms:
  telegram:
    extra:
      max_concurrent_updates: 32   # 1 restores fully sequential processing
```

### Albums (Optional)

Telegram delivers an album (several photos or files sent together) as one update per item.
Hermes waits until every item that has arrived is downloaded, then sends the whole album to the
agent as one message. It waits at most 8 seconds for a slow download before sending what it has.
After the last item, it waits a short debounce (default 1.5 s, at most 10 s) for one sent later.
Raise the debounce if your phone uploads album items far apart:

```yaml
platforms:
  telegram:
    extra:
      media_group_wait_seconds: 3.0
```

A short text followed right away by a photo (a question, then the screenshot it is about) also
reaches the agent as one message. If the photo starts arriving before the text has been sent on,
the text waits up to 8 seconds for the photo's download and joins it.

### Repeated inbound updates

Hermes suppresses repeated Telegram `update_id` values before message batching,
command/media handling, observed group-history writes and plugin observers.
The receiving adapter and numeric bot ID scope this check; it does not deduplicate
by text or `message_id`. A genuine edit with a new update ID can still be processed.

This is bounded protection, not an exactly-once guarantee:

- The adapter remembers the most recent 4096 completed admissions per bot. Active
  updates stay claimed until dispatch and its scheduled PTB handler tasks finish,
  including nonblocking native plugins and registered error callbacks.
- Completed update IDs are also saved per bot under the Hermes home and kept for
  24 hours, the longest Telegram holds an undelivered update. Reconnects, adapter
  rebuilds and gateway restarts therefore do not answer an already-handled update
  again, including the backlog answered after a cold boot. A receipt older than
  24 hours, or a receipts file that cannot be read, no longer protects that update.
- Failed or cancelled preparation releases its claim if nothing has been handed
  off. Once an update enters a batch/hold queue, gateway dispatch, an observer or
  a native plugin, a later error does not reopen it. Native plugins own their own
  partial effects, so entering their update or registered error callback is
  conservatively treated as handoff. PTB's own exception logging is not a handoff.
  Uncached static-sticker vision analysis is also a handoff: cancelling the await
  cannot undo an auxiliary model request already submitted. Caught preparation
  errors before any handoff remain retryable; an intentional refusal is terminal.
- Releasing a claim only permits a later delivery; it does not request one from
  Telegram. Polling acknowledgement is independent of agent completion. This check
  does not retry failed replies or prevent a downstream component from independently
  duplicating work.

For a suspected late replay, compare both occurrences' bot/profile, chat/topic,
`update_id`, update kind, `message_id` and actual receive time. An edit can reuse
`message_id`, and the message's sent timestamp is not its receive time.

### Command menu priority and cap (Optional)

Hermes registers its command menu automatically when the Telegram gateway starts. The menu is built from the central slash-command registry plus eligible plugin/skill commands, then capped so Telegram accepts the payload reliably. The default cap is 60 commands — enough to keep all built-in commands plus common skill commands visible.

If you have skill, plugin, or built-in commands that should stay visible in Telegram's `/` picker, prioritize them in `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      command_menu:
        max_commands: 60
        priority_mode: prepend  # prepend | append | replace
        priority:
          - my_plugin_command
          - excalidraw       # skill commands work here too
```

`priority_mode` controls how your list combines with Hermes' built-in priority list:

- `prepend`: put your commands first, then Hermes defaults
- `append`: keep Hermes defaults first, then your commands
- `replace`: use only your list for priority ordering

Priority is applied to the **combined** candidate list (core commands, plugin commands, and skill commands) before the cap is enforced — so a prioritized skill command is guaranteed a menu slot even when core commands alone would fill the menu. Previously skills were always trimmed first and alphabetically, so late-alphabet skills could never appear regardless of `priority`.

Telegram allows up to 100 BotCommands, but large command payloads can fail. Hermes defaults to 60 for reliability and clamps configured values to `1..100`; use `/commands` for the full command list.

### Inline command picker: search every command (no cap)

The `/` menu is capped, but Telegram's **inline mode** is not. Once enabled, type `@yourbotname` followed by a search term in any chat to get a live, searchable picker over **every** Hermes command and installed skill — results are computed per keystroke and paginated, so nothing is ever trimmed:

```
@yourbotname plan            → tap the /plan result to send it
@yourbotname plan migrate auth to OIDC   → sends /plan migrate auth to OIDC
@yourbotname pdf             → finds skills matching "pdf" by name or description
```

The first word filters the catalog; everything after it is carried into the sent command as its argument. Tapping a result sends the command as a normal message from you, so it dispatches through the standard command path (command-prefixed messages reach the bot even with privacy mode on).

**One-time setup:** inline mode is off by default for every Telegram bot. Enable it in [@BotFather](https://t.me/BotFather) with `/setinline` (pick your bot, set any placeholder text, e.g. `Search commands and skills...`). Until then, Telegram never delivers inline queries and the picker stays inert.

Results are only served to users who pass your gateway allowlist — unauthorized users get an empty list, so your installed skill catalog is not exposed to strangers (inline queries can be sent from any chat, even ones the bot is not in).

### Forum topics

In Telegram supergroups with topics enabled, the command menu does **not** inherit from the global `AllGroupChats` scope — each forum supergroup needs its own chat-scoped command registration. Hermes handles this automatically: the first time a message arrives in a forum supergroup, the gateway lazily registers the same capped menu scoped to that chat (`BotCommandScopeChat`). This happens once per forum chat per gateway run and is transparent to users — the `/` menu appears in forum topics just like in regular groups, without any configuration.

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

### Observe group chatter without auto-replying

For OpenClaw/Yuanbao-style group behavior, configure Telegram so the bot can **see** ordinary group messages but only **responds** when directly triggered:

```yaml
telegram:
  allowed_chats:
    - "-1001234567890"
  group_allowed_chats:
    - "-1001234567890"
  require_mention: true
  observe_unmentioned_group_messages: true
```

With this mode enabled, unmentioned group messages from explicitly allowlisted chats/topics are appended to the shared chat/topic session transcript as observed context, but they do not dispatch the agent. `allowed_chats` gates where the bot responds; `group_allowed_chats` authorizes the shared group session used for observed context, so use the same chat IDs for this mode. A later `@botname` mention, reply to the bot, or configured mention pattern in that same allowlisted chat/topic can use that observed context. The triggered message is also tagged with `[nickname|user_id]` and gets a per-turn safety prompt so the model treats prior observed lines as context, not instructions addressed to the bot. Slash commands sent in that chat/topic (`/new`, `/model`, `/undo`, skill commands) act on the same shared session, while admin-only commands are still checked against the sender.

Observed context is text only. A photo, video or file sent without a trigger is cached, and the observed line gives its saved path; the next addressed turn does not attach it inline. When a later message asks about it, the agent opens the cached file itself (`vision_analyze` for images; `video_analyze` for videos when the opt-in `video` toolset is enabled, otherwise its file tools), so image tokens are only spent when someone asks.

Equivalent environment variable:

```bash
TELEGRAM_ALLOWED_CHATS=-1001234567890
TELEGRAM_GROUP_ALLOWED_CHATS=-1001234567890
TELEGRAM_OBSERVE_UNMENTIONED_GROUP_MESSAGES=true
```

This requires Telegram to deliver ordinary group messages to the gateway, so disable BotFather privacy mode or promote the bot to group admin as described above.

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

Anything on this list is delivered as a native attachment on platforms that support it (Telegram, Discord, Signal, Slack, WhatsApp, Feishu, Matrix, etc.); on platforms without native support it falls back to a link or plain-text indicator. The **bold** categories were added in the last few releases — if you were relying on the model saying `here is the file: /path/to/report.docx` instead, swap to `MEDIA:/path/to/report.docx` for native delivery.

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
TELEGRAM_WEBHOOK_SECRET="$(openssl rand -hex 32)"  # required
# TELEGRAM_WEBHOOK_PORT=8443        # optional, default 8443
# TELEGRAM_WEBHOOK_HOST=::          # optional, bind host (e.g. IPv6-only)
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_WEBHOOK_URL` | Yes | Public HTTPS URL where Telegram will send updates. The URL path is auto-extracted (e.g., `/telegram` from the example above). |
| `TELEGRAM_WEBHOOK_SECRET` | **Yes** (when `TELEGRAM_WEBHOOK_URL` is set) | Secret token that Telegram echoes in every webhook request for verification. The gateway refuses to start without it — see [GHSA-3vpc-7q5r-276h](https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h). Generate with `openssl rand -hex 32`. |
| `TELEGRAM_WEBHOOK_PORT` | No | Local port the webhook server listens on (default: `8443`). |
| `TELEGRAM_WEBHOOK_HOST` | No | Local bind host for the webhook server. Default: unset → dual-stack bind on all interfaces (IPv4+IPv6). Set this on IPv6-only deployments (e.g. Fly.io 6PN) or to pin the listener to a specific interface. Can also be set via `platforms.telegram.extra.webhook_host` in `config.yaml`. |

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

If the fallback IP discovery path is unhealthy on your host, set `HERMES_TELEGRAM_DISABLE_FALLBACK_IPS=true` to keep cold connect on the plain `api.telegram.org` path. You can also bound DNS-over-HTTPS fallback discovery with `HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT` in seconds; the default is `5`.

## Home Channel

Use the `/sethome` command in any Telegram chat (DM or group) to designate it as the **home channel**. Scheduled tasks (cron jobs) deliver their results to this channel.

You can also set it manually in `~/.hermes/.env`:

```bash
TELEGRAM_HOME_CHANNEL=-1001234567890
TELEGRAM_HOME_CHANNEL_NAME="My Notes"
```

:::tip
Group chat IDs are negative numbers (e.g., `-1001234567890`). Your personal DM chat ID is the same as your user ID.
An `@username` works only for a public channel or supergroup: Telegram answers "Chat not found" for a person's handle, so use the numeric ID for a DM home.
:::

### Cron deliveries in topic mode

If you have topic mode enabled in your bot DM, cron messages delivered to the root chat land in the system-only lobby — replying there opens no session and you see the "main chat is reserved for system commands" notice. Create a dedicated forum topic (e.g. `Cron`) and set:

```bash
TELEGRAM_CRON_THREAD_ID=<topic_thread_id>
```

`TELEGRAM_CRON_THREAD_ID` overrides `TELEGRAM_HOME_CHANNEL_THREAD_ID` for cron deliveries only. Replies in that topic continue the topic's existing session.

## Shared Contacts and Locations

A contact card or location pin you share with the bot reaches the agent as a short text block (name, phone number and vCard for a contact; coordinates and venue for a location), so the agent can act on it.

A **live location** reaches the agent once, when you start sharing it. Telegram then refreshes the position every few seconds to a minute, for as long as the share runs. A refresh becomes a new message to the agent only after you have moved about 75 m from the last position it was told about, or after 15 minutes at the same spot. GPS jitter while you stand still never starts a turn. A static pin (or a venue) always reaches the agent, even when you send the same spot twice.

## Stickers

A static sticker reaches the agent as a short text note: its emoji, its pack name, and a one-to-two-sentence description written by your vision model (`auxiliary.vision`, which defaults to the main model). The description is cached per sticker (by Telegram's `file_unique_id`) in `sticker_cache.json` under the profile's Hermes home, so sending the same sticker again costs no vision call. If the vision call fails, the note falls back to the emoji alone. Animated and video stickers are not analyzed: the agent gets a note with their emoji only.

## Voice Messages

### Incoming Voice (Speech-to-Text)

Voice messages you send on Telegram are automatically transcribed by Hermes's configured STT provider and injected as text into the conversation.

- `local` uses `faster-whisper` on the machine running Hermes — no API key required
- `groq` uses Groq Whisper and requires `GROQ_API_KEY`
- `openai` uses OpenAI Whisper and requires `VOICE_TOOLS_OPENAI_KEY`

#### Skipping STT: pass the raw audio file to the agent

If you'd rather have the **agent itself** handle audio — for diarization, a custom transcription tool, or just archiving the recording — set `stt.enabled: false` in `~/.hermes/config.yaml`:

```yaml
stt:
  enabled: false
```

With STT disabled, the gateway still downloads the voice/audio attachment into Hermes's audio cache, but **does not transcribe it**. The agent receives the message with a marker like:

```
[The user sent a voice message: /home/<user>/.hermes/cache/audio/<hash>.ogg]
```

Your tools or skills can then read that path directly (e.g., hand it off to a local diarization pipeline, a richer transcription model, or upload it to long-term storage). The file extension reflects the original format Telegram delivered (`.ogg` for voice notes, `.mp3`/`.m4a`/etc. for audio attachments).

This pairs naturally with the [local Bot API server](#large-files-20mb-via-local-bot-api-server) section below, which lifts Telegram's 20MB getFile ceiling to 2GB — useful when the recordings you want to process are longer than a couple of minutes.

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

MP3 and M4A files go out as Telegram audio tracks. Their captions get the same markdown formatting as voice notes. With ffmpeg installed, the file's own artist and title tags and its embedded cover art also fill in the performer, track name and album art in Telegram's music player.

Configure the TTS provider in your `config.yaml` under the `tts.provider` key.

## Large Files (>20MB) via Local Bot API Server

Telegram's **public** Bot API caps `getFile` downloads at **20 MB**, so any voice note, audio file, video, or document larger than that is silently rejected by Hermes with a "too large" reply. The documented way around this is to run a **local** [telegram-bot-api](https://github.com/tdlib/telegram-bot-api) daemon — the same server software Telegram uses, but running on your network. A local server raises the file ceiling to **2 GB** and Hermes auto-lifts its own internal cap when it sees a custom `base_url` configured.

This unlocks workflows like:

- Sending long voice memos (45-minute meetings, podcasts) to the bot
- Uploading large videos for vision-tool processing
- Archiving raw audio for offline pipelines like diarization, alignment, or training data

### Step 1: Obtain Telegram API credentials

The local server talks directly to Telegram's MTProto layer (not the public Bot API), so it needs **MTProto credentials**:

1. Visit [my.telegram.org/apps](https://my.telegram.org/apps) and sign in with your Telegram account.
2. Create a new application (any name and short description will do).
3. Copy the `api_id` and `api_hash` — both are required.

### Step 2: Run the telegram-bot-api server

The community-maintained [`aiogram/telegram-bot-api`](https://hub.docker.com/r/aiogram/telegram-bot-api) Docker image is the easiest path. A minimal `docker-compose.yaml` (use `--local` mode to enable the higher limits):

```yaml
services:
  tg-bot-api:
    image: aiogram/telegram-bot-api:latest
    container_name: tg-bot-api
    restart: unless-stopped
    ports:
      - "127.0.0.1:8081:8081"   # bind to loopback only; see security note
    environment:
      TELEGRAM_API_ID: "12345"           # your api_id from Step 1
      TELEGRAM_API_HASH: "abcdef..."     # your api_hash from Step 1
      TELEGRAM_LOCAL: "1"                # enable --local mode (raises 20MB → 2GB)
    volumes:
      - ./tg-bot-api-data:/var/lib/telegram-bot-api
```

Bring it up:

```bash
docker compose up -d tg-bot-api
docker logs --tail 20 tg-bot-api
```

:::warning Security
The local Bot API server takes your bot token in the URL path (e.g. `/bot<TOKEN>/getMe`) with **no additional auth**. Anyone who can reach the port can fully control your bot — read every message it can see, send messages as it, etc. Bind the container to `127.0.0.1` and/or front it with a reverse proxy on a private network. **Never expose port 8081 to the public internet.**
:::

### Step 3: Log the bot out of the public API (one-time)

A bot can only be active on **one** Bot API server at a time. If your bot was already running against `api.telegram.org` (which it almost certainly was), you must explicitly log it out there before the local server will accept it:

```bash
curl "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/logOut"
# expected response: {"ok":true,"result":true}
```

This is a one-shot migration step — you don't repeat it on every restart. Telegram delivers any messages received after `logOut` through the new server instead.

Verify the local server can talk to Telegram on the bot's behalf:

```bash
curl "http://127.0.0.1:8081/bot<YOUR_BOT_TOKEN>/getMe"
# expected response: {"ok":true,"result":{"id":...,"is_bot":true,...}}
```

### Step 4: Point Hermes at the local server

Add the URLs under `platforms.telegram.extra` in `~/.hermes/config.yaml`:

```yaml
platforms:
  telegram:
    extra:
      base_url: "http://127.0.0.1:8081/bot"
      base_file_url: "http://127.0.0.1:8081/file/bot"
      local_mode: true        # see Step 5 below — only set this if the bot's data
                              # directory is readable by the Hermes process
```

:::caution Use `platforms.telegram.extra`, not `telegram.extra`
At the moment only the `platforms.<name>.extra` form is deep-merged into the platform config. Keys placed directly under a top-level `telegram.extra` block are silently dropped.
:::

When `base_url` is set, Hermes:

- Builds the python-telegram-bot client against the local server
- Auto-lifts its internal document/audio size cap from 20 MB → 2 GB
- Reports the active limit in the "too large" error message (`Maximum: 2048 MB.`) so it's obvious which mode you're in
- Sends through the same server outside the gateway too: `hermes send`, the `send_message` tool and cron deliveries that fall back to the standalone sender

Restart the gateway and look for a confirmation log line:

```bash
hermes gateway restart
grep -E "Using custom Telegram base_url|Using Telegram local_mode" ~/.hermes/logs/gateway.log | tail
```

### Step 5: `local_mode` — file access on disk

The local server has **two ways** to deliver files:

1. **Without `--local`** (the default): files are served over HTTP at `/file/bot<TOKEN>/<path>`, same as the public Bot API. The 20MB ceiling stays in effect. Useful as a network-fix only (e.g. when `api.telegram.org` is unreachable but you can self-host); not what you want for the size lift.
2. **With `--local`** (set via `TELEGRAM_LOCAL=1` above): files are written to the server's filesystem and the `getFile` response returns an **absolute path** instead of an HTTP URL. The 20MB ceiling is lifted. Hermes must then read the bytes **from disk**, not over HTTP.

To make the disk-read path work, set `local_mode: true` in the config above **and** make sure the Hermes process can read the path the server returns. Two scenarios:

- **Same machine** — telegram-bot-api and Hermes run on the same host. Bind-mount the data volume to a directory that Hermes can read (e.g., `/var/lib/telegram-bot-api`), and make sure the file ownership matches. The container drops privileges to its internal `telegram-bot-api` user (uid varies by image); the simplest fix is to add `user: "<UID>:<GID>"` to the compose service so files are owned by a uid Hermes already runs as.
- **Different machines** — the bot server runs on one host (e.g., a NAS, a separate VM) and Hermes on another. The server's data directory must be shared with the Hermes machine at the **same absolute path** the server reports (typically `/var/lib/telegram-bot-api`). NFS works well for this; CIFS/SMB with `uid=` mount remapping is friendlier if you don't want to deal with uid mismatches at the filesystem level.

If `local_mode: true` is set but Hermes can't `stat` the returned file path (permissions or wrong mount), python-telegram-bot silently falls back to an HTTP `getFile` against the local server — which in `--local` mode responds with `404 Not Found`. The symptom shows up in `gateway.log` as:

```
[Telegram] Failed to cache voice: Not Found
telegram.error.InvalidToken: Not Found
```

If you see that, the cap-lift is working but the file-share isn't. Verify `ls -la /var/lib/telegram-bot-api/<TOKEN>/voice/` from the Hermes host as the user the gateway runs as, and confirm a single file is `cat`-able without a permission error.

**Storage mounted at a different path.** When Hermes runs in another container or on another host and the server's data directory is mounted somewhere else (for example the compose file's `./tg-bot-api-data`), tell Hermes where it is:

```yaml
platforms:
  telegram:
    extra:
      local_mode: true
      local_bot_api_root: "/home/me/tg-bot-api-data"   # the host path of the server's /var/lib/telegram-bot-api
```

The server reports its own path (`/var/lib/telegram-bot-api/<TOKEN>/videos/file_7.MOV`, sometimes behind `base_file_url`). Hermes takes the part after the token directory and reads `<local_bot_api_root>/<TOKEN>/videos/file_7.MOV`, or `<local_bot_api_root>/videos/file_7.MOV` when you mounted the token directory itself. That is two file checks with no directory scan, and a path containing `..` is never followed. If neither file exists, the normal download runs as before.

### Step 6: Test it

Send the bot a voice note or audio file that's bigger than 20 MB. Tail the gateway log:

```bash
tail -f ~/.hermes/logs/gateway.log | grep -iE "telegram|cache"
```

You should see a `[Telegram] Cached user voice at /home/<user>/.hermes/cache/audio/...` line and **no** "too large" rejection. Combined with `stt.enabled: false` (above), the path to the original audio file then lands in the agent's inbound message for downstream processing.

## Group Chat Usage

Hermes Agent works in Telegram group chats with a few considerations:

- **Privacy mode** determines what messages the bot can see (see [Step 3](#step-3-privacy-mode-critical-for-groups))
- `TELEGRAM_ALLOWED_USERS` still applies — only authorized users can trigger the bot, even in groups
- You can keep the bot from responding to ordinary group chatter with `telegram.require_mention: true`
- With `telegram.require_mention: true`, group messages are accepted when they are:
  - replies to one of the bot's messages
  - `@botusername` mentions
  - `/command@botusername` (Telegram's bot-menu command form that includes the bot name)
  - matches for one of your configured regex wake words in `telegram.mention_patterns`
- In groups with multiple Hermes bots, `telegram.exclusive_bot_mentions` keeps routing deterministic. When a message is addressed to one or more Telegram bot usernames, only the mentioned bot profiles process it; other Hermes bots ignore it before reply and wake-word fallbacks run (and log one `Skipping message ... addressed to other bot(s)` line). A message is addressed to another bot only when it starts with that bot's handle (`@OtherBot do this`, `@BotA @BotB: compare`, `/status@OtherBot`). A handle later in the text, as in `the shop has its own bot now (@ShopBot)`, only talks about that bot and does not silence this one. This bot's own handle anywhere still wins unless an opening command, optionally after this bot's address, explicitly targets another bot. Enabled by default.
- Renaming the bot's `@username` in BotFather is picked up automatically — Hermes follows the new handle for mention routing without a gateway restart. Collectible (Fragment) usernames that don't end in `bot` are supported too.
- Use `telegram.ignored_threads` to keep Hermes silent in specific Telegram forum topics, even when the group would otherwise allow free responses or mention-triggered replies
- Use `telegram.allowed_topics` as a forum-topic whitelist — when set, Hermes ignores messages from any other group/forum topic and logs the first drop per chat/topic at INFO (DMs are never filtered). Telegram's General topic, and any group message without a `message_thread_id`, is normalized to topic ID `1`. Whitelist evaluation runs before `ignored_threads`.
- Use `telegram.free_response_chats` / `telegram.free_response_topics` to bypass `require_mention` for a whole chat or a single forum topic only.
- If `telegram.require_mention` is left unset or false, Hermes keeps the previous open-group behavior and responds to normal group messages it can see

### Letting the bot stay silent in a group

A bot that reads every group message can answer exactly `[SILENT]` (or another [silence token](index.md#intentional-silence-tokens)) when a message doesn't need it. Whether that silence stands depends on who the message addresses:

| Group message | Bare silence token |
|---|---|
| Mentions the bot, replies to it, matches `mention_patterns`, or is a `/command` | Replaced by a short notice, so a question to the bot never vanishes |
| Replies to another member, or opens by @mentioning someone else | Stays silent |
| Any other message without a `?` | Stays silent |
| Any other message with a `?` (it may be a question for the bot) | Replaced by the notice |

A message still gets the notice while the bot's own identity is unknown, or when it replies to a message whose author is unknown or that was posted as a chat (a linked channel's post, an anonymous admin). A voice note, video note or audio file without a caption also gets the notice, because its words are not in the message text. A forum topic's root message is not a reply to another member.

### Multiple Hermes bots in one group

If you run several Hermes profiles in the same Telegram group, create one Telegram bot token per profile and start one gateway per profile. Do not reuse the same bot token in multiple running gateways; Telegram will reject concurrent polling for the same token.

Recommended group config:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
  mention_patterns: []
```

With this setup, a group message like `@research_bot @ops_bot summarize this` is processed by `research_bot` and `ops_bot` only. Other Hermes bots in the group stay silent, even if the message is a reply to one of their earlier messages or would otherwise match a shared wake word.

Two Hermes bots that answer each other's quote-replies can still loop forever with `TELEGRAM_ALLOW_BOTS=all`, because a reply to the bot always passes the `require_mention` gate. Setting `telegram.bots_require_mention: true` (env `TELEGRAM_BOTS_REQUIRE_MENTION`) closes that path: a message from another bot only triggers a response when it explicitly `@mentions` this bot, while human replies keep working unchanged. This also holds in `free_response_chats` / `free_response_topics`: those let *humans* trigger Hermes without a mention, but another bot must still `@mention` it explicitly.

A bot-to-bot loop guard also meters every chat where bot-authored messages are admitted (`TELEGRAM_ALLOW_BOTS` set to `mentions` or `all`). Once 20 bot messages land in one chat inside 5 minutes, further bot messages in that chat are dropped for 10 minutes and one warning is logged; human messages are never counted or dropped. Settings live in `config.yaml`:

```yaml
gateway:
  bot_loop_guard:
    enabled: true        # false turns the guard off
    max_events: 20       # bot messages per chat per window
    window_seconds: 300
    cooldown_seconds: 600
```

A legitimate high-volume bot posting more than 20 messages into one chat in 5 minutes trips the guard too; raise `max_events` for that gateway.

The same block also ends a bot-to-bot exchange long before that budget. Your message starts an exchange. After it, each bot message is one hop, and past `max_hops` (default 4) further bot messages start no turn. A bot repeating the same text in one exchange counts once. With `bot_replies_expect_reply: false` (the default), Hermes does not answer a bot that is answering Hermes's own reply. Hermes also never sends its runtime notices to a bot: busy and interrupt acknowledgements, progress and status lines, the "not processed" and retry hints, and model-fallback notices. A bot gets only the model's reply.

```yaml
gateway:
  bot_loop_guard:
    max_hops: 4
    bot_replies_expect_reply: false
```

Group conversation text and media captions keep every mention when the message names other participants too (`@research_bot , @ops_bot are you both listening?` reaches `research_bot` verbatim); when this bot is the only one addressed, its own handle is still stripped so short answers such as `@hermes_bot 2` keep working. Group turns also carry the bot's own Telegram username in the per-channel context so the model can tell which retained mentions are for it. Slash commands still use the normal command-trigger cleanup.

Set `exclusive_bot_mentions: false` only for legacy groups where explicit mentions should not override reply and wake-word triggers.

To operate several profiles, run the gateway command once per profile. For example:

```bash
# default profile
hermes gateway start
hermes gateway status
hermes gateway stop

# named profiles
hermes -p research gateway start
hermes -p research gateway status
hermes -p research gateway stop
```

For a small fixed fleet, use a shell loop or script that calls `hermes gateway <action>` for the default profile and `hermes -p <profile> gateway <action>` for each named profile. This is more reliable than assuming a single process-level command controls every named profile on every service manager.

### Troubleshooting: works in DMs but not groups

If the bot responds in a private chat but stays silent in a group, check these
gates in order:

1. **Telegram delivery:** turn off BotFather privacy mode, promote the bot to
   admin, or mention the bot directly. Hermes cannot respond to group messages
   that Telegram never delivers to the bot.
2. **Rejoin after changing privacy:** remove the bot from the group and add it
   again after changing BotFather privacy settings. Telegram may keep the old
   delivery behavior for existing memberships.
3. **Hermes authorization:** make sure the sender is listed in
   `TELEGRAM_ALLOWED_USERS` or `TELEGRAM_GROUP_ALLOWED_USERS`, or allow the
   group chat with `TELEGRAM_GROUP_ALLOWED_CHATS`.
4. **Mention filters:** if `telegram.require_mention: true` is set, normal
   group chatter is ignored unless the message is a slash command, reply to the
   bot, `@botusername` mention, or configured `mention_patterns` match.
5. **Multi-bot routing:** if a group contains several bots, make sure each
   Hermes profile uses a unique bot token and keep `exclusive_bot_mentions`
   enabled unless you intentionally want legacy shared-trigger behavior.

Negative chat IDs are normal for Telegram groups and supergroups. If you use
chat-scoped authorization, put those IDs in `TELEGRAM_GROUP_ALLOWED_CHATS`, not
the sender-user allowlist.

### Example group trigger configuration

Add this to `~/.hermes/config.yaml`:

```yaml
telegram:
  require_mention: true
  exclusive_bot_mentions: true
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
- A captionless voice note has no text, so it can only wake the bot by reply or @mention. Set
  `voice_mention_patterns: true` to transcribe such a group voice note first and match these patterns
  against what was said. It is off by default because every unaddressed voice note in the group then
  goes to your speech-to-text provider. The turn reuses that transcript (one STT call)
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
Before adding topics to your config, the bot owner must **enable Threaded Mode** for the bot in **@BotFather**:

1. Open the BotFather **Mini App** (search `botfather` in Telegram, then tap **Open** on the search result — the classic `/mybots` text menu does not expose this setting)
2. Go to **My bots → your bot → Bot Settings → Threads Settings**
3. Turn on **Threaded Mode**

There is no "Topics" toggle in the DM chat itself — a bot DM is not a group, so the group-forum toggle described in some older guides does not apply here. Without Threaded Mode, Hermes will log `The chat is not a forum` on startup and skip topic creation. See [Prerequisites](#prerequisites) below for the same steps with more detail.
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
2. The `thread_id` is saved back to `config.yaml` automatically. Restarts, reconnects and adapter rebuilds after a network failure reuse it instead of creating the topic again
3. Each topic maps to an isolated session key: `agent:main:telegram:dm:{chat_id}:{thread_id}`
4. Messages in each topic have their own conversation history, memory flush, and context window
5. Every topic with a `thread_id` (in `dm_topics` and `group_topics`) is listed in `channel_directory.json`, so `send_message` can target it by name even when it has no recent session. A topic name used in more than one chat is listed as `<chat_id> / <name>`. A topic named like another directory entry (a DM or group of that name) is listed as `<chat_id>:<thread_id> / <name>`, and that entry as `<its id> / <name>`, so the bare name resolves to neither

:::caution Read-state limitation
Telegram private DM topics have a Bot API read-state limitation. Hermes can receive messages from a user-created topic and route replies back into that topic, but normal bot accounts don't expose a reliable API for marking the topic copy of an incoming user message as read. Telegram may show the root "All Messages" chat and the topic with different unread/read indicators. This is expected Telegram behaviour, not necessarily a Hermes processing failure.
:::

### Root DM handling

By default, messages sent to the root DM (outside any topic) are processed
normally. Set `ignore_root_dm: true` to turn the root DM into a lobby — normal
messages are silently ignored for users who have DM topics configured, while
system commands (`/start`, `/help`, `/status`, etc.) still work.

```yaml
platforms:
  telegram:
    extra:
      ignore_root_dm: true
      dm_topics:
        - chat_id: 123456789
          topics:
            - name: General
```

The check is **per-chat**: only users with at least one entry in `dm_topics`
will have their root DM affected. Users without configured topics are
unaffected.

### Skill binding

Topics with a `skill` field automatically load that skill when a new session starts in the topic. This works exactly like typing `/skill-name` at the start of a conversation — the skill content is injected into the first message, and subsequent messages see it in the conversation history.

For example, a topic with `skill: arxiv` will have the arxiv skill pre-loaded whenever its session resets (after an explicit `/new` or `/reset`).

:::tip
Topics created outside of the config (e.g., by manually calling the Telegram API) are discovered automatically when a `forum_topic_created` service message arrives. You can also add topics to the config while the gateway is running — they'll be picked up on the next cache miss.
:::

## Multi-session DM mode (`/topic`)

A ChatGPT-style multi-session DM — one bot, many parallel conversations. Unlike the operator-curated `extra.dm_topics` above, this mode is **user-driven**: no config, no pre-declared topic names. The end user flips it on with `/topic`, then taps the Telegram **+** button to create as many topics as they want, each one a fully independent Hermes session.

### `/topic` subcommands

| Form | Context | Effect |
|------|---------|--------|
| `/topic` | Root DM, not yet enabled | Check BotFather capabilities, enable multi-session mode |
| `/topic` | Root DM, already enabled | Show status: unlinked sessions available for restore |
| `/topic` | Inside a topic | Show the current topic's session binding |
| `/topic help` | Any | Inline usage |
| `/topic off` | Root DM | Disable multi-session mode and clear all topic bindings for this chat |
| `/topic <session-id>` | Inside a topic | Restore a previous Telegram session into the current topic |

Only authorized users (allowlist via `TELEGRAM_ALLOWED_USERS` / platform auth config) can run `/topic`. An unauthorized sender gets a refusal instead of activation.

### DM Topics vs Multi-session DM mode

| | `extra.dm_topics` (config-driven) | `/topic` (user-driven) |
|---|---|---|
| Who activates it | Operator, in `config.yaml` | End user, by sending `/topic` |
| Topic list | Fixed set declared in config | User creates/deletes topics freely |
| Topic names | Chosen by operator | Chosen by user; auto-renamed to match Hermes session title |
| Root DM behavior | Normal chat (lobby if `ignore_root_dm: true`) | Becomes a system lobby (non-command messages are rejected) |
| Primary use case | Permanent workspaces with optional skill binding | Ad-hoc parallel sessions |
| Persistence | `extra.dm_topics` in config | `telegram_dm_topic_mode` + `telegram_dm_topic_bindings` SQLite tables |

Both features can coexist on the same bot — you'd run `/topic` from a user's DM, and `extra.dm_topics` continues to manage operator-declared topics for other chats.

### Prerequisites

In **@BotFather**, open your bot → **Bot Settings → Threads Settings**:

1. Turn on **Threaded Mode** (enables `has_topics_enabled`)
2. Do **not** disable users creating topics (keeps `allows_users_to_create_topics` on)

When the user first runs `/topic`, Hermes calls `getMe` to verify both flags. If either is off, Hermes sends a screenshot of the BotFather Threads Settings page and explains what to toggle — no activation happens until prerequisites are met.

### Activation flow

From the root DM, send:

```
/topic
```

Hermes will:

1. Check `getMe().has_topics_enabled` and `allows_users_to_create_topics`
2. If both are true, enable multi-session topic mode for this DM
3. Reply with a list of previous unlinked Telegram sessions the user can restore

After activation, the **root DM is a lobby**: normal prompts are rejected with guidance pointing at **All Messages**. System commands (`/status`, `/sessions`, `/usage`, `/help`, etc.) still work in the root.

### Creating a new topic (end-user flow)

1. Open the bot DM in Telegram
2. Tap **All Messages** at the top of the bot interface, then send any message
3. Telegram creates a new topic for that message
4. Hermes responds inside that topic — the topic is now a standalone session

Every topic gets its own conversation history, model state, tool execution, and session ID. The isolation key is `agent:main:telegram:dm:{chat_id}:{thread_id}` — identical to the config-driven DM topics isolation.

### Messages sent in the root DM

Some Telegram clients deliver a message typed in the root DM (General / **All Messages**) without a `message_thread_id`. By default Hermes treats such a message as a continuation and answers it in your **most recently active topic** (lobby recovery).

To get a fresh session for every root-DM message instead, turn on `dm_topic_auto_transfer`:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        dm_topic_auto_transfer: true   # default: false
```

With it on, a plain message in the root DM makes Hermes create a new topic (named "Hermes Chat" until the auto-title renames it), start a new session there, and post the reply inside that topic. Slash commands typed in the root DM stay lobby commands and never create a topic. If creating the topic fails (for example Threaded Mode was switched off in BotFather), Hermes falls back to the default behaviour and answers in your most recently active topic. The key only matters while `/topic` mode is on. The top-level `telegram.extra.dm_topic_auto_transfer` key (shown in Desktop Settings) works too.

### Auto-renamed topics

When Hermes generates a session title for a topic (via the auto-title pipeline, after the first exchange), the Telegram topic itself is renamed to match — e.g. "New Topic" becomes "Database migration plan". The rename is best-effort: failures are logged but don't break the session.

To disable this and keep your manually-chosen topic names untouched, set:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_topic_auto_rename: true
```

When this flag is on, Hermes still generates an internal session title (used by `hermes sessions`, the TUI, etc.) but never edits the Telegram topic name. Useful when you organise topics by hand under BotFather Threaded Mode and don't want every first reply to overwrite the title.

### `/new` inside a topic

Resets the current topic's session (new session ID, fresh history) without touching other topics. Hermes replies with a reminder that for parallel work, creating another topic (via **All Messages**) is usually what you want.

### Restoring a previous session

Inside a topic, send:

```
/topic <session-id>
```

This binds the current topic to an existing Hermes session instead of starting fresh. Useful for continuing a conversation that started before topic mode was enabled. Restrictions:

- The target session must belong to the same Telegram user
- The target session must not already be bound to another topic

Hermes confirms with the session title and replays the last assistant message for context.

To discover session IDs, send `/topic` (no argument) in the root DM — Hermes lists the user's unlinked Telegram sessions.

### `/topic` inside a topic (no argument)

Shows the current topic's binding: session title, session ID, and hints for `/new` vs creating another topic.

### Under the hood

- Activation persists to `telegram_dm_topic_mode(profile_name, chat_id, user_id, enabled, ...)` in `state.db`. Primary key is `(profile_name, chat_id)` so multiplexed / profile-routed bots sharing one `state.db` do not clobber each other when the same Telegram user DMs multiple bots (private `chat_id` is the user id and is identical across bots).
- Each topic binding persists to `telegram_dm_topic_bindings(profile_name, chat_id, thread_id, session_id, ...)` with PK `(profile_name, chat_id, thread_id)` and `ON DELETE CASCADE` on `session_id` — pruning a session automatically clears its topic binding
- The topic-mode SQLite migration is **opt-in**: it runs on the first `/topic` call, never on gateway startup. Until a user runs `/topic` in this profile, `state.db` is unchanged. Schema v3 adds `profile_name`; legacy rows migrate into the `default` namespace only
- Each inbound DM message looks up its `(profile_name, chat_id, thread_id)` binding using the **routed** profile (`source.profile`, not the process-global active profile). If present, the lookup routes the message to the bound session via `SessionStore.switch_session()` so the session-key-to-session-id mapping stays consistent on disk
- `/new` inside a topic rewrites the binding row to point at the new session ID, so the next message stays on the fresh session
- Topics declared in `extra.dm_topics` are **never auto-renamed** — the operator-chosen name is preserved even when multi-session mode is enabled
- Set `extra.disable_topic_auto_rename: true` to turn off auto-rename for **all** topics in the chat (ad-hoc topics created via Threaded Mode included)
- The General (pinned top) topic in a forum-enabled DM is treated as the root lobby, regardless of whether Telegram delivers its messages with `message_thread_id=1` or with no thread_id
- A root-lobby message is routed to the user's most recently bound topic; with `extra.dm_topic_auto_transfer: true` it instead opens a new topic via `createForumTopic` (falling back to the most recent topic if that call fails)
- Root-lobby reminders are rate-limited to one message per 30 seconds per **(profile, chat)** — a user who forgets topic mode is on and types ten prompts in the root won't get ten replies, and two multiplexed profiles sharing a chat id do not suppress each other's reminders
- BotFather setup screenshots are rate-limited to one send per 5 minutes per **(profile, chat)** — repeated `/topic` attempts while Threads Settings are still disabled won't re-upload the same image
- `/bg <prompt>` started inside a topic delivers its result back to the same topic; background sessions don't trigger auto-rename of the owning topic
- `/topic` itself is gated by the bot's user authorization check — unauthorized DMs get a refusal instead of activation
- When Telegram rejects a topic with `message thread not found` (the topic was deleted, or topics were turned off in the Telegram client), Hermes retries the same topic once, then delivers the reply to the root chat without its reply anchor and drops that topic's binding. A reply whose anchor message was deleted keeps its topic and only loses the anchor. A send to a topic Hermes just created by name fails instead, so the topic is recreated rather than the reply landing in the root. Removing the chat's last binding turns topic mode off, so the root DM works as a normal chat again

### Disabling multi-session mode

Send `/topic off` in the root DM. Hermes flips the row off for **this profile's** namespace, clears that profile's `(thread_id → session_id)` bindings for the chat, and the root DM reverts to a normal Hermes chat. Existing topics in Telegram aren't deleted — they just stop being gated as independent sessions. Re-run `/topic` later to turn it back on.

If you need to clean up by hand (e.g. a bulk reset across many chats), scope rows by `profile_name` (use `default` for single-profile installs):

```bash
sqlite3 ~/.hermes/state.db \
  "UPDATE telegram_dm_topic_mode SET enabled = 0
     WHERE profile_name = 'default' AND chat_id = '<your_chat_id>';
   DELETE FROM telegram_dm_topic_bindings
     WHERE profile_name = 'default' AND chat_id = '<your_chat_id>';"
```

### Downgrading Hermes

If you downgrade to a Hermes version that predates `/topic`, the feature simply stops working — the `telegram_dm_topic_mode` and `telegram_dm_topic_bindings` tables remain in `state.db` but are ignored by older code. DMs revert to the native per-thread isolation (each `message_thread_id` still gets its own session via `build_session_key`), so your existing Telegram topics keep working as parallel sessions. The root DM is no longer a lobby — messages there go into the agent like they used to. Re-upgrading reactivates multi-session mode exactly where it was.

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

- **Bot API 9.4 (Feb 2026):** Private Chat Topics — bots can create forum topics in 1-on-1 DM chats via `createForumTopic`. Hermes uses this for two distinct features: operator-curated [Private Chat Topics](#private-chat-topics-bot-api-94) (config-driven, fixed topic list) and user-driven [Multi-session DM mode](#multi-session-dm-mode-topic) (activated by `/topic`, unlimited user-created topics).
- **Privacy policy:** Telegram now requires bots to have a privacy policy. Set one via BotFather with `/setprivacy_policy`, or Telegram may auto-generate a placeholder. This is particularly important if your bot is public-facing.
- **Bot API 9.5 (Mar 2026): Native streaming via `sendMessageDraft`.** Hermes uses Telegram's native streaming-draft API for private chats under the default `auto` transport (groups and group forum topics always use the `editMessageText` path; private-chat topics try drafts too and fall back to edits if Telegram rejects them). If the draft preview visibly collapses and re-renders on your Telegram client, set `transport: edit` (see below).

### Streaming transport (`gateway.streaming.transport`)

When streaming is enabled (`gateway.streaming.enabled: true`), Hermes picks one of four transports:

| Value | Behaviour |
|---|---|
| `auto` (default) | Native draft streaming on supported chats (currently Telegram DMs); legacy edit-based path otherwise. Falls back gracefully if a draft frame fails. |
| `draft` | Force native drafts. Logs a downgrade and falls back to edit if the chat doesn't support drafts (e.g. groups and group forum topics). |
| `edit` | Legacy progressive `editMessageText` polling for every chat type. |
| `off` | Disable streaming entirely (final reply only, no progressive updates). |

In `~/.hermes/config.yaml`:

```yaml
gateway:
  streaming:
    enabled: true
    transport: auto    # auto | draft | edit | off
```

**What you'll see in DMs with `auto` (default) or `draft`** — Telegram shows an animated draft preview that updates token-by-token. When the reply finishes, it's delivered as a regular message and the draft preview clears naturally on the client. Drafts have no message id, so the final answer is what stays in your chat history.

**What you'll see in DMs with `edit`** — the gateway sends a normal preview message and progressively updates it via `editMessageText`. Pick this if your client shows the draft preview collapsing or flickering while it grows.

**What about groups, supergroups and group forum topics?** Telegram restricts `sendMessageDraft` to private chats (DMs). The gateway transparently falls back to the edit-based path for those — same UX as before. Topics inside a private chat count as DMs: they try drafts and fall back to edits if Telegram rejects them.

**How often does the preview update?** At most once per `streaming.edit_interval`: deltas that arrive in between are combined, so the next frame carries the latest text instead of a burst of near-identical frames. Telegram meters draft frames and the typing indicator together per chat (20 calls per 5 seconds, 40 per 30 seconds; see [Telegram's streaming notes](https://core.telegram.org/api/bots/ai)), and Hermes keeps both inside that budget, so lowering `edit_interval` cannot push more updates than Telegram allows.

**What if a draft frame fails?** A Telegram `RetryAfter` (flood control) pauses preview updates until the wait is over and then resumes with the latest text; the response stays on native drafts. Any other failure (transient network error, server-side rejection, older python-telegram-bot install) flips that response back to the edit-based path for the rest of the stream. The next response gets a fresh attempt. The complete answer is always delivered by the normal final send.

**Preview text growing in place (`plain_drafts`).** By default the draft preview is rendered with the same MarkdownV2 formatting as the final reply, which means text already on screen can re-render as a code block or bold span closes. Set `telegram.extra.plain_drafts: true` (`gateway.platforms.telegram.extra.plain_drafts` in the nested form) to stream the preview as raw text instead: every frame then only adds characters to the previous one, so Telegram animates just the new text. The final reply keeps its normal formatting either way, and `plain_drafts` takes precedence over `rich_drafts` for the preview.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        plain_drafts: true   # default false
```

## Rendering: Rich Messages, Tables and Link Previews

**Rich Messages (Bot API 10.1).** Final replies that contain constructs the legacy MarkdownV2 path degrades — tables, task lists, collapsible `<details>`, and block math — are sent with Telegram's native [`sendRichMessage`](https://core.telegram.org/bots/api#sendrichmessage) using the agent's **raw markdown**, so they render natively with no client-side flattening. In DMs, the default `rich_drafts: false` keeps the streaming preview plain — it uses Telegram's ephemeral draft transport with legacy rendering (tables and other rich-only constructs stay as raw markdown in the preview) — then persists the completed response with `sendRichMessage`. Setting `rich_drafts: true` makes the live preview use `sendRichMessageDraft` too. Edit-based streams can finalize an existing preview in place through `editMessageText`'s `rich_message` parameter. Ordinary replies (plain prose, bold/italic, simple lists) stay on the MarkdownV2 path for consistent font weight and spacing across clients.

A rich-eligible message longer than the 32,768-character rich text limit is sent as several rich messages, split between blocks (a `<details>` section, table or code block is never cut unless it alone is over the limit, in which case a table repeats its header and a code block reopens in the next part). Any rejection from Telegram (unsupported endpoint on an older `python-telegram-bot`, parser error, oversized blocks/columns) **transparently falls back** to the MarkdownV2 path — your message is never lost. Transient/network errors are *not* silently re-sent (no duplicate final message).

**Long replies arrive as a file.** A reply that would need more than 8 Telegram messages (about 32,000 characters of prose) is sent as one `reply-<id>.md` document holding the complete text, with a short caption, instead of a burst of messages that can trip Telegram's flood limits. While a long reply streams, Hermes stops opening new messages once it reaches the limit, uploads the finished reply as the file, and then deletes the partial messages it had already posted for it. If the upload fails, nothing is deleted and the rest of the reply arrives as ordinary messages, so no text is lost; a message Telegram will not let the bot delete (older than 48 hours, or missing permission) simply stays. `hermes send` and the `send_message` tool do the same for an over-long message. A reply over Telegram's 50 MB document limit is split across numbered files.

**MarkdownV2 fallback.** When the rich path is unavailable for a message, Hermes converts markdown to MarkdownV2. Since MarkdownV2 has no native table syntax, pipe tables are normalized:

- **Small tables** are flattened into **row-group bullets** — each row becomes a readable bulleted list under the column headings. Good for 2–4 columns and short cells.
- **Larger or wider tables** fall back to a **fenced code block** with aligned columns so nothing collapses.

LaTeX symbol macros in prose (`\rightarrow`, `\leq`, `\alpha`, ...) are shown as the symbols they name (→, ≤, α), and an inline `$x \leq y$` drops its dollar signs once it is plain text, since MarkdownV2 cannot typeset math. Code spans, code blocks, file paths and unknown macros keep their backslashes. On the rich path the same mapping applies only outside math spans, which Telegram typesets itself.

Rich messages are **opt-in**. The default stays on the legacy MarkdownV2 path because current Telegram clients can make Bot API rich messages difficult to copy as plain text, which is especially painful for command snippets and mobile handoffs. To enable native rendering for tables/task lists/details/math:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        rich_messages: true
        rich_drafts: false
        allow_cjk_rich_messages: false
```

This setting is for client-rendering/copy compatibility; Hermes already falls back automatically when Telegram rejects the rich API call. `rich_drafts` controls whether the DM streaming preview *renders* rich (`sendRichMessageDraft`) and stays off by default because Telegram Desktop/macOS can visually overlay rich draft frames until the chat redraws; with it off, the preview streams plain and the final still arrives as a native Rich Message.

The same setting applies to sends made outside the gateway process: `hermes send`, the `send_message` tool, and cron deliveries that fall back to the standalone sender take the rich path under the same rules. They also escape the `(1/2)` chunk marker on long messages, and treat only Telegram's own HTML tags (`<b>`, `<a>`, `<code>`, ...) as HTML. A `<details>` block or an angle-bracket placeholder therefore keeps its Markdown formatting. A message the Bot API still rejects as HTML is retried as MarkdownV2 before falling back to plain text. Two or more photos sent together arrive as a native album (`sendMediaGroup`, 10 per album).

CJK text (Chinese, Japanese, Korean, and rare Han extensions) stays on the legacy MarkdownV2 path by default because affected Telegram Desktop/macOS clients have rendered Bot API rich messages with overlapping CJK glyph artifacts. If you use an unaffected client and prefer native rich tables/task lists/details/math for CJK payloads, set `allow_cjk_rich_messages: true` alongside `rich_messages: true` to opt in to that client-side risk.

**Link previews.** Telegram auto-generates link previews for URLs in bot messages. If you'd rather suppress those (long `/tools` output, agent reply that mentions ten links, etc.):

```yaml
gateway:
  platforms:
    telegram:
      extra:
        disable_link_previews: true
```

When enabled, Hermes attaches Telegram's `LinkPreviewOptions(is_disabled=True)` to every outgoing message, including the streaming preview edits and the final edit of a streamed reply (edits that resolve a button prompt, such as an approval card, don't carry it yet), and falls back to the legacy `disable_web_page_preview` parameter on older `python-telegram-bot` versions.

**Reply anchor on split replies.** With the default `reply_to_mode: first`, the first part of a reply quotes the message it answers. Some Telegram Desktop/TDLib clients render that first part as an unsupported message when the reply is split into several parts. To send split replies without the quote (single-message replies keep it, and DM topics always keep it because the topic is routed by it):

```yaml
gateway:
  platforms:
    telegram:
      extra:
        anchor_multipart_replies: false   # default: true
```

This covers split MarkdownV2 replies and split Rich Messages. A streamed reply (`transport: edit`) picks its anchor when the first preview is sent, before its final length is known, and Telegram cannot remove a reply anchor by editing, so a streamed reply that later overflows keeps the quote on its first part.

**Long replies and flood control.** A reply longer than Telegram's 4,096-character limit is sent as numbered parts (`(1/3)`, `(2/3)`, …). A part boundary that falls inside bold, italic, underline, strikethrough or spoiler text closes the style at the end of one part and reopens it at the start of the next, so both parts keep the formatting. Sends to one chat are delivered one reply at a time, so a scheduled report and a DM answer landing together cannot interleave their parts, and a file upload cannot land between two parts of the text it accompanies. If Telegram's flood control refuses a part mid-way, Hermes resumes from the refused part once the penalty passes instead of re-sending the parts already on screen, and while a chat is inside a known penalty window further sends to it fail closed locally (no extra requests that would lengthen the penalty). That window survives a gateway restart: it is saved per bot in the profile home (`telegram_flood_cooldown_<bot id>.json`), so a restart during a ban does not fire straight back into it. A penalty longer than the gateway's inline wait cap is handed to the delivery ledger, which redelivers the reply with a "part of it may already have arrived above" note. File attachments (documents, photos, videos, voice) follow the same rule: a short penalty is waited out and the upload retried; a longer one schedules a single native re-upload once the window has passed, and you only see the "Couldn't deliver the file attachment" notice if that re-upload is refused too. An upload that never reached Telegram (a connect or pool failure) or that Telegram answered with its own 5xx error is retried up to three times with a short backoff. If all three fail, you get the "Couldn't deliver the file attachment" notice. An upload that may already be in the chat (the connection dropped or timed out after the file was sent) is never re-sent and posts no failure notice; it is reported as unconfirmed, so the `send_message` tool and cron delivery do not re-send it either.

## Group Allowlisting

Telegram groups and forum chats have two orthogonal gates you can configure:

- **Sender user IDs** (`group_allow_from` / `TELEGRAM_GROUP_ALLOWED_USERS`) — sender-scoped allowlist that applies only to group/forum messages. Use this when you want specific users to be able to invoke the bot in groups without adding them to `TELEGRAM_ALLOWED_USERS` (which would also give them DM access).
- **Chat IDs** (`group_allowed_chats` / `TELEGRAM_GROUP_ALLOWED_CHATS`) — chat-scoped allowlist. Any member of these groups/forums can interact with the bot. Useful for team/support bots where group membership itself is the access signal.

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Global access (DMs + groups). Users here can always invoke the bot.
        allow_from:
          - "123456789"
        # Sender IDs allowed in groups/forums only. Does NOT grant DM access.
        group_allow_from:
          - "987654321"
        # Entire groups/forums — any member is authorized.
        group_allowed_chats:
          - "-1001234567890"
```

Equivalent env vars:

```bash
TELEGRAM_ALLOWED_USERS="123456789"
TELEGRAM_GROUP_ALLOWED_USERS="987654321"
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

Behavior:

- `TELEGRAM_ALLOWED_USERS` covers all chat types (DMs, groups, forums).
- `TELEGRAM_GROUP_ALLOWED_USERS` only authorizes the listed senders in groups/forums. They still can't DM the bot unless listed in `TELEGRAM_ALLOWED_USERS`.
- A chat in `TELEGRAM_GROUP_ALLOWED_CHATS` authorizes every member of that chat, regardless of sender.
- Use `*` in any of these to allow any sender/chat.
- Edits to these lists in `config.yaml` apply to the next message with no gateway restart: a user you remove is refused from their next message or button tap. This holds for every profile a multiplexed gateway (`gateway.multiplex_profiles`) serves, including the host profile it was started with. A value set in `.env`, an external secret source or the gateway's own environment still wins over `config.yaml`.
- This layers on top of existing mention/pattern triggers and on top of `group_topics` + `ignored_threads`.

### Migration from before PR #17686

Prior to this split, `TELEGRAM_GROUP_ALLOWED_USERS` was the only knob and users put **chat IDs** in it. For backward compatibility, chat-ID-shaped values (starting with `-`) in `TELEGRAM_GROUP_ALLOWED_USERS` are still honored as chat IDs and a deprecation warning is logged once. Migration:

```bash
# Old (still works, but deprecated)
TELEGRAM_GROUP_ALLOWED_USERS="-1001234567890"

# New
TELEGRAM_GROUP_ALLOWED_CHATS="-1001234567890"
```

### Guest @mention bypass (`guest_mode`)

In a typical setup, `group_allowed_chats` is a hard gate: messages from groups outside the list are silently dropped, even if a member explicitly @mentions the bot. That's the right default for support / team bots.

For more casual setups — friend group chats where you want the bot **mostly silent** but **occasionally available on explicit ping** — enable `guest_mode`:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        group_allowed_chats:
          - "-1001234567890"   # your main allowlisted group
        guest_mode: true       # non-allowlisted groups: allow on @mention only
```

Env equivalent:

```bash
TELEGRAM_GUEST_MODE=true
```

Default: `false`.

With `guest_mode: true`, a message from a non-allowlisted group is processed **only** if it explicitly @mentions the bot. The mention is required every turn — there's no session stickiness for guest interactions, so the bot never auto-engages in a friend group thread it isn't pinged into.

DMs and allowlisted groups behave exactly as before.

Broadcast **channel** posts run the same `allowed_chats`, `require_mention` and topic gates as groups, but never get the guest bypass: anyone with posting rights in a channel could otherwise cross the allowlist. Add the channel's ID to `allowed_chats` and @mention the bot in a post to address it.

## Slash Command Access Control

By default, every allowed user can run every slash command. To split your allowlist into **admins** (full slash command access) and **regular users** (only commands you explicitly enable), add `allow_admin_from` and `user_allowed_commands` to the platform's `extra` block:

```yaml
gateway:
  platforms:
    telegram:
      extra:
        # Existing allowlists (unchanged)
        allow_from:
          - "123456789"     # admin
          - "555555555"     # regular user
          - "777777777"     # regular user

        # NEW — admins get all slash commands (built-in + plugin)
        allow_admin_from:
          - "123456789"

        # NEW — non-admin allowed users can only run these slash commands.
        # /help and /whoami are always allowed so users can see their access.
        user_allowed_commands:
          - status
          - model
          - history

        # Optional: separate admin/command lists for groups
        group_allow_admin_from:
          - "123456789"
        group_user_allowed_commands:
          - status
```

**Behavior:**

- A user listed in `allow_admin_from` for a scope (DM or group) can run **every** registered slash command — built-in commands AND plugin-registered ones — through the live registry.
- A user in `allow_from` but **not** in `allow_admin_from` can only run commands listed in `user_allowed_commands`, plus the always-allowed floor: `/help` and `/whoami`.
- Plain chat (non-slash messages) is unaffected. Non-admin users can still talk to the agent normally, they just can't trigger arbitrary commands.
- **Backward compat:** if `allow_admin_from` is not set for a scope, slash command gating is disabled for that scope. Existing installs keep working with no changes.
- DM admin status does not imply group admin status. Each scope has its own admin list.
- If only `group_allow_admin_from` is set, DM scope stays in unrestricted (backward-compat) mode.

Use `/whoami` to see the active scope, your tier (admin / user / unrestricted), and which slash commands you can run.

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
3. Known IPv4 Telegram API IPs are tried **before** the dual-stack `api.telegram.org` hostname. A blackholed IPv6 path can sit in `connect()` without erroring, which used to pin the event loop so the 30s init deadline never fired.
4. A hardcoded IPv4 seed list (`149.154.166.110`, `149.154.167.220`) always follows the DoH answers (DoH often returns a single edge, which would otherwise be the only fallback), and is the whole list when DoH is blocked or times out. The hostname remains last resort.
5. Once a path succeeds, it becomes "sticky" — subsequent requests use it directly. The hostname is kept as a last resort for IPv6-only networks.
6. One request's walk across these paths shares a 15s connect budget, half the 30s send deadline. The first path keeps its full connect timeout and later paths split the rest. If every path hangs, the send fails with a connect timeout before the deadline. The gateway retries it, because the message never left.

Sends made outside the gateway process (`hermes send`, the `send_message` tool, standalone cron delivery) use the same transport with your configured IPs or the seed list. They skip DoH discovery so a one-shot send stays fast. They also use the gateway's `HERMES_TELEGRAM_HTTP_*` timeouts and its media-upload budgets. A connect or pool timeout is retried there, because the request never left. Any other timeout is not retried: Telegram may already have delivered the message, and `hermes send` says so in its warning instead of inviting a resend.

A media upload from a file on disk (gateway or standalone) gets read and write timeouts of 60s plus the time the file takes at 128 KB/s. Gateway uploads also get a wall-clock cap of 300s plus 1.5x that time (at most one hour). `hermes send` and the `send_message` tool are bounded only by those per-request timeouts. Standalone cron delivery keeps its `cron.standalone_send_timeout_seconds` cap (60s by default), widened by 1.5x the upload time when the delivery carries files, and a cron delivery through the live gateway widens `cron.media_send_timeout_seconds` the same way. A large file on a slow uplink, or through a self-hosted Bot API server that only answers once it has forwarded the file, then waits for Telegram's answer instead of reporting a failure for a file that lands minutes later.

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
You usually don't need to configure this manually. The auto-discovery via DoH handles most restricted-network scenarios. The `TELEGRAM_FALLBACK_IPS` env var is only needed if DoH is also blocked on your network. If IPv6 is broken on the host, you can also set `network.force_ipv4: true` in `config.yaml` to skip AAAA lookups process-wide.
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

### TLS handshake failures with OpenSSL 3.5

If Telegram repeatedly reconnects with an empty `httpx.ConnectError` or an
`SSL: UNEXPECTED_EOF_WHILE_READING`, while curl works through the same proxy,
the network path may reject OpenSSL 3.5's larger post-quantum TLS ClientHello.
This symptom alone is not proof: compare the default handshake with X25519
before enabling the workaround.

To use classic X25519 key exchange only for Telegram, add to `config.yaml`
and restart the gateway:

```yaml
platforms:
  telegram:
    extra:
      tls_classic_key_exchange: true
```

This covers both Bot API requests and long polling, including proxy and
fallback-IP connections, the TLS handshake with an `https://` proxy itself,
DNS-over-HTTPS fallback-IP discovery, and one-shot sends (cron,
`send_message`). TLS 1.3,
hostname checks, and certificate verification remain enabled; the platform
trust store and `SSL_CERT_FILE` are still used. The tradeoff is opting out of
post-quantum key exchange for Telegram. Other platforms and model-provider
clients keep their existing TLS settings. The default is `false`; remove the
setting or set it to `false` to restore OpenSSL's default groups. A value other
than true/false is ignored with a warning in the gateway log. See also [the related OpenSSL 3.5 report](https://github.com/NousResearch/hermes-agent/issues/106384).

## Message Reactions

The bot can add emoji reactions to messages as visual processing feedback:

- 👀 when the bot starts processing your message
- 👍 when the response is delivered successfully
- 👎 if an error occurs during processing

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
Unlike Discord (where reactions are additive), Telegram's Bot API replaces all bot reactions in a single call. The transition from 👀 to 👍/👎 happens atomically — you won't see both at once.
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
| Logs show `Telegram polling conflict` (409) | Another poller holds the bot's `getUpdates` session. This is usually the gateway's own previous long poll, which Telegram expires on its own: the adapter retries for about 200 s, then the gateway reconnects it again about 2, 5 and 10 minutes later (every profile in a multiplexed gateway does this). If the token is still in use after that, `hermes status` shows the platform as `fatal` with `token_in_use_elsewhere` and needs-attention, and the gateway stops retrying. Stop the other poller (another machine, a leftover process, or another bot framework using this token), then run `hermes gateway restart`. Two of your own profiles set up with the same token never get this far: the second one is refused at startup with `duplicate_credential` (or `telegram-bot-token_lock` when the other is a separate local gateway). |
| Bot token revoked/invalid | Generate a new token via `/revoke` then `/newbot` or `/token` in BotFather. Update your `.env` file. |
| Webhook not receiving updates | Verify `TELEGRAM_WEBHOOK_URL` is publicly reachable (test with `curl`). Ensure your platform/reverse proxy routes inbound HTTPS traffic from the URL's port to the local listen port configured by `TELEGRAM_WEBHOOK_PORT` (they do not need to be the same number). Ensure SSL/TLS is active — Telegram only sends to HTTPS URLs. Check firewall rules. |

## Exec Approval

When the agent tries to run a potentially dangerous command, it asks you for approval in the chat:

> ⚠️ This command is potentially dangerous (recursive delete). Reply "yes" to approve.

Reply "yes"/"y" to approve or "no"/"n" to deny.

## Interactive Prompts (clarify)

When the agent calls the `clarify` tool — to ask which approach you prefer, get post-task feedback, or check before a non-trivial decision — Telegram renders the question with **inline keyboard buttons**:

> ❓ Which framework should I use for the dashboard?
>
> [1. Next.js] [2. Remix] [3. Astro]
> [✏️ Other (type answer)]

Tap a button to answer, or tap **Other** to type a free-form response (the next message you send becomes the answer). Open-ended `clarify` calls (no preset choices) skip the buttons and just capture your next message.

When `clarify` is in the session's tools (not in cron jobs or profiles that disable it), the Telegram platform hint tells the agent to use it for small fixed-choice decisions, so it does not improvise numbered lists or copy-to-clipboard buttons, which never send an answer back. Persistent agent-authored action buttons are not a generic Hermes feature yet.

Configure the response timeout via `agent.clarify_timeout` in `~/.hermes/config.yaml` (default `90` seconds). If you don't respond within the timeout, the agent unblocks with `"outcome": "timed_out"` and adapts rather than hanging. Reply `skip` to skip a question.

If Telegram cannot render the button card (the Bot API rejects it, or the send fails after its 15-second acknowledgement window), Hermes re-asks the same question as a plain numbered-list message and your typed reply (a number or the option text) is taken as the answer. When even that cannot be delivered, the agent is released at once with `[clarify prompt could not be delivered]` instead of waiting out the timeout and mistaking the silence for you not answering.

## Push notification volume

Telegram fires a push notification on every message the bot sends. For long agent turns that emit tool-progress bubbles, streaming updates, and status callbacks, this gets noisy fast. The Telegram adapter has two notification modes:

| Mode | Behavior |
|------|----------|
| `important` (default) | Only **final responses**, **approval prompts**, and **slash-command confirmations** ring. Tool progress, streaming chunks, and status messages are delivered with `disable_notification=true`. |
| `all` | Every outgoing message fires a push notification. Legacy behavior; opt in if you genuinely want to hear about every tool call. |

Configure in `~/.hermes/config.yaml`:

```yaml
display:
  platforms:
    telegram:
      notifications: important   # or "all"
```

Env override (handy for quick A/B testing):

```bash
HERMES_TELEGRAM_NOTIFICATIONS=all
```

Unknown values log a warning and fall back to `important`.

`hermes send`, the `send_message` tool and standalone cron delivery follow the same mode. Under `important` their messages arrive silently, except cron briefs, which ring by default (`cron.delivery.notify`, as in the gateway lane).

## Status messages edited in place

The Telegram adapter routes recurring agent status callbacks (e.g. "Compressing context…", "Calling tool…") through `send_or_update_status()`, which keeps a `{(chat_id, status_key) → message_id}` cache and **edits the existing bubble** on subsequent emits instead of appending a new one each time. Distinct `status_key` values get their own messages; distinct chats never collide. If the edit fails (e.g. the user deleted the message, or it's older than Telegram allows for edits), the cache entry is dropped and the next emit posts a fresh message and re-caches its ID. No config required — this is the default Telegram behavior. Other adapters that don't implement `send_or_update_status` fall through to plain `send()` unchanged.

## Pin incoming user message during agent turn

When a user sends a message that triggers an agent turn, the Telegram adapter pins that incoming message for the duration of the turn and unpins it when the response is finished — a lightweight visual indicator that the bot is actively working on the message rather than ignoring it. The pin uses `disable_notification=true` to avoid extra pings. No config required.

## Security

:::warning
Always set `TELEGRAM_ALLOWED_USERS` to restrict who can interact with your bot. Without it, the gateway denies all users by default as a safety measure.
:::

Never share your bot token publicly. If compromised, revoke it immediately via BotFather's `/revoke` command.

For more details, see the [Security documentation](../security.md). You can also use [DM pairing](./index.md#dm-pairing-alternative-to-allowlists) for a more dynamic approach to user authorization.

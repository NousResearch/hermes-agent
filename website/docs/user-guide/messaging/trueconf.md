---
sidebar_position: 8
title: "TrueConf"
description: "Set up Hermes Agent as a TrueConf bot"
---

# TrueConf Setup

Hermes Agent integrates with TrueConf as a conversational bot via persistent WebSocket connection. Once connected, you can chat with your agent directly in TrueConf DMs or group chats, send voice messages, receive scheduled task results, and share files securely within your corporate TrueConf environment. The integration is built on the [`python-trueconf-bot` library](https://trueconf.github.io/python-trueconf-bot/latest/) and supports text, voice, images, and file attachments.

## Step 1: Get TrueConf Bot Credentials

To connect Hermes to your TrueConf server, you need a bot account with a username and password, plus your TrueConf server address.

1. Contact your TrueConf server administrator to create a bot account
2. Obtain the following credentials:
   - **TrueConf Server Address** (e.g., `trueconf.example.com` or `message.example.com`)
   - **Bot Username** (TrueConf ID e.g., `hermes_bot@trueconf.example.com`)
   - **Bot Password** (the password for the bot account)

:::warning
Keep your bot credentials secure. Anyone with these credentials can control your bot within your TrueConf environment.
:::

## Step 2: Find Your TrueConf User ID / Email

Hermes Agent uses TrueConf user IDs or chat IDs to control access. TrueConf user ID looks like email address but is another object (e.g., `user@example.com`). See details in [TrueConf documentation](https://trueconf.com/docs/server/en/admin/users/).

If you need to find your chat ID (UUID format), you can:
- Check the TrueConf client's chat info
- Contact your administrator
- Look at incoming message logs in `~/.hermes/logs/bot.log` after initial connection

Save this identifier; you'll need it for the next step.

## Step 3: Configure Hermes

### Option A: Interactive Setup (Recommended)

```bash
hermes gateway setup
```

Select **TrueConf** when prompted. The wizard asks for your server address, bot TrueConf ID, password, and allowed user IDs, then writes the configuration for you.

### Option B: Manual Configuration

Add the following to `~/.hermes/.env`:

```bash
TRUECONF_SERVER=trueconf.example.com
TRUECONF_USERNAME=hermes_bot@trueconf.example.com
TRUECONF_PASSWORD=your_bot_password
TRUECONF_ALLOWED_USERS=user1@example.com,user2@example.com    # Comma-separated for multiple users
TRUECONF_HOME_CHANNEL=hermes_bot@trueconf.example.com  # Optional: default chat for cron delivery
```

Or use `config.yaml`:

```yaml
gateway:
  platforms:
    trueconf:
      enabled: true
      extra:
        server: trueconf.example.com
        username: hermes_bot@trueconf.example.com
        password: your_bot_password
        allowed_users: "user1@example.com,user2@example.com"
        home_channel: "user@trueconf.example.com"
```

### Start the Gateway

```bash
hermes gateway start
```

The bot should connect within seconds via WebSocket. Send it a message on TrueConf to verify.

## Sending Generated Files (MEDIA: Support)

TrueConf adapter supports sending files via the `MEDIA:/path/to/file` syntax. Files are sent as native TrueConf attachments.

### Supported `MEDIA:` file extensions

| Category | Extensions |
|---|---|
| Images | `png`, `jpg`, `jpeg`, `gif`, `webp`, `bmp`, `tiff`, `svg` |
| Audio | `mp3`, `wav`, `ogg`, `m4a`, `opus`, `flac`, `aac` |
| Video | `mp4`, `mov`, `webm`, `mkv`, `avi` |
| **Documents** | `pdf`, `txt`, `md`, `csv`, `json`, `xml`, `html`, `yaml`, `yml`, `log` |
| **Office** | `docx`, `xlsx`, `pptx`, `odt`, `ods`, `odp` |
| **Archives** | `zip`, `rar`, `7z`, `tar`, `gz`, `bz2` |

Anything on this list is delivered as a native attachment on TrueConf; unsupported types fall back to a plain-text indicator.

## Connection Mode

TrueConf adapter uses **persistent WebSocket connection** to receive incoming messages in real-time. This is the default and only mode, optimized for corporate environments where the gateway can maintain a long-lived connection to the TrueConf server.

### SSL Verification

By default, the adapter verifies SSL certificates when connecting to the TrueConf server. If your server uses a self-signed certificate (common in internal corporate environments), you can disable verification:

**Option 1: config.yaml (recommended)**
```yaml
gateway:
  platforms:
    trueconf:
      extra:
        verify_ssl: false
```

**Option 2: environment variable**
```bash
TRUECONF_VERIFY_SSL=false
```

:::warning
Disabling SSL verification reduces security. Only use this for TrueConf servers you trust.
:::

## Parse mode

TrueConf Bot API supports three message formatting types:
*  `text` – plain text. All characters will be preserved and displayed.
*  `markdown` - basic Markdown formatting is supported. Special characters are converted into formatted text (bold, italic, links, etc.);
*  `html` - HTML tags will be interpreted and displayed as formatted text.

You can select parse mode for bot messages by the `TRUECONF_PARSE_MODE` variable.

HTML is the default mode for adapter because of better support by API.

## Home Channel

Use the `/sethome` command in any TrueConf chat (DM or group) to designate it as the **home channel**. Scheduled tasks (cron jobs) deliver their results to this channel.

You can also set it manually in `~/.hermes/.env`:

```bash
TRUECONF_HOME_CHANNEL=user@example.com
TRUECONF_HOME_CHANNEL_NAME="My Notes"
```

:::tip
TrueConf chat IDs are UUIDs (e.g., `123e4567-e89b-12d3-a456-426614174000`). Your personal DM chat ID is your user TrueConf ID.
:::

## Voice Messages

### Incoming Voice (Speech-to-Text)

Voice messages you send on TrueConf are automatically transcribed by Hermes's configured STT provider and injected as text into the conversation.

- `local` uses `faster-whisper` on the machine running Hermes — no API key required
- `groq` uses Groq Whisper and requires `GROQ_API_KEY`
- `openai` uses OpenAI Whisper and requires `VOICE_TOOLS_OPENAI_KEY`

### Outgoing Voice (Text-to-Speech)

When the agent generates audio via TTS, it's delivered as native TrueConf voice messages.

- **OpenAI and ElevenLabs** produce supported audio formats natively
- **Edge TTS** (the default free provider) outputs MP3 and may require conversion depending on TrueConf support

Configure the TTS provider in your `config.yaml` under the `tts.provider` key.

## Group Chat Usage

Hermes Agent works in TrueConf group chats with the following considerations:

- `TRUECONF_ALLOWED_USERS` still applies — only authorized users can trigger the bot, even in groups
- The bot sees all messages in groups where it's a member (similar to Telegram's privacy mode OFF)
- Use `trueconf.require_mention: true` to make the bot only respond when mentioned

### Example group configuration

Add this to `~/.hermes/config.yaml`:

```yaml
gateway:
  platforms:
    trueconf:
      extra:
        require_mention: true
```

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `TRUECONF_SERVER` | Yes | TrueConf server address (e.g., `trueconf.example.com`) |
| `TRUECONF_USERNAME` | Yes | Bot username (TrueConf ID) |
| `TRUECONF_PASSWORD` | Yes | Bot password |
| `TRUECONF_ALLOWED_USERS` | No | Comma-separated list of allowed user IDs or chat IDs |
| `TRUECONF_ALLOW_ALL_USERS` | No | Set to `true` to allow all users (not recommended for production) |
| `TRUECONF_HOME_CHANNEL` | No | Default chat ID for cron delivery |
| `TRUECONF_VERIFY_SSL` | No | Set to `false` to disable SSL verification (default: `true`) |

## Troubleshooting

### Bot not connecting
1. Verify server address, username, and password are correct
2. Check `~/.hermes/logs/gateway.log` for connection errors
3. Ensure the TrueConf server is reachable from the gateway machine
4. If using SSL, verify the certificate is trusted or set `TRUECONF_VERIFY_SSL=false`

### Messages not received
1. Check `~/.hermes/logs/bot.log` for incoming message logs
2. Verify your user email is in `TRUECONF_ALLOWED_USERS`
3. Ensure the bot is a member of the group chat (for group messages)

### File sending fails
1. Verify the file path in `MEDIA:` is readable by the gateway process
2. Check file extension is in the supported list
3. Look for file size limits in TrueConf server settings

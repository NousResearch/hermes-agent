---
sidebar_position: 10
title: "Voice Mode"
description: "Real-time voice conversations with Hermes Agent — CLI, Telegram, Discord (DMs, text channels, and voice channels)"
---

# Voice Mode

Hermes Agent supports full voice interaction across CLI and messaging platforms. Talk to the agent using your microphone, hear spoken replies, and have live voice conversations in Discord voice channels.

## Prerequisites

Before using voice features, make sure you have:

1. **Hermes Agent installed** — `pip install hermes-agent` (see [Getting Started](../../getting-started.md))
2. **An LLM provider configured** — set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL` in `~/.hermes/.env`
3. **A working base setup** — run `hermes` to verify the agent responds to text before enabling voice

:::tip
The `~/.hermes/` directory and default `config.yaml` are created automatically the first time you run `hermes`. You only need to create `~/.hermes/.env` manually for API keys.
:::

## Overview

| Feature | Platform | Description |
|---------|----------|-------------|
| **Interactive Voice** | CLI | Press Ctrl+B to record, agent auto-detects silence and responds |
| **Auto Voice Reply** | Telegram, Discord | Agent sends spoken audio alongside text responses |
| **Voice Channel** | Discord | Bot joins VC, listens to users speaking, speaks replies back |

## Requirements

### Python Packages

```bash
# CLI voice mode (microphone + audio playback)
pip install hermes-agent[voice]

# Discord + Telegram messaging (includes discord.py[voice] for VC support)
pip install hermes-agent[messaging]

# Premium TTS (ElevenLabs)
pip install hermes-agent[tts-premium]

# Everything at once
pip install hermes-agent[all]
```

| Extra | Packages | Required For |
|-------|----------|-------------|
| `voice` | `sounddevice`, `numpy` | CLI voice mode |
| `messaging` | `discord.py[voice]`, `python-telegram-bot`, `aiohttp` | Discord & Telegram bots |
| `tts-premium` | `elevenlabs` | ElevenLabs TTS provider |

:::info
`discord.py[voice]` installs **PyNaCl** (for voice encryption) and **opus bindings** automatically. This is required for Discord voice channel support.
:::

### System Dependencies

```bash
# macOS
brew install portaudio ffmpeg opus

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
```

| Dependency | Purpose | Required For |
|-----------|---------|-------------|
| **PortAudio** | Microphone input and audio playback | CLI voice mode |
| **ffmpeg** | Audio format conversion (MP3 → Opus, PCM → WAV) | All platforms |
| **Opus** | Discord voice codec | Discord voice channels |

### API Keys

Add to `~/.hermes/.env`:

```bash
# Speech-to-Text (at least one required)
GROQ_API_KEY=your-key              # Groq Whisper — fast, free tier available (recommended)
VOICE_TOOLS_OPENAI_KEY=your-key    # OpenAI Whisper — alternative

# Text-to-Speech (optional — Edge TTS works without any key)
ELEVENLABS_API_KEY=your-key        # ElevenLabs — premium quality
```

---

## CLI Voice Mode

### Quick Start

Start the CLI and enable voice mode:

```bash
hermes                # Start the interactive CLI
```

Then use these commands inside the CLI:

```
/voice          Toggle voice mode on/off
/voice on       Enable voice mode
/voice off      Disable voice mode
/voice tts      Toggle TTS output
/voice status   Show current state
```

### How It Works

1. Start the CLI with `hermes` and enable voice mode with `/voice on`
2. **Press Ctrl+B** — a beep plays (880Hz), recording starts
3. **Speak** — a live audio level bar shows your input: `● [▁▂▃▅▇▇▅▂] ❯`
4. **Stop speaking** — after 3 seconds of silence, recording auto-stops
5. **Two beeps** play (660Hz) confirming the recording ended
6. Audio is transcribed via Whisper and sent to the agent
7. If TTS is enabled, the agent's reply is spoken aloud
8. Recording **automatically restarts** — speak again without pressing any key

This loop continues until you press **Ctrl+B** during recording (exits continuous mode) or 3 consecutive recordings detect no speech.

:::tip
The record key is configurable via `voice.record_key` in `~/.hermes/config.yaml` (default: `ctrl+b`).
:::

### Silence Detection

Two-stage algorithm detects when you've finished speaking:

1. **Speech confirmation** — waits for audio above the RMS threshold (200) for at least 0.3s, tolerating brief dips between syllables
2. **End detection** — once speech is confirmed, triggers after 3.0 seconds of continuous silence

If no speech is detected at all for 15 seconds, recording stops automatically.

Both `silence_threshold` and `silence_duration` are configurable in `config.yaml`.

### Streaming TTS

When TTS is enabled, the agent speaks its reply **sentence-by-sentence** as it generates text — you don't wait for the full response:

1. Buffers text deltas into complete sentences (min 20 chars)
2. Strips markdown formatting and `<think>` blocks
3. Generates and plays audio per sentence in real-time

### Hallucination Filter

Whisper sometimes generates phantom text from silence or background noise ("Thank you for watching", "Subscribe", etc.). The agent filters these out using a set of 26 known hallucination phrases across multiple languages, plus a regex pattern that catches repetitive variations.

---

## Gateway Voice Reply (Telegram & Discord)

If you haven't set up your messaging bots yet, see the platform-specific guides:
- [Telegram Setup Guide](../messaging/telegram.md)
- [Discord Setup Guide](../messaging/discord.md)

Start the gateway to connect to your messaging platforms:

```bash
hermes gateway        # Start the gateway (connects to configured platforms)
hermes gateway setup  # Interactive setup wizard for first-time configuration
```

### Discord: Channels vs DMs

The bot supports two interaction modes on Discord:

| Mode | How to Talk | Mention Required | Setup |
|------|------------|-----------------|-------|
| **Direct Message (DM)** | Open the bot's profile → "Message" | No | Works immediately |
| **Server Channel** | Type in a text channel where the bot is present | Yes (`@botname`) | Bot must be invited to the server |

**DM (recommended for personal use):** Just open a DM with the bot and type — no @mention needed. Voice replies and all commands work the same as in channels.

**Server channels:** The bot only responds when you @mention it (e.g. `@hermesbyt4 hello`). Make sure you select the **bot user** from the mention popup, not the role with the same name.

:::tip
To disable the mention requirement in server channels, add to `~/.hermes/.env`:
```bash
DISCORD_REQUIRE_MENTION=false
```
Or set specific channels as free-response (no mention needed):
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### Commands

These work in both Telegram and Discord (DMs and text channels):

```
/voice          Toggle voice mode on/off
/voice on       Voice replies only when you send a voice message
/voice tts      Voice replies for ALL messages
/voice off      Disable voice replies
/voice status   Show current setting
```

### Modes

| Mode | Command | Behavior |
|------|---------|----------|
| `off` | `/voice off` | Text only (default) |
| `voice_only` | `/voice on` | Speaks reply only when you send a voice message |
| `all` | `/voice tts` | Speaks reply to every message |

Voice mode setting is persisted across gateway restarts.

### Platform Delivery

| Platform | Format | Notes |
|----------|--------|-------|
| **Telegram** | Voice bubble (Opus/OGG) | Plays inline in chat. ffmpeg converts MP3 → Opus if needed |
| **Discord** | Audio file attachment (MP3) | Sent alongside text response |

---

## Discord Voice Channels

The most immersive voice feature: the bot joins a Discord voice channel, listens to users speaking, transcribes their speech, processes through the agent, and speaks the reply back in the voice channel.

### Setup

#### 1. Discord Bot Permissions

If you already have a Discord bot set up for text (see [Discord Setup Guide](../messaging/discord.md)), you need to add voice permissions.

Go to the [Discord Developer Portal](https://discord.com/developers/applications) → your application → **Installation** → **Default Install Settings** → **Guild Install**:

**Add these permissions to the existing text permissions:**

| Permission | Purpose | Required |
|-----------|---------|----------|
| **Connect** | Join voice channels | Yes |
| **Speak** | Play TTS audio in voice channels | Yes |
| **Use Voice Activity** | Detect when users are speaking | Recommended |

**Updated Permissions Integer:**

| Level | Integer | What's Included |
|-------|---------|----------------|
| Text only | `274878286912` | View Channels, Send Messages, Read History, Embeds, Attachments, Threads, Reactions |
| Text + Voice | `274881432640` | All above + Connect, Speak |

**Re-invite the bot** with the updated permissions URL:

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274881432640
```

Replace `YOUR_APP_ID` with your Application ID from the Developer Portal.

:::warning
Re-inviting the bot to a server it's already in will update its permissions without removing it. You won't lose any data or configuration.
:::

#### 2. Privileged Gateway Intents

In the [Developer Portal](https://discord.com/developers/applications) → your application → **Bot** → **Privileged Gateway Intents**, enable all three:

| Intent | Purpose |
|--------|---------|
| **Presence Intent** | Detect user online/offline status |
| **Server Members Intent** | Map voice SSRC identifiers to Discord user IDs |
| **Message Content Intent** | Read text message content in channels |

All three are required for full voice channel functionality. **Server Members Intent** is especially critical — without it, the bot cannot identify who is speaking in the voice channel.

#### 3. Opus Codec

The Opus codec library must be installed on the machine running the gateway:

```bash
# macOS (Homebrew)
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

The bot auto-loads the codec from:
- **macOS:** `/opt/homebrew/lib/libopus.dylib`
- **Linux:** `libopus.so.0`

#### 4. Environment Variables

```bash
# ~/.hermes/.env

# Discord bot (already configured for text)
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — at least one required for voice channel listening
GROQ_API_KEY=your-key              # Recommended (fast, free tier)

# TTS — optional, Edge TTS (free) is the default
# ELEVENLABS_API_KEY=your-key      # Premium quality
```

### Start the Gateway

```bash
hermes gateway        # Start with existing configuration
```

The bot should come online in Discord within a few seconds.

### Commands

Use these in the Discord text channel where the bot is present:

```
/voice join      Bot joins your current voice channel
/voice channel   Alias for /voice join
/voice leave     Bot disconnects from voice channel
/voice status    Show voice mode and connected channel
```

:::info
You must be in a voice channel before running `/voice join`. The bot joins the same VC you're in.
:::

### How It Works

When the bot joins a voice channel, it:

1. **Captures audio** via Discord's UDP socket (RTP packets)
2. **Decrypts** using NaCl transport encryption (aead_xchacha20_poly1305_rtpsize)
3. **Decrypts** DAVE end-to-end encryption (Discord Audio/Video Encryption)
4. **Decodes** Opus audio to raw PCM (48kHz stereo, per-user decoder)
5. **Detects silence** — 1.5s of silence after at least 0.5s of speech triggers processing
6. **Converts** PCM to 16kHz mono WAV via ffmpeg
7. **Transcribes** via Whisper STT (Groq or OpenAI)
8. **Processes** through the full agent pipeline (session, tools, memory)
9. **Generates TTS** reply audio
10. **Plays** the reply in the voice channel

### Text Channel Integration

When the bot is in a voice channel:

- Transcripts appear in the text channel: `[Voice] @user: what you said`
- Agent responses are sent as text in the channel AND spoken in the VC
- The text channel is the one where `/voice join` was issued

### Echo Prevention

The bot automatically pauses its audio listener while playing TTS replies, preventing it from hearing and re-processing its own output.

### Access Control

Only users listed in `DISCORD_ALLOWED_USERS` can interact via voice. Other users' audio is silently ignored.

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## Configuration Reference

### config.yaml

```yaml
# Voice recording (CLI)
voice:
  record_key: "ctrl+b"            # Key to start/stop recording
  max_recording_seconds: 120       # Maximum recording length
  auto_tts: false                  # Auto-enable TTS when voice mode starts
  silence_threshold: 200           # RMS level (0-32767) below which counts as silence
  silence_duration: 3.0            # Seconds of silence before auto-stop

# Speech-to-Text
stt:
  enabled: true
  model: "whisper-1"               # Or: whisper-large-v3-turbo (Groq)

# Text-to-Speech
tts:
  provider: "edge"                 # "edge" (free) | "elevenlabs" | "openai"
  edge:
    voice: "en-US-AriaNeural"      # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
```

### Environment Variables

```bash
# Speech-to-Text providers
GROQ_API_KEY=...                   # Groq Whisper (recommended — fast, free tier)
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper (alternative)

# Text-to-Speech providers (Edge TTS needs no key)
ELEVENLABS_API_KEY=...             # ElevenLabs (premium quality)
# OpenAI TTS uses VOICE_TOOLS_OPENAI_KEY

# Discord voice channel
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### STT Provider Comparison

| Provider | Model | Speed | Quality | Cost |
|----------|-------|-------|---------|------|
| **Groq** | `whisper-large-v3-turbo` | Very fast (~0.5s) | Good | Free tier |
| **Groq** | `whisper-large-v3` | Fast (~1s) | Better | Free tier |
| **OpenAI** | `whisper-1` | Fast (~1s) | Good | Low |
| **OpenAI** | `gpt-4o-transcribe` | Medium (~2s) | Best | Higher |

### TTS Provider Comparison

| Provider | Quality | Cost | Latency | Key Required |
|----------|---------|------|---------|-------------|
| **Edge TTS** | Good | Free | ~1s | No |
| **ElevenLabs** | Excellent | Paid | ~2s | Yes |
| **OpenAI TTS** | Good | Paid | ~1.5s | Yes |

---

## Troubleshooting

### "No audio device found" (CLI)

PortAudio is not installed:

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

### Bot doesn't respond in Discord server channels

The bot requires an @mention by default in server channels. Make sure you:

1. Type `@` and select the **bot user** (with the #discriminator), not the **role** with the same name
2. Or use DMs instead — no mention needed
3. Or set `DISCORD_REQUIRE_MENTION=false` in `~/.hermes/.env`

### Bot joins VC but doesn't hear me

- Check your Discord user ID is in `DISCORD_ALLOWED_USERS`
- Make sure you're not muted in Discord
- The bot needs a SPEAKING event from Discord before it can map your audio — start speaking within a few seconds of joining

### Bot hears me but doesn't respond

- Verify STT key is set (`GROQ_API_KEY` or `VOICE_TOOLS_OPENAI_KEY`)
- Check the LLM model is configured and accessible
- Review gateway logs: `tail -f ~/.hermes/logs/gateway.log`

### Bot responds in text but not in voice channel

- TTS provider may be failing — check API key and quota
- Edge TTS (free, no key) is the default fallback
- Check logs for TTS errors

### Web UI not accessible from other devices on the network

The macOS firewall may block incoming connections. Allow the gateway through:

1. **System Settings** → **Network** → **Firewall** → **Options**
2. Add `/usr/local/bin/python3` (or your Python path) to the allowed list
3. Or temporarily disable the firewall for testing

On Linux, allow the port through `ufw`:

```bash
sudo ufw allow 8765/tcp
```

### Web UI microphone not working on mobile

Mobile browsers require **HTTPS** for microphone access (`navigator.mediaDevices` API). When accessing the Web UI over HTTP on a LAN IP (e.g. `http://192.168.1.x:8765`), the mic button will appear dimmed.

**Workarounds:**

**Android Chrome** — flag the LAN IP as secure:
1. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
2. Add your Web UI URL (e.g. `http://192.168.1.106:8765`)
3. Set to **Enabled** and relaunch Chrome

**iOS Safari / Chrome** — no flag bypass available. Use one of these instead:

1. **Self-signed HTTPS** with mkcert (recommended):
   ```bash
   # Install mkcert
   brew install mkcert
   mkcert -install

   # Generate cert for your LAN IP
   mkcert 192.168.1.106

   # Run a simple HTTPS reverse proxy (requires Node.js)
   npx local-ssl-proxy --source 8443 --target 8765 \
     --cert 192.168.1.106.pem --key 192.168.1.106-key.pem
   ```
   Then access `https://192.168.1.106:8443` on your iPhone. You'll need to trust the mkcert root CA on iOS: **Settings → General → About → Certificate Trust Settings**.

2. **Caddy reverse proxy** (auto-HTTPS for local networks):
   ```bash
   brew install caddy
   caddy reverse-proxy --from https://192.168.1.106:8443 --to http://127.0.0.1:8765
   ```

3. **SSH tunnel from mobile** (if you have an SSH client like Termius):
   ```bash
   ssh -L 8765:127.0.0.1:8765 user@your-mac-ip
   ```
   Then access `http://localhost:8765` on the mobile browser — localhost is exempt from HTTPS requirement.

:::tip
Text chat works on mobile over HTTP without any workaround — only the microphone feature requires HTTPS.
:::

### Whisper returns garbage text

The hallucination filter catches most cases automatically. If you're still getting phantom transcripts:

- Use a quieter environment
- Adjust `silence_threshold` in config (higher = less sensitive)
- Try a different STT model

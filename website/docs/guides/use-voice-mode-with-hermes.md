---
sidebar_position: 8
title: "Use Voice Mode with Hermes"
description: "A practical guide to setting up and using Hermes voice mode across CLI, Telegram, Discord, Discord voice channels, and voice server rooms"
---

# Use Voice Mode with Hermes

This guide is the practical companion to the [Voice Mode feature reference](/user-guide/features/voice-mode).

If the feature page explains what voice mode can do, this guide shows how to actually use it well.

:::tip
[Nous Portal](/integrations/nous-portal) bundles both the LLM and TTS through one OAuth — voice mode works end-to-end with no extra credentials.
:::

## What voice mode is good for

Voice mode is especially useful when:
- you want a hands-free CLI workflow
- you want spoken responses in Telegram or Discord
- you want Hermes sitting in a Discord voice channel for live conversation
- you want Hermes in a local voice-server/WebRTC room with browser debugging
- you want language practice, study help, or back-and-forth conversation while walking, cleaning, cooking, or doing other hands-busy work

## Choose your voice mode setup

There are four different voice experiences in Hermes.

| Mode | Best for | Platform |
|---|---|---|
| Interactive microphone loop | Personal hands-free use while coding or researching | CLI |
| Voice replies in chat | Spoken responses alongside normal messaging | Telegram, Discord |
| Live voice channel bot | Group or personal live conversation in a VC | Discord voice channels |
| Voice server room | Natural hands-free conversation, language practice, browser/WebRTC testing, and plugin-provided Discord/phone/WhatsApp transports | voice-server-compatible room |

A good path is:
1. get text working first
2. enable voice replies second
3. move to Discord voice channels or voice server rooms last if you want the full experience

## Step 1: make sure normal Hermes works first

Before touching voice mode, verify that:
- Hermes starts
- your provider is configured
- the agent can answer text prompts normally

```bash
hermes
```

Ask something simple:

```text
What tools do you have available?
```

If that is not solid yet, fix text mode first.

## Step 2: install the right extras

### CLI microphone + playback

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[voice]"
```

### Messaging platforms

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[messaging]"
```

### Premium ElevenLabs TTS

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[tts-premium]"
```

### Local NeuTTS (optional)

```bash
python -m pip install -U neutts[all]
```

### Everything

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[all]"
```

## Step 3: install system dependencies

### macOS

```bash
brew install portaudio ffmpeg opus
brew install espeak-ng
```

### Ubuntu / Debian

```bash
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng
```

Why these matter:
- `portaudio` → microphone input / playback for CLI voice mode
- `ffmpeg` → audio conversion for TTS and messaging delivery
- `opus` → Discord voice codec support
- `espeak-ng` → phonemizer backend for NeuTTS

## Step 4: choose STT and TTS providers

Hermes supports both local and cloud speech stacks.

### Easiest / cheapest setup

Use local STT and free Edge TTS:
- STT provider: `local`
- TTS provider: `edge`

This is usually the best place to start.

### Environment file example

Add to `~/.hermes/.env`:

```bash
# Cloud STT options (local needs no key)
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# Premium TTS (optional)
ELEVENLABS_API_KEY=***
```

### Provider recommendations

#### Speech-to-text

- `local` → best default for privacy and zero-cost use
- `groq` → very fast cloud transcription
- `openai` → good paid fallback

#### Text-to-speech

- `edge` → free and good enough for most users
- `neutts` → free local/on-device TTS
- `elevenlabs` → best quality
- `openai` → good middle ground
- `mistral` → multilingual, native Opus

### If you use `hermes setup`

If you choose NeuTTS in the setup wizard, Hermes checks whether `neutts` is already installed. If it is missing, the wizard tells you NeuTTS needs the Python package `neutts` and the system package `espeak-ng`, offers to install them for you, installs `espeak-ng` with your platform package manager, and then runs:

```bash
python -m pip install -U neutts[all]
```

If you skip that install or it fails, the wizard falls back to Edge TTS.

## Step 5: recommended config

```yaml
voice:
  record_key: "ctrl+b"
  max_recording_seconds: 120
  auto_tts: false
  beep_enabled: true
  silence_threshold: 200
  silence_duration: 3.0

stt:
  provider: "local"
  local:
    model: "base"

tts:
  provider: "edge"
  edge:
    voice: "en-US-AriaNeural"
```

This is a good conservative default for most people.

If you want local TTS instead, switch the `tts` block to:

```yaml
tts:
  provider: "neutts"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

## Use case 1: CLI voice mode

## Turn it on

Start Hermes:

```bash
hermes
```

Inside the CLI:

```text
/voice on
```

### Recording flow

Default key:
- `Ctrl+B`

Workflow:
1. press `Ctrl+B`
2. speak
3. wait for silence detection to stop recording automatically
4. Hermes transcribes and responds
5. if TTS is on, it speaks the answer
6. the loop can automatically restart for continuous use

### Useful commands

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### Good CLI workflows

#### Walk-up debugging

Say:

```text
I keep getting a docker permission error. Help me debug it.
```

Then continue hands-free:
- "Read the last error again"
- "Explain the root cause in simpler terms"
- "Now give me the exact fix"

#### Research / brainstorming

Great for:
- walking around while thinking
- dictating half-formed ideas
- asking Hermes to structure your thoughts in real time

#### Accessibility / low-typing sessions

If typing is inconvenient, voice mode is one of the fastest ways to stay in the full Hermes loop.

## Tuning CLI behavior

### Silence threshold

If Hermes starts/stops too aggressively, tune:

```yaml
voice:
  silence_threshold: 250
```

Higher threshold = less sensitive.

### Silence duration

If you pause a lot between sentences, increase:

```yaml
voice:
  silence_duration: 4.0
```

### Record key

If `Ctrl+B` conflicts with your terminal or tmux habits:

```yaml
voice:
  record_key: "ctrl+space"
```

## Use case 2: voice replies in Telegram or Discord

This mode is simpler than full voice channels.

Hermes stays a normal chat bot, but can speak replies.

### Start the gateway

```bash
hermes gateway
```

### Turn on voice replies

Inside Telegram or Discord:

```text
/voice on
```

or

```text
/voice tts
```

### Modes

| Mode | Meaning |
|---|---|
| `off` | text only |
| `voice_only` | speak only when the user sent voice |
| `all` | speak every reply |

### When to use which mode

- `/voice on` if you want spoken replies only for voice-originating messages
- `/voice tts` if you want a full spoken assistant all the time

### Good messaging workflows

#### Telegram assistant on your phone

Use when:
- you are away from your machine
- you want to send voice notes and get quick spoken replies
- you want Hermes to function like a portable research or ops assistant

#### Discord DMs with spoken output

Useful when you want private interaction without server-channel mention behavior.

## Use case 3: Discord voice channels

This is the most advanced mode.

Hermes joins a Discord VC, listens to user speech, transcribes it, runs the normal agent pipeline, and speaks replies back into the channel.

## Required Discord permissions

In addition to the normal text-bot setup, make sure the bot has:
- Connect
- Speak
- preferably Use Voice Activity

Also enable privileged intents in the Developer Portal:
- Presence Intent
- Server Members Intent
- Message Content Intent

## Join and leave

In a Discord text channel where the bot is present:

```text
/voice join
/voice leave
/voice status
```

### What happens when joined

- users speak in the VC
- Hermes detects speech boundaries
- transcripts are posted in the associated text channel
- Hermes responds in text and audio
- the text channel is the one where `/voice join` was issued

### Best practices for Discord VC use

- keep `DISCORD_ALLOWED_USERS` tight
- use a dedicated bot/testing channel at first
- verify STT and TTS work in ordinary text-chat voice mode before trying VC mode

## Use case 4: Voice-server room

This is the room-runtime mode.

Hermes subscribes to a voice-server event socket, receives finalized user speech as gateway messages, runs the normal agent pipeline, and sends replies back to the room. A local voice-server plugin provides the room runtime and owns WebRTC, RNNoise, VAD, Smart Turn, STT/TTS, barge-in, playback, and browser debugging.

This extra runtime is needed because live voice is not just "record, transcribe, reply." The room has to decide speech boundaries with VAD, aggregate partial STT into one user turn, detect when the user is done, and handle assistant audio while the user interrupts. It is also the right place for token streaming: Hermes can produce assistant text for one turn, while the voice server buffers, speaks, interrupts, and reports what was actually spoken for that same turn.

## One-time plugin setup

Install and enable a voice-server plugin once:

```bash
hermes plugins install <voice-server-plugin-url> --enable
```

Restart Hermes so the plugin and gateway platform are loaded:

```bash
hermes gateway install
hermes gateway restart
hermes gateway status
```

Configure the plugin runtime in the plugin checkout:

```bash
# voice-server-plugin/.env
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...

# voice-server-plugin/.config
VOICE_SERVER_STT_PROVIDER=local
VOICE_SERVER_LOCAL_STT_MODEL=base
VOICE_SERVER_TTS_PROVIDER=kokoro
VOICE_SERVER_TRANSPORT=webrtc
```

Point Hermes gateway at the room:

```bash
VOICE_SERVER_ENABLED=true
VOICE_SERVER_ROOM_URL=ws://127.0.0.1:7860/events
VOICE_SERVER_ROOM_ID=default
VOICE_SERVER_ALLOWED_USERS=caller
```

## Daily use

After setup, the normal path is just opening the local room URL:

```text
http://127.0.0.1:7860/auto-client/
```

The page may auto-connect the browser audio transport. That only readies the
room; it does not create the first Hermes voice session. Press the room's **New
Call** button to send a structured `inbound_call` event over `/events` with a
caller object and call id. Hermes uses those room/source fields to create a fresh
`voice_server` session for the call while the room server, WebSocket, and
gateway keep running. Repeated button presses create separate fresh sessions.
This is not a reset, delete, arbitrary resume, transcript injection, or
free-text command flow.

Keep the gateway and room runtime running as services:

```bash
hermes gateway run --accept-hooks
```

You do not need a skill prompt for normal use. The plugin setup helpers are still useful for first install, updates, and debugging: use its install/update helper to install or repair the plugin, and use its local-room helper when you want Hermes to start a foreground room or print the room URL.

Allow microphone access, then say a short test sentence. In `~/.hermes/logs/gateway.log`, you should see:

```text
inbound message: platform=voice_server
response ready: platform=voice_server
```

### What happens when connected

- the browser connects to the voice-server room
- Hermes sends `start_bot`
- the user presses **New Call**
- the voice server emits structured `inbound_call` caller data
- Hermes creates a fresh session for that call id/source
- the voice server emits `transcript`
- Hermes replies through that call session
- The voice server speaks `assistant_reply` or streams `assistant_llm_*` deltas
- The voice server emits `assistant_spoken`

Hermes does not call TTS directly in this mode. All audio output goes through the voice server.

### Outbound call protocol

The first outbound call protocol is new-session only. Hermes can ask the
connected voice server to start a call by sending `start_outbound_call` with:

- `room_id`
- `call_id`
- `target`
- optional `context`
- optional `metadata`

`target` is opaque to Hermes and may be either a string or an object. Hermes
does not define a phone/contact schema, does not configure the provider
transport, and does not send a `session_id` or other resume/acquire field for
the call. When the voice server emits `call_started` with the call id, Hermes
creates a fresh `voice_server` session for that call. Later `transcript` events
with the same `call_id` route into that fresh session; different call ids route
to separate sessions.

### Best practices for voice-server use

- keep `VOICE_SERVER_ALLOWED_USERS` tight
- verify the room reports `bot_started`, `transport_connected`, and `pipeline_started`
- use the local browser room first before testing phone, WhatsApp, Discord, or other transports
- keep STT/TTS provider settings in the plugin `.config`, not in Hermes core config

Interrupted speech is reconciled by turn id. If The voice server reports only partial speech, Hermes keeps only the spoken text in history; if nothing was spoken, Hermes removes that assistant turn.

## Voice quality recommendations

### Best quality setup

- STT: local `large-v3` or Groq `whisper-large-v3`
- TTS: ElevenLabs

### Best speed / convenience setup

- STT: local `base` or Groq
- TTS: Edge

### Best zero-cost setup

- STT: local
- TTS: Edge

## Common failure modes

### "No audio device found"

Install `portaudio`.

### "Bot joins but hears nothing"

Check:
- your Discord user ID is in `DISCORD_ALLOWED_USERS`
- you are not muted
- privileged intents are enabled
- the bot has Connect/Speak permissions

### "It transcribes but does not speak"

Check:
- TTS provider config
- API key / quota for ElevenLabs or OpenAI
- `ffmpeg` install for Edge conversion paths

### "Whisper outputs garbage"

Try:
- quieter environment
- higher `silence_threshold`
- different STT provider/model
- shorter, clearer utterances

### "It works in DMs but not in server channels"

That is often mention policy.

By default, the bot needs an `@mention` in Discord server text channels unless configured otherwise.

## Suggested first-week setup

If you want the shortest path to success:

1. get text Hermes working
2. install `hermes-agent[voice]`
3. use CLI voice mode with local STT + Edge TTS
4. then enable `/voice on` in Telegram or Discord
5. only after that, try Discord VC mode

That progression keeps the debugging surface small.

## Where to read next

- [Voice Mode feature reference](/user-guide/features/voice-mode)
- [Messaging Gateway](/user-guide/messaging)
- [Discord setup](/user-guide/messaging/discord)
- [Telegram setup](/user-guide/messaging/telegram)
- [Configuration](/user-guide/configuration)

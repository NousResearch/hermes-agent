---
sidebar_position: 7
title: "Use Voice Mode with Hermes"
description: "A practical guide to setting up and using Hermes voice mode across CLI, Telegram, Discord, and Discord voice channels"
---

# Use Voice Mode with Hermes

This guide is the practical companion to the [Voice Mode feature reference](/docs/user-guide/features/voice-mode).

If the feature page explains what voice mode can do, this guide shows how to actually use it well.

## What voice mode is good for

Voice mode is especially useful when:
- you want a hands-free CLI workflow
- you want spoken responses in Telegram or Discord
- you want Hermes sitting in a Discord voice channel for live conversation
- you want quick idea capture, debugging, or back-and-forth while walking around instead of typing

## Choose your voice mode setup

There are really three different voice experiences in Hermes.

| Mode | Best for | Platform |
|---|---|---|
| Interactive microphone loop | Personal hands-free use while coding or researching | CLI |
| Voice replies in chat | Spoken responses alongside normal messaging | Telegram, Discord |
| Live voice channel bot | Group or personal live conversation in a VC | Discord voice channels |

A good path is:
1. get text working first
2. enable voice replies second
3. move to Discord voice channels last if you want the full experience

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
pip install hermes-agent[voice]
```

### Messaging platforms

```bash
pip install hermes-agent[messaging]
```

### Premium ElevenLabs TTS

```bash
pip install hermes-agent[tts-premium]
```

### Local NeuTTS (optional)

```bash
python -m pip install -U neutts[all]
```

### Everything

```bash
pip install hermes-agent[all]
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

- [Voice Mode feature reference](/docs/user-guide/features/voice-mode)
- [Messaging Gateway](/docs/user-guide/messaging)
- [Discord setup](/docs/user-guide/messaging/discord)
- [Telegram setup](/docs/user-guide/messaging/telegram)
- [Configuration](/docs/user-guide/configuration)

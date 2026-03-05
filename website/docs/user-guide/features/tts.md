---
sidebar_position: 9
title: "Voice & TTS"
description: "Text-to-speech and voice message transcription across all platforms"
---

# Voice & TTS

Hermes Agent supports both text-to-speech output and voice message transcription across all messaging platforms.

## Text-to-Speech

Convert text to speech with three providers:

| Provider | Quality | Cost | API Key |
|----------|---------|------|---------|
| **Edge TTS** (default) | Good | Free | None needed |
| **ElevenLabs** | Excellent | Paid | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Good | Paid | `VOICE_TOOLS_OPENAI_KEY` |

### Platform Delivery

| Platform | Delivery | Format |
|----------|----------|--------|
| Telegram | Voice bubble (plays inline) | Opus `.ogg` |
| Discord | Audio file attachment | MP3 |
| WhatsApp | Audio file attachment | MP3 |
| CLI | Saved to `~/voice-memos/` | MP3 |

### Configuration

```yaml
# In ~/.hermes/config.yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai"
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
```

### Telegram Voice Bubbles & ffmpeg

Telegram voice bubbles require Opus/OGG audio format:

- **OpenAI and ElevenLabs** produce Opus natively — no extra setup
- **Edge TTS** (default) outputs MP3 and needs **ffmpeg** to convert:

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

Without ffmpeg, Edge TTS audio is sent as a regular audio file (playable, but shows as a rectangular player instead of a voice bubble).

:::tip
If you want voice bubbles without installing ffmpeg, switch to the OpenAI or ElevenLabs provider.
:::

## Voice Message Transcription

Voice messages sent on Telegram, Discord, WhatsApp, or Slack are automatically transcribed and injected as text into the conversation. The agent sees the transcript as normal text.

| Provider | Model | Quality | Cost |
|----------|-------|---------|------|
| **OpenAI Whisper** | `whisper-1` (default) | Good | Low |
| **OpenAI GPT-4o** | `gpt-4o-mini-transcribe` | Better | Medium |
| **OpenAI GPT-4o** | `gpt-4o-transcribe` | Best | Higher |

Requires `VOICE_TOOLS_OPENAI_KEY` in `~/.hermes/.env`.

### Configuration

```yaml
# In ~/.hermes/config.yaml
stt:
  enabled: true
  model: "whisper-1"
```

---
sidebar_position: 9
title: "Voice & TTS"
description: "Text-to-speech and voice message transcription across all platforms"
---

# Voice & TTS

Hermes Agent supports both text-to-speech output and voice message transcription across all messaging platforms.

:::tip Nous Subscribers
If you have a paid [Nous Portal](https://portal.nousresearch.com) subscription, OpenAI TTS is available through the **[Tool Gateway](tool-gateway.md)** without a separate OpenAI API key. Run `hermes model` or `hermes tools` to enable it.
:::

## Text-to-Speech

Convert text to speech with ten providers:

| Provider | Quality | Cost | API Key |
|----------|---------|------|---------|
| **Edge TTS** (default) | Good | Free | None needed |
| **ElevenLabs** | Excellent | Paid | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | Good | Paid | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | Excellent | Paid | `MINIMAX_API_KEY` |
| **Mistral (Voxtral TTS)** | Excellent | Paid | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | Excellent | Free tier | `GEMINI_API_KEY` |
| **xAI TTS** | Excellent | Paid | `XAI_API_KEY` |
| **NeuTTS** | Good | Free (local) | None needed |
| **KittenTTS** | Good | Free (local) | None needed |
| **Piper** | Good | Free (local) | None needed |

### Platform Delivery

| Platform | Delivery | Format |
|----------|----------|--------|
| Telegram | Voice bubble (plays inline) | Opus `.ogg` |
| Discord | Voice bubble (Opus/OGG), falls back to file attachment | Opus/MP3 |
| WhatsApp | Audio file attachment | MP3 |
| CLI | Saved to `~/.hermes/audio_cache/` | MP3 |

### Configuration

```yaml
# In ~/.hermes/config.yaml
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper"
  speed: 1.0                    # Global speed multiplier (provider-specific settings override this)
  edge:
    voice: "en-US-AriaNeural"   # 322 voices, 74 languages
    speed: 1.0                  # Converted to rate percentage (+/-%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # Override for OpenAI-compatible TTS endpoints
    speed: 1.0                  # 0.25 - 4.0
  minimax:
    model: "speech-2.8-hd"     # speech-2.8-hd (default), speech-2.8-turbo
    voice_id: "English_Graceful_Lady"  # See https://platform.minimax.io/faq/system-voice-id
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8"  # Paul - Neutral (default)
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # or gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 prebuilt voices: Zephyr, Puck, Kore, Enceladus, Gacrux, etc.
  xai:
    voice_id: "eve"             # or a custom voice ID — see docs below
    language: "en"              # ISO 639-1 code
    sample_rate: 24000          # 22050 / 24000 (default) / 44100 / 48000
    bit_rate: 128000            # MP3 bitrate; only applies when codec=mp3
    # base_url: "https://api.x.ai/v1"   # Override via XAI_BASE_URL env var
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8; also: kitten-tts-micro-0.8 (41MB), kitten-tts-mini-0.8 (80MB)
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # Expand numbers, currencies, units
  piper:
    voice: en_US-lessac-medium                  # voice name (auto-downloaded) OR absolute path to .onnx
    # voices_dir: ''                            # default: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # requires onnxruntime-gpu
    # length_scale: 1.0                         # 2.0 = twice as slow
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = half as loud
    # normalize_audio: true
```

**Speed control**: The global `tts.speed` value applies to all providers by default. Each provider can override it with its own `speed` setting (e.g., `tts.openai.speed: 1.5`). Provider-specific speed takes precedence over the global value. Default is `1.0` (normal speed).


### Input length limits

Each provider has a documented per-request input-character cap. Hermes truncates text before calling the provider so requests never fail with a length error:

| Provider | Default cap (chars) |
|----------|---------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 5000 |
| ElevenLabs | Model-aware (see below) |
| NeuTTS | 2000 |
| KittenTTS | 2000 |

**ElevenLabs** picks a cap from the configured `model_id`:

| `model_id` | Cap (chars) |
|------------|-------------|
| `eleven_flash_v2_5` | 40000 |
| `eleven_flash_v2` | 30000 |
| `eleven_multilingual_v2` (default), `eleven_multilingual_v1`, `eleven_english_sts_v2`, `eleven_english_sts_v1` | 10000 |
| `eleven_v3`, `eleven_ttv_v3` | 5000 |
| Unknown model | Falls back to provider default (10000) |

**Override per provider** with `max_text_length:` under the provider section of your TTS config:

```yaml
tts:
  openai:
    max_text_length: 8192   # raise or lower the provider cap
```

Only positive integers are honored. Zero, negative, non-numeric, or boolean values fall through to the provider default, so a broken config can't accidentally disable truncation.

### Telegram Voice Bubbles & ffmpeg

Telegram voice bubbles require Opus/OGG audio format:

- **OpenAI, ElevenLabs, and Mistral** produce Opus natively — no extra setup
- **Edge TTS** (default) outputs MP3 and needs **ffmpeg** to convert:
- **MiniMax TTS** outputs MP3 and needs **ffmpeg** to convert for Telegram voice bubbles
- **Google Gemini TTS** outputs raw PCM and uses **ffmpeg** to encode Opus directly for Telegram voice bubbles
- **xAI TTS** outputs MP3 and needs **ffmpeg** to convert for Telegram voice bubbles
- **NeuTTS** outputs WAV and also needs **ffmpeg** to convert for Telegram voice bubbles
- **KittenTTS** outputs WAV and also needs **ffmpeg** to convert for Telegram voice bubbles
- **Piper** outputs WAV and also needs **ffmpeg** to convert for Telegram voice bubbles

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

Without ffmpeg, Edge TTS, MiniMax TTS, NeuTTS, KittenTTS, and Piper audio are sent as regular audio files (playable, but shown as a rectangular player instead of a voice bubble).

:::tip
If you want voice bubbles without installing ffmpeg, switch to the OpenAI, ElevenLabs, or Mistral provider.
:::

### xAI Custom Voices (voice cloning)

xAI supports cloning your voice and using it with TTS. Create a custom voice in the [xAI Console](https://console.x.ai/team/default/voice/voice-library), then set the resulting `voice_id` in your config:

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # your custom voice ID
```

See the [xAI Custom Voices docs](https://docs.x.ai/developers/model-capabilities/audio/custom-voices) for details on recording, supported formats, and limits.

### Piper (local, 44 languages)

Piper is a fast, local neural TTS engine from the Open Home Foundation (the Home Assistant maintainers). It runs entirely on CPU, supports **44 languages** with pre-trained voices, and needs no API key.

**Install via `hermes tools`** → Voice & TTS → Piper — Hermes runs `pip install piper-tts` for you. Or install manually: `pip install piper-tts`.

**Switch to Piper:**

```yaml
tts:
  provider: piper
  piper:
    voice: en_US-lessac-medium
```

On the first TTS call for a voice that isn't cached locally, Hermes runs `python -m piper.download_voices <name>` and downloads the model (~20-90MB depending on quality tier) into `~/.hermes/cache/piper-voices/`. Subsequent calls reuse the cached model.

**Picking a voice.** The [full voice catalog](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md) covers English, Spanish, French, German, Italian, Dutch, Portuguese, Russian, Polish, Turkish, Chinese, Arabic, Hindi, and more — each with `x_low` / `low` / `medium` / `high` quality tiers. Sample voices at [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/).

**Using a pre-downloaded voice.** Set `tts.piper.voice` to an absolute path ending in `.onnx`:

```yaml
tts:
  piper:
    voice: /path/to/my-custom-voice.onnx
```

**Advanced knobs** (`tts.piper.length_scale` / `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio`, `use_cuda`) correspond 1:1 to Piper's `SynthesisConfig`. They're ignored on older `piper-tts` versions.

### Custom command providers

If a TTS engine you want isn't natively supported (VoxCPM, MLX-Kokoro, XTTS CLI, a voice-cloning script, anything else that exposes a CLI), you can wire it in as a **command-type provider** without writing any Python. Hermes writes the input text to a temp UTF-8 file, runs your shell command, and reads the audio file the command produced.

Declare one or more providers under `tts.providers.<name>` and switch between them with `tts.provider: <name>` — the same way you switch between built-ins like `edge` and `openai`.

```yaml
tts:
  provider: voxcpm                 # pick any name under tts.providers
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # try to deliver as a Telegram voice bubble

    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav

    piper-custom:                  # native Piper also supports custom .onnx via tts.piper.voice
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

#### Example: Doubao (Chinese seed-tts-2.0)

For high-quality Chinese TTS via ByteDance's [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) bidirectional-streaming API, install the [`doubao-speech`](https://pypi.org/project/doubao-speech/) PyPI package and wire it in as a command provider:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
```

```yaml
tts:
  provider: doubao
  providers:
    doubao:
      type: command
      command: "doubao-speech say --text-file {input_path} --out {output_path}"
      output_format: mp3
      max_text_length: 1024
      timeout: 30
```

Credentials come from your shell environment (`VOLCENGINE_APP_ID` / `VOLCENGINE_ACCESS_TOKEN`) or `~/.doubao-speech/config.yaml`. Pick a voice by adding `--voice zh-female-warm` (or any other alias from `doubao-speech list-voices`) to the command. `doubao-speech` also bundles streaming ASR — see the [STT section below](#example-doubao--volcengine-asr) for Hermes integration. Source and full docs: [github.com/Hypnus-Yuan/doubao-speech](https://github.com/Hypnus-Yuan/doubao-speech).

#### Placeholders

Your command template can reference these placeholders. Hermes substitutes them at render time and shell-quotes each value for the surrounding context (bare / single-quoted / double-quoted), so paths with spaces and other shell-sensitive characters are safe.

| Placeholder      | Meaning                                              |
|------------------|------------------------------------------------------|
| `{input_path}`   | Path to the temp UTF-8 text file Hermes wrote        |
| `{text_path}`    | Alias for `{input_path}`                             |
| `{output_path}`  | Path the command must write audio to                 |
| `{format}`       | `mp3` / `wav` / `ogg` / `flac`                       |
| `{voice}`        | `tts.providers.<name>.voice`, empty when unset       |
| `{model}`        | `tts.providers.<name>.model`                         |
| `{speed}`        | Resolved speed multiplier (provider or global)       |

Use `{{` and `}}` for literal braces.

#### Optional keys

| Key                | Default | Meaning                                                                                                    |
|--------------------|---------|------------------------------------------------------------------------------------------------------------|
| `timeout`          | `120`   | Seconds; the process tree is killed on expiry (Unix `killpg`, Windows `taskkill /T`).                       |
| `output_format`    | `mp3`   | One of `mp3` / `wav` / `ogg` / `flac`. Auto-inferred from the output extension if Hermes picks a path.      |
| `voice_compatible` | `false` | When `true`, Hermes converts MP3/WAV output to Opus/OGG via ffmpeg so Telegram renders a voice bubble.      |
| `max_text_length`  | `5000`  | Input is truncated to this length before rendering the command.                                             |
| `voice` / `model`  | empty   | Passed to the command as placeholder values only.                                                           |

#### Behavior notes

- **Built-in names always win.** A `tts.providers.openai` entry never shadows the native OpenAI provider, so no user config can silently replace a built-in.
- **Default delivery is a document.** Command providers deliver as regular audio attachments on every platform. Opt in to voice-bubble delivery per-provider with `voice_compatible: true`.
- **Command failures surface to the agent.** Non-zero exit, empty output, or timeout all return an error with the command's stderr/stdout included so you can debug the provider from the conversation.
- **`type: command` is the default when `command:` is set.** Writing `type: command` explicitly is good practice but not required; an entry with a non-empty `command` string is treated as a command provider.
- **`{input_path}` / `{text_path}` are interchangeable.** Use whichever reads better in your command.

#### Security

Command-type providers run whatever shell command you configure, with your user's permissions. Hermes quotes placeholder values and enforces the configured timeout, but the command template itself is trusted local input — treat it the same way you would a shell script on your PATH.

## Voice Message Transcription (STT)

Voice messages sent on Telegram, Discord, WhatsApp, Slack, or Signal are automatically transcribed and injected as text into the conversation. The agent sees the transcript as normal text.

| Provider | Quality | Cost | API Key |
|----------|---------|------|---------| 
| **Local Whisper** (default) | Good | Free | None needed |
| **Groq Whisper API** | Good–Best | Free tier | `GROQ_API_KEY` |
| **OpenAI Whisper API** | Good–Best | Paid | `VOICE_TOOLS_OPENAI_KEY` or `OPENAI_API_KEY` |

:::info Zero Config
Local transcription works out of the box when `faster-whisper` is installed. If that's unavailable, Hermes can also use a local `whisper` CLI from common install locations (like `/opt/homebrew/bin`) or a custom command via `HERMES_LOCAL_STT_COMMAND`.
:::

### Configuration

```yaml
# In ~/.hermes/config.yaml
stt:
  provider: "local"           # "local" | "groq" | "openai" | "mistral" | "xai"
  local:
    model: "base"             # tiny, base, small, medium, large-v3
  openai:
    model: "whisper-1"        # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
  mistral:
    model: "voxtral-mini-latest"  # voxtral-mini-latest, voxtral-mini-2602
  xai:
    model: "grok-stt"         # xAI Grok STT
```

### Provider Details

**Local (faster-whisper)** — Runs Whisper locally via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Uses CPU by default, GPU if available. Model sizes:

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `tiny` | ~75 MB | Fastest | Basic |
| `base` | ~150 MB | Fast | Good (default) |
| `small` | ~500 MB | Medium | Better |
| `medium` | ~1.5 GB | Slower | Great |
| `large-v3` | ~3 GB | Slowest | Best |

**Groq API** — Requires `GROQ_API_KEY`. Good cloud fallback when you want a free hosted STT option.

**OpenAI API** — Accepts `VOICE_TOOLS_OPENAI_KEY` first and falls back to `OPENAI_API_KEY`. Supports `whisper-1`, `gpt-4o-mini-transcribe`, and `gpt-4o-transcribe`.

**Mistral API (Voxtral Transcribe)** — Requires `MISTRAL_API_KEY`. Uses Mistral's [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) models. Supports 13 languages, speaker diarization, and word-level timestamps. Install with `pip install hermes-agent[mistral]`.

**xAI Grok STT** — Requires `XAI_API_KEY`. Posts to `https://api.x.ai/v1/stt` as multipart/form-data. Good choice if you're already using xAI for chat or TTS and want one API key for everything. Auto-detection order puts it after Groq — explicitly set `stt.provider: xai` to force it.

**Custom local CLI fallback** — Set `HERMES_LOCAL_STT_COMMAND` if you want Hermes to call a local transcription command directly. The command template supports `{input_path}`, `{output_dir}`, `{language}`, and `{model}` placeholders. Your command must write a `.txt` transcript somewhere under `{output_dir}`.

#### Example: Doubao / Volcengine ASR

If you use [`doubao-speech`](https://pypi.org/project/doubao-speech/) for Doubao TTS (see [above](#example-doubao-chinese-seed-tts-20)), the same package handles speech-to-text via the local-command STT surface:

```bash
pip install doubao-speech
export VOLCENGINE_APP_ID="your-app-id"
export VOLCENGINE_ACCESS_TOKEN="your-access-token"
export HERMES_LOCAL_STT_COMMAND='doubao-speech transcribe {input_path} --out {output_dir}/transcript.txt'
```

```yaml
stt:
  provider: local_command
```

Hermes writes the incoming voice message to `{input_path}`, runs the command, and reads the `.txt` file produced under `{output_dir}`. Language is auto-detected by the Volcengine bigmodel endpoint.

### Fallback Behavior

If your configured provider isn't available, Hermes automatically falls back:
- **Local faster-whisper unavailable** → Tries a local `whisper` CLI or `HERMES_LOCAL_STT_COMMAND` before cloud providers
- **Groq key not set** → Falls back to local transcription, then OpenAI
- **OpenAI key not set** → Falls back to local transcription, then Groq
- **Mistral key/SDK not set** → Skipped in auto-detect; falls through to next available provider
- **Nothing available** → Voice messages pass through with an accurate note to the user

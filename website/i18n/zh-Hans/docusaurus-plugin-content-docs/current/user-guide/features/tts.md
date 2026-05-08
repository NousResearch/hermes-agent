---
sidebar_position: 9
title: "语音与 TTS"
description: "跨所有平台的文本转语音和语音消息转录"
---

# 语音与 TTS

Hermes Agent 支持文本转语音输出和跨所有消息平台的语音消息转录。

:::tip Nous 订阅用户
如果你有付费的 [Nous Portal](https://portal.nousresearch.com) 订阅，OpenAI TTS 可通过**[工具网关](tool-gateway.md)**获得，无需单独的 OpenAI API 密钥。运行 `hermes model` 或 `hermes tools` 以启用它。
:::

## 文本转语音

使用十个提供者将文本转换为语音：

| 提供者 | 质量 | 成本 | API 密钥 |
|----------|---------|------|---------|
| **Edge TTS**（默认） | 良好 | 免费 | 无需 |
| **ElevenLabs** | 优秀 | 付费 | `ELEVENLABS_API_KEY` |
| **OpenAI TTS** | 良好 | 付费 | `VOICE_TOOLS_OPENAI_KEY` |
| **MiniMax TTS** | 优秀 | 付费 | `MINIMAX_API_KEY` |
| **Mistral (Voxtral TTS)** | 优秀 | 付费 | `MISTRAL_API_KEY` |
| **Google Gemini TTS** | 优秀 | 免费层级 | `GEMINI_API_KEY` |
| **xAI TTS** | 优秀 | 付费 | `XAI_API_KEY` |
| **NeuTTS** | 良好 | 免费（本地） | 无需 |
| **KittenTTS** | 良好 | 免费（本地） | 无需 |
| **Piper** | 良好 | 免费（本地） | 无需 |

### 平台传递

| 平台 | 传递方式 | 格式 |
|----------|----------|--------|
| Telegram | 语音气泡（内联播放） | Opus `.ogg` |
| Discord | 语音气泡 (Opus/OGG)，回退到文件附件 | Opus/MP3 |
| WhatsApp | 音频文件附件 | MP3 |
| CLI | 保存到 `~/.hermes/audio_cache/` | MP3 |

### 配置

```yaml
# 在 ~/.hermes/config.yaml 中
tts:
  provider: "edge"              # "edge" | "elevenlabs" | "openai" | "minimax" | "mistral" | "gemini" | "xai" | "neutts" | "kittentts" | "piper"
  speed: 1.0                    # 全局速度乘数（提供者特定设置会覆盖此值）
  edge:
    voice: "en-US-AriaNeural"   # 322 种声音，74 种语言
    speed: 1.0                  # 转换为速率百分比 (+/-%)
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"  # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"              # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # 覆盖 OpenAI 兼容的 TTS 端点
    speed: 1.0                  # 0.25 - 4.0
  minimax:
    model: "speech-2.8-hd"     # speech-2.8-hd (默认), speech-2.8-turbo
    voice_id: "English_Graceful_Lady"  # 参见 https://platform.minimax.io/faq/system-voice-id
    speed: 1                    # 0.5 - 2.0
    vol: 1                      # 0 - 10
    pitch: 0                    # -12 - 12
  mistral:
    model: "voxtral-mini-tts-2603"
    voice_id: "c69964a6-ab8b-4f8a-9465-ec0925096ec8" # Paul - Neutral (默认)
  gemini:
    model: "gemini-2.5-flash-preview-tts"  # or gemini-2.5-pro-preview-tts
    voice: "Kore"               # 30 种预置声音: Zephyr, Puck, Kore, Enceladus, Gacrux, etc.
  xai:
    voice_id: "eve"             # 或自定义声音 ID — 见下方文档
    language: "en"              # ISO 639-1 代码
    sample_rate: 24000          # 22050 / 24000 (默认) / 44100 / 48000
    bit_rate: 128000            # MP3 比特率；仅当 codec=mp3 时适用
    # base_url: "https://api.x.ai/v1"   # 通过 XAI_BASE_URL 环境变量覆盖
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
  kittentts:
    model: KittenML/kitten-tts-nano-0.8-int8   # 25MB int8；也: kitten-tts-micro-0.8 (41MB), kitten-tts-mini-0.8 (80MB)
    voice: Jasper                               # Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo
    speed: 1.0                                  # 0.5 - 2.0
    clean_text: true                            # 扩展数字、货币、单位
  piper:
    voice: en_US-lessac-medium                  # 声音名称（自动下载）或 .onnx 的绝对路径
    # voices_dir: ''                            # 默认: ~/.hermes/cache/piper-voices/
    # use_cuda: false                           # 需要 onnxruntime-gpu
    # length_scale: 1.0                         # 2.0 = 慢两倍
    # noise_scale: 0.667
    # noise_w_scale: 0.8
    # volume: 1.0                               # 0.5 = 音量减半
    # normalize_audio: true
```

**速度控制**：全局 `tts.speed` 值默认适用于所有提供者。每个提供者可以使用自己的 `speed` 设置（`tts.openai.speed: 1.5`）覆盖它。提供者特定速度优先于全局值。默认为 `1.0`（正常速度）。


### 输入长度限制

每个提供者都有文档化的每次请求输入字符上限。Hermes 在调用提供者之前截断文本，因此请求永远不会因长度错误而失败：

| 提供者 | 默认上限（字符） |
|----------|---------------------|
| Edge TTS | 5000 |
| OpenAI | 4096 |
| xAI | 15000 |
| MiniMax | 10000 |
| Mistral | 4000 |
| Google Gemini | 5000 |
| ElevenLabs | 模型感知（见下方） |
| NeuTTS | 2000 |
| KittenTTS | 2000 |

**ElevenLabs** 根据配置的 `model_id` 选择上限：

| `model_id` | 上限（字符） |
|------------|-------------|
| `eleven_flash_v2_5` | 40000 |
| `eleven_flash_v2` | 30000 |
| `eleven_multilingual_v2`（默认）、`eleven_multilingual_v1`、`eleven_english_sts_v2`、`eleven_english_sts_v1` | 10000 |
| `eleven_v3`、`eleven_ttv_v3` | 5000 |
| 未知模型 | 回退到提供者默认（10000） |

**按提供者覆盖** 在 TTS 配置的提供者部分下使用 `max_text_length:`：

```yaml
tts:
  openai:
    max_text_length: 8192   # 提高或降低提供者上限
```

只接受正整数。零、负数、非数字或布尔值会穿透到提供者默认值，因此损坏的配置不会意外禁用截断。

### Telegram 语音气泡与 ffmpeg

Telegram 语音气泡需要 Opus/OGG 音频格式：

- **OpenAI、ElevenLabs 和 Mistral** 原生产生 Opus — 无需额外设置
- **Edge TTS**（默认）输出 MP3，需要 **ffmpeg** 来转换：
- **MiniMax TTS** 输出 MP3，需要 **ffmpeg** 转换为 Telegram 语音气泡
- **Google Gemini TTS** 输出原始 PCM，使用 **ffmpeg** 直接编码为 Opus 以供 Telegram 语音气泡使用
- **xAI TTS** 输出 MP3，需要 **ffmpeg** 转换为 Telegram 语音气泡
- **NeuTTS** 输出 WAV，也需要 **ffmpeg** 转换为 Telegram 语音气泡
- **KittenTTS** 输出 WAV，也需要 **ffmpeg** 转换为 Telegram 语音气泡
- **Piper** 输出 WAV，也需要 **ffmpeg** 转换为 Telegram 语音气泡

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

如果没有 ffmpeg，Edge TTS、MiniMax TTS、NeuTTS、KittenTTS 和 Piper 音频会作为常规音频文件发送（可播放，但显示为矩形播放器而不是语音气泡）。

:::tip
如果你想在不安装 ffmpeg 的情况下使用语音气泡，请切换到 OpenAI、ElevenLabs 或 Mistral 提供者。
:::

### xAI 自定义声音（语音克隆）

xAI 支持克隆你的声音并将其与 TTS 一起使用。在 [xAI 控制台](https://console.x.ai/team/default/voice/voice-library) 中创建自定义声音，然后在配置中设置生成的 `voice_id`：

```yaml
tts:
  provider: xai
  xai:
    voice_id: "nlbqfwie"   # 你的自定义声音 ID
```

有关录制、支持的格式和限制的详细信息，请参阅 [xAI 自定义声音文档](https://docs.x.ai/developers/model-capabilities/audio/custom-voices)。

### Piper（本地，44 种语言）

Piper 是由 Open Home Foundation（Home Assistant 维护者）开发的快速本地神经 TTS 引擎。它完全在 CPU 上运行，支持**44 种语言**的预训练声音，且不需要 API 密钥。

**通过 `hermes tools` 安装** → 语音与 TTS → Piper — Hermes 为你运行 `pip install piper-tts`。或手动安装：`pip install piper-tts`。

**切换到 Piper：**

```yaml
tts:
  provider: piper
  piper:
    voice: en_US-lessac-medium
```

在首次为未本地缓存的声音调用 TTS 时，Hermes 运行 `python -m piper.download_voices <name>` 并将模型（约 20-90MB，取决于质量等级）下载到 `~/.hermes/cache/piper-voices/`。后续调用会重用缓存的模型。

**选择声音。** [完整声音目录](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md) 涵盖英语、西班牙语、法语、德语、意大利语、荷兰语、葡萄牙语、俄语、波兰语、土耳其语、中文、阿拉伯语、印地语等 — 每种都有 `x_low` / `low` / `medium` / `high` 质量等级。在 [rhasspy.github.io/piper-samples](https://rhasspy.github.io/piper-samples/) 试听示例声音。

**使用预下载的声音。** 将 `tts.piper.voice` 设置为以 `.onnx` 结尾的绝对路径：

```yaml
tts:
  piper:
    voice: /path/to/my-custom-voice.onnx
```

**高级旋钮**（`tts.piper.length_scale` / `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio`、`use_cuda`）与 Piper 的 `SynthesisConfig` 一一对应。在较旧的 `piper-tts` 版本上会被忽略。

### 自定义命令提供者

如果你想要一个 TTS 引擎但未原生支持（VoxCPM、MLX-Kokoro、XTTS CLI、语音克隆脚本，或任何暴露 CLI 的引擎），你可以将其连接为**命令类型提供者**，而无需编写任何 Python。Hermes 将输入文本写入临时 UTF-8 文件，运行你的 shell 命令，并读取命令生成的音频文件。

在 `tts.providers.<name>` 下声明一个或多个提供者，并使用 `tts.provider: <name>` 在它们之间切换 — 与你切换内置提供者（如 `edge` 和 `openai`）的方式相同。

```yaml
tts:
  provider: voxcpm                 # 选择 tts.providers 下的任何名称
  providers:
    voxcpm:
      type: command
      command: "voxcpm --ref ~/voice.wav --text-file {input_path} --out {output_path}"
      output_format: mp3
      timeout: 180
      voice_compatible: true       # 尝试作为 Telegram 语音气泡传递
    mlx-kokoro:
      type: command
      command: "python -m mlx_kokoro --in {input_path} --out {output_path} --voice {voice}"
      voice: af_sky
      output_format: wav
    piper-custom:                  # 原生 Piper 也支持通过 tts.piper.voice 使用自定义 .onnx
      type: command
      command: "piper -m /path/to/custom.onnx -f {output_path} < {input_path}"
      output_format: wav
```

#### 示例：豆包（中文 seed-tts-2.0）

要通过 ByteDance 的 [seed-tts-2.0](https://www.volcengine.com/docs/6561/1257544) 双向流式 API 实现高质量中文 TTS，请安装 [`doubao-speech`](https://pypi.org/project/doubao-speech/) PyPI 包并将其作为命令提供者连接：

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

凭证来自你的 shell 环境（`VOLCENGINE_APP_ID` / `VOLCENGINE_ACCESS_TOKEN`）或 `~/.doubao-speech/config.yaml`。通过向命令添加 `--voice zh-female-warm`（或 `doubao-speech list-voices` 中的任何其他别名）来选择声音。`doubao-speech` 还捆绑了流式 ASR — 请参阅下方[STT 部分](#示例豆包--volcengine-asr)了解 Hermes 集成。源代码和完整文档：[github.com/Hypnus-Yuan/doubao-speech](https://github.com/Hypnus-Yuan/doubao-speech)。

#### 占位符

你的命令模板可以引用这些占位符。Hermes 在渲染时替换它们，并对每个值进行 shell 引号处理（裸 / 单引号 / 双引号），因此带有空格和其他 shell 敏感字符的路径是安全的。

| 占位符      | 含义                                              |
|------------------|------------------------------------------------------|
| `{input_path}`   | Hermes 写入的临时 UTF-8 文本文件路径        |
| `{text_path}`    | `{input_path}` 的别名                             |
| `{output_path}`  | 命令必须写入音频的路径                 |
| `{format}`       | `mp3` / `wav` / `ogg` / `flac`                       |
| `{voice}`        | `tts.providers.<name>.voice`，未设置时为空       |
| `{model}`        | `tts.providers.<name>.model`                         |
| `{speed}`        | 解析的速度乘数（提供者或全局）       |

对字面的花括号使用 `{{` 和 `}}`。

#### 可选键

| 键                | 默认 | 含义                                                                                                    |
|--------------------|---------|------------------------------------------------------------------------------------------------------------|
| `timeout`          | `120`   | 秒数；到期时进程树被终止（Unix `killpg`，Windows `taskkill /T`）。                       |
| `output_format`    | `mp3`   | `mp3` / `wav` / `ogg` / `flac` 之一。如果 Hermes 选择路径，会自动从输出扩展名推断。      |
| `voice_compatible` | `false` | 为 `true` 时，Hermes 通过 ffmpeg 将 MP3/WAV 输出转换为 Opus/OGG，以便 Telegram 渲染语音气泡。      |
| `max_text_length`  | `5000`  | 渲染命令前，输入会被截断到此长度。                                             |
| `voice` / `model`  | 空   | 仅作为占位符值传递。                                                           |

#### 行为说明

- **内置名称总是获胜。** `tts.providers.openai` 条目永远不会遮蔽原生的 OpenAI 提供者，因此任何用户配置都无法静默替换内置提供者。
- **默认传递为文档。** 命令提供者默认在所有平台上作为常规音频附件传递。通过每个提供者的 `voice_compatible: true` 选择加入语音气泡传递。
- **命令失败会暴露给代理。** 非零退出、空输出或超时时都会返回错误，并包含命令的 stderr/stdout，以便你从对话中调试提供者。
- **`type: command` 是设置 `command:` 时的默认值。** 显式编写 `type: command` 是好的做法，但不是必需的；带有非空 `command` 字符串的条目被视为命令提供者。
- **`{input_path}` / `{text_path}` 可互换。** 在你的命令中使用哪个可读性更好就用哪个。

#### 安全性

命令类型提供者运行你配置的任何 shell 命令，使用你的用户权限。Hermes 会对占位符值进行引号处理并强制执行配置的超时，但命令模板本身是受信任的本地输入 — 像对待 PATH 上的 shell 脚本一样对待它。

## 语音消息转录 (STT)

在 Telegram、Discord、WhatsApp、Slack 或 Signal 上发送的语音消息会自动转录并作为文本注入到对话中。代理将转录视为普通文本。

| 提供者 | 质量 | 成本 | API 密钥 |
|----------|---------|------|---------| 
| **本地 Whisper**（默认） | 良好 | 免费 | 无需 |
| **Groq Whisper API** | 良好–最佳 | 免费层级 | `GROQ_API_KEY` |
| **OpenAI Whisper API** | 良好–最佳 | 付费 | `VOICE_TOOLS_OPENAI_KEY` 或 `OPENAI_API_KEY` |

:::info 零配置
当安装了 `faster-whisper` 时，本地转录开箱即用。如果不可用，Hermes 还可以使用来自常见安装位置（如 `/opt/homebrew/bin`）的本地 `whisper` CLI，或通过 `HERMES_LOCAL_STT_COMMAND` 使用自定义命令。
:::

### 配置

```yaml
# 在 ~/.hermes/config.yaml 中
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

### 提供者详情

**本地 (faster-whisper)** — 通过 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 在本地运行 Whisper。默认使用 CPU，如果可用则使用 GPU。模型大小：

| 模型 | 大小 | 速度 | 质量 |
|-------|------|-------|---------|
| `tiny` | ~75 MB | 最快 | 基础 |
| `base` | ~150 MB | 快速 | 良好（默认） |
| `small` | ~500 MB | 中等 | 更好 |
| `medium` | ~1.5 GB | 较慢 | 很棒 |
| `large-v3` | ~3 GB | 最慢 | 最佳 |

**Groq API** — 需要 `GROQ_API_KEY`。当你想要免费的托管 STT 选项时，是很好的云后备方案。

**OpenAI API** — 优先接受 `VOICE_TOOLS_OPENAI_KEY`，回退到 `OPENAI_API_KEY`。支持 `whisper-1`、`gpt-4o-mini-transcribe` 和 `gpt-4o-transcribe`。

**Mistral API (Voxtral Transcribe)** — 需要 `MISTRAL_API_KEY`。使用 Mistral 的 [Voxtral Transcribe](https://docs.mistral.ai/capabilities/audio/speech_to_text/) 模型。支持 13 种语言、说话人区分和词级时间戳。使用 `pip install hermes-agent[mistral]` 安装。

**xAI Grok STT** — 需要 `XAI_API_KEY`。以 multipart/form-data 形式发布到 `https://api.x.ai/v1/stt`。如果你已经将 xAI 用于聊天或 TTS，并希望一个 API 密钥处理所有事情，这是很好的选择。自动检测顺序将其放在 Groq 之后 — 显式设置 `stt.provider: xai` 以强制使用它。

**自定义本地 CLI 后备** — 如果你希望 Hermes 直接调用本地转录命令，请设置 `HERMES_LOCAL_STT_COMMAND`。命令模板支持 `{input_path}`、`{output_dir}`、`{language}` 和 `{model}` 占位符。你的命令必须在 `{output_dir}` 下的某个位置写入 `.txt` 转录文件。

#### 示例：豆包 / Volcengine ASR

如果你使用 [`doubao-speech`](https://pypi.org/project/doubao-speech/) 进行豆包 TTS（见[上方](#示例豆包中文-seed-tts-20)），同一个包通过本地命令 STT 接口处理语音转文本：

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

Hermes 将传入的语音消息写入 `{input_path}`，运行命令，并读取 `{output_dir}` 下生成的 `.txt` 文件。语言由 Volcengine bigmodel 端点自动检测。

### 回退行为

如果你的配置提供者不可用，Hermes 会自动回退：
- **本地 faster-whisper 不可用** → 在云提供者之前尝试本地 `whisper` CLI 或 `HERMES_LOCAL_STT_COMMAND`
- **未设置 Groq 密钥** → 回退到本地转录，然后 OpenAI
- **未设置 OpenAI 密钥** → 回退到本地转录，然后 Groq
- **未设置 Mistral 密钥/SDK** → 在自动检测中跳过；穿透到下一个可用提供者
- **没有可用的东西** → 语音消息传递，并向用户附上准确的说明

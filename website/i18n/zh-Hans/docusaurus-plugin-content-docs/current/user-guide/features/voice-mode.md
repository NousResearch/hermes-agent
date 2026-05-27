---
sidebar_position: 10
title: "语音模式"
description: "与 Hermes Agent 进行实时语音对话 — CLI、Telegram、Discord（私信、文字频道和语音频道）"
---

# 语音模式

Hermes Agent 支持在 CLI 和消息平台上进行完整的语音交互。通过麦克风与 Agent 对话，听取语音回复，并在 Discord 语音频道中进行实时语音对话。

如需包含推荐配置和实际使用模式的实践指南，请参阅 [使用 Hermes 的语音模式](/guides/use-voice-mode-with-hermes)。

## 前提条件

使用语音功能前，请确保已完成以下准备：

1. **已安装 Hermes Agent** — `pip install hermes-agent`（参见 [安装](/getting-started/installation)）
2. **已配置 LLM 提供商** — 运行 `hermes model` 或在 `~/.hermes/.env` 中设置首选提供商的凭据
3. **基础设置正常** — 运行 `hermes` 验证 Agent 能够响应文字消息，再启用语音功能

:::tip
`~/.hermes/` 目录和默认的 `config.yaml` 会在首次运行 `hermes` 时自动创建。只需手动创建 `~/.hermes/.env` 来存放 API 密钥。
:::

:::tip Nous Portal 同时覆盖两项
付费的 [Nous Portal](/user-guide/features/tool-gateway) 订阅通过 Tool Gateway 同时提供 LLM（第 2 步）**和** OpenAI TTS — 无需单独的 OpenAI 密钥。全新安装时，`hermes setup --portal` 可一次性完成两项配置。
:::

## 概览

| 功能 | 平台 | 说明 |
|---------|----------|-------------|
| **交互式语音** | CLI | 按 Ctrl+B 开始录音，Agent 自动检测静音并回复 |
| **自动语音回复** | Telegram、Discord | Agent 在文字回复的同时发送语音音频 |
| **语音频道** | Discord | Bot 加入语音频道，监听用户发言并语音回复 |

## 环境要求

### Python 包

```bash
# CLI 语音模式（麦克风 + 音频播放）
pip install "hermes-agent[voice]"

# Discord + Telegram 消息（包含 discord.py[voice] 以支持语音频道）
pip install "hermes-agent[messaging]"

# 高级 TTS（ElevenLabs）
pip install "hermes-agent[tts-premium]"

# 本地 TTS（NeuTTS，可选）
python -m pip install -U neutts[all]

# 一次性安装所有内容
pip install "hermes-agent[all]"
```

| 扩展包 | 包含的包 | 用途 |
|-------|----------|-------------|
| `voice` | `sounddevice`、`numpy` | CLI 语音模式 |
| `messaging` | `discord.py[voice]`、`python-telegram-bot`、`aiohttp` | Discord 和 Telegram 机器人 |
| `tts-premium` | `elevenlabs` | ElevenLabs TTS 提供商 |

可选本地 TTS 提供商：使用 `python -m pip install -U neutts[all]` 单独安装 `neutts`。首次使用时会自动下载模型。

:::info
`discord.py[voice]` 会自动安装 **PyNaCl**（用于语音加密）和 **opus 绑定**。这是 Discord 语音频道支持的必要条件。
:::

### 系统依赖

```bash
# macOS
brew install portaudio ffmpeg opus
brew install espeak-ng   # for NeuTTS

# Ubuntu/Debian
sudo apt install portaudio19-dev ffmpeg libopus0
sudo apt install espeak-ng   # for NeuTTS
```

| 依赖项 | 用途 | 适用场景 |
|-----------|---------|-------------|
| **PortAudio** | 麦克风输入和音频播放 | CLI 语音模式 |
| **ffmpeg** | 音频格式转换（MP3 → Opus、PCM → WAV） | 所有平台 |
| **Opus** | Discord 语音编解码器 | Discord 语音频道 |
| **espeak-ng** | Phonemizer 后端 | 本地 NeuTTS 提供商 |

### API 密钥

添加到 `~/.hermes/.env`：

```bash
# 语音转文字（STT）— 本地提供商完全不需要密钥
# pip install faster-whisper          # 免费，本地运行，推荐
GROQ_API_KEY=your-key                 # Groq Whisper — 速度快，有免费额度（云端）
VOICE_TOOLS_OPENAI_KEY=your-key       # OpenAI Whisper — 付费（云端）

# 文字转语音（TTS，可选 — Edge TTS 和 NeuTTS 无需任何密钥）
ELEVENLABS_API_KEY=***           # ElevenLabs — 高级音质
# 上方的 VOICE_TOOLS_OPENAI_KEY 同时启用 OpenAI TTS
```

:::tip
如果已安装 `faster-whisper`，语音模式的 STT 无需任何 API 密钥即可运行。模型（`base` 约 150 MB）会在首次使用时自动下载。
:::

---

## CLI 语音模式

语音模式在**经典 CLI**（`hermes chat`）和 **TUI**（`hermes --tui`）中均可使用。两者行为完全一致 — 相同的斜杠命令、相同的 VAD（语音活动检测）静音检测、相同的流式 TTS、相同的幻觉过滤器。TUI 额外将崩溃诊断日志转发至 `~/.hermes/logs/`，以便在异常音频后端出现按键录音失败时提供完整堆栈跟踪，而非静默消失。

### 快速开始

启动 CLI 并启用语音模式：

```bash
hermes                # 启动交互式 CLI
```

然后在 CLI 中使用以下命令：

```
/voice          切换语音模式开/关
/voice on       启用语音模式
/voice off      禁用语音模式
/voice tts      切换 TTS 输出
/voice status   显示当前状态
```

### 工作原理

1. 使用 `hermes` 启动 CLI，并通过 `/voice on` 启用语音模式
2. **按下 Ctrl+B** — 播放提示音（880Hz），开始录音
3. **开始说话** — 实时音频电平条显示输入状态：`● [▁▂▃▅▇▇▅▂] ❯`
4. **停止说话** — 静音 3 秒后自动停止录音
5. **两声提示音**（660Hz）确认录音结束
6. 音频通过 Whisper 转录后发送给 Agent
7. 如果启用了 TTS，Agent 的回复将以语音朗读
8. 录音**自动重新开始** — 无需按任何键即可继续说话

此循环持续进行，直到在录音过程中按下 **Ctrl+B**（退出连续模式），或连续 3 次录音均未检测到语音为止。

:::tip
录音键可通过 `~/.hermes/config.yaml` 中的 `voice.record_key` 配置（默认：`ctrl+b`）。
:::

### 静音检测

两阶段算法检测您是否已停止说话：

1. **语音确认** — 等待音频 RMS 值超过阈值（200）至少 0.3 秒，允许音节间的短暂停顿
2. **结束检测** — 语音确认后，持续静音 3.0 秒即触发停止

如果 15 秒内完全未检测到语音，录音自动停止。

`silence_threshold` 和 `silence_duration` 均可在 `config.yaml` 中配置。也可通过 `voice.beep_enabled: false` 禁用录音开始/结束提示音。

### 流式 TTS

启用 TTS 后，Agent 在生成文字的同时**逐句**朗读回复 — 无需等待完整响应：

1. 将文字增量缓冲为完整句子（最少 20 个字符）
2. 去除 Markdown 格式和 `<think>` 块
3. 实时逐句生成并播放音频

### 幻觉过滤器

Whisper 有时会从静音或背景噪音中生成幻觉文字（如"Thank you for watching"、"Subscribe"等）。Agent 使用包含 26 个已知幻觉短语（覆盖多种语言）的列表以及能捕获重复变体的正则表达式模式对其进行过滤。

---

## Gateway 语音回复（Telegram 和 Discord）

如果尚未设置消息机器人，请参阅对应平台的指南：
- [Telegram 设置指南](../messaging/telegram.md)
- [Discord 设置指南](../messaging/discord.md)

启动 gateway 以连接到消息平台：

```bash
hermes gateway        # 启动 gateway（连接到已配置的平台）
hermes gateway setup  # 首次配置的交互式设置向导
```

### Discord：频道与私信

Bot 在 Discord 上支持两种交互模式：

| 模式 | 交互方式 | 是否需要 @提及 | 设置 |
|------|------------|-----------------|-------|
| **私信（DM）** | 打开 Bot 的个人资料 → "发消息" | 否 | 立即可用 |
| **服务器频道** | 在 Bot 所在的文字频道中发言 | 是（`@botname`） | Bot 必须被邀请到服务器 |

**私信（个人使用推荐）：** 直接与 Bot 开启私信并发送消息 — 无需 @提及。语音回复和所有命令与在频道中使用完全相同。

**服务器频道：** Bot 仅在被 @提及时响应（例如 `@hermesbyt4 你好`）。请确保从提及弹窗中选择 **Bot 用户**，而非同名角色。

:::tip
如需在服务器频道中禁用提及要求，在 `~/.hermes/.env` 中添加：
```bash
DISCORD_REQUIRE_MENTION=false
```
或将特定频道设置为自由响应模式（无需提及）：
```bash
DISCORD_FREE_RESPONSE_CHANNELS=123456789,987654321
```
:::

### 命令

以下命令在 Telegram 和 Discord（私信和文字频道）中均可使用：

```
/voice          切换语音模式开/关
/voice on       仅在您发送语音消息时回复语音
/voice tts      对所有消息回复语音
/voice off      禁用语音回复
/voice status   显示当前设置
```

### 模式

| 模式 | 命令 | 行为 |
|------|---------|----------|
| `off` | `/voice off` | 仅文字（默认） |
| `voice_only` | `/voice on` | 仅当您发送语音消息时才语音回复 |
| `all` | `/voice tts` | 对每条消息均语音回复 |

语音模式设置在 gateway 重启后保持不变。

### 平台投递

| 平台 | 格式 | 说明 |
|----------|--------|-------|
| **Telegram** | 语音气泡（Opus/OGG） | 在聊天中内联播放。如需要，ffmpeg 将 MP3 转换为 Opus |
| **Discord** | 原生语音气泡（Opus/OGG） | 像用户语音消息一样内联播放。如语音气泡 API 失败则回退为文件附件 |

---

## Discord 语音频道

最具沉浸感的语音功能：Bot 加入 Discord 语音频道，监听用户发言，转录语音，通过 Agent 处理后，在语音频道中语音回复。

### 设置

#### 1. Discord Bot 权限

如果您已为文字功能设置了 Discord Bot（参见 [Discord 设置指南](../messaging/discord.md)），需要额外添加语音权限。

前往 [Discord 开发者门户](https://discord.com/developers/applications) → 您的应用 → **Installation** → **Default Install Settings** → **Guild Install**：

**在现有文字权限基础上添加以下权限：**

| 权限 | 用途 | 是否必需 |
|-----------|---------|----------|
| **Connect** | 加入语音频道 | 是 |
| **Speak** | 在语音频道中播放 TTS 音频 | 是 |
| **Use Voice Activity** | 检测用户是否正在说话 | 推荐 |

**更新后的权限整数：**

| 级别 | 整数 | 包含内容 |
|-------|---------|----------------|
| 仅文字 | `274878286912` | 查看频道、发送消息、读取历史、嵌入内容、附件、帖子、反应 |
| 文字 + 语音 | `274881432640` | 以上所有 + Connect、Speak |

**使用更新后的权限 URL 重新邀请 Bot：**

```
https://discord.com/oauth2/authorize?client_id=YOUR_APP_ID&scope=bot+applications.commands&permissions=274881432640
```

将 `YOUR_APP_ID` 替换为开发者门户中的应用 ID。

:::warning
将 Bot 重新邀请到已加入的服务器只会更新其权限，不会将其移除。不会丢失任何数据或配置。
:::

#### 2. 特权 Gateway Intents

在 [开发者门户](https://discord.com/developers/applications) → 您的应用 → **Bot** → **Privileged Gateway Intents** 中，启用以下三项：

| Intent | 用途 |
|--------|---------|
| **Presence Intent** | 检测用户在线/离线状态 |
| **Server Members Intent** | 将 `DISCORD_ALLOWED_USERS` 中的用户名解析为数字 ID（条件性） |
| **Message Content Intent** | 读取频道中的文字消息内容 |

**Message Content Intent** 为必需项。**Server Members Intent** 仅在 `DISCORD_ALLOWED_USERS` 列表使用用户名时才需要 — 如果使用数字用户 ID，可以关闭。语音频道中 SSRC → user_id 的映射来自 Discord 语音 WebSocket 上的 SPEAKING opcode，**不**需要 Server Members Intent。

#### 3. Opus 编解码器

运行 gateway 的机器上必须安装 Opus 编解码器库：

```bash
# macOS (Homebrew)
brew install opus

# Ubuntu/Debian
sudo apt install libopus0
```

Bot 会从以下路径自动加载编解码器：
- **macOS：** `/opt/homebrew/lib/libopus.dylib`
- **Linux：** `libopus.so.0`

#### 4. 环境变量

```bash
# ~/.hermes/.env

# Discord bot（已为文字功能配置）
DISCORD_BOT_TOKEN=your-bot-token
DISCORD_ALLOWED_USERS=your-user-id

# STT — 本地提供商无需密钥（pip install faster-whisper）
# GROQ_API_KEY=your-key            # 替代方案：云端，速度快，有免费额度

# TTS — 可选。Edge TTS 和 NeuTTS 无需密钥。
# ELEVENLABS_API_KEY=***      # 高级音质
# VOICE_TOOLS_OPENAI_KEY=***  # OpenAI TTS / Whisper
```

### 启动 Gateway

```bash
hermes gateway        # 使用现有配置启动
```

Bot 应在几秒内在 Discord 中上线。

### 命令

在 Bot 所在的 Discord 文字频道中使用以下命令：

```
/voice join      Bot 加入您当前所在的语音频道
/voice channel   /voice join 的别名
/voice leave     Bot 断开语音频道连接
/voice status    显示语音模式和已连接的频道
```

:::info
运行 `/voice join` 前，您必须已在某个语音频道中。Bot 会加入您所在的语音频道。
:::

### 工作原理

Bot 加入语音频道后：

1. **独立监听**每位用户的音频流
2. **检测静音** — 至少 0.5 秒语音后出现 1.5 秒静音即触发处理
3. **转录**音频（通过本地、Groq 或 OpenAI 的 Whisper STT）
4. **处理**完整的 Agent 流水线（会话、工具、记忆）
5. **语音回复**通过 TTS 在语音频道中播放

### 文字频道集成

Bot 在语音频道中时：

- 转录内容会出现在文字频道中：`[Voice] @user: 您说的内容`
- Agent 回复同时以文字发送到频道并在语音频道中朗读
- 文字频道为发出 `/voice join` 命令的那个频道

### 回声消除

Bot 在播放 TTS 回复时会自动暂停音频监听，防止听到并重复处理自身的输出。

### 访问控制

只有 `DISCORD_ALLOWED_USERS` 中列出的用户才能通过语音进行交互。其他用户的音频会被静默忽略。

```bash
# ~/.hermes/.env
DISCORD_ALLOWED_USERS=284102345871466496
```

---

## 配置参考

### config.yaml

```yaml
# 语音录制（CLI）
voice:
  record_key: "ctrl+b"            # 开始/停止录音的按键
  max_recording_seconds: 120       # 最大录音时长
  auto_tts: false                  # 启用语音模式时自动开启 TTS
  beep_enabled: true               # 播放录音开始/结束提示音
  silence_threshold: 200           # 静音判定的 RMS 电平（0-32767）
  silence_duration: 3.0            # 自动停止前的静音秒数

# 语音转文字（STT）
stt:
  enabled: true                     # 设为 false 可跳过自动转录 —
                                    # gateway 仍会缓存音频文件并将其路径
                                    # 作为入站消息的一部分传递给 Agent，
                                    # 适用于自定义流水线
                                    # （说话人分离、对齐、归档等）
  provider: "local"                  # "local"（免费）| "groq" | "openai"
  local:
    model: "base"                    # tiny, base, small, medium, large-v3
  # model: "whisper-1"              # 旧版：在未设置 provider 时使用

# 文字转语音（TTS）
tts:
  provider: "edge"                 # "edge"（免费）| "elevenlabs" | "openai" | "neutts" | "minimax"
  edge:
    voice: "en-US-AriaNeural"      # 322 种声音，74 种语言
  elevenlabs:
    voice_id: "pNInz6obpgDQGcFmaJgB"    # Adam
    model_id: "eleven_multilingual_v2"
  openai:
    model: "gpt-4o-mini-tts"
    voice: "alloy"                 # alloy, echo, fable, onyx, nova, shimmer
    base_url: "https://api.openai.com/v1"  # 可选：覆盖为自托管或兼容 OpenAI 的端点
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

### 环境变量

```bash
# 语音转文字提供商（本地无需密钥）
# pip install faster-whisper        # 免费本地 STT — 无需 API 密钥
GROQ_API_KEY=...                    # Groq Whisper（速度快，有免费额度）
VOICE_TOOLS_OPENAI_KEY=...         # OpenAI Whisper（付费）

# STT 高级覆盖（可选）
STT_GROQ_MODEL=whisper-large-v3-turbo    # 覆盖默认 Groq STT 模型
STT_OPENAI_MODEL=whisper-1               # 覆盖默认 OpenAI STT 模型
GROQ_BASE_URL=https://api.groq.com/openai/v1     # 自定义 Groq 端点
STT_OPENAI_BASE_URL=https://api.openai.com/v1    # 自定义 OpenAI STT 端点

# 文字转语音提供商（Edge TTS 和 NeuTTS 无需密钥）
ELEVENLABS_API_KEY=***             # ElevenLabs（高级音质）
# 上方的 VOICE_TOOLS_OPENAI_KEY 同时启用 OpenAI TTS

# Discord 语音频道
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=...
```

### STT 提供商对比

| 提供商 | 模型 | 速度 | 质量 | 费用 | 需要 API 密钥 |
|----------|-------|-------|---------|------|---------|
| **本地** | `base` | 快（取决于 CPU/GPU） | 良好 | 免费 | 否 |
| **本地** | `small` | 中等 | 较好 | 免费 | 否 |
| **本地** | `large-v3` | 慢 | 最佳 | 免费 | 否 |
| **Groq** | `whisper-large-v3-turbo` | 非常快（约 0.5 秒） | 良好 | 免费额度 | 是 |
| **Groq** | `whisper-large-v3` | 快（约 1 秒） | 较好 | 免费额度 | 是 |
| **OpenAI** | `whisper-1` | 快（约 1 秒） | 良好 | 付费 | 是 |
| **OpenAI** | `gpt-4o-transcribe` | 中等（约 2 秒） | 最佳 | 付费 | 是 |

提供商优先级（自动回退）：**本地** > **groq** > **openai**

### TTS 提供商对比

| 提供商 | 质量 | 费用 | 延迟 | 需要密钥 |
|----------|---------|------|---------|-------------|
| **Edge TTS** | 良好 | 免费 | 约 1 秒 | 否 |
| **ElevenLabs** | 优秀 | 付费 | 约 2 秒 | 是 |
| **OpenAI TTS** | 良好 | 付费 | 约 1.5 秒 | 是 |
| **NeuTTS** | 良好 | 免费 | 取决于 CPU/GPU | 否 |

NeuTTS 使用上方的 `tts.neutts` 配置块。

---

## 故障排查

### "No audio device found"（CLI）

PortAudio 未安装：

```bash
brew install portaudio    # macOS
sudo apt install portaudio19-dev  # Ubuntu
```

### Bot 在 Discord 服务器频道中不响应

Bot 在服务器频道中默认需要 @提及。请确认：

1. 输入 `@` 后选择 **Bot 用户**（带有 #discriminator），而非同名**角色**
2. 或改用私信 — 无需提及
3. 或在 `~/.hermes/.env` 中设置 `DISCORD_REQUIRE_MENTION=false`

### Bot 加入语音频道但听不到我说话

- 检查您的 Discord 用户 ID 是否在 `DISCORD_ALLOWED_USERS` 中
- 确认您在 Discord 中未被静音
- Bot 需要收到 Discord 的 SPEAKING 事件才能映射您的音频 — 加入后请在几秒内开始说话

### Bot 能听到我说话但不响应

- 验证 STT 是否可用：安装 `faster-whisper`（无需密钥）或设置 `GROQ_API_KEY` / `VOICE_TOOLS_OPENAI_KEY`
- 检查 LLM 模型是否已配置且可访问
- 查看 gateway 日志：`tail -f ~/.hermes/logs/gateway.log`

### Bot 有文字回复但语音频道中没有声音

- TTS 提供商可能出现故障 — 检查 API 密钥和配额
- Edge TTS（免费，无需密钥）是默认回退选项
- 检查日志中的 TTS 错误

### Whisper 返回乱码文字

幻觉过滤器会自动处理大多数情况。如果仍然出现幻觉转录：

- 在更安静的环境中使用
- 在配置中调高 `silence_threshold`（值越高，灵敏度越低）
- 尝试不同的 STT 模型
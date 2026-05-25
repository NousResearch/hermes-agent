---
sidebar_position: 8
title: "在 Hermes 中使用语音模式"
description: "在 CLI、Telegram、Discord 及 Discord 语音频道中设置和使用 Hermes 语音模式的实用指南"
---

# 在 Hermes 中使用语音模式

本指南是[语音模式功能参考](/user-guide/features/voice-mode)的实用配套文档。

功能页面介绍语音模式能做什么，本指南则说明如何真正用好它。

## 语音模式适合哪些场景

语音模式在以下情况特别有用：
- 需要免手持的 CLI 工作流
- 希望在 Telegram 或 Discord 中获得语音回复
- 希望 Hermes 加入 Discord 语音频道进行实时对话
- 边走动边快速记录想法、调试问题或来回交流，而不是打字

## 选择你的语音模式方案

Hermes 中实际上有三种不同的语音体验。

| 模式 | 最适合 | 平台 |
|---|---|---|
| 交互式麦克风循环 | 编码或研究时的个人免手持使用 | CLI |
| 聊天中的语音回复 | 在正常消息旁附带语音回复 | Telegram、Discord |
| 实时语音频道机器人 | 在语音频道中进行群组或个人实时对话 | Discord 语音频道 |

推荐路径：
1. 先让文本模式正常工作
2. 再启用语音回复
3. 最后如需完整体验，再切换到 Discord 语音频道

## 第一步：确保普通 Hermes 先正常运行

在接触语音模式之前，请确认：
- Hermes 能正常启动
- 已配置好 provider（提供商）
- Agent 能正常回答文本 prompt（提示词）

```bash
hermes
```

问一个简单的问题：

```text
What tools do you have available?
```

如果文本模式还不稳定，请先修复它。

## 第二步：安装所需的额外依赖

### CLI 麦克风 + 播放

```bash
pip install "hermes-agent[voice]"
```

### 消息平台

```bash
pip install "hermes-agent[messaging]"
```

### 高级 ElevenLabs TTS

```bash
pip install "hermes-agent[tts-premium]"
```

### 本地 NeuTTS（可选）

```bash
python -m pip install -U neutts[all]
```

### 全部安装

```bash
pip install "hermes-agent[all]"
```

## 第三步：安装系统依赖

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

各依赖的作用：
- `portaudio` → CLI 语音模式的麦克风输入与播放
- `ffmpeg` → TTS 和消息传递的音频转换
- `opus` → Discord 语音编解码器支持
- `espeak-ng` → NeuTTS 的 phonemizer 后端

## 第四步：选择 STT 和 TTS 提供商

Hermes 同时支持本地和云端语音处理方案。

### 最简单 / 最低成本的方案

使用本地 STT 和免费的 Edge TTS：
- STT provider：`local`
- TTS provider：`edge`

这通常是最好的起点。

### 环境变量文件示例

添加到 `~/.hermes/.env`：

```bash
# 云端 STT 选项（本地无需密钥）
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# 高级 TTS（可选）
ELEVENLABS_API_KEY=***
```

### Provider 推荐

#### 语音转文字（STT）

- `local` → 隐私保护和零成本使用的最佳默认选项
- `groq` → 极快的云端转录
- `openai` → 良好的付费备选

#### 文字转语音（TTS）

- `edge` → 免费，对大多数用户已足够
- `neutts` → 免费的本地/设备端 TTS
- `elevenlabs` → 最佳质量
- `openai` → 良好的中间选项
- `mistral` → 多语言，原生 Opus

### 如果使用 `hermes setup`

如果你在设置向导中选择了 NeuTTS，Hermes 会检查 `neutts` 是否已安装。如果缺失，向导会告知你 NeuTTS 需要 Python 包 `neutts` 和系统包 `espeak-ng`，并提供自动安装，使用平台包管理器安装 `espeak-ng`，然后运行：

```bash
python -m pip install -U neutts[all]
```

如果跳过安装或安装失败，向导会回退到 Edge TTS。

## 第五步：推荐配置

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

这是适合大多数人的保守默认配置。

如果想改用本地 TTS，将 `tts` 块替换为：

```yaml
tts:
  provider: "neutts"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

## 使用场景一：CLI 语音模式

## 开启方式

启动 Hermes：

```bash
hermes
```

在 CLI 内执行：

```text
/voice on
```

### 录音流程

默认按键：
- `Ctrl+B`

工作流程：
1. 按下 `Ctrl+B`
2. 说话
3. 等待静音检测自动停止录音
4. Hermes 转录并回复
5. 如果开启了 TTS，它会朗读答案
6. 循环可自动重启以持续使用

### 常用命令

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### 推荐的 CLI 工作流

#### 随走随调试

说：

```text
I keep getting a docker permission error. Help me debug it.
```

然后继续免手持操作：
- "再读一遍最后的错误"
- "用更简单的语言解释根本原因"
- "现在给我精确的修复方案"

#### 研究 / 头脑风暴

非常适合：
- 边走动边思考
- 口述半成形的想法
- 让 Hermes 实时整理你的思路

#### 无障碍 / 少打字场景

如果打字不方便，语音模式是保持完整 Hermes 工作流的最快方式之一。

## 调整 CLI 行为

### 静音阈值

如果 Hermes 开始/停止过于激进，调整：

```yaml
voice:
  silence_threshold: 250
```

阈值越高 = 灵敏度越低。

### 静音时长

如果你在句子之间经常停顿，增大该值：

```yaml
voice:
  silence_duration: 4.0
```

### 录音按键

如果 `Ctrl+B` 与你的终端或 tmux 习惯冲突：

```yaml
voice:
  record_key: "ctrl+space"
```

## 使用场景二：Telegram 或 Discord 中的语音回复

此模式比完整语音频道更简单。

Hermes 仍作为普通聊天机器人运行，但可以朗读回复。

### 启动 gateway

```bash
hermes gateway
```

### 开启语音回复

在 Telegram 或 Discord 中：

```text
/voice on
```

或

```text
/voice tts
```

### 模式说明

| 模式 | 含义 |
|---|---|
| `off` | 仅文本 |
| `voice_only` | 仅当用户发送语音时才朗读 |
| `all` | 朗读每条回复 |

### 何时使用哪种模式

- `/voice on`：仅对语音来源的消息给出语音回复
- `/voice tts`：始终作为完整语音助手运行

### 推荐的消息平台工作流

#### 手机上的 Telegram 助手

适用于：
- 离开电脑时
- 发送语音备忘并获取快速语音回复
- 希望 Hermes 充当便携式研究或运维助手

#### Discord 私信中的语音输出

适用于希望私密交互、避免服务器频道 @mention 行为的场景。

## 使用场景三：Discord 语音频道

这是最高级的模式。

Hermes 加入 Discord 语音频道（VC），监听用户语音，转录后运行正常的 agent 流水线，并将回复朗读回频道。

## 所需的 Discord 权限

除了普通文本机器人设置外，请确保机器人拥有：
- Connect（连接）
- Speak（发言）
- 最好还有 Use Voice Activity（使用语音活动）

同时在开发者门户中启用特权 intent（意图）：
- Presence Intent
- Server Members Intent
- Message Content Intent

## 加入与离开

在机器人所在的 Discord 文本频道中：

```text
/voice join
/voice leave
/voice status
```

### 加入后的行为

- 用户在语音频道中说话
- Hermes 检测语音边界
- 转录内容发布到关联的文本频道
- Hermes 以文字和音频形式回复
- 文本频道为执行 `/voice join` 的那个频道

### Discord 语音频道使用最佳实践

- 严格限制 `DISCORD_ALLOWED_USERS`
- 先使用专用的机器人/测试频道
- 在尝试语音频道模式之前，先确认 STT 和 TTS 在普通文本聊天语音模式下正常工作

## 语音质量建议

### 最佳质量方案

- STT：本地 `large-v3` 或 Groq `whisper-large-v3`
- TTS：ElevenLabs

### 最佳速度 / 便利性方案

- STT：本地 `base` 或 Groq
- TTS：Edge

### 最佳零成本方案

- STT：本地
- TTS：Edge

## 常见故障模式

### "No audio device found"

安装 `portaudio`。

### "机器人加入但听不到声音"

检查：
- 你的 Discord 用户 ID 是否在 `DISCORD_ALLOWED_USERS` 中
- 你是否处于静音状态
- 特权 intent 是否已启用
- 机器人是否拥有 Connect/Speak 权限

### "能转录但不说话"

检查：
- TTS provider 配置
- ElevenLabs 或 OpenAI 的 API 密钥 / 配额
- Edge 转换路径的 `ffmpeg` 安装情况

### "Whisper 输出乱码"

尝试：
- 更安静的环境
- 提高 `silence_threshold`
- 更换 STT provider/模型
- 更短、更清晰的表达

### "在私信中正常但在服务器频道中不工作"

这通常是 mention（提及）策略问题。

默认情况下，除非另行配置，机器人在 Discord 服务器文本频道中需要被 `@mention` 才会响应。

## 建议的第一周方案

如果你想走最短的成功路径：

1. 让文本 Hermes 正常工作
2. 安装 `hermes-agent[voice]`
3. 使用本地 STT + Edge TTS 的 CLI 语音模式
4. 然后在 Telegram 或 Discord 中启用 `/voice on`
5. 只有在此之后，再尝试 Discord 语音频道模式

这种递进方式可以将调试范围控制到最小。

## 下一步阅读

- [语音模式功能参考](/user-guide/features/voice-mode)
- [消息 Gateway](/user-guide/messaging)
- [Discord 设置](/user-guide/messaging/discord)
- [Telegram 设置](/user-guide/messaging/telegram)
- [配置](/user-guide/configuration)
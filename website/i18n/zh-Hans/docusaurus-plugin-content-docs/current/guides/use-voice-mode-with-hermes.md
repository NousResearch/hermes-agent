---
sidebar_position: 8
title: "在 Hermes 中使用语音模式"
description: "在 CLI、Telegram、Discord 和 Discord 语音频道中设置和使用 Hermes 语音模式的实用指南"
---

# 在 Hermes 中使用语音模式

本指南是 [语音模式功能参考](/user-guide/features/voice-mode) 的实战补充。

如果功能页讲的是“能做什么”，这篇指南讲的是“怎么真正用起来”。

## 语音模式适合什么场景

语音模式特别适合：
- 想要免手操作的 CLI 工作流
- 想在 Telegram 或 Discord 里收到语音回复
- 想让 Hermes 待在 Discord 语音频道里实时对话
- 一边走路一边捕捉想法、调试或来回问答

## 选择你的语音方案

Hermes 的语音体验大致有三种。

| 模式 | 适合场景 | 平台 |
|---|---|---|
| 交互式麦克风循环 | 编码或研究时的个人免手操作 | CLI |
| 聊天中的语音回复 | 在消息平台中获得语音输出 | Telegram、Discord |
| 语音频道机器人 | 在语音房间中实时对话 | Discord 语音频道 |

建议的路径是：
1. 先让文本工作正常
2. 再启用语音回复
3. 最后再上 Discord 语音频道

## 第 1 步：先确认普通 Hermes 可用

在碰语音之前，先确认：
- Hermes 能启动
- provider 已配置好
- 机器人可以正常回答文本问题

```bash
hermes
```

问一个简单问题：

```text
What tools do you have available?
```

如果这里还不稳定，先修文本模式。

## 第 2 步：安装正确的 extras

### CLI 麦克风 + 播放

```bash
pip install "hermes-agent[voice]"
```

### 消息平台

```bash
pip install "hermes-agent[messaging]"
```

### ElevenLabs 付费 TTS

```bash
pip install "hermes-agent[tts-premium]"
```

### 本地 NeuTTS（可选）

```bash
python -m pip install -U neutts[all]
```

### 全部

```bash
pip install "hermes-agent[all]"
```

## 第 3 步：安装系统依赖

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

这些依赖分别负责：
- `portaudio` → CLI 语音的麦克风输入 / 播放
- `ffmpeg` → TTS 与消息投递的音频转换
- `opus` → Discord 语音编解码
- `espeak-ng` → NeuTTS 的 phonemizer 后端

## 第 4 步：选择 STT 和 TTS provider

Hermes 支持本地和云端语音栈。

### 最省心 / 最便宜的组合

使用本地 STT + 免费 Edge TTS：
- STT provider: `local`
- TTS provider: `edge`

这通常是最好的起点。

### 环境文件示例

把这些写入 `~/.hermes/.env`：

```bash
# Cloud STT options (local needs no key)
GROQ_API_KEY=***
VOICE_TOOLS_OPENAI_KEY=***

# Premium TTS (optional)
ELEVENLABS_API_KEY=***
```

### Provider 建议

#### Speech-to-text

- `local` → 默认首选，兼顾隐私和零成本
- `groq` → 很快的云端转写
- `openai` → 不错的付费备用

#### Text-to-speech

- `edge` → 免费，够用
- `neutts` → 免费本地 TTS
- `elevenlabs` → 质量最好
- `openai` → 中间路线
- `mistral` → 多语言，原生 Opus

### 如果你用 `hermes setup`

如果你在安装向导里选择 NeuTTS，Hermes 会先检查 `neutts` 是否已安装。没有的话，向导会提示你 NeuTTS 需要 Python 包 `neutts` 和系统包 `espeak-ng`，并尝试帮你安装它们，然后执行：

```bash
python -m pip install -U neutts[all]
```

如果安装跳过或失败，向导会退回 Edge TTS。

## 第 5 步：推荐配置

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

这是大多数人的保守默认值。

如果你想用本地 TTS，把 `tts` 改成：

```yaml
tts:
  provider: "neutts"
  neutts:
    ref_audio: ''
    ref_text: ''
    model: neuphonic/neutts-air-q4-gguf
    device: cpu
```

## 用例 1：CLI 语音模式

## 打开它

启动 Hermes：

```bash
hermes
```

在 CLI 里输入：

```text
/voice on
```

### 录音流程

默认按键：
- `Ctrl+B`

流程：
1. 按 `Ctrl+B`
2. 开始说话
3. 等待静音检测自动结束录音
4. Hermes 转写并回复
5. 如果开启了 TTS，就会朗读答案
6. 循环可继续自动重启，实现连续使用

### 常用命令

```text
/voice
/voice on
/voice off
/voice tts
/voice status
```

### 常见 CLI 工作流

#### 边走边调试

可以直接说：

```text
I keep getting a docker permission error. Help me debug it.
```

然后继续免手操作：
- “再读一遍最后那个错误”
- “把根因讲得更简单一点”
- “现在给我精确的修复步骤”

#### 研究 / 头脑风暴

适合：
- 一边走一边想
- 口述半成型想法
- 让 Hermes 帮你实时整理思路

#### 无障碍 / 不方便打字时

如果打字不方便，语音模式是保持完整 Hermes 工作流的最快方式之一。

---
title: "Whisper — OpenAI 的通用语音识别模型"
sidebar_label: "Whisper"
description: "OpenAI 的通用语音识别模型"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Whisper

OpenAI 的通用语音识别模型。支持 99 种语言、转录、翻译为英语及语言识别。提供六种模型规格，从 tiny（3900 万参数）到 large（15.5 亿参数）。适用于语音转文字、播客转录或多语言音频处理。是鲁棒多语言 ASR（自动语音识别）的首选。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/whisper` 安装 |
| 路径 | `optional-skills/mlops/whisper` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `openai-whisper`, `transformers`, `torch` |
| 平台 | linux, macos |
| 标签 | `Whisper`, `Speech Recognition`, `ASR`, `Multimodal`, `Multilingual`, `OpenAI`, `Speech-To-Text`, `Transcription`, `Translation`, `Audio Processing` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Whisper - 鲁棒语音识别

OpenAI 的多语言语音识别模型。

## 何时使用 Whisper

**适用场景：**
- 语音转文字转录（99 种语言）
- 播客/视频转录
- 会议记录自动化
- 翻译为英语
- 嘈杂音频转录
- 多语言音频处理

**指标**：
- **GitHub 72,900+ 星**
- 支持 99 种语言
- 基于 68 万小时音频训练
- MIT 许可证

**改用其他替代方案的情况**：
- **AssemblyAI**：托管 API，支持说话人分离
- **Deepgram**：实时流式 ASR
- **Google Speech-to-Text**：基于云端

## 快速开始

### 安装

```bash
# Requires Python 3.8-3.11
pip install -U openai-whisper

# Requires ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: choco install ffmpeg
```

### 基本转录

```python
import whisper

# Load model
model = whisper.load_model("base")

# Transcribe
result = model.transcribe("audio.mp3")

# Print text
print(result["text"])

# Access segments
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

## 模型规格

```python
# Available models
models = ["tiny", "base", "small", "medium", "large", "turbo"]

# Load specific model
model = whisper.load_model("turbo")  # Fastest, good quality
```

| 模型 | 参数量 | 仅英语 | 多语言 | 速度 | 显存 |
|-------|------------|--------------|--------------|-------|------|
| tiny | 39M | ✓ | ✓ | ~32x | ~1 GB |
| base | 74M | ✓ | ✓ | ~16x | ~1 GB |
| small | 244M | ✓ | ✓ | ~6x | ~2 GB |
| medium | 769M | ✓ | ✓ | ~2x | ~5 GB |
| large | 1550M | ✗ | ✓ | 1x | ~10 GB |
| turbo | 809M | ✗ | ✓ | ~8x | ~6 GB |

**推荐**：追求最佳速度/质量比使用 `turbo`，原型开发使用 `base`

## 转录选项

### 语言指定

```python
# Auto-detect language
result = model.transcribe("audio.mp3")

# Specify language (faster)
result = model.transcribe("audio.mp3", language="en")

# Supported: en, es, fr, de, it, pt, ru, ja, ko, zh, and 89 more
```

### 任务选择

```python
# Transcription (default)
result = model.transcribe("audio.mp3", task="transcribe")

# Translation to English
result = model.transcribe("spanish.mp3", task="translate")
# Input: Spanish audio → Output: English text
```

### 初始 prompt（提示词）

```python
# Improve accuracy with context
result = model.transcribe(
    "audio.mp3",
    initial_prompt="This is a technical podcast about machine learning and AI."
)

# Helps with:
# - Technical terms
# - Proper nouns
# - Domain-specific vocabulary
```

### 时间戳

```python
# Word-level timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
```

### 温度回退

```python
# Retry with different temperatures if confidence low
result = model.transcribe(
    "audio.mp3",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
)
```

## 命令行用法

```bash
# Basic transcription
whisper audio.mp3

# Specify model
whisper audio.mp3 --model turbo

# Output formats
whisper audio.mp3 --output_format txt     # Plain text
whisper audio.mp3 --output_format srt     # Subtitles
whisper audio.mp3 --output_format vtt     # WebVTT
whisper audio.mp3 --output_format json    # JSON with timestamps

# Language
whisper audio.mp3 --language Spanish

# Translation
whisper spanish.mp3 --task translate
```

## 批量处理

```python
import os

audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]

for audio_file in audio_files:
    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file)

    # Save to file
    output_file = audio_file.replace(".mp3", ".txt")
    with open(output_file, "w") as f:
        f.write(result["text"])
```

## 实时转录

```python
# For streaming audio, use faster-whisper
# pip install faster-whisper

from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")

# Transcribe with streaming
segments, info = model.transcribe("audio.mp3", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## GPU 加速

```python
import whisper

# Automatically uses GPU if available
model = whisper.load_model("turbo")

# Force CPU
model = whisper.load_model("turbo", device="cpu")

# Force GPU
model = whisper.load_model("turbo", device="cuda")

# 10-20× faster on GPU
```

## 与其他工具集成

### 字幕生成

```bash
# Generate SRT subtitles
whisper video.mp4 --output_format srt --language English

# Output: video.srt
```

### 与 LangChain 集成

```python
from langchain.document_loaders import WhisperTranscriptionLoader

loader = WhisperTranscriptionLoader(file_path="audio.mp3")
docs = loader.load()

# Use transcription in RAG
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

### 从视频中提取音频

```bash
# Use ffmpeg to extract audio
ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav

# Then transcribe
whisper audio.wav
```

## 最佳实践

1. **使用 turbo 模型** — 英语场景下速度/质量最优
2. **指定语言** — 比自动检测更快
3. **添加初始 prompt** — 提升专业术语识别准确率
4. **使用 GPU** — 速度提升 10–20 倍
5. **批量处理** — 效率更高
6. **转换为 WAV** — 兼容性更好
7. **切分长音频** — 每段不超过 30 分钟
8. **确认语言支持情况** — 不同语言质量有差异
9. **使用 faster-whisper** — 比 openai-whisper 快 4 倍
10. **监控显存** — 根据硬件配置选择模型规格

## 性能

| 模型 | 实时倍率（CPU） | 实时倍率（GPU） |
|-------|------------------------|------------------------|
| tiny | ~0.32 | ~0.01 |
| base | ~0.16 | ~0.01 |
| turbo | ~0.08 | ~0.01 |
| large | ~1.0 | ~0.05 |

*实时倍率：0.1 表示比实时速度快 10 倍*

## 语言支持

主要支持语言：
- 英语（en）
- 西班牙语（es）
- 法语（fr）
- 德语（de）
- 意大利语（it）
- 葡萄牙语（pt）
- 俄语（ru）
- 日语（ja）
- 韩语（ko）
- 中文（zh）

完整列表：共 99 种语言

## 局限性

1. **幻觉问题** — 可能重复或生成不存在的文本
2. **长音频准确率** — 超过 30 分钟后质量下降
3. **说话人识别** — 不支持说话人分离
4. **口音** — 质量因口音而异
5. **背景噪音** — 可能影响准确率
6. **实时延迟** — 不适合实时字幕场景

## 资源

- **GitHub**：https://github.com/openai/whisper ⭐ 72,900+
- **论文**：https://arxiv.org/abs/2212.04356
- **模型卡片**：https://github.com/openai/whisper/blob/main/model-card.md
- **Colab**：可在仓库中获取
- **许可证**：MIT
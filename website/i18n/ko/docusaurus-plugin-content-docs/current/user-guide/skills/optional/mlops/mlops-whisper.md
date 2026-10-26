---
title: "Whisper — 99개 언어 음성 인식 및 번역"
sidebar_label: "Whisper"
description: "99개 언어 음성 인식 및 번역"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Whisper

99개 언어의 음성을 인식하고 번역합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/whisper`로 설치 |
| 경로 | `optional-skills/mlops/whisper` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `openai-whisper`, `transformers`, `torch` |
| 플랫폼 | linux, macos |
| 태그 | `Whisper`, `Speech Recognition`, `ASR`, `Multimodal`, `Multilingual`, `OpenAI`, `Speech-To-Text`, `Transcription`, `Translation`, `Audio Processing` |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침입니다.
:::

# Whisper - 강력한 음성 인식

OpenAI의 다국어 음성 인식 모델입니다.

## Whisper를 사용할 때

**다음과 같은 경우 사용하세요:**
- 음성-텍스트 변환(99개 언어)
- 팟캐스트/동영상 전사
- 회의록 자동화
- 영어로 번역
- 잡음이 있는 오디오 전사
- 다국어 오디오 처리

**측정 지표**:
- **GitHub 스타 72,900개 이상**
- 99개 언어 지원
- 680,000시간의 오디오로 학습
- MIT 라이선스

**대신 다음 대안을 사용하세요**:
- **AssemblyAI**: 관리형 API, 화자 분리
- **Deepgram**: 실시간 스트리밍 ASR
- **Google Speech-to-Text**: 클라우드 기반

## 빠른 시작

### 설치

```bash
# Requires Python 3.8-3.11
pip install -U openai-whisper

# Requires ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: choco install ffmpeg
```

### 기본 전사

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

## 모델 크기

```python
# Available models
models = ["tiny", "base", "small", "medium", "large", "turbo"]

# Load specific model
model = whisper.load_model("turbo")  # Fastest, good quality
```

| 모델 | 파라미터 | 영어 전용 | 다국어 | 속도 | VRAM |
|-------|------------|--------------|--------------|-------|------|
| tiny | 39M | ✓ | ✓ | ~32x | ~1 GB |
| base | 74M | ✓ | ✓ | ~16x | ~1 GB |
| small | 244M | ✓ | ✓ | ~6x | ~2 GB |
| medium | 769M | ✓ | ✓ | ~2x | ~5 GB |
| large | 1550M | ✗ | ✓ | 1x | ~10 GB |
| turbo | 809M | ✗ | ✓ | ~8x | ~6 GB |

**권장**: 속도/품질이 가장 중요하면 `turbo`, 프로토타이핑에는 `base`를 사용하세요.

## 전사 옵션

### 언어 지정

```python
# Auto-detect language
result = model.transcribe("audio.mp3")

# Specify language (faster)
result = model.transcribe("audio.mp3", language="en")

# Supported: en, es, fr, de, it, pt, ru, ja, ko, zh, and 89 more
```

### 작업 선택

```python
# Transcription (default)
result = model.transcribe("audio.mp3", task="transcribe")

# Translation to English
result = model.transcribe("spanish.mp3", task="translate")
# Input: Spanish audio → Output: English text
```

### 초기 프롬프트

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

### 타임스탬프

```python
# Word-level timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
```

### 온도 폴백

```python
# Retry with different temperatures if confidence low
result = model.transcribe(
    "audio.mp3",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
)
```

## 명령줄 사용법

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

## 일괄 처리

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

## 실시간 전사

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

## GPU 가속

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

## 다른 도구와의 통합

### 자막 생성

```bash
# Generate SRT subtitles
whisper video.mp4 --output_format srt --language English

# Output: video.srt
```

### LangChain과 함께 사용

```python
from langchain.document_loaders import WhisperTranscriptionLoader

loader = WhisperTranscriptionLoader(file_path="audio.mp3")
docs = loader.load()

# Use transcription in RAG
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

### 동영상에서 오디오 추출

```bash
# Use ffmpeg to extract audio
ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav

# Then transcribe
whisper audio.wav
```

## 모범 사례

1. **turbo 모델 사용** - 영어에서 속도/품질이 가장 좋습니다.
2. **언어 지정** - 자동 감지보다 빠릅니다.
3. **초기 프롬프트 추가** - 기술 용어 인식이 향상됩니다.
4. **GPU 사용** - 10-20배 빠릅니다.
5. **일괄 처리** - 더 효율적입니다.
6. **WAV로 변환** - 호환성이 더 좋습니다.
7. **긴 오디오 분할** - 30분 미만 청크로 나눕니다.
8. **언어 지원 확인** - 언어에 따라 품질이 다릅니다.
9. **faster-whisper 사용** - openai-whisper보다 4배 빠릅니다.
10. **VRAM 모니터링** - 하드웨어에 맞춰 모델 크기를 조정합니다.

## 성능

| 모델 | 실시간 계수(CPU) | 실시간 계수(GPU) |
|------|------------------------|------------------------|
| tiny | ~0.32 | ~0.01 |
| base | ~0.16 | ~0.01 |
| turbo | ~0.08 | ~0.01 |
| large | ~1.0 | ~0.05 |

*실시간 계수: 0.1 = 실시간보다 10배 빠름*

## 언어 지원

주요 지원 언어:
- 영어 (en)
- 스페인어 (es)
- 프랑스어 (fr)
- 독일어 (de)
- 이탈리아어 (it)
- 포르투갈어 (pt)
- 러시아어 (ru)
- 일본어 (ja)
- 한국어 (ko)
- 중국어 (zh)

전체 목록: 총 99개 언어

## 제한 사항

1. **환각** - 텍스트를 반복하거나 만들어낼 수 있습니다.
2. **장시간 오디오 정확도** - 30분을 초과하는 오디오에서 저하됩니다.
3. **화자 식별** - 화자 분리 기능이 없습니다.
4. **억양** - 품질이 다를 수 있습니다.
5. **배경 소음** - 정확도에 영향을 줄 수 있습니다.
6. **실시간 지연** - 실시간 자막에는 적합하지 않습니다.

## 리소스

- **GitHub**: https://github.com/openai/whisper ⭐ 72,900+
- **논문**: https://arxiv.org/abs/2212.04356
- **모델 카드**: https://github.com/openai/whisper/blob/main/model-card.md
- **Colab**: 저장소에서 이용 가능
- **라이선스**: MIT

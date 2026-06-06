---
title: "Whisper — OpenAI의 범용 음성 인식 모델"
sidebar_label: "Whisper"
description: "OpenAI의 범용 음성 인식 모델"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Whisper

OpenAI의 범용 음성 인식 모델. 99개 언어, 전사(transcription), 영어로의 번역, 언어 식별 기능을 지원합니다. tiny(39M 파라미터)부터 large(1550M 파라미터)까지 6가지 모델 크기를 제공합니다. 음성 텍스트 변환(Speech-to-Text), 팟캐스트 전사 또는 다국어 오디오 처리에 사용합니다. 강력하고 다국어를 지원하는 ASR에 가장 적합합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/whisper`로 설치 |
| Path | `optional-skills/mlops/whisper` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `openai-whisper`, `transformers`, `torch` |
| Platforms | linux, macos |
| Tags | `Whisper`, `Speech Recognition`, `ASR`, `Multimodal`, `Multilingual`, `OpenAI`, `Speech-To-Text`, `Transcription`, `Translation`, `Audio Processing` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Whisper - 강력한 음성 인식

OpenAI의 다국어 음성 인식 모델.

## Whisper를 사용해야 할 때

**다음과 같은 경우 사용하세요:**
- 음성 텍스트 변환 전사 (99개 언어)
- 팟캐스트/비디오 전사
- 회의록 자동화
- 영어로 번역
- 노이즈가 있는 오디오 전사
- 다국어 오디오 처리

**지표**:
- **72,900+ GitHub stars**
- 99개 언어 지원
- 680,000시간 분량의 오디오로 학습
- MIT 라이선스

**대신 다른 대안을 사용해야 할 때**:
- **AssemblyAI**: 관리형 API, 화자 분할(speaker diarization)
- **Deepgram**: 실시간 스트리밍 ASR
- **Google Speech-to-Text**: 클라우드 기반

## 빠른 시작

### 설치

```bash
# Python 3.8-3.11 필요
pip install -U openai-whisper

# ffmpeg 필요
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
# Windows: choco install ffmpeg
```

### 기본 전사

```python
import whisper

# 모델 로드
model = whisper.load_model("base")

# 전사
result = model.transcribe("audio.mp3")

# 텍스트 출력
print(result["text"])

# 세그먼트 접근
for segment in result["segments"]:
    print(f"[{segment['start']:.2f}s - {segment['end']:.2f}s] {segment['text']}")
```

## 모델 크기

```python
# 사용 가능한 모델
models = ["tiny", "base", "small", "medium", "large", "turbo"]

# 특정 모델 로드
model = whisper.load_model("turbo")  # 가장 빠름, 좋은 품질
```

| 모델 | 파라미터 | 영어 전용 | 다국어 | 속도 | VRAM |
|-------|------------|--------------|--------------|-------|------|
| tiny | 39M | ✓ | ✓ | ~32x | ~1 GB |
| base | 74M | ✓ | ✓ | ~16x | ~1 GB |
| small | 244M | ✓ | ✓ | ~6x | ~2 GB |
| medium | 769M | ✓ | ✓ | ~2x | ~5 GB |
| large | 1550M | ✗ | ✓ | 1x | ~10 GB |
| turbo | 809M | ✗ | ✓ | ~8x | ~6 GB |

**권장 사항**: 속도와 품질의 최적 조합을 원하면 `turbo`를 사용하고, 프로토타이핑에는 `base`를 사용하세요.

## 전사 옵션

### 언어 지정

```python
# 언어 자동 감지
result = model.transcribe("audio.mp3")

# 언어 지정 (더 빠름)
result = model.transcribe("audio.mp3", language="en")

# 지원: en, es, fr, de, it, pt, ru, ja, ko, zh 등 89개 언어
```

### 작업 선택

```python
# 전사 (기본값)
result = model.transcribe("audio.mp3", task="transcribe")

# 영어로 번역
result = model.transcribe("spanish.mp3", task="translate")
# 입력: 스페인어 오디오 → 출력: 영어 텍스트
```

### 초기 프롬프트 (Initial prompt)

```python
# 컨텍스트를 제공하여 정확도 향상
result = model.transcribe(
    "audio.mp3",
    initial_prompt="This is a technical podcast about machine learning and AI."
)

# 다음에 도움이 됩니다:
# - 전문 용어
# - 고유 명사
# - 도메인 특화 어휘
```

### 타임스탬프

```python
# 단어 수준 타임스탬프
result = model.transcribe("audio.mp3", word_timestamps=True)

for segment in result["segments"]:
    for word in segment["words"]:
        print(f"{word['word']} ({word['start']:.2f}s - {word['end']:.2f}s)")
```

### 온도 대체(Temperature fallback)

```python
# 신뢰도가 낮은 경우 다양한 온도로 재시도
result = model.transcribe(
    "audio.mp3",
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
)
```

## 명령줄 사용

```bash
# 기본 전사
whisper audio.mp3

# 모델 지정
whisper audio.mp3 --model turbo

# 출력 형식
whisper audio.mp3 --output_format txt     # 일반 텍스트
whisper audio.mp3 --output_format srt     # 자막
whisper audio.mp3 --output_format vtt     # WebVTT
whisper audio.mp3 --output_format json    # 타임스탬프가 포함된 JSON

# 언어
whisper audio.mp3 --language Spanish

# 번역
whisper spanish.mp3 --task translate
```

## 일괄 처리 (Batch processing)

```python
import os

audio_files = ["file1.mp3", "file2.mp3", "file3.mp3"]

for audio_file in audio_files:
    print(f"Transcribing {audio_file}...")
    result = model.transcribe(audio_file)

    # 파일에 저장
    output_file = audio_file.replace(".mp3", ".txt")
    with open(output_file, "w") as f:
        f.write(result["text"])
```

## 실시간 전사

```python
# 스트리밍 오디오의 경우 faster-whisper 사용
# pip install faster-whisper

from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")

# 스트리밍과 함께 전사
segments, info = model.transcribe("audio.mp3", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## GPU 가속

```python
import whisper

# 사용 가능한 경우 자동으로 GPU 사용
model = whisper.load_model("turbo")

# CPU 강제 지정
model = whisper.load_model("turbo", device="cpu")

# GPU 강제 지정
model = whisper.load_model("turbo", device="cuda")

# GPU에서 10-20배 더 빠름
```

## 다른 도구와의 통합

### 자막 생성

```bash
# SRT 자막 생성
whisper video.mp4 --output_format srt --language English

# 출력: video.srt
```

### LangChain과 함께 사용

```python
from langchain.document_loaders import WhisperTranscriptionLoader

loader = WhisperTranscriptionLoader(file_path="audio.mp3")
docs = loader.load()

# RAG에서 전사 사용
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings())
```

### 비디오에서 오디오 추출

```bash
# ffmpeg를 사용하여 오디오 추출
ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav

# 이후 전사
whisper audio.wav
```

## 모범 사례

1. **turbo 모델 사용** - 영어에서 최적의 속도/품질
2. **언어 지정** - 자동 감지보다 빠름
3. **초기 프롬프트 추가** - 기술 용어 인식 향상
4. **GPU 사용** - 10-20배 빠름
5. **일괄 처리** - 더 효율적임
6. **WAV로 변환** - 호환성 향상
7. **긴 오디오 분할** - 30분 미만의 청크로 분할
8. **언어 지원 확인** - 언어에 따라 품질 차이가 있음
9. **faster-whisper 사용** - openai-whisper보다 4배 빠름
10. **VRAM 모니터링** - 하드웨어에 맞게 모델 크기 조정

## 성능

| 모델 | 실시간 요인 (CPU) | 실시간 요인 (GPU) |
|-------|------------------------|------------------------|
| tiny | ~0.32 | ~0.01 |
| base | ~0.16 | ~0.01 |
| turbo | ~0.08 | ~0.01 |
| large | ~1.0 | ~0.05 |

*실시간 요인(Real-time factor): 0.1 = 실시간보다 10배 빠름*

## 언어 지원

상위 지원 언어:
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

## 한계점

1. **환각 (Hallucinations)** - 텍스트를 반복하거나 지어낼 수 있음
2. **장문 정확도** - 30분 이상의 오디오에서는 성능 저하
3. **화자 식별** - 화자 분할(diarization) 기능 없음
4. **억양** - 품질에 차이가 발생할 수 있음
5. **배경 소음** - 정확도에 영향을 미칠 수 있음
6. **실시간 지연** - 라이브 자막에는 적합하지 않음

## 리소스

- **GitHub**: https://github.com/openai/whisper ⭐ 72,900+
- **Paper**: https://arxiv.org/abs/2212.04356
- **Model Card**: https://github.com/openai/whisper/blob/main/model-card.md
- **Colab**: 저장소에서 사용 가능
- **License**: MIT

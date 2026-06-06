---
title: "Audiocraft Audio Generation — AudioCraft: MusicGen 텍스트-음악 변환, AudioGen 텍스트-효과음 변환"
sidebar_label: "Audiocraft Audio Generation"
description: "AudioCraft: MusicGen 텍스트-음악 변환, AudioGen 텍스트-효과음 변환"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Audiocraft Audio Generation

AudioCraft: MusicGen 텍스트-음악 변환, AudioGen 텍스트-효과음 변환.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/models/audiocraft` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `audiocraft`, `torch>=2.0.0`, `transformers>=4.30.0` |
| 플랫폼 | linux, macos |
| 태그 | `Multimodal`, `Audio Generation`, `Text-to-Music`, `Text-to-Audio`, `MusicGen` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# AudioCraft: 오디오 생성

MusicGen, AudioGen 및 EnCodec을 사용하여 텍스트-음악 및 텍스트-오디오 생성을 위한 Meta의 AudioCraft 사용에 대한 포괄적인 가이드입니다.

## AudioCraft 사용 시기

**다음과 같은 경우에 AudioCraft를 사용하세요:**
- 텍스트 설명을 바탕으로 음악을 생성해야 할 때
- 효과음(sound effects) 및 환경음(environmental audio)을 만들 때
- 음악 생성 애플리케이션을 구축할 때
- 멜로디 조건부(melody-conditioned) 음악 생성이 필요할 때
- 스테레오 오디오 출력을 원할 때
- 스타일 변환(style transfer)을 통해 제어 가능한 음악 생성이 필요할 때

**주요 특징:**
- **MusicGen**: 멜로디 조절(conditioning) 기능을 갖춘 텍스트-음악 변환
- **AudioGen**: 텍스트-효과음 변환
- **EnCodec**: 고음질 신경망 오디오 코덱
- **다양한 모델 크기**: Small (3억)부터 Large (33억)까지
- **스테레오 지원**: 풀 스테레오 오디오 생성
- **스타일 제어**: 참조(reference) 기반 생성을 위한 MusicGen-Style

**대신 다음 대안을 사용하는 것이 좋은 경우:**
- **Stable Audio**: 더 긴 상업용 음악 생성을 위한 경우
- **Bark**: 음악/효과음을 포함한 텍스트-음성(TTS) 변환의 경우
- **Riffusion**: 스펙트로그램 기반 음악 생성을 위한 경우
- **OpenAI Jukebox**: 가사가 포함된 원시 오디오(raw audio) 생성을 위한 경우

## 빠른 시작

### 설치

```bash
# PyPI에서 설치
pip install audiocraft

# GitHub에서 설치 (최신 버전)
pip install git+https://github.com/facebookresearch/audiocraft.git

# 또는 HuggingFace Transformers 사용
pip install transformers torch torchaudio
```

### 기본 텍스트-음악 변환 (AudioCraft)

```python
import torchaudio
from audiocraft.models import MusicGen

# 모델 로드
model = MusicGen.get_pretrained('facebook/musicgen-small')

# 생성 파라미터 설정
model.set_generation_params(
    duration=8,  # 초 단위
    top_k=250,
    temperature=1.0
)

# 텍스트로부터 생성
descriptions = ["happy upbeat electronic dance music with synths"]
wav = model.generate(descriptions)

# 오디오 저장
torchaudio.save("output.wav", wav[0].cpu(), sample_rate=32000)
```

### HuggingFace Transformers 사용

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

# 모델과 프로세서 로드
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.to("cuda")

# 음악 생성
inputs = processor(
    text=["80s pop track with bassy drums and synth"],
    padding=True,
    return_tensors="pt"
).to("cuda")

audio_values = model.generate(
    **inputs,
    do_sample=True,
    guidance_scale=3,
    max_new_tokens=256
)

# 저장
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
```

### AudioGen을 사용한 텍스트-효과음 변환

```python
from audiocraft.models import AudioGen

# AudioGen 로드
model = AudioGen.get_pretrained('facebook/audiogen-medium')

model.set_generation_params(duration=5)

# 효과음 생성
descriptions = ["dog barking in a park with birds chirping"]
wav = model.generate(descriptions)

torchaudio.save("sound.wav", wav[0].cpu(), sample_rate=16000)
```

## 핵심 개념

### 아키텍처 개요

<!-- ascii-guard-ignore -->
```
AudioCraft 아키텍처:
┌──────────────────────────────────────────────────────────────┐
│                    Text Encoder (T5)                          │
│                         │                                     │
│                    Text Embeddings                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│              Transformer Decoder (LM)                         │
│     Auto-regressively generates audio tokens                  │
│     Using efficient token interleaving patterns               │
└────────────────────────┬─────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                EnCodec Audio Decoder                          │
│        Converts tokens back to audio waveform                 │
└──────────────────────────────────────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

### 모델 변형(Variants)

| 모델 | 크기 | 설명 | 사용 사례 |
|-------|------|-------------|----------|
| `musicgen-small` | 300M | 텍스트-음악 | 빠른 생성 |
| `musicgen-medium` | 1.5B | 텍스트-음악 | 균형 잡힘 |
| `musicgen-large` | 3.3B | 텍스트-음악 | 최고 품질 |
| `musicgen-melody` | 1.5B | 텍스트 + 멜로디 | 멜로디 조건화 |
| `musicgen-melody-large` | 3.3B | 텍스트 + 멜로디 | 최고 품질 멜로디 |
| `musicgen-stereo-*` | 다양 | 스테레오 출력 | 스테레오 생성 |
| `musicgen-style` | 1.5B | 스타일 변환 | 참조 기반 |
| `audiogen-medium` | 1.5B | 텍스트-효과음 | 효과음 |

### 생성 파라미터

| 파라미터 | 기본값 | 설명 |
|-----------|---------|-------------|
| `duration` | 8.0 | 초 단위 길이 (1-120) |
| `top_k` | 250 | Top-k 샘플링 |
| `top_p` | 0.0 | 뉴클리어스(Nucleus) 샘플링 (0 = 비활성화) |
| `temperature` | 1.0 | 샘플링 온도(temperature) |
| `cfg_coef` | 3.0 | 분류기 없는 가이던스 (Classifier-free guidance) |

## MusicGen 사용법

### 텍스트-음악 생성

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained('facebook/musicgen-medium')

# 생성 구성
model.set_generation_params(
    duration=30,          # 최대 30초
    top_k=250,            # 샘플링 다양성
    top_p=0.0,            # 0 = top_k만 사용
    temperature=1.0,      # 창의성 (높을수록 더 다양함)
    cfg_coef=3.0          # 텍스트 준수도 (높을수록 더 엄격함)
)

# 여러 샘플 생성
descriptions = [
    "epic orchestral soundtrack with strings and brass",
    "chill lo-fi hip hop beat with jazzy piano",
    "energetic rock song with electric guitar"
]

# 생성 ([batch, channels, samples] 반환)
wav = model.generate(descriptions)

# 각각 저장
for i, audio in enumerate(wav):
    torchaudio.save(f"music_{i}.wav", audio.cpu(), sample_rate=32000)
```

### 멜로디 조건부 생성

```python
from audiocraft.models import MusicGen
import torchaudio

# 멜로디 모델 로드
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=30)

# 멜로디 오디오 로드
melody, sr = torchaudio.load("melody.wav")

# 멜로디 조건화를 통한 생성
descriptions = ["acoustic guitar folk song"]
wav = model.generate_with_chroma(descriptions, melody, sr)

torchaudio.save("melody_conditioned.wav", wav[0].cpu(), sample_rate=32000)
```

### 스테레오 생성

```python
from audiocraft.models import MusicGen

# 스테레오 모델 로드
model = MusicGen.get_pretrained('facebook/musicgen-stereo-medium')
model.set_generation_params(duration=15)

descriptions = ["ambient electronic music with wide stereo panning"]
wav = model.generate(descriptions)

# wav 형태: 스테레오의 경우 [batch, 2, samples]
print(f"Stereo shape: {wav.shape}")  # [1, 2, 480000]
torchaudio.save("stereo.wav", wav[0].cpu(), sample_rate=32000)
```

### 오디오 이어가기(Continuation)

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

# 이어갈 오디오 로드
import torchaudio
audio, sr = torchaudio.load("intro.wav")

# 텍스트와 오디오 처리
inputs = processor(
    audio=audio.squeeze().numpy(),
    sampling_rate=sr,
    text=["continue with a epic chorus"],
    padding=True,
    return_tensors="pt"
)

# 이어지는 부분 생성
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)
```

## MusicGen-Style 사용법

### 스타일 조건부 생성

```python
from audiocraft.models import MusicGen

# 스타일 모델 로드
model = MusicGen.get_pretrained('facebook/musicgen-style')

# 스타일 생성 구성
model.set_generation_params(
    duration=30,
    cfg_coef=3.0,
    cfg_coef_beta=5.0  # 스타일 영향력
)

# 스타일 조절기(Conditioner) 구성
model.set_style_conditioner_params(
    eval_q=3,          # RVQ quantizers (1-6)
    excerpt_length=3.0  # 스타일 발췌 길이
)

# 참조할 스타일 로드
style_audio, sr = torchaudio.load("reference_style.wav")

# 텍스트 + 스타일로 생성
descriptions = ["upbeat dance track"]
wav = model.generate_with_style(descriptions, style_audio, sr)
```

### 스타일 전용 생성 (텍스트 없음)

```python
# 텍스트 프롬프트 없이 일치하는 스타일 생성
model.set_generation_params(
    duration=30,
    cfg_coef=3.0,
    cfg_coef_beta=None  # 스타일 전용의 경우 double CFG 비활성화
)

wav = model.generate_with_style([None], style_audio, sr)
```

## AudioGen 사용법

### 효과음 생성

```python
from audiocraft.models import AudioGen
import torchaudio

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=10)

# 다양한 소리 생성
descriptions = [
    "thunderstorm with heavy rain and lightning",
    "busy city traffic with car horns",
    "ocean waves crashing on rocks",
    "crackling campfire in forest"
]

wav = model.generate(descriptions)

for i, audio in enumerate(wav):
    torchaudio.save(f"sound_{i}.wav", audio.cpu(), sample_rate=16000)
```

## EnCodec 사용법

### 오디오 압축

```python
from audiocraft.models import CompressionModel
import torch
import torchaudio

# EnCodec 로드
model = CompressionModel.get_pretrained('facebook/encodec_32khz')

# 오디오 로드
wav, sr = torchaudio.load("audio.wav")

# 올바른 샘플 레이트 확인
if sr != 32000:
    resampler = torchaudio.transforms.Resample(sr, 32000)
    wav = resampler(wav)

# 토큰으로 인코딩
with torch.no_grad():
    encoded = model.encode(wav.unsqueeze(0))
    codes = encoded[0]  # 오디오 코드

# 다시 오디오로 디코딩
with torch.no_grad():
    decoded = model.decode(codes)

torchaudio.save("reconstructed.wav", decoded[0].cpu(), sample_rate=32000)
```

## 일반적인 워크플로우

### 워크플로우 1: 음악 생성 파이프라인

```python
import torch
import torchaudio
from audiocraft.models import MusicGen

class MusicGenerator:
    def __init__(self, model_name="facebook/musicgen-medium"):
        self.model = MusicGen.get_pretrained(model_name)
        self.sample_rate = 32000

    def generate(self, prompt, duration=30, temperature=1.0, cfg=3.0):
        self.model.set_generation_params(
            duration=duration,
            top_k=250,
            temperature=temperature,
            cfg_coef=cfg
        )

        with torch.no_grad():
            wav = self.model.generate([prompt])

        return wav[0].cpu()

    def generate_batch(self, prompts, duration=30):
        self.model.set_generation_params(duration=duration)

        with torch.no_grad():
            wav = self.model.generate(prompts)

        return wav.cpu()

    def save(self, audio, path):
        torchaudio.save(path, audio, sample_rate=self.sample_rate)

# 사용법
generator = MusicGenerator()
audio = generator.generate(
    "epic cinematic orchestral music",
    duration=30,
    temperature=1.0
)
generator.save(audio, "epic_music.wav")
```

### 워크플로우 2: 사운드 디자인 배치 처리

```python
import json
from pathlib import Path
from audiocraft.models import AudioGen
import torchaudio

def batch_generate_sounds(sound_specs, output_dir):
    """
    사양(specifications)에 따라 여러 사운드를 생성합니다.

    Args:
        sound_specs: {"name": str, "description": str, "duration": float}의 리스트
        output_dir: 출력 디렉토리 경로
    """
    model = AudioGen.get_pretrained('facebook/audiogen-medium')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    results = []

    for spec in sound_specs:
        model.set_generation_params(duration=spec.get("duration", 5))

        wav = model.generate([spec["description"]])

        output_path = output_dir / f"{spec['name']}.wav"
        torchaudio.save(str(output_path), wav[0].cpu(), sample_rate=16000)

        results.append({
            "name": spec["name"],
            "path": str(output_path),
            "description": spec["description"]
        })

    return results

# 사용법
sounds = [
    {"name": "explosion", "description": "massive explosion with debris", "duration": 3},
    {"name": "footsteps", "description": "footsteps on wooden floor", "duration": 5},
    {"name": "door", "description": "wooden door creaking and closing", "duration": 2}
]

results = batch_generate_sounds(sounds, "sound_effects/")
```

### 워크플로우 3: Gradio 데모

```python
import gradio as gr
import torch
import torchaudio
from audiocraft.models import MusicGen

model = MusicGen.get_pretrained('facebook/musicgen-small')

def generate_music(prompt, duration, temperature, cfg_coef):
    model.set_generation_params(
        duration=duration,
        temperature=temperature,
        cfg_coef=cfg_coef
    )

    with torch.no_grad():
        wav = model.generate([prompt])

    # 임시 파일로 저장
    path = "temp_output.wav"
    torchaudio.save(path, wav[0].cpu(), sample_rate=32000)
    return path

demo = gr.Interface(
    fn=generate_music,
    inputs=[
        gr.Textbox(label="Music Description", placeholder="upbeat electronic dance music"),
        gr.Slider(1, 30, value=8, label="Duration (seconds)"),
        gr.Slider(0.5, 2.0, value=1.0, label="Temperature"),
        gr.Slider(1.0, 10.0, value=3.0, label="CFG Coefficient")
    ],
    outputs=gr.Audio(label="Generated Music"),
    title="MusicGen Demo"
)

demo.launch()
```

## 성능 최적화

### 메모리 최적화

```python
# 더 작은 모델 사용
model = MusicGen.get_pretrained('facebook/musicgen-small')

# 생성 사이에 캐시 지우기
torch.cuda.empty_cache()

# 더 짧은 길이 생성
model.set_generation_params(duration=10)  # 30 대신

# 반정밀도(half precision) 사용
model = model.half()
```

### 배치 처리 효율성

```python
# 여러 프롬프트를 한 번에 처리 (더 효율적)
descriptions = ["prompt1", "prompt2", "prompt3", "prompt4"]
wav = model.generate(descriptions)  # 단일 배치

# 다음과 같이 하지 마세요:
for desc in descriptions:
    wav = model.generate([desc])  # 다중 배치 (더 느림)
```

### GPU 메모리 요구 사항

| 모델 | FP32 VRAM | FP16 VRAM |
|-------|-----------|-----------|
| musicgen-small | ~4GB | ~2GB |
| musicgen-medium | ~8GB | ~4GB |
| musicgen-large | ~16GB | ~8GB |

## 일반적인 문제

| 문제 | 해결책 |
|-------|----------|
| CUDA OOM (메모리 부족) | 더 작은 모델 사용, 지속 시간 줄이기 |
| 낮은 품질 | cfg_coef 늘리기, 더 나은 프롬프트 사용 |
| 생성이 너무 짧음 | 최대 지속 시간 설정 확인 |
| 오디오 아티팩트 | 다른 temperature 값 시도 |
| 스테레오가 작동하지 않음 | 스테레오 모델 변형 사용 |

## 참조

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/audiocraft/references/advanced-usage.md)** - 학습, 미세 조정, 배포
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/models/audiocraft/references/troubleshooting.md)** - 일반적인 문제 및 해결책

## 리소스

- **GitHub**: https://github.com/facebookresearch/audiocraft
- **논문 (MusicGen)**: https://arxiv.org/abs/2306.05284
- **논문 (AudioGen)**: https://arxiv.org/abs/2209.15352
- **HuggingFace**: https://huggingface.co/facebook/musicgen-small
- **데모**: https://huggingface.co/spaces/facebook/MusicGen

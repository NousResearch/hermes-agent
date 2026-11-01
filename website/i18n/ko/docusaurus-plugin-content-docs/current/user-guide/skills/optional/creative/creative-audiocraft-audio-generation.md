---
title: "Audiocraft 오디오 생성 — AudioCraft: MusicGen 텍스트-음악, AudioGen 텍스트-사운드"
sidebar_label: "Audiocraft 오디오 생성"
description: "AudioCraft: MusicGen 텍스트-음악, AudioGen 텍스트-사운드"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Audiocraft 오디오 생성

AudioCraft: MusicGen 텍스트-음악, AudioGen 텍스트-사운드.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/creative/audiocraft-audio-generation`으로 설치 |
| 경로 | `optional-skills/creative/audiocraft-audio-generation` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `audiocraft`, `torch>=2.0.0`, `transformers>=4.30.0` |
| 플랫폼 | linux, macos |
| 태그 | `Multimodal`, `Audio Generation`, `Text-to-Music`, `Text-to-Audio`, `MusicGen` |
| 관련 스킬 | [`heartmula`](/docs/user-guide/skills/optional/creative/creative-heartmula), [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보게 되는 내용입니다.
:::

# AudioCraft: 오디오 생성

MusicGen, AudioGen, EnCodec을 사용해 텍스트로 음악과 오디오를 생성하는 방법을 안내합니다.

## AudioCraft를 사용할 때

**다음과 같은 경우 AudioCraft를 사용하세요:**
- 텍스트 설명으로 음악을 생성해야 할 때
- 음향 효과와 환경 오디오를 만들 때
- 음악 생성 애플리케이션을 구축할 때
- 멜로디 조건부 음악 생성이 필요할 때
- 스테레오 오디오 출력을 원할 때
- 스타일 전환을 사용한 제어 가능한 음악 생성이 필요할 때

**주요 기능:**
- **MusicGen**: 멜로디 조건부 기능을 지원하는 텍스트-음악 생성
- **AudioGen**: 텍스트-음향 효과 생성
- **EnCodec**: 고충실도 신경망 오디오 코덱
- **다양한 모델 크기**: Small(3억)부터 Large(33억)까지
- **스테레오 지원**: 완전한 스테레오 오디오 생성
- **스타일 조건부 기능**: 참조 기반 생성을 위한 MusicGen-Style

**다음 대안을 사용하세요:**
- **Stable Audio**: 더 긴 상업용 음악 생성
- **Bark**: 음악/음향 효과가 포함된 텍스트-음성 변환
- **Riffusion**: 스펙트로그램 기반 음악 생성
- **OpenAI Jukebox**: 가사가 포함된 원시 오디오 생성

## 빠른 시작

### 설치

```bash
# From PyPI
pip install audiocraft

# From GitHub (latest)
pip install git+https://github.com/facebookresearch/audiocraft.git

# Or use HuggingFace Transformers
pip install transformers torch torchaudio
```

### 기본 텍스트-음악 생성(AudioCraft)

```python
import torchaudio
from audiocraft.models import MusicGen

# Load model
model = MusicGen.get_pretrained('facebook/musicgen-small')

# Set generation parameters
model.set_generation_params(
    duration=8,  # seconds
    top_k=250,
    temperature=1.0
)

# Generate from text
descriptions = ["happy upbeat electronic dance music with synths"]
wav = model.generate(descriptions)

# Save audio
torchaudio.save("output.wav", wav[0].cpu(), sample_rate=32000)
```

### HuggingFace Transformers 사용

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy

# Load model and processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
model.to("cuda")

# Generate music
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

# Save
sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("output.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
```

### AudioGen을 사용한 텍스트-사운드

```python
from audiocraft.models import AudioGen

# Load AudioGen
model = AudioGen.get_pretrained('facebook/audiogen-medium')

model.set_generation_params(duration=5)

# Generate sound effects
descriptions = ["dog barking in a park with birds chirping"]
wav = model.generate(descriptions)

torchaudio.save("sound.wav", wav[0].cpu(), sample_rate=16000)
```

## 핵심 개념

### 아키텍처 개요

<!-- ascii-guard-ignore -->
```
AudioCraft Architecture:
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

### 모델 변형

| 모델 | 크기 | 설명 | 사용 사례 |
|-------|------|-------------|----------|
| `musicgen-small` | 300M | 텍스트-음악 | 빠른 생성 |
| `musicgen-medium` | 1.5B | 텍스트-음악 | 균형 잡힌 품질 |
| `musicgen-large` | 3.3B | 텍스트-음악 | 최고 품질 |
| `musicgen-melody` | 1.5B | 텍스트 + 멜로디 | 멜로디 조건부 생성 |
| `musicgen-melody-large` | 3.3B | 텍스트 + 멜로디 | 최고의 멜로디 품질 |
| `musicgen-stereo-*` | 다양함 | 스테레오 출력 | 스테레오 생성 |
| `musicgen-style` | 1.5B | 스타일 전환 | 참조 기반 |
| `audiogen-medium` | 1.5B | 텍스트-사운드 | 음향 효과 |

### 생성 매개변수

| 매개변수 | 기본값 | 설명 |
|-----------|---------|-------------|
| `duration` | 8.0 | 길이(초)(1~120) |
| `top_k` | 250 | Top-k 샘플링 |
| `top_p` | 0.0 | 뉴클리어스 샘플링(0 = 비활성화) |
| `temperature` | 1.0 | 샘플링 온도 |
| `cfg_coef` | 3.0 | 분류기 없는 가이던스 |

## MusicGen 사용법

### 텍스트-음악 생성

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained('facebook/musicgen-medium')

# Configure generation
model.set_generation_params(
    duration=30,          # Up to 30 seconds
    top_k=250,            # Sampling diversity
    top_p=0.0,            # 0 = use top_k only
    temperature=1.0,      # Creativity (higher = more varied)
    cfg_coef=3.0          # Text adherence (higher = stricter)
)

# Generate multiple samples
descriptions = [
    "epic orchestral soundtrack with strings and brass",
    "chill lo-fi hip hop beat with jazzy piano",
    "energetic rock song with electric guitar"
]

# Generate (returns [batch, channels, samples])
wav = model.generate(descriptions)

# Save each
for i, audio in enumerate(wav):
    torchaudio.save(f"music_{i}.wav", audio.cpu(), sample_rate=32000)
```

### 멜로디 조건부 생성

```python
from audiocraft.models import MusicGen
import torchaudio

# Load melody model
model = MusicGen.get_pretrained('facebook/musicgen-melody')
model.set_generation_params(duration=30)

# Load melody audio
melody, sr = torchaudio.load("melody.wav")

# Generate with melody conditioning
descriptions = ["acoustic guitar folk song"]
wav = model.generate_with_chroma(descriptions, melody, sr)

torchaudio.save("melody_conditioned.wav", wav[0].cpu(), sample_rate=32000)
```

### 스테레오 생성

```python
from audiocraft.models import MusicGen

# Load stereo model
model = MusicGen.get_pretrained('facebook/musicgen-stereo-medium')
model.set_generation_params(duration=15)

descriptions = ["ambient electronic music with wide stereo panning"]
wav = model.generate(descriptions)

# wav shape: [batch, 2, samples] for stereo
print(f"Stereo shape: {wav.shape}")  # [1, 2, 480000]
torchaudio.save("stereo.wav", wav[0].cpu(), sample_rate=32000)
```

### 오디오 이어 붙이기

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

# Load audio to continue
import torchaudio
audio, sr = torchaudio.load("intro.wav")

# Process with text and audio
inputs = processor(
    audio=audio.squeeze().numpy(),
    sampling_rate=sr,
    text=["continue with a epic chorus"],
    padding=True,
    return_tensors="pt"
)

# Generate continuation
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=512)
```

## MusicGen-Style 사용법

### 스타일 조건부 생성

```python
from audiocraft.models import MusicGen

# Load style model
model = MusicGen.get_pretrained('facebook/musicgen-style')

# Configure generation with style
model.set_generation_params(
    duration=30,
    cfg_coef=3.0,
    cfg_coef_beta=5.0  # Style influence
)

# Configure style conditioner
model.set_style_conditioner_params(
    eval_q=3,          # RVQ quantizers (1-6)
    excerpt_length=3.0  # Style excerpt length
)

# Load style reference
style_audio, sr = torchaudio.load("reference_style.wav")

# Generate with text + style
descriptions = ["upbeat dance track"]
wav = model.generate_with_style(descriptions, style_audio, sr)
```

### 스타일만 사용한 생성(텍스트 없음)

```python
# Generate matching style without text prompt
model.set_generation_params(
    duration=30,
    cfg_coef=3.0,
    cfg_coef_beta=None  # Disable double CFG for style-only
)

wav = model.generate_with_style([None], style_audio, sr)
```

## AudioGen 사용법

### 음향 효과 생성

```python
from audiocraft.models import AudioGen
import torchaudio

model = AudioGen.get_pretrained('facebook/audiogen-medium')
model.set_generation_params(duration=10)

# Generate various sounds
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

# Load EnCodec
model = CompressionModel.get_pretrained('facebook/encodec_32khz')

# Load audio
wav, sr = torchaudio.load("audio.wav")

# Ensure correct sample rate
if sr != 32000:
    resampler = torchaudio.transforms.Resample(sr, 32000)
    wav = resampler(wav)

# Encode to tokens
with torch.no_grad():
    encoded = model.encode(wav.unsqueeze(0))
    codes = encoded[0]  # Audio codes

# Decode back to audio
with torch.no_grad():
    decoded = model.decode(codes)

torchaudio.save("reconstructed.wav", decoded[0].cpu(), sample_rate=32000)
```

## 일반적인 워크플로

### 워크플로 1: 음악 생성 파이프라인

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

# Usage
generator = MusicGenerator()
audio = generator.generate(
    "epic cinematic orchestral music",
    duration=30,
    temperature=1.0
)
generator.save(audio, "epic_music.wav")
```

### 워크플로 2: 사운드 디자인 일괄 처리

```python
import json
from pathlib import Path
from audiocraft.models import AudioGen
import torchaudio

def batch_generate_sounds(sound_specs, output_dir):
    """
    Generate multiple sounds from specifications.

    Args:
        sound_specs: list of {"name": str, "description": str, "duration": float}
        output_dir: output directory path
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

# Usage
sounds = [
    {"name": "explosion", "description": "massive explosion with debris", "duration": 3},
    {"name": "footsteps", "description": "footsteps on wooden floor", "duration": 5},
    {"name": "door", "description": "wooden door creaking and closing", "duration": 2}
]

results = batch_generate_sounds(sounds, "sound_effects/")
```

### 워크플로 3: Gradio 데모

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

    # Save to temp file
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
# Use smaller model
model = MusicGen.get_pretrained('facebook/musicgen-small')

# Clear cache between generations
torch.cuda.empty_cache()

# Generate shorter durations
model.set_generation_params(duration=10)  # Instead of 30

# Use half precision
model = model.half()
```

### 일괄 처리 효율

```python
# Process multiple prompts at once (more efficient)
descriptions = ["prompt1", "prompt2", "prompt3", "prompt4"]
wav = model.generate(descriptions)  # Single batch

# Instead of
for desc in descriptions:
    wav = model.generate([desc])  # Multiple batches (slower)
```

### GPU 메모리 요구 사항

| 모델 | FP32 VRAM | FP16 VRAM |
|-------|-----------|-----------|
| musicgen-small | ~4GB | ~2GB |
| musicgen-medium | ~8GB | ~4GB |
| musicgen-large | ~16GB | ~8GB |

## 일반적인 문제

| 문제 | 해결 방법 |
|-------|----------|
| CUDA OOM | 더 작은 모델을 사용하고 duration을 줄이세요 |
| 품질 저하 | cfg_coef을 높이고 프롬프트를 개선하세요 |
| 생성 결과가 너무 짧음 | 최대 duration 설정을 확인하세요 |
| 오디오 아티팩트 | 다른 temperature를 시도하세요 |
| 스테레오가 작동하지 않음 | 스테레오 모델 변형을 사용하세요 |

## 참고 자료

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/audiocraft-audio-generation/references/advanced-usage.md)** - 학습, 파인튜닝, 배포
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/creative/audiocraft-audio-generation/references/troubleshooting.md)** - 일반적인 문제와 해결 방법

## 리소스

- **GitHub**: https://github.com/facebookresearch/audiocraft
- **논문(MusicGen)**: https://arxiv.org/abs/2306.05284
- **논문(AudioGen)**: https://arxiv.org/abs/2209.15352
- **HuggingFace**: https://huggingface.co/facebook/musicgen-small
- **데모**: https://huggingface.co/spaces/facebook/MusicGen

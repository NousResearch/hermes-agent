---
title: "Heartmula — HeartMuLa: 가사와 태그를 기반으로 Suno처럼 노래 생성"
sidebar_label: "Heartmula"
description: "HeartMuLa: 가사와 태그를 기반으로 Suno처럼 노래 생성"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Heartmula

HeartMuLa: 가사와 태그를 기반으로 Suno처럼 노래를 생성합니다.

## 스킬 메타데이터 (Skill metadata)

| | |
|---|---|
| Source | Bundled (installed by default) |
| Path | `skills/media/heartmula` |
| Version | `1.0.0` |
| Platforms | linux, macos, windows |
| Tags | `music`, `audio`, `generation`, `ai`, `heartmula`, `heartcodec`, `lyrics`, `songs` |
| Related skills | `audiocraft` |

## 참조: 전체 SKILL.md (Reference: full SKILL.md)

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# HeartMuLa - Open-Source Music Generation

## 개요 (Overview)
HeartMuLa는 다국어를 지원하며 가사와 태그를 조건으로 음악을 생성하는 오픈소스 음악 파운데이션 모델 제품군(Apache-2.0)입니다. 가사와 태그로부터 전체 노래를 생성합니다. 오픈소스계의 Suno라고 할 수 있습니다. 다음을 포함합니다:
- **HeartMuLa** - 가사와 태그로부터 생성을 위한 음악 언어 모델 (3B/7B)
- **HeartCodec** - 고충실도 오디오 재구성을 위한 12.5Hz 음악 코덱
- **HeartTranscriptor** - Whisper 기반 가사 전사(Transcription)
- **HeartCLAP** - 오디오-텍스트 정렬 모델

## 사용 시기 (When to Use)
- 사용자가 텍스트 설명을 기반으로 음악/노래를 생성하고 싶을 때
- 사용자가 오픈소스 Suno 대안을 원할 때
- 사용자가 로컬/오프라인 환경에서 음악 생성을 원할 때
- 사용자가 HeartMuLa, heartlib 또는 AI 음악 생성에 대해 물어볼 때

## 하드웨어 요구사항 (Hardware Requirements)
- **최소 (Minimum)**: `--lazy_load true` 사용 시 8GB VRAM (모델을 순차적으로 로드/언로드함)
- **권장 (Recommended)**: 편안한 단일 GPU 사용을 위해 16GB+ VRAM
- **다중 GPU (Multi-GPU)**: `--mula_device cuda:0 --codec_device cuda:1`를 사용하여 GPU 간에 분할 가능
- lazy_load 적용 시 3B 모델은 최대 ~6.2GB VRAM 사용

## 설치 단계 (Installation Steps)

### 1. 저장소 클론 (Clone Repository)
```bash
cd ~/  # 또는 원하는 디렉토리
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib
```

### 2. 가상 환경 생성 (Create Virtual Environment - Python 3.10 필요)
```bash
uv venv --python 3.10 .venv
. .venv/bin/activate
uv pip install -e .
```

### 3. 종속성 호환성 문제 수정 (Fix Dependency Compatibility Issues)

**중요 (IMPORTANT)**: 2026년 2월 기준으로 고정된 종속성 버전들이 최신 패키지와 충돌을 일으킵니다. 다음 수정을 적용하세요:

```bash
# datasets 업그레이드 (구 버전은 현재 pyarrow와 호환되지 않음)
uv pip install --upgrade datasets

# transformers 업그레이드 (huggingface-hub 1.x 호환성을 위해 필요)
uv pip install --upgrade transformers
```

### 4. 소스 코드 패치 (Patch Source Code - transformers 5.x에 필수)

**패치 1 - RoPE 캐시 수정 (RoPE cache fix)** - `src/heartlib/heartmula/modeling_heartmula.py`:

`HeartMuLa` 클래스의 `setup_caches` 메서드에서, `reset_caches`의 try/except 블록 이후 그리고 `with device:` 블록 전에 RoPE 재초기화 코드를 추가하세요:

```python
# 메타 디바이스 로드 중 건너뛴 RoPE 캐시를 재초기화합니다.
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
for module in self.modules():
    if isinstance(module, Llama3ScaledRoPE) and not module.is_cache_built:
        module.rope_init()
        module.to(device)
```

**이유 (Why)**: `from_pretrained`는 처음에 메타 디바이스에 모델을 생성합니다; `Llama3ScaledRoPE.rope_init()`는 메타 텐서에서의 캐시 빌드를 건너뛰고, 가중치가 실제 디바이스에 로드된 이후 다시 빌드하지 않습니다.

**패치 2 - HeartCodec 로딩 수정 (HeartCodec loading fix)** - `src/heartlib/pipelines/music_generation.py`:

모든 `HeartCodec.from_pretrained()` 호출(총 2곳: `__init__`의 즉시 로드(eager load)와 `codec` 프로퍼티의 지연 로드(lazy load))에 `ignore_mismatched_sizes=True`를 추가하세요.

**이유 (Why)**: VQ 코드북 `initted` 버퍼가 체크포인트에서는 `[1]` 형태(shape)지만, 모델에서는 `[]` 형태를 가집니다. 데이터는 같고 스칼라와 0차원 텐서의 차이일 뿐이므로 무시해도 안전합니다.

### 5. 모델 체크포인트 다운로드 (Download Model Checkpoints)
```bash
cd heartlib  # 프로젝트 루트
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123'
```

3개 모두 병렬로 다운로드할 수 있습니다. 총 크기는 수 GB입니다.

## GPU / CUDA

HeartMuLa는 기본적으로 CUDA를 사용합니다 (`--mula_device cuda --codec_device cuda`). 사용자에게 PyTorch CUDA 지원이 설치된 NVIDIA GPU가 있다면 별도의 설정이 필요 없습니다.

- 설치된 `torch==2.4.1`은 기본적으로 CUDA 12.1 지원을 포함합니다.
- `torchtune`이 `0.4.0+cpu` 버전을 보고할 수 있지만, 이는 단지 패키지 메타데이터일 뿐이며 여전히 PyTorch를 통해 CUDA를 사용합니다.
- GPU가 사용 중인지 확인하려면 출력에서 "CUDA memory"가 포함된 줄을 찾으세요(예: "CUDA memory before unloading: 6.20 GB").
- **GPU가 없나요? (No GPU?)** `--mula_device cpu --codec_device cpu`를 통해 CPU 모드로 실행할 수 있지만, 생성 속도가 **극도로 느려질 것**입니다(GPU에서 약 4분 걸리는 한 곡이 30~60분 이상 소요될 수 있음). CPU 모드는 또한 상당한 RAM(~12GB+ 여유)을 요구합니다. 사용자에게 NVIDIA GPU가 없다면 클라우드 GPU 서비스(Google Colab T4 무료 티어, Lambda Labs 등) 또는 https://heartmula.github.io/ 의 온라인 데모를 추천하세요.

## 사용법 (Usage)

### 기본 생성 (Basic Generation)
```bash
cd heartlib
. .venv/bin/activate
python ./examples/run_music_generation.py \
  --model_path=./ckpt \
  --version="3B" \
  --lyrics="./assets/lyrics.txt" \
  --tags="./assets/tags.txt" \
  --save_path="./assets/output.mp3" \
  --lazy_load true
```

### 입력 포맷팅 (Input Formatting)

**태그 (Tags)** (쉼표로 구분, 공백 없음):
```
piano,happy,wedding,synthesizer,romantic
```
또는
```
rock,energetic,guitar,drums,male-vocal
```

**가사 (Lyrics)** (대괄호로 된 구조 태그 사용):
```
[Intro]

[Verse]
여기에 가사 입력...

[Chorus]
코러스 가사...

[Bridge]
브릿지 가사...

[Outro]
```

### 주요 파라미터 (Key Parameters)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max_audio_length_ms` | 240000 | 최대 길이(ms) (240s = 4분) |
| `--topk` | 50 | Top-k 샘플링 |
| `--temperature` | 1.0 | 샘플링 온도 (Temperature) |
| `--cfg_scale` | 1.5 | Classifier-free guidance 스케일 |
| `--lazy_load` | false | 필요시 모델 로드/언로드 (VRAM 절약) |
| `--mula_dtype` | bfloat16 | HeartMuLa를 위한 데이터 타입 (bf16 권장) |
| `--codec_dtype` | float32 | HeartCodec을 위한 데이터 타입 (품질을 위해 fp32 권장) |

### 성능 (Performance)
- RTF (Real-Time Factor) ≈ 1.0 — 4분 길이의 노래를 생성하는 데 약 4분이 소요됩니다.
- 출력(Output): MP3, 48kHz 스테레오, 128kbps

## 주의사항 (Pitfalls)
1. **HeartCodec에 bf16을 사용하지 마세요 (Do NOT use bf16 for HeartCodec)** — 오디오 품질이 저하됩니다. fp32(기본값)를 사용하세요.
2. **태그가 무시될 수 있습니다 (Tags may be ignored)** — 알려진 문제(#90). 가사가 지배적인 경향이 있으니 태그 순서를 바꿔보며 실험해 보세요.
3. **macOS에서는 Triton을 사용할 수 없습니다 (Triton not available on macOS)** — GPU 가속은 Linux/CUDA 전용입니다.
4. 상류(upstream) 이슈 리포트에서 **RTX 5080 비호환성(incompatibility)**이 보고되었습니다.
5. 종속성 고정 충돌은 위에 설명된 수동 업그레이드 및 패치를 요구합니다.

## 링크 (Links)
- Repo: https://github.com/HeartMuLa/heartlib
- Models: https://huggingface.co/HeartMuLa
- Paper: https://arxiv.org/abs/2601.10547
- License: Apache-2.0

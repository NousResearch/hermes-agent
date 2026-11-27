---
title: "Heartmula — HeartMuLa: 가사 + 태그로 Suno와 유사한 노래 생성"
sidebar_label: "Heartmula"
description: "HeartMuLa: 가사 + 태그로 Suno와 유사한 노래 생성"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Heartmula

HeartMuLa: 가사 + 태그로 Suno와 유사한 노래를 생성합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | Optional — `hermes skills install official/creative/heartmula`로 설치 |
| 경로 | `optional-skills/creative/heartmula` |
| 버전 | `1.0.0` |
| 작성자 | Teknium (teknium1), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `music`, `audio`, `generation`, `ai`, `heartmula`, `heartcodec`, `lyrics`, `songs` |
| 관련 스킬 | [`audiocraft-audio-generation`](/docs/user-guide/skills/optional/creative/creative-audiocraft-audio-generation), [`songwriting-and-ai-music`](/docs/user-guide/skills/bundled/creative/creative-songwriting-and-ai-music) |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성 상태일 때 에이전트가 보게 되는 지침입니다.
:::

# HeartMuLa - 오픈 소스 음악 생성

## 개요
HeartMuLa는 가사와 태그를 조건으로 음악을 생성하고 다국어를 지원하는 오픈 소스 음악 파운데이션 모델 제품군입니다 (Apache-2.0). 가사와 태그로 완성된 노래를 생성합니다. 오픈 소스용 Suno에 해당합니다. 다음을 포함합니다.
- **HeartMuLa** - 가사와 태그로 생성하는 음악 언어 모델 (3B/7B)
- **HeartCodec** - 고충실도 오디오 복원을 위한 12.5Hz 음악 코덱
- **HeartTranscriptor** - Whisper 기반 가사 전사 모델
- **HeartCLAP** - 오디오-텍스트 정렬 모델

## 사용 시점
- 텍스트 설명으로 음악/노래를 생성하려는 경우
- 오픈 소스 Suno 대안을 찾는 경우
- 로컬/오프라인 음악 생성을 원하는 경우
- HeartMuLa, heartlib 또는 AI 음악 생성에 대해 묻는 경우

## 하드웨어 요구 사항
- **최소**: `--lazy_load true` 사용 시 VRAM 8GB (모델을 순차적으로 로드/언로드)
- **권장**: 한 대의 GPU를 편안하게 사용하려면 VRAM 16GB 이상
- **다중 GPU**: GPU 간에 나누려면 `--mula_device cuda:0 --codec_device cuda:1` 사용
- lazy_load를 사용하는 3B 모델은 VRAM이 최대 약 6.2GB까지 사용됩니다.

## 설치 단계

### 1. 저장소 클론
```bash
cd ~/  # or desired directory
git clone https://github.com/HeartMuLa/heartlib.git
cd heartlib
```

### 2. 가상 환경 생성 (Python 3.10 필요)
```bash
uv venv --python 3.10 .venv
. .venv/bin/activate
uv pip install -e .
```

### 3. 의존성 호환성 문제 수정

**중요**: 2026년 2월 기준으로 고정된 의존성이 최신 패키지와 충돌합니다. 다음 수정 사항을 적용하세요.

```bash
# Upgrade datasets (old version incompatible with current pyarrow)
uv pip install --upgrade datasets

# Upgrade transformers (needed for huggingface-hub 1.x compatibility)
uv pip install --upgrade transformers
```

### 4. 소스 코드 패치 (transformers 5.x에 필요)

**패치 1 - RoPE 캐시 수정** (`src/heartlib/heartmula/modeling_heartmula.py`):

`HeartMuLa` 클래스의 `setup_caches` 메서드에서 `reset_caches` try/except 블록 뒤, `with device:` 블록 앞에 RoPE 재초기화 코드를 추가합니다.

```python
# Re-initialize RoPE caches that were skipped during meta-device loading
from torchtune.models.llama3_1._position_embeddings import Llama3ScaledRoPE
for module in self.modules():
    if isinstance(module, Llama3ScaledRoPE) and not module.is_cache_built:
        module.rope_init()
        module.to(device)
```

**이유**: `from_pretrained`는 먼저 meta 디바이스에서 모델을 생성합니다. `Llama3ScaledRoPE.rope_init()`는 meta 텐서에서 캐시 생성을 건너뛰고, 이후 가중치가 실제 디바이스에 로드된 뒤에도 캐시를 다시 생성하지 않습니다.

**패치 2 - HeartCodec 로딩 수정** (`src/heartlib/pipelines/music_generation.py`):

모든 `HeartCodec.from_pretrained()` 호출에 `ignore_mismatched_sizes=True`를 추가합니다 (2곳: `__init__`의 즉시 로드와 `codec` 프로퍼티의 지연 로드).

**이유**: VQ 코드북의 `initted` 버퍼가 체크포인트에서는 `[1]`, 모델에서는 `[]` 형태입니다. 같은 데이터이며 스칼라와 0차원 텐서의 차이일 뿐입니다. 무시해도 안전합니다.

### 5. 모델 체크포인트 다운로드
```bash
cd heartlib  # project root
hf download --local-dir './ckpt' 'HeartMuLa/HeartMuLaGen'
hf download --local-dir './ckpt/HeartMuLa-oss-3B' 'HeartMuLa/HeartMuLa-oss-3B-happy-new-year'
hf download --local-dir './ckpt/HeartCodec-oss' 'HeartMuLa/HeartCodec-oss-20260123'
```

세 가지 모두 병렬로 다운로드할 수 있습니다. 전체 크기는 수 GB입니다.

## GPU / CUDA

HeartMuLa는 기본적으로 CUDA를 사용합니다 (`--mula_device cuda --codec_device cuda`). PyTorch CUDA 지원이 설치된 NVIDIA GPU가 있다면 별도의 설정은 필요하지 않습니다.

- 설치된 `torch==2.4.1`에는 CUDA 12.1 지원이 기본으로 포함되어 있습니다.
- `torchtune`에서 버전이 `0.4.0+cpu`로 표시될 수 있지만, 이는 패키지 메타데이터일 뿐이며 PyTorch를 통해 CUDA를 사용합니다.
- GPU가 사용되는지 확인하려면 출력에서 "CUDA memory" 줄을 찾으세요 (예: "CUDA memory before unloading: 6.20 GB").
- **GPU가 없나요?** `--mula_device cpu --codec_device cpu`로 CPU에서 실행할 수 있지만, 생성 속도가 **극도로 느릴 것**으로 예상해야 합니다 (GPU에서 약 4분인 노래 한 곡이 30-60분 이상 걸릴 수 있습니다). CPU 모드에는 상당한 RAM(여유 공간 약 12GB 이상)도 필요합니다. NVIDIA GPU가 없다면 클라우드 GPU 서비스(Google Colab 무료 등급의 T4, Lambda Labs 등) 또는 https://heartmula.github.io/의 온라인 데모를 권장하세요.

## 사용법

### 기본 생성
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

### 입력 형식

**태그** (공백 없이 쉼표로 구분):
```
piano,happy,wedding,synthesizer,romantic
```
또는
```
rock,energetic,guitar,drums,male-vocal
```

**가사** (대괄호로 묶은 구조 태그 사용):
```
[Intro]

[Verse]
Your lyrics here...

[Chorus]
Chorus lyrics...

[Bridge]
Bridge lyrics...

[Outro]
```

### 주요 매개변수
| 매개변수 | 기본값 | 설명 |
|-----------|---------|-------------|
| `--max_audio_length_ms` | 240000 | 최대 길이(ms) (240초 = 4분) |
| `--topk` | 50 | Top-k 샘플링 |
| `--temperature` | 1.0 | 샘플링 온도 |
| `--cfg_scale` | 1.5 | 분류기 없는 가이던스 스케일 |
| `--lazy_load` | false | 필요할 때 모델 로드/언로드 (VRAM 절약) |
| `--mula_dtype` | bfloat16 | HeartMuLa의 dtype (bf16 권장) |
| `--codec_dtype` | float32 | HeartCodec의 dtype (품질을 위해 fp32 권장) |

### 성능
- RTF (실시간 계수) ≈ 1.0 — 4분짜리 노래를 생성하는 데 약 4분이 걸립니다.
- 출력: MP3, 48kHz 스테레오, 128kbps

## 주의할 점
1. **HeartCodec에 bf16을 사용하지 마세요** — 오디오 품질이 저하됩니다. fp32(기본값)를 사용하세요.
2. **태그가 무시될 수 있습니다** — 알려진 문제 (#90)입니다. 가사가 우세한 경향이 있으므로 태그 순서를 바꿔 가며 실험해 보세요.
3. **macOS에서는 Triton을 사용할 수 없습니다** — GPU 가속은 Linux/CUDA에서만 가능합니다.
4. **RTX 5080 비호환성**이 업스트림 이슈에서 보고되었습니다.
5. 의존성 고정 충돌을 해결하려면 위에 설명한 수동 업그레이드와 패치가 필요합니다.

## 링크
- 저장소: https://github.com/HeartMuLa/heartlib
- 모델: https://huggingface.co/HeartMuLa
- 논문: https://arxiv.org/abs/2601.10547
- 라이선스: Apache-2.0

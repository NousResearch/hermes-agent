---
title: "Stable Diffusion Image Generation"
sidebar_label: "Stable Diffusion Image Generation"
description: "HuggingFace Diffusers를 통한 최신 텍스트-이미지 Stable Diffusion 모델"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Stable Diffusion 이미지 생성

HuggingFace Diffusers 라이브러리를 통한 Stable Diffusion 모델을 활용한 최신 텍스트-이미지 생성 기술입니다. 텍스트 프롬프트에서 이미지를 생성하거나, 이미지-이미지 변환, 인페인팅, 사용자 지정 디퓨전 파이프라인을 구축할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/stable-diffusion`로 설치 |
| 경로 | `optional-skills/mlops/stable-diffusion` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `diffusers>=0.30.0`, `transformers>=4.41.0`, `accelerate>=0.31.0`, `torch>=2.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Image Generation`, `Stable Diffusion`, `Diffusers`, `Text-to-Image`, `Multimodal`, `Computer Vision` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# Stable Diffusion 이미지 생성

HuggingFace Diffusers 라이브러리를 사용해 Stable Diffusion으로 이미지를 생성하기 위한 포괄적인 가이드입니다.

## Stable Diffusion을 사용하는 경우

**다음과 같은 경우에 Stable Diffusion을 사용하세요:**
- 텍스트 설명에서 이미지를 생성할 때
- 이미지-이미지 변환 (스타일 전이, 향상)을 수행할 때
- 인페인팅 (마스킹된 영역 채우기)
- 아웃페인팅 (경계 너머로 이미지 확장하기)
- 기존 이미지의 변형 생성
- 사용자 정의 이미지 생성 워크플로우 구축

**주요 기능:**
- **텍스트-이미지**: 자연어 프롬프트에서 이미지 생성
- **이미지-이미지**: 텍스트 가이드에 따라 기존 이미지 변환
- **인페인팅**: 문맥을 인식하여 마스킹된 영역을 콘텐츠로 채움
- **ControlNet**: 공간적 조건화(윤곽선, 포즈, 깊이) 추가
- **LoRA 지원**: 효율적인 파인튜닝 및 스타일 적응
- **다중 모델**: SD 1.5, SDXL, SD 3.0, Flux 지원

**다음을 대신 사용해 보세요:**
- **DALL-E 3**: GPU 없는 API 기반 생성의 경우
- **Midjourney**: 예술적이고 스타일라이즈된 결과물을 위한 경우
- **Imagen**: Google Cloud 연동을 위한 경우
- **Leonardo.ai**: 웹 기반 창작 워크플로우의 경우

## 빠른 시작

### 설치

```bash
pip install diffusers transformers accelerate torch
pip install xformers  # 선택사항: 메모리 효율적인 attention
```

### 기본 텍스트-이미지

```python
from diffusers import DiffusionPipeline
import torch

# 파이프라인 로드 (모델 타입 자동 감지)
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# 이미지 생성
image = pipe(
    "A serene mountain landscape at sunset, highly detailed",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

### SDXL 사용 (고품질)

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# 메모리 최적화 활성화
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A futuristic city with flying cars, cinematic lighting",
    height=1024,
    width=1024,
    num_inference_steps=30
).images[0]
```

## 아키텍처 개요

### 세 가지 축(Three-pillar) 디자인

Diffusers는 세 가지 핵심 구성요소를 중심으로 구축되었습니다:

<!-- ascii-guard-ignore -->
```
파이프라인 (오케스트레이션)
├── 모델 (신경망)
│   ├── UNet / Transformer (노이즈 예측)
│   ├── VAE (잠재 공간 인코딩/디코딩)
│   └── Text Encoder (CLIP/T5)
└── 스케줄러 (디노이징 알고리즘)
```
<!-- ascii-guard-ignore-end -->

### 파이프라인 추론 흐름

```
텍스트 프롬프트 → 텍스트 인코더 → 텍스트 임베딩
                                      ↓
랜덤 노이즈 → [디노이징 루프] ← 스케줄러
                       ↓
                   예측된 노이즈
                       ↓
               VAE 디코더 → 최종 이미지
```

## 핵심 개념

### 파이프라인 (Pipelines)

파이프라인은 전체 워크플로우를 오케스트레이션합니다:

| 파이프라인 | 목적 |
|----------|---------|
| `StableDiffusionPipeline` | 텍스트-이미지 (SD 1.x/2.x) |
| `StableDiffusionXLPipeline` | 텍스트-이미지 (SDXL) |
| `StableDiffusion3Pipeline` | 텍스트-이미지 (SD 3.0) |
| `FluxPipeline` | 텍스트-이미지 (Flux 모델들) |
| `StableDiffusionImg2ImgPipeline` | 이미지-이미지 |
| `StableDiffusionInpaintPipeline` | 인페인팅 |

### 스케줄러 (Schedulers)

스케줄러는 디노이징 과정을 제어합니다:

| 스케줄러 | 단계 수 | 품질 | 사용 사례 |
|-----------|-------|---------|----------|
| `EulerDiscreteScheduler` | 20-50 | 좋음 | 기본 선택 |
| `EulerAncestralDiscreteScheduler` | 20-50 | 좋음 | 더 많은 변형 |
| `DPMSolverMultistepScheduler` | 15-25 | 훌륭함 | 빠름, 고품질 |
| `DDIMScheduler` | 50-100 | 좋음 | 결정론적 |
| `LCMScheduler` | 4-8 | 좋음 | 매우 빠름 |
| `UniPCMultistepScheduler` | 15-25 | 훌륭함 | 빠른 수렴 |

### 스케줄러 교체

```python
from diffusers import DPMSolverMultistepScheduler

# 더 빠른 생성을 위한 교체
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# 이제 더 적은 단계로 생성
image = pipe(prompt, num_inference_steps=20).images[0]
```

## 생성 매개변수

### 주요 매개변수

| 매개변수 | 기본값 | 설명 |
|-----------|---------|-------------|
| `prompt` | 필수 | 원하는 이미지에 대한 텍스트 설명 |
| `negative_prompt` | None | 이미지에서 피해야 할 사항 |
| `num_inference_steps` | 50 | 디노이징 단계 수 (많을수록 = 품질 좋음) |
| `guidance_scale` | 7.5 | 프롬프트 준수도 (보통 7-12) |
| `height`, `width` | 512/1024 | 출력 크기 (8의 배수) |
| `generator` | None | 재현성을 위한 Torch 제너레이터 |
| `num_images_per_prompt` | 1 | 배치 크기 |

### 재현 가능한 생성

```python
import torch

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt="A cat wearing a top hat",
    generator=generator,
    num_inference_steps=50
).images[0]
```

### 네거티브 프롬프트

```python
image = pipe(
    prompt="Professional photo of a dog in a garden",
    negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
    guidance_scale=7.5
).images[0]
```

## 이미지-이미지 (Image-to-image)

텍스트 가이드에 따라 기존 이미지를 변환합니다:

```python
from diffusers import AutoPipelineForImage2Image
from PIL import Image

pipe = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("input.jpg").resize((512, 512))

image = pipe(
    prompt="A watercolor painting of the scene",
    image=init_image,
    strength=0.75,  # 얼마나 변환할지 (0-1)
    num_inference_steps=50
).images[0]
```

## 인페인팅 (Inpainting)

마스킹된 영역 채우기:

```python
from diffusers import AutoPipelineForInpainting
from PIL import Image

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("photo.jpg")
mask = Image.open("mask.png")  # 흰색 = 인페인팅할 영역

result = pipe(
    prompt="A red car parked on the street",
    image=image,
    mask_image=mask,
    num_inference_steps=50
).images[0]
```

## ControlNet

정확한 제어를 위해 공간적 조건을 추가합니다:

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

# 윤곽선 조건을 위한 ControlNet 로드
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# 제어를 위해 Canny 윤곽선 이미지를 사용
control_image = get_canny_image(input_image)

image = pipe(
    prompt="A beautiful house in the style of Van Gogh",
    image=control_image,
    num_inference_steps=30
).images[0]
```

### 사용 가능한 ControlNets

| ControlNet | 입력 타입 | 사용 사례 |
|------------|------------|----------|
| `canny` | 윤곽선 맵 | 구조 보존 |
| `openpose` | 포즈 스켈레톤 | 인간의 포즈 |
| `depth` | 깊이 맵 | 3D를 고려한 생성 |
| `normal` | 노말 맵 | 표면의 세부 사항 |
| `mlsd` | 선분 | 건축적인 선 |
| `scribble` | 거친 스케치 | 스케치-이미지 생성 |

## LoRA 어댑터

파인튜닝된 스타일 어댑터를 로드합니다:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# LoRA 가중치 로드
pipe.load_lora_weights("path/to/lora", weight_name="style.safetensors")

# LoRA 스타일로 생성
image = pipe("A portrait in the trained style").images[0]

# LoRA 강도 조절
pipe.fuse_lora(lora_scale=0.8)

# LoRA 언로드
pipe.unload_lora_weights()
```

### 다중 LoRA

```python
# 여러 개의 LoRA 로드
pipe.load_lora_weights("lora1", adapter_name="style")
pipe.load_lora_weights("lora2", adapter_name="character")

# 각각의 가중치 설정
pipe.set_adapters(["style", "character"], adapter_weights=[0.7, 0.5])

image = pipe("A portrait").images[0]
```

## 메모리 최적화

### CPU 오프로딩 활성화

```python
# 모델 CPU 오프로드 - 사용하지 않을 때 모델을 CPU로 이동시킵니다
pipe.enable_model_cpu_offload()

# 순차적 CPU 오프로드 - 더 공격적이고, 더 느립니다
pipe.enable_sequential_cpu_offload()
```

### Attention 슬라이싱

```python
# 청크 단위로 어텐션을 계산하여 메모리 감소
pipe.enable_attention_slicing()

# 또는 특정 청크 크기
pipe.enable_attention_slicing("max")
```

### xFormers 메모리 효율적인 attention

```python
# xformers 패키지가 필요합니다
pipe.enable_xformers_memory_efficient_attention()
```

### 큰 이미지를 위한 VAE 슬라이싱

```python
# 큰 이미지를 위해 잠재 공간을 타일 단위로 디코딩
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

## 모델 변형 (Model variants)

### 다른 정밀도로 로드

```python
# FP16 (GPU에 권장됨)
pipe = DiffusionPipeline.from_pretrained(
    "model-id",
    torch_dtype=torch.float16,
    variant="fp16"
)

# BF16 (더 나은 정밀도, Ampere 이상의 GPU 필요)
pipe = DiffusionPipeline.from_pretrained(
    "model-id",
    torch_dtype=torch.bfloat16
)
```

### 특정 컴포넌트 로드

```python
from diffusers import UNet2DConditionModel, AutoencoderKL

# 사용자 지정 VAE 로드
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# 파이프라인에 사용
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae=vae,
    torch_dtype=torch.float16
)
```

## 배치 생성

효율적으로 여러 이미지를 생성합니다:

```python
# 여러 개의 프롬프트
prompts = [
    "A cat playing piano",
    "A dog reading a book",
    "A bird painting a picture"
]

images = pipe(prompts, num_inference_steps=30).images

# 프롬프트당 여러 개의 이미지
images = pipe(
    "A beautiful sunset",
    num_images_per_prompt=4,
    num_inference_steps=30
).images
```

## 일반적인 워크플로우

### 워크플로우 1: 고품질 생성

```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

# 1. 최적화된 SDXL 로드
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# 2. 고품질 설정으로 생성
image = pipe(
    prompt="A majestic lion in the savanna, golden hour lighting, 8k, detailed fur",
    negative_prompt="blurry, low quality, cartoon, anime, sketch",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]
```

### 워크플로우 2: 빠른 프로토타이핑

```python
from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch

# 4-8 단계 생성을 위해 LCM 사용
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# 빠른 생성을 위한 LCM LoRA 로드
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()

# 약 1초 안에 생성
image = pipe(
    "A beautiful landscape",
    num_inference_steps=4,
    guidance_scale=1.0
).images[0]
```

## 흔한 문제들

**CUDA 메모리 부족:**
```python
# 메모리 최적화 활성화
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# 또는 낮은 정밀도 사용
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

**검은색/노이즈 이미지:**
```python
# VAE 구성 확인
# 필요한 경우 안전 검사 우회 사용
pipe.safety_checker = None

# 적절한 dtype 일관성 확인
pipe = pipe.to(dtype=torch.float16)
```

**느린 생성 속도:**
```python
# 더 빠른 스케줄러 사용
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 단계 수 감소
image = pipe(prompt, num_inference_steps=20).images[0]
```

## 참조 항목

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/stable-diffusion/references/advanced-usage.md)** - 맞춤형 파이프라인, 파인튜닝, 배포
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/stable-diffusion/references/troubleshooting.md)** - 흔한 문제와 해결책

## 리소스

- **문서**: https://huggingface.co/docs/diffusers
- **저장소**: https://github.com/huggingface/diffusers
- **모델 허브**: https://huggingface.co/models?library=diffusers
- **Discord**: https://discord.gg/diffusers

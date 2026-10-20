---
title: "Stable Diffusion — 텍스트-이미지 생성, 인페인팅 및 img2img"
sidebar_label: "Stable Diffusion"
description: "텍스트-이미지 생성, 인페인팅 및 img2img"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Stable Diffusion

텍스트-이미지 생성, 인페인팅 및 img2img.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/stable-diffusion`으로 설치 |
| 경로 | `optional-skills/mlops/stable-diffusion` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `diffusers>=0.30.0`, `transformers>=4.41.0`, `accelerate>=0.31.0`, `torch>=2.0.0` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Image Generation`, `Stable Diffusion`, `Diffusers`, `Text-to-Image`, `Multimodal`, `Computer Vision` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Stable Diffusion 이미지 생성

HuggingFace Diffusers 라이브러리를 사용하여 Stable Diffusion으로 이미지를 생성하는 안내서입니다.

## Stable Diffusion 사용 시점

**다음과 같은 경우 Stable Diffusion을 사용하세요:**
- 텍스트 설명에서 이미지 생성
- 텍스트 안내를 사용한 이미지-이미지 변환(스타일 전이, 품질 향상)
- 인페인팅(마스크된 영역 채우기)
- 아웃페인팅(경계 너머로 이미지 확장)
- 기존 이미지의 변형 생성
- 맞춤 이미지 생성 워크플로 구축

**다음과 같은 경우 대안을 사용하세요:**
- **DALL-E 3**: GPU 없이 API 기반으로 생성할 때
- **Midjourney**: 예술적이고 스타일화된 결과물을 원할 때
- **Imagen**: Google Cloud 통합이 필요할 때
- **Leonardo.ai**: 웹 기반의 창작 워크플로를 사용할 때

## 빠른 시작

### 설치

```bash
pip install diffusers transformers accelerate torch
pip install xformers  # Optional: memory-efficient attention
```

### 기본 텍스트-이미지 생성

```python
from diffusers import DiffusionPipeline
import torch

# Load pipeline (auto-detects model type)
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe.to("cuda")

# Generate image
image = pipe(
    "A serene mountain landscape at sunset, highly detailed",
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

image.save("output.png")
```

### SDXL 사용(더 높은 품질)

```python
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Enable memory optimization
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A futuristic city with flying cars, cinematic lighting",
    height=1024,
    width=1024,
    num_inference_steps=30
).images[0]
```

## 아키텍처 개요

### 세 기둥 설계

Diffusers는 세 가지 핵심 구성 요소를 중심으로 구축됩니다.

<!-- ascii-guard-ignore -->
```
Pipeline (orchestration)
├── Model (neural networks)
│   ├── UNet / Transformer (noise prediction)
│   ├── VAE (latent encoding/decoding)
│   └── Text Encoder (CLIP/T5)
└── Scheduler (denoising algorithm)
```
<!-- ascii-guard-ignore-end -->

### 파이프라인 추론 흐름

```
Text Prompt → Text Encoder → Text Embeddings
                                    ↓
Random Noise → [Denoising Loop] ← Scheduler
                      ↓
               Predicted Noise
                      ↓
              VAE Decoder → Final Image
```

## 핵심 개념

### 파이프라인

파이프라인은 전체 워크플로를 조정합니다.

| 파이프라인 | 용도 |
|----------|---------|
| `StableDiffusionPipeline` | 텍스트-이미지(SD 1.x/2.x) |
| `StableDiffusionXLPipeline` | 텍스트-이미지(SDXL) |
| `StableDiffusion3Pipeline` | 텍스트-이미지(SD 3.0) |
| `FluxPipeline` | 텍스트-이미지(Flux 모델) |
| `StableDiffusionImg2ImgPipeline` | 이미지-이미지 |
| `StableDiffusionInpaintPipeline` | 인페인팅 |

### 스케줄러

스케줄러는 디노이징 프로세스를 제어합니다.

| 스케줄러 | 단계 | 품질 | 사용 사례 |
|----------|-------|---------|----------|
| `EulerDiscreteScheduler` | 20-50 | 양호 | 기본 선택 |
| `EulerAncestralDiscreteScheduler` | 20-50 | 양호 | 더 많은 변형 |
| `DPMSolverMultistepScheduler` | 15-25 | 우수 | 빠르고 높은 품질 |
| `DDIMScheduler` | 50-100 | 양호 | 결정론적 |
| `LCMScheduler` | 4-8 | 양호 | 매우 빠름 |
| `UniPCMultistepScheduler` | 15-25 | 우수 | 빠른 수렴 |

### 스케줄러 교체

```python
from diffusers import DPMSolverMultistepScheduler

# Swap for faster generation
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# Now generate with fewer steps
image = pipe(prompt, num_inference_steps=20).images[0]
```

## 생성 매개변수

### 주요 매개변수

| 매개변수 | 기본값 | 설명 |
|-----------|---------|-------------|
| `prompt` | Required | 원하는 이미지의 텍스트 설명 |
| `negative_prompt` | None | 이미지에서 피할 요소 |
| `num_inference_steps` | 50 | 디노이징 단계(많을수록 품질 향상) |
| `guidance_scale` | 7.5 | 프롬프트 반영도(일반적으로 7-12) |
| `height`, `width` | 512/1024 | 출력 크기(8의 배수) |
| `generator` | None | 재현성을 위한 Torch 생성기 |
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

## 이미지-이미지

텍스트 안내를 사용하여 기존 이미지를 변환합니다.

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
    strength=0.75,  # How much to transform (0-1)
    num_inference_steps=50
).images[0]
```

## 인페인팅

마스크된 영역을 채웁니다.

```python
from diffusers import AutoPipelineForInpainting
from PIL import Image

pipe = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

image = Image.open("photo.jpg")
mask = Image.open("mask.png")  # White = inpaint region

result = pipe(
    prompt="A red car parked on the street",
    image=image,
    mask_image=mask,
    num_inference_steps=50
).images[0]
```

## ControlNet

정밀한 제어를 위해 공간 조건을 추가합니다.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch

# Load ControlNet for edge conditioning
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Use Canny edge image as control
control_image = get_canny_image(input_image)

image = pipe(
    prompt="A beautiful house in the style of Van Gogh",
    image=control_image,
    num_inference_steps=30
).images[0]
```

### 사용 가능한 ControlNet

| ControlNet | 입력 유형 | 사용 사례 |
|------------|------------|----------|
| `canny` | 에지 맵 | 구조 유지 |
| `openpose` | 포즈 스켈레톤 | 사람의 포즈 |
| `depth` | 깊이 맵 | 3D 인식 생성 |
| `normal` | 노멀 맵 | 표면 디테일 |
| `mlsd` | 선분 | 건축 선 |
| `scribble` | 대략적인 스케치 | 스케치-이미지 |

## LoRA 어댑터

미세 조정된 스타일 어댑터를 로드합니다.

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("path/to/lora", weight_name="style.safetensors")

# Generate with LoRA style
image = pipe("A portrait in the trained style").images[0]

# Adjust LoRA strength
pipe.fuse_lora(lora_scale=0.8)

# Unload LoRA
pipe.unload_lora_weights()
```

### 여러 LoRA

```python
# Load multiple LoRAs
pipe.load_lora_weights("lora1", adapter_name="style")
pipe.load_lora_weights("lora2", adapter_name="character")

# Set weights for each
pipe.set_adapters(["style", "character"], adapter_weights=[0.7, 0.5])

image = pipe("A portrait").images[0]
```

## 메모리 최적화

### CPU 오프로딩 활성화

```python
# Model CPU offload - moves models to CPU when not in use
pipe.enable_model_cpu_offload()

# Sequential CPU offload - more aggressive, slower
pipe.enable_sequential_cpu_offload()
```

### 어텐션 슬라이싱

```python
# Reduce memory by computing attention in chunks
pipe.enable_attention_slicing()

# Or specific chunk size
pipe.enable_attention_slicing("max")
```

### xFormers 메모리 효율적 어텐션

```python
# Requires xformers package
pipe.enable_xformers_memory_efficient_attention()
```

### 대형 이미지를 위한 VAE 슬라이싱

```python
# Decode latents in tiles for large images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

## 모델 변형

### 서로 다른 정밀도로 로드하기

```python
# FP16 (recommended for GPU)
pipe = DiffusionPipeline.from_pretrained(
    "model-id",
    torch_dtype=torch.float16,
    variant="fp16"
)

# BF16 (better precision, requires Ampere+ GPU)
pipe = DiffusionPipeline.from_pretrained(
    "model-id",
    torch_dtype=torch.bfloat16
)
```

### 특정 구성 요소 로드하기

```python
from diffusers import UNet2DConditionModel, AutoencoderKL

# Load custom VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

# Use with pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae=vae,
    torch_dtype=torch.float16
)
```

## 일괄 생성

여러 이미지를 효율적으로 생성합니다.

```python
# Multiple prompts
prompts = [
    "A cat playing piano",
    "A dog reading a book",
    "A bird painting a picture"
]

images = pipe(prompts, num_inference_steps=30).images

# Multiple images per prompt
images = pipe(
    "A beautiful sunset",
    num_images_per_prompt=4,
    num_inference_steps=30
).images
```

## 일반적인 워크플로

### 워크플로 1: 고품질 생성

```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

# 1. Load SDXL with optimizations
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# 2. Generate with quality settings
image = pipe(
    prompt="A majestic lion in the savanna, golden hour lighting, 8k, detailed fur",
    negative_prompt="blurry, low quality, cartoon, anime, sketch",
    num_inference_steps=30,
    guidance_scale=7.5,
    height=1024,
    width=1024
).images[0]
```

### 워크플로 2: 빠른 프로토타이핑

```python
from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch

# Use LCM for 4-8 step generation
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load LCM LoRA for fast generation
pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.fuse_lora()

# Generate in ~1 second
image = pipe(
    "A beautiful landscape",
    num_inference_steps=4,
    guidance_scale=1.0
).images[0]
```

## 일반적인 문제

**CUDA 메모리 부족:**
```python
# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Or use lower precision
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

**검은색/노이즈 이미지:**
```python
# Check VAE configuration
# Use safety checker bypass if needed
pipe.safety_checker = None

# Ensure proper dtype consistency
pipe = pipe.to(dtype=torch.float16)
```

**느린 생성:**
```python
# Use faster scheduler
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Reduce steps
image = pipe(prompt, num_inference_steps=20).images[0]
```

## 참조 자료

- **[고급 사용법](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/stable-diffusion/references/advanced-usage.md)** - 맞춤 파이프라인, 미세 조정, 배포
- **[문제 해결](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/stable-diffusion/references/troubleshooting.md)** - 일반적인 문제와 해결 방법

## 리소스

- **문서**: https://huggingface.co/docs/diffusers
- **저장소**: https://github.com/huggingface/diffusers
- **모델 허브**: https://huggingface.co/models?library=diffusers
- **Discord**: https://discord.gg/diffusers

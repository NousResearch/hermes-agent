---
title: "Stable Diffusion 图像生成"
sidebar_label: "Stable Diffusion 图像生成"
description: "通过 HuggingFace Diffusers 使用 Stable Diffusion 模型实现最先进的文本到图像生成"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Stable Diffusion 图像生成

通过 HuggingFace Diffusers 使用 Stable Diffusion 模型实现最先进的文本到图像生成。适用于从文本 prompt（提示词）生成图像、执行图像到图像转换、图像修复（inpainting），或构建自定义扩散 pipeline。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/stable-diffusion` 安装 |
| 路径 | `optional-skills/mlops/stable-diffusion` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `diffusers>=0.30.0`, `transformers>=4.41.0`, `accelerate>=0.31.0`, `torch>=2.0.0` |
| 平台 | linux, macos, windows |
| 标签 | `Image Generation`, `Stable Diffusion`, `Diffusers`, `Text-to-Image`, `Multimodal`, `Computer Vision` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发此 skill 时加载的完整 skill 定义。这是 agent 在 skill 激活时所看到的指令内容。
:::

# Stable Diffusion 图像生成

使用 HuggingFace Diffusers 库通过 Stable Diffusion 生成图像的综合指南。

## 何时使用 Stable Diffusion

**在以下情况下使用 Stable Diffusion：**
- 从文本描述生成图像
- 执行图像到图像转换（风格迁移、增强）
- Inpainting（填充遮罩区域）
- Outpainting（将图像扩展至边界之外）
- 创建现有图像的变体
- 构建自定义图像生成工作流

**核心功能：**
- **文本到图像**：从自然语言 prompt 生成图像
- **图像到图像**：在文本引导下转换现有图像
- **Inpainting**：用上下文感知内容填充遮罩区域
- **ControlNet**：添加空间条件控制（边缘、姿态、深度）
- **LoRA 支持**：高效微调与风格适配
- **多模型支持**：支持 SD 1.5、SDXL、SD 3.0、Flux

**改用以下替代方案：**
- **DALL-E 3**：无需 GPU 的 API 生成
- **Midjourney**：艺术化、风格化输出
- **Imagen**：Google Cloud 集成
- **Leonardo.ai**：基于 Web 的创意工作流

## 快速开始

### 安装

```bash
pip install diffusers transformers accelerate torch
pip install xformers  # Optional: memory-efficient attention
```

### 基础文本到图像

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

### 使用 SDXL（更高质量）

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

## 架构概览

### 三支柱设计

Diffusers 围绕三个核心组件构建：

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

### Pipeline 推理流程

```
Text Prompt → Text Encoder → Text Embeddings
                                    ↓
Random Noise → [Denoising Loop] ← Scheduler
                      ↓
               Predicted Noise
                      ↓
              VAE Decoder → Final Image
```

## 核心概念

### Pipeline

Pipeline 编排完整工作流：

| Pipeline | 用途 |
|----------|---------|
| `StableDiffusionPipeline` | 文本到图像（SD 1.x/2.x） |
| `StableDiffusionXLPipeline` | 文本到图像（SDXL） |
| `StableDiffusion3Pipeline` | 文本到图像（SD 3.0） |
| `FluxPipeline` | 文本到图像（Flux 模型） |
| `StableDiffusionImg2ImgPipeline` | 图像到图像 |
| `StableDiffusionInpaintPipeline` | Inpainting |

### Scheduler

Scheduler 控制去噪过程：

| Scheduler | 步数 | 质量 | 适用场景 |
|-----------|-------|---------|----------|
| `EulerDiscreteScheduler` | 20-50 | 良好 | 默认选择 |
| `EulerAncestralDiscreteScheduler` | 20-50 | 良好 | 更多变化 |
| `DPMSolverMultistepScheduler` | 15-25 | 优秀 | 快速、高质量 |
| `DDIMScheduler` | 50-100 | 良好 | 确定性生成 |
| `LCMScheduler` | 4-8 | 良好 | 极速生成 |
| `UniPCMultistepScheduler` | 15-25 | 优秀 | 快速收敛 |

### 切换 Scheduler

```python
from diffusers import DPMSolverMultistepScheduler

# Swap for faster generation
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config
)

# Now generate with fewer steps
image = pipe(prompt, num_inference_steps=20).images[0]
```

## 生成参数

### 关键参数

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `prompt` | 必填 | 目标图像的文本描述 |
| `negative_prompt` | None | 图像中需要避免的内容 |
| `num_inference_steps` | 50 | 去噪步数（越多质量越好） |
| `guidance_scale` | 7.5 | Prompt 遵循程度（通常为 7-12） |
| `height`, `width` | 512/1024 | 输出尺寸（8 的倍数） |
| `generator` | None | 用于可复现性的 Torch generator |
| `num_images_per_prompt` | 1 | 批量大小 |

### 可复现生成

```python
import torch

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt="A cat wearing a top hat",
    generator=generator,
    num_inference_steps=50
).images[0]
```

### Negative prompt

```python
image = pipe(
    prompt="Professional photo of a dog in a garden",
    negative_prompt="blurry, low quality, distorted, ugly, bad anatomy",
    guidance_scale=7.5
).images[0]
```

## 图像到图像

在文本引导下转换现有图像：

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

## Inpainting

填充遮罩区域：

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

添加空间条件控制以实现精确控制：

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

### 可用的 ControlNet

| ControlNet | 输入类型 | 适用场景 |
|------------|------------|----------|
| `canny` | 边缘图 | 保留结构 |
| `openpose` | 姿态骨架 | 人体姿态 |
| `depth` | 深度图 | 3D 感知生成 |
| `normal` | 法线图 | 表面细节 |
| `mlsd` | 线段 | 建筑线条 |
| `scribble` | 粗略草图 | 草图到图像 |

## LoRA 适配器

加载微调风格适配器：

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

### 多个 LoRA

```python
# Load multiple LoRAs
pipe.load_lora_weights("lora1", adapter_name="style")
pipe.load_lora_weights("lora2", adapter_name="character")

# Set weights for each
pipe.set_adapters(["style", "character"], adapter_weights=[0.7, 0.5])

image = pipe("A portrait").images[0]
```

## 内存优化

### 启用 CPU 卸载

```python
# Model CPU offload - moves models to CPU when not in use
pipe.enable_model_cpu_offload()

# Sequential CPU offload - more aggressive, slower
pipe.enable_sequential_cpu_offload()
```

### Attention 切片

```python
# Reduce memory by computing attention in chunks
pipe.enable_attention_slicing()

# Or specific chunk size
pipe.enable_attention_slicing("max")
```

### xFormers 内存高效 Attention

```python
# Requires xformers package
pipe.enable_xformers_memory_efficient_attention()
```

### 大图像的 VAE 切片

```python
# Decode latents in tiles for large images
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()
```

## 模型变体

### 加载不同精度

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

### 加载特定组件

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

## 批量生成

高效生成多张图像：

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

## 常见工作流

### 工作流 1：高质量生成

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

### 工作流 2：快速原型验证

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

## 常见问题

**CUDA 内存不足：**
```python
# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# Or use lower precision
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

**黑色/噪声图像：**
```python
# Check VAE configuration
# Use safety checker bypass if needed
pipe.safety_checker = None

# Ensure proper dtype consistency
pipe = pipe.to(dtype=torch.float16)
```

**生成速度慢：**
```python
# Use faster scheduler
from diffusers import DPMSolverMultistepScheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Reduce steps
image = pipe(prompt, num_inference_steps=20).images[0]
```

## 参考资料

- **[高级用法](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/stable-diffusion/references/advanced-usage.md)** - 自定义 pipeline、微调、部署
- **[故障排查](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/stable-diffusion/references/troubleshooting.md)** - 常见问题与解决方案

## 资源

- **文档**：https://huggingface.co/docs/diffusers
- **代码仓库**：https://github.com/huggingface/diffusers
- **模型中心**：https://huggingface.co/models?library=diffusers
- **Discord**：https://discord.gg/diffusers
---
title: "Llava — 大型语言与视觉助手"
sidebar_label: "Llava"
description: "大型语言与视觉助手"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llava

大型语言与视觉助手。支持视觉指令微调（instruction tuning）和基于图像的对话。将 CLIP 视觉编码器与 Vicuna/LLaMA 语言模型相结合。支持多轮图像对话、视觉问答（VQA）和指令跟随。适用于视觉语言聊天机器人或图像理解任务。最适合对话式图像分析。

## Skill 元数据

| | |
|---|---|
| 来源 | 可选 — 通过 `hermes skills install official/mlops/llava` 安装 |
| 路径 | `optional-skills/mlops/llava` |
| 版本 | `1.0.0` |
| 作者 | Orchestra Research |
| 许可证 | MIT |
| 依赖项 | `transformers`, `torch`, `pillow` |
| 平台 | linux, macos, windows |
| 标签 | `LLaVA`, `Vision-Language`, `Multimodal`, `Visual Question Answering`, `Image Chat`, `CLIP`, `Vicuna`, `Conversational AI`, `Instruction Tuning`, `VQA` |

## 参考：完整 SKILL.md

:::info
以下是 Hermes 在触发该 skill 时加载的完整 skill 定义。这是 skill 激活时 agent 所看到的指令内容。
:::

# LLaVA - 大型语言与视觉助手

用于对话式图像理解的开源视觉语言模型。

## 何时使用 LLaVA

**适用场景：**
- 构建视觉语言聊天机器人
- 视觉问答（VQA）
- 图像描述与字幕生成
- 多轮图像对话
- 视觉指令跟随
- 含图像的文档理解

**指标**：
- **GitHub 23,000+ 星标**
- GPT-4V 级别能力（目标）
- Apache 2.0 许可证
- 多种模型规格（7B–34B 参数）

**改用其他方案的情况**：
- **GPT-4V**：质量最高，基于 API
- **CLIP**：简单零样本分类
- **BLIP-2**：更适合纯字幕生成
- **Flamingo**：研究用途，非开源

## 快速开始

### 安装

```bash
# Clone repository
git clone https://github.com/haotian-liu/LLaVA
cd LLaVA

# Install
pip install -e .
```

### 基本用法

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from PIL import Image
import torch

# Load model
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# Load image
image = Image.open("image.jpg")
image_tensor = process_images([image], image_processor, model.config)
image_tensor = image_tensor.to(model.device, dtype=torch.float16)

# Create conversation
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nWhat is in this image?")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Generate response
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.2,
        max_new_tokens=512
    )

response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
print(response)
```

## 可用模型

| 模型 | 参数量 | 显存 | 质量 |
|-------|------------|------|---------|
| LLaVA-v1.5-7B | 7B | ~14 GB | 良好 |
| LLaVA-v1.5-13B | 13B | ~28 GB | 较好 |
| LLaVA-v1.6-34B | 34B | ~70 GB | 最佳 |

```python
# Load different models
model_7b = "liuhaotian/llava-v1.5-7b"
model_13b = "liuhaotian/llava-v1.5-13b"
model_34b = "liuhaotian/llava-v1.6-34b"

# 4-bit quantization for lower VRAM
load_4bit = True  # Reduces VRAM by ~4×
```

## CLI 用法

```bash
# Single image query
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file image.jpg \
    --query "What is in this image?"

# Multi-turn conversation
python -m llava.serve.cli \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-file image.jpg
# Then type questions interactively
```

## Web UI（Gradio）

```bash
# Launch Gradio interface
python -m llava.serve.gradio_web_server \
    --model-path liuhaotian/llava-v1.5-7b \
    --load-4bit  # Optional: reduce VRAM

# Access at http://localhost:7860
```

## 多轮对话

```python
# Initialize conversation
conv = conv_templates["llava_v1"].copy()

# Turn 1
conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\nWhat is in this image?")
conv.append_message(conv.roles[1], None)
response1 = generate(conv, model, image)  # "A dog playing in a park"

# Turn 2
conv.messages[-1][1] = response1  # Add previous response
conv.append_message(conv.roles[0], "What breed is the dog?")
conv.append_message(conv.roles[1], None)
response2 = generate(conv, model, image)  # "Golden Retriever"

# Turn 3
conv.messages[-1][1] = response2
conv.append_message(conv.roles[0], "What time of day is it?")
conv.append_message(conv.roles[1], None)
response3 = generate(conv, model, image)
```

## 常见任务

### 图像字幕生成

```python
question = "Describe this image in detail."
response = ask(model, image, question)
```

### 视觉问答

```python
question = "How many people are in the image?"
response = ask(model, image, question)
```

### 目标检测（文本形式）

```python
question = "List all the objects you can see in this image."
response = ask(model, image, question)
```

### 场景理解

```python
question = "What is happening in this scene?"
response = ask(model, image, question)
```

### 文档理解

```python
question = "What is the main topic of this document?"
response = ask(model, document_image, question)
```

## 训练自定义模型

```bash
# Stage 1: Feature alignment (558K image-caption pairs)
bash scripts/v1_5/pretrain.sh

# Stage 2: Visual instruction tuning (150K instruction data)
bash scripts/v1_5/finetune.sh
```

## 量化（降低显存占用）

```python
# 4-bit quantization
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="liuhaotian/llava-v1.5-13b",
    model_base=None,
    model_name=get_model_name_from_path("liuhaotian/llava-v1.5-13b"),
    load_4bit=True  # Reduces VRAM ~4×
)

# 8-bit quantization
load_8bit=True  # Reduces VRAM ~2×
```

## 最佳实践

1. **从 7B 模型开始** — 质量良好，显存需求可控
2. **使用 4-bit 量化** — 显著降低显存占用
3. **需要 GPU** — CPU 推理极慢
4. **清晰的 prompt** — 具体问题能获得更好的答案
5. **多轮对话** — 保持对话上下文
6. **温度 0.2–0.7** — 平衡创造性与一致性
7. **`max_new_tokens` 512–1024** — 用于详细回复
8. **批量处理** — 按顺序处理多张图像

## 性能

| 模型 | 显存（FP16） | 显存（4-bit） | 速度（tokens/s） |
|-------|-------------|--------------|------------------|
| 7B | ~14 GB | ~4 GB | ~20 |
| 13B | ~28 GB | ~8 GB | ~12 |
| 34B | ~70 GB | ~18 GB | ~5 |

*在 A100 GPU 上测试*

## 基准测试

LLaVA 在以下基准上取得了有竞争力的分数：
- **VQAv2**：78.5%
- **GQA**：62.0%
- **MM-Vet**：35.4%
- **MMBench**：64.3%

## 局限性

1. **幻觉** — 可能描述图像中不存在的内容
2. **空间推理** — 难以精确定位位置
3. **小字体文本** — 难以识别细小字体
4. **目标计数** — 对大量目标计数不精确
5. **显存需求** — 需要高性能 GPU
6. **推理速度** — 比 CLIP 慢

## 与框架集成

### LangChain

```python
from langchain.llms.base import LLM

class LLaVALLM(LLM):
    def _call(self, prompt, stop=None):
        # Custom LLaVA inference
        return response

llm = LLaVALLM()
```

### Gradio 应用

```python
import gradio as gr

def chat(image, text, history):
    response = ask_llava(model, image, text)
    return response

demo = gr.ChatInterface(
    chat,
    additional_inputs=[gr.Image(type="pil")],
    title="LLaVA Chat"
)
demo.launch()
```

## 资源

- **GitHub**：https://github.com/haotian-liu/LLaVA ⭐ 23,000+
- **论文**：https://arxiv.org/abs/2304.08485
- **演示**：https://llava.hliu.cc
- **模型**：https://huggingface.co/liuhaotian
- **许可证**：Apache 2.0
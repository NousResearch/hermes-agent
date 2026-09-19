---
title: "Llava — 시각-언어 채팅: VQA, 캡션 생성, 이미지 대화"
sidebar_label: "Llava"
description: "시각-언어 채팅: VQA, 캡션 생성, 이미지 대화"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Llava

시각-언어 채팅: VQA, 캡션 생성, 이미지 대화.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/llava`로 설치 |
| 경로 | `optional-skills/mlops/llava` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `transformers`, `torch`, `pillow` |
| 플랫폼 | linux, macos, windows |
| 태그 | `LLaVA`, `시각-언어`, `멀티모달`, `시각적 질의응답`, `이미지 채팅`, `CLIP`, `Vicuna`, `대화형 AI`, `지시 튜닝`, `VQA` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# LLaVA - 대규모 언어 및 시각 어시스턴트

대화형 이미지 이해를 위한 오픈 소스 시각-언어 모델입니다.

## LLaVA를 사용할 때

**다음과 같은 경우 사용하세요:**
- 시각-언어 챗봇 구축
- 시각적 질의응답(VQA)
- 이미지 설명 및 캡션 생성
- 멀티턴 이미지 대화
- 시각적 지시 따르기
- 이미지가 포함된 문서 이해

**메트릭**:
- **GitHub 스타 23,000개 이상**
- GPT-4V 수준의 기능(목표)
- Apache 2.0 라이선스
- 여러 모델 크기(매개변수 7B-34B)

**대신 다음 대안을 사용하세요:**
- **GPT-4V**: 최고 품질, API 기반
- **CLIP**: 단순한 제로샷 분류
- **BLIP-2**: 캡션 생성에만 더 적합
- **Flamingo**: 연구용, 오픈 소스 아님

## 빠른 시작

### 설치

```bash
# Clone repository
git clone https://github.com/haotian-liu/LLaVA
cd LLaVA

# Install
pip install -e .
```

### 기본 사용법

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

## 사용 가능한 모델

| 모델 | 매개변수 | VRAM | 품질 |
|-------|------------|------|---------|
| LLaVA-v1.5-7B | 7B | ~14 GB | 좋음 |
| LLaVA-v1.5-13B | 13B | ~28 GB | 더 좋음 |
| LLaVA-v1.6-34B | 34B | ~70 GB | 최고 |

```python
# Load different models
model_7b = "liuhaotian/llava-v1.5-7b"
model_13b = "liuhaotian/llava-v1.5-13b"
model_34b = "liuhaotian/llava-v1.6-34b"

# 4-bit quantization for lower VRAM
load_4bit = True  # Reduces VRAM by ~4×
```

## CLI 사용법

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

## 웹 UI(Gradio)

```bash
# Launch Gradio interface
python -m llava.serve.gradio_web_server \
    --model-path liuhaotian/llava-v1.5-7b \
    --load-4bit  # Optional: reduce VRAM

# Access at http://localhost:7860
```

## 멀티턴 대화

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

## 일반적인 작업

### 이미지 캡션 생성

```python
question = "Describe this image in detail."
response = ask(model, image, question)
```

### 시각적 질의응답

```python
question = "How many people are in the image?"
response = ask(model, image, question)
```

### 객체 감지(텍스트)

```python
question = "List all the objects you can see in this image."
response = ask(model, image, question)
```

### 장면 이해

```python
question = "What is happening in this scene?"
response = ask(model, image, question)
```

### 문서 이해

```python
question = "What is the main topic of this document?"
response = ask(model, document_image, question)
```

## 사용자 지정 모델 학습

```bash
# Stage 1: Feature alignment (558K image-caption pairs)
bash scripts/v1_5/pretrain.sh

# Stage 2: Visual instruction tuning (150K instruction data)
bash scripts/v1_5/finetune.sh
```

## 양자화(VRAM 줄이기)

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

## 모범 사례

1. **7B 모델로 시작** - 좋은 품질과 관리 가능한 VRAM 사용량
2. **4비트 양자화 사용** - VRAM을 크게 줄임
3. **GPU 필요** - CPU 추론은 매우 느림
4. **명확한 프롬프트** - 구체적인 질문일수록 더 나은 답변
5. **멀티턴 대화** - 대화 맥락 유지
6. **Temperature 0.2-0.7** - 창의성과 일관성의 균형
7. **max_new_tokens 512-1024** - 상세한 응답용
8. **배치 처리** - 여러 이미지를 순차적으로 처리

## 성능

| 모델 | VRAM (FP16) | VRAM (4비트) | 속도 (토큰/초) |
|-------|-------------|--------------|------------------|
| 7B | ~14 GB | ~4 GB | ~20 |
| 13B | ~28 GB | ~8 GB | ~12 |
| 34B | ~70 GB | ~18 GB | ~5 |

*A100 GPU 기준*

## 벤치마크

LLaVA는 다음 벤치마크에서 경쟁력 있는 점수를 달성합니다.
- **VQAv2**: 78.5%
- **GQA**: 62.0%
- **MM-Vet**: 35.4%
- **MMBench**: 64.3%

## 한계

1. **환각** - 이미지에 없는 내용을 설명할 수 있음
2. **공간 추론** - 정확한 위치 파악에 어려움
3. **작은 텍스트** - 작은 글씨를 읽기 어려움
4. **객체 수 세기** - 객체가 많으면 부정확함
5. **VRAM 요구 사항** - 강력한 GPU 필요
6. **추론 속도** - CLIP보다 느림

## 프레임워크 통합

### LangChain

```python
from langchain.llms.base import LLM

class LLaVALLM(LLM):
    def _call(self, prompt, stop=None):
        # Custom LLaVA inference
        return response

llm = LLaVALLM()
```

### Gradio 앱

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

## 리소스

- **GitHub**: https://github.com/haotian-liu/LLaVA ⭐ 23,000+
- **논문**: https://arxiv.org/abs/2304.08485
- **데모**: https://llava.hliu.cc
- **모델**: https://huggingface.co/liuhaotian
- **라이선스**: Apache 2.0

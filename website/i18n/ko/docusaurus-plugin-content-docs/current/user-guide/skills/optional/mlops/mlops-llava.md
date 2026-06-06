---
title: "Llava — 오픈 소스 대형 멀티모달 모델 (LMM)"
sidebar_label: "Llava"
description: "오픈 소스 대형 멀티모달 모델 (LMM)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llava

이미지와 텍스트를 모두 이해하는 오픈 소스 대형 멀티모달 모델(LMM)인 LLaVA(Large Language and Vision Assistant). 시각적 질문 답변, 이미지 캡션 생성, OCR, 시각적 추론 또는 멀티모달 채팅 시스템을 구축할 때 사용합니다. 이미지 입력이 포함된 GPT-4V 수준의 기능이 필요할 때 가장 적합합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/llava`로 설치 |
| Path | `optional-skills/mlops/llava` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `transformers>=4.35.0`, `torch`, `Pillow`, `accelerate` |
| Platforms | linux, macos, windows |
| Tags | `Multimodal`, `LMM`, `Computer Vision`, `LLaVA`, `Image Understanding`, `VQA` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# LLaVA (Large Language and Vision Assistant)

이미지와 텍스트를 함께 이해하는 오픈 소스 대형 멀티모달 모델(LMM)입니다.

## LLaVA를 사용해야 할 때

**다음과 같은 경우 LLaVA를 사용하세요:**
- 이미지 캡션을 작성하거나 이미지를 자세히 설명해야 할 때
- 이미지에 관한 질문에 답변할 때 (시각적 질문 답변 - VQA)
- 이미지에서 텍스트를 읽어야 할 때 (OCR)
- 차트나 다이어그램을 분석할 때
- 이미지와 텍스트 입력을 모두 받아들이는 챗봇을 구축할 때
- 로컬 또는 자체 서버에서 GPT-4V와 유사한 기능이 필요할 때

**대신 다른 대안을 사용해야 할 때:**
- 텍스트 입력만 있는 경우 → Llama 3 또는 Mistral 사용 (더 빠르고 가벼움)
- 이미지나 비디오를 *생성*해야 하는 경우 → Stable Diffusion 또는 Midjourney 사용
- 객체 감지나 분할(바운딩 박스)만 필요한 경우 → YOLO 또는 SAM 사용
- 비디오 분석의 경우 → Video-LLaVA 또는 프레임 추출 사용

## 빠른 시작

### 설치

```bash
pip install transformers torch Pillow accelerate
```

### 추론 예제 (Hugging Face Transformers 사용)

이것은 `llava-1.5-7b-hf`를 사용하는 가장 쉬운 방법입니다.

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import requests

# 1. 모델과 프로세서 로드
model_id = "llava-hf/llava-1.5-7b-hf"
# 메모리 절약을 위해 float16으로 로드 (약 15GB VRAM 필요)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# 2. 이미지 준비
url = "https://images.unsplash.com/photo-1517331156700-3c241d2b4d83?q=80&w=1000"
image = Image.open(requests.get(url, stream=True).raw)

# 3. 프롬프트 준비 (LLaVA 특정 형식 사용)
# <image> 토큰은 반드시 포함되어야 합니다!
prompt = "USER: <image>\nDescribe what you see in this image.\nASSISTANT:"

# 4. 입력 처리
inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

# 5. 응답 생성
generate_ids = model.generate(**inputs, max_new_tokens=200)

# 6. 디코딩 (프롬프트 부분 제외)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
```

## 핵심 개념

### 모델 버전

1. **LLaVA 1.5** (가장 인기 있고 안정적임)
   - 7B 파라미터 (Vicuna-7B 기반) - 16GB GPU에 적합
   - 13B 파라미터 (Vicuna-13B 기반) - 24GB GPU에 적합
   - 해상도: 336x336 이미지 입력

2. **LLaVA-NeXT (LLaVA 1.6)** (최신)
   - 다양한 언어 모델(Vicuna, Llama-3, Mistral) 기반
   - 동적 고해상도 지원 (그리드 패치)
   - OCR 및 추론 성능 향상

### 프롬프트 형식

LLaVA는 엄격한 프롬프트 템플릿을 요구합니다. 정확한 형식은 사용 중인 모델에 따라 다릅니다.

**LLaVA 1.5 (Vicuna 형식)**:
```text
USER: <image>\n{당신의 질문}
ASSISTANT:
```

**LLaVA-NeXT (Mistral 형식)**:
```text
[INST] <image>\n{당신의 질문} [/INST]
```

## 모범 사례

1. **메모리 최적화**: 
   - VRAM이 부족한 경우 `BitsAndBytesConfig`를 사용하여 모델을 4-bit 또는 8-bit로 로드하세요.
2. **다중 이미지**:
   - 표준 LLaVA 모델은 단일 이미지 프롬프트에 최적화되어 있습니다. 다중 이미지를 위해서는 특수한 모델(예: LLaVA-NeXT-Interleave)이 필요합니다.
3. **명확한 지침**:
   - 객체 수 세기, 텍스트 읽기 등 수행할 작업을 정확하게 지정하세요.
4. **로컬 실행(Ollama)**:
   - 프로그래밍 방식(Python)이 아닌 로컬에서 빠르게 실행하고 싶다면 Ollama를 사용하는 것이 가장 쉽습니다: `ollama run llava`

## 리소스

- **문서 (Hugging Face)**: [Transformers LLaVA 문서](https://huggingface.co/docs/transformers/model_doc/llava)
- **저장소**: [LLaVA 공식 저장소](https://github.com/haotian-liu/LLaVA)

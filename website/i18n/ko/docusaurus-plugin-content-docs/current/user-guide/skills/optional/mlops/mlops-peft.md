---
title: "Peft"
sidebar_label: "Peft"
description: "사전 학습된 대규모 모델(LLM 등)을 소비자용 하드웨어에서 효율적으로 파인튜닝할 수 있는 라이브러리"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Peft

사전 학습된 대규모 모델(LLM 등)을 소비자용 하드웨어에서 효율적으로 파인튜닝할 수 있는 라이브러리. 모델의 파라미터 전체를 수정하는 대신, 파라미터를 크게 줄인 어댑터를 학습시킵니다. LoRA, QLoRA 등 파라미터 효율적 파인튜닝 기법을 활용할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/peft`로 설치 |
| Path | `optional-skills/mlops/peft` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `peft`, `transformers`, `torch` |
| Platforms | linux, macos, windows |
| Tags | `Machine Learning`, `PEFT`, `LoRA`, `Fine-tuning`, `HuggingFace`, `LLM` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# PEFT (Parameter-Efficient Fine-Tuning)

사전 학습된 대규모 모델(예: LLM)을 모든 매개변수를 조정하지 않고 효율적으로 파인튜닝하기 위한 라이브러리입니다.

## PEFT를 사용해야 할 때

**다음과 같은 경우 PEFT를 사용하세요:**
- 소비자용 하드웨어(단일 GPU)에서 LLM(예: Llama 3)을 파인튜닝해야 할 때
- 전체 파인튜닝을 할 때 발생하는 메모리 부족(OOM) 오류를 피하고자 할 때
- 단일 기본 모델에 여러 개의 작은 어댑터(adapter)를 사용할 때
- 저장소 공간을 절약하고 싶을 때 (PEFT 어댑터는 수십 GB가 아닌 수 MB의 크기입니다)

**지원되는 주요 방법:**
- **LoRA** (Low-Rank Adaptation) - 가장 인기 있는 방식
- **QLoRA** - 양자화된 LoRA (가장 메모리 효율적임)
- **Prefix Tuning**
- **Prompt Tuning**
- **IA3**

## 빠른 시작 (LoRA)

### 설치

```bash
pip install peft transformers torch
```

### 1. 모델과 LoRA 어댑터 로드

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# 1. 기본 모델 로드
model_name = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 2. LoRA 구성 정의
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8,                     # 랭크 차원 (일반적으로 8, 16, 32, 64)
    lora_alpha=32,           # 스케일링 팩터 (일반적으로 r의 2배)
    lora_dropout=0.1,        # 드롭아웃 확률
    target_modules=["q_proj", "v_proj"] # 조정할 모듈 (기본적으로 어텐션 모듈)
)

# 3. 모델을 PEFT 형식으로 변환
model = get_peft_model(model, peft_config)

# 학습 가능한 매개변수의 수 출력
model.print_trainable_parameters()
# 출력 예시: trainable params: 4,194,304 || all params: 7,000,000,000 || trainable%: 0.0599%
```

### 2. 어댑터 저장 및 불러오기

전체 모델 가중치 대신 수 메가바이트 크기의 가중치 세트(어댑터) 하나만 저장합니다.

```python
# 학습 후 저장
model.save_pretrained("my-lora-adapter")

# 추론 시 불러오기
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
peft_model = PeftModel.from_pretrained(base_model, "my-lora-adapter")
```

## 워크플로: QLoRA (최대 메모리 절약)

QLoRA는 기본 모델을 4-bit 해상도로 양자화한 다음 모델 위에 LoRA 어댑터를 학습시킵니다. 이를 통해 일반 GPU에서 매우 큰 모델을 학습시킬 수 있습니다.

```bash
pip install bitsandbytes accelerate
```

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

# 1. 4-bit 양자화 구성 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 2. 4-bit 모드로 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. 양자화된 학습을 위한 모델 전처리
model = prepare_model_for_kbit_training(model)

# 4. 일반적인 LoRA 설정 (그러나 더 많은 대상을 지정할 수 있음)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
```

## 모범 사례

1. **대상 모듈(`target_modules`) 설정 규칙**: 
   - `"all-linear"`로 설정하면 지원되는 모든 선형(linear) 레이어가 자동으로 대상이 되어 가장 좋은 결과를 얻을 수 있습니다.
2. **알파(`lora_alpha`) 설정 규칙**:
   - `lora_alpha`는 일반적으로 랭크 `r`의 2배여야 합니다 (예: r=16일 때 alpha=32).
3. **병합(Merging)**:
   - 더 빠른 추론을 위해 어댑터를 기본 모델로 다시 병합할 수 있습니다: `model = model.merge_and_unload()`. 단, 이 작업은 메모리를 많이 소비합니다.

## 리소스

- **문서**: [Hugging Face PEFT 문서](https://huggingface.co/docs/peft/index)

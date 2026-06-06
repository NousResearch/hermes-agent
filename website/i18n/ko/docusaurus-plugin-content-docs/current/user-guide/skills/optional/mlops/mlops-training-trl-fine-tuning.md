---
title: "Trl (Transformer Reinforcement Learning)"
sidebar_label: "Trl (Transformer Reinforcement Learning)"
description: "Transformer 언어 모델 및 확산 모델을 파인튜닝하고 정렬하기 위한 전체 스택 라이브러리"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Trl (Transformer Reinforcement Learning)

Transformer 언어 모델 및 확산 모델을 파인튜닝하고 정렬하기 위한 전체 스택 라이브러리. SFT(지도 미세 조정), DPO(직접 선호도 최적화), PPO(근위 정책 최적화) 등을 지원합니다. LLM을 특정 작업에 적응시키거나, 선호도에 맞게 정렬하거나, 효율적인 어텐션(FlashAttention) 및 PEFT(LoRA)를 갖춘 다중 GPU 클러스터에서 학습시킬 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/training-trl-fine-tuning`로 설치 |
| Path | `optional-skills/mlops/training-trl-fine-tuning` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `trl`, `transformers`, `peft`, `accelerate`, `datasets` |
| Platforms | linux, macos, windows |
| Tags | `Machine Learning`, `Fine-tuning`, `TRL`, `LLM`, `SFT`, `DPO`, `PPO`, `HuggingFace`, `Alignment`, `Reinforcement Learning` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되어 있을 때 에이전트가 지침으로 보는 내용입니다.
:::

# TRL (Transformer Reinforcement Learning)

언어 모델 및 확산 모델의 파인튜닝과 정렬(Alignment)을 위한 Hugging Face의 전체 스택 라이브러리입니다.

## TRL을 사용해야 할 때

**다음과 같은 경우 TRL을 사용하세요:**
- 특정 도메인에 맞게 LLM을 미세 조정할 때 (SFT)
- 인간의 선호도에 맞게 모델을 정렬할 때 (DPO, PPO)
- 프롬프트를 사용하여 텍스트나 이미지를 생성할 때
- PEFT(LoRA) 및 양자화(bitsandbytes)로 메모리 사용량을 줄일 때
- 다중 GPU(Accelerate, DeepSpeed)에 걸쳐 학습을 확장할 때

**핵심 알고리즘:**
- **SFT (Supervised Fine-Tuning)**: 기본 지시문 따르기
- **DPO (Direct Preference Optimization)**: 보상 모델 없는 선호도 정렬
- **PPO (Proximal Policy Optimization)**: RLHF를 사용한 복잡한 정렬
- **ORPO/KTO/CPO**: 최신 선호도 최적화 방법

## 빠른 시작

### 설치

```bash
pip install trl transformers peft accelerate datasets bitsandbytes
```

### SFT (지도 미세 조정) 예제

```python
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 데이터셋 로드 (텍스트 열이 필요함)
dataset = load_dataset("imdb", split="train")

# 학습 인자 구성
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    max_steps=100,
)

# 트레이너 초기화
trainer = SFTTrainer(
    "facebook/opt-350m",            # 모델 (이름 또는 사전 학습된 모델 객체)
    train_dataset=dataset,
    dataset_text_field="text",      # 텍스트 열
    max_seq_length=512,
    args=training_args,
)

# 학습 시작
trainer.train()
```

## 워크플로 1: 지시문 튜닝 (SFTTrainer)

채팅 형식의 데이터를 SFTTrainer를 사용하여 LLM이 지시문을 따르도록 가르칩니다.

### 1. 데이터셋 형식

표준 `chat` 형식을 사용하는 것이 가장 좋습니다:

```python
dataset = load_dataset("HuggingFaceH4/no_robots")

# 데이터셋의 각 행은 메시지 목록을 포함해야 합니다:
# [
#   {"role": "user", "content": "Hello!"},
#   {"role": "assistant", "content": "Hi there!"}
# ]
```

### 2. SFTTrainer 사용

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_id = "meta-llama/Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 채팅 템플릿 사용
sft_config = SFTConfig(
    output_dir="./sft_output",
    max_seq_length=1024,
    packing=True, # 시퀀스를 포장하여 속도 향상
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
    tokenizer=tokenizer,
    # SFTTrainer는 데이터셋이 올바른 형식이면 자동으로 chat_template을 적용합니다.
)

trainer.train()
```

## 워크플로 2: DPO로 정렬 (DPOTrainer)

직접 선호도 최적화(DPO)를 사용하여 모델을 인간 선호도에 맞게 정렬합니다. PPO보다 훨씬 간단하고 안정적입니다.

### 1. 선호도 데이터셋 형식

데이터셋에는 세 가지 열이 필요합니다: `prompt`, `chosen`, `rejected`.

```python
dataset = load_dataset("Anthropic/hh-rlhf")

# 형식:
# prompt: "How do I fix a flat tire?"
# chosen: "Here are the steps to fix a flat tire: 1. ..." (선호되는 답변)
# rejected: "I don't know, maybe use tape?" (거부되는 답변)
```

### 2. DPOTrainer 사용

```python
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "my-sft-model"
model = AutoModelForCausalLM.from_pretrained(model_id)
# DPO는 참조 모델(학습 전의 초기 모델)이 필요합니다.
ref_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dpo_config = DPOConfig(
    output_dir="./dpo_output",
    beta=0.1, # DPO의 온도 파라미터 (보통 0.1 - 0.5)
)

trainer = DPOTrainer(
    model,
    ref_model,
    args=dpo_config,
    beta=0.1,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## 워크플로 3: LoRA로 효율적인 파인튜닝 (PEFT)

대규모 모델의 경우 LoRA를 사용하여 적은 수의 파라미터만 업데이트하여 메모리를 절약합니다.

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# LoRA 구성
peft_config = LoraConfig(
    r=16,               # 랭크 차원
    lora_alpha=32,      # 스케일링 팩터
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"] # 대상 모듈 (보통 어텐션 블록)
)

# 트레이너에 peft_config 전달 (자동으로 모델을 PEFT 형식으로 변환)
trainer = SFTTrainer(
    "meta-llama/Llama-3-8B",
    train_dataset=dataset,
    peft_config=peft_config,
    # ... 다른 인자들
)
```

### QLoRA (양자화 + LoRA)

더 적은 메모리를 위해 4-bit 양자화와 결합합니다:

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=quantization_config
)

# 그리고 앞의 예제와 같이 SFTTrainer + peft_config를 사용합니다.
```

## 워크플로 4: 보상 모델링 (RewardTrainer)

PPO에 사용될 보상 모델(주어진 입력에 대해 스칼라 보상을 출력하는 모델)을 학습합니다.

```python
from trl import RewardConfig, RewardTrainer
from transformers import AutoModelForSequenceClassification

# 보상 모델은 SequenceClassification 헤드를 사용합니다 (보통 num_labels=1)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=1)

# 데이터셋에는 'input_ids_chosen'과 'input_ids_rejected'가 필요합니다.
reward_config = RewardConfig(output_dir="./reward_model")

trainer = RewardTrainer(
    model=model,
    args=reward_config,
    train_dataset=dataset,
)
trainer.train()
```

## 워크플로 5: PPO를 사용한 RLHF (PPOTrainer)

전체 강화 학습 기반 파인튜닝을 위해 PPOTrainer를 사용합니다. 설정이 가장 복잡합니다.

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# PPO는 가치 헤드(Value Head)가 있는 모델이 필요합니다.
model = AutoModelForCausalLMWithValueHead.from_pretrained("my-sft-model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("my-sft-model")

ppo_config = PPOConfig(batch_size=16, mini_batch_size=4)
ppo_trainer = PPOTrainer(ppo_config, model, ref_model, tokenizer, dataset=dataset)

# PPO는 보상 모델(또는 휴리스틱)의 외부 피드백 루프가 필요합니다.
for epoch, batch in enumerate(ppo_trainer.dataloader):
    query_tensors = batch["input_ids"]

    # 모델에서 응답 생성
    response_tensors = ppo_trainer.generate(query_tensors)

    # 응답 스코어링 (사용자 지정 함수)
    rewards = get_rewards(query_tensors, response_tensors)

    # PPO 스텝 진행
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

## 모범 사례

1. **데이터 형식이 핵심입니다**: 학습을 시작하기 전에 항상 몇 가지 샘플을 확인하여 `tokenizer.apply_chat_template`이 올바르게 적용되었는지 확인하세요.
2. **먼저 SFT로 시작하세요**: DPO/PPO를 시도하기 전에 기본 모델에 대해 지도 미세 조정을 수행해야 합니다.
3. **PPO보다 DPO 선호**: 대부분의 사용 사례에서 DPO는 훨씬 설정하기 쉽고 PPO와 유사하거나 더 나은 결과를 얻습니다. 보상 모델 파이프라인이 필요하지 않습니다.
4. **메모리 절약 팁**: OOM(Out of Memory) 오류가 발생하는 경우:
   - `per_device_train_batch_size`를 줄이고 `gradient_accumulation_steps`를 늘립니다.
   - `gradient_checkpointing=True` 활성화
   - `peft`를 사용한 QLoRA 사용
5. **Flash Attention**: Ampere 이상의 GPU(A100, H100, RTX 30/40 시리즈)를 사용하는 경우 모델 로드 시 `attn_implementation="flash_attention_2"`를 사용하여 상당한 속도 향상과 메모리 절감을 얻을 수 있습니다.

## 일반적인 문제

**오류: Tokenizer에 pad_token이 없습니다.**
```python
# 해결책:
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**오류: CUDA Out of memory**
```python
# 해결책 (TrainingArguments에서):
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
gradient_checkpointing=True,
```

## 리소스

- **문서**: [Hugging Face TRL 문서](https://huggingface.co/docs/trl/index)
- **GitHub**: [huggingface/trl](https://github.com/huggingface/trl)

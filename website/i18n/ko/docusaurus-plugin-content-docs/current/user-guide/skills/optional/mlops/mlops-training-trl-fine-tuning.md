---
title: "TRL 파인 튜닝 — TRL: LLM RLHF를 위한 SFT, DPO, GRPO, RLOO 보상 모델링"
sidebar_label: "TRL 파인 튜닝"
description: "TRL: LLM RLHF를 위한 SFT, DPO, GRPO, RLOO 보상 모델링"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# TRL 파인 튜닝

TRL을 사용한 LLM RLHF용 SFT, DPO, GRPO, RLOO 보상 모델링입니다.

## Skill 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/trl-fine-tuning`으로 설치 |
| 경로 | `optional-skills/mlops/training/trl-fine-tuning` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `trl`, `transformers`, `datasets`, `peft`, `accelerate`, `torch` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Post-Training`, `TRL`, `Reinforcement Learning`, `Fine-Tuning`, `SFT`, `DPO`, `GRPO`, `RLOO`, `RLHF`, `Preference Alignment`, `HuggingFace` |

## 참조: 전체 SKILL.md

:::info
다음은 이 skill이 활성화될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성 상태일 때 에이전트가 보는 지침도 이것입니다.
:::

# TRL - Transformer 강화 학습

## 빠른 시작

TRL은 언어 모델을 인간의 선호에 맞게 정렬하기 위한 사후 학습 방법을 제공합니다.

**설치**:
```bash
pip install trl transformers datasets peft accelerate
```

**지도 파인 튜닝**(instruction tuning):
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,  # Prompt-completion pairs
)
trainer.train()
```

**DPO**(선호에 맞게 정렬):
```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(output_dir="model-dpo", beta=0.1)
trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_dataset,  # chosen/rejected pairs
    processing_class=tokenizer
)
trainer.train()
```

## 일반적인 워크플로

### 워크플로 1: 전체 RLHF 파이프라인(SFT → 보상 모델 → RLOO)

기본 모델부터 인간의 선호에 맞게 정렬된 모델까지 이어지는 전체 파이프라인입니다.

> **참고 (TRL 1.x):** PPO는 **TRL에서 제거되었습니다** — `PPOTrainer`, `PPOConfig`,
> `python -m trl.scripts.ppo`는 더 이상 존재하지 않습니다. TRL에 여전히 포함된 온라인 RL 트레이너를 사용하세요:
> **RLOO**(`RLOOTrainer` / `trl rloo`)는 보상 모델 기반 RLHF 파이프라인에 가장 가까운 대체재이며,
> **GRPO**(`GRPOTrainer` / `trl grpo`, 워크플로 3 참조)는 메모리 효율적인 대안입니다. 아래 단계에서는 RLOO를 사용합니다.

이 체크리스트를 복사하세요:

```
RLHF Training:
- [ ] Step 1: Supervised fine-tuning (SFT)
- [ ] Step 2: Train reward model
- [ ] Step 3: RLOO reinforcement learning
- [ ] Step 4: Evaluate aligned model
```

**1단계: 지도 파인 튜닝**

instruction-following 데이터로 기본 모델을 학습합니다:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load instruction dataset
dataset = load_dataset("trl-lib/Capybara", split="train")

# Configure training
training_args = SFTConfig(
    output_dir="Qwen2.5-0.5B-SFT",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="epoch"
)

# Train
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)
trainer.train()
trainer.save_model()
```

**2단계: 보상 모델 학습**

인간의 선호를 예측하도록 모델을 학습합니다:

```python
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

# Load SFT model as base
model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B-SFT",
    num_labels=1  # Single reward score
)
tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-SFT")

# Load preference data (chosen/rejected pairs)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

# Configure training
training_args = RewardConfig(
    output_dir="Qwen2.5-0.5B-Reward",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    learning_rate=1e-5
)

# Train reward model
trainer = RewardTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=dataset
)
trainer.train()
trainer.save_model()
```

**3단계: RLOO 강화 학습**

보상 모델을 사용해 정책을 최적화합니다. PPO는 TRL 1.x에서 제거되었으므로, 학습한 보상 모델을
`--reward_model_name_or_path`로 전달하는 RLOO CLI(`trl rloo`)를 사용합니다:

```bash
trl rloo \
    --model_name_or_path Qwen2.5-0.5B-SFT \
    --reward_model_name_or_path Qwen2.5-0.5B-Reward \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --output_dir Qwen2.5-0.5B-RLOO \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 64 \
    --num_generations 4
```

동등한 Python 코드(`RLOOTrainer` / `RLOOConfig`):
```python
from trl import RLOOTrainer, RLOOConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen2.5-0.5B-Reward", num_labels=1
)

config = RLOOConfig(
    output_dir="Qwen2.5-0.5B-RLOO",
    per_device_train_batch_size=64,
    learning_rate=3e-6,
    num_generations=4,
)

trainer = RLOOTrainer(
    model="Qwen2.5-0.5B-SFT",
    reward_funcs=reward_model,   # a reward model (or a callable reward function)
    args=config,
    train_dataset=dataset,       # prompt-only dataset
    processing_class=tokenizer,
)
trainer.train()
```

**4단계: 평가**

```python
from transformers import pipeline

# Load aligned model
generator = pipeline("text-generation", model="Qwen2.5-0.5B-RLOO")

# Test
prompt = "Explain quantum computing to a 10-year-old"
output = generator(prompt, max_length=200)[0]["generated_text"]
print(output)
```

### 워크플로 2: DPO를 사용한 간단한 선호 정렬

보상 모델 없이 선호에 맞게 모델을 정렬합니다.

이 체크리스트를 복사하세요:

```
DPO Training:
- [ ] Step 1: Prepare preference dataset
- [ ] Step 2: Configure DPO
- [ ] Step 3: Train with DPOTrainer
- [ ] Step 4: Evaluate alignment
```

**1단계: 선호 데이터셋 준비**

데이터셋 형식:
```json
{
  "prompt": "What is the capital of France?",
  "chosen": "The capital of France is Paris.",
  "rejected": "I don't know."
}
```

데이터셋 로드:
```python
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
# Or load your own
# dataset = load_dataset("json", data_files="preferences.json")
```

**2단계: DPO 구성**

```python
from trl import DPOConfig

config = DPOConfig(
    output_dir="Qwen2.5-0.5B-DPO",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-7,
    beta=0.1,  # KL penalty strength
    max_prompt_length=512,
    max_length=1024,
    logging_steps=10
)
```

**3단계: DPOTrainer로 학습**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

trainer = DPOTrainer(
    model=model,
    args=config,
    train_dataset=dataset,
    processing_class=tokenizer
)

trainer.train()
trainer.save_model()
```

**CLI 대안**:
```bash
trl dpo \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name argilla/Capybara-Preferences \
    --output_dir Qwen2.5-0.5B-DPO \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-7 \
    --beta 0.1
```

### 워크플로 3: GRPO를 사용한 메모리 효율적인 온라인 RL

최소한의 메모리로 강화 학습을 사용해 학습합니다.

GRPO에 대한 심층적인 지침 — 보상 함수 설계, 핵심 학습 인사이트(손실 동작, 모드 붕괴, 튜닝), 고급 다단계 패턴 — 은 **[references/grpo-training.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/grpo-training.md)**를 참조하세요. 바로 사용할 수 있는 학습 스크립트는 **[templates/basic_grpo_training.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/templates/basic_grpo_training.py)**에 있습니다.

이 체크리스트를 복사하세요:

```
GRPO Training:
- [ ] Step 1: Define reward function
- [ ] Step 2: Configure GRPO
- [ ] Step 3: Train with GRPOTrainer
```

**1단계: 보상 함수 정의**

```python
def reward_function(completions, **kwargs):
    """
    Compute rewards for completions.

    Args:
        completions: List of generated texts

    Returns:
        List of reward scores (floats)
    """
    rewards = []
    for completion in completions:
        # Example: reward based on length and unique words
        score = len(completion.split())  # Favor longer responses
        score += len(set(completion.lower().split()))  # Reward unique words
        rewards.append(score)
    return rewards
```

또는 보상 모델을 사용합니다:
```python
from transformers import pipeline

reward_model = pipeline("text-classification", model="reward-model-path")

def reward_from_model(completions, prompts, **kwargs):
    # Combine prompt + completion
    full_texts = [p + c for p, c in zip(prompts, completions)]
    # Get reward scores
    results = reward_model(full_texts)
    return [r["score"] for r in results]
```

**2단계: GRPO 구성**

```python
from trl import GRPOConfig

config = GRPOConfig(
    output_dir="Qwen2-GRPO",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    num_generations=4,  # Generate 4 completions per prompt
    max_new_tokens=128
)
```

**3단계: GRPOTrainer로 학습**

```python
from datasets import load_dataset
from trl import GRPOTrainer

# Load prompt-only dataset
dataset = load_dataset("trl-lib/tldr", split="train")

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_function,  # Your reward function
    args=config,
    train_dataset=dataset
)

trainer.train()
```

**CLI**:
```bash
trl grpo \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/tldr \
    --output_dir Qwen2-GRPO \
    --num_generations 4
```

## 언제 TRL을 사용하고 언제 대안을 사용할까요?

**다음과 같은 경우 TRL을 사용하세요:**
- 모델을 인간의 선호에 맞게 정렬해야 함
- 선호 데이터(선택/거부 쌍)가 있음
- 강화 학습(RLOO, GRPO)을 사용하고 싶음
- 보상 모델을 학습해야 함
- RLHF(전체 파이프라인)를 수행함

**방법 선택**:
- **SFT**: 프롬프트-완성 쌍이 있고 기본적인 instruction following을 원함
- **DPO**: 선호 데이터가 있고 간단한 정렬을 원함(보상 모델 불필요)
- **RLOO**: 보상 모델이 있고 온라인 RL을 원함(보상 모델 기반 RLHF 경로; PPO는 TRL 1.x에서 제거됨)
- **GRPO**: 메모리가 제한되어 있고 보상 함수를 사용한 온라인 RL을 원함
- **보상 모델**: RLHF 파이프라인을 구축하며 생성 결과에 점수를 매겨야 함

**대신 다음 대안을 사용하세요:**
- **HuggingFace Trainer**: RL이 필요 없는 기본 파인 튜닝
- **Axolotl**: YAML 기반 학습 구성
- **LitGPT**: 교육용 최소 파인 튜닝
- **Unsloth**: 빠른 LoRA 학습

## 일반적인 문제

**문제: DPO 학습 중 OOM**

배치 크기와 시퀀스 길이를 줄이세요:
```python
config = DPOConfig(
    per_device_train_batch_size=1,  # Reduce from 4
    max_length=512,  # Reduce from 1024
    gradient_accumulation_steps=8  # Maintain effective batch
)
```

또는 그래디언트 체크포인트를 사용하세요:
```python
model.gradient_checkpointing_enable()
```

**문제: 정렬 품질이 낮음**

beta 매개변수를 조정하세요:
```python
# Higher beta = more conservative (stays closer to reference)
config = DPOConfig(beta=0.5)  # Default 0.1

# Lower beta = more aggressive alignment
config = DPOConfig(beta=0.01)
```

**문제: 보상 모델이 학습되지 않음**

손실 유형과 학습률을 확인하세요:
```python
config = RewardConfig(
    learning_rate=1e-5,  # Try different LR
    num_train_epochs=3  # Train longer
)
```

선호 데이터셋에 명확한 승자가 있는지 확인하세요:
```python
# Verify dataset
print(dataset[0])
# Should have clear chosen > rejected
```

**문제: 온라인 RL(RLOO/GRPO) 학습이 불안정함**

참조 정책을 향한 KL/beta 정규화를 조정하세요:
```python
from trl import RLOOConfig

config = RLOOConfig(
    beta=0.05,          # KL coefficient toward the reference model (increase for stability)
    num_generations=4,  # more samples per prompt = lower-variance advantage estimates
)
```

## 고급 주제

**SFT 학습 가이드**: 데이터셋 형식, 채팅 템플릿, 패킹 전략, 멀티 GPU 학습은 [references/sft-training.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/sft-training.md)를 참조하세요.

**DPO 변형**: 권장 하이퍼파라미터와 IPO, cDPO, RPO 및 기타 DPO 손실 함수는 [references/dpo-variants.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/dpo-variants.md)를 참조하세요.

**보상 모델링**: 결과 보상과 과정 보상, Bradley-Terry 손실, 보상 모델 평가는 [references/reward-modeling.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/reward-modeling.md)를 참조하세요.

**온라인 RL 방법**: 자세한 구성과 함께 PPO, GRPO, RLOO, OnlineDPO를 살펴보려면 [references/online-rl.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/online-rl.md)를 참조하세요.

**GRPO 심층 분석**: 전문가 수준의 GRPO 패턴 — 보상 함수 설계 철학, 학습 인사이트(손실이 증가하는 이유, 모드 붕괴 감지), 하이퍼파라미터 튜닝, 다단계 학습, 문제 해결 — 은 [references/grpo-training.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/references/grpo-training.md)를 참조하세요. 바로 사용할 수 있는 템플릿은 [templates/basic_grpo_training.py](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/training/trl-fine-tuning/templates/basic_grpo_training.py)에 있습니다.

## 하드웨어 요구 사항

- **GPU**: NVIDIA(CUDA 필요)
- **VRAM**: 모델과 방법에 따라 다름
  - SFT 7B: 16GB(LoRA 사용 시)
  - DPO 7B: 24GB(참조 모델 저장)
  - RLOO 7B: 40GB(정책 + 보상 모델)
  - GRPO 7B: 24GB(더 높은 메모리 효율)
- **멀티 GPU**: `accelerate`를 통해 지원
- **혼합 정밀도**: BF16 권장(A100/H100)

**메모리 최적화**:
- 모든 방법에 LoRA/QLoRA 사용
- 그래디언트 체크포인트 활성화
- 그래디언트 누적과 함께 더 작은 배치 크기 사용

## 리소스

- 문서: https://huggingface.co/docs/trl/
- GitHub: https://github.com/huggingface/trl
- 논문:
  - "인간 피드백을 따르도록 언어 모델 학습" (InstructGPT, 2022)
  - "직접 선호 최적화: 당신의 언어 모델은 비밀스러운 보상 모델이다" (DPO, 2023)
  - "그룹 상대 정책 최적화" (GRPO, 2024)
- 예제: https://github.com/huggingface/trl/tree/main/examples/scripts

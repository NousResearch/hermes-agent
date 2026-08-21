---
title: "Simpo — DPO보다 간단한 참조 모델 없는 선호 정렬"
sidebar_label: "Simpo"
description: "DPO보다 간단한 참조 모델 없는 선호 정렬"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Simpo

참조 모델 없이 DPO보다 뛰어난 성능을 내는 더 간단한 선호 정렬 방법입니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 선택 사항 — `hermes skills install official/mlops/simpo`로 설치 |
| 경로 | `optional-skills/mlops/simpo` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `torch`, `transformers`, `datasets`, `trl`, `accelerate` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Post-Training`, `SimPO`, `Preference Optimization`, `Alignment`, `DPO Alternative`, `Reference-Free`, `LLM Alignment`, `Efficient Training` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# SimPO - 간단한 선호 최적화

## 빠른 시작

SimPO는 참조 모델이 필요하지 않으면서 DPO보다 뛰어난 성능을 내는 참조 모델 없는 선호 최적화 방법입니다.

**설치**:
```bash
# Create environment
conda create -n simpo python=3.10 && conda activate simpo

# Install PyTorch 2.2.2
# Visit: https://pytorch.org/get-started/locally/

# Install alignment-handbook
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
python -m pip install .

# Install Flash Attention 2
python -m pip install flash-attn --no-build-isolation
```

**학습** (Mistral 7B):
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py \
  training_configs/mistral-7b-base-simpo.yaml
```

## 일반적인 워크플로

### 워크플로 1: 기본 모델에서 학습 (Mistral 7B)

**설정** (`mistral-7b-base-simpo.yaml`):
```yaml
# Model
model_name_or_path: mistralai/Mistral-7B-v0.1
torch_dtype: bfloat16

# Dataset
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
  - train_prefs
  - test_prefs

# SimPO hyperparameters
beta: 2.0                  # Reward scaling (2.0-10.0)
gamma_beta_ratio: 0.5       # Target margin (0-1)
loss_type: sigmoid          # sigmoid or hinge
sft_weight: 0.0             # Optional SFT regularization

# Training
learning_rate: 5e-7         # Critical: 3e-7 to 1e-6
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# Output
output_dir: ./outputs/mistral-7b-simpo
```

**학습 시작**:
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml
```

### 워크플로 2: 지시 튜닝 모델 미세 조정 (Llama 3 8B)

**설정** (`llama3-8b-instruct-simpo.yaml`):
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

dataset_mixer:
  argilla/ultrafeedback-binarized-preferences-cleaned: 1.0

beta: 2.5
gamma_beta_ratio: 0.5
learning_rate: 5e-7
sft_weight: 0.1             # Add SFT loss to preserve capabilities

num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
output_dir: ./outputs/llama3-8b-simpo
```

**시작**:
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/llama3-8b-instruct-simpo.yaml
```

### 워크플로 3: 추론 집약적 작업 (더 낮은 LR)

**수학/코드 작업의 경우**:
```yaml
model_name_or_path: deepseek-ai/deepseek-math-7b-base

dataset_mixer:
  argilla/distilabel-math-preference-dpo: 1.0

beta: 5.0                   # Higher for stronger signal
gamma_beta_ratio: 0.7       # Larger margin
learning_rate: 3e-7         # Lower LR for reasoning
sft_weight: 0.0

num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

## 대안을 고려한 사용 시점

**다음과 같은 경우 SimPO를 사용하세요**:
- DPO보다 간단한 학습을 원할 때 (참조 모델 없음)
- 선호 데이터가 있을 때 (선택/거부 쌍)
- DPO보다 더 나은 성능이 필요할 때
- 컴퓨팅 리소스가 제한적일 때
- 단일 노드 학습으로 충분할 때

**알고리즘 선택**:
- **SimPO**: 가장 간단하고 성능이 뛰어나며 참조 모델이 필요 없음
- **DPO**: 참조 모델을 기준으로 삼아야 하고 더 보수적인 방식이 필요할 때
- **PPO**: 최대한의 제어가 필요하고 보상 모델과 복잡한 설정을 사용할 수 있을 때
- **GRPO**: 메모리 효율적인 RL이 필요하고 critic이 없을 때

**다음과 같은 경우에는 대안을 사용하세요**:
- **OpenRLHF**: 멀티 노드 분산 학습, PPO/GRPO
- **TRL**: 하나의 프레임워크에서 여러 방법이 필요할 때
- **DPO**: 검증된 기준선과 비교할 때

## 일반적인 문제

**문제: 손실 발산**

학습률을 낮추세요:
```yaml
learning_rate: 3e-7  # Reduce from 5e-7
```

beta를 낮추세요:
```yaml
beta: 1.0  # Reduce from 2.0
```

**문제: 모델이 기능을 잊음**

SFT 정규화를 추가하세요:
```yaml
sft_weight: 0.1  # Add SFT loss component
```

**문제: 선호 분리가 좋지 않음**

beta와 마진을 높이세요:
```yaml
beta: 5.0            # Increase from 2.0
gamma_beta_ratio: 0.8  # Increase from 0.5
```

**문제: 학습 중 OOM**

배치 크기를 줄이세요:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # Maintain effective batch
```

그래디언트 체크포인트를 활성화하세요:
```yaml
gradient_checkpointing: true
```

## 고급 주제

**손실 함수**: sigmoid와 hinge 손실, 수학적 공식, 각각을 사용할 시점은 [references/loss-functions.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/loss-functions.md)를 참조하세요.

**하이퍼파라미터 튜닝**: beta, gamma, 학습률 선택 가이드와 모델 크기별 권장 사항은 [references/hyperparameters.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/hyperparameters.md)를 참조하세요.

**데이터셋 준비**: 선호 데이터 형식, 품질 필터링, 사용자 지정 데이터셋 생성 방법은 [references/datasets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/datasets.md)를 참조하세요.

## 하드웨어 요구 사항

- **GPU**: NVIDIA A100/H100 권장
- **VRAM**:
  - 7B 모델: 1× A100 40GB (DeepSpeed ZeRO-3)
  - 8B 모델: 2× A100 40GB
  - 70B 모델: 8× A100 80GB
- **단일 노드**: DeepSpeed ZeRO-3로 충분
- **혼합 정밀도**: BF16 권장

**메모리 최적화**:
- DeepSpeed ZeRO-3 (기본 설정)
- 그래디언트 체크포인트
- Flash Attention 2

## 리소스

- 논문: https://arxiv.org/abs/2405.14734 (NeurIPS 2024)
- GitHub: https://github.com/princeton-nlp/SimPO
- 모델: https://huggingface.co/princeton-nlp
- Alignment Handbook: https://github.com/huggingface/alignment-handbook

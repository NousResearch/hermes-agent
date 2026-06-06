---
title: "Simpo Training — LLM 정렬을 위한 간단한 선호도 최적화(Simple Preference Optimization)"
sidebar_label: "Simpo Training"
description: "LLM 정렬을 위한 간단한 선호도 최적화(Simple Preference Optimization)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Simpo Training

LLM 정렬(alignment)을 위한 간단한 선호도 최적화 기법입니다. DPO에 대한 레퍼런스-프리(reference-free) 대안으로 더 나은 성능을 보여줍니다(AlpacaEval 2.0에서 +6.4점). 기준(reference) 모델이 필요하지 않아 DPO보다 효율적입니다. DPO/PPO보다 더 간단하고 빠른 학습으로 선호도 정렬을 원할 때 사용하세요.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/simpo`로 설치 |
| Path | `optional-skills/mlops/simpo` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `torch`, `transformers`, `datasets`, `trl`, `accelerate` |
| Platforms | linux, macos, windows |
| Tags | `Post-Training`, `SimPO`, `Preference Optimization`, `Alignment`, `DPO Alternative`, `Reference-Free`, `LLM Alignment`, `Efficient Training` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# SimPO - 간단한 선호도 최적화 (Simple Preference Optimization)

## 빠른 시작

SimPO는 기준 모델 없이도 DPO보다 뛰어난 성능을 발휘하는 레퍼런스-프리 선호도 최적화 방법입니다.

**설치**:
```bash
# 환경 생성
conda create -n simpo python=3.10 && conda activate simpo

# PyTorch 2.2.2 설치
# 방문: https://pytorch.org/get-started/locally/

# alignment-handbook 설치
git clone https://github.com/huggingface/alignment-handbook.git
cd alignment-handbook
python -m pip install .

# Flash Attention 2 설치
python -m pip install flash-attn --no-build-isolation
```

**학습** (Mistral 7B):
```bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py \
  training_configs/mistral-7b-base-simpo.yaml
```

## 일반적인 워크플로우

### 워크플로우 1: 베이스 모델부터 학습 (Mistral 7B)

**설정(Config)** (`mistral-7b-base-simpo.yaml`):
```yaml
# 모델
model_name_or_path: mistralai/Mistral-7B-v0.1
torch_dtype: bfloat16

# 데이터셋
dataset_mixer:
  HuggingFaceH4/ultrafeedback_binarized: 1.0
dataset_splits:
  - train_prefs
  - test_prefs

# SimPO 하이퍼파라미터
beta: 2.0                  # 보상 스케일링 (2.0-10.0)
gamma_beta_ratio: 0.5       # 목표 마진 (0-1)
loss_type: sigmoid          # sigmoid 또는 hinge
sft_weight: 0.0             # (선택) SFT 정규화

# 학습
learning_rate: 5e-7         # 중요: 3e-7 에서 1e-6 사이
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8

# 출력
output_dir: ./outputs/mistral-7b-simpo
```

**학습 실행**:
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/mistral-7b-base-simpo.yaml
```

### 워크플로우 2: 인스트럭트(Instruct) 모델 파인튜닝 (Llama 3 8B)

**설정** (`llama3-8b-instruct-simpo.yaml`):
```yaml
model_name_or_path: meta-llama/Meta-Llama-3-8B-Instruct

dataset_mixer:
  argilla/ultrafeedback-binarized-preferences-cleaned: 1.0

beta: 2.5
gamma_beta_ratio: 0.5
learning_rate: 5e-7
sft_weight: 0.1             # 능력을 보존하기 위해 SFT 손실을 추가

num_train_epochs: 1
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
output_dir: ./outputs/llama3-8b-simpo
```

**실행**:
```bash
accelerate launch --config_file accelerate_configs/deepspeed_zero3.yaml \
  scripts/run_simpo.py training_configs/llama3-8b-instruct-simpo.yaml
```

### 워크플로우 3: 추론 중심 작업 (낮은 학습률)

**수학/코드 작업의 경우**:
```yaml
model_name_or_path: deepseek-ai/deepseek-math-7b-base

dataset_mixer:
  argilla/distilabel-math-preference-dpo: 1.0

beta: 5.0                   # 더 강력한 신호를 위해 높게 설정
gamma_beta_ratio: 0.7       # 더 큰 마진
learning_rate: 3e-7         # 추론을 위해 더 낮은 LR 사용
sft_weight: 0.0

num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

## 사용 시기 및 대안

**SimPO를 사용하는 경우**:
- DPO보다 더 간단한 학습을 원할 때 (기준 모델 불필요)
- 선호도 데이터가 있을 때 (선택/거절된 쌍)
- DPO보다 더 나은 성능이 필요할 때
- 컴퓨팅 리소스가 제한적일 때
- 단일 노드 학습으로 충분할 때

**알고리즘 선택**:
- **SimPO**: 가장 간단하고 성능이 뛰어나며 기준 모델이 없음
- **DPO**: 기준 모델 베이스라인이 필요하며 더 보수적임
- **PPO**: 제어력이 가장 높지만 보상 모델이 필요하며 설정이 복잡함
- **GRPO**: 메모리 효율적인 RL, 크리틱(critic) 모델이 없음

**대신 사용할 수 있는 대안**:
- **OpenRLHF**: 다중 노드 분산 학습, PPO/GRPO
- **TRL**: 하나의 프레임워크에 여러 방법이 필요할 때
- **DPO**: 확립된 베이스라인과 비교할 때

## 일반적인 문제

**문제: 손실 발산(Loss divergence)**

학습률을 줄이세요:
```yaml
learning_rate: 3e-7  # 5e-7에서 줄임
```

beta를 줄이세요:
```yaml
beta: 1.0  # 2.0에서 줄임
```

**문제: 모델이 기존 능력을 잊어버림**

SFT 정규화를 추가하세요:
```yaml
sft_weight: 0.1  # SFT 손실 구성 요소를 추가
```

**문제: 선호도 분리가 잘 되지 않음**

beta와 마진을 높이세요:
```yaml
beta: 5.0            # 2.0에서 높임
gamma_beta_ratio: 0.8  # 0.5에서 높임
```

**문제: 학습 중 메모리 부족 (OOM)**

배치 크기를 줄이세요:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16  # 유효 배치는 유지
```

그래디언트 체크포인팅을 활성화하세요:
```yaml
gradient_checkpointing: true
```

## 고급 주제

**손실 함수(Loss functions)**: 시그모이드(sigmoid)와 힌지(hinge) 손실 비교, 수학적 공식, 각각의 사용 시기에 대해서는 [references/loss-functions.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/loss-functions.md)를 참조하세요.

**하이퍼파라미터 튜닝**: beta, gamma, 학습률 선택 가이드와 모델 크기별 권장 사항은 [references/hyperparameters.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/hyperparameters.md)를 참조하세요.

**데이터셋 준비**: 선호도 데이터 형식, 품질 필터링, 커스텀 데이터셋 생성에 대해서는 [references/datasets.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/simpo/references/datasets.md)를 참조하세요.

## 하드웨어 요구 사항

- **GPU**: NVIDIA A100/H100 권장
- **VRAM**:
  - 7B 모델: 1× A100 40GB (DeepSpeed ZeRO-3)
  - 8B 모델: 2× A100 40GB
  - 70B 모델: 8× A100 80GB
- **단일 노드(Single-node)**: DeepSpeed ZeRO-3으로 충분
- **혼합 정밀도(Mixed precision)**: BF16 권장

**메모리 최적화**:
- DeepSpeed ZeRO-3 (기본 설정)
- 그래디언트 체크포인팅
- Flash Attention 2

## 리소스

- 논문: https://arxiv.org/abs/2405.14734 (NeurIPS 2024)
- GitHub: https://github.com/princeton-nlp/SimPO
- 모델: https://huggingface.co/princeton-nlp
- Alignment Handbook: https://github.com/huggingface/alignment-handbook

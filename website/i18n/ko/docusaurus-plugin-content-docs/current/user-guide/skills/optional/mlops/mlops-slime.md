---
title: "Slime — Megatron 및 SGLang을 활용한 LLM RL 후학습"
sidebar_label: "Slime"
description: "Megatron 및 SGLang을 활용한 LLM RL 후학습"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Slime

LLM을 Megatron 및 SGLang으로 RL 후학습합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/slime`으로 설치 |
| 경로 | `optional-skills/mlops/slime` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `sglang-router>=0.2.3`, `ray`, `torch>=2.0.0`, `transformers>=4.40.0` |
| 플랫폼 | linux, macos |
| 태그 | `Reinforcement Learning`, `Megatron-LM`, `SGLang`, `GRPO`, `Post-Training`, `GLM` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 불러오는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 보게 되는 내용입니다.
:::

# slime: RL 확장을 위한 LLM 후학습 프레임워크

slime은 칭화대학교 THUDM 팀이 만든 LLM 후학습 프레임워크로, GLM-4.5, GLM-4.6, GLM-4.7을 구동합니다. Megatron-LM을 학습에, SGLang을 고처리량 롤아웃 생성에 연결합니다.

## slime을 사용하는 경우

**다음이 필요하다면 slime을 선택하세요:**
- SGLang 추론과 함께 사용하는 Megatron-LM 네이티브 학습
- 유연한 데이터 버퍼를 활용한 사용자 정의 데이터 생성 워크플로
- GLM, Qwen3, DeepSeek V3 또는 Llama 3 모델 학습
- 프로덕션 지원(Z.ai)을 갖춘 연구용 프레임워크

**다음과 같은 경우에는 대안을 고려하세요:**
- 엔터프라이즈급 안정성 기능이 필요함 → **miles** 사용
- 백엔드를 유연하게 교체하고 싶음 → **verl** 사용
- PyTorch 네이티브 추상화가 필요함 → **torchforge** 사용

## 주요 기능

- **학습**: 전체 병렬화(TP, PP, DP, SP)를 지원하는 Megatron-LM
- **롤아웃**: 라우터를 사용하는 SGLang 기반 고처리량 생성
- **데이터 버퍼**: 유연한 프롬프트 관리 및 샘플 저장
- **모델**: GLM-4.x, Qwen3, DeepSeek V3/R1, Llama 3

## 아키텍처 개요

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────────┐
│                    Data Buffer                          │
│ - Prompt initialization and management                  │
│ - Custom data generation and filtering                  │
│ - Rollout sample storage                                │
└─────────────┬───────────────────────────┬───────────────┘
              │                           │
┌─────────────▼───────────┐ ┌─────────────▼───────────────┐
│ Training (Megatron-LM)  │ │ Rollout (SGLang + Router)   │
│ - Actor model training  │ │ - Response generation       │
│ - Critic (optional)     │ │ - Reward/verifier output    │
│ - Weight sync to rollout│ │ - Multi-turn support        │
└─────────────────────────┘ └─────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

## 설치

```bash
# Recommended: Docker
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -it slimerl/slime:latest /bin/bash

# Inside container
cd /root/slime && pip install -e . --no-deps
```

### 소스에서 설치

```bash
git clone https://github.com/THUDM/slime.git
cd slime
pip install -r requirements.txt
pip install -e .
```

## 빠른 시작: GRPO 학습

```bash
# Source model configuration
source scripts/models/qwen3-4B.sh

# Launch training
python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 4 \
    --rollout-num-gpus 4 \
    --advantage-estimator grpo \
    --use-kl-loss --kl-loss-coef 0.001 \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --global-batch-size 256 \
    --num-rollout 3000 \
    --prompt-data /path/to/data.jsonl \
    ${MODEL_ARGS[@]} ${CKPT_ARGS[@]}
```

---

## 워크플로 1: 표준 GRPO 학습

그룹 상대 어드밴티지를 사용해 추론 모델을 학습할 때 이 워크플로를 사용합니다.

### 사전 요구 사항 체크리스트
- [ ] Docker 환경 또는 Megatron-LM + SGLang 설치
- [ ] 모델 체크포인트(HuggingFace 또는 Megatron 형식)
- [ ] JSONL 형식의 학습 데이터

### 1단계: 데이터 준비

```python
# data.jsonl format
{"prompt": "What is 2 + 2?", "label": "4"}
{"prompt": "Solve: 3x = 12", "label": "x = 4"}
```

또는 채팅 형식을 사용할 수 있습니다.
```python
{
    "prompt": [
        {"role": "system", "content": "You are a math tutor."},
        {"role": "user", "content": "What is 15 + 27?"}
    ],
    "label": "42"
}
```

### 2단계: 모델 구성

미리 구성된 모델 스크립트를 선택합니다.

```bash
# List available models
ls scripts/models/
# glm4-9B.sh, qwen3-4B.sh, qwen3-30B-A3B.sh, deepseek-v3.sh, llama3-8B.sh, ...

# Source your model
source scripts/models/qwen3-4B.sh
```

### 3단계: 학습 시작

```bash
python train.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --advantage-estimator grpo \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --prompt-data /path/to/train.jsonl \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --rollout-batch-size 32 \
    --n-samples-per-prompt 8 \
    --global-batch-size 256 \
    --num-rollout 3000 \
    --save-interval 100 \
    --eval-interval 50 \
    ${MODEL_ARGS[@]}
```

### 4단계: 학습 모니터링
- [ ] TensorBoard 확인: `tensorboard --logdir outputs/`
- [ ] 보상 곡선이 상승하는지 확인
- [ ] 노드 전체의 GPU 사용률 모니터링

---

## 워크플로 2: 비동기 학습

롤아웃과 학습을 겹쳐 처리해 처리량을 높이려면 비동기 모드를 사용합니다.

### 비동기를 사용하는 경우
- 긴 생성 시간이 걸리는 대형 모델
- 동기 모드에서 GPU 유휴 시간이 많음
- 버퍼링에 충분한 메모리가 있음

### 비동기 학습 시작

```bash
python train_async.py \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus 8 \
    --advantage-estimator grpo \
    --async-buffer-size 4 \
    --prompt-data /path/to/train.jsonl \
    ${MODEL_ARGS[@]}
```

### 비동기 전용 매개변수

```bash
--async-buffer-size 4        # Number of rollouts to buffer
--update-weights-interval 2  # Sync weights every N rollouts
```

---

## 워크플로 3: 멀티턴 에이전트 학습

도구 사용 또는 다단계 추론을 수행하는 에이전트를 학습할 때 이 워크플로를 사용합니다.

### 사전 요구 사항
- [ ] 멀티턴 로직을 위한 사용자 정의 generate 함수
- [ ] 도구/환경 인터페이스

### 1단계: 사용자 정의 Generate 함수 정의

```python
# custom_generate.py
async def custom_generate(args, samples, evaluation=False):
    """Multi-turn generation with tool calling."""
    for sample in samples:
        conversation = sample.prompt

        for turn in range(args.max_turns):
            # Generate response
            response = await generate_single(conversation)

            # Check for tool call
            tool_call = extract_tool_call(response)
            if tool_call:
                tool_result = execute_tool(tool_call)
                conversation.append({"role": "assistant", "content": response})
                conversation.append({"role": "tool", "content": tool_result})
            else:
                break

        sample.response = response
        sample.reward = compute_reward(sample)

    return samples
```

### 2단계: 사용자 정의 함수로 시작

```bash
python train.py \
    --custom-generate-function-path custom_generate.py \
    --max-turns 5 \
    --prompt-data /path/to/agent_data.jsonl \
    ${MODEL_ARGS[@]}
```

전체 멀티턴 검색 예시는 `examples/search-r1/`를 참조하세요.

---

## 구성 참고

### 세 가지 인자 범주

slime은 세 가지 유형의 인자를 사용합니다.

**1. Megatron 인자**(직접 전달):
```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--num-layers 32
--hidden-size 4096
```

**2. SGLang 인자**(`--sglang-` 접두사 사용):
```bash
--sglang-mem-fraction-static 0.8
--sglang-context-length 8192
--sglang-log-level INFO
```

**3. slime 인자**:
```bash
# Resource allocation
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--rollout-num-gpus 8
--colocate  # Share GPUs between training/inference

# Data
--prompt-data /path/to/data.jsonl
--input-key prompt
--label-key label

# Training loop
--num-rollout 3000
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256

# Algorithm
--advantage-estimator grpo  # or: gspo, ppo, reinforce_plus_plus
--use-kl-loss
--kl-loss-coef 0.001
```

### 주요 제약 조건

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

예: 32 × 8 = 256 × 1

---

## 데이터 버퍼 시스템

slime의 데이터 버퍼는 유연한 데이터 관리를 지원합니다.

### 기본 데이터 소스

```python
class RolloutDataSource:
    def get_samples(self, num_samples):
        """Fetch prompts from dataset."""
        return self.dataset.sample(num_samples)

    def add_samples(self, samples):
        """Called after generation (no-op by default)."""
        pass
```

### 버퍼링된 데이터 소스(오프폴리시)

```python
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self):
        self.buffer = []

    def add_samples(self, samples):
        """Store generated samples for reuse."""
        self.buffer.extend(samples)

    def buffer_filter(self, args, buffer, num_samples):
        """Custom selection logic (prioritized, stratified, etc.)."""
        return select_best(buffer, num_samples)
```

---

## 일반적인 문제와 해결 방법

### 문제: SGLang 엔진 충돌

**증상**: 학습 중 추론 엔진이 종료됨

**해결 방법**:
```bash
# Enable fault tolerance
--use-fault-tolerance

# Increase memory allocation
--sglang-mem-fraction-static 0.85

# Reduce batch size
--rollout-batch-size 16
```

### 문제: 가중치 동기화 시간 초과

**증상**: 롤아웃 후 학습이 멈춤

**해결 방법**:
```bash
# Increase sync interval
--update-weights-interval 5

# Use colocated mode (no network transfer)
--colocate
```

### 문제: 학습 중 OOM

**증상**: 역전파 중 CUDA OOM 발생

**해결 방법**:
```bash
# Enable gradient checkpointing
--recompute-activations

# Reduce micro-batch size
--micro-batch-size 1

# Enable sequence parallelism
--sequence-parallel
```

### 문제: 데이터 로딩 속도 저하

**증상**: 데이터를 가져오는 동안 GPU가 유휴 상태임

**해결 방법**:
```bash
# Increase data workers
--num-data-workers 4

# Use streaming dataset
--streaming-data
```

---

## 지원 모델

| 모델 제품군 | 구성 |
|--------------|----------------|
| GLM | GLM-4.5, GLM-4.6, GLM-4.7, GLM-Z1-9B |
| Qwen | Qwen3 (4B, 8B, 30B-A3B), Qwen3-MoE, Qwen2.5 |
| DeepSeek | V3, V3.1, R1 |
| Llama | Llama 3 (8B, 70B) |
| 기타 | Kimi K2, Moonlight-16B |

각 모델에는 `scripts/models/`에 미리 구성된 스크립트가 있습니다.

---

## 고급 주제

### 공동 배치 모드

메모리 사용량을 줄이려면 학습과 추론에서 GPU를 공유합니다.

```bash
python train.py \
    --colocate \
    --actor-num-gpus-per-node 8 \
    --sglang-mem-fraction-static 0.4 \
    ${MODEL_ARGS[@]}
```

### 사용자 정의 보상 모델

```python
# custom_rm.py
class CustomRewardModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def compute_reward(self, prompts, responses):
        inputs = self.tokenize(prompts, responses)
        scores = self.model(inputs)
        return scores.tolist()
```

```bash
--custom-rm-path custom_rm.py
```

### 평가 멀티태스크

```bash
--eval-prompt-data aime /path/to/aime.jsonl \
--eval-prompt-data gsm8k /path/to/gsm8k.jsonl \
--n-samples-per-eval-prompt 16
```

---

## 리소스

- **문서**: https://thudm.github.io/slime/
- **GitHub**: https://github.com/THUDM/slime
- **블로그**: https://lmsys.org/blog/2025-07-09-slime/
- **예시**: 14개 이상의 완성된 예시는 `examples/` 디렉터리를 참조하세요.

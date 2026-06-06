---
title: "Slime Rl Training — Megatron+SGLang 프레임워크인 slime을 사용한 LLM 강화 학습(RL) 포스트 트레이닝(post-training) 가이드 제공"
sidebar_label: "Slime Rl Training"
description: "Megatron+SGLang 프레임워크인 slime을 사용한 LLM 강화 학습(RL) 포스트 트레이닝(post-training) 가이드 제공"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# Slime RL Training

Megatron+SGLang 프레임워크인 slime을 사용한 LLM RL 포스트 트레이닝에 대한 가이드를 제공합니다. GLM 모델을 학습하거나, 커스텀 데이터 생성 워크플로우를 구현하거나, RL 확장을 위해 긴밀하게 결합된 Megatron-LM 연동이 필요할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/slime`로 설치 |
| 경로 | `optional-skills/mlops/slime` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `sglang-router>=0.2.3`, `ray`, `torch>=2.0.0`, `transformers>=4.40.0` |
| 플랫폼 | linux, macos |
| 태그 | `Reinforcement Learning`, `Megatron-LM`, `SGLang`, `GRPO`, `Post-Training`, `GLM` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# slime: RL 확장을 위한 LLM 포스트 트레이닝 프레임워크

slime은 칭화대의 THUDM 팀이 개발한 LLM 포스트 트레이닝 프레임워크로, GLM-4.5, GLM-4.6, GLM-4.7의 기반 기술입니다. 이 프레임워크는 학습을 위한 Megatron-LM과 고처리량 롤아웃(rollout) 생성을 위한 SGLang을 연결합니다.

## slime을 사용하는 경우

**다음의 경우에 slime을 선택하세요:**
- SGLang 추론 기능이 포함된 Megatron-LM 네이티브 학습이 필요할 때
- 유연한 데이터 버퍼를 통해 커스텀 데이터 생성 워크플로우가 필요할 때
- GLM, Qwen3, DeepSeek V3, Llama 3 모델을 학습할 때
- 프로덕션(Z.ai)이 지원하는 연구용 프레임워크가 필요할 때

**대신 다음 대안을 고려해 보세요:**
- 엔터프라이즈 수준의 안정성 기능이 필요할 때 → **miles** 사용
- 유연하게 백엔드를 교체하고 싶을 때 → **verl** 사용
- PyTorch 네이티브 추상화가 필요할 때 → **torchforge** 사용

## 주요 기능

- **학습 (Training)**: 전체 병렬 처리 기능(TP, PP, DP, SP)을 지원하는 Megatron-LM
- **롤아웃 (Rollout)**: 라우터를 사용한 SGLang 기반의 고처리량 생성
- **데이터 버퍼 (Data Buffer)**: 유연한 프롬프트 관리 및 샘플 저장 기능
- **모델 지원 (Models)**: GLM-4.x, Qwen3, DeepSeek V3/R1, Llama 3

## 아키텍처 개요

<!-- ascii-guard-ignore -->
```
┌─────────────────────────────────────────────────────────┐
│                    Data Buffer (데이터 버퍼)            │
│ - 프롬프트 초기화 및 관리                               │
│ - 커스텀 데이터 생성 및 필터링                          │
│ - 롤아웃 샘플 저장                                      │
└─────────────┬───────────────────────────┬───────────────┘
              │                           │
┌─────────────▼───────────┐ ┌─────────────▼───────────────┐
│ Training (Megatron-LM)  │ │ Rollout (SGLang + Router)   │
│ - Actor 모델 학습       │ │ - 응답 생성                 │
│ - Critic 모델 (선택적)  │ │ - 보상(Reward)/검증기 결과  │
│ - 롤아웃으로의 가중치 동기화│ │ - 멀티 턴(Multi-turn) 지원│
└─────────────────────────┘ └─────────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

## 설치

```bash
# 권장 방식: Docker
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -it slimerl/slime:latest /bin/bash

# 컨테이너 내부에서 실행
cd /root/slime && pip install -e . --no-deps
```

### 소스로부터 설치

```bash
git clone https://github.com/THUDM/slime.git
cd slime
pip install -r requirements.txt
pip install -e .
```

## 빠른 시작: GRPO 학습

```bash
# 소스 모델 구성
source scripts/models/qwen3-4B.sh

# 학습 시작
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

## 워크플로우 1: 표준 GRPO 학습

그룹-상대적 이점(group-relative advantages)을 이용해 추론 모델을 학습할 때 이 워크플로우를 사용하세요.

### 필수 조건 체크리스트
- [ ] Docker 환경 혹은 Megatron-LM + SGLang 설치 완료
- [ ] 모델 체크포인트 (HuggingFace 혹은 Megatron 포맷)
- [ ] JSONL 포맷의 학습 데이터

### 1단계: 데이터 준비

```python
# data.jsonl 포맷
{"prompt": "What is 2 + 2?", "label": "4"}
{"prompt": "Solve: 3x = 12", "label": "x = 4"}
```

또는 채팅(chat) 포맷을 사용:
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

미리 구성된 모델 스크립트를 선택하세요:

```bash
# 지원되는 모델 확인
ls scripts/models/
# glm4-9B.sh, qwen3-4B.sh, qwen3-30B-A3B.sh, deepseek-v3.sh, llama3-8B.sh, ...

# 해당 모델 파일 소싱
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
- [ ] 보상(Reward) 곡선이 증가하는지 확인
- [ ] 여러 노드에 걸친 GPU 사용량(utilization) 모니터링

---

## 워크플로우 2: 비동기식(Asynchronous) 학습

롤아웃과 학습을 오버랩시켜 더 높은 처리량을 원한다면 비동기 모드를 사용하세요.

### 비동기 모드를 사용하는 경우
- 생성 시간이 오래 걸리는 대규모 모델
- 동기 모드에서 GPU 유휴 시간(idle time)이 높을 경우
- 버퍼링에 사용할 충분한 메모리가 있는 경우

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

### 비동기 전용 파라미터

```bash
--async-buffer-size 4        # 버퍼에 유지할 롤아웃 개수
--update-weights-interval 2  # 매 N회 롤아웃마다 가중치 동기화
```

---

## 워크플로우 3: 멀티 턴 에이전트(Multi-Turn Agentic) 학습

툴(tool) 사용이나 다단계 추론이 필요한 에이전트를 학습시킬 때 이 워크플로우를 사용하세요.

### 필수 조건
- [ ] 멀티 턴 로직을 위한 커스텀 생성 함수(generate function)
- [ ] 툴/환경 인터페이스

### 1단계: 커스텀 생성 함수(Generate Function) 정의

```python
# custom_generate.py
async def custom_generate(args, samples, evaluation=False):
    """툴 사용을 포함한 멀티 턴 생성."""
    for sample in samples:
        conversation = sample.prompt

        for turn in range(args.max_turns):
            # 응답 생성
            response = await generate_single(conversation)

            # 툴 사용 호출 확인
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

### 2단계: 커스텀 함수와 함께 시작

```bash
python train.py \
    --custom-generate-function-path custom_generate.py \
    --max-turns 5 \
    --prompt-data /path/to/agent_data.jsonl \
    ${MODEL_ARGS[@]}
```

완전한 멀티 턴 검색 예제는 `examples/search-r1/`를 참고하세요.

---

## 구성(Configuration) 참고

### 3가지의 인자 카테고리

slime은 세 가지 유형의 인자(argument)를 사용합니다:

**1. Megatron 인자** (직접 전달됨):
```bash
--tensor-model-parallel-size 2
--pipeline-model-parallel-size 1
--num-layers 32
--hidden-size 4096
```

**2. SGLang 인자** (`--sglang-`를 접두사로 사용):
```bash
--sglang-mem-fraction-static 0.8
--sglang-context-length 8192
--sglang-log-level INFO
```

**3. slime 인자**:
```bash
# 리소스 할당
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--rollout-num-gpus 8
--colocate  # 학습/추론 간 GPU를 공유

# 데이터
--prompt-data /path/to/data.jsonl
--input-key prompt
--label-key label

# 학습 루프
--num-rollout 3000
--rollout-batch-size 32
--n-samples-per-prompt 8
--global-batch-size 256

# 알고리즘
--advantage-estimator grpo  # 또는: gspo, ppo, reinforce_plus_plus
--use-kl-loss
--kl-loss-coef 0.001
```

### 핵심 제약조건

```
rollout_batch_size × n_samples_per_prompt = global_batch_size × num_steps_per_rollout
```

예시: 32 × 8 = 256 × 1

---

## 데이터 버퍼 시스템

slime의 데이터 버퍼는 유연한 데이터 관리를 가능하게 합니다:

### 기본 데이터 소스(Data Source)

```python
class RolloutDataSource:
    def get_samples(self, num_samples):
        """데이터셋에서 프롬프트 가져오기."""
        return self.dataset.sample(num_samples)

    def add_samples(self, samples):
        """생성 이후 호출됨 (기본적으로는 아무 일도 하지 않음)."""
        pass
```

### 버퍼 적용된 데이터 소스 (Off-Policy 방식)

```python
class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self):
        self.buffer = []

    def add_samples(self, samples):
        """재사용을 위해 생성된 샘플 저장."""
        self.buffer.extend(samples)

    def buffer_filter(self, args, buffer, num_samples):
        """커스텀 선택 로직 (우선순위, 계층화 등)."""
        return select_best(buffer, num_samples)
```

---

## 흔한 문제와 해결책

### 문제: SGLang 엔진 크래시 (Crash)

**증상**: 학습 중반에 추론 엔진이 종료됨

**해결책**:
```bash
# 내결함성(Fault tolerance) 활성화
--use-fault-tolerance

# 메모리 할당량 증가
--sglang-mem-fraction-static 0.85

# 배치 크기 줄이기
--rollout-batch-size 16
```

### 문제: 가중치 동기화 타임아웃 (Timeout)

**증상**: 롤아웃 이후에 학습이 멈춰서 진행 안됨 (Hangs)

**해결책**:
```bash
# 동기화 주기(interval) 늘리기
--update-weights-interval 5

# colocated 모드 사용 (네트워크 전송 없음)
--colocate
```

### 문제: 학습 도중 메모리 부족 (OOM)

**증상**: Backward pass 도중 CUDA 메모리 부족(OOM)

**해결책**:
```bash
# 그래디언트 체크포인트 활성화 (Gradient checkpointing)
--recompute-activations

# 마이크로 배치 크기 줄이기
--micro-batch-size 1

# 시퀀스 병렬 처리 활성화 (Sequence parallelism)
--sequence-parallel
```

### 문제: 데이터 로딩이 너무 느림

**증상**: 데이터 패치(fetch) 도중 GPU가 유휴 상태에 빠짐

**해결책**:
```bash
# 데이터 작업자 수 늘리기
--num-data-workers 4

# 스트리밍 데이터셋 사용
--streaming-data
```

---

## 지원되는 모델

| 모델 제품군 | 지원 구성 (Configurations) |
|--------------|----------------|
| GLM | GLM-4.5, GLM-4.6, GLM-4.7, GLM-Z1-9B |
| Qwen | Qwen3 (4B, 8B, 30B-A3B), Qwen3-MoE, Qwen2.5 |
| DeepSeek | V3, V3.1, R1 |
| Llama | Llama 3 (8B, 70B) |
| 기타 | Kimi K2, Moonlight-16B |

각 모델은 `scripts/models/` 경로에 미리 구성된 스크립트가 있습니다.

---

## 고급 주제

### Co-location 모드

메모리 절약을 위해 학습과 추론에 같은 GPU를 공유합니다:

```bash
python train.py \
    --colocate \
    --actor-num-gpus-per-node 8 \
    --sglang-mem-fraction-static 0.4 \
    ${MODEL_ARGS[@]}
```

### 커스텀 보상 모델 (Reward Model)

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

### 다중 태스크 평가(Evaluation)

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
- **예제**: `examples/` 디렉토리에 14개 이상의 다양한 예제가 준비되어 있습니다.

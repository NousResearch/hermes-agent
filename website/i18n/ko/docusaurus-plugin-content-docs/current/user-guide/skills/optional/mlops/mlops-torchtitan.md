---
title: "Distributed Llm Pretraining Torchtitan"
sidebar_label: "Distributed Llm Pretraining Torchtitan"
description: "4D 병렬 처리 (FSDP2, TP, PP, CP)를 갖춘 torchtitan을 활용하여 PyTorch 네이티브 분산 LLM 사전 학습 환경 제공"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py를 통해 스킬의 SKILL.md에서 자동으로 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 수정하세요. */}

# 분산 LLM 사전 학습 Torchtitan (Distributed Llm Pretraining Torchtitan)

4D 병렬 처리 (FSDP2, TP, PP, CP)를 갖춘 torchtitan을 활용하여 PyTorch 네이티브 분산 LLM 사전 학습 환경을 제공합니다. 8개부터 512개 이상의 GPU 규모에서 Float8, torch.compile, 분산 체크포인트 등을 적용하여 Llama 3.1, DeepSeek V3 또는 사용자 정의 모델을 사전 학습할 때 사용하세요.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택사항 — `hermes skills install official/mlops/torchtitan`로 설치 |
| 경로 | `optional-skills/mlops/torchtitan` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `torch>=2.6.0`, `torchtitan>=0.2.0`, `torchao>=0.5.0` |
| 플랫폼 | linux, macos |
| 태그 | `Model Architecture`, `Distributed Training`, `TorchTitan`, `FSDP2`, `Tensor Parallel`, `Pipeline Parallel`, `Context Parallel`, `Float8`, `Llama`, `Pretraining` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이는 스킬이 활성화되었을 때 에이전트가 지시사항으로 보는 내용입니다.
:::

# TorchTitan - PyTorch 네이티브 분산 LLM 사전 학습

## 빠른 시작

TorchTitan은 조립 가능한 4D 병렬 처리(FSDP2, TP, PP, CP) 기능을 갖춘 대규모 LLM 사전 학습용 PyTorch 공식 플랫폼으로, H100 GPU 기준 베이스라인 대비 65% 이상의 속도 향상을 달성했습니다.

**설치**:
```bash
# PyPI를 통한 설치 (stable)
pip install torchtitan

# 소스로부터 설치 (최신 기능, PyTorch nightly 필요)
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
```

**토크나이저 다운로드**:
```bash
# https://huggingface.co/settings/tokens 에서 HF 토큰 발급
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

**8개 GPU에서 학습 시작**:
```bash
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh
```

## 일반적인 워크플로우

### 워크플로우 1: 단일 노드(Single node)에서 Llama 3.1 8B 사전 학습

다음 체크리스트를 복사하세요:

```
단일 노드 사전 학습:
- [ ] 1단계: 토크나이저 다운로드
- [ ] 2단계: 학습 환경 구성
- [ ] 3단계: 학습 시작
- [ ] 4단계: 모니터링 및 체크포인트
```

**1단계: 토크나이저 다운로드**

```bash
python scripts/download_hf_assets.py \
  --repo_id meta-llama/Llama-3.1-8B \
  --assets tokenizer \
  --hf_token=YOUR_HF_TOKEN
```

**2단계: 학습 환경 구성**

TOML 구성 파일을 수정하거나 생성하세요:

```toml
# llama3_8b_custom.toml
[job]
dump_folder = "./outputs"
description = "Llama 3.1 8B training"

[model]
name = "llama3"
flavor = "8B"
hf_assets_path = "./assets/hf/Llama-3.1-8B"

[optimizer]
name = "AdamW"
lr = 3e-4

[lr_scheduler]
warmup_steps = 200

[training]
local_batch_size = 2
seq_len = 8192
max_norm = 1.0
steps = 1000
dataset = "c4"

[parallelism]
data_parallel_shard_degree = -1  # FSDP를 위해 모든 GPU 사용

[activation_checkpoint]
mode = "selective"
selective_ac_option = "op"

[checkpoint]
enable = true
folder = "checkpoint"
interval = 500
```

**3단계: 학습 시작**

```bash
# 단일 노드 8 GPU 환경
CONFIG_FILE="./llama3_8b_custom.toml" ./run_train.sh

# 또는 torchrun을 통해 직접 실행
torchrun --nproc_per_node=8 \
  -m torchtitan.train \
  --job.config_file ./llama3_8b_custom.toml
```

**4단계: 모니터링 및 체크포인트**

TensorBoard 로그는 `./outputs/tb/` 에 저장됩니다:
```bash
tensorboard --logdir ./outputs/tb
```

### 워크플로우 2: SLURM을 사용한 다중 노드(Multi-node) 학습

```
다중 노드 학습:
- [ ] 1단계: 확장을 위한 병렬 처리 구성
- [ ] 2단계: SLURM 스크립트 설정
- [ ] 3단계: 작업(Job) 제출
- [ ] 4단계: 체크포인트에서 재개(Resume)
```

**1단계: 확장을 위한 병렬 처리 구성**

256개 GPU(32 노드) 환경에서 70B 모델을 학습할 경우:
```toml
[parallelism]
data_parallel_shard_degree = 32  # 32개 랭크(rank)에 걸쳐 FSDP 수행
tensor_parallel_degree = 8        # 노드 내부에서 TP 수행
pipeline_parallel_degree = 1      # 70B 모델에는 PP 미적용
context_parallel_degree = 1       # 긴 시퀀스 길이를 위해서는 이 값을 늘립니다.
```

**2단계: SLURM 스크립트 설정**

```bash
#!/bin/bash
#SBATCH --job-name=llama70b
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

srun torchrun \
  --nnodes=32 \
  --nproc_per_node=8 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m torchtitan.train \
  --job.config_file ./llama3_70b.toml
```

**3단계: 작업(Job) 제출**

```bash
sbatch multinode_trainer.slurm
```

**4단계: 체크포인트에서 재개(Resume)**

구성된 폴더 내에 체크포인트가 존재하면 학습은 자동으로 재개됩니다.

### 워크플로우 3: H100 GPU 환경에서 Float8 학습 활성화

Float8은 H100 GPU에서 30-50%의 속도 향상을 제공합니다.

```
Float8 학습:
- [ ] 1단계: torchao 설치
- [ ] 2단계: Float8 구성
- [ ] 3단계: compile 옵션과 함께 시작
```

**1단계: torchao 설치**

```bash
USE_CPP=0 pip install git+https://github.com/pytorch/ao.git
```

**2단계: Float8 구성**

TOML 구성 파일에 다음을 추가합니다:
```toml
[model]
converters = ["quantize.linear.float8"]

[quantize.linear.float8]
enable_fsdp_float8_all_gather = true
precompute_float8_dynamic_scale_for_fsdp = true
filter_fqns = ["output"]  # 출력(output) 레이어는 제외합니다

[compile]
enable = true
components = ["model", "loss"]
```

**3단계: compile 옵션과 함께 시작**

```bash
CONFIG_FILE="./llama3_8b.toml" ./run_train.sh \
  --model.converters="quantize.linear.float8" \
  --quantize.linear.float8.enable_fsdp_float8_all_gather \
  --compile.enable
```

### 워크플로우 4: 405B 모델을 위한 4D 병렬 처리

```
4D 병렬 처리 (FSDP + TP + PP + CP):
- [ ] 1단계: 시드(seed) 체크포인트 생성
- [ ] 2단계: 4D 병렬 처리 구성
- [ ] 3단계: 512개의 GPU에서 시작
```

**1단계: 시드(seed) 체크포인트 생성**

PP 스테이지 전반에 걸쳐 일관된 초기화를 위해 필요합니다:
```bash
NGPU=1 CONFIG_FILE=./llama3_405b.toml ./run_train.sh \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1
```

**2단계: 4D 병렬 처리 구성**

```toml
[parallelism]
data_parallel_shard_degree = 8   # FSDP
tensor_parallel_degree = 8       # 노드 내부에서 TP
pipeline_parallel_degree = 8     # 노드 간 PP
context_parallel_degree = 1      # 긴 시퀀스를 위한 CP

[training]
local_batch_size = 32
seq_len = 8192
```

**3단계: 512개의 GPU에서 시작**

```bash
# 64 노드 x 8 GPU = 512 GPU
srun torchrun --nnodes=64 --nproc_per_node=8 \
  -m torchtitan.train \
  --job.config_file ./llama3_405b.toml
```

## 대안 솔루션과의 비교 및 사용 시기

**다음의 경우에 TorchTitan을 사용하세요:**
- 처음부터 LLM을 사전 학습할 때 (8B ~ 405B+)
- 제3자(Third-party) 의존성 없는 PyTorch 네이티브 솔루션이 필요할 때
- 조립 가능한 4D 병렬 처리 (FSDP2, TP, PP, CP)가 필요할 때
- H100에서 Float8 지원으로 학습할 때
- torchtune이나 HuggingFace와 상호 호환되는 체크포인트가 필요할 때

**다음을 대신 사용해 보세요:**
- **Megatron-LM**: NVIDIA 환경 전용 최고 성능 발휘
- **DeepSpeed**: 더 넓은 ZeRO 최적화 생태계, 추론 지원
- **Axolotl/TRL**: 사전 학습이 아닌 파인튜닝용
- **LitGPT**: 교육 목적, 소규모 학습

## 흔한 문제들

**문제: 대규모 모델에서 메모리 부족(OOM)**

Activation checkpointing을 켜고 배치 크기를 줄입니다:
```toml
[activation_checkpoint]
mode = "full"  # "selective" 대신 "full" 사용

[training]
local_batch_size = 1
```

또는 그래디언트 누적(gradient accumulation)을 사용합니다:
```toml
[training]
local_batch_size = 1
global_batch_size = 32  # 그래디언트 누적 적용됨
```

**문제: 비동기 collectives에서 TP로 인한 높은 메모리 사용량**

환경 변수 설정:
```bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
```

**문제: Float8 학습 속도가 향상되지 않음**

Float8은 오직 큰 크기의 GEMM 연산에서만 이득이 있습니다. 작은 레이어는 필터링하세요:
```toml
[quantize.linear.float8]
filter_fqns = ["attention.wk", "attention.wv", "output", "auto_filter_small_kn"]
```

**문제: 병렬 처리 구성 변경 후 체크포인트 로드 실패**

DCP의 리샤딩(resharding) 기능을 사용하세요:
```bash
# 조각난(sharded) 체크포인트를 단일 파일로 변환
python -m torch.distributed.checkpoint.format_utils \
  dcp_to_torch checkpoint/step-1000 checkpoint.pt
```

**문제: 파이프라인 병렬 처리(Pipeline parallelism) 초기화 문제**

먼저 시드 체크포인트를 생성하세요 (워크플로우 4의 1단계 참조).

## 지원되는 모델

| 모델 | 크기 | 상태 |
|-------|-------|--------|
| Llama 3.1 | 8B, 70B, 405B | 프로덕션 |
| Llama 4 | 다양한 크기 | 실험적 단계 |
| DeepSeek V3 | 16B, 236B, 671B (MoE) | 실험적 단계 |
| GPT-OSS | 20B, 120B (MoE) | 실험적 단계 |
| Qwen 3 | 다양한 크기 | 실험적 단계 |
| Flux | 디퓨전 | 실험적 단계 |

## 성능 벤치마크 (H100 기준)

| 모델 | GPU 수 | 병렬 처리 | TPS/GPU | 기법 |
|-------|------|-------------|---------|------------|
| Llama 8B | 8 | FSDP | 5,762 | 베이스라인 |
| Llama 8B | 8 | FSDP+compile+FP8 | 8,532 | +48% |
| Llama 70B | 256 | FSDP+TP+AsyncTP | 876 | 2D 병렬 |
| Llama 405B | 512 | FSDP+TP+PP | 128 | 3D 병렬 |

## 고급 주제

**FSDP2 구성**: 상세한 FSDP2 대 FSDP1 비교 및 ZeRO 기능 대응에 대해서는 [references/fsdp.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/fsdp.md)를 참조하세요.

**Float8 학습**: 텐서별(tensorwise) 스케일링 대 행별(rowwise) 스케일링 레시피에 대해서는 [references/float8.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/float8.md)를 참조하세요.

**체크포인트**: HuggingFace 변환 및 비동기 체크포인트 관리에 대해서는 [references/checkpoint.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/checkpoint.md)를 참조하세요.

**사용자 정의 모델 추가**: TrainSpec 프로토콜에 대해서는 [references/custom-models.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/custom-models.md)를 참조하세요.

## 리소스

- GitHub: https://github.com/pytorch/torchtitan
- 논문: https://arxiv.org/abs/2410.06511
- ICLR 2025: https://iclr.cc/virtual/2025/poster/29620
- PyTorch 포럼: https://discuss.pytorch.org/c/distributed/torchtitan/44

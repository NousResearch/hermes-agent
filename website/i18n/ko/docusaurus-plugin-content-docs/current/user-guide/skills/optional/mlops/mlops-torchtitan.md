---
title: "Torchtitan — PyTorch 4D 병렬 처리로 대규모 LLM 사전 학습"
sidebar_label: "Torchtitan"
description: "PyTorch 4D 병렬 처리로 대규모 LLM 사전 학습"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Torchtitan

PyTorch 4D 병렬 처리로 대규모 LLM을 사전 학습합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/torchtitan`으로 설치 |
| 경로 | `optional-skills/mlops/torchtitan` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `torch>=2.6.0`, `torchtitan>=0.2.0`, `torchao>=0.5.0` |
| 플랫폼 | linux, macos |
| 태그 | `모델 아키텍처`, `분산 학습`, `TorchTitan`, `FSDP2`, `Tensor Parallel`, `Pipeline Parallel`, `Context Parallel`, `Float8`, `Llama`, `사전 학습` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# TorchTitan - PyTorch 네이티브 분산 LLM 사전 학습

## 빠른 시작

TorchTitan은 조합 가능한 4D 병렬 처리(FSDP2, TP, PP, CP)를 지원하는 PyTorch의 공식 대규모 LLM 사전 학습 플랫폼으로, H100 GPU에서 기준선보다 65% 이상의 속도 향상을 달성합니다.

**설치**:
```bash
# From PyPI (stable)
pip install torchtitan

# From source (latest features, requires PyTorch nightly)
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
```

**토크나이저 다운로드**:
```bash
# Get HF token from https://huggingface.co/settings/tokens
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

**GPU 8개에서 학습 시작**:
```bash
# Configs are selected by name from the Python config registry
# (torchtitan/models/llama3/config_registry.py), not by TOML path
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

## 일반적인 워크플로

### 워크플로 1: 단일 노드에서 Llama 3.1 8B 사전 학습

다음 체크리스트를 복사하세요:

```
Single Node Pretraining:
- [ ] Step 1: Download tokenizer
- [ ] Step 2: Configure training
- [ ] Step 3: Launch training
- [ ] Step 4: Monitor and checkpoint
```

**1단계: 토크나이저 다운로드**

```bash
python scripts/download_hf_assets.py \
  --repo_id meta-llama/Llama-3.1-8B \
  --assets tokenizer \
  --hf_token=YOUR_HF_TOKEN
```

**2단계: 학습 구성**

현재 torchtitan 레이아웃에서는 실행 구성이 Python **구성 레지스트리**(`torchtitan/models/llama3/config_registry.py`)에 정의되며, `CONFIG=<name>`(또는 `--config <name>`)을 통해 이름으로 선택됩니다. 사용자 지정이 필요하면 레지스트리에 자체 구성을 등록하거나 명령줄에서 개별 필드를 재정의하세요(예: `--optimizer.lr 3e-4 --training.steps 1000`).

다음은 8B 실행에 해당하는 설정입니다(필드로 표시했으며, 레지스트리 항목에서 설정하거나 `--section.key value` 재정의로 전달하세요):

```toml
# fields for a llama3 8B run (register in config_registry.py or pass as --overrides)
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
data_parallel_shard_degree = -1  # Use all GPUs for FSDP

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
# 8 GPUs on single node (config selected by name from the registry)
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh

# Override individual fields on the command line
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh --optimizer.lr 3e-4 --training.steps 1000

# Or explicitly with torchrun (run_train.sh wraps this)
torchrun --nproc_per_node=8 \
  -m torchtitan.train \
  --module llama3 --config llama3_8b
```

**4단계: 모니터링 및 체크포인트**

TensorBoard 로그는 `./outputs/tb/`에 저장됩니다:
```bash
tensorboard --logdir ./outputs/tb
```

### 워크플로 2: SLURM을 사용한 다중 노드 학습

```
Multi-Node Training:
- [ ] Step 1: Configure parallelism for scale
- [ ] Step 2: Set up SLURM script
- [ ] Step 3: Submit job
- [ ] Step 4: Resume from checkpoint
```

**1단계: 규모에 맞게 병렬 처리 구성**

256개 GPU(32개 노드)에서 70B 모델을 실행하는 경우:
```toml
[parallelism]
data_parallel_shard_degree = 32  # FSDP across 32 ranks
tensor_parallel_degree = 8        # TP within node
pipeline_parallel_degree = 1      # No PP for 70B
context_parallel_degree = 1       # Increase for long sequences
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
  --module llama3 --config llama3_70b
```

**3단계: 작업 제출**

```bash
sbatch multinode_trainer.slurm
```

**4단계: 체크포인트에서 재개**

구성된 폴더에 체크포인트가 있으면 학습이 자동으로 재개됩니다.

### 워크플로 3: H100에서 Float8 학습 활성화

Float8은 H100 GPU에서 30~50%의 속도 향상을 제공합니다.

```
Float8 Training:
- [ ] Step 1: Install torchao
- [ ] Step 2: Configure Float8
- [ ] Step 3: Launch with compile
```

**1단계: torchao 설치**

```bash
USE_CPP=0 pip install git+https://github.com/pytorch/ao.git
```

**2단계: Float8 구성**

현재 torchtitan에서 Float8은 `[quantize.linear.float8]` TOML 섹션이 아니라 구성 레지스트리 내부의 `model_registry()` 호출에 전달하는 `quantization` 매개변수를 통해 구성 시 적용됩니다. `Float8LinearConverter.Config`를 추가하세요:

```python
# in torchtitan/models/llama3/config_registry.py (your model_registry(...) call)
from torchtitan.components.quantization import Float8LinearConverter

model_spec = model_registry(
    "8B",
    quantization=[
        Float8LinearConverter.Config(
            recipe_name="rowwise",          # or "rowwise_with_gw_hp"
            filter_fqns=["output"],          # skip layers too small to benefit
            model_compile_enabled=True,      # requires torch.compile for competitive perf
        ),
    ],
)
```

실행 구성에서도 `torch.compile`을 활성화하세요:
```toml
[compile]
enable = true
components = ["model", "loss"]
```

**3단계: compile로 학습 시작**

```bash
# Float8 config is baked into the registered config; just select it and enable compile
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh --compile.enable
```

### 워크플로 4: 405B 모델을 위한 4D 병렬 처리

```
4D Parallelism (FSDP + TP + PP + CP):
- [ ] Step 1: Create seed checkpoint
- [ ] Step 2: Configure 4D parallelism
- [ ] Step 3: Launch on 512 GPUs
```

**1단계: 시드 체크포인트 생성**

PP 단계 전체에서 일관된 초기화를 수행하려면 필요합니다:
```bash
NGPU=1 MODULE=llama3 CONFIG=llama3_405b ./run_train.sh \
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
tensor_parallel_degree = 8       # TP within node
pipeline_parallel_degree = 8     # PP across nodes
context_parallel_degree = 1      # CP for long sequences

[training]
local_batch_size = 32
seq_len = 8192
```

**3단계: GPU 512개에서 시작**

```bash
# 64 nodes x 8 GPUs = 512 GPUs
srun torchrun --nnodes=64 --nproc_per_node=8 \
  -m torchtitan.train \
  --module llama3 --config llama3_405b
```

## 언제 사용하고 언제 대안을 사용할지

**다음과 같은 경우 TorchTitan을 사용하세요:**
- LLM을 처음부터 사전 학습하는 경우(8B~405B 이상)
- 서드파티 의존성 없는 PyTorch 네이티브 솔루션이 필요한 경우
- 조합 가능한 4D 병렬 처리(FSDP2, TP, PP, CP)가 필요한 경우
- Float8을 지원하는 H100에서 학습하는 경우
- torchtune/HuggingFace와 상호 운용 가능한 체크포인트가 필요한 경우

**다음과 같은 경우에는 대안을 사용하세요:**
- **Megatron-LM**: NVIDIA 전용 배포에서 최대 성능이 필요한 경우
- **DeepSpeed**: 더 폭넓은 ZeRO 최적화 생태계와 추론 지원이 필요한 경우
- **Axolotl/TRL**: 사전 학습이 아닌 파인튜닝이 필요한 경우
- **LitGPT**: 교육 목적 또는 소규모 학습인 경우

## 일반적인 문제

**문제: 대형 모델에서 메모리 부족**

활성화 체크포인팅을 활성화하고 배치 크기를 줄이세요:
```toml
[activation_checkpoint]
mode = "full"  # Instead of "selective"

[training]
local_batch_size = 1
```

또는 그래디언트 누적을 사용하세요:
```toml
[training]
local_batch_size = 1
global_batch_size = 32  # Accumulates gradients
```

**문제: TP가 비동기 collective에서 높은 메모리 사용량을 유발함**

환경 변수를 설정하세요:
```bash
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
```

**문제: Float8 학습이 더 빠르지 않음**

Float8은 큰 GEMM에서만 이점이 있습니다. converter의 `filter_fqns`를 통해 작은 레이어를 필터링하세요:
```python
from torchtitan.components.quantization import Float8LinearConverter

Float8LinearConverter.Config(
    # add "auto_filter_small_kn" to auto-skip layers too small to benefit
    filter_fqns=["attention.wk", "attention.wv", "output", "auto_filter_small_kn"],
    model_compile_enabled=True,
)
```

**문제: 병렬 처리 변경 후 체크포인트 로드 실패**

DCP의 resharding 기능을 사용하세요:
```bash
# Convert sharded checkpoint to single file
python -m torch.distributed.checkpoint.format_utils \
  dcp_to_torch checkpoint/step-1000 checkpoint.pt
```

**문제: 파이프라인 병렬 처리 초기화**

먼저 시드 체크포인트를 생성하세요(워크플로 4, 1단계 참조).

## 지원 모델

| 모델 | 크기 | 상태 |
|-------|-------|--------|
| Llama 3.1 | 8B, 70B, 405B | 프로덕션 |
| Llama 4 | 다양함 | 실험적 |
| DeepSeek V3 | 16B, 236B, 671B (MoE) | 실험적 |
| GPT-OSS | 20B, 120B (MoE) | 실험적 |
| Qwen 3 | 다양함 | 실험적 |
| Flux | Diffusion | 실험적 |

## 성능 벤치마크(H100)

| 모델 | GPU | 병렬 처리 | TPS/GPU | 기법 |
|-------|------|-------------|---------|------------|
| Llama 8B | 8 | FSDP | 5,762 | 기준선 |
| Llama 8B | 8 | FSDP+compile+FP8 | 8,532 | +48% |
| Llama 70B | 256 | FSDP+TP+AsyncTP | 876 | 2D 병렬 처리 |
| Llama 405B | 512 | FSDP+TP+PP | 128 | 3D 병렬 처리 |

## 고급 주제

**FSDP2 구성**: 자세한 FSDP2와 FSDP1 비교 및 ZeRO 대응 항목은 [references/fsdp.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/fsdp.md)를 참조하세요.

**Float8 학습**: tensorwise 및 rowwise 스케일링 레시피는 [references/float8.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/float8.md)를 참조하세요.

**체크포인트**: HuggingFace 변환 및 비동기 체크포인팅은 [references/checkpoint.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/checkpoint.md)를 참조하세요.

**사용자 지정 모델 추가**: TrainSpec 프로토콜은 [references/custom-models.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/torchtitan/references/custom-models.md)를 참조하세요.

## 리소스

- GitHub: https://github.com/pytorch/torchtitan
- 논문: https://arxiv.org/abs/2410.06511
- ICLR 2025: https://iclr.cc/virtual/2025/poster/29620
- PyTorch 포럼: https://discuss.pytorch.org/c/distributed/torchtitan/44

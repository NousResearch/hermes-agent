---
title: "Accelerate — 최소한의 변경으로 여러 GPU에서 PyTorch 학습 실행"
sidebar_label: "Accelerate"
description: "최소한의 변경으로 여러 GPU에서 PyTorch 학습 실행"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Accelerate

최소한의 변경으로 여러 GPU에서 PyTorch 학습을 실행합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/accelerate`로 설치 |
| 경로 | `optional-skills/mlops/accelerate` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `accelerate`, `torch`, `transformers` |
| 플랫폼 | linux, macos, windows |
| 태그 | `Distributed Training`, `HuggingFace`, `Accelerate`, `DeepSpeed`, `FSDP`, `Mixed Precision`, `PyTorch`, `DDP`, `Unified API`, `Simple` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# HuggingFace Accelerate - 통합 분산 학습

## 빠른 시작

Accelerate를 사용하면 4줄의 코드만으로 분산 학습을 간소화할 수 있습니다.

**설치**:
```bash
pip install accelerate
```

**PyTorch 스크립트 변환** (4줄):
```python
import torch
+ from accelerate import Accelerator

+ accelerator = Accelerator()

  model = torch.nn.Transformer()
  optimizer = torch.optim.Adam(model.parameters())
  dataloader = torch.utils.data.DataLoader(dataset)

+ model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

  for batch in dataloader:
      optimizer.zero_grad()
      loss = model(batch)
-     loss.backward()
+     accelerator.backward(loss)
      optimizer.step()
```

**실행** (단일 명령):
```bash
accelerate launch train.py
```

## 일반적인 워크플로

### 워크플로 1: 단일 GPU에서 다중 GPU로

**원본 스크립트**:
```python
# train.py
import torch

model = torch.nn.Linear(10, 2).to('cuda')
optimizer = torch.optim.Adam(model.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

for epoch in range(10):
    for batch in dataloader:
        batch = batch.to('cuda')
        optimizer.zero_grad()
        loss = model(batch).mean()
        loss.backward()
        optimizer.step()
```

**Accelerate 사용** (4줄 추가):
```python
# train.py
import torch
from accelerate import Accelerator  # +1

accelerator = Accelerator()  # +2

model = torch.nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)  # +3

for epoch in range(10):
    for batch in dataloader:
        # No .to('cuda') needed - automatic!
        optimizer.zero_grad()
        loss = model(batch).mean()
        accelerator.backward(loss)  # +4
        optimizer.step()
```

**구성** (대화형):
```bash
accelerate config
```

**질문**:
- 어떤 머신인가요? (단일/다중 GPU/TPU/CPU)
- 머신은 몇 대인가요? (1)
- 혼합 정밀도인가요? (no/fp16/bf16/fp8)
- DeepSpeed를 사용하나요? (no/yes)

**실행** (모든 설정에서 작동):
```bash
# Single GPU
accelerate launch train.py

# Multi-GPU (8 GPUs)
accelerate launch --multi_gpu --num_processes 8 train.py

# Multi-node
accelerate launch --multi_gpu --num_processes 16 \
  --num_machines 2 --machine_rank 0 \
  --main_process_ip $MASTER_ADDR \
  train.py
```

### 워크플로 2: 혼합 정밀도 학습

**FP16/BF16 활성화**:
```python
from accelerate import Accelerator

# FP16 (with gradient scaling)
accelerator = Accelerator(mixed_precision='fp16')

# BF16 (no scaling, more stable)
accelerator = Accelerator(mixed_precision='bf16')

# FP8 (H100+)
accelerator = Accelerator(mixed_precision='fp8')

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# Everything else is automatic!
for batch in dataloader:
    with accelerator.autocast():  # Optional, done automatically
        loss = model(batch)
    accelerator.backward(loss)
```

### 워크플로 3: DeepSpeed ZeRO 통합

**DeepSpeed ZeRO-2 활성화** (`DeepSpeedPlugin`을 전달하며, 일반 dict는 전달하지 않음):
```python
from accelerate import Accelerator, DeepSpeedPlugin

deepspeed_plugin = DeepSpeedPlugin(
    zero_stage=2,                     # ZeRO-2
    offload_optimizer_device="none",  # or "cpu" to offload
    gradient_accumulation_steps=4,
)

accelerator = Accelerator(
    mixed_precision='bf16',
    deepspeed_plugin=deepspeed_plugin,  # DeepSpeedPlugin instance (or dict[str, DeepSpeedPlugin])
)

# Same code as before!
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**또는 플러그인을 통해 전체 DeepSpeed JSON 구성 지정**:
```python
from accelerate import Accelerator, DeepSpeedPlugin

# hf_ds_config accepts a path to a DeepSpeed config JSON (or a dict)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config="ds_config.json")
accelerator = Accelerator(mixed_precision='bf16', deepspeed_plugin=deepspeed_plugin)
```

**ds_config.json** (플러그인을 통해 전달하는 원시 DeepSpeed 구성 — `--config_file`을 통해 전달하지 않음):
```json
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"},
        "allgather_bucket_size": 5e8,
        "reduce_bucket_size": 5e8
    }
}
```

**또는 대화형 구성 사용**:
```bash
accelerate config
# Select: DeepSpeed → ZeRO-2
# This writes an accelerate YAML config (default: ~/.cache/huggingface/accelerate/default_config.yaml)
```

**실행** (`--config_file`은 원시 DeepSpeed JSON이 아니라 accelerate YAML을 요구):
```bash
# Uses the default accelerate config written by `accelerate config`
accelerate launch train.py

# Or point at a specific accelerate YAML
accelerate launch --config_file accelerate_deepspeed.yaml train.py
```

### 워크플로 4: FSDP (Fully Sharded Data Parallel)

**FSDP 활성화**:
```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",  # ZeRO-3 equivalent
    auto_wrap_policy="transformer_based_wrap",  # valid: transformer_based_wrap | size_based_wrap | no_wrap
    cpu_offload=False
)

accelerator = Accelerator(
    mixed_precision='bf16',
    fsdp_plugin=fsdp_plugin
)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**또는 구성 사용**:
```bash
accelerate config
# Select: FSDP → Full Shard → No CPU Offload
```

### 워크플로 5: 그래디언트 누적

**그래디언트 누적**:
```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):  # Handles accumulation
        optimizer.zero_grad()
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
```

**유효 배치 크기**: `batch_size * num_gpus * gradient_accumulation_steps`

## 대안과 비교해 언제 사용할지

**다음과 같은 경우 Accelerate 사용**:
- 가장 간단한 분산 학습을 원할 때
- 모든 하드웨어에서 사용할 단일 스크립트가 필요할 때
- HuggingFace 생태계를 사용할 때
- 유연성(DDP/DeepSpeed/FSDP/Megatron)을 원할 때
- 빠른 프로토타이핑이 필요할 때

**주요 장점**:
- **4줄**: 최소한의 코드 변경
- **통합 API**: DDP, DeepSpeed, FSDP, Megatron에서 동일한 코드
- **자동화**: 디바이스 배치, 혼합 정밀도, 샤딩
- **대화형 구성**: 수동 런처 설정 불필요
- **단일 실행 명령**: 어디서나 작동

**다음과 같은 경우 대안 사용**:
- **PyTorch Lightning**: 콜백과 고수준 추상화가 필요할 때
- **Ray Train**: 다중 노드 오케스트레이션과 하이퍼파라미터 튜닝이 필요할 때
- **DeepSpeed**: 직접적인 API 제어와 고급 기능이 필요할 때
- **Raw DDP**: 최대한의 제어와 최소한의 추상화가 필요할 때

## 일반적인 문제

**문제: 잘못된 디바이스 배치**

디바이스로 수동 이동하지 마세요:
```python
# WRONG
batch = batch.to('cuda')

# CORRECT
# Accelerate handles it automatically after prepare()
```

**문제: 그래디언트 누적이 작동하지 않음**

컨텍스트 매니저를 사용하세요:
```python
# CORRECT
with accelerator.accumulate(model):
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
```

**문제: 분산 환경에서 체크포인트 저장**

Accelerator 메서드를 사용하세요:
```python
# Save only on main process
if accelerator.is_main_process:
    accelerator.save_state('checkpoint/')

# Load on all processes
accelerator.load_state('checkpoint/')
```

**문제: FSDP에서 결과가 다름**

동일한 랜덤 시드를 사용하세요:
```python
from accelerate.utils import set_seed
set_seed(42)
```

## 고급 주제

**Megatron 통합**: 텐서 병렬, 파이프라인 병렬 및 시퀀스 병렬 설정은 [references/megatron-integration.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/megatron-integration.md)를 참조하세요.

**사용자 지정 플러그인**: 사용자 지정 분산 플러그인 생성과 고급 구성은 [references/custom-plugins.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/custom-plugins.md)를 참조하세요.

**성능 튜닝**: 프로파일링, 메모리 최적화 및 모범 사례는 [references/performance.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/performance.md)를 참조하세요.

## 하드웨어 요구 사항

- **CPU**: 작동함(느림)
- **단일 GPU**: 작동함
- **다중 GPU**: DDP(기본값), DeepSpeed 또는 FSDP
- **다중 노드**: DDP, DeepSpeed, FSDP, Megatron
- **TPU**: 지원됨
- **Apple MPS**: 지원됨

**런처 요구 사항**:
- **DDP**: `torch.distributed.run`(내장)
- **DeepSpeed**: `deepspeed`(`pip install deepspeed`)
- **FSDP**: PyTorch 1.12+(내장)
- **Megatron**: 사용자 지정 설정

## 리소스

- 문서: https://huggingface.co/docs/accelerate
- GitHub: https://github.com/huggingface/accelerate
- 버전: 1.11.0+
- 튜토리얼: "스크립트 Accelerate 적용"
- 예제: https://github.com/huggingface/accelerate/tree/main/examples
- 사용처: HuggingFace Transformers, TRL, PEFT, 모든 HF 라이브러리

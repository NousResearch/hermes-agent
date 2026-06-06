---
title: "Huggingface Accelerate — 가장 단순한 분산 학습 API"
sidebar_label: "Huggingface Accelerate"
description: "가장 단순한 분산 학습 API"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Huggingface Accelerate

가장 단순한 분산 학습 API입니다. 단 4줄의 코드로 모든 PyTorch 스크립트에 분산 지원을 추가할 수 있습니다. DeepSpeed/FSDP/Megatron/DDP를 위한 통합 API입니다. 자동 디바이스 배치, 혼합 정밀도(FP16/BF16/FP8)를 지원합니다. 대화형 설정 및 단일 실행 명령을 제공하며, HuggingFace 생태계의 표준입니다.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/accelerate`로 설치 |
| Path | `optional-skills/mlops/accelerate` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `accelerate`, `torch`, `transformers` |
| Platforms | linux, macos, windows |
| Tags | `Distributed Training`, `HuggingFace`, `Accelerate`, `DeepSpeed`, `FSDP`, `Mixed Precision`, `PyTorch`, `DDP`, `Unified API`, `Simple` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# HuggingFace Accelerate - 통합 분산 학습

## 빠른 시작

Accelerate는 단 4줄의 코드만으로 분산 학습을 간단하게 만들어줍니다.

**설치**:
```bash
pip install accelerate
```

**PyTorch 스크립트 변환** (4줄 추가):
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

**실행** (단일 명령어):
```bash
accelerate launch train.py
```

## 일반적인 워크플로우

### 워크플로우 1: 단일 GPU에서 다중 GPU로

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

**Accelerate 적용** (4줄 추가):
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
        # .to('cuda')가 필요 없습니다 - 자동으로 처리됨!
        optimizer.zero_grad()
        loss = model(batch).mean()
        accelerator.backward(loss)  # +4
        optimizer.step()
```

**설정** (대화형):
```bash
accelerate config
```

**질문 내용**:
- 어떤 기계입니까? (단일/다중 GPU/TPU/CPU)
- 기계는 몇 대입니까? (1)
- 혼합 정밀도를 사용합니까? (no/fp16/bf16/fp8)
- DeepSpeed를 사용합니까? (no/yes)

**실행** (어떤 설정에서도 작동):
```bash
# 단일 GPU
accelerate launch train.py

# 다중 GPU (8 GPU)
accelerate launch --multi_gpu --num_processes 8 train.py

# 다중 노드
accelerate launch --multi_gpu --num_processes 16 \
  --num_machines 2 --machine_rank 0 \
  --main_process_ip $MASTER_ADDR \
  train.py
```

### 워크플로우 2: 혼합 정밀도 학습

**FP16/BF16 활성화**:
```python
from accelerate import Accelerator

# FP16 (그래디언트 스케일링 포함)
accelerator = Accelerator(mixed_precision='fp16')

# BF16 (스케일링 없음, 더 안정적임)
accelerator = Accelerator(mixed_precision='bf16')

# FP8 (H100 이상)
accelerator = Accelerator(mixed_precision='fp8')

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 나머지는 자동으로 처리됩니다!
for batch in dataloader:
    with accelerator.autocast():  # (선택 사항) 자동으로 수행됨
        loss = model(batch)
    accelerator.backward(loss)
```

### 워크플로우 3: DeepSpeed ZeRO 통합

**DeepSpeed ZeRO-2 활성화**:
```python
from accelerate import Accelerator

accelerator = Accelerator(
    mixed_precision='bf16',
    deepspeed_plugin={
        "zero_stage": 2,  # ZeRO-2
        "offload_optimizer": False,
        "gradient_accumulation_steps": 4
    }
)

# 코드는 이전과 같습니다!
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**또는 설정을 통해**:
```bash
accelerate config
# 선택: DeepSpeed → ZeRO-2
```

**deepspeed_config.json**:
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

**실행**:
```bash
accelerate launch --config_file deepspeed_config.json train.py
```

### 워크플로우 4: FSDP (Fully Sharded Data Parallel)

**FSDP 활성화**:
```python
from accelerate import Accelerator, FullyShardedDataParallelPlugin

fsdp_plugin = FullyShardedDataParallelPlugin(
    sharding_strategy="FULL_SHARD",  # ZeRO-3에 해당
    auto_wrap_policy="TRANSFORMER_AUTO_WRAP",
    cpu_offload=False
)

accelerator = Accelerator(
    mixed_precision='bf16',
    fsdp_plugin=fsdp_plugin
)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

**또는 설정을 통해**:
```bash
accelerate config
# 선택: FSDP → Full Shard → No CPU Offload
```

### 워크플로우 5: 그래디언트 누적(Gradient accumulation)

**그래디언트 누적 처리**:
```python
from accelerate import Accelerator

accelerator = Accelerator(gradient_accumulation_steps=4)

model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

for batch in dataloader:
    with accelerator.accumulate(model):  # 누적을 처리합니다
        optimizer.zero_grad()
        loss = model(batch)
        accelerator.backward(loss)
        optimizer.step()
```

**유효 배치 크기**: `batch_size * num_gpus * gradient_accumulation_steps`

## 사용 시기 및 대안

**Accelerate를 사용하는 경우**:
- 가장 간단한 분산 학습을 원할 때
- 모든 하드웨어에서 작동하는 단일 스크립트가 필요할 때
- HuggingFace 생태계를 사용할 때
- 유연성(DDP/DeepSpeed/FSDP/Megatron)이 필요할 때
- 빠른 프로토타이핑이 필요할 때

**주요 장점**:
- **4줄**: 최소한의 코드 변경
- **통합 API**: DDP, DeepSpeed, FSDP, Megatron에 대해 동일한 코드
- **자동화**: 디바이스 배치, 혼합 정밀도, 샤딩 자동화
- **대화형 설정**: 런처를 수동으로 설정할 필요 없음
- **단일 실행**: 어디서나 동작함

**대신 사용할 수 있는 대안**:
- **PyTorch Lightning**: 콜백과 고수준의 추상화가 필요할 때
- **Ray Train**: 다중 노드 오케스트레이션 및 하이퍼파라미터 튜닝
- **DeepSpeed**: API 직접 제어와 고급 기능
- **Raw DDP**: 최소한의 추상화로 최대한의 제어 권한

## 일반적인 문제

**문제: 잘못된 디바이스 배치(Device placement)**

디바이스로 직접 이동하지 마세요:
```python
# 잘못된 예
batch = batch.to('cuda')

# 올바른 예
# prepare() 이후에 Accelerate가 자동으로 처리합니다.
```

**문제: 그래디언트 누적이 작동하지 않음**

컨텍스트 매니저를 사용하세요:
```python
# 올바른 예
with accelerator.accumulate(model):
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
```

**문제: 분산 환경에서의 체크포인트 처리**

accelerator 메서드를 사용하세요:
```python
# 메인 프로세스에서만 저장
if accelerator.is_main_process:
    accelerator.save_state('checkpoint/')

# 모든 프로세스에서 불러오기
accelerator.load_state('checkpoint/')
```

**문제: FSDP 사용 시 결과가 다름**

랜덤 시드를 동일하게 보장하세요:
```python
from accelerate.utils import set_seed
set_seed(42)
```

## 고급 주제

**Megatron 통합**: 텐서 병렬 처리, 파이프라인 병렬 처리 및 시퀀스 병렬 처리 설정에 대해서는 [references/megatron-integration.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/megatron-integration.md)를 참조하세요.

**커스텀 플러그인**: 커스텀 분산 플러그인 생성 및 고급 설정은 [references/custom-plugins.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/custom-plugins.md)를 참조하세요.

**성능 튜닝**: 프로파일링, 메모리 최적화 및 모범 사례는 [references/performance.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/accelerate/references/performance.md)를 참조하세요.

## 하드웨어 요구 사항

- **CPU**: 지원 (느림)
- **단일 GPU**: 지원
- **다중 GPU**: DDP (기본값), DeepSpeed, 또는 FSDP
- **다중 노드**: DDP, DeepSpeed, FSDP, Megatron
- **TPU**: 지원
- **Apple MPS**: 지원

**런처 요구 사항**:
- **DDP**: `torch.distributed.run` (기본 내장)
- **DeepSpeed**: `deepspeed` (pip install deepspeed)
- **FSDP**: PyTorch 1.12 이상 (기본 내장)
- **Megatron**: 별도 설정 필요

## 리소스

- 공식 문서: https://huggingface.co/docs/accelerate
- GitHub: https://github.com/huggingface/accelerate
- 버전: 1.11.0 이상
- 튜토리얼: "Accelerate your scripts"
- 예제: https://github.com/huggingface/accelerate/tree/main/examples
- 사용자: HuggingFace Transformers, TRL, PEFT 및 모든 HF 라이브러리

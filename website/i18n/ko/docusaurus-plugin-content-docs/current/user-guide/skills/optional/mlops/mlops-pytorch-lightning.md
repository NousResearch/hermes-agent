---
title: "Pytorch Lightning — 기본 분산 지원을 갖춘 깔끔한 학습 루프"
sidebar_label: "Pytorch Lightning"
description: "기본 분산 지원을 갖춘 깔끔한 학습 루프"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동으로 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Pytorch Lightning

기본 분산 지원을 갖춘 깔끔한 학습 루프입니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/pytorch-lightning`로 설치 |
| 경로 | `optional-skills/mlops/pytorch-lightning` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `lightning`, `torch`, `transformers` |
| 플랫폼 | linux, macos, windows |
| 태그 | `PyTorch Lightning`, `훈련 프레임워크`, `분산 학습`, `DDP`, `FSDP`, `DeepSpeed`, `고수준 API`, `콜백`, `모범 사례`, `확장 가능` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 활성화될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성 상태일 때 에이전트가 보는 지침입니다.
:::

# PyTorch Lightning - 고수준 학습 프레임워크

## 빠른 시작

PyTorch Lightning은 유연성을 유지하면서 반복적인 코드를 없애도록 PyTorch 코드를 구성합니다.

**설치**:
```bash
pip install lightning
```

**PyTorch를 Lightning으로 변환** (3단계):

```python
import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Step 1: Define LightningModule (organize your PyTorch code)
class LitModel(L.LightningModule):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)  # Auto-logged to TensorBoard
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Step 2: Create data
train_loader = DataLoader(train_dataset, batch_size=32)

# Step 3: Train with Trainer (handles everything else!)
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=2)
model = LitModel()
trainer.fit(model, train_loader)
```

**이게 전부입니다!** Trainer가 다음을 처리합니다.
- GPU/TPU/CPU 전환
- 분산 학습 (DDP, FSDP, DeepSpeed)
- 혼합 정밀도 (FP16, BF16)
- 그래디언트 누적
- 체크포인트 저장
- 로깅
- 진행률 표시줄

## 일반적인 워크플로

### 워크플로 1: PyTorch에서 Lightning으로

**기존 PyTorch 코드**:
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
model.to('cuda')

for epoch in range(max_epochs):
    for batch in train_loader:
        batch = batch.to('cuda')
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
```

**Lightning 버전**:
```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)  # No .to('cuda') needed!
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# Train
trainer = L.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(LitModel(), train_loader)
```

**이점**: 40줄 이상 → 15줄, 디바이스 관리 불필요, 자동 분산

### 워크플로 2: 검증 및 테스트

```python
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        val_loss = nn.functional.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', val_loss)
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        test_loss = nn.functional.cross_entropy(y_hat, y)
        self.log('test_loss', test_loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# Train with validation
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

# Test
trainer.test(model, test_loader)
```

**자동 기능**:
- 기본적으로 매 에포크마다 검증 실행
- TensorBoard에 메트릭 기록
- val_loss를 기준으로 최적 모델 체크포인트 저장

### 워크플로 3: 분산 학습 (DDP)

```python
# Same code as single GPU!
model = LitModel()

# 8 GPUs with DDP (automatic!)
trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    strategy='ddp'  # Or 'fsdp', 'deepspeed'
)

trainer.fit(model, train_loader)
```

**실행**:
```bash
# Single command, Lightning handles the rest
python train.py
```

**변경할 필요 없음**:
- 자동 데이터 분배
- 그래디언트 동기화
- 멀티 노드 지원 (`num_nodes=2`만 설정)

### 워크플로 4: 모니터링을 위한 콜백

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# Create callbacks
checkpoint = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    filename='model-{epoch:02d}-{val_loss:.2f}'
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min'
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

# Add to Trainer
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint, early_stop, lr_monitor]
)

trainer.fit(model, train_loader, val_loader)
```

**결과**:
- 최적 모델 3개 자동 저장
- 5 에포크 동안 개선이 없으면 조기 중단
- TensorBoard에 학습률 기록

### 워크플로 5: 학습률 스케줄링

```python
class LitModel(L.LightningModule):
    # ... (training_step, etc.)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # Update per epoch
                'frequency': 1
            }
        }

# Learning rate auto-logged!
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader)
```

## 언제 사용하고 언제 대안을 사용할까요

**다음과 같은 경우 PyTorch Lightning을 사용합니다**:
- 깔끔하고 체계적인 코드를 원할 때
- 프로덕션 수준의 학습 루프가 필요할 때
- 단일 GPU, 멀티 GPU, TPU 사이를 전환할 때
- 기본 제공 콜백과 로깅을 원할 때
- 팀 협업 (표준화된 구조)

**주요 장점**:
- **체계적**: 연구 코드와 엔지니어링을 분리
- **자동화**: 한 줄로 DDP, FSDP, DeepSpeed 사용
- **콜백**: 모듈식 학습 확장
- **재현 가능**: 반복 코드 감소 = 버그 감소
- **검증됨**: 월 100만 회 이상 다운로드, 실전에서 검증됨

**다음과 같은 경우에는 대안을 사용합니다**:
- **Accelerate**: 기존 코드의 변경을 최소화하면서 더 많은 유연성
- **Ray Train**: 멀티 노드 오케스트레이션, 하이퍼파라미터 튜닝
- **Raw PyTorch**: 최대한의 제어, 학습 목적
- **Keras**: TensorFlow 생태계

## 일반적인 문제

**문제: 손실이 감소하지 않음**

데이터와 모델 설정을 확인합니다:
```python
# Add to training_step
def training_step(self, batch, batch_idx):
    if batch_idx == 0:
        print(f"Batch shape: {batch[0].shape}")
        print(f"Labels: {batch[1]}")
    loss = ...
    return loss
```

**문제: 메모리 부족**

배치 크기를 줄이거나 그래디언트 누적을 사용합니다:
```python
trainer = L.Trainer(
    accumulate_grad_batches=4,  # Effective batch = batch_size × 4
    precision='bf16'  # Or 'fp16', reduces memory 50%
)
```

**문제: 검증이 실행되지 않음**

val_loader를 전달했는지 확인합니다:
```python
# WRONG
trainer.fit(model, train_loader)

# CORRECT
trainer.fit(model, train_loader, val_loader)
```

**문제: DDP가 예기치 않게 여러 프로세스를 생성함**

Lightning은 GPU를 자동으로 감지합니다. devices를 명시적으로 설정합니다:
```python
# Test on CPU first
trainer = L.Trainer(accelerator='cpu', devices=1)

# Then GPU
trainer = L.Trainer(accelerator='gpu', devices=1)
```

## 고급 주제

**콜백**: EarlyStopping, ModelCheckpoint, 사용자 지정 콜백 및 콜백 훅은 [references/callbacks.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/callbacks.md)를 참고하세요.

**분산 전략**: DDP, FSDP, DeepSpeed ZeRO 통합 및 멀티 노드 설정은 [references/distributed.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/distributed.md)를 참고하세요.

**하이퍼파라미터 튜닝**: Optuna, Ray Tune 및 WandB sweep 통합은 [references/hyperparameter-tuning.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/hyperparameter-tuning.md)를 참고하세요.

## 하드웨어 요구 사항

- **CPU**: 작동함 (디버깅에 적합)
- **단일 GPU**: 작동함
- **멀티 GPU**: DDP (기본값), FSDP 또는 DeepSpeed
- **멀티 노드**: DDP, FSDP, DeepSpeed
- **TPU**: 지원됨 (8코어)
- **Apple MPS**: 지원됨

**정밀도 옵션**:
- FP32 (기본값)
- FP16 (V100, 구형 GPU)
- BF16 (A100/H100, 권장)
- FP8 (H100)

## 리소스

- 문서: https://lightning.ai/docs/pytorch/stable/
- GitHub: https://github.com/Lightning-AI/pytorch-lightning ⭐ 29,000+
- 버전: 2.5.5+
- 예제: https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples
- Discord: https://discord.gg/lightning-ai
- 사용처: Kaggle 우승자, 연구소, 프로덕션 팀

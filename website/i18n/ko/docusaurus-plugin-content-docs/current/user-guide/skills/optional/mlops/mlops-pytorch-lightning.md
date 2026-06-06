---
title: "Pytorch Lightning"
sidebar_label: "Pytorch Lightning"
description: "고수준 PyTorch 프레임워크로 Trainer 클래스, 자동 분산 학습(DDP/FSDP/DeepSpeed), 콜백 시스템을 갖추고 있으며 보일러플레이트 코드를 최소화합니다."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Pytorch Lightning

고수준 PyTorch 프레임워크로 Trainer 클래스, 자동 분산 학습(DDP/FSDP/DeepSpeed), 콜백 시스템을 지원하며 보일러플레이트(상용구) 코드를 최소화합니다. 노트북에서 슈퍼컴퓨터까지 동일한 코드로 확장할 수 있습니다. 모범 사례가 내장된 깔끔한 학습 루프를 원할 때 사용하세요.

## Skill metadata

| | |
|---|---|
| Source | Optional — `hermes skills install official/mlops/pytorch-lightning`로 설치 |
| Path | `optional-skills/mlops/pytorch-lightning` |
| Version | `1.0.0` |
| Author | Orchestra Research |
| License | MIT |
| Dependencies | `lightning`, `torch`, `transformers` |
| Platforms | linux, macos, windows |
| Tags | `PyTorch Lightning`, `Training Framework`, `Distributed Training`, `DDP`, `FSDP`, `DeepSpeed`, `High-Level API`, `Callbacks`, `Best Practices`, `Scalable` |

## Reference: full SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# PyTorch Lightning - 고수준 학습 프레임워크

## 빠른 시작

PyTorch Lightning은 유연성을 유지하면서 보일러플레이트를 제거하기 위해 PyTorch 코드를 구조화합니다.

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

# 1단계: LightningModule 정의 (PyTorch 코드 구조화)
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
        self.log('train_loss', loss)  # TensorBoard에 자동 기록
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# 2단계: 데이터 준비
train_loader = DataLoader(train_dataset, batch_size=32)

# 3단계: Trainer로 학습 (나머지 모든 것을 처리!)
trainer = L.Trainer(max_epochs=10, accelerator='gpu', devices=2)
model = LitModel()
trainer.fit(model, train_loader)
```

**이게 전부입니다!** Trainer가 다음을 처리합니다:
- GPU/TPU/CPU 전환
- 분산 학습 (DDP, FSDP, DeepSpeed)
- 혼합 정밀도 (FP16, BF16)
- 그래디언트 누적 (Gradient accumulation)
- 체크포인트 저장
- 로깅
- 진행률 표시줄

## 일반적인 워크플로우

### 워크플로우 1: PyTorch에서 Lightning으로 전환

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
        loss = self.model(batch)  # .to('cuda')가 필요 없습니다!
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# 학습
trainer = L.Trainer(max_epochs=10, accelerator='gpu')
trainer.fit(LitModel(), train_loader)
```

**이점**: 40+줄 → 15줄, 디바이스 관리 불필요, 자동 분산 처리

### 워크플로우 2: 검증(Validation) 및 테스트

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

# 검증을 포함하여 학습
trainer = L.Trainer(max_epochs=10)
trainer.fit(model, train_loader, val_loader)

# 테스트
trainer.test(model, test_loader)
```

**자동화 기능**:
- 기본적으로 매 에포크마다 검증 실행
- TensorBoard에 메트릭 기록
- val_loss를 기반으로 최상의 모델 체크포인트 저장

### 워크플로우 3: 분산 학습 (DDP)

```python
# 단일 GPU와 동일한 코드!
model = LitModel()

# 8개 GPU와 DDP (자동!)
trainer = L.Trainer(
    accelerator='gpu',
    devices=8,
    strategy='ddp'  # 또는 'fsdp', 'deepspeed'
)

trainer.fit(model, train_loader)
```

**실행**:
```bash
# 단일 명령으로 Lightning이 나머지를 처리합니다
python train.py
```

**변경할 필요가 없음**:
- 자동 데이터 분산
- 그래디언트 동기화
- 다중 노드 지원 (`num_nodes=2`만 설정하면 됨)

### 워크플로우 4: 모니터링을 위한 콜백

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

# 콜백 생성
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

# Trainer에 추가
trainer = L.Trainer(
    max_epochs=100,
    callbacks=[checkpoint, early_stop, lr_monitor]
)

trainer.fit(model, train_loader, val_loader)
```

**결과**:
- 상위 3개의 최고 모델 자동 저장
- 5에포크 동안 개선이 없으면 조기 종료
- TensorBoard에 학습률 기록

### 워크플로우 5: 학습률 스케줄링

```python
class LitModel(L.LightningModule):
    # ... (training_step 등)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # 코사인 어닐링 (Cosine annealing)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 에포크 단위로 업데이트
                'frequency': 1
            }
        }

# 학습률이 자동으로 기록됩니다!
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_loader)
```

## 사용 시기 및 대안

**PyTorch Lightning을 사용하는 경우**:
- 깔끔하고 체계적인 코드를 원할 때
- 프로덕션 수준의 학습 루프가 필요할 때
- 단일 GPU, 다중 GPU, TPU 간의 전환
- 내장된 콜백 및 로깅 기능을 원할 때
- 팀 협업 (표준화된 구조)

**주요 장점**:
- **구조화**: 연구 코드와 엔지니어링 코드의 분리
- **자동화**: 한 줄의 코드로 DDP, FSDP, DeepSpeed 설정
- **콜백**: 모듈식 학습 확장 기능
- **재현성**: 보일러플레이트 감소 = 버그 감소
- **신뢰성**: 월 100만 회 이상의 다운로드, 입증된 성능

**대신 사용할 수 있는 대안**:
- **Accelerate**: 기존 코드에 최소한의 변경, 더 많은 유연성
- **Ray Train**: 다중 노드 오케스트레이션, 하이퍼파라미터 튜닝
- **Raw PyTorch**: 최대한의 제어, 학습 목적
- **Keras**: TensorFlow 생태계

## 일반적인 문제

**문제: 손실(Loss)이 감소하지 않음**

데이터와 모델 설정 확인:
```python
# training_step에 추가
def training_step(self, batch, batch_idx):
    if batch_idx == 0:
        print(f"Batch shape: {batch[0].shape}")
        print(f"Labels: {batch[1]}")
    loss = ...
    return loss
```

**문제: 메모리 부족 (Out of memory)**

배치 크기를 줄이거나 그래디언트 누적을 사용:
```python
trainer = L.Trainer(
    accumulate_grad_batches=4,  # 유효 배치 = batch_size × 4
    precision='bf16'  # 또는 'fp16', 메모리 50% 감소
)
```

**문제: 검증(Validation)이 실행되지 않음**

val_loader를 전달했는지 확인:
```python
# 잘못된 예
trainer.fit(model, train_loader)

# 올바른 예
trainer.fit(model, train_loader, val_loader)
```

**문제: DDP가 예기치 않게 여러 프로세스를 생성함**

Lightning은 GPU를 자동으로 감지합니다. 명시적으로 디바이스를 설정하세요:
```python
# 먼저 CPU에서 테스트
trainer = L.Trainer(accelerator='cpu', devices=1)

# 그 다음 GPU
trainer = L.Trainer(accelerator='gpu', devices=1)
```

## 고급 주제

**콜백**: EarlyStopping, ModelCheckpoint, 커스텀 콜백 및 콜백 후크에 대해서는 [references/callbacks.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/callbacks.md)를 참조하세요.

**분산 전략**: DDP, FSDP, DeepSpeed ZeRO 통합, 다중 노드 설정에 대해서는 [references/distributed.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/distributed.md)를 참조하세요.

**하이퍼파라미터 튜닝**: Optuna, Ray Tune, WandB sweeps와의 통합에 대해서는 [references/hyperparameter-tuning.md](https://github.com/NousResearch/hermes-agent/blob/main/optional-skills/mlops/pytorch-lightning/references/hyperparameter-tuning.md)를 참조하세요.

## 하드웨어 요구 사항

- **CPU**: 지원 (디버깅에 좋음)
- **단일 GPU**: 지원
- **다중 GPU**: DDP (기본값), FSDP 또는 DeepSpeed
- **다중 노드**: DDP, FSDP, DeepSpeed
- **TPU**: 지원 (8코어)
- **Apple MPS**: 지원

**정밀도 옵션**:
- FP32 (기본값)
- FP16 (V100, 구형 GPU)
- BF16 (A100/H100, 권장)
- FP8 (H100)

## 리소스

- 공식 문서: https://lightning.ai/docs/pytorch/stable/
- GitHub: https://github.com/Lightning-AI/pytorch-lightning ⭐ 29,000+
- 버전: 2.5.5+
- 예제: https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples
- Discord: https://discord.gg/lightning-ai
- 사용자: Kaggle 우승자, 연구실, 프로덕션 팀

---
title: "Weights And Biases — W&B: ML 실험, 스윕, 모델 레지스트리, 대시보드 기록"
sidebar_label: "Weights And Biases"
description: "W&B: ML 실험, 스윕, 모델 레지스트리, 대시보드 기록"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py가 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하십시오. */}

# Weights And Biases

W&B: ML 실험, 스윕, 모델 레지스트리, 대시보드를 기록합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함(기본 설치) |
| 경로 | `skills/mlops/evaluation/weights-and-biases` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `wandb` |
| 플랫폼 | linux, macos, windows |
| 태그 | `MLOps`, `Weights And Biases`, `WandB`, `Experiment Tracking`, `Hyperparameter Tuning`, `Model Registry`, `Collaboration`, `Real-Time Visualization`, `PyTorch`, `TensorFlow`, `HuggingFace` |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Weights & Biases: ML 실험 추적 및 MLOps

## 이 스킬을 사용하는 경우

다음과 같은 경우 Weights & Biases(W&B)를 사용합니다:
- **ML 실험 추적** — 메트릭을 자동으로 기록
- **학습 시각화** — 실시간 대시보드에서 확인
- **실행 비교** — 하이퍼파라미터와 구성 전반에서 비교
- **하이퍼파라미터 최적화** — 자동화된 스윕으로 최적화
- **모델 레지스트리 관리** — 버전 관리 및 계보 추적
- **ML 프로젝트 협업** — 팀 워크스페이스 사용
- **아티팩트 추적** — 계보를 포함한 데이터셋, 모델, 코드 추적

**사용자**: 200,000명 이상의 ML 실무자 | **GitHub Stars**: 10.5k+ | **통합**: 100+

## 설치

```bash
# Install W&B
pip install wandb

# Login (creates API key)
wandb login

# Or set API key programmatically
export WANDB_API_KEY=your_api_key_here
```

## 빠른 시작

### 기본 실험 추적

```python
import wandb

# Initialize a run
run = wandb.init(
    project="my-project",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "ResNet50"
    }
)

# Training loop
for epoch in range(run.config.epochs):
    # Your training code
    train_loss = train_epoch()
    val_loss = validate()

    # Log metrics
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "train/accuracy": train_acc,
        "val/accuracy": val_acc
    })

# Finish the run
wandb.finish()
```

### PyTorch 사용

```python
import torch
import wandb

# Initialize
wandb.init(project="pytorch-demo", config={
    "lr": 0.001,
    "epochs": 10
})

# Access config
config = wandb.config

# Training loop
for epoch in range(config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                "loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })

# Save model
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")  # Upload to W&B

wandb.finish()
```

## 핵심 개념

### 1. 프로젝트와 실행

**Project**: 관련 실험의 모음
**Run**: 학습 스크립트가 한 번 실행된 것

```python
# Create/use project
run = wandb.init(
    project="image-classification",
    name="resnet50-experiment-1",  # Optional run name
    tags=["baseline", "resnet"],    # Organize with tags
    notes="First baseline run"      # Add notes
)

# Each run has unique ID
print(f"Run ID: {run.id}")
print(f"Run URL: {run.url}")
```

### 2. 구성 추적

하이퍼파라미터를 자동으로 추적합니다:

```python
config = {
    # Model architecture
    "model": "ResNet50",
    "pretrained": True,

    # Training params
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "optimizer": "Adam",

    # Data params
    "dataset": "ImageNet",
    "augmentation": "standard"
}

wandb.init(project="my-project", config=config)

# Access config during training
lr = wandb.config.learning_rate
batch_size = wandb.config.batch_size
```

### 3. 메트릭 기록

```python
# Log scalars
wandb.log({"loss": 0.5, "accuracy": 0.92})

# Log multiple metrics
wandb.log({
    "train/loss": train_loss,
    "train/accuracy": train_acc,
    "val/loss": val_loss,
    "val/accuracy": val_acc,
    "learning_rate": current_lr,
    "epoch": epoch
})

# Log with custom x-axis
wandb.log({"loss": loss}, step=global_step)

# Log media (images, audio, video)
wandb.log({"examples": [wandb.Image(img) for img in images]})

# Log histograms
wandb.log({"gradients": wandb.Histogram(gradients)})

# Log tables
table = wandb.Table(columns=["id", "prediction", "ground_truth"])
wandb.log({"predictions": table})
```

### 4. 모델 체크포인트 저장

```python
import torch
import wandb

# Save model checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}

torch.save(checkpoint, 'checkpoint.pth')

# Upload to W&B
wandb.save('checkpoint.pth')

# Or use Artifacts (recommended)
artifact = wandb.Artifact('model', type='model')
artifact.add_file('checkpoint.pth')
wandb.log_artifact(artifact)
```

## 하이퍼파라미터 스윕

최적의 하이퍼파라미터를 자동으로 검색합니다.

### 스윕 구성 정의

```python
sweep_config = {
    'method': 'bayes',  # or 'grid', 'random'
    'metric': {
        'name': 'val/accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")
```

### 학습 함수 정의

```python
def train():
    # Initialize run
    run = wandb.init()

    # Access sweep parameters
    lr = wandb.config.learning_rate
    batch_size = wandb.config.batch_size
    optimizer_name = wandb.config.optimizer

    # Build model with sweep config
    model = build_model(wandb.config)
    optimizer = get_optimizer(optimizer_name, lr)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, optimizer, batch_size)
        val_acc = validate(model)

        # Log metrics
        wandb.log({
            "train/loss": train_loss,
            "val/accuracy": val_acc
        })

# Run sweep
wandb.agent(sweep_id, function=train, count=50)  # Run 50 trials
```

### 스윕 전략

```python
# Grid search - exhaustive
sweep_config = {
    'method': 'grid',
    'parameters': {
        'lr': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64]}
    }
}

# Random search
sweep_config = {
    'method': 'random',
    'parameters': {
        'lr': {'distribution': 'uniform', 'min': 0.0001, 'max': 0.1},
        'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5}
    }
}

# Bayesian optimization (recommended)
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/loss', 'goal': 'minimize'},
    'parameters': {
        'lr': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-1}
    }
}
```

## 아티팩트

계보와 함께 데이터셋, 모델 및 기타 파일을 추적합니다.

### 아티팩트 기록

```python
# Create artifact
artifact = wandb.Artifact(
    name='training-dataset',
    type='dataset',
    description='ImageNet training split',
    metadata={'size': '1.2M images', 'split': 'train'}
)

# Add files
artifact.add_file('data/train.csv')
artifact.add_dir('data/images/')

# Log artifact
wandb.log_artifact(artifact)
```

### 아티팩트 사용

```python
# Download and use artifact
run = wandb.init(project="my-project")

# Download artifact
artifact = run.use_artifact('training-dataset:latest')
artifact_dir = artifact.download()

# Use the data
data = load_data(f"{artifact_dir}/train.csv")
```

### 모델 레지스트리

```python
# Log model as artifact
model_artifact = wandb.Artifact(
    name='resnet50-model',
    type='model',
    metadata={'architecture': 'ResNet50', 'accuracy': 0.95}
)

model_artifact.add_file('model.pth')
wandb.log_artifact(model_artifact, aliases=['best', 'production'])

# Link to model registry
run.link_artifact(model_artifact, 'model-registry/production-models')
```

## 통합 예시

### HuggingFace Transformers

```python
from transformers import Trainer, TrainingArguments
import wandb

# Initialize W&B
wandb.init(project="hf-transformers")

# Training arguments with W&B
training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # Enable W&B logging
    run_name="bert-finetuning",
    logging_steps=100,
    save_steps=500
)

# Trainer automatically logs to W&B
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

### PyTorch Lightning

```python
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

# Create W&B logger
wandb_logger = WandbLogger(
    project="lightning-demo",
    log_model=True  # Log model checkpoints
)

# Use with Trainer
trainer = Trainer(
    logger=wandb_logger,
    max_epochs=10
)

trainer.fit(model, datamodule=dm)
```

### Keras/TensorFlow

```python
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Initialize
wandb.init(project="keras-demo")

# Add callbacks (the monolithic WandbCallback was removed;
# use the dedicated callbacks from wandb.integration.keras instead)
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[
        WandbMetricsLogger(),                        # Auto-logs metrics
        WandbModelCheckpoint("models/model-{epoch}")  # Saves checkpoints
    ]
)
```

## 시각화 및 분석

### 사용자 지정 차트

```python
# Log custom visualizations
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(x, y)
wandb.log({"custom_plot": wandb.Image(fig)})

# Log confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=ground_truth,
    preds=predictions,
    class_names=class_names
)})
```

### 리포트

W&B UI에서 공유 가능한 리포트를 만듭니다:
- 실행, 차트 및 텍스트 결합
- Markdown 지원
- 삽입 가능한 시각화
- 팀 협업

## 모범 사례

### 1. 태그와 그룹으로 정리

```python
wandb.init(
    project="my-project",
    tags=["baseline", "resnet50", "imagenet"],
    group="resnet-experiments",  # Group related runs
    job_type="train"             # Type of job
)
```

### 2. 관련된 모든 항목 기록

```python
# Log system metrics
wandb.log({
    "gpu/util": gpu_utilization,
    "gpu/memory": gpu_memory_used,
    "cpu/util": cpu_utilization
})

# Log code version
wandb.log({"git_commit": git_commit_hash})

# Log data splits
wandb.log({
    "data/train_size": len(train_dataset),
    "data/val_size": len(val_dataset)
})
```

### 3. 설명적인 이름 사용

```python
# ✅ Good: Descriptive run names
wandb.init(
    project="nlp-classification",
    name="bert-base-lr0.001-bs32-epoch10"
)

# ❌ Bad: Generic names
wandb.init(project="nlp", name="run1")
```

### 4. 중요한 아티팩트 저장

```python
# Save final model
artifact = wandb.Artifact('final-model', type='model')
artifact.add_file('model.pth')
wandb.log_artifact(artifact)

# Save predictions for analysis
predictions_table = wandb.Table(
    columns=["id", "input", "prediction", "ground_truth"],
    data=predictions_data
)
wandb.log({"predictions": predictions_table})
```

### 5. 불안정한 연결에서는 오프라인 모드 사용

```python
import os

# Enable offline mode
os.environ["WANDB_MODE"] = "offline"

wandb.init(project="my-project")
# ... your code ...

# Sync later
# wandb sync <run_directory>
```

## 팀 협업

### 실행 공유

```python
# Runs are automatically shareable via URL
run = wandb.init(project="team-project")
print(f"Share this URL: {run.url}")
```

### 팀 프로젝트

- wandb.ai에서 팀 계정 생성
- 팀 구성원 추가
- 프로젝트 공개 범위 설정(비공개/공개)
- 팀 수준의 아티팩트와 모델 레지스트리 사용

## 요금

- **무료**: 공개 프로젝트 무제한, 100GB 저장 공간
- **Academic**: 학생/연구자에게 무료
- **Teams**: 시트당 월 $50, 비공개 프로젝트, 무제한 저장 공간
- **Enterprise**: 맞춤형 요금, 온프레미스 옵션

## 리소스

- **문서**: https://docs.wandb.ai
- **GitHub**: https://github.com/wandb/wandb (10.5k+ stars)
- **예시**: https://github.com/wandb/examples
- **커뮤니티**: https://wandb.ai/community
- **Discord**: https://wandb.me/discord

## 함께 보기

- `references/sweeps.md` - 종합 하이퍼파라미터 최적화 가이드
- `references/artifacts.md` - 데이터 및 모델 버전 관리 패턴
- `references/integrations.md` - 프레임워크별 예시

---
title: "Evaluating Llms Weights And Biases — Weights & Biases (W&B): 실험 추적 및 모델 평가"
sidebar_label: "Evaluating Llms Weights And Biases"
description: "Weights & Biases (W&B): 실험 추적 및 모델 평가"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Evaluating Llms Weights And Biases

Weights & Biases (W&B): 실험 추적 및 모델 평가.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/evaluation/weights-and-biases` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `wandb` |
| 플랫폼 | linux, macos |
| 태그 | `Evaluation`, `Experiment Tracking`, `W&B`, `Model Evaluation`, `Hyperparameter Tuning`, `Artifacts`, `MLOps` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Weights & Biases (W&B) - 모델 평가 및 추적

Weights & Biases (wandb)는 기계 학습을 위한 개발자 도구 세트입니다. 모델 평가 및 실험 추적을 위해 이 스킬을 사용하세요.

## 포함된 내용

기계 학습 모델 및 대규모 언어 모델(LLM)을 평가하기 위한 도구로, 지표 추적, 모델 예측값 로깅, 데이터셋 및 모델 버전 관리(Artifacts) 기능을 제공합니다.

## 빠른 시작

1. **설치 및 로그인**:
```bash
pip install wandb
wandb login
```

2. **기본 추적**:
```python
import wandb

# 새로운 실험(Run) 초기화
wandb.init(
    project="my-awesome-project",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
)

# 학습 루프 시뮬레이션
for epoch in range(10):
    acc = 1 - (2 ** -epoch) # 더미 정확도
    loss = 2 ** -epoch      # 더미 손실값

    # W&B에 지표(metrics) 로깅
    wandb.log({"accuracy": acc, "loss": loss})

# 실험 종료
wandb.finish()
```

## 핵심 개념

- **Runs (실험)**: wandb의 핵심 추적 단위입니다.
- **Projects (프로젝트)**: 서로 비교하기 위한 실험(Runs)들의 모음입니다.
- **Config (구성)**: 학습률 등 하이퍼파라미터를 저장하여 나중에 재현하고 비교할 수 있게 합니다.
- **Metrics (지표)**: 시간/에포크(epoch)에 따라 로깅되는 시계열 데이터(예: 손실값, 정확도)입니다.
- **Artifacts (아티팩트)**: 재현성을 위한 데이터셋 및 모델의 버전 관리 시스템입니다.
- **Tables (테이블)**: 대시보드 내에서 대화형으로 필터링하고 그룹화할 수 있는 예측 데이터(텍스트, 이미지 등)를 저장합니다.

## 일반적인 워크플로우

### 워크플로우 1: LLM 예측값 평가

표 형태로 프롬프트에 대한 모델 응답을 기록하고 시각화합니다. 정성적 평가(qualitative evaluation)에 유용합니다.

```python
import wandb
import random

# 데이터셋
prompts = [
    "Explain quantum computing in one sentence.",
    "Write a haiku about a robot.",
    "What is the capital of France?"
]

# 예측 함수 시뮬레이션 (여기선 모델 호출로 대체)
def generate_response(prompt, temperature):
    responses = {
        "Explain quantum computing in one sentence.": [
            "Quantum computing uses quantum bits to perform calculations exponentially faster.",
            "It's like computing with states that are 0 and 1 at the same time."
        ],
         "Write a haiku about a robot.": [
             "Metal hands so cold,\nThinking with a mind of code,\nSoul of electric.",
             "Gears turn in the night,\nComputing the stars above,\nQuiet metal friend."
         ],
         "What is the capital of France?": ["Paris.", "Paris is the capital."]
    }
    return random.choice(responses[prompt])

def evaluate_model(temperature=0.7):
    # run 초기화
    run = wandb.init(
        project="llm-eval-demo",
        config={"model": "dummy-llm", "temperature": temperature}
    )

    # 결과를 저장할 테이블 생성
    columns = ["Prompt", "Temperature", "Generated Text", "Target Length (chars)", "Actual Length (chars)"]
    eval_table = wandb.Table(columns=columns)

    for prompt in prompts:
        # 모델 출력 생성
        response = generate_response(prompt, temperature)

        # 행(row) 추가
        eval_table.add_data(
            prompt,
            temperature,
            response,
            100, # 목표 길이 (더미값)
            len(response)
        )

    # 테이블 로그
    run.log({"evaluation_results": eval_table})
    run.finish()

evaluate_model(0.5)
```

### 워크플로우 2: 데이터셋 및 모델 버전 관리 (Artifacts)

Artifacts를 사용해 평가에 사용된 데이터셋과 모델의 버전을 관리합니다.

```python
import wandb
import os

# 더미 데이터셋 생성
with open("test_dataset.csv", "w") as f:
    f.write("text,label\nhello,1\ngoodbye,0\n")

run = wandb.init(project="artifact-demo")

# 1. 데이터셋 아티팩트 생성 및 로깅
dataset_artifact = wandb.Artifact(
    name="my_test_dataset",
    type="dataset",
    description="Test dataset for evaluation",
    metadata={"source": "user_generated", "size": 2}
)
dataset_artifact.add_file("test_dataset.csv")
run.log_artifact(dataset_artifact)

# 2. 나중에 평가 스크립트에서 (다른 run 내에서)
# 특정 버전 사용 가능. 예: "my_test_dataset:v0" 또는 "my_test_dataset:latest"
downloaded_artifact = run.use_artifact('my_test_dataset:latest')
dataset_dir = downloaded_artifact.download()
print(f"Dataset downloaded to: {dataset_dir}")

run.finish()
```

### 워크플로우 3: 멀티모달 로깅

이미지, 오디오, 기타 리치 미디어(rich media)를 평가합니다.

```python
import wandb
import numpy as np

run = wandb.init(project="multimodal-demo")

# 시각화할 더미 이미지
images = []
for i in range(3):
    img_data = np.random.randint(255, size=(100, 100, 3), dtype=np.uint8)
    # 캡션이 있는 wandb.Image 생성
    images.append(wandb.Image(img_data, caption=f"Random image {i}"))

# 이미지 로깅 (동일 키의 리스트는 UI에서 갤러리로 표시됨)
run.log({"generated_images": images})

run.finish()
```

### 워크플로우 4: 프레임워크 통합

Hugging Face, PyTorch Lightning 등 널리 사용되는 프레임워크에 대한 기본 통합(built-in integrations)을 사용합니다.

**Hugging Face Transformers 예시:**
```python
from transformers import TrainingArguments, Trainer
import wandb

# 환경 변수로 W&B 로깅 활성화 (또는 TrainingArguments 내 report_to="wandb")
import os
os.environ["WANDB_PROJECT"] = "hf-integration-demo"

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb", # 중요!
    run_name="my-hf-run",
    # ... 기타 학습 인자
)

# Trainer 초기화 및 학습 (wandb는 자동으로 loss, 지표, 모델 구성을 기록함)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
# )
# trainer.train()
```

## 성능 최적화 / 모범 사례

1. **루프 밖에서 init 호출**: `wandb.init()`은 루프(예: 에포크마다) 내에서 호출하지 마세요. 각 실행(run) 당 한 번만 호출해야 합니다.
2. **지표 그룹화**: `wandb.log({"train/loss": 0.5, "val/loss": 0.4})`와 같이 슬래시(`/`)를 사용하면 UI에서 차트가 자동으로 그룹화됩니다.
3. **요약(Summary) 지표**: wandb는 각 지표의 마지막 값(또는 최소/최대값)을 자동으로 요약합니다. 특정 지표(예: 최고 정확도)를 덮어쓰려면 `wandb.run.summary["best_accuracy"] = 0.95`를 사용하세요.

## 일반적인 문제

**문제: 오프라인이거나 네트워크가 제한된 환경**
- **해결책**: `WANDB_MODE=offline` 환경 변수를 사용하거나, `wandb.init(mode="offline")`을 전달하세요. 네트워크 연결이 가능할 때 `wandb sync wandb/run-folder`를 사용해 동기화합니다.

**문제: UI에 표시되는 표/차트가 너무 많음**
- **해결책**: `wandb.log` 호출의 빈도(stride)를 확인하세요(예: 매 스텝마다가 아니라 `step % 100 == 0`일 때만 로깅).

## 리소스

- **공식 문서**: https://docs.wandb.ai/
- **LLM 평가 문서**: https://docs.wandb.ai/guides/prompts

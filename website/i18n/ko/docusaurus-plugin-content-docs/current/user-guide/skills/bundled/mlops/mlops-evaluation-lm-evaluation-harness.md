---
title: "Evaluating Llms Harness — lm-eval-harness: LLM 벤치마킹 (MMLU, GSM8K 등)"
sidebar_label: "Evaluating Llms Harness"
description: "lm-eval-harness: LLM 벤치마킹 (MMLU, GSM8K 등)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Evaluating Llms Harness

lm-eval-harness: LLM 벤치마킹 (MMLU, GSM8K 등).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 내장 (기본으로 설치됨) |
| 경로 | `skills/mlops/evaluation/lm-evaluation-harness` |
| 버전 | `1.0.0` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 의존성 | `lm-eval`, `transformers`, `vllm` |
| 플랫폼 | linux, macos |
| 태그 | `Evaluation`, `LM Evaluation Harness`, `Benchmarking`, `MMLU`, `HumanEval`, `GSM8K`, `EleutherAI`, `Model Quality`, `Academic Benchmarks`, `Industry Standard` |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# lm-evaluation-harness - LLM 벤치마킹

## 포함된 내용

60개 이상의 학술 벤치마크(MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag)에서 LLM을 평가합니다. 모델 품질 벤치마킹, 모델 비교, 학술 결과 보고 또는 학습 진행 상황을 추적할 때 사용합니다. EleutherAI, HuggingFace 및 주요 연구소에서 사용하는 업계 표준입니다. HuggingFace, vLLM, API를 지원합니다.

## 빠른 시작

lm-evaluation-harness는 표준화된 프롬프트와 지표를 사용하여 60개 이상의 학술 벤치마크에서 LLM을 평가합니다.

**설치**:
```bash
pip install lm-eval
```

**모든 HuggingFace 모델 평가**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size 8
```

**사용 가능한 작업(tasks) 보기**:
```bash
lm_eval --tasks list
```

## 일반적인 워크플로우

### 워크플로우 1: 표준 벤치마크 평가

핵심 벤치마크(MMLU, GSM8K, HumanEval)에서 모델을 평가합니다.

다음 체크리스트를 복사하여 사용하세요:

```
벤치마크 평가:
- [ ] 1단계: 벤치마크 제품군(suite) 선택
- [ ] 2단계: 모델 구성
- [ ] 3단계: 평가 실행
- [ ] 4단계: 결과 분석
```

**1단계: 벤치마크 제품군 선택**

**핵심 추론 벤치마크**:
- **MMLU** (Massive Multitask Language Understanding) - 57개 주제, 객관식
- **GSM8K** - 초등학교 수학 활용 문제
- **HellaSwag** - 상식 추론
- **TruthfulQA** - 진실성(Truthfulness) 및 사실성
- **ARC** (AI2 Reasoning Challenge) - 과학 문제

**코드 벤치마크**:
- **HumanEval** - Python 코드 생성 (164개 문제)
- **MBPP** (Mostly Basic Python Problems) - Python 코딩

**표준 제품군** (모델 출시에 권장):
```bash
--tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge
```

**2단계: 모델 구성**

**HuggingFace 모델**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16 \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size auto  # 최적의 배치 크기 자동 감지
```

**양자화된 모델 (4-bit/8-bit)**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,load_in_4bit=True \
  --tasks mmlu \
  --device cuda:0
```

**커스텀 체크포인트**:
```bash
lm_eval --model hf \
  --model_args pretrained=/path/to/my-model,tokenizer=/path/to/tokenizer \
  --tasks mmlu \
  --device cuda:0
```

**3단계: 평가 실행**

```bash
# 전체 MMLU 평가 (57개 주제)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --num_fewshot 5 \  # 5-shot 평가 (표준)
  --batch_size 8 \
  --output_path results/ \
  --log_samples  # 개별 예측 저장

# 한 번에 여러 벤치마크 실행
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path results/llama2-7b-eval.json
```

**4단계: 결과 분석**

결과는 `results/llama2-7b-eval.json`에 저장됩니다:

```json
{
  "results": {
    "mmlu": {
      "acc": 0.459,
      "acc_stderr": 0.004
    },
    "gsm8k": {
      "exact_match": 0.142,
      "exact_match_stderr": 0.006
    },
    "hellaswag": {
      "acc_norm": 0.765,
      "acc_norm_stderr": 0.004
    }
  },
  "config": {
    "model": "hf",
    "model_args": "pretrained=meta-llama/Llama-2-7b-hf",
    "num_fewshot": 5
  }
}
```

### 워크플로우 2: 학습 진행 상황 추적

학습 중 체크포인트를 평가합니다.

```
학습 진행 추적:
- [ ] 1단계: 주기적 평가 설정
- [ ] 2단계: 빠른 벤치마크 선택
- [ ] 3단계: 평가 자동화
- [ ] 4단계: 학습 곡선 그리기
```

**1단계: 주기적 평가 설정**

N번의 학습 단계마다 평가:

```bash
#!/bin/bash
# eval_checkpoint.sh

CHECKPOINT_DIR=$1
STEP=$2

lm_eval --model hf \
  --model_args pretrained=$CHECKPOINT_DIR/checkpoint-$STEP \
  --tasks gsm8k,hellaswag \
  --num_fewshot 0 \  # 속도를 위해 0-shot
  --batch_size 16 \
  --output_path results/step-$STEP.json
```

**2단계: 빠른 벤치마크 선택**

자주 평가하기 위한 빠른 벤치마크:
- **HellaSwag**: 1 GPU에서 약 10분
- **GSM8K**: 약 5분
- **PIQA**: 약 2분

잦은 평가에 피해야 할 것(너무 느림):
- **MMLU**: 약 2시간 (57개 주제)
- **HumanEval**: 코드 실행 필요

**3단계: 평가 자동화**

학습 스크립트와 통합:

```python
# 학습 루프 내
if step % eval_interval == 0:
    model.save_pretrained(f"checkpoints/step-{step}")

    # 평가 실행
    os.system(f"./eval_checkpoint.sh checkpoints step-{step}")
```

또는 PyTorch Lightning 콜백 사용:

```python
from pytorch_lightning import Callback

class EvalHarnessCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        step = trainer.global_step
        checkpoint_path = f"checkpoints/step-{step}"

        # 체크포인트 저장
        trainer.save_checkpoint(checkpoint_path)

        # lm-eval 실행
        os.system(f"lm_eval --model hf --model_args pretrained={checkpoint_path} ...")
```

**4단계: 학습 곡선 그리기**

```python
import json
import matplotlib.pyplot as plt

# 모든 결과 로드
steps = []
mmlu_scores = []

for file in sorted(glob.glob("results/step-*.json")):
    with open(file) as f:
        data = json.load(f)
        step = int(file.split("-")[1].split(".")[0])
        steps.append(step)
        mmlu_scores.append(data["results"]["mmlu"]["acc"])

# 플롯
plt.plot(steps, mmlu_scores)
plt.xlabel("Training Step")
plt.ylabel("MMLU Accuracy")
plt.title("Training Progress")
plt.savefig("training_curve.png")
```

### 워크플로우 3: 여러 모델 비교

모델 비교를 위한 벤치마크 제품군.

```
모델 비교:
- [ ] 1단계: 모델 목록 정의
- [ ] 2단계: 평가 실행
- [ ] 3단계: 비교 표 생성
```

**1단계: 모델 목록 정의**

```bash
# models.txt
meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-hf
mistralai/Mistral-7B-v0.1
microsoft/phi-2
```

**2단계: 평가 실행**

```bash
#!/bin/bash
# eval_all_models.sh

TASKS="mmlu,gsm8k,hellaswag,truthfulqa"

while read model; do
    echo "Evaluating $model"

    # 출력 파일을 위한 모델 이름 추출
    model_name=$(echo $model | sed 's/\//-/g')

    lm_eval --model hf \
      --model_args pretrained=$model,dtype=bfloat16 \
      --tasks $TASKS \
      --num_fewshot 5 \
      --batch_size auto \
      --output_path results/$model_name.json

done < models.txt
```

**3단계: 비교 표 생성**

```python
import json
import pandas as pd

models = [
    "meta-llama-Llama-2-7b-hf",
    "meta-llama-Llama-2-13b-hf",
    "mistralai-Mistral-7B-v0.1",
    "microsoft-phi-2"
]

tasks = ["mmlu", "gsm8k", "hellaswag", "truthfulqa"]

results = []
for model in models:
    with open(f"results/{model}.json") as f:
        data = json.load(f)
        row = {"Model": model.replace("-", "/")}
        for task in tasks:
            # 각 작업의 기본 지표 가져오기
            metrics = data["results"][task]
            if "acc" in metrics:
                row[task.upper()] = f"{metrics['acc']:.3f}"
            elif "exact_match" in metrics:
                row[task.upper()] = f"{metrics['exact_match']:.3f}"
        results.append(row)

df = pd.DataFrame(results)
print(df.to_markdown(index=False))
```

출력:
```
| Model                  | MMLU  | GSM8K | HELLASWAG | TRUTHFULQA |
|------------------------|-------|-------|-----------|------------|
| meta-llama/Llama-2-7b  | 0.459 | 0.142 | 0.765     | 0.391      |
| meta-llama/Llama-2-13b | 0.549 | 0.287 | 0.801     | 0.430      |
| mistralai/Mistral-7B   | 0.626 | 0.395 | 0.812     | 0.428      |
| microsoft/phi-2        | 0.560 | 0.613 | 0.682     | 0.447      |
```

### 워크플로우 4: vLLM으로 평가 (더 빠른 추론)

5~10배 더 빠른 평가를 위해 vLLM 백엔드를 사용합니다.

```
vLLM 평가:
- [ ] 1단계: vLLM 설치
- [ ] 2단계: vLLM 백엔드 구성
- [ ] 3단계: 평가 실행
```

**1단계: vLLM 설치**

```bash
pip install vllm
```

**2단계: vLLM 백엔드 구성**

```bash
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
  --tasks mmlu \
  --batch_size auto
```

**3단계: 평가 실행**

vLLM은 표준 HuggingFace보다 5-10배 빠릅니다:

```bash
# 표준 HF: 7B 모델에서 MMLU 약 2시간
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --batch_size 8

# vLLM: 7B 모델에서 MMLU 약 15-20분
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=2 \
  --tasks mmlu \
  --batch_size auto
```

## 언제 사용하고 대안은 무엇인가

**다음과 같은 경우 lm-evaluation-harness를 사용하세요:**
- 학술 논문을 위한 모델 벤치마킹
- 표준 작업을 통한 모델 품질 비교
- 학습 진행 상황 추적
- 표준화된 지표 보고 (모두가 동일한 프롬프트를 사용함)
- 재현 가능한 평가가 필요할 때

**대신 다음 대안을 사용하는 것이 좋은 경우:**
- **HELM** (Stanford): 더 넓은 범위의 평가 (공정성, 효율성, 캘리브레이션)
- **AlpacaEval**: LLM 심사관을 이용한 지시 수행(instruction-following) 평가
- **MT-Bench**: 대화형 멀티턴(multi-turn) 평가
- **커스텀 스크립트**: 도메인 특화 평가

## 일반적인 문제

**문제: 평가가 너무 느림**

vLLM 백엔드 사용:
```bash
lm_eval --model vllm \
  --model_args pretrained=model-name,tensor_parallel_size=2
```

또는 fewshot 예제 줄이기:
```bash
--num_fewshot 0  # 5 대신 0 사용
```

또는 MMLU의 부분 집합만 평가:
```bash
--tasks mmlu_stem  # STEM 주제만
```

**문제: 메모리 부족 (Out of memory)**

배치 크기 줄이기:
```bash
--batch_size 1  # 또는 --batch_size auto
```

양자화 사용:
```bash
--model_args pretrained=model-name,load_in_8bit=True
```

CPU 오프로딩 활성화:
```bash
--model_args pretrained=model-name,device_map=auto,offload_folder=offload
```

**문제: 보고된 것과 결과가 다름**

fewshot 횟수 확인:
```bash
--num_fewshot 5  # 대부분의 논문은 5-shot 사용
```

정확한 작업 이름 확인:
```bash
--tasks mmlu  # mmlu_direct나 mmlu_fewshot이 아님
```

모델과 토크나이저 일치 여부 확인:
```bash
--model_args pretrained=model-name,tokenizer=same-model-name
```

**문제: HumanEval이 코드를 실행하지 않음**

실행 종속성 설치:
```bash
pip install human-eval
```

코드 실행 활성화:
```bash
lm_eval --model hf \
  --model_args pretrained=model-name \
  --tasks humaneval \
  --allow_code_execution  # HumanEval에 필수
```

## 고급 주제

**벤치마크 설명**: 60개 이상의 모든 작업에 대한 상세 설명, 측정 대상, 해석에 대해서는 [references/benchmark-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/benchmark-guide.md)를 참조하세요.

**커스텀 작업**: 도메인 특화 평가 작업 생성에 대해서는 [references/custom-tasks.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/custom-tasks.md)를 참조하세요.

**API 평가**: OpenAI, Anthropic 및 기타 API 모델 평가에 대해서는 [references/api-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/api-evaluation.md)를 참조하세요.

**다중 GPU 전략**: 데이터 병렬(Data parallel) 및 텐서 병렬(Tensor parallel) 평가에 대해서는 [references/distributed-eval.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/lm-evaluation-harness/references/distributed-eval.md)를 참조하세요.

## 하드웨어 요구 사항

- **GPU**: NVIDIA (CUDA 11.8+), CPU에서도 작동함 (매우 느림)
- **VRAM**:
  - 7B 모델: 16GB (bf16) 또는 8GB (8-bit)
  - 13B 모델: 28GB (bf16) 또는 14GB (8-bit)
  - 70B 모델: 다중 GPU 또는 양자화 필요
- **시간** (7B 모델, 단일 A100):
  - HellaSwag: 10분
  - GSM8K: 5분
  - MMLU (전체): 2시간
  - HumanEval: 20분

## 리소스

- GitHub: https://github.com/EleutherAI/lm-evaluation-harness
- Docs: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs
- 작업 라이브러리: MMLU, GSM8K, HumanEval, TruthfulQA, HellaSwag, ARC, WinoGrande 등을 포함한 60개 이상의 작업
- 리더보드: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard (이 harness를 사용함)

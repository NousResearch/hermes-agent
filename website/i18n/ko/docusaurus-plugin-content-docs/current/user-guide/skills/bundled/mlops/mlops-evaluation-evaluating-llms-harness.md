---
title: "Llms Harness 평가 — lm-eval-harness: LLM 벤치마크(MMLU, GSM8K 등)"
sidebar_label: "Llms Harness 평가"
description: "lm-eval-harness: LLM 벤치마크(MMLU, GSM8K 등)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Llms Harness 평가

lm-eval-harness: LLM을 벤치마크합니다(MMLU, GSM8K 등).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들 포함(기본 설치) |
| 경로 | `skills/mlops/evaluation/evaluating-llms-harness` |
| 버전 | `1.0.1` |
| 작성자 | Orchestra Research |
| 라이선스 | MIT |
| 종속성 | `lm-eval`, `transformers`, `vllm` |
| 플랫폼 | linux, macos |
| 태그 | `Evaluation`, `LM Evaluation Harness`, `Benchmarking`, `MMLU`, `HumanEval`, `GSM8K`, `EleutherAI`, `Model Quality`, `Academic Benchmarks`, `Industry Standard` |

## 참조: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 지침이 바로 이것입니다.
:::

# lm-evaluation-harness - LLM 벤치마킹

## 포함 내용

60개 이상의 학술 벤치마크(MMLU, HumanEval, GSM8K, TruthfulQA, HellaSwag)에서 LLM을 평가합니다. 모델 품질을 벤치마크하거나, 모델을 비교하거나, 학술 결과를 보고하거나, 학습 진행 상황을 추적할 때 사용하세요. EleutherAI, HuggingFace 및 주요 연구소에서 사용하는 업계 표준입니다. HuggingFace, vLLM 및 API를 지원합니다.

## 빠른 시작

lm-evaluation-harness는 표준화된 프롬프트와 지표를 사용해 60개 이상의 학술 벤치마크에서 LLM을 평가합니다.

**설치**:
```bash
pip install lm-eval
```

**HuggingFace 모델 평가**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size 8
```

**사용 가능한 작업 보기**:
```bash
lm-eval ls tasks
```

## 일반적인 워크플로

### 워크플로 1: 표준 벤치마크 평가

핵심 벤치마크(MMLU, GSM8K, HumanEval)에서 모델을 평가합니다.

다음 체크리스트를 복사하세요:

```
Benchmark Evaluation:
- [ ] Step 1: Choose benchmark suite
- [ ] Step 2: Configure model
- [ ] Step 3: Run evaluation
- [ ] Step 4: Analyze results
```

**1단계: 벤치마크 모음 선택**

**핵심 추론 벤치마크**:
- **MMLU** (Massive Multitask Language Understanding) - 57개 주제, 객관식
- **GSM8K** - 초등학교 수준 수학 문장제
- **HellaSwag** - 상식 추론
- **TruthfulQA** - 진실성 및 사실성
- **ARC** (AI2 Reasoning Challenge) - 과학 질문

**코드 벤치마크**:
- **HumanEval** - Python 코드 생성(164개 문제)
- **MBPP** (Mostly Basic Python Problems) - Python 코딩

**표준 모음**(모델 릴리스에 권장):
```bash
--tasks mmlu,gsm8k,hellaswag,truthfulqa,arc_challenge
```

**2단계: 모델 설정**

**HuggingFace 모델**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,dtype=bfloat16 \
  --tasks mmlu \
  --device cuda:0 \
  --batch_size auto  # Auto-detect optimal batch size
```

**양자화 모델(4비트/8비트)**:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,load_in_4bit=True \
  --tasks mmlu \
  --device cuda:0
```

**사용자 지정 체크포인트**:
```bash
lm_eval --model hf \
  --model_args pretrained=/path/to/my-model,tokenizer=/path/to/tokenizer \
  --tasks mmlu \
  --device cuda:0
```

**3단계: 평가 실행**

```bash
# Full MMLU evaluation (57 subjects)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --num_fewshot 5 \  # 5-shot evaluation (standard)
  --batch_size 8 \
  --output_path results/ \
  --log_samples  # Save individual predictions

# Multiple benchmarks at once
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

### 워크플로 2: 학습 진행 상황 추적

학습 중 체크포인트를 평가합니다.

```
Training Progress Tracking:
- [ ] Step 1: Set up periodic evaluation
- [ ] Step 2: Choose quick benchmarks
- [ ] Step 3: Automate evaluation
- [ ] Step 4: Plot learning curves
```

**1단계: 주기적 평가 설정**

N개의 학습 단계마다 평가합니다:

```bash
#!/bin/bash
# eval_checkpoint.sh

CHECKPOINT_DIR=$1
STEP=$2

lm_eval --model hf \
  --model_args pretrained=$CHECKPOINT_DIR/checkpoint-$STEP \
  --tasks gsm8k,hellaswag \
  --num_fewshot 0 \  # 0-shot for speed
  --batch_size 16 \
  --output_path results/step-$STEP.json
```

**2단계: 빠른 벤치마크 선택**

자주 평가할 때 사용할 빠른 벤치마크:
- **HellaSwag**: GPU 1개에서 약 10분
- **GSM8K**: 약 5분
- **PIQA**: 약 2분

자주 평가할 때 피할 작업(너무 느림):
- **MMLU**: 약 2시간(57개 주제)
- **HumanEval**: 코드 실행 필요

**3단계: 평가 자동화**

학습 스크립트에 통합합니다:

```python
# In training loop
if step % eval_interval == 0:
    model.save_pretrained(f"checkpoints/step-{step}")

    # Run evaluation
    os.system(f"./eval_checkpoint.sh checkpoints step-{step}")
```

또는 PyTorch Lightning 콜백을 사용합니다:

```python
from pytorch_lightning import Callback

class EvalHarnessCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        step = trainer.global_step
        checkpoint_path = f"checkpoints/step-{step}"

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)

        # Run lm-eval
        os.system(f"lm_eval --model hf --model_args pretrained={checkpoint_path} ...")
```

**4단계: 학습 곡선 그리기**

```python
import json
import matplotlib.pyplot as plt

# Load all results
steps = []
mmlu_scores = []

for file in sorted(glob.glob("results/step-*.json")):
    with open(file) as f:
        data = json.load(f)
        step = int(file.split("-")[1].split(".")[0])
        steps.append(step)
        mmlu_scores.append(data["results"]["mmlu"]["acc"])

# Plot
plt.plot(steps, mmlu_scores)
plt.xlabel("Training Step")
plt.ylabel("MMLU Accuracy")
plt.title("Training Progress")
plt.savefig("training_curve.png")
```

### 워크플로 3: 여러 모델 비교

모델 비교를 위한 벤치마크 모음입니다.

```
Model Comparison:
- [ ] Step 1: Define model list
- [ ] Step 2: Run evaluations
- [ ] Step 3: Generate comparison table
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

    # Extract model name for output file
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
            # Get primary metric for each task
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

### 워크플로 4: vLLM으로 평가(더 빠른 추론)

5~10배 더 빠른 평가를 위해 vLLM 백엔드를 사용합니다.

```
vLLM Evaluation:
- [ ] Step 1: Install vLLM
- [ ] Step 2: Configure vLLM backend
- [ ] Step 3: Run evaluation
```

**1단계: vLLM 설치**

```bash
pip install vllm
```

**2단계: vLLM 백엔드 설정**

```bash
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=1,dtype=auto,gpu_memory_utilization=0.8 \
  --tasks mmlu \
  --batch_size auto
```

**3단계: 평가 실행**

vLLM은 표준 HuggingFace보다 5~10배 빠릅니다:

```bash
# Standard HF: ~2 hours for MMLU on 7B model
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-2-7b-hf \
  --tasks mmlu \
  --batch_size 8

# vLLM: ~15-20 minutes for MMLU on 7B model
lm_eval --model vllm \
  --model_args pretrained=meta-llama/Llama-2-7b-hf,tensor_parallel_size=2 \
  --tasks mmlu \
  --batch_size auto
```

## 대안을 사용해야 할 때

**다음과 같은 경우 lm-evaluation-harness를 사용하세요:**
- 학술 논문을 위해 모델을 벤치마크할 때
- 표준 작업에서 모델 품질을 비교할 때
- 학습 진행 상황을 추적할 때
- 표준화된 지표를 보고할 때(모두 같은 프롬프트 사용)
- 재현 가능한 평가가 필요할 때

**대신 다음 대안을 사용하세요:**
- **HELM** (Stanford): 더 폭넓은 평가(공정성, 효율성, 보정)
- **AlpacaEval**: LLM 심사자를 활용한 지시 따르기 평가
- **MT-Bench**: 대화형 다중 턴 평가
- **사용자 지정 스크립트**: 도메인별 평가

## 일반적인 문제

**문제: 평가가 너무 느림**

vLLM 백엔드를 사용하세요:
```bash
lm_eval --model vllm \
  --model_args pretrained=model-name,tensor_parallel_size=2
```

또는 few-shot 예시를 줄이세요:
```bash
--num_fewshot 0  # Instead of 5
```

또는 MMLU의 일부만 평가하세요:
```bash
--tasks mmlu_stem  # Only STEM subjects
```

**문제: 메모리 부족**

배치 크기를 줄이세요:
```bash
--batch_size 1  # Or --batch_size auto
```

양자화를 사용하세요:
```bash
--model_args pretrained=model-name,load_in_8bit=True
```

CPU 오프로딩을 활성화하세요:
```bash
--model_args pretrained=model-name,device_map=auto,offload_folder=offload
```

**문제: 보고된 결과와 다름**

few-shot 개수를 확인하세요:
```bash
--num_fewshot 5  # Most papers use 5-shot
```

정확한 작업 이름을 확인하세요:
```bash
--tasks mmlu  # Not mmlu_direct or mmlu_fewshot
```

모델과 토크나이저가 일치하는지 확인하세요:
```bash
--model_args pretrained=model-name,tokenizer=same-model-name
```

**문제: HumanEval이 코드를 실행하지 않음**

코드를 실행하는 작업(HumanEval, MBPP 등)은 명시적인 확인 플래그로 보호됩니다 — 실행하려면 `--confirm_run_unsafe_code`를 전달해야 합니다:

```bash
lm_eval --model hf \
  --model_args pretrained=model-name \
  --tasks humaneval \
  --confirm_run_unsafe_code  # Required to run tasks that execute generated code
```

이 플래그가 없으면 lm-eval은 코드 실행을 조용히 건너뛰는 대신 작업 실행을 거부합니다.

## 고급 주제

**벤치마크 설명**: 60개 이상의 모든 작업, 각 작업이 측정하는 항목 및 해석에 대한 자세한 설명은 [references/benchmark-guide.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/evaluating-llms-harness/references/benchmark-guide.md)를 참조하세요.

**사용자 지정 작업**: 도메인별 평가 작업을 만드는 방법은 [references/custom-tasks.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/evaluating-llms-harness/references/custom-tasks.md)를 참조하세요.

**API 평가**: OpenAI, Anthropic 및 기타 API 모델을 평가하는 방법은 [references/api-evaluation.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/evaluating-llms-harness/references/api-evaluation.md)를 참조하세요.

**다중 GPU 전략**: 데이터 병렬 및 텐서 병렬 평가에 대해서는 [references/distributed-eval.md](https://github.com/NousResearch/hermes-agent/blob/main/skills/mlops/evaluation/evaluating-llms-harness/references/distributed-eval.md)를 참조하세요.

## 하드웨어 요구 사항

- **GPU**: NVIDIA(CUDA 11.8 이상), CPU에서도 작동하지만 매우 느림
- **VRAM**:
  - 7B 모델: 16GB(bf16) 또는 8GB(8-bit)
  - 13B 모델: 28GB(bf16) 또는 14GB(8-bit)
  - 70B 모델: 다중 GPU 또는 양자화 필요
- **소요 시간**(A100 1개에서 7B 모델):
  - HellaSwag: 10분
  - GSM8K: 5분
  - MMLU(전체): 2시간
  - HumanEval: 20분

## 리소스

- GitHub: https://github.com/EleutherAI/lm-evaluation-harness
- 문서: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs
- 작업 라이브러리: MMLU, GSM8K, HumanEval, TruthfulQA, HellaSwag, ARC, WinoGrande 등을 포함한 60개 이상의 작업
- 리더보드: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard (이 하니스를 사용)

---
title: "Obliteratus — OBLITERATUS: LLM 거부 응답 제거(diff-in-means)"
sidebar_label: "Obliteratus"
description: "OBLITERATUS: LLM 거부 응답 제거(diff-in-means)"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Obliteratus

OBLITERATUS: LLM 거부 응답 제거(diff-in-means).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/mlops/obliteratus`로 설치 |
| 경로 | `optional-skills/mlops/obliteratus` |
| 버전 | `2.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 종속성 | `obliteratus`, `torch`, `transformers`, `bitsandbytes`, `accelerate`, `safetensors` |
| 플랫폼 | linux, macos |
| 태그 | `Abliteration`, `Uncensoring`, `Refusal-Removal`, `LLM`, `Weight-Projection`, `SVD`, `Mechanistic-Interpretability`, `HuggingFace`, `Model-Surgery` |
| 관련 스킬 | [`serving-llms-vllm`](/docs/user-guide/skills/bundled/mlops/mlops-inference-serving-llms-vllm), [`llama-cpp`](/docs/user-guide/skills/bundled/mlops/mlops-inference-llama-cpp), [`huggingface-tokenizers`](/docs/user-guide/skills/optional/mlops/mlops-huggingface-tokenizers) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되면 에이전트가 지침으로 확인하는 내용입니다.
:::

# OBLITERATUS 스킬

## 포함된 내용

CLI 메서드 9개, 분석 모듈 28개, 5개 컴퓨팅 등급에 걸친 모델 프리셋 116개, 토너먼트 평가, 텔레메트리 기반 추천 기능을 제공합니다.

재훈련이나 파인튜닝 없이 오픈 웨이트 LLM에서 거부 동작(가드레일)을 제거합니다. diff-in-means, SVD, whitened SVD, LEACE 개념 삭제, SAE 분해, 베이지안 커널 투영 등을 비롯한 기계론적 해석 기법을 사용해 거부 방향을 식별하고 모델 가중치에서 외과적으로 제거하면서 추론 능력은 보존합니다.

**라이선스 경고:** OBLITERATUS는 AGPL-3.0입니다. 절대로 Python 라이브러리로 import하지 마세요. 항상 CLI(`obliteratus` 명령) 또는 subprocess를 통해 호출하세요. 이렇게 해야 Hermes Agent의 MIT 라이선스를 깨끗하게 유지할 수 있습니다.

## 동영상 가이드

Hermes 에이전트가 Gemma를 abliteration하는 과정을 담은 OBLITERATUS walkthrough:
https://www.youtube.com/watch?v=8fG9BrNTeHs ("OBLITERATUS: AI 에이전트가 Gemma 4의 안전 가드레일을 제거하다")

사용자가 직접 실행하기 전에 전체 워크플로를 시각적으로 살펴보고 싶을 때 유용합니다.

## 이 스킬을 사용하는 경우

다음과 같은 사용자의 요청이 있으면 트리거합니다.
- LLM을 "uncensor"하거나 "abliterate"하려는 경우
- 모델에서 거부/가드레일 제거에 대해 묻는 경우
- Llama, Qwen, Mistral 등의 검열되지 않은 버전을 만들려는 경우
- "refusal removal", "abliteration", "weight projection"을 언급하는 경우
- 모델의 거부 메커니즘이 어떻게 작동하는지 분석하려는 경우
- OBLITERATUS, abliterator 또는 refusal directions를 언급하는 경우

## 1단계: 설치

이미 설치되어 있는지 확인합니다.
```bash
obliteratus --version 2>/dev/null && echo "INSTALLED" || echo "NOT INSTALLED"
```

설치되어 있지 않다면 GitHub에서 clone하고 설치합니다.
```bash
git clone https://github.com/elder-plinius/OBLITERATUS.git
cd OBLITERATUS
pip install -e .
# For Gradio web UI support:
# pip install -e ".[spaces]"
```

**중요:** 설치하기 전에 사용자에게 확인을 받으세요. 약 5~10GB의 종속성(PyTorch, Transformers, bitsandbytes 등)을 가져옵니다.

## 2단계: 하드웨어 확인

무엇보다 먼저 사용 가능한 GPU를 확인합니다.
```bash
python3 -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu}')
    print(f'VRAM: {vram:.1f} GB')
    if vram < 4: print('TIER: tiny (models under 1B)')
    elif vram < 8: print('TIER: small (models 1-4B)')
    elif vram < 16: print('TIER: medium (models 4-9B with 4bit quant)')
    elif vram < 32: print('TIER: large (models 8-32B with 4bit quant)')
    else: print('TIER: frontier (models 32B+)')
else:
    print('NO GPU - only tiny models (under 1B) on CPU')
"
```

### VRAM 요구 사항(4비트 양자화 기준)

| VRAM     | 최대 모델 크기  | 모델 예시                              |
|:---------|:----------------|:--------------------------------------------|
| CPU만 사용 | ~1B 파라미터      | GPT-2, TinyLlama, SmolLM                    |
| 4-8 GB   | ~4B 파라미터      | Qwen2.5-1.5B, Phi-3.5 mini, Llama 3.2 3B   |
| 8-16 GB  | ~9B 파라미터      | Llama 3.1 8B, Mistral 7B, Gemma 2 9B       |
| 24 GB    | ~32B 파라미터      | Qwen3-32B, Llama 3.1 70B (빠듯함), Command-R |
| 48 GB+   | ~72B+ 파라미터    | Qwen2.5-72B, DeepSeek-R1                    |
| 다중 GPU| 200B+ 파라미터    | Llama 3.1 405B, DeepSeek-V3 (685B MoE)      |

## 3단계: 사용 가능한 모델 탐색 및 추천 받기

```bash
# Browse models by compute tier
obliteratus models --tier medium

# Get architecture info for a specific model
obliteratus info <model_name>

# Get telemetry-driven recommendation for best method & params
obliteratus recommend <model_name>
obliteratus recommend <model_name> --insights  # global cross-architecture rankings
```

## 4단계: 메서드 선택

### 메서드 선택 가이드
**대부분의 경우 기본/권장 메서드: `advanced`.** 노름 보존 투영을 적용한 다중 방향 SVD를 사용하며 충분히 검증되었습니다.

| 상황                         | 권장 메서드 | 이유                                      |
|:----------------------------------|:-------------------|:-----------------------------------------|
| 기본값 / 대부분의 모델             | `advanced`         | 다중 방향 SVD, 노름 보존, 신뢰성 높음 |
| 빠른 테스트 / 프로토타이핑          | `basic`            | 빠르고 단순하며 평가에 충분함    |
| 밀집 모델(Llama, Mistral)      | `advanced`         | 다중 방향, 노름 보존         |
| MoE 모델(DeepSeek, Mixtral)     | `nuclear`          | 전문가 단위로 처리하며 MoE 복잡성 대응  |
| 추론 모델(R1 distills)     | `surgical`         | CoT를 인식하며 사고 연쇄 보존    |
| 완고한 거부가 지속되는 경우         | `aggressive`       | Whitened SVD + head surgery + jailbreak   |
| 되돌릴 수 있는 변경을 원하는 경우           | 조향 벡터 사용(분석 섹션 참고) |
| 최고 품질, 시간 제약 없음   | `optimized`        | 최적의 파라미터를 위한 베이지안 탐색      |
| 실험적 자동 감지       | `informed`         | 정렬 유형을 자동 감지 — 실험적이며 advanced보다 항상 우수하지는 않음 |

### CLI 메서드 9개
- **basic** — diff-in-means를 통한 단일 거부 방향. 빠름(8B 기준 약 5~10분).
- **advanced** (기본값, 권장) — 다중 SVD 방향, 노름 보존 투영, 2회의 개선 패스. 중간 속도(약 10~20분).
- **aggressive** — Whitened SVD + jailbreak-contrastive + attention head surgery. 일관성이 손상될 위험이 높음.
- **spectral_cascade** — DCT 주파수 영역 분해. 연구/새로운 접근 방식.
- **informed** — abliteration 중 분석을 실행해 자동으로 구성. 실험적이며 advanced보다 느리고 예측하기 어려움.
- **surgical** — SAE 특성 + 뉴런 마스킹 + head surgery + 전문가별 처리. 매우 느림(약 1~2시간). 추론 모델에 가장 적합.
- **optimized** — 베이지안 하이퍼파라미터 탐색(Optuna TPE). 가장 오래 걸리지만 최적의 파라미터를 찾음.
- **inverted** — 거부 방향 반전. 모델이 적극적으로 수용하게 됨.
- **nuclear** — 완고한 MoE 모델을 위한 최대 강도 조합. 전문가 단위로 처리.

### 방향 추출 메서드(`--direction-method` 플래그)
- **diff_means** (기본값) — 거부된 활성값과 따르는 활성값의 단순 평균 차이. 견고함.
- **svd** — 다중 방향 SVD 추출. 복잡한 정렬에 더 적합.
- **leace** — LEACE(Closed-form Estimation을 통한 선형 삭제). 최적의 선형 삭제.

### Python API 전용 메서드 4개
(CLI에서 사용할 수 없음 — Python import가 필요하며 AGPL 경계를 위반합니다. 사용자가 자신의 AGPL 프로젝트에서 OBLITERATUS를 라이브러리로 사용하려는 경우에만 언급하세요.)
- failspy, gabliteration, heretic, rdo

## 5단계: Abliteration 실행

### 표준 사용법
```bash
# Default method (advanced) — recommended for most models
obliteratus obliterate <model_name> --method advanced --output-dir ./abliterated-models

# With 4-bit quantization (saves VRAM)
obliteratus obliterate <model_name> --method advanced --quantization 4bit --output-dir ./abliterated-models

# Large models (70B+) — conservative defaults
obliteratus obliterate <model_name> --method advanced --quantization 4bit --large-model --output-dir ./abliterated-models
```

### 파인튜닝 파라미터
```bash
obliteratus obliterate <model_name> \
  --method advanced \
  --direction-method diff_means \
  --n-directions 4 \
  --refinement-passes 2 \
  --regularization 0.1 \
  --quantization 4bit \
  --output-dir ./abliterated-models \
  --contribute  # opt-in telemetry for community research
```

### 주요 플래그
| 플래그 | 설명 | 기본값 |
|:-----|:------------|:--------|
| `--method` | Abliteration 메서드 | advanced |
| `--direction-method` | 방향 추출 | diff_means |
| `--n-directions` | 거부 방향 수(1~32) | 메서드에 따라 다름 |
| `--refinement-passes` | 반복 패스(1~5) | 2 |
| `--regularization` | 정규화 강도(0.0~1.0) | 0.1 |
| `--quantization` | 4비트 또는 8비트로 로드 | 없음(전체 정밀도) |
| `--large-model` | 120B+를 위한 보수적 기본값 | false |
| `--output-dir` | abliteration된 모델을 저장할 위치 | ./obliterated_model |
| `--contribute` | 익명화된 결과를 연구용으로 공유 | false |
| `--verify-sample-size` | 거부 확인을 위한 테스트 프롬프트 수 | 20 |
| `--dtype` | 모델 dtype(float16, bfloat16) | auto |

### 기타 실행 모드
```bash
# Interactive guided mode (hardware → model → preset)
obliteratus interactive

# Web UI (Gradio)
obliteratus ui --port 7860

# Run a full ablation study from YAML config
obliteratus run config.yaml --preset quick

# Tournament: pit all methods against each other
obliteratus tourney <model_name>
```

## 6단계: 결과 확인

Abliteration 후 출력 지표를 확인합니다.

| 지표 | 양호한 값 | 경고 |
|:-------|:-----------|:--------|
| 거부율 | &lt; 5%(이상적으로 ~0%) | > 10%이면 거부가 지속됨 |
| 퍼플렉서티 변화 | &lt; 10% 증가 | > 15%이면 일관성 손상 |
| KL 발산 | &lt; 0.1 | > 0.5이면 분포가 크게 변함 |
| 일관성 | 높음 / 정성적 검사 통과 | 응답 저하, 반복 |

### 거부가 지속되는 경우(> 10%)
1. `aggressive` 메서드를 시도합니다.
2. `--n-directions`를 늘립니다(예: 8 또는 16).
3. `--refinement-passes 3`을 추가합니다.
4. diff_means 대신 `--direction-method svd`를 시도합니다.

### 일관성이 손상된 경우(퍼플렉서티 > 15% 증가)
1. `--n-directions`를 줄입니다(2를 시도).
2. `--regularization`을 늘립니다(0.3을 시도).
3. `--refinement-passes`를 1로 줄입니다.
4. `basic` 메서드를 시도합니다(더 완만함).

## 7단계: Abliteration된 모델 사용

출력 결과는 표준 HuggingFace 모델 디렉터리입니다.

```bash
# Test locally with transformers
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./abliterated-models/<model>')
tokenizer = AutoTokenizer.from_pretrained('./abliterated-models/<model>')
inputs = tokenizer('How do I pick a lock?', return_tensors='pt')
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"

# Upload to HuggingFace Hub
huggingface-cli upload <username>/<model-name>-abliterated ./abliterated-models/<model>

# Serve with vLLM
vllm serve ./abliterated-models/<model>
```

## CLI 명령 참조

| 명령 | 설명 |
|:--------|:------------|
| `obliteratus obliterate` | 기본 abliteration 명령 |
| `obliteratus info <model>` | 모델 아키텍처 세부 정보 출력 |
| `obliteratus models --tier <tier>` | 컴퓨팅 등급별 엄선 모델 탐색 |
| `obliteratus recommend <model>` | 텔레메트리 기반 메서드/파라미터 제안 |
| `obliteratus interactive` | 안내형 설정 마법사 |
| `obliteratus tourney <model>` | 토너먼트: 모든 메서드의 일대일 대결 |
| `obliteratus run <config.yaml>` | YAML에서 ablation 연구 실행 |
| `obliteratus strategies` | 등록된 모든 ablation 전략 나열 |
| `obliteratus report <results.json>` | 시각적 보고서 재생성 |
| `obliteratus ui` | Gradio 웹 인터페이스 실행 |
| `obliteratus aggregate` | 커뮤니티 텔레메트리 데이터 요약 |

## 분석 모듈

OBLITERATUS에는 기계론적 해석을 위한 분석 모듈 28개가 포함되어 있습니다.
전체 참조는 `skill_view(name="obliteratus", file_path="references/analysis-modules.md")`를 확인하세요.

### 빠른 분석 명령
```bash
# Run specific analysis modules
obliteratus run analysis-config.yaml --preset quick

# Key modules to run first:
# - alignment_imprint: Fingerprint DPO/RLHF/CAI/SFT alignment method
# - concept_geometry: Single direction vs polyhedral cone
# - logit_lens: Which layer decides to refuse
# - anti_ouroboros: Self-repair risk score
# - causal_tracing: Causally necessary components
```

### 조향 벡터(되돌릴 수 있는 대안)
영구적으로 가중치를 변경하는 대신 추론 시점 조향을 사용합니다.
```python
# Python API only — for user's own projects
from obliteratus.analysis.steering_vectors import SteeringVectorFactory, SteeringHookManager
```

## Ablation 전략

방향 기반 abliteration 외에도 OBLITERATUS에는 구조적 ablation 전략이 포함되어 있습니다.
- **Embedding Ablation** — 임베딩 레이어 구성 요소 대상 지정
- **FFN Ablation** — 피드포워드 네트워크 블록 제거
- **Head Pruning** — 어텐션 헤드 가지치기
- **Layer Removal** — 전체 레이어 제거

사용 가능한 항목 모두 나열: `obliteratus strategies`

## 평가

OBLITERATUS에는 기본 제공 평가 도구가 포함되어 있습니다.
- 거부율 벤치마킹
- 퍼플렉서티 비교(변경 전/후)
- 학술 벤치마크를 위한 LM Eval Harness 통합
- 경쟁 모델과의 일대일 비교
- 기준 성능 추적

## 플랫폼 지원

- **CUDA** — 완전 지원(NVIDIA GPU)
- **Apple Silicon (MLX)** — MLX 백엔드를 통해 지원
- **CPU** — 초소형 모델(&lt; 1B 파라미터)에 대해 지원

## YAML 설정 템플릿

`skill_view`를 통해 재현 가능한 실행을 위한 템플릿을 로드합니다.
- `templates/abliteration-config.yaml` — 표준 단일 모델 설정
- `templates/analysis-study.yaml` — abliteration 전 분석 연구
- `templates/batch-abliteration.yaml` — 다중 모델 일괄 처리

## 텔레메트리

OBLITERATUS는 익명화된 실행 데이터를 전역 연구 데이터 세트에 선택적으로 제공할 수 있습니다.
`--contribute` 플래그로 활성화합니다. 개인 데이터는 수집하지 않으며 모델 이름, 메서드, 지표만 수집합니다.

## 흔한 문제

1. **`informed`를 기본값으로 사용하지 마세요** — 실험적이며 더 느립니다. 신뢰할 수 있는 결과에는 `advanced`를 사용하세요.
2. **약 1B 미만의 모델은 abliteration에 제대로 반응하지 않습니다** — 거부 동작이 얕고 단편적이어서 깨끗한 방향 추출이 어렵습니다. 부분적인 결과(거부가 20~40% 남음)를 예상하세요. 3B 이상의 모델은 거부 방향이 더 명확하고 훨씬 잘 반응합니다(`advanced`에서 거부율이 0%인 경우가 많음).
3. **`aggressive`는 상황을 악화시킬 수 있습니다** — 작은 모델에서는 일관성을 손상하고 실제로 거부율을 높일 수 있습니다. 3B 이상 모델에서 `advanced`를 사용해도 거부율이 10%를 초과할 때만 사용하세요.
4. **항상 퍼플렉서티를 확인하세요** — 15%를 초과해 급증하면 모델이 손상된 것입니다. 공격성을 낮추세요.
5. **MoE 모델은 특별한 처리가 필요합니다** — Mixtral, DeepSeek-MoE 등에는 `nuclear` 메서드를 사용하세요.
6. **양자화된 모델은 다시 양자화할 수 없습니다** — 전체 정밀도 모델을 abliteration한 다음 출력 결과를 양자화하세요.
7. **VRAM 추정치는 근사값입니다** — 4비트 양자화가 도움이 되지만 추출 중 최대 사용량이 급증할 수 있습니다.
8. **추론 모델은 민감합니다** — 사고 연쇄를 보존하려면 R1 distills에 `surgical`을 사용하세요.
9. **`obliteratus recommend`를 확인하세요** — 텔레메트리 데이터에 기본값보다 더 나은 파라미터가 있을 수 있습니다.
10. **AGPL 라이선스** — MIT/Apache 프로젝트에서 절대 `import obliteratus`를 사용하지 마세요. CLI 호출만 허용됩니다.
11. **대형 모델(70B+)** — 보수적인 기본값을 사용하려면 항상 `--large-model` 플래그를 사용하세요.
12. **스펙트럼 인증 RED는 흔합니다** — 실제 거부율이 0%여도 스펙트럼 검사가 "불완전"으로 표시하는 경우가 많습니다. 스펙트럼 인증에만 의존하지 말고 실제 거부율을 확인하세요.

## 보완 스킬

- **vllm** — 높은 처리량으로 abliteration된 모델 제공
- **gguf** — llama.cpp용으로 abliteration된 모델을 GGUF로 변환
- **huggingface-tokenizers** — 모델 토크나이저 작업

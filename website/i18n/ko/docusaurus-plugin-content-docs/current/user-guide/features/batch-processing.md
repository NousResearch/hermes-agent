---
sidebar_position: 12
title: "배치 처리"
description: "대규모로 에이전트 궤적 생성 — 병렬 처리, 체크포인트, 도구 세트 분포"
---

# 배치 처리

배치 처리를 사용하면 수백 또는 수천 개의 프롬프트에 대해 Hermes 에이전트를 병렬로 실행하여 구조화된 궤적 데이터를 생성할 수 있습니다. 주로 **학습 데이터 생성**에 사용되며, 도구 사용 통계가 포함된 ShareGPT 형식의 궤적을 생성하여 파인튜닝이나 평가에 활용할 수 있습니다.

## 개요

배치 러너(`batch_runner.py`)는 프롬프트로 구성된 JSONL 데이터 세트를 처리하고, 각 프롬프트를 도구 접근 권한이 있는 전체 에이전트 세션으로 실행합니다. 각 프롬프트에는 격리된 환경이 할당됩니다. 출력은 전체 대화 기록, 도구 호출 통계, 추론 포함률 지표를 갖춘 구조화된 궤적 데이터입니다.

## 빠른 시작

```bash
# Basic batch run
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=4

# Resume an interrupted run
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --resume

# List available toolset distributions
python batch_runner.py --list_distributions
```

:::tip 대규모 실행에서도 예측 가능한 비용
배치 실행은 여러 동시 에이전트 세션을 시작하며, 각 세션은 모델 호출과 도구 호출을 수행합니다. [Nous Portal](/user-guide/features/tool-gateway) 구독은 웹 검색, 이미지 생성, TTS, 클라우드 브라우저와 함께 모델 이용을 하나의 청구서로 묶어 줍니다. 5개 공급업체 계정의 요청 제한을 따로 관리하지 않고 궤적당 비용을 안정적으로 유지하려 할 때 유용합니다. `hermes setup --portal`로 설정한 다음 `--model`에 Nous 모델을 지정하세요.
:::

## 데이터 세트 형식

입력 데이터 세트는 JSONL 파일입니다(한 줄에 JSON 객체 하나). 각 항목에는 `prompt` 필드가 있어야 합니다.

```jsonl
{"prompt": "Write a Python function that finds the longest palindromic substring"}
{"prompt": "Create a REST API endpoint for user authentication using Flask"}
{"prompt": "Debug this error: TypeError: cannot unpack non-iterable NoneType object"}
```

항목에는 다음 필드를 선택적으로 포함할 수 있습니다.
- `image` 또는 `docker_image`: 이 프롬프트의 샌드박스에 사용할 컨테이너 이미지(Docker, Modal, Singularity 백엔드에서 작동)
- `cwd`: 작업의 터미널 세션에 사용할 작업 디렉터리 재정의

## 구성 옵션

| 매개변수 | 기본값 | 설명 |
|-----------|---------|-------------|
| `--dataset_file` | (필수) | JSONL 데이터 세트 경로 |
| `--batch_size` | (필수) | 배치당 프롬프트 수 |
| `--run_name` | (필수) | 이 실행의 이름(출력 디렉터리와 체크포인트에 사용) |
| `--distribution` | `"default"` | 샘플링할 도구 세트 분포 |
| `--model` | `claude-sonnet-4.6` | 사용할 모델 |
| `--base_url` | `https://openrouter.ai/api/v1` | API 기본 URL |
| `--api_key` | (환경 변수) | 모델 API 키 |
| `--max_turns` | `10` | 프롬프트당 최대 도구 호출 반복 횟수 |
| `--num_workers` | `4` | 병렬 워커 프로세스 수 |
| `--resume` | `false` | 체크포인트에서 재개 |
| `--verbose` | `false` | 자세한 로깅 활성화 |
| `--max_samples` | all | 데이터 세트에서 처음 N개 샘플만 처리 |
| `--max_tokens` | 모델 기본값 | 모델 응답당 최대 토큰 수 |

### 공급자 라우팅(OpenRouter)

| 매개변수 | 설명 |
|-----------|-------------|
| `--providers_allowed` | 허용할 공급자를 쉼표로 구분(예: `"anthropic,openai"`) |
| `--providers_ignored` | 무시할 공급자를 쉼표로 구분(예: `"together,deepinfra"`) |
| `--providers_order` | 선호하는 공급자 순서(쉼표로 구분) |
| `--provider_sort` | `"price"`, `"throughput"` 또는 `"latency"` 기준으로 정렬 |

### 추론 제어

| 매개변수 | 설명 |
|-----------|-------------|
| `--reasoning_effort` | 추론 수준: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `ultra` |
| `--reasoning_disabled` | 추론/사고 토큰을 완전히 비활성화 |

### 고급 옵션

| 매개변수 | 설명 |
|-----------|-------------|
| `--ephemeral_system_prompt` | 실행 중 사용하지만 궤적에는 저장하지 않는 시스템 프롬프트 |
| `--log_prefix_chars` | 로그 미리보기에 표시할 문자 수(기본값: 100) |
| `--prefill_messages_file` | 퓨샷 프라이밍을 위한 사전 입력 메시지가 담긴 JSON 파일 경로 |

## 도구 세트 분포

각 프롬프트에는 **분포**에서 무작위로 샘플링된 도구 세트가 할당됩니다. 이를 통해 학습 데이터가 다양한 도구 조합을 포함하도록 합니다. 사용 가능한 모든 분포를 보려면 `--list_distributions`를 사용하세요.

현재 구현에서 분포는 **각 개별 도구 세트**에 확률을 할당합니다. 샘플러는 각 도구 세트를 독립적으로 선택한 다음, 최소 하나의 도구 세트가 활성화되도록 보장합니다. 이는 미리 작성된 조합 표를 사용하는 방식과 다릅니다.

## 출력 형식

모든 출력은 `data/<run_name>/`에 저장됩니다.

```text
data/my_run/
├── trajectories.jsonl    # Combined final output (all batches merged)
├── batch_0.jsonl         # Individual batch results
├── batch_1.jsonl
├── ...
├── checkpoint.json       # Resume checkpoint
└── statistics.json       # Aggregate tool usage stats
```

### 궤적 형식

`trajectories.jsonl`의 각 줄은 JSON 객체입니다.

```json
{
  "prompt_index": 42,
  "conversations": [
    {"from": "human", "value": "Write a function..."},
    {"from": "gpt", "value": "I'll create that function...",
     "tool_calls": [...]},
    {"from": "tool", "value": "..."},
    {"from": "gpt", "value": "Here's the completed function..."}
  ],
  "metadata": {
    "batch_num": 2,
    "timestamp": "2026-01-15T10:30:00",
    "model": "anthropic/claude-sonnet-4.6"
  },
  "completed": true,
  "partial": false,
  "api_calls": 3,
  "toolsets_used": ["terminal", "file"],
  "tool_stats": {
    "terminal": {"count": 2, "success": 2, "failure": 0},
    "read_file": {"count": 1, "success": 1, "failure": 0}
  },
  "tool_error_counts": {
    "terminal": 0,
    "read_file": 0
  }
}
```

`conversations` 필드는 `from` 및 `value` 필드를 사용하는 ShareGPT 유사 형식입니다. 도구 통계는 가능한 모든 도구를 기본값 0과 함께 포함하도록 정규화되어 HuggingFace 데이터 세트와 호환되는 일관된 스키마를 보장합니다.

## 체크포인트

배치 러너에는 내결함성을 위한 강력한 체크포인트 기능이 있습니다.

- **체크포인트 파일:** 각 배치가 완료될 때 저장되며, 완료된 프롬프트 인덱스를 추적합니다.
- **콘텐츠 기반 재개:** `--resume` 시 러너는 기존 배치 파일을 스캔하고 실제 텍스트 콘텐츠로 완료된 프롬프트를 일치시킵니다(단순히 인덱스만 사용하지 않음). 따라서 데이터 세트 순서가 바뀌어도 복구할 수 있습니다.
- **실패한 프롬프트:** 성공적으로 완료된 프롬프트만 완료된 것으로 표시하며, 실패한 프롬프트는 재개 시 다시 시도합니다.
- **배치 병합:** 완료되면 이전 실행의 파일을 포함한 모든 배치 파일을 하나의 `trajectories.jsonl`로 병합합니다.

### 재개 방식

1. 모든 `batch_*.jsonl` 파일에서 완료된 프롬프트를 스캔합니다(콘텐츠 일치 기준).
2. 이미 완료된 프롬프트를 제외하도록 데이터 세트를 필터링합니다.
3. 남은 프롬프트를 다시 배치로 나눕니다.
4. 남은 프롬프트만 처리합니다.
5. 모든 배치 파일(이전 파일 + 새 파일)을 최종 출력으로 병합합니다.

## 품질 필터링

배치 러너는 자동 품질 필터링을 적용합니다.

- **추론 없음 필터:** 추론이 포함된 어시스턴트 턴이 하나도 없는 샘플(`<REASONING_SCRATCHPAD>` 또는 네이티브 사고 토큰이 없음)을 폐기합니다.
- **손상된 항목 필터:** 가상의 도구 이름(유효한 도구 목록에 없는 이름)이 포함된 항목을 최종 병합 중 필터링합니다.
- **추론 통계:** 전체 실행에서 추론이 있거나 없는 턴의 비율을 추적합니다.

## 통계

완료 후 러너는 종합 통계를 출력합니다.

- **도구 사용:** 도구별 호출 횟수와 성공/실패율
- **추론 포함률:** 추론이 포함된 어시스턴트 턴의 비율
- **폐기된 샘플:** 추론 부족으로 필터링된 샘플 수
- **소요 시간:** 전체 처리 시간

통계는 프로그래밍 방식으로 분석할 수 있도록 `statistics.json`에도 저장됩니다.

## 사용 사례

### 학습 데이터 생성

파인튜닝을 위한 다양한 도구 사용 궤적을 생성합니다.

```bash
python batch_runner.py \
    --dataset_file=data/coding_prompts.jsonl \
    --batch_size=20 \
    --run_name=coding_v1 \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=8 \
    --distribution=default \
    --max_turns=15
```

### 모델 평가

표준화된 프롬프트를 통해 모델이 도구를 얼마나 잘 사용하는지 평가합니다.

```bash
python batch_runner.py \
    --dataset_file=data/eval_suite.jsonl \
    --batch_size=10 \
    --run_name=eval_gpt4 \
    --model=openai/gpt-4o \
    --num_workers=4 \
    --max_turns=10
```

### 프롬프트별 컨테이너 이미지

특정 환경이 필요한 벤치마크의 경우 각 프롬프트에 자체 컨테이너 이미지를 지정할 수 있습니다.

```jsonl
{"prompt": "Install numpy and compute eigenvalues of a 3x3 matrix", "image": "python:3.11-slim"}
{"prompt": "Compile this Rust program and run it", "image": "rust:1.75"}
{"prompt": "Set up a Node.js Express server", "image": "node:20-alpine", "cwd": "/app"}
```

배치 러너는 각 프롬프트를 실행하기 전에 Docker 이미지에 접근할 수 있는지 확인합니다.

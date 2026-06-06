---
sidebar_position: 12
title: "일괄 처리"
description: "대규모 에이전트 궤적 생성 — 병렬 처리, 체크포인트 및 도구 세트 분포"
---

# 일괄 처리 (Batch Processing)

일괄 처리 기능을 사용하면 수백 또는 수천 개의 프롬프트에서 Hermes 에이전트를 병렬로 실행하여 구조화된 궤적(trajectory) 데이터를 생성할 수 있습니다. 이는 주로 미세 조정(fine-tuning)이나 평가에 사용할 수 있는 도구 사용 통계가 포함된 ShareGPT 형식의 궤적을 생성하는 **훈련 데이터 생성**에 사용됩니다.

## 개요

일괄 처리 실행기(`batch_runner.py`)는 프롬프트의 JSONL 데이터 세트를 처리하여, 각각의 프롬프트를 도구 액세스가 가능한 전체 에이전트 세션을 통해 실행합니다. 각 프롬프트는 고유한 격리된 환경을 갖게 됩니다. 출력은 전체 대화 기록, 도구 호출 통계 및 추론(reasoning) 범위 메트릭이 포함된 구조화된 궤적 데이터입니다.

## 빠른 시작

```bash
# 기본 일괄 처리 실행
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --model=anthropic/claude-sonnet-4.6 \
    --num_workers=4

# 중단된 실행 재개
python batch_runner.py \
    --dataset_file=data/prompts.jsonl \
    --batch_size=10 \
    --run_name=my_first_run \
    --resume

# 사용 가능한 도구 세트 분포 나열
python batch_runner.py --list_distributions
```

:::tip 대규모 실행 시 예측 가능한 비용
일괄 처리 실행은 다수의 동시 에이전트 세션을 회전시키며, 각 세션은 모델 호출과 도구 호출을 수행합니다. 유료 [Nous Portal](/user-guide/features/tool-gateway) 구독을 사용하면 하나의 청구서에 모델 액세스와 웹 검색, 이미지 생성, TTS 및 클라우드 브라우저가 포함되므로 5개 공급업체 계정에 걸쳐 속도 제한을 저글링할 필요 없이 안정적인 궤적당 비용을 원할 때 유용합니다. `hermes setup --portal`로 설정한 다음 `--model`이 Nous 모델을 가리키도록 설정하세요.
:::

## 데이터 세트 형식

입력 데이터 세트는 JSONL 파일(줄당 하나의 JSON 객체)입니다. 각 항목에는 `prompt` 필드가 있어야 합니다:

```jsonl
{"prompt": "Write a Python function that finds the longest palindromic substring"}
{"prompt": "Create a REST API endpoint for user authentication using Flask"}
{"prompt": "Debug this error: TypeError: cannot unpack non-iterable NoneType object"}
```

항목에는 다음이 선택적으로 포함될 수 있습니다:
- `image` 또는 `docker_image`: 이 프롬프트의 샌드박스에 사용할 컨테이너 이미지 (Docker, Modal 및 Singularity 백엔드와 함께 작동)
- `cwd`: 작업의 터미널 세션에 대한 작업 디렉토리 재정의(override)

## 구성 옵션

| 매개변수 | 기본값 | 설명 |
|-----------|---------|-------------|
| `--dataset_file` | (필수) | JSONL 데이터 세트의 경로 |
| `--batch_size` | (필수) | 배치당 프롬프트 수 |
| `--run_name` | (필수) | 이 실행의 이름 (출력 디렉토리 및 체크포인트에 사용됨) |
| `--distribution` | `"default"` | 샘플링할 도구 세트 분포 |
| `--model` | `claude-sonnet-4.6` | 사용할 모델 |
| `--base_url` | `https://openrouter.ai/api/v1` | API 기본 URL |
| `--api_key` | (환경 변수) | 모델에 대한 API 키 |
| `--max_turns` | `10` | 프롬프트당 최대 도구 호출 반복 횟수 |
| `--num_workers` | `4` | 병렬 워커 프로세스 수 |
| `--resume` | `false` | 체크포인트에서 재개 |
| `--verbose` | `false` | 상세 로깅 활성화 |
| `--max_samples` | 전체 | 데이터 세트에서 처음 N개의 샘플만 처리 |
| `--max_tokens` | 모델 기본값 | 모델 응답당 최대 토큰 수 |

### 제공자 라우팅 (OpenRouter)

| 매개변수 | 설명 |
|-----------|-------------|
| `--providers_allowed` | 쉼표로 구분된 허용할 제공자 (예: `"anthropic,openai"`) |
| `--providers_ignored` | 쉼표로 구분된 무시할 제공자 (예: `"together,deepinfra"`) |
| `--providers_order` | 쉼표로 구분된 선호하는 제공자 순서 |
| `--provider_sort` | `"price"`, `"throughput"`, 또는 `"latency"` 로 정렬 |

### 추론 제어 (Reasoning Control)

| 매개변수 | 설명 |
|-----------|-------------|
| `--reasoning_effort` | 노력 수준: `none`, `minimal`, `low`, `medium`, `high`, `xhigh` |
| `--reasoning_disabled` | 추론/생각 토큰을 완전히 비활성화 |

### 고급 옵션

| 매개변수 | 설명 |
|-----------|-------------|
| `--ephemeral_system_prompt` | 실행 중에는 사용되지만 궤적에 저장되지 **않는** 시스템 프롬프트 |
| `--log_prefix_chars` | 로그 미리보기에 표시할 문자 수 (기본값: 100) |
| `--prefill_messages_file` | 퓨샷(few-shot) 프라이밍을 위한 프리필(prefill) 메시지가 포함된 JSON 파일 경로 |

## 도구 세트 분포 (Toolset Distributions)

각 프롬프트는 **분포(distribution)**에서 무작위로 샘플링된 도구 세트 그룹을 얻습니다. 이를 통해 훈련 데이터가 다양한 도구 조합을 커버하도록 보장합니다. `--list_distributions`를 사용하여 사용 가능한 모든 분포를 확인하세요.

현재 구현에서 분포는 **각 개별 도구 세트**에 확률을 할당합니다. 샘플러는 각 도구 세트를 독립적으로 전환한 다음, 적어도 하나의 도구 세트가 활성화되도록 보장합니다. 이는 사전 구축된 조합의 직접 작성된 테이블과는 다릅니다.

## 출력 형식

모든 출력은 `data/<run_name>/`로 이동합니다:

```text
data/my_run/
├── trajectories.jsonl    # 결합된 최종 출력 (모든 배치가 병합됨)
├── batch_0.jsonl         # 개별 배치 결과
├── batch_1.jsonl
├── ...
├── checkpoint.json       # 재개(Resume) 체크포인트
└── statistics.json       # 통합 도구 사용 통계
```

### 궤적 형식

`trajectories.jsonl`의 각 줄은 JSON 객체입니다:

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

`conversations` 필드는 `from` 및 `value` 필드가 있는 ShareGPT와 유사한 형식을 사용합니다. 도구 통계는 0의 기본값이 있는 가능한 모든 도구를 포함하도록 정규화되어 HuggingFace 데이터 세트 호환성을 위한 항목 전반에 걸쳐 일관된 스키마를 보장합니다.

## 체크포인트 (Checkpointing)

일괄 처리 실행기에는 내결함성(fault tolerance)을 위한 강력한 체크포인트 기능이 있습니다:

- **체크포인트 파일:** 각 배치가 완료된 후 저장되며 어떤 프롬프트 인덱스가 완료되었는지 추적합니다.
- **콘텐츠 기반 재개:** `--resume` 사용 시 실행기는 기존 배치 파일을 스캔하고 (인덱스만이 아니라) 실제 텍스트 콘텐츠로 완료된 프롬프트를 일치시키므로 데이터 세트 순서가 변경되더라도 복구가 가능합니다.
- **실패한 프롬프트:** 성공적으로 완료된 프롬프트만 완료로 표시됩니다 — 실패한 프롬프트는 재개 시 다시 시도됩니다.
- **배치 병합:** 완료 시 (이전 실행을 포함한) 모든 배치 파일이 단일 `trajectories.jsonl`로 병합됩니다.

### 재개(Resume) 작동 방식

1. 내용 일치를 통해 완료된 프롬프트에 대해 모든 `batch_*.jsonl` 파일을 스캔합니다.
2. 이미 완료된 프롬프트를 제외하도록 데이터 세트를 필터링합니다.
3. 남은 프롬프트를 다시 배치 처리합니다.
4. 남은 프롬프트만 처리합니다.
5. 모든 배치 파일(이전 + 새)을 최종 출력으로 병합합니다.

## 품질 필터링

일괄 처리 실행기는 자동 품질 필터링을 적용합니다:

- **무추론(No-reasoning) 필터:** 어시스턴트 턴 중 추론이 포함된 턴이 하나도 없는 샘플(`<REASONING_SCRATCHPAD>` 또는 기본 생각 토큰이 없는 경우)은 삭제됩니다.
- **손상된 항목 필터:** 환각적인 도구 이름(유효한 도구 목록에 없음)이 있는 항목은 최종 병합 중에 필터링됩니다.
- **추론 통계:** 전체 실행에 걸쳐 추론이 있는/없는 어시스턴트 턴의 비율을 추적합니다.

## 통계

완료 후 실행기는 포괄적인 통계를 인쇄합니다:

- **도구 사용량:** 호출 횟수, 도구별 성공/실패율
- **추론 범위:** 추론이 포함된 어시스턴트 턴의 비율
- **삭제된 샘플:** 추론 부족으로 인해 필터링된 샘플 수
- **소요 시간:** 총 처리 시간

통계는 프로그래밍 방식의 분석을 위해 `statistics.json`에도 저장됩니다.

## 사용 사례

### 훈련 데이터 생성

미세 조정을 위한 다양한 도구 사용 궤적을 생성합니다:

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

표준화된 프롬프트 전체에서 모델이 도구를 얼마나 잘 사용하는지 평가합니다:

```bash
python batch_runner.py \
    --dataset_file=data/eval_suite.jsonl \
    --batch_size=10 \
    --run_name=eval_gpt4 \
    --model=openai/gpt-4o \
    --num_workers=4 \
    --max_turns=10
```

### 프롬프트당 컨테이너 이미지

특정 환경이 필요한 벤치마크의 경우 각 프롬프트에 자체 컨테이너 이미지를 지정할 수 있습니다:

```jsonl
{"prompt": "Install numpy and compute eigenvalues of a 3x3 matrix", "image": "python:3.11-slim"}
{"prompt": "Compile this Rust program and run it", "image": "rust:1.75"}
{"prompt": "Set up a Node.js Express server", "image": "node:20-alpine", "cwd": "/app"}
```

일괄 처리 실행기는 각 프롬프트를 실행하기 전에 Docker 이미지에 접근할 수 있는지 확인합니다.

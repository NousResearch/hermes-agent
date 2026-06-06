# 컨텍스트 압축 및 캐싱 (Context Compression and Caching)

Hermes 에이전트는 긴 대화에 걸쳐 컨텍스트 윈도우 사용을 효율적으로 관리하기 위해 이중 압축 시스템과 Anthropic 프롬프트 캐싱을 사용합니다.

소스 파일: `agent/context_engine.py` (ABC), `agent/context_compressor.py` (기본 엔진), `agent/prompt_caching.py`, `gateway/run.py` (세션 위생 검사), `run_agent.py` (`_compress_context` 검색)

## 플러그형 컨텍스트 엔진 (Pluggable Context Engine)

컨텍스트 관리는 `ContextEngine` 추상 기본 클래스(ABC) (`agent/context_engine.py`)를 기반으로 합니다. 내장된 `ContextCompressor`가 기본 구현체지만, 플러그인은 대안적인 엔진(예: 무손실 컨텍스트 관리)으로 이를 교체할 수 있습니다.

```yaml
context:
  engine: "compressor"    # 기본값 — 내장된 손실 요약 (lossy summarization)
  engine: "lcm"           # 예제 — 무손실 컨텍스트를 제공하는 플러그인
```

엔진의 책임은 다음과 같습니다:
- 압축(compaction) 시점을 결정 (`should_compress()`)
- 압축 수행 (`compress()`)
- 선택적으로 에이전트가 호출할 수 있는 도구 노출 (예: `lcm_grep`)
- API 응답으로부터의 토큰 사용량 추적

선택은 `config.yaml`의 `context.engine`에 의해 구동됩니다. 해석 순서:
1. `plugins/context_engine/<name>/` 디렉토리 확인
2. 일반 플러그인 시스템 확인 (`register_context_engine()`)
3. 내장 `ContextCompressor`로 폴백

플러그인 엔진은 **절대 자동 활성화되지 않습니다**. 사용자가 명시적으로 `context.engine`을 플러그인 이름으로 설정해야 합니다. 기본값 `"compressor"`는 항상 내장 엔진을 사용합니다.

`hermes plugins` → Provider Plugins → Context Engine을 통해 구성하거나, `config.yaml`을 직접 편집하세요.

컨텍스트 엔진 플러그인 작성에 대해서는 [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin)을 참고하세요.

## 이중 압축 시스템 (Dual Compression System)

Hermes는 독립적으로 동작하는 두 가지 분리된 압축 레이어를 가지고 있습니다:

```
                      ┌──────────────────────────┐
  수신 메시지           │   게이트웨이 세션 위생    │  컨텍스트의 85%에서 실행
  ─────────────────►  │   (에이전트 전, 대략적)   │  큰 세션을 위한 안전망
                      └─────────────┬────────────┘
                                    │
                                    ▼
                      ┌──────────────────────────┐
                      │   에이전트 ContextCompressor│ 컨텍스트의 50%에서 실행 (기본)
                      │   (루프 내, 실제 토큰)      │ 정상적인 컨텍스트 관리
                      └──────────────────────────┘
```

### 1. 게이트웨이 세션 위생 (Gateway Session Hygiene - 85% 임계값)

`gateway/run.py`에 위치해 있습니다 (`Session hygiene: auto-compress`로 검색). 이것은 에이전트가 메시지를 처리하기 전에 실행되는 **안전망(safety net)**입니다. 대화 턴 간에 세션이 너무 커질 경우(예: Telegram/Discord에서의 밤사이 축적 등) API 실패를 방지합니다.

- **임계값**: 모델 컨텍스트 길이의 85%로 고정
- **토큰 소스**: 마지막 턴에서 실제 API가 보고한 토큰 선호; 글자(character) 기반의 대략적 추정치(`estimate_messages_tokens_rough`)로 폴백
- **실행 조건**: `len(history) >= 4`이고 압축이 활성화되어 있을 때만
- **목적**: 에이전트의 자체 압축기를 벗어난 세션을 잡기 위함

게이트웨이 위생 임계값은 의도적으로 에이전트의 압축기보다 높게 설정되어 있습니다.
(에이전트와 동일하게) 50%로 설정하면 긴 게이트웨이 세션에서 매 턴마다 시기상조로 압축이 일어나는 문제를 일으켰기 때문입니다.

### 2. 에이전트 ContextCompressor (50% 임계값, 구성 가능)

`agent/context_compressor.py`에 위치해 있습니다. 이는 정확한 API 보고 토큰 개수에 접근하여 에이전트의 도구 루프 내에서 실행되는 **기본 압축 시스템**입니다.

## 구성 (Configuration)

모든 압축 설정은 `config.yaml`의 `compression` 키 아래에서 읽힙니다:

```yaml
compression:
  enabled: true              # 압축 활성화/비활성화 (기본값: true)
  threshold: 0.50            # 컨텍스트 윈도우의 비율 (기본값: 0.50 = 50%)
  target_ratio: 0.20         # 보호할 꼬리(tail)를 위한 임계값 비율 (기본값: 0.20)
  protect_last_n: 20         # 보호할 최근 메시지의 최소 개수 (기본값: 20)

# 보조(auxiliary) 설정 하에서 요약 모델/프로바이더 구성:
auxiliary:
  compression:
    model: null              # 요약을 위한 모델 오버라이드 (기본값: 자동 감지)
    provider: auto           # 프로바이더: "auto", "openrouter", "nous", "main", 등
    base_url: null           # 사용자 지정 OpenAI 호환 엔드포인트
```

### 매개변수 상세 정보

| 매개변수 | 기본값 | 범위 | 설명 |
|-----------|---------|-------|-------------|
| `threshold` | `0.50` | 0.0-1.0 | 프롬프트 토큰 ≥ `threshold × context_length`일 때 압축 트리거 |
| `target_ratio` | `0.20` | 0.10-0.80 | 꼬리(tail) 보호 토큰 예산 제어: `threshold_tokens × target_ratio` |
| `protect_last_n` | `20` | ≥1 | 항상 보존되는 최근 메시지의 최소 개수 |
| `protect_first_n` | `3` | (하드코딩됨) | 시스템 프롬프트 + 첫 대화(exchange)는 항상 보존됨 |

### 계산 값 (기본값인 200K 컨텍스트 모델의 경우)

```
context_length       = 200,000
threshold_tokens     = 200,000 × 0.50 = 100,000
tail_token_budget    = 100,000 × 0.20 = 20,000
max_summary_tokens   = min(200,000 × 0.05, 12,000) = 10,000
```

:::note 임계값은 MAIN 모델의 컨텍스트 윈도우에서 파생됩니다
`threshold_tokens`는 항상 `threshold × context_length`이며, 여기서 `context_length`는 보조/요약 모델이 아닌 **메인 에이전트 모델의** 컨텍스트 윈도우입니다. 262,144 토큰 모델에서 기본값인 `0.50`을 사용할 때 임계값은 `262,144 × 0.50 = 131,072`입니다. 이 수치가 일반적인 "128K 컨텍스트"와 가까운 것은 단순한 퍼센트 상의 우연이며, 보조 모델의 윈도우가 트리거라는 신호가 아닙니다. 보조 모델의 컨텍스트 윈도우는 별개의 문제입니다. — 요약 생성 가능성에 영향을 주며 압축 시점과는 무관한 "요약 모델 컨텍스트 길이(Summary model context length)" 경고는 아래를 참조하세요.
:::

## 압축 알고리즘 (Compression Algorithm)

`ContextCompressor.compress()` 메서드는 4단계 알고리즘을 따릅니다:

### 1단계: 오래된 도구 결과 잘라내기 (저렴함, LLM 호출 없음)

보호되는 꼬리(tail) 외부에 있는 오래된 도구 결과(200자 초과)는 다음으로 교체됩니다:
```
[Old tool output cleared to save context space]
```

이것은 장황한 도구 출력(파일 내용, 터미널 출력, 검색 결과)에서 상당한 토큰을 절약하는 저렴한 사전 패스(pre-pass)입니다.

### 2단계: 경계 결정 (Determine Boundaries)

```
┌─────────────────────────────────────────────────────────────┐
│  메시지 목록                                                  │
│                                                             │
│  [0..2]  ← protect_first_n (시스템 + 첫 대화)                 │
│  [3..N]  ← 중간 턴 → 요약됨(SUMMARIZED)                       │
│  [N..end] ← 꼬리 (토큰 예산 또는 protect_last_n에 따름)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

꼬리 보호는 **토큰 예산 기반**입니다: 토큰 예산이 소진될 때까지 끝에서부터 거꾸로 걸어가며 누적합니다. 예산으로 보호할 메시지가 적은 경우 고정된 `protect_last_n` 개수로 폴백합니다.

경계는 `tool_call`/`tool_result` 그룹이 쪼개지지 않도록 정렬됩니다.
`_align_boundary_backward()` 메서드는 그룹을 온전하게 유지하기 위해 연속된 도구 결과를 지나 상위 어시스턴트 메시지를 찾습니다.

### 3단계: 구조화된 요약 생성

:::warning 요약 모델 컨텍스트 길이 (Summary model context length)
요약 모델은 **최소한 메인 에이전트 모델만큼 큰** 컨텍스트 윈도우를 가져야 합니다. 전체 중간 섹션은 단일 `call_llm(task="compression")` 호출로 요약 모델에 전송됩니다. 요약 모델의 컨텍스트가 더 작다면 API는 컨텍스트 길이 오류를 반환합니다 — `_generate_summary()`가 이를 잡아서 경고를 로깅하고 `None`을 반환합니다. 압축기는 대화 컨텍스트를 조용히 잃으며 **요약 없이** 중간 턴을 버리게 됩니다. 이것이 압축 품질 저하의 가장 일반적인 원인입니다.
:::

중간 턴은 구조화된 템플릿과 함께 보조 LLM을 사용하여 요약됩니다:

```
## Goal
[사용자가 성취하려는 목표]

## Constraints & Preferences
[사용자 선호도, 코딩 스타일, 제약 조건, 중요한 결정 사항]

## Progress
### Done
[완료된 작업 — 특정 파일 경로, 실행한 명령어, 결과]
### In Progress
[현재 진행 중인 작업]
### Blocked
[직면한 문제나 차단 요소(Blockers)]

## Key Decisions
[중요한 기술적 결정과 그 이유]

## Relevant Files
[읽었거나 수정한, 또는 생성한 파일 — 각각에 대한 짧은 메모와 함께]

## Next Steps
[다음에 수행해야 할 작업]

## Critical Context
[특정 값, 오류 메시지, 구성 세부 정보]
```

요약 예산은 압축되는 내용의 양에 비례하여 조정됩니다:
- 공식: `content_tokens × 0.20` (`_SUMMARY_RATIO` 상수)
- 최소: 2,000 토큰
- 최대: `min(context_length × 0.05, 12,000)` 토큰

### 4단계: 압축된 메시지 조립

압축된 메시지 목록은 다음과 같습니다:
1. 머리(Head) 메시지 (첫 번째 압축 시 시스템 프롬프트에 메모 추가됨)
2. 요약 메시지 (동일한 역할이 연속되는 것을 피하기 위해 선택된 역할)
3. 꼬리(Tail) 메시지 (수정되지 않음)

고립된(Orphaned) `tool_call`/`tool_result` 쌍은 `_sanitize_tool_pairs()`에 의해 정리됩니다:
- 제거된 호출을 참조하는 도구 결과 → 제거됨
- 결과가 제거된 도구 호출 → 스터브(stub) 결과가 주입됨

### 반복적인 재압축 (Iterative Re-compression)

후속 압축에서는 이전 요약본이 LLM에 전달되며, 처음부터 다시 요약하는 대신 **업데이트**하라는 지시를 받습니다. 이를 통해 여러 번의 압축에도 정보가 보존됩니다 — 항목들이 "In Progress"에서 "Done"으로 이동하고, 새로운 진행 상황이 추가되며, 불필요한 정보는 삭제됩니다.

이 목적을 위해 압축기 인스턴스의 `_previous_summary` 필드에 마지막 요약 텍스트를 저장합니다.

## 전/후 예제

### 압축 전 (메시지 45개, 토큰 약 95K)

```
[0] system:    "당신은 유용한 어시스턴트입니다..." (시스템 프롬프트)
[1] user:      "FastAPI 프로젝트 설정을 도와줘"
[2] assistant: <tool_call> terminal: mkdir project </tool_call>
[3] tool:      "directory created"
[4] assistant: <tool_call> write_file: main.py </tool_call>
[5] tool:      "file written (2.3KB)"
    ... 파일 수정, 테스트, 디버깅 등 30개 이상의 턴 ...
[38] assistant: <tool_call> terminal: pytest </tool_call>
[39] tool:      "8 passed, 2 failed\n..."  (5KB 출력)
[40] user:      "실패하는 테스트를 고쳐줘"
[41] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[42] tool:      "import pytest\n..."  (3KB)
[43] assistant: "테스트 픽스처(fixtures)에 문제가 보이네요..."
[44] user:      "좋아, 에러 핸들링도 추가해 줘"
```

### 압축 후 (메시지 25개, 토큰 약 45K)

```
[0] system:    "당신은 유용한 어시스턴트입니다...
               [참고: 이전 대화 턴의 일부가 압축되었습니다...]"
[1] user:      "FastAPI 프로젝트 설정을 도와줘"
[2] assistant: "[CONTEXT COMPACTION] 이전 턴들이 압축되었습니다...

               ## Goal
               테스트와 에러 핸들링을 포함한 FastAPI 프로젝트 설정

               ## Progress
               ### Done
               - 프로젝트 구조 생성: main.py, tests/, requirements.txt
               - main.py에 5개의 API 엔드포인트 구현
               - tests/test_api.py에 10개의 테스트 케이스 작성
               - 10개 중 8개 테스트 통과

               ### In Progress
               - 실패하는 2개의 테스트 수정 (test_create_user, test_delete_user)

               ## Relevant Files
               - main.py — 5개의 엔드포인트가 있는 FastAPI 앱
               - tests/test_api.py — 10개의 테스트 케이스
               - requirements.txt — fastapi, pytest, httpx

               ## Next Steps
               - 실패하는 테스트 픽스처(fixtures) 수정
               - 에러 핸들링 추가"
[3] user:      "실패하는 테스트를 고쳐줘"
[4] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[5] tool:      "import pytest\n..."
[6] assistant: "테스트 픽스처(fixtures)에 문제가 보이네요..."
[7] user:      "좋아, 에러 핸들링도 추가해 줘"
```

## 프롬프트 캐싱 (Anthropic)

소스: `agent/prompt_caching.py`

대화형 프롬프트를 캐싱하여 다중 턴 대화에서 입력 토큰 비용을 최대 ~75% 절감합니다. Anthropic의 `cache_control` 브레이크포인트를 사용합니다.

### 전략: system_and_3

Anthropic은 요청당 최대 4개의 `cache_control` 브레이크포인트를 허용합니다. Hermes는 "system_and_3" 전략을 사용합니다:

```
브레이크포인트 1: 시스템 프롬프트           (모든 턴에 걸쳐 안정적임)
브레이크포인트 2: 끝에서 3번째 비시스템(non-system) 메시지  ─┐
브레이크포인트 3: 끝에서 2번째 비시스템 메시지               ├─ 롤링 윈도우 (Rolling window)
브레이크포인트 4: 마지막 비시스템 메시지                    ─┘
```

### 작동 방식

`apply_anthropic_cache_control()`은 메시지를 깊은 복사(deep-copy)하고 `cache_control` 마커를 주입합니다:

```python
# 캐시 마커 형식
marker = {"type": "ephemeral"}
# 또는 1시간 TTL의 경우:
marker = {"type": "ephemeral", "ttl": "1h"}
```

마커는 내용 유형(content type)에 따라 다르게 적용됩니다:

| 콘텐츠 유형 | 마커 위치 |
|-------------|-------------------|
| 문자열 내용 | `[{"type": "text", "text": ..., "cache_control": ...}]` 로 변환 |
| 목록 내용 | 마지막 요소의 딕셔너리에 추가됨 |
| 없음/비어있음 | `msg["cache_control"]`로 추가됨 |
| 도구 메시지 | `msg["cache_control"]`로 추가됨 (네이티브 Anthropic 전용) |

### 캐시 인식 설계 패턴

1. **안정적인 시스템 프롬프트**: 시스템 프롬프트는 첫 번째 브레이크포인트이며 모든 턴에 걸쳐 캐시됩니다. 대화 도중에 이를 수정하는 것을 피하세요 (압축은 첫 번째 수행 시에만 참고 메모를 추가합니다).

2. **메시지 순서 중요**: 캐시 적중(hit)은 접두사(prefix) 일치가 필요합니다. 중간에 메시지를 추가하거나 제거하면 그 뒤의 모든 것에 대한 캐시가 무효화됩니다.

3. **압축과 캐시 간의 상호작용**: 압축 후에는 압축된 영역에 대해 캐시가 무효화되지만 시스템 프롬프트 캐시는 살아남습니다. 롤링 3메시지 윈도우가 1-2턴 내에 캐싱을 다시 설정합니다.

4. **TTL 선택**: 기본값은 `5m`(5분)입니다. 사용자가 턴 사이에 휴식을 취하는 장기 세션의 경우 `1h`를 사용하세요.

### 프롬프트 캐싱 활성화

다음의 경우 프롬프트 캐싱이 자동으로 활성화됩니다:
- 모델이 Anthropic Claude 모델인 경우 (모델 이름으로 감지됨)
- 프로바이더가 `cache_control`을 지원하는 경우 (네이티브 Anthropic API 또는 OpenRouter)

```yaml
# config.yaml — TTL은 구성 가능합니다 ("5m" 또는 "1h"이어야 함)
prompt_caching:
  cache_ttl: "5m"
```

CLI는 시작 시 캐싱 상태를 표시합니다:
```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```

## 컨텍스트 압력 경고 (Context Pressure Warnings)

중간 컨텍스트 압력 경고가 제거되었습니다 (이와 관련해 "중간 압력 경고가 없습니다 — 이는 모델들이 복잡한 작업을 너무 일찍 '포기'하게 만들었습니다"라고 언급하는 `run_agent.py`의 iteration-budget 블록을 참고하세요). 압축은 별도의 사전 경고 단계 없이 프롬프트 토큰이 구성된 `compression.threshold` (기본값 50%)에 도달할 때 시작됩니다. 게이트웨이 세션 위생은 모델 컨텍스트 윈도우의 85% 지점에서 두 번째 안전망으로 실행됩니다.

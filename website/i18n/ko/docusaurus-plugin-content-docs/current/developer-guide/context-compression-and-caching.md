# 컨텍스트 압축 및 캐싱

Hermes Agent는 긴 대화에서 컨텍스트 창 사용량을 효율적으로 관리하기 위해 이중 압축 시스템과 Anthropic 프롬프트 캐싱을 사용합니다.

소스 파일: `agent/context_engine.py` (ABC), `agent/context_compressor.py` (기본 엔진),
`agent/prompt_caching.py`, `gateway/run.py` (세션 관리), `run_agent.py` (`_compress_context` 검색)


## 플러그형 컨텍스트 엔진

컨텍스트 관리는 `ContextEngine` ABC (`agent/context_engine.py`)를 기반으로 합니다. 기본 제공되는 `ContextCompressor`가 기본 구현이지만, 플러그인이 다른 엔진(예: Lossless Context Management)으로 교체할 수 있습니다.

```yaml
context:
  engine: "compressor"    # default — built-in lossy summarization
  engine: "lcm"           # example — plugin providing lossless context
```

엔진은 다음을 담당합니다.
- 압축을 실행할 시점 결정(`should_compress()`)
- 압축 수행(`compress()`)
- 에이전트가 호출할 수 있는 도구를 선택적으로 제공(예: `lcm_grep`)
- API 응답에서 토큰 사용량 추적

선택은 `config.yaml`의 `context.engine` 설정으로 결정됩니다. 해석 순서는 다음과 같습니다.
1. `plugins/context_engine/<name>/` 디렉터리 확인
2. 일반 플러그인 시스템(`register_context_engine()`) 확인
3. 기본 제공 `ContextCompressor`로 대체

플러그인 엔진은 **자동으로 활성화되지 않습니다**. 사용자가 명시적으로 `context.engine`을 플러그인 이름으로 설정해야 합니다. 기본값인 `"compressor"`는 항상 기본 제공 엔진을 사용합니다.

`hermes plugins` → Provider Plugins → Context Engine에서 설정하거나 `config.yaml`을 직접 편집하세요.

컨텍스트 엔진 플러그인 빌드 방법은 [Context Engine Plugins](/developer-guide/context-engine-plugin)을 참조하세요.

## 이중 압축 시스템

Hermes에는 서로 독립적으로 동작하는 두 개의 압축 계층이 있습니다.

```
                     ┌──────────────────────────┐
  Incoming message   │   Gateway Session Hygiene │  Fires at 85% of context
  ─────────────────► │   (pre-agent, rough est.) │  Safety net for large sessions
                     └─────────────┬────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │   Agent ContextCompressor │  Fires at 50% of context (default)
                     │   (in-loop, real tokens)  │  Normal context management
                     └──────────────────────────┘
```

### 1. 게이트웨이 세션 관리(85% 임계값)

`gateway/run.py`에 있으며(`Session hygiene: auto-compress` 검색), 에이전트가 메시지를 처리하기 전에 실행되는 **안전망**입니다. 세션이 턴 사이에 너무 커져 API 오류가 발생하는 것을 방지합니다(예: Telegram/Discord에서 밤새 누적되는 경우).

- **임계값**: 모델 컨텍스트 길이의 85%로 고정
- **토큰 출처**: 최근 턴에서 API가 보고한 실제 토큰을 우선 사용하고, 없으면 대략적인 문자 기반 추정(`estimate_messages_tokens_rough`) 사용
- **실행 조건**: `len(history) >= 4`이고 압축이 활성화된 경우에만 실행
- **목적**: 에이전트 자체 압축기를 빠져나간 세션 포착

게이트웨이 관리 임계값은 의도적으로 에이전트 압축기보다 높습니다. 이를 50%(에이전트와 동일)로 설정하면 긴 게이트웨이 세션에서 매 턴마다 조기 압축이 발생했습니다.

### 2. 에이전트 ContextCompressor(50% 임계값, 설정 가능)

`agent/context_compressor.py`에 있습니다. 정확한 API 보고 토큰 수를 사용할 수 있는 에이전트의 도구 루프 내부에서 실행되는 **주요 압축 시스템**입니다.


## 설정

모든 압축 설정은 `compression` 키 아래의 `config.yaml`에서 읽습니다.

```yaml
compression:
  enabled: true              # Enable/disable compression (default: true)
  threshold: 0.50            # Fraction of context window (default: 0.50 = 50%)
  # model_thresholds:        # Per-model threshold overrides (substring match,
  #   "glm-5.2": 0.40        # longest key wins). See "Per-model threshold
  #   "claude-sonnet": 0.35  # overrides" below.
  target_ratio: 0.20         # How much of threshold to keep as tail (default: 0.20)
  protect_last_n: 20         # Minimum protected tail messages (default: 20)
  min_tail_user_messages: 1  # Real user messages guaranteed in the tail (default: 1)
  codex_gpt55_autoraise: true  # gpt-5.5 on Codex OAuth: raise trigger to 85% (default: true)
  codex_gpt55_autoraise_notice: true  # Show the one-time autoraise notice (default: true)
  codex_app_server_auto: native  # native|hermes|off for Codex app-server thread compaction
  codex_responses_native: false  # gpt-5.6 on direct OpenAI/Codex: server-side compaction (opt-in)
  codex_responses_compact_threshold: 200000  # Server-side compaction trigger (input tokens)
  in_place: true             # Compact on the same session id, no rotation (default: true)

# Summarization model/provider configured under auxiliary:
auxiliary:
  compression:
    model: null              # Override model for summaries (default: auto-detect)
    provider: auto           # Provider: "auto", "openrouter", "nous", "main", etc.
    base_url: null           # Custom OpenAI-compatible endpoint
```

### 매개변수 세부 정보

| 매개변수 | 기본값 | 범위 | 설명 |
|-----------|---------|-------|-------------|
| `threshold` | `0.50` | 0.0-1.0 | 프롬프트 토큰이 `threshold × context_length` 이상이면 압축 실행 |
| `model_thresholds` | `{}` | map | `threshold`의 모델별 재정의. 키는 모델 이름과 부분 문자열로 일치하며(가장 긴 일치 항목 우선), 작은 컨텍스트 하한도 계속 적용됨(아래 참조) |
| `target_ratio` | `0.20` | 0.10-0.80 | 테일 보호 토큰 예산 제어: `threshold_tokens × target_ratio` |
| `protect_last_n` | `20` | ≥1 | 항상 보존되는 최근 메시지의 최소 개수 |
| `min_tail_user_messages` | `1` | ≥1 | 압축되지 않은 테일에 보존되는 실제(실행 가능한) 사용자 메시지의 최소 개수. `1`은 기존의 마지막 사용자 앵커 하나이며(동작을 보존하는 기본값), 도구 출력이 많아 테일 토큰 예산을 채우더라도 마지막 실제 사용자 턴 3개를 그대로 유지하려면 `3` 등으로 높입니다. 빈 플랫폼 에코, 압축 핸드오프, 합성 연속 행은 N에 포함되지 않습니다. 이 보장이 테일 토큰 예산보다 우선하므로 앵커가 컷을 뒤로 당기면 테일이 예산을 초과할 수 있습니다 |
| `protect_first_n` | `3` | (하드코딩) | 시스템 프롬프트와 첫 교환을 항상 보존 |
| `idle_compact_after_seconds` | `0` | ≥0초 | 선택 사항: 세션이 지정된 유휴 시간 후 재개되면 처음부터 압축(0 = 비활성화). 컨텍스트가 `threshold × target_ratio` 이하이면 건너뛰며, 쿨다운/과도한 실행 방지/잠금 보호를 준수 |
| `codex_gpt55_autoraise` | `true` | bool | ChatGPT Codex OAuth 경로에서 gpt-5.5의 트리거를 85%로 올림(아래 참조). 전역 `threshold`를 유지하려면 `false`로 설정 |
| `codex_gpt55_autoraise_notice` | `true` | bool | Codex gpt-5.5 자동 상향 알림을 한 번 표시. 85% 자동 상향은 유지하되 배너를 숨기려면 `false`로 설정 |
| `codex_app_server_auto` | `native` | `native`, `hermes`, `off` | Codex app-server 세션의 스레드 압축 모드(아래 참조) |
| `codex_responses_native` | `false` | bool | Responses API에서 OpenAI의 서버 측 압축을 사용. 직접 OpenAI API 또는 ChatGPT Codex 구독의 gpt-5.6 계열 모델에서만 작동(아래 참조) |
| `codex_responses_compact_threshold` | `200000` | ≥1 토큰 | 입력 토큰 기준 서버 측 압축 트리거. 요청 시 로컬 압축 임계값보다 낮게 제한되어 서버가 먼저 압축함 |
| `in_place` | `true` | bool | 새 세션으로 교체하지 않고 동일한 세션 ID에서 압축(아래 참조) |

### 인플레이스 압축(안정적인 단일 세션 ID)

`compression.in_place: true`(기본값)이면 압축이 **동일한 세션 ID에서 실행 중인 메시지 목록을 다시 작성**합니다. 시스템 프롬프트를 재구성하고, 요약된 중간 부분을 교체하며, 압축 전 턴을 동일한 ID 아래 소프트 아카이브합니다(세션 저장소에서 `active=0, compacted=1`). 해당 데이터는 여전히 `session_search`로 검색하고 복구할 수 있으며 삭제되지 않습니다. `parent_session_id` 체인이나 `name #N` 재번호 매기기는 없습니다. 하나의 대화가 전체 수명 동안 하나의 영구 ID를 유지합니다. 이 방식으로 세션 교체와 관련된 버그 묶음(손실된 `/goal` 상태, 고아 세션, 경계 간 검색 공백)이 해결되었습니다.

소비자는 세션 ID를 비교하는 대신 모드를 확인합니다.

- `session:compress` 이벤트에는 `in_place: true/false`와 `old_session_id`가 포함됩니다(인플레이스 모드에서는 기존 ID가 없으므로 빈 문자열).
- 게이트웨이는 ID 변경 차이가 아니라 에이전트의 회전과 무관한 `_last_compaction_in_place` 플래그를 기준으로 트랜스크립트 처리를 재설정합니다.

`in_place: false`로 설정하면 기존의 교체 경로로 돌아갑니다. 이 경우 각 압축이 새 세션 ID를 커밋하고 `parent_session_id`를 통해 이전 세션과 연결합니다.

### 모델별 임계값 재정의

`compression.model_thresholds`를 사용하면 활성 모델에 따라 서로 다른 시점에 압축을 실행할 수 있습니다. 컨텍스트 창 크기가 매우 다른 모델 사이를 전환할 때 유용합니다(예: 1M 컨텍스트 모델은 더 늦게 압축하고 128K 모델은 더 일찍 압축해야 함).

```yaml
compression:
  threshold: 0.50
  model_thresholds:
    "glm-5.2": 0.40
    "glm-5.2-1M": 0.25
    "claude-sonnet": 0.35
```

해석 규칙:

- 키는 모델 이름과 **부분 문자열로 일치**하며, **가장 긴 일치 키가 우선**합니다(`glm-5.2-1M`은 `glm-5.2-1M` 모델에서 `glm-5.2`보다 우선).
- 일치하는 키가 없거나 맵이 비어 있으면 전역 `threshold`가 적용됩니다.
- `/model` 전환마다 재정의가 다시 해석됩니다. 일치하는 키가 없는 모델로 전환하면 전역 `threshold`로 돌아갑니다.
- **작은 컨텍스트 하한은 재정의 위에도 적용**됩니다(상향만 적용). 컨텍스트 창이 512K 미만인 모델의 하한은 `0.75`이므로 하한보다 낮은 재정의는 `0.75`로 올라가고, 그보다 높은 재정의(예: `0.80`)가 우선합니다.

플러그인 컨텍스트 엔진은 `from agent.context_compressor import resolve_model_threshold`를 통해 동일한 해석 로직을 재사용할 수 있습니다. `update_model()`을 재정의하는 엔진은 자체 압축 정책을 소유하므로 맵을 무시할 수 있습니다.

### Codex gpt-5.5 임계값 자동 상향

ChatGPT Codex OAuth 백엔드는 gpt-5.5를 **272K** 컨텍스트 창으로 제한합니다(동일한 슬러그가 OpenAI 직접 API와 OpenRouter에서는 1.05M, GitHub Copilot에서는 400K를 노출). 기본 50% 트리거에서는 약 136K에서 압축이 실행되어 모델이 실제로 사용할 수 있는 창의 절반만 사용하게 됩니다. 활성 경로가 Codex OAuth(`provider: openai-codex`)이고 모델이 gpt-5.5이면 Hermes는 트리거를 **85%**(약 231K)로 올리고 옵트아웃 명령이 포함된 알림을 표시합니다. 알림은 프로필당 한 번 표시됩니다. `$HERMES_HOME` 아래의 마커(`.codex_gpt55_autoraise_notice`)가 실행 여부를 기록하므로, 반복되는 에이전트/세션 초기화(예: 수신되는 게이트웨이 메시지마다)에서도 다시 표시되지 않습니다. 상향된 임계값이 나중에 변경되면 한 번 다시 알립니다. 이 영향은 정확히 이 경로에만 적용됩니다. 다른 제공자의 gpt-5.5는 전역 `threshold`를 유지합니다. 전역 값으로 다시 낮추려면 다음을 실행하세요.

```bash
hermes config set compression.codex_gpt55_autoraise false
```

85% 자동 상향은 유지하되 일회성 알림만 숨기려면 다음을 실행하세요.

```bash
hermes config set compression.codex_gpt55_autoraise_notice false
```

### Codex app-server 스레드 압축

Codex app-server 세션(`api_mode: codex_app_server` — codex CLI/에이전트 런타임)은 다른 모든 경로와 다릅니다. codex 에이전트가 백업 스레드 컨텍스트를 소유하므로 Hermes의 보조 요약기는 이를 줄일 수 없습니다. 로컬 트랜스크립트 미러를 다시 작성해도 실제 스레드는 강제 컨텍스트 초기화가 발생할 때까지 무한히 커집니다. 따라서 이 런타임에서는 app-server 자체 메커니즘을 통해 압축합니다.

- 수동 압축(`/compress`)은 app-server에 스레드 압축(`thread/compact/start`)을 요청하고 압축 턴이 완료될 때까지 기다립니다.
- 자동 압축은 `compression.codex_app_server_auto`로 제어합니다. 기본값인 `native`는 app-server가 압축 시점을 결정하도록 하며 Hermes는 결과 압축 이벤트(압축 카운터, 세션 이벤트)를 기록합니다. `hermes`로 설정하면 Hermes의 압축 임계값이 app-server 압축을 시작하고, `off`로 설정하면 Hermes가 시작하는 자동 압축을 완전히 비활성화합니다(codex는 여전히 자체적으로 압축할 수 있음).

이 런타임에서는 Hermes의 로컬 트랜스크립트를 절대 다시 작성하지 않습니다. state.db가 압축 경계를 기록하고 보이는 트랜스크립트는 그대로 유지됩니다. 다른 모든 경로( Codex OAuth 채팅 세션 포함)는 Hermes 요약 압축기를 유지합니다.

### 네이티브 Responses 압축(직접 OpenAI / Codex 구독의 gpt-5.6)

OpenAI의 Responses API는 서버 측 압축을 지원합니다. 요청에 `context_management: [{type: "compaction", compact_threshold: N}]`이 포함되고 렌더링된 입력이 N 토큰을 넘으면, 서버는 오래된 컨텍스트를 불투명한 암호화 `compaction` 출력 항목으로 정리합니다. Hermes는 이 항목을 기존 어시스턴트 메시지의 재생 사이드카에 저장하고 이후 턴에 다시 전송하여 제거된 기록을 대신하도록 합니다. 이를 통해 클라이언트 측 요약 단계 없이 긴 범위의 회상이 가능하고, ZDR에도 적합합니다(`store: false`, `previous_response_id` 없음).

`compression.codex_responses_native: true`로 옵트인합니다. 게이트는 의도적으로 좁으며 매 요청마다 다시 확인됩니다.

- **모델:** gpt-5.6 계열만 해당합니다. 다른 모델은 이 필드가 있으면 서버 측에서 실패합니다(gpt-5.1/5.2는 HTTP 500을 반환하거나 스트림이 멈춥니다. 2026년 8월 실시간 검증 결과, 다운그레이드할 구조화된 거부 응답은 없습니다).
- **경로:** `api.openai.com`(OpenAI API 키) 또는 ChatGPT Codex 백엔드(Codex 구독 OAuth)만 해당합니다. xAI, GitHub/Copilot, OpenRouter, 릴레이 및 로컬 서버에는 이 필드가 절대 전달되지 않습니다.

압축의 나머지 동작은 변경되지 않습니다. 로컬 압축기는 폴백 담당자로 계속 활성화되어 있고(네이티브 임계값은 로컬 트리거보다 약 8K 토큰 낮게 제한되어 서버가 먼저 압축), 이 필드에 대한 구조화된 제공자 거부가 발생하면 세션의 네이티브 압축을 비활성화하고 해당 필드 없이 요청을 재시도합니다. 세션을 대상이 아닌 모델이나 경로로 전환하면 해당 필드가 전송되지 않습니다. 엔드포인트가 변경될 때 기존 교차 발급자 보호가 캡처된 체크포인트를 재생에서 제거하기 때문입니다.

### 계산된 값(기본값의 200K 컨텍스트 모델 기준)

```
context_length       = 200,000
threshold_tokens     = 200,000 × 0.50 = 100,000
tail_token_budget    = 100,000 × 0.20 = 20,000
max_summary_tokens   = min(200,000 × 0.05, 12,000) = 10,000
```

:::note 임계값은 MAIN 모델의 컨텍스트 창에서 파생됩니다
`threshold_tokens`는 항상 `threshold × context_length`이며, 여기서 `context_length`는 **주 에이전트 모델**의 컨텍스트 창입니다. 보조/요약 모델의 컨텍스트 창이 아닙니다. 기본값 `0.50`인 262,144 토큰 모델에서는 임계값이 `262,144 × 0.50 = 131,072`입니다. 이 값이 흔히 말하는 "128K 컨텍스트"와 가까운 것은 비율이 만들어 낸 우연일 뿐, 보조 모델의 창이 트리거라는 뜻이 아닙니다. 보조 모델의 컨텍스트 창은 별개의 문제입니다. 요약 생성 가능 여부에 어떤 영향을 미치는지는 아래의 "요약 모델 컨텍스트 길이" 경고를 참조하세요.
:::


## 압축 알고리즘

`ContextCompressor.compress()` 메서드는 4단계 알고리즘을 따릅니다.

### 1단계: 오래된 도구 결과 정리(저렴하며 LLM 호출 없음)

보호된 테일 바깥의 오래된 도구 결과(>200자)는 다음으로 대체됩니다.
```
[Old tool output cleared to save context space]
```

이는 파일 내용, 터미널 출력, 검색 결과와 같은 장황한 도구 출력에서 상당한 토큰을 절약하는 저렴한 사전 처리 단계입니다.

### 2단계: 경계 결정

```
┌─────────────────────────────────────────────────────────────┐
│  Message list                                               │
│                                                             │
│  [0..2]  ← protect_first_n (system + first exchange)        │
│  [3..N]  ← middle turns → SUMMARIZED                        │
│  [N..end] ← tail (by token budget OR protect_last_n)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

테일 보호는 **토큰 예산 기반**입니다. 끝에서부터 거꾸로 이동하며 예산이 소진될 때까지 토큰을 누적합니다. 예산으로 보호되는 메시지가 더 적어질 경우 고정된 `protect_last_n` 개수로 대체됩니다.

경계는 도구 호출/도구 결과 그룹이 분할되지 않도록 정렬됩니다. `_align_boundary_backward()` 메서드는 연속된 도구 결과를 지나 부모 어시스턴트 메시지를 찾으며 그룹을 온전히 유지합니다.

### 3단계: 구조화된 요약 생성

:::warning 요약 모델 컨텍스트 길이
요약 모델의 컨텍스트 창은 주 에이전트 모델의 컨텍스트 창보다 **크거나 같아야** 합니다. 전체 중간 구간이 하나의 `call_llm(task="compression")` 호출로 요약 모델에 전달됩니다. 요약 모델의 컨텍스트가 더 작으면 API가 컨텍스트 길이 오류를 반환하고, `_generate_summary()`가 이를 포착하여 경고를 기록한 뒤 `None`을 반환합니다. 그러면 압축기는 요약 없이 중간 턴을 삭제하여 대화 컨텍스트가 조용히 손실됩니다. 이는 압축 품질 저하의 가장 흔한 원인입니다.
:::

중간 턴은 다음 구조화된 템플릿을 사용하여 보조 LLM으로 요약됩니다.

```
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[User preferences, coding style, constraints, important decisions]

## Progress
### Done
[Completed work — specific file paths, commands run, results]
### In Progress
[Work currently underway]
### Blocked
[Any blockers or issues encountered]

## Key Decisions
[Important technical decisions and why]

## Relevant Files
[Files read, modified, or created — with brief note on each]

## Next Steps
[What needs to happen next]

## Critical Context
[Specific values, error messages, configuration details]
```

요약 예산은 압축되는 콘텐츠의 양에 따라 조정됩니다.
- 공식: `content_tokens × 0.20`(`_SUMMARY_RATIO` 상수)
- 최소: 2,000 토큰
- 최대: `min(context_length × 0.05, 12,000)` 토큰

### 4단계: 압축된 메시지 조립

압축된 메시지 목록은 다음과 같습니다.
1. 헤드 메시지(첫 압축 시 시스템 프롬프트에 메모 추가)
2. 요약 메시지(연속된 동일 역할 위반을 피하도록 역할 선택)
3. 테일 메시지(수정하지 않음)

고립된 도구 호출/도구 결과 쌍은 `_sanitize_tool_pairs()`로 정리합니다.
- 제거된 호출을 참조하는 도구 결과 → 제거
- 결과가 제거된 도구 호출 → 스텁 결과 주입

### 반복 압축

두 번째 이후의 압축에서는 이전 요약을 LLM에 전달하고 처음부터 다시 요약하는 대신 **업데이트**하도록 지시합니다. 이를 통해 여러 번의 압축에서도 정보가 유지됩니다. 항목은 "진행 중"에서 "완료"로 이동하고, 새로운 진행 상황이 추가되며, 더 이상 유효하지 않은 정보는 제거됩니다.

압축기 인스턴스의 `_previous_summary` 필드가 이 목적을 위해 마지막 요약 텍스트를 저장합니다.


## 전/후 예시

### 압축 전(메시지 45개, 약 95K 토큰)

```
[0] system:    "You are a helpful assistant..." (system prompt)
[1] user:      "Help me set up a FastAPI project"
[2] assistant: <tool_call> terminal: mkdir project </tool_call>
[3] tool:      "directory created"
[4] assistant: <tool_call> write_file: main.py </tool_call>
[5] tool:      "file written (2.3KB)"
    ... 30 more turns of file editing, testing, debugging ...
[38] assistant: <tool_call> terminal: pytest </tool_call>
[39] tool:      "8 passed, 2 failed\n..."  (5KB output)
[40] user:      "Fix the failing tests"
[41] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[42] tool:      "import pytest\n..."  (3KB)
[43] assistant: "I see the issue with the test fixtures..."
[44] user:      "Great, also add error handling"
```

### 압축 후(메시지 25개, 약 45K 토큰)

```
[0] system:    "You are a helpful assistant...
               [Note: Some earlier conversation turns have been compacted...]"
[1] user:      "Help me set up a FastAPI project"
[2] assistant: "[CONTEXT COMPACTION] Earlier turns were compacted...

               ## Goal
               Set up a FastAPI project with tests and error handling

               ## Progress
               ### Done
               - Created project structure: main.py, tests/, requirements.txt
               - Implemented 5 API endpoints in main.py
               - Wrote 10 test cases in tests/test_api.py
               - 8/10 tests passing

               ### In Progress
               - Fixing 2 failing tests (test_create_user, test_delete_user)

               ## Relevant Files
               - main.py — FastAPI app with 5 endpoints
               - tests/test_api.py — 10 test cases
               - requirements.txt — fastapi, pytest, httpx

               ## Next Steps
               - Fix failing test fixtures
               - Add error handling"
[3] user:      "Fix the failing tests"
[4] assistant: <tool_call> read_file: tests/test_api.py </tool_call>
[5] tool:      "import pytest\n..."
[6] assistant: "I see the issue with the test fixtures..."
[7] user:      "Great, also add error handling"
```


## 프롬프트 캐싱(Anthropic)

소스: `agent/prompt_caching.py`

여러 턴의 대화에서 대화 접두사를 캐싱하여 입력 토큰 비용을 약 75% 줄입니다. Anthropic의 `cache_control` 중단점을 사용합니다.

### 전략: system_and_3

Anthropic은 요청당 최대 4개의 `cache_control` 중단점을 허용합니다. Hermes는 `system_and_3` 전략을 사용합니다.

```
Breakpoint 1: System prompt           (stable across all turns)
Breakpoint 2: 3rd-to-last non-system message  ─┐
Breakpoint 3: 2nd-to-last non-system message   ├─ Rolling window
Breakpoint 4: Last non-system message          ─┘
```

### 작동 방식

`apply_anthropic_cache_control()`은 메시지를 깊은 복사한 뒤 `cache_control` 마커를 주입합니다.

```python
# Cache marker format
marker = {"type": "ephemeral"}
# Or for 1-hour TTL:
marker = {"type": "ephemeral", "ttl": "1h"}
```

마커는 콘텐츠 유형에 따라 다르게 적용됩니다.

| 콘텐츠 유형 | 마커 위치 |
|-------------|-------------|
| 문자열 콘텐츠 | `[{"type": "text", "text": ..., "cache_control": ...}]`로 변환 |
| 목록 콘텐츠 | 마지막 요소의 딕셔너리에 추가 |
| None/빈 콘텐츠 | `msg["cache_control"]`으로 추가 |
| 도구 메시지 | `msg["cache_control"]`으로 추가(네이티브 Anthropic 전용) |

### 캐시 인식 설계 패턴

1. **안정적인 시스템 프롬프트**: 시스템 프롬프트는 중단점 1이며 모든 턴에서 캐시됩니다. 대화 중간에 변경하지 마세요(압축은 첫 압축 시에만 메모를 추가).

2. **메시지 순서가 중요함**: 캐시 적중에는 접두사 일치가 필요합니다. 중간에 메시지를 추가하거나 제거하면 이후 모든 항목의 캐시가 무효화됩니다.

3. **압축과 캐시의 상호작용**: 압축 후에는 압축된 영역의 캐시가 무효화되지만 시스템 프롬프트 캐시는 유지됩니다. 순환 3개 메시지 창이 1~2턴 안에 캐싱을 다시 설정합니다.

4. **TTL 선택**: 기본값은 `5m`(5분)입니다. 턴 사이에 사용자가 쉬는 장시간 세션에는 `1h`를 사용하세요.

5. **모델 ID는 캐시 키의 일부임**: 제공자 측 캐시는 요청을 처리하는 모델(및 계정/API 키)을 기준으로 범위가 정해집니다. 대화 중간에 모델을 변경하면(명시적인 `/model` 전환, 주 모델 폴백, 다른 계정으로의 자격 증명 풀 교체 등) 다음 요청은 캐시 적중이 0이 되고 할인되지 않은 입력 가격으로 전체 대화를 다시 읽습니다. 이는 제공자 캐시의 작동 방식에 따른 본질적인 특성이므로 Hermes가 피할 수 없습니다. 이러한 이유로 `/model`, 폴백 제공자, 자격 증명 풀에 대한 사용자 문서에는 비용 경고가 포함되어 있습니다. 모델이나 자격 증명을 세션 중간에 조용히 교체하는 기능을 추가하지 마세요.

### 프롬프트 캐싱 활성화

다음 조건을 만족하면 프롬프트 캐싱이 자동으로 활성화됩니다.
- 모델이 Anthropic Claude 모델임(모델 이름으로 감지)
- 제공자가 `cache_control`을 지원함(네이티브 Anthropic API 또는 OpenRouter)

```yaml
# config.yaml — TTL is configurable (must be "5m" or "1h")
prompt_caching:
  cache_ttl: "5m"
```

CLI는 시작 시 캐싱 상태를 표시합니다.
```
💾 Prompt caching: ENABLED (Claude via OpenRouter, 5m TTL)
```


## 컨텍스트 압력 경고

중간 컨텍스트 압력 경고는 제거되었습니다(`run_agent.py`의 반복 예산 블록에는 "No intermediate pressure warnings — they caused models to 'give up' prematurely on complex tasks"라고 명시되어 있음). 압축은 프롬프트 토큰이 설정된 `compression.threshold`(기본 50%)에 도달하면 사전 경고 단계 없이 실행됩니다. 게이트웨이 세션 관리는 모델 컨텍스트 창의 85%에서 보조 안전망으로 실행됩니다.

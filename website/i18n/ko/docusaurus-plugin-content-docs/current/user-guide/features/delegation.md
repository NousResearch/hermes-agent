---
sidebar_position: 7
title: "서브에이전트 위임 (Subagent Delegation)"
description: "delegate_task를 사용하여 병렬 작업을 위한 격리된 하위 에이전트 생성"
---

# 서브에이전트 위임 (Subagent Delegation)

`delegate_task` 도구는 격리된 컨텍스트, 제한된 도구 세트 및 자체 터미널 세션을 갖춘 하위 AIAgent 인스턴스를 생성합니다. 각 하위 에이전트는 새로운 대화로 시작하여 독립적으로 작업하며, 오직 최종 요약본만이 부모 에이전트의 컨텍스트로 전달됩니다.

## 단일 작업 (Single Task)

```python
delegate_task(
    goal="테스트가 실패하는 이유 디버그",
    context="오류: test_foo.py 42번째 줄의 어서션(assertion)",
    toolsets=["terminal", "file"]
)
```

## 병렬 일괄 처리 (Parallel Batch)

기본적으로 최대 3개의 동시 서브에이전트를 지원합니다 (구성 가능, 엄격한 상한선 없음):

```python
delegate_task(tasks=[
    {"goal": "주제 A 조사", "toolsets": ["web"]},
    {"goal": "주제 B 조사", "toolsets": ["web"]},
    {"goal": "빌드 오류 수정", "toolsets": ["terminal", "file"]}
])
```

## 서브에이전트 컨텍스트 작동 방식 (How Subagent Context Works)

:::warning 주의: 서브에이전트는 아무것도 모릅니다
서브에이전트는 **완전히 새로운 대화**로 시작합니다. 부모의 대화 기록, 이전 도구 호출 또는 위임 전에 논의된 내용에 대해 전혀 알지 못합니다. 서브에이전트의 유일한 컨텍스트는 부모 에이전트가 `delegate_task`를 호출할 때 채워 넣는 `goal`과 `context` 필드에서만 제공됩니다.
:::

즉, 부모 에이전트는 서브에이전트가 필요로 하는 **모든 것**을 호출 시 전달해야 합니다:

```python
# 나쁨 - 서브에이전트는 "오류"가 무엇인지 전혀 모릅니다.
delegate_task(goal="오류 수정")

# 좋음 - 서브에이전트는 필요한 모든 컨텍스트를 가집니다.
delegate_task(
    goal="api/handlers.py에서 TypeError 수정",
    context="""api/handlers.py 파일의 47번째 줄에 TypeError가 있습니다:
    'NoneType' object has no attribute 'get'.
    process_request() 함수는 parse_body()로부터 딕셔너리를 받지만,
    Content-Type이 누락된 경우 parse_body()는 None을 반환합니다.
    프로젝트는 /home/user/myproject에 있으며 Python 3.11을 사용합니다."""
)
```

서브에이전트는 제공된 `goal` 및 `context`를 기반으로 구성된 집중적인 시스템 프롬프트를 받게 되며, 작업을 완료하고 수행한 작업, 발견한 사항, 수정한 파일 및 발생한 문제에 대한 구조화된 요약을 제공하도록 지시받습니다.

## 실제 사용 예시 (Practical Examples)

### 병렬 조사 (Parallel Research)

여러 주제를 동시에 조사하고 요약을 수집합니다:

```python
delegate_task(tasks=[
    {
        "goal": "2025년 WebAssembly의 현재 상태 조사",
        "context": "초점: 브라우저 지원, 비 브라우저 런타임, 언어 지원",
        "toolsets": ["web"]
    },
    {
        "goal": "2025년 RISC-V 채택의 현재 상태 조사",
        "context": "초점: 서버 칩, 임베디드 시스템, 소프트웨어 생태계",
        "toolsets": ["web"]
    },
    {
        "goal": "2025년 양자 컴퓨팅 진행 상황 조사",
        "context": "초점: 오류 정정 혁신, 실용적 응용, 주요 기업",
        "toolsets": ["web"]
    }
])
```

### 코드 리뷰 및 수정 (Code Review + Fix)

새로운 컨텍스트에 리뷰 및 수정 워크플로우를 위임합니다:

```python
delegate_task(
    goal="인증 모듈의 보안 문제를 검토하고 발견된 문제 수정",
    context="""/home/user/webapp에 있는 프로젝트입니다.
    인증 모듈 파일: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py.
    이 프로젝트는 Flask, PyJWT 및 bcrypt를 사용합니다.
    초점: SQL 인젝션, JWT 검증, 비밀번호 처리, 세션 관리.
    발견된 문제를 수정하고 테스트 스위트(pytest tests/auth/)를 실행하세요.""",
    toolsets=["terminal", "file"]
)
```

### 다중 파일 리팩토링 (Multi-File Refactoring)

부모 에이전트의 컨텍스트를 넘치게 할 대규모 리팩토링 작업을 위임합니다:

```python
delegate_task(
    goal="src/의 모든 Python 파일을 리팩토링하여 print()를 적절한 로깅으로 교체",
    context="""/home/user/myproject에 있는 프로젝트입니다.
    logger = logging.getLogger(__name__)와 함께 'logging' 모듈을 사용하세요.
    print() 호출을 적절한 로그 레벨로 교체합니다:
    - print(f"Error: ...") -> logger.error(...)
    - print(f"Warning: ...") -> logger.warning(...)
    - print(f"Debug: ...") -> logger.debug(...)
    - 기타 print -> logger.info(...)
    테스트 파일이나 CLI 출력의 print()는 변경하지 마세요.
    이후 pytest를 실행하여 손상된 내용이 없는지 확인하세요.""",
    toolsets=["terminal", "file"]
)
```

## 일괄 처리 모드 세부 정보 (Batch Mode Details)

`tasks` 배열을 제공하면 서브에이전트는 스레드 풀을 사용하여 **병렬**로 실행됩니다:

- **최대 동시성:** 기본적으로 3개의 작업 (`delegation.max_concurrent_children` 또는 `DELEGATION_MAX_CONCURRENT_CHILDREN` 환경 변수를 통해 구성 가능; 최솟값 1, 엄격한 상한선 없음). 제한보다 큰 일괄 처리는 조용히 잘리지 않고 도구 오류를 반환합니다.
- **스레드 풀:** 구성된 동시성 제한을 최대 작업자 수로 하여 `ThreadPoolExecutor`를 사용합니다.
- **진행 상황 표시:** CLI 모드에서는 트리 뷰를 통해 각 서브에이전트의 도구 호출을 작업별 완료 줄과 함께 실시간으로 표시합니다. 게이트웨이 모드에서는 진행 상황이 일괄 처리되어 부모 에이전트의 진행 콜백으로 전달됩니다.
- **결과 순서:** 완료 순서와 관계없이 입력 순서와 일치하도록 결과를 작업 인덱스별로 정렬합니다.
- **중단 전파:** 부모를 중단시키면(예: 새 메시지 전송) 활성화된 모든 자식도 중단됩니다.

단일 작업 위임은 스레드 풀 오버헤드 없이 직접 실행됩니다.

## 모델 재정의 (Model Override)

`config.yaml`을 통해 서브에이전트에 다른 모델을 구성할 수 있습니다 — 간단한 작업을 더 저렴하거나 빠른 모델에 위임할 때 유용합니다:

```yaml
# ~/.hermes/config.yaml 파일 내
delegation:
  model: "google/gemini-flash-2.0"    # 서브에이전트를 위한 더 저렴한 모델
  provider: "openrouter"              # 선택 사항: 서브에이전트를 다른 제공자로 라우팅
```

생략하면 서브에이전트는 부모와 동일한 모델을 사용합니다.

## 도구 세트 선택 팁 (Toolset Selection Tips)

`toolsets` 매개변수는 서브에이전트가 액세스할 수 있는 도구를 제어합니다. 작업에 따라 선택하세요:

| 도구 세트 패턴 (Toolset Pattern) | 사용 사례 (Use Case) |
|----------------|----------|
| `["terminal", "file"]` | 코드 작업, 디버깅, 파일 편집, 빌드 |
| `["web"]` | 조사, 팩트 체크, 문서 조회 |
| `["terminal", "file", "web"]` | 풀스택 작업 (기본값) |
| `["file"]` | 읽기 전용 분석, 실행 없는 코드 리뷰 |
| `["terminal"]` | 시스템 관리, 프로세스 관리 |

지정한 내용에 관계없이 서브에이전트에 대해 특정 도구 세트가 차단됩니다:
- `delegation` — 리프(leaf) 서브에이전트에 대해 차단됨 (기본값). `max_spawn_depth`에 의해 제한되는 `role="orchestrator"` 자식에 대해서는 유지됨 — 아래의 [깊이 제한 및 중첩 오케스트레이션](#깊이-제한-및-중첩-오케스트레이션-depth-limit-and-nested-orchestration)을 참조하세요.
- `clarify` — 서브에이전트는 사용자와 상호 작용할 수 없습니다.
- `memory` — 공유된 영구 메모리에 기록할 수 없습니다.
- `code_execution` — 자식 에이전트는 단계별로 추론해야 합니다.
- `send_message` — 플랫폼 간 부작용 발생 금지 (예: Telegram 메시지 전송 불가).

## 최대 반복 횟수 (Max Iterations)

각 서브에이전트는 도구 호출 턴의 수를 제어하는 반복 제한(기본값: 50)을 가집니다:

```python
delegate_task(
    goal="빠른 파일 확인",
    context="/etc/nginx/nginx.conf가 존재하는지 확인하고 처음 10줄을 인쇄",
    max_iterations=10  # 간단한 작업이므로 많은 턴이 필요하지 않음
)
```

## 자식 시간 초과 (Child Timeout)

서브에이전트가 `delegation.child_timeout_seconds` 물리적 시간(wall-clock seconds) 이상 조용히 있으면 중단된 것으로 간주하고 종료(kill)됩니다. 기본값은 **600초**(10분)입니다 — 복잡한 연구 작업에서 고성능 추론 모델이 생각하는 도중에 강제 종료되는 문제를 해결하기 위해 이전 릴리스의 300초에서 상향 조정되었습니다. 필요에 따라 설치 환경별로 조정하세요:

```yaml
delegation:
  child_timeout_seconds: 600   # 기본값
```

빠른 로컬 모델의 경우 낮추고, 어려운 문제에 대해 느린 추론 모델을 사용하는 경우 높이세요. 타이머는 자식 에이전트가 API 호출이나 도구 호출을 할 때마다 재설정됩니다 — 진정으로 유휴 상태인 작업자만 강제 종료를 트리거합니다.

:::tip 호출 없이 시간 초과 시의 진단 덤프
서브에이전트가 **단 한 번의 API 호출도 없이** 시간 초과된 경우(보통 제공자 연결 불가, 인증 실패, 또는 도구 스키마 거부 발생 시), `delegate_task`는 서브에이전트의 구성 스냅샷, 자격 증명 해결 추적 과정, 및 초기 오류 메시지를 포함한 구조화된 진단 내용을 `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log`에 기록합니다. 이전의 조용히 시간 초과되던 동작 방식보다 근본 원인을 찾기 훨씬 쉽습니다.
:::

## 실행 중인 서브에이전트 모니터링 (`/agents`)

TUI는 재귀적인 `delegate_task` 팬아웃(fan-out)을 일급 감사(audit) 표면으로 바꿔주는 `/agents` 오버레이(별칭 `/tasks`)를 제공합니다:

- 부모별로 그룹화된, 실행 중이거나 최근에 완료된 서브에이전트의 실시간 트리 뷰.
- 브랜치별 비용, 토큰 및 수정된 파일에 대한 요약(rollups).
- 종료 및 일시 정지 제어 — 형제 에이전트를 방해하지 않고 특정 서브에이전트를 실행 중에 취소할 수 있습니다.
- 사후 검토: 서브에이전트가 부모에게 돌아간 후에도 각 서브에이전트의 턴별(turn-by-turn) 기록을 단계별로 살펴볼 수 있습니다.

클래식 CLI는 `/agents`를 텍스트 요약으로 출력할 뿐이지만, TUI 오버레이에서는 그 진가가 발휘됩니다. [TUI — 슬래시 명령어 (TUI — Slash commands)](/user-guide/tui#slash-commands)를 참고하세요.

## 깊이 제한 및 중첩 오케스트레이션 (Depth Limit and Nested Orchestration)

기본적으로 위임은 **단층적(flat)**입니다. 부모(깊이 0)가 자식(깊이 1)을 생성하고, 그 자식은 더 이상 위임할 수 없습니다. 이것은 제어할 수 없는 재귀적 위임을 방지합니다.

다단계 워크플로우(연구 → 종합, 또는 하위 문제들에 대한 병렬 오케스트레이션)의 경우, 부모는 자체 작업자를 위임할 수 **있는** **오케스트레이터(orchestrator)** 자식을 생성할 수 있습니다:

```python
delegate_task(
    goal="세 가지 코드 검토 접근 방식을 조사하고 하나를 추천",
    role="orchestrator",  # 이 자식이 자체 작업자를 생성할 수 있도록 허용
    context="...",
)
```

- `role="leaf"` (기본값): 자식은 더 이상 위임할 수 없습니다 — 단층적 위임 동작과 동일합니다.
- `role="orchestrator"`: 자식이 `delegation` 도구 세트를 유지합니다. `delegation.max_spawn_depth`에 의해 제한됩니다 (기본값 **1** = 단층적이므로 기본값에서는 `role="orchestrator"`가 아무런 효과가 없습니다). 오케스트레이터 자식이 리프(leaf) 손자를 생성할 수 있게 하려면 `max_spawn_depth`를 2로 올리십시오. 더 깊은 트리의 경우 3 이상으로 설정하십시오. 상한선은 없습니다 — 비용이 실질적인 한계입니다.
- `delegation.orchestrator_enabled: false`: `role` 매개변수에 관계없이 모든 자식을 강제로 `leaf`로 만드는 전역 킬 스위치(kill switch)입니다.

**비용 경고:** `max_spawn_depth: 3` 및 `max_concurrent_children: 3`일 경우 트리는 3×3×3 = 27개의 동시 리프 에이전트에 도달할 수 있습니다. 추가 단계마다 비용이 배가되므로 `max_spawn_depth`를 의도적으로 올리십시오.

## 수명 및 지속성 (Lifetime and Durability)

:::warning delegate_task는 동기식입니다 — 지속되지 않습니다
`delegate_task`는 **부모의 현재 턴 내에서** 실행됩니다. 모든 자식이 완료되거나 취소될 때까지 부모를 차단합니다. 이것은 백그라운드 작업 큐가 **아닙니다**:

- 만약 부모가 중단되면(사용자가 새 메시지 전송, `/stop`, `/new`), 활성화된 모든 자식이 취소되고 `status="interrupted"`를 반환합니다. 진행 중이던 작업은 폐기됩니다.
- 부모 턴이 종료된 후에는 자식이 계속 실행되지 **않습니다**.
- 취소된 자식은 구조화된 결과(`status="interrupted"`, `exit_reason="interrupted"`)를 반환하지만, 부모 역시 중단되었기 때문에 그 결과가 사용자에게 보이는 응답으로 전달되는 일은 거의 없습니다.

중단 상황에서도 살아남아야 하거나 현재 턴보다 오래 유지되어야 하는 **지속적인 장기 작업(durable long-running work)**의 경우 다음을 사용하십시오:

- `cronjob` (action=`create`) — 별도의 에이전트 실행을 예약합니다. 부모 턴 중단에 면역입니다.
- `terminal(background=True, notify_on_complete=True)` — 에이전트가 다른 작업을 수행하는 동안에도 계속 실행되는 장기 실행 셸 명령어입니다.
:::

## 핵심 특성 (Key Properties)

- 각 서브에이전트는 부모와 분리된 **자체 터미널 세션**을 가집니다.
- **중첩 위임은 선택 사항(opt-in)입니다** — 오직 `role="orchestrator"` 자식만 더 위임할 수 있으며, 그것도 `max_spawn_depth`가 기본값 1(단층적)에서 올려졌을 때만 가능합니다. `orchestrator_enabled: false`를 통해 전역적으로 비활성화할 수 있습니다.
- 리프 서브에이전트는 `delegate_task`, `clarify`, `memory`, `send_message`, `execute_code`를 호출할 수 **없습니다**. 오케스트레이터 서브에이전트는 `delegate_task`를 유지하지만 여전히 다른 4개는 사용할 수 없습니다.
- **중단 전파(Interrupt propagation)** — 부모를 중단시키면 활성화된 모든 자식(오케스트레이터 산하의 손자 포함)이 중단됩니다.
- 오직 최종 요약본만이 부모의 컨텍스트에 들어가 토큰 사용을 효율적으로 유지합니다.
- 서브에이전트는 부모의 **API 키, 제공자 구성 및 자격 증명 풀(credential pool)**을 상속합니다 (속도 제한에 따른 키 순환(key rotation)을 가능하게 함).

## Delegation vs execute_code

| 요인 (Factor) | delegate_task | execute_code |
|--------|--------------|-------------|
| **추론 (Reasoning)** | 전체 LLM 추론 루프 | 단지 Python 코드 실행 |
| **컨텍스트 (Context)** | 분리된 새로운 대화 | 대화 없음, 스크립트만 |
| **도구 액세스 (Tool access)** | 차단되지 않은 모든 도구와 추론 결합 | RPC를 통한 7개 도구, 추론 없음 |
| **병렬성 (Parallelism)** | 기본적으로 3개의 동시 서브에이전트 (구성 가능) | 단일 스크립트 |
| **가장 적합한 용도 (Best for)** | 판단이 필요한 복잡한 작업 | 기계적인 다단계 파이프라인 |
| **토큰 비용 (Token cost)** | 더 높음 (전체 LLM 루프) | 더 낮음 (stdout만 반환) |
| **사용자 상호 작용 (User interaction)** | 없음 (서브에이전트는 clarify 불가능) | 없음 |

**경험 법칙 (Rule of thumb):** 하위 작업이 추론, 판단 또는 다단계 문제 해결을 필요로 할 때는 `delegate_task`를 사용하세요. 기계적인 데이터 처리나 스크립팅된 워크플로우가 필요할 때는 `execute_code`를 사용하세요.

## 구성 (Configuration)

```yaml
# ~/.hermes/config.yaml 파일 내
delegation:
  max_iterations: 50                        # 자식당 최대 턴 수 (기본값: 50)
  # max_concurrent_children: 3              # 일괄 처리당 병렬 자식 수 (기본값: 3)
  # max_spawn_depth: 1                      # 트리 깊이 (최솟값 1, 상한선 없음, 기본값 1 = 단층적). 오케스트레이터 자식이 리프를 생성할 수 있게 하려면 2로 올림. 3 이상은 더 깊은 트리.
  # orchestrator_enabled: true              # 모든 자식을 리프 역할로 강제하려면 비활성화.
  model: "google/gemini-3-flash-preview"             # 선택적 제공자/모델 재정의
  provider: "openrouter"                             # 선택적 내장 제공자
  api_mode: anthropic_messages                       # 선택 사항; anthropic_messages 엔드포인트의 경우 base_url에서 자동 감지됨

# 또는 제공자 대신 직접 사용자 지정 엔드포인트 사용:
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
  # api_mode: "anthropic_messages"  # 선택 사항. base_url에 대한 와이어 프로토콜 재정의 ("chat_completions", "codex_responses" 또는 "anthropic_messages"). 비워두면 URL에서 자동 감지됨(예: /anthropic 접미사). 휴리스틱이 분류할 수 없는 엔드포인트(Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM 프록시 등)의 경우 명시적으로 설정.
```

`base_url`이 Anthropic 호환 엔드포인트를 가리킬 때 — 예를 들어 `/anthropic`으로 끝나는 경로, Azure Foundry Claude 라우트, 또는 MiniMax `/anthropic` 프록시 — `api_mode`는 `anthropic_messages`로 자동 감지되므로 별도 설정 없이 서브에이전트가 올바른 와이어 형식을 사용합니다. 자동 감지 추측이 틀린 경우(드문 경우) `api_mode`를 명시적으로 설정하세요.

:::tip
에이전트는 작업 복잡도에 따라 위임을 자동으로 처리합니다. 위임하라고 명시적으로 요청할 필요가 없습니다 — 에이전트가 알아서 판단하여 필요할 때 수행합니다.
:::

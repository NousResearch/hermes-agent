---
sidebar_position: 7
title: "서브에이전트 위임"
description: "delegate_task로 격리된 자식 에이전트를 생성하여 병렬 작업 스트림 실행"
---

# 서브에이전트 위임

`delegate_task` 도구는 격리된 컨텍스트, 상속된 도구 액세스 권한, 자체 터미널 세션을 갖는 자식 AIAgent 인스턴스를 생성합니다. 각 자식은 새로운 대화에서 시작해 독립적으로 작업하며, 최종 요약만 부모의 컨텍스트에 들어갑니다.

최상위 모델 호출은 자동으로 백그라운드에서 실행됩니다. Hermes는 즉시 핸들을 반환하여 대화를 계속할 수 있게 하고, 결과가 새 메시지로 돌아오면 게시합니다. 오케스트레이터 서브에이전트는 합성 결과를 반환하기 전에 자체 워커가 끝나기를 기다립니다.

## 단일 작업

```python
delegate_task(
    goal="Debug why tests fail",
    context="Error: assertion in test_foo.py line 42"
)
```

## 병렬 배치

기본적으로 최대 3개의 동시 서브에이전트가 실행됩니다(구성 가능하며 엄격한 상한은 없음).

```python
delegate_task(tasks=[
    {"goal": "Research topic A", "context": "Focus on recent primary sources"},
    {"goal": "Research topic B", "context": "Compare the leading explanations"},
    {"goal": "Fix the build", "context": "Project root: /home/user/project"}
])
```

## 서브에이전트 컨텍스트 작동 방식

:::warning 중요: 서브에이전트는 아무것도 알지 못합니다
서브에이전트는 완전히 새로운 대화에서 시작합니다. 부모 대화 기록, 이전 도구 호출, 또는 이전에 논의한 내용에 대한 지식이 전혀 없습니다. 서브에이전트가 받는 유일한 컨텍스트는 부모가 `delegate_task`를 호출할 때 채우는 `goal` 및 `context` 필드에서 옵니다.
:::

따라서 부모 에이전트는 호출할 때 서브에이전트에 필요한 모든 것을 전달해야 합니다.

```python
# BAD - subagent has no idea what "the error" is
delegate_task(goal="Fix the error")

# GOOD - subagent has all context it needs
delegate_task(
    goal="Fix the TypeError in api/handlers.py",
    context="""The file api/handlers.py has a TypeError on line 47:
    'NoneType' object has no attribute 'get'.
    The function process_request() receives a dict from parse_body(),
    but parse_body() returns None when Content-Type is missing.
    The project is at /home/user/myproject and uses Python 3.11."""
)
```

서브에이전트는 부모가 제공한 목표와 컨텍스트로 구성된 집중 시스템 프롬프트를 받으며, 이를 통해 부모는 작업을 완료하고 수행한 작업, 발견한 내용, 수정한 파일, 발생한 문제에 대한 구조화된 요약을 반환하도록 지시합니다.

## 실용 예시

### 병렬 조사

여러 주제를 동시에 조사하고 요약을 수집합니다.

```python
delegate_task(tasks=[
    {
        "goal": "Research the current state of WebAssembly in 2025",
        "context": "Focus on: browser support, non-browser runtimes, language support"
    },
    {
        "goal": "Research the current state of RISC-V adoption in 2025",
        "context": "Focus on: server chips, embedded systems, software ecosystem"
    },
    {
        "goal": "Research quantum computing progress in 2025",
        "context": "Focus on: error correction breakthroughs, practical applications, key players"
    }
])
```

### 코드 검토 + 수정

새로운 컨텍스트에서 검토 및 수정 워크플로를 위임합니다.

```python
delegate_task(
    goal="Review the authentication module for security issues and fix any found",
    context="""Project at /home/user/webapp.
    Auth module files: src/auth/login.py, src/auth/jwt.py, src/auth/middleware.py.
    The project uses Flask, PyJWT, and bcrypt.
    Focus on: SQL injection, JWT validation, password handling, session management.
    Fix any issues found and run the test suite (pytest tests/auth/)."""
)
```

### 다중 파일 리팩터링

부모 컨텍스트를 가득 채울 수 있는 대규모 리팩터링을 위임합니다.

```python
delegate_task(
    goal="Refactor all Python files in src/ to replace print() with proper logging",
    context="""Project at /home/user/myproject.
    Use the 'logging' module with logger = logging.getLogger(__name__).
    Replace print() calls with appropriate log levels:
    - print(f"Error: ...") -> logger.error(...)
    - print(f"Warning: ...") -> logger.warning(...)
    - print(f"Debug: ...") -> logger.debug(...)
    - Other prints -> logger.info(...)
    Don't change print() in test files or CLI output.
    Run pytest after to verify nothing broke."""
)
```

## 배치 모드 세부 사항

최상위 에이전트가 `tasks` 배열을 제공하면 Hermes는 하나의 백그라운드 핸들을 반환하고, 동시성 제한에 따라 서브에이전트를 병렬로 실행한 뒤 모든 자식이 완료되면 통합된 결과 하나를 게시합니다.

- **최대 동시성:** 기본 배치에서는 3개이며(`delegation.max_concurrent_children` 또는 `DELEGATION_MAX_CONCURRENT_CHILDREN` 환경 변수로 구성 가능), 엄격한 상한은 없습니다. 제한을 초과한 배치는 조용히 잘리지 않고 도구 오류를 반환합니다.
- **스레드 풀:** 구성된 동시성 제한을 최대 워커 수로 사용하는 `ThreadPoolExecutor`
- **진행 상황 표시:** CLI 모드에서는 실시간으로 도구 호출을 표시하는 트리 뷰와 작업별 완료 줄을 보여주고, 게이트웨이 모드에서는 진행 콜백을 통해 배치로 중계합니다.
- **결과 순서:** 완료 순서와 관계없이 입력 순서에 맞춰 결과를 정렬합니다.
- **취소:** 후속 메시지는 활성 백그라운드 배치를 취소하지 않습니다. `/stop` 또는 소유 세션을 닫거나 초기화하면 실행 중인 자식이 취소됩니다. 동기 오케스트레이터 자식은 부모의 인터럽트 상태를 따릅니다.

### 내구성 있는 백그라운드 완료

백그라운드 위임이 끝나면 Hermes는 결과를 일반적인 새 턴 큐에 게시하기 전에 활성 프로필의 `state.db`에 완료 이벤트를 저장합니다. 완료 후 전달 전에 Hermes가 재시작되면 보류 중인 이벤트가 복원되어 동일한 소유권 확인을 거쳐 라우팅됩니다. 경쟁 소비자는 내구성 있는 클레임을 사용하므로, 성공적으로 수락한 소비자만 전달을 확인하며 실패한 시도는 재시도를 위해 클레임을 해제합니다.

이는 충돌 후 자식 실행을 재개하지 않습니다. 실행 중이던 위임은 외부 부작용이 발생했는지 Hermes가 증명할 수 없으므로 `unknown`으로 기록됩니다. 보류 및 전달된 레코드는 제한되며 프로필별로 분리됩니다.

## 모델 재정의

서브에이전트에 다른 모델을 사용하도록 `config.yaml`에서 구성할 수 있습니다. 간단한 작업에 더 저렴하고 빠른 모델을 위임할 때 유용합니다.

```yaml
# In ~/.hermes/config.yaml
delegation:
  model: "google/gemini-flash-2.0"    # Cheaper model for subagents
  provider: "openrouter"              # Optional: route subagents to a different provider
```

생략하면 서브에이전트는 부모와 같은 모델을 사용합니다.

### 비용 전략: 프런티어 플래너, 저렴한 워커

잘 정의된 하위 작업으로 문제를 분해하려면 프런티어 수준의 판단이 필요하지만, 명확한 목표와 전체 컨텍스트, 출력 계약이 이미 주어진 하위 작업을 실행하는 데에는 보통 그렇지 않습니다. 한편 토큰은 자식이 소비하는데, 서브에이전트의 병렬 배치는 일반적으로 전체 실행 토큰의 대부분을 사용하므로 실제 비용이 발생하는 곳은 워커 모델입니다. 주 세션은 프런티어 모델로 유지하면서 `delegation.model`을 저렴한 모델로 고정하면 중요한 계획 품질은 유지하고 작업량이 많은 부분의 비용을 줄일 수 있습니다.

```yaml
# ~/.hermes/config.yaml
model:
  default: "your-frontier-model"     # parent (planner) stays on the frontier model
delegation:
  model: "your-inexpensive-model"    # all delegate_task children run on this
  provider: "openrouter"             # optional: route children to a different provider
```

해결 순서는 다음과 같습니다. `delegation.base_url`(직접 엔드포인트)가 우선하고, 그다음 `delegation.provider`(런타임 프로바이더 시스템을 통해 전체 자격 증명 번들을 확인)가 오며, 둘 다 설정되지 않으면 자식은 부모의 프로바이더와 자격 증명을 상속합니다. `delegation.model`은 모든 경우에 적용되며, 비어 있으면 자식은 부모 모델을 상속합니다.

고정 설정은 전역이라는 점에 유의하세요. `delegate_task`에는 작업별 모델 매개변수가 없으므로 배치의 모든 자식이 구성된 위임 모델로 실행됩니다. 더 강력한 모델이 필요한 품질 민감 하위 작업에는 해당 세션에서 `delegation.model`을 설정하지 않거나 [kanban board](kanban.md#per-task-model-override)에 작업을 전달하세요. kanban board는 작업별 모델 재정의를 지원합니다.

## 상속된 도구 액세스

`delegate_task`는 모델이 사용하는 `toolsets` 매개변수를 허용하지 않습니다. 각 서브에이전트는 부모의 활성화된 도구 세트를 상속하므로 모델은 부모에게 없는 기능을 자식에게 부여할 수 없습니다. 위임된 작업에 추가 기능이 필요하다면 대화를 시작하기 전에 부모의 도구를 구성하세요.

부모에게 있더라도 특정 도구는 서브에이전트에서 차단됩니다.

- `delegate_task` — 리프 서브에이전트에서는 차단됩니다(기본값). `role="orchestrator"` 자식에서는 유지되며 `max_spawn_depth`로 제한됩니다. 아래 [깊이 제한 및 중첩 오케스트레이션](#depth-limit-and-nested-orchestration)을 참조하세요.
- `clarify` — 서브에이전트는 사용자와 상호작용할 수 없습니다.
- `memory` — 공유 영속 메모리에 쓰지 않습니다.
- `send_message` — 플랫폼 간 부작용이 없습니다.
- `cronjob` — 부모의 이름으로 더 많은 작업을 예약할 수 없습니다.

두 역할 모두 `execute_code`(프로그래밍 방식의 도구 호출)를 유지하므로 자식은 기계적인 작업을 일괄 처리할 수 있습니다.

## 최대 반복 횟수

각 서브에이전트에는 도구 호출 턴 수를 제어하는 반복 제한(기본값: 50)이 있습니다.

```python
delegate_task(
    goal="Quick file check",
    context="Check if /etc/nginx/nginx.conf exists and print its first 10 lines",
    max_iterations=10  # Simple task, don't need many turns
)
```

## 자식 타임아웃

기본적으로 서브에이전트에는 벽시계 시간 초과가 없습니다. 자식은 실제로 수행 중인 작업에서 API 오류, 도구 오류, 또는 반복 예산 소진이 발생할 때만 실패하며, 위임 수준의 스톱워치 때문에 실패하지는 않습니다. 이전 릴리스에는 하드 캡(300초, 이후 600초)이 있었지만, 꾸준히 진행 중인 정당하게 바쁜 자식을 중간에 종료했습니다. 심층 코드 검토, 대규모 조사 팬아웃, 느린 추론 모델에는 10분 이상이 일상적으로 필요합니다.

진정으로 멈춘 자식은 여전히 감지됩니다. 하트비트 정체 모니터는 자식이 진행하지 않을 때(모델 API 호출, 도구 시작, 활동 타임스탬프 갱신이 전혀 없을 때) 부모의 활동 갱신을 멈추게 하여 실제로 멈춘 경우 게이트웨이 비활성 타임아웃이 작동하도록 합니다. 진행 중인 모델 대기는 진행으로 간주됩니다. 서브에이전트는 프로바이더를 기다리는 동안 활동 시계를 갱신하므로 느린 로컬 프로바이더나 긴 프리필 완료가 정체로 처리되지 않습니다.

어쨌든 하드 캡이 필요하다면(예: 관리되지 않는 cron 기반 위임의 비용을 제어하려는 경우) 설치별로 명시적으로 활성화하세요.

```yaml
delegation:
  child_timeout_seconds: 0     # default: 0 = no timeout
  # child_timeout_seconds: 1800  # opt-in hard cap (floor 30s)
```

양수 값은 각 자식에 하드 벽시계 제한을 적용하고, `0` 또는 음수 값은 이를 비활성화합니다.

구성된 제한에 도달한 경우 결과에는 구조화된 타임아웃 메타데이터가 오류 메시지와 함께 포함됩니다. `timeout_seconds`(구성된 제한), `timed_out_after_seconds`(실제 벽시계 시간), `timeout_phase`(자식이 첫 요청에 도달하지 못했으면 `before_first_llm_call`, 그 외에는 `after_llm_calls`)가 해당합니다. 타임아웃이 아닌 오류에서는 세 값 모두 `null`입니다.

:::tip 호출 0회 타임아웃 진단 덤프
하드 캡이 구성되어 있고 서브에이전트가 API 호출을 **0회** 수행한 채 타임아웃되면(대개 프로바이더에 연결할 수 없거나 인증에 실패했거나 도구 스키마가 거부된 경우), `delegate_task`는 `~/.hermes/logs/subagent-timeout-<session>-<timestamp>.log`에 구조화된 진단 정보를 기록합니다. 여기에는 서브에이전트의 구성 스냅샷, 자격 증명 확인 추적, 초기 오류 메시지, **모든** 활성 스레드의 스택 추적(자식 자체뿐 아니라)이 포함됩니다. 중첩 헬퍼 스레드를 기다리며 멈춘 자식은 전체 상황 없이는 느린 프로바이더와 구분할 수 없기 때문입니다.
:::

## 백그라운드 서브에이전트의 정체 감지

백그라운드 위임(`delegate_task(background=true)`)은 기본적으로 활성화되고 별도 구성 없이 사용할 수 있는 **진행 기반 정체 모니터**가 감시합니다. 벽시계 시간 초과와 달리, 얼마나 오래 실행되든 진행 중인 자식에는 절대 개입하지 않습니다.

모니터는 분리된 각 자식의 진행 신호(API 호출 횟수, 현재 도구, 마지막 활동 타임스탬프)를 샘플링합니다. 마지막 활동 타임스탬프는 **스트리밍되는 모든 토큰**, 도구 전환, API 호출 경계에서 갱신되므로 긴 응답을 스트리밍 중인 자식도 항상 살아 있는 것으로 간주됩니다.

1. **진행 중인 자식에는 절대 개입하지 않습니다.** 어떤 진행 신호라도 증가하면 시계가 초기화됩니다.
2. 진행이 완전히 멈춘 자식이 정체 임계값을 넘기면(유휴 상태 450초, 도구 안에서는 1200초 — 정상적으로 느린 터미널 명령과 웹 가져오기에 더 높은 한도를 적용) **인터럽트**하고 120초의 유예 기간을 부여합니다. 제때 풀려나는 자식은 부분 결과를 일반 완료 경로로 전달합니다.
3. 반환하지 않는 자식은 최종 상태가 `stalled`인 완료 이벤트로 강제 종료되어 소유 세션이 결과를 받게 되고, 비동기 슬롯이 새 작업을 위해 해제됩니다.

`stalled` 이벤트는 동기 경로의 타임아웃 필드와 대응하는 구조화된 메타데이터를 포함합니다. `stalled_after_quiet_seconds`, `stall_threshold_seconds`, `stall_phase`(`idle` / `in_tool`), `stall_grace_seconds`가 해당합니다.

이로써 멈춘 백그라운드 자식 때문에 프로세스를 재시작할 때까지 세션이 죽은 것처럼 보이던 오랜 실패 모드가 해결되었습니다. 근본 원인도 수정되었습니다. 수일간 게이트웨이를 실행한 뒤 첫 API 호출에서 자식이 멈추던 문제를 피하도록, 위임된 자식은 이제 중첩 워커 스레드가 아니라 자체 대화 스레드에서 OpenAI-wire API 요청을 인라인으로 실행합니다. 정체 모니터는 다른 모든 상황을 위한 안전망으로 남아 있습니다.

## 실행 중인 서브에이전트 모니터링 (`/agents`)

TUI에는 `/agents` 오버레이(`/tasks` 별칭)가 포함되어 있으며, 재귀적인 `delegate_task` 팬아웃을 일급 감사 화면으로 바꿉니다.

- 실행 중인 서브에이전트와 최근 완료된 서브에이전트의 실시간 트리 뷰(부모별 그룹화)
- 브랜치별 비용, 토큰, 접촉한 파일의 집계
- 종료 및 일시 중지 제어 — 형제 서브에이전트를 인터럽트하지 않고 특정 서브에이전트를 실행 중간에 취소
- 사후 검토: 부모에게 돌아온 후에도 각 서브에이전트의 턴별 기록을 단계별로 확인

클래식 CLI는 `/agents`를 텍스트 요약으로만 출력하고, 오버레이의 장점은 TUI에서 드러납니다. [TUI — 슬래시 명령](/user-guide/tui#slash-commands)을 참조하세요.

클래식 CLI와 모든 게이트웨이 플랫폼(Telegram, Discord, Slack, ...)에서 `/agents`는 각 실행 중인 자식에서 직접 샘플링한 **자식별 실시간 활동이 포함된 백그라운드 위임**도 나열합니다.

```
Background delegations: 1 running
- deleg_ab12cd34 · running · research the delegation stall monitor
  - child 1: 4 api calls · in web_search · active 12s ago
  - child 2: 7 api calls · between turns · active 3s ago
```

정체 모니터가 플래그를 지정한 위임은 `stalling · no progress 450s — interrupting`으로 표시되고, 오래 조용하지만 정상인 자식은 조용했던 시간을 표시하므로 한눈에 "느린 것"과 "멈춘 것"을 구분할 수 있습니다.

## 실행 중인 서브에이전트 조종

자식을 인터럽트하면 진행 중인 작업이 버려지며, 실제로는 단지 방향을 바꾸고 싶을 때가 많습니다.

### 부모 에이전트에서 (모델 대상)

부모 에이전트는 자식을 생성할 때 사용한 것과 같은 `delegate_task` 도구로 실행 중인 자식을 오케스트레이션합니다. 별도의 제어 도구는 없습니다.

```json
{"action": "list"}
{"action": "steer", "subagent_id": "sa-0-1a2b3c4d", "message": "focus on pricing instead"}
{"action": "stop",  "subagent_id": "sa-0-1a2b3c4d"}
```

- `list`는 대화의 실시간 자식인 `subagent_id`, 목표, 상태, `running_seconds`, `accepting_steer`, 실시간 트랜스크립트 경로를 반환합니다. ID는 생성 디스패치 응답에서도 `subagent_ids`로 반환됩니다.
- `steer`는 실행 중인 자식을 멈추지 않고 방향 수정을 대기열에 넣습니다(전달 의미는 아래 참조).
- `stop`은 다음 반복 경계에서 자식을 조기에 종료하며, 부분 결과는 정상 완료 메시지로 대화에 다시 들어옵니다.

제어 작업은 현재 턴에서 동기적으로 실행되고(백그라운드로 전환되지 않음), 호출자 자신의 생성 트리로 범위가 제한됩니다. 한 대화는 다른 세션의 자식을 보거나 제어할 수 없습니다. 또한 턴별 서브에이전트 생성 한도를 소비하지 않으므로 한도에 도달한 뒤에도 `stop`은 계속 작동합니다.

### TUI / 게이트웨이에서 (세션 대상)

`tools/delegate_tool.py`의 `steer_subagent(subagent_id, text)`는 `interrupt_subagent()`의 방향 전환 측 미러입니다. [`/steer`](/reference/slash-commands)와 동일한 메커니즘으로 실행 중인 자식에 텍스트를 대기열에 넣습니다. 텍스트는 다음 반복 경계에서 자식의 마지막 도구 결과에 추가되며, 진행 중인 도구 호출은 중단되지 않고 자식은 이를 대역 외 사용자 메시지로 받습니다. 프로그래밍 방식의 호스트는 `subagent.interrupt`와 나란히 있는 세션 범위의 `subagent.steer` 게이트웨이 RPC를 통해 접근합니다.

```json
{"method": "subagent.steer", "params": {"session_id": "owning-ui-session", "subagent_id": "sa-0-1a2b3c4d", "text": "focus on pricing instead"}}
```

서브에이전트 ID는 `delegation.status`(또는 `list_active_subagents()`)에서 가져옵니다. `subagent.interrupt`도 같은 곳에서 ID를 가져옵니다. 게이트웨이는 자식을 생성한 정확히 동일한 실시간 UI/게이트웨이 세션에서만 조종을 허용합니다. 누락되었거나, 다른 세션이거나, 모호하거나, 오래되어 재사용된 세션 ID는 거부됩니다. 전역 서브에이전트 ID를 알고 있다는 사실은 권한이 아닙니다. 직접 프로세스 내 호출자는 의도적으로 범위가 지정되지 않은 헬퍼 계약을 유지합니다.

**대기열에 들어갔다고 전달된 것은 아니지만, 결코 가짜 성공이 아닙니다.** `"queued"` 응답은 자식의 완료 경계 전에 텍스트가 수락되었다는 뜻이지, 자식이 이를 보았다는 뜻은 아닙니다. 수락과 완료는 동기화됩니다. 자식이 여전히 텍스트를 소비할 수 있거나, 정확한 텍스트가 `pending_steer`로 결과에 반영됩니다. 종료 후 호출은 `"rejected"`를 반환합니다. 자식이 조종을 수락했지만 이미 최종 답변을 생성한 경우, 부모가 받는 완료 항목에는 `missed_steer`가 유지되고 요약에 다음과 같은 메모가 추가됩니다.

```
[steer did not land — the subagent finished before it could be delivered: focus on pricing instead]
```

따라서 부모(또는 이를 조종하는 운영자)는 조종된 자식과 이전 지시로 완료한 자식을 구분할 수 있고, 실제로 전달되었다고 믿는 대신 후속 조치로 지침을 다시 보낼 수 있습니다.

## 실시간 트랜스크립트

모든 `delegate_task` 디스패치는 작업별 **추가 전용, 사람이 읽을 수 있는 로그**도 생성합니다. 따라서 통합 요약을 기다리는 대신 사용자(또는 부모 에이전트)가 서브에이전트의 작업을 실시간으로 지켜볼 수 있습니다.

```
<hermes_home>/cache/delegation/live/<delegation_id>/task-<n>.log
```

디스패치 응답에는 `live_transcripts`로 경로가 포함되며, 파일은 디스패치 시 미리 생성되므로 즉시 사용할 수 있습니다.

```bash
tail -f ~/.hermes/cache/delegation/live/deleg_ab12cd34/task-0.log
```

각 줄에는 타임스탬프가 표시되고 자식의 어시스턴트 텍스트, 사고 단편, 도구 호출(`-> tool_name({args})`), 도구 결과, 최종 상태 마커가 기록됩니다. 같은 디렉터리의 `manifest.json`은 배치(목표, 작업 수, 작업별 상태)를 설명합니다. 로그는 완료 후에도 유지되며 요약과 함께 완전한 운영 기록 역할을 합니다. `cache/delegation` 아래에 있으므로 원격 터미널 백엔드(Docker/Modal/SSH)에서도 읽을 수 있습니다.

## 깊이 제한 및 중첩 오케스트레이션

기본적으로 위임은 **평면적**입니다. 부모(깊이 0)가 자식(깊이 1)을 생성하고, 해당 자식은 더 이상 위임할 수 없습니다. 이는 재귀 위임이 폭주하는 것을 방지합니다.

다단계 워크플로(조사 → 합성 또는 하위 문제에 대한 병렬 오케스트레이션)의 경우 부모는 자체 워커를 위임할 수 있는 오케스트레이터 자식을 생성할 수 있습니다.

```python
delegate_task(
    goal="Survey three code review approaches and recommend one",
    role="orchestrator",  # Allows this child to spawn its own workers
    context="...",
)
```

- `role="leaf"`(기본값): 자식은 더 이상 위임할 수 없습니다. 평면 위임 동작과 동일합니다.
- `role="orchestrator"`: 자식은 `delegation` 도구 세트를 유지합니다. `delegation.max_spawn_depth`로 제한되며, 기본값 **1**(평면)에서는 `role="orchestrator"`가 아무 효과가 없습니다. `max_spawn_depth`를 2로 높이면 오케스트레이터 자식이 리프 손자 자식을 생성하고, 3 이상이면 더 깊은 트리를 만들 수 있습니다. 상한은 없으며 실제 제한은 비용입니다.
- `delegation.orchestrator_enabled: false`: `role` 매개변수와 관계없이 모든 자식을 `leaf`로 강제하는 전역 비상 차단 스위치입니다.

**비용 경고:** `max_spawn_depth: 3` 및 `max_concurrent_children: 3`에서는 트리가 3×3×3 = 27개의 동시 리프 에이전트에 도달할 수 있습니다. 각 추가 레벨은 비용을 배가하므로 `max_spawn_depth`는 의도적으로 높이세요.

## 수명 및 내구성

:::warning 백그라운드 완료의 내구성은 내구성 있는 실행이 아닙니다
세션이 나중에 전달을 지원하는 경우 최상위 모델 대상 `delegate_task` 호출은 자동으로 백그라운드에서 실행됩니다. Hermes는 즉시 핸들을 반환하고 자식 또는 배치가 끝난 후 결과를 대화에 다시 넣습니다. 오케스트레이터 서브에이전트는 반환하기 전에 현재 턴에서 워커를 기다립니다. 반환 전에 결과를 합성해야 하기 때문입니다. 분리된 결과를 나중에 전달할 수 없는 상태 비저장 요청/응답 엔드포인트는 동기 실행으로 대체됩니다.

- 일반적인 후속 메시지는 백그라운드 자식을 취소하지 않습니다. `/stop`은 실행 중인 백그라운드 위임을 취소하고, 소유 세션을 닫거나 초기화하면 활성 자식이 버려집니다.
- 명시적인 세션 닫기/초기화는 해당 세션의 백그라운드 자식을 인터럽트합니다. 게이트웨이가 소유한 세션의 TUI 뷰어를 닫는 것은 게이트웨이의 작업을 종료하지 않습니다.
- Hermes 프로세스를 재시작해도 실행 중인 자식은 재개되지 않습니다. Hermes가 어느 쪽에서 어떤 부작용을 일으켰는지 증명할 수 없으므로 시도 결과는 `unknown`이 됩니다.
- 재시작 전에 완료했지만 결과가 전달되지 않은 자식은 복원되어 소유 세션의 일반 확인 절차를 통해 다시 라우팅됩니다.
- 취소된 자식은 구조화된 결과(`status="interrupted"`, `exit_reason="interrupted"`)를 반환하지만 부모도 인터럽트된 상태이므로 이 결과가 사용자에게 보이는 답변에 들어가지 않는 경우가 많습니다.

세션 종료 또는 프로세스 재시작을 견뎌야 하는 **내구성 있는 실행**에는 다음을 사용하세요.

- `cronjob` (`action="create"`) — 별도의 에이전트 실행을 예약하며 부모 턴의 인터럽트 영향을 받지 않습니다.
- `terminal(background=True, notify_on_complete=True)` — 에이전트가 다른 작업을 하는 동안에도 계속 실행되는 장시간 셸 명령입니다.
:::

## 핵심 속성

- 각 서브에이전트는 부모와 분리된 **자체 터미널 세션**을 갖습니다.
- 서브에이전트는 부모의 활성화된 도구 세트를 상속하며, 모델은 호출마다 이를 선택하거나 확장할 수 없습니다.
- **중첩 위임은 선택 사항입니다.** `role="orchestrator"` 자식만 더 위임할 수 있고, 기본값 1(평면)에서 `max_spawn_depth`를 높인 경우에만 가능합니다. `orchestrator_enabled: false`로 전역 비활성화할 수 있습니다.
- 리프 서브에이전트는 `delegate_task`, `clarify`, `memory`, `send_message`, `cronjob`을 호출할 수 없습니다. 오케스트레이터 서브에이전트는 `delegate_task`를 유지하지만 다른 차단 항목은 그대로 적용됩니다. 두 역할 모두 `execute_code`(프로그래밍 방식의 도구 호출)를 유지하므로 추론 반복을 소모하지 않고 기계적인 작업을 일괄 처리할 수 있습니다.
- **취소는 소유권을 따릅니다.** `/stop` 또는 소유 세션을 닫거나 초기화하면 백그라운드 자식이 취소되고, 오케스트레이터 아래의 동기식 후손은 부모의 인터럽트 상태를 따릅니다.
- 부모 컨텍스트에는 최종 요약만 들어가므로 토큰 사용량이 효율적으로 유지됩니다.
- 서브에이전트는 부모의 **API 키, 프로바이더 구성, 자격 증명 풀**을 상속합니다(속도 제한 시 키 교체 가능).

## 워크트리 격리

기본적으로 서브에이전트는 부모의 작업 디렉터리를 공유합니다. 조사 및 읽기 위주의 작업에는 적합하지만, 같은 저장소를 편집하는 병렬 자식은 충돌할 수 있습니다. `delegation.worktree_isolation: true`를 설정하면 각 자식에 현재 저장소의 `HEAD`에서 분기한 자체 git 워크트리를 제공합니다(내부적으로 Muse Code의 `--subagent-worktree-isolation`에서 영감을 받음).

```yaml
delegation:
  worktree_isolation: true   # default: false
```

격리를 활성화하면 다음과 같습니다.

- 각 자식은 자체 브랜치 `hermes-subagent/subagent-<id>`에 있는 `<repo>/.worktrees/subagent-<id>`에서 터미널을 시작하며, 목표 메시지는 해당 위치에서 작업하고 커밋하라고 지시합니다.
- 부모 체크아웃은 변경되지 않으며, 자식끼리 서로의 편집 내용을 덮어쓸 수 없습니다.
- 자식이 끝나면 결과 항목에 `path`, `branch`, `commits`(기준보다 앞선 커밋), `dirty`를 보고하는 `worktree` 필드가 추가됩니다. 부모가 각 브랜치를 검토하거나 병합합니다(`git log <branch>`, `git merge <branch>`).
- 커밋이 없고 트리가 깨끗한 워크트리는 자동으로 정리됩니다(`pruned: true`). 작업이 남아 있는 것은 유지됩니다.

범위는 선택 사항이며 git 저장소와 로컬 터미널 백엔드에만 적용됩니다. git이 아닌 디렉터리, docker/ssh/modal 백엔드, 또는 워크트리 생성 실패 시 설정은 조용히 현재의 공유 작업 디렉터리 동작으로 저하되며 오류가 발생하지 않습니다.

## 위임과 execute_code 비교

| 기준 | delegate_task | execute_code |
|--------|--------------|-------------|
| **추론** | 완전한 LLM 추론 루프 | Python 코드 실행만 수행 |
| **컨텍스트** | 격리된 새로운 대화 | 대화 없음, 스크립트만 실행 |
| **도구 액세스** | 차단되지 않은 모든 도구와 추론 | RPC를 통한 7개 도구, 추론 없음 |
| **병렬성** | 기본적으로 동시 서브에이전트 3개(구성 가능) | 단일 스크립트 |
| **최적 용도** | 판단, 추론 또는 여러 단계의 문제 해결이 필요한 복잡한 작업 | 기계적인 다단계 파이프라인 |
| **토큰 비용** | 더 높음(완전한 LLM 루프) | 더 낮음(stdout만 반환) |
| **사용자 상호작용** | 없음(서브에이전트는 질문할 수 없음) | 없음 |

**경험칙:** 하위 작업에 추론, 판단 또는 여러 단계의 문제 해결이 필요하면 `delegate_task`를 사용하세요. 기계적인 데이터 처리나 스크립트 기반 파이프라인이 필요하면 `execute_code`를 사용하세요.

## 구성

```yaml
# In ~/.hermes/config.yaml
delegation:
  max_iterations: 50                        # Max turns per child (default: 50)
  # max_concurrent_children: 3              # Parallel children per batch (default: 3)
  # worktree_isolation: false               # Give each child its own git worktree (see Worktree Isolation above)
  # max_spawn_depth: 1                      # Tree depth (floor 1, no ceiling, default 1 = flat). Raise to 2 to allow orchestrator children to spawn leaves; 3+ for deeper trees.
  # orchestrator_enabled: true              # Disable to force all children to leaf role.
  model: "google/gemini-3-flash-preview"             # Optional provider/model override
  provider: "openrouter"                             # Optional built-in provider
  api_mode: anthropic_messages                       # optional; auto-detected from base_url for anthropic_messages endpoints

# Or use a direct custom endpoint instead of provider:
delegation:
  model: "qwen2.5-coder"
  base_url: "http://localhost:1234/v1"
  api_key: "local-key"
  # api_mode: "anthropic_messages"  # Optional. Wire protocol override for base_url ("chat_completions", "codex_responses", or "anthropic_messages"). Empty = auto-detect from URL (e.g. /anthropic suffix). Set explicitly for endpoints the heuristic can't classify (Azure AI Foundry, MiniMax, Zhipu GLM, LiteLLM proxies, …).
```

`base_url`이 Anthropic 호환 엔드포인트를 가리키면(예: `/anthropic`으로 끝나는 경로, Azure Foundry Claude 경로, MiniMax `/anthropic` 프록시) `api_mode`는 `anthropic_messages`로 자동 감지되므로 직접 설정할 필요가 없습니다. 자동 감지 추정이 잘못되는 드문 경우에는 `api_mode`를 명시적으로 설정하세요.

:::tip
에이전트는 작업 복잡도에 따라 위임을 자동으로 처리합니다. 명시적으로 위임을 요청할 필요가 없습니다. 적절할 때 자동으로 위임합니다.
:::

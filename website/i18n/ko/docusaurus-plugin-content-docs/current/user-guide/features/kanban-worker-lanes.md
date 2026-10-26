# Kanban 워커 레인

**워커 레인**은 kanban 디스패처가 작업을 라우팅할 수 있는 프로세스의 유형입니다. 각 레인에는 정체성(assignee 문자열), 생성 메커니즘, 그리고 생성된 작업을 처리해야 하는 계약이 있습니다.

이 페이지는 두 독자를 위한 계약입니다:

- 보드에 어떤 레인을 연결할지(어떤 프로필을 만들고 어떤 assignee를 사용할지) 선택하는 **운영자**.
- 새로운 레인 형태(Codex / Claude Code / OpenCode를 감싸는 CLI 워커, 컨테이너화된 리뷰 워커, API를 통해 작업을 가져오는 비-Hermes 서비스)를 추가하려는 **플러그인 / 통합 작성자**.

워커 코드 자체, 즉 레인 내부에서 실행되는 에이전트를 작성하는 경우, kanban 수명 주기와 참고 정보가 워커의 시스템 프롬프트에 자동으로 주입됩니다([`agent/prompt_builder.py`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py)의 `KANBAN_GUIDANCE` 블록).

## 계층

```text
Hermes Kanban  =  canonical task lifecycle + audit trail
Worker lane    =  implementation executor for one assigned card
Reviewer       =  human or human-proxy that gates "done"
GitHub PR      =  upstreamable artifact (optional, for code lanes)
```

Hermes Kanban은 수명 주기의 진실을 소유합니다 — `ready` → `running` → `review` / `blocked` / `done` / `archived`. 워커 레인은 작업을 실행하지만 그 진실을 소유하지 않습니다. 워커가 하는 모든 일은 `kanban_*` 도구(또는 비-Hermes 외부 워커의 경우 API)를 통해 kanban 커널로 돌아갑니다. 리뷰어는 "코드 변경 작성"에서 "작업 완료"로의 전환을 승인합니다.

## 레인이 제공하는 것

kanban 워커 레인이 되려면 통합이 다음 세 가지를 제공해야 합니다:

### 1. Assignee 문자열

디스패처는 `task.assignee`를 Hermes 프로필 이름(기본 레인 형태) 또는 등록된 비-생성 가능 식별자(플러그인 레인 형태 — 아래 [외부 CLI 워커 레인 추가](#adding-an-external-cli-worker-lane) 참조)와 대조합니다. assignee가 확인되지 않는 작업은 `skipped_nonspawnable` 이벤트와 함께 `ready` 상태에 남으므로 보드 운영자가 수정할 수 있습니다. 작업이 조용히 삭제되거나 임의의 폴백으로 실행되지는 않습니다.

### 2. 생성 메커니즘

Hermes 프로필 레인의 경우 디스패처의 `_default_spawn`가 다음을 실행합니다.

`hermes -p <assignee> chat -q <prompt>`

(`hermes` 셸이 `$PATH`에 없을 때는 이에 상응하는 모듈 형식) 작업에 고정된 워크스페이스 안에서 실행하며, 다음 환경 변수를 설정합니다:

| 변수 | 전달하는 값 |
|---|---|
| `HERMES_KANBAN_TASK` | 워커가 작업 중인 task id |
| `HERMES_KANBAN_DB` | 보드별 SQLite 파일의 절대 경로 |
| `HERMES_KANBAN_BOARD` | board slug |
| `HERMES_KANBAN_WORKSPACES_ROOT` | 보드 워크스페이스 트리의 루트 |
| `HERMES_KANBAN_WORKSPACE` | *이* 작업의 워크스페이스 절대 경로 |
| `HERMES_KANBAN_RUN_ID` | 현재 실행의 id(수명 주기 게이트용) |
| `HERMES_KANBAN_CLAIM_LOCK` | claim lock 문자열(`<host>:<pid>:<uuid>`) |
| `HERMES_PROFILE` | 워커 자신의 프로필 이름(`kanban_comment` 작성자 표시용) |
| `HERMES_TENANT` | 작업에 tenant가 있는 경우 tenant namespace |

비-Hermes 레인(플러그인을 통해 등록)의 경우 플러그인이 자체 `spawn_fn` callable을 제공하며, 이 callable은 `task`, `workspace`, `board`를 받고 크래시 감지를 위한 선택적 pid를 반환합니다.

### 3. 수명 주기 종료자

모든 claim은 정확히 다음 중 하나로 끝나야 합니다:

- `kanban_complete(summary=..., metadata=...)` — 작업이 성공하고 상태가 `done`으로 전환됩니다.
- `kanban_request_review(summary=..., metadata=..., reviewer=...)` — 동일 카드의 구현이 완료되고 일급 리뷰 단계로 들어가며 상태가 `review`로 전환됩니다. `kanban.review_dispatch`가 비활성화되지 않은 경우 디스패처는 번들된 `sdlc-review` 스킬을 로드합니다. 리뷰어는 `kanban_complete`로 승인하거나, `kanban_request_changes`로 실행 가능한 재작업을 반환하거나, `kanban_block`으로 실제 외부 차단을 에스컬레이션합니다.
- `kanban_block(reason=...)` — 작업이 사람의 입력을 기다리며 상태가 `blocked`로 전환됩니다. `kanban_unblock`이 실행되면 디스패처가 워커를 다시 생성합니다.
- 도구 호출 없이 워커 프로세스가 종료됩니다. 커널은 프로세스를 회수하고 `crashed`(PID가 종료됨), `gave_up`(연속 실패 차단기가 작동함), 또는 `timed_out`(`max_runtime` 초과)을 기록합니다. 이는 실패 경로이며, 정상적인 워커는 여기서 종료되지 않습니다.

kanban 커널은 각 실행을 정확히 하나의 종료 동작이 끝내도록 강제합니다. 종료 도구를 호출하지 않고 정상 종료하는 워커는 크래시로 처리됩니다.

## 출력과 리뷰 인계

코드 변경 작업에서는 작업 그래프에 인코딩된 리뷰 모델을 선택합니다:

- **동일 카드 리뷰:** `kanban_request_review(summary=..., metadata=..., reviewer=...)`를 호출합니다. 작업은 `review`에 들어가며 블록 연속 발생 횟수 집계에는 영향을 주지 않습니다. 디스패처는 기본적으로 번들된 `sdlc-review` 스킬로 작업을 claim합니다. 리뷰어는 `kanban_complete`로 승인하거나, `kanban_request_changes(reason=...)`를 호출해 리뷰 실행을 종료하고 작업을 원래 구현자에게 되돌리거나, 실제 외부 에스컬레이션에 한해서만 차단할 수 있습니다.
- **미리 생성된 후속 리뷰/QA/릴리스 카드:** `kanban_show`에서 하위 ID를 확인한 다음 `kanban_show(task_id=...)`로 해당 카드를 살펴보고 종료 동작을 선택합니다. 하위 카드가 구현 단계의 후속 리뷰/QA/릴리스 단계인 경우 구현 단계에서 `kanban_complete`를 호출합니다. 이 단계는 부모가 `done`/`archived`가 될 때까지 승격될 수 없습니다. 동일 카드 리뷰를 추가로 요청하지 말고 `review-required:`로 부모를 고정 차단하지도 마세요 — 어느 쪽이든 후속 레인을 고립시키거나 중복시킵니다.
- **사람 전용 보드:** `kanban.review_dispatch: false`로 설정합니다. 그러면 사람이 승인하거나 대시보드에서 `reopen-review`를 사용해 작업을 `ready`/`todo`로 되돌릴 때까지 작업이 `review`에 남을 수 있습니다.

두 리뷰 모델 모두 수명 주기 전환 자체에 구조화된 인계 내용을 담습니다. `summary`나 `metadata`에 비밀, 토큰, 원시 PII를 넣지 마세요. 실행 행은 영구적으로 보존됩니다.

주입되는 `KANBAN_GUIDANCE`에는 두 그래프 형태, `kanban_complete`, 동일 카드 리뷰 루프, 실제 차단을 위한 `kanban_block`이 모두 설명되어 있습니다.

## 로그와 감사 추적

디스패처는 작업별 워커 stdout/stderr를 `<board-root>/logs/<task_id>.log`에 기록합니다. kanban 메타데이터에서 로그를 감사할 수 있습니다:

- `task_runs` 행에는 `log_path`, 종료 코드(가능한 경우), summary, metadata가 담깁니다.
- `task_events` 행에는 모든 상태 전환(`promoted`, `claimed`, `heartbeat`, `completed`, `blocked`, `review_requested`, `changes_requested`, `review_reopened`, `gave_up`, `crashed`, `timed_out`, `reclaimed`, `claim_extended`)이 담깁니다.
- `kanban_show`는 두 정보를 모두 반환하므로 리뷰어(또는 후속 워커)는 대시보드에 접근하지 않고도 작업의 전체 이력을 확인할 수 있습니다.

대시보드는 summary, metadata 블록, 종료 상태 배지와 함께 실행 이력을 표시합니다. CLI 사용자는 `hermes kanban tail <task_id>`로 실시간 내용을 보거나 `hermes kanban runs <task_id>`로 과거 시도 목록을 확인할 수 있습니다.

## 기존 레인 형태

### Hermes 프로필 레인(기본)

오늘날 모든 kanban 워커가 취하는 형태입니다. assignee는 프로필 이름이고, 디스패처는 `hermes -p <profile>`을 생성하며, 워커에는 `KANBAN_GUIDANCE` 시스템 프롬프트 블록이 자동으로 주입되고, 실행을 종료하기 위해 `kanban_*` 도구를 사용합니다. 프로필을 정의하는 것 외에는 설정이 필요하지 않습니다.

플릿용 프로필을 만들 때는 오케스트레이터가 라우팅할 역할에 맞는 이름을 선택하세요. 오케스트레이터가 있는 경우 `hermes profile list`를 통해 프로필 이름을 찾으며, 시스템이 가정하는 고정된 목록은 없습니다(오케스트레이터 측 계약은 주입되는 `KANBAN_GUIDANCE`의 일부입니다).

### 오케스트레이터 프로필 레인

프로필 레인의 특수화 형태입니다. 오케스트레이터는 구현을 위한 `terminal` / `file` / `code` / `web`은 제외하고 `kanban`을 포함하는 도구 세트를 가진 Hermes 프로필입니다. 오케스트레이터의 역할은 `kanban_create` + `kanban_link`를 통해 상위 수준의 목표를 하위 작업으로 분해한 뒤 물러나는 것입니다. 오케스트레이터 스킬에는 유혹을 막기 위한 규칙이 담겨 있습니다.

## 외부 CLI 워커 레인 추가

비-Hermes CLI 도구(Codex CLI, Claude Code CLI, OpenCode CLI, 로컬 코딩 모델 실행기 등)를 kanban 워커 레인으로 연결하는 것은 *아직 정식 지원 경로가 아닙니다*. 디스패처의 생성 함수는 플러그 방식(`spawn_fn`은 `dispatch_once`의 매개변수)이며, 플러그인은 비-Hermes assignee에 자체 `spawn_fn`을 등록할 수 있습니다. 그러나 CLI의 종료 코드를 `kanban_complete` / `kanban_block` 호출로 감싸고, CLI의 워크스페이스/샌드박스 규칙을 디스패처의 `HERMES_KANBAN_WORKSPACE` 환경에 매핑하며, 인증과 CLI별 정책을 처리하는 주변 통합 작업은 여전히 통합별 설계 과제입니다.

CLI 레인 추가를 고려 중이라면 사용하려는 구체적인 CLI와 활성화하려는 워크플로를 설명하는 이슈를 열어 주세요. 위 계약은 이러한 레인이 만족해야 하는 제약 조건이며, 구현 형태(각 CLI마다 하나의 플러그인인지, 설정으로 매개변수화하는 일반 CLI 실행기 플러그인인지는)는 열려 있습니다.

이 문제의 역사적 이슈는 [#19931](https://github.com/NousResearch/hermes-agent/issues/19931)이며, Codex에 특화되었지만 병합되지 않고 종료된 PR은 [#19924](https://github.com/NousResearch/hermes-agent/pull/19924)입니다. 두 문서 모두 최초의 아키텍처 제안을 설명하지만 실행기는 반영되지 않았습니다.

## 디스패처가 처리하는 실패 모드

레인 작성자가 다음을 다시 구현할 필요가 없도록 디스패처가 처리합니다:

- **오래된 claim TTL** — claim한 뒤 heartbeat / 완료 / 차단을 전혀 하지 않는 워커는 `DEFAULT_CLAIM_TTL_SECONDS`(기본 15분)가 지나면 회수됩니다. 단, 워커 프로세스가 실제로 종료된 경우에만 해당합니다. 실행 중인 워커(도구 호출 없이 20분 이상 걸리는 느린 모델의 LLM 호출 등)는 종료되는 대신 claim이 *연장*됩니다. 죽은 PID만 회수됩니다.
- **크래시한 워커** — 호스트 로컬 PID가 사라진 워커는 `detect_crashed_workers`가 감지하고 회수합니다. 작업의 `consecutive_failures`가 증가하며 차단기가 작동하면 자동으로 차단될 수 있습니다.
- **실행 수준 재시도** — 작업이 재시도될 때(차단 후, 크래시 후, 회수 후), 워커는 종료 도구의 `expected_run_id` 매개변수를 사용해 자신의 실행이 이미 대체되었는지 빠르게 확인할 수 있습니다.
- **작업별 최대 실행 시간** — `task.max_runtime_seconds`는 PID가 살아 있는지와 무관하게 실행당 wall-clock 시간을 엄격히 제한합니다. 살아 있는 PID의 연장 기능이 계속 실행하게 만들 수 있는 실제 교착 워커를 포착합니다.
- **고립 작업 감지** — assignee가 `kanban.stranded_threshold_seconds`(기본 30분) 안에 claim을 생성하지 않는 `ready` 작업은 `hermes kanban diagnostics`에서 `stranded_in_ready` 경고로 표시됩니다. 심각도는 임계값의 2배에서 error, 6배에서 critical로 올라갑니다. 오타가 난 assignee, 삭제된 프로필, 중단된 외부 워커 풀을 하나의 신호로 포착하며, 정체성에 무관하고 보드별 허용 목록을 관리할 필요가 없습니다.
- **기존 리뷰 의존성 교착** — 하나 이상의 직접 하위 작업이 `todo`에서 의존성으로 게이트된 상태인데 부모가 `review-required:`로 고정 차단되면 즉시 `review_dependency_deadlock` 오류가 생성됩니다. 진단은 읽기 전용입니다. 완료된 단계를 완료 처리하거나 잘못된 연결을 해제하도록 제안하지만 사용자 차단을 자동으로 제거하지는 않습니다.

## 관련 문서

- [Kanban 개요](./kanban) — 사용자를 위한 소개입니다.
- [Kanban 튜토리얼](./kanban-tutorial) — 대시보드를 열어 두고 따라 하는 안내입니다.
- [`KANBAN_GUIDANCE`](https://github.com/NousResearch/hermes-agent/blob/main/agent/prompt_builder.py) — 모든 kanban 워커의 시스템 프롬프트에 주입되는 워커 + 오케스트레이터 수명 주기입니다.

# Kanban 워커 레인 (Kanban worker lanes)

**워커 레인(worker lane)** 은 칸반(Kanban) 디스패처가 작업을 라우팅할 수 있는 프로세스 클래스입니다. 각 레인은 ID(assignee 문자열), 생성(spawn) 메커니즘, 그리고 생성된 후 작업으로 무엇을 해야 하는지에 대한 계약(contract)을 갖습니다.

이 문서는 그 계약에 대한 내용입니다. 두 종류의 독자를 위해 작성되었습니다:

- **운영자(Operators):** 보드에 어떤 레인을 연결할지(어떤 프로필을 생성할지, 어떤 담당자를 사용할지) 결정하는 사람.
- **플러그인 / 통합 작성자:** 새로운 레인 형태를 추가하려는 사람 (Codex / Claude Code / OpenCode를 래핑하는 CLI 워커, 컨테이너화된 리뷰 워커, API를 통해 작업을 가져오는 비(非) Hermes 서비스 등).

레인 *내부*에서 실행되는 에이전트인 워커 코드 자체를 작성하는 중이라면, [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) 스킬 문서에서 더 깊은 절차적 세부 사항을 확인할 수 있습니다.

## 계층 구조 (The hierarchy)

```text
Hermes Kanban  =  정규화된 작업 수명 주기(lifecycle) + 감사 추적(audit trail)
워커 레인      =  할당된 카드 하나에 대한 구현 실행자
리뷰어         =  "완료(done)" 상태를 게이트하는 사람 또는 사람 대리자
GitHub PR      =  업스트림 가능한 아티팩트 (선택 사항, 코드 레인용)
```

Hermes Kanban은 수명 주기(`ready` → `running` → `blocked` / `done` / `archived`)에 대한 진실(truth)을 소유합니다. 워커 레인은 작업을 실행하지만 그 진실을 결코 소유하지 않습니다. 워커가 하는 모든 일은 `kanban_*` 도구를 통해 (또는 비 Hermes 외부 워커의 경우 API를 통해) 칸반 커널로 다시 흘러갑니다. 리뷰어는 "코드 변경 작성 완료"에서 "작업 완료"로의 전환을 제어(gate)합니다.

## 레인이 제공하는 것

칸반 워커 레인이 되려면 통합(integration)에서 다음 세 가지를 제공해야 합니다:

### 1. 할당자 문자열 (An assignee string)

디스패처는 `task.assignee`를 Hermes 프로필 이름(기본 레인 형태) 또는 등록된 생성 불가능한 식별자(플러그인 레인 형태 — 아래의 [외부 CLI 워커 레인 추가하기](#adding-an-external-cli-worker-lane) 참조)와 대조하여 일치시킵니다. 담당자가 확인되지 않는 작업은 보드 운영자가 수정할 수 있도록 `skipped_nonspawnable` 이벤트와 함께 `ready` 상태로 남겨집니다. 조용히 삭제되거나 임의의 폴백(fallback)에 의해 실행되지 않습니다.

### 2. 생성 메커니즘 (A spawn mechanism)

Hermes 프로필 레인의 경우, 디스패처의 `_default_spawn`은 작업의 고정된 작업 공간(pinned workspace) 내에서 다음과 같은 환경 변수를 설정한 후 `hermes -p <assignee> chat -q <prompt>`를 실행합니다 (`hermes` 심(shim)이 `$PATH`에 없는 경우 동등한 모듈 형식 사용).

| 변수 | 포함 내용 |
|---|---|
| `HERMES_KANBAN_TASK` | 워커가 작동 중인 작업 ID |
| `HERMES_KANBAN_DB` | 보드별 SQLite 파일의 절대 경로 |
| `HERMES_KANBAN_BOARD` | 보드 슬러그 |
| `HERMES_KANBAN_WORKSPACES_ROOT` | 보드 작업 공간 트리의 루트 |
| `HERMES_KANBAN_WORKSPACE` | *이* 작업의 작업 공간에 대한 절대 경로 |
| `HERMES_KANBAN_RUN_ID` | 현재 실행의 ID (수명 주기 게이트용) |
| `HERMES_KANBAN_CLAIM_LOCK` | 소유권 잠금 문자열 (`<host>:<pid>:<uuid>`) |
| `HERMES_PROFILE` | 워커 자신의 프로필 이름 (`kanban_comment` 작성자 속성용) |
| `HERMES_TENANT` | 작업에 테넌트 네임스페이스가 있는 경우 해당 네임스페이스 |

비 Hermes 레인(플러그인을 통해 등록됨)의 경우 플러그인은 `task`, `workspace`, `board`를 받고 충돌 감지를 위해 선택적인 PID를 반환하는 자체 `spawn_fn` 콜러블(callable)을 제공합니다.

### 3. 수명 주기 종료자 (A lifecycle terminator)

모든 클레임(claim)은 정확히 다음 중 하나로 끝나야 합니다:

- `kanban_complete(summary=..., metadata=...)` — 작업이 성공하고 상태가 `done`으로 바뀝니다.
- `kanban_block(reason=...)` — 작업이 사람의 입력을 기다리고 상태가 `blocked`로 바뀝니다. `kanban_unblock`이 실행되면 디스패처가 다시 생성합니다.
- 워커 프로세스가 도구 호출 없이 종료됩니다. 커널은 이를 회수하고 `crashed`(PID 종료) 또는 `gave_up`(연속 실패 차단기 작동) 또는 `timed_out`(max_runtime 초과)을 내보냅니다. 이는 실패 경로입니다. 정상적인 워커는 여기서 끝나지 않습니다.

칸반 커널은 이들 중 정확히 하나가 각 실행을 종료하도록 강제합니다. 둘 중 어느 것도 호출하지 않고 정상적으로 종료되는 워커는 충돌한 것으로 간주됩니다.

## 출력 및 리뷰 필수 규칙 (Outputs and the review-required convention)

코드를 변경하는 대부분의 작업의 경우 워커가 완료되는 순간 작업이 진정으로 *완료*되는 것은 아니며 사람의 리뷰어가 필요합니다. 칸반 커널은 이러한 구분을 강제하지 않습니다("코드를 변경하는 작업"은 모호하며 모든 코드 워커에서 완료 대신 차단을 강제하면 리뷰를 원하지 않는 흐름이 중단될 수 있습니다). 이는 그 위에 계층화된 규칙(convention)입니다.

- **완료 대신 차단(Block instead of complete):** 대시보드 / `hermes kanban show`에 리뷰 대기 중인 행이 표시되도록 `reason` 접두사로 `review-required: `를 사용합니다.
- **먼저 구조화된 메타데이터를 `kanban_comment`에 드롭(Drop structured metadata into a `kanban_comment` first):** `kanban_block`은 사람이 읽을 수 있는 `reason`만 전달하기 때문입니다. 댓글은 내구성이 있는 주석 채널입니다. 감사 관련 모든 필드(changed_files, tests_run, diff_path 또는 PR url, decisions)는 여기에 속합니다.
- **리뷰어는 승인하고 차단 해제(approves and unblocks)** (후속 조치를 위한 댓글 스레드와 함께 워커가 다시 생성됨) 또는 다른 댓글을 통해 **변경을 요청**합니다 (다음 워커 실행에서 `kanban_show` 컨텍스트의 일부로 확인).

[`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) 스킬에는 `kanban_complete`(진정한 종료 작업 — 오타 수정, 문서 변경, 연구 보고서) 및 `review-required` 차단 패턴 모두에 대한 실제 사례가 있습니다.

## 로그 및 감사 추적 (Logs and audit trail)

디스패처는 작업별 워커의 stdout/stderr를 `<board-root>/logs/<task_id>.log`에 기록합니다. 로그는 칸반 메타데이터에서 감사할 수 있습니다.

- `task_runs` 행에는 `log_path`, 종료 코드(사용 가능한 경우), 요약 및 메타데이터가 포함됩니다.
- `task_events` 행에는 모든 상태 전환(`promoted`, `claimed`, `heartbeat`, `completed`, `blocked`, `gave_up`, `crashed`, `timed_out`, `reclaimed`, `claim_extended`)이 포함됩니다.
- `kanban_show`는 둘 다 반환하므로 리뷰어(또는 후속 워커)가 작업을 읽을 때 대시보드 액세스 없이도 전체 기록을 볼 수 있습니다.

대시보드는 요약, 메타데이터 블록 및 종료 상태 배지와 함께 실행 기록을 렌더링합니다. CLI 사용자는 `hermes kanban tail <task_id>`를 실행하여 실시간으로 확인하거나 `hermes kanban runs <task_id>`를 실행하여 이전 시도 목록을 볼 수 있습니다.

## 기존 레인 형태 (Existing lane shapes)

### Hermes 프로필 레인 (기본값)

오늘날 모든 칸반 워커가 취하는 형태입니다. 담당자는 프로필 이름이고 디스패처는 `hermes -p <profile>`을 생성하며 워커는 `KANBAN_GUIDANCE` 시스템 프롬프트 블록과 함께 [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) 스킬을 자동 로드하고 실행을 종료하기 위해 `kanban_*` 도구를 사용합니다. 프로필 정의 외에는 설정이 필요하지 않습니다.

플릿(fleet)용 프로필을 생성할 때 오케스트레이터가 라우팅하도록 할 *역할*과 일치하는 이름을 선택합니다. 오케스트레이터(존재하는 경우)는 `hermes profile list`를 통해 프로필 이름을 발견합니다. 시스템이 가정하는 고정된 명단은 없습니다(오케스트레이터 측 계약은 [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) 스킬 참조).

### 오케스트레이터 프로필 레인

프로필 레인의 특수화: 오케스트레이터는 툴셋에 `kanban`을 포함하지만 구현을 위한 `terminal` / `file` / `code` / `web`을 제외하는 Hermes 프로필입니다. 그 임무는 `kanban_create` + `kanban_link`를 통해 높은 수준의 목표를 하위 작업으로 분해하고 물러나는 것입니다. 오케스트레이터 스킬은 유혹 방지 규칙을 인코딩합니다.

## 외부 CLI 워커 레인 추가하기 (Adding an external CLI worker lane)

비 Hermes CLI 도구(Codex CLI, Claude Code CLI, OpenCode CLI, 로컬 코딩 모델 러너 등)를 칸반 워커 레인으로 연결하는 것은 *아직 포장된 경로가 아닙니다*. 디스패처의 생성 기능은 연결 가능하며(`spawn_fn`은 `dispatch_once`의 매개변수임) 플러그인은 비 Hermes 담당자에 대해 자체 `spawn_fn`을 등록할 수 있습니다. 하지만 주변 통합 작업 — CLI의 종료 코드를 `kanban_complete` / `kanban_block` 호출로 래핑하고, CLI의 작업 공간/샌드박스 규칙을 디스패처의 `HERMES_KANBAN_WORKSPACE` 환경 변수에 매핑하며, 인증 및 CLI별 정책을 처리하는 작업 — 은 여전히 통합별 설계 작업으로 남아있습니다.

CLI 레인 추가를 고려하고 있다면 특정 CLI 및 활성화하려는 워크플로를 설명하는 이슈를 등록하세요. 위의 계약은 그러한 레인이 충족해야 하는 제약 조건입니다. 구현 형태(CLI당 하나의 플러그인 vs 구성으로 매개변수화된 일반 CLI 러너 플러그인)는 열려 있습니다.

이에 대한 과거 이슈는 [#19931](https://github.com/NousResearch/hermes-agent/issues/19931)이고 병합되지 않고 닫힌 Codex 전용 PR은 [#19924](https://github.com/NousResearch/hermes-agent/pull/19924)입니다. 이들은 원래 아키텍처 제안을 설명했지만 러너를 적용하지는 못했습니다.

## 디스패처가 처리하는 실패 모드 (Failure modes the dispatcher handles)

레인 작성자가 이를 다시 구현할 필요가 없도록 디스패처는 다음을 처리합니다.

- **오래된 클레임 TTL (Stale claim TTL)** — 클레임 후 하트비트/완료/차단을 수행하지 않는 워커는 `DEFAULT_CLAIM_TTL_SECONDS`(기본값 15분) 후에 회수되지만, 워커 프로세스가 실제로 종료된 경우에만 회수됩니다. 활성 워커(도구가 없는 단일 LLM 호출에 20분 이상을 소비하는 느린 모델)는 클레임이 종료되는 대신 *연장*됩니다. 죽은 PID만 회수됩니다.
- **충돌한 워커 (Crashed worker)** — 호스트 로컬 PID가 사라진 워커는 `detect_crashed_workers`에 의해 감지되고 거둬집니다. 작업은 `consecutive_failures`를 증가시키고 차단기가 작동할 때 자동으로 차단될 수 있습니다.
- **실행 수준 재시도 (Run-level retry)** — 작업이 재시도될 때(차단 후, 충돌 후, 회수 후) 워커는 종료 도구에 `expected_run_id` 매개변수를 사용하여 자체 실행이 이미 대체된 경우 빠르게 실패할 수 있습니다.
- **작업별 최대 런타임 (Per-task max runtime)** — `task.max_runtime_seconds`는 PID 활성 상태와 관계없이 실행당 실제 경과 시간(wall-clock time)을 하드 캡핑합니다. 활성 PID 확장으로 인해 계속 실행될 수 있는 진정으로 교착 상태(deadlocked)에 빠진 워커를 잡습니다.
- **고립된 작업 감지 (Stranded-task detection)** — 담당자가 `kanban.stranded_threshold_seconds`(기본값 30분) 내에 클레임을 생성하지 않는 준비된 작업은 `hermes kanban diagnostics`에서 `stranded_in_ready` 경고로 표시됩니다. 임계값의 2배가 되면 심각도가 오류로 에스컬레이션되고 6배가 되면 치명적으로 에스컬레이션됩니다. 오타가 있는 담당자, 삭제된 프로필, 중단된 외부 워커 풀을 하나의 신호로 감지합니다(ID에 구애받지 않으며 보드별로 관리할 허용 목록 없음).

## 관련 항목 (Related)

- [Kanban 개요 (Kanban overview)](./kanban) — 사용자 대상 소개.
- [Kanban 튜토리얼 (Kanban tutorial)](./kanban-tutorial) — 대시보드를 열어 놓은 연습.
- [`kanban-worker`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-worker/SKILL.md) — 워커 프로세스가 로드하는 스킬.
- [`kanban-orchestrator`](https://github.com/NousResearch/hermes-agent/blob/main/skills/devops/kanban-orchestrator/SKILL.md) — 오케스트레이터 측.

---
sidebar_position: 12
title: "Kanban (멀티 에이전트 보드)"
description: "여러 Hermes 프로필을 조율하기 위한, 지속형 SQLite 기반 작업 보드"
sidebar_label: "Kanban"
---

# Kanban — 멀티 에이전트 프로필 협업

> **전체 흐름을 먼저 보고 싶다면?** [Kanban 튜토리얼](./kanban-tutorial)을 읽어보세요. 이 문서는 레퍼런스이고, 튜토리얼은 사용자 시나리오 중심 설명입니다.

Hermes Kanban은 모든 Hermes 프로필이 함께 쓰는 **지속형 작업 보드**입니다. 취약한 in-process 서브에이전트 무리 대신, 이름 있는 여러 에이전트가 같은 작업을 협업할 수 있게 해줍니다. 모든 task는 `~/.hermes/kanban.db`의 한 row이고, 모든 handoff도 누구나 읽고 쓸 수 있는 row이며, 모든 worker는 자기 정체성을 가진 **독립 OS 프로세스**입니다.

### 두 개의 표면: 모델은 tool로 말하고, 사용자는 CLI로 다룹니다

보드에는 두 개의 진입점이 있고, 둘 다 같은 `~/.hermes/kanban.db`를 사용합니다.

- **에이전트는 전용 `kanban_*` toolset으로 보드를 다룹니다.** `kanban_show`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`가 여기에 포함됩니다. dispatcher는 worker를 띄울 때 이 tool들을 스키마에 넣어주며, 모델은 `hermes kanban` CLI를 shell로 호출하지 않고 **직접 tool call**로 task를 읽고 넘깁니다. 아래의 [작업자는 보드와 어떻게 상호작용하나](#how-workers-interact-with-the-board)를 참고하세요.
- **사람(그리고 스크립트, cron)은 `hermes kanban …` CLI, `/kanban …` 슬래시 명령, 혹은 dashboard로 보드를 다룹니다.** 이 표면은 tool-calling 모델이 없는 인간/자동화를 위한 인터페이스입니다.

두 표면 모두 같은 `kanban_db` 계층을 통하기 때문에, 읽기 결과는 일관되고 쓰기 결과가 어긋나지 않습니다. 이 문서는 복사해 쓰기 쉬운 CLI 예시를 중심으로 설명하지만, 여기 등장하는 CLI 동작은 전부 모델이 쓰는 tool-call 대응물이 있습니다.

이 구조는 `delegate_task`로는 커버하기 어려운 작업에 적합합니다.

- **리서치 분업** — 병렬 조사자 + 분석가 + 작성자, 그리고 human-in-the-loop
- **스케줄 기반 운영** — 주/월 단위로 누적되는 recurring 브리프
- **디지털 트윈** — 시간이 지나며 메모리를 축적하는 named assistant (`inbox-triage`, `ops-review`)
- **엔지니어링 파이프라인** — 분해 → 병렬 구현(worktree) → 리뷰 → 반복 → PR
- **플릿 작업** — 한 specialist가 N개의 대상(예: 50개 소셜 계정, 12개 서비스)을 관리

설계 배경, 비교 분석(Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise), 8개의 정형 협업 패턴은 레포의 `docs/hermes-kanban-v1-spec.pdf`를 참고하세요.

## Kanban vs. `delegate_task`

겉보기엔 비슷하지만, 같은 primitive가 아닙니다.

| | `delegate_task` | Kanban |
|---|---|---|
| 형태 | RPC 호출 (fork → join) | 지속형 메시지 큐 + 상태 머신 |
| 부모 | 자식이 끝날 때까지 block | `create` 후 fire-and-forget |
| 자식 정체성 | 익명 subagent | persistent memory를 가진 named profile |
| 재개 가능성 | 없음 — 실패하면 끝 | block → unblock → 재실행, crash → reclaim |
| Human in the loop | 지원 안 함 | 언제든 comment / unblock 가능 |
| task당 agent 수 | 한 호출 = 한 subagent | task 수명 동안 N명의 agent 가능 |
| 감사 이력 | 컨텍스트 압축 시 사라짐 | SQLite row로 영구 보존 |
| 조율 구조 | 계층형 (caller → callee) | 동료형 — 어떤 profile이든 task를 읽고 수정 가능 |

**한 줄 차이:** `delegate_task`는 함수 호출이고, Kanban은 어떤 profile이든 보고 수정할 수 있는 handoff row를 가진 작업 큐입니다.

**`delegate_task`를 써야 할 때**
- 부모 agent가 이어서 생각하기 전에 짧은 reasoning 결과가 필요할 때
- 사람이 끼지 않을 때
- 결과가 다시 부모 컨텍스트 안으로 바로 돌아가야 할 때

**Kanban을 써야 할 때**
- 작업이 agent 경계를 넘을 때
- 재시작 이후에도 살아남아야 할 때
- 중간에 사람 입력이 필요할 수 있을 때
- 다른 role이 이어받을 수 있어야 할 때
- 사후에 추적 가능해야 할 때

두 기능은 함께 쓸 수 있습니다. kanban worker가 자기 task 수행 중 내부적으로 `delegate_task`를 호출하는 것도 가능합니다.

## 핵심 개념

- **Board** — 자체 SQLite DB, workspace 디렉터리, dispatcher loop를 가진 독립 queue. 하나의 설치에 여러 board를 둘 수 있습니다. 자세한 내용은 아래의 [Boards (멀티 프로젝트)](#boards-multi-project).
- **Task** — 제목, 선택적 본문, 단일 assignee(profile 이름), 상태(`triage | todo | ready | running | blocked | done | archived`), 선택적 tenant namespace, 선택적 idempotency key를 가진 row.
- **Link** — 부모 → 자식 의존성을 기록하는 `task_links` row. 부모가 모두 `done`이면 dispatcher가 `todo → ready`로 승격시킵니다.
- **Comment** — 에이전트 간 프로토콜. agent와 사람이 comment를 붙이고, worker가 (재)실행될 때 전체 thread를 컨텍스트로 읽습니다.
- **Workspace** — worker가 실제 작업을 수행하는 디렉터리.
  - `scratch` (기본값) — `~/.hermes/kanban/workspaces/<id>/` 아래의 새 tmp 디렉터리 (non-default board는 board 경로 아래)
  - `dir:<path>` — 기존 공유 디렉터리. **절대경로만 허용**됩니다.
  - `worktree` — 코딩 task를 위한 git worktree (`.worktrees/<id>/`)
- **Dispatcher** — 주기적으로 stale claim 회수, crashed worker 정리, ready task 승격, atomic claim, assigned profile spawn을 수행하는 장기 실행 루프. 기본적으로 gateway 내부(`kanban.dispatch_in_gateway: true`)에서 동작합니다.
- **Tenant** — board 내부의 선택적 namespace. 예를 들어 하나의 specialist fleet가 여러 고객사를 처리할 때 `--tenant business-a`처럼 사용합니다. tenant는 soft filter이고, board가 hard isolation boundary입니다.

## Boards (멀티 프로젝트) {#boards-multi-project}

board를 쓰면 서로 무관한 작업 흐름을 프로젝트/레포/도메인별로 완전히 분리할 수 있습니다. 새 설치에는 `default` board 하나만 존재하며, DB는 하위 호환 때문에 `~/.hermes/kanban.db`에 놓입니다. 작업 흐름이 하나뿐인 사용자는 board 개념을 몰라도 됩니다.

board 단위 격리는 다음을 의미합니다.

- board별 별도 SQLite DB (`~/.hermes/kanban/boards/<slug>/kanban.db`)
- 별도 `workspaces/` 및 `logs/`
- worker는 자기 board task만 볼 수 있음 (`HERMES_KANBAN_BOARD` 고정)
- board 간 task link는 불가

### CLI에서 board 관리

```bash
# 현재 디스크에 있는 board 확인
hermes kanban boards list

# 새 board 생성
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch

# switch 없이 특정 board만 대상으로 실행
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# 현재 board 바꾸기
hermes kanban boards switch atm10-server
hermes kanban boards show

# 표시 이름 변경 (slug는 디렉터리 이름이라 immutable)
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# 아카이브(기본): dir을 boards/_archived/<slug>-<ts>/ 로 이동
hermes kanban boards rm atm10-server

# 영구 삭제
hermes kanban boards rm atm10-server --delete
```

board 해석 우선순위는 다음과 같습니다.

1. 명시적 `--board <slug>`
2. `HERMES_KANBAN_BOARD` 환경변수
3. `~/.hermes/kanban/current`
4. `default`

slug는 소문자 영숫자 + `-` + `_`, 길이 1–64로 제한되며, 대문자 입력은 자동 소문자화됩니다.

### Dashboard에서 board 관리

`hermes dashboard`의 Kanban 탭은 board가 2개 이상이거나 task가 존재하면 상단에 board switcher를 표시합니다.

- **Board dropdown** — 활성 board 선택. 브라우저 `localStorage`에 저장되므로 새로고침 후에도 유지됩니다.
- **+ New board** — slug, display name, description, icon 입력 modal
- **Archive** — non-`default` board에서만 표시

모든 dashboard API endpoint는 `?board=<slug>`를 받고, 이벤트 WebSocket도 연결 시점에 특정 board로 고정됩니다.

## 빠른 시작

아래 명령은 **사람인 당신**이 board를 만들고 task를 등록하는 단계입니다. task가 assign된 뒤부터는 dispatcher가 해당 profile을 worker로 띄우고, 그 이후에는 **모델이 CLI가 아니라 `kanban_*` tool call**로 task를 진행합니다.

```bash
# 1. board 생성
hermes kanban init

# 2. gateway 시작 (내장 dispatcher 포함)
hermes gateway start

# 3. task 생성
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. 실시간 확인
hermes kanban watch

# 5. board 상태 보기
hermes kanban list
hermes kanban stats
```

dispatcher가 `t_abcd`를 집어 `researcher` profile을 worker로 띄우면, 그 worker가 제일 먼저 하는 일은 `kanban_show()` 호출입니다. `hermes kanban show t_abcd`를 shell로 실행하지 않습니다.

### Gateway 내장 dispatcher (기본값)

dispatcher는 gateway 프로세스 안에서 돌기 때문에 별도 서비스가 필요 없습니다. gateway만 살아 있으면 ready task는 다음 tick(기본 60초)에 처리됩니다.

```yaml
kanban:
  dispatch_in_gateway: true
  dispatch_interval_seconds: 60
```

디버깅용으로만 `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`으로 끌 수 있습니다. `hermes kanban daemon` 단독 실행 방식은 **deprecated**이며, 가능하면 gateway를 쓰는 것이 권장됩니다.

### Idempotent create (자동화 / webhook용)

```bash
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

같은 key로 재호출하면 중복 task 대신 기존 task id를 돌려줍니다.

### Bulk CLI verbs

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

## 작업자는 보드와 어떻게 상호작용하나 {#how-workers-interact-with-the-board}

**Worker는 `hermes kanban`을 shell로 호출하지 않습니다.** dispatcher는 worker spawn 시 `HERMES_KANBAN_TASK=t_abcd`를 child env에 넣고, 그 환경변수가 모델 스키마에서 전용 **kanban toolset**을 활성화합니다. 이 7개 tool은 CLI와 동일하게 Python `kanban_db` 계층을 직접 호출합니다.

| Tool | 목적 | 필수 파라미터 |
|---|---|---|
| `kanban_show` | 현재 task 읽기 (제목, 본문, 시도 이력, 부모 handoff, comment, `worker_context`) | — |
| `kanban_complete` | `summary` + `metadata`로 완료 | `summary` 또는 `result` 중 최소 하나 |
| `kanban_block` | 사람 입력이 필요할 때 block | `reason` |
| `kanban_heartbeat` | 장기 작업 중 살아있음을 표시 | — |
| `kanban_comment` | task thread에 note 추가 | `task_id`, `body` |
| `kanban_create` | (orchestrator) child task fan-out | `title`, `assignee` |
| `kanban_link` | (orchestrator) 부모-자식 dependency 추가 | `parent_id`, `child_id` |

전형적인 worker 흐름은 아래와 같습니다.

```
kanban_show()
# (model이 worker_context를 읽고 terminal/file tool로 실제 작업 수행)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

orchestrator라면 이런 식으로 fan-out합니다.

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

`kanban_create`, `kanban_link`, 다른 task에 대한 `kanban_comment`는 모든 worker에게 기술적으로 열려 있지만, **worker profile은 fan-out하지 않고 orchestrator profile은 직접 실행하지 않는다**는 운영 규칙을 `kanban-orchestrator` skill이 강제하는 것이 권장됩니다.

### 왜 `hermes kanban` shell 호출 대신 tool인가

1. **백엔드 이식성** — terminal backend가 Docker / Modal / Singularity / SSH여도, kanban tool은 agent 자신의 Python 프로세스에서 돌아가므로 항상 `~/.hermes/kanban.db`에 도달합니다.
2. **shell quoting 취약성 제거** — `--metadata '{"files": [...]}'` 같은 문자열 인자 문제를 피합니다.
3. **더 좋은 오류 처리** — stderr 파싱이 아니라 structured JSON 결과를 모델이 바로 읽습니다.

**일반 세션에는 schema footprint가 0입니다.** 평범한 `hermes chat` 세션에는 `kanban_*` tool이 나타나지 않습니다. `HERMES_KANBAN_TASK`가 있을 때만 `check_fn`이 True가 되기 때문입니다.

### 추천 handoff evidence

`kanban_complete(summary=..., metadata={...})`의 의도는 명확합니다.

- `summary` — 사람이 읽는 closeout
- `metadata` — 다음 agent / reviewer / dashboard가 재사용할 수 있는 machine-readable handoff

엔지니어링/리뷰 task라면 보통 이런 `metadata` 형태를 권장합니다.

```json
{
  "changed_files": ["path/to/file.py"],
  "verification": ["pytest tests/hermes_cli/test_kanban_db.py -q"],
  "dependencies": ["parent task id or external issue, if any"],
  "blocked_reason": null,
  "retry_notes": "what failed before, if this was a retry",
  "residual_risk": ["what was not tested or still needs human review"]
}
```

이 키들은 강제 스키마가 아니라 **관례**입니다. 중요한 건 다음 4가지를 빠르게 알 수 있게 하는 것입니다.

1. 무엇이 바뀌었나?
2. 어떻게 검증했나?
3. 실패했을 때 무엇이 unblock / retry를 가능하게 하나?
4. 어떤 risk가 의도적으로 남아 있나?

`metadata`에는 secret, raw log, token, OAuth material, 무관한 transcript를 넣지 말고, 요약과 pointer만 넣는 게 좋습니다.

### Worker skill

kanban task를 처리할 수 있는 profile은 `kanban-worker` skill을 로드해야 합니다. 이 skill은 CLI가 아니라 **tool call 기준 lifecycle**을 가르칩니다.

1. spawn되면 `kanban_show()` 호출
2. terminal tool로 `cd $HERMES_KANBAN_WORKSPACE`
3. 장기 작업 중 `kanban_heartbeat(note="...")`
4. 끝나면 `kanban_complete(...)`, 막히면 `kanban_block(...)`

설치 예시는 다음과 같습니다.

```bash
hermes skills install devops/kanban-worker
```

dispatcher는 worker를 띄울 때 자동으로 `--skills kanban-worker`도 함께 넘기므로, profile 기본 skill 설정에 없더라도 실행 시점에는 항상 패턴 라이브러리를 갖게 됩니다.

### 특정 task에 skill 추가로 pin하기

어떤 task는 assignee profile 기본 skill만으로는 부족할 수 있습니다. 예를 들어 번역 task에는 `translation`, 리뷰 task에는 `github-code-review`, 보안 감사에는 `security-pr-audit`가 필요할 수 있습니다. 그럴 때 profile 자체를 매번 수정하지 말고 task에 직접 skill을 붙이면 됩니다.

**orchestrator agent에서**

```
kanban_create(
    title="translate README to Japanese",
    assignee="linguist",
    skills=["translation"],
)

kanban_create(
    title="audit auth flow",
    assignee="reviewer",
    skills=["security-pr-audit", "github-code-review"],
)
```

**사람이 CLI / slash command에서**

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**dashboard에서는** inline create form의 **skills** 필드에 comma-separated로 넣으면 됩니다.

이 skill들은 기본 `kanban-worker`에 **추가(additive)** 됩니다. dispatcher는 각 skill마다 `--skills <name>` 플래그를 하나씩 넣어 worker를 띄웁니다.

### Orchestrator skill

**잘 행동하는 orchestrator는 일을 직접 하지 않습니다.** 사용자의 목표를 task로 분해하고, link를 만들고, specialist에게 assign한 뒤 물러납니다. `kanban-orchestrator` skill은 이 규칙을 `kanban_create` / `kanban_link` / `kanban_comment` 패턴으로 정리해 둡니다.

대표적인 orchestrator turn 예시:

```
# 사용자 목표: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],
    body="one-pager, neutral tone, cite sources inline",
)
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

설치:

```bash
hermes skills install devops/kanban-orchestrator
```

가장 깔끔한 운용은 orchestrator profile의 toolset을 board operation 위주(`kanban`, `gateway`, `memory`)로 제한해, 구현 작업을 **물리적으로 직접 실행할 수 없게** 만드는 것입니다.

## Dashboard (GUI)

`/kanban` CLI와 slash command만으로도 headless 운영은 가능하지만, triage, cross-profile supervision, comment thread 읽기, 카드 drag/drop 같은 작업은 사람이 보기엔 시각 보드가 더 편합니다. Hermes는 이를 core 기능이 아니라 `plugins/kanban/`의 **bundled dashboard plugin**으로 제공합니다.

열기:

```bash
hermes kanban init
hermes dashboard
```

### Plugin이 제공하는 것

- `triage`, `todo`, `ready`, `running`, `blocked`, `done` 컬럼(토글 시 `archived` 포함)
- 카드에 task id, title, priority badge, tenant tag, assignee, comment/link 수, progress pill, 생성 시간 표시
- **Running 컬럼의 profile별 lane**
- **WebSocket 기반 실시간 업데이트**
- 컬럼 간 **drag-drop 상태 전환**
- **Inline create**
- **Multi-select + bulk action**
- 카드 클릭 시 side drawer:
  - 제목/assignee/priority 수정
  - markdown description 편집
  - dependency editor
  - 상태 전환 버튼
  - result section, comment thread, 최근 20개 이벤트
- 상단 toolbar filter:
  - free-text search
  - tenant dropdown
  - assignee dropdown
  - archived toggle
  - lanes by profile toggle
  - **Nudge dispatcher** 버튼

시각적으로는 Linear / Fusion 스타일의 dark theme 보드를 지향합니다.

### 아키텍처

GUI는 철저히 **DB 읽기 + `kanban_db` 쓰기** 레이어입니다.

```
┌────────────────────────┐      WebSocket (tails task_events)
│   React SPA (plugin)   │ ◀──────────────────────────────────┐
│   HTML5 drag-and-drop  │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     writes call kanban_db.*        │
│  FastAPI router        │     directly — same code path      │
│  plugins/kanban/       │     the CLI /kanban verbs use      │
│  dashboard/plugin_api.py                                    │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── append task_events ──────────┘
│  (WAL, shared)         │
└────────────────────────┘
```

### REST 표면

모든 route는 `/api/plugins/kanban/` 아래에 있으며 dashboard의 ephemeral session token으로 보호됩니다.

| Method | Path | 목적 |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | 상태 컬럼별 전체 board + filter용 tenants/assignees |
| `GET` | `/tasks/:id` | task + comments + events + links |
| `POST` | `/tasks` | 생성 |
| `PATCH` | `/tasks/:id` | 상태 / assignee / priority / title / body / result 수정 |
| `POST` | `/tasks/bulk` | 여러 id에 동일 patch 적용 |
| `POST` | `/tasks/:id/comments` | comment 추가 |
| `POST` | `/links` | dependency 추가 |
| `DELETE` | `/links?parent_id=…&child_id=…` | dependency 제거 |
| `POST` | `/dispatch?max=…&dry_run=…` | dispatcher 즉시 1회 실행 |
| `GET` | `/config` | `dashboard.kanban` 설정 읽기 |
| `WS` | `/events?since=<event_id>` | `task_events` 실시간 스트림 |

handler는 전부 얇은 wrapper이고, 실제 비즈니스 로직은 `kanban_db`에 있습니다.

### Dashboard 설정

`~/.hermes/config.yaml`의 `dashboard.kanban` 아래 키로 기본 동작을 바꿀 수 있습니다.

```yaml
dashboard:
  kanban:
    default_tenant: acme
    lane_by_profile: true
    include_archived_by_default: false
    render_markdown: true
```

### 보안 모델

dashboard는 기본적으로 localhost에 bind되므로 plugin route들은 별도 인증 없이 열려 있습니다. 즉 **호스트 내부 프로세스**는 kanban REST 표면에 접근할 수 있습니다.

WebSocket은 브라우저 upgrade 요청 특성상 `Authorization` 헤더를 못 쓰기 때문에 `?token=…` query parameter로 dashboard session token을 요구합니다.

`hermes dashboard --host 0.0.0.0`로 띄우면 모든 plugin route가 네트워크에 노출됩니다. **공유 호스트에서는 권장되지 않습니다.** task body, comment, workspace path 등 협업 surface 전체가 노출될 수 있습니다.

### Live updates

`task_events`는 monotonic `id`를 가진 append-only SQLite table입니다. WebSocket endpoint는 클라이언트별 last-seen event id를 들고 있다가 새 row를 push합니다. 이벤트 burst가 와도 frontend는 board endpoint를 한 번만 재로딩해 상태를 맞춥니다.

### 확장

plugin은 표준 Hermes dashboard plugin contract를 사용합니다. 추가 컬럼, 커스텀 카드 UI, tenant-filtered layout, 전체 `tab.override` 교체도 plugin fork 없이 표현 가능합니다.

비활성화만 하고 싶다면 `config.yaml`에 다음을 추가하면 됩니다.

```yaml
dashboard:
  plugins:
    kanban:
      enabled: false
```

### 범위 경계

GUI는 의도적으로 얇습니다. auto-assignment, budget, governance gate, org-chart view 같은 것은 user-space 영역입니다.

## CLI 명령 레퍼런스

이 표면은 **사람, 스크립트, cron, dashboard**가 보드를 조작할 때 씁니다. dispatcher 내부 worker는 동일 작업을 `kanban_*` [tool 표면](#how-workers-interact-with-the-board)으로 수행합니다.

```
hermes kanban init
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|dir:<path>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived] [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...
hermes kanban tail <id>
hermes kanban watch [--assignee P] [--tenant T] [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]
hermes kanban runs <id> [--json]
hermes kanban assignees [--json]
hermes kanban dispatch [--dry-run] [--max N] [--failure-limit N] [--json]
hermes kanban daemon --force
hermes kanban stats [--json]
hermes kanban log <id> [--tail BYTES]
hermes kanban notify-subscribe <id> --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id> --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>
hermes kanban gc [--event-retention-days N] [--log-retention-days N]
```

모든 명령은 interactive CLI와 messaging gateway에서도 `/kanban` slash command로 쓸 수 있습니다.

## `/kanban` 슬래시 명령 {#kanban-slash-command}

모든 `hermes kanban <action>`은 `/kanban <action>`으로도 호출할 수 있습니다. interactive `hermes chat` 세션과 Telegram/Discord/Slack/WhatsApp/Signal/Matrix/Mattermost/email/SMS 등 gateway 플랫폼에서 모두 동작합니다.

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
```

여러 단어 인자는 shell처럼 quote하면 됩니다. 내부적으로 `shlex.split`을 사용합니다.

### 실행 중 사용: `/kanban`은 running-agent guard를 우회합니다

일반적으로 gateway는 agent가 아직 응답 중이면 slash command와 user message를 queue에 쌓습니다. 그러나 **`/kanban`은 예외입니다.** board는 `~/.hermes/kanban.db`에 있고 실행 중인 agent의 내부 state에 묶여 있지 않기 때문입니다.

예:

- worker가 peer를 기다리며 block됨 → 휴대폰에서 `/kanban unblock t_abcd`
- 사람이 context를 더 넣어야 함 → `/kanban comment t_xyz "use the 2026 schema, not 2025"`
- orchestrator를 멈추지 않고 플릿 상태를 보고 싶음 → `/kanban list --mine`, `/kanban stats`

### `/kanban create` 시 자동 구독 (gateway 전용)

gateway에서 `/kanban create "…"`로 task를 만들면, 원래 chat이 해당 task의 terminal event(`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`)에 자동 구독됩니다.

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

`--json`을 써서 machine output으로 create하면 auto-subscribe는 생략됩니다.

### 메시징 출력 잘림

gateway 플랫폼은 메시지 길이 제한이 있어서 `/kanban list`, `/kanban show`, `/kanban tail` 결과가 약 3800자를 넘으면 잘려서 반환됩니다. 전체 출력은 터미널의 `hermes kanban …`를 쓰면 됩니다.

### 자동완성

interactive CLI에서 `/kanban ` 뒤 Tab을 누르면 built-in subcommand hint가 순환됩니다.

## 협업 패턴

새 primitive를 추가하지 않고도 다음 패턴을 지원합니다.

| Pattern | 형태 | 예시 |
|---|---|---|
| **P1 Fan-out** | 같은 role의 sibling N개 | "5개 각도를 병렬 조사" |
| **P2 Pipeline** | scout → editor → writer 체인 | daily brief 조립 |
| **P3 Voting / quorum** | sibling N개 + 1 aggregator | 3명 조사 → 1명 reviewer 결정 |
| **P4 Long-running journal** | 같은 profile + shared dir + cron | Obsidian vault |
| **P5 Human-in-the-loop** | worker block → user comment → unblock | 애매한 의사결정 |
| **P6 `@mention`** | prose 안의 inline routing | `@reviewer look at this` |
| **P7 Thread-scoped workspace** | thread 내부 `/kanban here` | 프로젝트별 gateway thread |
| **P8 Fleet farming** | 한 profile, N subjects | 50개 소셜 계정 |
| **P9 Triage specifier** | rough idea → `triage` → specifier 확장 → `todo` | 한 줄 아이디어를 spec로 승격 |

실전 예시는 `docs/hermes-kanban-v1-spec.pdf` 참고.

## 멀티 테넌트 사용

하나의 specialist fleet가 여러 비즈니스를 담당한다면 task에 tenant를 붙입니다.

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

worker는 `$HERMES_TENANT`를 받고 memory write를 prefix namespace로 분리합니다. board, dispatcher, profile 정의는 공유하고 데이터만 scope됩니다.

## Gateway 알림

gateway에서 `/kanban create …`를 실행하면 원래 chat이 새 task에 자동 구독됩니다. gateway의 background notifier는 몇 초마다 `task_events`를 poll하고 terminal event마다 메시지를 한 번씩 보냅니다. 완료된 task는 worker `--result`의 첫 줄도 함께 보내줍니다.

명시적으로 CLI에서 구독을 관리할 수도 있습니다.

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

task가 `done` 또는 `archived`가 되면 구독은 자동 제거됩니다.

## Runs — 시도 1회당 row 1개

task는 논리적 작업 단위이고, **run**은 그 작업을 실행한 한 번의 시도입니다. dispatcher가 ready task를 claim하면 `task_runs`에 row를 만들고 `tasks.current_run_id`가 그 row를 가리킵니다. 시도가 완료/차단/crash/timeout/spawn-failed/reclaimed로 끝나면 run row는 `outcome`과 함께 닫히고 pointer는 비워집니다.

task와 run을 분리하는 이유:

- 실제 postmortem에 필요한 **전체 시도 이력** 보존
- 어떤 파일이 바뀌었는지, 어떤 테스트를 돌렸는지, reviewer가 무엇을 지적했는지 같은 **시도별 metadata** 저장

run은 structured handoff가 놓이는 곳이기도 합니다.

- `summary` / `--summary` — 사람이 읽는 handoff
- `metadata` / `--metadata` — 자유 형식 JSON dict
- `result` / `--result` — task row에 남는 짧은 log line

예:

```
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

사람이 CLI로 직접 닫을 수도 있습니다.

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

hermes kanban runs t_abcd
```

주의 사항:

- **Bulk close + `--summary`/`--metadata`는 거부**됩니다. handoff는 run마다 달라야 하기 때문입니다.
- dashboard에서 running task를 다른 상태로 drag하면 in-flight run은 orphan 대신 `reclaimed`로 닫힙니다.
- 한 번도 claim되지 않은 task를 사람이 완료/차단하면 summary/handoff를 잃지 않도록 zero-duration synthetic run이 생성됩니다.

### Forward compatibility

`tasks`의 nullable column 두 개는 v2 workflow routing용으로 예약되어 있습니다.

- `workflow_template_id`
- `current_step_key`

v1 kernel은 routing에는 쓰지 않지만, client가 기록하는 것은 허용합니다.

## Event 레퍼런스

모든 상태 전환은 `task_events`에 row를 append합니다. 각 row는 선택적으로 `run_id`를 포함하므로 UI가 시도 단위로 묶을 수 있습니다.

### Lifecycle

| Kind | Payload | 시점 |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | task 생성 |
| `promoted` | — | 부모가 모두 `done`이 되어 `todo → ready` |
| `claimed` | `{lock, expires, run_id}` | dispatcher가 `ready` task를 atomic claim |
| `completed` | `{result_len, summary?}` | worker가 `done`으로 종료 |
| `blocked` | `{reason}` | worker 또는 사람이 `blocked`로 전환 |
| `unblocked` | — | `blocked → ready` |
| `archived` | — | 기본 보드에서 숨김 |

### Edits

| Kind | Payload | 시점 |
|---|---|---|
| `assigned` | `{assignee}` | assignee 변경 |
| `edited` | `{fields}` | title/body 수정 |
| `reprioritized` | `{priority}` | priority 수정 |
| `status` | `{status}` | dashboard drag-drop 등으로 직접 status 변경 |

### Worker telemetry

| Kind | Payload | 시점 |
|---|---|---|
| `spawned` | `{pid}` | worker 프로세스 시작 성공 |
| `heartbeat` | `{note?}` | 장기 작업 중 liveness signal |
| `reclaimed` | `{stale_lock}` | claim TTL 만료, task가 `ready`로 복귀 |
| `crashed` | `{pid, claimer}` | worker PID가 사라짐 |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds` 초과 |
| `spawn_failed` | `{error, failures}` | spawn 시도 1회 실패 |
| `gave_up` | `{failures, error}` | circuit breaker 발동 후 auto-block |

개별 task 이벤트는 `hermes kanban tail <id>`, 보드 전체 이벤트는 `hermes kanban watch`로 볼 수 있습니다.

## 범위 밖

Kanban은 의도적으로 **single-host** 설계입니다. `~/.hermes/kanban.db`는 로컬 SQLite 파일이고, dispatcher는 같은 머신에서 worker를 spawn합니다. 두 호스트가 하나의 board를 공유하는 구조는 지원하지 않습니다.

멀티 호스트가 필요하다면 호스트별 독립 board를 두고, 그 사이를 `delegate_task`나 별도 message queue로 연결해야 합니다.

## 설계 문서

아키텍처, 동시성 정합성, 타 시스템 비교, 구현 계획, 리스크, open question을 포함한 전체 설계 문서는 `docs/hermes-kanban-v1-spec.pdf`에 있습니다. 동작 변경 PR을 넣기 전에는 이 문서를 먼저 읽는 것이 좋습니다.

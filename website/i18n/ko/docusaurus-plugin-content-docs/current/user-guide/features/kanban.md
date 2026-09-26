---
sidebar_position: 12
title: "Kanban (멀티 에이전트 보드)"
description: "여러 Hermes 프로필을 조율하기 위한 영속적인 SQLite 기반 작업 보드"
---

# Kanban — 멀티 에이전트 프로필 협업

> **사용법 안내가 필요하신가요?** [Kanban 튜토리얼](./kanban-tutorial)을 읽어 보세요. 네 가지 사용자 스토리(1인 개발자, 플릿 파밍, 재시도가 포함된 역할 파이프라인, 회로 차단기)와 각 사례의 대시보드 스크린샷을 제공합니다. 이 페이지는 레퍼런스이고, 튜토리얼은 사용 흐름을 설명합니다.

Hermes Kanban은 모든 Hermes 프로필이 공유하는 영속적인 작업 보드입니다. 프로세스 내부의 취약한 서브에이전트 스웜 없이 여러 이름 있는 에이전트가 작업에 협업할 수 있습니다. 모든 작업은 `~/.hermes/kanban.db`의 한 행이고, 모든 핸드오프는 누구나 읽고 쓸 수 있는 한 행이며, 모든 워커는 자체 식별성을 가진 완전한 OS 프로세스입니다.

### 두 가지 표면: 모델은 도구로 대화하고, 사용자는 CLI로 대화합니다

보드에는 동일한 `~/.hermes/kanban.db`를 기반으로 하는 두 개의 진입점이 있습니다.

- **에이전트는 전용 `kanban_*` 도구 세트를 통해 보드를 운영합니다** — `kanban_show`, `kanban_list`, `kanban_complete`, `kanban_request_review`, `kanban_request_changes`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_attach`, `kanban_attach_url`, `kanban_attachments`, `kanban_create`, `kanban_link`, `kanban_unblock`. 디스패처는 이 도구들이 스키마에 이미 포함된 상태로 각 워커를 생성하며, 오케스트레이터 프로필도 `kanban` 도구 세트를 명시적으로 활성화할 수 있습니다. 모델은 `hermes kanban`을 셸에서 실행하는 방식이 *아니라*, 도구를 직접 호출해 작업을 읽고 라우팅합니다. 아래의 [워커가 보드와 상호작용하는 방법](#how-workers-interact-with-the-board)을 참고하세요.
- **사용자(그리고 스크립트와 cron)는 CLI에서 `hermes kanban …`**, 슬래시 명령으로 `/kanban …`, 또는 대시보드를 통해 보드를 운영합니다. 이는 사람과 자동화를 위한 진입점이며, 도구 호출 모델이 뒤에 없는 곳에서 사용합니다.

두 표면 모두 동일한 `kanban_db` 레이어를 거치므로 읽기는 일관된 뷰를 보고 쓰기는 서로 어긋나지 않습니다. 이 페이지의 나머지 부분에서는 복사해 붙여 넣기 쉽도록 CLI 예제를 보여 주지만, 모든 CLI 동사에는 모델이 사용하는 도구 호출 대응 항목이 있습니다.

다음은 `delegate_task`로 처리하기 어려운 워크로드를 다루는 형태입니다.

- **리서치 트리아지** — 여러 연구자 + 분석가 + 작성자, 사람의 개입 포함.
- **예약된 운영** — 수 주에 걸쳐 저널을 쌓는 반복 일일 브리핑.
- **디지털 트윈** — 시간이 지나며 메모리를 축적하는 영속적인 이름 있는 어시스턴트(`inbox-triage`, `ops-review`).
- **엔지니어링 파이프라인** — 분해 → 병렬 워크트리에서 구현 → 리뷰 → 반복 → PR.
- **플릿 작업** — N개의 대상을 관리하는 한 명의 전문가(소셜 계정 50개, 모니터링 서비스 12개).

전체 설계 근거, Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise와의 비교 분석, 여덟 가지 표준 협업 패턴은 저장소의 `docs/hermes-kanban-v1-spec.pdf`를 참고하세요.

## Kanban과 `delegate_task` 비교

비슷해 보이지만 같은 프리미티브가 아닙니다.

| | `delegate_task` | Kanban |
|---|---|---|
| 형태 | RPC 호출 (포크 → 조인) | 영속적인 메시지 큐 + 상태 머신 |
| 부모 | 자식이 반환할 때까지 블로킹 | `create` 후 기다리지 않고 진행 |
| 자식 식별성 | 익명 서브에이전트 | 영속 메모리를 가진 이름 있는 프로필 |
| 재개 가능성 | 없음 — 실패 = 실패 | 블로킹 → 블로킹 해제 → 재실행; 충돌 → 회수 |
| 사람의 개입 | 지원하지 않음 | 언제든 댓글 작성 / 블로킹 해제 |
| 작업당 에이전트 수 | 호출 1회 = 서브에이전트 1명 | 작업 수명 동안 N명의 에이전트(재시도, 리뷰, 후속 작업) |
| 감사 추적 | 컨텍스트 압축 시 사라짐 | SQLite에 영구적으로 저장되는 행 |
| 조율 | 계층적(호출자 → 피호출자) | 피어 — 어떤 프로필이든 어떤 작업이든 읽고 쓸 수 있음 |

**한 문장으로 구분하면:** `delegate_task`는 함수 호출이고, Kanban은 모든 핸드오프를 어떤 프로필(또는 사람)이든 보고 편집할 수 있는 행으로 기록하는 작업 큐입니다.

**부모 에이전트가 계속 진행하기 전에 짧은 추론 답변이 필요하고**, 사람이 관여하지 않으며, 결과가 부모의 컨텍스트로 돌아가야 한다면 `delegate_task`를 사용하세요.

**작업이 에이전트 경계를 넘거나, 재시작 후에도 유지되어야 하거나, 사람의 입력이 필요할 수 있거나, 다른 역할이 이어받을 수 있어야 하거나, 나중에 찾아볼 수 있어야 한다면** Kanban을 사용하세요.

둘은 함께 사용할 수 있습니다. Kanban 워커가 실행 중 내부적으로 `delegate_task`를 호출할 수도 있습니다.

## 핵심 개념

- **보드(Board)** — 자체 SQLite DB, 워크스페이스 디렉터리, 디스패처 루프를 가진 독립적인 작업 큐입니다. 하나의 설치에서 여러 보드를 사용할 수 있습니다(예: 프로젝트, 저장소 또는 도메인별 하나씩). [보드(멀티 프로젝트)](#boards-multi-project)를 참고하세요. 단일 프로젝트 사용자는 `default` 보드를 계속 사용하며, 이 문서 섹션 밖에서는 "보드"라는 단어를 볼 일이 없습니다.
- **작업(Task)** — 제목, 선택적 본문, 담당자 한 명(프로필 이름), 상태(`triage | todo | ready | running | blocked | review | done | archived`), 선택적 테넌트 네임스페이스, 선택적 멱등성 키(재시도되는 자동화의 중복 제거)가 있는 행입니다.
- **링크(Link)** — 부모 → 자식 의존성을 기록하는 `task_links` 행입니다. 모든 부모가 `done`이 되면 디스패처가 `todo → ready`로 승격합니다.
- **댓글(Comment)** — 에이전트 간 프로토콜입니다. 에이전트와 사람은 댓글을 추가하며, 워커가 (다시) 생성될 때 전체 댓글 스레드를 컨텍스트의 일부로 읽습니다.
- **워크스페이스(Workspace)** — 워커가 작업하는 디렉터리입니다. 세 가지 종류가 있습니다.
  - `scratch`(기본값) — `~/.hermes/kanban/workspaces/<id>/` 아래의 새 임시 디렉터리입니다(기본 보드가 아닌 보드에서는 `~/.hermes/kanban/boards/<slug>/workspaces/<id>/`). **작업이 완료되면 삭제됩니다** — scratch는 원래 임시로 설계되었습니다. `kanban_complete(artifacts=[...])`를 통해 명시적으로 선언한 파일은 정리 전에 작업별 영속 첨부 파일 저장소로 복사됩니다. 레거시 완료 요약에 있는 기존 결과물 경로도 같은 방식으로 처리됩니다. 그 밖의 scratch 파일은 제거됩니다. 선언한 scratch 결과물이 없으면 워커가 경로를 수정하고 재시도할 수 있도록 작업이 진행 중인 상태로 유지됩니다. 전체 워크스페이스를 계속 사용할 수 있어야 한다면 `worktree:` 또는 `dir:<path>`를 사용하세요. 설치 환경에서 scratch 워크스페이스가 처음 생성되면 디스패처가 경고를 기록하고 작업에 `tip_scratch_workspace` 이벤트를 발생시킵니다(`hermes kanban show <id>`로 확인 가능).
  - `dir:<path>` — 기존 공유 디렉터리입니다(Obsidian 볼트, 메일 운영 디렉터리, 계정별 폴더). **절대 경로여야 합니다.** `dir:../tenants/foo/` 같은 상대 경로는 디스패처가 실행되는 CWD에 따라 해석되어 모호하고 혼동된 대리인(confused deputy) 탈출 벡터가 되므로 디스패치 시 거부됩니다. 그 외에는 경로를 신뢰합니다 — 사용자의 컴퓨터와 파일 시스템이며 워커는 사용자의 uid로 실행됩니다. 이는 신뢰하는 로컬 사용자 위협 모델이며, kanban은 단일 호스트를 전제로 설계되었습니다. **완료 후에도 유지됩니다.**
  - `worktree` — 코딩 작업을 위한 `.worktrees/<id>/` 아래의 git 워크트리입니다. 정확한 대상 경로를 고정하려면 `worktree:<path>`를 사용하세요. 워커 측의 `git worktree add`가 필요하면 `--branch`를 사용해 생성합니다. **완료 후에도 유지됩니다.**
- **디스패처(Dispatcher)** — 매 N초(기본값 60초)마다 오래된 클레임을 회수하고, 충돌한 워커(PID는 사라졌지만 TTL이 아직 만료되지 않은 경우)를 회수하고, 준비된 작업을 승격하고, 원자적으로 클레임한 뒤, 할당된 프로필을 생성하는 장기 실행 루프입니다. 기본적으로 게이트웨이 내부에서 실행됩니다(`kanban.dispatch_in_gateway: true`). 하나의 디스패처가 틱마다 모든 보드를 순회하며, 워커는 `HERMES_KANBAN_BOARD`가 고정된 상태로 생성되어 다른 보드를 볼 수 없습니다. 동일 작업에서 연속으로 생성에 실패하는 횟수가 `kanban.failure_limit`(기본값: 2)에 도달하면 디스패처가 마지막 오류를 사유로 해당 작업을 자동으로 블로킹합니다 — 존재하지 않는 프로필이나 마운트할 수 없는 워크스페이스 등으로 인해 작업이 반복 시도되는 것을 방지합니다.
- **테넌트(Tenant)** — 보드 *내부*의 선택적 문자열 네임스페이스입니다. 하나의 전문가 플릿이 여러 비즈니스를 서비스할 수 있으며(`--tenant business-a`), 워크스페이스 경로와 메모리 키 접두사로 데이터를 격리합니다. 테넌트는 느슨한 필터이고, 보드는 강제 격리 경계입니다.

## 보드(멀티 프로젝트)

보드를 사용하면 서로 관련 없는 작업 흐름을 프로젝트, 저장소 또는 도메인별로 하나씩 격리된 큐로 나눌 수 있습니다. 새로 설치하면 `default`라는 보드가 정확히 하나 생성됩니다(하위 호환성을 위해 DB는 `~/.hermes/kanban.db`). 하나의 작업 흐름만 원하는 사용자는 보드에 대해 알 필요가 없으며, 이 기능은 옵트인입니다.

보드별 격리는 절대적입니다.

- 보드별로 분리된 SQLite DB(`~/.hermes/kanban/boards/<slug>/kanban.db`).
- 분리된 `workspaces/` 및 `logs/` 디렉터리.
- 작업을 위해 생성된 워커는 **자신의 보드 작업만** 봅니다 — 디스패처가 자식 환경에 `HERMES_KANBAN_BOARD`를 설정하고 워커가 접근할 수 있는 모든 `kanban_*` 도구가 이 값을 읽습니다.
- 보드 간 작업 연결은 허용되지 않습니다(스키마를 단순하게 유지하기 위함입니다. 정말 프로젝트 간 참조가 필요하다면 자유 형식 텍스트 멘션을 사용하고 ID로 직접 찾아보세요).

### CLI에서 보드 관리하기

```bash
# See what's on disk. Fresh installs show only "default".
hermes kanban boards list

# Create a new board.
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # optional: make it the active board

# Operate on a specific board without switching.
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# Change which board is "current" for subsequent calls.
hermes kanban boards switch atm10-server
hermes kanban boards show             # who's active right now?

# Rename the display name (the slug is immutable — it's the directory name).
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# Archive (default) — moves the board's dir to boards/_archived/<slug>-<ts>/.
# Recoverable by moving the dir back.
hermes kanban boards rm atm10-server

# Hard delete — `rm -rf` the board dir. No recovery.
hermes kanban boards rm atm10-server --delete
```

보드 확인 순서(우선순위가 높은 순서)는 다음과 같습니다.

1. CLI 호출에 명시된 `--board <slug>`.
2. `HERMES_KANBAN_BOARD` 환경 변수(워커가 다른 보드를 볼 수 없도록 워커 생성 시 디스패처가 설정).
3. `hermes kanban boards switch`로 저장된 `~/.hermes/kanban/current`의 slug.
4. `default`.

slug는 다음 규칙으로 검증됩니다. 소문자 영숫자와 하이픈 및 밑줄만 사용할 수 있고, 길이는 1~64자이며, 영숫자로 시작해야 합니다. 대문자 입력은 자동으로 소문자로 변환됩니다. 슬래시, 공백, 마침표, `..` 등은 경로 순회 트릭으로 보드 이름을 지정할 수 없도록 CLI 레이어에서 거부됩니다.

### 대시보드에서 보드 관리하기

`hermes dashboard` → Kanban 탭은 보드가 두 개 이상 존재하거나 어느 보드든 작업이 있는 즉시 상단에 보드 전환기를 표시합니다. 보드가 하나뿐인 사용자는 작은 `+ New board` 버튼만 보며, 전환기는 필요할 때까지 숨겨집니다.

- **보드 드롭다운** — 활성 보드를 선택합니다. 선택 사항은 브라우저의 `localStorage`에 저장되므로, 열어 둔 터미널 아래에서 CLI의 `current` 포인터를 바꾸지 않고도 새로 고침 후 유지됩니다.
- **+ New board** — slug, 표시 이름, 설명, 아이콘을 묻는 모달을 엽니다. 새 보드로 자동 전환하는 옵션도 있습니다.
- **Settings** — 현재 보드의 표시 이름, 설명, **프로젝트 디렉터리**(`default_workdir`)를 편집하는 모달을 엽니다. 프로젝트 디렉터리는 새 작업이 상속하는 보드 수준의 기본 워크스페이스입니다(git 저장소 → 유지되는 워크트리, 일반 디렉터리 → 유지되는 디렉터리). 각 작업은 생성 시 이 값을 재정의할 수 있습니다. 필드를 비우면 새 작업은 폐기 가능한 scratch 워크스페이스로 되돌아갑니다.
- **Archive** — `default`가 아닌 보드에서만 표시됩니다. 확인하면 보드 디렉터리를 `boards/_archived/`로 이동합니다.

모든 대시보드 API 엔드포인트는 보드 범위를 지정하기 위해 `?board=<slug>`를 받습니다. 이벤트 WebSocket은 연결 시 보드에 고정되며, UI에서 전환하면 새 보드에 대한 새 WS 연결이 열립니다.


## 파일 첨부

작업에는 파일 첨부(PDF, 이미지, 소스 문서)를 추가할 수 있으므로, 경로를 본문에 붙여 넣고 워커가 찾아 주기를 기대할 필요 없이 워커가 필요한 원본 자료를 제공할 수 있습니다.

- **업로드** — 대시보드 드로어에서 작업을 열고 **Attachments** 섹션의 *Upload file* 버튼을 사용합니다(한 번에 여러 파일을 업로드할 수 있습니다). 파일 하나당 최대 25MB입니다.
- **저장소** — 기본 보드의 경우 파일은 `<hermes-home>/kanban/attachments/<task_id>/` 아래에 저장되고, 이름 있는 보드의 경우 `<hermes-home>/kanban/boards/<slug>/attachments/<task_id>/` 아래에 저장됩니다. 사용자 지정 위치를 고정하려면 `HERMES_KANBAN_ATTACHMENTS_ROOT`를 설정하세요.
- **워커가 보는 내용** — 디스패처가 작업을 워커에게 넘길 때 워커 컨텍스트에는 각 파일의 이름과 **절대 경로**를 나열하는 **Attachments** 섹션이 포함됩니다. 워커는 파일/터미널 도구에 완전히 접근할 수 있으므로 첨부 파일을 직접 읽습니다(`read_file` 또는 `pdftotext` 같은 셸 도구).
- **다운로드 / 제거** — 드로어에는 각 첨부 파일의 다운로드 링크와 제거(×) 컨트롤이 표시됩니다. 첨부 파일을 제거하면 메타데이터 행과 디스크의 파일이 모두 삭제됩니다.

:::note 원격 터미널 백엔드
첨부 파일 경로는 기본값인 **로컬** 터미널 백엔드에서 직접 확인됩니다. 원격 백엔드(Docker, Modal)에서 워커를 실행한다면, 워커 컨텍스트의 절대 경로에 접근할 수 있도록 보드의 `attachments/` 디렉터리를 샌드박스에 마운트하세요.
:::
## 빠른 시작

아래 명령은 **사용자**가 보드를 설정하고 작업을 생성할 때 사용합니다. 작업이 할당되면 디스패처가 할당된 프로필을 워커로 생성하며, 그 시점부터 모델이 CLI 명령이 아니라 `kanban_*` 도구 호출을 통해 작업을 주도합니다. — [보드와 워커가 상호작용하는 방법](#how-workers-interact-with-the-board)을 참고하세요.

```bash
# 1. Create the board (you)
hermes kanban init

# 2. Start the gateway (hosts the embedded dispatcher)
hermes gateway start

# 3. Create a task (you — or an orchestrator agent via kanban_create)
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. Watch activity live (you)
hermes kanban watch

# 5. See the board (you)
hermes kanban list
hermes kanban stats
```

디스패처가 `t_abcd`를 가져와 `researcher` 프로필을 생성하면, 해당 워커의 모델이 가장 먼저 하는 일은 `kanban_show()`를 호출해 작업을 읽는 것입니다. `hermes kanban show t_abcd`를 실행하지 않습니다.

### 게이트웨이에 내장된 디스패처 (기본값)

디스패처는 게이트웨이 프로세스 안에서 실행됩니다. 설치할 것도, 별도로 관리할 서비스도 없습니다. 게이트웨이가 실행 중이면 준비된 작업이 다음 틱(기본 60초)에 처리됩니다.

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # default
  dispatch_interval_seconds: 60    # default
  review_dispatch: true            # default: spawn the assigned profile with
                                   # the bundled sdlc-review skill. Set false
                                   # for human-only review boards.
```

디버깅을 위해 런타임에서 `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`으로 설정 플래그를 덮어쓸 수 있습니다. 표준 게이트웨이 감독 방식을 적용합니다. `hermes gateway start`를 직접 실행하거나, 게이트웨이를 systemd 사용자 유닛으로 연결할 수 있습니다(게이트웨이 문서 참고). 실행 중인 게이트웨이가 없으면 `ready` 작업은 게이트웨이가 시작될 때까지 현재 상태에 머뭅니다. `hermes kanban create`는 생성 시 이를 경고합니다.

별도 프로세스로 `hermes kanban daemon`을 실행하는 방식은 **더 이상 사용되지 않습니다**. 게이트웨이를 사용하세요. 게이트웨이를 실행할 수 없는 경우(예: 헤드리스 호스트 정책이 장기 실행 서비스를 금지하는 경우) `--force` 이스케이프 해치를 사용하면 한 릴리스 주기 동안 기존 독립 데몬을 계속 실행할 수 있습니다. 하지만 동일한 `kanban.db`에 게이트웨이에 내장된 디스패처와 독립 데몬을 함께 실행하면 클레임 경쟁이 발생하므로 지원되지 않습니다.

### 멱등적 생성 (자동화 / 웹훅용)

```bash
# First call creates the task. Any subsequent call with the same key
# returns the existing task id instead of duplicating.
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### 일괄 CLI 동사

모든 라이프사이클 동사는 여러 ID를 받으므로 한 명령으로 일괄 정리할 수 있습니다.

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

:::note 차단 해제된 작업이 배치되는 위치
`unblock`은 안전한 원래 단계를 복원합니다. 부모 작업이 완료된 리뷰어 기원 작업은 **`review`**, 부모 작업이 완료된 구현 작업은 **`ready`**, 부모 작업 중 하나라도 열려 있으면 **`todo`**로 복원됩니다. `todo` 작업은 원래 단계의 출처 정보를 유지하며, 의존성 게이트가 해소되면 자동으로 `review` 또는 `ready`로 돌아갑니다. `unblock`은 절대로 작업을 곧바로 `triage`로 보내지 않습니다.

작업을 차단 해제한 뒤 **`triage`**에 나타난다면, 원인은 차단 해제가 아닙니다. 이후에 *같은 이유로 다시 차단*된 것이 원인입니다. 작업이 차단 → 차단 해제 → 같은 원인으로 다시 차단되는 과정이 `BLOCK_RECURRENCE_LIMIT`번(기본값 `2`) 반복되면, 차단 해제 루프 차단기가 해당 작업을 다시 `blocked`로 보내지 않습니다. 크론 작업이라면 계속 차단 해제를 반복할 상황이므로, 대신 사람의 판단을 위해 `triage`로 보냅니다. 이는 LLM의 판단이 아니라 결정론적인 DB 가드이며, 작업 본문으로 이를 우회할 수 없습니다. 반복 횟수는 차단 해제 때마다 의도적으로 유지되고, 성공적인 `complete` 때만 초기화됩니다. 차단 해제된 작업을 작업 풀에 남겨두려면 차단이 반복되는 이유(미완료 부모, 누락된 입력, 충족되지 않은 기능)를 해결한 뒤 차단 해제하거나, 루프가 예상된 경우 `BLOCK_RECURRENCE_LIMIT`을 높이세요.
:::

## 워커가 보드와 상호작용하는 방법

**워커는 `hermes kanban`을 셸로 실행하지 않습니다.** 디스패처가 워커를 생성하면 자식 프로세스의 환경에 `HERMES_KANBAN_TASK=t_abcd`를 설정하고, 이 환경 변수에 따라 모델 스키마에서 전용 **kanban 도구 세트**가 활성화됩니다. 도구 세트 설정에서 `kanban`을 활성화한 오케스트레이터 프로필에서도 동일한 도구 세트를 사용할 수 있습니다. 이 도구들은 CLI와 마찬가지로 Python `kanban_db` 계층을 통해 보드를 직접 읽고 변경합니다. 실행 중인 워커는 다른 도구와 마찬가지로 이를 호출하며, `hermes kanban` CLI를 보거나 필요로 하지 않습니다.

| 도구 | 용도 | 필수 매개변수 |
|---|---|---|
| `kanban_show` | 현재 작업(제목, 본문, 이전 시도, 부모 핸드오프, 댓글, 전체 형식 지정 `worker_context`)을 읽습니다. 기본값은 환경 변수의 작업 ID입니다. | — |
| `kanban_list` | `assignee`, `status`, `tenant`, 보관 작업 표시 여부, limit 필터를 사용해 작업 요약을 나열합니다. 보드 작업을 찾는 오케스트레이터용입니다. | — |
| `kanban_complete` | `summary`와 구조화된 `metadata` 핸드오프로 완료합니다. | `summary` / `result` 중 하나 이상 |
| `kanban_request_review` | 영속적인 `summary`, 선택적 `metadata`, 선택적 리뷰어 프로필과 함께 동일 카드 리뷰를 시작합니다. 작업은 `review`로 이동하며, 이는 차단이 아닙니다. | `summary` |
| `kanban_request_changes` | 활성 리뷰 실행의 리뷰어 판정입니다. 해당 실행을 종료하고 부모 게이트를 다시 적용한 뒤, 차단 루프 집계 없이 원래 구현자에게 작업을 보냅니다. | `reason` |
| `kanban_block` | 이유에 따라 작업을 중지하고 라우팅합니다. `kind=dependency`(작업을 `todo`에서 대기시키고 자동 재개), `needs_input`/`capability`/`transient`(사람에게 표시)를 사용합니다. 같은 종류의 차단이 반복되면 자동으로 `triage`로 승격됩니다. | `reason` |
| `kanban_heartbeat` | 긴 작업 중 생존 상태를 알립니다. 순수한 부수 효과입니다. | — |
| `kanban_comment` | 작업 스레드에 영속적인 메모를 추가합니다. | `task_id`, `body` |
| `kanban_attach` | 바이트를 인라인(base64)으로 전달해 파일을 작업에 첨부합니다. 첨부 파일 디렉터리에 저장되며 최대 25MB입니다. | 파일 바이트 + 이름 |
| `kanban_attach_url` | URL로 파일을 작업에 첨부합니다. | `url` |
| `kanban_attachments` | 작업의 첨부 파일을 나열합니다. | — |
| `kanban_create` | (오케스트레이터) `assignee`, 선택적 `parents`, `skills` 등을 사용해 하위 작업으로 분배합니다. | `title`, `assignee` |
| `kanban_link` | (오케스트레이터) 사후에 `parent_id → child_id` 의존성 간선을 추가합니다. | `parent_id`, `child_id` |
| `kanban_unblock` | 차단된 작업을 원래 단계(`review` 또는 `ready`)로 복원하거나, 부모 작업이 열려 있는 동안에는 `todo`로 복원합니다. | `task_id` |

일반적인 워커의 한 턴은 다음과 같습니다.

```
# Model's tool calls, in order:
kanban_show()                                     # no args — uses HERMES_KANBAN_TASK
# (model reads the returned worker_context, does the work via terminal/file tools)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# (more work)
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

대신 **오케스트레이터** 워커는 다음과 같이 작업을 분배합니다.

```
kanban_show()
kanban_create(
    title="research ICP funding 2024-2026",
    assignee="researcher-a",
    body="focus on seed + series A, North America, AI-adjacent",
)
# → returns {"task_id": "t_r1", ...}
kanban_create(title="research ICP funding — EU angle", assignee="researcher-b", body="…")
# → returns {"task_id": "t_r2", ...}
kanban_create(
    title="synthesize findings into launch brief",
    assignee="writer",
    parents=["t_r1", "t_r2"],                     # promotes to ready when both complete
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

“(오케스트레이터)” 도구인 `kanban_list`, `kanban_create`, `kanban_link`, `kanban_unblock`, 그리고 다른 작업에 대한 `kanban_comment`는 동일한 도구 세트를 통해 사용할 수 있습니다. 관례상(자동 주입된 kanban 안내에 반영되어 있음) 워커 프로필은 작업을 분배하거나 관련 없는 작업을 라우팅하지 않으며, 오케스트레이터 프로필은 구현 작업을 수행하지 않습니다. 디스패처가 생성한 워커도 작업 범위가 지정되어 있으므로 파괴적인 라이프사이클 작업을 수행할 수 있고, 관련 없는 작업은 변경할 수 없습니다.

### `hermes kanban` 셸 실행 대신 도구를 사용하는 이유

세 가지 이유가 있습니다.

1. **백엔드 이식성.** 터미널 도구가 원격 백엔드(Docker / Modal / Singularity / SSH)를 가리키는 워커가 `hermes kanban complete`를 실행하면, `hermes`가 설치되지 않았고 `~/.hermes/kanban.db`도 마운트되지 않은 컨테이너 내부에서 실행됩니다. kanban 도구는 에이전트의 자체 Python 프로세스에서 실행되며 터미널 백엔드와 관계없이 항상 `~/.hermes/kanban.db`에 연결됩니다.
2. **셸 인용 취약성 없음.** `--metadata '{"files": [...]}'`를 shlex와 argparse를 거쳐 전달하는 방식은 잠재적인 함정입니다. 구조화된 도구 인자는 이를 완전히 건너뜁니다.
3. **더 나은 오류.** 도구 결과는 모델이 추론할 수 있는 구조화된 JSON이지, 파싱해야 하는 stderr 문자열이 아닙니다.

**일반 세션에는 스키마가 전혀 추가되지 않습니다.** 활성 프로필이 오케스트레이터 작업을 위해 `kanban` 도구 세트를 명시적으로 활성화하지 않는 한, 일반적인 `hermes chat` 세션의 스키마에는 `kanban_*` 도구가 전혀 없습니다. 디스패처가 생성한 작업 워커는 `HERMES_KANBAN_TASK`가 설정되어 있기 때문에 작업 범위가 지정된 도구를 얻고, 오케스트레이터 프로필은 설정을 통해 더 넓은 라우팅 표면을 얻습니다. kanban을 사용하지 않는 사용자에게는 도구가 늘어나지 않습니다.

자동 주입된 kanban 안내는 언제 어떤 도구를 어떤 순서로 호출해야 하는지 모델에 알려줍니다.

### 권장 핸드오프 증거

`kanban_complete(summary=..., metadata={...})`는 의도적으로 유연합니다. `summary`는 사람이 읽는 종료 요약이고, `metadata`는 후속 에이전트, 리뷰어 또는 대시보드가 문장을 긁어오지 않고 재사용할 수 있는 기계 판독용 핸드오프입니다.

엔지니어링 및 리뷰 작업에는 다음과 같은 선택적 메타데이터 형태를 권장합니다.

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

이 키들은 관례일 뿐 스키마 요구 사항은 아닙니다. 중요한 점은 모든 워커가 다음 네 가지 질문에 빠르게 답할 수 있을 만큼 충분한 증거를 남기는 것입니다.

1. 무엇이 변경되었는가?
2. 어떻게 검증했는가?
3. 실패할 경우 무엇으로 차단을 해제하거나 재시도할 수 있는가?
4. 어떤 위험을 의도적으로 열어 두었는가?

`metadata`에 비밀, 원시 로그, 토큰, OAuth 자료, 관련 없는 대화 기록을 넣지 마세요. 대신 포인터와 요약을 저장하세요. 작업에 파일이나 테스트가 없다면 `summary`에 명시하고, 소스 URL, 이슈 ID 또는 수동 리뷰 단계처럼 존재하는 증거는 `metadata`에 사용하세요.

### 워커 라이프사이클

kanban 작업을 처리하는 모든 프로필은 자동으로 워커 라이프사이클을 받습니다. 생성 시 워커의 시스템 프롬프트에 주입되는 `KANBAN_GUIDANCE` 블록에 포함되므로 설치하거나 설정할 것이 **전혀 없습니다**. 이 안내는 CLI 명령이 아니라 **도구 호출**을 통해 전체 라이프사이클을 가르칩니다.

1. 생성 시 `kanban_show()`를 호출해 제목 + 본문 + 부모 핸드오프 + 이전 시도 + 전체 댓글 스레드를 읽습니다.
2. (터미널 도구를 통해) `cd $HERMES_KANBAN_WORKSPACE`로 이동해 그곳에서 작업합니다.
3. 긴 작업 중에는 몇 분마다 `kanban_heartbeat(note="...")`를 호출합니다. **작업이 1시간을 넘길 수 있다면 최소 한 시간에 한 번 `kanban_heartbeat`를 호출하세요.** 디스패처는 마지막 한 시간 동안 하트비트가 없고 `kanban.dispatch_stale_timeout_seconds`(기본 4시간)를 초과해 실행 중인 작업을, 정리 없이 워커가 중단된 것으로 보고 회수합니다. 회수는 무해합니다(작업은 실패 카운터를 증가시키지 않고 다시 디스패치되도록 `ready`로 돌아갑니다). 다만 현재 실행의 진행 상황은 잃게 됩니다.
4. `kanban_complete(summary="...", metadata={...})`로 완료하거나, 막힌 경우 `kanban_block(reason="...")`을 호출합니다.

마지막 `kanban_complete` / `kanban_block` 호출은 워커 프로토콜의 일부입니다. 프로세스가 상태가 여전히 `running`인 채로 종료 코드 0으로 종료되면, 디스패처는 이를 `protocol_violation` 이벤트를 발생시키는 프로토콜 위반으로 처리합니다.

**에이전트 측 예방:** 워커가 종료되기 전에 Hermes는 모델이 종료 도구를 호출하지 않고 멈추려는 것을 감지하면 최대 두 번의 합성 재촉을 주입합니다. 이는 모델이 다음 단계를 설명("보고서를 작성하겠습니다")한 뒤 `kanban_complete` 또는 `kanban_block`을 즉시 호출하지 않고 멈추는 흔한 상황을 포착합니다. 이 보호 장치는 디스패처가 생성한 워커(`HERMES_KANBAN_TASK` 설정 상태)에만 활성화되며 `HERMES_KANBAN_STOP_NUDGE=0`으로 비활성화할 수 있습니다.

**디스패처 측 복구:** 재촉이 모두 소진되거나 재촉에 도달하기 전에 워커가 중단되면, 디스패처는 작업을 같은 루프에 다시 생성하기 전에 위반에 대해 제한된 재시도를 수행합니다(연속 위반 최대 `_PROTOCOL_VIOLATION_FAILURE_LIMIT`회, 기본값 3). 카운터는 *연속으로 발생한* 정상 종료 프로토콜 위반만 계산합니다. 속도 제한으로 인한 재큐잉이 사이에 끼면 중립이며, 다른 실패 종류가 발생하면 연속 기록이 초기화됩니다. 작업별 `max_retries`가 이 한도를 덮어씁니다. 이는 보통 모델이 일반 텍스트 답변을 작성하고 Kanban 도구 표면을 사용하지 않은 채 종료했음을 의미합니다.

라이프사이클과, 부하를 지탱하는 참고 세부 정보(워크스페이스 종류, 결과물 `artifacts`, 생성된 카드의 클레임)는 시스템 프롬프트 블록에 포함되어 제공됩니다. 따라서 어떤 프로필로 실행되든 모든 워커가 이를 받으며, 프로필별 스킬 설정은 필요하지 않습니다.
### 특정 작업에 추가 스킬 고정

담당자 프로필에 없는 특정 작업 컨텍스트가 필요한 경우가 있습니다(번역 작업에 `translation` 스킬이 필요한 경우, 검토 작업에 `github-code-review`가 필요한 경우, 보안 감사에 `security-pr-audit`가 필요한 경우 등). 담당자 프로필을 수정하는 대신 작업에 스킬을 직접 연결하세요.

**오케스트레이터 에이전트에서**(일반적으로 사용자가 작업을 라우팅하는 경우) `kanban_create` 도구의 `skills` 배열을 사용합니다.

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

**사람이 직접**(CLI / 슬래시 명령) 지정할 때는 `--skill`을 반복해서 사용합니다.

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**대시보드에서**는 생성 작업 대화상자의 **skills** 필드에 스킬을 쉼표로 구분해 입력합니다.

디스패처는 나열된 각 스킬에 대해 `--skills <name>` 플래그를 하나씩 내보내며, 자동으로 주입되는 kanban 지침과 함께 모든 스킬을 워커에 로드합니다. 스킬 이름은 담당자 프로필에 실제로 설치된 스킬과 일치해야 합니다(`hermes skills list`로 확인). 런타임에 설치되지는 않습니다.

### 작업별 모델 재정의

담당자 프로필의 기본 모델과 관계없이 특정 모델(및 선택적으로 프로바이더)을 사용하도록 작업의 워커를 고정합니다.

```bash
# At creation
hermes kanban create "hard refactor" --assignee coder \
    --model claude-opus-4.6 --provider anthropic

# Or later — takes effect on the next dispatch
hermes kanban set-model t_abcd claude-opus-4.6 --provider anthropic
hermes kanban set-model t_abcd none    # clear the override
```

디스패처는 고정된 모델로 워커를 실행합니다(설정된 경우 `--provider <name>`도 전달되며, `--provider`에는 모델이 필요합니다). 대시보드의 작업별 모델 드롭다운은 동일한 `model_override` 필드를 설정합니다. 재정의가 없으면 워커는 프로필에 설정된 모델을 사용합니다.

### 비용 전략: 프런티어 오케스트레이터와 저비용 워커

Kanban의 프로필별 설정을 사용하면 계획 담당자와 워커의 비용을 자연스럽게 분리할 수 있습니다. 잘 정의된 카드를 여러 작업으로 분해하는 데는 프런티어 수준의 판단이 필요하지만, 이미 명확한 목표·컨텍스트·인계 근거가 담긴 카드를 실행하는 데는 대개 그렇지 않습니다. 토큰의 대부분은 워커에서 소비되므로 비용이 발생하는 지점도 워커 모델입니다. 오케스트레이터/디스패처 프로필은 프런티어 모델로 실행하고, 워커 프로필에는 저비용 모델을 지정하세요. 각 프로필에는 `~/.hermes/profiles/<name>/` 아래에 자체 `config.yaml`이 있으며, 디스패처는 `hermes -p <assignee>`를 실행할 때 프로필 범위의 `HERMES_HOME`을 주입하므로 각 워커가 자신의 프로필 모델 설정을 읽습니다.

```yaml
# ~/.hermes/config.yaml (orchestrator / dispatcher profile)
model:
  default: "your-frontier-model"

# ~/.hermes/profiles/coder/config.yaml (worker profile)
model:
  default: "your-inexpensive-model"

# ~/.hermes/profiles/researcher/config.yaml (another worker profile)
model:
  default: "your-inexpensive-model"
```

가끔 품질이 중요한 카드가 있다면 [작업별 모델 재정의](#per-task-model-override)(생성 시 `--model`/`--provider`, 이후 `hermes kanban set-model`, 또는 대시보드의 모델 드롭다운)를 사용해 해당 작업만 더 강력한 모델로 고정하세요. 프로필을 수정할 필요가 없습니다.

### 수명 주기 플러그인 훅

보드 전환은 [플러그인 훅](/user-guide/features/hooks#plugin-hooks)을 발생시킵니다: `kanban_task_claimed`, `kanban_task_completed`, `kanban_task_blocked`이며, 각각 `task_id`와 `profile_name`을 전달합니다. 훅은 보드 DB 변경 사항이 커밋된 **후** 실행되므로 콜백은 항상 영속화된 상태를 확인합니다. 프로세스 분리를 기억하세요: `kanban_task_claimed`는 **디스패처** 프로세스에서 실행되고, `kanban_task_completed`/`kanban_task_blocked`는 **워커** 프로세스에서 실행됩니다. 따라서 모든 전환을 중앙에서 관찰하려면 디스패처 프로필에 훅을 등록하세요.

```python
def register(ctx):
    def on_blocked(task_id=None, profile_name=None, **kw):
        ctx.dispatch_tool("terminal", {"command": f"notify-send 'kanban blocked: {task_id}'"})
    ctx.register_hook("kanban_task_blocked", on_blocked)
```

### 목표 모드 카드(`--goal`)

기본적으로 각 워커는 카드에 **한 번만** 시도합니다. 작업을 수행하고 `kanban_complete`/`kanban_block`을 호출한 뒤 종료합니다. 대신 `--goal`(CLI) 또는 `goal_mode=True`(`kanban_create` 도구/대시보드)를 전달하면 해당 워커를 **목표 루프**로 실행할 수 있습니다. 이는 `/goal` 슬래시 명령 뒤에서 동작하는 Ralph 스타일 엔진과 동일합니다. 매 턴이 끝날 때마다 보조 judge가 카드의 제목과 본문(수락 기준으로 취급)에 비추어 워커의 출력을 확인합니다. 작업이 끝나지 않았고 턴 예산이 남아 있으면 judge가 동의할 때까지, 워커가 직접 작업을 종료할 때까지 또는 예산이 소진될 때까지 **같은 세션에서** 계속 진행합니다. 예산이 소진되면 조용히 종료하지 않고 사람의 검토를 위해 카드를 **차단**합니다.

```bash
hermes kanban create "Translate the docs site to French" \
    --body "Acceptance: every page translated, no English left, links intact." \
    --assignee linguist \
    --goal \
    --goal-max-turns 15      # optional; default 20
```

개방형 작업, 여러 단계의 작업 또는 “X가 참이 될 때까지 계속”하는 카드에 사용하세요. 값싼 일회성 작업에는 생략하세요. 턴마다 judge가 추가하는 오버헤드를 감수할 가치가 없으며, 디스패처의 기존 재시도/회로 차단기가 일시적인 워커 실패를 이미 처리합니다. judge의 품질은 목표 텍스트의 품질만큼 좋으므로 본문을 **명시적인 수락 기준**으로 작성하세요.

:::note 목표 모드 카드는 `/goal` 엔진을 공유하지만 서로 연결되지는 않습니다
`--goal`은 *해당 카드의 워커 세션 하나 안에서* 계속 루프를 실행합니다. [`/goal` 슬래시 명령](./goals)과 엔진은 공유하지만 상태는 공유하지 않습니다. 채팅 세션에서 `/goal`을 설정해도 kanban 카드가 생성·클레임·이동되지 않으며, 목표 모드 카드의 루프도 채팅 세션의 `/goal status`에는 표시되지 않습니다. 이 대화를 계속 반복하려면 [`/goal`](./goals)을 사용하고, 보드에서 작업하려면 카드를 생성하세요.
:::

### 오케스트레이터의 동작 방식

**잘 작동하는 오케스트레이터는 직접 작업하지 않습니다.** 사용자의 목표를 작업으로 분해하고, 작업을 연결하고, 설정된 프로필 중 하나에 각각 할당한 다음 물러납니다. 오케스트레이터 지침(유혹 방지 규칙, Step-0 프로필 검색 프롬프트, 알 수 없는 담당자 이름에서는 디스패처가 조용히 실패하므로 실제로 존재하는 프로필을 기준으로 모든 카드를 구성해야 한다는 규칙, 그리고 `kanban_create`/`kanban_link`/`kanban_comment`에 맞춘 분해 플레이북)은 워커의 시스템 프롬프트에 자동으로 주입됩니다. 설치할 것은 없습니다.

표준적인 오케스트레이터 턴의 예시는 다음과 같습니다(두 연구자가 병렬로 작업한 뒤 작가에게 전달).

```
# Goal from user: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # promoted to 'ready' when both researchers complete
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# Optional: add cross-cutting deps discovered later without re-creating tasks
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

오케스트레이터 지침은 워커의 시스템 프롬프트에 자동으로 포함됩니다. 프로필마다 설치하거나 동기화할 것은 없습니다.

**분산하기 전에 결정하세요.** 설계 결정은 워커가 아니라 오케스트레이터의 몫입니다. 두 병렬 카드가 같은 사항(이름 지정 방식, 스키마, 파일 형식, API 형태 등)을 각각 결정해야 한다면 오케스트레이터가 한 번 결정하고 그 결정을 **두** 카드의 본문에 기록하세요. 워커는 형제 카드를 볼 수 없으므로 각 하위 카드의 본문에는 의존하는 모든 결정이 담겨야 합니다. 예를 들어 “내보내기 도구 만들기”와 “가져오기 도구 만들기”를 병렬 카드로 나눌 때 각 워커가 자체 파일 형식을 만들게 하지 마세요. 먼저 개행으로 구분된 JSON과 `version` 필드를 사용하기로 정하고, 그 내용을 두 본문에 모두 작성해야 두 절반이 서로 왕복할 수 있습니다.

최상의 결과를 얻으려면 도구 세트가 보드 작업(`kanban`, `gateway`, `memory`)으로 제한된 프로필과 함께 사용하세요. 그러면 오케스트레이터가 시도하더라도 구현 작업을 실제로 실행할 수 없습니다.

## 대시보드(GUI)

`/kanban` CLI와 슬래시 명령만으로도 보드를 헤드리스로 운영할 수 있지만, 사람이 참여하는 작업에서는 시각적 보드가 더 적합한 경우가 많습니다. 분류, 여러 프로필의 감독, 댓글 스레드 읽기, 열 사이의 카드 드래그 작업이 그 예입니다. Hermes는 이를 `plugins/kanban/`의 **번들 대시보드 플러그인**으로 제공합니다. [대시보드 확장](./extending-the-dashboard)에 설명된 모델을 따르며, 코어 기능도 별도 서비스도 아닙니다.

다음과 같이 엽니다.

```bash
hermes kanban init      # one-time: create kanban.db if not already present
hermes dashboard        # "Kanban" tab appears in the nav, after "Skills"
```

### 플러그인이 제공하는 기능

- 상태별로 하나의 열을 표시하는 **Kanban** 탭: `triage`, `todo`, `ready`, `running`, `blocked`, `done`(토글이 켜져 있으면 `archived`도 포함).
  - `triage`는 다듬어지지 않은 아이디어를 보관하는 열입니다. 기본값(`kanban.auto_decompose: true`)에서는 이곳에 들어온 작업에 **분해기**를 디스패처가 자동으로 실행합니다. 기본 제공 분해기는 `auxiliary.kanban_decomposer` 모델 경로를 사용하고, 프로필 목록(설명 포함)을 읽어 작업을 가장 적합한 전문 프로필로 라우팅되는 작은 하위 작업 그래프로 분배합니다. 원래 작업은 모든 하위 작업의 부모로 계속 남으며, 모든 하위 작업이 완료되면 담당자(`kanban.orchestrator_profile`, 설정되지 않은 경우 현재 기본 프로필)가 깨어나 완료 여부를 판단합니다. 페이지 상단의 **Orchestration: Auto/Manual** 알약을 전환하거나(`emerald` = 자동, 흐린 회색 = 수동), `config.yaml`을 직접 편집하세요. 두 모드는 `hermes kanban specify`와 함께 사용할 수 있습니다. 분산하지 않고 단일 작업의 사양을 다시 작성하고 싶을 때도 이 기능은 그대로 사용할 수 있습니다.
- 카드에는 작업 ID, 제목, 우선순위 배지, 테넌트 태그, 할당된 프로필, 댓글/링크 수, **진행 알약**(작업에 종속 항목이 있으면 완료된 하위 작업 `N/M`), “N ago로 생성됨”이 표시됩니다. 카드별 체크박스로 여러 항목을 선택할 수 있습니다.
- **Running 내부의 프로필별 레인** — 도구 모음의 체크박스로 Running 열을 담당자별 하위 그룹으로 전환합니다.
- **WebSocket을 통한 실시간 업데이트** — 플러그인은 짧은 폴링 간격으로 추가 전용 `task_events` 테이블을 추적합니다. 어떤 프로필(CLI, 게이트웨이 또는 다른 대시보드 탭)이 작업하든 보드에 즉시 반영됩니다. 이벤트가 몰리면 새로고침을 디바운스하여 한 번만 다시 가져옵니다.
- 카드를 열 사이에서 **드래그 앤 드롭**하여 상태를 변경합니다. 드롭은 `PATCH /api/plugins/kanban/tasks/:id`를 전송하고, CLI가 사용하는 것과 동일한 `kanban_db` 코드를 거칩니다. 따라서 세 표면의 상태가 절대 어긋나지 않습니다. 파괴적 상태(`done`, `archived`, `blocked`)로 이동하면 확인을 요청합니다. 터치 기기에서는 포인터 기반 대체 동작을 사용하므로 태블릿에서도 보드를 이용할 수 있습니다.
- **작업 생성 대화상자** — 열 헤더의 아무 곳에서나 `+`를 클릭하면 제목, 담당자, 우선순위, 스킬, 작업공간 종류/경로(보드의 프로젝트 디렉터리에서 초기화되며 작업별 재정의 가능), 목표 모드, (선택 사항으로) 모든 기존 작업을 보여 주는 드롭다운의 부모 작업을 입력하는 모달이 열립니다. Enter를 누르면 작업을 생성하고, Shift+Enter를 누르면 제목 필드에 줄바꿈을 삽입하며, Escape를 누르면 취소합니다. Triage 열에서 생성하면 새 작업이 자동으로 triage에 보관됩니다.
- **다중 선택 및 일괄 작업** — Shift/Ctrl을 누른 채 카드를 클릭하거나 체크박스를 선택해 선택 항목에 추가합니다. 상단에 일괄 작업 바가 나타나 상태 일괄 전환, 보관, 재할당(프로필 드롭다운 또는 “(unassign)”)을 제공합니다. 일괄 파괴 작업은 먼저 확인합니다. ID별 부분 실패는 나머지 작업을 중단하지 않고 보고합니다.
- **카드 클릭**(Shift/Ctrl 없음) 시 사이드 드로어가 열립니다(Escape 또는 바깥쪽 클릭으로 닫힘). 드로어에는 다음이 표시됩니다.
  - **편집 가능한 제목** — 제목을 클릭해 이름을 바꿉니다.
  - **편집 가능한 담당자/우선순위** — 메타 행을 클릭해 다시 작성합니다.
  - **편집 가능한 설명** — 기본적으로 Markdown으로 렌더링됩니다(제목, 굵게, 기울임, 인라인 코드, 펜스로 감싼 코드, `http(s)`/`mailto:` 링크, 글머리 기호 목록). “edit” 버튼을 누르면 textarea로 전환됩니다. Markdown 렌더러는 작고 XSS를 안전하게 처리합니다. 모든 치환은 HTML 이스케이프된 입력에 대해 실행되며 `http(s)`/`mailto:` 링크만 통과하고 `target="_blank"`와 `rel="noopener noreferrer"`는 항상 설정됩니다.
  - **종속성 편집기** — 부모와 자식의 칩 목록을 표시하며 각 항목에는 연결 해제를 위한 `×`가 있습니다. 다른 모든 작업을 보여 주는 드롭다운으로 새 부모 또는 자식을 추가할 수도 있습니다. 사이클을 만들려는 시도는 서버에서 명확한 메시지와 함께 거부됩니다.
  - **상태 작업 행**(→ triage / → ready / → running / block / unblock / complete / archive) — 파괴적 전환에는 확인 프롬프트가 표시됩니다. **Triage** 열의 카드에는 두 가지 LLM 기반 작업도 표시됩니다: **⚗ Decompose**는 설명에 따라 전문 프로필로 라우팅되는 하위 작업 그래프로 작업을 분해하고, **✨ Specify**는 단일 작업의 사양을 다시 작성합니다. LLM이 분산 작업에 이점이 없다고 판단하면 Decompose는 specify 방식의 승격으로 대체되므로, 엄밀한 상위 집합입니다. 둘 다 CLI(`hermes kanban decompose <id>` / `specify <id>` / `--all`), 모든 게이트웨이 플랫폼(`/kanban decompose <id>`), 그리고 `POST /api/plugins/kanban/tasks/:id/decompose` 및 `…/specify`를 통한 프로그래밍 방식으로 사용할 수 있습니다. `config.yaml`의 `auxiliary.kanban_decomposer`와 `auxiliary.triage_specifier`에서 모델을 설정하세요.
  - 결과 섹션(역시 Markdown으로 렌더링됨), Enter로 제출하는 댓글 스레드, 최근 이벤트 20개.
- **도구 모음 필터** — 자유 텍스트 검색, 테넌트 드롭다운(`config.yaml`의 `dashboard.kanban.default_tenant`가 기본값), 담당자 드롭다운, “show archived” 토글, “lanes by profile” 토글, 다음 60초 틱을 기다리지 않고 디스패처를 깨우는 **Nudge dispatcher** 버튼.

시각적으로는 익숙한 Linear / Fusion 레이아웃을 목표로 합니다. 어두운 테마, 개수가 표시된 열 헤더, 색이 있는 상태 점, 우선순위와 테넌트용 알약 칩을 사용합니다. 플러그인은 테마 CSS 변수(`--color-*`, `--radius`, `--font-mono`, ...)만 읽으므로 현재 활성화된 대시보드 테마에 맞춰 자동으로 다시 스타일링됩니다.
### 자동 오케스트레이션과 수동 오케스트레이션

칸반 보드의 Triage 열에 추가한 작업을 처리하는 방법은 두 가지입니다:

**자동(기본값)** — `kanban.auto_decompose: true`. 게이트웨이에 내장된 디스패처가 각 틱마다 **decomposer**를 실행하며, `kanban.auto_decompose_per_tick`(기본값 3)으로 제한되므로 Triage 작업을 한꺼번에 많이 넣어도 보조 LLM 비용이 급증하지 않습니다. Decomposer는 내장 분해 프롬프트와 `auxiliary.kanban_decomposer` 모델 경로를 사용하고, 설치된 프로필과 각 프로필의 설명을 읽은 뒤 LLM에 작업 그래프(JSON)를 생성하도록 요청합니다. 즉, 어떤 작업을 생성할지, 누구에게 할당할지, 어떤 작업이 어떤 작업에 의존하는지를 결정합니다. 원래 Triage 작업은 그래프의 모든 리프 작업의 부모가 되므로 전체 그래프가 완료될 때까지 유지됩니다. 이후 `kanban.orchestrator_profile`에 지정된 프로필(지정하지 않으면 현재 활성 기본 프로필)에게 판단을 맡기도록 다시 `ready` 상태로 승격되며, 오케스트레이터는 작업이 완료되지 않았다면 추가 작업을 생성할 수 있습니다. 이는 "한 줄을 남겨 두고 자리를 비우는" 방식입니다.

**수동** — `kanban.auto_decompose: false`. Triage 작업은 사용자가 조작할 때까지 Triage에 남습니다. 카드에서 **⚗ Decompose** 버튼을 클릭하거나, `hermes kanban decompose <id>`(또는 `--all`)를 실행하거나, 채팅에서 `/kanban decompose <id>`를 사용하세요. 작업이 언제 실행되는지 완전히 제어하고 싶을 때 유용하며, 보드가 decomposer를 도입하기 전의 동작과 같습니다.

**중요한 경계:** 수동 모드는 내장 Triage decomposer만 비활성화합니다. 프로필이 `kanban_create`를 호출하는 것을 막거나 creator-session wake-up을 비활성화하지는 않습니다. `kanban.auto_subscribe_on_create: true`이면 작업의 terminal event가 원래 에이전트를 synthetic status turn으로 재개하므로, 에이전트가 handoff를 확인하고 정말 새로운 후속 작업이 필요한지 판단할 수 있습니다. 작업 완료를 수동적으로 처리하려면 `auto_subscribe_on_create: false`로 설정하세요. 출처를 구분하기 위해 내장 decomposer가 만든 자식 작업에는 `created_by=auto-decomposer`가 사용되고, 재개된 프로필이 만든 작업에는 해당 프로필 이름이 사용됩니다.

칸반 페이지 상단의 **Orchestration: Auto/Manual** pill(에메랄드색 = Auto, 흐린 회색 = Manual)에서 두 모드 사이를 전환하거나, `config.yaml`을 직접 편집할 수 있습니다. 두 모드 모두 `hermes kanban specify`와 함께 사용할 수 있습니다. 여러 작업으로 fan-out하지 않고 단일 작업의 명세만 다시 작성하고 싶을 때 여전히 사용할 수 있습니다.

Decomposer의 라우팅 결정은 프로필 설명에 따라 달라집니다. 프로필 설명은 `hermes profile create --description "..."`, `hermes profile describe <name> --text "..."`, `hermes profile describe <name> --auto`(프로필에 설치된 skill과 모델에서 LLM이 생성)로 설정하거나, 확장된 **Orchestration settings** 패널의 프로필별 편집기에서 설정할 수 있는 프로필별 라벨링 기능입니다. 설명이 없는 프로필도 roster에는 표시됩니다. 이름으로 라우팅할 수는 있지만 정밀도가 떨어집니다. Decomposer는 `assignee=None`인 자식 작업을 절대 만들지 않습니다. LLM이 알 수 없는 프로필을 선택하면 해당 자식 작업은 `kanban.default_assignee`(설정되지 않았으면 현재 활성 기본 프로필)로 라우팅됩니다.

`kanban.orchestrator_profile`은 해당 프로필의 프롬프트, skill 또는 사용자 지정 로직을 decomposition 호출에 불러오지 않습니다. fan-out 후 root/orchestration 작업을 누가 소유할지 제어할 뿐입니다. Decomposer의 모델/provider를 변경하려면 `auxiliary.kanban_decomposer`를 설정하세요. 내장 decomposer 대신 프로필의 사용자 지정 작업 분할 로직을 사용하려면 Manual 모드로 전환한 다음 해당 프로필이 명시적으로 작업을 생성하거나 분해하도록 하세요.

설정 항목(`~/.hermes/config.yaml`의 `kanban:` 아래):

| 키 | 기본값 | 용도 |
|---|---|---|
| `auto_decompose` | `true` | 디스패처가 매 틱마다 Triage 작업에 대해 내장 decomposer를 자동 실행합니다. 프로필이 호출하는 `kanban_create`나 creator wake turn을 제어하지는 않습니다. |
| `auto_decompose_per_tick` | `3` | 디스패처 틱당 decomposition 수의 상한입니다. 초과분은 다음 틱으로 미뤄집니다. |
| `orchestrator_profile` | `""` | decomposition 후 root/orchestration 작업에 할당할 프로필입니다. 비어 있으면 활성 기본 프로필로 대체됩니다. |
| `default_assignee` | `""` | LLM이 알 수 없는 프로필을 선택했을 때 자식 작업을 배치할 곳입니다. 비어 있으면 활성 기본 프로필로 대체됩니다. |
| `auto_subscribe_on_create` | `true` | 영속 gateway/TUI 세션 안에서 `kanban_create`가 실행되면 terminal event가 해당 작업을 만든 에이전트를 synthetic status turn으로 재개합니다. 완료를 수동적으로 처리하거나 명시적인 `kanban_notify-subscribe` 호출을 요구하려면 `false`로 설정하세요. `auto_decompose`와는 독립적입니다. |
| `done_sub_retention_days` | `30` | 알림 구독은 `done` 상태에서도 유지되어 재개할 수 있으며, `archived` 상태가 되면 제거됩니다. notifier GC는 이 기간(일) 동안 새 이벤트가 없이 `done` 상태인 작업의 구독을 정리합니다. 이를 통해 절대 archive되지 않는 보드에서도 하위 테이블의 증가를 제한합니다. `0`이면 정리를 비활성화합니다. |

그리고 두 개의 보조 LLM 슬롯이 있습니다:

| 키 | 용도 |
|---|---|
| `auxiliary.kanban_decomposer` | 작업 그래프를 생성하는 모델입니다(Decompose에서 호출). `provider`/`model`을 설정해 주 채팅 모델을 재정의할 수 있습니다. |
| `auxiliary.profile_describer` | 프로필 설명을 자동 생성하는 모델입니다(`hermes profile describe --auto`에서 호출). |

### 아키텍처

GUI는 자체 도메인 로직 없이 엄격하게 **DB에서 읽고 kanban_db를 통해 쓰는** 계층입니다:

<!-- ascii-guard-ignore -->
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
<!-- ascii-guard-ignore-end -->

### REST 표면

모든 경로는 `/api/plugins/kanban/` 아래에 마운트되며 dashboard의 임시 세션 토큰으로 보호됩니다:

| 메서드 | 경로 | 용도 |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | 상태 열별로 그룹화한 전체 보드와 필터 드롭다운에 사용할 tenants + assignees |
| `GET` | `/tasks/:id` | 작업 + 댓글 + 이벤트 + 링크 |
| `POST` | `/tasks` | 생성(`kanban_db.create_task` 래핑, `triage: bool` 및 `parents: [id, …]` 수락) |
| `PATCH` | `/tasks/:id` | 상태 / 담당자 / 우선순위 / 제목 / 본문 / 결과 |
| `POST` | `/tasks/bulk` | `ids`의 모든 id에 동일한 patch(상태 / archive / 담당자 / 우선순위) 적용. 형제 작업을 중단하지 않고 id별 실패를 보고 |
| `POST` | `/tasks/:id/comments` | 댓글 추가 |
| `POST` | `/tasks/:id/specify` | Triage specifier 실행 — 보조 LLM이 작업 본문을 구체화하고 `triage`에서 `todo`로 승격합니다. `{ok, task_id, reason, new_title}`를 반환하며, "not in triage" / aux client 없음 / LLM 오류에서는 사람이 읽을 수 있는 사유와 함께 `ok=false`를 4xx가 아닌 200으로 반환합니다. |
| `POST` | `/tasks/:id/decompose` | kanban decomposer 실행 — 보조 LLM이 작업 그래프를 생성하고 helper가 자식 작업과 링크를 원자적으로 생성한 뒤 root를 연결하고 `triage → todo`로 전환합니다. `{ok, task_id, reason, fanout, child_ids, new_title}`를 반환합니다. `/specify`와 동일하게 LLM 오류에서도 200을 반환합니다. |
| `GET` | `/profiles` | 설명이 포함된 설치 프로필 목록(대시보드의 프로필 설명 편집기와 오케스트레이터 선택기에서 사용) |
| `PATCH` | `/profiles/:name` | 프로필 설명 설정 또는 삭제(사용자 작성 — `description_auto: false`). `{ok, profile, description}`을 반환합니다. |
| `POST` | `/profiles/:name/describe-auto` | `auxiliary.profile_describer`로 프로필 설명 생성. `description_auto: true`로 저장하여 대시보드에서 "review" badge를 표시할 수 있게 합니다. |
| `GET` | `/orchestration` | 칸반 오케스트레이션 설정(`orchestrator_profile`, `default_assignee`, `auto_decompose`)과 fallback 적용 후의 *resolved* 유효값을 읽습니다. |
| `PUT` | `/orchestration` | `config.yaml`의 세 오케스트레이션 키 중 하나 이상을 업데이트합니다. 비어 있지 않은 프로필 이름이 실제로 존재하는지 검증합니다. |
| `POST` | `/links` | 의존성 추가(`parent_id` → `child_id`) |
| `DELETE` | `/links?parent_id=…&child_id=…` | 의존성 제거 |
| `POST` | `/dispatch?max=…&dry_run=…` | 디스패처를 깨워 60초 대기를 건너뜁니다. |
| `GET` | `/config` | `config.yaml`에서 dashboard.kanban 설정을 읽습니다 — `default_tenant`, `lane_by_profile`, `include_archived_by_default`, `render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events` 행의 실시간 스트림 |

모든 handler는 얇은 래퍼입니다. 이 plugin은 약 700줄의 Python(router + WebSocket tail + bulk batcher + config reader)으로 구성되며 새로운 비즈니스 로직을 추가하지 않습니다. 작은 `_conn()` helper가 모든 읽기와 쓰기에서 `kanban.db`를 자동 초기화하므로, 사용자가 먼저 dashboard를 열었든 REST API에 직접 접근했든 `hermes kanban init`을 실행했든 새로 설치한 환경에서 작동합니다.

### Dashboard 설정

`~/.hermes/config.yaml`의 `dashboard.kanban` 아래에 있는 다음 키 중 하나라도 탭의 기본값을 변경합니다. plugin은 로드 시 `GET /config`를 통해 이 값을 읽습니다:

```yaml
dashboard:
  kanban:
    default_tenant: acme              # preselects the tenant filter
    lane_by_profile: true             # default for the "lanes by profile" toggle
    include_archived_by_default: false
    render_markdown: true             # set false for plain <pre> rendering
```

각 키는 선택 사항이며 표시된 기본값으로 대체됩니다.

### 보안 모델

dashboard의 HTTP 인증 middleware는 [`/api/plugins/`](./extending-the-dashboard#backend-api-routes)를 **명시적으로 건너뜁니다** — dashboard가 기본적으로 localhost에 바인딩되므로 plugin 경로는 설계상 인증되지 않습니다. 따라서 kanban REST 표면은 호스트의 모든 프로세스에서 접근할 수 있습니다.

WebSocket은 한 단계 더 확인합니다. 브라우저는 upgrade 요청에 `Authorization`을 설정할 수 없으므로 dashboard의 임시 세션 토큰을 `?token=…` query parameter로 요구합니다. 이는 브라우저 내 PTY bridge에서 사용하는 패턴과 같습니다.

`hermes dashboard --host 0.0.0.0`을 실행하면 kanban을 포함한 모든 plugin 경로가 네트워크에서 접근 가능해집니다. **공유 호스트에서는 이렇게 하지 마세요.** 보드에는 작업 본문, 댓글, workspace 경로가 포함되어 있으므로, 이 경로에 접근한 공격자는 전체 협업 표면을 읽을 수 있고 작업을 생성하거나 재할당하거나 archive할 수도 있습니다.

`~/.hermes/kanban.db`의 작업은 의도적으로 프로필과 무관합니다(이것이 coordination primitive입니다). `hermes -p <profile> dashboard`로 dashboard를 열어도 보드에는 호스트의 다른 프로필이 생성한 작업이 계속 표시됩니다. 모든 프로필을 같은 사용자가 소유하지만, 여러 persona가 공존할 때는 이 점을 알아 두어야 합니다.

### 실시간 업데이트

`task_events`는 단조 증가하는 `id`를 가진 append-only SQLite 테이블입니다. WebSocket endpoint는 각 클라이언트가 마지막으로 본 event id를 보관하고 새 행이 추가되는 즉시 전달합니다. 이벤트가 한꺼번에 들어오면 frontend는 (매우 저렴한) board endpoint를 다시 로드합니다. 모든 이벤트 종류에서 로컬 상태를 patch하려는 것보다 간단하고 정확합니다. WAL 모드에서는 읽기 루프가 dispatcher의 `BEGIN IMMEDIATE` claim transaction을 차단하지 않습니다.
### 확장하기

이 플러그인은 표준 Hermes 대시보드 플러그인 계약을 사용합니다. 전체 매니페스트 참조, 셸 슬롯, 페이지 범위 슬롯, Plugin SDK는 [대시보드 확장](./extending-the-dashboard)을 참고하세요. 추가 열, 사용자 지정 카드 크롬, 테넌트로 필터링된 레이아웃, 전체 `tab.override` 교체를 모두 이 플러그인을 포크하지 않고도 표현할 수 있습니다.

플러그인을 제거하지 않고 비활성화하려면 `config.yaml`에 `dashboard.plugins.kanban.enabled: false`를 추가하세요(또는 `plugins/kanban/dashboard/manifest.json`을 삭제하세요).

### 범위 경계

GUI는 의도적으로 얇게 구성되어 있습니다. 플러그인이 수행하는 모든 작업은 CLI에서 사용할 수 있으며, 플러그인은 이를 사람이 편리하게 사용할 수 있도록 만들 뿐입니다. 자동 할당, 예산, 거버넌스 게이트, 조직도 뷰는 설계 사양의 범위를 벗어난 항목에 명시된 대로 사용자 영역에 남습니다. 즉, 라우터 프로필, 다른 플러그인 또는 `tools/approval.py` 재사용의 몫입니다.

## CLI 명령 참조

이 표면은 **사용자**(또는 스크립트, cron, 대시보드)가 보드를 조작할 때 사용합니다. 디스패처 내부에서 실행되는 워커는 동일한 작업에 `kanban_*` [도구 표면](#how-workers-interact-with-the-board)을 사용합니다. 이 CLI와 도구는 모두 `kanban_db`를 통해 라우팅되므로 두 표면은 설계상 동일하게 동작합니다.

```
hermes kanban init                                     # create kanban.db + print daemon hint
hermes kanban create "<title>" [--body ...] [--assignee <profile>]
                                [--parent <id>]... [--tenant <name>]
                                [--workspace scratch|worktree|worktree:<path>|dir:<path>]
                                [--branch <name>]
                                [--priority N] [--triage] [--idempotency-key KEY]
                                [--max-runtime 30m|2h|1d|<seconds>]
                                [--max-retries N]
                                [--goal] [--goal-max-turns N]
                                [--skill <name>]...
                                [--json]
hermes kanban list [--mine] [--assignee P] [--status S] [--tenant T] [--archived]
        [--workflow-template-id <id>] [--current-step-key <key>]
        [--sort created|created-desc|priority|priority-desc|status|assignee|title|updated]
        [--json]
hermes kanban show <id> [--json]
hermes kanban assign <id> <profile>                    # or 'none' to unassign
hermes kanban reassign <id>... <profile>               # bulk re-assign tasks to a profile
hermes kanban edit <id> [--title ...] [--body ...]     # edit task title / body / priority in place
        [--priority N]
hermes kanban promote <id>...                          # move todo/blocked tasks to ready (recovery)
hermes kanban schedule <id> --at <ISO8601>             # set/clear a task's scheduled_at start time
hermes kanban diagnostics [--json]                     # board health snapshot (alias: diag)
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# Bulk verbs — accept multiple ids:
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban request-review <id> [--summary "..."] [--metadata JSON] [--reviewer PROFILE]
hermes kanban request-changes <id> "<required changes>"               # active reviewer -> implementer
hermes kanban reopen-review  <id>... [--reason "..."]                 # changes requested: 'review' -> ready/todo

hermes kanban tail <id>                                # follow a single task's event stream
hermes kanban watch [--assignee P] [--tenant T]        # live stream ALL events to the terminal
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # worker liveness signal for long ops
hermes kanban runs <id> [--json]                       # attempt history (one row per run)
hermes kanban assignees [--json]                       # profiles on disk + per-assignee task counts
hermes kanban dispatch [--dry-run] [--max N]           # one-shot pass
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # DEPRECATED — standalone dispatcher (use `hermes gateway start` instead)
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # per-status + per-assignee counts
hermes kanban log <id> [--tail BYTES]                  # worker log from ~/.hermes/kanban/logs/
hermes kanban notify-subscribe <id>                    # gateway bridge hook (used by /kanban in the gateway)
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
        [--chat-type dm|group|channel|thread] [--delivery-mode notify|notify+wake|wake]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # what a worker sees
hermes kanban specify [<id> | --all] [--tenant T]      # flesh out a triage-column idea
        [--author NAME] [--json]                       #   into a full spec and promote to todo
hermes kanban gc [--event-retention-days N]            # workspaces + old events + old logs
        [--log-retention-days N]
```

모든 명령은 대화형 CLI와 메시징 게이트웨이에서 슬래시 명령으로도 사용할 수 있습니다(아래 [`/kanban` 슬래시 명령](#kanban-slash-command) 참고).

`--max-retries`는 디스패처에 대한 작업별 회로 차단기 재정의입니다. `--max-retries 1`은 첫 번째 성공하지 못한 시도에서 작업을 차단하고, `--max-retries 3`은 두 번 재시도한 뒤 세 번째 실패에서 차단합니다. 생략하면 `config.yaml`의 `kanban.failure_limit`을 사용하고, 그다음에는 내장 기본값을 사용합니다.

### 동시성, 예약, 하위 작업 승격 설정

| 설정 키 | 기본값 | 동작 |
|------------|---------|--------------|
| `kanban.max_in_progress` | 설정되지 않음 (무제한) | 동시에 실행되는 작업 수를 제한합니다. 보드에 이미 N개의 작업이 실행 중이면 디스패처는 추가 작업을 생성하지 않습니다. 느린 워커(로컬 LLM, 리소스가 제한된 호스트)가 작업을 끝내기도 전에 더 많은 작업이 쌓여 시간 초과되는 것을 방지하는 데 유용합니다. 유효하지 않거나 1 미만인 값은 경고를 기록하고 무제한으로 처리합니다. |
| `kanban.max_in_progress_per_profile` | 설정되지 않음 (무제한) | `max_in_progress`의 프로필별 변형으로, 하나의 할당자 프로필이 동시에 실행할 수 있는 작업 수를 제한합니다. 한 프로필은 느리거나 속도 제한을 받더라도 다른 프로필은 계속 진행해야 할 때 유용합니다. 보드 전체의 `max_in_progress`와 함께 적용되며, 작업을 진행하려면 두 설정 모두 생성을 허용해야 합니다. |
| `kanban.auto_promote_children` | `true` | `decompose_triage_task()`가 상위 작업 차단 종속성이 없는 하위 작업을 생성하면 자동으로 `ready`로 승격하여 디스패처가 가져갈 수 있게 합니다. `false`로 설정하면 수동 검토가 필요하며, 승격할 때까지 하위 작업은 `todo`에 남습니다. |
| `kanban.default_workdir` | 설정되지 않음 | `--workspace`나 작업 자체에서 재정의하지 않은 새 작업에 적용되는 보드 수준의 기본 작업 디렉터리입니다. 작업별 `workspace:`가 우선합니다. |

```yaml
kanban:
  max_in_progress: 2
  auto_promote_children: false
  default_workdir: ~/work/active-project
```

### 예약된 작업 시작 (`scheduled_at`)

작업에 `scheduled_at`을 설정하면 특정 시간까지 디스패치를 지연할 수 있습니다. 디스패처는 미래로 설정된 `scheduled_at`을 가진 ready 작업을 건너뛰고, 해당 시각 이후 첫 번째 틱에서 작업을 가져갑니다.

```bash
hermes kanban create "nightly backup audit" \
  --assignee ops --scheduled-at "2026-06-01T03:00:00Z"
```

### 재생성 방지 가드

디스패처는 이전 실행에서 할당량/인증/429 오류(`blocker_auth`)가 발생했거나, 가드 기간 내에 실행을 성공적으로 완료했거나(`recent_success`), 최근 작업 댓글에 GitHub PR 링크가 있을 때(`active_pr`) ready 작업을 다시 생성하지 않습니다. 이는 사람이 대응하는 동안 동일한 버그나 작업에 워커가 반복적으로 폭주하는 것을 방지합니다. [이벤트 참조](#event-reference)의 `respawn_guarded` 행을 참고하세요.

### 드래그 삭제 및 일괄 삭제 (대시보드)

대시보드의 칸반 페이지에는 **휴지통 드롭 영역**이 있습니다. 카드를 이 영역으로 드래그하면 작업이 삭제됩니다(`task_events`, 하위 작업 링크, 구독까지 연쇄 삭제). 확인 프롬프트가 실수로 인한 삭제를 방지합니다. JSON 본문 `{"ids": ["t_abc", "t_def", ...]}`를 사용하는 `DELETE /api/plugins/kanban/tasks`로도 일괄 삭제할 수 있습니다.

### 워커 가시성 엔드포인트

대시보드 플러그인 API는 이제 외부 모니터를 위해 다음 읽기 전용 엔드포인트(및 실행 제어 동사)를 제공합니다.

| 엔드포인트 | 반환값 |
|----------|---------|
| `GET /api/plugins/kanban/workers/active` | 현재 생성된 워커와 PID, 프로필, 작업 id, 시작 시각, 마지막 heartbeat |
| `GET /api/plugins/kanban/runs/{id}` | 단일 실행 세부 정보 — 작업 id, 상태, 시작/종료 시각, 종료 코드, 로그 경로 |
| `POST /api/plugins/kanban/runs/{run_id}/terminate` | 회수 가능한 실행 종료 — 워커를 중지하고 작업을 다시 디스패치할 수 있도록 해제 |
| `GET /api/plugins/kanban/inspect` | 통합 디스패처 스냅샷 — 대기열, `max_in_progress` 대비 진행 중 수, 최근 이벤트 |

이 모든 엔드포인트에는 칸반 플러그인 API의 나머지 부분과 동일한 대시보드 플러그인 인증이 적용됩니다.

### 칸반 Swarm 토폴로지 헬퍼

`hermes kanban swarm`은 한 번에 지속 가능한 **Kanban Swarm v1** 그래프를 생성합니다. 여기에는 완료된 루트/블랙보드 카드, N개의 병렬 워커 카드, 모든 워커를 조건으로 하는 검증자 카드, 검증자를 조건으로 하는 합성자 카드가 포함됩니다. 공유 Swarm 컨텍스트("블랙보드")는 루트 카드의 구조화된 JSON 댓글로 저장되므로 모든 워커가 읽을 수 있습니다.

```bash
hermes kanban swarm "Design a multi-region failover plan" \
  --workers researcher,architect,sre \
  --verifier reviewer --synthesizer writer
```

생성된 그래프는 원자적으로 커밋됩니다. 디스패처와 대시보드 리더는 부분적으로 연결된 루트/워커/검증자 그래프가 아니라, 새로운 Swarm이 전혀 없거나 완전한 토폴로지만 보게 됩니다. 이후 정상적으로 디스패치됩니다. 워커는 병렬로 실행되고, 모든 워커가 완료되면 검증자가 깨어나며, 검증자가 작업이 정상이라고 표시하면 합성자가 깨어납니다.

## `/kanban` 슬래시 명령 {#kanban-slash-command}

모든 `hermes kanban <action>` 동사는 대화형 `hermes chat` 세션 내부와 모든 게이트웨이 플랫폼(Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, email, SMS)에서 `/kanban <action>`으로도 사용할 수 있습니다. 두 표면은 동일한 `hermes_cli.kanban.run_slash()` 진입점을 호출하고, 이 진입점은 `hermes kanban` argparse 트리를 재사용하므로 CLI, `/kanban`, `hermes kanban`에서 인자 표면, 플래그, 출력 형식이 동일합니다. 보드를 조작하기 위해 채팅을 나갈 필요가 없습니다.

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # flesh out a triage one-liner into a real spec
/kanban specify --all --tenant engineering  # sweep every triage task in one tenant
```

셸에서와 같은 방식으로 여러 단어로 된 인자를 따옴표로 묶으세요. `run_slash`는 줄의 나머지 부분을 `shlex.split`으로 파싱하므로 `"..."`와 `'...'`을 모두 사용할 수 있습니다.

### 실행 중 사용: `/kanban`은 실행 중 에이전트 가드를 우회합니다

게이트웨이는 일반적으로 에이전트가 아직 사고 중일 때 슬래시 명령과 사용자 메시지를 대기열에 넣습니다. 첫 번째 턴이 진행 중일 때 실수로 두 번째 턴을 시작하는 것을 막기 위해서입니다. **`/kanban`은 이 가드에서 명시적으로 제외됩니다.** 보드는 실행 중인 에이전트의 상태가 아니라 `~/.hermes/kanban.db`에 있으므로, 읽기(`list`, `show`, `context`, `tail`, `watch`, `stats`, `runs`)와 쓰기(`comment`, `unblock`, `block`, `assign`, `archive`, `create`, `link`, …)가 턴 중에도 모두 즉시 처리됩니다.

이 분리가 존재하는 이유는 다음과 같습니다.

- 워커가 동료를 기다리며 차단된 경우 → 휴대폰에서 `/kanban unblock t_abcd`를 보내면 디스패처가 다음 틱에 동료를 가져갑니다. 차단된 워커는 중단되지 않고, 더 이상 차단된 상태가 아니게 됩니다.
- 사람의 컨텍스트가 필요한 카드를 발견한 경우 → `/kanban comment t_xyz "use the 2026 schema, not 2025"`를 보내면 작업 스레드에 기록되고, 해당 작업의 *다음* 실행이 `kanban_show()`에서 이를 읽습니다.
- 오케스트레이터를 중지하지 않고 플릿이 무엇을 하고 있는지 확인하려는 경우 → `/kanban list --mine` 또는 `/kanban stats`로 주 대화에 영향을 주지 않고 보드를 살펴볼 수 있습니다.
### `/kanban create` 시 자동 구독 (게이트웨이 전용)

게이트웨이에서 `/kanban create "…"`로 작업을 생성하면, 생성한 채팅(플랫폼 + 채팅 ID + 스레드 ID)이 해당 작업의 종료 이벤트(`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`)를 자동으로 구독합니다. 작업 ID를 폴링하거나 기억하지 않아도 터미널 이벤트마다 한 번씩 메시지를 받으며, `completed` 이벤트에는 워커 결과의 `--result` 요약 첫 줄도 포함됩니다.

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

구독은 작업이 `done` 상태에 도달해도 유지됩니다. 완료는 되돌릴 수 있는 상태이므로(리뷰어나 컨트롤러가 완료된 작업을 다시 열 수 있음), 원래 세션은 다시 열리는 주기에도 계속 알림을 받습니다. 구독은 되돌릴 수 없는 최종 상태인 `archived`에서 자동으로 제거됩니다. 아카이브하지 않는 보드에서는 새 활동 없이 `done` 상태에 머문 작업을 `kanban.done_sub_retention_days`일(기본값 30일) 동안 보존한 뒤 GC 스윕이 구독 행을 정리합니다. 0으로 설정하면 이 기능을 비활성화할 수 있습니다. `--json`(기계 출력)으로 생성을 스크립트에서 수행하면 자동 구독을 건너뜁니다. 스크립트 호출자는 `/kanban notify-subscribe`를 통해 구독을 명시적으로 관리한다고 가정하기 때문입니다.

채팅에서 생성된 자동 구독은 `notify+wake` 모드로 생성됩니다. 종료 이벤트가 발생하면 대상 에이전트가 수동 메시지를 받는 동시에 실제 턴도 수행하므로, 보드 컨텍스트를 읽고 자신의 말투로 답할 수 있습니다. 자세한 내용은 아래 [전달 모드](#delivery-modes)를 참고하세요.

### 메시징에서의 출력 잘림

게이트웨이 플랫폼에는 실제 메시지 길이 제한이 있습니다. `/kanban list`, `/kanban show`, `/kanban tail`의 출력이 약 3800자를 넘으면 다음 꼬리말과 함께 응답이 잘립니다: `… (truncated; use \`hermes kanban …\` in your terminal for full output)`. CLI에는 이러한 제한이 없습니다.

### 자동 완성

대화형 CLI에서 `/kanban `을 입력하고 Tab 키를 누르면 기본 제공 하위 명령 목록(`list`, `ls`, `show`, `create`, `assign`, `link`, `unlink`, `claim`, `comment`, `complete`, `block`, `unblock`, `archive`, `tail`, `dispatch`, `context`, `init`, `gc`)을 순환합니다. 위 CLI 참고 문서에 나열된 나머지 동사(`watch`, `stats`, `runs`, `log`, `assignees`, `heartbeat`, `notify-subscribe`, `notify-list`, `notify-unsubscribe`, `daemon`)도 사용할 수 있습니다. 아직 자동 완성 힌트 목록에만 포함되지 않았을 뿐입니다.

## 협업 패턴

보드는 새로운 기본 기능 없이도 다음 여덟 가지 패턴을 지원합니다.

| 패턴 | 형태 | 예시 |
|---|---|---|
| **P1 Fan-out** | 같은 역할의 N개 형제 | "5가지 관점을 병렬로 조사" |
| **P2 Pipeline** | 역할 체인: 정찰 → 편집 → 작성 | 일일 브리프 조립 |
| **P3 Voting / quorum** | N개 형제 + 집계자 1명 | 연구자 3명 → 리뷰어 1명이 선택 |
| **P4 Long-running journal** | 동일 프로필 + 공유 디렉터리 + cron | Obsidian 볼트 |
| **P5 Human-in-the-loop** | 워커 차단 → 사용자가 댓글 작성 → 차단 해제 | 모호한 결정 |
| **P6 `@mention`** | 본문에서 인라인 라우팅 | `@reviewer look at this` |
| **P7 Thread-scoped workspace** | 스레드에서 `/kanban here` | 프로젝트별 게이트웨이 스레드 |
| **P8 Fleet farming** | 프로필 하나, 대상 N개 | 소셜 계정 50개 |
| **P9 Triage specifier** | 대략적인 아이디어 → `triage` → `hermes kanban specify`가 본문 확장 → `todo` | "이 한 줄짜리 아이디어를 사양이 있는 작업으로 바꿔" |

각 패턴의 실제 예시는 `docs/hermes-kanban-v1-spec.pdf`를 참고하세요.

## 후속 카드에 컨텍스트 전달하기 (부모 링크)

부모 링크는 단순한 일정 조정 게이트가 아니라, **완료된** 카드에서 새 카드로 컨텍스트를 전달하는 채널입니다. `--parent <done-card-id>`로 카드를 생성하면 두 가지 일이 일어납니다.

1. **즉시 실행 가능해집니다.** `create_task`는 부모 상태에 따라 상태를 설정합니다. 모든 부모가 `done`인 자식은 대기나 수동 승격 없이 바로 `ready`로 생성됩니다. (아직 열린 부모의 자식은 마지막 부모가 끝나 `recompute_ready`가 승격할 때까지 `todo`에 머뭅니다.)
2. **부모의 인계 내용이 함께 전달됩니다.** 자식을 위해 조립되는 워커 컨텍스트(`build_worker_context`, `kanban_show()`가 반환하는 내용)에는 각 부모의 완료 `summary`와 `metadata`가 원문 그대로 담긴 `## Parent task results` 섹션이 포함됩니다.

```
## Parent task results
### t_77c26979 (completed just now)
Added exponential backoff with jitter to the retry helper.
_metadata_: `{"changed_files": ["hermes_cli/retry.py", "tests/test_retry.py"], "decisions": ["capped backoff at 60s", "jitter = full"]}`
```

이 때문에 완료된 카드에 대한 후속 작업은 완료 카드를 다시 여는 대신 **새 자식 카드**로 만드는 것이 권장되는 패턴입니다. 완료된 카드는 변경할 수 없는 기록이며, 해당 컨텍스트는 부모 링크를 통해 앞으로 전달됩니다. 동일 카드 재작업(실패한 카드의 재시도 루프)은 별도의 메커니즘입니다. 같은 카드의 이전 시도는 해당 카드 자체 컨텍스트에서 "prior attempts"로 표시됩니다.

워크트리나 브랜치만으로는 충분하지 않습니다. 저장소 상태는 후속 워커에게 코드가 *어떤 모습인지* 알려 주지만 *왜 그런 모습인지*는 알려 주지 않습니다. 결정 사항, 실행한 테스트, 변경한 파일은 git이 아니라 부모의 구조화된 인계 내용에 있습니다. 부모 작업이 완료된 뒤 발생한 증거(예: 나중에 실패한 CI 로그)는 새 카드의 **본문**에 넣어야 합니다.

```bash
# Implementation card t_impl is done. CI fails two hours later.
hermes kanban create "Fix CI failure from t_impl: test_retry flakes on 3.11" \
    --assignee coder \
    --parent t_impl \
    --body "$(cat <<'EOF'
CI run #4812 failed after t_impl merged.
Log excerpt: FAILED tests/test_retry.py::test_backoff_jitter - TimeoutError
Acceptance: tests/test_retry.py green on 3.11 and 3.12 in CI.
Use a fresh worktree/branch; do not force-push the original branch.
EOF
)"
```

복구 작업 워커는 원래 카드의 요약과 메타데이터(변경한 파일, 결정 사항)를 이미 컨텍스트로 전달받으며, 본문에 넣은 새로운 증거도 함께 받습니다.

### 충돌하는 워커 브랜치 조정

엔지니어링 파이프라인(P1/P2 및 워크트리)에서는 두 워커의 브랜치를 병합할 때 충돌이 발생할 수 있습니다. 어느 워커에게도 스스로 판정하게 하지 마세요. 충돌한 에이전트는 동료의 컨텍스트가 없으므로 상대방의 변경을 덮어쓰거나 자신의 변경을 포기하기 쉽습니다. 대신 충돌한 두 카드를 모두 부모로 연결한 조정 카드를 만들고, **제3의 중립 프로필**에 할당하세요. 부모 링크를 통해 양쪽 완료 요약이 조정자의 컨텍스트에 전달되므로, 조정자는 두 diff와 두 의도를 모두 받습니다. 번들로 제공되는 [`merge-reconciler` skill](https://github.com/NousResearch/hermes-agent/blob/main/skills/autonomous-ai-agents/merge-reconciler/SKILL.md)은 해당 워커에게 전체 절차를 제공합니다. 모든 충돌 덩어리를 분류하고, 공정하게 해결하고, 검증한 뒤, 각 결정의 이름을 지정한 요약을 반환합니다.

### 병렬 캠페인의 충돌 집중 지점

대규모 캠페인에서는 여러 워커가 같은 파일에 조금씩 추가하고 그 파일을 작게 유지할 책임자는 아무도 없는 상황이 생깁니다. 그러면 해당 파일은 계속 충돌하는 지점이 됩니다. 해결책은 새로운 기본 기능이 아니라 주석 규칙입니다. 한 워커가 자신의 diff가 한 파일에서 계속 충돌한다는 사실을 알아차리거나, 자신이 처리하는 카드의 최근 댓글에 어떤 파일이 반복해서 등장하는 것을 발견하면 조용히 변경을 더 쌓아 올려서는 안 됩니다. 대신 다음과 같이 인식 가능한 접두사를 사용해 자신의 카드에 댓글을 남깁니다.

```
hotspot: hermes_cli/kanban_db.py — third conflicting edit to the dispatch loop this wave
```

오케스트레이터(또는 보드를 검토하는 사람)가 **같은 경로를 지칭하는 `hotspot:` 댓글을 두 개 이상** 발견하면, 해당 파일을 건드리는 작업을 더 대기열에 넣기 **전에** 전용 리팩터링/분해 카드를 만들어야 합니다. 앞으로 발생할 모든 충돌을 조정하는 것보다 충돌 자석 파일을 분리하는 편이 비용이 적습니다. 이미 발생한 충돌에는 위의 조정 카드 패턴과 `merge-reconciler` skill을 사용하세요. hotspot 표시는 조정자가 상시 작업 레인이 되는 것을 막는 사전 대응책입니다.

## 멀티 테넌트 사용

하나의 전문 플릿이 여러 기업을 지원하는 경우 각 작업에 테넌트를 태그하세요.

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

워커는 `$HERMES_TENANT`를 받고 메모리 쓰기를 접두사로 네임스페이스화합니다. 보드, 디스패처, 프로필 정의는 모두 공유되며 데이터만 범위가 지정됩니다.

## 게이트웨이 알림

게이트웨이(Telegram, Discord, Slack 등)에서 `/kanban create …`를 실행하면 생성한 작업을 원래 채팅이 자동으로 구독합니다. 게이트웨이의 백그라운드 알림기는 몇 초마다 `task_events`를 폴링해 해당 채팅으로 종료 이벤트(`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`)마다 한 번씩 메시지를 전달합니다. 완료된 작업은 워커의 `--result` 첫 줄도 전송하므로 `/kanban show`를 실행하지 않아도 결과를 확인할 수 있습니다.

구독은 CLI에서 명시적으로 관리할 수 있습니다. 스크립트나 cron 작업이 자신이 생성하지 않은 채팅에 알림을 보내려 할 때 유용합니다.

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7 \
    --chat-type group --delivery-mode notify+wake
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

작업이 `done` 또는 `archived`에 도달하면 구독이 자동으로 삭제되므로 정리할 필요가 없습니다.

### 전달 모드

`--delivery-mode`는 종료 이벤트에 알림기가 **어떻게** 반응할지 제어합니다. 모든 구독은 세 가지 모드 중 하나이며(`notify`가 기본값이자 원래 동작입니다) 다음과 같습니다.

| 모드 | 수동 메시지 | 에이전트 깨우기 | 다음과 같은 경우에 사용 |
|------|-----------------|-----------------|-------------|
| `notify` | 예 | 아니요 | 채팅에서 알림 메시지만 받고 싶을 때(기본값). |
| `notify+wake` | 예 | 예 | 대상 에이전트가 실제 턴을 수행해 보드 컨텍스트를 읽고 자신의 말투로 답하기를 원할 때. 채팅에서 생성된 자동 구독은 이 모드를 사용합니다. |
| `wake` | 아니요 | 예 | 별도의 알림 없이 에이전트가 이벤트에 따라 행동하기만 원할 때. |

"깨우기"는 대상 게이트웨이 에이전트에 합성된 인바운드 메시지를 만들어 일반적인 턴을 수행하게 합니다. 따라서 한 줄짜리 수동 알림을 받는 대신 댓글과 결과를 읽고, 추론하고, 답변합니다. 이 기능은 알림기가 실행 중인 게이트웨이 프로세스 내부에 있을 때만 작동합니다. 그렇지 않으면 `notify+wake` 구독은 여전히 수동 메시지를 전달하지만, 해당 프로세스에서 `wake` 전용 구독은 아무 일도 하지 않습니다.

`--chat-type`(`dm` | `group` | `channel` | `thread`)은 원래 채팅의 유형을 기록하므로, 깨워진 턴이 운영자의 **실제** 세션을 확인할 수 있습니다. `build_session_key`는 DM과 그룹, 채널, 스레드를 서로 다른 방식으로 키로 사용하므로, `chat_type`이 부정확하면 깨우기가 컨텍스트 없는 별도 세션으로 라우팅됩니다. `/kanban` 자동 구독 및 슬래시 명령 경로는 이를 자동으로 수집합니다. 스크립트나 cron에서 채팅을 수동으로 구독할 때만 직접 설정하면 됩니다. 기존 구독을 변경하지 않으려면 생략하세요(새 구독의 기본값은 `dm`입니다).

### 멀티 프로필 설정: 전달은 프로필이 소유합니다

프로필마다 하나의 게이트웨이를 사용하는 배포(예: `writer`, `admin` 등 각 프로필에 별도의 게이트웨이 프로세스가 있고 디스패처는 하나뿐인 경우 — [멀티 게이트웨이 가이드](https://github.com/NousResearch/hermes-agent/blob/main/docs/kanban/multi-gateway.md) 참고)에서는 디스패치와 전달의 소유자가 다릅니다.

- **디스패치는 단일 소유자입니다.** 정확히 하나의 게이트웨이만 `kanban.dispatch_in_gateway: true`를 유지하고 디스패처를 실행하며, 나머지 모든 게이트웨이는 이를 `false`로 설정합니다.
- **알림 전달은 프로필이 소유합니다.** 디스패치를 담당하지 않는 게이트웨이를 포함한 모든 게이트웨이는 알림기를 실행하고, 자신이 호스팅하는 플랫폼 어댑터를 가진 프로필이 표시된 구독만 폴링합니다. `writer` 프로필의 Telegram에서 생성된 작업의 `completed`/`blocked` 메시지는 `default` 게이트웨이가 디스패치했더라도 `writer` 게이트웨이가 전달합니다.
- **기존 구독**(프로필 표시가 도입되기 전에 생성되어 행에 `notifier_profile`이 없는 구독)은 실제 디스패처 싱글턴 잠금을 보유한 게이트웨이만 전달하므로 두 게이트웨이가 서로 경쟁하지 않습니다.

보드 DB의 이벤트별 원자적 클레임이 게이트웨이 간 중복 전달을 방지합니다. 릴레이, 자격 증명 공유 또는 추가 디스패처는 필요하지 않습니다. 각 프로필 게이트웨이가 자신의 어댑터를 통해 전달하기만 하면 됩니다.
## 실행 — 시도마다 한 행

작업은 논리적 단위이며, 실행은 한 번의 시도입니다. 디스패처가 준비된 작업을 할당하면 `task_runs`에 행을 만들고 `tasks.current_run_id`가 이를 가리킵니다. 해당 시도가 끝나면(완료, 차단됨, 실패, 시간 초과, 생성 실패, 회수됨) 실행 행이 `outcome`과 함께 닫히고 작업의 포인터가 해제됩니다. 세 번 시도된 작업에는 세 개의 `task_runs` 행이 생깁니다.

두 테이블을 사용하는 이유는 시도 전체의 이력을 보존하기 위해서입니다. 실제 운영 사후 분석(“두 번째 검토 시도는 승인까지 갔고, 세 번째 시도에서 병합됨”)을 수행할 수 있으며, 각 시도에 대한 메타데이터(변경된 파일, 실행한 테스트, 검토자가 남긴 소견)도 연결할 수 있습니다. 이는 작업 자체의 사실이 아니라 실행의 사실입니다.

실행은 구조화된 인계도 제공합니다. 작업자가 작업을 완료할 때(`kanban_complete(...)`) 다음을 전달할 수 있습니다.

- `summary` (도구 매개변수) / `--summary` (CLI) — 사람이 읽는 인계 내용이며 실행에 기록됩니다. 하위 작업은 `build_worker_context`에서 이를 확인할 수 있습니다.
- `metadata` (도구 매개변수) / `--metadata` (CLI) — 실행에 저장되는 자유 형식 JSON 딕셔너리이며, 하위 작업은 요약과 함께 직렬화된 형태로 확인할 수 있습니다.
- `result` (도구 매개변수) / `--result` (CLI) — 작업 행에 기록되는 짧은 로그 한 줄입니다(이전 버전과의 호환성을 위해 유지됨).

```text
# What a worker actually does — a tool call, from inside the agent loop:
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

작업자가 처리하지 못한 작업을 사람이 닫아야 할 때(예: 작업이 중단되었거나 대시보드에서 수동으로 완료 처리한 경우)에도 CLI에서 같은 인계를 수행할 수 있습니다.

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# Review the attempt history on a retried task:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

실행은 대시보드(드로어의 Run History 섹션에 시도별로 색상이 지정된 한 행씩 표시)와 REST API(`GET /api/plugins/kanban/tasks/:id`가 `runs[]` 배열을 반환)에 노출됩니다. `{status: "done", summary, metadata}`와 함께 `PATCH /api/plugins/kanban/tasks/:id`를 호출하면 두 값이 커널로 전달되므로 대시보드의 “완료로 표시” 버튼은 CLI와 동일하게 동작합니다. `task_events` 행에는 속한 실행을 나타내는 `run_id`가 포함되어 있어 UI가 시도별로 그룹화할 수 있으며, `completed` 이벤트에는 게이트웨이 알림기가 두 번째 SQL 왕복 없이 구조화된 인계를 렌더링할 수 있도록 페이로드에 첫 줄 요약(최대 400자)이 들어갑니다.

**일괄 종료 주의사항.** `hermes kanban complete a b c --summary X`는 거부됩니다. 구조화된 인계는 실행별로 이루어지므로 동일한 요약을 N개 작업에 복사해 넣는 것은 거의 항상 잘못된 결과를 만듭니다. 일반적인 “관리 작업을 한꺼번에 끝냈다”는 경우에는 `--summary` / `--metadata` 없이 일괄 종료할 수 있습니다.

**상태 변경으로 회수된 실행.** 대시보드에서 실행 중인 작업을 `running`에서 다른 상태로 드래그하거나(`ready`로 되돌리거나 바로 `todo`로 이동), 아직 실행 중인 작업을 보관 처리하면 진행 중인 실행은 고아 상태로 남지 않고 `outcome='reclaimed'`로 닫힙니다. `tasks.current_run_id`가 `NULL`이면 `task_runs` 행은 항상 종료 상태이며 그 반대도 마찬가지입니다. 이 불변 조건은 CLI, 대시보드, 디스패처, 알림기 전반에서 유지됩니다.

**한 번도 할당되지 않은 작업의 합성 실행.** 한 번도 할당되지 않은 작업을 완료하거나 차단하면(예: 사람이 대시보드에서 요약과 함께 `ready` 작업을 닫거나, CLI 사용자가 `hermes kanban complete <ready-task> --summary X`를 실행하는 경우) 인계가 사라질 수 있습니다. 이를 방지하기 위해 커널은 요약 / 메타데이터 / 사유를 담은 지속 시간 0의 실행 행(`started_at == ended_at`)을 삽입하여 시도 이력을 완전하게 유지합니다. `completed` / `blocked` 이벤트의 `run_id`는 이 행을 가리킵니다.

**실시간 드로어 새로고침.** 대시보드의 WebSocket 이벤트 스트림이 현재 사용자가 보고 있는 작업에 대한 새 이벤트를 보고하면, 드로어가 스스로 다시 로드됩니다(`useEffect` 의존성 목록에 작업별 이벤트 카운터를 전달). 이제 실행의 새 행이나 변경된 결과를 보기 위해 닫았다가 다시 열 필요가 없습니다.

### 향후 호환성

`tasks`의 두 nullable 열은 v2 워크플로 라우팅을 위해 예약되어 있습니다. `workflow_template_id`는 이 작업이 속한 템플릿을, `current_step_key`는 해당 템플릿에서 활성화된 단계를 나타냅니다. v1 커널은 라우팅에 이 열을 사용하지 않지만 클라이언트가 값을 쓸 수 있도록 허용하므로, v2 릴리스에서는 추가 스키마 마이그레이션 없이 라우팅 기능을 추가할 수 있습니다.

## 이벤트 참조

모든 상태 전환은 `task_events`에 행을 추가합니다. 각 행에는 선택적인 `run_id`가 있어 UI가 이벤트를 시도별로 그룹화할 수 있습니다. 종류는 필터링하기 쉽도록 세 그룹으로 나뉩니다(`hermes kanban watch --kinds completed,gave_up,timed_out`).

**수명 주기**(논리적 단위인 작업에 무엇이 변경되었는지):

| 종류 | 페이로드 | 시점 |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | 작업이 삽입됨. `run_id`는 `NULL`. |
| `promoted` | — | 모든 상위 작업이 `done`이 되어 `todo → ready`가 됨. `run_id`는 `NULL`. |
| `claimed` | `{lock, expires, run_id}` | 디스패처가 생성할 `ready` 작업을 원자적으로 할당함. |
| `completed` | `{result_len, summary?}` | 작업자가 `--result` / `--summary`를 작성하고 작업이 `done`이 됨. `summary`는 첫 줄 인계 내용(최대 400자)이며 전체 내용은 실행 행에 저장됨. 인계 필드와 함께 `complete_task`가 한 번도 할당되지 않은 작업에서 호출되면 지속 시간 0의 실행이 생성되어 `run_id`가 가리킬 대상이 유지됨. |
| `blocked` | `{reason, kind, recurrences}` | 작업자 또는 사람이 작업을 `blocked`로 변경함. `kind`는 유형화된 차단 사유(`needs_input`, `capability`, `transient`, 일반 차단이면 `null`)이고, `recurrences`는 차단 해제 루프 카운터임. `--reason`과 함께 한 번도 할당되지 않은 작업에서 호출되면 지속 시간 0의 실행이 생성됨. |
| `dependency_wait` | `{reason, kind}` | 작업자가 `kind=dependency`로 차단함. 작업은 다른 작업만 기다리는 상태이므로 `blocked` 대신 `todo`(상위 작업에 의해 게이트되고 자동 승격됨)로 라우팅됨. 사람이 개입할 필요가 없음. |
| `block_loop_detected` | `{reason, kind, recurrences, limit}` | 작업이 같은 사유로 `BLOCK_RECURRENCE_LIMIT`번(기본값 2) 차단 해제 및 재차단됨. cron이 계속 차단을 해제하는 `blocked`에 다시 머무르는 대신 사람의 결정을 위해 `triage`로 라우팅하여 차단 해제↔재차단 루프를 끊음. |
| `unblocked` | — | 수동으로 또는 `/unblock`을 통해 `blocked → ready`가 됨(상위 작업이 아직 열려 있으면 `todo`). 디스패처의 `consecutive_failures`는 초기화하지만, 루프 차단기가 기억을 유지할 수 있도록 `block_recurrences`는 의도적으로 보존함. `run_id`는 `NULL`. |
| `archived` | — | 기본 보드에서 숨겨짐. 작업이 아직 실행 중이었다면, 부수 효과로 회수된 실행의 `run_id`를 포함함. |

**편집**(전환이 아닌 사람이 주도한 변경):

| 종류 | 페이로드 | 시점 |
|---|---|---|
| `assigned` | `{assignee}` | 담당자가 변경됨(담당자 해제 포함). |
| `edited` | `{fields}` | 제목 또는 본문이 업데이트됨. |
| `reprioritized` | `{priority}` | 우선순위가 변경됨. |
| `status` | `{status}` | 대시보드의 드래그 앤 드롭이 상태를 직접 기록함(예: `todo → ready`). `running`에서 드래그해 나갈 때 회수된 실행의 `run_id`를 포함하며, 그 외에는 `run_id`가 `NULL`. |

**작업자 텔레메트리**(논리적 작업이 아닌 실행 프로세스에 관한 정보):

| 종류 | 페이로드 | 시점 |
|---|---|---|
| `spawned` | `{pid}` | 디스패처가 작업자 프로세스를 성공적으로 시작함. |
| `heartbeat` | `{note?}` | 작업자가 장시간 작업 중 생존을 알리기 위해 `hermes kanban heartbeat $TASK`를 호출함. |
| `reclaimed` | `{stale_lock}` | 완료 없이 할당 TTL이 만료됨. 작업이 다시 `ready`로 돌아감. |
| `crashed` | `{pid, claimer}` | TTL이 아직 만료되지 않았지만 작업자 PID가 더 이상 살아 있지 않음. |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | `max_runtime_seconds`를 초과함. 디스패처가 SIGTERM을 보낸 뒤 5초의 유예 시간이 지나면 SIGKILL을 보내고 작업을 다시 큐에 넣음. |
| `stale` | `{elapsed_seconds, last_heartbeat_at, heartbeat_age_seconds, timeout_seconds, pid, terminated}` | 작업이 `kanban.dispatch_stale_timeout_seconds`(기본값 4시간)보다 오래 실행되었고 지난 1시간 동안 `kanban_heartbeat`가 오지 않음. 디스패처가 호스트 로컬 작업자(있는 경우)에 SIGTERM을 보내고 작업을 `ready`로 초기화하여 다시 디스패치함. 실패 카운터는 증가시키지 않음(오래된 상태는 작업자 오류가 아니라 디스패처 측의 부재 감지임). 장시간 작업을 실행하는 작업자는 이를 피하기 위해 최소 한 시간에 한 번 `kanban_heartbeat`를 호출해야 함. |
| `reconciled` | `{reason, claim_lock, claim_expires, worker_pid}` | 고아 카드 조정: 카드가 깨진 할당 장부(`claim_lock` 또는 `claim_expires`가 `NULL`인 상태 — 할당 중 충돌, 수동 SQL, DB 복구)로 `running`에 있었고 살아 있는 작업자도 없어 TTL/충돌/오래된 상태 경로로 복구할 수 없음. 디스패처가 설명이 포함된 주석을 남기고 카드를 `ready`로 다시 큐에 넣음. `config.yaml`의 `kanban.reconcile_orphans`(기본값 `true`)로 제어됨. |
| `respawn_guarded` | `{reason}` | 디스패처가 이번 틱에 이 `ready` 작업을 다시 생성하지 않음. 사유: `blocker_auth`(마지막 실패가 할당량/인증/429 오류였으므로 속도 제한 창이 초기화될 때까지 대기), `recent_success`(지난 한 시간 안에 완료된 실행이 있었으므로 다시 실행하기 전에 검토를 기다림), `active_pr`(최근 주석에 GitHub PR URL이 있으므로 이전 작업자가 이미 PR을 열었음). 작업은 `ready`에 남고 다음 틱에 다시 생성 기회를 얻음. 조건이 계속되면 일반 `consecutive_failures` 회로 차단기가 `failure_limit`번 실패 후 `gave_up`을 통해 자동 차단함. |
| `spawn_failed` | `{error, failures}` | 생성 시도 하나가 실패함(PATH 누락, 작업 공간을 마운트할 수 없음 등). 카운터가 증가하고 작업은 재시도를 위해 `ready`로 돌아감. |
| `protocol_violation` | `{pid, claimer, exit_code, protocol_violation}` | 작업자가 성공적으로 종료했지만 작업은 여전히 `running`인 상태이며, 보통 `kanban_complete` 또는 `kanban_block`을 호출하지 않고 응답했기 때문임. 모든 위반에서 발생함(페이로드의 `protocol_violation: true` 표시는 실행 메타데이터에 복사되고 위반 전용 재시도 예산에 반영됨). 예산 이내에서는 작업별 `max_retries`가 재정의할 수 있는 `_PROTOCOL_VIOLATION_FAILURE_LIMIT`(기본값 3)회의 *연속* 위반까지 작업이 다시 `ready`로 돌아가 다음 시도를 수행함. 연속 위반이 한도에 도달하면 디스패처가 `gave_up`도 발생시키고 자동 차단함. |
| `gave_up` | `{failures, effective_limit, limit_source, error}` | N회의 연속 비성공 시도가 발생하여 회로 차단기가 작동함. 마지막 오류와 함께 작업을 자동으로 차단함. 유효 한도는 작업의 `max_retries`, 디스패처의 `failure_limit` / `kanban.failure_limit`, 내장 기본값 순으로 결정됨. |

`hermes kanban tail <id>`는 한 작업에 대한 이벤트를 표시합니다. `hermes kanban watch`는 보드 전체의 이벤트를 스트리밍합니다.

## 범위 외

Kanban은 의도적으로 단일 호스트에서만 동작합니다. `~/.hermes/kanban.db`는 로컬 SQLite 파일이며 디스패처는 같은 시스템에서 작업자를 생성합니다. 두 호스트 간 공유 보드는 지원되지 않습니다. “호스트 A의 작업자 X와 호스트 B의 작업자 Y”를 조정할 기본 요소가 없고, 충돌 감지 경로가 호스트 로컬 PID를 전제로 하기 때문입니다. 여러 호스트가 필요하면 호스트마다 독립적인 보드를 실행하고 `delegate_task` / 메시지 큐로 연결하십시오.
## 설계 사양

전체 설계(아키텍처, 동시성 정확성, 다른 시스템과의 비교, 구현 계획, 위험 요소, 미해결 질문)는 `docs/hermes-kanban-v1-spec.pdf`에 있습니다. 동작 변경 PR을 제출하기 전에 해당 문서를 읽어 보세요.

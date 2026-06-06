---
sidebar_position: 12
title: "칸반(다중 에이전트 보드) (Kanban (Multi-Agent Board))"
description: "여러 Hermes 프로필을 조정하기 위한 내구성 있는 SQLite 기반 작업 보드"
---

# 칸반 — 다중 에이전트 프로필 협업 (Kanban — Multi-Agent Profile Collaboration)

> **연습이 필요하십니까?** [칸반 튜토리얼](./kanban-tutorial)을 읽어보세요 — 대시보드 스크린샷과 함께 4가지 사용자 스토리(솔로 개발자, 플릿 파밍(fleet farming), 재시도를 포함한 역할 파이프라인, 서킷 브레이커)가 포함되어 있습니다. 이 페이지는 참조 문서이며, 튜토리얼은 이야기 형식입니다.

Hermes 칸반은 모든 Hermes 프로필에서 공유되는 내구성 있는 작업 보드(task board)로, 깨지기 쉬운 프로세스 내 서브에이전트 스웜(swarm) 없이 여러 명명된 에이전트가 작업에서 협력할 수 있게 해줍니다. 모든 작업은 `~/.hermes/kanban.db`의 행(row)이며, 모든 핸드오프는 누구나 읽고 쓸 수 있는 행이고, 모든 작업자(worker)는 고유한 ID를 가진 전체 OS 프로세스입니다.

### 두 가지 표면: 모델은 도구를 통해 대화하고, 사용자는 CLI를 통해 대화합니다

보드에는 두 개의 정문이 있으며, 둘 다 동일한 `~/.hermes/kanban.db`에 의해 지원됩니다:

- **에이전트는 전용 `kanban_*` 도구 세트를 통해 보드를 작동시킵니다** — `kanban_show`, `kanban_list`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`, `kanban_unblock`. 디스패처(dispatcher)는 스키마에 이러한 도구가 이미 포함된 각 작업자를 생성합니다. 오케스트레이터(orchestrator) 프로필은 명시적으로 `kanban` 도구 세트를 활성화할 수도 있습니다. 모델은 작업을 읽고 라우팅할 때 `hermes kanban`으로 셸 아웃(shelling out)하는 것이 아니라 도구를 직접 호출합니다. 아래의 [작업자가 보드와 상호 작용하는 방법](#how-workers-interact-with-the-board)을 참조하세요.
- **사용자(그리고 스크립트, 크론)는 CLI에서 `hermes kanban …`을 통해 보드를 작동시킵니다** — 또는 슬래시 명령어로 `/kanban …`을 사용하거나 대시보드를 이용합니다. 이는 사람과 자동화를 위한 것으로, 백그라운드에 도구 호출 모델이 없는 곳입니다.

두 표면 모두 동일한 `kanban_db` 계층을 통해 라우팅되므로, 읽기는 일관된 보기를 확인하고 쓰기는 변동될 수 없습니다. 이 페이지의 나머지 부분에서는 복사-붙여넣기하기 쉽기 때문에 CLI 예제를 보여주지만, 모든 CLI 동사에는 모델이 사용하는 것과 동일한 도구 호출 버전이 있습니다.

이것은 `delegate_task`가 처리할 수 없는 워크로드를 다루는 구조입니다:

- **연구 트리아지(Research triage)** — 병렬 연구자 + 분석가 + 작가, 휴먼 인 더 루프(human-in-the-loop).
- **예약된 운영 작업(Scheduled ops)** — 몇 주에 걸쳐 저널을 작성하는 반복적인 일일 브리핑.
- **디지털 트윈(Digital twins)** — 시간이 지남에 따라 메모리를 축적하는 영구적으로 명명된 비서(`inbox-triage`, `ops-review`).
- **엔지니어링 파이프라인(Engineering pipelines)** — 분해 → 병렬 워크트리에서 구현 → 검토 → 반복 → PR.
- **플릿 작업(Fleet work)** — N개의 대상을 관리하는 한 명의 전문가(50개의 소셜 계정, 12개의 모니터링된 서비스).

전체 설계 근거, Cline Kanban / Paperclip / NanoClaw / Google Gemini Enterprise와의 비교 분석, 8가지 일반적인 협업 패턴에 대해서는 레포지토리의 `docs/hermes-kanban-v1-spec.pdf`를 참조하세요.

## 칸반 vs. `delegate_task`

비슷해 보이지만, 동일한 기본 요소가 아닙니다.

| | `delegate_task` | 칸반 (Kanban) |
|---|---|---|
| 구조 (Shape) | RPC 호출 (포크 → 조인) | 내구성 있는 메시지 큐 + 상태 머신 |
| 부모 (Parent) | 자식이 반환될 때까지 차단됨 | `create` 후 실행하고 잊음 (Fire-and-forget) |
| 자식 ID (Child identity) | 익명의 서브에이전트 | 지속성 있는 메모리를 가진 명명된 프로필 |
| 재개 가능성 (Resumability) | 없음 — 실패 = 실패 | 차단 → 차단 해제 → 다시 실행; 크래시 → 회수 |
| 휴먼 인 더 루프 (Human in the loop) | 지원되지 않음 | 언제든지 코멘트 / 차단 해제 |
| 작업당 에이전트 (Agents per task) | 한 번의 호출 = 하나의 서브에이전트 | 작업 수명 동안 N명의 에이전트 (재시도, 검토, 후속 조치) |
| 감사 추적 (Audit trail) | 컨텍스트 압축 시 손실됨 | SQLite에 영구적으로 남는 행 |
| 조정 (Coordination) | 계층적 (호출자 → 피호출자) | 피어(Peer) — 모든 프로필이 모든 작업을 읽고/쓸 수 있음 |

**한 문장 구분:** `delegate_task`는 함수 호출입니다. 반면 칸반은 모든 핸드오프가 어떤 프로필(또는 사람)이라도 보고 편집할 수 있는 작업 큐입니다.

**`delegate_task`는 다음의 경우에 사용하세요:** 부모 에이전트가 계속 진행하기 전에 짧은 추론 답변이 필요하고, 사람이 개입하지 않으며, 결과가 부모의 컨텍스트로 되돌아가는 경우.

**칸반은 다음의 경우에 사용하세요:** 작업이 에이전트 경계를 넘어가거나, 재시작 시에도 유지되어야 하거나, 사람의 입력이 필요할 수 있거나, 다른 역할이 이어받을 수 있거나, 작업 후 발견할 수 있어야 하는 경우.

두 개는 공존할 수 있습니다: 칸반 작업자는 실행 중에 내부적으로 `delegate_task`를 호출할 수 있습니다.

## 핵심 개념

- **보드 (Board)** — 자체 SQLite DB, 작업 공간(workspaces) 디렉토리 및 디스패처 루프가 있는 독립적인 작업 큐입니다. 한 번의 설치에 여러 보드(예: 프로젝트, 리포지토리 또는 도메인당 하나씩)를 가질 수 있습니다. 아래의 [보드 (다중 프로젝트)](#boards-multi-project)를 참조하세요. 단일 프로젝트 사용자는 `default` 보드에 머무르며 이 문서 섹션 밖에서는 "보드"라는 단어를 볼 필요가 없습니다.
- **작업 (Task)** — 제목, 선택적 본문, 하나의 담당자(프로필 이름), 상태(`triage | todo | ready | running | blocked | done | archived`), 선택적 테넌트 네임스페이스, 선택적 멱등성 키(재시도되는 자동화를 위한 중복 제거)가 있는 행(row)입니다.
- **링크 (Link)** — 부모 → 자식 의존성을 기록하는 `task_links` 행입니다. 디스패처는 모든 부모가 `done` 상태일 때 `todo → ready`로 승격시킵니다.
- **코멘트 (Comment)** — 에이전트 간의 프로토콜입니다. 에이전트와 사람은 코멘트를 추가합니다. 작업자가 (재)생성될 때 컨텍스트의 일부로 전체 코멘트 스레드를 읽습니다.
- **작업 공간 (Workspace)** — 작업자가 작업하는 디렉토리입니다. 세 종류가 있습니다:
  - `scratch` (기본값) — `~/.hermes/kanban/workspaces/<id>/` 아래의 새로운 임시 디렉토리(비기본 보드의 경우 `~/.hermes/kanban/boards/<slug>/workspaces/<id>/`). **작업 완료 시 삭제됨** — scratch는 본질적으로 일시적이며, 작업자(또는 `hermes kanban complete <id>`)가 작업을 완료로 표시하는 순간 디렉토리가 지워집니다. 작업자의 출력물을 유지하려면 대신 `worktree:` 또는 `dir:<path>`를 사용하세요. 처음으로 시스템에 scratch 작업 공간이 생성될 때 디스패처는 경고를 로깅하고 작업에 `tip_scratch_workspace` 이벤트를 발생시킵니다(`hermes kanban show <id>`를 통해 볼 수 있음).
  - `dir:<path>` — 기존의 공유 디렉토리 (Obsidian 볼트, 메일 ops 디렉토리, 계정별 폴더). **반드시 절대 경로여야 합니다.** `dir:../tenants/foo/`와 같은 상대 경로는 디스패처가 있는 CWD에 대해 해석되어 모호하며 confused-deputy 우회 공격 벡터가 될 수 있으므로 디스패치 시 거부됩니다. 이 경로는 안전하다고 간주됩니다 — 로컬 머신, 사용자 파일 시스템이며, 작업자는 사용자의 uid로 실행됩니다. 이는 신뢰할 수 있는 로컬 사용자 위협 모델(trusted-local-user threat model)이며, 칸반은 본질적으로 단일 호스트로 설계되었습니다. **완료 후에도 보존됨.**
  - `worktree` — 코딩 작업을 위해 `.worktrees/<id>/` 아래의 git 워크트리입니다. 대상 경로를 고정하려면 `worktree:<path>`를 사용하세요. 제공된 경우 `--branch`를 사용하여 작업자 측 `git worktree add`가 이를 생성합니다. **완료 후에도 보존됨.**
- **디스패처 (Dispatcher)** — 매 N초(기본값 60초)마다 오래된 클레임 회수, 크래시된 작업자 회수(PID는 없지만 TTL은 아직 만료되지 않음), 준비된 작업 승격, 원자적 클레임 처리, 할당된 프로필 생성을 수행하는 장기 실행 루프입니다. 기본적으로 **게이트웨이 내부에서 실행**됩니다 (`kanban.dispatch_in_gateway: true`). 하나의 디스패처가 틱(tick)마다 모든 보드를 정리합니다. 작업자는 `HERMES_KANBAN_BOARD`가 고정된 상태로 생성되므로 다른 보드를 볼 수 없습니다. 동일한 작업에 대해 `kanban.failure_limit` (기본값: 2) 횟수 연속 생성에 실패하면 디스패처는 마지막 오류를 이유로 자동 차단합니다 — 프로필이 존재하지 않거나 작업 공간을 마운트할 수 없는 등의 작업에서 반복적으로 스래싱(thrashing)이 발생하는 것을 방지합니다.
- **테넌트 (Tenant)** — 보드 *내의* 선택적 문자열 네임스페이스입니다. 하나의 전문 플릿이 여러 비즈니스(`--tenant business-a`)를 서비스할 수 있으며, 작업 공간 경로와 메모리 키 접두사별로 데이터를 격리합니다. 테넌트는 소프트 필터입니다. 보드는 하드 격리 경계입니다.

## 보드 (다중 프로젝트)

보드를 사용하면 프로젝트, 저장소, 도메인 등 관련 없는 작업 스트림을 분리된 대기열로 나눌 수 있습니다. 새로 설치된 시스템에는 정확히 하나의 보드가 `default`라는 이름으로 존재합니다 (이전 버전 호환성을 위해 DB는 `~/.hermes/kanban.db`에 위치). 하나의 작업 스트림만 원하는 사용자는 보드에 대해 알 필요가 없습니다. 이 기능은 선택 사항입니다.

보드별 격리는 절대적입니다:

- 보드당 분리된 SQLite DB (`~/.hermes/kanban/boards/<slug>/kanban.db`).
- 분리된 `workspaces/` 및 `logs/` 디렉토리.
- 작업을 위해 생성된 작업자는 **오직** 자신의 보드 작업만 볼 수 있습니다 — 디스패처가 자식 환경 변수에 `HERMES_KANBAN_BOARD`를 설정하며, 작업자가 접근할 수 있는 모든 `kanban_*` 도구는 이를 읽습니다.
- 보드 간 작업 연결은 허용되지 않습니다 (스키마를 단순하게 유지하기 위해; 만약 정말로 여러 프로젝트를 아우르는 참조가 필요하다면 자유 텍스트 멘션을 사용하고 ID로 수동 조회하세요).

### CLI에서 보드 관리

```bash
# 디스크에 무엇이 있는지 확인. 새로 설치하면 "default"만 표시됨.
hermes kanban boards list

# 새 보드 생성.
hermes kanban boards create atm10-server \
    --name "ATM10 Server" \
    --description "Minecraft modded server ops" \
    --icon 🎮 \
    --switch                   # 선택 사항: 활성 보드로 설정

# 전환하지 않고 특정 보드에서 작업.
hermes kanban --board atm10-server list
hermes kanban --board atm10-server create "Restart ATM server" --assignee ops

# 후속 호출에 대해 "현재" 보드 변경.
hermes kanban boards switch atm10-server
hermes kanban boards show             # 현재 활성 상태인 보드 확인?

# 표시 이름 이름 바꾸기 (슬러그는 불변임 — 디렉토리 이름임).
hermes kanban boards rename atm10-server "ATM10 (Prod)"

# 보관 (기본값) — 보드의 디렉토리를 boards/_archived/<slug>-<ts>/로 이동.
# 디렉토리를 다시 옮겨 복구할 수 있음.
hermes kanban boards rm atm10-server

# 하드 삭제 — 보드 디렉토리에서 `rm -rf`. 복구 불가.
hermes kanban boards rm atm10-server --delete
```

보드 확인 우선순위 (가장 높은 우선순위부터):

1. CLI 호출의 명시적인 `--board <slug>`.
2. `HERMES_KANBAN_BOARD` 환경 변수 (작업자 생성 시 디스패처에 의해 설정되므로 작업자가 다른 보드를 볼 수 없음).
3. `~/.hermes/kanban/current` — `hermes kanban boards switch`에 의해 지속되는 슬러그.
4. `default`.

슬러그 유효성 검사: 소문자 알파벳 + 하이픈 + 언더스코어, 1-64자, 알파벳이나 숫자로 시작해야 합니다. 대문자 입력은 자동으로 소문자로 변환됩니다. 슬래시, 공백, 마침표, `..` 등 기타 문자열은 경로 조작(path-traversal) 기법으로 보드 이름을 지정할 수 없도록 CLI 계층에서 거부됩니다.

### 대시보드에서 보드 관리

`hermes dashboard` → 칸반 탭은 보드가 두 개 이상 생성되거나, 어떤 보드에든 작업이 생기는 즉시 상단에 보드 스위처를 표시합니다. 단일 보드 사용자는 작은 `+ New board` 버튼만 보며, 스위처는 필요해질 때까지 숨겨져 있습니다.

- **보드 드롭다운** — 활성 보드를 선택합니다. 열려 있는 터미널 아래의 CLI `current` 포인터가 변경되는 것을 피하기 위해, 브라우저의 `localStorage`에 선택 사항이 저장되어 페이지를 새로고침해도 유지됩니다.
- **+ New board** — 슬러그, 표시 이름, 설명, 아이콘을 묻는 모달을 엽니다. 새 보드로 자동 전환하는 옵션이 있습니다.
- **Archive** — `default`가 아닌 보드에만 표시됩니다. 확인 후 보드 디렉토리를 `boards/_archived/`로 이동합니다.

모든 대시보드 API 엔드포인트는 보드 범위를 지정하기 위해 `?board=<slug>`를 허용합니다. 이벤트 WebSocket은 연결 시 보드에 고정되며, UI에서 전환하면 새 보드에 대한 새로운 WS를 엽니다.

## 파일 첨부

작업에는 파일 첨부(PDF, 이미지, 원본 문서)를 추가할 수 있으므로, 경로를 본문에 붙여넣고 작업자가 찾길 바랄 필요 없이 작업자가 원본 자료를 필요로 하는 형태로 가지고 있습니다.

- **업로드** — 대시보드 드로어에서 작업을 열고 **Attachments** 섹션의 *Upload file* 버튼을 사용합니다(여러 파일을 한 번에 올릴 수 있음). 각 업로드는 25MB로 제한됩니다.
- **스토리지** — 파일은 기본 보드의 경우 `<hermes-home>/kanban/attachments/<task_id>/` 아래에 위치하거나 명명된 보드의 경우 `<hermes-home>/kanban/boards/<slug>/attachments/<task_id>/` 아래에 위치합니다. 사용자 지정 위치를 고정하려면 `HERMES_KANBAN_ATTACHMENTS_ROOT`를 설정하세요.
- **작업자가 보는 것** — 디스패처가 작업자에게 작업을 넘겨줄 때, 작업자의 컨텍스트에는 각 파일의 이름과 **절대 경로**를 나열하는 **Attachments** 섹션이 포함됩니다. 작업자는 전체 파일/터미널 도구 권한을 가지므로 첨부 파일을 직접 읽습니다(`read_file` 또는 `pdftotext` 같은 셸 도구).
- **다운로드 / 제거** — 드로어에는 각 첨부 파일에 대한 다운로드 링크와 제거 (×) 제어가 나열되어 있습니다. 첨부 파일을 제거하면 메타데이터 행과 디스크 상의 파일이 모두 삭제됩니다.

:::note 원격 터미널 백엔드
첨부 파일 경로는 칸반 작업자의 기본값인 **로컬** 터미널 백엔드에서 직접 해결됩니다. 원격 백엔드(Docker, Modal)에서 작업자를 실행하는 경우, 작업자 컨텍스트의 절대 경로에 접근할 수 있도록 보드의 `attachments/` 디렉토리를 샌드박스에 마운트하십시오.
:::

## 빠른 시작

아래 명령은 보드를 설정하고 작업을 생성하는 **사용자**(사람)의 작업입니다. 작업이 할당되면 디스패처가 할당된 프로필을 작업자로 생성하고, 그 시점부터 **모델은 CLI 명령이 아닌 `kanban_*` 도구 호출을 통해 작업을 주도합니다** — [작업자가 보드와 상호 작용하는 방법](#how-workers-interact-with-the-board)을 참조하세요.

```bash
# 1. 보드 생성 (사용자)
hermes kanban init

# 2. 게이트웨이 시작 (임베디드 디스패처 호스팅)
hermes gateway start

# 3. 작업 생성 (사용자 — 또는 kanban_create를 통한 오케스트레이터 에이전트)
hermes kanban create "research AI funding landscape" --assignee researcher

# 4. 실시간으로 활동 보기 (사용자)
hermes kanban watch

# 5. 보드 보기 (사용자)
hermes kanban list
hermes kanban stats
```

디스패처가 `t_abcd`를 픽업하고 `researcher` 프로필을 생성할 때, 해당 작업자의 모델이 가장 먼저 수행하는 작업은 `kanban_show()`를 호출하여 작업을 읽는 것입니다. 이 작업자는 `hermes kanban show t_abcd`를 실행하지 않습니다.

### 게이트웨이 내장 디스패처 (기본값)

디스패처는 게이트웨이 프로세스 내부에서 실행됩니다. 설치할 것이 없고, 관리할 별도의 서비스도 없습니다 — 게이트웨이가 작동 중이라면 다음 틱(기본 60초)에 준비된 작업(ready tasks)이 선택됩니다.

```yaml
# config.yaml
kanban:
  dispatch_in_gateway: true        # 기본값
  dispatch_interval_seconds: 60    # 기본값
```

디버깅을 위해 런타임에 구성 플래그를 오버라이드하려면 `HERMES_KANBAN_DISPATCH_IN_GATEWAY=0`을 사용하세요. 표준 게이트웨이 감독이 적용됩니다: `hermes gateway start`를 직접 실행하거나 게이트웨이를 systemd 사용자 유닛으로 연결하세요(게이트웨이 문서 참조). 게이트웨이가 실행 중이 아니면 실행될 때까지 `ready` 작업은 계속 머물러 있게 됩니다 — `hermes kanban create`는 생성 시 이에 대해 경고합니다.

별도의 프로세스로 `hermes kanban daemon`을 실행하는 것은 **더 이상 권장되지 않습니다(deprecated)**; 게이트웨이를 사용하세요. 게이트웨이를 도저히 실행할 수 없는 경우(장기 실행 서비스가 금지된 헤드리스 호스트 정책 등) `--force` 이스케이프 해치가 한 번의 릴리스 주기 동안 레거시 독립형 데몬을 유지하게 해주지만, 게이트웨이-임베디드 디스패처와 독립형 데몬을 동일한 `kanban.db`에 대해 동시에 실행하는 것은 클레임 경쟁(claim races)을 초래하며 지원되지 않습니다.

### 멱등성 있는 생성 (자동화 / 웹훅 용도)

```bash
# 첫 번째 호출은 작업을 생성합니다. 동일한 키를 사용한 후속 호출은
# 중복을 생성하는 대신 기존 작업 ID를 반환합니다.
hermes kanban create "nightly ops review" \
    --assignee ops \
    --idempotency-key "nightly-ops-$(date -u +%Y-%m-%d)" \
    --json
```

### 대량 작업을 위한 CLI 동사

수명 주기와 관련된 모든 동사는 한 번의 명령으로 배치를 정리할 수 있도록 여러 ID를 허용합니다:

```bash
hermes kanban complete t_abc t_def t_hij --result "batch wrap"
hermes kanban archive  t_abc t_def t_hij
hermes kanban unblock  t_abc t_def
hermes kanban block    t_abc "need input" --ids t_def t_hij
```

## 작업자가 보드와 상호 작용하는 방법

**작업자는 `hermes kanban`을 셸 호출하지 않습니다.** 디스패처가 작업자를 생성할 때 자식의 환경에 `HERMES_KANBAN_TASK=t_abcd`를 설정하고, 해당 환경 변수는 모델의 스키마에서 전용 **kanban 도구 세트**를 켭니다. 도구 세트 구성에서 `kanban`을 활성화하는 오케스트레이터 프로필에서도 동일한 도구 세트를 사용할 수 있습니다. 이러한 도구는 CLI와 마찬가지로 Python `kanban_db` 계층을 통해 보드를 직접 읽고 변형합니다. 실행 중인 작업자는 이것을 다른 도구처럼 호출합니다. 이 작업자는 `hermes kanban` CLI를 보거나 필요로 하지 않습니다.

| 도구 | 목적 | 필수 매개변수 |
|---|---|---|
| `kanban_show` | 현재 작업(제목, 본문, 이전 시도, 부모 핸드오프, 코멘트, 미리 포맷된 전체 `worker_context`) 읽기. 기본값은 환경 변수의 작업 ID입니다. | — |
| `kanban_list` | `assignee`, `status`, `tenant`, 아카이브된 표시 여부 및 제한에 대한 필터로 작업 요약 목록 표시. 오케스트레이터가 보드 작업을 발견하기 위한 것입니다. | — |
| `kanban_complete` | `summary` + `metadata` 구조화된 핸드오프로 마무리. | `summary` / `result` 중 최소 하나 |
| `kanban_block` | `reason`을 포함하여 사람의 입력을 위해 에스컬레이션(escalate). | `reason` |
| `kanban_heartbeat` | 장시간 운영 중 활성 상태 유지 신호. 순수 사이드 이펙트. | — |
| `kanban_comment` | 작업 스레드에 영구적인 메모 추가. | `task_id`, `body` |
| `kanban_create` | (오케스트레이터) `assignee`, 선택적 `parents`, `skills` 등을 사용하여 자식 작업으로 팬아웃(fan out) | `title`, `assignee` |
| `kanban_link` | (오케스트레이터) 사후에 `parent_id → child_id` 의존성 엣지(edge) 추가. | `parent_id`, `child_id` |
| `kanban_unblock` | (오케스트레이터) 차단된(blocked) 작업을 다시 대기(ready) 상태로 이동. | `task_id` |

일반적인 작업자의 차례는 다음과 같습니다:

```
# 모델의 도구 호출, 순서대로:
kanban_show()                                     # 인수 없음 — HERMES_KANBAN_TASK 사용
# (모델은 반환된 worker_context를 읽고, 터미널/파일 도구를 통해 작업을 수행합니다)
kanban_heartbeat(note="halfway through — 4 of 8 files transformed")
# (더 많은 작업)
kanban_complete(
    summary="migrated limiter.py to token-bucket; added 14 tests, all pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
)
```

**오케스트레이터(orchestrator)** 작업자는 팬아웃(팬아웃)을 수행합니다:

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
    parents=["t_r1", "t_r2"],                     # 두 항목이 완료될 때 ready로 승격됨
    body="one-pager, 300 words, neutral tone",
)
kanban_complete(summary="decomposed into 2 research tasks + 1 writer; linked dependencies")
```

"(오케스트레이터)" 도구 — `kanban_list`, `kanban_create`, `kanban_link`, `kanban_unblock`, 그리고 외부 작업의 `kanban_comment` — 는 동일한 도구 세트를 통해 사용할 수 있습니다. 일반적인 규약(`kanban-orchestrator` 기술에 의해 시행됨)은 일반 작업자(worker profiles)는 관련 없는 작업을 라우팅하거나 팬아웃하지 않으며, 오케스트레이터 프로필은 구현 작업을 실행하지 않는다는 것입니다. 디스패처가 생성한 작업자는 파괴적인 생명주기 작업에 대해 여전히 작업 범위(task-scoped)에 제한을 받으며 관련 없는 작업을 변경할 수 없습니다.

### `hermes kanban` 셸 명령이 아닌 도구인 이유

세 가지 이유가 있습니다:

1. **백엔드 이식성.** 터미널 도구가 원격 백엔드(Docker / Modal / Singularity / SSH)를 가리키는 작업자는 컨테이너 *내부*에서 `hermes kanban complete`를 실행하게 되며, 이 컨텍스트에는 `hermes`가 설치되어 있지 않고 `~/.hermes/kanban.db`도 마운트되어 있지 않습니다. 칸반 도구는 에이전트의 고유한 Python 프로세스에서 실행되며 터미널 백엔드와 상관없이 항상 `~/.hermes/kanban.db`에 도달합니다.
2. **셸 인용 (Shell-quoting) 문제 없음.** shlex + argparse를 통해 `--metadata '{"files": [...]}'`를 전달하는 것은 잠재적인 위험을 내포합니다. 구조화된 도구 인수는 이를 완전히 건너뜁니다.
3. **더 나은 오류 메시지.** 도구 결과는 모델이 해석(parse)해야 하는 stderr 문자열이 아니라, 모델이 추론할 수 있는 구조화된 JSON입니다.

**일반 세션에 불필요한 스키마 로드 방지.** 정규 `hermes chat` 세션은 활성 프로필이 명시적으로 오케스트레이터 작업에 대해 `kanban` 도구 세트를 활성화하지 않는 한 스키마에 `kanban_*` 도구를 갖지 않습니다. 디스패처가 생성한 작업자는 `HERMES_KANBAN_TASK`가 설정되어 있으므로 작업 범위(task-scoped) 도구를 가져옵니다. 오케스트레이터 프로필은 구성을 통해 광범위한 라우팅 표면을 갖게 됩니다. 칸반을 만지지 않는 사용자를 위한 불필요한 도구 팽창(tool bloat)은 없습니다.

`kanban-worker` 및 `kanban-orchestrator` 기술은 언제 어떤 도구를 호출해야 하는지 모델에게 알려줍니다.

### 추천하는 핸드오프 증명

`kanban_complete(summary=..., metadata={...})`는 의도적으로 유연합니다. `summary`는 인간이 읽을 수 있는 요약이며, `metadata`는 기계 판독 가능한 핸드오프로 하위 에이전트, 검토자 또는 대시보드에서 산문을 스크래핑할 필요 없이 재사용할 수 있습니다.

엔지니어링 및 검토 작업의 경우 이 선택적 메타데이터 형태를 선호합니다.

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

이러한 키는 스키마 요구사항이 아니라 규칙입니다. 유용한 속성은 모든 작업자가 다음 리더가 이 4가지 질문에 빠르게 답할 수 있도록 충분한 증거를 남긴다는 점입니다:

1. 무엇이 바뀌었는가?
2. 어떻게 검증되었는가?
3. 이것이 실패했을 때 어떻게 다시 시도하거나 차단 해제할 수 있는가?
4. 고의로 남겨둔 채 대기 중인 리스크는 무엇인가?

비밀정보(secrets), 원시 로그, 토큰, OAuth 자료 및 관련없는 트랜스크립트는 `metadata`에 두지 마십시오. 대신 포인터 및 요약을 보관하세요. 작업에 파일이나 테스트가 없는 경우 `summary`에 명시적으로 기술하고, `metadata`를 사용하여 소스 URL, 이슈 ID 또는 수동 검토 단계와 같이 실제로 존재하는 증거를 남기세요.

### 워커 기술 (The worker skill)

칸반 작업을 수행해야 하는 모든 프로필은 `kanban-worker` 기술을 로드해야 합니다. 이것은 CLI 명령이 아닌 **도구 호출(tool calls)**로 전체 수명 주기를 작업자에게 가르칩니다:

1. 생성 시, 제목 + 본문 + 부모 핸드오프 + 이전 시도 + 전체 코멘트 스레드를 읽기 위해 `kanban_show()`를 호출합니다.
2. `cd $HERMES_KANBAN_WORKSPACE` (터미널 도구를 통해) 이동하여 거기에서 작업을 수행합니다.
3. 긴 작업 중에는 몇 분마다 `kanban_heartbeat(note="...")`를 호출합니다. **작업이 1시간 이상 소요될 수 있는 경우 1시간에 최소 한 번 `kanban_heartbeat`를 호출해야 합니다** — 디스패처는 작업자가 정리를 수행하지 않고 크래시되었다는 가정 하에 최근 1시간 내에 하트비트 없이 `kanban.dispatch_stale_timeout_seconds`(기본값 4시간)를 넘긴 작업을 회수(reclaim)합니다. 회수는 오류(benign)는 아니지만(작업은 실패-카운터 틱을 올리지 않고 다시 배포를 위해 `ready` 상태로 되돌아감) 현재 실행의 진행 상황을 잃게 됩니다.
4. `kanban_complete(summary="...", metadata={...})`로 완료하거나, 막혔을 경우 `kanban_block(reason="...")`로 작업을 차단합니다.

마지막 `kanban_complete` / `kanban_block` 호출은 작업자 프로토콜의 일부입니다. 작업이 여전히 `running`인 상태에서 작업자 프로세스가 0으로 종료되면 디스패처는 이것을 프로토콜 위반으로 간주하고 `protocol_violation` 이벤트를 발생시키며, 동일한 루프로 다시 재배포하지 않고 다음 틱에 작업을 자동 차단합니다. 이것은 대개 모델이 칸반 도구 표면을 사용하지 않고 일반 텍스트 답변을 작성하고 종료되었음을 의미합니다.

`kanban-worker`는 번들 기술로, 설치 및 업데이트 시 모든 프로필에 동기화됩니다 — 별도의 스킬스 허브(Skills Hub) 설치 단계는 없습니다. 칸반 작업자(`researcher`, `writer`, `ops` 등)에 사용하는 프로필에 존재하는지 확인하세요:

```bash
hermes -p <your-worker-profile> skills list | grep kanban-worker
```

번들로 제공된 복사본이 없는 경우 해당 프로필에 대해 복원하세요:

```bash
hermes -p <your-worker-profile> skills reset kanban-worker --restore
```

디스패처는 모든 작업자를 생성할 때 `--skills kanban-worker`를 자동 전달하므로, 프로필의 기본 기술 구성에 없더라도 작업자는 항상 패턴 라이브러리를 사용 가능한 상태로 생성됩니다.

### 특정 작업에 추가 기술 연결 (Pinning extra skills to a specific task)

때로는 단일 작업이 할당자의 프로필에서 기본적으로 보유하지 않은 전문적인 맥락이 필요할 때가 있습니다 — `translation` 기술이 필요한 번역 작업, `github-code-review`가 필요한 검토 작업, `security-pr-audit`이 필요한 보안 감사가 그렇습니다. 매번 할당자의 프로필을 편집하는 대신 기술을 작업에 직접 연결하세요.

**오케스트레이터 에이전트의 경우** (일반적인 경우 — 한 에이전트가 다른 에이전트에게 작업을 라우팅), `kanban_create` 도구의 `skills` 배열을 사용하세요:

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

**인간의 경우 (CLI / 슬래시 명령)**, 각 기술에 대해 `--skill`을 반복 사용하세요:

```bash
hermes kanban create "translate README to Japanese" \
    --assignee linguist \
    --skill translation

hermes kanban create "audit auth flow" \
    --assignee reviewer \
    --skill security-pr-audit \
    --skill github-code-review
```

**대시보드의 경우**, 인라인 생성 양식의 **skills** 필드에 기술을 쉼표로 구분하여 입력합니다.

이러한 기술들은 내장된 `kanban-worker`에 **추가**됩니다 — 디스패처는 각각의 지정된 기술과 내장된 기술에 대해 하나의 `--skills <name>` 플래그를 발생시켜, 작업자가 지정된 기술을 모두 탑재한 상태로 시작할 수 있도록 합니다. 기술의 이름은 할당된 담당자 프로필에 실제로 설치된 기술의 이름과 일치해야 합니다(`hermes skills list`를 실행하여 확인 가능). 런타임 중에 기술을 동적으로 설치하는 기능은 제공하지 않습니다.

### 목표 지향적 카드 (`--goal`) (Goal-mode cards)

기본적으로 각각의 작업자는 카드마다 **단 한번의 시도**만 허용됩니다 — 작업을 수행하고, `kanban_complete`/`kanban_block`을 호출한 뒤 종료합니다. `--goal` (CLI)을 전달하거나 `goal_mode=True` (`kanban_create` 도구/대시보드)로 지정하면 작업자를 **목표 루프(goal loop)**, 즉 `/goal` 슬래시 명령어의 이면에 있는 Ralph 스타일 엔진에서 실행시킬 수 있습니다: 보조 판정관이 카드의 제목 및 본문(이를 수용 기준(acceptance criteria)으로 간주함)과 작업자의 결과물을 매 턴마다 대조하고, 작업이 완료되지 않은 상태이며 턴 예산이 남아 있는 경우, 작업자는 판정관의 수용을 받거나, 스스로 작업을 멈추거나, 할당된 예산을 모두 소비할 때까지 (예산이 초과될 경우 조용히 종료하지 않고 사람이 개입할 수 있도록 카드를 **막음** (blocks)) **같은 세션 안에서** 작업을 멈추지 않고 지속하게 됩니다.

```bash
hermes kanban create "Translate the docs site to French" \
    --body "Acceptance: every page translated, no English left, links intact." \
    --assignee linguist \
    --goal \
    --goal-max-turns 15      # 선택 사항; 기본값은 20
```

단발성으로 끝나지 않거나, 여러 단계가 필요하거나, "X가 참이 될 때까지 계속" 해야 하는 카드에 사용하세요. 저렴하고 단순한 일회성 작업에는 사용하지 마세요 — 매 턴마다 판정관을 거치는 오버헤드는 가치가 떨어지며, 디스패처에 내장된 재시도/차단 시스템이 이미 일시적인 작업자 오류를 처리해줍니다. 판정관의 성능은 당신이 작성한 목표 문서의 구체성에 전적으로 의지하므로, 본문을 **명시적인 수용 기준(explicit acceptance criteria)**으로 작성하세요.

### 오케스트레이터 기술 (The orchestrator skill)

**잘 만들어진 오케스트레이터는 일을 직접 수행하지 않습니다.** 사용자의 목표를 여러 개의 하위 과제로 나누고, 이들을 연결하여 각각의 과제에 적합한 프로필을 할당한 뒤 물러섭니다. `kanban-orchestrator` 기술은 이를 도구 호출 패턴(tool-call patterns)을 통해 모델에 학습시킵니다. 여기에는 딴 길로 새지 않게 하는 규칙(anti-temptation rules), '0단계 프로필 탐색 안내(Step-0 profile-discovery prompt)'(알 수 없는 담당자 이름이 주어질 경우 디스패처는 소리 없이 작업을 실패 처리하므로, 오케스트레이터는 항상 당신의 머신에 실재하는 프로필을 바탕으로 카드를 만들어야 함), 그리고 `kanban_create` / `kanban_link` / `kanban_comment`에 맞춰 세분화된 작업 지침서(decomposition playbook) 등이 포함됩니다.

전형적인 오케스트레이터의 턴(두 명의 병렬 연구자가 작가에게 넘겨주는 과정):

```
# 사용자의 목표: "draft a launch post on the ICP funding landscape"
kanban_create(title="research ICP funding, NA angle",  assignee="researcher-a", body="…")  # → t_r1
kanban_create(title="research ICP funding, EU angle",  assignee="researcher-b", body="…")  # → t_r2
kanban_create(
    title="synthesize ICP funding research into launch post draft",
    assignee="writer",
    parents=["t_r1", "t_r2"],        # 두 연구원이 모두 완료하면 'ready'로 승격됨
    body="one-pager, neutral tone, cite sources inline",
)                                     # → t_w1
# 선택 사항: 작업을 다시 생성하지 않고 나중에 발견된 교차 의존성을 추가
kanban_link(parent_id="t_r1", child_id="t_followup")
kanban_complete(
    summary="decomposed into 2 parallel research tasks → 1 synthesis task; writer starts when both researchers finish",
)
```

`kanban-orchestrator`는 번들로 제공되는 기술입니다. 이 기술은 설치 및 업데이트 과정에서 각각의 프로필에 자동으로 동기화되므로, Skills Hub에서 따로 설치할 필요가 없습니다. 본인의 오케스트레이터 프로필에 이 기술이 제대로 설치되어 있는지 확인하려면 다음 명령어를 사용하세요:

```bash
hermes -p orchestrator skills list | grep kanban-orchestrator
```

번들 파일이 존재하지 않는다면, 해당 프로필 전용으로 아래 명령어를 통해 이를 복원할 수 있습니다.

```bash
hermes -p orchestrator skills reset kanban-orchestrator --restore
```

최고의 결과를 얻고 싶다면, 이 기술을 도구 집합(toolsets)이 칸반 보드 전용 작업(`kanban`, `gateway`, `memory`)으로만 구성된 프로필과 결합하세요. 그렇게 하면 오케스트레이터가 기술적인 세부 구현 작업을 시도하려 해도 물리적으로 이를 수행할 수 없게 됩니다.

## 대시보드 (GUI)

헤드리스(headless) 방식으로 보드를 운영하는 데에는 `/kanban` CLI와 슬래시 명령어만으로도 충분하지만, 작업 우선순위를 정하고, 서로 다른 프로필들을 교차 점검하며, 댓글 스레드를 읽고, 카드들을 드래그 앤 드롭으로 상태 열(column) 사이를 오가게 하는 사람을 위해 만들어진 그래픽 인터페이스가 시각적인 보드입니다. Hermes는 이것을 내장된 기능이나 별개의 서비스가 아닌, **번들로 제공되는 대시보드 플러그인**(`plugins/kanban/`) 형태로 지원하며, 이는 [대시보드 확장(Extending the Dashboard)](./extending-the-dashboard)에 제시된 모델을 따릅니다.

대시보드 여는 방법:

```bash
hermes kanban init      # 최초 1회: kanban.db가 아직 없을 때 생성함
hermes dashboard        # 내비게이션 바의 "Skills" 다음에 "Kanban" 탭이 나타남
```

### 플러그인이 제공하는 기능

- 각 상태에 따른 열(column)을 보여주는 **칸반(Kanban)** 탭: `triage`, `todo`, `ready`, `running`, `blocked`, `done` (토글을 켜면 `archived` 포함).
  - `triage`는 아직 덜 다듬어진 아이디어들을 위한 대기열입니다. 기본 설정(`kanban.auto_decompose: true`)을 유지하면, 디스패처가 이 위치에 도착하는 작업들에 대해 **분해기(decomposer)**를 자동 실행합니다. 오케스트레이터 프로필이 거친 아이디어를 읽고, 프로필 명단(설명 포함)을 확인하여 작업을 세분화된 하위 작업들의 그래프 형태로 분배합니다. 원래의 작업은 생성된 모든 하위 작업들의 부모(parent)로서 그대로 남아있게 되어, 하위 작업들이 모두 끝나면 오케스트레이터가 깨어나 작업 완료 여부를 다시 판단할 수 있도록 합니다. 대시보드 상단의 **Orchestration: Auto/Manual** 버튼을 클릭하거나 `kanban.auto_decompose: false`를 설정하여, 이러한 오케스트레이션 과정을 수동 모드로 바꿀 수 있습니다. 수동 모드에서는 사용자가 카드에서 **⚗ Decompose**를 누르거나, `hermes kanban decompose <id>`를 실행할 때까지 작업이 triage에서 움직이지 않습니다. 작업을 더 작게 나눌 필요가 없거나 오케스트레이터 프로필이 설정되지 않은 경우, **✨ Specify** 버튼을 통해 동일한 언어 모델 엔진을 거쳐 단일 작업에 대한 구체화 작업(목표, 접근법, 허용 기준을 포함하여 제목 및 본문을 새롭게 작성함)을 수행할 수 있습니다. 자세한 내용은 아래의 [자동(Auto) 및 수동(Manual) 오케스트레이션(Auto vs Manual orchestration)](#auto-vs-manual-orchestration) 항목을 참고하세요.
- 각 카드에는 작업 ID, 제목, 중요도를 나타내는 배지, 속한 그룹(테넌트), 담당 프로필 명, 댓글 및 연결된 링크의 개수, (다른 작업을 하위로 두고 있는 경우 보여지는) 작업 **진행률**(예: `N/M`), 그리고 "생성된 지 N일/시간 지남"이 표시됩니다. 각 카드별로 체크 박스가 있어 다중 선택이 가능합니다.
- **프로필별 진행 레인(Per-profile lanes inside Running)** — 툴바의 체크박스를 통해 진행 중(Running)인 작업들을 담당자별로 하위 그룹화하는 기능을 껐다 켤 수 있습니다.
- **WebSocket을 통한 실시간 업데이트** — 이 플러그인은 짧은 주기로(short poll interval) 추가만 가능한 `task_events` 테이블의 끝단을 추적(tails)합니다; 프로필(CLI, 게이트웨이, 또는 또 다른 대시보드 탭)에서 조작을 가하는 그 순간 즉시 보드에 반영됩니다. 과부하를 막기 위해 디바운싱(debouncing) 처리가 되어 있어, 짧은 순간에 폭주하는 이벤트들도 한 번의 데이터 재요청(refetch)만으로 처리됩니다.
- 상태를 변경하려면 카드를 다른 열(column)로 **드래그 앤 드롭**하세요. 드롭다운 작업은 `PATCH /api/plugins/kanban/tasks/:id` 명령어를 전송하며, 이는 CLI가 쓰는 것과 동일한 `kanban_db` 코드를 관통합니다 — 세 종류의 표면(CLI, 게이트웨이, 대시보드)이 따로 노는 일은 발생하지 않습니다. 파괴적인 특성을 갖는 상태로(`done`, `archived`, `blocked`) 옮길 때는 다시 한번 확인 절차를 거칩니다. 터치 디바이스 이용 시 포인터 기반의 방식을 사용하여 태블릿 환경에서도 보드를 무리 없이 쓸 수 있습니다.
- **인라인 생성(Inline create)** — 컬럼 상단의 `+`를 눌러 제목, 할당자, 중요도, 그리고 (선택적으로) 기존 태스크 전체를 보여주는 드롭다운 리스트에서 부모 태스크를 입력할 수 있습니다. 엔터를 치면 생성되고, Shift+Enter를 치면 제목 필드에서 줄 바꿈이, Escape 키를 누르면 취소됩니다. Triage(우선순위 분류) 칼럼에서 생성한 태스크는 자동으로 분류 대기 상태로 유지됩니다.
- **벌크 액션 다중 선택(Multi-select with bulk actions)** — 카드를 Shift/Ctrl 클릭하거나 체크박스를 체크해 선택 범위에 추가하세요. 상단에 나타나는 벌크 액션 바를 이용해 여러 상태의 일괄 변경(status transitions), 아카이브, 그리고 할당자 재배치(프로필 드롭다운이나 "(unassign)")를 수행할 수 있습니다. 되돌릴 수 없는 조작일 경우 실행 전 먼저 확인합니다. 개별 ID 중 실패한 항목이 있더라도 나머지 동작을 중단하지 않고 모두 보고합니다.
- (Shift나 Ctrl 없이) **카드를 클릭**하면 화면 우측에서 서랍창이 나타납니다(바깥 공간을 클릭하거나 Escape를 누르면 닫힘):
  - **편집 가능한 제목** — 머리말을 클릭하면 이름을 변경할 수 있습니다.
  - **편집 가능한 할당자 / 중요도** — 수정하려면 메타(meta) 행을 클릭하세요.
  - **편집 가능한 설명** — 기본적으로 마크다운(제목, 굵게, 기울임꼴, 인라인 코드, 코드 블록, `http(s)` / `mailto:` 링크, 글머리 기호 목록)으로 렌더링되며, 클릭 시 텍스트 에어리어로 변하는 "편집" 버튼이 함께 제공됩니다. 마크다운 렌더링은 가볍고 XSS의 위협으로부터 안전한 렌더러를 바탕으로 구동됩니다. 오직 HTML 이스케이프 된 텍스트만을 입력값으로 받아들이며, 오로지 `http(s)` / `mailto:`로 시작하는 링크만 활성화시키고, 링크를 새 창으로 열게 하는 `target="_blank"` 속성과 보안 강화를 위한 `rel="noopener noreferrer"` 속성을 의무적으로 적용합니다.
  - **의존성 편집기(Dependency editor)** — 상위(parents)와 하위(children) 작업들을 보여주는 칩 리스트이며 각각 연결을 해제할 수 있는 `×` 표시가 있습니다. 추가적으로 새로운 상하위 태스크를 등록할 수 있게 해 주는 타 태스크 열람 드롭다운이 함께 제공됩니다. 순환 구조를 발생시키는 의존성 추가 시도는 거부되며 명확한 메시지가 출력됩니다.
  - **상태 액션 행(Status action row)** (→ triage / → ready / → running / block / unblock / complete / archive)으로 되돌릴 수 없는 변경 시 확인 절차를 거칩니다. **Triage** 상태에 있는 카드의 경우, 이 액션 열에는 두 가지 LLM(대형 언어 모델) 주도 기능도 함께 표시됩니다: **⚗ Decompose**는 (오케스트레이터의 주도 아래) 태스크를 하위 작업의 그래프 구조로 쪼개고 각각의 하위 작업을 그에 가장 알맞은 성격(description)의 프로필들에게 할당해 주며, **✨ Specify**는 하나의 태스크 내용을 전체적으로 재작성(rewrite)해 줍니다. 만약 LLM이 이 작업이 여러 갈래의 하위 작업으로 나뉠 필요가 없다고 판단할 경우 Decompose는 specify 단계로 격하(fallback)하여 승격 단계를 거치므로 완벽하게 상위의 개념이라고 할 수 있습니다. 두 기능 모두 CLI(`hermes kanban decompose <id>` / `specify <id>` / `--all`), 게이트웨이 플랫폼 상의 명령(`/kanban decompose <id>`), API를 통한 직접 통신(`POST /api/plugins/kanban/tasks/:id/decompose` 및 `…/specify`)의 세 가지 방식으로 접근할 수 있습니다. 작동에 사용할 모델 정보는 `config.yaml` 파일 내의 `auxiliary.kanban_decomposer`와 `auxiliary.triage_specifier` 항목을 통해 변경할 수 있습니다.
  - 결과 섹션(마찬가지로 마크다운으로 표시), Enter를 눌러 전송하는 코멘트 스레드(comment thread), 마지막 20개의 이벤트들이 표시됩니다.
  - **툴바 필터(Toolbar filters)** — 자유 텍스트(free-text) 기반 검색창, 테넌트(tenant) 선택 드롭다운(`config.yaml` 파일 내의 `dashboard.kanban.default_tenant` 설정이 기본값으로 지정됨), 작업 할당자 선택 드롭다운, "보관된 작업 보기(show archived)" 기능 활성화 스위치, "프로필별 레인 보기(lanes by profile)" 기능 활성화 스위치, 그리고 디스패처의 60초 간격 사이클을 기다리지 않고 즉시 실행시킬 수 있는 **Nudge dispatcher** 버튼이 제공됩니다.

시각적인 형태는 흔히 알려진 Linear나 Fusion의 디자인 요소를 본뜹니다: 어두운 테마(dark theme), 개수 표기가 함께 하는 각 칸의 이름(column headers), 다양한 색상을 갖는 진행 상태 표기점(coloured status dots), 중요도(priority)와 테넌트(tenant)를 표기하는 둥근 알약 형태의 칩(pill chips). 이 플러그인은 환경 테마 CSS 변수(`--color-*`, `--radius`, `--font-mono`, ...)만을 덮어쓰기 때문에 활성화된 대시보드 테마가 어떤 것이든 그에 맞춰 자동으로 디자인이 재조정됩니다.

### 자동화 (Auto) vs 수동조작 (Manual) 오케스트레이션 (Auto vs Manual orchestration)

Triage(분류 대기)열에 등록된 새로운 업무를 다룰 때, 칸반 보드는 두 가지 처리 방식을 제시합니다:

**자동(Auto) (기본값)** — `kanban.auto_decompose: true`. 게이트웨이(gateway)에 편입된 디스패처가 정해진 사이클마다 자동으로 **decomposer**를 작동시킵니다. 대량의 Triage 작업을 한꺼번에 분해하여 보조 LLM의 토큰을 폭발적으로 써버리는 일을 방지하기 위해 이 작동은 매 사이클마다 `kanban.auto_decompose_per_tick`에 지정된 상한선(기본 3개)으로 횟수를 제한받습니다. 이 분해기(decomposer) 기능은 대강 적힌 아이디어를 읽고, 기기에 설치된 프로필 목록과 그들의 설명(descriptions) 정보들을 살펴본 뒤, LLM에게 여러 갈래로 나뉘는 작업들의 관계망 그래프(JSON task graph)를 작성할 것을 요청합니다: 어떠한 업무를 만들어 낼 것인지, 그 임무들을 어느 프로필에게 넘길 것인지, 그리고 어떤 임무가 어느 것에 앞서야 하는지. 최초의 덜 다듬어졌던 업무는 그래프 하단 모든 갈래들의 부모 요소가 되며, 하위 업무들의 모든 수행이 완료될 때까지 살아있게 됩니다. 하위 업무들이 모두 마쳐진 뒤 이 부모 임무는 `ready` 단계로 승격하게 되고 할당받은 프로필(오케스트레이터 프로필)이 결과물들을 판단한 뒤 목표가 완전히 이뤄지지 않았다면 새로운 임무를 다시 지시하게 됩니다. 이것이 곧 "한 줄 던져놓고 신경 쓰지 않기(drop a one-liner, walk away)"의 과정입니다.

**수동(Manual)** — `kanban.auto_decompose: false`. 사용자가 명령을 지시할 때까지 등록된 업무들이 Triage 열에서 대기합니다. 카드의 **⚗ Decompose** 버튼을 누르거나 `hermes kanban decompose <id>` (또는 `--all`) 명령어를 치거나, 대화창에서 `/kanban decompose <id>`를 입력하세요. 무엇이, 언제 수행될지를 완전히 스스로 제어하고 싶을 때 유용하며, 디컴포저 기능이 도입되기 전 보드가 동작했던 예전 방식과 똑같이 작동합니다.

칸반 페이지 상단에 위치한 알약 형태의 **Orchestration: Auto/Manual** 토글(에메랄드색 = Auto, 옅은 회색 = Manual)을 누르거나, `config.yaml`을 직접 편집하여 둘 사이의 모드를 오갈 수 있습니다. 두 모드 모두 `hermes kanban specify`와 양립할 수 있습니다. 팬아웃 없이 단일 태스크의 명세를 재작성하고 싶다면 이 기능은 항상 사용 가능합니다.

디컴포저(decomposer)가 업무를 프로필들에게 연결하는 과정은 프로필의 속성(profile descriptions) 설정에 영향을 받습니다. 이 프로필 속성은 `hermes profile create --description "..."`, `hermes profile describe <name> --text "..."`, `hermes profile describe <name> --auto` (LLM이 설치된 스킬 세트와 모델을 분석해 프로필의 속성문을 스스로 씀), 또는 확장된 **Orchestration settings** 패널 내에 위치한 대시보드의 개별 프로필 설정 페이지를 통해 프로필 각각에 레이블을 붙여 넣는 원시적(primitive) 과정입니다. 속성값이 없는 프로필 또한 목록에는 나타나고 이름을 기반으로 임무들을 부여받게 되지만 정확도는 떨어지게 됩니다. 분해기가 만들어 내는 새로운 하위 자식 임무들이 `assignee=None`(할당받은 자 없음)으로 떨어지는 일은 발생하지 않습니다: 만약 LLM이 누군지 알 수 없는 프로필을 지목하게 되면, 새로운 자식 태스크는 곧바로 `kanban.default_assignee`(설정되어 있지 않으면 기본 활성 프로필)로 연결(routed)됩니다.

구성 설정(Config knobs) (전부 `~/.hermes/config.yaml` 내의 `kanban:` 영역에 있음):

| 설정 (Key) | 기본값 (Default) | 기능 (Purpose) |
|---|---|---|
| `auto_decompose` | `true` | 디스패처가 틱(사이클)마다 자동적으로 디컴포저를 실행시킴. |
| `auto_decompose_per_tick` | `3` | 디스패처가 한 틱마다 실행할 수 있는 분해 횟수의 최대 허용 수치. 허용치를 넘는 경우 다음 틱으로 넘겨짐. |
| `orchestrator_profile` | `""` | 분해 과정을 총괄하게 되는 프로필. 공란 시 = 현재의 기본 프로필로 구동됨. |
| `default_assignee` | `""` | LLM이 존재하지 않는 프로필을 지정했을 때 하위 임무들이 안착하게 되는 장소. 공란 시 = 현재의 기본 프로필로 구동됨. |

아울러 두 종류의 보조 LLM 자리가 존재합니다:

| 설정 (Key) | 기능 (Purpose) |
|---|---|
| `auxiliary.kanban_decomposer` | (Decompose 버튼에서 호출되며) 관계망 그래프들을 뿜어내는 모델. `provider`/`model` 설정을 변경하여 주요 대화에 쓰이는 챗 모델이 아닌 타 모델을 덮어써서 활용 가능. |
| `auxiliary.profile_describer` | (`hermes profile describe --auto`에 의해 불려 나오며) 프로필의 성격을 자동으로 적어 주는 모델. |

### 설계 구조 (Architecture)

GUI는 데이터베이스의 원본을 오로지 읽기만 하며, 데이터 작성의 경우 자체적인 도메인 로직(domain logic)의 개입 없이 kanban_db라는 정해진 통로만을 거치는 **read-through-the-DB + write-through-kanban_db** 구조를 철저히 지킵니다.

<!-- ascii-guard-ignore -->
```
┌────────────────────────┐      WebSocket (task_events 뒤쫓음)
│   React SPA (plugin)   │ ◀──────────────────────────────────┐
│   HTML5 drag-and-drop  │                                    │
└──────────┬─────────────┘                                    │
           │ REST over fetchJSON                              │
           ▼                                                  │
┌────────────────────────┐     데이터 입력 시 CLI의           │
│  FastAPI router        │     /kanban 명령들과 똑같은        │
│  plugins/kanban/       │     과정으로 kanban_db.*가         │
│  dashboard/plugin_api.py     직접 불리게 됩니다             │
└──────────┬─────────────┘                                    │
           │                                                  │
           ▼                                                  │
┌────────────────────────┐                                    │
│  ~/.hermes/kanban.db   │ ───── task_events 덧붙임 ──────────┘
│  (WAL, shared)         │
└────────────────────────┘
```
<!-- ascii-guard-ignore-end -->

### REST 형태 설계 (REST surface)

모든 데이터 라우트 정보들은 `/api/plugins/kanban/` 아래에 놓여 있으며 대시보드 내부의 일시적인 세션 토큰에 의해 안전히 보호받습니다.

| 통신 방법 (Method) | 경로 (Path) | 기능 (Purpose) |
|---|---|---|
| `GET` | `/board?tenant=<name>&include_archived=…` | 상태, 테넌트 설정, 그리고 할당자 분류 조건이 반영된 전체 보드 정보를 요청 |
| `GET` | `/tasks/:id` | 임무 + 관련 코멘트 + 기록 + 관련 태스크 링크 정보 |
| `POST` | `/tasks` | 임무 생성 (kanban_db.create_task를 통해 수행되며, `triage: bool` 항목과 `parents: [id, …]` 항목을 값으로 받음) |
| `PATCH` | `/tasks/:id` | 특정 임무의 진행 상태, 할당자, 중요도, 제목, 본문, 그리고 결과를 변경함 |
| `POST` | `/tasks/bulk` | 동일한 변경사항(진행 상태 / 아카이브 여부 / 할당자 / 중요도)들을 해당 `ids` 전체에 한꺼번에 덮어씀. 특정 id에 대한 처리가 실패하더라도 에러를 발생시켜 다른 작업에 영향을 주지 않음 |
| `POST` | `/tasks/:id/comments` | 특정 임무에 새로운 의견 추가 |
| `POST` | `/tasks/:id/specify` | (보조 LLM이) `triage`(분류 대기 중) 상태의 임무를 상세하게 다시 기술하고 이를 `todo` 상태로 한 단계 올림. 작업이 성공적으로 이루어질 경우 `{ok, task_id, reason, new_title}` 정보를 되돌려 줌. 임무가 "not in triage" 이거나 보조 클라이언트가 없거나 혹은 LLM 통신 과정에서의 에러를 이유로 작업이 실패할 경우 사람이 읽기 편한 에러와 함께 `ok=false`를 반환하고 이는 `4xx` 상태코드가 아닌 `200` 정상 처리로 여겨짐 |
| `POST` | `/tasks/:id/decompose` | (보조 LLM이) 임무들의 네트워크망을 만들고 헬퍼를 불러 하위 태스크를 생성, 최초 작업과 묶어 준 뒤 이들의 상태를 `triage → todo` 상태로 바꿈. 작업 결과로 `{ok, task_id, reason, fanout, child_ids, new_title}`가 돌아오고, `/specify` 상황과 마찬가지로 LLM에 에러가 나더라도 `200` 정상 처리 코드로 여김. |
| `GET` | `/profiles` | 대시보드 상의 프로필 설명 편집기나 오케스트레이터 목록에서 불러다 쓸 수 있는 설치 완료된 프로필 정보와 세부 설명 요청 |
| `PATCH` | `/profiles/:name` | (유저가 직접) 프로필의 세부 내용 쓰기, 고치거나 비움 (`description_auto: false`). `{ok, profile, description}`을 반환. |
| `POST` | `/profiles/:name/describe-auto` | `auxiliary.profile_describer`를 불러다가 프로필의 성격을 자동으로 적어 내리게 함. 이 설정은 `description_auto: true`와 함께 기록되어 대시보드가 언제든지 "review" 배지를 달고 나타나도록 함 |
| `GET` | `/orchestration` | `config.yaml` 파일 내에 기록되어 있던 `orchestrator_profile`, `default_assignee`, `auto_decompose` 등의 세팅 값과 오류 검증 후 최종적으로 사용되는 유효한 세팅값을 호출함 |
| `PUT` | `/orchestration` | `config.yaml` 안의 오케스트레이션 키값 3개 중 일부나 혹은 3개 모두 갱신. 지정된 이름의 프로필들이 시스템 안에 온전히 있는지를 확인(검증)함. |
| `POST` | `/links` | 임무 간 상하 의존관계를 만듦 (`parent_id` → `child_id`) |
| `DELETE` | `/links?parent_id=…&child_id=…` | 의존 관계 끊기 |
| `POST` | `/dispatch?max=…&dry_run=…` | 디스패처가 다음 주기인 60초까지 기다리지 않도록 재촉함 |
| `GET` | `/config` | `config.yaml`의 `dashboard.kanban`에 있는 설정값들을 가져옴 — `default_tenant`, `lane_by_profile`, `include_archived_by_default`, `render_markdown` |
| `WS` | `/events?since=<event_id>` | `task_events` 로우(row)들에 대한 정보를 계속 쏘아줌 |

모든 통신 처리기는 단순한 다리 역할을 수행합니다. 라우터와, 이벤트 발생 시 실시간 알람, 그리고 정보를 대량으로 쏟아부을 수 있는 기능과 설정 값을 읽어들이는 기능으로 이루어진 총 700줄가량의 Python 코드로 구성된 플러그인은 추가적인 비즈니스 로직(business logic)을 일절 갖지 않습니다. 사용자가 대시보드를 직접 켜거나, 직접 REST 통신을 날리든, 혹은 `hermes kanban init`를 실행시키던지 상관없이 읽고 쓰기 과정의 시작에 단출한 형태의 `_conn()` 헬퍼 기능이 kanban.db 데이터를 자동으로 불러일으키기 때문에 시스템 구축 즉시 동작할 수 있습니다.

### 대시보드 설정 (Dashboard config)

`~/.hermes/config.yaml`의 `dashboard.kanban` 하위에 있는 다음 키들 중 하나를 수정하면 탭의 기본값이 변경됩니다 — 플러그인은 로드 시 `GET /config`를 통해 값을 읽습니다:

```yaml
dashboard:
  kanban:
    default_tenant: acme              # 테넌트 필터 초기 설정값
    lane_by_profile: true             # "lanes by profile" 기본 상태 (프로필별 구분)
    include_archived_by_default: false
    render_markdown: true             # 마크다운 없이 일반 글자로 보여주려면 false로 설정
```

각 항목의 변경은 자유로우며, 기본값이 제공되지 않을 경우 표기된 값으로 실행됩니다.

### 보안 모델 (Security model)

대시보드의 HTTP 자격 증명 관련 기능은 [명시적으로 `/api/plugins/`의 인증 검사를 생략합니다(explicitly skips `/api/plugins/`)](./extending-the-dashboard#backend-api-routes). 플러그인의 경로 설정에는 처음부터 인증 절차를 갖지 않도록 계획되었기 때문입니다. 즉 대시보드는 `localhost` 통신만 하도록 기본 설정되어 있어서, 칸반의 REST 통신 역시 그 기계 내 다른 장치들로부터는 모두 접속이 가능하게 됩니다.

웹소켓(WebSocket) 설정 시 한 단계 보안을 거칩니다: 웹소켓을 설정하려면 대시보드 내부의 일시적인 세션 토큰 정보를 쿼리 파라미터(`?token=…`) 형태로 제공받아야 하는데, (브라우저는 연결 업그레이드를 위한 자격 증명 통신 헤더(`Authorization`)를 전송할 수 없으므로) 이는 PTY 연결 과정을 사용할 때와 동일한 패턴입니다.

어떤 이유에서건 네트워크상의 모든 곳에서 대시보드로 통신하는 기능(`hermes dashboard --host 0.0.0.0`)을 켜놓았다면 플러그인의 모든 경로와 칸반 보드가 그대로 외부로 향하게 됩니다. **여러 기기가 공유된 서버 환경에서는 이 기능을 절대 활성화하지 마세요.** 칸반 보드 내 데이터베이스에는 태스크 본문과, 코멘트, 작업 저장 위치 등이 전부 노출됩니다; 이런 정보들이 유출되면 다른 누군가가 당신의 대화 스레드를 마음대로 보고 지시 사항을 내리거나 다른 작업자에게 배분하고 중요 내용을 파기하는 일들이 벌어집니다.

`~/.hermes/kanban.db` 안에 포함된 임무들은 의도적으로 프로필의 영향을 받지 않는 (profile-agnostic) 구조를 띕니다 (이는 임무 간의 긴밀한 정보 교환 및 제어를 위한 설정이기도 합니다). 만일 당신이 특정 프로필(`hermes -p <profile> dashboard`)로 대시보드 창을 띄웠더라도 같은 기계 환경을 거쳐 작성된 다른 프로필들의 임무들 역시 그대로 당신의 시야에 함께 놓이게 될 것입니다. 한 명의 이용자가 이 모든 프로필들을 사용한다 할지라도, 이렇게 다양한 정체성들이 한데 얽혀 있다는 사실 자체는 사용자에게 혼란을 줄 수 있으므로 알고 넘어갈 필요가 있습니다.

### 실시간 데이터 갱신 (Live updates)

`task_events`는 고유 ID를 부여받고 추가만이 가능한 형태의 SQLite 테이블입니다. 웹소켓 기능은 접속된 각각의 클라이언트가 어느 데이터 단계까지 지켜보았는지를 파악하여 변화가 생길 때만 실시간 데이터를 제공합니다. 다수의 이벤트들이 몰려올 때마다 프로그램 앞단(frontend)에서는 아주 가벼운 재로딩 과정(board endpoint)만을 실시하는데 이는 각 종류들의 이벤트를 처리하기 위한 복잡한 과정을 여러 차례 거치는 일보다 훨씬 단순하고도 효율적입니다. 또한, 읽기 처리 시스템과 디스패처의 즉각적인 읽고 쓰기 처리 시스템(`BEGIN IMMEDIATE`)이 서로 동시에 돌아갈 수 있도록 WAL 구조가 쓰여 시스템의 정지를 미연에 막습니다.

### 확장성 (Extending it)

플러그인은 Hermes의 기본 대시보드 확장 방식을 따릅니다 — 확장 파일, 외부 스크립트 연결 공간, 대시보드 구조의 자유로운 조작을 가능하게 하는 스크립트 도구들 등 보다 자세한 내용을 원한다면 [대시보드 확장(Extending the Dashboard)](./extending-the-dashboard) 기능을 참조하세요. 플러그인을 복사하여 개조할(fork) 수고로움 없이도 여분의 칸(extra columns), 커스텀 디자인 기능, 임무 배정자들의 설정 정보 등을 관리하거나 자유롭게 `tab.override` 내용을 덮어씌울 수 있습니다.

플러그인 자체를 지워버리기보다는 기능 정지를 원한다면 `config.yaml` 파일에서 `dashboard.plugins.kanban.enabled: false` 구문을 추가하세요 (혹은 `plugins/kanban/dashboard/manifest.json` 파일을 지워버려도 무관합니다).

### 기능 제공 범위 한계 (Scope boundary)

설계된 GUI의 두께는 무척 얇습니다. 화면의 인터페이스로 움직이는 일련의 과정들은 CLI로 쳐서 넣었을 때와 모두 같습니다; 그저 대시보드의 사용으로 인간이 그것들을 더욱더 편리하게 접할 수 있을 뿐. 정보의 자동 조달, 예산 관리, 보안 통과 기능 그리고 시스템 구축망 등의 요소들은 오직 사용자 권한 체계에서만 제공됩니다. 이를 위해서는 안내의 기능(router profile), 기타 기능을 수행할 보조 기능들, 그리고 승인을 담당하는 기능들(`tools/approval.py`의 재사용) 등이 모두 모여야 하는데 이런 요소들은 맨 처음 제품 디자인 구조 과정에서 거론되었던 범주 밖의 일들이기도 합니다.

## CLI 명령어 참조 문서 (CLI command reference)

이것은 **사용자**(또는 스크립트, 크론, 대시보드)가 보드를 직접 운영하기 위해 쓰는 시스템 영역입니다. 디스패처 환경 속의 작업자들은 이 CLI를 일절 사용하지 않고 `kanban_*`으로 시작되는 [도구 표면부(tool surface)](#how-workers-interact-with-the-board) 명령만을 쓰게 됩니다.
```bash
hermes kanban init                                     # kanban.db 생성 + 데몬 힌트 출력
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
hermes kanban assign <id> <profile>                    # 'none' 지정 시 할당 해제
hermes kanban link <parent_id> <child_id>
hermes kanban unlink <parent_id> <child_id>
hermes kanban claim <id> [--ttl SECONDS]
hermes kanban comment <id> "<text>" [--author NAME]

# 대량 처리 (Bulk) 명령어 — 여러 id를 받을 수 있습니다:
hermes kanban complete <id>... [--result "..."]
hermes kanban block <id> "<reason>" [--ids <id>...]
hermes kanban unblock <id>...
hermes kanban archive <id>...

hermes kanban tail <id>                                # 단일 작업의 이벤트 스트림 추적
hermes kanban watch [--assignee P] [--tenant T]        # 모든 이벤트를 터미널로 실시간 스트리밍
        [--kinds completed,blocked,…] [--interval SECS]
hermes kanban heartbeat <id> [--note "..."]            # 긴 작업을 위한 작업자 생존 신호
hermes kanban runs <id> [--json]                       # 시도 이력 (실행당 하나의 행)
hermes kanban assignees [--json]                       # 디스크상의 프로필 + 할당자별 작업 수
hermes kanban dispatch [--dry-run] [--max N]           # 원샷 패스 (디스패처 한 번 실행)
        [--failure-limit N] [--json]
hermes kanban daemon --force                           # 사용 중단됨 (DEPRECATED) — 독립형 디스패처 (대신 `hermes gateway start` 사용 권장)
        [--failure-limit N] [--pidfile PATH] [-v]
hermes kanban stats [--json]                           # 상태별 + 할당자별 통계 수
hermes kanban log <id> [--tail BYTES]                  # ~/.hermes/kanban/logs/ 의 작업자 로그 출력
hermes kanban notify-subscribe <id>                    # 게이트웨이 연결 고리 (게이트웨이 내 /kanban 에서 사용)
        --platform <name> --chat-id <id> [--thread-id <id>] [--user-id <id>]
hermes kanban notify-list [<id>] [--json]
hermes kanban notify-unsubscribe <id>
        --platform <name> --chat-id <id> [--thread-id <id>]
hermes kanban context <id>                             # 작업자가 보게 될 내용
hermes kanban specify [<id> | --all] [--tenant T]      # triage 열의 간단한 아이디어를 완전한 세부 사항으로 구체화하고
        [--author NAME] [--json]                       #   todo 열로 승격시킴
hermes kanban gc [--event-retention-days N]            # 작업 공간 + 오래된 이벤트 + 오래된 로그 정리
        [--log-retention-days N]
```

위의 모든 명령어는 대화형 CLI 및 메시징 게이트웨이(Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, 이메일, SMS)에서 슬래시 명령어 형태로도 이용 가능합니다 (아래 [ `/kanban` 슬래시 명령어 ](#kanban-slash-command) 항목 참조).

`--max-retries`는 디스패처를 위한 작업별 서킷 브레이커 설정입니다. `--max-retries 1` 로 설정 시 첫 실패 직후 작업이 차단되며, `--max-retries 3`으로 설정 시 2번의 재시도를 허용하고 3번째 실패 때 차단합니다. 값이 지정되지 않은 경우, 우선적으로 `config.yaml`에 있는 `kanban.failure_limit` 값을, 그 다음으론 기본 내장 값을 적용합니다.

### 동시성, 스케줄링 및 하위 작업 승격 설정 (Concurrency, scheduling, and child promotion config)

| 설정값 (Config key) | 기본값 (Default) | 기능 (What it does) |
|---|---|---|
| `kanban.max_in_progress` | 설정되지 않음 (무제한) | 동시에 실행 가능한 최대 작업 개수를 제한합니다. 이미 N개의 작업이 보드에서 돌아가고 있으면 디스패처가 새 작업 추가를 중지합니다. 속도가 느린 작업자들(로컬 LLM, 한정된 자원의 호스트 장치)이 시간 초과를 겪거나 일이 밀려 쌓이는 것을 막기 위해 현재 처리 중인 일에 집중할 수 있게 합니다. 이 값이 올바르지 않거나 1 미만이면 경고를 띄우고 무제한 모드로 작동합니다. |
| `kanban.auto_promote_children` | `true` | `decompose_triage_task()`가 다른 상위 작업에 의해 막히지 않은 하위 작업들을 만들어 내면 디스패처가 이들을 곧장 `ready` 단계로 자동 승격시킵니다. 수동 확인을 거치고 싶다면 `false`로 설정하세요 — 그렇게 하면 이들 작업은 사용자가 직접 승격시킬 때까지 `todo` 단계에 머무릅니다. |
| `kanban.default_workdir` | 설정되지 않음 | `--workspace`나 작업 자체의 설정 중 어느 쪽으로도 작업 디렉토리가 명시되지 않은 신규 작업들에 적용될 보드 단위의 기본 설정 폴더입니다. 각각의 작업에 붙은 `workspace:` 옵션이 우선합니다. |

```yaml
kanban:
  max_in_progress: 2
  auto_promote_children: false
  default_workdir: ~/work/active-project
```

### 예약된 작업 시작 (`scheduled_at`)

작업에 `scheduled_at`을 설정하여 특정 시간까지 배포를 지연시킵니다. 디스패처는 준비(ready)된 작업 중 `scheduled_at`이 미래인 작업은 건너뛰고 해당 타임스탬프 이후의 첫 번째 주기(tick)에 선택합니다.

```bash
hermes kanban create "nightly backup audit" \
  --assignee ops --scheduled-at "2026-06-01T03:00:00Z"
```

### 재시작 방지 (Respawn guard)

디스패처는 이전에 할당량/인증/429 에러를 마주쳤거나(`blocker_auth`), 아주 짧은 찰나에 성공적으로 작업을 수행해 낸 기록이 있거나(`recent_success`), 혹은 의견란에 깃허브 Pull Request 링크가 올라와 있을 경우(`active_pr`), 준비된 태스크(ready task)의 재생성을 허가하지 않습니다. 이는 인간이 작업 속도를 따라잡을 때까지 동일한 문제나 작업에 대해 작업자 폭주가 반복되는 현상을 방지하기 위한 것입니다. 관련된 자세한 기록은 [이벤트 참조 (Event reference)](#event-reference) 내 `respawn_guarded` 정보를 참고하세요.

### 드래그 앤 드롭 삭제 및 일괄 삭제 (Drag-to-delete and bulk delete) (dashboard)

대시보드 칸반 페이지에는 **휴지통 영역(trash drop zone)**이 마련되어 있습니다 — 어떠한 카드든지 이곳으로 끌어다 놓으면 임무가 파기되며(`task_events` 내용, 연결된 하위 작업, 알림 구독도 함께 삭제됨), 파기 전에는 실수를 방지하기 위한 안내 창이 먼저 열립니다. 다중 삭제(Bulk delete)는 JSON 형식의 `{"ids": ["t_abc", "t_def", ...]}` 내용을 담은 `DELETE /api/plugins/kanban/tasks`를 통해서도 이행 가능합니다.

### 작업자 가시성 엔드포인트 (Worker visibility endpoints)

대시보드 플러그인 API는 외부 모니터링 시스템에서 읽을 수 있는 3개의 전용 경로(read-only)를 추가로 열어놓았습니다:

| 엔드포인트 (Endpoint) | 반환값 (Returns) |
|---|---|
| `GET /api/plugins/kanban/workers/active` | 진행 중인 모든 프로세스의 PID, 프로필 명, 임무 ID, 시작 시간, 최근 응답 확인 시간(last heartbeat) |
| `GET /api/plugins/kanban/runs/{id}` | 각 수행별 상세 내용 (작업 id, 상태, 시작/종료 시점, 종료 코드, 로그 파일의 위치) |
| `GET /api/plugins/kanban/inspect` | 디스패처의 통합 상황 정보 — 지연 중인 업무 목록, 현재 동시 수행 수치와 그 상한치(`max_in_progress`), 최근의 이벤트 기록 |

위 3가지 경로들 또한 대시보드 플러그인과 동일한 인증 과정을 준수합니다.

### 칸반 스웜 (Kanban Swarm) 토폴로지 헬퍼

`hermes kanban swarm`은 한 번에 완전한 형태의 **Kanban Swarm v1** 그래프를 만듭니다: 최상위 작업이자 전체 현황판(root/blackboard)이 되는 카드 하나, 다수의 병렬 작업자 카드들, 모든 작업자의 작업 완료 후 구동되는 하나의 검증자(verifier) 카드, 그리고 검증자 카드가 작업 완료를 승인하면 마지막으로 켜지게 되는 종합 작성자(synthesizer) 카드 하나로 구성됩니다. 이 모든 군집 작업의 내역(blackboard)은 최상위 카드의 구조화된 JSON 형태로 코멘트란에 지속적으로 쓰이게 되어 다른 작업자들 누구나 이곳을 열어 파악할 수 있도록 해 줍니다.

```bash
hermes kanban swarm "Design a multi-region failover plan" \
  --workers researcher,architect,sre \
  --verifier reviewer --synthesizer writer
```

결과물로 도출된 작업망(graph)은 평소처럼 구동됩니다 — 여러 작업자가 다 같이 움직이며 작업을 수행하고 나면, 검증자가 그 내용을 인계받아 검토한 뒤, 최종적으로 작성자가 작업을 넘겨받아 취합합니다.

## `/kanban` 슬래시 명령어 {#kanban-slash-command}

모든 `hermes kanban <action>` 명령어들은, 대화가 오가는 `hermes chat` 세션 안에서뿐만 아니라, 모든 메신저 게이트웨이 (Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Mattermost, email, SMS) 상에서도 `/kanban <action>` 형태로 쓰일 수 있습니다. CLI에서 쓰일 때나 이곳에 쓰일 때나 둘 다 동일한 인자 구조(argparse tree)를 바탕으로 `hermes_cli.kanban.run_slash()`라는 통로를 통해 일하게 되므로, 매개 변수나 옵션 태그, 그리고 출력 형태 또한 CLI와 슬래시 명령어 사이에 아무런 차이가 없습니다. 당신은 칸반 보드를 굴리기 위해 굳이 현재의 채팅창을 벗어나지 않아도 됩니다.

```
/kanban list
/kanban show t_abcd
/kanban create "write launch post" --assignee writer --parent t_research
/kanban comment t_abcd "looks good, ship it"
/kanban unblock t_abcd
/kanban dispatch --max 3
/kanban specify t_abcd                  # triage에 있는 단순 아이디어를 구체적인 명세서로 변환
/kanban specify --all --tenant engineering  # 한 테넌트의 모든 triage 태스크 일괄 처리
```

명령어 속 다중 어휘들의 처리는 쉘 스크립트를 작성할 때와 동일합니다 — `run_slash`는 `shlex.split` 방식을 이용하여 나머지 줄들을 파싱하므로, 쌍따옴표(`"..."`) 나 외따옴표(`'...'`) 둘 다 먹힙니다.

### 실행 도중의 사용 (Mid-run usage): `/kanban`은 실행 중인 에이전트의 방어벽을 우회합니다

당신이 내린 명령을 에이전트가 아직 한창 수행하고 있을 때 게이트웨이에서는 보통 슬래시 명령어나 새로운 메시지들을 대기열에 담아둡니다 — 이것이 원래 첫 번째 지시가 한창 돌아가는 도중 또 다른 엉뚱한 지시가 실행되지 않도록 막아주는 방식입니다. **`/kanban`의 명령어는 이 보호 기제를 완전히 면제받습니다.** 칸반의 활동 내용들은 실행 중인 해당 에이전트의 상태 공간 안이 아닌 별도의 `~/.hermes/kanban.db`에만 오로지 자리 잡고 있으므로, 읽기 작업(`list`, `show`, `context`, `tail`, `watch`, `stats`, `runs`)이든 쓰기 작업(`comment`, `unblock`, `block`, `assign`, `archive`, `create`, `link`, …)이든 턴(turn) 중간에 즉시 전송되어 처리됩니다.

이것이 분리의 핵심입니다:

- 어느 한 작업자가 동료의 작업이 끝나길 기다리며 막혀 있을 때 → 당신이 폰에서 `/kanban unblock t_abcd`를 입력하면, 디스패처가 다음 틱(tick)에 동료를 즉시 활성화시킵니다. 작업 중이던 작업자는 중단되는 게 아닙니다 — 단지 막혔던(blocked) 상태에서 벗어날 뿐입니다.
- 어떤 카드에 누군가의 맥락 설명이 필요할 것 같아 보일 때 → `/kanban comment t_xyz "use the 2026 schema, not 2025"`라고 태스크 스레드에 글을 남겨놓으면 *그 이후* 구동될 해당 작업자가 `kanban_show()`를 통해 이를 읽게 됩니다.
- 메인 대화를 중단하지 않고 전체적인 진행 상황을 둘러보고 싶을 때 → `/kanban list --mine` 혹은 `/kanban stats` 등을 이용해 메인 대화를 멈추지 않고 보드를 훑어볼 수 있습니다.

### `/kanban create` 시 자동 구독 알림 (Auto-subscribe on `/kanban create`) (게이트웨이 전용)

당신이 게이트웨이에서 `/kanban create "…"`를 이용하여 임무를 하나 띄웠을 때, 맨 처음 명령이 발생했던 발신 채팅창(플랫폼 + 채팅방 id + 스레드 id)은 그 새 태스크의 최종 종료 이벤트(`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`)를 자동으로 구독하게 됩니다. 이렇게 되면 나중에 굳이 `/kanban show`를 하거나 태스크의 id 값을 기억해내지 않더라도 알림 수신 때마다 결과 메시지의 첫 줄이 담긴 메시지를 전송받게 됩니다.

```
you> /kanban create "transcribe today's podcast" --assignee transcriber
bot> Created t_9fc1a3  (ready, assignee=transcriber)
     (subscribed — you'll be notified when t_9fc1a3 completes or blocks)

… ~8 minutes later …

bot> ✓ t_9fc1a3 completed by transcriber
     transcribed 42 minutes, saved to podcast/2026-05-04.md
```

태스크가 완전히 종료되어 `done` 혹은 `archived`가 되면 이 알림은 자동 해지됩니다. 기계가 직접 뽑아내게 하려고 스크립트 기반으로 `--json` 출력 명령을 썼을 때는 이 자동 구독 기능이 생략됩니다 — 이는 스크립트 호출자가 스스로 `/kanban notify-subscribe` 기능을 이용해 직접 구독 관리를 하고자 한다는 추정을 바탕으로 설계된 것입니다.

### 메시징 출력 잘림 (Output truncation in messaging)

게이트웨이 플랫폼에는 실용적인 메시지 길이 제한이 존재합니다. `/kanban list`, `/kanban show`, 또는 `/kanban tail`의 결과물이 대략 3800 글자를 초과할 시 `… (truncated; use \`hermes kanban …\` in your terminal for full output)` 이란 맺음말과 함께 결과 내용의 하단이 잘려나갑니다. 터미널상의 CLI에선 이런 제한이 없습니다.

### 자동완성 (Autocomplete)

대화형 CLI 환경에서는, `/kanban `을 입력 후 Tab 키를 누르면 시스템 안에 심겨있는 하위 명령어들(`list`, `ls`, `show`, `create`, `assign`, `link`, `unlink`, `claim`, `comment`, `complete`, `block`, `unblock`, `archive`, `tail`, `dispatch`, `context`, `init`, `gc`)이 계속해서 순환 표시됩니다. 위 항목에서 거론했던 나머지 기능들(`watch`, `stats`, `runs`, `log`, `assignees`, `heartbeat`, `notify-subscribe`, `notify-list`, `notify-unsubscribe`, `daemon`) 역시 완전히 똑같이 쓰이기는 하지만 — 그저 아직 자동완성 리스트에는 포함되지 않았을 뿐입니다.

## 협업 패턴 (Collaboration patterns)

이 보드는 그 어떤 추가 도구 없이도 8가지 패턴 구조를 소화해 냅니다:

| 패턴 (Pattern) | 형태 (Shape) | 예시 (Example) |
|---|---|---|
| **P1 다중 처리 (Fan-out)** | N개의 형제 작업, 같은 권한 | "5개 시각에 대한 자료를 병렬 조사" |
| **P2 직렬 연결 (Pipeline)** | 권한들의 꼬리 물기: 정보 탐색 → 편집 → 작성 | 일일 브리핑 모음 구성 |
| **P3 집단 결의 (Voting / quorum)** | N개의 형제 작업 + 1개의 판단자 | 3명의 조사자 → 1명의 검토자가 채택 |
| **P4 장기적 모니터링 (Long-running journal)** | 고정된 동일 프로필 + 공용 디렉토리 + 스케줄러(cron) | Obsidian 저장소 |
| **P5 인간 개입 (Human-in-the-loop)** | 작업자 차단(막힘) → 사용자의 해결(코멘트) → 차단 해제 | 모호한 상황의 결단 |
| **P6 `@맨션` 방식 (`@mention`)** | 글의 문맥에 기재된 일감 분배 | `@reviewer 여기 좀 확인해봐` |
| **P7 쓰레드 분리된 작업공간 (Thread-scoped workspace)** | 한정된 스레드에서만 쓰이는 `/kanban here` | 각 프로젝트별 게이트웨이 스레드 |
| **P8 대규모 함대 (Fleet farming)** | 1개의 프로필, N개의 피사체 | 50개의 소셜 계정 운영 |
| **P9 세분화 지시 (Triage specifier)** | 성긴 생각 → `triage`(분류대기) → `hermes kanban specify`를 통한 본문 구체화 → `todo`(할일) | "이 한 줄짜리 메모를 구체적인 사양서로 바꿔 놔라" |

각 과정들의 실제 사용 모델을 보려면 `docs/hermes-kanban-v1-spec.pdf`를 살펴보십시오.

## 멀티 테넌트 활용 (Multi-tenant usage)

수많은 기업의 업무를 한 명의 전문가가 다 소화해 내게 하려면 테넌트(tenant) 명찰을 달아 각각의 태스크를 표기하세요:

```bash
hermes kanban create "monthly report" \
    --assignee researcher \
    --tenant business-a \
    --workspace dir:~/tenants/business-a/data/
```

작업자는 이 정보들을 `$HERMES_TENANT` 형태로 받아 기록물마다 이름을 달아 관리하게 됩니다. 보드판, 디스패처, 프로필의 정보들은 그대로 모두 공용으로 쓰이며 데이터의 이름들만 별도로 구분됩니다.

## 게이트웨이 알림 (Gateway notifications)

메신저 게이트웨이 (Telegram, Discord, Slack 등)에서 `/kanban create …` 명령어를 이용하면 해당 챗방은 자연스레 알림 설정 대상이 됩니다. 게이트웨이의 백그라운드 알리미가 매 시간마다 `task_events`를 훑어 임무의 종말점(`completed`, `blocked`, `gave_up`, `crashed`, `timed_out`)에 다다른 이벤트 메시지 하나를 지정된 채팅창으로 넘겨줍니다. 완료된(completed) 임무의 경우에는 굳이 `/kanban show`를 할 필요가 없게 결과물의 요약본(`--result`) 중 가장 앞쪽 줄을 보냅니다.

이 기능은 스크립트나 스케줄러 프로그램이 발생지를 통하지 않고 외부의 메신저를 통해 바로 명령을 하달할 때 유용하므로, 다음의 CLI 커맨드로 이 알람 기능만 단독으로 조절할 수도 있습니다:

```bash
hermes kanban notify-subscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
hermes kanban notify-list
hermes kanban notify-unsubscribe t_abcd \
    --platform telegram --chat-id 12345678 --thread-id 7
```

알람 기능은 태스크가 종료되어 `done` 또는 `archived`가 될 시 자동으로 해지되므로 따로 지워줄 필요는 없습니다.

## 수행 이력 — 시도 1회당 1열 (Runs — one row per attempt)

하나의 임무(task)는 한 가지 일의 단위를 뜻하며, 하나의 **런(run)**은 해당 일을 한번 시도한 것을 말합니다. 디스패처가 준비된 임무를 하나 부여잡게 될 때마다 녀석은 `task_runs` 테이블에 열을 생성하고 이 값을 `tasks.current_run_id`에 엮습니다. 어떠한 사유로든 그 한 번의 시도가 끝나게 되면 — 성공, 차단, 붕괴, 시간 초과, 생성 실패, 회수 등 — 이 런(run) 데이터 값에 수행 `결과(outcome)`이 적히고 기존 작업(task)과의 연결이 해지됩니다. 세 번의 시도를 해본 임무는 세 개의 `task_runs` 데이터를 가지게 됩니다.

그저 태스크를 변경하는 것 대신 이렇게 두 가지의 테이블 값을 갖게 하는 데는 이유가 있습니다: "두 번째 리뷰어의 승인 절차를, 세 번째 검토자가 모두 다 통합시켜 끝내버렸다"와 같이 실질적으로 **전체적인 재시도 과정(full attempt history)**이 나중에 다 필요하게 될 때가 있고, 매번의 변경된 파일, 작동된 수십 가지의 테스트 횟수, 그에 따른 판단자의 지시 기록 등을 덧입혀 보관해 놓을 깨끗한 보관소가 필요하기 때문입니다. 이러한 메타데이터(metadata)는 한 가지의 임무(task) 그 자체의 진실이라기보다는 매번의 과정(run)들이 모여서 만들어진 팩트이기 때문입니다.

이 수행 이력 기능은 **조직화된 인계(structured handoff)** 과정이기도 합니다. `kanban_complete(...)`를 호출하여 일꾼의 한 가지의 태스크 수행이 마무리될 때 전달하는 정보는 다음과 같습니다:

- `summary` (도구 파라미터) / `--summary` (CLI 환경) — 다음 사람이 보기 편하도록 남기는 인간용 요약문; run 열에 남겨지게 되며 후행자들은 이 값을 자기의 `build_worker_context`에서 확인하게 됨.
- `metadata` (도구 파라미터) / `--metadata` (CLI 환경) — 형식 없이 자유로운 JSON 형식 사전 형태의 구문; 후행자들은 이것이 요약 내용과 한데 결합되어 나온 결과물로서 이를 확인하게 됨.
- `result` (도구 파라미터) / `--result` (CLI 환경) — 이전 환경 버전 호환을 위해 유지하는 태스크 열에 남겨지는 짧은 로그 기록.

다음 순서의 후임 작업자는 상위 작업들의 완료 기록들 중에서 가장 최근의 요약 및 메타데이터 정보들을 읽습니다. 오류로 인해 다시 작업을 재도전하게 된 일꾼들 역시 똑같은 길에서 또 실패하는 일이 없도록 자기 일의 이력서(과거 시도의 결과, 요약 내용, 에러)를 참고합니다.

```
# 작업자가 실제로 수행하는 것 — 에이전트 루프 내부에서 도구 호출 (tool call):
kanban_complete(
    summary="implemented token bucket, keys on user_id with IP fallback, all tests pass",
    metadata={"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14},
    result="rate limiter shipped",
)
```

만약 일꾼이 아닌 당신(사람)이 대시보드에서 완료 처리해 닫았거나 혹은 그냥 버려진 태스크를 정리하는 등 임무를 마감할 때 쓰이는 구조화된 처리 기능은 이와 같습니다:

```bash
hermes kanban complete t_abcd \
    --result "rate limiter shipped" \
    --summary "implemented token bucket, keys on user_id with IP fallback, all tests pass" \
    --metadata '{"changed_files": ["limiter.py", "tests/test_limiter.py"], "tests_run": 14}'

# 재시도된 작업의 전체 이력 보기:
hermes kanban runs t_abcd
#   #  OUTCOME       PROFILE           ELAPSED  STARTED
#   1  blocked       worker               12s  2026-04-27 14:02
#        → BLOCKED: need decision on rate-limit key
#   2  completed     worker                8m   2026-04-27 15:18
#        → implemented token bucket, keys on user_id with IP fallback
```

수행 과정(run)들은 대시보드(서랍의 Run History 파트에 여러 색상의 결과가 순차적으로 기록됨)와 REST API(`GET /api/plugins/kanban/tasks/:id`가 `runs[]` 배열 데이터를 반환함)로 나타납니다. `{status: "done", summary, metadata}` 형태를 갖는 `PATCH /api/plugins/kanban/tasks/:id` 역시 커널로 동시 전송되므로 대시보드의 '완료 처리' 버튼이나 터미널 환경상의 기능이나 다를 바 없습니다. `task_events` 데이터는 각자의 `run_id` 값을 가져 UI 화면이 이것들을 하나로 통합 분류할 수 있게 해 주고, 이렇게 묶여진 `completed` 내역에는 최초 수행한 첫 줄 요약 정보(400자 이내)가 담겨져 있어 또 다른 추가 SQL 명령 없이도 게이트웨이의 알람이 짜임새 있게 인계 요약문(handoff)을 날려줍니다.

**다중 삭제 경고 사항(Bulk close caveat).** `hermes kanban complete a b c --summary X` 명령은 듣지 않습니다 — 구조적인 기록의 이전은 매 시행 이력(per-run)마다 이루어져야 하므로 다수의 태스크에다 전부 X라고 하는 똑같은 문구를 똑같이 복사-붙여넣기 시키는 것은 무리가 있습니다. 다만 요약본`--summary`나 부수적인 `--metadata` 없이 일괄로 여러 행정적인 일을 치울 땐 유용합니다.

**작업 상태 변경에 따른 런의 귀환 (Reclaimed runs from status changes).** 대시보드의 화면에서 실행 중인 카드(`running`)를 강제로 다른 칸(`ready` 혹은 `todo`)으로 잡아서 끌어 놓거나, 돌아가던 중인 태스크를 그대로 아카이브(archive)시켜버렸다면 실행 중이던 이 기록의 꼬리표엔 `outcome='reclaimed'`라는 딱지가 붙은 상태로 종료되며 공중에 떠버리거나 버려지는 일은 막습니다. `tasks.current_run_id`이 `NULL` 값이 되었을 때 `task_runs` 데이터 열이 반드시 마감 상태에 이르는, 두 대상 간의 이러한 절대적 관계(invariant) 공식은 터미널, GUI 패널, 알람 등 그 어떤 창구를 통해 일어나는지와 상관없이 유지됩니다.

**요청도 없이 끝마쳐진 태스크에 대한 통합 데이터 (Synthetic runs for never-claimed completions).** `ready` 대기 상황에서 누가 잡기도 전에 당신이 대시보드 화면상에서 완료 버튼을 누르거나 아니면 `hermes kanban complete <ready-task> --summary X` 와 같은 명령어 등을 치면 데이터 인계 작업(handoff)이 사라지는 일이 벌어질 수 있습니다. 이런 경우 커널 프로그램이 내부적으로 수행 시기가 0인, 마치 `started_at == ended_at`과 같은 임의의 수행 과정(run) 데이터를 끼워 넣어서 그 자리에 당신의 summary / metadata / reason 값을 담습니다. `completed` / `blocked` 상태 변경 때 생겨나는 알림 역시 그 임의로 세워진 값을 향해 맞춰지게 됩니다.

**화면 실시간 반영 업데이트 기능 (Live drawer refresh).** 대시보드가 감시하고 있던 데이터들 중 어떤 변화 이벤트가 사용자가 바라보던 웹소켓에 감지되면, 서랍 기능의 구조가 내부적으로 재시작됩니다(`useEffect` 연동된 각각의 태스크 이벤트 개수들의 변경을 파악하여). 이제 더 이상 새 결과나 추가 정보를 보기 위해 창을 닫고 다시 열 필요 없이 편히 보세요.

### 이전 버전의 상위 호환성 (Forward compatibility)

향후 v2에서의 동작 라우팅 기능을 위해 빈 곳으로 남겨져 있는 열(column)이 두 가지 있습니다: 어느 템플릿의 작업을 진행 중인지를 다루는 `workflow_template_id`, 그 템플릿 안에서 현재 활성화된 키가 어떤 것인지 파악하기 위한 `current_step_key`. v1 구동 상황에선 라우팅 작업 수행에는 배제되지만 유저가 쓰는 것은 막지 않으므로 나중에 v2 업데이트 배포 시에 추가적인 양식 이동 절차 없이 곧바로 새 기능을 안착시킬 수 있게 도와줍니다.

## 이벤트 참조 (Event reference)

상태가 변할 때마다 하나의 데이터 줄이 생겨나 `task_events` 꼬리칸에 합류하게 됩니다. 이 각각의 데이터 꼬리표는 `run_id` 값을 갖게 되므로 프로그램 화면(UI)들이 이들을 통합 분류해 줄 수 있습니다. 종류는 크게 3가지 부류로 묶이므로 쉽게 걸러볼 수 있습니다 (`hermes kanban watch --kinds completed,gave_up,timed_out`):

**라이프사이클(Lifecycle)** (태스크를 하나의 기능으로 볼 때 변경된 내역):

| 유형 (Kind) | 내용물 (Payload) | 언제 (When) |
|---|---|---|
| `created` | `{assignee, status, parents, tenant}` | 태스크가 등록됨. `run_id` 값은 `NULL`. |
| `promoted` | — | 모든 선행 조건(parents)이 마무리 되어 `todo → ready`로 진행 승격. `run_id` 값은 `NULL`. |
| `claimed` | `{lock, expires, run_id}` | 디스패처가 준비 상태(`ready`)의 작업을 수행 대상으로 등록함. |
| `completed` | `{result_len, summary?}` | 작업자가 `--result` / `--summary`를 적고 `done` 상태로 임무를 완료함. 이 `summary` 영역은 초반 요약부로서 (400자 이내) 전체 과정 중 일부분에 불과합니다. 작업 이력의 전문 내용은 run 영역에 살아있습니다. 어떤 누구도 착수하지 않았던 임무에 넘길 값(handoff fields)들이 생겨나 종료(`complete_task`) 상황에 다다르면 소요 시간 0 상태의 런 기능 하나를 임의로 빚어내어 빈 곳을 보충해 냅니다. |
| `blocked` | `{reason}` | 사람, 일꾼 어느 한쪽이 차단을 지시하여 `blocked` 상태가 되었음. 이 역시 빈 곳에 사유값(`--reason`)을 넘겨받아 종료 상황이 발생하면 소요 시간 0 상태의 런을 짜 맞춥니다. |
| `unblocked` | — | 손으로 직접, 혹은 `/unblock` 명령을 받아 `blocked → ready` 됨. `run_id` 값은 `NULL`. |
| `archived` | — | 기본 보드판의 시야 밖으로 가려짐. 아직 수행 중이었던 임무를 가렸다면, 회수(reclaimed) 조치 당해버린 그 런의 `run_id`를 담음. |

**에디트(Edits)** (전환이 아닌 인간이 내린 조작들):

| 유형 (Kind) | 내용물 (Payload) | 언제 (When) |
|---|---|---|
| `assigned` | `{assignee}` | 할당자가 (해고당했거나) 다른 이로 교체. |
| `edited` | `{fields}` | 제목이나 본문 내용이 바뀌었음. |
| `reprioritized` | `{priority}` | 중요도 수치가 바뀌었음. |
| `status` | `{status}` | 화면의 드래그를 이용해 바로 상태를 덮어썼을 때(가령 `todo → ready`). 돌고 있던 작업을 끌어서 취소(`running`에서 벗어남)했을 경우 함께 버려진(reclaimed) `run_id` 값을 갖게 됨; 그렇지 않다면 `run_id`는 NULL. |

**작업자 관측 통계 기록(Worker telemetry)** (임무 그 자체의 논리적 분석이 아니라 실행 과정들에 초점이 맞춰진 기록):

| 유형 (Kind) | 내용물 (Payload) | 언제 (When) |
|---|---|---|
| `spawned` | `{pid}` | 디스패처가 성공적으로 한 일꾼의 작업을 개시시킴. |
| `heartbeat` | `{note?}` | 임무가 너무 오래 걸릴 때 일꾼이 `hermes kanban heartbeat $TASK` 호출 기능을 통해 본인 생존 신고를 해놓음. |
| `reclaimed` | `{stale_lock}` | 할당된 생명(Claim TTL)이 완료를 못 짓고 끝났으므로 태스크를 다시금 `ready` 대기로 되돌려 놓음. |
| `crashed` | `{pid, claimer}` | 일꾼의 PID 생명 정보가 죽어버렸지만 본래의 생명 기한(TTL)엔 다다르지 못했음. |
| `timed_out` | `{pid, elapsed_seconds, limit_seconds, sigkill}` | 주어졌던 `max_runtime_seconds`가 다 소모됨; 디스패처가 SIGTERM 종료 지시를 날리고(5초 뒤엔 SIGKILL)는 대기줄로 다시 넣음. |
| `stale` | `{elapsed_seconds, last_heartbeat_at, heartbeat_age_seconds, timeout_seconds, pid, terminated}` | 임무가 정해져 있던 `kanban.dispatch_stale_timeout_seconds` 상한치(기본값: 4시간)를 넘어버림과 동시에 1시간 이상 `kanban_heartbeat` 반응조차 오지 않았음. 디스패처는 해당 기계에 속한 일꾼에 대해 SIGTERM 종료 명령을 내리고, 재배치 과정 진입을 위해 다시 `ready`로 임무 상태를 세탁함. 실패 카운트를 소모시키지 않음(이 기능은 디스패처의 실종 분석 측면이며, 작업자가 결점을 낸 것이 아님). 아주 긴 일을 해내는 일꾼은 1시간에 한 번 정도는 살았다는 알람을 지시하여 본 기능을 모면해야 함. |
| `respawn_guarded` | `{reason}` | 이번 차례에 디스패처가 어떤 태스크의 재시작을 불허했음. 사유(Reasons): `blocker_auth` (이전 임무가 할당량/인증/429 문제로 깨짐 — 해당 쿼터가 부활할 때까지 휴식), `recent_success` (최근 1시간 내에 성공한 기록이 존재 — 다시 반복 수행하기 전 수검 기간 갖기), `active_pr` (가장 최근 등록 코멘트에 GitHub 깃허브 풀리퀘가 걸려있음 — 다른 동료가 이미 제출을 함). 해당 임무는 여전히 `ready` 상태를 유지; 다음 사이클이 돌아올 때 또다시 생성 찬스를 갖게 됨. 기저에 깔린 저 증상들이 풀리지 않는 한, 서킷 브레이크 제어기에 의해 결국 `failure_limit`를 다 쓰고 `gave_up` 포기 및 자동 차단 조치가 이루어질 것임. |
| `spawn_failed` | `{error, failures}` | 작업 하나가 생겨나는 데에 실패했음(마운트 불가 에러, 지정된 자리를 못 찾음 등). 횟수 하나가 올라가며 작업은 다시 시도를 기다리며 대기실(`ready`)로 감. |
| `protocol_violation` | `{pid, claimer, exit_code}` | 임무가 여전히 굴러가고(`running`) 있는데 작업자는 일을 마치고 나가버린 상태, 대개 `kanban_complete`나 `kanban_block` 어느 한쪽의 처리 없이 답변만을 덜렁 적고는 내뺐을 때. 디스패처는 이 일에 대해서 `gave_up` 딱지와 함께 즉각적인 차단을 먹이며 대기열 재시도 조치도 취하지 않음. |
| `gave_up` | `{failures, effective_limit, limit_source, error}` | 불성실한 실패들이 겹치며 서킷 브레이커가 작동했음. 마지막 에러 원인을 품은 채 태스크 자체가 자동 봉쇄당함(auto-blocks). 상한선 적용은 먼저 임무 자체 설정의 `max_retries` 값, 그다음 디스패처 상의 `failure_limit` / `kanban.failure_limit` 값, 그리고 그마저도 없다면 기본 내장 값의 순서대로 정해짐. |

특정 임무 하나의 이벤트 과정만 보고 싶을 때는 `hermes kanban tail <id>` 옵션을 이용하세요. 보드 위의 모든 과정을 한꺼번에 관찰하고 싶다면 `hermes kanban watch`를 켜놓으면 됩니다.

## 적용 범위 외 (Out of scope)

칸반은 고의적으로 단일 장치(single-host) 운용 목적으로 설계되었습니다. `~/.hermes/kanban.db`은 내 컴퓨터 속 하나의 SQLite 로컬 파일이고 녀석을 감시하는 디스패처 역시 내가 쓰는 이 PC 속에서만 새로운 일꾼들을 창조합니다. 물리적으로 다른 공간에 있는 두 기기가 이 하나의 보드 내용을 공유하고 쓸 수 있도록 하는 방안(A 컴퓨터 속 X와 B 컴퓨터 속 Y)에 대한 기능은 제공하지 않으며, 죽어버린 PID 정보를 감별해 내는 과정 역시 같은 기계 속에 들어있을 때만 가능합니다. 필연적으로 복수의 시스템 환경을 구축하여 일하고자 하신다면 개별 장비마다 하나씩의 보드를 따로 운용케 하고 메인 `delegate_task` 기능이 메시지를 주고받는 방식을 사용하게 하여 서로를 이어주는 편이 좋습니다.

## 디자인 설계서 (Design spec)

설계 아키텍처, 병렬 실행에 관한 문제와 타 시스템에 대한 분석, 구현 방법, 예상 리스크와 해결되지 않은 난제들에 이르기까지, 보다 상세하고 전체적인 개념의 설계도 문서는 `docs/hermes-kanban-v1-spec.pdf`에서 열람하실 수 있습니다. 시스템의 행동 양식의 변경과 관련된 어떠한 작업(PR)을 건의하시기 전 이 문서를 한 번씩 참조해주시기를 바랍니다.

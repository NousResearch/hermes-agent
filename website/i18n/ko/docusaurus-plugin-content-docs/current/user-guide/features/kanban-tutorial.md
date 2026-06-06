---
title: "칸반 튜토리얼 (Kanban tutorial)"
description: "Hermes 칸반 시스템의 4가지 주요 사용 사례 튜토리얼"
---

# 칸반 튜토리얼 (Kanban tutorial)

이 문서는 브라우저에 대시보드를 띄워놓고 진행하는, Hermes 칸반 시스템이 설계된 네 가지 주요 사용 사례에 대한 설명서입니다. 아직 [칸반 개요 (Kanban overview)](./kanban)를 읽지 않았다면 그곳부터 시작하세요 — 이 문서는 당신이 작업(task), 실행(run), 담당자(assignee), 디스패처(dispatcher)가 무엇인지 알고 있다고 가정합니다.

## 설정 (Setup)

```bash
hermes kanban init           # 선택 사항; 첫 번째 `hermes kanban <명령어>` 실행 시 자동 초기화됨
hermes dashboard             # 브라우저에서 http://127.0.0.1:9119 열기
# 왼쪽 탐색 메뉴에서 Kanban 클릭
```

대시보드는 **사용자**가 시스템을 관찰하기에 가장 편안한 장소입니다. 디스패처가 생성한 에이전트 작업자는 대시보드나 CLI를 보지 못하며, 오직 전용 `kanban_*` [도구 세트](./kanban#how-workers-interact-with-the-board)(`kanban_show`, `kanban_list`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`, `kanban_unblock`)를 통해서만 보드를 구동합니다. 세 가지 표면 — 대시보드, CLI, 작업자 도구 — 은 모두 동일한 보드별 SQLite DB(`default` 보드의 경우 `~/.hermes/kanban.db`, 나중에 생성하는 보드의 경우 `~/.hermes/kanban/boards/<slug>/kanban.db`)를 통과하므로 어느 경로에서 변경 사항이 발생하든 각 보드는 일관성을 유지합니다.

이 튜토리얼 전체에서는 `default` 보드를 사용합니다. 여러 개의 격리된 대기열(프로젝트 / 저장소 / 도메인당 하나씩)이 필요한 경우 개요의 [보드 (다중 프로젝트)](./kanban#boards-multi-project)를 참조하세요 — 동일한 CLI / 대시보드 / 작업자 흐름이 보드별로 적용되며, 작업자는 물리적으로 다른 보드의 작업을 볼 수 없습니다.

튜토리얼 전반에 걸쳐 **`bash`로 레이블이 지정된 코드 블록은 *사용자*가 실행하는 명령입니다.** `# worker tool calls`로 레이블이 지정된 코드 블록은 생성된 작업자의 모델이 도구 호출로 내보내는 내용입니다 — 사용자가 직접 실행할 일이 있어서가 아니라 루프를 처음부터 끝까지 보여주기 위해 여기에 표시됩니다.

## 보드 한눈에 보기 (The board at a glance)

![칸반 보드 개요](/img/kanban-tutorial/01-board-overview.png)

왼쪽에서 오른쪽으로 6개의 열이 있습니다:

- **Triage (분류 대기)** — 다듬어지지 않은 아이디어. 기본적으로 디스패처는 이곳의 작업에 대해 **분해기(decomposer)** (오케스트레이터 기반의 팬아웃)를 자동 실행합니다: 프로필 목록과 설명을 읽고 원본 작업을 부모로 유지한 채 가장 적합한 전문가에게 라우팅된 자식 작업 그래프를 생성합니다. 이를 통해 모든 하위 작업이 완료되면 오케스트레이터가 다시 깨어나 완료를 판단할 수 있습니다. 칸반 페이지 상단의 **Orchestration: Auto/Manual** 버튼을 눌러 모드를 전환하세요. 수동 모드(또는 오케스트레이터 프로필이 없는 설정)에서는 카드에서 **⚗ Decompose**를 클릭하거나 `hermes kanban decompose <id>` / `/kanban decompose <id>`를 실행하세요. 팬아웃이 필요 없는 단일 작업의 경우, **✨ Specify**가 단발성 명세 재작성(목표, 접근 방식, 수용 기준)을 수행하고 `todo`로 승격시킵니다. 모델은 `config.yaml`의 `auxiliary.kanban_decomposer` 및 `auxiliary.triage_specifier`에서 구성합니다. 메인 칸반 가이드의 [자동 및 수동 오케스트레이션 (Auto vs Manual orchestration)](./kanban#auto-vs-manual-orchestration)을 참조하세요.
- **Todo (할 일)** — 생성되었지만 종속성(dependencies)을 기다리고 있거나 아직 할당되지 않은 작업.
- **Ready (준비됨)** — 할당되어 디스패처가 가져가기를 기다리는 작업.
- **In progress (진행 중)** — 작업자가 작업을 활발하게 실행 중인 상태. "Lanes by profile(프로필별 레인)"이 켜져 있으면(기본값) 이 열은 담당자별로 하위 그룹화되어 각 작업자가 무엇을 하고 있는지 한눈에 볼 수 있습니다.
- **Blocked (차단됨)** — 작업자가 사람의 입력을 요청했거나 회로 차단기(circuit breaker)가 작동한 상태.
- **Done (완료됨)** — 완료됨.

상단 바에는 검색, 테넌트, 담당자를 위한 필터와 `Lanes by profile` 토글, 그리고 데몬의 다음 주기를 기다리지 않고 바로 한 번의 배포 틱(dispatch tick)을 실행하는 `Nudge dispatcher` 버튼이 있습니다. 카드를 클릭하면 오른쪽에 서랍이 열립니다.

### 평면 뷰 (Flat view)

프로필 레인이 너무 복잡하다면 "Lanes by profile" 토글을 끄세요. 그러면 In Progress 열이 선택된(claim) 시간순으로 정렬된 단일 평면 목록으로 축소됩니다:

![프로필별 레인이 꺼진 보드](/img/kanban-tutorial/02-board-flat.png)

## 스토리 1 — 솔로 개발자의 기능 배포 (Solo dev shipping a feature)

기능을 구축하고 있습니다. 전형적인 흐름: 스키마를 설계하고, API를 구현하고, 테스트를 작성합니다. 부모→자식 종속성이 있는 세 개의 작업입니다.

```bash
SCHEMA=$(hermes kanban create "Design auth schema" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --body "Design the user/session/token schema for the auth module." \
    --json | jq -r .id)

API=$(hermes kanban create "Implement auth API endpoints" \
    --assignee backend-dev --tenant auth-project --priority 2 \
    --parent $SCHEMA \
    --body "POST /register, POST /login, POST /refresh, POST /logout." \
    --json | jq -r .id)

hermes kanban create "Write auth integration tests" \
    --assignee qa-dev --tenant auth-project --priority 2 \
    --parent $API \
    --body "Cover happy path, wrong password, expired token, concurrent refresh."
```

`API`는 `SCHEMA`를 부모로 갖고, `tests`는 `API`를 부모로 갖기 때문에 오직 `SCHEMA`만 `ready` 상태로 시작합니다. 나머지 두 개는 부모가 완료될 때까지 `todo` 상태에 머무릅니다. 이것이 종속성 승격 엔진이 하는 일입니다 — 테스트할 API가 생길 때까지 어떤 작업자도 테스트 작성을 시작하지 않습니다.

다음 디스패처 틱(기본값 60초, 또는 **Nudge dispatcher**를 누르면 즉시)에 `backend-dev` 프로필이 환경 변수에 `HERMES_KANBAN_TASK=$SCHEMA`를 가진 작업자로 생성됩니다. 다음은 에이전트 내부에서 본 작업자의 도구 호출 루프입니다:

```python
# 작업자 도구 호출 — 사용자가 실행하는 명령이 아닙니다
kanban_show()
# → 제목, 본문, worker_context, 부모, 이전 시도, 댓글을 반환합니다

# (작업자는 worker_context를 읽고, terminal/file 도구를 사용하여 스키마를 설계하고,
#  마이그레이션을 작성하고, 자체 검사를 실행하고, 커밋합니다 — 실제 작업이 여기서 일어납니다)

kanban_heartbeat(note="schema drafted, writing migrations now")

kanban_complete(
    summary="users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); "
            "refresh tokens stored as sessions with type='refresh'",
    metadata={
        "changed_files": ["migrations/001_users.sql", "migrations/002_sessions.sql"],
        "decisions": ["bcrypt for hashing", "JWT for session tokens",
                      "7-day refresh, 15-min access"],
    },
)
```

`kanban_show`는 `task_id`를 기본적으로 `$HERMES_KANBAN_TASK`로 설정하므로 작업자는 자신의 ID를 알 필요가 없습니다. `kanban_complete`는 요약 + 메타데이터를 현재 `task_runs` 행에 기록하고, 해당 실행을 종료하며, 작업을 `done`으로 전환합니다 — 이 모든 것이 `kanban_db`를 통한 원자적(atomic) 한 번의 단계로 이루어집니다.

`SCHEMA`가 `done`이 되면 종속성 엔진은 `API`를 `ready`로 자동 승격시킵니다. API 작업자는 실행을 시작할 때 `kanban_show()`를 호출하고 부모 인계 내용에 첨부된 `SCHEMA`의 요약과 메타데이터를 확인합니다 — 따라서 긴 설계 문서를 다시 읽지 않고도 스키마 결정을 알 수 있습니다.

보드에서 완료된 스키마 작업을 클릭하면 서랍에 모든 것이 표시됩니다:

![솔로 개발자 — 완료된 스키마 작업 서랍](/img/kanban-tutorial/03-drawer-schema-task.png)

맨 아래의 **실행 기록(Run History)** 섹션이 핵심 추가 사항입니다. 하나의 시도: 결과 `completed`, 작업자 `@backend-dev`, 소요 시간, 타임스탬프, 그리고 전체 인계 요약문입니다. 메타데이터 덩어리(`changed_files`, `decisions`)도 실행 기록에 저장되며 이 부모를 읽는 모든 하위 작업자에게 표출됩니다.

터미널에서 언제든지 동일한 데이터를 검사할 수 있습니다 — 이 명령들은 작업자가 아니라 **사용자**가 보드를 엿보는 것입니다:

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## 스토리 2 — 대규모 함대 운용 (Fleet farming)

세 명의 작업자(번역가, 전사자(transcriber), 카피라이터)와 독립적인 작업 더미가 있습니다. 세 명 모두 병렬로 작업을 끌어와 눈에 띄는 진전을 보이기를 원합니다. 이것은 가장 간단한 칸반 사용 사례이자 원래 디자인이 최적화된 사례입니다.

작업을 생성합니다:

```bash
for lang in Spanish French German; do
    hermes kanban create "Translate homepage to $lang" \
        --assignee translator --tenant content-ops
done
for i in 1 2 3 4 5; do
    hermes kanban create "Transcribe Q3 customer call #$i" \
        --assignee transcriber --tenant content-ops
done
for sku in 1001 1002 1003 1004; do
    hermes kanban create "Generate product description: SKU-$sku" \
        --assignee copywriter --tenant content-ops
done
```

게이트웨이를 시작하고 자리를 비웁니다 — 동일한 kanban.db에서 세 가지 전문가 프로필의 작업을 모두 가져오는 내장 디스패처를 호스팅합니다:

```bash
hermes gateway start
```

이제 보드를 `content-ops`로 필터링하거나 "Transcribe"로 검색하면 다음과 같이 나타납니다:

![전사(transcribe) 작업으로 필터링된 함대 보기](/img/kanban-tutorial/07-fleet-transcribes.png)

두 개의 전사가 완료되었고, 하나는 실행 중이며, 두 개는 다음 디스패처 틱을 기다리며 준비 상태에 있습니다. In Progress 열은 프로필별로 그룹화되어 있으므로(기본값인 "Lanes by profile") 섞인 목록을 훑어보지 않고도 각 작업자의 활성 작업을 볼 수 있습니다. 디스패처는 현재 작업이 완료되는 즉시 다음 준비된 작업을 진행 중으로 승격시킵니다. 3개의 담당자 풀에서 3개의 데몬이 병렬로 작동하므로 인간의 추가 개입 없이 전체 콘텐츠 대기열이 비워집니다.

**스토리 1에서 언급된 구조화된 인계(structured handoff)에 관한 모든 내용이 여기에도 적용됩니다.** 통화를 완료한 번역가 작업자는 `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})`를 발생시킵니다 — 분석 및 이 작업에 의존하는 모든 다운스트림 작업에 유용합니다.

## 스토리 3 — 재시도를 포함한 역할 파이프라인 (Role pipeline with retry)

이것이 평범한 TODO 목록에 비해 칸반이 제값을 하는 곳입니다. PM이 명세를 작성합니다. 엔지니어가 구현합니다. 리뷰어가 첫 번째 시도를 거부합니다. 엔지니어가 변경 사항을 적용하여 다시 시도합니다. 리뷰어가 승인합니다.

`auth-project`로 필터링된 대시보드 보기:

![다중 역할 기능을 위한 파이프라인 보기](/img/kanban-tutorial/08-pipeline-auth.png)

3단계 체인이 한 번에 표시됩니다: `Spec: password reset flow` (완료, pm), `Implement password reset flow` (완료, backend-dev), `Review password reset PR` (준비, reviewer). 각각 하단에 녹색으로 된 부모가 있고 자식이 종속성으로 표시됩니다.

흥미로운 것은 구현 작업입니다. 차단되었다가 다시 시도되었기 때문입니다. 다음은 각 작업자의 모델이 수행하는 도구 호출로 표시된 전체 세 에이전트의 움직임입니다:

```python
# --- PM 작업자가 $SPEC에 생성되어 수용 기준(acceptance criteria)을 작성합니다 ---
# 작업자 도구 호출
kanban_show()
kanban_complete(
    summary="spec approved; POST /forgot-password sends email, "
            "GET /reset/:token renders form, POST /reset applies new password",
    metadata={"acceptance": [
        "expired token returns 410",
        "reused last-3 password returns 400 with message",
        "successful reset invalidates all active sessions",
    ]},
)
# → $SPEC이 완료됩니다; $IMPL이 todo에서 ready로 자동 승격됩니다

# --- 엔지니어 작업자가 $IMPL에 생성됩니다 (첫 번째 시도) ---
# 작업자 도구 호출
kanban_show()   # worker_context에서 $SPEC의 요약 + 수용 메타데이터를 읽습니다
# (엔지니어가 코드를 작성하고, 테스트를 실행하고, PR을 엽니다)
# 리뷰어의 피드백이 도착합니다 — 엔지니어는 우려 사항이 타당하다고 판단하고 차단합니다
kanban_block(
    reason="Review: password strength check missing, reset link isn't "
           "single-use (can be replayed within 30min)",
)
# → $IMPL이 blocked로 전환됩니다; 실행 1은 outcome='blocked'로 종료됩니다
```

이제 사용자(사람, 또는 별도의 리뷰어 프로필)가 차단 이유를 읽고, 수정 방향이 명확하다고 판단하여 대시보드의 "Unblock" 버튼이나 CLI / 슬래시 명령어를 통해 차단을 해제합니다:

```bash
hermes kanban unblock $IMPL
# 또는 채팅창에서: /kanban unblock $IMPL
```

디스패처는 `$IMPL`을 다시 `ready`로 승격시키고 다음 틱에 `backend-dev` 작업자를 재생성합니다. 이 두 번째 생성은 동일한 작업에 대한 **새로운 실행(new run)**입니다:

```python
# --- 엔지니어 작업자가 $IMPL에 생성됩니다 (두 번째 시도) ---
# 작업자 도구 호출
kanban_show()
# → worker_context는 이제 실행 1의 차단 이유를 포함하므로, 이 작업자는 전체 명세를 다시
#   읽는 대신 어떤 두 가지를 수정해야 하는지 알고 있습니다.
# (엔지니어는 zxcvbn 검사를 추가하고, 재설정 토큰을 일회용으로 만들고, 테스트를 다시 실행합니다)
kanban_complete(
    summary="added zxcvbn strength check, reset tokens are now single-use "
            "(stored + deleted on success)",
    metadata={
        "changed_files": [
            "auth/reset.py",
            "auth/tests/test_reset.py",
            "migrations/003_single_use_reset_tokens.sql",
        ],
        "tests_run": 11,
        "review_iteration": 2,
    },
)
```

구현 작업을 클릭하세요. 서랍에 **두 번의 시도**가 표시됩니다:

![차단되었다가 완료된 두 번의 실행이 있는 구현 작업](/img/kanban-tutorial/04b-drawer-retry-history-scrolled.png)

- **Run 1** — `@backend-dev`에 의해 `blocked`. 결과 바로 아래에 리뷰 피드백이 있습니다: "password strength check missing, reset link isn't single-use (can be replayed within 30min)".
- **Run 2** — `@backend-dev`에 의해 `completed`. 새로운 요약, 새로운 메타데이터.

각 실행은 고유한 결과, 요약 및 메타데이터를 가진 `task_runs`의 한 행입니다. 재시도 기록은 "최신 상태" 작업 위에 쌓아 올려진 개념적인 사후 고려 사항이 아니라 기본 표현입니다. 재시도 중인 작업자가 작업을 열 때 `build_worker_context`는 이전 시도를 보여주므로, 두 번째 작업을 수행하는 작업자는 첫 번째 작업이 차단된 이유를 확인하고 처음부터 다시 실행하는 대신 해당 특정 발견 사항을 해결합니다.

리뷰어가 다음을 맡습니다. `Review password reset PR`을 열면 다음을 볼 수 있습니다:

![파이프라인에 대한 리뷰어의 서랍 보기](/img/kanban-tutorial/09-drawer-pipeline-review.png)

부모 링크는 완료된 구현입니다. 리뷰어의 작업자가 `Review password reset PR`에 생성되어 `kanban_show()`를 호출할 때 반환되는 `worker_context`에는 부모의 가장 최근에 완료된 실행의 요약 + 메타데이터가 포함됩니다 — 따라서 리뷰어는 "added zxcvbn strength check, reset tokens are now single-use"를 읽고 diff를 보기 전에 변경된 파일 목록을 미리 파악합니다.

## 스토리 4 — 회로 차단기 및 충돌 복구 (Circuit breaker and crash recovery)

실제 작업자는 실패합니다. 누락된 자격 증명, OOM 종료(Out-Of-Memory kills), 일시적인 네트워크 오류. 디스패처에는 두 가지 방어선이 있습니다: 보드가 영원히 요동치지 않도록 연속적으로 N번 실패한 후 자동 차단하는 **회로 차단기(circuit breaker)**와, TTL이 만료되기 전에 작업자 PID가 사라진 작업을 회수하는 **충돌 감지(crash detection)**입니다.

### 회로 차단기 — 영구적으로 보이는 실패 (Circuit breaker — permanent-looking failure)

프로필 환경에 `AWS_ACCESS_KEY_ID`가 설정되어 있지 않아 작업자를 생성할 수 없는 배포 작업입니다:

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops \
    --max-retries 3
```

디스패처가 작업자 생성을 시도합니다. 생성이 실패합니다 (`RuntimeError: AWS_ACCESS_KEY_ID not set`). 디스패처는 클레임(claim)을 해제하고, 실패 카운터를 증가시키며, 다음 틱에 다시 시도합니다. 이 예제에서는 `--max-retries 3`을 설정했기 때문에 세 번 연속 실패하면 회로가 차단됩니다: 작업은 `gave_up` 결과와 함께 `blocked` 상태가 됩니다. 플래그를 생략하면 Hermes는 `kanban.failure_limit` (기본값: 2)를 사용합니다. 사람이 차단을 해제할 때까지 더 이상 재시도하지 않습니다.

차단된 작업을 클릭하세요:

![회로 차단기 — 2번의 spawn_failed + 1번의 gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

세 번의 실행 모두 `error` 필드에 동일한 오류가 있습니다. 처음 두 번은 `spawn_failed`(재시도 가능)이고 세 번째는 `gave_up`(단말)입니다. 위의 이벤트 로그에는 전체 시퀀스가 표시됩니다: `created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`.

터미널에서:

```bash
hermes kanban runs t_ef5d
# #   OUTCOME        PROFILE        ELAPSED  STARTED
# 1   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 2   spawn_failed   deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
# 3   gave_up        deploy-bot          0s  2026-04-27 19:34
#       ! AWS_ACCESS_KEY_ID not set in deploy-bot env
```

Telegram / Discord / Slack이 연결되어 있는 경우 `gave_up` 이벤트 발생 시 게이트웨이 알림이 실행되므로 보드를 확인하지 않고도 중단 사실을 알 수 있습니다.

### 충돌 복구 — 실행 도중 작업자 사망 (Crash recovery — worker dies mid-flight)

때로는 생성이 성공하지만 작업자 프로세스가 나중에 종료되기도 합니다 — 세그폴트(segfault), OOM, `systemctl stop` 등. 디스패처는 `kill(pid, 0)`을 폴링하여 죽은 pid를 감지합니다. 클레임이 해제되고, 작업은 다시 `ready`로 돌아가며, 다음 틱에 새로운 작업자에게 작업이 주어집니다.

시드 데이터의 예는 메모리가 부족했던 마이그레이션입니다:

```bash
# 작업자가 가져와서 240만 행을 스캔하기 시작하고, 약 230만 행에서 OOM 종료가 발생합니다
# 디스패처가 죽은 pid를 감지하고, 클레임을 해제하며, 시도 카운터를 증가시킵니다
# 청크(chunked) 전략을 사용한 재시도가 성공합니다
```

서랍에 전체 두 번의 시도 기록이 표시됩니다:

![충돌 및 복구 — 1번의 crashed + 1번의 completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

Run 1 — `crashed`, 오류 내용 `OOM kill at row 2.3M (process 99999 gone)`. Run 2 — `completed`, 메타데이터에 `"strategy": "chunked with LIMIT + WHERE id > last_id"` 포함. 재시도 중인 작업자는 자신의 컨텍스트에서 실행 1의 충돌을 확인하고 더 안전한 전략을 선택했습니다. 이 메타데이터는 미래의 관찰자(또는 사후 보고서 작성자)가 무엇이 변경되었는지 명확하게 알 수 있도록 해줍니다.

## 구조화된 인계 — `summary`와 `metadata`가 중요한 이유 (Structured handoff)

위의 모든 스토리에서 작업자는 마지막에 `kanban_complete(summary=..., metadata=...)`를 호출했습니다. 이는 단순한 장식이 아니라 워크플로우 단계 간의 주요 인계 채널입니다.

작업 B의 작업자가 생성되어 `kanban_show()`를 호출할 때 반환되는 `worker_context`에는 다음이 포함됩니다:

- B의 **이전 시도** (이전 실행: 결과, 요약, 오류, 메타데이터) — 재시도 중인 작업자가 실패한 경로를 반복하지 않도록 합니다.
- **부모 작업 결과** — 각 부모에 대해 가장 최근에 완료된 실행의 요약 및 메타데이터 — 다운스트림 작업자가 업스트림 작업이 왜, 어떻게 수행되었는지 확인할 수 있도록 합니다.

이는 평면적인 칸반 시스템을 괴롭히는 "댓글과 작업 결과물을 뒤적거리는" 과정을 대체합니다. PM은 명세의 메타데이터에 수용 기준을 작성하고, 엔지니어의 작업자는 부모 인계에서 구조적으로 이를 확인합니다. 엔지니어는 실행한 테스트와 통과한 수를 기록하고, 리뷰어의 작업자는 diff를 열기 전에 그 목록을 손에 넣습니다.

일괄 닫기 보호 장치가 존재하는 이유는 이 데이터가 각 실행마다 고유하기 때문입니다. `hermes kanban complete a b c --summary X` (CLI에서 사용자가 실행)는 거부됩니다 — 세 개의 작업에 동일한 요약을 복사하여 붙여넣는 것은 거의 항상 잘못된 것이기 때문입니다. 인계 플래그가 없는 일괄 닫기는 일반적인 "밀린 관리 작업을 마쳤다"는 경우에는 여전히 작동합니다. 도구 인터페이스는 일괄 변형을 전혀 노출하지 않습니다. 같은 이유로 `kanban_complete`는 항상 한 번에 단일 작업에만 수행됩니다.

## 현재 실행 중인 작업 검사하기 (Inspecting a task currently running)

완전성을 위해 — 다음은 아직 진행 중인 작업의 서랍 모습입니다 (스토리 1의 API 구현, `backend-dev`가 차지했지만 아직 완료되지 않음):

![클레임되어 진행 중인 작업](/img/kanban-tutorial/10-drawer-in-flight.png)

상태는 `Running`입니다. 활성 실행은 Run History 섹션에 결과 `active`로 표시되며 `ended_at`이 없습니다. 이 작업자가 종료되거나 시간 초과가 발생하면 디스패처는 이 실행을 적절한 결과와 함께 닫고 다음 클레임 시 새로운 실행을 엽니다 — 시도 행은 결코 사라지지 않습니다.

## 다음 단계 (Next steps)

- [칸반 개요 (Kanban overview)](./kanban) — 전체 데이터 모델, 이벤트 어휘 및 CLI 참조.
- `hermes kanban --help` — 모든 하위 명령어, 모든 플래그.
- `hermes kanban watch --kinds completed,gave_up,timed_out` — 전체 보드에 걸친 실시간 스트림 터미널 이벤트.
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 특정 작업이 완료될 때 게이트웨이 알림 받기.

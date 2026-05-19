# Kanban 튜토리얼

브라우저에 dashboard를 띄운 상태에서, Hermes Kanban 시스템이 설계된 4가지 대표 사용 사례를 따라가는 walkthrough입니다. 아직 [Kanban 개요](./kanban)를 읽지 않았다면 먼저 그 문서부터 보세요. 이 튜토리얼은 task, run, assignee, dispatcher의 의미를 이미 안다고 가정합니다.

## 설정

```bash
hermes kanban init           # 선택 사항; 첫 `hermes kanban <anything>` 호출 시 자동 초기화됨
hermes dashboard             # 브라우저에서 http://127.0.0.1:9119 열기
# 왼쪽 네비게이션에서 Kanban 클릭
```

dashboard는 시스템을 지켜보는 **사람인 당신**에게 가장 편한 인터페이스입니다. dispatcher가 spawn하는 agent worker는 dashboard나 CLI를 직접 보지 않습니다. 이들은 전용 `kanban_*` [toolset](./kanban#how-workers-interact-with-the-board) (`kanban_show`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_create`, `kanban_link`)으로 보드를 다룹니다. dashboard, CLI, worker tool은 모두 같은 board별 SQLite DB(기본 board는 `~/.hermes/kanban.db`, 이후 만든 board는 `~/.hermes/kanban/boards/<slug>/kanban.db`)를 통하므로, 어느 쪽에서 바꿔도 보드 상태는 일관됩니다.

이 튜토리얼은 계속 `default` board를 사용합니다. 프로젝트/레포/도메인별로 여러 개의 격리된 queue를 원한다면 개요 문서의 [Boards (멀티 프로젝트)](./kanban#boards-multi-project)를 보세요. CLI / dashboard / worker 흐름은 똑같고, worker는 물리적으로 다른 board의 task를 볼 수 없습니다.

이 문서 전체에서 **`bash`로 표시된 code block은 사람이 직접 실행하는 명령**입니다. **`# worker tool calls`**로 표시된 블록은 spawn된 worker의 모델이 실제로 내보내는 tool call 예시입니다. end-to-end 루프를 보여주기 위해 넣은 것이지, 사용자가 직접 실행하라는 뜻은 아닙니다.

## 보드 한눈에 보기

![Kanban board overview](/img/kanban-tutorial/01-board-overview.png)

왼쪽부터 오른쪽으로 6개 컬럼이 있습니다.

- **Triage** — 아직 거친 아이디어 상태인 항목. specifier가 구체 스펙으로 다듬기 전의 주차 구역입니다.
- **Todo** — 만들어졌지만 dependency를 기다리거나, 아직 assign되지 않은 task입니다.
- **Ready** — assign되었고 dispatcher가 claim하기만 기다리는 상태입니다.
- **In progress** — worker가 현재 실행 중인 task입니다. 기본값인 `Lanes by profile`이 켜져 있으면 assignee별로 하위 그룹이 생겨, 각 worker가 무엇을 하는지 한눈에 볼 수 있습니다.
- **Blocked** — worker가 사람 입력을 요청했거나 circuit breaker가 발동한 상태입니다.
- **Done** — 완료된 task입니다.

상단 바에는 search, tenant, assignee filter가 있고, `Lanes by profile` 토글과 `Nudge dispatcher` 버튼이 있습니다. `Nudge dispatcher`는 daemon의 다음 주기를 기다리지 않고 **지금 바로** dispatch tick을 한 번 실행합니다. 카드를 클릭하면 오른쪽 drawer가 열립니다.

### Flat view

profile lane이 너무 복잡하게 느껴지면 `Lanes by profile`을 끄세요. 그러면 In Progress 컬럼이 claim 시각 순의 단일 평면 리스트로 접힙니다.

![Board with lanes by profile off](/img/kanban-tutorial/02-board-flat.png)

## Story 1 — 혼자 기능을 출하하는 개발자

기능 하나를 만든다고 해봅시다. 전형적인 흐름은 스키마 설계 → API 구현 → 테스트 작성입니다. 부모→자식 dependency를 가진 task 3개로 구성됩니다.

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

`API`는 `SCHEMA`를 부모로 가지며, `tests`는 `API`를 부모로 가집니다. 그래서 처음에 `ready`로 시작하는 것은 `SCHEMA` 하나뿐입니다. 나머지 두 task는 부모가 끝나기 전까지 `todo`에 머뭅니다. 이것이 dependency promotion engine의 역할입니다. 아직 테스트할 API가 없는데 테스트 작성 worker가 먼저 집어가는 일은 생기지 않습니다.

다음 dispatcher tick(기본 60초, 또는 **Nudge dispatcher** 즉시 실행)에서 `backend-dev` profile이 `HERMES_KANBAN_TASK=$SCHEMA`를 가진 worker로 spawn됩니다. agent 내부에서 이 worker의 tool-call 루프는 대략 다음과 같습니다.

```python
# worker tool calls — 직접 실행하는 명령 아님
kanban_show()
# → title, body, worker_context, parents, prior attempts, comments를 반환

# (worker가 worker_context를 읽고 terminal/file tool로 스키마를 설계하고,
#  migration을 작성하고, 자체 체크를 돌리고, commit하는 실제 작업이 여기서 일어남)

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

`kanban_show`는 기본적으로 `task_id`를 `$HERMES_KANBAN_TASK`에서 가져오므로 worker는 자기 id를 몰라도 됩니다. `kanban_complete`는 summary + metadata를 현재 `task_runs` row에 기록하고, run을 닫고, task를 `done`으로 바꾸는 일을 **한 번의 atomic hop**으로 처리합니다.

`SCHEMA`가 `done`이 되면 dependency engine이 `API`를 자동으로 `ready`로 승격시킵니다. 이후 API worker가 `kanban_show()`를 호출하면, 부모 handoff에 붙은 `SCHEMA`의 summary와 metadata를 바로 보게 됩니다. 긴 설계 문서를 다시 읽지 않아도 스키마 결정을 이해할 수 있습니다.

보드에서 완료된 schema task를 클릭하면 drawer에 모든 것이 보입니다.

![Solo dev — completed schema task drawer](/img/kanban-tutorial/03-drawer-schema-task.png)

핵심은 하단의 **Run History** 섹션입니다. 한 번의 시도, outcome `completed`, worker `@backend-dev`, 소요 시간, 타임스탬프, 그리고 전체 handoff summary가 표시됩니다. metadata blob(`changed_files`, `decisions`)도 run에 함께 저장되어 이후 parent를 읽는 downstream worker에 전달됩니다.

같은 데이터는 언제든 터미널에서도 볼 수 있습니다. 아래 명령은 **사람인 당신**이 보드를 들여다보는 행위이지 worker 동작이 아닙니다.

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## Story 2 — Fleet farming

세 명의 worker(번역가, 전사 담당자, 카피라이터)와 서로 독립적인 task 묶음이 있다고 해봅시다. 세 명이 병렬로 일하면서 가시적인 진척을 내길 원합니다. 이것이 Kanban의 가장 단순하고도 원래 설계가 최적화된 대표 use-case입니다.

작업을 생성합니다.

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

gateway를 시작하고 잠시 자리를 떠나도 됩니다. gateway 안의 embedded dispatcher가 세 specialist profile의 task를 같은 `kanban.db`에서 동시에 끌어갑니다.

```bash
hermes gateway start
```

이제 보드를 `content-ops`로 filter하거나, "Transcribe"로 검색해보면 다음과 같은 화면이 나옵니다.

![Fleet view filtered to transcribe tasks](/img/kanban-tutorial/07-fleet-transcribes.png)

두 개의 전사 task는 완료, 하나는 실행 중, 둘은 다음 dispatcher tick을 기다리며 `ready` 상태입니다. In Progress 컬럼은 기본값인 profile lane으로 묶여 있기 때문에, mixed list를 훑지 않아도 각 worker의 현재 작업을 볼 수 있습니다. 현재 작업 하나가 끝나면 dispatcher가 다음 ready task를 즉시 running으로 승격합니다. 세 daemon이 세 개의 assignee pool을 병렬로 처리하므로, 사람 손을 더 대지 않아도 전체 content queue가 빠르게 소진됩니다.

**Story 1에서 본 structured handoff는 여기에도 그대로 적용됩니다.** 번역 worker가 task를 끝내며 `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})`처럼 남기면, 분석과 후속 task, 대시보드 통계에 바로 쓸 수 있습니다.

## Story 3 — 재시도가 포함된 역할 파이프라인

여기서 Kanban은 평평한 TODO 리스트보다 확실한 장점을 보여줍니다. PM이 spec을 쓰고, 엔지니어가 구현하고, reviewer가 첫 시도를 reject하고, 엔지니어가 수정 후 재시도하고, reviewer가 승인합니다.

`auth-project`로 filter된 dashboard 화면:

![Pipeline view for a multi-role feature](/img/kanban-tutorial/08-pipeline-auth.png)

세 단계 체인이 동시에 보입니다. `Spec: password reset flow` (DONE, pm), `Implement password reset flow` (DONE, backend-dev), `Review password reset PR` (READY, reviewer). 각 task는 하단에 녹색 parent link와 child dependency를 보여줍니다.

가장 흥미로운 것은 구현 task입니다. 한 번 blocked되었다가 retry되었기 때문입니다. 아래는 세 agent가 어떻게 맞물리는지 worker tool call 기준으로 보여주는 예시입니다.

```python
# --- PM worker가 $SPEC에 spawn되어 acceptance criteria를 작성 ---
# worker tool calls
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
# → $SPEC is done; $IMPL auto-promotes from todo to ready

# --- Engineer worker가 $IMPL에 spawn (첫 시도) ---
# worker tool calls
kanban_show()   # worker_context에서 $SPEC의 summary + acceptance metadata를 읽음
# (engineer가 코드를 작성하고, 테스트를 돌리고, PR을 엶)
# Reviewer feedback arrives — engineer decides the concerns are valid and blocks
kanban_block(
    reason="Review: password strength check missing, reset link isn't "
           "single-use (can be replayed within 30min)",
)
# → $IMPL transitions to blocked; run 1 closes with outcome='blocked'
```

이제 **사람인 당신**(혹은 별도 reviewer profile)이 block reason을 읽고, 수정 방향이 명확하다고 판단해 dashboard의 "Unblock" 버튼을 누르거나, CLI / slash command로 unblock합니다.

```bash
hermes kanban unblock $IMPL
# 또는 chat에서: /kanban unblock $IMPL
```

dispatcher는 `$IMPL`을 다시 `ready`로 돌려놓고, 다음 tick에 `backend-dev` worker를 다시 spawn합니다. 이 두 번째 spawn은 **같은 task에 대한 새로운 run**입니다.

```python
# --- Engineer worker가 $IMPL에 다시 spawn (두 번째 시도) ---
# worker tool calls
kanban_show()
# → 이제 worker_context에 run 1의 block reason이 포함되어 있으므로,
#   worker는 스펙 전체를 다시 읽기보다 어떤 두 가지를 고쳐야 하는지 바로 안다.
# (engineer가 zxcvbn check 추가, reset token을 single-use로 만들고, 테스트 재실행)
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

구현 task를 클릭하면 drawer에 **두 번의 시도**가 보입니다.

![Implementation task with two runs — blocked then completed](/img/kanban-tutorial/04b-drawer-retry-history-scrolled.png)

- **Run 1** — `@backend-dev`가 `blocked`. review feedback이 outcome 아래에 그대로 남아 있습니다: "password strength check missing, reset link isn't single-use (can be replayed within 30min)".
- **Run 2** — `@backend-dev`가 `completed`. 새로운 summary와 metadata를 가집니다.

각 run은 `task_runs`의 독립 row이며, 고유한 outcome, summary, metadata를 가집니다. retry history는 단지 최신 상태 task 위에 얹힌 부가 기능이 아니라, 시스템의 **1차 표현 방식**입니다. 재시도 worker가 task를 열면 `build_worker_context`가 이전 시도를 보여주므로, 두 번째 worker는 첫 번째 시도가 왜 막혔는지 알고 **같은 실수를 반복하지 않습니다**.

이제 reviewer 차례입니다. `Review password reset PR`을 열면 다음을 보게 됩니다.

![Reviewer's drawer view of the pipeline](/img/kanban-tutorial/09-drawer-pipeline-review.png)

부모 link는 완료된 구현 task를 가리킵니다. reviewer worker가 `Review password reset PR`에 spawn되어 `kanban_show()`를 호출하면, `worker_context`는 부모의 **가장 최근 completed run의 summary + metadata**를 포함합니다. 그래서 diff를 보기 전부터 "added zxcvbn strength check, reset tokens are now single-use"라는 요약과 changed file 목록을 손에 쥐고 시작할 수 있습니다.

## Story 4 — Circuit breaker와 crash recovery

현실의 worker는 실패합니다. credential 누락, OOM kill, 일시적 네트워크 오류가 생깁니다. dispatcher는 이에 대해 두 겹의 방어선을 가집니다.

- **circuit breaker** — 연속 N회 실패 시 자동 block하여 보드가 영원히 thrash하지 않도록 함
- **crash detection** — TTL이 만료되기 전에 worker PID가 사라진 task를 reclaim함

### Circuit breaker — 영구 장애처럼 보이는 실패

예를 들어 `AWS_ACCESS_KEY_ID`가 profile 환경에 없는 deploy task:

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops
```

dispatcher가 worker spawn을 시도하지만 `RuntimeError: AWS_ACCESS_KEY_ID not set`로 실패합니다. dispatcher는 claim을 release하고 failure counter를 증가시키며 다음 tick에 다시 시도합니다. 기본 `failure_limit`인 3회 연속 실패 후 circuit이 열리면 task는 outcome `gave_up`과 함께 `blocked`가 됩니다. 사람이 unblock하기 전까지는 더 이상 재시도하지 않습니다.

blocked task를 클릭하면:

![Circuit breaker — 2 spawn_failed + 1 gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

같은 error가 적힌 세 개의 run이 보입니다. 앞의 두 개는 `spawn_failed`(재시도 가능), 세 번째는 `gave_up`(종결 상태)입니다. 위쪽 event log는 `created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`의 전체 순서를 보여줍니다.

터미널에서는:

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

Telegram / Discord / Slack이 연결되어 있다면 `gave_up` 이벤트에 대해 gateway notification이 발송되므로, 보드를 수동으로 확인하지 않아도 장애를 알 수 있습니다.

### Crash recovery — worker가 실행 중 중간에 죽는 경우

spawn은 성공했지만 이후 worker 프로세스가 죽는 경우도 있습니다. segfault, OOM, `systemctl stop` 등이 여기에 해당합니다. dispatcher는 `kill(pid, 0)` polling으로 죽은 pid를 감지하고, claim을 release하고, task를 `ready`로 되돌린 뒤 다음 tick에 새 worker에게 넘깁니다.

seed data 예시에서는 migration이 메모리를 다 써버리는 상황입니다.

```bash
# Worker claims, starts scanning 2.4M rows, OOM kills it at ~2.3M
# Dispatcher detects dead pid, releases claim, increments attempt counter
# Retry with a chunked strategy succeeds
```

drawer는 두 번의 시도 전체를 보여줍니다.

![Crash and recovery — 1 crashed + 1 completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

Run 1은 `crashed`, error는 `OOM kill at row 2.3M (process 99999 gone)`입니다. Run 2는 `completed`, metadata에는 `"strategy": "chunked with LIMIT + WHERE id > last_id"`가 들어 있습니다. retrying worker는 run 1의 crash를 컨텍스트에서 보고 더 안전한 전략을 선택했습니다. metadata는 나중에 보는 사람이나 postmortem 작성자에게 **무엇이 바뀌었는지**를 즉시 보여줍니다.

## Structured handoff — 왜 `summary`와 `metadata`가 중요한가

위의 모든 story에서 worker는 마지막에 `kanban_complete(summary=..., metadata=...)`를 호출했습니다. 이것은 장식이 아니라, workflow 단계 사이를 잇는 **주요 handoff 채널**입니다.

task B의 worker가 spawn되어 `kanban_show()`를 호출하면 `worker_context`에는 다음이 포함됩니다.

- B 자신의 **이전 시도들** (outcome, summary, error, metadata) — retry worker가 실패한 경로를 반복하지 않도록 함
- **부모 task 결과** — 각 부모에 대해 가장 최근 completed run의 summary와 metadata — downstream worker가 upstream 작업의 이유와 방법을 이해하도록 함

이 구조는 flat kanban 시스템에서 흔한 "comment와 결과물을 뒤져서 맥락을 복원하는 일"을 대체합니다. PM이 spec metadata에 acceptance criteria를 쓰면 engineer worker는 부모 handoff에서 이를 구조적으로 읽습니다. engineer가 어떤 테스트를 돌렸고 몇 개가 통과했는지 기록하면, reviewer worker는 diff를 열기 전부터 그 목록을 손에 쥐게 됩니다.

bulk-close guard가 존재하는 이유도 이것이 **run별 데이터**이기 때문입니다. `hermes kanban complete a b c --summary X`는 CLI에서 거부됩니다. 같은 summary를 세 task에 복붙하는 일은 거의 항상 잘못이기 때문입니다. handoff flag 없이 bulk close하는 기능은 "행정성 task 여러 개를 한꺼번에 끝냈다" 같은 일반적인 경우를 위해 여전히 남아 있습니다. 반면 tool 표면에는 bulk variant 자체가 없습니다. 같은 이유로 `kanban_complete`는 언제나 single-task 단위입니다.

## 현재 실행 중인 task 들여다보기

완전성을 위해, 아직 끝나지 않은 in-flight task의 drawer도 보겠습니다. 아래는 Story 1의 API 구현 task가 `backend-dev`에 claim되어 실행 중이지만 아직 완료되지 않은 상태입니다.

![Claimed, in-flight task](/img/kanban-tutorial/10-drawer-in-flight.png)

상태는 `Running`입니다. 활성 run은 Run History 섹션에 outcome `active`, `ended_at` 없음으로 표시됩니다. 만약 이 worker가 죽거나 timeout되면, dispatcher는 이 run을 적절한 outcome으로 닫고 다음 claim에서 새 run을 엽니다. **시도 row는 사라지지 않습니다.**

## 다음 단계

- [Kanban 개요](./kanban) — 전체 데이터 모델, event vocabulary, CLI reference
- `hermes kanban --help` — 모든 subcommand와 flag
- `hermes kanban watch --kinds completed,gave_up,timed_out` — 보드 전체 terminal event 실시간 스트림
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 특정 task가 끝날 때 gateway 알림 받기

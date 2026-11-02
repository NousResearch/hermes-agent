# Kanban 튜토리얼

브라우저에서 대시보드를 열고 Hermes Kanban 시스템이 설계된 네 가지 사용 사례를 따라가 봅니다. 아직 [Kanban 개요](./kanban)를 읽지 않았다면 먼저 읽어 보세요. 여기서는 task, run, assignee, dispatcher가 무엇인지 알고 있다고 가정합니다.

## 설정

```bash
hermes kanban init           # optional; first `hermes kanban <anything>` auto-inits
hermes dashboard             # opens http://127.0.0.1:9119 in your browser
# click Kanban in the left nav
```

대시보드는 **여러분이** 시스템을 지켜보기에 가장 편한 장소입니다. dispatcher가 생성하는 agent worker는 대시보드나 CLI를 전혀 보지 못합니다. 대신 전용 `kanban_*` [toolset](./kanban#how-workers-interact-with-the-board) (`kanban_show`, `kanban_list`, `kanban_complete`, `kanban_block`, `kanban_heartbeat`, `kanban_comment`, `kanban_attach`, `kanban_attach_url`, `kanban_attachments`, `kanban_create`, `kanban_link`, `kanban_unblock`)을 통해 보드를 조작합니다. 세 표면인 대시보드, CLI, worker tool은 모두 동일한 보드별 SQLite DB(기본 보드는 `~/.hermes/kanban.db`, 이후 생성하는 보드는 `~/.hermes/kanban/boards/<slug>/kanban.db`)를 사용하므로, 어느 쪽에서 변경했든 각 보드의 상태가 일관됩니다.

이 튜토리얼에서는 처음부터 끝까지 `default` 보드를 사용합니다. 여러 개의 격리된 큐(프로젝트 / 저장소 / 도메인별 하나씩)가 필요하다면 개요의 [Boards (multi-project)](./kanban#boards-multi-project)를 참조하세요. 동일한 CLI / dashboard / worker 흐름이 보드별로 적용되며, worker는 다른 보드의 task를 물리적으로 볼 수 없습니다.

튜토리얼 전체에서 `bash`로 표시된 **code block은 여러분이 실행하는 명령**입니다. `# worker tool calls`로 표시된 code block은 생성된 worker의 model이 tool call로 내보내는 내용입니다. 전체 루프를 직접 볼 수 있도록 표시한 것이며, 여러분이 직접 실행할 내용은 아닙니다.

## 보드 한눈에 보기

![Kanban 보드 개요](/img/kanban-tutorial/01-board-overview.png)

왼쪽에서 오른쪽으로 여섯 개의 열이 있습니다.

- **Triage** — 가공되지 않은 아이디어입니다. 기본적으로 dispatcher는 이 열의 task에서 **decomposer**를 자동 실행합니다. 내장 decomposer는 `auxiliary.kanban_decomposer`를 사용하고, 여러분의 profile roster와 설명을 읽어 가장 적합한 specialist로 라우팅되는 child task 그래프를 만듭니다. 원래 task는 parent로 유지되므로, 모든 작업이 끝나면 그 assignee(`kanban.orchestrator_profile`, 설정되지 않은 경우 활성 default profile)가 다시 깨어나 완료 여부를 판단합니다. kanban 페이지 상단의 **Orchestration: Auto/Manual** pill을 전환해 모드를 바꿀 수 있습니다. Manual 모드에서는 카드의 **⚗ Decompose**를 클릭하거나 `hermes kanban decompose <id>` / `/kanban decompose <id>`를 실행하세요. fan-out이 필요 없는 단일 task에는 **✨ Specify**가 한 번 실행되는 spec 재작성(goal, approach, acceptance criteria)을 수행하고 `todo`로 승격합니다. `config.yaml`의 `auxiliary.kanban_decomposer`와 `auxiliary.triage_specifier`에서 model을 설정하세요. 메인 Kanban guide의 [Auto vs Manual orchestration](./kanban#auto-vs-manual-orchestration)를 참조하세요.
- **Todo** — 생성되었지만 dependency를 기다리는 task이거나 아직 할당되지 않은 task입니다.
- **Ready** — 할당되었으며 dispatcher가 claim하기를 기다리는 task입니다.
- **In progress** — worker가 활발히 task를 실행 중인 상태입니다. (기본값인) "Lanes by profile"이 켜져 있으면 이 열이 assignee별 하위 그룹으로 나뉘어 각 worker가 무엇을 하고 있는지 한눈에 볼 수 있습니다.
- **Blocked** — worker가 사람의 입력을 요청했거나 circuit breaker가 작동한 상태입니다.
- **Done** — 완료된 task입니다.

상단 바에는 search, tenant, assignee 필터와 `Lanes by profile` toggle, 그리고 daemon의 다음 interval을 기다리지 않고 지금 즉시 dispatch tick을 한 번 실행하는 `Nudge dispatcher` button이 있습니다. 카드를 클릭하면 오른쪽에 해당 drawer가 열립니다.

### Flat view

profile lane이 복잡하게 보이면 "Lanes by profile"을 끄세요. 그러면 In Progress 열이 claim time 순으로 정렬된 하나의 flat list로 접힙니다.

![profile lane을 끈 보드](/img/kanban-tutorial/02-board-flat.png)

## Story 1 — 기능을 출시하는 Solo dev

기능을 개발하고 있습니다. 전형적인 흐름은 schema를 설계하고, API를 구현하고, test를 작성하는 것입니다. parent→child dependency가 있는 세 개의 task입니다.

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

`API`의 parent가 `SCHEMA`이고 `tests`의 parent가 `API`이므로, 처음에는 `SCHEMA`만 `ready` 상태가 됩니다. 나머지 두 task는 parent가 완료될 때까지 `todo`에 머뭅니다. 이것이 dependency promotion engine이 제 역할을 하는 모습입니다. API가 테스트할 대상이 생길 때까지 다른 worker는 test-writing task를 가져가지 않습니다.

다음 dispatcher tick(기본값 60s, 또는 **Nudge dispatcher**를 누르면 즉시)에 `backend-dev` profile이 worker로 생성되고, 환경에 `HERMES_KANBAN_TASK=$SCHEMA`가 설정됩니다. 다음은 agent 내부에서 worker의 tool-call loop가 어떻게 보이는지 보여 줍니다.

```python
# worker tool calls — NOT commands you run
kanban_show()
# → returns title, body, worker_context, parents, prior attempts, comments

# (worker reads worker_context, uses terminal/file tools to design the schema,
#  write migrations, run its own checks, commit — the real work happens here)

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

`kanban_show`는 기본적으로 `task_id`를 `$HERMES_KANBAN_TASK`로 설정하므로 worker가 자신의 id를 알 필요가 없습니다. `kanban_complete`는 현재 `task_runs` row에 summary + metadata를 기록하고 해당 run을 닫은 다음, `kanban_db`를 통한 하나의 atomic hop으로 task를 `done`으로 전환합니다.

`SCHEMA`가 `done` 상태가 되면 dependency engine이 자동으로 `API`를 `ready`로 승격합니다. API worker가 task를 가져가면 `kanban_show()`를 호출하고 parent handoff에 첨부된 `SCHEMA`의 summary와 metadata를 확인하므로, 긴 design doc을 다시 읽지 않고도 schema 결정 사항을 알 수 있습니다.

보드에서 완료된 schema task를 클릭하면 drawer에 모든 내용이 표시됩니다.

![Solo dev — 완료된 schema task drawer](/img/kanban-tutorial/03-drawer-schema-task.png)

아래쪽의 Run History section이 핵심적인 추가 요소입니다. 하나의 attempt에 outcome `completed`, worker `@backend-dev`, duration, timestamp, 그리고 handoff summary 전체가 표시됩니다. metadata blob(`changed_files`, `decisions`)도 run에 저장되며, 이 parent를 읽는 downstream worker라면 누구나 확인할 수 있습니다.

같은 데이터를 언제든지 terminal에서 확인할 수 있습니다. 이 명령은 worker가 아니라 **여러분이** 보드를 살펴보는 것입니다.

```bash
hermes kanban show $SCHEMA
hermes kanban runs $SCHEMA
# #  OUTCOME       PROFILE       ELAPSED  STARTED
# 1  completed     backend-dev        0s  2026-04-27 19:34
#     → users(id, email, pw_hash), sessions(id, user_id, jti, expires_at); refresh tokens ...
```

## Story 2 — Fleet farming

세 명의 worker(translator, transcriber, copywriter)와 서로 독립적인 task 더미가 있습니다. 세 worker가 모두 병렬로 작업을 가져가고 진행 상황을 눈에 보이게 만들고 싶습니다. 이것은 가장 단순한 kanban 사용 사례이며, 원래 설계가 최적화된 대상이기도 합니다.

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

gateway를 시작하고 맡겨 두세요. gateway가 세 specialist profile의 task를 동일한 kanban.db에서 가져가는 embedded dispatcher를 호스팅합니다.

```bash
hermes gateway start
```

이제 보드를 `content-ops`로 필터링하거나(또는 "Transcribe"를 검색하면) 다음과 같이 표시됩니다.

![transcribe task로 필터링한 Fleet view](/img/kanban-tutorial/07-fleet-transcribes.png)

transcribe 두 개는 완료되었고, 하나는 실행 중이며, 두 개는 다음 dispatcher tick을 기다리며 ready 상태입니다. In Progress 열은 profile별로 그룹화되어 있으므로("Lanes by profile" 기본값), 섞여 있는 목록을 훑지 않고도 각 worker의 활성 task를 확인할 수 있습니다. 현재 task가 완료되는 즉시 dispatcher가 다음 ready task를 running으로 승격합니다. 세 개의 daemon이 세 assignee pool에서 병렬로 작업하므로, 추가적인 사람의 입력 없이 전체 content queue가 비워집니다.

**Story 1에서 설명한 structured handoff는 여기에도 모두 적용됩니다.** call을 완료한 translator worker는 `kanban_complete(summary="translated 4 pages, style matched existing marketing voice", metadata={"duration_seconds": 720, "tokens_used": 2100})`를 emit합니다. 이는 analytics에 유용하며, 이 task에 의존하는 downstream task에도 유용합니다.

## Story 3 — retry가 있는 Role pipeline

여기서 Kanban은 단순한 flat TODO list보다 진가를 발휘합니다. PM이 spec을 작성하고, engineer가 구현하며, reviewer가 첫 시도를 거절합니다. engineer는 변경 사항을 반영해 다시 시도하고, reviewer가 승인합니다.

`auth-project`로 필터링한 dashboard view입니다.

![multi-role feature를 위한 Pipeline view](/img/kanban-tutorial/08-pipeline-auth.png)

스크린샷은 **미리 생성한 downstream card** model을 사용합니다. implementation card에는 전용 reviewer child가 있습니다. 이 model에서는 engineer가 implementation이 준비되었을 때 `kanban_complete`를 호출해야 reviewer child가 `todo`에서 벗어날 수 있습니다. review를 요청하기 위해 implementation parent를 절대 block하지 마세요.

같은 card가 implementation과 review를 모두 담당하는 workflow라면 first-class review lifecycle을 사용하세요. 전체 implement → review → changes → re-review choreography는 다음과 같습니다.

```python
# --- Engineer: first implementation attempt ---
kanban_show()
# (write code, run tests, prepare the candidate)
kanban_request_review(
    summary="implemented reset flow; candidate is ready for review",
    metadata={"changed_files": ["auth/reset.py"], "tests_run": 8},
    reviewer="reviewer",
)
# → the same card enters review; the implementation run closes as
#   outcome='review_requested'

# --- Reviewer: request concrete changes ---
kanban_show()
# (inspect the handoff and candidate)
kanban_request_changes(
    reason="Add password-strength validation and make reset tokens single-use."
)
# → the review run closes as outcome='changes_requested'; the card returns
#   to backend-dev in ready/todo without touching block-loop accounting

# --- Engineer: second implementation attempt ---
kanban_show()  # prior review evidence is in worker_context
# (apply feedback and re-run tests)
kanban_request_review(
    summary="added zxcvbn validation and single-use reset tokens",
    metadata={
        "changed_files": [
            "auth/reset.py",
            "auth/tests/test_reset.py",
            "migrations/003_single_use_reset_tokens.sql",
        ],
        "tests_run": 11,
        "review_iteration": 2,
    },
    reviewer="reviewer",
)

# --- Reviewer: approve ---
kanban_complete(summary="review passed; acceptance criteria verified")
# → done
```

이제 task의 run history에는 `review_requested → changes_requested → review_requested → completed`가 기록됩니다. 각 attempt에는 별도의 actor, summary, metadata, outcome이 있으므로 두 번째 engineer는 reviewer가 거절한 내용을 정확히 확인할 수 있고, 최종 승인은 audit할 수 있는 상태로 남습니다. `kanban_block`은 정상적인 review feedback이 아니라 실제 외부 escalation(접근 권한 누락, product decision, 사용 불가능한 infrastructure)을 위해 사용해야 합니다.

스크린샷처럼 downstream-card model을 의도적으로 사용하는 경우 reviewer는 implementation parent가 완료된 후 `Review password reset PR`을 엽니다.

![Pipeline의 reviewer's drawer view](/img/kanban-tutorial/09-drawer-pipeline-review.png)

reviewer card의 `worker_context`에는 완료된 implementation handoff가 포함됩니다. 이는 별도의 card workflow입니다. same-card `kanban_request_review`와 결합하면 review lane이 중복되므로 함께 사용하지 마세요.

## Story 4 — Circuit breaker와 crash recovery

실제 worker는 실패합니다. credential 누락, OOM kill, 일시적인 network error 등이 그 예입니다. dispatcher에는 두 가지 방어선이 있습니다. 하나는 N회 연속 실패 후 자동으로 block해 보드가 영원히 thrash하지 않도록 하는 **circuit breaker**이고, 다른 하나는 worker PID가 TTL 만료 전에 사라진 task를 회수하는 **crash detection**입니다.

### Circuit breaker — 영구적으로 보이는 failure

profile 환경에 `AWS_ACCESS_KEY_ID`가 설정되지 않아 worker를 spawn할 수 없는 deploy task입니다.

```bash
hermes kanban create "Deploy to staging (missing creds)" \
    --assignee deploy-bot --tenant ops \
    --max-retries 3
```

dispatcher가 worker spawn을 시도합니다. spawn이 실패(`RuntimeError: AWS_ACCESS_KEY_ID not set`)하면 dispatcher는 claim을 해제하고 failure counter를 증가시킨 다음, 다음 tick에 다시 시도합니다. 이 예시에서는 `--max-retries 3`을 설정했으므로 세 번 연속 실패한 후 circuit이 작동합니다. task는 outcome `gave_up`과 함께 `blocked`로 이동합니다. flag를 생략하면 Hermes는 `kanban.failure_limit`(기본값: 2)을 사용합니다. 사람이 unblock하기 전까지는 더 이상 retry하지 않습니다.

blocked task를 클릭합니다.

![Circuit breaker — 2 spawn_failed + 1 gave_up](/img/kanban-tutorial/11-drawer-gave-up.png)

세 run 모두 `error` field에 같은 error가 표시됩니다. 첫 두 run은 retry 가능한 `spawn_failed`이고, 세 번째는 terminal 상태인 `gave_up`입니다. 위의 event log에는 전체 sequence가 표시됩니다. `created → claimed → spawn_failed → claimed → spawn_failed → claimed → gave_up`입니다.

terminal에서 확인합니다.

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

Telegram / Discord / Slack이 연결되어 있다면 `gave_up` event가 발생할 때 gateway notification이 전송됩니다. 보드를 직접 확인하지 않아도 outage를 알 수 있습니다.

### Crash recovery — worker가 실행 도중 종료되는 경우

spawn은 성공했지만 이후 worker process가 종료되는 경우도 있습니다. segfault, OOM, `systemctl stop` 등이 그 예입니다. dispatcher는 `kill(pid, 0)`을 polling해 dead pid를 감지합니다. 그러면 claim이 해제되고 task는 다시 `ready`로 돌아가며, 다음 tick에 새 worker가 task를 받습니다.

seed data의 예시는 메모리가 부족해 실행 중이던 migration입니다.

```bash
# Worker claims, starts scanning 2.4M rows, OOM kills it at ~2.3M
# Dispatcher detects dead pid, releases claim, increments attempt counter
# Retry with a chunked strategy succeeds
```

drawer에는 두 번의 attempt 전체 history가 표시됩니다.

![Crash and recovery — 1 crashed + 1 completed](/img/kanban-tutorial/06-drawer-crash-recovery.png)

Run 1은 `crashed`이며 error는 `OOM kill at row 2.3M (process 99999 gone)`입니다. Run 2는 `completed`이며 metadata에 `"strategy": "chunked with LIMIT + WHERE id > last_id"`가 있습니다. retry한 worker는 context에서 run 1의 crash를 확인하고 더 안전한 strategy를 선택했습니다. metadata 덕분에 무엇이 바뀌었는지 future observer(또는 postmortem writer)가 쉽게 알 수 있습니다.

## Structured handoff — `summary`와 `metadata`가 중요한 이유

위의 모든 story에서 worker는 마지막에 `kanban_complete(summary=..., metadata=...)`를 호출했습니다. 이것은 장식이 아닙니다. workflow 단계 사이의 primary handoff channel입니다.

task B의 worker가 생성되어 `kanban_show()`를 호출하면, worker가 돌려받는 `worker_context`에는 다음이 포함됩니다.

- B의 **prior attempts**(이전 run의 outcome, summary, error, metadata)가 포함되므로 retry하는 worker가 실패한 경로를 반복하지 않습니다.
- **Parent task results** — 각 parent의 가장 최근 완료 run에서 summary와 metadata — 가 포함되므로 downstream worker가 upstream 작업이 왜, 어떻게 수행되었는지 확인할 수 있습니다.

이는 flat kanban system에서 흔히 발생하는 "comment와 작업 결과를 뒤지는" 과정을 대신합니다. PM은 spec의 metadata에 acceptance criteria를 기록하고, engineer의 worker는 parent handoff에서 이를 구조적으로 확인합니다. engineer가 실행한 test와 통과한 수를 기록하면, reviewer의 worker는 diff를 열기 전에 그 목록을 확인할 수 있습니다.

bulk-close guard가 존재하는 이유는 이 데이터가 run별로 관리되기 때문입니다. `hermes kanban complete a b c --summary X`(CLI에서 여러분이 실행)는 거부됩니다. 동일한 summary를 세 task에 복사해 붙이는 것은 거의 항상 잘못이기 때문입니다. handoff flag 없이 bulk close하는 것은 흔한 "admin task 더미를 끝냈다"는 경우를 위해 여전히 작동합니다. tool surface에는 bulk variant가 아예 노출되지 않습니다. 같은 이유로 `kanban_complete`는 항상 한 번에 하나의 task만 처리합니다.

## 완료된 card의 후속 작업 — parent link를 통한 CI remediation

Story 1의 implementation card가 `done` 상태입니다. 두 시간 후 merge된 branch에서 CI가 실패했습니다. 완료된 card는 history이며 handoff가 앞으로 흐르므로, done card를 다시 열지 마세요. done card를 **parent**로 지정해 remediation card를 생성합니다.

```bash
hermes kanban create "Fix CI: test_backoff_jitter flakes on 3.11" \
    --assignee backend-dev \
    --parent t_impl \
    --workspace worktree --branch wt/ci-fix-backoff \
    --body "CI run #4812 failed after t_impl completed.
FAILED tests/test_retry.py::test_backoff_jitter - TimeoutError
Acceptance: tests/test_retry.py green on 3.11 and 3.12."
```

이 작업이 가능한 이유는 세 가지입니다.

- **Immediate dispatch.** Parent가 이미 `done` 상태이므로 child가 곧바로 `ready`로 생성되고, dispatcher는 다음 tick에 이를 claim할 수 있습니다. (아직 열린 parent의 child라면 `todo`에서 기다립니다.)
- **Inherited context.** Remediation worker의 context에는 *Parent task results* section이 포함됩니다. 이 section에는 `t_impl`의 완료 summary와 metadata, 즉 원래 worker가 기록한 changed files와 decisions가 담겨 있습니다. 따라서 한 줄도 읽기 전에 code가 왜 그런 형태가 되었는지 알 수 있습니다.
- **Fresh evidence in the body.** `t_impl`이 완료될 때는 CI log가 존재하지 않았으므로 parent handoff에 들어갈 수 없습니다. CI log는 명시적인 acceptance criteria와 함께 새 card의 body에 넣습니다.

remediation card에는 새로운 worktree/branch를 사용하는 것이 좋습니다. 원래 branch를 checkout하면 worker에게 repo *state*만 주어지고 *rationale*은 주어지지 않습니다. parent handoff가 rationale을 전달합니다. 같은 assignee profile을 사용하는 것이 보통 올바릅니다. code를 작성한 profile이 이를 수정할 skill도 갖고 있기 때문입니다.

## 현재 실행 중인 task 확인

참고로 아직 실행 중인 task의 drawer는 다음과 같습니다. (Story 1의 API implementation으로, `backend-dev`가 claim했지만 아직 완료되지 않았습니다.)

![Claimed, 실행 중인 task](/img/kanban-tutorial/10-drawer-in-flight.png)

상태는 `Running`입니다. active run은 Run History section에 outcome `active`와 `ended_at` 없음으로 표시됩니다. 이 worker가 종료되거나 timeout되면 dispatcher는 적절한 outcome으로 이 run을 닫고 다음 claim에서 새 run을 엽니다. attempt row는 사라지지 않습니다.

## 다음 단계

- [Kanban overview](./kanban) — 전체 data model, event vocabulary, CLI reference입니다.
- `hermes kanban --help` — 모든 subcommand와 모든 flag입니다.
- `hermes kanban watch --kinds completed,gave_up,timed_out` — 전체 보드의 terminal event를 실시간으로 stream합니다.
- `hermes kanban notify-subscribe <task> --platform telegram --chat-id <id>` — 특정 task가 완료될 때 gateway ping을 받습니다.

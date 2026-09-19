---
sidebar_position: 16
title: "지속적인 목표"
description: "지속적인 목표를 설정하고 완료될 때까지 Hermes가 여러 턴에 걸쳐 계속 작업하도록 합니다. Ralph 루프에 대한 Hermes의 구현입니다."
---

# 지속적인 목표 (`/goal`)

`/goal`은 여러 턴에 걸쳐 유지되는 목표를 Hermes에 설정합니다. 매 턴이 끝나면 가벼운 judge 모델이 assistant의 마지막 응답으로 목표가 달성되었는지 확인합니다. 달성되지 않았다면 Hermes는 같은 세션에 continuation prompt를 자동으로 전달하고 계속 작업합니다. 목표가 달성되거나, 일시 중지 또는 삭제되거나, 턴 예산이 소진될 때까지 이어집니다.

이는 [Codex CLI 0.128.0의 `/goal`](https://github.com/openai/codex)에서 Eric Traut(OpenAI)이 직접 영감을 받은 **Ralph loop**에 대한 Hermes의 구현입니다. 여러 턴에 걸쳐 목표를 유지하고 달성될 때까지 멈추지 않는다는 핵심 아이디어는 그들의 것이며, 구현은 독립적으로 Hermes 아키텍처에 맞게 적용되었습니다.

## 언제 사용하나요

사용자가 매 턴 다시 프롬프트를 보내지 않아도 Hermes가 스스로 반복 작업하기를 원하는 경우 `/goal`을 사용하세요.

- "`src/`의 모든 lint 오류를 수정하고 `ruff check`가 통과하는지 확인해 줘"
- "repo Y의 feature X를 테스트까지 포함해 포팅하고 CI를 통과시켜 줘"
- "중간 실행 compression 중 session ID가 가끔 어긋나는 이유를 조사하고 보고서를 작성해 줘"
- "EXIF 날짜로 파일 이름을 바꾸는 작은 CLI를 만들고 photos/ 폴더를 대상으로 테스트해 줘"

에이전트가 한 턴 실행한 뒤 멈추는 작업에는 `/goal`이 필요하지 않습니다. *"계속 진행해"라고 세 번 말해야 할 작업*이라면 `/goal`이 적합합니다.

## Goals와 Kanban: 어느 것을 사용해야 하나요?

`/goal`과 [Kanban](./kanban)은 모두 사용자가 다시 프롬프트를 보내지 않아도 Hermes가 작업을 계속하도록 합니다. 그래서 둘이 연결되어 있다고 생각하기 쉽지만, 실제 경계는 분명합니다.

- **`/goal`은 단일 세션입니다.** 루프는 이 대화에 continuation prompt를 전달하며 judge가 완료되었다고 판단할 때까지 계속합니다. 목표를 설정해도 kanban card가 생성되지 않고, 다른 profile에 작업이 할당되지 않으며, 작업이 분산되지 않습니다.
- **Kanban은 여러 작업으로 이루어진 보드입니다.** 각 card는 자체 세션을 가진 별도의 worker process에 dispatch됩니다. Card, dependency, assignee, handoff는 `/goal`이 아니라 보드에 저장됩니다.
- **겹치는 부분은 의도적이며 작습니다.** `--goal`로 생성한 kanban card는 해당 card의 worker session 안에서 `/goal`과 같은 Ralph-style continuation engine을 실행합니다. 보드가 아니라 engine을 가져다 씁니다. [Goal-mode cards](./kanban#goal-mode-cards---goal)를 참고하세요.

| 원하는 것 | 사용할 기능 |
|---|---|
| 이 대화에서 한 작업을 완료할 때까지 계속 반복하기 | `/goal <text>` |
| dependency, handoff 또는 여러 profile이 필요한 여러 독립 작업 | [Kanban](./kanban) — `hermes kanban create …` |
| 완료 조건이 충족될 때까지 계속 반복해야 하는 보드의 한 card | `--goal`을 사용하는 kanban card |

:::note
보드에서 작업하려면 직접 보드에 추가하세요(`hermes kanban create …`). `/goal`이 대신 해 주지는 않습니다. 반대도 마찬가지입니다. 이 대화에서 목표를 일시 중지하거나, 재개하거나, 삭제해도 kanban card가 생성되거나, 점유되거나, 이동하지 않습니다.
:::

## 빠른 시작

```
/goal Fix every failing test in tests/hermes_cli/ and make sure scripts/run_tests.sh passes for that directory
```

다음과 같은 내용을 보게 됩니다.

1. **목표 수락** — `⊙ Goal set (20-turn budget): <your goal>`
2. **Turn 1 실행** — Hermes가 목표를 일반 메시지로 받은 것처럼 작업을 시작합니다.
3. **Judge 실행** — 턴이 끝나면 judge model이 `done` 또는 `continue`를 결정합니다.
4. **필요하면 루프 실행** — `continue`이면 `↻ Continuing toward goal (1/20): <judge's reason>`이 표시되고 Hermes가 다음 단계를 자동으로 수행합니다.
5. **종료** — 최종적으로 `✓ Goal achieved: <reason>` 또는 `⏸ Goal paused — N/20 turns used`가 표시됩니다.

## Commands

| Command | 동작 |
|---|---|
| `/goal <text>` | 지속적인 목표를 설정하거나 교체합니다. 별도의 메시지를 보내지 않아도 첫 번째 턴을 즉시 시작합니다. |
| `/goal draft <text>` | 일반 언어의 목표에서 구조화된 completion contract를 작성한 뒤 설정합니다. [Completion contracts](#completion-contracts)를 참고하세요. |
| `/goal show` | 활성 목표의 completion contract를 출력합니다. |
| `/goal` 또는 `/goal status` | 현재 목표, 상태, 사용한 턴 수를 표시합니다. |
| `/goal pause` | 목표를 삭제하지 않고 자동 continuation loop를 중지합니다. |
| `/goal resume` | 루프를 재개합니다(턴 카운터를 0으로 초기화). |
| `/goal clear` | 목표를 완전히 삭제합니다. |
| `/goal wait <pid> [reason]` | background process가 실행되는 동안 루프를 대기시킵니다. process가 종료되면 자동으로 재개합니다. |
| `/goal unwait` | wait barrier를 해제하고 즉시 루프를 재개합니다. |
| `/goal gate add <command>` | 목표가 완료되었다고 판단되기 전에 통과해야 하는 **quality gate**인 shell command를 추가합니다. [Quality gates](#quality-gates)를 참고하세요. |
| `/goal gate` 또는 `/goal gate list` | 목표의 gate와 통과/실패 상태를 나열합니다. |
| `/goal gate remove <N>` | N번째 gate를 삭제합니다(1부터 시작). |
| `/goal gate clear` | 모든 gate를 삭제합니다. |

CLI와 모든 gateway platform(Telegram, Discord, Slack, Matrix, Signal, WhatsApp, SMS, iMessage, Webhook, API server, web dashboard)에서 동일하게 동작합니다.

## Completion contracts

기본 `/goal <text>`도 잘 동작하지만, *모호한* 목표는 판단도 모호하게 만듭니다. judge는 무엇을 원하는지 알려 준 내용만 확인할 수 있기 때문입니다. Codex의 `/goal` 안내도 같은 점을 강조합니다. 오래 유지되는 목표는 **완료의 의미, 입증 방법, 변경해서는 안 되는 것, 범위, 중단하고 입력을 요청할 시점**을 명시할 때 가장 효과적입니다. Hermes는 이를 기존 goal loop 위에 선택적으로 추가할 수 있는 **completion contract**로 적용합니다.

Contract에는 모두 선택 사항인 다섯 가지 field가 있습니다.

| Field | 의미 |
|---|---|
| `outcome` | 완료 시 반드시 참이어야 하는 하나의 최종 상태입니다. |
| `verification` | outcome을 *입증하는* 구체적인 test / command / artifact입니다. |
| `constraints` | 변경되거나 regression이 발생해서는 안 되는 항목입니다. |
| `boundaries` | 범위에 포함되는 file, dir, tool 또는 system입니다. |
| `stop_when` | Hermes가 멈추고 입력을 요청해야 하는 조건입니다. |

Contract가 설정되면 두 prompt가 모두 바뀝니다. **continuation prompt**는 agent에게 verification surface를 목표로 삼고 constraints를 지키도록 안내하며, **judge prompt**는 구체적인 증거(command result, file excerpt, test output)가 있는 verification criterion이 충족된 경우에만 `done`으로 판단합니다. 단순히 "완료된 것 같다"는 주장은 충분하지 않습니다. 이는 `/goal`에서 가장 흔한 실패 모드인 조기 완료 또는 불완전하게 정의된 목표에 대한 끝없는 continuation을 직접 줄입니다.

### Contract를 설정하는 두 가지 방법

**1. Hermes가 작성하도록 하기** (권장 — Codex의 "let the agent draft the goal" 팁을 적용한 방식):

```
/goal draft Migrate the auth service from session cookies to JWT
```

Hermes는 `goal_judge` auxiliary model을 통해 한 줄짜리 목표를 full contract로 확장하고, 검토하거나 각 field를 더 구체화할 수 있도록 결과를 표시한 뒤 설정합니다. aux model을 사용할 수 없으면 일반적인 free-form goal로 대체합니다. 작성 기능의 문제로 목표 설정이 차단되지는 않습니다.

**2. Inline으로 작성하기** — `field: value` 줄을 사용합니다.

```
/goal Migrate auth to JWT
verify: pytest tests/auth passes
constraints: keep the /login response shape unchanged
boundaries: only touch services/auth and its tests
stop when: a DB schema migration is required
```

첫 번째 비-field 줄이 goal headline이 됩니다. 인식되는 field prefix(`verify:`, `verified by:`, `constraints:`, `preserve:`, `boundaries:`, `scope:`, `stop when:`, `blocked:`, …)가 있으면 contract에 해당 내용을 채웁니다. 일반 목표에 우연히 콜론이 포함된 경우(`Fix bug: the parser drops commas`)에는 변형하지 않습니다. 알려진 field prefix만 추출합니다.

활성 contract를 확인하려면 `/goal show`를 사용하세요. Contract는 목표와 함께 `SessionDB.state_meta`에 저장되므로 `/resume` 후에도 유지됩니다. 이 기능이 추가되기 전의 기존 목표는 변경 없이 불러옵니다(contract 없음). Contract와 `/subgoal` criteria는 함께 구성됩니다. Subgoal은 contract에 추가 criteria로 합쳐지며 judge는 그것까지 모두 충족해야 합니다.

## 목표 중간에 criteria 추가하기: `/subgoal`

목표가 활성화된 동안 `/subgoal <text>`를 사용하면 루프를 초기화하지 않고 acceptance criteria를 추가할 수 있습니다. 호출할 때마다 goal의 subgoal list에 번호가 매겨진 항목 하나가 추가됩니다. 다음 턴에 agent가 보게 되는 **continuation prompt**에는 원래 목표와 함께 "Additional criteria the user added mid-loop" block이 포함되고, **judge prompt**는 모든 subgoal을 고려하도록 다시 작성됩니다. 원래 objective **와** 모든 subgoal이 충족될 때까지 목표는 완료로 표시되지 않습니다.

| Command | 동작 |
|---|---|
| `/subgoal <text>` | 활성 목표에 새 criterion을 추가합니다. 활성 `/goal`이 필요합니다. |
| `/subgoal` (인자 없음) | 현재 번호가 매겨진 subgoal list를 표시합니다. |
| `/subgoal remove <N>` | N번째 subgoal을 삭제합니다(1부터 시작). |
| `/subgoal clear` | 원래 목표는 유지하면서 모든 subgoal을 삭제합니다. |

Subgoal은 `SessionDB.state_meta`에서 목표와 함께 저장되므로 `/resume` 후에도 유지됩니다. 새 `/goal <text>`을 설정하면 목표가 교체되고 subgoal list가 삭제됩니다. `/goal clear`도 동일하게 동작합니다.

루프를 시작한 뒤("실패하는 test를 수정해 줘") 중간에 "방금 수정한 bug에 대한 regression test도 추가해 줘"가 필요해졌을 때 사용하세요. `/subgoal add a regression test`는 실행 중인 루프를 중단하지 않고 성공 criteria를 강화합니다.

## Quality gates

Completion contract는 judge를 더 엄격하게 만들지만, judge는 여전히 prose를 읽는 LLM입니다. **Quality gate**는 더 강력합니다. 목표가 완료되려면 반드시 exit 0을 반환해야 하는 deterministic shell command입니다. Prime-Agent의 bounded autonomous mode(`--autonomous-gate`)에서 영감을 받았습니다.

```
/goal Fix the flaky session tests
/goal gate add scripts/run_tests.sh tests/hermes_cli/test_goals.py
```

매 턴 다음 순서로 동작합니다.

1. **Gate가 judge보다 먼저 실행됩니다.** Gate 하나라도 실패하면 judge를 호출하지 않습니다. 이는 목표가 완료되지 않았다는 deterministic evidence입니다. Gate의 exit code와 output tail(마지막 약 3 KB)이 continuation prompt에 포함되므로 agent는 실제 실패를 기준으로 다음 작업을 수행합니다.
2. **모든 gate 통과 → 일반 judging.** 모든 gate가 통과하면 LLM judge가 평소처럼 done/continue/wait verdict를 결정합니다.
3. **workspace가 변경되지 않으면 재실행하지 않습니다.** Gate가 실패한 뒤 workspace에 변경이 없으면(git fingerprint는 HEAD와 working-tree status로 추적) gate를 재실행하지 않고 기록된 실패를 재생하며 attempt count를 증가시킵니다. Git repository 밖에서는 항상 gate를 다시 실행합니다.
4. **Retry는 제한됩니다.** 각 gate에는 기본적으로 3회의 retry와 5분 timeout이 적용됩니다. Retry를 모두 소진하면 goal은 자동으로 일시 중지되며, 수동으로 수정하거나 gate를 삭제하거나 `/goal resume`하라는 메시지가 표시됩니다.

Gate는 목표와 함께 `SessionDB.state_meta`에 저장되므로 `/resume` 후에도 유지됩니다. Gate 관리(`/goal gate …`)는 gateway에서 실행 중에도 안전합니다. Gate는 turn boundary에서만 실행됩니다.

Gate와 contract는 함께 구성됩니다. Contract는 agent가 무엇을 목표로 할지 정하고, gate는 "완료"를 기계적으로 확인합니다. 둘 다 사용하면 gate가 먼저 실행됩니다.

## Background process 대기: 자동 처리

일부 목표는 자체적으로 실행되며 몇 분이 걸리는 작업에 의존합니다. 예를 들어 PR의 CI, 긴 build, deploy, rate-limit cooldown 등이 있습니다. 도움 없이 goal loop를 사용하면 agent는 매 턴마다 "완료됐나?"를 확인하며 busy-work를 반복하게 됩니다.

**이 작업은 자동으로 처리됩니다.** 매 턴 judge에는 agent의 live background process(`terminal(background=true)` registry — pid, session id, command, uptime, recent output, `watch_patterns` / `notify_on_complete` trigger)가 목표 및 agent 응답과 함께 제공됩니다. Agent의 진행이 그중 하나를 실제로 기다려야 하는 상황이면 judge는 `continue` 대신 `wait` verdict를 반환하고 loop가 대기 상태가 됩니다. 대기가 충족될 때까지 다음 턴을 건너뛰며( judge 호출 없음, continuation 없음, 턴 미소모), 이후 결과를 받은 상태로 정상 재개합니다. Judge는 time basis(`wait_for_seconds`)로도 대기할 수 있습니다. `/goal status`에서는 대기 중일 때 `⏳ Goal (parked …)`가 표시됩니다.

Judge는 process 자체의 signal을 보고 적절한 wait 유형을 선택합니다.

- **`wait_on_session <id>`** — process의 자체 trigger가 발생하면 해제됩니다. process가 종료되거나, `watch_patterns`로 시작된 경우 pattern이 일치하면 해제됩니다. 장시간 실행되는 watcher / server / poller가 중간에 signal을 보내고 스스로 종료되지 않을 수 있는 경우에 사용합니다(예: `BUILD SUCCESSFUL`을 출력하고 계속 실행되는 build, 또는 `notify_on_complete` watcher).
- **`wait_on_pid <pid>`** — process가 종료될 때만 해제됩니다.
- **`wait_for_seconds <n>`** — 지정한 지연 시간이 지나면 해제됩니다.

이 대기 방식은 직접 입력할 필요가 없습니다. Judge가 process context를 보고 결정합니다. 수동 command는 override로 존재합니다.

| Command | 동작 |
|---|---|
| `/goal wait <pid> [reason]` | 해당 PID의 process가 종료될 때까지 loop를 수동으로 대기시킵니다. |
| `/goal unwait` | judge 또는 수동으로 설정된 wait barrier를 지우고 즉시 재개합니다. |

Barrier(pid 또는 time 기반)는 목표와 함께 저장되므로 `/resume` 후에도 유지됩니다. `/goal pause`, `/goal resume`, `/goal clear`는 모두 barrier를 삭제합니다. Barrier가 설정될 때 PID가 이미 종료되었거나 대기 중에 종료되면 다음 확인 시 barrier가 해제됩니다. 오래된 barrier가 loop를 영원히 막을 수는 없습니다.

일반적인 흐름은 다음과 같습니다. Agent가 PR을 push하고 `terminal(background=true, notify_on_complete=true)`로 CI watcher를 시작한 뒤 "CI를 확인하고 있다"고 보고합니다. Judge는 watcher process가 아직 실행 중인 것을 보고 해당 pid를 기준으로 wait를 반환하고, loop는 조용히 대기합니다. CI가 끝나는 즉시 loop가 다시 시작되어 실제 결과를 기준으로 목표를 판단합니다.

## 동작 세부 사항

### Judge

매 턴이 끝나면 Hermes는 다음 정보를 사용해 auxiliary model을 호출합니다.

- 지속적인 목표 텍스트
- agent의 가장 최근 최종 응답(텍스트의 마지막 약 4 KB)
- judge가 strict one-line JSON으로 응답하도록 지시하는 system prompt: `{"verdict": "done" | "continue" | "wait", "reason": "<one-sentence rationale>"}` (wait verdict에는 `wait_on_session` / `wait_on_pid` / `wait_for_seconds`가 추가됩니다. 기존 `{"done": <bool>, "reason": "..."}` 형식도 계속 지원됩니다.)

Judge는 의도적으로 보수적입니다. 응답이 목표 완료를 **명시적으로** 확인하거나, 최종 deliverable이 분명히 생성되었거나, 목표를 달성할 수 없는 상태로 막혔을 때(불가능한 작업에 예산을 소모하지 않도록 DONE으로 처리)에만 goal을 `done`으로 표시합니다.

### Fail-open semantics

Judge에서 오류가 발생하면(network blip, 잘못된 응답, aux client를 사용할 수 없는 경우) Hermes는 verdict를 `continue`로 처리합니다. 고장 난 judge 때문에 진행이 막히지 않도록 하며, **turn budget**이 실제 backstop 역할을 합니다.

### Turn budget

기본값은 continuation turn 20회입니다(`config.yaml`의 `goals.max_turns`). 예산에 도달하면 Hermes는 자동으로 일시 중지하고 다음과 같은 방법을 정확히 안내합니다.

```
⏸ Goal paused — 20/20 turns used. Use /goal resume to keep going, or /goal clear to stop.
```

`/goal resume`은 카운터를 0으로 초기화하므로, 제한된 단위로 계속 작업할 수 있습니다.

### 사용자의 메시지는 항상 우선합니다

목표가 활성화된 동안 사용자가 실제 메시지를 보내면 continuation loop보다 우선 처리됩니다. CLI에서는 사용자의 메시지가 queued continuation보다 먼저 `_pending_input`에 들어가고, gateway에서는 adapter FIFO를 통해 동일한 방식으로 처리됩니다. 턴이 끝난 뒤 judge가 다시 실행되므로 사용자의 메시지가 우연히 목표를 완료했다면 judge가 이를 감지하고 중지합니다.

### 중간 실행 안전성(gateway)

Agent가 이미 실행 중이어도 `/goal status`, `/goal pause`, `/goal clear`, `/goal wait`, `/goal unwait`는 안전하게 실행할 수 있습니다. 현재 턴을 중단하지 않고 control-plane state만 변경합니다. 실행 중 새 목표를 설정하는 `/goal <new text>`는 `/stop`을 먼저 사용하라는 메시지와 함께 거부됩니다. 이전 continuation이 새 목표와 경쟁할 수 있기 때문입니다.

### Persistence

Goal state는 `SessionDB.state_meta`에 `goal:<session_id>`를 key로 저장됩니다. 따라서 `/resume`하면 중단된 지점(활성, 일시 중지, 완료 상태 그대로)에서 다시 시작할 수 있습니다. 목표를 설정하고 노트북을 닫았다가 다음 날 돌아와 `/resume`해도 정확히 그대로 유지됩니다.

### Prompt cache

Continuation prompt는 history에 추가되는 일반 user-role message입니다. System prompt를 변경하거나, toolset을 교체하거나, conversation을 어떤 방식으로든 수정해 Hermes의 prompt cache를 무효화하지 않습니다. 20턴 목표 실행 비용은 일반 대화 20턴과 cache 측면에서 동일합니다.

## Configuration

`~/.hermes/config.yaml`에 다음을 추가하세요.

```yaml
goals:
  # Max continuation turns before Hermes auto-pauses and asks you to
  # /goal resume. Default 20. Lower this if you want tighter loops;
  # raise it for long-running refactors.
  max_turns: 20
```

### Judge model 선택

Judge는 `goal_judge` auxiliary task를 사용합니다. 기본적으로 main model로 resolve됩니다([Auxiliary Models](/user-guide/configuration#auxiliary-models) 참고). 비용을 줄이기 위해 judge를 저렴하고 빠른 model로 route하려면 override를 추가하세요.

```yaml
auxiliary:
  goal_judge:
    provider: openrouter
    model: google/gemini-3-flash-preview
```

Judge call은 작고(출력 약 200 token) 턴마다 한 번 실행되므로, 저렴하고 빠른 model이 보통 적합합니다.

## Example walkthrough

```
You: /goal Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

  ⊙ Goal set (20-turn budget): Create four files /tmp/note_{1..4}.txt, one per turn, each containing its number as text

Hermes: Creating /tmp/note_1.txt now.
  💻 echo "1" > /tmp/note_1.txt   (0.1s)
  I've created /tmp/note_1.txt with the content "1". I'll continue with the remaining files on the next turn as you specified.

  ↻ Continuing toward goal (1/20): Only 1 of 4 files has been created; 3 files remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "2" > /tmp/note_2.txt   (0.1s)
  Created /tmp/note_2.txt. Two more to go.

  ↻ Continuing toward goal (2/20): 2 of 4 files created; 2 remain.

Hermes: [Continuing toward your standing goal]
  💻 echo "3" > /tmp/note_3.txt   (0.1s)
  Created /tmp/note_3.txt.

  ↻ Continuing toward goal (3/20): 3 of 4 files created; 1 remains.

Hermes: [Continuing toward your standing goal]
  💻 echo "4" > /tmp/note_4.txt   (0.1s)
  All four files have been created: /tmp/note_1.txt through /tmp/note_4.txt, each containing its number.

  ✓ Goal achieved: All four files were created with the specified content, completing the goal.

You: _
```

네 번의 턴, `/goal` 한 번의 호출, 사용자가 "계속해"라고 말할 필요가 전혀 없습니다.

## Judge가 잘못 판단할 때

어떤 judge도 완벽하지 않습니다. 주의할 두 가지 실패 모드는 다음과 같습니다.

**False negative — 실제로는 완료되었지만 judge가 continue라고 판단하는 경우.** Turn budget이 이를 방지합니다. `⏸ Goal paused`가 표시되면 `/goal clear`를 사용하거나 새 메시지를 보내면 됩니다.

**False positive — 아직 작업이 남았지만 judge가 done이라고 판단하는 경우.** `✓ Goal achieved`가 표시되었는데도 작업이 남아 있다면 후속 메시지를 보내 계속 진행하거나, 목표를 더 구체적으로 다시 설정하세요: `/goal <more specific text>`. Judge의 system prompt는 false negative가 false positive보다 드물도록 의도적으로 보수적으로 작성되었습니다.

Judge verdict가 납득되지 않으면 `↻ Continuing toward goal` 또는 `✓ Goal achieved` 줄의 reason text에서 judge가 무엇을 보았는지 정확히 확인할 수 있습니다. 이는 대개 목표가 모호했는지 또는 model의 응답이 문제였는지 진단하는 데 충분합니다.

## Attribution

`/goal`은 **Ralph loop** pattern에 대한 Hermes의 구현입니다. 여러 턴 동안 목표를 유지하고, 달성할 때까지 멈추지 않으며, create/pause/resume/clear control을 제공하는 user-facing design은 OpenAI Codex team의 Eric Traut이 [Codex CLI 0.128.0](https://github.com/openai/codex)에 대중화하고 구현했습니다. Hermes의 구현은 독립적입니다(중앙 `CommandDef` registry, `SessionDB.state_meta` persistence, auxiliary-client judge, gateway 측 adapter-FIFO continuation). 하지만 아이디어는 그들의 것입니다. 공로는 공로자에게.

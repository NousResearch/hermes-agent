---
title: "Kanban Codex Lane"
sidebar_label: "Kanban Codex Lane"
description: "Use when a Hermes Kanban worker wants to run Codex CLI as an isolated implementation lane while Hermes keeps ownership of task lifecycle, reconciliation, tes..."
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Kanban Codex Lane

Hermes Kanban 워커가 작업의 수명 주기, 조정(reconciliation), 테스트 및 핸드오프에 대한 소유권을 유지하면서 Codex CLI를 격리된 구현 레인(implementation lane)으로 실행하고자 할 때 사용합니다.

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/autonomous-ai-agents/kanban-codex-lane` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Tags | `kanban`, `codex`, `worktrees`, `autonomous-agents`, `prediction-market-bot` |
| Related skills | [`kanban-worker`](/docs/user-guide/skills/bundled/devops/devops-kanban-worker), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Kanban Codex Lane

## 개요

이 스킬은 Kanban 워커를 위한 가벼운 Hermes+Codex 듀얼 레인 규칙을 정의합니다. Hermes는 항상 작업의 소유자입니다. Hermes는 `kanban_show`를 호출하고, Codex가 적절한지 결정하며, 격리된 작업 공간(workspace)을 생성하거나 선택하고, Codex를 시작 및 모니터링하며, 모든 변경 사항(diff)을 조정(reconcile)하고, 검증을 실행한 뒤, 최종적으로 `kanban_complete` 또는 `kanban_block` 핸드오프를 작성합니다. Codex는 오직 입력 레인 역할만 수행합니다. Codex의 결과물은 작업 완료 신호가 아니며, 신뢰할 수 있는 리뷰어가 아니고, 내구성이 있는 Kanban 상태를 직접 작성하는 것이 허용되지 않습니다.

이 규칙은 Hermes 워커가 디스패처(dispatcher)를 변경하지 않고도 제한된 범위의 구현 도움을 위해 Codex를 사용할 수 있도록 존재합니다. 디스패처는 여전히 Hermes 워커를 생성(spawn)해야 합니다. 워커는 자신의 실행 주기 내에서 선택적으로 Codex를 생성한 다음, 독립적인 검토와 테스트를 거친 후 레인의 결과를 승인(accept), 부분 승인(partially accept) 또는 거부(reject)할 수 있습니다.

## 사용 시기

다음 조건이 모두 충족될 때 Codex 레인을 사용하세요:

- Kanban 작업이 명확한 인수 기준(acceptance criteria)을 가진 코딩, 리팩터링, 문서화, 테스트 또는 기계적인 마이그레이션 작업일 때.
- Hermes가 한 번의 실행으로 범위가 제한된 diff를 평가할 수 있을 때.
- 저장소를 격리된 git worktree/branch에 복사하거나 체크아웃할 수 있을 때.
- Codex가 종료된 후 Hermes가 직접 관련 테스트를 실행할 수 있을 때.
- 변경해서는 안 되는 파일과 모든 안전 제약 사항을 프롬프트에 명시할 수 있을 때.

다음 조건 중 하나라도 해당하면 Codex 레인을 사용하지 마세요:

- 작업에 Kanban 본문에 아직 캡처되지 않은 인간의 판단이 필요할 때.
- 워커에게 저장소 접근 권한, Codex 인증, 또는 결과를 조정할 시간이 부족할 때.
- 변경 사항이 비밀 정보(secrets), 자격 증명 저장소, 개인 사용자 데이터, 또는 프로덕션 주문 입력 시스템(production order-entry systems)을 건드릴 때.
- 또 다른 에이전트를 생성하는 것보다 작고 직접적인 수정이 더 빠르고 안전할 때.
- 작업이 조사(research) 전용이며 diff 대신 서면 핸드오프를 생성해야 할 때.
- 워커가 Codex의 자체 보고에만 의존하여 완료(Done) 표시를 하려는 유혹을 받을 때.

## 소유권 규칙 (Ownership Rules)

1. Hermes가 Kanban 수명 주기를 소유합니다. Codex는 워커를 대신하여 `kanban_complete`, `kanban_block`, `kanban_create`, 게이트웨이 메시징 또는 기타 Hermes 보드 CLI를 호출해서는 안 됩니다.
2. Hermes가 최종 승인을 소유합니다. 검토 및 검증되기 전까지 Codex의 커밋/diff는 신뢰할 수 없는 패치로 취급하세요.
3. Hermes가 테스트 실행을 소유합니다. Codex가 테스트를 실행할 수는 있지만, 이는 참고용일 뿐입니다. 저장소의 공식 래퍼(canonical wrapper)를 사용하여 Hermes에서 필수 검증을 반복하세요.
4. Hermes가 안전을 소유합니다. 테스트가 통과하더라도 Codex가 안전 경계, 위험 게이트(risk gates), 실제 거래(live trading) 동작 또는 비밀 정보 처리 방식을 변경하는 경우 레인을 거부하세요.
5. Hermes가 정리를 소유합니다. 더 이상 필요하지 않을 때 멈춰 있는 Codex 프로세스를 종료하고 임시 worktree를 제거하세요.

## 필수 Worktree 및 Branch 패턴

절대로 공유된 지저분한(dirty) 체크아웃에서 직접 Codex를 실행하지 마세요. Kanban 작업과 레인을 연결하고 신뢰할 수 없는 편집을 격리시키는 branch/worktree 이름을 사용하세요.

권장되는 변수:

```bash
TASK_ID="${HERMES_KANBAN_TASK:-t_manual}"
REPO="/path/to/repo"
BASE="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
SAFE_TASK="$(printf '%s' "$TASK_ID" | tr -cd '[:alnum:]_-')"
BRANCH="codex/${SAFE_TASK}/$(date -u +%Y%m%d%H%M%S)"
WORKTREE="/tmp/${SAFE_TASK}-codex-lane"
```

격리된 레인 생성:

```bash
git -C "$REPO" fetch --all --prune
git -C "$REPO" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
git -C "$WORKTREE" status --short --branch
```

현재 Kanban 작업 공간이 이 작업을 위해 생성된 격리된 git worktree이고, 의도적인 Hermes 편집을 제외하고 `git status --short`가 깨끗한(clean) 경우에만 그 안에 형제(sibling) Codex 브랜치를 생성할 수 있습니다. 그렇지 않은 경우에는 별도의 임시 worktree를 생성하고 조정을 마친 후 승인된 커밋을 체리픽(cherry-pick)하거나 다시 복사하세요.

조정 후 정리(Cleanup):

```bash
git -C "$REPO" worktree remove "$WORKTREE"
git -C "$REPO" branch -D "$BRANCH"  # 승인된 커밋이 복사/체리픽되었거나 의도적으로 거부된 후에만 실행
```

검토를 위한 아티팩트로 worktree가 필요하다면 유지하세요. `codex_lane.artifacts`에 이를 기록하고 핸드오프에 언급하세요.

## Codex 역량 검사 (Capability Checks)

Codex를 생성하기 전에 이 검사를 실행하세요. Codex가 없는 것은 레인을 건너뛸 정상적인 이유이며, Hermes가 직접 작업을 수행할 수 있다면 작업 차단 요인이 아닙니다.

```bash
command -v codex
codex --version
codex features list | grep -i goals || true
```

`/goal` 지원이 필요한 경우 가용성을 확인한 후에만 기능 플래그(feature flag)를 활성화하거나 실행하세요:

```bash
codex features enable goals || true
codex --enable goals --version
```

인증은 `OPENAI_API_KEY` 또는 Codex CLI OAuth 상태(주로 `~/.codex/auth.json`)를 통해 이루어집니다. 토큰 파일을 출력하지 마세요. `OPENAI_API_KEY`가 없다고 해서 인증을 사용할 수 없는 것은 아닙니다.

## 모드 선택 (Mode Selection)

Codex가 자체적으로 종료되어야 하는 범위가 제한된 일회성(one-shot) 편집의 경우 `codex exec`를 사용하세요:

```python
terminal(
    command="codex exec --full-auto '$(cat /tmp/codex_prompt.md)'",
    workdir=WORKTREE,
    background=True,
    pty=True,
    notify_on_complete=True,
)
```

내구성 있는 목표 추적이 도움이 되는 더 넓은 다단계(multi-step) 작업에만 Codex `/goal`을 사용하세요. 기능이 기본적으로 비활성화된 경우 PTY/tmux 세션에서 대화식으로 시작하거나 `codex --enable goals`를 사용하여 시작하세요. 목표(goal objective)를 자립적으로 유지하세요: 저장소 경로, 작업 ID, 안전 제약 조건, 허용된 범위, 인수 기준, 테스트 및 커밋 기대치를 포함합니다.

Codex에 붙여넣을 예제 `/goal` 목표 텍스트:

```text
/goal Work in this repository only: <WORKTREE>. Task: <TASK_ID> <TITLE>.
Hermes owns the Kanban lifecycle; do not call Hermes kanban tools or messaging.
Create small commits on branch <BRANCH>. Follow the PMB safety constraints in the prompt.
Run the requested verification commands and report exact outputs. Stop after producing a diff and summary.
```

prediction-market-bot 또는 안전에 민감한 저장소에서는 `--yolo`를 사용하지 마세요. 격리된 worktree 내에서 `--full-auto`를 선호한 다음, Hermes 조정을 신뢰하세요.

## 프롬프트 구성 (Prompt Construction)

prediction-market-bot 작업의 경우 `templates/pmb-codex-lane-prompt.md`에 링크된 템플릿을 사용하세요. 다른 저장소의 경우 동일한 구조를 유지하고 PMB 전용 안전 블록을 저장소별 불변 사항(invariants)으로 교체하세요.

모든 Codex 프롬프트에는 다음이 포함되어야 합니다:

- `task_id`, 제목 및 전체 Kanban 인수 기준(acceptance criteria).
- 저장소 경로, worktree 경로, 브랜치 이름 및 허용된 파일 범위.
- 명시적 진술: Hermes가 Kanban 수명 주기를 소유합니다. Codex는 입력 레인 역할만 수행합니다.
- 필수 출력: 간결한 요약, 변경된 파일, 커밋, 실행된 테스트 및 알려진 위험(risks).
- 금지된 조치: 비밀 정보 접근, 외부 메시징, 보드(board) 변경, 관련 없는 리팩터링, 요구되지 않은 종속성(dependency) 업그레이드.
- Codex가 실행할 수 있는 검증 명령어와 이후 Hermes가 실행할 명령어.

PMB의 경우, 다음 필수 안전 제약 조건을 원문 그대로 포함하세요:

```text
PMB safety constraints:
- live-SIM is paper-only; do not add or enable live REST order entry.
- Never use market orders.
- Do not add execution crossing or bypass price/risk checks.
- Do not fake passive fills, fills, PnL, order states, or reconciliation evidence.
- Do not weaken risk gates, limits, kill switches, or fail-closed behavior.
- Keep research/selection outside the C++ hot path unless explicitly requested.
- Do not read, print, write, or require secrets/tokens/credentials.
```

## 모니터링, 시간 초과 및 강제 종료 (Kill) 동작

PTY 및 완료 알림과 함께 백그라운드에서 장기 Codex 레인을 시작합니다:

```python
result = terminal(
    command="codex exec --full-auto '$(cat /tmp/codex_prompt.md)'",
    workdir=WORKTREE,
    background=True,
    pty=True,
    notify_on_complete=True,
)
session_id = result["session_id"]
```

간섭하지 않고 모니터링합니다:

```python
process(action="poll", session_id=session_id)
process(action="log", session_id=session_id, limit=200)
process(action="wait", session_id=session_id, timeout=300)
```

레인이 2분 이상 걸리는 경우, 몇 분마다 Kanban 하트비트를 보냅니다. 예: `kanban_heartbeat(note="Codex lane running in <WORKTREE>; waiting for tests/diff")`.

강제 종료(Kill) 조건:

- 작업의 남은 런타임 예산 동안 유용한 출력이 없음.
- Codex가 비밀 정보, 프로덕션 자격 증명 또는 외부 권한을 요청함.
- Codex가 worktree 외부의 파일을 수정하려고 시도함.
- Codex가 관련 없는 재작성(rewrites) 또는 종속성 이탈(churn)을 시작함.
- 워커 타임아웃에 근접하여 Codex가 여전히 실행 중이며 안전하고 부분적인 아티팩트가 존재하지 않음.

강제 종료 명령어:

```python
process(action="kill", session_id=session_id)
```

강제 종료 후 `git status --short`를 검사하고, 안전한 경우에만 유용한 패치를 보존하며, 구체적인 `rejected_reason`과 함께 `codex_lane.result: timed_out` 또는 `rejected`를 기록합니다.

## 조정 체크리스트 (Reconciliation Checklist)

Hermes는 Codex 레인 결과를 승인하기 전에 이 체크리스트를 반드시 수행해야 합니다:

- [ ] `git -C <WORKTREE> status --short --branch`가 예상된 파일만 표시함.
- [ ] Hermes가 `git -C <WORKTREE> diff --stat` 및 `git diff`를 검토함.
- [ ] 비밀 정보, 자격 증명, 생성된 캐시, 관련 없는 데이터 또는 로컬 아티팩트가 포함되어 있지 않음.
- [ ] PMB 안전 제약 조건이 보존됨: 실제 REST 주문 입력 없음, 시장가 주문 없음, 체결 교차(execution crossing) 없음, 가짜 패시브 체결/PnL 없음, 위험 게이트 약화 없음, 비밀 정보 없음.
- [ ] Codex 커밋이 깔끔하게 체리픽(cherry-pick) 또는 스쿼시(squash)할 수 있을 만큼 충분히 작음.
- [ ] Hermes가 자체적으로 공식 테스트를 실행함. Hermes Agent의 경우 `scripts/run_tests.sh`를 사용하고 기타 저장소의 경우 문서화된 래퍼를 사용함.
- [ ] Codex가 실행한 테스트는 Hermes가 실행한 테스트와 별도로 나열됨.
- [ ] 승인된 커밋/diff가 Hermes 소유의 workspace/branch에 적용됨.
- [ ] 거부되거나 부분적인 작업에는 구체적인 이유가 있으며, 유용한 경우 아티팩트 경로가 있음.

승인 결과(Acceptance outcomes):

- `accepted`: Codex diff/commits를 검토, 적용 및 검증함.
- `partial`: 편집 또는 체리픽 후 일부 Codex 작업이 승인됨; 거부된 부분은 문서화됨.
- `rejected`: 승인된 Codex 변경 사항이 없음; 이유가 문서화됨.
- `timed_out`: Codex가 레인 예산을 초과함; 유용한 아티팩트가 존재할 수도 있고 그렇지 않을 수도 있음.

## `kanban_complete` 메타데이터 스키마

레인을 고려한 모든 작업에 대해 이 객체를 `metadata.codex_lane` 아래에 포함하세요. Codex를 사용하지 않은 경우 `used: false`로 설정하고 `rejected_reason` 또는 형제 필드인 `notes`에 그 이유를 설명하세요.

```json
{
  "codex_lane": {
    "used": true,
    "mode": "exec | goal | skipped",
    "worktree": "/absolute/path/to/codex/worktree",
    "branch": "codex/t_caa69668/20260508100000",
    "command": "codex exec --full-auto ...",
    "result": "accepted | rejected | partial | timed_out",
    "accepted_commits": ["<sha1>", "<sha2>"],
    "rejected_reason": "완전히 승인된 경우 비어 있음; 그렇지 않으면 구체적인 이유",
    "tests_run": [
      {"command": "scripts/run_tests.sh tests/tools/test_x.py", "exit_code": 0, "owner": "hermes"},
      {"command": "codex-reported: npm test", "exit_code": 0, "owner": "codex"}
    ],
    "artifacts": ["/absolute/path/to/log-or-patch"]
  }
}
```

의도적으로 Codex를 건너뛴 작업의 경우:

```json
{
  "codex_lane": {
    "used": false,
    "mode": "skipped",
    "worktree": null,
    "branch": null,
    "command": null,
    "result": "rejected",
    "accepted_commits": [],
    "rejected_reason": "직접적인 Hermes 편집이 Codex를 생성하는 것보다 더 작고 안전했습니다.",
    "tests_run": [],
    "artifacts": []
  }
}
```

## 자주 발생하는 함정 (Common Pitfalls)

1. Codex의 자체 보고를 검증으로 취급하는 것. 항상 diff를 검사하고 Hermes에서 테스트를 다시 실행하세요.
2. 사용자의 지저분한 메인 체크아웃에서 Codex를 실행하는 것. 항상 worktree/branch에 격리하세요.
3. Codex가 Kanban을 소유하게 하는 것. Codex는 진행 상황을 요약할 수 있지만, 보드 상태는 Hermes가 기록합니다.
4. 프롬프트에 PMB 안전 불변 사항(invariants)을 잊어버리는 것. 안전 텍스트가 누락된 것은 레인 설정 실패입니다.
5. 빠른 편집을 위해 `/goal`을 사용하는 것. 내구성 있는 다단계 연속 작업이 필요하지 않은 한 `codex exec`를 선호하세요.
6. 이유를 기록하지 않고 멈춘 레인을 종료하는 것. `rejected_reason`은 결정을 설명해야 합니다.
7. 테스트가 통과한다고 해서 관련 없는 광범위한 정리(cleanup) 작업을 승인하는 것. 범위를 벗어난 변경 사항은 거부하거나 체리픽만 하세요.

## 검증 체크리스트 (Verification Checklist)

- [ ] `command -v codex`, `codex --version` 및 선택적인 goals 기능 검사를 마친 후에만 Codex를 건너뛰거나 시작함.
- [ ] Codex는 격리된 worktree/branch에서만 실행됨.
- [ ] 프롬프트에 작업 범위, 소유권 규칙, 적용 가능한 경우 PMB 안전 제약 조건, 검증 명령어가 포함됨.
- [ ] Hermes가 `git diff` 및 안전에 민감한 파일을 검토함.
- [ ] Hermes가 공식 테스트를 독립적으로 실행함.
- [ ] `kanban_complete.metadata.codex_lane`이 위의 스키마를 따름.
- [ ] 임시 프로세스와 불필요한 worktrees가 정리됨.

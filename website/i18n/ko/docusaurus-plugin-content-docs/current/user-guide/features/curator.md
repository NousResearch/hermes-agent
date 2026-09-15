---
sidebar_position: 3
title: "큐레이터"
description: "에이전트가 생성한 스킬의 백그라운드 유지 관리 — 사용 추적, 오래됨 판정, 보관 및 LLM 기반 검토"
---

# 큐레이터

큐레이터는 **에이전트가 생성한 스킬**을 대상으로 백그라운드 유지 관리 작업을 수행합니다. 각 스킬의 조회, 사용, 패치 횟수를 추적하고, 오랫동안 사용되지 않은 스킬을 `active → stale → archived` 상태로 이동하며, 주기적으로 짧은 보조 모델 검토를 실행해 통합 또는 드리프트 패치를 제안합니다.

[자기 개선 루프](/user-guide/features/skills#agent-managed-skills-skill_manage-tool)를 통해 생성된 스킬이 영원히 쌓이지 않도록 하는 기능입니다. 에이전트가 새로운 문제를 해결하고 스킬을 저장할 때마다 해당 스킬은 `~/.hermes/skills/`에 들어갑니다. 유지 관리가 없으면 좁은 범위의 거의 중복된 스킬 수십 개가 쌓여 카탈로그를 오염시키고 토큰을 낭비하게 됩니다.

기본값(`prune_builtins: true`)에서는 큐레이터가 주로 관리하는 에이전트 생성 스킬과 함께, 사용되지 않은 **번들 내장 스킬**(저장소와 함께 제공됨)도 `archive_after_days` 동안 사용되지 않으면 보관할 수 있습니다. [agentskills.io](https://agentskills.io)에서 허브를 통해 설치한 스킬은 항상 제외됩니다. `curator.prune_builtins: false`로 설정하면 번들 스킬에는 절대 손대지 않는 기존의 에이전트 생성 스킬 전용 동작으로 되돌릴 수 있습니다. 또한 큐레이터는 **자동으로 삭제하지 않습니다** — 가장 심각한 결과도 복구 가능한 `~/.hermes/skills/.archive/`로의 보관입니다.

[이슈 #7816](https://github.com/NousResearch/hermes-agent/issues/7816)을 추적합니다.

## 실행 방식

큐레이터는 cron 데몬이 아니라 비활성 상태 확인에 의해 실행됩니다. CLI 세션이 시작될 때와 게이트웨이의 cron-ticker 스레드에서 반복 틱이 발생할 때 Hermes는 다음을 확인합니다.

1. 마지막 큐레이터 실행 이후 충분한 시간이 지났는지(`interval_hours`, 기본값 **7일**), 그리고
2. 에이전트가 충분히 오래 유휴 상태였는지(`min_idle_hours`, 기본값 **2시간**).

두 조건이 모두 충족되면 메모리/스킬 자기 개선 알림과 동일한 방식으로 `AIAgent`의 백그라운드 포크를 실행합니다. 이 포크는 자체 프롬프트 캐시에서 실행되며 활성 대화에는 절대 손대지 않습니다.

:::info 최초 실행 동작
새로 설치한 경우(또는 `hermes update` 후 큐레이터가 도입되기 전 설치에서 처음 틱이 발생한 경우), 큐레이터는 즉시 실행되지 **않습니다**. 최초 관찰 시 `last_run_at`을 "현재"로 기록하고, 한 번의 `interval_hours` 전체 기간이 지난 뒤로 첫 실제 작업을 미룹니다. 이를 통해 큐레이터가 스킬 라이브러리에 손대기 전에 전체 간격 동안 라이브러리를 검토하고, 중요한 스킬을 고정하거나, 완전히 옵트아웃할 수 있습니다.

큐레이터가 실제로 실행되었을 때 무엇을 할지 미리 확인하려면 `hermes curator run --dry-run`을 실행하세요 — 라이브러리를 변경하지 않고 동일한 검토 보고서를 생성합니다.
:::

한 번의 실행은 두 단계로 이루어집니다.

1. **자동 전환** (결정론적이며 LLM을 사용하지 않음). `stale_after_days`(30) 동안 사용되지 않은 스킬은 `stale` 상태가 되고, `archive_after_days`(90) 동안 사용되지 않은 스킬은 `~/.hermes/skills/.archive/`로 이동합니다. 이는 항상 켜져 있는 정리 동작으로, 큐레이터가 활성화되어 있으면 보조 모델 비용 없이 실행됩니다.
   - **고정된 스킬**과 **모든 cron 작업에서 참조하는 스킬**(일시 중지/비활성화된 작업 포함)은 완전히 건너뜁니다 — 느리거나 일시 중지된 일정 때문에 작업에 필요한 스킬이 보관되는 일이 없도록 자동 전환에서 고정된 스킬처럼 취급합니다. 통합 시에는 cron 스킬 참조도 우산 스킬에 맞게 다시 작성합니다.
   - **한 번도 사용되지 않은 스킬**(`use_count == 0`)에는 유예 하한이 적용됩니다. 최소 `stale_after_days`일 이상 지나기 전에는 보관되지 않습니다. 사용 횟수가 0이라는 것은 증거가 없다는 뜻이지, 스킬을 폐기해도 된다는 증거가 아닙니다.
2. **LLM 통합** (반복 횟수 상한이 높은 단일 보조 모델 실행 — 전체 큐레이션 스윕에는 일반적으로 50–100회의 API 호출이 필요함) — **기본적으로 꺼져 있습니다**. `curator.consolidate: true`이면 포크된 에이전트가 에이전트 생성 스킬을 조사하고, `skill_view`로 스킬을 읽으며, 스킬별로 유지, 패치(`skill_manage` 사용), 겹치는 스킬을 클래스 수준의 우산 스킬로 통합, 또는 터미널 도구로 보관할지를 결정할 수 있습니다. 통합에서는 스킬을 완전한 패키지로 취급합니다. 스킬에 `references/`, `templates/`, `scripts/`, `assets/` 또는 해당 경로로 향하는 상대 링크가 있다면 큐레이터는 독립적으로 유지하거나, 필요한 지원 파일을 옮기고 경로를 다시 작성하거나, 전체 패키지를 변경 없이 보관해야 합니다 — `SKILL.md`만 다른 스킬의 `references/` 파일로 평탄화해서는 안 됩니다.

:::info 통합은 옵트인입니다
기본적으로 큐레이터는 **정리**만 수행합니다 — 결정론적 비활성 상태 단계가 스킬을 오래 사용되지 않은 상태로 표시하고 보관합니다. 의견이 개입되는 LLM **통합** 단계(우산 스킬 구축, 겹치는 스킬 병합)는 실행할 때마다 보조 모델 토큰을 사용하고 라이브러리에 광범위한 구조 변경을 가하므로 기본적으로 꺼져 있습니다. `curator.consolidate: true`로 켜거나 `hermes curator run --consolidate`로 필요할 때 한 번 실행하세요.
:::

고정된 스킬은 큐레이터의 자동 전환과 에이전트의 자체 `skill_manage` 도구 모두에서 제외됩니다. 아래 [스킬 고정](#pinning-a-skill)을 참조하세요.

## 구성

모든 설정은 `.env`가 아닌 `config.yaml`의 `curator:` 아래에 둡니다(비밀이 아니기 때문입니다). 기본값:

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7 days
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
  consolidate: false           # LLM umbrella-building pass — opt-in (prune-only by default)
  prune_builtins: true         # archive unused bundled built-in skills too (hub skills always exempt)
```

완전히 비활성화하려면 `curator.enabled: false`로 설정하세요. 항상 실행되는 정리는 유지하면서 LLM 통합을 옵트인하려면 `curator.consolidate: true`로 설정하세요.

### 더 저렴한 보조 모델에서 검토 실행

큐레이터의 LLM 검토 단계는 Vision, Compression, Session Search 등과 함께 실행되는 일반 보조 작업 슬롯인 `auxiliary.curator`를 사용합니다. "Auto"는 "내 주 채팅 모델 사용"을 의미합니다. 대신 이 슬롯을 특정 제공자와 모델로 지정해 검토 단계에만 고정할 수 있습니다.

**가장 쉬운 방법 — `hermes model`:**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → pick "Curator" → pick provider → pick model
```

웹 대시보드의 **Models** 탭에서도 같은 선택기를 사용할 수 있습니다.

**직접 config.yaml에 설정(동일한 결과):**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # generous — reviews can take several minutes
```

`provider: auto`(기본값)로 두면 검토 단계는 주 채팅 모델이 무엇이든 그 모델을 통해 실행됩니다. 이는 다른 모든 보조 작업과 동일한 동작입니다.

:::note 레거시 구성
이전 릴리스에서는 일회성 `curator.auxiliary.{provider,model}` 블록을 사용했습니다. 이 경로도 계속 작동하지만 지원 중단 로그를 출력합니다 — 큐레이터가 다른 모든 보조 작업과 동일한 연결(`hermes model`, 대시보드 Models 탭, `base_url`, `api_key`, `timeout`, `extra_body`)을 공유하도록 위의 `auxiliary.curator`로 마이그레이션하세요.
:::

## CLI

```bash
hermes curator status         # last run, counts, pinned list, LRU top 5
hermes curator run            # trigger a run now (blocks until done). Prune-only unless curator.consolidate: true
hermes curator run --consolidate # force the LLM consolidation pass on for this run, overriding the config default
hermes curator run --background  # fire-and-forget: start the run in a background thread
hermes curator run --dry-run  # preview only — report without any mutations
hermes curator backup         # take a manual snapshot of ~/.hermes/skills/
hermes curator rollback       # restore from the newest snapshot
hermes curator rollback --list     # list available snapshots
hermes curator rollback --id <ts>  # restore a specific snapshot
hermes curator rollback -y         # skip the confirmation prompt
hermes curator pause          # stop runs until resumed
hermes curator resume
hermes curator pin <skill>    # never auto-transition this skill
hermes curator unpin <skill>
hermes curator adopt <skill>    # hand an unmanaged skill to the curator
hermes curator adopt --all-unmanaged   # hand over every unmanaged skill
hermes curator list-unmanaged   # itemize skills with no provenance marker
hermes curator restore <skill>  # move an archived skill back to active
hermes curator list-archived    # list skills currently in ~/.hermes/skills/.archive/
hermes curator archive <skill>  # manually archive a single skill now
hermes curator prune [--days N] # bulk-archive agent-created skills idle >= N days (default 90)
```

## 백업 및 롤백

실제 큐레이터 작업을 실행하기 전에 Hermes는 `~/.hermes/skills/`의 tar.gz 스냅샷을 `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz`에 저장합니다. 작업에서 원하지 않았던 항목이 보관되거나 통합되었다면 한 번의 명령으로 전체 실행을 되돌릴 수 있습니다.

```bash
hermes curator rollback        # restore newest snapshot (with confirmation)
hermes curator rollback -y     # skip the prompt
hermes curator rollback --list # see all snapshots with reason + size
```

롤백 자체도 되돌릴 수 있습니다. 스킬 트리를 교체하기 전에 Hermes는 `pre-rollback to <target-id>` 태그가 붙은 또 다른 스냅샷을 저장하므로, 잘못된 롤백도 `--id`로 해당 스냅샷으로 다시 진행해 취소할 수 있습니다.

`hermes curator backup --reason "before-refactor"`로 언제든 수동 스냅샷을 만들 수도 있습니다. `--reason` 문자열은 스냅샷의 `manifest.json`에 저장되며 `--list`에 표시됩니다.

스냅샷은 디스크 사용량을 제한하기 위해 `curator.backup.keep`(기본값 5) 개로 정리됩니다.

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

자동 스냅샷을 비활성화하려면 `curator.backup.enabled: false`로 설정하세요. 수동 `hermes curator backup` 명령은 백업이 비활성화된 경우에도 먼저 `enabled: true`로 설정해야 작동합니다 — 이 플래그는 두 경로를 대칭적으로 제어하므로 변경 작업에서 실행 전 스냅샷을 실수로 건너뛸 수 없습니다.

`hermes curator status`에는 최근 사용되지 않은 스킬 5개도 표시됩니다 — 다음에 오래됨 상태가 될 가능성이 높은 스킬을 빠르게 확인할 수 있습니다.

동일한 하위 명령을 실행 중인 세션(CLI 또는 게이트웨이 플랫폼) 안의 `/curator` 슬래시 명령으로도 사용할 수 있습니다.

## "에이전트 생성"의 의미

큐레이터는 `~/.hermes/skills/.usage.json`에서 **에이전트 생성**으로 명시적으로 표시된 스킬만 관리합니다. 스킬이 해당 조건을 충족하려면 다음을 모두 만족해야 합니다.

1. 이름이 `~/.hermes/skills/.bundled_manifest`에 없어야 합니다(저장소와 함께 제공되는 번들 스킬).
2. 이름이 `~/.hermes/skills/.hub/lock.json`에 없어야 합니다(허브에서 설치한 스킬).
3. `.usage.json` 항목의 `"created_by": "agent"` 또는 `"agent_created": true`여야 합니다.

현재 이 표식을 설정하는 것은 **백그라운드 자기 개선 검토 포크**뿐입니다 — 주기적인 검토 단계에서 새 우산 스킬을 만들 때 `skill_manage`의 `mark_agent_created()` 호출이 실행됩니다(~에이전트 10턴마다). 백그라운드 포크는 `tools/skill_provenance.py`를 통해 쓰기 출처가 `"background_review"`로 실행되며, 이것이 `mark_agent_created()` 호출을 트리거하는 유일한 경로입니다.

대화 중 포그라운드 에이전트가 `skill_manage(action="create")`로 만드는 스킬에는 에이전트 생성 표식이 설정되지 않습니다 — 사용자가 지시한 것으로 간주하므로 큐레이터가 의도적으로 그대로 둡니다.

:::warning 직접 작성한 스킬은 큐레이션되지 않습니다
`SKILL.md`를 직접 만들었거나 Hermes가 외부 스킬 디렉터리를 가리키도록 했다면 해당 스킬의 `.usage.json` 항목에는 `created_by: null`이 저장되거나(또는 필드가 없음) 표시됩니다. 큐레이터는 이 스킬에 손대지 않습니다. 포그라운드 에이전트가 사용자의 요청에 따라 만든 스킬에도 동일하게 적용됩니다.

**큐레이터가 실제로 관리하는 스킬을 확인하려면** `hermes curator status`를 실행하세요.
에이전트 생성 수가 0이면 큐레이터가 관리하는 스킬이 없으므로 LLM 검토 단계가 건너뛰어지고 보고서에
`Model: (not resolved) via (not resolved)`와 `Duration: 0s`가 표시됩니다.
:::

### 관리되지 않는 스킬 채택

`hermes curator status`는 관리되는 스킬 수와 함께 **관리되지 않는** 스킬 수도 보고합니다.

```
curator-managed skills: 43 total  (agent-created=43  bundled=0)
  active     41
  stale       2
  archived    0

unmanaged (no provenance marker): 112 total
  pre-dates marker    34
  foreground-created  78
  never auto-staled or archived — `hermes curator adopt <name>` hands one over
```

이 112개는 큐레이션-*대상*이 될 수 있지만 수명 주기에서는 영구적으로 보이지 않습니다. 이유는 다음 두 가지 중 하나입니다.

- **표식보다 오래됨** — `created_by`가 존재하기 전에 기록이 작성되어 출처 신호가 전혀 없습니다. 기록만으로는 작성자를 실제로 알 수 없습니다.
- **포그라운드에서 생성됨** — 포그라운드 `skill_manage(create)`는 사용자가 요청한 스킬은 사용자의 것이라는 설계에 따라 표식을 설정하지 않은 채로 둡니다.

따라서 규모가 큰 라이브러리도 대부분이 큐레이션된 것처럼 보이면서 실제로는 많은 부분에 손댈 수 없을 수 있습니다. `adopt`는 **선언**을 통해 이 간극을 메웁니다.

```bash
hermes curator list-unmanaged                    # itemize them, with reasons
hermes curator adopt <name> [<name> ...]         # hand specific skills over
hermes curator adopt --all-unmanaged --dry-run   # preview the full list
hermes curator adopt --all-unmanaged             # hand over everything (prompts)
hermes curator adopt --all-unmanaged --yes       # skip the prompt
```

채택하면 백그라운드 검토 포크가 기록하는 것과 동일한 `created_by: agent` 표식이 기록됩니다. 비활성 상태 시계는 초기화되지 않습니다 — 채택한 스킬은 기존 `last_activity_at`을 유지하므로, 이미 사용을 중단한 라이브러리를 넘겨도 90일 기간이 새로 시작되지 않습니다. 오래 유휴 상태인 채택 스킬은 다음 실행에서 `stale`(또는 `archived`) 상태가 될 수 있습니다. 그것이 바로 채택의 목적입니다.

채택은 자율 **개선**을 가능하게 하는 방법이기도 합니다. 백그라운드 검토 포크는 큐레이터가 관리하지 않는 스킬의 패치를 거부하므로, 사용자의 스킬이 오래되었다고 판단하면 편집하는 대신 채택을 권장합니다. 포그라운드(사용자 지시) 편집에는 영향이 없습니다 — 사용자는 언제든 요청에 따라 자신의 스킬을 편집하게 할 수 있습니다.

:::note `created_by`는 출처 주장이 아니라 정책 플래그입니다
저장된 필드의 이름은 `created_by`이지만, 이 값은 "자율 큐레이션이 이 항목을 수정해도 되는가?"로 사용되며 "이 파일을 누가 작성했는가"를 의미하지 않습니다. 두 질문은 서로 다르고, 표식보다 오래된 기록의 경우 작성자에 대한 답은 복구할 수 없습니다. 모든 `.usage.json`에 이미 이 이름이 저장되어 있으므로 그대로 유지합니다 — 이 값을 정책으로 읽으세요. `hermes curator adopt`는 정책을 변경할 뿐, 파일 작성자가 누구였는지는 말해주지 않습니다.
:::

:::note 출처는 추론하지 않고 선언합니다
채택은 의도적으로 수동 작업입니다. 텔레메트리로는 작성자를 확인할 수 없습니다. 수천 번 패치된 스킬은 에이전트가 **유지 관리한다**는 사실을 보여줄 뿐, 에이전트가 **작성했다**는 뜻은 아닙니다 — Hermes는 사용자가 작성한 스킬을 대신해 자주 편집합니다. "에이전트가 만든 것처럼 보이면 채택"하는 자동 휴리스틱은 결국 사용자가 직접 작성한 것을 보관하게 만들 수 있습니다. `adopt`는 번들, 허브 설치, 외부 및 보호된 내장 스킬(사용자 외의 소유자가 있는 스킬)을 거부합니다.
:::

**에이전트 생성**으로 표시된 스킬은 전체 수명 주기를 따릅니다.

- `active` → (30일 미사용) `stale` → (90일 미사용) `archived`
- 고정된 스킬은 모든 자동 전환을 우회합니다.
- 보관된 스킬은 `hermes curator restore <name>`으로 복구할 수 있습니다.

특정 스킬을 절대 건드리지 않도록 보호하려면 — 예를 들어 직접 작성했고 의존하고 있는 스킬이라면 — `hermes curator pin <name>`을 사용하세요. 다음 섹션을 참조하세요.

## 스킬 고정

고정은 큐레이터의 자동 보관 단계와 에이전트의 `skill_manage(action="delete")` 도구 호출 모두에서 스킬이 삭제되지 않도록 보호합니다. 스킬이 고정되면:

- **큐레이터**는 자동 전환(`active → stale → archived`) 중 해당 스킬을 건너뛰며, LLM 검토 단계에도 그대로 두라는 지시가 전달됩니다.
- **에이전트의 `skill_manage` 도구**는 삭제를 거부하고 `hermes curator unpin <name>`을 사용하라고 안내합니다. 패치와 편집은 계속 가능하므로, 문제가 생길 때마다 고정 해제/재고정 작업을 반복하지 않고도 에이전트가 고정된 스킬의 내용을 개선할 수 있습니다.

다음 명령으로 고정하거나 고정을 해제합니다.

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

이 플래그는 `~/.hermes/skills/.usage.json`에 있는 해당 스킬 항목의 `"pinned": true`로 저장되므로 세션이 바뀌어도 유지됩니다.

cron 작업의 `skills:` 목록에 이름이 있는 스킬도 **자동 전환**에 대해서는 같은 방식으로 보호됩니다(참조가 남아 있는 동안 큐레이터가 해당 스킬을 오래됨/보관 상태로 전환하지 않음). 작업이 일시 중지되었거나 비활성화된 경우에도 마찬가지입니다. `skill_manage delete`도 차단하려면 명시적으로 고정하는 것이 좋습니다.

**에이전트 생성** 스킬만 고정할 수 있습니다 — 번들 및 허브 설치 스킬에 `hermes curator pin`을 사용하면 설명 메시지와 함께 거부됩니다. 허브 설치 스킬은 큐레이터 변경의 대상이 되지 않습니다. 번들 내장 스킬은 `curator.prune_builtins: true`(기본값)일 때만 대상으로 삼으며, 그 경우에도 사용하지 않은 기간이 `archive_after_days`에 도달했을 때 보관만 할 뿐 패치, 통합 또는 삭제하지 않습니다. 번들 스킬을 완전히 제외하려면 `curator.prune_builtins: false`로 설정하세요.

일부 **보호된 내장 스킬**은 `curator.prune_builtins`, 고정 상태, LLM 판단과 관계없이 보관 및 통합이 불가능하도록 코드에 고정되어 있습니다. 이러한 스킬은 예를 들어 `/plan` 슬래시 명령 흐름을 제공하는 `plan`처럼 핵심 UX를 뒷받침하므로, 아무 신호 없이 보관하면 슬래시 명령이 "Unknown command" 오류로 바뀝니다. 보호된 내장 스킬은 큐레이터의 후보 목록에서 완전히 제외되므로 통합 단계에서도 볼 수 없습니다.

"삭제하지 않음"보다 강한 보장이 필요하다면 — 에이전트가 계속 읽기는 하되 스킬의 내용을 완전히 고정하려는 경우처럼 — 편집기로 `~/.hermes/skills/<name>/SKILL.md`를 직접 편집하세요. 고정은 도구를 통한 삭제를 막을 뿐, 사용자가 직접 파일 시스템에 접근하는 것은 막지 않습니다.

## 사용 텔레메트리

큐레이터는 스킬별 항목 하나씩을 가진 사이드카 `~/.hermes/skills/.usage.json`을 유지합니다.

```json
{
  "my-skill": {
    "use_count": 12,
    "view_count": 34,
    "last_used_at": "2026-04-24T18:12:03Z",
    "last_viewed_at": "2026-04-23T09:44:17Z",
    "patch_count": 3,
    "last_patched_at": "2026-04-20T22:01:55Z",
    "created_at": "2026-03-01T14:20:00Z",
    "state": "active",
    "pinned": false,
    "archived_at": null
  }
}
```

다음 시점에 카운터가 증가합니다.

- `view_count`: 에이전트가 스킬에 `skill_view`를 호출할 때.
- `use_count`: 스킬이 대화의 프롬프트에 로드될 때.
- `patch_count`: 스킬에 대해 `skill_manage patch/edit/write_file/remove_file`이 실행될 때.

번들 및 허브 설치 스킬은 텔레메트리 기록에서 명시적으로 제외됩니다.

## 실행별 보고서

모든 큐레이터 실행은 `~/.hermes/logs/curator/` 아래에 타임스탬프가 붙은 디렉터리를 생성합니다.

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # machine-readable: full fidelity, stats, LLM output
    └── REPORT.md     # human-readable summary
```

`REPORT.md`는 특정 실행에서 수행한 작업을 빠르게 확인하는 방법입니다 — 어떤 스킬이 전환되었는지, LLM 검토자가 무엇을 말했는지, 어떤 스킬이 패치되었는지를 보여줍니다. `agent.log`를 grep하지 않고도 감사할 때 유용합니다.

:::note 후보가 없으면 보고서에 `(not resolved)`가 표시됩니다
검토할 **에이전트 생성 스킬이 없으면** LLM 검토 단계 전체를 건너뜁니다. 보고서 헤더에는 `Model: (not resolved) via (not resolved)`와 `Duration: 0s`가 표시됩니다 — 이는 구성 오류나 모델 해결 실패를 의미하지 **않습니다**. 후보가 없어서 모델을 호출하지 않았다는 뜻일 뿐입니다. 자동 전환 단계는 계속 실행되며 평소처럼 수치를 보고합니다.
:::

### 요약의 이름 변경 맵

실행에서 여러 스킬을 하나의 우산 스킬 아래로 통합했거나 거의 중복된 스킬을 병합했다면, 실행 종료 시 출력되는 사용자 대상 요약에 큐레이터가 적용한 모든 `old-name → new-name` 쌍을 보여주는 명시적인 이름 변경 맵이 포함됩니다. 이는 스킬별 전환 행에 더해 제공되므로, JSON 보고서를 비교하지 않고도 이름이 대량으로 변경된 시점을 한눈에 확인할 수 있습니다. 이 안내는 `hermes curator pin`에도 표시되므로, 원한다면 즉시 우산 이름을 고정해 새 레이블을 잠글 수 있습니다.

## 보관된 스킬 복원

큐레이터가 보관했지만 여전히 필요한 스킬이라면:

```bash
hermes curator restore <skill-name>
```

이 명령은 스킬을 `~/.hermes/skills/.archive/`에서 활성 트리로 옮기고 상태를 `active`로 초기화합니다. 이후 같은 이름의 번들 또는 허브 설치 스킬이 설치되어 있다면 복원을 거부합니다(업스트림 스킬을 가리게 되기 때문입니다).

## 환경별 비활성화

큐레이터는 기본적으로 켜져 있습니다. 끄려면:

- **특정 프로필 하나에서만:** `~/.hermes/config.yaml`(또는 활성 프로필의 구성)을 편집하고 `curator.enabled: false`로 설정합니다.
- **한 번의 실행에서만:** `hermes curator pause` — 일시 중지는 세션이 바뀌어도 유지되며, 다시 활성화하려면 `resume`을 사용합니다.

또한 `min_idle_hours`가 지나지 않았으면 큐레이터는 실행을 거부하므로, 작업이 활발한 개발 머신에서는 자연스럽게 한가한 시간에만 실행됩니다.

## 관련 문서

- [스킬 시스템](/user-guide/features/skills) — 스킬의 일반적인 동작과 스킬을 생성하는 자기 개선 루프
- [메모리](/user-guide/features/memory) — 장기 메모리를 유지 관리하는 병렬 백그라운드 검토
- [번들 스킬 카탈로그](/reference/skills-catalog)
- [이슈 #7816](https://github.com/NousResearch/hermes-agent/issues/7816) — 최초 제안 및 설계 논의

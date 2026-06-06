---
sidebar_position: 3
title: "큐레이터 (Curator)"
description: "에이전트가 만든 기술에 대한 백그라운드 유지 관리 — 사용량 추적, 비활성화(staleness), 보관 및 LLM 기반 리뷰"
---

# 큐레이터 (Curator)

큐레이터는 **에이전트가 생성한 기술(skills)**을 위한 백그라운드 유지 관리 프로세스입니다. 각 기술이 얼마나 자주 조회되고, 사용되며, 패치되었는지 추적하고, 오랫동안 사용되지 않은 기술을 `활성(active) → 오래된(stale) → 보관됨(archived)` 상태로 이동시키며, 주기적으로 보조 모델(auxiliary model) 리뷰를 생성하여 통합을 제안하거나 드리프트를 패치합니다.

이것은 [자가 개선 루프(self-improvement loop)](/user-guide/features/skills#agent-managed-skills-skill_manage-tool)를 통해 생성된 기술이 끝없이 쌓이지 않도록 하기 위해 존재합니다. 에이전트가 새로운 문제를 해결하고 기술을 저장할 때마다 해당 기술은 `~/.hermes/skills/`에 저장됩니다. 유지 관리 없이 방치하면 카탈로그를 오염시키고 토큰을 낭비하는 수십 개의 좁고 유사한 중복 기술이 쌓이게 됩니다.

큐레이터는 레포지토리와 함께 제공되는 번들 기술이나 허브에서 설치된 기술(예: [agentskills.io](https://agentskills.io))은 **절대 건드리지 않습니다**. 에이전트가 직접 작성한 기술만 리뷰합니다. 또한 **절대 자동 삭제하지 않습니다**. 최악의 결과는 `~/.hermes/skills/.archive/`로의 보관이며, 이는 복구 가능합니다.

[issue #7816](https://github.com/NousResearch/hermes-agent/issues/7816)를 추적합니다.

## 작동 방식

큐레이터는 크론 데몬(cron daemon)이 아닌 비활동(inactivity) 검사에 의해 트리거됩니다. CLI 세션 시작 시, 그리고 게이트웨이의 크론-티커 스레드 내의 반복적인 틱(tick)에서 Hermes는 다음을 확인합니다:

1. 마지막 큐레이터 실행 이후 충분한 시간이 지났는지 (`interval_hours`, 기본값 **7일**)
2. 에이전트가 충분히 오랫동안 유휴 상태였는지 (`min_idle_hours`, 기본값 **2시간**)

두 조건이 모두 참이면 `AIAgent`의 백그라운드 포크(fork)를 생성합니다 — 이것은 메모리/기술 자가 개선 넛지(nudge)에 사용되는 것과 동일한 패턴입니다. 포크는 자체 프롬프트 캐시에서 실행되며 활성 대화에는 절대 영향을 주지 않습니다.

:::info 첫 실행 동작
새로 설치하거나 `hermes update` 후 이전 큐레이터가 처음으로 작동할 때, 큐레이터는 **즉시 실행되지 않습니다**. 첫 번째 관찰은 `last_run_at`을 "지금"으로 시드(seed)하고 첫 번째 실제 실행을 전체 `interval_hours`만큼 연기합니다. 이를 통해 기술 라이브러리를 검토하고, 중요한 항목을 고정(pin)하거나, 큐레이터가 건드리기 전에 완전히 거부할 수 있는 전체 간격을 확보할 수 있습니다.

실제로 실행되기 전에 큐레이터가 *어떻게* 할지 확인하고 싶다면 `hermes curator run --dry-run`을 실행하세요 — 라이브러리를 변형하지 않고 동일한 리뷰 보고서를 생성합니다.
:::

실행에는 두 가지 단계가 있습니다:

1. **자동 전환 (Automatic transitions)** (결정론적, LLM 없음). `stale_after_days` (30일) 동안 사용되지 않은 기술은 `오래됨(stale)`이 되고, `archive_after_days` (90일) 동안 사용되지 않은 기술은 `~/.hermes/skills/.archive/`로 이동됩니다.
2. **LLM 리뷰 (LLM review)** (단일 보조 모델 패스, `max_iterations=8`). 포크된 에이전트는 에이전트가 생성한 기술을 조사하고, `skill_view`로 기술을 읽을 수 있으며, 기술별로 유지, 패치(`skill_manage` 사용), 중복 기술 통합, 또는 터미널 도구를 통해 보관할지 결정합니다. 통합 시 기술을 전체 패키지로 취급합니다: 기술에 `references/`, `templates/`, `scripts/`, `assets/` 또는 이러한 경로에 대한 상대 링크가 있는 경우, 큐레이터는 이를 독립 실행형으로 유지하거나, 필요한 지원 파일을 재배치하고 경로를 다시 작성하거나, 전체 패키지를 변경하지 않고 보관해야 합니다 — `SKILL.md`만 다른 기술의 `references/` 파일로 평면화하지 않습니다.

고정된(pinned) 기술은 큐레이터의 자동 전환과 에이전트 자체의 `skill_manage` 도구 모두에 대해 접근이 금지됩니다. 아래의 [기술 고정하기](#pinning-a-skill)를 참조하세요.

## 구성

모든 설정은 `config.yaml`의 `curator:` 아래에 있습니다 (`.env`가 아님 — 이것은 비밀이 아닙니다). 기본값:

```yaml
curator:
  enabled: true
  interval_hours: 168          # 7일
  min_idle_hours: 2
  stale_after_days: 30
  archive_after_days: 90
```

완전히 비활성화하려면 `curator.enabled: false`로 설정하세요.

### 더 저렴한 보조 모델에서 리뷰 실행

큐레이터의 LLM 리뷰 패스는 일반적인 보조 작업 슬롯(`auxiliary.curator`)으로, 비전, 압축, 세션 검색 등과 함께 위치합니다. "Auto"는 "기본 채팅 모델 사용"을 의미합니다. 리뷰 패스에 대해 특정 제공자 + 모델을 지정하려면 슬롯을 덮어쓰세요.

**가장 쉬운 방법 — `hermes model`:**

```bash
hermes model                   # → "Auxiliary models — side-task routing"
                               # → "Curator" 선택 → 제공자 선택 → 모델 선택
```

웹 대시보드의 **Models** 탭에서도 동일한 선택기를 사용할 수 있습니다.

**직접 config.yaml 수정 (동일한 효과):**

```yaml
auxiliary:
  curator:
    provider: openrouter
    model: google/gemini-3-flash-preview
    timeout: 600               # 넉넉하게 설정 — 리뷰는 몇 분이 걸릴 수 있습니다
```

`provider: auto` (기본값)로 두면, 다른 모든 보조 작업의 동작과 일치하도록 기본 채팅 모델을 통해 리뷰 패스가 라우팅됩니다.

:::note 레거시 구성
이전 릴리스에서는 일회성 `curator.auxiliary.{provider,model}` 블록을 사용했습니다. 해당 경로는 여전히 작동하지만 지원 중단(deprecation) 로그 라인을 출력합니다 — 큐레이터가 다른 모든 보조 작업과 동일한 배관(`hermes model`, 대시보드 Models 탭, `base_url`, `api_key`, `timeout`, `extra_body`)을 공유할 수 있도록 위의 `auxiliary.curator`로 마이그레이션하세요.
:::

## CLI

```bash
hermes curator status         # 마지막 실행, 카운트, 고정된 목록, LRU(가장 오래 전에 사용된) 상위 5개
hermes curator run            # 지금 리뷰 트리거 (LLM 패스가 끝날 때까지 차단됨)
hermes curator run --background  # 실행 후 잊기: 백그라운드 스레드에서 LLM 패스 시작
hermes curator run --dry-run  # 미리보기 전용 — 변형 없이 보고서만 생성
hermes curator backup         # ~/.hermes/skills/의 수동 스냅샷 생성
hermes curator rollback       # 가장 최신 스냅샷에서 복원
hermes curator rollback --list     # 사용 가능한 스냅샷 나열
hermes curator rollback --id <ts>  # 특정 스냅샷 복원
hermes curator rollback -y         # 확인 프롬프트 건너뛰기
hermes curator pause          # 재개될 때까지 실행 중지
hermes curator resume
hermes curator pin <skill>    # 이 기술을 자동 전환하지 않음
hermes curator unpin <skill>
hermes curator restore <skill>  # 보관된 기술을 다시 활성 상태로 이동
```

## 백업 및 롤백

모든 실제 큐레이터 패스 이전에, Hermes는 `~/.hermes/skills/.curator_backups/<utc-iso>/skills.tar.gz`에 `~/.hermes/skills/`의 tar.gz 스냅샷을 생성합니다. 원하지 않는 보관이나 통합이 수행된 경우, 명령 하나로 전체 실행을 취소할 수 있습니다:

```bash
hermes curator rollback        # 최신 스냅샷 복원 (확인 메시지 포함)
hermes curator rollback -y     # 프롬프트 건너뛰기
hermes curator rollback --list # 모든 스냅샷을 이유 + 크기와 함께 보기
```

롤백 자체도 되돌릴 수 있습니다. 기술 트리를 교체하기 전에 Hermes는 `pre-rollback to <target-id>`로 태그가 지정된 또 다른 스냅샷을 만듭니다. 따라서 잘못된 롤백은 `--id`를 사용하여 해당 스냅샷으로 롤포워드(roll-forward)하여 취소할 수 있습니다.

`hermes curator backup --reason "before-refactor"`와 같이 사용하여 언제든지 수동으로 스냅샷을 찍을 수도 있습니다. `--reason` 문자열은 스냅샷의 `manifest.json`에 저장되며 `--list`에 표시됩니다.

디스크 사용량을 제한하기 위해 스냅샷은 `curator.backup.keep` (기본값 5)으로 잘립니다:

```yaml
curator:
  backup:
    enabled: true
    keep: 5
```

자동 스냅샷을 비활성화하려면 `curator.backup.enabled: false`로 설정하세요. 백업이 비활성화된 경우, `enabled: true`를 먼저 설정해야만 수동 `hermes curator backup` 명령이 작동합니다 — 플래그가 두 경로를 대칭적으로 통제하므로 변경이 일어나는 실행에서 실수로 실행 전 스냅샷을 건너뛸 방법은 없습니다.

`hermes curator status`는 가장 오랫동안 사용되지 않은 상위 5개의 기술도 나열합니다 — 다음에 오래된 상태가 될 가능성이 있는 항목을 빠르게 확인할 수 있습니다.

실행 중인 세션(CLI 또는 게이트웨이 플랫폼) 내에서 `/curator` 슬래시 명령으로 동일한 하위 명령을 사용할 수 있습니다.

## "에이전트 생성"의 의미

큐레이터는 `~/.hermes/skills/.usage.json`에서 명시적으로 **에이전트가 생성한 것(agent-created)**으로 표시된 기술만 관리합니다. 다음 조건이 모두 참일 때 기술이 이에 해당합니다:

1. 이름이 `~/.hermes/skills/.bundled_manifest`에 **없어야 합니다** (레포지토리와 함께 제공된 번들 기술).
2. 이름이 `~/.hermes/skills/.hub/lock.json`에 **없어야 합니다** (허브에서 설치된 기술).
3. `.usage.json` 항목에 `"created_by": "agent"` 또는 `"agent_created": true`가 있어야 합니다.

현재, **백그라운드 자가 개선 리뷰 포크**만이 이 마커를 설정합니다 — 주기적 리뷰 패스(대략 에이전트 턴 10회마다) 중에 새 포괄적 기술(umbrella skill)을 생성할 때. 백그라운드 포크는 `"background_review"`라는 쓰기 출처(write origin)로 실행되며(`tools/skill_provenance.py`를 통해), 이것이 `skill_manage`에서 `mark_agent_created()` 호출을 트리거하는 유일한 경로입니다.

대화 중에 포그라운드 에이전트가 `skill_manage(action="create")`를 통해 생성한 기술은 에이전트 생성으로 **표시되지 않습니다** — 사용자 지시로 간주되며 큐레이터는 의도적으로 이를 건드리지 않습니다.

:::warning 직접 작성한 기술은 큐레이트되지 않습니다
`SKILL.md`를 수동으로 생성하거나 Hermes에 외부 기술 디렉토리를 지정한 경우, 해당 기술은 `created_by: null`(또는 필드 없음)인 `.usage.json` 항목을 갖게 됩니다. 큐레이터는 이를 건드리지 않습니다. 사용자의 요청으로 포그라운드 에이전트가 생성한 기술에도 동일하게 적용됩니다.

**큐레이터가 실제로 관리하는 기술을 확인하려면** `hermes curator status`를 실행하세요.
에이전트 생성 개수가 0이면 현재 큐레이터의 관할권에 있는 기술이 없습니다 — LLM 리뷰 패스를 건너뛰고 보고서에 `Duration: 0s`와 함께 `Model: (not resolved) via (not resolved)`가 표시됩니다.
:::

에이전트가 생성한 기술은 전체 수명 주기를 따릅니다:

- `활성(active)` → (30일 미사용) `오래됨(stale)` → (90일 미사용) `보관됨(archived)`
- 고정된 기술은 모든 자동 전환을 우회합니다
- 보관본은 `hermes curator restore <name>`을 통해 복구할 수 있습니다

특정 기술이 건드려지지 않도록 보호하고 싶다면(예: 의존하는 수동 작성 기술) `hermes curator pin <name>`을 사용하세요. 다음 섹션을 참조하세요.

## 기술 고정하기 (Pinning a skill)

고정(Pinning)은 삭제로부터 기술을 보호합니다 — 큐레이터의 자동 보관 패스와 에이전트의 `skill_manage(action="delete")` 도구 호출 모두로부터 보호합니다. 기술이 고정되면:

- **큐레이터**는 자동 전환(`active → stale → archived`) 중에 이를 건너뛰고, LLM 리뷰 패스에서 이를 건드리지 않도록 지시받습니다.
- **에이전트의 `skill_manage` 도구**는 해당 기술에 대한 `delete`를 거부하고, 사용자에게 `hermes curator unpin <name>`을 안내합니다. 패치 및 편집은 여전히 통과되므로, 고정/고정 해제/다시 고정 절차 없이도 함정(pitfalls)이 발생할 때 에이전트가 고정된 기술의 내용을 개선할 수 있습니다.

다음과 같이 고정 및 고정 해제할 수 있습니다:

```bash
hermes curator pin <skill>
hermes curator unpin <skill>
```

플래그는 `~/.hermes/skills/.usage.json`의 기술 항목에 `"pinned": true`로 저장되므로 세션 간에 유지됩니다.

오직 **에이전트가 생성한(agent-created)** 기술만 고정할 수 있습니다 — 번들로 제공되거나 허브에서 설치된 기술은 애초에 큐레이터의 변형 대상이 아니며, 고정을 시도할 경우 `hermes curator pin`이 설명 메시지와 함께 거부할 것입니다.

에이전트가 여전히 기술을 읽는 동안 내용 전체를 완전히 동결하는 것과 같이 "삭제 방지"보다 더 강력한 보증을 원한다면, 에디터에서 `~/.hermes/skills/<name>/SKILL.md`를 직접 편집하세요. 고정은 도구에 의한 삭제를 방지할 뿐, 사용자의 파일 시스템 접근을 막지는 않습니다.

## 사용량 원격 분석 (Usage telemetry)

큐레이터는 각 기술마다 하나의 항목으로 구성된 사이드카를 `~/.hermes/skills/.usage.json`에 유지합니다:

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

카운터는 다음과 같은 경우에 증가합니다:

- `view_count`: 에이전트가 해당 기술에 대해 `skill_view`를 호출할 때.
- `use_count`: 대화의 프롬프트에 기술이 로드될 때.
- `patch_count`: `skill_manage patch/edit/write_file/remove_file`이 기술에 대해 실행될 때.

번들 기술 및 허브 설치 기술은 명시적으로 원격 분석 기록에서 제외됩니다.

## 실행별 보고서 (Per-run reports)

모든 큐레이터 실행은 타임스탬프가 지정된 디렉토리를 `~/.hermes/logs/curator/` 아래에 작성합니다:

```
~/.hermes/logs/curator/
└── 20260429-111512/
    ├── run.json      # 기계 판독 가능: 전체 충실도, 통계, LLM 출력
    └── REPORT.md     # 인간 판독 가능 요약
```

`REPORT.md`는 해당 실행이 수행한 작업 — 어떤 기술이 전환되었는지, LLM 리뷰어가 무엇을 제안했는지, 어떤 기술이 패치되었는지 — 을 확인하는 빠른 방법입니다. `agent.log`를 검색(grep)하지 않고도 감사(auditing)하기 좋습니다.

:::note 후보가 없습니까? 보고서에 `(not resolved)`로 표시됩니다
큐레이터가 리뷰할 **에이전트 생성 기술**이 없을 때, LLM 리뷰 패스는 완전히 생략됩니다. 보고서 헤더에는 `Duration: 0s`와 함께 `Model: (not resolved) via (not resolved)`가 표시됩니다 — 이것은 구성 오류나 모델 해결 실패를 나타내는 것이 **아닙니다**. 단지 후보가 없어서 어떤 모델도 호출되지 않았음을 의미할 뿐입니다. 자동 전환 단계는 여전히 실행되고 평소처럼 카운트를 보고합니다.
:::

### 요약의 이름 변경 매핑 (Rename map in the summary)

실행이 여러 기술을 포괄적 기술로 통합하거나 중복에 가까운 기술을 병합한 경우, 실행이 끝날 때 출력되는 사용자 표시 요약에는 큐레이터가 적용한 모든 `old-name → new-name` 쌍을 보여주는 명시적 이름 변경 매핑(rename map)이 포함됩니다. 이것은 기술별 전환 행(transition lines)에 추가되므로 이름 변경 작업이 진행될 때 JSON 보고서에서 차이를 찾지 않고 한눈에 파악할 수 있습니다. 이 힌트는 새로운 레이블을 즉시 확정하고 싶은 경우에 대비하여 `hermes curator pin` 아래에도 표시됩니다.

## 보관된 기술 복원하기 (Restoring an archived skill)

여전히 원하는 기술을 큐레이터가 보관한 경우:

```bash
hermes curator restore <skill-name>
```

이 명령은 기술을 `~/.hermes/skills/.archive/`에서 활성 트리로 다시 이동시키고 상태를 `active`로 재설정합니다. 만약 동일한 이름의 번들 기술이나 허브 설치 기술이 그 사이에 설치된 경우, 업스트림을 덮어쓰게 되므로 복원이 거부됩니다.

## 환경별로 비활성화하기 (Disabling per environment)

큐레이터는 기본적으로 켜져 있습니다. 끄려면:

- **한 프로필에 대해서만:** `~/.hermes/config.yaml`(또는 활성 프로필의 구성)을 편집하고 `curator.enabled: false`를 설정합니다.
- **한 번의 실행에 대해서만:** `hermes curator pause` — 일시 중지는 세션 간에 유지됩니다. 다시 활성화하려면 `resume`을 사용하세요.

큐레이터는 또한 `min_idle_hours`가 지나지 않으면 실행을 거부하므로, 활발하게 개발 중인 머신에서는 자연스럽게 조용한 시간대에만 실행됩니다.

## 참고 항목

- [기술 시스템 (Skills System)](/user-guide/features/skills) — 기술이 일반적으로 작동하는 방식과 기술을 생성하는 자가 개선 루프
- [메모리 (Memory)](/user-guide/features/memory) — 장기 메모리를 유지 관리하는 병렬 백그라운드 리뷰
- [번들 기술 카탈로그 (Bundled Skills Catalog)](/reference/skills-catalog)
- [이슈 #7816 (Issue #7816)](https://github.com/NousResearch/hermes-agent/issues/7816) — 초기 제안 및 설계 논의

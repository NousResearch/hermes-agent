---
sidebar_position: 3
title: "영구 메모리"
description: "Hermes Agent가 세션 간에 기억하는 방식 — MEMORY.md, USER.md 및 세션 검색"
---

# 영구 메모리

Hermes Agent에는 세션 간에 유지되는 제한적이고 엄선된 메모리가 있습니다. 이를 통해 에이전트는 사용자의 선호 사항, 프로젝트, 환경 및 학습한 내용을 기억할 수 있습니다.

## 작동 방식

에이전트의 메모리는 두 파일로 구성됩니다.

| 파일 | 용도 | 글자 수 제한 |
|------|---------|------------|
| **MEMORY.md** | 에이전트의 개인 메모 — 환경 정보, 규칙, 학습한 내용 | 2,200자(~800토큰) |
| **USER.md** | 사용자 프로필 — 선호 사항, 의사소통 방식, 기대치 | 1,375자(~500토큰) |

두 파일은 모두 `~/.hermes/memories/`에 저장되며 세션 시작 시 시스템 프롬프트에 고정된 스냅샷으로 주입됩니다. 에이전트는 `memory` 도구를 통해 자체 메모리를 관리하며 항목을 추가, 교체 또는 제거할 수 있습니다.

:::caution Hermes 홈 디렉터리 하나당 에이전트 하나
두 에이전트 프로세스가 동일한 Hermes 홈 디렉터리를 가리키지 않도록 하세요. 메모리 쓰기는 자동으로 이루어지고 세션 시작 시 시스템 프롬프트에 다시 로드되므로, 하나의 홈을 공유하는 두 작성자는 어느 쪽도 직접 작성하지 않은 상태로 서로의 항목을 누적하게 됩니다. 메모리는 설계상 [프로필](/user-guide/profiles)별로 분리됩니다. 두 번째 에이전트에는 별도의 프로필을 제공하고, 메모리를 공유해야 한다면 대신 [외부 메모리 제공자](/user-guide/features/memory-providers)를 사용하세요.
:::

:::info
글자 수 제한은 메모리를 집중된 상태로 유지합니다. 메모리는 **자동으로 압축되지 않습니다**. 쓰기 작업으로 제한을 초과하게 되면 `memory` 도구는 항목을 조용히 삭제하는 대신 오류를 반환합니다. 그러면 에이전트가 같은 턴에 재시도하기 전에 항목을 통합하거나 제거하여 직접 공간을 만듭니다([메모리가 가득 차면 어떻게 되나요](#what-happens-when-memory-is-full) 참조). `replace` 역시 제한을 받습니다. 더 긴 항목으로 교체하면 여전히 초과할 수 있으므로, 새 콘텐츠를 줄이거나 다른 항목을 제거해 맞춰야 합니다.
:::

## 시스템 프롬프트에 메모리가 표시되는 방식

각 세션이 시작되면 디스크의 메모리 항목이 다음과 같이 고정된 블록으로 렌더링되어 시스템 프롬프트에 로드됩니다.

```
══════════════════════════════════════════════
MEMORY (your personal notes) [67% — 1,474/2,200 chars]
══════════════════════════════════════════════
User's project is a Rust web service at ~/code/myapi using Axum + SQLx
§
This machine runs Ubuntu 22.04, has Docker and Podman installed
§
User prefers concise responses, dislikes verbose explanations
```

형식에는 다음이 포함됩니다.

- 어떤 저장소인지(MEMORY 또는 USER PROFILE)를 나타내는 헤더
- 에이전트가 용량을 파악할 수 있도록 사용률과 글자 수
- `§`(섹션 기호) 구분자로 나뉜 개별 항목
- 여러 줄로 구성된 항목

**고정 스냅샷 패턴:** 시스템 프롬프트 주입 내용은 세션 시작 시 한 번 캡처되며 세션 중에는 변경되지 않습니다. 이는 의도된 동작으로, 성능을 위해 LLM의 접두사 캐시를 보존합니다. 에이전트가 세션 중 메모리 항목을 추가하거나 제거하면 변경 사항은 즉시 디스크에 저장되지만 다음 세션이 시작될 때까지 시스템 프롬프트에는 나타나지 않습니다. 도구 응답에는 항상 최신 상태가 표시됩니다.

## 메모리 도구 동작

에이전트는 다음 동작과 함께 `memory` 도구를 사용합니다.

- **add** — 새 메모리 항목 추가
- **replace** — 업데이트된 내용으로 기존 항목 교체(`old_text`를 통한 부분 문자열 일치 사용)
- **remove** — 더 이상 관련 없는 항목 제거(`old_text`를 통한 부분 문자열 일치 사용)

`read` 동작은 없습니다. 메모리 콘텐츠는 세션 시작 시 시스템 프롬프트에 자동으로 주입됩니다. 에이전트는 대화 컨텍스트의 일부로 자신의 메모리를 확인합니다.

### 부분 문자열 일치

`replace`와 `remove` 동작은 짧고 고유한 부분 문자열 일치를 사용하므로 전체 항목 텍스트가 필요하지 않습니다. `old_text` 매개변수에는 정확히 하나의 항목을 식별하는 고유한 부분 문자열만 지정하면 됩니다.

```python
# If memory contains "User prefers dark mode in all editors"
memory(action="replace", target="memory",
       old_text="dark mode",
       content="User prefers light mode in VS Code, dark mode in terminal")
```

부분 문자열이 여러 항목과 일치하면 더 구체적인 일치를 요청하는 오류가 반환됩니다.

## 두 대상 설명

### `memory` — 에이전트의 개인 메모

환경, 워크플로 및 교훈에 대해 에이전트가 기억해야 하는 정보입니다.

- 환경 정보(OS, 도구, 프로젝트 구조)
- 프로젝트 규칙과 설정
- 발견한 도구의 특이점과 해결 방법
- 완료한 작업의 기록
- 효과가 있었던 스킬과 기법

### `user` — 사용자 프로필

사용자의 신원, 선호 사항 및 의사소통 방식을 설명하는 정보입니다.

- 이름, 역할, 시간대
- 의사소통 선호 사항
- 싫어하는 것과 피해야 할 것
- 워크플로 습관
- 기술 숙련도

## 저장할 것과 건너뛸 것

### 저장할 항목(사전 대응)

에이전트가 자동으로 저장하므로 요청할 필요가 없습니다. 다음과 같은 내용을 알게 되면 저장합니다.

- **사용자 선호 사항:** "JavaScript보다 TypeScript를 선호해" → `user`에 저장
- **환경 정보:** "이 서버는 PostgreSQL 16이 설치된 Debian 12에서 실행돼" → `memory`에 저장
- **수정 사항:** "Docker 명령에 `sudo`를 사용하지 마, 사용자가 docker 그룹에 속해 있어" → `memory`에 저장
- **규칙:** "프로젝트는 탭과 120자 줄 너비, Google 스타일 docstring을 사용해" → `memory`에 저장
- **완료한 작업:** "2026-01-15에 데이터베이스를 MySQL에서 PostgreSQL로 마이그레이션했어" → `memory`에 저장
- **명시적 요청:** "내 API 키는 매달 교체한다는 것을 기억해" → `memory`에 저장

### 건너뛸 항목

- **사소하거나 명백한 정보:** "사용자가 Python에 대해 물어봤다" — 유용하기에는 너무 모호함
- **쉽게 다시 찾을 수 있는 사실:** "Python 3.12는 f-string 중첩을 지원한다" — 웹 검색으로 확인 가능
- **원시 데이터 덤프:** 큰 코드 블록, 로그 파일, 데이터 표 — 메모리에 넣기에는 너무 큼
- **세션에만 해당하는 임시 정보:** 임시 파일 경로, 일회성 디버깅 컨텍스트
- **컨텍스트 파일에 이미 있는 정보:** SOUL.md와 AGENTS.md 내용

## 용량 관리

메모리는 시스템 프롬프트의 크기를 제한하기 위해 엄격한 글자 수 제한을 사용합니다.

| 저장소 | 제한 | 일반적인 항목 수 |
|------|---------|----------------|
| memory | 2,200자 | 8~15개 항목 |
| user | 1,375자 | 5~10개 항목 |

### 메모리가 가득 차면 어떻게 되나요

추가하려는 항목으로 제한을 초과하게 되면 도구가 오류를 반환합니다.

```json
{
  "success": false,
  "error": "Memory at 2,100/2,200 chars. Adding this entry (250 chars) would exceed the limit. Consolidate now: use 'replace' to merge overlapping entries into shorter ones or 'remove' stale or less important entries (see current_entries below), then retry this add — all in this turn.",
  "current_entries": ["..."],
  "usage": "2,100/2,200"
}
```

그러면 에이전트는 다음을 수행해야 합니다.
1. 오류 응답에 표시된 현재 항목 읽기
2. 제거하거나 통합할 수 있는 항목 식별
3. `replace`를 사용해 관련 항목을 더 짧게 합치기
4. 새 항목을 추가하기 전에 추가 작업 재시도

**권장 사항:** 시스템 프롬프트 헤더에 용량이 80%를 넘었다고 표시되면 항목을 추가하기 전에 통합하세요. 예를 들어 여러 개의 "프로젝트가 X를 사용함" 항목을 하나의 포괄적인 프로젝트 설명으로 합칠 수 있습니다.

### 좋은 메모리 항목의 실제 예

정보를 조밀하게 담은 간결한 항목이 가장 좋습니다.

```
# Good: Packs multiple related facts
User runs macOS 14 Sonoma, uses Homebrew, has Docker Desktop and Podman. Shell: zsh with oh-my-zsh. Editor: VS Code with Vim keybindings.

# Good: Specific, actionable convention
Project ~/code/api uses Go 1.22, sqlc for DB queries, chi router. Run tests with 'make test'. CI via GitHub Actions.

# Good: Lesson learned with context
The staging server (10.0.1.50) needs SSH port 2222, not 22. Key is at ~/.ssh/staging_ed25519.

# Bad: Too vague
User has a project.

# Bad: Too verbose
On January 5th, 2026, the user asked me to look at their project which is
located at ~/code/api. I discovered it uses Go version 1.22 and...
```

## 중복 방지

메모리 시스템은 동일한 중복 항목을 자동으로 거부합니다. 이미 존재하는 콘텐츠를 추가하려고 하면 "중복 항목이 추가되지 않음"이라는 메시지와 함께 성공을 반환합니다.

## 보안 검사

메모리 항목은 시스템 프롬프트에 주입되기 전에 인젝션 및 유출 패턴 검사를 받습니다. 위협 패턴(프롬프트 인젝션, 자격 증명 유출, SSH 백도어)과 일치하거나 보이지 않는 유니코드 문자를 포함한 콘텐츠는 차단됩니다.

## 세션 검색

MEMORY.md와 USER.md 외에도 에이전트는 `session_search` 도구를 사용해 과거 대화를 검색할 수 있습니다.

- 모든 CLI 및 메시징 세션은 FTS5 전문 검색 기능을 사용하는 SQLite(`~/.hermes/state.db`)에 저장됩니다.
- 검색 쿼리는 DB의 실제 메시지를 반환하며 LLM 요약이나 잘림이 없습니다.
- 활성 메모리에 없더라도 몇 주 전에 나눈 대화를 찾을 수 있습니다.
- 찾은 세션 안에서 앞뒤로 이동할 수도 있습니다.

```bash
hermes sessions list    # Browse past sessions
```

세 가지 호출 형태(탐색 / 스크롤 / 찾아보기)와 응답 형식은 [세션 검색 도구](/user-guide/sessions#session-search-tool)를 참조하세요.

### session_search와 memory 비교

| 기능 | 영구 메모리 | 세션 검색 |
|------|---------|----------------|
| **용량** | 총 약 1,300토큰 | 무제한(모든 세션) |
| **속도** | 시스템 프롬프트에서 즉시 사용 가능 | 약 20ms FTS5 쿼리, 약 1ms 스크롤 |
| **비용** | 모든 프롬프트에서 토큰 비용 발생 | 무료 — LLM 호출 없음 |
| **용도** | 항상 컨텍스트에 있어야 하는 핵심 사실 | 특정 과거 대화 찾기 |
| **관리** | 에이전트가 수동으로 엄선 | 자동 — 모든 세션 저장 |
| **토큰 비용** | 세션당 고정(약 1,300토큰) | 필요할 때만(필요한 경우 검색) |

**Memory**는 항상 컨텍스트에 있어야 하는 중요한 사실을 위한 것입니다. **Session search**는 에이전트가 과거 대화의 구체적인 내용을 기억해야 하는 "지난주에 X에 대해 이야기했나?" 같은 질문을 위한 것입니다.

## 학습 여정(`/journey`)

학습 여정은 Hermes가 학습한 모든 내용을 보여주는 타임라인입니다. 저장된 스킬과 메모리 항목을 시간순으로 표시하며(위쪽이 가장 오래된 항목, 아래쪽이 가장 최신 항목), 재생 가능한 "별자리" 스크러버로 축적 과정을 다시 볼 수 있습니다. 동일한 그래프 데이터가 다음 세 가지 화면에서 사용됩니다.

- **클래식 CLI / 독립 실행형** — `hermes journey`(별칭: `hermes learning`, `hermes memory-graph`)는 터미널에 타임라인을 렌더링합니다. 플래그: `--play`는 축적 과정을 애니메이션으로 재생하고(`--fps`로 조정), `--width`/`--height`는 렌더링 크기를 재정의하며, `--no-color`는 색상을 비활성화하고, `--json`은 원시 그래프 페이로드를 덤프합니다.
- **TUI** — `/journey`(별칭: `/learning`, `/memory-graph`)는 타임라인을 오버레이로 엽니다.
- **데스크톱 앱** — `/journey`는 동일한 노드를 대화형으로 보여주는 별 지도 / 메모리 그래프 패널을 엽니다.

여정은 보는 것뿐 아니라 Hermes가 학습한 내용을 **정리하고 수정하는** 곳이기도 합니다.

| 명령 | 동작 |
|---------|--------------|
| `hermes journey list` | 노드 ID를 나열합니다 — 스킬 이름 및 메모리 청크의 `memory:<source>:<index>` ID. |
| `hermes journey delete <node> [-y]` | 노드를 삭제합니다. 스킬은 **보관**되어 복원할 수 있고, 메모리 청크는 제거됩니다. `-y`는 확인을 건너뜁니다. |
| `hermes journey edit <node>` | 노드의 콘텐츠(스킬의 `SKILL.md` 또는 메모리 청크)를 `$EDITOR`에서 엽니다. |

동일한 `list` / `delete <id>` / `edit <id>` 하위 명령은 CLI의 인채팅 `/journey` 명령에서도 사용할 수 있으며, 데스크톱 패널에서는 노드에서 직접 편집/삭제할 수 있습니다.

## 설정

```yaml
# In ~/.hermes/config.yaml
memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200   # ~800 tokens
  user_char_limit: 1375     # ~500 tokens
  write_approval: false     # false = write freely (default) | true = require approval
```

## 메모리 쓰기 제어(`write_approval`)

기본적으로 에이전트는 턴 후 실행되는 백그라운드 자기 개선 검토를 포함해 메모리를 자유롭게 저장합니다. 먼저 저장을 승인하고 싶다면 `memory.write_approval: true`로 설정하세요. 이는 **포그라운드 턴과 백그라운드 검토 모두**에 적용되는 간단한 on/off 게이트입니다.

| `write_approval` | 동작 |
|------------------|-----------|
| `false`(기본값) | 자유롭게 작성 — 게이트가 꺼져 있습니다(게이트 적용 전 동작). |
| `true` | 무엇이든 저장하기 전에 승인 필요. 대화형 CLI에서는 포그라운드 쓰기를 인라인으로 확인하며(항목이 전체를 읽을 수 있을 만큼 작음), 그 외 모든 곳(메시징 플랫폼, 스크립트, 백그라운드 자기 개선 검토)에서는 `/memory pending`으로 검토할 수 있도록 쓰기를 **대기 상태로 둡니다**. |

> 메모리를 게이트만 하는 것이 아니라 완전히 끄려면 `memory_enabled: false`로 설정하세요.

CLI 또는 모든 메시징 플랫폼에서 대기 중인 쓰기를 검토하세요.

```
/memory pending             # list staged memory writes (auto ones tagged [auto])
/memory approve <id>        # apply one (or 'all')
/memory reject <id>         # drop one (or 'all')
/memory approval on         # turn the gate on (or 'off') and persist it
```

이는 "에이전트가 나에 대해 잘못된 가정을 저장했어"라는 문제에 대한 해결책입니다. `write_approval: true`로 설정하면 모든 저장, 특히 묻지 않고 실행되는 백그라운드 저장이 프로필에 들어가기 전에 사용자의 예/아니요 승인을 기다립니다.

## 백그라운드 검토 알림(`display.memory_notifications`)

턴이 끝나면 백그라운드 자기 개선 검토가 조용히 메모리를 저장하거나 스킬을 업데이트할 수 있습니다. 이는 Hermes의 동의 인식 학습 루프입니다. 반복된 수정 사항과 오래 유지되는 워크플로 교훈은 간결한 메모리 항목 또는 절차적 스킬이 되며, `write_approval`을 사용하면 해당 쓰기가 향후 세션에 영향을 주기 전에 검토를 위해 대기시킬 수 있습니다. 기본적으로 채팅에 짧은 `💾 Memory updated` 줄을 표시하여 이런 일이 일어났음을 알려줍니다. 얼마나 많은 알림을 표시할지 제어하세요.

```yaml
display:
  memory_notifications: on    # off | on (default) | verbose
```

| 값 | 동작 |
|-----------|-----------|
| `off` | 채팅 알림 없음. 검토는 계속 실행되고 계속 쓰지만, 해당 줄은 표시되지 않습니다. |
| `on`(기본값) | 일반적인 줄 표시. 예: `💾 Memory updated`, `💾 Skill 'foo' patched`. |
| `verbose` | 변경된 내용의 간결한 미리보기 표시. 예: `💾 Memory ➕ User prefers terse replies` 또는 `"old" → "new"` 스킬 diff 일부. |

> 이는 **게이트웨이** 채팅 알림만 제어합니다. 검토 자체와 메모리/스킬 저장소에 대한 쓰기는 이 설정의 영향을 받지 않습니다. `display.platforms.<platform>.memory_notifications`를 통해 플랫폼별로 설정하세요.

## 더 저렴한 모델에서 검토 실행(`auxiliary.background_review`)

기본적으로 검토는 **주 채팅 모델**에서 실행되며 대화를 재생합니다. 대화가 이미 프롬프트 캐시에 준비되어 있으므로 캐시 읽기 비용이 저렴합니다. 비용이 높은 주 모델을 사용한다면 대신 더 저렴한 모델에서 검토를 실행할 수 있습니다.

```yaml
auxiliary:
  background_review:
    provider: openrouter
    model: google/gemini-3-flash-preview   # auto (default) = main chat model
```

주 모델과 **다른** 모델을 지정하면 검토는 그 모델에서 실행되어 상당히 낮은 비용이 듭니다(벤치마크에서 약 3~5배). 다른 모델은 어차피 주 모델의 프롬프트 캐시를 재사용할 수 없으므로, 새 캐시에 기록하는 내용을 최소화하기 위해 분기는 전체 대화 기록 대신 간결한 **다이제스트**(최근 턴은 원문 그대로, 이전 턴은 요약)를 자동으로 재생합니다. 테스트에서 메모리 캡처는 동일했고 스킬 캡처도 주 모델 검토와 거의 동일했습니다.

`auto`로 두거나 주 모델로 설정하면 아무것도 바뀌지 않습니다. 검토는 전체 대화를 따뜻한 캐시에 재생하면서 주 모델에서 계속 실행됩니다.

## 스킬 쓰기 제어(`skills.write_approval`)

스킬도 동일한 on/off 게이트를 사용하지만 `SKILL.md`는 채팅 말풍선으로 전체를 읽기에는 너무 크므로 검토 UX가 다릅니다.

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

`write_approval: true`이면 출처와 관계없이 스킬 쓰기(create / edit / patch / write_file / delete)가 항상 **대기 상태가 됩니다**. 인라인으로 한 줄 요약을 검토하지만 전체 diff는 대화 밖에 유지됩니다.

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

메시징 플랫폼에서는 요약과 메타데이터로 스킬을 승인하거나, 전체 변경 사항을 읽고 싶을 때 CLI/대시보드의 `/skills diff` 또는 `~/.hermes/pending/skills/<id>.json`의 대기 파일을 열 수 있습니다. 자세한 내용은 [에이전트 스킬 쓰기 게이트](/user-guide/features/skills#gating-agent-skill-writes-skillswrite_approval)를 참조하세요.

## 외부 메모리 제공자

MEMORY.md와 USER.md를 넘어서는 더 깊고 지속적인 메모리를 위해 Hermes는 Honcho, OpenViking, Mem0, Hindsight, Holographic, RetainDB, ByteRover, Supermemory를 포함한 8개의 외부 메모리 제공자 플러그인을 제공합니다.

외부 제공자는 내장 메모리와 **함께 실행되며**(절대 대체하지 않음), 지식 그래프, 의미 검색, 자동 사실 추출, 세션 간 사용자 모델링 등의 기능을 추가합니다.

```bash
hermes memory setup      # pick a provider and configure it
hermes memory status     # check what's active
```

각 제공자의 자세한 내용, 설정 방법 및 비교는 [메모리 제공자](./memory-providers.md) 가이드를 참조하세요.

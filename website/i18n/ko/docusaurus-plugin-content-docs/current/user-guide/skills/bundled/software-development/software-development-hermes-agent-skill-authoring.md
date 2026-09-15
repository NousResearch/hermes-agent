---
title: "Hermes Agent 스킬 작성 — 리포지토리 내 SKILL.md 파일 작성: 프런트매터와 구조"
sidebar_label: "Hermes Agent 스킬 작성"
description: "리포지토리 내 SKILL.md 파일 작성: 프런트매터와 구조"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent 스킬 작성

리포지토리 내 SKILL.md 파일의 프런트매터와 구조를 작성합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 제공 (기본 설치) |
| 경로 | `skills/software-development/hermes-agent-skill-authoring` |
| 버전 | `2.0.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `skills`, `authoring`, `hermes-agent`, `conventions`, `skill-md` |
| 관련 스킬 | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 참고: 전체 SKILL.md

:::info
다음은 Hermes가 이 스킬이 트리거될 때 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 보는 내용입니다.
:::

# Hermes-Agent 스킬 작성 (리포지토리 내)

## 개요

SKILL.md가 위치할 수 있는 곳은 두 군데입니다.

1. **사용자 로컬:** `~/.hermes/skills/<maybe-category>/<name>/SKILL.md` — 개인용이며 공유되지 않습니다. `skill_manage(action='create')`를 통해 생성합니다.
2. **리포지토리 내 (이 스킬이 다루는 경우):** hermes-agent 리포지토리 내부의 `skills/<category>/<name>/SKILL.md` 또는 `optional-skills/<category>/<name>/SKILL.md` — 커밋되어 패키지와 함께 제공됩니다. `write_file` + `git add`를 사용합니다. `skill_manage(action='create')`는 이 트리를 대상으로 하지 않습니다.

리포지토리 내 스킬은 리포지토리의 **엄격한 작성 표준**을 충족해야 합니다 (AGENTS.md의 "스킬 작성 표준 (HARDLINE)" 절을 참조하세요 — 이 스킬은 해당 표준을 실제로 적용하는 방법을 안내합니다). 해당 표준을 위반한 PR은 리뷰어가 거부하므로, 처음부터 충족하는 편이 사후 보완 작업보다 효율적입니다.

## 사용 시점

- 사용자 로컬 스킬에는 사용하지 마세요: `~/.hermes/skills/`의 개인 스킬은 `skill_manage`만 사용합니다.
- 사용자가 "이 브랜치 / 리포지토리 / 커밋에" 스킬을 추가해 달라고 요청한 경우
- hermes-agent와 함께 제공되어야 하는 재사용 가능한 워크플로를 커밋하는 경우
- `skills/` 또는 `optional-skills/` 아래의 기존 스킬을 편집하는 경우 (`patch`는 작은 편집에, `write_file`은 다시 작성할 때 사용합니다. `skill_manage`는 리포지토리 내 스킬의 patch에는 계속 사용할 수 있지만 `create`에는 사용할 수 없습니다.)

## 먼저 계층 결정: 기본 제공 vs 선택 사항

- **기본 제공 (`skills/<category>/`)** — 폭넓은 사용자에게 유용하고, 풋프린트가 작으며, 매일 사용하는 동작입니다. "사용자가 한 달에 5회 이상 세션에서 이 스킬을 로드할 것"이라고 자신 있게 말할 수 있어야 합니다.
- **선택 사항 (`optional-skills/<category>/`)** — 틈새·수직 분야(블록체인, 게임, 금융, 특정 앱), 반복 작업/작업 스킬, 또는 규모가 큰 항목입니다. `hermes skills install official/<category>/<skill>`을 통해 설치합니다.

확신이 없으면 선택 사항으로 두세요. 나중에 기본 제공으로 승격하는 것은 쉽지만, 다시 내리는 것은 사용자에게 불편을 줍니다. "누구든 필요할 때 유용할 것"이라는 판단은 기본 제공이 아니라 선택 사항에 해당합니다.

카테고리는 스킬이 무엇인지에 따라 정하고, 어떤 느낌인지에 따라 정하지 마세요 (AI 에이전트 CLI는 "생산성"처럼 느껴지더라도 `autonomous-ai-agents/`에 둡니다). 기존 카테고리는 `search_files(pattern='*', target='files', path='skills')`로 확인하고, 특별한 이유 없이 새로운 최상위 카테고리를 만들지 마세요.

**라우터 / 인덱스 / 허브 스킬은 만들지 마세요.** 핵심 내용이 자매 스킬을 가리키는 라우팅 표뿐인 스킬은 자매 스킬 자체의 `When to Use` 트리거를 복제하고 간접 참조 단계를 추가합니다. "스킬 X를 대신 로드"하는 포인터가 없으면 내용이 비게 되는 스킬은 작성하지 마세요 — 카탈로그와 각 자매 스킬의 트리거가 이미 그 역할을 합니다.

## 필수 프런트매터

검증 기준의 출처: `tools/skill_manager_tool.py::_validate_frontmatter`. 검증기의 필수 요구 사항은 다음과 같습니다.

- 첫 바이트부터 `---`로 시작해야 합니다 (앞에 빈 줄이 없어야 합니다).
- 본문 앞에 `\n---\n`으로 닫혀야 합니다.
- YAML 매핑으로 파싱되어야 합니다.
- `name` 필드가 있어야 합니다.
- `description` 필드가 있어야 합니다 (검증기 한도는 1024자이지만, 아래 리포지토리 엄격 기준은 훨씬 더 엄격합니다).
- 닫는 `---` 뒤에 비어 있지 않은 본문이 있어야 합니다.

리포지토리 표준 형태(검증기가 강제하지 않는 경우에도 모든 필드를 예상합니다)는 다음과 같습니다.

```yaml
---
name: my-skill-name               # lowercase, hyphens, ≤64 chars (MAX_NAME_LENGTH)
description: Concise capability statement, under sixty chars.
version: 0.1.0                    # semver; new skills start at 0.1.0
author: Real Name (github-handle), Hermes Agent
license: MIT
platforms: [linux, macos, windows]   # audit, don't guess — see Platform Gating
metadata:
  hermes:
    tags: [Short, Descriptive, Tags]
    related_skills: [other-in-repo-skill]
---
```

### `description` 규칙 (엄격 기준 — 검증기의 1024자는 표준이 아닙니다)

- **60자 이하.** 한 문장으로 작성합니다. 마침표로 끝납니다.
- 구현이 아니라 기능을 설명하고, 스킬 이름을 반복하지 마세요.
- 마케팅 용어("powerful", "comprehensive", "seamless", "advanced")를 사용하지 마세요.
- 시스템 프롬프트의 스킬 인덱스는 57자 + "..."에서 잘립니다 — 트리거/기능이 이 범위 안에서 완결되어야 합니다.
- 설명에 `:`가 포함되면 큰따옴표로 감싸세요. 그렇지 않으면 YAML이 이를 매핑으로 파싱하여 문서 생성기가 중단됩니다. 따옴표는 글자 수에 포함되지 않습니다.

좋은 예: `Track named companies for material news with cited digests.`
나쁜 예: `Use when a user asks to monitor named competitors or companies for product launches, pricing changes, funding, ...` (240자 — 리뷰에서 거부됨)

### `author` 규칙

- **사람을 먼저**, 그 다음 보조 협력자로 "Hermes Agent"를 기재합니다: `Ben Barclay (benbarclay), Hermes Agent`.
- 기여된 스킬에 `author: Hermes Agent`만 단독으로 기재하지 마세요 — 에이전트가 초안을 작성했더라도 사람을, 도구가 아니라 사람을 크레딧으로 기재합니다.
- 메인테이너가 작성한 스킬: `Teknium (teknium1), Hermes Agent`.

### `related_skills` 규칙

- 모든 항목은 PR과 동일한 트리 상태에서 같은 트리에 있는 기존 리포지토리 내 스킬로 해석되어야 합니다. 계획만 되었거나 다른 PR에 있거나 `~/.hermes/skills/`에만 있는 스킬은 참조하지 마세요.
- 각 항목을 확인하세요: `search_files(pattern='<name>', target='files', path='skills')` (및 `optional-skills/`).

## 플랫폼 게이팅: 확인하고, 믿기만 하지 마세요

`platforms:`는 호스트 OS에 따라 로딩을 제한합니다. 스킬의 설명과 스크립트가 실제로 호출하는 것을 기준으로 설정하세요.

| 스킬이 사용하는 것 | `platforms:` |
|---|---|
| Hermes 도구 + 표준 라이브러리 Python + 크로스 플랫폼 CLI만 사용 | `[linux, macos, windows]` |
| bash 파이프라인, `grep`/`awk`/`sed` 체인 | `[linux, macos]` |
| `osascript`, `defaults`, `pmset` | `[macos]` |
| `apt`/`systemctl`/`/proc` | `[linux]` |

스크립트에서 찾아야 할 POSIX 전용 신호: `fcntl`, `termios`, `pty`, `os.fork`, `os.killpg`, `signal.SIGKILL`, `os.kill(pid, 0)` 활성 상태 확인, 하드코딩된 `/tmp` `/proc` `/etc`. 기본 방향은 먼저 크로스 플랫폼으로 수정하는 것입니다 (`tempfile.gettempdir()`, `pathlib.Path`, `psutil.pid_exists`); 의존성이 실제로 플랫폼에 종속될 때만 더 좁게 제한하고, `## Pitfalls`에서 이유를 설명하세요.

## 크기 제한

- 전체 SKILL.md: 시행되는 최대 100,000자이지만, 단순한 스킬은 **약 100줄**, 복잡한 스킬은 **약 200줄**을 목표로 합니다. 동료 스킬은 8~14k자입니다.
- 부피가 크거나 브랜치에 종속된 내용은 `references/*.md`, `templates/` 또는 `scripts/`에 두고, SKILL.md에서 가리키세요. 인라인으로 넣지 마세요.
- 매번 모델이 파서나 사소하지 않은 로직을 인라인으로 작성할 것이라고 기대하지 마세요 — `scripts/`에 보조 스크립트를 제공하고 경로를 참조하세요.

## 본문 구조 (현대적인 절 순서)

```
# <Skill> Skill
2-3 sentence intro: what it does, what it doesn't do, dependency stance.

## When to Use          — bulleted triggers (+ "Don't use for:" counter-triggers)
## Prerequisites        — exact env vars, installs, API key sourcing
## How to Run           — canonical invocation through the `terminal` tool
## Quick Reference      — flat command list, no narration
## Procedure            — numbered steps, each with a checkable completion criterion
## Pitfalls             — known limits, things that look broken but aren't
## Verification         — how to prove the skill worked
```

모든 스킬에 모든 절이 필요한 것은 아닙니다 (순수 절차 작업 스킬에는 Quick Reference가 없을 수 있습니다). 하지만 When to Use + 실행 가능한 본문 + Pitfalls + Verification은 최소 요구 사항입니다. 홍보성 서론, "Setup Check" 같은 무의미한 단계, Prerequisites에 이미 있는 환경 변수의 재설명은 삭제하세요.

### Hermes 도구를 참조하고 원시 셸은 참조하지 마세요

스킬에 기능이 필요할 때는 적절한 Hermes 도구를 백틱으로 이름을 지정하세요: `terminal`, `read_file`, `write_file`, `patch`, `search_files`, `web_search`, `web_extract`, `browser_navigate`, `vision_analyze`, `delegate_task`, `cronjob`. 에이전트가 이미 래핑해 둔 셸 유틸리티의 이름을 직접 지정하지 마세요 (`grep` → `search_files`, `cat` → `read_file`, `sed`/`awk` → `patch`, `find`/`ls` → `search_files target='files'`). CLI 래퍼 스킬은 `terminal(command="<tool> ...", timeout=...)` 호출 형태로 작성하세요 — 원시 셸 문장("`foo --version`을 실행하세요")은 리뷰를 막는 비준수 사항입니다. 스킬이 MCP 서버에 의존한다면 해당 서버의 이름을 지정하고 Prerequisites에 설정 방법을 문서화하세요.

### 머신 로컬 경로는 절대 사용하지 마세요

리포지토리 상대 경로(`skills/...`, `tools/skill_manager_tool.py`)를 작성하세요. 커밋된 스킬에 `/home/<you>/...` 경로를 넣으면 다른 모든 사용자의 환경에서 깨지며 즉시 리뷰 경고 대상이 됩니다.

## 작성 품질 원칙

스킬은 에이전트의 프로세스를 더 예측 가능하게 만들어 — 에이전트가 매번 동일하게 유용한 규율을 따르도록 합니다.

1. **프로세스 예측 가능성을 최적화하세요.** 동작을 바꾸지 않는 문장은 삭제하세요.
2. **적절한 컨텍스트 로드를 선택하세요.** 설명은 매 턴 비용이 발생하므로 세부 사항은 본문이나 연결된 참고 자료에 둡니다.
3. **단계를 완료 기준으로 끝내세요.** 확인 가능하고, 중요한 경우에는 빠짐없이 작성합니다. "변경된 모든 파일을 빠짐없이 기록"이 "변경 사항을 요약"보다 낫습니다.
4. **규칙이 적용되는 개념 옆에 규칙을 배치하세요.**
5. **강한 선행 단어를 사용하세요** ("tight loop", "root cause", "regression test"). 길게 반복해서 설명하지 마세요.
6. **중복과 무의미한 단계를 제거하세요.** "주의하세요"와 "모범 사례를 사용하세요"는 모델의 동작을 바꾸지 않습니다 — 확인 가능한 기준으로 바꾸거나 삭제하세요.

## 테스트 및 문서 (리포지토리 스킬 필수 사항)

1. **테스트**는 `tests/skills/test_<skill>_skill.py`에 둡니다 — 표준 라이브러리 + pytest + `unittest.mock`만 사용하며 실제 네트워크는 사용하지 않습니다. `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`로 실행합니다. (일반적인 `tests/tools/test_skill_manager_tool.py`가 통과하는 것은 **여러분의** 스킬에 대해 아무것도 증명하지 않습니다.)
2. **문서 재생성:** `python3 website/scripts/generate-skill-docs.py`를 실행한 뒤 범위를 엄격히 관리하세요 — 생성기는 모든 자동 생성 페이지를 다시 작성합니다. 자신의 것이 아닌 항목은 `git checkout --`으로 되돌리세요. 최종 diff에는 자신의 SKILL.md, 자신의 스킬 문서 페이지 하나, 카탈로그 행 한 줄, `website/sidebars.ts` 삽입 한 줄만 있어야 합니다 ( `search_files(pattern='<your-slug>', path='website/sidebars.ts')`로 확인 — 정확히 한 번 검색되어야 하며, 그렇지 않으면 페이지가 고아가 됩니다).
3. **`.env.example`** (새 환경 변수가 필요한 스킬인 경우에만): 명확히 구분된 주석 블록 하나만 추가하고 파일의 다른 부분은 건드리지 마세요.

## 워크플로

1. **동료 스킬을 조사하세요**: 대상 카테고리에서 `search_files(target='files')`를 사용하고, 어조와 구조를 맞추기 위해 동료 SKILL.md 파일 2~3개를 읽으세요. 좁은 자매 스킬을 새로 만들기보다 기존 스킬을 확장하는 것을 우선하세요.
2. **계층과 카테고리를 결정하세요** (위 참조). 확신이 없으면 선택 사항으로 두고, 기본 제공으로 정하지 말고 푸시하기 전에 질문하세요.
3. **초안을 작성하세요**: `skills/<category>/<name>/SKILL.md` (또는 `optional-skills/...`)에 `write_file`을 사용합니다.
4. **로컬에서 검증하세요**:
   ```python
   import yaml, re, pathlib
   content = pathlib.Path("skills/<category>/<name>/SKILL.md").read_text()
   assert content.startswith("---")
   m = re.search(r'\n---\s*\n', content[3:])
   fm = yaml.safe_load(content[3:m.start()+3])
   assert "name" in fm and "description" in fm
   assert len(fm["description"]) <= 60, f"description {len(fm['description'])} chars — hardline is 60"
   assert fm["description"].endswith(".")
   assert "platforms" in fm
   assert len(content) <= 100_000
   ```
   또한 모든 `related_skills` 항목이 리포지토리 내에 존재하는지 확인하세요.
5. **테스트를 추가하고 문서를 재생성하세요** (이전 절 참조).
6. 활성 브랜치에서 Git add + 커밋을 수행하고 PR을 엽니다.
7. **참고:** 현재 세션의 스킬 로더는 캐시됩니다 — 새 세션이 시작될 때까지 `skill_view` / `skills_list`에서 새 스킬을 볼 수 없습니다. 이는 버그가 아닙니다.

## 기존 리포지토리 내 스킬 편집

- **작은 수정:** `skill_manage(action='patch', ...)`는 `patch`와 마찬가지로 리포지토리 내 스킬에서 작동합니다.
- **대규모 다시 작성:** 전체 SKILL.md를 `write_file`합니다.
- **지원 파일:** 스킬 디렉터리 아래의 `references/`, `templates/` 또는 `scripts/`에 `write_file`합니다.
- **항상 커밋하세요** — 리포지토리 내 스킬은 런타임 상태가 아니라 소스입니다. 프런트매터가 변경되면 문서 생성기를 다시 실행하세요.

## 일반적인 함정

1. **리포지토리 내 스킬에 `skill_manage(action='create')`를 사용하는 것.** `~/.hermes/skills/`에 작성되므로 `write_file`을 사용해 리포지토리 트리에 작성하세요.
2. **검증기의 한도를 표준으로 믿는 것.** 검증기는 설명에 1024자를 허용하지만, 리뷰에서는 60자를 넘으면 거부합니다. 검증기는 `platforms:`, 작성자 형식, 테스트 또는 문서를 확인하지 않습니다 — 리뷰에서는 이를 확인합니다.
3. **기여된 스킬에 `author: Hermes Agent`를 작성하는 것.** 사람 기여자를 먼저 크레딧으로 기재하세요.
4. **`---` 앞에 공백을 두는 것.** 앞에 빈 줄이나 BOM이 있으면 검증에 실패합니다.
5. **설명이 너무 일반적이거나 트리거가 57자 이후에 묻히는 것.**
6. **존재하지 않는 리포지토리 내 스킬을 `related_skills`에 지정하는 것** (사용자 로컬, 계획 중, 또는 자매 PR에만 있는 스킬).
7. **동료 스킬을 복제하는 것.** 새로운 자매 스킬보다 확장을 우선하세요.
8. **문서 생성기를 건너뛰거나 관련 없는 변경 사항을 무분별하게 재생성하는 것.** 둘 다 잘못입니다: 재생성하지 않으면 사이드바에 없는 고아 스킬이 되고, 무분별한 재생성은 다른 스킬의 변경 사항으로 diff를 부풀립니다.
9. **현재 세션에서 새 스킬이 보일 것이라고 기대하는 것.** 로더는 세션 시작 시 초기화됩니다.
10. **스킬에 잔여물을 쌓이게 두는 것.** 규칙을 추가할 때 기존 규칙을 대체하는 낡은 표현을 제거하세요.

## 검증 체크리스트

- [ ] 계층을 의도적으로 결정했는가 (기본 제공 기준: 월 5회 이상 세션; 아니면 `optional-skills/`)
- [ ] `skills/<category>/<name>/SKILL.md` 또는 `optional-skills/<category>/<name>/SKILL.md`에 파일이 있는가
- [ ] 프런트매터가 바이트 0에서 `---`로 시작하고 `\n---\n`으로 닫히는가
- [ ] `name`, `description`, `version`, `author`, `license`, `platforms`, `metadata.hermes.{tags, related_skills}`가 모두 있는가
- [ ] 설명이 60자 이하이고, 한 문장이며, 마침표로 끝나고, 마케팅 용어가 없는가
- [ ] `author`가 사람 기여자를 먼저 크레딧으로 기재하는가
- [ ] `platforms:`가 동료 스킬을 복사하지 않고 실제 본문/스크립트에 따라 점검되었는가
- [ ] 모든 `related_skills` 항목이 리포지토리 내에서 해석되는가
- [ ] 본문이 현대적인 절 순서를 따르고 명령이 Hermes 도구를 통해 표현되는가
- [ ] 파일 어디에도 머신 로컬 경로가 없는가
- [ ] 순서가 있는 각 단계에 확인 가능한 완료 기준이 있는가
- [ ] `scripts/run_tests.sh`에서 `tests/skills/test_<skill>_skill.py`가 통과하는가
- [ ] 문서가 범위를 엄격히 관리하며 재생성되었고, 슬러그에 대해 사이드바 항목이 정확히 하나인가
- [ ] 의도한 브랜치에서 `git add` + 커밋을 했고 PR을 열었는가

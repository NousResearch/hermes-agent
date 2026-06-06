---
title: "Hermes Agent Skill Authoring — 저장소 내 SKILL 작성"
sidebar_label: "Hermes Agent Skill Authoring"
description: "저장소 내 SKILL 작성"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Hermes Agent Skill Authoring

저장소 내(in-repo) SKILL.md 작성: 프론트매터(frontmatter), 유효성 검사기, 구조.

## 스킬 메타데이터

| | |
|---|---|
| Source | Bundled (기본 설치됨) |
| Path | `skills/software-development/hermes-agent-skill-authoring` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `skills`, `authoring`, `hermes-agent`, `conventions`, `skill-md` |
| Related skills | [`plan`](/docs/user-guide/skills/bundled/software-development/software-development-plan), [`requesting-code-review`](/docs/user-guide/skills/bundled/software-development/software-development-requesting-code-review) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화될 때 에이전트가 지침으로 보는 내용입니다.
:::

# Hermes-Agent 스킬 작성하기 (저장소 내)

## 개요

SKILL.md가 존재할 수 있는 위치는 두 곳입니다:

1. **사용자 로컬(User-local):** `~/.hermes/skills/<maybe-category>/<name>/SKILL.md` — 개인적이며 공유되지 않습니다. `skill_manage(action='create')`를 통해 생성됩니다.
2. **저장소 내 (이 스킬이 다루는 케이스):** `/home/bb/hermes-agent/skills/<category>/<name>/SKILL.md` — 커밋되어 패키지와 함께 배포됩니다. `write_file` + `git add`를 사용하세요. `skill_manage(action='create')`는 이 트리를 대상으로 하지 **않습니다.**

## 사용 시기

- 사용자가 "이 브랜치 / 저장소 / 커밋에" 스킬을 추가해 달라고 요청할 때
- hermes-agent와 함께 배포되어야 하는 재사용 가능한 워크플로를 커밋할 때
- `/home/bb/hermes-agent/skills/` 아래에 있는 기존 스킬을 편집할 때 (작은 수정에는 `patch`를 사용하고, 전체 재작성에는 `write_file`을 사용하세요; `skill_manage`는 저장소 내 스킬의 patch에는 동작하지만, `create`에는 동작하지 않습니다)

## 필수 프론트매터 (Frontmatter)

신뢰할 수 있는 출처: `tools/skill_manager_tool.py::_validate_frontmatter`. 필수 요구 사항:

- 첫 번째 바이트가 `---`로 시작해야 합니다 (앞에 빈 줄이 없어야 함).
- 본문이 시작되기 전 `\n---\n`으로 닫혀야 합니다.
- YAML 매핑으로 정상적으로 파싱되어야 합니다.
- `name` 필드가 존재해야 합니다.
- `description` 필드가 존재해야 하며, **1024자**(`MAX_DESCRIPTION_LENGTH`) 이하여야 합니다.
- 닫는 `---` 이후의 본문이 비어 있지 않아야 합니다.

`skills/software-development/` 아래의 모든 피어(peer) 스킬이 사용하는 형태:

```yaml
---
name: my-skill-name               # 소문자, 하이픈 사용, 64자 이하 (MAX_NAME_LENGTH)
description: Use when <trigger>. <one-line behavior>.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [short, descriptive, tags]
    related_skills: [other-skill, another-skill]
---
```

`version` / `author` / `license` / `metadata`는 유효성 검사기에 의해 강제되지는 않지만 모든 동료 스킬들이 이를 가지고 있습니다 — 생략하면 당신의 스킬만 튀어 보일 것입니다.

## 크기 제한

- 설명(Description): 1024자 이하 (강제됨).
- SKILL.md 전체: 100,000자 이하 (`MAX_SKILL_CONTENT_CHARS`로 강제됨, 약 36k 토큰).
- `software-development/`에 있는 기존 스킬들은 **8~14,000자** 범위에 있습니다. 이 범위를 목표로 하세요. 20,000자를 넘길 것 같다면 `references/*.md`로 분리하고 SKILL.md에서 참조하세요.

## 기존 스킬과의 구조 일치시키기

저장소 내 모든 스킬은 대략적으로 다음 구조를 따릅니다:

```
# <제목>

## 개요 (Overview)
한두 문단: 무엇이고 왜 필요한지.

## 사용 시기 (When to Use)
- 글머리 기호로 작성된 트리거(조건)
- "Don't use for:" 반대 트리거(사용하지 말아야 할 때)

## <스킬에 특화된 주제 섹션들>
- 빠른 참조용 표가 일반적입니다.
- 정확한 명령어가 포함된 코드 블록
- Hermes 관련 특정 레시피 (scripts/run_tests.sh를 통한 테스트, ui-tui 경로 등)

## 흔한 실수들 (Common Pitfalls)
실수들과 그에 대한 해결책을 번호가 매겨진 목록으로 정리합니다.

## 검증 체크리스트 (Verification Checklist)
- [ ] 작업 후 검증을 위한 체크박스 목록

## 단발성 레시피 (One-Shot Recipes) (선택 사항)
명명된 시나리오 → 구체적인 명령 시퀀스.
```

모든 섹션이 필수는 아니지만, `개요(Overview)` + `사용 시기(When to Use)` + 실행 가능한 본문 + `흔한 실수들(Pitfalls)`은 스킬이 동료처럼 느껴지기 위한 최소 요건입니다.

## 디렉토리 위치

```
skills/<category>/<skill-name>/SKILL.md
```

현재 저장소에 있는 카테고리들 (`ls skills/`로 확인하세요): `autonomous-ai-agents`, `creative`, `data-science`, `devops`, `dogfood`, `email`, `gaming`, `github`, `leisure`, `mcp`, `media`, `mlops/*`, `note-taking`, `productivity`, `red-teaming`, `research`, `smart-home`, `social-media`, `software-development`.

가장 가까운 기존 카테고리를 선택하세요. 가볍게 새로운 최상위 카테고리를 만들지 마세요.

## 워크플로

1. 대상 카테고리에서 **동료 스킬들을 조사**하세요:
   ```
   ls skills/<category>/
   ```
   어조와 구조를 맞추기 위해 2~3개의 동료 SKILL.md 파일을 읽어보세요.
2. 불확실한 경우 `tools/skill_manager_tool.py`에서 **유효성 검사 제약 조건**을 확인하세요.
3. `write_file`을 사용하여 `skills/<category>/<name>/SKILL.md`에 초안을 작성하세요.
4. **로컬에서 유효성 검사**를 실행하세요:
   ```python
   import yaml, re, pathlib
   content = pathlib.Path("skills/<category>/<name>/SKILL.md").read_text()
   assert content.startswith("---")
   m = re.search(r'\n---\s*\n', content[3:])
   fm = yaml.safe_load(content[3:m.start()+3])
   assert "name" in fm and "description" in fm
   assert len(fm["description"]) <= 1024
   assert len(content) <= 100_000
   ```
5. 현재 활성 브랜치에서 **Git add + commit**을 수행하세요.
6. **참고:** 현재 세션의 스킬 로더는 캐시되어 있습니다 — 새 세션을 시작할 때까지 `skill_view` / `skills_list`는 새 스킬을 보지 못합니다. 이는 버그가 아니라 예상된 동작입니다.

## 다른 스킬 상호 참조

`metadata.hermes.related_skills`는 로드 시 두 트리(저장소 내 `skills/` 및 `~/.hermes/skills/`)를 통합합니다. 저장소 내 스킬에서 사용자 로컬 스킬을 참조할 **수는** 있지만, 저장소를 새로 클론하는 다른 사용자에게는 해당 참조가 해결되지 않습니다. 저장소 내 스킬에서는 가급적 저장소 내 스킬만 참조하는 것을 선호하세요. 자주 참조되는 스킬이 `~/.hermes/skills/`에만 있다면, 이를 저장소로 승격하는 것을 고려하세요.

## 기존 저장소 내 스킬 편집하기

- **작은 수정 (오타, 누락된 실수 추가, 트리거 세분화):** 저장소 내 스킬에도 `skill_manage(action='patch', name=..., old_string=..., new_string=...)`가 잘 작동합니다.
- **주요 재작성:** 전체 SKILL.md를 `write_file` 하세요. `skill_manage(action='edit')`도 작동하지만 전체 새 내용을 제공해야 합니다.
- **지원 파일 추가:** `skills/<category>/<name>/references/<file>.md`, `templates/<file>`, 또는 `scripts/<file>`에 `write_file`을 사용하세요. `skill_manage(action='write_file')`도 작동하며 references/templates/scripts/assets 하위 디렉토리 허용 목록을 강제합니다.
- **항상 커밋하세요** — 저장소 내 스킬은 소스 코드이며 런타임 상태가 아닙니다.

## 일반적인 함정

1. **저장소 내 스킬에 `skill_manage(action='create')` 사용하기.** 이 명령은 저장소 트리가 아닌 `~/.hermes/skills/`에 씁니다. 저장소 내 스킬 생성에는 `write_file`을 사용하세요.

2. **`---` 앞의 선행 공백.** 유효성 검사기는 `content.startswith("---")`를 확인합니다; 앞서 있는 빈 줄이나 BOM은 유효성 검사를 실패하게 만듭니다.

3. **설명이 너무 일반적임.** 동료 스킬의 설명은 "Use when ..."으로 시작하며, 단일 작업이 아닌 *트리거 클래스*를 설명합니다. "Debug X"보다 "Use when debugging X"가 낫습니다.

4. **작성자/라이선스/메타데이터 블록을 잊는 것.** 유효성 검사기가 강제하지는 않지만, 모든 동료 스킬이 이를 가지고 있습니다; 이를 생략하면 스킬이 미완성된 것처럼 보입니다.

5. **동료 스킬과 중복되는 스킬 작성하기.** 생성하기 전에 `ls skills/<category>/`를 확인하고 2~3개의 동료 스킬을 열어보세요. 매우 좁은 범위의 형제 스킬을 새로 만드는 것보다 기존 스킬을 확장하는 것을 선호하세요.

6. **현재 세션이 새 스킬을 인식할 것이라 기대하기.** 그렇지 않습니다. 스킬 로더는 세션 시작 시 초기화됩니다. 새 세션에서 확인하거나 정확한 경로를 사용하여 `skill_view`를 통해 확인하세요.

7. **저장소에 없는 스킬 링크하기.** `related_skills: [some-user-local-skill]`은 당신에게는 작동하지만 다른 클론 사용자에게는 깨집니다. 가급적 저장소 내 링크만 사용하세요.

## 검증 체크리스트

- [ ] 파일 위치가 `~/.hermes/skills/`가 아닌 `skills/<category>/<name>/SKILL.md`에 존재함
- [ ] 프론트매터가 0바이트에서 `---`로 시작하고 `\n---\n`으로 닫힘
- [ ] `name`, `description`, `version`, `author`, `license`, `metadata.hermes.{tags, related_skills}`가 모두 존재함
- [ ] 이름은 64자 이하, 소문자 + 하이픈 구조
- [ ] 설명은 1024자 이하이며 "Use when ..."으로 시작함
- [ ] 전체 파일은 100,000자 이하 (8~15k자를 목표로 함)
- [ ] 구조: `# 제목` → `## 개요` → `## 사용 시기` → 본문 → `## 흔한 실수들` → `## 검증 체크리스트`
- [ ] `related_skills` 참조가 저장소 내에서 해결됨 (또는 사용자 로컬 스킬임을 명시적으로 밝힘)
- [ ] 의도한 브랜치에서 `git add skills/<category>/<name>/ && git commit`이 완료됨

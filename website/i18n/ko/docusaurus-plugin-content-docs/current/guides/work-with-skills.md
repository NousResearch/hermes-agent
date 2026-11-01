---
sidebar_position: 12
title: "스킬 활용하기"
description: "스킬을 찾고, 설치하고, 사용하고, 만들기 — Hermes에 새로운 워크플로를 가르치는 온디맨드 지식"
---

# 스킬 활용하기

스킬은 Hermes가 특정 작업을 처리하는 방법을 가르치는 온디맨드 지식 문서입니다. ASCII 아트 생성부터 GitHub PR 관리까지 다양한 작업을 다룹니다. 이 가이드는 스킬을 일상적으로 사용하는 방법을 설명합니다.

전체 기술 참조는 [스킬 시스템](/user-guide/features/skills)을 참고하세요.

---

## 스킬 찾기

모든 Hermes 설치에는 번들 스킬이 포함되어 있습니다. 사용 가능한 스킬을 확인하세요.

```bash
# In any chat session:
/skills

# Or from the CLI:
hermes skills list
```

이 명령은 이름과 설명을 간결한 목록으로 보여줍니다.

```
ascii-art         Generate ASCII art using pyfiglet, cowsay, boxes...
arxiv             Search and retrieve academic papers from arXiv...
github-pr-workflow Full PR lifecycle — create branches, commit...
plan              Plan mode — inspect context, write a markdown...
excalidraw        Create hand-drawn style diagrams using Excalidraw...
```

### 스킬 검색하기

```bash
# Search by keyword
/skills search docker
/skills search music
```

### 스킬 허브

공식 선택 스킬(기본으로 활성화되지 않는 무겁거나 전문적인 스킬)은 허브에서 이용할 수 있습니다.

```bash
# Browse official optional skills
/skills browse

# Search the hub
/skills search blockchain
```

---

## 스킬 사용하기

설치된 모든 스킬은 자동으로 슬래시 명령이 됩니다. 이름만 입력하세요.

```bash
# Load a skill and give it a task
/ascii-art Make a banner that says "HELLO WORLD"
/plan Design a REST API for a todo app
/github-pr-workflow Create a PR for the auth refactor

# Just the skill name (no task) loads it and lets you describe what you need
/excalidraw
```

자연어 대화로도 스킬을 실행할 수 있습니다. 특정 스킬을 사용해 달라고 Hermes에 요청하면 `skill_view` 도구를 통해 해당 스킬을 로드합니다.

### 점진적 공개

스킬은 토큰을 효율적으로 사용하는 로딩 방식을 따릅니다. 에이전트는 모든 내용을 한 번에 로드하지 않습니다.

1. **`skills_list()`** — 모든 스킬의 간결한 목록(약 3천 토큰). 세션 시작 시 로드됩니다.
2. **`skill_view(name)`** — 스킬 하나의 전체 SKILL.md 내용. 에이전트가 필요하다고 판단할 때 로드됩니다.
3. **`skill_view(name, file_path)`** — 스킬에 포함된 특정 참조 파일. 필요한 경우에만 로드됩니다.

따라서 실제로 사용하기 전까지는 스킬이 토큰을 소모하지 않습니다.

---

## 허브에서 설치하기

공식 선택 스킬은 Hermes에 포함되어 있지만 기본으로 활성화되지는 않습니다. 명시적으로 설치하세요.

```bash
# Install an official optional skill
hermes skills install official/research/arxiv

# Install from the hub in a chat session
/skills install official/creative/songwriting-and-ai-music

# Install SKILL.md and its referenced support files from an HTTP(S) URL
hermes skills install https://sharethis.chat/SKILL.md
/skills install https://example.com/SKILL.md --name my-skill
```

수행되는 작업:
1. 스킬 디렉터리를 `~/.hermes/skills/`로 복사합니다.
2. `skills_list` 출력에 표시됩니다.
3. 슬래시 명령으로 사용할 수 있게 됩니다.

:::tip
설치한 스킬은 새 세션부터 적용됩니다. 현재 세션에서 사용하려면 `/reset`으로 새로 시작하거나, `--now`를 추가해 프롬프트 캐시를 즉시 무효화하세요(다음 턴에서 토큰 비용이 증가합니다).
:::

### 설치 확인하기

```bash
# Check it's there
hermes skills list | grep arxiv

# Or in chat
/skills search arxiv
```

---

## 플러그인이 제공하는 스킬

플러그인은 네임스페이스 이름(`plugin:skill`)으로 자체 스킬을 번들할 수 있습니다. 이를 통해 기본 제공 스킬과 이름이 충돌하는 것을 방지합니다.

```bash
# Load a plugin skill by its qualified name
skill_view("superpowers:writing-plans")

# Built-in skill with the same base name is unaffected
skill_view("writing-plans")
```

플러그인 스킬은 시스템 프롬프트에 나열되지 않으며 `skills_list`에도 표시되지 않습니다. 옵트인 방식이므로 플러그인이 제공하는 스킬임을 알고 있을 때 명시적으로 로드하세요. 로드되면 에이전트는 같은 플러그인의 형제 스킬 목록을 보여주는 배너를 확인합니다.

자체 플러그인에 스킬을 제공하는 방법은 [Hermes 플러그인 구축 → 스킬 번들링](/developer-guide/plugins#bundle-skills)을 참고하세요.

---

## 스킬 설정 구성하기

일부 스킬은 프런트매터에 필요한 설정을 선언합니다.

```yaml
metadata:
  hermes:
    config:
      - key: tenor.api_key
        description: "Tenor API key for GIF search"
        prompt: "Enter your Tenor API key"
        url: "https://developers.google.com/tenor/guides/quickstart"
```

설정이 있는 스킬을 처음 로드하면 Hermes가 값을 요청합니다. 값은 `skills.config.*` 아래의 `config.yaml`에 저장됩니다.

CLI에서 스킬 설정을 관리하세요.

```bash
# Interactive config for a specific skill
hermes skills config gif-search

# View all skill config
hermes config get skills.config --json
```

---

## 직접 스킬 만들기

스킬은 YAML 프런트매터가 있는 Markdown 파일일 뿐입니다. 만드는 데 5분도 걸리지 않습니다.

### 1. 디렉터리 만들기

```bash
mkdir -p ~/.hermes/skills/my-category/my-skill
```

### 2. SKILL.md 작성하기

```markdown title="~/.hermes/skills/my-category/my-skill/SKILL.md"
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
metadata:
  hermes:
    tags: [my-tag, automation]
    category: my-category
---

# My Skill

## When to Use
Use this skill when the user asks about [specific topic] or needs to [specific task].

## Procedure
1. First, check if [prerequisite] is available
2. Run `command --with-flags`
3. Parse the output and present results

## Pitfalls
- Common failure: [description]. Fix: [solution]
- Watch out for [edge case]

## Verification
Run `check-command` to confirm the result is correct.
```

### 3. 참조 파일 추가하기(선택 사항)

스킬에는 에이전트가 필요할 때 로드하는 지원 파일을 포함할 수 있습니다.

```
my-skill/
├── SKILL.md                    # Main skill document
├── references/
│   ├── api-docs.md             # API reference the agent can consult
│   └── examples.md             # Example inputs/outputs
├── templates/
│   └── config.yaml             # Template files the agent can use
└── scripts/
    └── setup.sh                # Scripts the agent can execute
```

SKILL.md에서 이를 참조하세요.

```markdown
For API details, load the reference: `skill_view("my-skill", "references/api-docs.md")`
```

### 4. 테스트하기

새 세션을 시작하고 스킬을 사용해 보세요.

```bash
hermes chat -q "/my-skill help me with the thing"
```

스킬은 자동으로 표시되며 등록할 필요가 없습니다. `~/.hermes/skills/`에 넣으면 즉시 활성화됩니다.

:::info
에이전트는 `skill_manage`를 사용해 스킬을 직접 만들고 업데이트할 수도 있습니다. 복잡한 문제를 해결한 뒤 Hermes가 다음에 사용할 수 있도록 접근 방식을 스킬로 저장할지 제안할 수 있습니다.
:::

---

## 플랫폼별 스킬 관리

어떤 플랫폼에서 어떤 스킬을 사용할 수 있는지 제어하세요.

```bash
hermes skills
```

이 명령은 플랫폼별로 스킬을 활성화하거나 비활성화할 수 있는 대화형 TUI를 엽니다(CLI, Telegram, Discord 등). 특정 컨텍스트에서만 특정 스킬을 사용할 때 유용합니다. 예를 들어 개발 스킬을 Telegram에서 사용할 수 없게 할 수 있습니다.

---

## 스킬과 메모리 비교

둘 다 세션 간에 유지되지만 서로 다른 역할을 합니다.

| | 스킬 | 메모리 |
|---|---|---|
| **무엇** | 절차적 지식 — 일을 처리하는 방법 | 사실 지식 — 사물이 무엇인지 |
| **언제** | 관련 있을 때만 온디맨드로 로드 | 모든 세션에 자동으로 주입 |
| **크기** | 클 수 있음(수백 줄) | 간결해야 함(핵심 사실만) |
| **비용** | 로드할 때까지 토큰 0 | 작지만 지속적인 토큰 비용 |
| **예시** | "Kubernetes에 배포하는 방법" | "사용자는 다크 모드를 선호하고 PST에 거주" |
| **누가 생성** | 사용자, 에이전트 또는 허브에서 설치 | 대화를 기반으로 에이전트가 생성 |

**경험 법칙:** 참조 문서에 넣을 내용이라면 스킬입니다. 포스트잇에 적을 내용이라면 메모리입니다.

---

## 팁

**스킬은 집중된 범위로 유지하세요.** "모든 DevOps"를 다루려는 스킬은 너무 길고 모호해집니다. "Fly.io에 Python 앱 배포하기"를 다루는 스킬은 실제로 유용할 만큼 구체적입니다.

**에이전트가 스킬을 만들게 하세요.** 복잡한 다단계 작업이 끝나면 Hermes는 종종 접근 방식을 스킬로 저장할지 제안합니다. 동의하세요. 에이전트가 만든 스킬에는 발견한 문제점까지 포함해 정확한 워크플로가 담깁니다.

**카테고리를 사용하세요.** 스킬을 하위 디렉터리(`~/.hermes/skills/devops/`, `~/.hermes/skills/research/` 등)에 정리하세요. 목록을 관리하기 쉬워지고 에이전트가 스킬을 더 빠르게 찾을 수 있습니다.

**오래된 스킬을 업데이트하세요.** 스킬을 사용하다가 다루지 않은 문제를 만나면 배운 내용을 반영해 스킬을 업데이트하도록 Hermes에 요청하세요. 관리되지 않는 스킬은 문제가 될 수 있습니다.

---

*전체 스킬 참조(프런트매터 필드, 조건부 활성화, 외부 디렉터리 등)는 [스킬 시스템](/user-guide/features/skills)을 참고하세요.*

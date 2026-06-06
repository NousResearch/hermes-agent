---
sidebar_position: 12
title: "스킬 사용하기 (Working with Skills)"
description: "스킬을 찾고, 설치하고, 사용하고, 생성하는 방법 — Hermes에게 새로운 워크플로우를 가르치는 온디맨드 지식"
---

# 스킬 사용하기 (Working with Skills)

스킬(Skills)은 Hermes에게 ASCII 아트 생성부터 GitHub PR 관리에 이르기까지 특정 작업을 처리하는 방법을 가르치는 온디맨드 지식 문서입니다. 이 가이드에서는 스킬을 일상적으로 사용하는 방법을 설명합니다.

전체 기술 레퍼런스는 [Skills System](/user-guide/features/skills)을 참조하세요.

---

## 스킬 찾기

모든 Hermes 설치에는 기본적으로 포함된 스킬들이 제공됩니다. 사용 가능한 스킬을 확인하세요:

```bash
# 채팅 세션 중 어디서나:
/skills

# 또는 CLI에서:
hermes skills list
```

이 명령어는 이름과 설명이 포함된 요약된 목록을 보여줍니다:

```
ascii-art         pyfiglet, cowsay, boxes 등을 사용하여 ASCII 아트를 생성합니다...
arxiv             arXiv에서 학술 논문을 검색하고 검색합니다...
github-pr-workflow 전체 PR 수명 주기 — 브랜치 생성, 커밋...
plan              계획 모드 — 컨텍스트를 검사하고 마크다운 문서를 작성합니다...
excalidraw        Excalidraw를 사용하여 손으로 그린 스타일의 다이어그램을 생성합니다...
```

### 스킬 검색하기

```bash
# 키워드로 검색
/skills search docker
/skills search music
```

### 스킬 허브 (Skills Hub)

공식 옵션 스킬(무겁거나 특정 목적을 위해 기본적으로 활성화되지 않은 스킬)은 허브를 통해 사용할 수 있습니다:

```bash
# 공식 옵션 스킬 찾아보기
/skills browse

# 허브 검색
/skills search blockchain
```

---

## 스킬 사용하기

설치된 모든 스킬은 자동으로 슬래시 명령어가 됩니다. 스킬 이름을 입력하기만 하면 됩니다:

```bash
# 스킬을 로드하고 작업 지시하기
/ascii-art "HELLO WORLD"라고 적힌 배너를 만들어줘
/plan 할 일 앱을 위한 REST API 설계해줘
/github-pr-workflow 인증 리팩토링을 위한 PR을 생성해줘

# 스킬 이름만 입력(작업 없음)하면 스킬이 로드되고 필요한 것을 설명할 수 있습니다
/excalidraw
```

자연스러운 대화를 통해 스킬을 트리거할 수도 있습니다 — Hermes에게 특정 스킬을 사용해 달라고 요청하면 `skill_view` 도구를 통해 스킬을 로드합니다.

### 점진적 공개 (Progressive Disclosure)

스킬은 토큰 효율적인 로딩 패턴을 사용합니다. 에이전트는 한 번에 모든 것을 로드하지 않습니다:

1. **`skills_list()`** — 모든 스킬의 요약된 목록 (~3k 토큰). 세션 시작 시 로드됩니다.
2. **`skill_view(name)`** — 단일 스킬의 전체 SKILL.md 콘텐츠. 에이전트가 해당 스킬이 필요하다고 판단할 때 로드됩니다.
3. **`skill_view(name, file_path)`** — 스킬 내의 특정 참조 파일. 필요할 때만 로드됩니다.

이는 스킬이 실제로 사용될 때까지 토큰 비용이 들지 않음을 의미합니다.

---

## 허브에서 설치하기

공식 옵션 스킬은 Hermes와 함께 제공되지만 기본적으로 활성화되어 있지는 않습니다. 명시적으로 설치해야 합니다:

```bash
# 공식 옵션 스킬 설치
hermes skills install official/research/arxiv

# 채팅 세션에서 허브로부터 설치
/skills install official/creative/songwriting-and-ai-music

# 임의의 HTTP(S) URL에서 단일 파일 SKILL.md 직접 설치
hermes skills install https://sharethis.chat/SKILL.md
/skills install https://example.com/SKILL.md --name my-skill
```

설치 시 일어나는 일:
1. 스킬 디렉토리가 `~/.hermes/skills/`에 복사됩니다.
2. `skills_list` 출력에 나타납니다.
3. 슬래시 명령어로 사용할 수 있게 됩니다.

:::tip
설치된 스킬은 새 세션에서 적용됩니다. 현재 세션에서 바로 사용하고 싶다면 `/reset`을 사용하여 새로 시작하거나, `--now`를 추가하여 프롬프트 캐시를 즉시 무효화하세요 (다음 턴에 토큰 비용이 더 발생합니다).
:::

### 설치 확인

```bash
# 스킬이 있는지 확인
hermes skills list | grep arxiv

# 또는 채팅에서
/skills search arxiv
```

---

## 플러그인 제공 스킬

플러그인은 네임스페이스 이름(`plugin:skill`)을 사용하여 자체 스킬을 번들로 제공할 수 있습니다. 이렇게 하면 내장 스킬과의 이름 충돌을 방지할 수 있습니다.

```bash
# 정규화된 이름으로 플러그인 스킬 로드
skill_view("superpowers:writing-plans")

# 같은 기본 이름을 가진 내장 스킬은 영향을 받지 않음
skill_view("writing-plans")
```

플러그인 스킬은 시스템 프롬프트에 **표시되지 않으며** `skills_list`에 나타나지 않습니다. 이들은 옵트인(opt-in) 방식이며 — 플러그인이 스킬을 제공한다는 것을 알 때 명시적으로 로드해야 합니다. 로드되면 에이전트는 동일한 플러그인의 형제 스킬들을 나열하는 배너를 보게 됩니다.

직접 만든 플러그인에 스킬을 포함하는 방법은 [Build a Hermes Plugin → Bundle skills](/guides/build-a-hermes-plugin#bundle-skills)를 참조하세요.

---

## 스킬 설정 구성

일부 스킬은 프런트매터(frontmatter)에 필요한 구성을 선언합니다:

```yaml
metadata:
  hermes:
    config:
      - key: tenor.api_key
        description: "Tenor API key for GIF search"
        prompt: "Enter your Tenor API key"
        url: "https://developers.google.com/tenor/guides/quickstart"
```

구성이 있는 스킬이 처음 로드될 때, Hermes는 값을 입력하라는 메시지를 표시합니다. 이 값들은 `config.yaml`의 `skills.config.*` 아래에 저장됩니다.

CLI에서 스킬 구성을 관리하세요:

```bash
# 특정 스킬을 위한 대화형 구성
hermes skills config gif-search

# 모든 스킬 구성 보기
hermes config show | grep '^skills\.config'
```

---

## 자신만의 스킬 만들기

스킬은 YAML 프런트매터가 있는 마크다운 파일일 뿐입니다. 스킬을 만드는 데는 5분도 채 걸리지 않습니다.

### 1. 디렉토리 생성

```bash
mkdir -p ~/.hermes/skills/my-category/my-skill
```

### 2. SKILL.md 작성

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

## 언제 사용하나요 (When to Use)
사용자가 [특정 주제]에 대해 질문하거나 [특정 작업]을 해야 할 때 이 스킬을 사용하세요.

## 절차 (Procedure)
1. 먼저 [사전 조건]이 사용 가능한지 확인합니다.
2. `command --with-flags`를 실행합니다.
3. 출력을 구문 분석하여 결과를 제시합니다.

## 주의 사항 (Pitfalls)
- 흔한 실패: [설명]. 해결책: [해결책]
- [엣지 케이스]를 주의하세요.

## 검증 (Verification)
결과가 올바른지 확인하려면 `check-command`를 실행하세요.
```

### 3. 참조 파일 추가 (선택 사항)

스킬에는 에이전트가 온디맨드로 로드할 수 있는 지원 파일을 포함할 수 있습니다:

```
my-skill/
├── SKILL.md                    # 메인 스킬 문서
├── references/
│   ├── api-docs.md             # 에이전트가 참조할 수 있는 API 레퍼런스
│   └── examples.md             # 예제 입력/출력
├── templates/
│   └── config.yaml             # 에이전트가 사용할 수 있는 템플릿 파일
└── scripts/
    └── setup.sh                # 에이전트가 실행할 수 있는 스크립트
```

SKILL.md에서 이를 참조하세요:

```markdown
API 세부 정보는 레퍼런스를 로드하세요: `skill_view("my-skill", "references/api-docs.md")`
```

### 4. 테스트하기

새 세션을 시작하고 스킬을 시도해 보세요:

```bash
hermes chat -q "/my-skill 이 작업을 도와줘"
```

스킬은 등록할 필요 없이 자동으로 나타납니다. `~/.hermes/skills/`에 넣기만 하면 즉시 사용 가능합니다.

:::info
에이전트는 `skill_manage`를 사용하여 스스로 스킬을 생성하고 업데이트할 수도 있습니다. 복잡한 문제를 해결한 후, Hermes는 다음번을 위해 해당 접근 방식을 스킬로 저장하겠냐고 제안할 수 있습니다.
:::

---

## 플랫폼별 스킬 관리

어떤 플랫폼에서 어떤 스킬을 사용할 수 있는지 제어하세요:

```bash
hermes skills
```

이 명령어는 플랫폼별로(CLI, Telegram, Discord 등) 스킬을 활성화하거나 비활성화할 수 있는 대화형 TUI를 엽니다. 특정 컨텍스트에서만 특정 스킬을 사용할 수 있도록 제한할 때 유용합니다 — 예를 들어, Telegram에서는 개발 관련 스킬을 숨기는 등의 설정이 가능합니다.

---

## 스킬 vs 메모리

둘 다 세션 간에 지속되지만 서로 다른 목적을 제공합니다:

| | 스킬 (Skills) | 메모리 (Memory) |
|---|---|---|
| **무엇을** | 절차적 지식 — 어떻게 하는지 | 사실적 지식 — 무엇인지 |
| **언제** | 관련이 있을 때만 온디맨드로 로드됨 | 매 세션에 자동으로 주입됨 |
| **크기** | 클 수 있음 (수백 줄) | 요약되어야 함 (주요 사실만) |
| **비용** | 로드될 때까지 토큰 비용 제로 | 작지만 지속적인 토큰 비용 발생 |
| **예시** | "Kubernetes에 배포하는 방법" | "사용자는 다크 모드를 선호하고, PST 시간대에 거주함" |
| **생성 주체** | 사용자, 에이전트 또는 Hub에서 설치 | 대화를 기반으로 에이전트가 생성 |

**경험 법칙:** 참조 문서에 넣을 만한 내용이라면 스킬입니다. 포스트잇에 적어둘 만한 내용이라면 메모리입니다.

---

## 팁

**스킬을 집중적으로 유지하세요.** "모든 DevOps"를 다루려는 스킬은 너무 길고 모호해질 것입니다. "Fly.io에 Python 앱 배포하기"를 다루는 스킬이 실제로 유용할 만큼 충분히 구체적입니다.

**에이전트가 스킬을 만들도록 하세요.** 복잡한 다단계 작업 후에 Hermes는 종종 해당 접근 방식을 스킬로 저장하겠냐고 제안할 것입니다. 이를 수락하세요 — 에이전트가 작성한 이러한 스킬은 과정 중에 발견된 주의 사항(pitfalls)을 포함하여 정확한 워크플로우를 포착합니다.

**카테고리를 사용하세요.** 스킬을 하위 디렉토리로 구성하세요 (`~/.hermes/skills/devops/`, `~/.hermes/skills/research/` 등). 이렇게 하면 목록을 관리하기 쉽고 에이전트가 관련 스킬을 더 빨리 찾는 데 도움이 됩니다.

**기한이 지난 스킬은 업데이트하세요.** 스킬을 사용하다가 문서에 없는 문제에 부딪히면, 새로 배운 내용으로 스킬을 업데이트하라고 Hermes에게 지시하세요. 유지 관리되지 않는 스킬은 방해 요소가 됩니다.

---

*프런트매터 필드, 조건부 활성화, 외부 디렉토리 등을 포함한 전체 스킬 레퍼런스는 [Skills System](/user-guide/features/skills)을 참조하세요.*

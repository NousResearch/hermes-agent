---
sidebar_position: 2
title: "스킬 시스템"
description: "필요할 때 불러오는 지식 문서 — 점진적 공개, 에이전트가 관리하는 스킬, Skills Hub"
---

# 스킬 시스템

스킬은 에이전트가 필요할 때 불러올 수 있는 지식 문서입니다. 토큰 사용량을 최소화하는 **점진적 공개** 패턴을 따르며, [agentskills.io](https://agentskills.io/specification) 오픈 표준과 호환됩니다.

모든 스킬은 **`~/.hermes/skills/`**에 저장됩니다 — 이곳이 기본 디렉터리이자 기준 원본입니다. 새로 설치하면 저장소의 번들 스킬이 복사됩니다. Hub에서 설치한 스킬과 에이전트가 만든 스킬도 이곳에 저장됩니다. 에이전트는 모든 스킬을 수정하거나 삭제할 수 있습니다.

Hermes가 **외부 스킬 디렉터리**를 가리키도록 할 수도 있습니다 — 로컬 디렉터리와 함께 스캔되는 추가 폴더입니다. 아래의 [외부 스킬 디렉터리](#external-skill-directories)를 참고하세요.

참고:

- [번들 스킬 카탈로그](/reference/skills-catalog)
- [공식 선택적 스킬 카탈로그](/reference/optional-skills-catalog)

## 빈 상태에서 시작하기

기본적으로 모든 프로필에는 번들 스킬 카탈로그가 시드되고, 이후 `hermes update`를 실행할 때마다 새로 번들된 스킬이 추가됩니다. **번들 스킬이 전혀 없고** 업데이트 후에도 빈 상태로 유지되는 프로필을 원한다면 두 가지 방법이 있습니다.

**설치 시** (기본 `~/.hermes` 프로필에 적용):

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --no-skills
```

**프로필 생성 시** (이름이 있는 프로필):

```bash
hermes profile create research --no-skills
```

**이미 설치된 프로필에서** (기본 또는 이름이 있는 프로필), 런타임에 토글합니다:

```bash
hermes skills opt-out            # stop future seeding — nothing on disk is touched
hermes skills opt-out --remove   # also delete UNMODIFIED bundled skills (confirms first)
hermes skills opt-in --sync      # undo: remove the marker and re-seed now
```

세 경로 모두 프로필 디렉터리에 `.no-bundled-skills` 마커를 기록합니다. 마커가 있는 동안에는 설치 프로그램, `hermes update`, 모든 스킬 동기화가 해당 프로필의 번들 스킬 시드를 건너뜁니다. 마커를 삭제하거나 `hermes skills opt-in`을 실행하면 다시 활성화됩니다.

:::note 기본적으로 안전
`hermes skills opt-out`은 **향후** 시드만 중지하며, 이미 디스크에 있는 것은 절대 삭제하지 않습니다. 선택적 `--remove` 플래그는 수정되지 않은 경우에만 번들 스킬을 삭제합니다(Hermes가 설치한 버전과 바이트 단위로 동일한 경우). 수정한 스킬, Hub에서 설치한 스킬, 직접 작성한 스킬은 항상 보존됩니다.
:::

## 스킬 사용하기

설치된 모든 스킬은 자동으로 슬래시 명령으로 사용할 수 있습니다:

```bash
# In the CLI or any messaging platform:
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor
/plan design a rollout for migrating our auth provider

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

### 하나의 명령에서 여러 스킬 쌓기

메시지의 시작 부분에서 슬래시 명령을 연결하면 여러 스킬을 한 번에 호출할 수 있습니다 — 맨 앞의 `/skill` 토큰이 최대 5개까지 로드되고, 나머지는 지시가 됩니다:

```bash
/github-pr-workflow /test-driven-development fix issue #123 and open a PR
```

설치는 첫 번째 토큰이 설치된 스킬이 아닐 때 중지되므로, `/`로 시작하는 인자(파일 경로 등)가 실수로 소비되지 않습니다:

```bash
/ocr-and-documents /tmp/scan.pdf extract the tables   # loads one skill; /tmp/scan.pdf is the argument
```

반복해서 사용하는 조합이라면 [스킬 번들](#skill-bundles)을 사용하는 것이 좋습니다 — 하나의 짧은 명령으로 같은 효과를 얻을 수 있습니다.

번들된 `plan` 스킬이 좋은 예입니다. `/plan [request]`를 실행하면 스킬의 지시에 따라 Hermes가 필요할 경우 컨텍스트를 조사하고, 작업을 실행하는 대신 마크다운 구현 계획을 작성하며, 활성 워크스페이스/백엔드 작업 디렉터리를 기준으로 `.hermes/plans/` 아래에 결과를 저장합니다.

스킬은 자연어 대화로도 사용할 수 있습니다:

```bash
hermes chat --toolsets skills -q "What skills do you have?"
hermes chat --toolsets skills -q "Show me the axolotl skill"
```

## 소스에서 스킬 학습하기 (`/learn`)

`/learn`은 이미 알고 있는 내용이나 참고 자료 모음을 손으로 `SKILL.md`를 작성하지 않고 재사용 가능한 스킬로 빠르게 바꾸는 방법입니다. 열린 범위의 기능으로, 에이전트가 이미 가진 도구로 자료를 수집한 다음 [하우스 작성 표준](#skillmd-format)(60자 이하 설명, 표준 섹션 순서, Hermes 도구 프레이밍, 명령어를 지어내지 않음)에 맞는 스킬을 작성합니다.

```bash
# A local SDK or doc directory — read with read_file / search_files
/learn the REST client in ~/projects/acme-sdk, focus on auth + pagination

# An online doc page — fetched with web_extract
/learn https://docs.example.com/api/quickstart

# The workflow you just walked the agent through in this conversation
/learn how I just deployed the staging server

# Pasted notes / a described procedure
/learn filing an expense: open the portal, New > Expense, attach the receipt, submit

# A whole book, paper stack, or large docs corpus — becomes a knowledge-base skill
/learn ~/books/designing-data-intensive-applications.pdf
```

### 대규모 소스는 지식 기반 스킬이 됩니다

소스가 책, 논문 모음, 사양서 또는 대규모 문서 폴더인 경우 에이전트는 이를 하나의 파일에 억지로 넣거나 손실이 큰 요약으로 줄이지 않습니다. 대신 소스의 핵심 멘탈 모델과 함께 색인을 담은 간결한 `SKILL.md`와, 장 또는 주제별로 정리된 요약 파일을 `references/` 아래에 두는 **확장형 지식 기반 스킬**을 작성합니다(필요한 경우 용어집 또는 치트시트도 포함). 참고 파일은 질문에 필요할 때까지 비용이 들지 않으며, 에이전트는 `skill_view`로 필요할 때 불러옵니다. 따라서 질의 비용은 소스의 크기가 아니라 답변에 필요한 양에 비례합니다. 같은 주제에 새 자료로 `/learn`을 다시 실행하면 중복을 만들지 않고 기존 스킬에 통합합니다.

이렇게 추출한 내용은 구조(프레임워크, 정의, 결정 규칙, 안티패턴)를 종합하며, 원문을 그대로 재현하지 않습니다.

실시간 에이전트가 소스를 수집하므로 `/learn`은 CLI, 메시징 게이트웨이, TUI, 대시보드에서 동일하게 작동하며 모든 터미널 백엔드(로컬, Docker, 원격)에서도 작동합니다. 별도의 수집 엔진이 필요하지 않기 때문입니다. **대시보드**의 Skills 페이지에는 디렉터리 필드, URL 필드, 자유 입력 텍스트 상자가 있는 패널을 여는 **스킬 학습** 버튼이 있습니다. 이 버튼은 `/learn` 요청을 구성하여 채팅에서 실행합니다.

모델 도구의 표면적은 늘어나지 않습니다: `/learn`은 표준에 맞춘 프롬프트를 만들고 일반적인 턴으로 에이전트에 전달합니다. 에이전트는 `skill_manage` 도구로 결과를 저장하므로, [쓰기 승인 게이트](#gating-agent-skill-writes-skillswrite_approval)를 켜 둔 경우 해당 게이트가 적용됩니다.

## 점진적 공개

스킬은 토큰을 절약하는 로딩 패턴을 사용합니다:

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

에이전트는 실제로 필요할 때만 스킬의 전체 내용을 로드합니다.

## SKILL.md 형식

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
platforms: [macos, linux]     # Optional — restrict to specific OS platforms
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # Optional — conditional activation (see below)
    requires_toolsets: [terminal]   # Optional — conditional activation (see below)
    config:                          # Optional — config.yaml settings
      - key: my.setting
        description: "What this controls"
        default: "value"
        prompt: "Prompt for setup"
---

# Skill Title

## When to Use
Trigger conditions for this skill.

## Procedure
1. Step one
2. Step two

## Pitfalls
- Known failure modes and fixes

## Verification
How to confirm it worked.
```

### 플랫폼별 스킬

스킬은 `platforms` 필드를 사용하여 특정 운영 체제로 제한할 수 있습니다:

| 값 | 일치하는 플랫폼 |
|-------|---------|
| `macos` | macOS (Darwin) |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders, FindMy)
platforms: [macos, linux]     # macOS and Linux
```

설정하면 호환되지 않는 플랫폼에서는 시스템 프롬프트, `skills_list()`, 슬래시 명령에서 스킬이 자동으로 숨겨집니다. 생략하면 모든 플랫폼에서 스킬이 로드됩니다.

## 스킬 출력 및 미디어 전달

스킬 응답(또는 모든 에이전트 응답)에 미디어 파일의 절대 경로가 그대로 포함되면 — 예를 들어 `/home/user/screenshots/diagram.png` — 게이트웨이가 이를 자동으로 감지하고 화면에 표시되는 텍스트에서 경로를 제거한 다음, 원시 경로를 메시지에 남기는 대신 사용자의 채팅으로 파일을 네이티브 방식으로 전달합니다(Telegram 사진, Discord 첨부 파일 등).

오디오의 경우 `[[audio_as_voice]]` 지시어를 사용하면 이를 지원하는 플랫폼(Telegram, WhatsApp)에서 오디오 파일이 네이티브 음성 메시지 버블로 전달됩니다.

### 문서 스타일 전달 강제하기: `[[as_document]]`

때로는 인라인 미리보기와 반대로 파일을 다운로드 가능한 첨부 파일로 전달하고 싶을 수 있습니다. 대표적인 예는 고해상도 스크린샷이나 차트입니다 — Telegram의 `sendPhoto`는 이를 약 200KB, 1280px로 다시 압축하여 가독성을 해치지만, 1~2MB PNG를 `sendDocument`로 보내면 원본 바이트를 그대로 유지합니다.

응답(일반적으로 마지막 줄 등 응답 안의 텍스트)에 `[[as_document]]`라는 지시어를 그대로 포함하면 해당 응답에서 추출된 모든 미디어 경로가 이미지 버블 대신 문서/파일 첨부로 전달됩니다:

```
Here is your rendered chart:

/home/user/.hermes/cache/chart-q4-2025.png

[[as_document]]
```

다음과 같은 경우 스킬에서 사용하세요:

- 사용자가 다른 도구에서 편집하거나 보관하거나 원본 그대로 공유할 수 있도록 스크린샷이나 차트를 파일로 제공하는 경우
- 기본 손실 미리보기가 세부 정보(작은 글자, 픽셀 단위로 정확한 다이어그램, 색상에 민감한 렌더링)를 가릴 경우

별도의 문서 경로가 없는 플랫폼(예: SMS)은 해당 플랫폼이 제공하는 첨부 방식으로 대체합니다.

### 조건부 활성화 (대체 스킬)

스킬은 현재 세션에서 사용할 수 있는 도구에 따라 자동으로 표시되거나 숨겨질 수 있습니다. 이는 **대체 스킬**에 가장 유용합니다 — 프리미엄 도구를 사용할 수 없을 때만 표시되어야 하는 무료 또는 로컬 대안입니다.

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

| 필드 | 동작 |
|----------|----------|
| `fallback_for_toolsets` | 나열된 도구 세트를 사용할 수 있을 때 스킬을 **숨깁니다**. 사용할 수 없을 때 표시합니다. |
| `fallback_for_tools` | 동일하지만 도구 세트가 아닌 개별 도구를 확인합니다. |
| `requires_toolsets` | 나열된 도구 세트를 사용할 수 없을 때 스킬을 **숨깁니다**. 사용할 수 있을 때 표시합니다. |
| `requires_tools` | 동일하지만 개별 도구를 확인합니다. |

**예시:** 기본 제공 `duckduckgo-search` 스킬은 `fallback_for_toolsets: [web]`을 사용합니다. `FIRECRAWL_API_KEY`가 설정되어 있으면 웹 도구 세트를 사용할 수 있으므로 에이전트는 `web_search`를 사용하고 DuckDuckGo 스킬은 숨겨진 상태로 유지됩니다. API 키가 없으면 웹 도구 세트를 사용할 수 없으므로 DuckDuckGo 스킬이 자동으로 대체 스킬로 표시됩니다.

조건부 필드가 없는 스킬은 이전과 정확히 동일하게 동작합니다 — 항상 표시됩니다.

## 로드 시 보안 설정

스킬은 검색 목록에서 사라지지 않고 필요한 환경 변수를 선언할 수 있습니다:

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

누락된 값이 발견되면 Hermes는 로컬 CLI에서 스킬이 실제로 로드될 때만 안전하게 값을 요청합니다. 설정을 건너뛰고 스킬을 계속 사용할 수도 있습니다. 메시징 화면은 채팅에서 비밀을 요청하지 않습니다 — 대신 로컬에서 `hermes setup` 또는 `~/.hermes/.env`를 사용하라고 안내합니다.

설정되면 선언된 환경 변수는 `execute_code` 및 `terminal` 샌드박스로 **자동 전달**됩니다 — 스킬의 스크립트에서 `$TENOR_API_KEY`를 직접 사용할 수 있습니다. 스킬이 아닌 환경 변수에는 `terminal.env_passthrough` 설정 옵션을 사용하세요. 자세한 내용은 [환경 변수 전달](/user-guide/security#environment-variable-passthrough)을 참고하세요.

### 스킬 구성 설정

스킬은 `config.yaml`에 저장되는 비밀이 아닌 구성 설정(경로, 기본 설정)도 선언할 수 있습니다:

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
```

설정은 config.yaml의 `skills.config` 아래에 저장됩니다. `hermes config migrate`는 구성되지 않은 설정의 입력을 요청하고, `hermes config show`는 설정을 표시합니다. 스킬이 로드되면 확인된 구성 값이 컨텍스트에 주입되므로 에이전트가 설정된 값을 자동으로 알 수 있습니다.

[스킬 설정](/user-guide/configuration#skill-settings) 및 [스킬 만들기 — 구성 설정](/developer-guide/creating-skills#config-settings-configyaml)에서 자세히 알아보세요.

## 스킬 디렉터리 구조

```text
~/.hermes/skills/                  # Single source of truth
├── mlops/                         # Category directory
│   ├── axolotl/
│   │   ├── SKILL.md               # Main instructions (required)
│   │   ├── references/            # Additional docs
│   │   ├── templates/             # Output formats
│   │   ├── scripts/               # Helper scripts callable from the skill
│   │   ├── examples/              # Referenced example outputs
│   │   └── assets/                # Supplementary files
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # Agent-created skill
│       ├── SKILL.md
│       └── references/
├── .hub/                          # Skills Hub state
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # Tracks seeded bundled skills
```

서드파티 URL 및 GitHub 설치에는 `SKILL.md`와 `references/`, `templates/`, `scripts/`, `assets/`, `examples/` 아래에서 해당 파일이 참조하는 정확한 로컬 파일이 포함됩니다. 참조되지 않은 저장소 파일은 복사되지 않습니다. Hermes는 격리된 전체 번들을 스캔하고 소스 URL, 정확한 콘텐츠 해시, 스캐너 버전, 발견 항목, 타임스탬프, 신규/캐시 상태를 `skills/.hub/lock.json`에 기록합니다.

## 외부 스킬 디렉터리

Hermes 외부에서 스킬을 관리하는 경우 — 예를 들어 여러 AI 도구가 함께 사용하는 `~/.agents/skills/` 디렉터리 — Hermes에 해당 디렉터리도 스캔하도록 지정할 수 있습니다.

`~/.hermes/config.yaml`의 `skills` 섹션 아래에 `external_dirs`를 추가합니다:

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

경로는 `~` 확장과 `${VAR}` 환경 변수 치환을 지원합니다.

### 작동 방식

- **로컬에서 생성, 제자리에서 업데이트**: 새 에이전트 생성 스킬은 `~/.hermes/skills/`에 기록됩니다. 기존 스킬은 `external_dirs` 아래의 스킬을 포함하여 발견된 위치에서 수정되며, 에이전트가 `patch`, `edit`, `write_file`, `remove_file`, `delete` 같은 `skill_manage` 작업을 사용할 때 적용됩니다.
- **외부 디렉터리는 쓰기 보호 경계가 아닙니다**: 외부 스킬 디렉터리가 Hermes 프로세스에 쓰기 가능한 경우 에이전트가 관리하는 스킬 업데이트로 해당 디렉터리의 파일이 변경될 수 있습니다. 공유 외부 스킬을 읽기 전용으로 유지해야 한다면 파일 시스템 권한 또는 별도의 프로필/도구 세트 설정을 사용하세요.
- **로컬 우선순위**: 같은 스킬 이름이 로컬 디렉터리와 외부 디렉터리에 모두 있으면 로컬 버전이 우선합니다.
- **완전한 통합**: 외부 스킬은 시스템 프롬프트 색인, `skills_list`, `skill_view`, `/skill-name` 슬래시 명령에 표시되며 로컬 스킬과 다르지 않습니다.
- **존재하지 않는 경로는 조용히 건너뜁니다**: 구성된 디렉터리가 존재하지 않으면 Hermes는 오류 없이 무시합니다. 모든 환경에 없을 수 있는 선택적 공유 디렉터리에 유용합니다.

### 예시

```text
~/.hermes/skills/               # Local (primary, read-write)
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # External (shared, mutable if writable)
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

네 개의 스킬이 모두 스킬 색인에 표시됩니다. `my-custom-workflow`라는 새 스킬을 로컬에서 만들면 외부 버전을 가립니다.

## 스킬 번들

스킬 번들은 여러 스킬을 하나의 슬래시 명령으로 묶는 작은 YAML 파일입니다. `/<bundle-name>`을 실행하면 번들에 나열된 모든 스킬이 한 번에 로드되며, 특정 작업에서 항상 함께 사용하면 좋은 스킬들을 묶을 때 유용합니다.

### 간단한 예시

```bash
# Create a bundle for backend feature work
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "Backend feature work — review, test, PR workflow"
```

그런 다음 CLI 또는 모든 게이트웨이 플랫폼에서:

```
/backend-dev refactor the auth middleware
```

에이전트는 세 스킬을 하나의 사용자 메시지에 로드한 상태로 받으며, 슬래시 명령 뒤의 모든 텍스트는 사용자 지시로 첨부됩니다.

### YAML 스키마

번들은 **`~/.hermes/skill-bundles/<slug>.yaml`**에 저장되며 다음과 같은 형태입니다:

```yaml
name: backend-dev
description: Backend feature work — review, test, PR workflow.
skills:
  - github-code-review
  - test-driven-development
  - github-pr-workflow
instruction: |
  Always start by writing failing tests, then implement.
  Open the PR through the standard workflow with co-author tags.
```

필드:
- `name` (선택 사항 — 기본값은 파일명에서 확장자를 뺀 부분) — 번들의 표시 이름입니다. 슬래시 명령에서는 하이픈 슬러그로 정규화됩니다(`Backend Dev` → `/backend-dev`).
- `description` (선택 사항) — `/bundles` 및 `hermes bundles list`에 표시되는 짧은 텍스트입니다.
- `skills` (필수, 비어 있지 않은 목록) — 스킬 이름 또는 스킬 디렉터리에 대한 상대 경로입니다. `/<skill-name>`에 전달할 때와 같은 식별자를 사용하세요.
- `instruction` (선택 사항) — 로드된 스킬 콘텐츠 앞에 추가되는 지침입니다. 이 스킬들을 함께 사용할 때의 방식을 정형화하는 데 유용합니다.

### 번들 관리

```bash
# List all installed bundles
hermes bundles list

# Inspect one bundle
hermes bundles show backend-dev

# Create a bundle interactively (omit --skill flags to enter them one per line)
hermes bundles create research

# Overwrite an existing bundle
hermes bundles create backend-dev --skill ... --force

# Delete a bundle
hermes bundles delete backend-dev

# Re-scan ~/.hermes/skill-bundles/ and report changes
hermes bundles reload
```

채팅 세션 안에서 `/bundles`를 실행하면 설치된 모든 번들과 해당 스킬이 나열됩니다.

### 동작

- **번들은 개별 스킬보다 우선합니다**: 슬러그가 충돌하는 경우 번들이 우선합니다. 번들 이름을 `research`로 지정했는데 `research`라는 스킬도 있다면 `/research`는 번들을 호출합니다. 이는 이름을 지정하여 번들을 선택적으로 활성화했기 때문에 의도된 동작입니다.
- **없는 스킬은 치명적 오류가 아니라 건너뜁니다**: 번들에 `skill-foo`가 나열되어 있지만 설치하지 않았다면 번들은 확인되는 스킬을 계속 로드하고, 건너뛴 항목을 나열한 메모를 에이전트에 전달합니다.
- **번들은 모든 화면에서 작동합니다**: 대화형 CLI, TUI, 대시보드 채팅, 모든 게이트웨이 플랫폼(Telegram, Discord, Slack, …)에서 작동합니다. 개별 스킬 명령과 같은 위치에서 디스패치가 중앙화되어 있기 때문입니다.
- **번들은 프롬프트 캐시를 무효화하지 않습니다**: `/<skill-name>`과 같은 방식으로 호출 시점에 새로운 사용자 메시지를 생성하며, 시스템 프롬프트를 변경하지 않습니다.

### 각 스킬을 수동으로 설치하는 것보다 번들이 유리한 경우

다음과 같은 경우 번들을 사용하세요:
- 반복 작업에서 항상 같은 스킬을 함께 사용하는 경우(`/backend-dev`, `/release-prep`, `/incident-response`).
- 여러 `/skill` 호출을 연달아 입력하는 것보다 한 글자 짧은 멘탈 모델을 원하는 경우.
- 팀 전체의 "작업 프로필"을 공유하려는 경우 — 번들 YAML을 공유 dotfiles 저장소에 커밋하고 `~/.hermes/skill-bundles/`에 심볼릭 링크하면 됩니다.

번들은 YAML 별칭일 뿐이며 스킬을 대신 설치해 주지 않습니다. 스킬 자체가 이미 `~/.hermes/skills/` 또는 외부 스킬 디렉터리에 있어야 합니다. 그렇지 않으면 번들을 호출해도 없는 스킬은 건너뜁니다.

## 에이전트가 관리하는 스킬 (`skill_manage` 도구)

에이전트는 `skill_manage` 도구를 사용하여 자신의 스킬을 만들고, 업데이트하고, 삭제할 수 있습니다. 이는 에이전트의 **절차적 메모리**입니다 — 복잡한 작업 방식을 알아내면 나중에 다시 사용할 수 있도록 스킬로 저장합니다.

스킬과 메모리는 자기 개선 루프에서 함께 작동합니다. 메모리는 항상 컨텍스트에 있어야 하는 작고 지속적인 사실을 저장하고, 스킬은 관련 있을 때만 로드해야 하는 더 긴 절차를 저장합니다. 백그라운드 검토는 세션 후 스킬 변경을 제안하거나 준비할 수 있지만, 아래의 쓰기 승인 게이트를 사용하면 해당 변경 사항이 반영되기 전에 사람의 검토를 요구할 수 있습니다.

### 에이전트가 스킬을 만드는 경우

시스템 프롬프트는 에이전트가 중요하고 반복할 가치가 있는 작업 방식을 `skill_manage`로 기록하도록 요청합니다. 실제로는 다음과 같은 경우가 해당됩니다:

- 반복할 가치가 있는 여러 단계의 작업 방식을 알아낸 경우
- 오류나 막다른 길을 만났지만 작동하는 경로를 찾은 경우
- 사용자가 에이전트의 접근 방식을 수정한 경우

### 작업

| 작업 | 용도 | 주요 매개변수 |
|--------|--------|------------|
| `create` | 처음부터 새 스킬 만들기 | `name`, `content` (전체 SKILL.md), 선택 사항인 `category` |
| `patch` | 대상 수정(권장) | `name`, `old_string`, `new_string` |
| `edit` | 주요 구조 재작성 | `name`, `content` (전체 SKILL.md 교체) |
| `delete` | 스킬 완전히 제거 | `name` |
| `write_file` | 지원 파일 추가/업데이트 | `name`, `file_path`, `file_content` |
| `remove_file` | 지원 파일 제거 | `name`, `file_path` |

:::tip
업데이트에는 `patch` 작업을 권장합니다 — 변경된 텍스트만 도구 호출에 포함되므로 `edit`보다 토큰 효율이 높습니다.
:::

### 에이전트 스킬 쓰기 제한 (`skills.write_approval`)

기본적으로 에이전트는 백그라운드 [자기 개선 검토](/user-guide/features/memory#controlling-memory-writes-write_approval)에서 발생한 변경을 포함하여 스킬을 자유롭게 작성합니다. 모든 스킬 쓰기를 먼저 승인하고 싶다면(작은 모델이 학습한 내용을 잘못 판단하는 경우, 보안 환경, 또는 자기 개선 루프를 직접 확인하고 싶은 경우) 쓰기 승인 게이트를 켜세요:

```yaml
skills:
  write_approval: false     # false = write freely (default) | true = require approval
```

`write_approval: true`이면 모든 `skill_manage` 쓰기(create / edit / patch / delete / write_file / remove_file)가 커밋되지 않고 **준비 상태**가 됩니다 — SKILL.md는 인라인으로 검토하기에 너무 크므로, 쓰기가 포그라운드 턴에서 발생했는지 백그라운드 검토에서 발생했는지와 무관하게 준비 상태로 처리됩니다. 준비된 쓰기는 `~/.hermes/pending/skills/`에 재시작 후에도 남아 있으며 위험한 명령과 동일한 익숙한 승인/거부 흐름으로 검토합니다:

```
/skills pending             # list staged skill writes + a one-line gist each
/skills diff <id>           # full unified diff (best viewed in CLI or dashboard)
/skills approve <id>        # apply it (or 'all')
/skills reject <id>         # drop it (or 'all')
/skills approval on         # turn the gate on (or 'off') and persist it
```

검토 화면은 대화형 CLI와 메시징 플랫폼에서 작동합니다(채팅 버블에서는 diff 출력이 잘리므로 CLI 또는 pending JSON 파일에서 전체 diff를 읽으세요). 메모리 쓰기에도 `memory.write_approval` 아래에 동일한 게이트가 적용됩니다 — [메모리 쓰기 제어](/user-guide/features/memory#controlling-memory-writes-write_approval)를 참고하세요.

> 별도의 `skills.guard_agent_created` 설정은 콘텐츠 스캐너(위험 패턴 휴리스틱)이지 승인 게이트가 아닙니다 — 두 설정은 서로 독립적입니다. [에이전트가 만든 스킬 쓰기 시 보호](/user-guide/configuration#guard-on-agent-created-skill-writes)를 참고하세요.

## Skills Hub

온라인 레지스트리, `skills.sh`, 직접 지정하는 well-known 스킬 엔드포인트 및 공식 선택적 스킬에서 스킬을 탐색하고, 검색하고, 설치하고, 관리합니다.

### 일반적인 명령

```bash
hermes skills browse                              # Browse all hub skills (official first)
hermes skills browse --source official            # Browse only official optional skills
hermes skills search kubernetes                   # Search all sources
hermes skills search react --source skills-sh     # Search the skills.sh directory
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # Preview before installing
hermes skills install openai/skills/k8s           # Install with security scan
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # Direct URL (+ referenced support files)
hermes skills install https://example.com/SKILL.md --name my-skill # Override name when frontmatter has none
hermes skills list --source hub                   # List hub-installed skills
hermes skills check                               # Check installed hub skills for upstream updates
hermes skills update                              # Reinstall hub skills with upstream changes when needed
hermes skills audit                               # Re-scan all hub skills for security
hermes skills uninstall k8s                       # Remove a hub skill
hermes skills reset google-workspace              # Un-stick a bundled skill from "user-modified" (see below)
hermes skills reset google-workspace --restore    # Also restore the bundled version, deleting your local edits
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # Export skill config
hermes skills tap add myorg/skills-repo           # Add a custom GitHub source
```

### 지원되는 Hub 소스

| 소스 | 예시 | 비고 |
|--------|---------|-------|
| `official` | `official/security/1password` | Hermes와 함께 제공되는 선택적 스킬입니다. |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | `hermes skills search <query> --source skills-sh`로 검색할 수 있습니다. skills.sh 슬러그가 저장소 폴더와 다르면 Hermes가 별칭 스타일 스킬을 확인합니다. |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | 웹사이트의 `/.well-known/skills/index.json`에서 직접 제공되는 스킬입니다. 사이트 또는 문서 URL을 사용하여 검색합니다. |
| `url` | `https://sharethis.chat/SKILL.md` | 직접 HTTP(S) URL입니다. 명칭 확인 순서는 프런트매터 → URL 슬러그 → 대화형 프롬프트 → `--name` 플래그입니다. |
| `github` | `openai/skills/k8s` | 직접 GitHub 저장소/경로 설치와 사용자 지정 tap입니다. |
| `clawhub`, `lobehub`, `browse-sh` | 소스별 식별자 | 커뮤니티 또는 마켓플레이스 통합입니다. |

### 통합된 Hub 및 레지스트리

Hermes는 현재 다음 스킬 생태계 및 검색 소스와 통합됩니다:

#### 1. 공식 선택적 스킬 (`official`)

Hermes 저장소 자체에서 관리되며 기본 신뢰로 설치되는 스킬입니다.

- 카탈로그: [공식 선택적 스킬 카탈로그](../../reference/optional-skills-catalog)
- 저장소 내 소스: `optional-skills/`
- 예시:

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh (`skills-sh`)

Vercel의 공개 스킬 디렉터리입니다. Hermes는 이를 직접 검색하고, 스킬 상세 페이지를 확인하고, 별칭 스타일 슬러그를 확인하며, 실제 소스 저장소에서 설치할 수 있습니다.

- 디렉터리: [skills.sh](https://skills.sh/)
- CLI/도구 저장소: [vercel-labs/skills](https://github.com/vercel-labs/skills)
- 공식 Vercel 스킬 저장소: [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- 예시:

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. Well-known 스킬 엔드포인트 (`well-known`)

`/.well-known/skills/index.json`을 게시하는 사이트에서 URL 기반으로 검색합니다. 하나의 중앙화된 Hub가 아니라 웹 검색 규약입니다.

- 실제 엔드포인트 예시: [Mintlify 문서 스킬 색인](https://mintlify.com/docs/.well-known/skills/index.json)
- 참조 서버 구현: [vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- 예시:

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. 직접 GitHub 스킬 (`github`)

Hermes는 GitHub 저장소 및 GitHub 기반 tap에서 직접 설치할 수 있습니다. 저장소/경로를 이미 알고 있거나 자체 사용자 지정 소스 저장소를 추가하려는 경우 유용합니다.

기본 tap(설정 없이 탐색 가능):
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [huggingface/skills](https://github.com/huggingface/skills)
- [NVIDIA/skills](https://github.com/NVIDIA/skills) — NVIDIA 검증 스킬(서명된 `skill.oms.sig` 및 거버넌스 `skill-card.md`)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- 예시:

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

**카테고리 그룹화(`skills.sh.json`).** GitHub tap은 저장소 루트에 [skills.sh 스키마](https://skills.sh/schemas/skills.sh.schema.json)에 맞는 `skills.sh.json` 파일을 제공할 수 있습니다. 해당 파일의 `groupings`(`title`과 스킬 이름 목록으로 구성)는 색인 생성 시 읽히고 [Skills Hub](https://hermes-agent.nousresearch.com/docs) 페이지에 표시되는 카테고리 라벨이 됩니다 — 태그에서 추정한 분류 대신 사용됩니다. 이는 범용적이므로 해당 파일을 제공하는 모든 tap에 실제 분류가 적용되며 Hermes 측 변경은 필요하지 않습니다.

```json
{
  "$schema": "https://skills.sh/schemas/skills.sh.schema.json",
  "groupings": [
    { "title": "Inference AI", "skills": ["dynamo-recipe-runner", "dynamo-router-sla"] },
    { "title": "Decision Optimization", "skills": ["cuopt-developer", "cuopt-install"] }
  ]
}
```

#### 5. ClawHub (`clawhub`)

커뮤니티 소스로 통합된 서드파티 스킬 마켓플레이스입니다.

- 사이트: [clawhub.ai](https://clawhub.ai/)
- Hermes 소스 ID: `clawhub`

#### 6. LobeHub (`lobehub`)

LobeHub의 공개 카탈로그에서 에이전트 항목을 검색하고 설치 가능한 Hermes 스킬로 변환할 수 있습니다.

- 사이트: [LobeHub](https://lobehub.com/)
- 공개 에이전트 색인: [chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- 기반 저장소: [lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Hermes 소스 ID: `lobehub`

#### 7. browse.sh (`browse-sh`)

Hermes는 Browserbase의 200개 이상 사이트별 브라우저 자동화 `SKILL.md` 파일 카탈로그인 [browse.sh](https://browse.sh)와 통합됩니다(Airbnb, Amazon, arXiv, 12306.cn, Etsy, Xero 등). 각 스킬은 하나의 웹사이트를 처음부터 끝까지 조작하는 방법을 설명하며 Hermes의 브라우저 도구 및 이미 설치한 브라우저 자동화 스킬과 함께 사용하기에 적합합니다.

- 사이트: [browse.sh](https://browse.sh/)
- 카탈로그 API: `https://browse.sh/api/skills`
- Hermes 소스 ID: `browse-sh`
- 신뢰 수준: `community`

```bash
hermes skills search airbnb --source browse-sh
hermes skills inspect browse-sh/airbnb.com/search-listings-ddgioa
hermes skills install browse-sh/airbnb.com/search-listings-ddgioa
```

식별자는 `browse-sh/<hostname>/<task-id>` 형식을 사용하며 browse.sh 카탈로그에 공개된 슬러그와 일치합니다. 콘텐츠는 카탈로그의 GitHub `sourceUrl`이 아니라 스킬별 상세 엔드포인트(`/api/skills/<slug>` → `skillMdUrl`)를 통해 확인됩니다.

#### 8. 직접 URL (`url`)

모든 HTTP(S) URL에서 `SKILL.md`를 직접 설치할 수 있습니다 — Hub 목록이나 입력할 GitHub 경로 없이 작성자가 자신의 사이트에 호스팅한 스킬에 유용합니다. Hermes는 `references/`, `templates/`, `scripts/`, `assets/`, `examples/` 아래에서 명시적으로 참조된 파일도 가져온 다음 전체 번들을 스캔하고 설치합니다.

- Hermes 소스 ID: `url`
- 식별자: URL 자체(접두사 없음)
- 범위: 허용 목록에 있는 디렉터리의 `SKILL.md`와 정확히 참조된 지원 파일입니다. Hermes는 호스트에서 관련 없는 파일을 열거하거나 복사하지 않습니다.

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

명칭 확인 순서:
1. SKILL.md YAML 프런트매터의 `name:` 필드(권장 — 형식이 올바른 모든 스킬에는 이 필드가 있습니다).
2. URL 경로의 상위 디렉터리 이름(예: `.../my-skill/SKILL.md` → `my-skill`, 또는 `.../my-skill.md` → `my-skill`) — 유효한 식별자(`^[a-z][a-z0-9_-]*$`)인 경우.
3. TTY가 있는 터미널에서 대화형 프롬프트.
4. 비대화형 화면(`/skills install` 슬래시 명령을 TUI, 게이트웨이 플랫폼, 스크립트에서 실행하는 경우)에서는 `--name` 재정의를 안내하는 명확한 오류가 발생합니다.

```bash
# Frontmatter has no name and the URL slug is unhelpful — supply one:
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# Or inside a chat session:
/skills install https://example.com/SKILL.md --name sharethis-chat
```

신뢰 수준은 항상 `community`입니다 — 다른 모든 소스와 동일한 보안 스캔이 실행됩니다. URL은 설치 식별자로 저장되므로, 새로 고치고 싶을 때 `hermes skills update`가 동일한 URL에서 자동으로 다시 가져옵니다.

### 보안 스캔 및 `--force`

Hub에서 설치한 모든 스킬은 데이터 유출, 프롬프트 인젝션, 파괴적인 명령, 공급망 신호 및 기타 위협을 확인하는 **보안 스캐너**를 거칩니다.

이제 `hermes skills inspect ...`는 가능한 경우 업스트림 메타데이터도 표시합니다:
- 저장소 URL
- skills.sh 상세 페이지 URL
- 설치 명령
- 주간 설치 수
- 업스트림 보안 감사 상태
- well-known 색인/엔드포인트 URL

서드파티 스킬을 검토했고 위험하지 않은 정책 차단을 재정의하려는 경우 `--force`를 사용하세요:

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

중요한 동작:
- `--force`는 주의/경고 유형의 발견 항목에 대한 정책 차단을 재정의할 수 있습니다.
- `--force`는 `dangerous` 스캔 판정을 재정의하지 않습니다.
- 공식 선택적 스킬(`official/...`)은 기본 제공 신뢰로 처리되며 서드파티 경고 패널을 표시하지 않습니다.

### 신뢰 수준

| 수준 | 소스 | 정책 |
|-------|--------|--------|
| `builtin` | Hermes와 함께 제공 | 항상 신뢰 |
| `official` | 저장소의 `optional-skills/` | 기본 제공 신뢰, 서드파티 경고 없음 |
| `trusted` | `openai/skills`, `anthropics/skills`, `huggingface/skills`, `NVIDIA/skills` 같은 신뢰 레지스트리/저장소 | 커뮤니티 소스보다 허용적인 정책 |
| `community` | 그 외 모든 것(`skills.sh`, well-known 엔드포인트, 사용자 지정 GitHub 저장소, 대부분의 마켓플레이스) | 위험하지 않은 발견 항목은 `--force`로 재정의 가능; `dangerous` 판정은 계속 차단 |

### 업데이트 수명 주기

Hub는 설치된 스킬의 업스트림 사본을 다시 확인할 수 있도록 충분한 출처 정보를 추적합니다:

```bash
hermes skills check          # Report which installed hub skills changed upstream
hermes skills update         # Reinstall only the skills with updates available
hermes skills update react   # Update one specific installed hub skill
```

이는 저장된 소스 식별자와 현재 업스트림 번들 콘텐츠 해시를 사용하여 변경을 감지합니다.

:::tip GitHub 요청 제한
Skills Hub 작업은 시간당 인증되지 않은 사용자에게 60건의 요청 제한이 있는 GitHub API를 사용합니다. 설치 또는 검색 중 요청 제한 오류가 표시되면 `.env` 파일에 `GITHUB_TOKEN`을 설정하여 제한을 시간당 5,000건으로 늘리세요. 이 경우 오류 메시지에 실행 가능한 안내가 포함됩니다.
:::

### 사용자 지정 스킬 tap 게시

팀, 조직 또는 공개적으로 선별한 스킬 세트를 공유하려면 **tap**으로 게시할 수 있습니다. 다른 Hermes 사용자는 `hermes skills tap add <owner/repo>`로 추가합니다. 서버도, 레지스트리 가입도, 릴리스 파이프라인도 필요하지 않습니다. `SKILL.md` 파일로 구성된 디렉터리 하나면 됩니다.

#### 저장소 구조

tap은 다음과 같은 구조를 갖는 모든 GitHub 저장소(공개 또는 비공개 — 비공개 저장소에는 `GITHUB_TOKEN` 필요)입니다:

```
owner/repo
├── skills/                       # default path; configurable per-tap
│   ├── my-workflow/
│   │   ├── SKILL.md              # required
│   │   ├── references/           # optional supporting files
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # optional but helpful
```

규칙:
- 각 스킬은 tap의 루트 경로(기본값 `skills/`) 아래의 자체 디렉터리에 있습니다.
- 디렉터리 이름이 스킬의 설치 슬러그가 됩니다.
- 각 스킬 디렉터리에는 표준 [SKILL.md 프런트매터](#skillmd-format)(`name`, `description`, 선택 사항인 `metadata.hermes.tags`, `version`, `author`, `platforms`, `metadata.hermes.config` 포함)가 있는 `SKILL.md`가 있어야 합니다.
- `references/`, `templates/`, `scripts/`, `assets/` 같은 하위 디렉터리는 설치 시 `SKILL.md`와 함께 다운로드됩니다.
- 디렉터리 이름이 `.` 또는 `_`로 시작하는 스킬은 무시됩니다.

Hermes는 tap 경로의 모든 하위 디렉터리를 나열하고 각각에 `SKILL.md`가 있는지 확인하여 스킬을 검색합니다.

#### 최소 tap 예시

```
my-org/hermes-skills
└── skills/
    └── deploy-runbook/
        └── SKILL.md
```

`skills/deploy-runbook/SKILL.md`:

```markdown
---
name: deploy-runbook
description: Our deployment runbook — services, rollback, Slack channels
version: 1.0.0
author: My Org Platform Team
metadata:
  hermes:
    tags: [deployment, runbook, internal]
---

# Deploy Runbook

Step 1: ...
```

이를 GitHub에 푸시하면 모든 Hermes 사용자가 구독하고 설치할 수 있습니다:

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### 기본 경로가 아닌 경로

스킬이 `skills/` 아래에 없을 때(기존 프로젝트에 `skills/` 하위 트리를 추가하는 경우 등)는 `~/.hermes/skills/.hub/taps.json`의 tap 항목을 편집합니다:

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

`hermes skills tap add` CLI는 새 tap의 기본 경로를 `path: "skills/"`로 설정합니다. 다른 경로가 필요하면 파일을 직접 편집하세요. `hermes skills tap list`는 각 tap의 유효 경로를 표시합니다.

#### tap을 추가하지 않고 개별 스킬 직접 설치

사용자는 전체 저장소를 tap으로 추가하지 않고 공개 GitHub 저장소에서 단일 스킬을 설치할 수도 있습니다:

```bash
hermes skills install owner/repo/skills/my-workflow
```

사용자에게 전체 레지스트리 구독을 요청하지 않고 하나의 스킬만 공유하고 싶을 때 유용합니다.

#### tap의 신뢰 수준

새 tap에는 기본적으로 `community` 신뢰가 할당됩니다. 여기에서 설치한 스킬은 표준 보안 스캔을 거치며 최초 설치 시 서드파티 경고 패널이 표시됩니다. 조직 또는 널리 신뢰되는 소스에 더 높은 신뢰를 부여해야 한다면 `tools/skills_hub.py`의 `TRUSTED_REPOS`에 저장소를 추가하세요(Hermes 핵심 PR이 필요합니다).

#### tap 관리

```bash
hermes skills tap list                                # show all configured taps
hermes skills tap add myorg/skills-repo               # add (default path: skills/)
hermes skills tap remove myorg/skills-repo            # remove
```

실행 중인 세션 안에서는:

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

tap은 `~/.hermes/skills/.hub/taps.json`에 저장됩니다(필요할 때 생성됨).

## 번들 스킬 업데이트 (`hermes skills reset`)

Hermes는 저장소 내부 `skills/`에 번들 스킬 세트를 제공합니다. 설치 시와 `hermes update`를 실행할 때마다 동기화 과정에서 이를 `~/.hermes/skills/`로 복사하고, 동기화 시점의 콘텐츠 해시( **원본 해시** )와 각 스킬 이름의 매핑을 `~/.hermes/skills/.bundled_manifest`에 기록합니다.

각 동기화에서 Hermes는 로컬 사본의 해시를 다시 계산하고 원본 해시와 비교합니다:

- **변경되지 않음** → 업스트림 변경을 안전하게 가져옵니다. 새 번들 버전을 복사하고 새 원본 해시를 기록합니다.
- **변경됨** → **사용자 수정**으로 처리하고 영원히 건너뛰므로 편집 내용이 덮어써지지 않습니다.

이 보호 기능은 훌륭하지만 한 가지 날카로운 부분이 있습니다. 번들 스킬을 편집한 후 `~/.hermes/hermes-agent/skills/`에서 내용을 복사하여 변경을 버리고 번들 버전으로 되돌리려 해도, 매니페스트에는 마지막으로 성공한 동기화 시점의 **이전** 원본 해시가 남아 있습니다. 새로 복사한 내용(현재 번들 해시)은 오래된 원본 해시와 일치하지 않으므로 동기화는 계속 이를 사용자 수정으로 표시합니다.

`hermes skills reset`은 이 탈출구입니다:

```bash
# Safe: clears the manifest entry for this skill. Your current copy is preserved,
# but the next sync re-baselines against it so future updates work normally.
hermes skills reset google-workspace

# Full restore: also deletes your local copy and re-copies the current bundled
# version. Use this when you want the pristine upstream skill back.
hermes skills reset google-workspace --restore

# Non-interactive (e.g. in scripts or TUI mode) — skip the --restore confirmation.
hermes skills reset google-workspace --restore --yes
```

같은 명령을 채팅에서 슬래시 명령으로도 사용할 수 있습니다:

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note 프로필
각 프로필에는 자체 `HERMES_HOME` 아래에 자체 `.bundled_manifest`가 있으므로 `hermes -p coder skills reset <name>`은 해당 프로필에만 영향을 줍니다.
:::

### 슬래시 명령 (채팅 내부)

동일한 모든 명령을 `/skills`와 함께 사용할 수 있습니다:

```text
/skills browse
/skills search react --source skills-sh
/skills search https://mintlify.com/docs --source well-known
/skills inspect skills-sh/vercel-labs/json-render/json-render-react
/skills install openai/skills/skill-creator --force
/skills check
/skills update
/skills reset google-workspace
/skills list
```

공식 선택적 스킬은 여전히 `official/security/1password`, `official/migration/openclaw-migration` 같은 식별자를 사용합니다.

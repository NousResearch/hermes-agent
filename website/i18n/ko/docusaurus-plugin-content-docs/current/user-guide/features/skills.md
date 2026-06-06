---
sidebar_position: 2
title: "스킬 시스템 (Skills System)"
description: "온디맨드 지식 문서 — 점진적 공개, 에이전트 관리 스킬 및 스킬 허브"
---

# 스킬 시스템 (Skills System)

스킬(Skills)은 에이전트가 필요할 때 로드할 수 있는 온디맨드 지식 문서입니다. 토큰 사용량을 최소화하기 위해 **점진적 공개(progressive disclosure)** 패턴을 따르며, [agentskills.io](https://agentskills.io/specification) 개방형 표준과 호환됩니다.

모든 스킬은 **`~/.hermes/skills/`**에 저장되며, 이는 기본 디렉토리이자 단일 진실 공급원(source of truth)입니다. 새로 설치 시 번들로 제공되는 스킬이 저장소에서 복사됩니다. 허브에서 설치하거나 에이전트가 생성한 스킬도 여기에 저장됩니다. 에이전트는 모든 스킬을 수정하거나 삭제할 수 있습니다.

Hermes가 **외부 스킬 디렉토리**를 가리키도록 설정할 수도 있습니다 — 이는 로컬 디렉토리와 함께 스캔되는 추가 폴더입니다. 아래의 [외부 스킬 디렉토리](#external-skill-directories)를 참조하세요.

참고:

- [번들 스킬 카탈로그 (Bundled Skills Catalog)](/reference/skills-catalog)
- [공식 선택 스킬 카탈로그 (Official Optional Skills Catalog)](/reference/optional-skills-catalog)

## 빈 상태로 시작하기

기본적으로 모든 프로필에는 번들 스킬 카탈로그가 시드(seed)되며, `hermes update`를 실행할 때마다 새로 번들된 스킬이 추가됩니다. **번들 스킬이 없는** 프로필을 원하고 업데이트 시에도 비워두려면 두 가지 방법이 있습니다.

**설치 시** (기본 `~/.hermes` 프로필에 적용됨):

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --no-skills
```

**프로필 생성 시** (명명된 프로필):

```bash
hermes profile create research --no-skills
```

**이미 설치된 프로필** (기본 또는 명명된 프로필)에서 런타임에 전환하려면:

```bash
hermes skills opt-out            # 향후 시딩 중지 — 디스크에 있는 어떤 항목도 건드리지 않음
hermes skills opt-out --remove   # 수정되지 않은 번들 스킬도 삭제함 (먼저 확인을 거침)
hermes skills opt-in --sync      # 실행 취소: 마커를 제거하고 지금 다시 시드함
```

세 가지 경로 모두 프로필 디렉토리에 `.no-bundled-skills` 마커를 작성합니다. 마커가 있는 동안 설치 프로그램, `hermes update` 및 모든 스킬 동기화는 해당 프로필에 대한 번들 스킬 시딩을 건너뜁니다. 마커를 삭제하거나 `hermes skills opt-in`을 실행하여 다시 활성화하세요.

:::note 기본적으로 안전함
`hermes skills opt-out`은 *향후* 시딩만 중지하며 이미 디스크에 있는 항목은 삭제하지 않습니다. 선택적인 `--remove` 플래그는 번들 스킬이 수정되지 않은 경우(Hermes가 설치한 버전과 바이트가 동일한 경우)에**만** 번들 스킬을 삭제합니다. 사용자가 편집한 스킬, 허브에서 설치한 스킬, 직접 작성한 스킬은 항상 보존됩니다.
:::

## 스킬 사용

설치된 모든 스킬은 슬래시 명령어로 자동으로 사용할 수 있습니다.

```bash
# CLI 또는 모든 메시징 플랫폼에서:
/gif-search 재미있는 고양이
/axolotl 내 데이터셋에서 Llama 3을 파인튜닝하도록 도와줘
/github-pr-workflow 인증 리팩토링에 대한 PR을 만들어 줘
/plan 인증 제공업체를 마이그레이션하기 위한 롤아웃을 설계해 줘

# 스킬 이름만 입력하면 스킬이 로드되고 에이전트가 필요한 것을 묻습니다.
/excalidraw
```

번들로 제공되는 `plan` 스킬이 좋은 예입니다. `/plan [요청]`을 실행하면 스킬의 명령이 로드되어 Hermes에게 필요한 경우 컨텍스트를 검사하고, 작업을 실행하는 대신 마크다운 구현 계획을 작성하며, 활성 작업 공간/백엔드 작업 디렉토리에 상대적인 `.hermes/plans/` 아래에 결과를 저장하도록 지시합니다.

자연스러운 대화를 통해 스킬과 상호 작용할 수도 있습니다.

```bash
hermes chat --toolsets skills -q "어떤 스킬을 가지고 있어?"
hermes chat --toolsets skills -q "axolotl 스킬을 보여줘"
```

## 점진적 공개 (Progressive Disclosure)

스킬은 토큰 효율적인 로딩 패턴을 사용합니다.

```
Level 0: skills_list()           → [{name, description, category}, ...]   (~3k tokens)
Level 1: skill_view(name)        → Full content + metadata       (varies)
Level 2: skill_view(name, path)  → Specific reference file       (varies)
```

에이전트는 실제로 필요할 때만 전체 스킬 콘텐츠를 로드합니다.

## SKILL.md 형식

```markdown
---
name: my-skill
description: 이 스킬이 수행하는 작업에 대한 간략한 설명
version: 1.0.0
platforms: [macos, linux]     # 선택 사항 — 특정 OS 플랫폼으로 제한
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    fallback_for_toolsets: [web]    # 선택 사항 — 조건부 활성화 (아래 참조)
    requires_toolsets: [terminal]   # 선택 사항 — 조건부 활성화 (아래 참조)
    config:                          # 선택 사항 — config.yaml 설정
      - key: my.setting
        description: "이 설정이 제어하는 것"
        default: "value"
        prompt: "설정을 위한 프롬프트"
---

# 스킬 제목

## 사용 시기 (When to Use)
이 스킬의 트리거 조건입니다.

## 절차 (Procedure)
1. 첫 번째 단계
2. 두 번째 단계

## 함정 (Pitfalls)
- 알려진 실패 모드 및 수정 사항

## 확인 (Verification)
작동을 확인하는 방법.
```

### 플랫폼별 스킬

스킬은 `platforms` 필드를 사용하여 특정 운영 체제로 자신을 제한할 수 있습니다.

| 값 | 일치 항목 |
|-------|---------|
| `macos` | macOS (Darwin) |
| `linux` | Linux |
| `windows` | Windows |

```yaml
platforms: [macos]            # macOS 전용 (예: iMessage, Apple 미리 알림, FindMy)
platforms: [macos, linux]     # macOS 및 Linux
```

설정된 경우 해당 스킬은 호환되지 않는 플랫폼의 시스템 프롬프트, `skills_list()` 및 슬래시 명령에서 자동으로 숨겨집니다. 생략하면 스킬이 모든 플랫폼에 로드됩니다.

## 스킬 출력 및 미디어 전달

스킬 응답(또는 모든 에이전트 응답)에 미디어 파일의 베어(bare) 절대 경로(예: `/home/user/screenshots/diagram.png`)가 포함된 경우 게이트웨이는 이를 자동 감지하고 표시되는 텍스트에서 경로를 제거한 다음, 원시 경로를 메시지에 남겨두는 대신 사용자 채팅(Telegram 사진, Discord 첨부 파일 등)으로 파일을 기본적으로 전달합니다.

특히 오디오의 경우 `[[audio_as_voice]]` 지시문은 이를 지원하는 플랫폼(Telegram, WhatsApp)에서 오디오 파일을 기본 음성 메시지 버블로 승격시킵니다.

### 문서 스타일 전달 강제: `[[as_document]]`

때로는 인라인 미리보기의 **반대**를 원할 때가 있습니다. 즉, 파일이 재압축된 이미지 버블이 아니라 다운로드 가능한 첨부 파일로 전달되기를 원합니다. 전형적인 예는 고해상도 스크린샷이나 차트입니다. Telegram의 `sendPhoto`는 1280px에서 약 200KB로 재압축하여 가독성을 파괴합니다. `sendDocument`를 통해 전송된 1-2MB PNG는 원본 바이트를 그대로 유지합니다.

응답(또는 응답 내의 텍스트 — 일반적으로 마지막 줄)에 리터럴 지시문 `[[as_document]]`가 포함되어 있으면 해당 응답에서 추출된 모든 이미지 경로가 이미지 버블이 아닌 문서/파일 첨부 파일로 전달됩니다.

```
여기에 렌더링된 차트가 있습니다:

/home/user/.hermes/cache/chart-q4-2025.png

[[as_document]]
```

지시문은 전달 전에 제거되므로 사용자에게는 보이지 않습니다. 세분성(Granularity)은 의도적으로 응답당 전부 아니면 전무(all-or-nothing)입니다. `[[as_document]]`를 한 번 내보내면 동일한 응답의 모든 이미지 경로가 문서로 전달됩니다. 이는 `[[audio_as_voice]]`의 범위와 동일합니다.

스킬에서 다음과 같은 경우에 사용하세요.

- 사용자가 파일로 필요로 하는 스크린샷이나 차트를 생성하는 경우(다른 도구에서 편집, 보관, 그대로 공유하기 위해).
- 기본 손실이 있는 미리보기가 세부 정보(작은 텍스트, 픽셀 단위로 정확한 다이어그램, 색상에 민감한 렌더링)를 가리는 경우.

별도의 문서 경로가 없는 플랫폼(예: SMS)은 보유한 첨부 메커니즘으로 폴백합니다.

### 조건부 활성화 (폴백 스킬)

스킬은 현재 세션에서 사용할 수 있는 도구에 따라 자동으로 자신을 표시하거나 숨길 수 있습니다. 이는 프리미엄 도구를 사용할 수 없을 때만 나타나야 하는 무료 또는 로컬 대안인 **폴백 스킬(fallback skills)**에 가장 유용합니다.

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # 이러한 툴셋을 사용할 수 없을 때만 표시
    requires_toolsets: [terminal]     # 이러한 툴셋을 사용할 수 있을 때만 표시
    fallback_for_tools: [web_search]  # 이러한 특정 도구를 사용할 수 없을 때만 표시
    requires_tools: [terminal]        # 이러한 특정 도구를 사용할 수 있을 때만 표시
```

| 필드 | 동작 |
|-------|----------|
| `fallback_for_toolsets` | 나열된 툴셋을 사용할 수 있을 때 스킬이 **숨겨집니다**. 툴셋이 없으면 표시됩니다. |
| `fallback_for_tools` | 동일하지만 툴셋 대신 개별 도구를 확인합니다. |
| `requires_toolsets` | 나열된 툴셋을 사용할 수 없을 때 스킬이 **숨겨집니다**. 툴셋이 있으면 표시됩니다. |
| `requires_tools` | 동일하지만 개별 도구를 확인합니다. |

**예시:** 기본 제공되는 `duckduckgo-search` 스킬은 `fallback_for_toolsets: [web]`을 사용합니다. `FIRECRAWL_API_KEY`가 설정되어 있으면 웹 툴셋을 사용할 수 있고 에이전트는 `web_search`를 사용합니다. 즉, DuckDuckGo 스킬은 숨겨진 상태로 유지됩니다. API 키가 없으면 웹 툴셋을 사용할 수 없으므로 DuckDuckGo 스킬이 폴백으로 자동 표시됩니다.

조건부 필드가 없는 스킬은 이전과 동일하게 동작합니다(항상 표시됨).

## 로드 시 보안 설정

스킬은 검색(discovery)에서 사라지지 않고 필요한 환경 변수를 선언할 수 있습니다.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: https://developers.google.com/tenor에서 키를 받으세요
    required_for: 전체 기능
```

누락된 값이 발견되면 Hermes는 스킬이 로컬 CLI에 실제로 로드될 때만 보안을 유지하며 요청합니다. 설정을 건너뛰고 스킬을 계속 사용할 수 있습니다. 메시징 표면(surfaces)은 채팅에서 비밀을 요구하지 않습니다. 대신 사용자에게 `hermes setup` 또는 `~/.hermes/.env`를 로컬로 사용하도록 알려줍니다.

설정된 후 선언된 환경 변수는 `execute_code` 및 `terminal` 샌드박스로 **자동 패스스루**됩니다. 즉, 스킬의 스크립트는 `$TENOR_API_KEY`를 직접 사용할 수 있습니다. 스킬과 관련 없는 환경 변수의 경우 `terminal.env_passthrough` 설정 옵션을 사용하세요. 자세한 내용은 [환경 변수 패스스루](/user-guide/security#environment-variable-passthrough)를 참조하세요.

### 스킬 설정

스킬은 `config.yaml`에 저장된 비밀이 아닌 구성 설정(경로, 환경설정)을 선언할 수도 있습니다.

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: 플러그인 데이터 디렉토리 경로
        default: "~/myplugin-data"
        prompt: 플러그인 데이터 디렉토리 경로
```

설정은 config.yaml의 `skills.config` 아래에 저장됩니다. `hermes config migrate`는 구성되지 않은 설정에 대한 프롬프트를 표시하고 `hermes config show`는 이를 표시합니다. 스킬이 로드되면 에이전트가 구성된 값을 자동으로 알 수 있도록 확인된 구성 값이 컨텍스트에 주입됩니다.

자세한 내용은 [스킬 설정](/user-guide/configuration#skill-settings) 및 [스킬 생성 — 구성 설정](/developer-guide/creating-skills#config-settings-configyaml)을 참조하세요.

## 스킬 디렉토리 구조

```text
~/.hermes/skills/                  # 단일 진실 공급원
├── mlops/                         # 카테고리 디렉토리
│   ├── axolotl/
│   │   ├── SKILL.md               # 주요 지침 (필수)
│   │   ├── references/            # 추가 문서
│   │   ├── templates/             # 출력 형식
│   │   ├── scripts/               # 스킬에서 호출할 수 있는 헬퍼 스크립트
│   │   └── assets/                # 보충 파일
│   └── vllm/
│       └── SKILL.md
├── devops/
│   └── deploy-k8s/                # 에이전트가 생성한 스킬
│       ├── SKILL.md
│       └── references/
├── .hub/                          # 스킬 허브 상태
│   ├── lock.json
│   ├── quarantine/
│   └── audit.log
└── .bundled_manifest              # 시드된 번들 스킬 추적
```

## 외부 스킬 디렉토리

예를 들어 여러 AI 도구에서 사용하는 공유 `~/.agents/skills/` 디렉토리와 같이 Hermes 외부에서 스킬을 유지 관리하는 경우, Hermes에 해당 디렉토리도 스캔하도록 지시할 수 있습니다.

`~/.hermes/config.yaml`의 `skills` 섹션 아래에 `external_dirs`를 추가하세요.

```yaml
skills:
  external_dirs:
    - ~/.agents/skills
    - /home/shared/team-skills
    - ${SKILLS_REPO}/skills
```

경로는 `~` 확장 및 `${VAR}` 환경 변수 대체를 지원합니다.

### 작동 방식

- **로컬에서 생성, 제자리 업데이트**: 에이전트가 생성한 새 스킬은 `~/.hermes/skills/`에 기록됩니다. 기존 스킬은 에이전트가 `patch`, `edit`, `write_file`, `remove_file` 또는 `delete`와 같은 `skill_manage` 작업을 사용할 때 `external_dirs` 아래의 스킬을 포함하여 발견된 곳에서 수정됩니다.
- **외부 디렉토리는 쓰기 보호 경계가 아닙니다**: Hermes 프로세스가 외부 스킬 디렉토리에 쓸 수 있는 경우, 에이전트 관리 스킬 업데이트는 해당 디렉토리의 파일을 변경할 수 있습니다. 공유 외부 스킬을 읽기 전용으로 유지해야 하는 경우 파일 시스템 권한 또는 별도의 프로필/툴셋 설정을 사용하세요.
- **로컬 우선**: 로컬 디렉토리와 외부 디렉토리 모두에 동일한 스킬 이름이 존재하는 경우 로컬 버전이 우선합니다.
- **전체 통합**: 외부 스킬은 시스템 프롬프트 인덱스, `skills_list`, `skill_view` 및 `/skill-name` 슬래시 명령어에 표시되며 로컬 스킬과 다르지 않습니다.
- **존재하지 않는 경로는 조용히 건너뜁니다**: 구성된 디렉토리가 없으면 Hermes는 오류 없이 이를 무시합니다. 모든 시스템에 존재하지 않을 수 있는 선택적 공유 디렉토리에 유용합니다.

### 예시

```text
~/.hermes/skills/               # 로컬 (기본, 읽기-쓰기)
├── devops/deploy-k8s/
│   └── SKILL.md
└── mlops/axolotl/
    └── SKILL.md

~/.agents/skills/               # 외부 (공유됨, 쓰기 가능한 경우 변경 가능)
├── my-custom-workflow/
│   └── SKILL.md
└── team-conventions/
    └── SKILL.md
```

네 가지 스킬이 모두 스킬 인덱스에 나타납니다. 로컬에 `my-custom-workflow`라는 새 스킬을 만들면 외부 버전을 섀도잉(shadow)합니다.

## 스킬 번들

스킬 번들은 여러 스킬을 단일 슬래시 명령어로 그룹화하는 작은 YAML 파일입니다. `/<bundle-name>`을 실행하면 번들에 나열된 모든 스킬이 한 번에 로드됩니다. 특정 작업이 항상 동일한 스킬 세트와 함께 유용할 때 사용합니다.

### 빠른 예시

```bash
# 백엔드 기능 작업을 위한 번들 생성
hermes bundles create backend-dev \
  --skill github-code-review \
  --skill test-driven-development \
  --skill github-pr-workflow \
  -d "백엔드 기능 작업 — 리뷰, 테스트, PR 워크플로"
```

그런 다음 CLI 또는 게이트웨이 플랫폼에서:

```
/backend-dev 인증 미들웨어 리팩토링해 줘
```

에이전트는 세 가지 스킬이 모두 하나의 사용자 메시지로 로드된 상태로 수신하며, 슬래시 명령 뒤의 모든 텍스트는 사용자 지침으로 첨부됩니다.

### YAML 스키마

번들은 **`~/.hermes/skill-bundles/<slug>.yaml`**에 저장되며 다음과 같은 형태를 가집니다:

```yaml
name: backend-dev
description: 백엔드 기능 작업 — 리뷰, 테스트, PR 워크플로.
skills:
  - github-code-review
  - test-driven-development
  - github-pr-workflow
instruction: |
  항상 실패하는 테스트를 먼저 작성한 다음 구현을 시작하세요.
  co-author 태그가 포함된 표준 워크플로를 통해 PR을 엽니다.
```

필드:
- `name` (선택 사항 — 기본값은 파일 이름 어간) — 번들의 표시 이름입니다. 슬래시 명령에 맞게 하이픈 슬러그로 정규화됩니다 (`Backend Dev` → `/backend-dev`).
- `description` (선택 사항) — `/bundles` 및 `hermes bundles list`에 표시되는 짧은 텍스트입니다.
- `skills` (필수, 비어 있지 않은 목록) — 스킬 디렉토리에 상대적인 스킬 이름 또는 경로입니다. `/<skill-name>`에 전달하는 것과 동일한 식별자를 사용합니다.
- `instruction` (선택 사항) — 로드된 스킬 콘텐츠 앞에 추가되는 추가 지침입니다. "우리가 항상 함께 사용하는 방법"을 성문화하는 데 유용합니다.

### 번들 관리

```bash
# 설치된 모든 번들 목록 보기
hermes bundles list

# 특정 번들 검사
hermes bundles show backend-dev

# 대화형으로 번들 생성 (--skill 플래그를 생략하여 한 줄에 하나씩 입력)
hermes bundles create research

# 기존 번들 덮어쓰기
hermes bundles create backend-dev --skill ... --force

# 번들 삭제
hermes bundles delete backend-dev

# ~/.hermes/skill-bundles/ 다시 스캔 및 변경 사항 보고
hermes bundles reload
```

채팅 세션 내에서 `/bundles`를 실행하면 설치된 모든 번들과 해당 스킬이 나열됩니다.

### 동작 (Behavior)

- **슬러그가 충돌할 때 번들이 개별 스킬보다 우선합니다.** 번들 이름을 `research`로 지정하고 `research`라는 스킬도 있는 경우 `/research`는 번들을 호출합니다. 사용자가 번들 이름을 지정하여 번들을 사용하도록 선택했으므로 이는 의도된 것입니다.
- **누락된 스킬은 치명적이지 않으며 건너뜁니다.** 번들에 `skill-foo`가 나열되어 있고 아직 설치하지 않은 경우에도 번들은 확인 가능한 스킬을 로드하고 에이전트는 건너뛴 항목이 나열된 노트를 받습니다.
- **번들은 모든 표면에서 작동합니다** — 대화형 CLI, TUI, 대시보드 채팅 및 모든 게이트웨이 플랫폼(Telegram, Discord, Slack 등) — 디스패치가 개별 스킬 명령과 동일한 곳에서 중앙 집중화되기 때문입니다.
- **번들은 프롬프트 캐시를 무효화하지 않습니다.** `/<skill-name>`이 수행하는 것과 동일한 방식으로 호출 시 새로운 사용자 메시지를 생성하며 시스템 프롬프트를 변경하지 않습니다.

### 각 스킬을 수동으로 설치하는 것보다 번들을 사용하는 것이 나은 경우

다음과 같은 경우에 번들을 사용하세요.
- 반복적인 작업에 항상 동일한 스킬을 짝지어 사용하는 경우 (`/backend-dev`, `/release-prep`, `/incident-response`).
- 여러 `/skill` 호출을 연속으로 입력하는 것보다 한 글자 짧은 멘탈 모델을 원하는 경우.
- 번들 YAML을 공유 dotfiles 저장소에 체크인하고 `~/.hermes/skill-bundles/`에 심볼릭 링크를 연결하여 팀 전체의 "작업 프로필"을 배포하려는 경우.

번들은 단지 YAML 별칭일 뿐이며 사용자를 위해 스킬을 설치하지 않습니다. 스킬 자체는 이미 (`~/.hermes/skills/` 또는 외부 스킬 디렉토리에) 존재해야 합니다. 그렇지 않으면 번들 호출은 누락된 스킬을 건너뛰기만 합니다.

## 에이전트 관리 스킬 (skill_manage 도구)

에이전트는 `skill_manage` 도구를 통해 자체 스킬을 생성, 업데이트 및 삭제할 수 있습니다. 이것은 에이전트의 **절차적 기억(procedural memory)**입니다. 간단하지 않은 워크플로를 파악하면 향후 재사용을 위해 해당 접근 방식을 스킬로 저장합니다.

### 에이전트가 스킬을 생성하는 경우

- 복잡한 작업(5개 이상의 도구 호출)을 성공적으로 완료한 후
- 오류나 막다른 골목에 부딪혀 작동하는 경로를 찾았을 때
- 사용자가 접근 방식을 수정했을 때
- 간단하지 않은 워크플로를 발견했을 때

### 작업 (Actions)

| 작업 | 용도 | 주요 파라미터 |
|--------|---------|------------|
| `create` | 처음부터 새 스킬 생성 | `name`, `content` (전체 SKILL.md), 선택적 `category` |
| `patch` | 대상 수정 (권장됨) | `name`, `old_string`, `new_string` |
| `edit` | 주요 구조 재작성 | `name`, `content` (전체 SKILL.md 교체) |
| `delete` | 스킬 완전히 제거 | `name` |
| `write_file` | 지원 파일 추가/업데이트 | `name`, `file_path`, `file_content` |
| `remove_file` | 지원 파일 제거 | `name`, `file_path` |

:::tip
업데이트에는 `patch` 작업이 권장됩니다. 변경된 텍스트만 도구 호출에 나타나기 때문에 `edit`보다 토큰 효율적입니다.
:::

## 스킬 허브 (Skills Hub)

온라인 레지스트리, `skills.sh`, 직접적으로 잘 알려진 스킬 엔드포인트 및 공식 선택 스킬에서 스킬을 탐색, 검색, 설치 및 관리합니다.

### 일반 명령어

```bash
hermes skills browse                              # 모든 허브 스킬 탐색 (공식 항목 먼저)
hermes skills browse --source official            # 공식 선택 스킬만 탐색
hermes skills search kubernetes                   # 모든 소스 검색
hermes skills search react --source skills-sh     # skills.sh 디렉토리 검색
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect openai/skills/k8s           # 설치 전 미리보기
hermes skills install openai/skills/k8s           # 보안 스캔과 함께 설치
hermes skills install official/security/1password
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install https://sharethis.chat/SKILL.md              # 직접 URL (단일 파일 SKILL.md)
hermes skills install https://example.com/SKILL.md --name my-skill # 프론트매터에 이름이 없는 경우 무시하고 지정
hermes skills list --source hub                   # 허브에서 설치된 스킬 목록
hermes skills check                               # 업스트림 업데이트에 대해 설치된 허브 스킬 확인
hermes skills update                              # 필요한 경우 업스트림 변경 사항과 함께 허브 스킬 재설치
hermes skills audit                               # 모든 허브 스킬의 보안 재스캔
hermes skills uninstall k8s                       # 허브 스킬 제거
hermes skills reset google-workspace              # 번들 스킬의 "사용자 수정" 상태 해제 (아래 참조)
hermes skills reset google-workspace --restore    # 로컬 편집 내용을 삭제하여 번들 버전도 복원
hermes skills publish skills/my-skill --to github --repo owner/repo
hermes skills snapshot export setup.json          # 스킬 설정 내보내기
hermes skills tap add myorg/skills-repo           # 사용자 지정 GitHub 소스 추가
```

### 지원되는 허브 소스

| 소스 | 예시 | 참고 |
|--------|---------|-------|
| `official` | `official/security/1password` | Hermes와 함께 제공되는 선택 스킬입니다. |
| `skills-sh` | `skills-sh/vercel-labs/agent-skills/vercel-react-best-practices` | `hermes skills search <query> --source skills-sh`를 통해 검색할 수 있습니다. skills.sh 슬러그가 리포지토리 폴더와 다를 때 Hermes는 별칭 스타일 스킬을 확인합니다. |
| `well-known` | `well-known:https://mintlify.com/docs/.well-known/skills/mintlify` | 웹사이트의 `/.well-known/skills/index.json`에서 직접 제공되는 스킬입니다. 사이트 또는 문서 URL을 사용하여 검색합니다. |
| `url` | `https://sharethis.chat/SKILL.md` | 단일 파일 `SKILL.md`에 대한 직접 HTTP(S) URL입니다. 이름 확인 순서: 프론트매터 → URL 슬러그 → 대화형 프롬프트 → `--name` 플래그. |
| `github` | `openai/skills/k8s` | 직접 GitHub 리포지토리/경로 설치 및 사용자 지정 탭. |
| `clawhub`, `lobehub`, `browse-sh` | 소스 특정 식별자 | 커뮤니티 또는 마켓플레이스 통합. |

### 통합 허브 및 레지스트리

Hermes는 현재 다음과 같은 스킬 생태계 및 검색 소스와 통합되어 있습니다.

#### 1. 공식 선택 스킬 (`official`)

이들은 Hermes 저장소 자체에서 유지 관리되며 기본 제공 신뢰(built-in trust)로 설치됩니다.

- 카탈로그: [공식 선택 스킬 카탈로그](../../reference/optional-skills-catalog)
- 리포지토리의 소스: `optional-skills/`
- 예시:

```bash
hermes skills browse --source official
hermes skills install official/security/1password
```

#### 2. skills.sh (`skills-sh`)

Vercel의 공개 스킬 디렉토리입니다. Hermes는 이를 직접 검색하고, 스킬 세부 정보 페이지를 검사하고, 별칭 스타일의 슬러그를 확인하고, 기본 소스 저장소에서 설치할 수 있습니다.

- 디렉토리: [skills.sh](https://skills.sh/)
- CLI/도구 저장소: [vercel-labs/skills](https://github.com/vercel-labs/skills)
- 공식 Vercel 스킬 저장소: [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- 예시:

```bash
hermes skills search react --source skills-sh
hermes skills inspect skills-sh/vercel-labs/json-render/json-render-react
hermes skills install skills-sh/vercel-labs/json-render/json-render-react --force
```

#### 3. 잘 알려진 스킬 엔드포인트 (`well-known`)

`/.well-known/skills/index.json`을 게시하는 사이트의 URL 기반 검색입니다. 단일 중앙 집중식 허브가 아니며 웹 검색 규칙입니다.

- 라이브 엔드포인트 예: [Mintlify 문서 스킬 인덱스](https://mintlify.com/docs/.well-known/skills/index.json)
- 참조 서버 구현: [vercel-labs/skills-handler](https://github.com/vercel-labs/skills-handler)
- 예시:

```bash
hermes skills search https://mintlify.com/docs --source well-known
hermes skills inspect well-known:https://mintlify.com/docs/.well-known/skills/mintlify
hermes skills install well-known:https://mintlify.com/docs/.well-known/skills/mintlify
```

#### 4. 직접 GitHub 스킬 (`github`)

Hermes는 GitHub 저장소 및 GitHub 기반 탭에서 직접 설치할 수 있습니다. 리포지토리/경로를 이미 알고 있거나 고유한 사용자 지정 소스 리포지토리를 추가하려는 경우에 유용합니다.

기본 탭 (설정 없이 탐색 가능):
- [openai/skills](https://github.com/openai/skills)
- [anthropics/skills](https://github.com/anthropics/skills)
- [huggingface/skills](https://github.com/huggingface/skills)
- [NVIDIA/skills](https://github.com/NVIDIA/skills) — NVIDIA 검증 스킬 (서명 `skill.oms.sig` + 거버넌스 `skill-card.md`)
- [garrytan/gstack](https://github.com/garrytan/gstack)

- 예시:

```bash
hermes skills install openai/skills/k8s
hermes skills tap add myorg/skills-repo
```

**카테고리 그룹화 (`skills.sh.json`).** GitHub 탭은 리포지토리 루트에 [skills.sh 스키마](https://skills.sh/schemas/skills.sh.schema.json)를 따르는 `skills.sh.json` 파일을 포함할 수 있습니다. (각각 `title`과 스킬 이름 목록을 가진) `groupings`는 인덱싱 시점에 읽히며 태그 파생 추측 대신 [스킬 허브(Skills Hub)](https://hermes-agent.nousresearch.com/docs) 페이지에 표시되는 카테고리 레이블이 됩니다. 이는 일반적입니다. 파일을 제공하는 모든 탭은 Hermes 측 변경 없이 실제 분류를 얻습니다.

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

커뮤니티 소스로 통합된 타사 스킬 마켓플레이스입니다.

- 사이트: [clawhub.ai](https://clawhub.ai/)
- Hermes 소스 ID: `clawhub`

#### 6. Claude 마켓플레이스 스타일 리포지토리 (`claude-marketplace`)

Hermes는 Claude 호환 플러그인/마켓플레이스 매니페스트를 게시하는 마켓플레이스 리포지토리를 지원합니다.

알려진 통합 소스는 다음과 같습니다.
- [anthropics/skills](https://github.com/anthropics/skills)
- [aiskillstore/marketplace](https://github.com/aiskillstore/marketplace)

Hermes 소스 ID: `claude-marketplace`

#### 7. LobeHub (`lobehub`)

Hermes는 LobeHub의 공용 카탈로그에서 에이전트 항목을 검색하고 설치 가능한 Hermes 스킬로 변환할 수 있습니다.

- 사이트: [LobeHub](https://lobehub.com/)
- 공개 에이전트 인덱스: [chat-agents.lobehub.com](https://chat-agents.lobehub.com/)
- 백업 리포지토리: [lobehub/lobe-chat-agents](https://github.com/lobehub/lobe-chat-agents)
- Hermes 소스 ID: `lobehub`

#### 8. browse.sh (`browse-sh`)

Hermes는 200개 이상의 사이트별 브라우저 자동화 SKILL.md 파일(Airbnb, Amazon, arXiv, 12306.cn, Etsy, Xero 등)로 구성된 Browserbase의 카탈로그인 [browse.sh](https://browse.sh)와 통합됩니다. 각 스킬은 웹사이트를 처음부터 끝까지 구동하는 방법을 설명하며 Hermes의 브라우저 도구 및 이미 설치된 모든 브라우저 자동화 스킬과 함께 사용하기에 적합합니다.

- 사이트: [browse.sh](https://browse.sh/)
- 카탈로그 API: `https://browse.sh/api/skills`
- Hermes 소스 ID: `browse-sh`
- 신뢰 수준: `community`

```bash
hermes skills search airbnb --source browse-sh
hermes skills inspect browse-sh/airbnb.com/search-listings-ddgioa
hermes skills install browse-sh/airbnb.com/search-listings-ddgioa
```

식별자는 `browse-sh/<hostname>/<task-id>` 형식을 사용하고 browse.sh 카탈로그에 노출된 슬러그와 일치합니다. 콘텐츠는 카탈로그의 GitHub `sourceUrl`을 통해서가 아니라 스킬별 세부 정보 엔드포인트(`/api/skills/<slug>` → `skillMdUrl`)를 통해 확인됩니다.

#### 9. 직접 URL (`url`)

작성자가 자신의 사이트에서 스킬을 호스팅할 때 (허브 목록이나 입력할 GitHub 경로가 없음) 모든 HTTP(S) URL에서 단일 파일 `SKILL.md`를 직접 설치합니다. Hermes는 URL을 가져오고 YAML 프론트매터를 구문 분석하며 보안 검사를 수행하고 설치합니다.

- Hermes 소스 ID: `url`
- 식별자: URL 자체 (접두사 필요 없음)
- 범위: **단일 파일 `SKILL.md`**만. `references/` 또는 `scripts/`가 있는 다중 파일 스킬은 매니페스트가 필요하며 위의 다른 소스 중 하나를 통해 게시해야 합니다.

```bash
hermes skills install https://sharethis.chat/SKILL.md
hermes skills install https://example.com/my-skill/SKILL.md --category productivity
```

이름 확인 순서:
1. SKILL.md YAML 프론트매터의 `name:` 필드 (권장 — 잘 구성된 모든 스킬에는 이 필드가 있습니다).
2. URL 경로의 상위 디렉토리 이름 (예: `.../my-skill/SKILL.md` → `my-skill`, 또는 `.../my-skill.md` → `my-skill`)이 유효한 식별자(`^[a-z][a-z0-9_-]*$`)인 경우.
3. TTY가 있는 터미널에서의 대화형 프롬프트.
4. 대화형이 아닌 표면(TUI 내부의 `/skills install` 슬래시 명령, 게이트웨이 플랫폼, 스크립트)에서 `--name` 오버라이드를 가리키는 깨끗한 오류 발생.

```bash
# 프론트매터에 이름이 없고 URL 슬러그가 도움이 되지 않는 경우, 제공하세요.
hermes skills install https://example.com/SKILL.md --name sharethis-chat

# 또는 채팅 세션 내에서:
/skills install https://example.com/SKILL.md --name sharethis-chat
```

신뢰 수준은 항상 `community`입니다. 다른 모든 소스에 대해 동일한 보안 스캔이 실행됩니다. URL은 설치 식별자로 저장되므로 새로 고침하려는 경우 `hermes skills update`가 동일한 URL에서 자동으로 다시 가져옵니다.

### 보안 스캔 및 `--force`

허브에 설치된 모든 스킬은 데이터 유출, 프롬프트 인젝션, 파괴적인 명령어, 공급망 신호 및 기타 위협을 검사하는 **보안 스캐너**를 거칩니다.

`hermes skills inspect ...`는 이제 가능한 경우 업스트림 메타데이터도 표면화합니다.
- 저장소 URL
- skills.sh 세부 정보 페이지 URL
- 설치 명령
- 주간 설치 횟수
- 업스트림 보안 감사 상태
- 잘 알려진 인덱스/엔드포인트 URL

타사 스킬을 검토했고 위험하지 않은 정책 차단을 무시하려는 경우 `--force`를 사용하세요.

```bash
hermes skills install skills-sh/anthropics/skills/pdf --force
```

중요한 동작:
- `--force`는 주의/경고 스타일 결과에 대한 정책 차단을 재정의할 수 있습니다.
- `--force`는 `dangerous`(위험) 스캔 판정을 재정의하지 **않습니다**.
- 공식 선택 스킬 (`official/...`)은 기본 제공 신뢰로 취급되며 타사 경고 패널을 표시하지 않습니다.

### 신뢰 수준

| 수준 | 소스 | 정책 |
|-------|--------|--------|
| `builtin` | Hermes와 함께 제공됨 | 항상 신뢰됨 |
| `official` | 리포지토리의 `optional-skills/` | 기본 제공 신뢰, 타사 경고 없음 |
| `trusted` | `openai/skills`, `anthropics/skills`, `huggingface/skills`, `NVIDIA/skills`와 같은 신뢰할 수 있는 레지스트리/저장소 | 커뮤니티 소스보다 허용적인 정책 |
| `community` | 기타 모든 소스 (`skills.sh`, 잘 알려진 엔드포인트, 사용자 지정 GitHub 저장소, 대부분의 마켓플레이스) | 위험하지 않은 결과는 `--force`로 재정의 가능; `dangerous` 판정은 계속 차단됨 |

### 업데이트 수명 주기

허브는 이제 설치된 스킬의 업스트림 복사본을 다시 확인할 수 있을 만큼 충분한 출처를 추적합니다.

```bash
hermes skills check          # 설치된 허브 스킬 중 업스트림이 변경된 스킬 보고
hermes skills update         # 업데이트가 있는 스킬만 재설치
hermes skills update react   # 특정 설치된 허브 스킬 하나 업데이트
```

이것은 저장된 소스 식별자와 현재 업스트림 번들 콘텐츠 해시를 사용하여 변경(drift)을 감지합니다.

:::tip GitHub API 속도 제한
스킬 허브 작업은 인증되지 않은 사용자에 대해 시간당 60회 요청의 속도 제한이 있는 GitHub API를 사용합니다. 설치 또는 검색 중에 속도 제한 오류가 표시되면 `.env` 파일에 `GITHUB_TOKEN`을 설정하여 제한을 시간당 5,000회로 늘리세요. 이 문제가 발생할 때 오류 메시지에는 실행 가능한 힌트가 포함되어 있습니다.
:::

### 사용자 지정 스킬 탭 게시하기

팀, 조직 또는 공개적으로 선별된 스킬 세트를 공유하려는 경우, 이를 **탭(tap)**으로 게시할 수 있습니다. 탭은 다른 Hermes 사용자가 `hermes skills tap add <owner/repo>`를 사용하여 추가하는 GitHub 저장소입니다. 서버, 레지스트리 가입, 릴리스 파이프라인이 필요 없습니다. `SKILL.md` 파일들의 디렉토리일 뿐입니다.

#### 저장소 레이아웃

탭은 다음과 같이 구성된 모든 GitHub 저장소입니다 (공개 또는 비공개 — 비공개는 `GITHUB_TOKEN` 필요):

```
owner/repo
├── skills/                       # 기본 경로; 탭마다 구성 가능
│   ├── my-workflow/
│   │   ├── SKILL.md              # 필수
│   │   ├── references/           # 선택적 지원 파일
│   │   ├── templates/
│   │   └── scripts/
│   ├── another-skill/
│   │   └── SKILL.md
│   └── third-skill/
│       └── SKILL.md
└── README.md                     # 선택 사항이지만 유용함
```

규칙:
- 각 스킬은 탭의 루트 경로(기본값 `skills/`) 아래의 고유한 디렉토리에 있습니다.
- 디렉토리 이름이 스킬의 설치 슬러그가 됩니다.
- 각 스킬 디렉토리에는 표준 [SKILL.md 프론트매터](#skillmd-format) (`name`, `description`, 추가적으로 선택 사항인 `metadata.hermes.tags`, `version`, `author`, `platforms`, `metadata.hermes.config`)가 포함된 `SKILL.md`가 있어야 합니다.
- `references/`, `templates/`, `scripts/`, `assets/`와 같은 하위 디렉토리는 설치 시 `SKILL.md`와 함께 다운로드됩니다.
- 디렉토리 이름이 `.` 또는 `_`로 시작하는 스킬은 무시됩니다.

Hermes는 탭 경로의 모든 하위 디렉토리를 나열하고 각각에서 `SKILL.md`를 검색하여 스킬을 발견합니다.

#### 최소 탭 예시

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
description: 우리의 배포 런북 — 서비스, 롤백, Slack 채널
version: 1.0.0
author: My Org Platform Team
metadata:
  hermes:
    tags: [deployment, runbook, internal]
---

# 배포 런북 (Deploy Runbook)

1단계: ...
```

GitHub에 푸시한 후 모든 Hermes 사용자는 구독하고 설치할 수 있습니다.

```bash
hermes skills tap add my-org/hermes-skills
hermes skills search deploy
hermes skills install my-org/hermes-skills/deploy-runbook
```

#### 기본값이 아닌 경로

스킬이 `skills/` 아래에 있지 않은 경우 (기존 프로젝트에 `skills/` 하위 트리를 추가할 때 일반적임) `~/.hermes/.hub/taps.json`의 탭 항목을 편집합니다.

```json
{
  "taps": [
    {"repo": "my-org/platform-docs", "path": "internal/skills/"}
  ]
}
```

`hermes skills tap add` CLI는 기본적으로 새 탭을 `path: "skills/"`로 지정합니다. 다른 경로가 필요한 경우 파일을 직접 편집하세요. `hermes skills tap list`는 탭별 유효 경로를 표시합니다.

#### 개별 스킬 직접 설치 (탭 추가 없이)

사용자는 전체 저장소를 탭으로 추가하지 않고 공개 GitHub 저장소에서 단일 스킬을 설치할 수도 있습니다.

```bash
hermes skills install owner/repo/skills/my-workflow
```

사용자에게 전체 레지스트리를 구독하도록 요구하지 않고 하나의 스킬을 공유하려는 경우에 유용합니다.

#### 탭 신뢰 수준

새 탭에는 기본적으로 `community` 신뢰가 할당됩니다. 해당 탭에서 설치된 스킬은 표준 보안 검사를 실행하고 첫 번째 설치 시 타사 경고 패널을 표시합니다. 조직이나 널리 신뢰받는 소스에 더 높은 신뢰를 부여해야 하는 경우, `tools/skills_hub.py`의 `TRUSTED_REPOS`에 해당 리포지토리를 추가하세요 (Hermes 코어 PR 필요).

#### 탭 관리

```bash
hermes skills tap list                                # 구성된 모든 탭 표시
hermes skills tap add myorg/skills-repo               # 추가 (기본 경로: skills/)
hermes skills tap remove myorg/skills-repo            # 제거
```

실행 중인 세션 내에서:

```
/skills tap list
/skills tap add myorg/skills-repo
/skills tap remove myorg/skills-repo
```

탭은 `~/.hermes/.hub/taps.json`에 저장됩니다(요청 시 생성됨).

## 번들 스킬 업데이트 (`hermes skills reset`)

Hermes는 저장소 내부의 `skills/`에 번들로 제공되는 스킬 세트와 함께 제공됩니다. 설치 시 그리고 모든 `hermes update` 시, 동기화(sync) 패스가 이들을 `~/.hermes/skills/`로 복사하고, 각 스킬 이름을 동기화 당시의 콘텐츠 해시(**origin hash**, 원본 해시)에 매핑하는 매니페스트를 `~/.hermes/skills/.bundled_manifest`에 기록합니다.

각 동기화에서 Hermes는 로컬 사본의 해시를 다시 계산하고 원본 해시와 비교합니다.

- **변경되지 않음** → 업스트림 변경 사항을 안전하게 가져오고, 새 번들 버전을 복사하고, 새 원본 해시를 기록합니다.
- **변경됨** → **사용자가 수정한(user-modified)** 것으로 취급되어 영원히 건너뛰므로 편집 내용이 덮어써지지 않습니다.

이 보호 기능은 훌륭하지만 한 가지 날카로운 모서리가 있습니다. 번들 스킬을 편집한 후 나중에 변경 사항을 포기하고 `~/.hermes/hermes-agent/skills/`에서 복사하여 붙여넣는 방식으로 번들 버전으로 돌아가려는 경우, 매니페스트에는 여전히 마지막으로 성공한 동기화가 실행되었을 때의 *이전* 원본 해시가 유지됩니다. 새로 복사-붙여넣기 한 내용(현재 번들 해시)이 오래된 원본 해시와 일치하지 않으므로 동기화는 계속 이를 사용자가 수정한 것으로 표시합니다.

`hermes skills reset`은 이를 위한 탈출구(escape hatch)입니다.

```bash
# 안전: 이 스킬에 대한 매니페스트 항목을 지웁니다. 현재 사본은 보존되지만,
# 다음 동기화 시 이를 기준으로 다시 기준선을 설정하므로 향후 업데이트는 정상적으로 작동합니다.
hermes skills reset google-workspace

# 전체 복원: 로컬 사본도 삭제하고 현재 번들 버전을 다시 복사합니다.
# 원래의 깨끗한 업스트림 스킬을 되돌리려는 경우에 사용하세요.
hermes skills reset google-workspace --restore

# 비대화형 (예: 스크립트 또는 TUI 모드에서) — --restore 확인을 건너뜁니다.
hermes skills reset google-workspace --restore --yes
```

동일한 명령이 채팅에서 슬래시 명령으로 작동합니다.

```text
/skills reset google-workspace
/skills reset google-workspace --restore
```

:::note 프로필
각 프로필에는 자체 `HERMES_HOME` 아래에 자체 `.bundled_manifest`가 있으므로, `hermes -p coder skills reset <name>`은 해당 프로필에만 영향을 미칩니다.
:::

### 슬래시 명령어 (채팅 내)

모든 동일한 명령이 `/skills`와 함께 작동합니다.

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

공식 선택 스킬은 계속해서 `official/security/1password` 및 `official/migration/openclaw-migration`과 같은 식별자를 사용합니다.

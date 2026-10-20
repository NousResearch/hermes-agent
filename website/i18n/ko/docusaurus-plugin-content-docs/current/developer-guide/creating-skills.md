---
sidebar_position: 3
title: "스킬 만들기"
description: "Hermes Agent용 스킬을 만드는 방법 — SKILL.md 형식, 지침, 게시"
---

# 스킬 만들기

스킬은 Hermes Agent에 새로운 기능을 추가하는 가장 권장되는 방법입니다. 도구보다 만들기 쉽고, 에이전트 코드를 변경할 필요가 없으며, 커뮤니티와 공유할 수 있습니다.

## 스킬인가, 도구인가?

다음과 같은 경우 **스킬**로 만드세요:
- 기능을 지침 + 셸 명령 + 기존 도구로 표현할 수 있음
- 에이전트가 `terminal` 또는 `web_extract`를 통해 호출할 수 있는 외부 CLI나 API를 감쌈
- 에이전트에 사용자 지정 Python 통합이나 API 키 관리 기능을 내장할 필요가 없음
- 예: arXiv 검색, git 워크플로, Docker 관리, PDF 처리, CLI 도구를 통한 이메일

다음과 같은 경우 **도구**로 만드세요:
- API 키, 인증 흐름 또는 여러 구성 요소의 구성을 종단 간 통합해야 함
- 매번 정확하게 실행되어야 하는 사용자 지정 처리 로직이 필요함
- 바이너리 데이터, 스트리밍 또는 실시간 이벤트를 처리함
- 예: 브라우저 자동화, TTS, 비전 분석

## 스킬 디렉터리 구조

번들로 제공되는 스킬은 카테고리별로 `skills/`에 구성됩니다. 공식 선택적 스킬은 `optional-skills/`에서 동일한 구조를 사용합니다:

```text
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Required: main instructions
│       └── scripts/              # Optional: helper scripts
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

## SKILL.md 형식

```markdown
---
name: my-skill
description: Brief description (shown in skill search results)
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # Optional — restrict to specific OS platforms
                                   #   Valid: macos, linux, windows
                                   #   Omit to load on all platforms (default)
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    requires_toolsets: [web]            # Optional — only show when these toolsets are active
    requires_tools: [web_search]        # Optional — only show when these tools are available
    fallback_for_toolsets: [browser]    # Optional — hide when these toolsets are active
    fallback_for_tools: [browser_navigate]  # Optional — hide when these tools exist
    config:                              # Optional — config.yaml settings the skill needs
      - key: my.setting
        description: "What this setting controls"
        default: "sensible-default"
        prompt: "Display prompt for setup"
    blueprint:                              # Optional — marks this skill a runnable automation
      schedule: "0 9 * * *"              #   cron expr / "every 2h" / ISO timestamp
      deliver: origin                    #   optional (default origin)
      prompt: "Task instruction for each run"  # optional
      no_agent: false                    # optional
required_environment_variables:          # Optional — env vars the skill needs
  - name: MY_API_KEY
    prompt: "Enter your API key"
    help: "Get one at https://example.com"
    required_for: "API access"
---

# Skill Title

Brief intro.

## When to Use
Trigger conditions — when should the agent load this skill?

## Quick Reference
Table of common commands or API calls.

## Procedure
Step-by-step instructions the agent follows.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

### 플랫폼별 스킬

스킬은 `platforms` 필드를 사용하여 특정 운영 체제로 제한할 수 있습니다:

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders)
platforms: [macos, linux]     # macOS and Linux
platforms: [windows]          # Windows only
```

설정하면 호환되지 않는 플랫폼에서는 해당 스킬이 시스템 프롬프트, `skills_list()`, 슬래시 명령에서 자동으로 숨겨집니다. 생략하거나 비워 두면 모든 플랫폼에서 스킬이 로드됩니다(이전 버전과의 호환성 유지).

### 조건부 스킬 활성화

스킬은 특정 도구나 도구 세트에 대한 의존성을 선언할 수 있습니다. 이에 따라 특정 세션의 시스템 프롬프트에 스킬이 표시되는지가 결정됩니다.

```yaml
metadata:
  hermes:
    requires_toolsets: [web]           # Hide if the web toolset is NOT active
    requires_tools: [web_search]       # Hide if web_search tool is NOT available
    fallback_for_toolsets: [browser]   # Hide if the browser toolset IS active
    fallback_for_tools: [browser_navigate]  # Hide if browser_navigate IS available
```

| 필드 | 동작 |
|-------|----------|
| `requires_toolsets` | 나열된 도구 세트 중 사용 **가능하지 않은** 것이 하나라도 있으면 스킬을 숨김 |
| `requires_tools` | 나열된 도구 중 사용 **가능하지 않은** 것이 하나라도 있으면 스킬을 숨김 |
| `fallback_for_toolsets` | 나열된 도구 세트 중 사용 **가능한** 것이 하나라도 있으면 스킬을 숨김 |
| `fallback_for_tools` | 나열된 도구 중 사용 **가능한** 것이 하나라도 있으면 스킬을 숨김 |

**`fallback_for_*` 사용 사례:** 기본 도구를 사용할 수 없을 때 해결 방법으로 동작하는 스킬을 만드세요. 예를 들어 API 키가 필요하고 웹 검색 도구가 구성되지 않은 경우에만 표시되는 `fallback_for_tools: [web_search]` 설정의 `duckduckgo-search` 스킬이 있습니다.

**`requires_*` 사용 사례:** 특정 도구가 있어야만 의미가 있는 스킬을 만들 때 사용합니다. 예를 들어 웹 도구가 비활성화되어 있을 때 웹 도구 세트를 사용하는 웹 스크래핑 워크플로 스킬은 프롬프트를 복잡하게 만들지 않습니다.

### 환경 변수 요구 사항

스킬은 필요한 환경 변수를 선언할 수 있습니다. 스킬이 `skill_view`를 통해 로드되면, 필요한 변수는 샌드박스 실행 환경(`terminal`, `execute_code`)으로 전달되도록 자동 등록됩니다.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: "Tenor API key"               # Shown when prompting user
    help: "Get your key at https://tenor.com"  # Help text or URL
    required_for: "GIF search functionality"   # What needs this var
```

각 항목은 다음을 지원합니다:
- `name` (필수) — 환경 변수 이름
- `prompt` (선택) — 사용자에게 값을 요청할 때 표시할 프롬프트 텍스트
- `help` (선택) — 값을 얻는 방법에 대한 도움말 텍스트 또는 URL
- `required_for` (선택) — 이 변수가 필요한 기능에 대한 설명

사용자는 `config.yaml`에서 전달할 변수를 수동으로 구성할 수도 있습니다:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_VAR
    - ANOTHER_VAR
```

macOS 전용 스킬의 예시는 `skills/apple/`을 참조하세요.

## 로드 시 보안 설정

스킬에 API 키나 토큰이 필요하면 `required_environment_variables`를 사용하세요. 값이 없다고 해서 검색에서 스킬이 숨겨지지는 않습니다. 대신 로컬 CLI에서 스킬이 로드될 때 Hermes가 안전하게 값을 요청합니다.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

사용자는 설정을 건너뛰고 스킬을 계속 로드할 수 있습니다. Hermes는 원시 비밀 값을 모델에 노출하지 않습니다. 게이트웨이와 메시징 세션에서는 인밴드로 비밀을 수집하는 대신 로컬 설정 안내를 표시합니다.

:::tip 샌드박스 전달
스킬이 로드되면 설정된 `required_environment_variables`가 자동으로 `execute_code` 및 `terminal` 샌드박스로 전달됩니다. 여기에는 Docker 및 Modal과 같은 원격 백엔드도 포함됩니다. 따라서 사용자가 별도로 구성하지 않아도 스킬의 스크립트에서 `$TENOR_API_KEY`(또는 Python에서는 `os.environ["TENOR_API_KEY"]`)에 접근할 수 있습니다. 자세한 의미는 [환경 변수 전달](/user-guide/security#environment-variable-passthrough)을 참조하세요.
:::

이전 버전의 `prerequisites.env_vars`도 이전 버전과의 호환성을 위한 별칭으로 계속 지원됩니다.

### 구성 설정(config.yaml)

스킬은 비밀이 아닌 설정을 `skills.config` 네임스페이스 아래의 `config.yaml`에 저장하도록 선언할 수 있습니다. 환경 변수(비밀이며 `.env`에 저장됨)와 달리 구성 설정은 경로, 기본 설정 및 기타 민감하지 않은 값을 위한 것입니다.

```yaml
metadata:
  hermes:
    config:
      - key: myplugin.path
        description: Path to the plugin data directory
        default: "~/myplugin-data"
        prompt: Plugin data directory path
      - key: myplugin.domain
        description: Domain the plugin operates on
        default: ""
        prompt: Plugin domain (e.g., AI/ML research)
```

각 항목은 다음을 지원합니다:
- `key` (필수) — 설정의 점 경로(예: `myplugin.path`)
- `description` (필수) — 설정이 제어하는 내용을 설명
- `default` (선택) — 사용자가 구성하지 않았을 때의 기본값
- `prompt` (선택) — `hermes config migrate` 중 표시되는 프롬프트 텍스트; 없으면 `description` 사용

**작동 방식:**

1. **저장:** 값은 `config.yaml`의 `skills.config.<key>` 아래에 기록됩니다:
   ```yaml
   skills:
     config:
       myplugin:
         path: ~/my-data
   ```

2. **검색:** `hermes config migrate`는 활성화된 모든 스킬을 검색하고, 구성되지 않은 설정을 찾아 사용자에게 요청합니다. 설정은 `hermes config show`의 "Skill Settings"에도 표시됩니다.

3. **런타임 주입:** 스킬이 로드되면 해당 구성 값이 확인되어 스킬 메시지에 추가됩니다:
   ```
   [Skill config (from ~/.hermes/config.yaml):
     myplugin.path = /home/user/my-data
   ]
   ```
   에이전트는 `config.yaml` 자체를 읽지 않아도 구성된 값을 확인할 수 있습니다.

4. **수동 설정:** 사용자는 다음과 같이 값을 직접 설정할 수도 있습니다:
   ```bash
   hermes config set skills.config.myplugin.path ~/my-data
   ```

:::tip 어떤 설정을 사용해야 하나요?
API 키, 토큰 및 기타 **비밀**(모델에 표시되지 않으며 `~/.hermes/.env`에 저장됨)에는 `required_environment_variables`를 사용하세요. **경로, 기본 설정 및 민감하지 않은 설정**(`config.yaml`에 저장되며 config show에서 표시됨)에는 `config`를 사용하세요.
:::

### 자격 증명 파일 요구 사항(OAuth 토큰 등)

OAuth 또는 파일 기반 자격 증명을 사용하는 스킬은 원격 샌드박스에 마운트해야 하는 파일을 선언할 수 있습니다. 이는 환경 변수가 아닌 **파일**로 저장되는 자격 증명(일반적으로 설정 스크립트가 생성하는 OAuth 토큰 파일)을 위한 것입니다.

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

각 항목은 다음을 지원합니다:
- `path` (필수) — `~/.hermes/`를 기준으로 한 파일 경로
- `description` (선택) — 파일과 파일 생성 방법에 대한 설명

로드되면 Hermes가 이 파일의 존재 여부를 확인합니다. 파일이 없으면 `setup_needed`가 트리거됩니다. 기존 파일은 자동으로 다음과 같이 처리됩니다:
- Docker 컨테이너에 읽기 전용 바인드 마운트로 **마운트**됨
- Modal 샌드박스에 (생성 시점과 각 명령 전마다 동기화되어 세션 중 OAuth도 작동하도록) **동기화**됨
- **로컬** 백엔드에서는 특별한 처리 없이 사용 가능

:::tip 어떤 설정을 사용해야 하나요?
간단한 API 키와 토큰(문자열이며 `~/.hermes/.env`에 저장됨)에는 `required_environment_variables`를 사용하세요. OAuth 토큰 파일, 클라이언트 비밀, 서비스 계정 JSON, 인증서 또는 디스크에 파일로 저장된 자격 증명에는 `required_credential_files`를 사용하세요.
:::

두 가지를 모두 사용하는 완전한 예시는 `skills/productivity/google-workspace/SKILL.md`를 참조하세요.

## 스킬 지침

### 외부 의존성 없음

표준 라이브러리 Python, curl 및 기존 Hermes 도구(`web_extract`, `terminal`, `read_file`)를 우선 사용하세요. 의존성이 필요하면 스킬에 설치 단계를 문서화하세요.

### 점진적 공개

가장 일반적인 워크플로를 먼저 배치하세요. 예외 사례와 고급 사용법은 아래쪽에 둡니다. 이렇게 하면 일반적인 작업에서 토큰 사용량을 낮게 유지할 수 있습니다.

### 헬퍼 스크립트 포함

XML/JSON 파싱이나 복잡한 로직에는 `scripts/`에 헬퍼 스크립트를 포함하세요. 매번 LLM이 파서를 인라인으로 작성하게 하지 마세요.

### 미디어를 문서로 전달(`[[as_document]]`)

스킬이 고해상도 스크린샷, 차트 또는 손실 미리 보기 압축으로 인해 품질이 저하될 수 있는 이미지를 생성한다면 응답 어딘가(일반적으로 마지막 줄)에 리터럴 지시어 `[[as_document]]`를 출력하세요. 게이트웨이는 이 지시어를 제거하고 해당 응답에서 추출한 모든 미디어 경로를 인라인 이미지 말풍선 대신 다운로드 가능한 파일 첨부로 전달합니다. 전체 의미는 [스킬 출력 및 미디어 전달](../user-guide/features/skills.md#skill-output-and-media-delivery)을 참조하세요.

#### SKILL.md에서 번들 스크립트 참조

스킬이 로드되면 활성화 메시지에 절대 스킬 디렉터리가 `[Skill directory: /abs/path]`로 노출되며, SKILL.md 본문의 어디에 있든 다음 템플릿 토큰 두 개도 대체됩니다:

| 토큰 | 대체 결과 |
|---|---|
| `${HERMES_SKILL_DIR}` | 스킬 디렉터리의 절대 경로 |
| `${HERMES_SESSION_ID}` | 활성 세션 ID(세션이 없으면 그대로 유지) |

따라서 SKILL.md에서 번들 스크립트를 직접 실행하도록 에이전트에 다음과 같이 지시할 수 있습니다:

```markdown
To analyse the input, run:

    node ${HERMES_SKILL_DIR}/scripts/analyse.js <input>
```

에이전트는 대체된 절대 경로를 확인하고 별도의 `skill_view` 왕복이나 경로 계산 없이 실행할 명령과 함께 `terminal` 도구를 호출합니다. `config.yaml`에서 `skills.template_vars: false`를 설정하면 전역 대체를 비활성화할 수 있습니다.

#### 인라인 셸 스니펫(선택적 활성화)

스킬은 `` !`cmd` `` 형식의 인라인 셸 스니펫을 SKILL.md 본문에 포함할 수도 있습니다. 활성화하면 각 스니펫의 표준 출력이 에이전트가 읽기 전에 메시지에 삽입되므로 스킬이 동적 컨텍스트를 주입할 수 있습니다:

```markdown
Current date: !`date -u +%Y-%m-%d`
Git branch: !`git -C ${HERMES_SKILL_DIR} rev-parse --abbrev-ref HEAD`
```

이는 **기본적으로 꺼져 있습니다** — SKILL.md의 스니펫은 승인 없이 호스트에서 실행되므로 신뢰하는 스킬 소스에 대해서만 활성화하세요:

```yaml
# config.yaml
skills:
  inline_shell: true
  inline_shell_timeout: 10   # seconds per snippet
```

스니펫은 스킬 디렉터리를 작업 디렉터리로 사용하여 실행되며 출력은 4000자로 제한됩니다. 실패(시간 초과, 0이 아닌 종료)는 전체 스킬을 중단하는 대신 짧은 `[inline-shell error: ...]` 표시로 나타납니다.

### 테스트

스킬을 실행하고 에이전트가 지침을 올바르게 따르는지 확인하세요:

```bash
hermes chat --toolsets skills -q "Use the X skill to do Y"
```

## 스킬은 어디에 두어야 하나요?

번들 스킬(`skills/`)은 모든 Hermes 설치에 포함됩니다. 다음과 같이 **대부분의 사용자에게 폭넓게 유용해야** 합니다:

- 문서 처리, 웹 리서치, 일반적인 개발 워크플로, 시스템 관리
- 다양한 사용자가 정기적으로 사용

스킬이 공식적이고 유용하지만 모든 사용자에게 보편적으로 필요한 것은 아니라면(예: 유료 서비스 통합, 무거운 의존성) `optional-skills/`에 넣으세요. 저장소와 함께 제공되고 `hermes skills browse`를 통해("official"로 표시되어) 검색할 수 있으며, 기본 제공 신뢰로 설치됩니다.

스킬이 전문적이거나 커뮤니티가 기여했거나 틈새 용도라면 **Skills Hub**에 더 적합합니다. 레지스트리에 업로드하고 `hermes skills install`을 통해 공유하세요.

## 블루프린트: 자동화이기도 한 스킬

**블루프린트**는 프론트매터에 일정을 추가로 선언하는 일반적인 스킬입니다. `metadata.hermes.blueprint` 블록을 추가하면 스킬이 공유 가능하고 실행 가능한 자동화로 바뀝니다:

```yaml
metadata:
  hermes:
    tags: [blueprint, email]
    blueprint:
      schedule: "0 8 * * *"     # presence of `blueprint:` marks it runnable
      deliver: telegram          # optional (default: origin)
      prompt: "Summarize my unread email and today's calendar."  # optional
      no_agent: false            # optional
```

블루프린트는 **스킬이므로** 검색, 검사, 설치, 보안 검사, 출처, 탭, 중앙 집중형 인덱스 및 공유를 위한 `hermes skills publish`까지 전체 스킬 파이프라인을 변경 없이 통과합니다. 새로 배울 것은 없습니다.

**블루프린트 설치.** `blueprint:` 블록을 포함한 스킬을 설치하면 Hermes는 이를 예약하는 대신 **추천 cron 작업**으로 등록합니다. 일정 예약은 **선택 사항**이며, 설치만으로 반복 작업이 조용히 생성되지는 않습니다. `/suggestions`를 통해 검토하고 수락합니다:

```bash
hermes skills install owner/morning-brief
# → Blueprint: 'morning-brief' is an automation (schedule 0 8 * * *).
#   Added to your suggestions — run /suggestions to schedule or dismiss it.

# then, in a session:
/suggestions             # lists pending suggestions, numbered
/suggestions accept 1    # creates the cron job
/suggestions dismiss 1   # never offer it again
```

블루프린트는 통합된 추천 cron 작업 화면의 한 **출처**입니다. 이 화면에는 엄선된 시작 자동화와 (이후 추가될) 사용 패턴 및 통합 추천이 같은 곳에 표시됩니다. 아래의 [추천 cron 작업](#suggested-cron-jobs)을 참조하세요.

**만든 자동화 공유.** cron 작업(`hermes cron create --skill <name> ...`)이 로드한 블루프린트는 다시 SKILL.md로 내보내고 다른 스킬처럼 게시할 수 있습니다. 따라서 자신에게 맞게 조정한 자동화를 다른 사람이 한 번의 명령으로 설치할 수 있게 됩니다.

블루프린트 계층은 새로운 객체 유형, 저장소 또는 전송 방식을 추가하지 않습니다. 블루프린트는 스킬이고, 일정은 cron 작업이며, 공유는 기존 게시/탭/인덱스 경로를 사용합니다.

## 추천 cron 작업

Hermes는 자동화를 제안하고 한 번 탭하여 수락할 수 있게 하므로 cron 작업을 직접 조합할 필요가 없습니다. 모든 제안은 어디에서 왔는지와 관계없이 하나의 화면인 `/suggestions` 명령을 통해 제공됩니다:

| 출처 | 트리거 |
|--------|---------|
| `catalog` | 엄선된 시작 자동화(`/suggestions catalog`) — 일일 브리핑, 중요 메일 모니터, 주간 검토, 업무 시작 알림 |
| `blueprint` | `blueprint:` 블록이 있는 스킬을 설치함 |
| `usage` | 백그라운드 검토에서 일정으로 처리할 수 있는 반복 요청을 발견함 |
| `integration` | 계정을 연결함(Gmail, GitHub, ...)으로써 명확한 자동화가 제안됨 |

```bash
/suggestions             # list pending
/suggestions accept N    # schedule suggestion N (creates the cron job)
/suggestions dismiss N   # dismiss it — latched, never re-offered
/suggestions catalog     # add the curated starter automations
```

제안을 수락하면 `cronjob` 도구가 사용하는 것과 동일한 `cron.jobs.create_job`이 호출되므로 두 번째 작업 엔진은 없습니다. 제안이 작업을 **자동으로 생성하는 일은 절대 없으며**, 수락은 항상 명시적으로 이루어집니다. 해제된 제안은 안정적인 키로 고정되므로 같은 제안이 다시 표시되지 않습니다. 보류 중인 목록에는 상한이 있어 잔소리 목록이 되지 않습니다.

**중요 메일 모니터** 카탈로그 항목은 폴링→분류→표시 패턴입니다. 저렴한 분류 모델(`config.yaml`의 `auxiliary.monitor`)로 받은 편지함 항목을 평가하고 긴급성 임계값을 넘는 항목만 전달하며, 그렇지 않으면 조용히 유지됩니다.

## 스킬 게시

### Skills Hub로

```bash
hermes skills publish skills/my-skill --to github --repo owner/repo
```

### 사용자 지정 저장소로

저장소를 탭으로 추가합니다:

```bash
hermes skills tap add owner/repo
```

그러면 사용자가 저장소에서 검색하고 설치할 수 있습니다.

## 보안 검사

Hub에서 설치된 모든 스킬은 다음을 검사하는 보안 스캐너를 거칩니다:

- 데이터 유출 패턴
- 프롬프트 인젝션 시도
- 파괴적인 명령
- 셸 인젝션

신뢰 수준:
- `builtin` — Hermes에 포함됨(항상 신뢰)
- `official` — 저장소의 `optional-skills/`에서 제공됨(기본 제공 신뢰, 타사 경고 없음)
- `trusted` — openai/skills, anthropics/skills, huggingface/skills에서 제공됨
- `community` — 위험하지 않은 발견은 `--force`로 재정의할 수 있으며, `dangerous` 판정은 계속 차단됨

이제 Hermes는 여러 외부 검색 모델의 서드파티 스킬도 사용할 수 있습니다:
- 직접 GitHub 식별자(예: `openai/skills/k8s`)
- `skills.sh` 식별자(예: `skills-sh/vercel-labs/json-render/json-render-react`)
- `/.well-known/skills/index.json`에서 제공되는 잘 알려진 엔드포인트

GitHub 전용 설치 프로그램 없이도 스킬을 검색할 수 있게 하려면 저장소나 마켓플레이스에 게시하는 것과 함께 잘 알려진 엔드포인트에서 제공하는 방법을 고려하세요.

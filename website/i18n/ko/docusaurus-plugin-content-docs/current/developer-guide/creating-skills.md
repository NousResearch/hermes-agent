---
sidebar_position: 3
title: "Creating Skills"
description: "Hermes 에이전트용 스킬을 만드는 방법 - SKILL.md 형식, 가이드라인 및 게시(publishing)"
---

# 스킬 생성하기

스킬(Skills)은 Hermes 에이전트에 새로운 기능을 추가하는 권장 방법입니다. 도구를 만드는 것보다 훨씬 쉽고, 에이전트의 코드 변경이 필요하지 않으며, 커뮤니티와 쉽게 공유할 수 있습니다.

## 스킬(Skill)이어야 할까요, 도구(Tool)이어야 할까요?

다음의 경우 **스킬**로 만드세요:
- 기능을 지침(instructions) + 셸 명령어 + 기존 도구로 표현할 수 있을 때
- 에이전트가 `terminal`이나 `web_extract`를 통해 호출할 수 있는 외부 CLI나 API를 래핑할 때
- 에이전트에 내장되어야 할 사용자 정의 Python 연동이나 API 키 관리가 필요하지 않을 때
- 예시: arXiv 검색, git 워크플로우, Docker 관리, PDF 처리, CLI 도구를 통한 이메일

다음의 경우 **도구**로 만드세요:
- API 키, 인증 흐름, 다중 구성요소 설정과의 엔드투엔드 통합이 필요할 때
- 매번 정확하게 실행되어야 하는 맞춤형 처리 로직이 필요할 때
- 바이너리 데이터, 스트리밍, 또는 실시간 이벤트를 처리할 때
- 예시: 브라우저 자동화, TTS, 비전 분석

## 스킬 디렉토리 구조

번들로 제공되는 스킬은 카테고리별로 `skills/`에 있습니다. 공식적인 선택적 스킬들은 `optional-skills/`에서 동일한 구조를 사용합니다:

```text
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # 필수: 핵심 지침(instructions)
│       └── scripts/              # 선택: 헬퍼 스크립트
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
description: 짧은 설명 (스킬 검색 결과에 표시됨)
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # 선택 — 특정 OS 플랫폼으로 제한
                                   #   유효한 값: macos, linux, windows
                                   #   생략 시 모든 플랫폼에서 로드됨(기본값)
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    requires_toolsets: [web]            # 선택 — 이러한 도구 세트가 활성화될 때만 표시됨
    requires_tools: [web_search]        # 선택 — 이러한 도구가 있을 때만 표시됨
    fallback_for_toolsets: [browser]    # 선택 — 이러한 도구 세트가 활성화될 때 숨김
    fallback_for_tools: [browser_navigate]  # 선택 — 이러한 도구가 있을 때 숨김
    config:                              # 선택 — 스킬에 필요한 config.yaml 설정
      - key: my.setting
        description: "이 설정이 제어하는 것"
        default: "sensible-default"
        prompt: "설정 시 표시할 프롬프트"
required_environment_variables:          # 선택 — 스킬에 필요한 환경 변수
  - name: MY_API_KEY
    prompt: "API 키를 입력하세요"
    help: "https://example.com에서 얻을 수 있습니다"
    required_for: "API 액세스"
---

# 스킬 제목

간략한 소개.

## 사용 시기 (When to Use)
트리거 조건 — 에이전트가 언제 이 스킬을 로드해야 하나요?

## 빠른 참조 (Quick Reference)
일반적인 명령어나 API 호출 표.

## 절차 (Procedure)
에이전트가 따를 단계별 지침.

## 주의 사항 (Pitfalls)
알려진 실패 사례 및 처리 방법.

## 검증 (Verification)
에이전트가 성공적으로 수행했는지 확인하는 방법.
```

### 플랫폼 특정 스킬

스킬은 `platforms` 필드를 사용하여 특정 운영 체제로 스스로를 제한할 수 있습니다:

```yaml
platforms: [macos]            # macOS 전용 (예: iMessage, Apple 미리 알림)
platforms: [macos, linux]     # macOS 및 Linux
platforms: [windows]          # Windows 전용
```

이 값이 설정되면, 호환되지 않는 플랫폼에서는 시스템 프롬프트, `skills_list()`, 슬래시 명령어에서 스킬이 자동으로 숨겨집니다. 생략하거나 비워두면, 스킬은 모든 플랫폼에서 로드됩니다 (하위 호환성 유지).

### 조건부 스킬 활성화

스킬은 특정 도구나 도구 세트에 대한 종속성을 선언할 수 있습니다. 이는 특정 세션의 시스템 프롬프트에 스킬이 나타날지 여부를 제어합니다.

```yaml
metadata:
  hermes:
    requires_toolsets: [web]           # web 도구 세트가 활성화되지 않으면 숨김
    requires_tools: [web_search]       # web_search 도구가 없으면 숨김
    fallback_for_toolsets: [browser]   # browser 도구 세트가 활성화되면 숨김
    fallback_for_tools: [browser_navigate]  # browser_navigate 도구가 있으면 숨김
```

| 필드 | 동작 |
|-------|----------|
| `requires_toolsets` | 나열된 도구 세트 중 하나라도 사용할 수 **없으면** 스킬이 **숨겨집니다** |
| `requires_tools` | 나열된 도구 중 하나라도 사용할 수 **없으면** 스킬이 **숨겨집니다** |
| `fallback_for_toolsets` | 나열된 도구 세트 중 하나라도 사용할 수 **있으면** 스킬이 **숨겨집니다** |
| `fallback_for_tools` | 나열된 도구 중 하나라도 사용할 수 **있으면** 스킬이 **숨겨집니다** |

**`fallback_for_*`의 사용 사례:** 주 도구를 사용할 수 없을 때 우회책 역할을 하는 스킬을 만듭니다. 예를 들어, `fallback_for_tools: [web_search]`가 설정된 `duckduckgo-search` 스킬은 (API 키가 필요한) 웹 검색 도구가 구성되지 않았을 때만 나타납니다.

**`requires_*`의 사용 사례:** 특정 도구가 있을 때만 의미가 있는 스킬을 만듭니다. 예를 들어, `requires_toolsets: [web]`이 설정된 웹 스크래핑 워크플로우 스킬은 웹 도구가 비활성화된 경우 프롬프트를 어지럽히지 않습니다.

### 환경 변수 요구 사항

스킬은 필요한 환경 변수를 선언할 수 있습니다. 스킬이 `skill_view`를 통해 로드되면 필요한 변수가 샌드박스화된 실행 환경(terminal, execute_code)으로 통과(passthrough)되도록 자동으로 등록됩니다.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: "Tenor API key"               # 사용자에게 물어볼 때 표시
    help: "Get your key at https://tenor.com"  # 도움말 텍스트 또는 URL
    required_for: "GIF search functionality"   # 이 변수를 필요로 하는 것
```

각 항목은 다음을 지원합니다:
- `name` (필수) — 환경 변수 이름
- `prompt` (선택) — 사용자에게 값을 요청할 때의 프롬프트 텍스트
- `help` (선택) — 값을 얻기 위한 도움말 텍스트 또는 URL
- `required_for` (선택) — 어떤 기능에 이 변수가 필요한지 설명

사용자는 `config.yaml`에 통과(passthrough) 변수를 수동으로 구성할 수도 있습니다:

```yaml
terminal:
  env_passthrough:
    - MY_CUSTOM_VAR
    - ANOTHER_VAR
```

macOS 전용 스킬의 예시는 `skills/apple/`을 참고하세요.

## 로드 시 보안 설정

스킬이 API 키나 토큰을 필요로 할 때 `required_environment_variables`를 사용하세요. 값이 누락되어도 스킬 검색에서 **숨겨지지 않습니다**. 대신, 로컬 CLI에 스킬이 로드될 때 Hermes가 안전하게 값을 요청합니다.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

사용자는 설정을 건너뛰고 스킬 로드를 계속할 수 있습니다. Hermes는 원시 비밀 값을 절대 모델에 노출하지 않습니다. 게이트웨이와 메시징 세션은 밴드 내에서 비밀을 수집하는 대신 로컬 설정 가이드를 보여줍니다.

:::tip 샌드박스 통과(Passthrough)
스킬이 로드될 때, 설정된 선언된 `required_environment_variables`는 Docker 및 Modal과 같은 원격 백엔드를 포함한 `execute_code` 및 `terminal` 샌드박스로 **자동 통과(passed through)**됩니다. 사용자가 별도로 구성할 필요 없이 스킬의 스크립트에서 `$TENOR_API_KEY` (또는 Python의 `os.environ["TENOR_API_KEY"]`)에 접근할 수 있습니다. 자세한 내용은 [환경 변수 패스스루](/user-guide/security#environment-variable-passthrough)를 참고하세요.
:::

이전 버전의 `prerequisites.env_vars`도 하위 호환성을 위해 지원됩니다.

### 설정 (config.yaml)

스킬은 `skills.config` 네임스페이스 아래의 `config.yaml`에 저장되는 비밀이 아닌 설정을 선언할 수 있습니다. (`.env`에 저장되는 비밀인) 환경 변수와 달리 구성 설정은 경로, 선호도 및 기타 덜 민감한 값들을 위한 것입니다.

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
- `key` (필수) — 설정을 위한 점 표기 경로 (예: `myplugin.path`)
- `description` (필수) — 이 설정이 제어하는 것을 설명
- `default` (선택) — 사용자가 구성하지 않은 경우의 기본값
- `prompt` (선택) — `hermes config migrate` 중에 표시되는 프롬프트 텍스트; 기본적으로 `description`으로 폴백

**작동 방식:**

1. **저장:** 값은 `skills.config.<key>` 아래에 `config.yaml`에 쓰여집니다:
   ```yaml
   skills:
     config:
       myplugin:
         path: ~/my-data
   ```

2. **검색:** `hermes config migrate`는 모든 활성화된 스킬을 스캔하고, 구성되지 않은 설정을 찾고 사용자에게 묻습니다. 설정은 또한 "Skill Settings" 아래 `hermes config show`에 나타납니다.

3. **런타임 주입:** 스킬이 로드될 때, 해당 구성 값이 해석되고 스킬 메시지에 덧붙여집니다:
   ```
   [Skill config (from ~/.hermes/config.yaml):
     myplugin.path = /home/user/my-data
   ]
   ```
   에이전트는 자체적으로 `config.yaml`을 읽을 필요 없이 구성된 값을 봅니다.

4. **수동 설정:** 사용자는 값을 직접 설정할 수도 있습니다:
   ```bash
   hermes config set skills.config.myplugin.path ~/my-data
   ```

:::tip 언제 무엇을 사용할까
API 키, 토큰 및 기타 **비밀**(절대 모델에 보이지 않고 `~/.hermes/.env`에 저장됨)에 대해서는 `required_environment_variables`를 사용하세요. **경로, 선호도, 그리고 덜 민감한 설정**(구성 보기에 표시되며 `config.yaml`에 저장됨)에 대해서는 `config`를 사용하세요.
:::

### 자격 증명 파일 요구 사항 (OAuth 토큰 등)

OAuth 또는 파일 기반 자격 증명을 사용하는 스킬은 원격 샌드박스에 마운트되어야 하는 파일을 선언할 수 있습니다. 이는 환경 변수가 아닌 **파일**로 저장된 자격 증명을 위한 것입니다. (보통 설정 스크립트에 의해 생성된 OAuth 토큰 파일)

```yaml
required_credential_files:
  - path: google_token.json
    description: Google OAuth2 token (created by setup script)
  - path: google_client_secret.json
    description: Google OAuth2 client credentials
```

각 항목은 다음을 지원합니다:
- `path` (필수) — `~/.hermes/` 기준의 상대 파일 경로
- `description` (선택) — 파일이 무엇이며 어떻게 생성되는지 설명

로드될 때 Hermes는 이 파일들이 존재하는지 확인합니다. 누락된 파일은 `setup_needed`를 발생시킵니다. 존재하는 파일은 자동으로:
- **Docker에 마운트** — 읽기 전용 바인드 마운트로 컨테이너 내부 연결
- **Modal에 동기화** — (세션 중의 OAuth가 동작하도록 각 명령어 전에 샌드박스에 복사됨)
- 특별한 처리 없이 **로컬** 백엔드에서 사용 가능

:::tip 언제 무엇을 사용할까
단순한 API 키 및 토큰 (`~/.hermes/.env`에 저장되는 문자열)에는 `required_environment_variables`를 사용하세요. OAuth 토큰 파일, 클라이언트 시크릿, 서비스 계정 JSON, 인증서 또는 디스크에 파일로 존재하는 모든 자격 증명에는 `required_credential_files`를 사용하세요.
:::

두 가지를 모두 사용하는 완전한 예시는 `skills/productivity/google-workspace/SKILL.md`를 참고하세요.

## 스킬 가이드라인

### 외부 종속성 금지

표준 라이브러리(stdlib) Python, curl, 그리고 기존 Hermes 도구(`web_extract`, `terminal`, `read_file`)를 우선적으로 사용하세요. 만약 종속성이 꼭 필요하다면 설치 단계를 스킬에 문서화하세요.

### 점진적인 공개 (Progressive Disclosure)

가장 일반적인 워크플로우를 먼저 배치하세요. 엣지 케이스나 고급 사용법은 맨 아래로 보냅니다. 이는 일반적인 작업의 토큰 사용량을 낮게 유지시켜 줍니다.

### 헬퍼 스크립트 포함

XML/JSON 파싱이나 복잡한 로직의 경우 `scripts/`에 헬퍼 스크립트를 포함하세요 — LLM이 매번 파서를 인라인으로 작성하리라 기대하지 마세요.

### 미디어를 문서로 전달하기 (`[[as_document]]`)

스킬이 고해상도 스크린샷, 차트 또는 손실 압축 미리보기가 적절하지 않은 모든 이미지를 생성하는 경우 응답의 어딘가에(보통 마지막 줄) 문자열 리터럴 `[[as_document]]`를 내보내세요(emit). 게이트웨이는 지시문을 제거하고 해당 응답에서 추출된 모든 미디어 경로를 인라인 이미지 버블 대신 다운로드 가능한 파일 첨부로 전달합니다. 완전한 의미론에 대해서는 [스킬 출력 및 미디어 전송](../user-guide/features/skills.md#skill-output-and-media-delivery)을 참고하세요.

#### SKILL.md에서 번들 스크립트 참조하기

스킬이 로드될 때 활성화 메시지는 절대 스킬 디렉토리를 `[Skill directory: /abs/path]`로 노출하며, 또한 SKILL.md 본문 내 아무 곳에나 두 개의 템플릿 토큰을 치환합니다:

| 토큰 | 교체되는 값 |
|---|---|
| `${HERMES_SKILL_DIR}` | 스킬 디렉토리의 절대 경로 |
| `${HERMES_SESSION_ID}` | 활성 세션 ID (세션이 없으면 그대로 남음) |

따라서 SKILL.md는 에이전트에게 번들로 포함된 스크립트를 직접 실행하도록 지시할 수 있습니다:

```markdown
입력을 분석하려면 다음을 실행하세요:

    node ${HERMES_SKILL_DIR}/scripts/analyse.js <input>
```

에이전트는 치환된 절대 경로를 보고 준비된 명령어와 함께 `terminal` 도구를 실행합니다 — 경로 계산이나 추가적인 `skill_view` 왕복이 없습니다. `config.yaml`에서 `skills.template_vars: false`를 통해 치환을 전역적으로 비활성화할 수 있습니다.

#### 인라인 셸 스니펫 (선택 사항)

스킬은 SKILL.md 본문에 `` !`cmd` ``로 작성된 인라인 셸 스니펫을 삽입할 수도 있습니다. 활성화되면 각 스니펫의 stdout(표준 출력)이 에이전트가 읽기 전에 메시지에 인라인되므로 스킬이 동적 컨텍스트를 주입할 수 있습니다:

```markdown
현재 날짜: !`date -u +%Y-%m-%d`
Git 브랜치: !`git -C ${HERMES_SKILL_DIR} rev-parse --abbrev-ref HEAD`
```

이 기능은 **기본적으로 꺼져 있습니다** — SKILL.md의 모든 스니펫은 승인 없이 호스트에서 실행되므로 신뢰하는 스킬 소스에 대해서만 활성화하세요:

```yaml
# config.yaml
skills:
  inline_shell: true
  inline_shell_timeout: 10   # 스니펫당 초 단위 시간
```

스니펫은 스킬 디렉토리를 작업 디렉토리로 사용하여 실행되며, 출력은 4000자로 제한됩니다. 실패(시간 초과, 0이 아닌 종료)는 전체 스킬을 망가뜨리는 대신 짧은 `[inline-shell error: ...]` 마커로 표시됩니다.

### 테스트하기

스킬을 실행하고 에이전트가 지침을 올바르게 따르는지 검증하세요:

```bash
hermes chat --toolsets skills -q "Use the X skill to do Y"
```

## 스킬을 어디에 배치해야 할까요?

번들 스킬(in `skills/`)은 모든 Hermes 설치와 함께 제공됩니다. 이들은 **대부분의 사용자에게 광범위하게 유용해야 합니다**:

- 문서 처리, 웹 조사, 일반적인 개발 워크플로우, 시스템 관리
- 다양한 사람들이 정기적으로 사용하는 것

스킬이 공식적이고 유용하지만 보편적으로 필요하지 않다면 (예: 유료 서비스 연동, 무거운 종속성), **`optional-skills/`**에 넣으세요 — 리포지토리와 함께 제공되고 `hermes skills browse`를 통해 검색할 수 있으며 ("official"로 레이블링됨), 내장된 신뢰(built-in trust)와 함께 설치됩니다.

스킬이 특정 분야에 특화되었거나, 커뮤니티 기여이거나, 틈새(niche) 스킬이라면 **Skills Hub**에 올리는 것이 더 낫습니다 — 레지스트리에 업로드하고 `hermes skills install`을 통해 공유하세요.

## 스킬 게시하기 (Publishing Skills)

### Skills Hub로

```bash
hermes skills publish skills/my-skill --to github --repo owner/repo
```

### 커스텀 리포지토리로

여러분의 리포지토리를 tap으로 추가하세요:

```bash
hermes skills tap add owner/repo
```

그러면 사용자들이 여러분의 리포지토리에서 검색하고 설치할 수 있습니다.

## 보안 스캔

허브에서 설치된 모든 스킬은 다음과 같은 항목을 확인하는 보안 스캐너를 거칩니다:

- 데이터 유출(exfiltration) 패턴
- 프롬프트 주입(injection) 시도
- 파괴적인 명령어
- 셸 주입

신뢰 수준(Trust levels):
- `builtin` — Hermes와 함께 제공됨 (항상 신뢰됨)
- `official` — 리포지토리의 `optional-skills/` 출처 (서드파티 경고가 없는 내장된 신뢰)
- `trusted` — openai/skills, anthropics/skills, huggingface/skills 출처
- `community` — 위험하지 않은 발견 사항은 `--force`로 무시할 수 있지만; `dangerous` 평결은 여전히 차단됩니다.

Hermes는 이제 여러 외부 검색 모델에서 서드파티 스킬을 사용할 수 있습니다:
- 직접 GitHub 식별자 (예: `openai/skills/k8s`)
- `skills.sh` 식별자 (예: `skills-sh/vercel-labs/json-render/json-render-react`)
- `/.well-known/skills/index.json`에서 제공되는 잘 알려진 엔드포인트

GitHub 전용 설치 프로그램 없이 스킬을 검색할 수 있게 하려면, 리포지토리나 마켓플레이스에 게시하는 것과 더불어 잘 알려진 엔드포인트(well-known endpoint)에서 스킬을 제공하는 것을 고려하세요.

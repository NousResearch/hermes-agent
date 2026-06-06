---
title: "Antigravity Cli — Antigravity CLI(agy) 작동: 플러그인, 인증, 샌드박스"
sidebar_label: "Antigravity Cli"
description: "Antigravity CLI(agy) 작동: 플러그인, 인증, 샌드박스"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Antigravity Cli

Antigravity CLI(agy)를 작동합니다: 플러그인, 인증, 샌드박스 등.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/antigravity-cli` 명령어로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/antigravity-cli` |
| 버전 | `0.1.0` |
| 작성자 | Tony Simons (asimons81), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Antigravity`, `CLI`, `Auth`, `Plugins`, `Sandbox` |
| 관련 스킬 | [`grok`](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Antigravity CLI (`agy`)

`agy`로 호출되는 Antigravity CLI를 위한 운영자(Operator) 가이드입니다. Hermes `terminal` 도구를 통해 모든 `agy` 명령어를 실행하고, `read_file`로 해당 구성과 로그를 검사하세요. 이 스킬은 참조 및 절차입니다 — 네트워크 API를 래핑하지 않으므로 Hermes 자체에서 인증할 항목이 없습니다.

## 사용 시기

- `agy` 바이너리를 설치, 업데이트 또는 스모크 테스트할 때
- 대화형이 아닌 `agy --print` / `agy -p` 원샷 작업을 구동할 때
- Antigravity 인증, 샌드박스, 권한 또는 플러그인 상태를 디버깅할 때
- Antigravity 설정, 키 바인딩, 대화 기록 또는 로그를 읽을 때

## 멘탈 모델

Antigravity는 두 개의 레이어로 구성되어 있습니다 — 이를 명확히 구분하지 않으면 지침이 잘못될 수 있습니다:

1. **셸 래퍼(Shell wrapper) 명령어** — `agy help`, `agy install`, `agy plugin`, `agy update`, `agy changelog`. `terminal` 도구를 통해 실행합니다.
2. **세션 내 대화형 슬래시 명령어** — `/config`, `/permissions`, `/skills`, `/agents` 등. 이들은 실행 중인 `agy` TUI 세션 내에만 존재하며, 셸 래퍼에는 존재하지 않습니다.

`agy help`는 세션 내 슬래시 명령어가 아닌 셸 래퍼의 표면을 보여줍니다.

## 전제 조건

- PATH에 `agy` 바이너리가 있어야 합니다. `terminal` 도구를 통해 다음 명령어로 확인하세요:
  `command -v agy && agy --version`.
- 이 스킬에는 환경 변수나 API 키가 필요하지 않습니다 — Antigravity는 OS 키링 / 브라우저 로그인을 통해 자체 인증을 관리합니다 (아래 인증 섹션 참조).

## 실행 방법

`terminal` 도구를 통해 모든 `agy` 명령어를 호출하세요. 예시:

```
terminal(command="agy --version")
terminal(command="agy help")
terminal(command="agy plugin list")
terminal(command="agy --print 'Summarize the repo in 3 bullets'", workdir="/path/to/project")
```

대화형 다중 턴(multi-turn) TUI 세션의 경우 `codex` / `claude-code` 스킬에서 사용하는 것과 동일한 패턴인 `pty=true`(및 캡처/모니터링용 tmux)와 함께 `agy`를 실행하세요. 원샷 스모크 테스트와 스크립트 기반 프롬프트의 경우, 대화형이 아닌 `agy --print`를 선호하세요.

Antigravity의 자체 파일을 검사하려면 아래 코어(Core) 경로에 있는 경로에서 `read_file`을 사용하세요 — 터미널을 통해 `cat`을 실행하지 마세요.

## 코어 경로 (Core paths)

- 바이너리 / 진입점: `agy`
- 앱 데이터 디렉토리: `~/.gemini/antigravity-cli/`
- 설정 파일: `~/.gemini/antigravity-cli/settings.json`
- 키 바인딩 파일: `~/.gemini/antigravity-cli/keybindings.json`
- 로그: `~/.gemini/antigravity-cli/log/cli-*.log`
- 대화: `~/.gemini/antigravity-cli/conversations/`
- 두뇌 아티팩트 (Brain artifacts): `~/.gemini/antigravity-cli/brain/`
- 히스토리: `~/.gemini/antigravity-cli/history.jsonl`
- 플러그인 준비 공간 (staging): `~/.gemini/antigravity-cli/plugins/<plugin_name>/`

## 빠른 참조

### 래퍼 명령어
- `agy changelog`
- `agy help`
- `agy install`
- `agy plugin` / `agy plugins`
- `agy update`

### 유용한 플래그
- `--add-dir`
- `--continue` / `-c`
- `--conversation`
- `--dangerously-skip-permissions`
- `--print` / `-p`
- `--print-timeout`
- `--prompt`
- `--prompt-interactive` / `-i`
- `--sandbox`
- `--log-file`
- `--version`

### 플러그인 하위 명령어 (`agy plugin --help`)
- `list`, `import [source]`, `install <target>`, `uninstall <name>`,
  `enable <name>`, `disable <name>`, `validate [path]`, `link <mp> <target>`,
  `help`

### 설치 플래그 (`agy install --help`)
- `--dir`, `--skip-aliases`, `--skip-path`

### 세션 내 슬래시 명령어
- **대화 제어:** `/resume` (`/switch`), `/rewind` (`/undo`), `/rename <name>`, `/clear`, `/fork`, `/reset`, `/new`
- **설정 및 도구:** `/config`, `/settings`, `/permissions`, `/model`, `/keybindings`, `/statusline`, `/tasks`, `/skills`, `/mcp`, `/open <path>`, `/usage`, `/logout`, `/agents`
- **프롬프트 도우미:** `@` 경로 자동 완성, `esc esc` 프롬프트 지우기(스트리밍 중이 아닐 때), `!` 터미널 명령어 직접 실행, `?` 도움말 열기

## 설정 및 권한

### 일반 설정 키 (`settings.json`)
- `allowNonWorkspaceAccess`
- `colorScheme`
- `permissions.allow`
- `trustedWorkspaces`

### 권한 모드
`request-review`, `always-proceed`, `strict`, `proceed-in-sandbox`.

### 샌드박스 동작
- `enableTerminalSandbox`는 `settings.json`의 불리언 값입니다. 기본값은 `false`입니다.
- 시작 시의 재정의 값(`--sandbox`, `--dangerously-skip-permissions`)은 현재 세션에 대한 영구 설정을 대체할 수 있습니다.

## 인증 동작

- CLI는 먼저 OS의 보안 키링(keyring)을 시도합니다.
- 저장된 세션이 없는 경우 브라우저 기반 Google 로그인으로 대체됩니다.
- 로컬 환경에서는 기본 브라우저를 열고; SSH 환경에서는 승인 URL을 인쇄하고 복사해 다시 붙여넣을 인증 코드를 기대합니다.
- `/logout`은 저장된 자격 증명을 제거합니다.

## 플러그인

- 플러그인은 `~/.gemini/antigravity-cli/plugins/<plugin_name>/` 아래에 준비됩니다.
- 스킬, 에이전트, 규칙, MCP 서버 및 후크(hooks)를 번들로 제공할 수 있습니다.
- `agy plugin list`가 아무 것도 반환하지 않는 것은 유효한 빈(empty) 상태입니다.

## 주의 사항 (Pitfalls)

- `agy help`는 대화형 슬래시 명령어가 아닌 래퍼 명령어를 보여줍니다.
- `agy --version`은 안전한 비대화형 버전 확인 방법입니다; `agy version`은 대화형이며 실제 TTY 없이는 실패할 수 있습니다.
- 실패 시 가장 먼저 확인할 곳: `~/.gemini/antigravity-cli/log/cli-*.log` (`read_file`로 읽으세요).
- 영구적인 JSON 설정과 시작 시의 재정의(override) 값을 혼동하지 마세요.
- `~/.gemini/antigravity-cli/bin/agentapi`는 `agy agentapi`에 대한 얇은 래퍼입니다.
- WSL에서 토큰 저장소는 파일 기반이므로 인증 문제는 주로 로컬 파일 / 세션 상태 문제이며 브라우저 전용 문제는 아닙니다.
- 작업 공간 신원(Workspace identity)은 시작 디렉토리와 `.antigravitycli` 프로젝트 마커에 따라 달라질 수 있습니다.

## 검증 (Verification)

다음과 같이 `terminal` 도구를 통해 설치가 정상이고 사용할 수 있는지 확인하세요 (파일은 `read_file`로 읽으세요):

1. `terminal(command="command -v agy")`
2. `terminal(command="agy --version")`
3. `terminal(command="agy help")`
4. `terminal(command="agy plugin list")`
5. `~/.gemini/antigravity-cli/settings.json`에 대한 `read_file`
6. 가장 최근의 `~/.gemini/antigravity-cli/log/cli-*.log`에 대한 `read_file`
7. 필요한 경우 `~/.gemini/antigravity-cli/keybindings.json`에 대한 `read_file`

## 지원 파일

- `references/cli-docs.md` — 시작하기, 사용법 및 기능 문서에서 요약된 노트.

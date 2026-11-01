---
title: "Antigravity CLI — Antigravity CLI(agy) 운용: 플러그인, 인증, 샌드박스"
sidebar_label: "Antigravity CLI"
description: "Antigravity CLI(agy) 운용: 플러그인, 인증, 샌드박스"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Antigravity CLI

Antigravity CLI(agy)를 운용합니다: 플러그인, 인증, 샌드박스.

## 스킬 메타데이터

| | |
|---|---|
| 원본 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/antigravity-cli`로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/antigravity-cli` |
| 버전 | `0.2.0` |
| 작성자 | Tony Simons (asimons81), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Antigravity`, `CLI`, `Auth`, `Plugins`, `Sandbox` |
| 관련 스킬 | [`grok`](/docs/user-guide/skills/optional/autonomous-ai-agents/autonomous-ai-agents-grok), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트가 보게 되는 지침입니다.
:::

# Antigravity CLI (`agy`)

`agy`로 호출하는 Antigravity CLI의 운용 안내입니다. 모든 `agy` 명령은 Hermes `terminal` 도구를 통해 실행하고, 설정과 로그는 `read_file`로 확인하세요. 이 스킬은 참고 자료이자 절차이며 네트워크 API를 감싸지 않으므로 Hermes 자체에서 인증할 항목은 없습니다.

## 사용 시점

- `agy` 바이너리 설치, 업데이트 또는 스모크 테스트
- 비대화형 `agy --print` / `agy -p` 일회성 실행
- Antigravity 인증, 샌드박스, 권한 또는 플러그인 상태 디버깅
- Antigravity 설정, 키 바인딩, 대화 또는 로그 읽기

## 개념 모델

Antigravity에는 두 계층이 있습니다. 둘을 구분하지 않으면 안내가 잘못됩니다.

1. **셸 래퍼 명령** — `agy help`, `agy install`, `agy plugin`, `agy update`, `agy changelog`. `terminal` 도구를 통해 실행합니다.
2. **대화형 세션 내 슬래시 명령** — `/config`, `/permissions`, `/skills`, `/agents` 등. 실행 중인 `agy` TUI 세션 내부에만 존재하며 셸 래퍼에는 없습니다.

`agy help`는 셸 래퍼의 기능을 보여주며 세션 내 슬래시 명령은 보여주지 않습니다.

## 사전 요구 사항

- PATH에 `agy` 바이너리가 있어야 합니다. `terminal` 도구로 다음을 확인하세요: `command -v agy && agy --version`.
- 이 스킬에 필요한 환경 변수나 API 키는 없습니다 — Antigravity가 OS 키링 / 브라우저 로그인을 통해 자체적으로 인증을 관리합니다(아래 인증 참조).

## 실행 방법

모든 `agy` 명령은 `terminal` 도구를 통해 호출하세요. 예:

```
terminal(command="agy --version")
terminal(command="agy help")
terminal(command="agy plugin list")
terminal(command="agy --print 'Summarize the repo in 3 bullets'", workdir="/path/to/project")
```

대화형 다중 턴 TUI 세션은 `agy`를 `pty=true`로 실행하고(캡처/모니터링에는 tmux 사용), `codex` / `claude-code` 스킬과 동일한 패턴을 따르세요. 일회성 스모크 테스트와 스크립트 프롬프트에는 `agy --print`(비대화형)를 우선 사용하세요.

Antigravity 자체 파일을 확인할 때는 Core paths 아래의 경로에 `read_file`을 사용하세요 — `terminal`을 통해 `cat`하지 마세요.

## 위임 패턴

`agy`는 `codex` / `claude-code`와 같은 계열의 코딩 에이전트 백엔드이므로, 단순 스모크 테스트가 아니라 실제 작업(기능, 수정, 리뷰, 2차 의견)을 맡길 때는 동일한 위임 형태를 사용하세요.

### 일회성 실행(스크립트 프롬프트와 2차 의견에 권장)

```
terminal(command="agy -p 'Review this diff for bugs and security issues' --model 'Gemini 3.1 Pro (High)'", workdir="/path/to/repo", timeout=300)
```

`-p`는 비대화형으로 실행하고 종료합니다. `--model`로 엔진을 선택하세요(`agy models`를 실행해 정확한 표시 문자열을 확인합니다. 예: `'Gemini 3.1 Pro (High)'`, `'Claude Opus 4.6 (Thinking)'`). 반복 사용 가능한 `--add-dir`로 추가 컨텍스트 루트를 지정할 수 있습니다.

### 장시간 / 제한된 실행(테스트, 빌드, 여러 파일 변경)

백그라운드로 실행하고 완료 알림을 받으세요. `codex` 스킬과 동일합니다:

```
terminal(command="agy -p 'Implement the change described in TASK.md and run the tests' --dangerously-skip-permissions", workdir="/path/to/repo", background=true, notify_on_complete=true)
# then: process(action="poll"/"log"/"wait", session_id=<id>)
```

### 대화형 다중 턴(PTY + tmux)

대화형 세션은 `agy -i`(또는 인자 없이 `agy`)를 `pty=true`로 실행하고, `capture-pane` / `send-keys`를 위한 tmux를 사용하세요. 이는 `codex` / `claude-code` 스킬에 설명된 패턴과 동일합니다. 나중에 `--continue` / `-c` 또는 특정 `--conversation <id>`로 재개할 수 있습니다.

### 병렬 인스턴스(하위 이슈 / worktree 분산)

작업마다 하나의 git worktree를 만들고 각 worktree에서 독립적인 `agy -p`를 백그라운드로 실행한 뒤 결과를 수집하세요. 동시성은 장비 성능과 검토 가능 범위에 맞게 제한하세요.

### 출력 + 실행 제한 주의 사항(Claude Code와 다름)

- `agy -p`는 일반 텍스트를 반환합니다 — `--output-format json`이 없고 `session_id` / 비용 / 턴 수가 담긴 결과 래퍼도 없습니다. stdout을 직접 분석하고 JSON 객체를 예상하지 마세요.
- `--max-turns`가 없습니다. print 실행은 `--print-timeout`(기본값 `5m`)으로 제한됩니다. 장시간 작업에는 값을 높이세요: `--print-timeout 20m`. 바깥 `terminal` `timeout=`과 함께 사용해 외부 호출이 너무 일찍 종료되지 않게 하세요.

### 오케스트레이션 경계

Antigravity는 작업을 실행하는 백엔드 또는 제3자 의견 검토자입니다 — 작업을 실행하는 에이전트/프로필이 소유하는 실행 세부 사항이지, 일급 오케스트레이션 프리미티브가 아닙니다. `agy`를 칸반 보드에 자체 카드로 올리거나 조정 계층으로 취급하지 마세요. 일반 작업 그래프를 통해 위임하고, 할당된 작업자가 방법(codex/claude-code/직접 도구 중 하나)으로 `agy`를 선택하게 하세요. 사용자가 명시적으로 요청했거나, 작업자가 이를 감싸도록 구성되었거나, 다른 에이전트의 계획 또는 diff를 Gemini 계열로 교차 확인하려는 경우에만 명시적으로 사용하세요.

## 핵심 경로

- 바이너리 / 진입점: `agy`
- 앱 데이터 디렉터리: `~/.gemini/antigravity-cli/`
- 설정 파일: `~/.gemini/antigravity-cli/settings.json`
- 키 바인딩 파일: `~/.gemini/antigravity-cli/keybindings.json`
- 로그: `~/.gemini/antigravity-cli/log/cli-*.log`
- 대화: `~/.gemini/antigravity-cli/conversations/`
- 브레인 아티팩트: `~/.gemini/antigravity-cli/brain/`
- 기록: `~/.gemini/antigravity-cli/history.jsonl`
- 플러그인 스테이징: `~/.gemini/antigravity-cli/plugins/<plugin_name>/`

## 빠른 참조

### 래퍼 명령
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

### 플러그인 하위 명령(`agy plugin --help`)
- `list`, `import [source]`, `install <target>`, `uninstall <name>`,
  `enable <name>`, `disable <name>`, `validate [path]`, `link <mp> <target>`,
  `help`

### 설치 플래그(`agy install --help`)
- `--dir`, `--skip-aliases`, `--skip-path`

### 세션 내 슬래시 명령
- **대화 제어:** `/resume` (`/switch`), `/rewind` (`/undo`),
  `/rename <name>`, `/clear`, `/fork`, `/reset`, `/new`
- **설정 및 도구:** `/config`, `/settings`, `/permissions`, `/model`,
  `/keybindings`, `/statusline`, `/tasks`, `/skills`, `/mcp`, `/open <path>`,
  `/usage`, `/logout`, `/agents`
- **프롬프트 도우미:** `@` 경로 자동 완성, `esc esc`는 스트리밍 중이 아닐 때 프롬프트를 지움, `!`는 터미널 명령을 직접 실행, `?`는 도움말을 엶

## 설정 및 권한

### 일반 설정 키(`settings.json`)
- `allowNonWorkspaceAccess`
- `colorScheme`
- `permissions.allow`
- `trustedWorkspaces`

### 권한 모드
`request-review`, `always-proceed`, `strict`, `proceed-in-sandbox`.

### 샌드박스 동작
- `enableTerminalSandbox`는 `settings.json`의 불리언 값이며 기본값은 `false`입니다.
- 실행 시 재정의(`--sandbox`, `--dangerously-skip-permissions`)는 현재 세션의 영구 설정을 덮어쓸 수 있습니다.

## 인증 동작

- CLI는 먼저 OS 보안 키링을 시도합니다.
- 저장된 세션이 없으면 브라우저 기반 Google 로그인으로 대체합니다.
- 로컬에서는 기본 브라우저를 열고, SSH 환경에서는 인증 URL을 출력한 뒤 인증 코드를 붙여 넣도록 요청합니다.
- `/logout`은 저장된 자격 증명을 삭제합니다.

## 플러그인

- 플러그인은 `~/.gemini/antigravity-cli/plugins/<plugin_name>/` 아래에 스테이징됩니다.
- 스킬, 에이전트, 규칙, MCP 서버 및 훅을 포함할 수 있습니다.
- `agy plugin list`가 가져온 플러그인을 표시하지 않는 것은 유효한 빈 상태입니다.

## 주의 사항

- `agy help`는 대화형 슬래시 명령이 아니라 래퍼 명령을 보여줍니다.
- `agy --version`은 안전한 비대화형 버전 확인 명령입니다. `agy version`은 대화형이며 실제 TTY가 없으면 실패할 수 있습니다.
- 장애가 발생했을 때 가장 먼저 확인할 곳은 `~/.gemini/antigravity-cli/log/cli-*.log`입니다(`read_file`로 읽으세요).
- 영구 JSON 설정과 실행 시 재정의를 혼동하지 마세요.
- `~/.gemini/antigravity-cli/bin/agentapi`는 `agy agentapi`를 호출하는 얇은 래퍼입니다.
- WSL에서는 토큰 저장이 파일 기반이므로 인증 문제는 대개 브라우저만의 문제가 아니라 로컬 파일 / 세션 상태 문제입니다.
- 작업 공간 식별자는 실행 디렉터리와 `.antigravitycli` 프로젝트 마커에 따라 달라질 수 있습니다.
- `agy -p`는 일반 텍스트만 출력합니다 — `--output-format json`이나 결과 래퍼가 없습니다. JSON 객체를 추출하려 하지 마세요(`claude-code`와 다릅니다).
- print 실행은 `--max-turns`(agy에는 존재하지 않음)가 아니라 `--print-timeout`(기본값 `5m`)으로 제한하세요.

## 검증

`terminal` 도구로 모두 수행하여 설치가 실제로 완료되고 사용할 수 있는지 확인하세요(파일은 `read_file`로 읽습니다).

1. `terminal(command="command -v agy")`
2. `terminal(command="agy --version")`
3. `terminal(command="agy help")`
4. `terminal(command="agy plugin list")`
5. `~/.gemini/antigravity-cli/settings.json`에 `read_file` 사용
6. 최신 `~/.gemini/antigravity-cli/log/cli-*.log`에 `read_file` 사용
7. 필요하면 `~/.gemini/antigravity-cli/keybindings.json`에 `read_file` 사용

## 지원 파일

- `references/cli-docs.md` — 시작하기, 사용법 및 기능 문서의 요약 참고 자료.

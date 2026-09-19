---
title: "Grok — xAI Grok Build CLI에 코딩 위임 (기능, PR)"
sidebar_label: "Grok"
description: "xAI Grok Build CLI에 코딩 위임 (기능, PR)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py가 스킬의 SKILL.md에서 자동 생성합니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Grok

기능과 PR을 위해 코딩을 xAI Grok Build CLI에 위임합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/grok`으로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/grok` |
| 버전 | `0.1.1` |
| 작성자 | Matt Maximo (MattMaximo), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Grok`, `xAI`, `Code-Review`, `Refactoring`, `Automation` |
| 관련 스킬 | [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Grok Build CLI — Hermes 오케스트레이션 가이드

Hermes 터미널을 통해 [Grok Build](https://docs.x.ai/build/overview)(xAI의
자율 코딩 에이전트 CLI인 `grok` 명령)에 코딩 작업을 위임합니다. Grok은 파일을
읽고, 코드를 작성하고, 셸 명령을 실행하고, 하위 에이전트를 생성하고, git
워크플로를 관리할 수 있습니다. 세 가지 방식으로 실행됩니다. 대화형 TUI,
**헤드리스**(`-p`), 그리고 JSON-RPC를 통한 **ACP 에이전트**입니다.

이는 `codex`와 `claude-code`에 이은 세 번째 형제입니다. 오케스트레이션
패턴은 거의 동일합니다 — 일회성 작업에는 **헤드리스 `-p`를 우선 사용**하고,
대화형 세션에는 PTY를 사용하세요.

## 사용 시점

- 기능 구현
- 리팩터링
- PR 리뷰
- 일괄 이슈 수정
- 그 외 Codex / Claude Code를 사용하고 싶지만 Grok을 선택하는 모든 작업

## 사전 요구 사항

- **설치(권장):** `npm install -g @xai-official/grok`
  - 공식 설치 프로그램 `curl -fsSL https://x.ai/cli/install.sh | bash`도
    작동하지만, 일부 환경에서는 `x.ai` 호스트가 Cloudflare에 의해 차단됩니다. npm
    경로를 사용하면 이 의존성을 완전히 피할 수 있습니다.
- **인증 — SuperGrok / X Premium+ 구독(기본 경로):**
  - `grok login`을 한 번 실행하면 브라우저에서 OAuth가 열리고 토큰이
    `~/.grok/auth.json`에 캐시됩니다. 이 방식은 **SuperGrok 또는 X Premium+**
    구독을 사용합니다(토큰별 API 과금 없음).
  - `~/.grok/auth.json`을 확인하거나, 저렴한 헤드리스 스모크 테스트를 실행하여
    로그인 상태를 확인하세요: `grok --no-auto-update -p "Say ok."`
  - TUI에서는 `/logout`으로 로그아웃하고 `/login`(또는 재실행)으로 다시 로그인합니다.
- **git 저장소 불필요** — Codex와 달리 Grok은 git 디렉터리 외부에서도 정상적으로
  실행됩니다(스크래치/일회성 작업에 유용).
- **Claude Code / AGENTS.md와 설정 없이 호환** — Grok은 `CLAUDE.md`, `.claude/`
  (스킬, 에이전트, MCP, 훅, 규칙), 그리고 `AGENTS.md` 계열을 자동으로 읽습니다.
  기존 프로젝트 컨텍스트가 그대로 적용됩니다.

> **API 키 대체 경로(이 사용자에게는 기본값 아님):** Grok은 `api.x.ai`를 통한
> 종량제 결제를 위해 `XAI_API_KEY` 환경 변수 설정도 지원합니다. `grok login` /
> SuperGrok 인증을 사용할 수 없을 때만 사용하세요. 여기서는 구독 경로
> (`grok login`)가 의도된 설정입니다.

## 두 가지 오케스트레이션 모드

### 모드 1: 헤드리스(`-p`) — 비대화형(권장)

일회성 작업을 실행하고 결과를 출력한 뒤 종료합니다. PTY가 필요 없고 탐색해야
하는 대화형 대화상자도 없습니다. 가장 깔끔한 통합 경로이며 `claude -p` 및
`codex exec`와 같은 방식입니다.

```
terminal(command="grok --no-auto-update -p 'Add a dark mode toggle to settings'", workdir="/path/to/project", timeout=180)
```

자동화에서는 백그라운드 업데이트 확인을 건너뛰도록 항상 `--no-auto-update`를
전달하세요.

**헤드리스를 사용할 때:**
- 일회성 코딩 작업(버그 수정, 기능 추가, 리팩터링)
- CI/CD 자동화 및 스크립팅
- `--output-format json`을 사용한 구조화된 출력 파싱
- 다중 턴 대화가 필요하지 않은 모든 작업

### 모드 2: 대화형 PTY — 다중 턴 TUI 세션

TUI는 전체 화면 마우스 대화형 앱입니다. `pty=true`로 구동하세요. 안정적인
모니터링/입력에는 tmux를 사용합니다(`claude-code` 스킬과 동일한 패턴).

```
# Launch in a tmux session for capture-pane monitoring
terminal(command="tmux new-session -d -s grok-work -x 140 -y 40")
terminal(command="tmux send-keys -t grok-work 'cd /path/to/project && grok' Enter")

# Wait for startup, then send a task
terminal(command="sleep 5 && tmux send-keys -t grok-work 'Refactor the auth module to use JWT' Enter")

# Monitor progress
terminal(command="sleep 15 && tmux capture-pane -t grok-work -p -S -50")

# Exit when done
terminal(command="tmux send-keys -t grok-work '/quit' Enter && sleep 1 && tmux kill-session -t grok-work")
```

**인라인 헤드리스 출력 팁:** 전체 화면 대체 화면 전환 없이 TUI 스타일 출력을
보려면 `--no-alt-screen`을 추가하세요. 순수 자동화에는 여전히 헤드리스 `-p`가
더 깔끔합니다.

## 헤드리스 상세

### 주요 플래그

| 플래그 | 효과 |
|--------|------|
| `-p, --single <PROMPT>` | 단일 프롬프트를 보내고 헤드리스로 실행한 후 종료 |
| `-m, --model <MODEL>` | 모델 선택 |
| `-s, --session-id <UUID>` | 새 세션에 할당할 **새로운** 유효한 UUID(기존 세션이면 안 됨). 재개하지 않으며, 이를 위해서는 `--resume`/`--continue`를 사용합니다. `--resume`/`--continue`와 함께 사용할 때만 `--fork-session`과 같이 사용할 수 있습니다 |
| `-r, --resume [<UUID>]` | UUID로 기존 세션 재개(생략하면 가장 최근 세션) |
| `-c, --continue` | 현재 디렉터리에서 가장 최근 세션 계속하기 |
| `--fork-session` | 재개할 때 원래 세션 ID 대신 새 세션 ID 생성 |
| `--max-turns <N>` | 에이전트 턴 최대 개수 제한 |
| `--cwd <PATH>` | 작업 디렉터리 설정 |
| `--output-format <FMT>` | `plain`(기본값), `json`, 또는 `streaming-json` |
| `--always-approve` | 모든 도구 실행 자동 승인(`--full-auto` / `--yolo`와 동일) |
| `--no-alt-screen` | 인라인 실행, 전체 화면 TUI 전환 없음 |
| `--no-auto-update` | 백그라운드 업데이트 확인 생략(`--help`에는 숨겨져 있지만 여전히 작동하며 모든 자동화에서 사용) |

### 출력 형식

- `plain` — 사람이 읽을 수 있는 텍스트(기본값)
- `json` — 실행 종료 시 하나의 JSON 객체
- `streaming-json` — 도착하는 이벤트를 줄바꿈으로 구분한 JSON

```
# Structured result for parsing
terminal(command="grok --no-auto-update -p 'List all TODO comments in src/' --output-format json", workdir="/project", timeout=120)

# Auto-approve for autonomous building
terminal(command="grok --no-auto-update --always-approve -p 'Refactor the database layer and run the tests'", workdir="/project", timeout=300)
```

### 백그라운드 모드(장시간 작업)

```
# Start headless in background
terminal(command="grok --no-auto-update --always-approve -p 'Refactor the auth module'", workdir="/project", background=true, notify_on_complete=true)
# Returns session_id

# Monitor
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Kill if needed
process(action="kill", session_id="<id>")
```

대화형(TUI) 백그라운드 세션에는 `pty=true`와 tmux를 사용하고, `claude-code` /
`codex` 스킬과 정확히 같은 방식으로 `tmux capture-pane`을 모니터링하세요.

### 세션 계속하기

세션은 이름이 아니라 UUID를 기준으로 합니다. `--session-id`는 새로 시작하는
실행에 새 UUID를 할당하며(재개하지 않음), `--resume`은 기존 세션의 UUID를
사용합니다(생략하면 이 디렉터리에서 가장 최근 세션을 재개).

```
# Start a session with a self-assigned UUID (must be a valid, unused UUID)
SID=$(uuidgen)
terminal(command="grok --no-auto-update -s $SID -p 'Start refactoring the database layer' --always-approve", workdir="/project", timeout=240)

# Resume that exact session later by its UUID
terminal(command="grok --no-auto-update -r $SID -p 'Now add connection pooling' --always-approve", workdir="/project", timeout=180)

# Or just continue the most recent session in this directory (no UUID needed)
terminal(command="grok --no-auto-update -c -p 'What did you change last time?'", workdir="/project", timeout=60)
```

## 읽기 전용 감사 → Markdown 메모 패턴

Grok이 로컬 아티팩트를 검토하고 변경 없이 Obsidian 또는 저장소용 깔끔한
Markdown 메모를 반환하게 하려면 다음과 같이 하세요.

1. 먼저 Hermes 도구(`read_file`, `write_file`)로 안정적인 입력 파일을 준비합니다. 원본 경로를 그대로 출력하는 대신 관련 컨텍스트만 임시 파일에 스냅샷합니다.
2. `--always-approve` 없이 Grok을 헤드리스로 실행하여 자동 쓰기를 할 수 없게 하고, `markdown only, no preamble`을 요구합니다.
3. Grok의 표준 출력을 `write_file()`로 대상 메모에 바로 저장합니다.

```
grok --no-auto-update -p "Read /tmp/current.md and /tmp/inventory.md. Produce markdown only, no preamble. Output a clean note titled 'Cleanup Review'." --output-format plain
```

**주의 사항(Claude Code와 동일):** 문서 재작성에서 단순히 "rewrite this"라고
요청하면 전체 파일 대신 변경 요약을 반환할 수 있습니다. 대신 파일을 입력으로
전달하고 `Return ONLY the full revised markdown document. No intro, no explanation,
no code fences. Start immediately with '# Title'.`을 요구하세요. 덮어쓰기 전에
`read_file()`로 첫 줄을 확인합니다.

## PR 리뷰 패턴

### 빠른 리뷰(헤드리스)

```
terminal(command="cd /path/to/repo && git diff main...feature-branch | grok --no-auto-update -p 'Review this diff for bugs, security issues, and style problems. Be thorough.'", timeout=120)
```

### 임시 디렉터리에 복제하여 리뷰(안전, 저장소 변경 없음)

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && grok --no-auto-update -p 'Review the changes vs origin/main. Check bugs, security, race conditions, missing tests.'", pty=true, timeout=300)
```

### 리뷰 게시

```
terminal(command="gh pr comment 42 --body '<review text>'", workdir="/path/to/repo")
```

## 작업 트리를 사용한 병렬 이슈 수정

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Grok headless in each (background)
terminal(command="grok --no-auto-update --always-approve -p 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, notify_on_complete=true)
terminal(command="grok --no-auto-update --always-approve -p 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, notify_on_complete=true)

# Monitor
process(action="list")

# After completion: push and open PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## 유용한 하위 명령 및 TUI 명령

| 명령 | 용도 |
|---------|---------|
| `grok` | 대화형 TUI 시작 |
| `grok -p "query"` | 헤드리스 일회성 실행 |
| `grok login` / `grok logout` | 로그인 / 로그아웃(SuperGrok / X Premium+ OAuth) |
| `grok inspect` | cwd에서 Grok이 발견한 항목 표시: 구성 소스, 지침, 스킬, 플러그인, 훅, MCP 서버 |
| `grok agent stdio` | JSON-RPC를 통한 ACP 에이전트 실행(IDE/도구 통합용) |
| `grok update` | CLI 업데이트(`x.ai` 호스트 필요, 자동화에서는 생략) |

TUI 슬래시 명령(대화형 전용): `/model <name>`, `/always-approve`,
`/plan`, `/context`, `/compact`, `/resume`, `/sessions`, `/fork`, `/usage`,
`/quit`. `Shift+Tab`으로 세션 모드를 순환합니다(쓰기 도구가 세션 계획 파일을
제외하고 차단되는 Plan mode 포함).

## 구성(`~/.grok/config.toml`)

```toml
[cli]
auto_update = false          # skip background update checks persistently

[ui]
permission_mode = "ask"      # or "always-approve" to skip tool prompts by default

[models]
default = "grok-build-0.1"
```

전역 설정은 프로젝트 범위의 `.grok/config.toml`이 아닌
`~/.grok/config.toml`에 저장하세요. `permission_mode`가 기존의
`approval_mode` / `yolo = true` 키를 대체합니다.

## 주의 사항 및 함정

1. **인증은 구독으로 제한됩니다.** `grok login`에는 SuperGrok 또는 X Premium+ 구독이 필요합니다. 로그인에 실패하거나 `~/.grok/auth.json`이 없으면 `XAI_API_KEY`로 대체하기 전에 구독이 활성 상태인지 확인하세요.
2. **Hermes의 xAI 인증과 `grok` CLI의 인증을 혼동하지 마세요.** Hermes의 `x_search`는 자체 xAI OAuth로 실행되고, 독립 실행형 `grok` CLI는 `~/.grok/auth.json`에 별도 토큰을 저장합니다. `x_search`가 작동한다고 해서 `grok`이 로그인된 것은 **아닙니다**.
3. **자동화에서는 항상 `--no-auto-update`를 전달하세요** — 그렇지 않으면 Grok이 업데이트 확인을 위해 외부에 연결하며(`x.ai`/`storage.googleapis.com`에 접근하지 못할 수 있음) 문제가 발생할 수 있습니다.
4. **curl 설치 프로그램보다 npm 설치를 우선하세요** — `npm install -g
   @xai-official/grok`은 Cloudflare에 의해 차단된 `x.ai` 호스트를 피합니다.
5. **`--always-approve`는 자율 빌드 스위치입니다.** 이 옵션이 없으면 헤드리스 실행이 도구 승인 프롬프트를 기다리느라 멈출 수 있습니다. 읽기 전용 리뷰/감사 작업에서는 Grok이 파일을 변경하지 못하도록 의도적으로 생략하세요.
6. **헤드리스 `-p`는 TUI 대화상자를 건너뜁니다**. TUI에는 Claude Code와 마찬가지로 `pty=true`(모니터링에는 tmux 추가)가 필요합니다.
7. 인라인으로 TUI를 실행하고 전체 화면 대체 화면 전환 때문에 캡처된 출력이 깨진다면 `--no-alt-screen`을 사용하세요.
8. **git 저장소는 필요하지 않지만**, PR/커밋 워크플로에서는 저장소가 있는 편이 좋습니다 — 스크래치 커밋 작업에는 `mktemp -d && git init`을 사용하세요.
9. 작업이 끝나면 `tmux kill-session -t <name>`으로 tmux 세션을 정리하세요.

## Hermes 에이전트를 위한 규칙

1. **단일 작업에는 헤드리스 `-p`를 우선 사용** — `--output-format json`을 통한 가장 깔끔한 통합 및 구조화된 출력.
2. Grok이 올바른 프로젝트를 대상으로 하도록 **항상 `workdir`**(또는 `--cwd`)을 설정하세요.
3. 모든 자동 호출에 `--no-auto-update`를 전달하세요.
4. Grok이 자율적으로 작성해야 할 때만 `--always-approve`를 사용하고, 읽기 전용 리뷰와 감사에서는 생략하세요.
5. 장시간 작업은 `background=true, notify_on_complete=true`로 백그라운드 실행하고 `process` 도구로 모니터링하세요.
6. 다중 턴 대화형 작업에는 tmux를 사용하고 `tmux capture-pane -t <session> -p -S -50`으로 모니터링하세요.
7. **사용하기 전에 인증을 확인하세요** — `~/.grok/auth.json`을 확인하거나 저렴한 `grok -p "Say ok."` 스모크 테스트를 실행하세요. Hermes의 xAI 인증이 이어진다고 가정하지 마세요.
8. **결과를 사용자에게 보고하세요** — Grok이 변경한 내용과 남은 작업을 요약하세요.

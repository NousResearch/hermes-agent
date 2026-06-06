---
title: "Grok — 코딩 작업을 xAI Grok Build CLI에 위임 (기능, PR 등)"
sidebar_label: "Grok"
description: "코딩 작업을 xAI Grok Build CLI에 위임 (기능, PR 등)"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Grok

코딩 작업을 xAI Grok Build CLI(기능 개발, PR 리뷰 등)에 위임합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/grok` 명령어로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/grok` |
| 버전 | `0.1.0` |
| 작성자 | Matt Maximo (MattMaximo), Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Grok`, `xAI`, `Code-Review`, `Refactoring`, `Automation` |
| 관련 스킬 | [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Grok Build CLI — Hermes 오케스트레이션 가이드

Hermes 터미널을 통해 코딩 작업을 [Grok Build](https://docs.x.ai/build/overview) (xAI의 자율 코딩 에이전트 CLI, `grok` 명령어)에 위임합니다. Grok은 파일을 읽고, 코드를 작성하고, 셸 명령어를 실행하고, 하위 에이전트를 생성하고, git 워크플로우를 관리할 수 있습니다. Grok은 대화형 TUI, **헤드리스** (`-p`), 그리고 JSON-RPC 기반의 **ACP 에이전트**라는 세 가지 방식으로 실행됩니다.

이것은 `codex`와 `claude-code`에 이은 세 번째 형제 스킬입니다. 오케스트레이션 패턴은 거의 동일합니다 — **원샷(one-shot) 작업에는 헤드리스 `-p`를 선호**하고, 대화형 세션에는 PTY를 사용하세요.

## 사용 시기

- 기능 개발 (Building features)
- 리팩토링 (Refactoring)
- PR 리뷰 (PR reviews)
- 일괄 이슈 수정 (Batch issue fixing)
- 다른 상황이라면 Codex / Claude Code를 사용했겠지만 Grok을 원할 때

## 전제 조건

- **설치 (권장):** `npm install -g @xai-official/grok`
  - 공식 설치 프로그램인 `curl -fsSL https://x.ai/cli/install.sh | bash`도 작동하지만, `x.ai` 호스트는 일부 환경에서 Cloudflare 방화벽에 의해 차단될 수 있습니다. npm 경로는 해당 종속성을 완전히 피합니다.
- **인증 — SuperGrok 또는 X Premium+ 구독 (기본 경로):**
  - `grok login`을 한 번 실행합니다 → OAuth를 위한 브라우저가 열립니다 → 토큰이 `~/.grok/auth.json`에 캐시됩니다. 이것은 **SuperGrok 또는 X Premium+** 구독을 사용합니다(토큰별 API 과금 없음).
  - `~/.grok/auth.json`을 찾아서 로그인 상태를 확인하거나 비용이 적게 드는 헤드리스 스모크 테스트를 실행하세요: `grok --no-auto-update -p "Say ok."`
  - TUI에서 `/logout`은 로그아웃을 수행하고 `/login` (또는 재실행)은 다시 로그인합니다.
- **git 저장소가 필요하지 않음** — Codex와 달리 Grok은 git 디렉토리 외부에서도 잘 실행됩니다 (스크래치/일회성 작업에 적합).
- **설정 없이 Claude Code / AGENTS.md와 호환 가능** — Grok은 자동으로 `CLAUDE.md`, `.claude/` (스킬, 에이전트, MCP, 후크, 규칙), 그리고 `AGENTS.md` 계열을 읽습니다. 기존 프로젝트 컨텍스트가 그대로 작동합니다.

> **API 키 예비 수단 (이 사용자의 기본값이 아님):** Grok은 또한 사용한 만큼 비용을 지불하는 종량제(pay-as-you-go) 결제를 위해 `api.x.ai`를 통한 `XAI_API_KEY` 환경 변수 설정을 지원합니다.
> 이 기능은 `grok login` / SuperGrok 인증을 사용할 수 없을 때만 사용하세요. 여기서는 구독 방식(`grok login`)이 의도된 설정입니다.

## 두 가지 오케스트레이션 모드

### 모드 1: 헤드리스 (`-p`) — 비대화형 (권장)

원샷 작업을 실행하고, 결과를 출력한 후 종료합니다. PTY나 탐색해야 할 대화형 다이얼로그가 없습니다. 이것이 가장 깔끔한 통합 경로입니다 — `claude -p` 및 `codex exec`와 유사합니다.

```
terminal(command="grok --no-auto-update -p 'Add a dark mode toggle to settings'", workdir="/path/to/project", timeout=180)
```

백그라운드 업데이트 확인을 건너뛰려면 자동화 시 항상 `--no-auto-update`를 전달하세요.

**헤드리스 모드 사용 시기:**
- 원샷 코딩 작업 (버그 수정, 기능 추가, 리팩토링)
- CI/CD 자동화 및 스크립팅
- `--output-format json`을 사용한 구조화된 출력 파싱
- 다중 턴(multi-turn) 대화가 필요하지 않은 모든 작업

### 모드 2: 대화형 PTY — 다중 턴 TUI 세션

TUI는 전체 화면 마우스 대화형 앱입니다. `pty=true`로 구동하세요. 강력한 모니터링/입력을 위해 tmux를 사용하세요 (`claude-code` 스킬과 동일한 패턴).

```
# 캡처 페인(capture-pane) 모니터링을 위해 tmux 세션에서 실행
terminal(command="tmux new-session -d -s grok-work -x 140 -y 40")
terminal(command="tmux send-keys -t grok-work 'cd /path/to/project && grok' Enter")

# 시작을 기다린 후 작업 전송
terminal(command="sleep 5 && tmux send-keys -t grok-work 'Refactor the auth module to use JWT' Enter")

# 진행 상황 모니터링
terminal(command="sleep 15 && tmux capture-pane -t grok-work -p -S -50")

# 완료 시 종료
terminal(command="tmux send-keys -t grok-work '/quit' Enter && sleep 1 && tmux kill-session -t grok-work")
```

**헤드리스지만 인라인 출력을 원할 때 팁:** 전체 화면 교체(alt-screen takeover) 없이 TUI 스타일의 출력을 원한다면 (예: 더 깨끗한 로그를 위해) `--no-alt-screen`을 추가하세요. 하지만 순수한 자동화를 위해서는 TUI보다 헤드리스 `-p`가 여전히 더 깔끔합니다.

## 헤드리스 심층 분석

### 일반 플래그

| 플래그 | 효과 |
|------|--------|
| `-p, --single <PROMPT>` | 하나의 프롬프트를 보내고, 헤드리스로 실행한 후 종료 |
| `-m, --model <MODEL>` | 모델 선택 |
| `-s, --session-id <ID>` | 이름이 지정된 헤드리스 세션 생성 또는 재개 |
| `-r, --resume <ID>` | 기존 세션 재개 |
| `-c, --continue` | 현재 디렉토리에서 가장 최근 세션 계속하기 |
| `--cwd <PATH>` | 작업 디렉토리 설정 |
| `--output-format <FMT>` | `plain` (기본값), `json`, 또는 `streaming-json` |
| `--always-approve` | 모든 도구 실행을 자동 승인 (`--full-auto` / `--yolo`와 동일) |
| `--no-alt-screen` | 전체 화면 TUI 교체 없이 인라인으로 실행 |
| `--no-auto-update` | 백그라운드 업데이트 확인 건너뛰기 (모든 자동화에 사용) |

### 출력 형식

- `plain` — 사람이 읽을 수 있는 텍스트 (기본값)
- `json` — 실행이 끝날 때 하나의 JSON 객체 반환 (결과를 깔끔하게 파싱)
- `streaming-json` — JSON 이벤트가 도착하는 대로 줄 바꿈으로 구분하여 반환

```
# 파싱을 위한 구조화된 결과
terminal(command="grok --no-auto-update -p 'List all TODO comments in src/' --output-format json", workdir="/project", timeout=120)

# 자율 구축을 위한 자동 승인
terminal(command="grok --no-auto-update --always-approve -p 'Refactor the database layer and run the tests'", workdir="/project", timeout=300)
```

### 백그라운드 모드 (긴 작업)

```
# 백그라운드에서 헤드리스로 시작
terminal(command="grok --no-auto-update --always-approve -p 'Refactor the auth module'", workdir="/project", background=true, notify_on_complete=true)
# session_id 반환

# 모니터링
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# 필요 시 강제 종료
process(action="kill", session_id="<id>")
```

대화형(TUI) 백그라운드 세션의 경우 `pty=true` + tmux를 사용하고 `claude-code` / `codex` 스킬과 동일하게 `tmux capture-pane`으로 모니터링합니다.

### 세션 이어하기

```
# 이름이 지정된 세션 시작
terminal(command="grok --no-auto-update -s refactor-db -p 'Start refactoring the database layer' --always-approve", workdir="/project", timeout=240)

# 나중에 재개
terminal(command="grok --no-auto-update -r refactor-db -p 'Now add connection pooling' --always-approve", workdir="/project", timeout=180)

# 또는 이 디렉토리에서 가장 최근 세션 계속하기
terminal(command="grok --no-auto-update -c -p 'What did you change last time?'", workdir="/project", timeout=60)
```

## 읽기 전용 감사 → 마크다운 노트 패턴

아무것도 변경하지 않고 Grok이 로컬 아티팩트를 검토하고 (Obsidian이나 저장소용으로) 깔끔한 마크다운 노트를 반환하도록 하려면:

1. 먼저 Hermes 도구(`read_file`, `write_file`)로 안정적인 입력 파일을 준비합니다. 원시 경로를 전부 던져주는 것보다, 관련 컨텍스트만 임시 파일에 스냅샷으로 저장하세요.
2. Grok이 파일을 마음대로 작성할 수 없도록 `--always-approve`를 **제외하고** 헤드리스로 실행하며, `markdown only, no preamble` (서문 없이 마크다운만)을 요구합니다.
3. `write_file()`을 사용하여 Grok의 stdout을 대상 노트에 직접 저장합니다.

```
grok --no-auto-update -p "Read /tmp/current.md and /tmp/inventory.md. Produce markdown only, no preamble. Output a clean note titled 'Cleanup Review'." --output-format plain
```

**주의 사항 (Claude Code와 동일):** 문서 재작성의 경우, 단순히 "이걸 다시 써줘"라고 요청하면 전체 파일 대신 변경 요약이 반환될 수 있습니다. 대신: 파이프로 파일을 입력하고, `Return ONLY the full revised markdown document. No intro, no explanation, no code fences. Start immediately with '# Title'.` (수정된 전체 마크다운 문서만 반환하세요. 서론, 설명, 코드 펜스 없이 즉시 '# Title'로 시작하세요)라고 명확히 요구하세요. 대상 파일을 덮어쓰기 전에 `read_file()`로 처음 몇 줄을 확인하세요.

## PR 리뷰 패턴

### 빠른 리뷰 (헤드리스)

```
terminal(command="cd /path/to/repo && git diff main...feature-branch | grok --no-auto-update -p 'Review this diff for bugs, security issues, and style problems. Be thorough.'", timeout=120)
```

### 임시 저장소 복제 리뷰 (안전함, 저장소 변형 없음)

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && grok --no-auto-update -p 'Review the changes vs origin/main. Check bugs, security, race conditions, missing tests.'", pty=true, timeout=300)
```

### 리뷰 게시

```
terminal(command="gh pr comment 42 --body '<review text>'", workdir="/path/to/repo")
```

## 워크트리(Worktrees)를 이용한 병렬 이슈 수정

```
# 워크트리 생성
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# 각 워크트리에서 헤드리스 Grok 실행 (백그라운드)
terminal(command="grok --no-auto-update --always-approve -p 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, notify_on_complete=true)
terminal(command="grok --no-auto-update --always-approve -p 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, notify_on_complete=true)

# 모니터링
process(action="list")

# 완료 후: 푸시 및 PR 열기
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# 정리
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## 유용한 하위 명령어 및 TUI 명령어

| 명령어 | 목적 |
|---------|---------|
| `grok` | 대화형 TUI 시작 |
| `grok -p "query"` | 헤드리스 원샷 실행 |
| `grok login` / `grok logout` | 로그인 / 로그아웃 (SuperGrok / X Premium+ OAuth) |
| `grok inspect` | Grok이 현재 작업 디렉토리(cwd)에서 발견한 사항을 보여줌: 구성 소스, 지시사항, 스킬, 플러그인, 후크, MCP 서버 |
| `grok agent stdio` | JSON-RPC 기반 ACP 에이전트로 실행 (IDE/도구 통합용) |
| `grok update` | CLI 업데이트 (`x.ai` 호스트 필요; 자동화 시 건너뛰기) |

TUI 슬래시 명령어 (대화형 전용): `/model <name>`, `/always-approve`, `/plan`, `/context`, `/compact`, `/resume`, `/sessions`, `/fork`, `/usage`, `/quit`. `Shift+Tab`은 세션 모드(계획(Plan) 모드 포함)를 전환합니다 (계획 모드는 세션 계획 파일 이외의 쓰기 도구를 차단합니다).

## 설정 (`~/.grok/config.toml`)

```toml
[cli]
auto_update = false          # 백그라운드 업데이트 확인 영구 건너뛰기

[ui]
permission_mode = "ask"      # 또는 "always-approve" (기본적으로 도구 확인 프롬프트를 건너뛰기)

[models]
default = "grok-build-0.1"
```

전역 환경 설정은 `~/.grok/config.toml`에 지정하세요 (프로젝트 범위의 `.grok/config.toml`이 아님). `permission_mode`는 레거시의 `approval_mode` / `yolo = true` 키를 대체합니다.

## 주의 사항 및 팁 (Gotchas)

1. **인증은 구독 기반입니다.** `grok login`에는 SuperGrok 또는 X Premium+ 구독이 필요합니다. 로그인이 실패하거나 `~/.grok/auth.json`이 없는 경우 `XAI_API_KEY` 예비 수단을 사용하기 전에 구독이 활성화되어 있는지 확인하세요.
2. **Hermes의 xAI 인증과 `grok` CLI 인증을 혼동하지 마세요.** Hermes의 `x_search`는 자체 xAI OAuth로 실행됩니다. 반면 독립 실행형 `grok` CLI는 `~/.grok/auth.json`에 별도의 토큰을 가지고 있습니다. `x_search`가 작동한다고 해서 `grok`이 로그인되어 있다는 의미는 아닙니다.
3. **자동화 시 항상 `--no-auto-update`를 전달하세요.** 그렇지 않으면 Grok이 업데이트를 확인하려고 시도하며 (`x.ai`/`storage.googleapis.com`에 접근하지 못할 수 있습니다).
4. **curl 설치 프로그램보다 npm 설치를 선호하세요.** `npm install -g @xai-official/grok`은 Cloudflare 방화벽이 있는 `x.ai` 호스트를 우회합니다.
5. **`--always-approve`는 자율 빌드(autonomous-build) 스위치입니다.** 이것이 없으면 헤드리스 실행은 도구 승인 프롬프트를 기다리며 멈출 수 있습니다. 읽기 전용 리뷰/감사 작업의 경우 Grok이 파일을 변경할 수 없도록 이 스위치를 의도적으로 생략하세요.
6. **헤드리스 `-p`는 TUI 다이얼로그를 건너뜁니다.** Claude Code와 마찬가지로 TUI에는 모니터링을 위한 `pty=true` (+ tmux)가 필요합니다.
7. **`--no-alt-screen` 사용:** TUI를 인라인으로 실행하는데 전체 화면 교체가 캡처된 출력을 알아볼 수 없게 만드는 경우 사용하세요.
8. **git 저장소가 필요하지 않습니다.** 하지만 PR/커밋 워크플로우를 위해서는 여전히 필요합니다 — 스크래치 커밋 작업에는 `mktemp -d && git init`을 사용하세요.
9. 완료 시 `tmux kill-session -t <name>`으로 **tmux 세션을 정리하세요.**

## Hermes 에이전트를 위한 규칙

1. 단일 작업에는 **헤드리스 `-p` 모드를 선호**하세요. 가장 깔끔한 통합 방식이며 `--output-format json`을 통해 구조화된 출력을 제공합니다.
2. Grok이 올바른 프로젝트를 대상으로 하도록 **항상 `workdir` (또는 `--cwd`)를 설정하세요.**
3. 모든 자동화된 호출에는 **`--no-auto-update`를 전달하세요.**
4. **Grok이 자율적으로 작성해야 할 때만 `--always-approve`를 사용하세요**. 읽기 전용 리뷰 및 감사 시에는 생략하세요.
5. `background=true, notify_on_complete=true`로 **긴 작업을 백그라운드로 실행**하고 `process` 도구를 통해 모니터링하세요.
6. 다중 턴 대화형 작업에는 **tmux를 사용**하고 `tmux capture-pane -t <session> -p -S -50`으로 모니터링하세요.
7. 인증에 의존하기 전에 **인증 상태를 확인하세요.** `~/.grok/auth.json`을 확인하거나 비용이 적게 드는 `grok -p "Say ok."` 스모크 테스트를 실행하세요. Hermes의 xAI 인증이 이어진다고 가정하지 마세요.
8. **사용자에게 결과를 보고하세요.** Grok이 변경한 사항과 남은 사항을 요약하세요.

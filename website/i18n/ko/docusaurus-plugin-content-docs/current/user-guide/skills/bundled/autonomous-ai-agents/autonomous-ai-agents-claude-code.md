---
title: "Claude Code — Claude Code CLI에 코딩 위임(기능, PR)"
sidebar_label: "Claude Code"
description: "Claude Code CLI에 코딩 위임(기능, PR)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Claude Code

Claude Code CLI에 코딩을 위임합니다(기능, PR).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 번들됨(기본 설치) |
| 경로 | `skills/autonomous-ai-agents/claude-code` |
| 버전 | `2.2.1` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Claude`, `Anthropic`, `Code-Review`, `Refactoring`, `PTY`, `Automation` |
| 관련 스킬 | [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Claude Code — Hermes 오케스트레이션 가이드

[Claude Code](https://code.claude.com/docs/en/cli-reference)(Anthropic의 자율 코딩 에이전트 CLI)에 코딩 작업을 위임하며, Hermes 터미널을 통해 사용합니다. Claude Code v2.x는 파일 읽기, 코드 작성, 셸 명령 실행, 하위 에이전트 생성, git 워크플로 관리를 자율적으로 수행할 수 있습니다.

## 사전 요구 사항

- **설치:** `npm install -g @anthropic-ai/claude-code`
- **인증:** 로그인하려면 `claude`를 한 번 실행합니다(Pro/Max는 브라우저 OAuth를 사용하거나 `ANTHROPIC_API_KEY`를 설정).
- **콘솔 인증:** API 키 결제를 사용하려면 `claude auth login --console`
- **SSO 인증:** Enterprise용 `claude auth login --sso`
- **상태 확인:** `claude auth status`(JSON) 또는 `claude auth status --text`(사람이 읽을 수 있는 형식)
- **상태 점검:** `claude doctor` — 자동 업데이터와 설치 상태를 확인합니다.
- **버전 확인:** `claude --version`(v2.x 이상 필요)
- **업데이트:** `claude update` 또는 `claude upgrade`

## 두 가지 오케스트레이션 모드

Hermes는 Claude Code와 근본적으로 다른 두 가지 방식으로 상호작용합니다. 작업에 따라 선택하세요.

### 모드 1: Print 모드(`-p`) — 비대화형(대부분의 작업에 권장)

Print 모드는 일회성 작업을 실행하고, 결과를 반환한 뒤 종료합니다. PTY가 필요하지 않습니다. 대화형 프롬프트도 없습니다. 가장 깔끔한 통합 경로입니다.

```
terminal(command="claude -p 'Add error handling to all API calls in src/' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

**Print 모드를 사용할 때:**
- 일회성 코딩 작업(버그 수정, 기능 추가, 리팩터링)
- CI/CD 자동화 및 스크립팅
- `--json-schema`를 사용하는 구조화된 데이터 추출
- 파이프 입력 처리(`cat file | claude -p "analyze this"`)
- 여러 턴의 대화가 필요하지 않은 모든 작업

**Print 모드는 모든 대화형 대화상자를 건너뜁니다** — 워크스페이스 신뢰 프롬프트와 권한 확인이 없습니다. 따라서 자동화에 적합합니다.

### 모드 2: tmux를 통한 대화형 PTY — 여러 턴 세션

대화형 모드는 후속 프롬프트를 보내고, 슬래시 명령을 사용하며, Claude의 작업을 실시간으로 확인할 수 있는 완전한 대화형 REPL을 제공합니다. **tmux 오케스트레이션이 필요합니다.**

```
# Start a tmux session
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")

# Launch Claude Code inside it
terminal(command="tmux send-keys -t claude-work 'cd /path/to/project && claude' Enter")

# Wait for startup, then send your task
# (after ~3-5 seconds for the welcome screen)
terminal(command="sleep 5 && tmux send-keys -t claude-work 'Refactor the auth module to use JWT tokens' Enter")

# Monitor progress by capturing the pane
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -50")

# Send follow-up tasks
terminal(command="tmux send-keys -t claude-work 'Now add unit tests for the new JWT code' Enter")

# Exit when done
terminal(command="tmux send-keys -t claude-work '/exit' Enter")
```

**대화형 모드를 사용할 때:**
- 여러 턴의 반복 작업(리팩터링 → 검토 → 수정 → 테스트 사이클)
- 사람의 개입을 통한 결정이 필요한 작업
- 탐색적 코딩 세션
- Claude의 슬래시 명령(`/compact`, `/review`, `/model`)을 사용해야 할 때

## PTY 대화상자 처리(대화형 모드에서 중요)

Claude Code는 최초 실행 시 확인 대화상자를 최대 두 개 표시합니다. 반드시 tmux send-keys로 처리해야 합니다.

### 대화상자 1: 워크스페이스 신뢰(디렉터리를 처음 방문할 때)
```
❯ 1. Yes, I trust this folder    ← DEFAULT (just press Enter)
  2. No, exit
```
**처리:** `tmux send-keys -t <session> Enter` — 기본 선택이 올바릅니다.

### 대화상자 2: 권한 우회 경고(`--dangerously-skip-permissions` 사용 시에만)
```
❯ 1. No, exit                    ← DEFAULT (WRONG choice!)
  2. Yes, I accept
```
**처리:** 먼저 아래로 이동한 다음 Enter를 누릅니다.
```
tmux send-keys -t <session> Down && sleep 0.3 && tmux send-keys -t <session> Enter
```

### 안정적인 대화상자 처리 패턴
```
# Launch with permissions bypass
terminal(command="tmux send-keys -t claude-work 'claude --dangerously-skip-permissions \"your task\"' Enter")

# Handle trust dialog (Enter for default "Yes")
terminal(command="sleep 4 && tmux send-keys -t claude-work Enter")

# Handle permissions dialog (Down then Enter for "Yes, I accept")
terminal(command="sleep 3 && tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")

# Now wait for Claude to work
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")
```

**참고:** 디렉터리에 대한 최초 신뢰 수락 후에는 신뢰 대화상자가 다시 나타나지 않습니다. `--dangerously-skip-permissions`를 사용할 때마다 권한 대화상자만 다시 나타납니다.

## CLI 하위 명령

| 하위 명령 | 용도 |
|------------|---------|
| `claude` | 대화형 REPL 시작 |
| `claude "query"` | 초기 프롬프트와 함께 REPL 시작 |
| `claude -p "query"` | Print 모드(비대화형, 완료 시 종료) |
| `cat file \| claude -p "query"` | stdin 콘텐츠를 컨텍스트로 파이프 입력 |
| `claude -c` | 이 디렉터리에서 가장 최근 대화 계속하기 |
| `claude -r "id"` | 특정 세션을 ID 또는 이름으로 재개 |
| `claude auth login` | 로그인(`--console`을 추가하면 API 결제, `--sso`를 추가하면 Enterprise) |
| `claude auth status` | 로그인 상태 확인(JSON 반환, 사람이 읽을 수 있는 형식은 `--text`) |
| `claude mcp add <name> -- <cmd>` | MCP 서버 추가 |
| `claude mcp list` | 구성된 MCP 서버 나열 |
| `claude mcp remove <name>` | MCP 서버 제거 |
| `claude agents` | 구성된 에이전트 나열 |
| `claude doctor` | 설치 및 자동 업데이터 상태 점검 실행 |
| `claude update` / `claude upgrade` | Claude Code를 최신 버전으로 업데이트 |
| `claude remote-control` | claude.ai 또는 모바일 앱에서 Claude를 제어할 서버 시작 |
| `claude install [target]` | 네이티브 빌드 설치(stable, latest 또는 특정 버전) |
| `claude setup-token` | 장기 인증 토큰 설정(구독 필요) |
| `claude plugin` / `claude plugins` | Claude Code 플러그인 관리 |
| `claude auto-mode` | 자동 모드 분류기 구성 검사 |

## Print 모드 상세

### 구조화된 JSON 출력
```
terminal(command="claude -p 'Analyze auth.py for security issues' --output-format json --max-turns 5", workdir="/project", timeout=120)
```

다음과 같은 JSON 객체를 반환합니다.
```json
{
  "type": "result",
  "subtype": "success",
  "result": "The analysis text...",
  "session_id": "75e2167f-...",
  "num_turns": 3,
  "total_cost_usd": 0.0787,
  "duration_ms": 10276,
  "stop_reason": "end_turn",
  "terminal_reason": "completed",
  "usage": { "input_tokens": 5, "output_tokens": 603, ... },
  "modelUsage": { "claude-sonnet-4-6": { "costUSD": 0.078, "contextWindow": 200000 } }
}
```

**주요 필드:** 재개에 사용하는 `session_id`, 에이전트 루프의 턴 수인 `num_turns`, 지출 추적용 `total_cost_usd`, 성공/오류 감지용 `subtype`(`success`, `error_max_turns`, `error_budget`)입니다.

### 스트리밍 JSON 출력
실시간 토큰 스트리밍에는 `--verbose`와 함께 `stream-json`을 사용합니다.
```
terminal(command="claude -p 'Write a summary' --output-format stream-json --verbose --include-partial-messages", timeout=60)
```

줄바꿈으로 구분된 JSON 이벤트를 반환합니다. 실시간 텍스트는 jq로 필터링합니다.
```
claude -p "Explain X" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
```

스트림 이벤트에는 `attempt`, `max_retries`, `error` 필드가 있는 `system/api_retry` 이벤트가 포함됩니다(예: `rate_limit`, `billing_error`).

### 양방향 스트리밍
실시간 입력과 출력 스트리밍을 모두 사용하려면 다음을 실행합니다.
```
claude -p "task" --input-format stream-json --output-format stream-json --replay-user-messages
```
`--replay-user-messages`는 확인을 위해 사용자 메시지를 stdout으로 다시 내보냅니다.

### 파이프 입력
```
# Pipe a file for analysis
terminal(command="cat src/auth.py | claude -p 'Review this code for bugs' --max-turns 1", timeout=60)

# Pipe multiple files
terminal(command="cat src/*.py | claude -p 'Find all TODO comments' --max-turns 1", timeout=60)

# Pipe command output
terminal(command="git diff HEAD~3 | claude -p 'Summarize these changes' --max-turns 1", timeout=60)
```

### 구조화된 추출을 위한 JSON Schema
```
terminal(command="claude -p 'List all functions in src/' --output-format json --json-schema '{\"type\":\"object\",\"properties\":{\"functions\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"functions\"]}' --max-turns 5", workdir="/project", timeout=90)
```

JSON 결과에서 `structured_output`을 파싱합니다. Claude는 반환 전에 출력이 스키마에 맞는지 검증합니다.

### 세션 계속하기
```
# Start a task
terminal(command="claude -p 'Start refactoring the database layer' --output-format json --max-turns 10 > /tmp/session.json", workdir="/project", timeout=180)

# Resume with session ID
terminal(command="claude -p 'Continue and add connection pooling' --resume $(cat /tmp/session.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"session_id\"])') --max-turns 5", workdir="/project", timeout=120)

# Or resume the most recent session in the same directory
terminal(command="claude -p 'What did you do last time?' --continue --max-turns 1", workdir="/project", timeout=30)

# Fork a session (new ID, keeps history)
terminal(command="claude -p 'Try a different approach' --resume <id> --fork-session --max-turns 10", workdir="/project", timeout=120)
```

지정한 모델이 기본 모델이 과부하 상태일 때 자동으로 대체합니다(Print 모드에만 해당).

### CI/스크립팅을 위한 Bare 모드
```
terminal(command="claude --bare -p 'Run all tests and report failures' --allowedTools 'Read,Bash' --max-turns 10", workdir="/project", timeout=180)
```

`--bare`는 훅, 플러그인, MCP 검색, CLAUDE.md 로드를 건너뜁니다. 가장 빠르게 시작할 수 있습니다. `ANTHROPIC_API_KEY`가 필요합니다(OAuth는 건너뜁니다).

베어 모드에서 컨텍스트를 선택적으로 로드하려면 다음 플래그를 사용합니다.
| 로드할 항목 | 플래그 |
|------|------|
| 시스템 프롬프트 추가 | `--append-system-prompt "text"` 또는 `--append-system-prompt-file path` |
| 설정 | `--settings <file-or-json>` |
| MCP 서버 | `--mcp-config <file-or-json>` |
| 사용자 지정 에이전트 | `--agents '<json>'` |

### 과부하 시 대체 모델
```
terminal(command="claude -p 'task' --fallback-model haiku --max-turns 5", timeout=90)
```
기본 모델이 과부하 상태일 때 지정한 모델로 자동 대체합니다(Print 모드에만 해당).

## 전체 CLI 플래그 참조

### 세션 및 환경
| 플래그 | 효과 |
|------|--------|
| `-p, --print` | 비대화형 일회성 모드(완료 시 종료) |
| `-c, --continue` | 현재 디렉터리에서 가장 최근 대화 재개 |
| `-r, --resume <id>` | ID 또는 이름으로 특정 세션 재개(ID가 없으면 대화형 선택기) |
| `--fork-session` | 재개할 때 원래 세션을 재사용하지 않고 새 세션 ID 생성 |
| `--session-id <uuid>` | 대화에 특정 UUID 사용 |
| `--no-session-persistence` | 세션을 디스크에 저장하지 않음(Print 모드에만 해당) |
| `--add-dir <paths...>` | Claude에 추가 작업 디렉터리 접근 권한 부여 |
| `-w, --worktree [name]` | `.claude/worktrees/<name>`의 격리된 git worktree에서 실행 |
| `--tmux` | worktree용 tmux 세션 생성(`--worktree` 필요) |
| `--ide` | 시작 시 유효한 IDE에 자동 연결 |
| `--chrome` / `--no-chrome` | 웹 테스트용 Chrome 브라우저 통합 활성화/비활성화 |
| `--from-pr [number]` | 특정 GitHub PR에 연결된 세션 재개 |
| `--file <specs...>` | 시작 시 파일 리소스 다운로드(형식: `file_id:relative_path`) |

### 모델 및 성능
| 플래그 | 효과 |
|------|--------|
| `--model <alias>` | 모델 선택: `sonnet`, `opus`, `haiku` 또는 `claude-sonnet-4-6` 같은 전체 이름 |
| `--effort <level>` | 추론 깊이: `low`, `medium`, `high`, `xhigh`, `max` |
| `--max-turns <n>` | 에이전트 루프 제한(Print 모드에만 해당, 무한 실행 방지) |
| `--max-budget-usd <n>` | API 지출을 달러 단위로 제한(Print 모드에만 해당) |
| `--fallback-model <model>` | 기본 모델이 과부하일 때 자동 대체(Print 모드에만 해당) |
| `--betas <betas...>` | API 요청에 포함할 베타 헤더(API 키 사용자만 해당) |

### 권한 및 안전
| 플래그 | 효과 |
|------|--------|
| `--dangerously-skip-permissions` | 모든 도구 사용(파일 쓰기, bash, 네트워크 등)을 자동 승인 |
| `--allow-dangerously-skip-permissions` | 기본적으로 활성화하지 않고 우회를 *옵션*으로 허용 |
| `--permission-mode <mode>` | `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions` |
| `--allowedTools <tools...>` | 특정 도구를 허용 목록에 추가(쉼표 또는 공백으로 구분) |
| `--disallowedTools <tools...>` | 특정 도구를 차단 목록에 추가 |
| `--tools <tools...>` | 내장 도구 집합 재정의(`""` = 없음, `"default"` = 모두 또는 도구 이름) |

### 출력 및 입력 형식
| 플래그 | 효과 |
|------|--------|
| `--output-format <fmt>` | `text`(기본값), `json`(단일 결과 객체), `stream-json`(줄바꿈으로 구분) |
| `--input-format <fmt>` | `text`(기본값) 또는 `stream-json`(실시간 스트리밍 입력) |
| `--json-schema <schema>` | 스키마에 맞는 구조화된 JSON 출력 강제 |
| `--verbose` | 전체 턴별 출력 |
| `--include-partial-messages` | 도착하는 부분 메시지 청크 포함(`stream-json` + print) |
| `--replay-user-messages` | stdout으로 사용자 메시지 재출력(`stream-json` 양방향) |

### 시스템 프롬프트 및 컨텍스트
| 플래그 | 효과 |
|------|--------|
| `--append-system-prompt <text>` | 기본 시스템 프롬프트에 **추가**(내장 기능 유지) |
| `--append-system-prompt-file <path>` | 기본 시스템 프롬프트에 파일 콘텐츠 **추가** |
| `--system-prompt <text>` | 전체 시스템 프롬프트 **교체**(일반적으로 --append 사용) |
| `--system-prompt-file <path>` | 파일 콘텐츠로 시스템 프롬프트 **교체** |
| `--bare` | 훅, 플러그인, MCP 검색, CLAUDE.md, OAuth 건너뜀(가장 빠른 시작) |
| `--agents '<json>'` | JSON으로 사용자 지정 하위 에이전트를 동적으로 정의 |
| `--mcp-config <path>` | JSON 파일에서 MCP 서버 로드(반복 가능) |
| `--strict-mcp-config` | `--mcp-config`의 MCP 서버만 사용하고 다른 MCP 구성은 무시 |
| `--settings <file-or-json>` | JSON 파일 또는 인라인 JSON에서 추가 설정 로드 |
| `--setting-sources <sources>` | 로드할 쉼표로 구분된 출처: `user`, `project`, `local` |
| `--plugin-dir <paths...>` | 이 세션에서만 디렉터리의 플러그인 로드 |
| `--disable-slash-commands` | 모든 스킬/슬래시 명령 비활성화 |

### 디버깅
| 플래그 | 효과 |
|------|--------|
| `-d, --debug [filter]` | 선택적 카테고리 필터로 디버그 로깅 활성화(예: `"api,hooks"`, `"!1p,!file"`) |
| `--debug-file <path>` | 파일에 디버그 로그 기록(디버그 모드도 암시적으로 활성화) |

### 에이전트 팀
| 플래그 | 효과 |
|------|--------|
| `--teammate-mode <mode>` | 에이전트 팀 표시 방식: `auto`, `in-process` 또는 `tmux` |
| `--brief` | 에이전트-사용자 통신용 `SendUserMessage` 도구 활성화 |

### --allowedTools / --disallowedTools의 도구 이름 구문
```
Read                    # All file reading
Edit                    # File editing (existing files)
Write                   # File creation (new files)
Bash                    # All shell commands
Bash(git *)             # Only git commands
Bash(git commit *)      # Only git commit commands
Bash(npm run lint:*)    # Pattern matching with wildcards
WebSearch               # Web search capability
WebFetch                # Web page fetching
mcp__<server>__<tool>   # Specific MCP tool
```

## 설정 및 구성

### 설정 계층(높은 우선순위부터 낮은 우선순위)
1. **CLI 플래그** — 모든 항목을 재정의
2. **로컬 프로젝트:** `.claude/settings.local.json`(개인용, gitignored)
3. **프로젝트:** `.claude/settings.json`(공유, git-tracked)
4. **사용자:** `~/.claude/settings.json`(전역)

### 설정의 권한
```json
{
  "permissions": {
    "allow": ["Bash(npm run lint:*)", "WebSearch", "Read"],
    "ask": ["Write(*.ts)", "Bash(git push*)"],
    "deny": ["Read(.env)", "Bash(rm -rf *)"]
  }
}
```

### 메모리 파일(CLAUDE.md) 계층
1. **전역:** `~/.claude/CLAUDE.md` — 모든 프로젝트에 적용
2. **프로젝트:** `./CLAUDE.md` — 프로젝트별 컨텍스트
3. **로컬:** `.claude/CLAUDE.local.md` — 개인 프로젝트 재정의(gitignored)

대화형 모드에서 `#` 접두사를 사용하면 메모리에 빠르게 추가할 수 있습니다. `# Always use 2-space indentation`.

## 대화형 세션: 슬래시 명령

### 세션 및 컨텍스트
| 명령 | 용도 |
|---------|---------|
| `/help` | 사용자 지정 및 MCP 명령을 포함한 모든 명령 표시 |
| `/compact [focus]` | 토큰 절약을 위해 컨텍스트 압축; CLAUDE.md는 압축 후에도 유지됩니다. 예: `/compact focus on auth logic` |
| `/clear` | 대화 기록을 지우고 새로 시작 |
| `/context` | 최적화 팁과 함께 컨텍스트 사용량을 색상 그리드로 시각화 |
| `/cost` | 모델별 및 캐시 적중률별 토큰 사용량 확인 |
| `/resume` | 다른 세션으로 전환하거나 재개 |
| `/rewind` | 대화 또는 코드의 이전 체크포인트로 되돌리기 |
| `/btw <question>` | 컨텍스트 비용을 추가하지 않고 부가 질문 |
| `/status` | 버전, 연결 상태 및 세션 정보 표시 |
| `/todos` | 대화에서 추적 중인 작업 항목 나열 |
| `/exit` 또는 `Ctrl+D` | 세션 종료 |

### 개발 및 검토
| 명령 | 용도 |
|---------|---------|
| `/review` | 현재 변경 사항의 코드 검토 요청 |
| `/security-review` | 현재 변경 사항의 보안 분석 수행 |
| `/plan [description]` | 계획 수립을 위해 Plan 모드 진입 |
| `/loop [interval]` | 세션 내 반복 작업 예약 |
| `/batch` | 대규모 병렬 변경을 위해 worktree 자동 생성(5-30개 worktree) |

### 구성 및 도구
| 명령 | 용도 |
|---------|---------|
| `/model [model]` | 세션 중 모델 전환(화살표 키로 effort 조정) |
| `/effort [level]` | 추론 수준 설정: `low`, `medium`, `high`, `xhigh` 또는 `max` |
| `/init` | CLAUDE.md 파일 생성 |
| `/memory` | 편집을 위해 CLAUDE.md 열기 |
| `/config` | 대화형 설정 구성 열기 |
| `/permissions` | 도구 권한 확인/업데이트 |
| `/agents` | 전문 하위 에이전트 관리 |
| `/mcp` | MCP 서버를 관리하는 대화형 UI |
| `/add-dir` | 추가 작업 디렉터리 추가(모노레포에 유용) |
| `/usage` | 플랜 제한 및 속도 제한 상태 표시 |
| `/voice` | 누르고 말하기 음성 모드 활성화(20개 언어, Space를 누르고 녹음, 놓으면 전송) |
| `/release-notes` | 버전 릴리스 노트 대화형 선택기 |

### 사용자 지정 슬래시 명령
프로젝트 공유용 `.claude/commands/<name>.md` 또는 개인용 `~/.claude/commands/<name>.md`를 생성합니다.

```markdown
# .claude/commands/deploy.md
Run the deploy pipeline:
1. Run all tests
2. Build the Docker image
3. Push to registry
4. Update the $ARGUMENTS environment (default: staging)
```

사용법: `/deploy production` — `$ARGUMENTS`가 사용자의 입력으로 대체됩니다.

### 스킬(자연어 호출)
수동으로 호출하는 슬래시 명령과 달리, `.claude/skills/`의 스킬은 작업이 일치하면 Claude가 자연어를 통해 자동으로 호출하는 Markdown 가이드입니다.

```markdown
# .claude/skills/database-migration.md
When asked to create or modify database migrations:
1. Use Alembic for migration generation
2. Always create a rollback function
3. Test migrations against a local database copy
```

## 대화형 세션: 키보드 단축키

### 일반 제어
| 키 | 동작 |
|-----|--------|
| `Ctrl+C` | 현재 입력 또는 생성 취소 |
| `Ctrl+D` | 세션 종료 |
| `Ctrl+R` | 명령 기록 역방향 검색 |
| `Ctrl+B` | 실행 중인 작업을 백그라운드로 전환 |
| `Ctrl+V` | 이미지를 대화에 붙여넣기 |
| `Ctrl+O` | 트랜스크립트 모드 — Claude의 사고 과정 확인 |
| `Ctrl+G` 또는 `Ctrl+X Ctrl+E` | 외부 편집기에서 프롬프트 열기 |
| `Esc Esc` | 대화 또는 코드 상태 되돌리기 / 요약 |

### 모드 전환
| 키 | 동작 |
|-----|--------|
| `Shift+Tab` | 권한 모드 순환(Normal → Auto-Accept → Plan) |
| `Alt+P` | 모델 전환 |
| `Alt+T` | 사고 모드 전환 |
| `Alt+O` | Fast Mode 전환 |

### 여러 줄 입력
| 키 | 동작 |
|-----|--------|
| `\\` + `Enter` | 빠른 줄바꿈 |
| `Shift+Enter` | 줄바꿈(대안) |
| `Ctrl+J` | 줄바꿈(대안) |

### 입력 접두사
| 접두사 | 동작 |
|--------|--------|
| `!` | AI를 우회하여 bash 직접 실행(예: `!npm test`). 셸 모드를 전환하려면 `!`만 사용합니다. |
| `@` | 자동 완성으로 파일/디렉터리 참조(예: `@./src/api/`) |
| `#` | CLAUDE.md 메모리에 빠르게 추가(예: `# Use 2-space indentation`) |
| `/` | 슬래시 명령 |

### 전문가 팁: "ultrathink"
특정 턴에서 최대 추론 수준을 사용하려면 프롬프트에 "ultrathink" 키워드를 사용합니다. 현재 `/effort` 설정과 관계없이 가장 깊은 사고 모드를 트리거합니다.

## PR 검토 패턴

### 빠른 검토(Print 모드)
```
terminal(command="cd /path/to/repo && git diff main...feature-branch | claude -p 'Review this diff for bugs, security issues, and style problems. Be thorough.' --max-turns 1", timeout=60)
```

### 심층 검토(대화형 + Worktree)
```
terminal(command="tmux new-session -d -s review -x 140 -y 40")
terminal(command="tmux send-keys -t review 'cd /path/to/repo && claude -w pr-review' Enter")
terminal(command="sleep 5 && tmux send-keys -t review Enter")  # Trust dialog
terminal(command="sleep 2 && tmux send-keys -t review 'Review all changes vs main. Check for bugs, security issues, race conditions, and missing tests.' Enter")
terminal(command="sleep 30 && tmux capture-pane -t review -p -S -60")
```

### 번호로 PR 검토
```
terminal(command="claude -p 'Review this PR thoroughly' --from-pr 42 --max-turns 10", workdir="/path/to/repo", timeout=120)
```

### tmux를 사용한 Claude Worktree
```
terminal(command="claude -w feature-x --tmux", workdir="/path/to/repo")
```
격리된 git worktree를 `.claude/worktrees/feature-x`에 생성하고 tmux 세션도 생성합니다. iTerm2 네이티브 창을 사용할 수 있으면 이를 사용하며, 기존 tmux를 사용하려면 `--tmux=classic`을 추가합니다.

## 병렬 Claude 인스턴스

여러 개의 독립적인 Claude 작업을 동시에 실행합니다.

```
# Task 1: Fix backend
terminal(command="tmux new-session -d -s task1 -x 140 -y 40 && tmux send-keys -t task1 'cd ~/project && claude -p \"Fix the auth bug in src/auth.py\" --allowedTools \"Read,Edit\" --max-turns 10' Enter")

# Task 2: Write tests
terminal(command="tmux new-session -d -s task2 -x 140 -y 40 && tmux send-keys -t task2 'cd ~/project && claude -p \"Write integration tests for the API endpoints\" --allowedTools \"Read,Write,Bash\" --max-turns 15' Enter")

# Task 3: Update docs
terminal(command="tmux new-session -d -s task3 -x 140 -y 40 && tmux send-keys -t task3 'cd ~/project && claude -p \"Update README.md with the new API endpoints\" --allowedTools \"Read,Edit\" --max-turns 5' Enter")

# Monitor all
terminal(command="sleep 30 && for s in task1 task2 task3; do echo '=== '$s' ==='; tmux capture-pane -t $s -p -S -5 2>/dev/null; done")
```

## CLAUDE.md — 프로젝트 컨텍스트 파일

Claude Code는 프로젝트 루트에서 `CLAUDE.md`를 자동으로 로드합니다. 프로젝트 컨텍스트를 저장하는 데 사용하세요.

```markdown
# Project: My API

## Architecture
- FastAPI backend with SQLAlchemy ORM
- PostgreSQL database, Redis cache
- pytest for testing with 90% coverage target

## Key Commands
- `make test` — run full test suite
- `make lint` — ruff + mypy
- `make dev` — start dev server on :8000

## Code Standards
- Type hints on all public functions
- Docstrings in Google style
- 2-space indentation for YAML, 4-space for Python
- No wildcard imports
```

**구체적으로 작성하세요.** "좋은 코드를 작성하라" 대신 "JS에는 2칸 들여쓰기를 사용하라" 또는 "테스트 파일 이름에는 `.test.ts` 접미사를 사용하라"고 작성합니다. 구체적인 지침은 수정 사이클을 줄여 줍니다.

### 규칙 디렉터리(모듈식 CLAUDE.md)
규칙이 많은 프로젝트에서는 하나의 거대한 CLAUDE.md 대신 규칙 디렉터리를 사용합니다.
- **프로젝트 규칙:** `.claude/rules/*.md` — 팀 공유, git-tracked
- **사용자 규칙:** `~/.claude/rules/*.md` — 개인, 전역

rules 디렉터리의 각 `.md` 파일은 추가 컨텍스트로 로드됩니다. 하나의 CLAUDE.md에 모든 내용을 억지로 넣는 것보다 깔끔합니다.

### 자동 메모리
Claude는 학습한 프로젝트 컨텍스트를 `~/.claude/projects/<project>/memory/`에 자동 저장합니다.
- **제한:** 프로젝트당 25KB 또는 200줄
- CLAUDE.md와는 별개이며, Claude가 축적하는 자체 메모리입니다.

## 사용자 지정 하위 에이전트

프로젝트의 `.claude/agents/`, 개인용 `~/.claude/agents/` 또는 `--agents` CLI 플래그(세션)에 전문 에이전트를 정의합니다.

### 에이전트 위치 우선순위
1. `.claude/agents/` — 프로젝트 수준, 팀 공유
2. `--agents` CLI 플래그 — 세션별, 동적
3. `~/.claude/agents/` — 사용자 수준, 개인용

### 에이전트 생성
```markdown
# .claude/agents/security-reviewer.md
---
name: security-reviewer
description: Security-focused code review
model: opus
tools: [Read, Bash]
---
You are a senior security engineer. Review code for:
- Injection vulnerabilities (SQL, XSS, command injection)
- Authentication/authorization flaws
- Secrets in code
- Unsafe deserialization
```

호출: `@security-reviewer review the auth module`

### CLI를 통한 동적 에이전트
```
terminal(command="claude --agents '{\"reviewer\": {\"description\": \"Reviews code\", \"prompt\": \"You are a code reviewer focused on performance\"}}' -p 'Use @reviewer to check auth.py'", timeout=120)
```

Claude는 여러 에이전트를 오케스트레이션할 수 있습니다. "@db-expert를 사용해 쿼리를 최적화한 다음 @security로 변경 사항을 감사하세요."

## 훅 — 이벤트 자동화

프로젝트의 `.claude/settings.json` 또는 전역 `~/.claude/settings.json`에서 구성합니다.

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write(*.py)",
      "hooks": [{"type": "command", "command": "ruff check --fix $CLAUDE_FILE_PATHS"}]
    }],
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{"type": "command", "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -q 'rm -rf'; then echo 'Blocked!' && exit 2; fi"}]
    }],
    "Stop": [{
      "hooks": [{"type": "command", "command": "echo 'Claude finished a response' >> /tmp/claude-activity.log"}]
    }]
  }
}
```

### 8가지 훅 유형
| 훅 | 실행 시점 | 일반적인 용도 |
|-----|--------------|------------|
| `UserPromptSubmit` | Claude가 사용자 프롬프트를 처리하기 전 | 입력 검증, 로깅 |
| `PreToolUse` | 도구 실행 전 | 보안 게이트, 위험한 명령 차단(exit 2 = 차단) |
| `PostToolUse` | 도구 완료 후 | 코드 자동 포맷, 린터 실행 |
| `Notification` | 권한 요청 또는 입력 대기 시 | 데스크톱 알림, 경고 |
| `Stop` | Claude가 응답을 완료할 때 | 완료 로깅, 상태 업데이트 |
| `SubagentStop` | 하위 에이전트 완료 시 | 에이전트 오케스트레이션 |
| `PreCompact` | 컨텍스트 메모리 삭제 전 | 세션 트랜스크립트 백업 |
| `SessionStart` | 세션 시작 시 | 개발 컨텍스트 로드(예: `git status`) |

### 훅 환경 변수
| 변수 | 내용 |
|----------|---------|
| `CLAUDE_PROJECT_DIR` | 현재 프로젝트 경로 |
| `CLAUDE_FILE_PATHS` | 수정 중인 파일 |
| `CLAUDE_TOOL_INPUT` | JSON 형식의 도구 매개변수 |

### 보안 훅 예시
```json
{
  "PreToolUse": [{
    "matcher": "Bash",
    "hooks": [{"type": "command", "command": "if echo \"$CLAUDE_TOOL_INPUT\" | grep -qE 'rm -rf|git push.*--force|:(){ :|:& };:'; then echo 'Dangerous command blocked!' && exit 2; fi"}]
  }]
}
```

## MCP 통합

데이터베이스, API 및 서비스에 외부 도구 서버를 추가합니다.

```
# GitHub integration
terminal(command="claude mcp add -s user github -- npx @modelcontextprotocol/server-github", timeout=30)

# PostgreSQL queries
terminal(command="claude mcp add -s local postgres -- npx @anthropic-ai/server-postgres --connection-string postgresql://localhost/mydb", timeout=30)

# Puppeteer for web testing
terminal(command="claude mcp add puppeteer -- npx @anthropic-ai/server-puppeteer", timeout=30)
```

### MCP 범위
| 플래그 | 범위 | 저장 위치 |
|---------|-------|---------|
| `-s user` | 전역(모든 프로젝트) | `~/.claude.json` |
| `-s local` | 이 프로젝트(개인용) | `.claude/settings.local.json`(gitignored) |
| `-s project` | 이 프로젝트(팀 공유) | `.claude/settings.json`(git-tracked) |

### Print/CI 모드의 MCP
```
terminal(command="claude --bare -p 'Query database' --mcp-config mcp-servers.json --strict-mcp-config", timeout=60)
```
`--strict-mcp-config`는 `--mcp-config`에 지정된 서버를 제외한 모든 MCP 서버를 무시합니다.

채팅에서 MCP 리소스를 참조합니다: `@github:issue://123`

### MCP 제한 및 조정
- **도구 설명:** 서버당 도구 설명과 서버 지침은 2KB로 제한됩니다.
- **결과 크기:** 기본적으로 제한됩니다. 대규모 출력을 위해 `maxResultSizeChars` 주석을 사용하면 최대 **500K** 문자까지 허용됩니다.
- **출력 토큰:** 컨텍스트 범람을 방지하려면 `export MAX_MCP_OUTPUT_TOKENS=50000` — MCP 서버 출력 상한을 설정합니다.
- **전송 방식:** `stdio`(로컬 프로세스), `http`(원격), `sse`(서버 전송 이벤트)

## 대화형 세션 모니터링

### TUI 상태 읽기
```
# Periodic capture to check if Claude is still working or waiting for input
terminal(command="tmux capture-pane -t dev -p -S -10")
```

다음 표시를 확인합니다.
- 하단의 `❯` = 입력을 기다리는 중(Claude가 완료했거나 질문 중)
- `●` 줄 = Claude가 도구를 적극적으로 사용하는 중(읽기, 쓰기, 실행)
- `⏵⏵ bypass permissions on` = 권한 모드 상태 표시줄
- `◐ medium · /effort` = 상태 표시줄의 현재 effort 수준
- `ctrl+o to expand` = 도구 출력이 잘림(대화형으로 확장 가능)

### 컨텍스트 창 상태
대화형 모드에서 `/context`를 사용하여 색상 그리드로 컨텍스트 사용량을 확인합니다. 주요 임계값은 다음과 같습니다.
- **&lt; 70%** — 정상 작동, 최대 정밀도
- **70-85%** — 정밀도가 떨어지기 시작하므로 `/compact` 고려
- **> 85%** — 환각 위험이 크게 증가하므로 `/compact` 또는 `/clear` 사용

## 환경 변수

| 변수 | 효과 |
|----------|--------|
| `ANTHROPIC_API_KEY` | 인증용 API 키(OAuth의 대안) |
| `CLAUDE_CODE_EFFORT_LEVEL` | 기본 effort: `low`, `medium`, `high`, `max` 또는 `auto` |
| `MAX_THINKING_TOKENS` | 사고 토큰 상한(`0`으로 설정하면 사고를 완전히 비활성화) |
| `MAX_MCP_OUTPUT_TOKENS` | MCP 서버 출력 상한(기본값은 다양함, 예: `50000`) |
| `CLAUDE_CODE_NO_FLICKER=1` | 화면 깜박임 제거를 위해 alt-screen 렌더링 활성화 |
| `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` | 보안을 위해 하위 프로세스에서 자격 증명 제거 |

## 비용 및 성능 팁

1. Print 모드에서 **`--max-turns` 사용** — 무한 루프를 방지합니다. 대부분의 작업은 5-10부터 시작하세요.
2. 비용 상한에는 **`--max-budget-usd`**를 사용합니다. 시스템 프롬프트 캐시 생성만으로도 약 $0.05가 필요하다는 점에 유의하세요.
3. 간단한 작업에는 **`--effort low`**를 사용합니다(더 빠르고 저렴). 복잡한 추론에는 `high` 또는 `max`를 사용합니다.
4. CI/스크립팅에서는 **`--bare`**를 사용하여 플러그인/훅 검색 오버헤드를 건너뜁니다.
5. 필요한 것만 허용하도록 **`--allowedTools`**를 사용합니다(예: 검토에는 `Read`만).
6. 대화형 세션에서 컨텍스트가 커지면 **`/compact`**를 사용합니다.
7. 이미 알고 있는 콘텐츠를 분석할 때는 Claude가 파일을 읽게 하는 대신 **파이프 입력**을 사용합니다.
8. 간단한 작업에는 **`--model haiku`**(저렴), 복잡한 다단계 작업에는 **`--model opus`**를 사용합니다.
9. Print 모드에서 **`--fallback-model haiku`**를 사용하여 모델 과부하를 원활하게 처리합니다.
10. 서로 다른 작업에는 **새 세션을 시작**합니다 — 세션은 5시간 지속되며, 새 컨텍스트가 더 효율적입니다.
11. CI에서 세션이 누적되지 않도록 **`--no-session-persistence`**를 사용합니다.

## 함정 및 주의 사항

1. **대화형 모드에는 tmux가 필수입니다** — Claude Code는 완전한 TUI 앱입니다. Hermes 터미널에서 `pty=true`만 사용해도 작동하지만, 모니터링과 입력에 필수적인 `capture-pane`과 `send-keys`를 제공하므로 tmux가 더 좋습니다.
2. **`--dangerously-skip-permissions` 대화상자의 기본값은 "No, exit"입니다** — 수락하려면 Down 다음 Enter를 보내야 합니다. Print 모드(`-p`)에서는 이를 건너뜁니다.
3. **`--max-budget-usd`의 최솟값은 약 $0.05입니다** — 시스템 프롬프트 캐시 생성만으로도 이 정도 비용이 듭니다. 더 낮게 설정하면 즉시 오류가 발생합니다.
4. **`--max-turns`는 Print 모드 전용입니다** — 대화형 세션에서는 무시됩니다.
5. **Claude는 `python3` 대신 `python`을 사용할 수 있습니다** — `python` 심볼릭 링크가 없는 시스템에서는 Claude의 bash 명령이 첫 시도에 실패하지만 스스로 수정합니다.
6. **세션 재개에는 같은 디렉터리가 필요합니다** — `--continue`는 현재 작업 디렉터리의 가장 최근 세션을 찾습니다.
7. **`--json-schema`에는 충분한 `--max-turns`가 필요합니다** — Claude가 구조화된 출력을 생성하기 전에 파일을 읽어야 하므로 여러 턴이 걸립니다.
8. **신뢰 대화상자는 디렉터리당 한 번만 나타납니다** — 최초 방문 시에만 나타나며 이후 캐시됩니다.
9. **백그라운드 tmux 세션은 유지됩니다** — 완료 후에는 항상 `tmux kill-session -t <name>`으로 정리합니다.
10. **슬래시 명령(`/commit` 등)은 대화형 모드에서만 작동합니다** — `-p` 모드에서는 대신 자연어로 작업을 설명합니다.
11. **`--bare`는 OAuth를 건너뜁니다** — `ANTHROPIC_API_KEY` 환경 변수 또는 설정의 `apiKeyHelper`가 필요합니다.
12. **컨텍스트 저하는 실제로 발생합니다** — 컨텍스트 창 사용량이 70%를 넘으면 AI 출력 품질이 측정 가능하게 저하됩니다. `/context`로 모니터링하고 사전에 `/compact`를 사용하세요.

## Hermes 에이전트를 위한 규칙

1. **단일 작업에는 Print 모드(`-p`)를 우선 사용** — 더 깔끔하고 대화상자 처리가 필요 없으며 구조화된 출력을 제공합니다.
2. **여러 턴의 대화형 작업에는 tmux 사용** — TUI를 오케스트레이션하는 유일하게 신뢰할 수 있는 방법입니다.
3. **항상 `workdir` 설정** — Claude가 올바른 프로젝트 디렉터리에 집중하도록 합니다.
4. **Print 모드에서 `--max-turns` 설정** — 무한 실행과 비용 폭주를 방지합니다.
5. **tmux 세션 모니터링** — 진행 상황을 확인하려면 `tmux capture-pane -t <session> -p -S -50`을 사용합니다.
6. **`❯` 프롬프트 확인** — 입력을 기다리고 있음을 나타냅니다(완료했거나 질문 중).
7. **tmux 세션 정리** — 리소스 누수를 방지하려면 완료 후 종료합니다.
8. **결과를 사용자에게 보고** — 완료 후 Claude가 수행한 작업과 변경 사항을 요약합니다.
9. **느린 세션을 종료하지 않기** — Claude가 여러 단계의 작업을 수행 중일 수 있으므로 먼저 진행 상황을 확인합니다.
10. **`--allowedTools` 사용** — 작업에 실제로 필요한 기능만 허용합니다.

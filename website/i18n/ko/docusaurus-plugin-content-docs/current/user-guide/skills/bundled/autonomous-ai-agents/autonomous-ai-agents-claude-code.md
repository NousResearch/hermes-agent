---
title: "Claude Code — Delegate coding to Claude Code CLI (features, PRs)"
sidebar_label: "Claude Code"
description: "Delegate coding to Claude Code CLI (features, PRs)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Claude Code

코딩 작업을 Claude Code CLI에 위임하기 (기능 구현, PR).

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/autonomous-ai-agents/claude-code` |
| Version | `2.2.0` |
| Author | Hermes Agent + Teknium |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `Claude`, `Anthropic`, `Code-Review`, `Refactoring`, `PTY`, `Automation` |
| Related skills | [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent), [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Claude Code — Hermes 오케스트레이션 가이드

Hermes 터미널을 통해 코딩 작업을 [Claude Code](https://code.claude.com/docs/en/cli-reference) (Anthropic의 자율 코딩 에이전트 CLI)에 위임합니다. Claude Code v2.x는 파일을 읽고, 코드를 작성하며, 쉘 명령을 실행하고, 하위 에이전트(subagent)를 생성하고, git 워크플로우를 자율적으로 관리할 수 있습니다.

## 사전 요구 사항

- **설치:** `npm install -g @anthropic-ai/claude-code`
- **인증:** 한 번 `claude`를 실행하여 로그인합니다 (Pro/Max의 경우 브라우저 OAuth 사용, 또는 `ANTHROPIC_API_KEY` 설정).
- **콘솔 인증:** API 키 결제를 위해 `claude auth login --console`
- **SSO 인증:** 엔터프라이즈를 위해 `claude auth login --sso`
- **상태 확인:** `claude auth status` (JSON) 또는 `claude auth status --text` (사람이 읽기 쉬운 형식)
- **상태 점검(Health check):** `claude doctor` — 자동 업데이트 및 설치 상태 확인
- **버전 확인:** `claude --version` (v2.x+ 필요)
- **업데이트:** `claude update` 또는 `claude upgrade`

## 두 가지 오케스트레이션 모드

Hermes는 두 가지 근본적으로 다른 방식으로 Claude Code와 상호 작용합니다. 작업에 따라 선택하세요.

### 모드 1: 인쇄(Print) 모드 (`-p`) — 비대화형 (대부분의 작업에 권장)

Print 모드는 일회성(one-shot) 작업을 실행하고 결과를 반환한 다음 종료합니다. PTY가 필요 없습니다. 대화형 프롬프트도 없습니다. 이것이 가장 깔끔한 통합 방식입니다.

```
terminal(command="claude -p 'Add error handling to all API calls in src/' --allowedTools 'Read,Edit' --max-turns 10", workdir="/path/to/project", timeout=120)
```

**Print 모드 사용 시기:**
- 일회성 코딩 작업 (버그 수정, 기능 추가, 리팩터링)
- CI/CD 자동화 및 스크립팅
- `--json-schema`를 사용한 구조화된 데이터 추출
- 파이프 입력 처리 (`cat file | claude -p "analyze this"`)
- 다중 턴(multi-turn) 대화가 필요 없는 모든 작업

**Print 모드는 모든 대화형 대화창을 건너뜁니다** — 작업 공간 신뢰(workspace trust) 프롬프트나 권한 확인 절차가 없습니다. 따라서 자동화에 이상적입니다.

### 모드 2: tmux를 통한 대화형 PTY — 다중 턴(Multi-Turn) 세션

대화형 모드는 추가 프롬프트를 보내고, 슬래시(/) 명령을 사용하며, Claude가 실시간으로 작업하는 것을 볼 수 있는 완전한 대화형 REPL을 제공합니다. **tmux 오케스트레이션이 필요합니다.**

```
# tmux 세션 시작
terminal(command="tmux new-session -d -s claude-work -x 140 -y 40")

# 그 안에서 Claude Code 시작
terminal(command="tmux send-keys -t claude-work 'cd /path/to/project && claude' Enter")

# 시작될 때까지 기다린 후 작업 전송
# (환영 화면이 나올 때까지 약 3-5초 후)
terminal(command="sleep 5 && tmux send-keys -t claude-work 'Refactor the auth module to use JWT tokens' Enter")

# 창(pane)을 캡처하여 진행 상태 모니터링
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -50")

# 후속 작업 전송
terminal(command="tmux send-keys -t claude-work 'Now add unit tests for the new JWT code' Enter")

# 완료 시 종료
terminal(command="tmux send-keys -t claude-work '/exit' Enter")
```

**대화형 모드 사용 시기:**
- 다중 턴의 반복적인 작업 (리팩터링 → 리뷰 → 수정 → 테스트 사이클)
- Human-in-the-loop 결정이 필요한 작업
- 탐색적 코딩 세션
- Claude의 슬래시 명령(`/compact`, `/review`, `/model`)을 사용해야 할 때

## PTY 대화창 처리 (대화형 모드에서 매우 중요)

Claude Code는 첫 실행 시 최대 두 개의 확인 대화창을 표시합니다. 반드시 tmux `send-keys`를 통해 이를 처리해야 합니다:

### 대화창 1: 작업 공간 신뢰 (디렉토리 첫 방문 시)
```
❯ 1. Yes, I trust this folder    ← 기본값 (그냥 Enter 누르기)
  2. No, exit
```
**처리 방법:** `tmux send-keys -t <session> Enter` — 기본 선택이 맞습니다.

### 대화창 2: 권한 경고 우회 (`--dangerously-skip-permissions` 사용 시에만)
```
❯ 1. No, exit                    ← 기본값 (잘못된 선택!)
  2. Yes, I accept
```
**처리 방법:** 반드시 먼저 '아래(DOWN)'로 이동한 후 Enter를 눌러야 합니다:
```
tmux send-keys -t <session> Down && sleep 0.3 && tmux send-keys -t <session> Enter
```

### 견고한 대화창 처리 패턴
```
# 권한 우회 옵션과 함께 시작
terminal(command="tmux send-keys -t claude-work 'claude --dangerously-skip-permissions \"your task\"' Enter")

# 신뢰 대화창 처리 (기본값인 "Yes"를 위해 Enter)
terminal(command="sleep 4 && tmux send-keys -t claude-work Enter")

# 권한 대화창 처리 ("Yes, I accept"를 위해 Down 후 Enter)
terminal(command="sleep 3 && tmux send-keys -t claude-work Down && sleep 0.3 && tmux send-keys -t claude-work Enter")

# 이제 Claude가 작업하기를 기다림
terminal(command="sleep 15 && tmux capture-pane -t claude-work -p -S -60")
```

**참고:** 특정 디렉토리에 대해 한 번 신뢰(trust)를 수락한 후에는 신뢰 대화창이 다시 나타나지 않습니다. 권한(permissions) 대화창만 `--dangerously-skip-permissions`를 사용할 때마다 매번 반복해서 나타납니다.

## CLI 하위 명령어 (Subcommands)

| 하위 명령어 | 목적 |
|------------|---------|
| `claude` | 대화형 REPL 시작 |
| `claude "query"` | 초기 프롬프트와 함께 REPL 시작 |
| `claude -p "query"` | Print 모드 (비대화형, 완료 시 종료) |
| `cat file \| claude -p "query"` | 파이프 콘텐츠를 stdin 컨텍스트로 전달 |
| `claude -c` | 현재 디렉토리에서 가장 최근 대화 계속하기 |
| `claude -r "id"` | ID 또는 이름으로 특정 세션 재개하기 |
| `claude auth login` | 로그인 (API 청구용은 `--console`, 엔터프라이즈용은 `--sso` 추가) |
| `claude auth status` | 로그인 상태 확인 (JSON 반환; 사람이 읽기 쉬운 형식은 `--text` 추가) |
| `claude mcp add <name> -- <cmd>` | MCP 서버 추가 |
| `claude mcp list` | 구성된 MCP 서버 목록 보기 |
| `claude mcp remove <name>` | MCP 서버 제거 |
| `claude agents` | 구성된 에이전트 목록 보기 |
| `claude doctor` | 설치 및 자동 업데이터 상태 점검 실행 |
| `claude update` / `claude upgrade` | Claude Code를 최신 버전으로 업데이트 |
| `claude remote-control` | claude.ai 또는 모바일 앱에서 Claude를 제어하는 서버 시작 |
| `claude install [target]` | 네이티브 빌드 설치 (stable, latest, 또는 특정 버전) |
| `claude setup-token` | 수명이 긴 인증 토큰 설정 (구독 필요) |
| `claude plugin` / `claude plugins` | Claude Code 플러그인 관리 |
| `claude auto-mode` | 자동 모드 분류기 설정 검사 |

## Print 모드 심층 분석

### 구조화된 JSON 출력
```
terminal(command="claude -p 'Analyze auth.py for security issues' --output-format json --max-turns 5", workdir="/project", timeout=120)
```

다음과 같은 JSON 객체를 반환합니다:
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

**주요 필드:** 재개를 위한 `session_id`, 에이전틱 루프 횟수인 `num_turns`, 비용 추적을 위한 `total_cost_usd`, 성공/오류 감지를 위한 `subtype` (`success`, `error_max_turns`, `error_budget`).

### 스트리밍 JSON 출력
실시간 토큰 스트리밍의 경우 `--verbose`와 함께 `stream-json`을 사용합니다:
```
terminal(command="claude -p 'Write a summary' --output-format stream-json --verbose --include-partial-messages", timeout=60)
```

줄 바꿈(newline)으로 구분된 JSON 이벤트를 반환합니다. jq를 사용하여 라이브 텍스트를 필터링합니다:
```
claude -p "Explain X" --output-format stream-json --verbose --include-partial-messages | \
  jq -rj 'select(.type == "stream_event" and .event.delta.type? == "text_delta") | .event.delta.text'
```

스트림 이벤트에는 `attempt`, `max_retries`, `error` 필드(예: `rate_limit`, `billing_error`)가 포함된 `system/api_retry`가 포함됩니다.

### 양방향 스트리밍
실시간 입력 및 출력 스트리밍의 경우:
```
claude -p "task" --input-format stream-json --output-format stream-json --replay-user-messages
```
`--replay-user-messages`는 확인을 위해 사용자 메시지를 stdout에 다시 보냅니다(re-emit).

### 파이프 입력
```
# 분석을 위해 파일을 파이프로 전달
terminal(command="cat src/auth.py | claude -p 'Review this code for bugs' --max-turns 1", timeout=60)

# 여러 파일 전달
terminal(command="cat src/*.py | claude -p 'Find all TODO comments' --max-turns 1", timeout=60)

# 명령어 출력 전달
terminal(command="git diff HEAD~3 | claude -p 'Summarize these changes' --max-turns 1", timeout=60)
```

### 구조화된 데이터 추출을 위한 JSON 스키마
```
terminal(command="claude -p 'List all functions in src/' --output-format json --json-schema '{\"type\":\"object\",\"properties\":{\"functions\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},\"required\":[\"functions\"]}' --max-turns 5", workdir="/project", timeout=90)
```

JSON 결과에서 `structured_output`을 파싱하세요. Claude는 결과를 반환하기 전에 스키마에 대해 출력을 검증합니다.

### 세션 이어하기 (Continuation)
```
# 작업 시작
terminal(command="claude -p 'Start refactoring the database layer' --output-format json --max-turns 10 > /tmp/session.json", workdir="/project", timeout=180)

# 세션 ID로 재개
terminal(command="claude -p 'Continue and add connection pooling' --resume $(cat /tmp/session.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[\"session_id\"])') --max-turns 5", workdir="/project", timeout=120)

# 또는 동일한 디렉토리에서 가장 최근 세션 재개
terminal(command="claude -p 'What did you do last time?' --continue --max-turns 1", workdir="/project", timeout=30)

# 세션 분기 (fork - 히스토리를 유지하며 새 ID 생성)
terminal(command="claude -p 'Try a different approach' --resume <id> --fork-session --max-turns 10", workdir="/project", timeout=120)
```

### CI/스크립팅을 위한 Bare 모드
```
terminal(command="claude --bare -p 'Run all tests and report failures' --allowedTools 'Read,Bash' --max-turns 10", workdir="/project", timeout=180)
```

`--bare`는 훅, 플러그인, MCP 검색, CLAUDE.md 로딩을 건너뜁니다. 가장 빠른 시작 속도를 제공합니다. `ANTHROPIC_API_KEY`가 필요합니다(OAuth 건너뜀).

bare 모드에서 컨텍스트를 선택적으로 로드하려면:
| 로드할 대상 | 플래그 |
|---------|------|
| 시스템 프롬프트 추가 | `--append-system-prompt "text"` 또는 `--append-system-prompt-file path` |
| 설정 | `--settings <file-or-json>` |
| MCP 서버 | `--mcp-config <file-or-json>` |
| 사용자 지정 에이전트 | `--agents '<json>'` |

### 과부하 시 대체 모델 (Fallback Model)
```
terminal(command="claude -p 'task' --fallback-model haiku --max-turns 5", timeout=90)
```
기본 모델이 과부하 상태일 때 지정된 모델로 자동 대체합니다 (print 모드 전용).

## 전체 CLI 플래그 참조

### 세션 & 환경
| 플래그 | 효과 |
|------|--------|
| `-p, --print` | 비대화형 일회성 모드 (완료 시 종료) |
| `-c, --continue` | 현재 디렉토리에서 가장 최근 대화 계속하기 |
| `-r, --resume <id>` | ID 또는 이름으로 특정 세션 재개 (ID가 없으면 대화형 선택기 표시) |
| `--fork-session` | 재개 시 원래 ID를 재사용하지 않고 새 세션 ID 생성 |
| `--session-id <uuid>` | 대화에 특정 UUID 사용 |
| `--no-session-persistence` | 디스크에 세션을 저장하지 않음 (print 모드 전용) |
| `--add-dir <paths...>` | Claude에게 추가 작업 디렉토리에 대한 접근 권한 부여 |
| `-w, --worktree [name]` | `.claude/worktrees/<name>`의 격리된 git worktree에서 실행 |
| `--tmux` | worktree를 위한 tmux 세션 생성 (`--worktree` 필요) |
| `--ide` | 시작 시 유효한 IDE에 자동 연결 |
| `--chrome` / `--no-chrome` | 웹 테스트를 위한 Chrome 브라우저 통합 활성화/비활성화 |
| `--from-pr [number]` | 특정 GitHub PR에 연결된 세션 재개 |
| `--file <specs...>` | 시작 시 다운로드할 파일 리소스 (형식: `file_id:relative_path`) |

### 모델 & 성능
| 플래그 | 효과 |
|------|--------|
| `--model <alias>` | 모델 선택: `sonnet`, `opus`, `haiku` 또는 `claude-sonnet-4-6` 같은 전체 이름 |
| `--effort <level>` | 추론(Reasoning) 깊이: `low`, `medium`, `high`, `max`, `auto` |
| `--max-turns <n>` | 에이전틱 루프 제한 (print 모드 전용; 폭주 방지) |
| `--max-budget-usd <n>` | API 최대 지출 비용 한도 (달러) 설정 (print 모드 전용) |
| `--fallback-model <model>` | 기본 모델이 과부하 시 자동 대체 모델 (print 모드 전용) |
| `--betas <betas...>` | API 요청에 포함할 베타 헤더 (API 키 사용자 전용) |

### 권한 & 안전
| 플래그 | 효과 |
|------|--------|
| `--dangerously-skip-permissions` | 모든 도구 사용(파일 쓰기, bash, 네트워크 등)을 자동 승인 |
| `--allow-dangerously-skip-permissions` | 우해 기능을 기본적으로 활성화하지 않으면서 *옵션*으로 사용할 수 있게 함 |
| `--permission-mode <mode>` | `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions` |
| `--allowedTools <tools...>` | 허용할 특정 도구 목록 (쉼표 또는 공백으로 구분) |
| `--disallowedTools <tools...>` | 차단할 특정 도구 목록 |
| `--tools <tools...>` | 내장 도구 세트 무시 및 대체 (`""` = 없음, `"default"` = 모두, 또는 특정 도구 이름들) |

### 출력 & 입력 형식
| 플래그 | 효과 |
|------|--------|
| `--output-format <fmt>` | `text` (기본값), `json` (단일 결과 객체), `stream-json` (줄바꿈 구분) |
| `--input-format <fmt>` | `text` (기본값) 또는 `stream-json` (실시간 스트리밍 입력) |
| `--json-schema <schema>` | 스키마와 일치하는 구조화된 JSON 출력을 강제함 |
| `--verbose` | 턴(turn)별 전체 출력 |
| `--include-partial-messages` | 부분 메시지 청크가 도착하는 대로 포함 (stream-json + print) |
| `--replay-user-messages` | 사용자 메시지를 stdout에 다시 보냄(re-emit) (stream-json 양방향) |

### 시스템 프롬프트 & 컨텍스트
| 플래그 | 효과 |
|------|--------|
| `--append-system-prompt <text>` | 기본 시스템 프롬프트에 텍스트 **추가** (내장 기능 보존) |
| `--append-system-prompt-file <path>` | 기본 시스템 프롬프트에 파일 내용 **추가** |
| `--system-prompt <text>` | 전체 시스템 프롬프트를 텍스트로 **교체** (대개 --append 사용 권장) |
| `--system-prompt-file <path>` | 시스템 프롬프트를 파일 내용으로 **교체** |
| `--bare` | 훅, 플러그인, MCP 검색, CLAUDE.md, OAuth 건너뜀 (가장 빠른 시작) |
| `--agents '<json>'` | JSON으로 사용자 정의 하위 에이전트 동적 정의 |
| `--mcp-config <path>` | JSON 파일에서 MCP 서버 로드 (반복 가능) |
| `--strict-mcp-config` | 다른 모든 MCP 구성을 무시하고 `--mcp-config`의 MCP 서버만 사용 |
| `--settings <file-or-json>` | JSON 파일이나 인라인 JSON에서 추가 설정 로드 |
| `--setting-sources <sources>` | 쉼표로 구분된 로드할 소스: `user`, `project`, `local` |
| `--plugin-dir <paths...>` | 이번 세션에 대해서만 해당 디렉토리에서 플러그인 로드 |
| `--disable-slash-commands` | 모든 스킬/슬래시 명령어 비활성화 |

### 디버깅
| 플래그 | 효과 |
|------|--------|
| `-d, --debug [filter]` | 선택적 카테고리 필터를 사용하여 디버그 로깅 활성화 (예: `"api,hooks"`, `"!1p,!file"`) |
| `--debug-file <path>` | 디버그 로그를 파일에 기록 (디버그 모드를 암시적으로 활성화) |

### 에이전트 팀
| 플래그 | 효과 |
|------|--------|
| `--teammate-mode <mode>` | 에이전트 팀 표시 방식: `auto`, `in-process` 또는 `tmux` |
| `--brief` | 에이전트 대 사용자 간 통신을 위한 `SendUserMessage` 도구 활성화 |

### --allowedTools / --disallowedTools를 위한 도구 이름 구문
```
Read                    # 모든 파일 읽기
Edit                    # 파일 편집 (기존 파일)
Write                   # 파일 생성 (새 파일)
Bash                    # 모든 쉘 명령어
Bash(git *)             # git 명령어만
Bash(git commit *)      # git commit 명령어만
Bash(npm run lint:*)    # 와일드카드를 사용한 패턴 매칭
WebSearch               # 웹 검색 기능
WebFetch                # 웹 페이지 가져오기
mcp__<server>__<tool>   # 특정 MCP 도구
```

## 설정 & 구성

### 설정 계층 (우선순위가 높은 순서대로)
1. **CLI 플래그** — 모든 것을 덮어씁니다.
2. **로컬 프로젝트:** `.claude/settings.local.json` (개인용, git에 무시됨)
3. **프로젝트:** `.claude/settings.json` (공유용, git 추적 대상)
4. **사용자:** `~/.claude/settings.json` (전역)

### 설정 내 권한(Permissions)
```json
{
  "permissions": {
    "allow": ["Bash(npm run lint:*)", "WebSearch", "Read"],
    "ask": ["Write(*.ts)", "Bash(git push*)"],
    "deny": ["Read(.env)", "Bash(rm -rf *)"]
  }
}
```

### 메모리 파일 (CLAUDE.md) 계층
1. **전역:** `~/.claude/CLAUDE.md` — 모든 프로젝트에 적용
2. **프로젝트:** `./CLAUDE.md` — 프로젝트별 컨텍스트 (git 추적 대상)
3. **로컬:** `.claude/CLAUDE.local.md` — 개인적인 프로젝트 덮어쓰기 (git에 무시됨)

대화형 모드에서 접두사 `#`을 사용하면 메모리에 빠르게 추가할 수 있습니다: `# Always use 2-space indentation`.

## 대화형 세션: 슬래시 명령어

### 세션 & 컨텍스트
| 명령어 | 목적 |
|---------|---------|
| `/help` | 모든 명령어 보기 (커스텀 및 MCP 명령어 포함) |
| `/compact [focus]` | 토큰을 절약하기 위해 컨텍스트 압축; CLAUDE.md는 압축 후에도 살아남습니다. 예: `/compact focus on auth logic` |
| `/clear` | 처음부터 시작하기 위해 대화 기록 지우기 |
| `/context` | 최적화 팁과 함께 컨텍스트 사용량을 컬러 그리드로 시각화 |
| `/cost` | 모델별 및 캐시 적중(cache-hit) 내역이 포함된 토큰 사용량 보기 |
| `/resume` | 다른 세션으로 전환하거나 재개 |
| `/rewind` | 대화나 코드의 이전 체크포인트로 되돌리기 |
| `/btw <question>` | 컨텍스트 비용 추가 없이 간단한 질문하기 |
| `/status` | 버전, 연결 및 세션 정보 보기 |
| `/todos` | 대화에서 추적된 작업 항목(action items) 나열 |
| `/exit` 또는 `Ctrl+D` | 세션 종료 |

### 개발 & 리뷰
| 명령어 | 목적 |
|---------|---------|
| `/review` | 현재 변경 사항에 대한 코드 리뷰 요청 |
| `/security-review` | 현재 변경 사항에 대한 보안 분석 수행 |
| `/plan [description]` | 작업 계획을 자동으로 시작하는 Plan 모드 시작 |
| `/loop [interval]` | 세션 내에서 반복되는 작업 예약 |
| `/batch` | 대규모 병렬 변경을 위한 worktree 자동 생성 (5-30개의 worktree) |

### 구성 & 도구
| 명령어 | 목적 |
|---------|---------|
| `/model [model]` | 세션 중간에 모델 전환 (화살표 키를 사용하여 effort 조정) |
| `/effort [level]` | 추론 노력 수준(effort) 설정: `low`, `medium`, `high`, `max`, `auto` |
| `/init` | 프로젝트 메모리를 위한 CLAUDE.md 파일 생성 |
| `/memory` | CLAUDE.md 파일을 열어 편집 |
| `/config` | 대화형 설정 구성 열기 |
| `/permissions` | 도구 권한 보기/업데이트 |
| `/agents` | 특수 하위 에이전트 관리 |
| `/mcp` | MCP 서버를 관리하는 대화형 UI |
| `/add-dir` | 작업 디렉토리 추가 (모노레포에 유용) |
| `/usage` | 요금제 한도 및 요금 제한(rate limit) 상태 보기 |
| `/voice` | 푸시 투 토크(push-to-talk) 음성 모드 활성화 (20개 언어 지원, Space를 길게 눌러 녹음, 떼면 전송) |
| `/release-notes` | 버전 릴리스 노트를 위한 대화형 선택기 |

### 커스텀 슬래시 명령어
`.claude/commands/<name>.md` (프로젝트 공유) 또는 `~/.claude/commands/<name>.md` (개인용)를 생성하세요:

```markdown
# .claude/commands/deploy.md
Run the deploy pipeline:
1. Run all tests
2. Build the Docker image
3. Push to registry
4. Update the $ARGUMENTS environment (default: staging)
```

사용법: `/deploy production` — `$ARGUMENTS`가 사용자의 입력으로 대체됩니다.

### 스킬 (자연어 호출)
수동으로 호출되는 슬래시 명령어와 달리, `.claude/skills/` 안의 스킬들은 마크다운 가이드이며, 작업이 일치할 때 Claude가 자연어를 통해 자동으로 호출합니다:

```markdown
# .claude/skills/database-migration.md
When asked to create or modify database migrations:
1. Use Alembic for migration generation
2. Always create a rollback function
3. Test migrations against a local database copy
```

## 대화형 세션: 키보드 단축키

### 일반 제어
| 키 | 액션 |
|-----|--------|
| `Ctrl+C` | 현재 입력 또는 생성 취소 |
| `Ctrl+D` | 세션 종료 |
| `Ctrl+R` | 역방향 명령어 히스토리 검색 |
| `Ctrl+B` | 실행 중인 작업을 백그라운드로 전환 |
| `Ctrl+V` | 대화에 이미지 붙여넣기 |
| `Ctrl+O` | 트랜스크립트(Transcript) 모드 — Claude의 사고 과정 보기 |
| `Ctrl+G` 또는 `Ctrl+X Ctrl+E` | 외부 편집기에서 프롬프트 열기 |
| `Esc Esc` | 대화 또는 코드 상태를 되감거나 요약 |

### 모드 전환(Toggles)
| 키 | 액션 |
|-----|--------|
| `Shift+Tab` | 권한 모드 순환 (Normal → Auto-Accept → Plan) |
| `Alt+P` | 모델 전환 |
| `Alt+T` | 생각(thinking) 모드 전환 |
| `Alt+O` | 빠른 모드(Fast Mode) 전환 |

### 여러 줄 입력
| 키 | 액션 |
|-----|--------|
| `\` + `Enter` | 빠른 줄바꿈 |
| `Shift+Enter` | 줄바꿈 (대안) |
| `Ctrl+J` | 줄바꿈 (대안) |

### 입력 접두사(Prefixes)
| 접두사 | 액션 |
|--------|--------|
| `!` | AI를 거치지 않고 bash를 직접 실행 (예: `!npm test`). `!`만 사용하면 셸 모드가 전환됨. |
| `@` | 자동 완성을 사용하여 파일/디렉토리 참조 (예: `@./src/api/`) |
| `#` | CLAUDE.md 메모에 빠르게 추가 (예: `# Use 2-space indentation`) |
| `/` | 슬래시 명령어 |

### Pro Tip: "ultrathink"
특정 턴(turn)에서 최대의 추론 능력이 필요할 때 프롬프트에 "ultrathink" 키워드를 사용하세요. 이는 현재 `/effort` 설정과 관계없이 가장 깊은 생각 모드를 작동시킵니다.

## PR 리뷰 패턴

### 빠른 리뷰 (Print 모드)
```
terminal(command="cd /path/to/repo && git diff main...feature-branch | claude -p 'Review this diff for bugs, security issues, and style problems. Be thorough.' --max-turns 1", timeout=60)
```

### 심층 리뷰 (대화형 + Worktree)
```
terminal(command="tmux new-session -d -s review -x 140 -y 40")
terminal(command="tmux send-keys -t review 'cd /path/to/repo && claude -w pr-review' Enter")
terminal(command="sleep 5 && tmux send-keys -t review Enter")  # 신뢰 대화창
terminal(command="sleep 2 && tmux send-keys -t review 'Review all changes vs main. Check for bugs, security issues, race conditions, and missing tests.' Enter")
terminal(command="sleep 30 && tmux capture-pane -t review -p -S -60")
```

### PR 번호로 리뷰
```
terminal(command="claude -p 'Review this PR thoroughly' --from-pr 42 --max-turns 10", workdir="/path/to/repo", timeout=120)
```

### tmux를 사용한 Claude Worktree
```
terminal(command="claude -w feature-x --tmux", workdir="/path/to/repo")
```
`.claude/worktrees/feature-x`에 격리된 git worktree를 생성하고 이를 위한 tmux 세션을 생성합니다. 사용 가능할 때 iTerm2의 네이티브 창(pane)을 사용하며, 전통적인 tmux를 원할 경우 `--tmux=classic`을 추가하세요.

## 병렬 Claude 인스턴스

서로 독립적인 여러 Claude 작업을 동시에 실행하세요:

```
# 작업 1: 백엔드 수정
terminal(command="tmux new-session -d -s task1 -x 140 -y 40 && tmux send-keys -t task1 'cd ~/project && claude -p \"Fix the auth bug in src/auth.py\" --allowedTools \"Read,Edit\" --max-turns 10' Enter")

# 작업 2: 테스트 작성
terminal(command="tmux new-session -d -s task2 -x 140 -y 40 && tmux send-keys -t task2 'cd ~/project && claude -p \"Write integration tests for the API endpoints\" --allowedTools \"Read,Write,Bash\" --max-turns 15' Enter")

# 작업 3: 문서 업데이트
terminal(command="tmux new-session -d -s task3 -x 140 -y 40 && tmux send-keys -t task3 'cd ~/project && claude -p \"Update README.md with the new API endpoints\" --allowedTools \"Read,Edit\" --max-turns 5' Enter")

# 모두 모니터링
terminal(command="sleep 30 && for s in task1 task2 task3; do echo '=== '$s' ==='; tmux capture-pane -t $s -p -S -5 2>/dev/null; done")
```

## CLAUDE.md — 프로젝트 컨텍스트 파일

Claude Code는 프로젝트 루트에서 `CLAUDE.md`를 자동으로 로드합니다. 프로젝트 컨텍스트를 유지하기 위해 사용하세요:

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

**구체적으로 작성하세요.** "좋은 코드를 작성하라" 대신 "JS에는 2칸 들여쓰기를 사용하라" 또는 "테스트 파일명에 `.test.ts` 접미사를 붙여라"와 같이 쓰세요. 구체적인 지침은 수정에 드는 주기를 절약합니다.

### Rules 디렉토리 (모듈식 CLAUDE.md)
규칙이 많은 프로젝트의 경우 하나의 거대한 CLAUDE.md 대신 rules 디렉토리를 사용하세요:
- **프로젝트 규칙:** `.claude/rules/*.md` — 팀 공유, git 추적 대상
- **사용자 규칙:** `~/.claude/rules/*.md` — 개인용, 전역

rules 디렉토리의 각 `.md` 파일은 추가 컨텍스트로 로드됩니다. 이것이 모든 것을 하나의 CLAUDE.md에 쑤셔 넣는 것보다 훨씬 깔끔합니다.

### 자동 메모리 (Auto-Memory)
Claude는 학습된 프로젝트 컨텍스트를 `~/.claude/projects/<project>/memory/`에 자동으로 저장합니다.
- **제한:** 프로젝트당 25KB 또는 200줄
- 이것은 CLAUDE.md와는 별개입니다 — 여러 세션에 걸쳐 누적된 프로젝트에 대한 Claude의 자체 메모입니다.

## 사용자 정의 하위 에이전트 (Custom Subagents)

특화된 에이전트를 `.claude/agents/` (프로젝트), `~/.claude/agents/` (개인용), 또는 `--agents` CLI 플래그 (세션)를 통해 정의하세요:

### 에이전트 위치 우선순위
1. `.claude/agents/` — 프로젝트 레벨, 팀 공유
2. `--agents` CLI 플래그 — 특정 세션용, 동적
3. `~/.claude/agents/` — 사용자 레벨, 개인용

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

호출 방법: `@security-reviewer review the auth module`

### CLI를 통한 동적 에이전트
```
terminal(command="claude --agents '{\"reviewer\": {\"description\": \"Reviews code\", \"prompt\": \"You are a code reviewer focused on performance\"}}' -p 'Use @reviewer to check auth.py'", timeout=120)
```

Claude는 여러 에이전트를 조율할 수 있습니다: "Use @db-expert to optimize queries, then @security to audit the changes."

## 훅(Hooks) — 이벤트 기반 자동화

`.claude/settings.json` (프로젝트) 또는 `~/.claude/settings.json` (전역)에서 구성합니다:

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
|------|--------------|------------|
| `UserPromptSubmit` | Claude가 사용자 프롬프트를 처리하기 전 | 입력 검증, 로깅 |
| `PreToolUse` | 도구 실행 전 | 보안 게이트, 위험한 명령 차단 (exit 2 = 차단) |
| `PostToolUse` | 도구 완료 후 | 코드 자동 포맷팅, 린터 실행 |
| `Notification` | 권한 요청이나 입력 대기 시 | 데스크톱 알림, 경고 |
| `Stop` | Claude가 응답을 완료했을 때 | 완료 로깅, 상태 업데이트 |
| `SubagentStop` | 하위 에이전트 완료 시 | 에이전트 오케스트레이션 |
| `PreCompact` | 컨텍스트 메모리가 지워지기 전 | 세션 기록(transcript) 백업 |
| `SessionStart` | 세션이 시작될 때 | 개발 컨텍스트 로드 (예: `git status`) |

### 훅 환경 변수
| 변수 | 내용 |
|----------|---------|
| `CLAUDE_PROJECT_DIR` | 현재 프로젝트 경로 |
| `CLAUDE_FILE_PATHS` | 수정 중인 파일 |
| `CLAUDE_TOOL_INPUT` | JSON 형태의 도구 매개변수 |

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

데이터베이스, API 및 서비스를 위한 외부 도구 서버를 추가하세요:

```
# GitHub 통합
terminal(command="claude mcp add -s user github -- npx @modelcontextprotocol/server-github", timeout=30)

# PostgreSQL 쿼리
terminal(command="claude mcp add -s local postgres -- npx @anthropic-ai/server-postgres --connection-string postgresql://localhost/mydb", timeout=30)

# 웹 테스트용 Puppeteer
terminal(command="claude mcp add puppeteer -- npx @anthropic-ai/server-puppeteer", timeout=30)
```

### MCP 범위 (Scopes)
| 플래그 | 범위 | 저장 위치 |
|------|-------|---------|
| `-s user` | 전역 (모든 프로젝트) | `~/.claude.json` |
| `-s local` | 이 프로젝트 (개인용) | `.claude/settings.local.json` (git에 무시됨) |
| `-s project` | 이 프로젝트 (팀 공유) | `.claude/settings.json` (git 추적 대상) |

### Print/CI 모드에서의 MCP
```
terminal(command="claude --bare -p 'Query database' --mcp-config mcp-servers.json --strict-mcp-config", timeout=60)
```
`--strict-mcp-config`는 `--mcp-config`의 것을 제외한 모든 MCP 서버를 무시합니다.

채팅에서 MCP 리소스를 참조하세요: `@github:issue://123`

### MCP 한도 & 튜닝
- **도구 설명:** 도구 설명 및 서버 지침에 대한 서버당 2KB 한도
- **결과 크기:** 기본적으로 한도가 정해져 있음; 큰 출력을 위해 **500K** 문자까지 허용하려면 `maxResultSizeChars` 주석 사용
- **출력 토큰:** `export MAX_MCP_OUTPUT_TOKENS=50000` — 컨텍스트 초과(flooding)를 방지하기 위해 MCP 서버 출력 제한
- **전송 방식(Transports):** `stdio` (로컬 프로세스), `http` (원격), `sse` (서버 전송 이벤트)

## 대화형 세션 모니터링

### TUI 상태 읽기
```
# Claude가 여전히 작업 중인지 또는 입력을 기다리고 있는지 확인하기 위한 주기적인 캡처
terminal(command="tmux capture-pane -t dev -p -S -10")
```

다음 표시를 찾으세요:
- 하단의 `❯` = 입력을 기다리는 중 (Claude가 작업을 마쳤거나 질문 중)
- `●` 라인 = Claude가 적극적으로 도구를 사용 중 (읽기, 쓰기, 명령어 실행)
- `⏵⏵ bypass permissions on` = 권한 모드를 보여주는 상태 표시줄
- `◐ medium · /effort` = 상태 표시줄의 현재 노력(effort) 수준
- `ctrl+o to expand` = 도구 출력이 잘림 (대화형으로 확장 가능)

### 컨텍스트 창(Window) 건강 상태
대화형 모드에서 `/context`를 사용하면 컨텍스트 사용량을 컬러 그리드로 볼 수 있습니다. 주요 임계값:
- **&lt; 70%** — 정상 작동, 전체 정확도(precision)
- **70-85%** — 정확도가 떨어지기 시작함, `/compact` 사용 고려
- **> 85%** — 환각(Hallucination) 위험이 급격히 높아짐, `/compact` 또는 `/clear` 사용

## 환경 변수

| 변수 | 효과 |
|----------|--------|
| `ANTHROPIC_API_KEY` | 인증용 API 키 (OAuth 대안) |
| `CLAUDE_CODE_EFFORT_LEVEL` | 기본 추론 노력 수준: `low`, `medium`, `high`, `max`, `auto` |
| `MAX_THINKING_TOKENS` | 생각 토큰 제한 (생각 모드를 완전히 비활성화하려면 `0`으로 설정) |
| `MAX_MCP_OUTPUT_TOKENS` | MCP 서버 출력 한도 (기본값은 가변적; 예: `50000`으로 설정) |
| `CLAUDE_CODE_NO_FLICKER=1` | 터미널 깜빡임을 없애기 위해 대체 화면(alt-screen) 렌더링 활성화 |
| `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` | 보안을 위해 하위 프로세스에서 자격 증명(credentials) 제거 |

## 비용 & 성능 팁

1. **Print 모드에서 `--max-turns` 사용** — 폭주 루프를 방지합니다. 대부분의 작업에 5-10으로 시작하세요.
2. **비용 제한을 위해 `--max-budget-usd` 사용** — 참고: 시스템 프롬프트 캐시 생성에만 최소 ~$0.05가 듭니다.
3. **간단한 작업에는 `--effort low` 사용** (더 빠르고 저렴). 복잡한 추론에는 `high` 또는 `max`를 사용하세요.
4. **CI/스크립팅에는 `--bare` 사용** — 플러그인/훅 검색 오버헤드를 건너뜁니다.
5. **필요한 것만 제한하려면 `--allowedTools` 사용** (예: 리뷰의 경우 `Read`만 허용).
6. **컨텍스트가 커지면 대화형 세션에서 `/compact` 사용**.
7. **이미 알고 있는 콘텐츠를 분석해야 할 경우** — Claude가 파일을 읽게 하지 말고 입력값을 파이프로 전달하세요.
8. **간단한 작업은 `--model haiku`** (저렴), 복잡한 다단계 작업은 `--model opus`를 사용하세요.
9. **모델 과부하를 우아하게 처리하려면 Print 모드에서 `--fallback-model haiku` 사용**.
10. **독립적인 작업을 위해서는 새 세션을 시작** — 세션은 5시간 동안 지속됩니다; 깨끗한 컨텍스트가 더 효율적입니다.
11. **CI 환경에서는 `--no-session-persistence`를 사용** — 디스크에 저장된 세션이 쌓이는 것을 방지합니다.

## 주의 사항 (Pitfalls & Gotchas)

1. **대화형 모드는 반드시 tmux가 필요합니다** — Claude Code는 완전한 TUI 앱입니다. Hermes 터미널에서 `pty=true`만 사용하는 것도 작동하지만, tmux는 모니터링을 위한 `capture-pane`과 입력을 위한 `send-keys`를 제공하며 이는 오케스트레이션에 필수적입니다.
2. **`--dangerously-skip-permissions` 대화창은 "No, exit"이 기본값입니다** — 수락하려면 반드시 Down을 누르고 Enter를 눌러야 합니다. Print 모드(`-p`)는 이를 완전히 건너뜁니다.
3. **`--max-budget-usd`의 최소 한도는 ~$0.05입니다** — 시스템 프롬프트 캐시 생성 비용만 이 정도입니다. 더 낮게 설정하면 즉시 오류가 발생합니다.
4. **`--max-turns`는 Print 모드 전용입니다** — 대화형 세션에서는 무시됩니다.
5. **Claude는 `python3` 대신 `python`을 사용할 수 있습니다** — `python` 심볼릭 링크가 없는 시스템에서 Claude의 bash 명령어는 첫 번째 시도에서 실패하지만 자체적으로 수정(self-corrects)합니다.
6. **세션을 재개하려면 같은 디렉토리에 있어야 합니다** — `--continue`는 현재 작업 디렉토리의 가장 최근 세션을 찾습니다.
7. **`--json-schema`는 충분한 `--max-turns`를 필요로 합니다** — Claude가 구조화된 출력을 생성하기 전에 파일을 읽어야 하므로 여러 턴(turns)이 소요됩니다.
8. **신뢰(Trust) 대화창은 디렉토리당 한 번만 나타납니다** — 처음 실행 시에만 표시되고 이후에는 캐시됩니다.
9. **백그라운드 tmux 세션은 계속 유지됩니다** — 완료 시 항상 `tmux kill-session -t <name>`으로 정리하세요.
10. **슬래시 명령어(`/commit` 등)는 대화형 모드에서만 작동합니다** — `-p` 모드에서는 작업 내용을 자연어로 설명해야 합니다.
11. **`--bare`는 OAuth를 건너뜁니다** — `ANTHROPIC_API_KEY` 환경 변수 또는 설정 내 `apiKeyHelper`가 필요합니다.
12. **컨텍스트 성능 저하(degradation)는 실제 발생합니다** — AI 출력 품질은 컨텍스트 창 사용량이 70%를 넘으면 측정 가능할 정도로 떨어집니다. `/context`로 모니터링하고 미리 `/compact`를 실행하세요.

## Hermes 에이전트를 위한 규칙

1. **단일 작업에는 Print 모드(`-p`)를 우선 사용하세요** — 더 깔끔하고, 대화창 처리가 필요 없으며, 구조화된 출력을 제공합니다.
2. **다중 턴(multi-turn) 대화형 작업에는 tmux를 사용하세요** — TUI를 오케스트레이션하는 유일하게 안정적인 방법입니다.
3. **항상 `workdir`을 설정하세요** — Claude가 올바른 프로젝트 디렉토리에 집중하도록 합니다.
4. **Print 모드에서는 `--max-turns`를 설정하세요** — 무한 루프와 비용 폭주를 방지합니다.
5. **tmux 세션을 모니터링하세요** — 진행 상황을 확인하려면 `tmux capture-pane -t <session> -p -S -50`을 사용하세요.
6. **`❯` 프롬프트를 찾으세요** — Claude가 입력을 기다리고 있음을 나타냅니다 (완료되었거나 질문이 있는 상태).
7. **tmux 세션을 정리하세요** — 완료되면 리소스 누수를 방지하기 위해 세션을 종료(kill)하세요.
8. **결과를 사용자에게 보고하세요** — 완료 후, Claude가 수행한 작업과 변경된 내용을 요약하세요.
9. **느린 세션을 죽이지 마세요** — Claude가 여러 단계의 작업을 수행 중일 수 있습니다; 대신 진행 상황을 확인하세요.
10. **`--allowedTools`를 사용하세요** — 실제로 작업에 필요한 기능으로 권한을 제한하세요.

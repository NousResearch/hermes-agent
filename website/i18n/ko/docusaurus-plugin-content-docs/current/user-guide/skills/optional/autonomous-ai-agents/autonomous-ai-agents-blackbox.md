---
title: "Blackbox — Blackbox AI 멀티 모델 CLI에 코딩 작업 위임"
sidebar_label: "Blackbox"
description: "Blackbox AI 멀티 모델 CLI에 코딩 작업 위임"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 소스 SKILL.md를 편집하세요. */}

# Blackbox

Blackbox AI 멀티 모델 CLI에 코딩 작업을 위임합니다.

## Skill 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/blackbox`로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/blackbox` |
| 버전 | `1.0.1` |
| 작성자 | Hermes Agent (Nous Research) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Blackbox`, `Multi-Agent`, `Judge`, `Multi-Model` |
| 관련 skills | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 skill이 트리거될 때 Hermes가 로드하는 전체 skill 정의입니다. skill이 활성화되면 에이전트가 보게 되는 내용입니다.
:::

# Blackbox CLI

Hermes 터미널을 통해 [Blackbox AI](https://www.blackbox.ai/)에 코딩 작업을 위임합니다. Blackbox는 여러 LLM(Claude, Codex, Gemini, Blackbox Pro)에 작업을 배정하고 judge를 사용해 최상의 구현을 선택하는 멀티 모델 코딩 에이전트 CLI입니다.

CLI(npm `@blackbox_ai/blackbox-cli`, 바이너리 `blackbox`)는 TypeScript 코딩 에이전트(Gemini CLI에서 포크됨)이며 대화형 세션, 비대화형 단일 실행, 체크포인트, MCP, 비전 모델 전환을 지원합니다.

## 사전 요구 사항

- Node.js 20 이상 설치
- Blackbox CLI 설치: `npm install -g @blackbox_ai/blackbox-cli` (바이너리: `blackbox`)
- [app.blackbox.ai/dashboard](https://app.blackbox.ai/dashboard)의 API 키
- 구성 완료: `blackbox configure`를 실행하고 API 키를 입력
- 터미널 호출에서 `pty=true` 사용 — Blackbox CLI는 대화형 터미널 앱입니다.

## 단일 실행 작업

```
terminal(command="blackbox --prompt 'Add JWT authentication with refresh tokens to the Express API'", workdir="/path/to/project", pty=true)
```

빠른 임시 작업에는 다음을 사용합니다.
```
terminal(command="cd $(mktemp -d) && git init && blackbox --prompt 'Build a REST API for todos with SQLite'", pty=true)
```

## 백그라운드 모드 (장시간 작업)

몇 분이 걸리는 작업은 진행 상황을 모니터링할 수 있도록 백그라운드 모드를 사용하세요.

```
# Start in background with PTY
terminal(command="blackbox --prompt 'Refactor the auth module to use OAuth 2.0'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Blackbox asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## 체크포인트 및 재개

Blackbox CLI에는 작업을 일시 중지하고 재개할 수 있는 체크포인트 기능이 내장되어 있습니다.

```
# After a task completes, Blackbox shows a checkpoint tag
# Resume with a follow-up task:
terminal(command="blackbox --resume-checkpoint 'task-abc123-2026-03-06' --prompt 'Now add rate limiting to the endpoints'", workdir="~/project", pty=true)
```

## 세션 명령

대화형 세션 중에는 다음 명령을 사용합니다.

| 명령 | 동작 |
|--------|--------|
| `/compress` | 토큰을 절약하도록 대화 기록 축소 |
| `/clear` | 기록 삭제 및 새 세션 시작 |
| `/stats` | 현재 토큰 사용량 확인 |
| `Ctrl+C` | 현재 작업 취소 |

## PR 검토

작업 트리를 수정하지 않도록 임시 디렉터리에 복제합니다.

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && blackbox --prompt 'Review this PR against main. Check for bugs, security issues, and code quality.'", pty=true)
```

## 병렬 작업

독립적인 작업에는 여러 Blackbox 인스턴스를 생성합니다.

```
terminal(command="blackbox --prompt 'Fix the login bug'", workdir="/tmp/issue-1", background=true, pty=true)
terminal(command="blackbox --prompt 'Add unit tests for auth'", workdir="/tmp/issue-2", background=true, pty=true)

# Monitor all
process(action="list")
```

## 멀티 모델 모드

Blackbox의 고유 기능은 동일한 작업을 여러 모델로 실행하고 결과를 판정하는 것입니다. `blackbox configure`를 통해 사용할 모델을 구성하세요 — 여러 제공자를 선택하면 CLI가 서로 다른 모델의 결과를 평가하고 Chairman/judge 워크플로로 최상의 결과를 선택합니다.

## 주요 플래그

| 플래그 | 동작 |
|--------|--------|
| `--prompt "task"` (`-p`) | 비대화형 단일 실행 수행 |
| `--resume-checkpoint "tag"` | 태그를 사용해 체크포인트에서 재개 |
| `--yolo` (`-y`) | 모든 작업과 모델 전환을 자동 승인 |
| `--vlm-switch-mode <mode>` | 이미지 처리: `once`, `session`, `persist` |
| `-c, --checkpointing` | 편집 체크포인트 기능 활성화 |
| `blackbox configure` | 설정, 제공자, 모델 변경 |
| `blackbox update` | CLI를 최신 버전으로 업데이트 |
| `blackbox mcp` | MCP 서버 관리 |
| `blackbox extensions` | CLI 확장 기능 관리 |
| `blackbox voice <action>` / `blackbox shortcut` | 음성 입력 및 `b` 단축키 구성 |

## 비전 지원

Blackbox는 입력에서 이미지를 자동으로 감지하고 멀티모달 분석으로 전환할 수 있습니다. VLM 모드는 다음과 같습니다.
- `"once"` — 현재 쿼리에만 모델 전환
- `"session"` — 전체 세션에 모델 전환
- `"persist"` — 현재 모델 유지(전환하지 않음)

## 토큰 제한

`.blackboxcli/settings.json`에서 토큰 사용량을 제어합니다.
```json
{
  "sessionTokenLimit": 32000
}
```

## 규칙

1. **항상 `pty=true` 사용** — Blackbox CLI는 대화형 터미널 앱이며 PTY 없이 실행하면 멈춥니다.
2. **`workdir` 사용** — 에이전트가 올바른 작업에 집중하도록 합니다.
3. **장시간 작업에는 백그라운드 사용** — `background=true`로 실행하고 `process` 도구로 모니터링합니다.
4. **방해하지 않기** — `poll`/`log`로 모니터링하고 느리다는 이유로 세션을 종료하지 마세요.
5. **결과 보고** — 완료 후 무엇이 변경되었는지 확인하고 사용자에게 요약합니다.
6. **크레딧에는 비용이 듭니다** — Blackbox는 크레딧 기반이며 멀티 모델 모드는 크레딧을 더 빠르게 소모합니다.
7. **사전 요구 사항 확인** — 위임을 시도하기 전에 `blackbox` CLI가 설치되어 있는지 확인합니다.

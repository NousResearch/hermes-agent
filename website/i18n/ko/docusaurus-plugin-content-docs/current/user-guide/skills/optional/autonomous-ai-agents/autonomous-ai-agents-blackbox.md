---
title: "Blackbox — 코딩 작업을 Blackbox AI CLI 에이전트에 위임"
sidebar_label: "Blackbox"
description: "코딩 작업을 Blackbox AI CLI 에이전트에 위임"
---

{/* 이 페이지는 website/scripts/generate-skill-docs.py에 의해 스킬의 SKILL.md에서 자동 생성되었습니다. 이 페이지가 아닌 원본 SKILL.md를 편집하세요. */}

# Blackbox

코딩 작업을 Blackbox AI CLI 에이전트에 위임합니다. 여러 LLM을 통해 작업을 실행하고 가장 좋은 결과를 선택하는 내장 심판(judge)이 있는 멀티 모델 에이전트입니다. blackbox CLI와 Blackbox AI API 키가 필요합니다.

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 선택 사항 — `hermes skills install official/autonomous-ai-agents/blackbox` 명령어로 설치 |
| 경로 | `optional-skills/autonomous-ai-agents/blackbox` |
| 버전 | `1.0.0` |
| 작성자 | Hermes Agent (Nous Research) |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Blackbox`, `Multi-Agent`, `Judge`, `Multi-Model` |
| 관련 스킬 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이것은 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Blackbox CLI

Hermes 터미널을 통해 코딩 작업을 [Blackbox AI](https://www.blackbox.ai/)에 위임합니다. Blackbox는 여러 LLM(Claude, Codex, Gemini, Blackbox Pro)으로 작업을 분배하고 심판(judge)을 사용하여 최상의 구현을 선택하는 멀티 모델 코딩 에이전트 CLI입니다.

이 CLI는 [오픈소스](https://github.com/blackboxaicode/cli) (GPL-3.0, TypeScript, Gemini CLI에서 포크됨)이며 대화형 세션, 비대화형 원샷(one-shots), 체크포인트, MCP 및 비전 모델 전환을 지원합니다.

## 전제 조건

- Node.js 20+ 설치됨
- Blackbox CLI 설치됨: `npm install -g @blackboxai/cli`
- 또는 소스에서 설치:
  ```
  git clone https://github.com/blackboxaicode/cli.git
  cd cli && npm install && npm install -g .
  ```
- [app.blackbox.ai/dashboard](https://app.blackbox.ai/dashboard)에서 발급받은 API 키
- 구성: `blackbox configure`를 실행하고 API 키 입력
- 터미널 호출 시 `pty=true` 사용 — Blackbox CLI는 대화형 터미널 앱입니다

## 원샷 작업 (One-Shot Tasks)

```
terminal(command="blackbox --prompt 'Add JWT authentication with refresh tokens to the Express API'", workdir="/path/to/project", pty=true)
```

빠른 스크래치(scratch) 작업용:
```
terminal(command="cd $(mktemp -d) && git init && blackbox --prompt 'Build a REST API for todos with SQLite'", pty=true)
```

## 백그라운드 모드 (긴 작업)

몇 분이 걸리는 작업의 경우 진행 상황을 모니터링할 수 있도록 백그라운드 모드를 사용하세요:

```
# PTY와 함께 백그라운드에서 시작
terminal(command="blackbox --prompt 'Refactor the auth module to use OAuth 2.0'", workdir="~/project", background=true, pty=true)
# session_id 반환

# 진행 상황 모니터링
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Blackbox가 질문을 하면 입력값 보내기
process(action="submit", session_id="<id>", data="yes")

# 필요할 때 강제 종료(Kill)
process(action="kill", session_id="<id>")
```

## 체크포인트 및 재개 (Checkpoints & Resume)

Blackbox CLI에는 작업을 일시 중지하고 재개할 수 있는 체크포인트 지원 기능이 내장되어 있습니다:

```
# 작업이 완료된 후 Blackbox는 체크포인트 태그를 표시합니다
# 후속 작업으로 재개:
terminal(command="blackbox --resume-checkpoint 'task-abc123-2026-03-06' --prompt 'Now add rate limiting to the endpoints'", workdir="~/project", pty=true)
```

## 세션 명령어

대화형 세션 중에는 다음 명령어를 사용하세요:

| 명령어 | 효과 |
|---------|--------|
| `/compress` | 토큰을 절약하기 위해 대화 기록 축소 |
| `/clear` | 기록을 지우고 새로 시작 |
| `/stats` | 현재 토큰 사용량 보기 |
| `Ctrl+C` | 현재 작업 취소 |

## PR 리뷰

작업 트리 수정을 피하기 위해 임시 디렉토리에 복제합니다:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && blackbox --prompt 'Review this PR against main. Check for bugs, security issues, and code quality.'", pty=true)
```

## 병렬 작업

독립적인 작업을 위해 여러 Blackbox 인스턴스를 생성합니다:

```
terminal(command="blackbox --prompt 'Fix the login bug'", workdir="/tmp/issue-1", background=true, pty=true)
terminal(command="blackbox --prompt 'Add unit tests for auth'", workdir="/tmp/issue-2", background=true, pty=true)

# 모두 모니터링
process(action="list")
```

## 멀티 모델 모드

Blackbox의 고유한 기능은 동일한 작업을 여러 모델을 통해 실행하고 결과를 판단(judge)하는 것입니다. `blackbox configure`를 통해 사용할 모델을 구성합니다 — 여러 제공자를 선택하여 CLI가 다른 모델의 출력을 평가하고 가장 좋은 것을 선택하는 심판/위원장(Chairman/judge) 워크플로우를 활성화하세요.

## 주요 플래그

| 플래그 | 효과 |
|------|--------|
| `--prompt "task"` | 비대화형 원샷 실행 |
| `--resume-checkpoint "tag"` | 저장된 체크포인트에서 재개 |
| `--yolo` | 모든 작업 및 모델 전환 자동 승인 |
| `blackbox session` | 대화형 채팅 세션 시작 |
| `blackbox configure` | 설정, 제공자, 모델 변경 |
| `blackbox info` | 시스템 정보 표시 |

## 비전(Vision) 지원

Blackbox는 입력된 이미지를 자동으로 감지하고 멀티모달 분석으로 전환할 수 있습니다. VLM 모드:
- `"once"` — 현재 쿼리에 대해서만 모델 전환
- `"session"` — 전체 세션에 대해 전환
- `"persist"` — 현재 모델 유지 (전환 안 함)

## 토큰 한도

`.blackboxcli/settings.json`을 통해 토큰 사용량을 제어합니다:
```json
{
  "sessionTokenLimit": 32000
}
```

## 규칙

1. **항상 `pty=true`를 사용하세요** — Blackbox CLI는 대화형 터미널 앱이므로 PTY 없이는 멈춥니다.
2. **`workdir`를 사용하세요** — 에이전트가 올바른 디렉토리에 집중하게 하세요.
3. **긴 작업에는 백그라운드** — `background=true`를 사용하고 `process` 도구로 모니터링하세요.
4. **간섭하지 마세요** — `poll`/`log`로 모니터링하고 속도가 느리다는 이유로 세션을 강제 종료하지 마세요.
5. **결과 보고** — 완료 후 변경된 사항을 확인하고 사용자를 위해 요약하세요.
6. **크레딧은 돈입니다** — Blackbox는 크레딧 기반 시스템을 사용합니다. 멀티 모델 모드는 크레딧을 더 빨리 소비합니다.
7. **전제 조건 확인** — 위임을 시도하기 전에 `blackbox` CLI가 설치되어 있는지 확인하세요.

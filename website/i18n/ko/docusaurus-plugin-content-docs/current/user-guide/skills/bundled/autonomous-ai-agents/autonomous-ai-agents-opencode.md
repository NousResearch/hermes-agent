---
title: "Opencode — Delegate coding to OpenCode CLI (features, PR review)"
sidebar_label: "Opencode"
description: "Delegate coding to OpenCode CLI (features, PR review)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Opencode

코딩 작업을 OpenCode CLI에 위임하기 (기능 구현, PR 리뷰).

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/autonomous-ai-agents/opencode` |
| Version | `1.2.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `OpenCode`, `Autonomous`, `Refactoring`, `Code-Review` |
| Related skills | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# OpenCode CLI

[OpenCode](https://opencode.ai)를 Hermes 터미널/프로세스 도구에 의해 오케스트레이션되는 자율 코딩 워커(worker)로 사용합니다. OpenCode는 제공자(provider)에 종속되지 않는 오픈 소스 AI 코딩 에이전트로, TUI 및 CLI를 제공합니다.

## 사용 시기

- 사용자가 명시적으로 OpenCode 사용을 요청할 때
- 외부 코딩 에이전트가 코드를 구현/리팩터링/리뷰하도록 하고 싶을 때
- 진행 상태를 확인하며 장시간 실행되는 코딩 세션이 필요할 때
- 격리된 작업 디렉토리(workdirs)/worktrees에서 작업을 병렬로 실행하고 싶을 때

## 사전 요구 사항

- OpenCode 설치됨: `npm i -g opencode-ai@latest` 또는 `brew install anomalyco/tap/opencode`
- 인증 구성됨: `opencode auth login` 또는 제공자 환경 변수(OPENROUTER_API_KEY 등) 설정
- 확인: `opencode auth list`가 하나 이상의 제공자를 표시해야 함
- 코드 작업을 위한 Git 저장소 (권장)
- 대화형 TUI 세션을 위한 `pty=true`

## 바이너리 확인 (중요)

쉘(Shell) 환경에 따라 다른 OpenCode 바이너리가 실행될 수 있습니다. 터미널과 Hermes 간의 동작이 다른 경우 다음을 확인하세요:

```
terminal(command="which -a opencode")
terminal(command="opencode --version")
```

필요한 경우 명시적인 바이너리 경로를 고정하세요:

```
terminal(command="$HOME/.opencode/bin/opencode run '...'", workdir="~/project", pty=true)
```

## 일회성 작업 (One-Shot Tasks)

범위가 제한된 비대화형 작업의 경우 `opencode run`을 사용하세요:

```
terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

`-f`를 사용하여 컨텍스트 파일을 첨부하세요:

```
terminal(command="opencode run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

`--thinking`을 사용하여 모델의 사고(thinking) 과정을 확인하세요:

```
terminal(command="opencode run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

특정 모델 강제 지정:

```
terminal(command="opencode run 'Refactor auth module' --model openrouter/anthropic/claude-sonnet-4", workdir="~/project")
```

## 대화형 세션 (백그라운드)

여러 번의 상호 작용이 필요한 반복적인 작업의 경우 TUI를 백그라운드에서 시작하세요:

```
terminal(command="opencode", workdir="~/project", background=true, pty=true)
# session_id 반환

# 프롬프트 전송
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")

# 진행 상태 모니터링
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# 후속 입력 전송
process(action="submit", session_id="<id>", data="Now add error handling for token expiry")

# 깔끔하게 종료 — Ctrl+C
process(action="write", session_id="<id>", data="\x03")
# 또는 단순히 프로세스 강제 종료
process(action="kill", session_id="<id>")
```

**중요:** `/exit`를 사용하지 마십시오 — 유효한 OpenCode 명령어가 아니며 대신 에이전트 선택 대화창이 열립니다. 종료하려면 Ctrl+C (`\x03`) 또는 `process(action="kill")`을 사용하세요.

### TUI 키 바인딩 (Keybindings)

| 키 | 액션 |
|-----|--------|
| `Enter` | 메시지 제출 (필요한 경우 두 번 누르기) |
| `Tab` | 에이전트 전환 (build/plan) |
| `Ctrl+P` | 명령어 팔레트(command palette) 열기 |
| `Ctrl+X L` | 세션 전환 |
| `Ctrl+X M` | 모델 전환 |
| `Ctrl+X N` | 새 세션 |
| `Ctrl+X E` | 에디터 열기 |
| `Ctrl+C` | OpenCode 종료 |

### 세션 재개 (Resuming Sessions)

종료 후 OpenCode는 세션 ID를 출력합니다. 다음 방법으로 재개할 수 있습니다:

```
terminal(command="opencode -c", workdir="~/project", background=true, pty=true)  # 마지막 세션 계속
terminal(command="opencode -s ses_abc123", workdir="~/project", background=true, pty=true)  # 특정 세션
```

## 자주 사용하는 플래그 (Common Flags)

| 플래그 | 용도 |
|------|-----|
| `run 'prompt'` | 일회성 실행 및 종료 |
| `--continue` / `-c` | 마지막 OpenCode 세션 계속하기 |
| `--session <id>` / `-s` | 특정 세션 계속하기 |
| `--agent <name>` | OpenCode 에이전트 선택 (build 또는 plan) |
| `--model provider/model` | 특정 모델 강제 지정 |
| `--format json` | 기계가 읽을 수 있는 출력/이벤트 (Machine-readable) |
| `--file <path>` / `-f` | 메시지에 파일 첨부 |
| `--thinking` | 모델의 사고(thinking) 블록 표시 |
| `--variant <level>` | 추론(Reasoning) 노력 수준 (high, max, minimal) |
| `--title <name>` | 세션 이름 지정 |
| `--attach <url>` | 실행 중인 opencode 서버에 연결 |

## 절차 (Procedure)

1. 도구 준비 상태 확인:
   - `terminal(command="opencode --version")`
   - `terminal(command="opencode auth list")`
2. 제한된 작업의 경우 `opencode run '...'`을 사용합니다 (pty 필요 없음).
3. 반복적인 작업의 경우 `background=true, pty=true`와 함께 `opencode`를 시작합니다.
4. 장기 작업은 `process(action="poll"|"log")`를 사용하여 모니터링합니다.
5. OpenCode가 입력을 요구하면 `process(action="submit", ...)`을 통해 응답합니다.
6. `process(action="write", data="\x03")` 또는 `process(action="kill")`로 종료합니다.
7. 파일 변경 사항, 테스트 결과 및 다음 단계를 요약하여 사용자에게 다시 알려줍니다.

## PR 리뷰 워크플로우

OpenCode에는 내장된 PR 명령어가 있습니다:

```
terminal(command="opencode pr 42", workdir="~/project", pty=true)
```

또는 격리를 위해 임시 복제본에서 리뷰를 진행합니다:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')", pty=true)
```

## 병렬 작업 패턴 (Parallel Work Pattern)

충돌을 피하기 위해 분리된 작업 디렉토리(workdirs/worktrees)를 사용하세요:

```
terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## 세션 및 비용 관리

과거 세션 나열:

```
terminal(command="opencode session list")
```

토큰 사용량 및 비용 확인:

```
terminal(command="opencode stats")
terminal(command="opencode stats --days 7 --models anthropic/claude-sonnet-4")
```

## 주의 사항 (Pitfalls)

- 대화형 `opencode` (TUI) 세션은 `pty=true`가 필요합니다. `opencode run` 명령어는 pty가 필요하지 **않습니다**.
- `/exit`는 유효한 명령어가 아닙니다 — 에이전트 선택 대화창이 열립니다. TUI를 종료하려면 Ctrl+C를 사용하세요.
- PATH가 일치하지 않으면 잘못된 OpenCode 바이너리/모델 구성이 선택될 수 있습니다.
- OpenCode가 멈춘 것 같다면 강제 종료하기 전에 로그를 검사하세요:
  - `process(action="log", session_id="<id>")`
- 병렬 OpenCode 세션 간에 하나의 작업 디렉토리를 공유하지 마세요.
- TUI에서 제출하려면 Enter 키를 두 번 눌러야 할 수 있습니다 (한 번은 텍스트 완료, 한 번은 전송).

## 검증 (Verification)

스모크 테스트 (Smoke test):

```
terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'")
```

성공 기준:
- 출력에 `OPENCODE_SMOKE_OK`가 포함됨
- 제공자/모델 오류 없이 명령어가 종료됨
- 코드 작업의 경우: 예상된 파일이 변경되고 테스트를 통과함

## 규칙 (Rules)

1. 일회성(one-shot) 자동화의 경우 `opencode run`을 우선 사용하세요 — 더 간단하고 pty가 필요하지 않습니다.
2. 반복이 필요할 때만 대화형 백그라운드 모드를 사용하세요.
3. 항상 OpenCode 세션의 범위를 단일 저장소/작업 디렉토리(workdir)로 지정하세요.
4. 긴 작업의 경우 `process` 로그를 통해 진행 상황 업데이트를 제공하세요.
5. 구체적인 결과(변경된 파일, 테스트 결과, 남은 위험 요소)를 보고하세요.
6. 대화형 세션은 절대로 `/exit`를 사용하지 말고 Ctrl+C 또는 kill 명령으로 종료하세요.

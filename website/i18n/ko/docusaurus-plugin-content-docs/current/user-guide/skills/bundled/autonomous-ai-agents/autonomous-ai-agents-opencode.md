---
title: "Opencode — OpenCode CLI에 코딩 위임하기 (기능, PR 검토)"
sidebar_label: "Opencode"
description: "OpenCode CLI에 코딩 위임하기 (기능, PR 검토)"
---

{/* 이 페이지는 스킬의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Opencode

OpenCode CLI에 코딩을 위임합니다(기능, PR 검토).

## 스킬 메타데이터

| | |
|---|---|
| 출처 | 기본 포함됨(기본 설치) |
| 경로 | `skills/autonomous-ai-agents/opencode` |
| 버전 | `1.2.0` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `OpenCode`, `Autonomous`, `Refactoring`, `Code-Review` |
| 관련 스킬 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 완전한 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보게 되는 내용입니다.
:::

# OpenCode CLI

Hermes의 터미널/프로세스 도구가 오케스트레이션하는 자율 코딩 작업자로 [OpenCode](https://opencode.ai)를 사용하세요. OpenCode는 TUI와 CLI를 갖춘 공급자 비종속 오픈 소스 AI 코딩 에이전트입니다.

## 사용 시점

- 사용자가 OpenCode 사용을 명시적으로 요청한 경우
- 외부 코딩 에이전트가 코드 구현/리팩터링/검토를 수행하도록 하려는 경우
- 진행 상황을 확인하는 장시간 코딩 세션이 필요한 경우
- 격리된 작업 디렉터리/워크트리에서 병렬 작업을 실행하려는 경우

## 사전 요구 사항

- OpenCode 설치: `npm i -g opencode-ai@latest` 또는 `brew install anomalyco/tap/opencode`
- 인증 구성: `opencode auth login`을 실행하거나 공급자 환경 변수를 설정하세요(OPENROUTER_API_KEY 등).
- 확인: `opencode auth list`에 최소 하나의 공급자가 표시되어야 합니다.
- 코드 작업을 위한 Git 저장소(권장)
- 대화형 TUI 세션에는 `pty=true`

## 바이너리 확인(중요)

셸 환경에 따라 서로 다른 OpenCode 바이너리가 확인될 수 있습니다. 터미널과 Hermes에서 동작이 다르면 다음을 확인하세요:

```
terminal(command="which -a opencode")
terminal(command="opencode --version")
```

필요하면 명시적인 바이너리 경로를 고정하세요:

```
terminal(command="$HOME/.opencode/bin/opencode run '...'", workdir="~/project", pty=true)
```

## 일회성 작업

범위가 정해진 비대화형 작업에는 `opencode run`을 사용하세요:

```
terminal(command="opencode run 'Add retry logic to API calls and update tests'", workdir="~/project")
```

`-f`로 컨텍스트 파일을 첨부하세요:

```
terminal(command="opencode run 'Review this config for security issues' -f config.yaml -f .env.example", workdir="~/project")
```

`--thinking`으로 모델의 사고 과정을 표시하세요:

```
terminal(command="opencode run 'Debug why tests fail in CI' --thinking", workdir="~/project")
```

특정 모델을 강제하세요:

```
terminal(command="opencode run 'Refactor auth module' --model openrouter/anthropic/claude-sonnet-4", workdir="~/project")
```

## 대화형 세션(백그라운드)

여러 차례의 교환이 필요한 반복 작업은 TUI를 백그라운드에서 시작하세요:

```
terminal(command="opencode", workdir="~/project", background=true, pty=true)
# Returns session_id

# Send a prompt
process(action="submit", session_id="<id>", data="Implement OAuth refresh flow and add tests")

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send follow-up input
process(action="submit", session_id="<id>", data="Now add error handling for token expiry")

# Exit cleanly — Ctrl+C
process(action="write", session_id="<id>", data="\x03")
# Or just kill the process
process(action="kill", session_id="<id>")
```

**중요:** `/exit`를 사용하지 마세요. 유효한 OpenCode 명령이 아니며 대신 에이전트 선택기 대화 상자를 엽니다. Ctrl+C(`\x03`)를 사용하거나 `process(action="kill")`로 TUI를 종료하세요.

### TUI 키 바인딩

| 키 | 동작 |
|--------|--------|
| `Enter` | 메시지 제출(필요하면 두 번 누름) |
| `Tab` | 에이전트 전환(build 또는 plan) |
| `Ctrl+P` | 명령 팔레트 열기 |
| `Ctrl+X L` | 세션 전환 |
| `Ctrl+X M` | 모델 전환 |
| `Ctrl+X N` | 새 세션 |
| `Ctrl+X E` | 편집기 열기 |
| `Ctrl+C` | OpenCode 종료 |

### 세션 재개

종료한 뒤 OpenCode가 세션 ID를 출력합니다. 다음과 같이 재개하세요:

```
terminal(command="opencode -c", workdir="~/project", background=true, pty=true)  # Continue last session
terminal(command="opencode -s ses_abc123", workdir="~/project", background=true, pty=true)  # Specific session
```

## 일반 플래그

| 플래그 | 용도 |
|-----|-----|
| `run 'prompt'` | 일회성 실행 후 종료 |
| `--continue` / `-c` | 마지막 OpenCode 세션 계속하기 |
| `--session <id>` / `-s` | 특정 세션 계속하기 |
| `--agent <name>` | OpenCode 에이전트 선택(build 또는 plan) |
| `--model provider/model` | 특정 모델 강제 지정 |
| `--format json` | 기계가 읽을 수 있는 출력/이벤트 |
| `--file <path>` / `-f` | 메시지에 파일 첨부 |
| `--thinking` | 모델 사고 블록 표시 |
| `--variant <level>` | 추론 노력 수준(high, max, minimal) |
| `--title <name>` | 세션 이름 지정 |
| `--attach <url>` | 실행 중인 opencode 서버에 연결 |

## 절차

1. 도구 준비 상태를 확인합니다:
   - `terminal(command="opencode --version")`
   - `terminal(command="opencode auth list")`
2. 범위가 정해진 작업에는 `opencode run '...'`을 사용합니다(`pty` 불필요).
3. 반복 작업에는 `background=true, pty=true`로 `opencode`를 시작합니다.
4. 장시간 작업은 `process(action="poll"|"log")`로 모니터링합니다.
5. OpenCode가 입력을 요청하면 `process(action="submit", ...)`으로 응답합니다.
6. `process(action="write", data="\x03")` 또는 `process(action="kill")`로 종료합니다.
7. 파일 변경, 테스트 결과, 다음 단계를 사용자에게 구체적으로 요약합니다.

## PR 검토 워크플로

OpenCode에는 내장 PR 명령이 있습니다:

```
terminal(command="opencode pr 42", workdir="~/project", pty=true)
```

또는 격리를 위해 임시 클론에서 검토하세요:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && opencode run 'Review this PR vs main. Report bugs, security risks, test gaps, and style issues.' -f $(git diff origin/main --name-only | head -20 | tr '\n' ' ')", pty=true)
```

## 병렬 작업 패턴

충돌을 피하려면 별도의 작업 디렉터리/워크트리를 사용하세요:

```
terminal(command="opencode run 'Fix issue #101 and commit'", workdir="/tmp/issue-101", background=true, pty=true)
terminal(command="opencode run 'Add parser regression tests and commit'", workdir="/tmp/issue-102", background=true, pty=true)
process(action="list")
```

## 세션 및 비용 관리

이전 세션을 나열하세요:

```
terminal(command="opencode session list")
```

토큰 사용량과 비용을 확인하세요:

```
terminal(command="opencode stats")
terminal(command="opencode stats --days 7 --models anthropic/claude-sonnet-4")
```

## 주의 사항

- 대화형 `opencode`(TUI) 세션에는 `pty=true`가 필요합니다. `opencode run` 명령에는 pty가 필요하지 않습니다.
- `/exit`는 유효한 명령이 아니며 에이전트 선택기를 엽니다. Ctrl+C(`\x03`)를 사용해 TUI 세션을 종료하세요.
- PATH 불일치로 잘못된 OpenCode 바이너리/모델 구성이 선택될 수 있습니다.
- OpenCode가 멈춘 것처럼 보이면 종료하기 전에 로그를 확인하세요:
  - `process(action="log", session_id="<id>")`
- 여러 OpenCode 세션에서 하나의 작업 디렉터리를 공유하지 마세요.
- Enter로 제출하려면 두 번 눌러야 할 수 있습니다(한 번은 텍스트 확정, 한 번은 전송).

## 검증

스모크 테스트:

```
terminal(command="opencode run 'Respond with exactly: OPENCODE_SMOKE_OK'")
```

성공 기준:
- 출력에 `OPENCODE_SMOKE_OK`가 포함됩니다.
- 공급자/모델 오류 없이 명령이 종료됩니다.
- 코드 작업의 경우 예상 파일이 변경되고 테스트가 통과합니다.

## 규칙

1. 일회성 자동화에는 `opencode run`을 우선 사용하세요. 더 간단하고 pty가 필요하지 않습니다.
2. 반복이 필요한 경우에만 대화형 백그라운드 모드를 사용하세요.
3. OpenCode 세션의 범위를 하나의 저장소/작업 디렉터리로 항상 제한하세요.
4. 장시간 작업은 `process` 로그로 진행 상황을 제공하세요.
5. 구체적인 결과(변경된 파일, 테스트, 남은 위험)를 보고하세요.
6. 대화형 세션은 Ctrl+C 또는 kill로 종료하고 `/exit`는 사용하지 마세요.

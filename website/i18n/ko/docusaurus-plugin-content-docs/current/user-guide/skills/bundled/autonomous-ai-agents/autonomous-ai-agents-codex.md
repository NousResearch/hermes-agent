---
title: "Codex — OpenAI Codex CLI에 코딩 위임하기(기능, PR)"
sidebar_label: "Codex"
description: "OpenAI Codex CLI에 코딩 위임하기(기능, PR)"
---

{/* 이 페이지는 skill의 SKILL.md에서 website/scripts/generate-skill-docs.py로 자동 생성됩니다. 이 페이지가 아니라 원본 SKILL.md를 편집하세요. */}

# Codex

OpenAI Codex CLI에 코딩 작업(기능, PR)을 위임합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 번들됨(기본 설치) |
| 경로 | `skills/autonomous-ai-agents/codex` |
| 버전 | `1.0.1` |
| 작성자 | Hermes Agent |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `Coding-Agent`, `Codex`, `OpenAI`, `Code-Review`, `Refactoring` |
| 관련 스킬 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참조: 전체 SKILL.md

:::info
다음은 이 스킬이 실행될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되면 에이전트는 이 내용을 지침으로 봅니다.
:::

# Codex CLI

Hermes 터미널을 통해 [Codex](https://github.com/openai/codex)에 코딩 작업을 위임합니다. Codex는 OpenAI의 자율 코딩 에이전트 CLI입니다.

## 사용 시점

- 기능 개발
- 리팩터링
- PR 검토
- 여러 이슈 일괄 수정

codex CLI와 git 저장소가 필요합니다.

## 사전 요구 사항

- Codex 설치: `npm install -g @openai/codex`
- OpenAI 인증 구성: `OPENAI_API_KEY` 또는 Codex CLI 로그인 흐름을 통한 Codex OAuth 자격 증명
  중 하나
- **반드시 git 저장소 안에서 실행해야 합니다** — Codex는 저장소 밖에서 실행되지 않습니다
- 터미널 호출에 `pty=true` 사용 — Codex는 대화형 터미널 앱입니다

Hermes 자체에서는 `model.provider: openai-codex`가 `hermes auth add openai-codex` 이후 `~/.hermes/auth.json`의 Hermes 관리 Codex OAuth를 사용합니다. 독립 실행형 Codex CLI의 경우 유효한 CLI OAuth 세션이 `~/.codex/auth.json`에 있을 수 있으므로, `OPENAI_API_KEY`가 없다는 사실만으로 Codex 인증이 없다고 판단하지 마세요.

## 단일 작업

```
terminal(command="codex exec 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

임시 작업의 경우(Codex에는 git 저장소가 필요함):
```
terminal(command="cd $(mktemp -d) && git init && codex exec 'Build a snake game in Python'", pty=true)
```

## 백그라운드 모드(장시간 작업)

```
# Start in background with PTY
terminal(command="codex exec --sandbox workspace-write 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Codex asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## 주요 플래그

| 플래그 | 효과 |
|------|--------|
| `exec "prompt"` | 단일 실행, 작업이 끝나면 종료 |
| `--sandbox workspace-write` (`-s`) | 샌드박스에서 실행하지만 작업 공간의 파일 변경을 자동 승인(권장 자동 빌드 모드) |
| `--dangerously-bypass-approvals-and-sandbox` | 샌드박스와 승인 없이 실행(가장 빠르지만 가장 위험함; `--yolo`도 숨겨진 별칭으로 작동) |
| `--sandbox danger-full-access` | Codex 샌드박스 없음; 호스트 서비스 컨텍스트에서 bubblewrap이 작동하지 않을 때 유용 |

> **지원 중단 예정:** `--full-auto`도 여전히 작동하지만, 실행 중인 CLI에서 대신 `--sandbox workspace-write`를 사용하라는 경고를 표시합니다.

## Hermes 게이트웨이 주의 사항

Hermes 게이트웨이/서비스 컨텍스트(예: Telegram 기반 에이전트 세션)에서 Codex CLI를 호출할 때는, 사용자의 대화형 셸에서는 같은 명령이 작동하더라도 Codex `workspace-write` 샌드박싱이 실패할 수 있습니다. 일반적인 증상은 `setting up uid map: Permission denied` 또는 `loopback: Failed RTM_NEWADDR: Operation not permitted`와 같은 bubblewrap/사용자 네임스페이스 오류입니다.

이 컨텍스트에서는 다음을 우선 사용하세요:

```
codex exec --sandbox danger-full-access "<task>"
```

대신 프로세스 경계를 안전 계층으로 사용하세요. 명시적인 `workdir`, 실행 전 깨끗한 git 상태, 범위를 좁힌 작업 프롬프트, `git diff` 검토, 대상 테스트, 커밋 전 사람/에이전트 확인을 적용해 광범위한 변경을 통제합니다.

## PR 검토

안전한 검토를 위해 임시 디렉터리에 복제하세요:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## worktree를 사용한 병렬 이슈 수정

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Codex in each
terminal(command="codex --sandbox workspace-write exec 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="codex --sandbox workspace-write exec 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## 일괄 PR 검토

```
# Fetch all PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# Review multiple PRs in parallel
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86'", workdir="~/project", background=true, pty=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87'", workdir="~/project", background=true, pty=true)

# Post results
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## 규칙

1. **항상 `pty=true` 사용** — Codex는 대화형 터미널 앱이므로 PTY 없이 실행하면 멈춥니다
2. **git 저장소 필요** — Codex는 git 디렉터리 밖에서 실행되지 않습니다. 임시 작업에는 `mktemp -d && git init`을 사용하세요
3. **단일 작업에는 `exec` 사용** — `codex exec "prompt"`는 깔끔하게 실행 후 종료됩니다
4. **빌드에는 `--sandbox workspace-write` 사용** — 샌드박스 내 변경을 자동 승인합니다(`--full-auto`는 지원 중단 예정)
5. **장시간 작업에는 백그라운드 사용** — `background=true`로 실행하고 `process` 도구로 모니터링하세요
6. **방해하지 않기** — `poll`/`log`로 모니터링하고 장시간 작업을 기다리세요
7. **병렬 실행 가능** — 여러 Codex 프로세스를 동시에 실행해도 됩니다

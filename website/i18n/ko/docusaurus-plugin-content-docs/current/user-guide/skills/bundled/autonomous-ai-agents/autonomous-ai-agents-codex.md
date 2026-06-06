---
title: "Codex — Delegate coding to OpenAI Codex CLI (features, PRs)"
sidebar_label: "Codex"
description: "Delegate coding to OpenAI Codex CLI (features, PRs)"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Codex

코딩 작업을 OpenAI Codex CLI에 위임하기 (기능 구현, PR).

## 스킬 메타데이터

| | |
|---|---|
| Source | 번들 (기본 설치) |
| Path | `skills/autonomous-ai-agents/codex` |
| Version | `1.0.0` |
| Author | Hermes Agent |
| License | MIT |
| Platforms | linux, macos, windows |
| Tags | `Coding-Agent`, `Codex`, `OpenAI`, `Code-Review`, `Refactoring` |
| Related skills | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`hermes-agent`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-hermes-agent) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 이 스킬이 활성화되었을 때 에이전트가 지시 사항으로 보는 내용입니다.
:::

# Codex CLI

Hermes 터미널을 통해 코딩 작업을 [Codex](https://github.com/openai/codex)에 위임합니다. Codex는 OpenAI의 자율 코딩 에이전트 CLI입니다.

## 사용 시기

- 기능 구축
- 리팩터링
- PR 리뷰
- 대량 이슈 수정 (Batch issue fixing)

codex CLI와 git 저장소가 필요합니다.

## 사전 요구 사항

- Codex 설치: `npm install -g @openai/codex`
- OpenAI 인증 구성: `OPENAI_API_KEY` 설정 또는 Codex CLI 로그인 절차를 통한 Codex OAuth 자격 증명 사용
- **반드시 git 저장소 안에서 실행해야 함** — Codex는 저장소가 아닌 곳에서는 실행을 거부합니다.
- 터미널 호출 시 `pty=true` 사용 — Codex는 대화형 터미널 앱입니다.

Hermes 자체의 경우, `hermes auth add openai-codex` 명령 실행 후 `model.provider: openai-codex`는 `~/.hermes/auth.json`에서 Hermes가 관리하는 Codex OAuth를 사용합니다. 독립 실행형 Codex CLI의 경우, 유효한 CLI OAuth 세션이 `~/.codex/auth.json`에 있을 수 있습니다. 따라서 `OPENAI_API_KEY`가 없다는 것만으로 Codex 인증이 없다고 간주하지 마세요.

## 일회성 작업 (One-Shot Tasks)

```
terminal(command="codex exec 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

연습용이나 스크래치 작업의 경우 (Codex는 git 저장소가 필요합니다):
```
terminal(command="cd $(mktemp -d) && git init && codex exec 'Build a snake game in Python'", pty=true)
```

## 백그라운드 모드 (장기 작업)

```
# PTY와 함께 백그라운드에서 시작
terminal(command="codex exec --full-auto 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# session_id 반환

# 진행 상태 모니터링
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Codex가 질문할 경우 입력 전송
process(action="submit", session_id="<id>", data="yes")

# 필요한 경우 강제 종료
process(action="kill", session_id="<id>")
```

## 주요 플래그

| 플래그 | 효과 |
|------|--------|
| `exec "prompt"` | 일회성 실행, 완료 시 종료됨 |
| `--full-auto` | 샌드박스 환경에서 실행되지만 작업 공간 내의 파일 변경을 자동 승인함 |
| `--yolo` | 샌드박스 및 승인 없이 실행 (가장 빠르지만 가장 위험함) |

## PR 리뷰

안전한 리뷰를 위해 임시 디렉토리에 클론합니다:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## Worktree를 사용한 병렬 이슈 수정

```
# worktree 생성
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# 각 worktree에서 Codex 실행
terminal(command="codex --yolo exec 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="codex --yolo exec 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# 모니터링
process(action="list")

# 완료 후 푸시 및 PR 생성
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# 정리
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## 대량 PR 리뷰

```
# 모든 PR refs 가져오기
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# 여러 PR을 병렬로 리뷰
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86'", workdir="~/project", background=true, pty=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87'", workdir="~/project", background=true, pty=true)

# 리뷰 결과 게시
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## 규칙

1. **항상 `pty=true`를 사용하세요** — Codex는 대화형 터미널 앱이며 PTY 없이는 중단(hang)됩니다.
2. **Git 저장소가 필요합니다** — Codex는 git 디렉토리 외부에서는 실행되지 않습니다. 임시 작업에는 `mktemp -d && git init`를 사용하세요.
3. **일회성 작업에는 `exec`를 사용하세요** — `codex exec "prompt"`는 실행 후 깔끔하게 종료됩니다.
4. **빌드에는 `--full-auto`를 사용하세요** — 샌드박스 내에서 변경 사항을 자동으로 승인합니다.
5. **장기 작업에는 백그라운드 모드를 사용하세요** — `background=true`를 사용하고 `process` 도구로 모니터링하세요.
6. **간섭하지 마세요** — `poll`/`log`로 모니터링하고, 오래 실행되는 작업은 인내심을 갖고 기다리세요.
7. **병렬 처리가 가능합니다** — 대량 작업을 위해 여러 Codex 프로세스를 동시에 실행하세요.

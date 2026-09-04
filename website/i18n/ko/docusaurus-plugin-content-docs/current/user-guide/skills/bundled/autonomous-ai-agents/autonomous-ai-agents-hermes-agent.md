---
title: "Hermes Agent — Hermes Agent 사용, 구성, 테마 지정, 확장 및 오케스트레이션"
sidebar_label: "Hermes Agent"
description: "Hermes Agent를 사용하고, 구성하고, 테마를 지정하고, 확장하고, 오케스트레이션합니다"
---

{/* This page is auto-generated from the skill's SKILL.md by website/scripts/generate-skill-docs.py. Edit the source SKILL.md, not this page. */}

# Hermes Agent

Hermes Agent를 사용하고, 구성하고, 테마를 지정하고, 확장하고, 오케스트레이션합니다.

## 스킬 메타데이터

| | |
|---|---|
| 소스 | 기본 제공 (기본 설치) |
| 경로 | `skills/autonomous-ai-agents/hermes-agent` |
| 버전 | `3.1.0` |
| 작성자 | Hermes Agent + Teknium |
| 라이선스 | MIT |
| 플랫폼 | linux, macos, windows |
| 태그 | `hermes`, `setup`, `configuration`, `multi-agent`, `spawning`, `cli`, `gateway`, `themes`, `skins`, `desktop-plugins`, `tui-widgets`, `petdex`, `development` |
| 관련 스킬 | [`claude-code`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-claude-code), [`codex`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-codex), [`opencode`](/docs/user-guide/skills/bundled/autonomous-ai-agents/autonomous-ai-agents-opencode) |

## 참고: 전체 SKILL.md

:::info
다음은 이 스킬이 트리거될 때 Hermes가 로드하는 전체 스킬 정의입니다. 스킬이 활성화되었을 때 에이전트가 지침으로 보는 내용입니다.
:::

# Hermes Agent

Hermes Agent는 Nous Research가 만든 오픈 소스 AI 에이전트 프레임워크로, 터미널, 네이티브 데스크톱 앱, 메시징 플랫폼 및 IDE에서 실행됩니다. Claude Code (Anthropic), Codex (OpenAI), OpenClaw와 같은 범주에 속하며, 도구 호출을 사용해 시스템과 상호작용하는 자율 코딩 및 작업 실행 에이전트입니다. Hermes는 모든 LLM 제공자(OpenRouter, Anthropic, OpenAI, Google, DeepSeek, xAI, 로컬 모델 및 20개 이상의 기타 제공자)와 작동하며 Linux, macOS, Windows 및 WSL에서 실행됩니다.

Hermes를 차별화하는 점:

- **스킬을 통한 자체 개선** — Hermes는 재사용 가능한 절차를 스킬로 저장해 경험에서 학습하고, 이후 세션에 로드합니다.
- **세션 간 영구 메모리** — 사용자가 누구인지, 선호 사항, 환경 세부 정보 및 학습한 내용을 기억합니다. 메모리 백엔드는 플러그 방식으로 연결할 수 있습니다.
- **멀티 플랫폼 게이트웨이** — 동일한 에이전트가 Telegram, Discord, Slack, WhatsApp, iMessage, Signal, Matrix, Teams, Email 및 12개 이상의 플랫폼에서 채팅만이 아니라 전체 도구 액세스와 함께 실행됩니다.
- **다양한 표면** — 동일한 에이전트 코어가 CLI, Ink TUI, 네이티브 Electron 데스크톱 앱, 웹 대시보드 및 IDE용 ACP 서버(VS Code / Zed / JetBrains)를 구동합니다.
- **제공자에 종속되지 않음** — 워크플로 중간에 모델과 제공자를 교체할 수 있으며, 자격 증명 풀이 여러 API 키를 자동으로 순환합니다.
- **프로필** — 구성, 세션, 스킬 및 메모리가 격리된 여러 독립 Hermes 인스턴스를 실행합니다.
- **확장 가능하고 테마 지정 가능** — 플러그인, MCP 서버, 사용자 지정 도구, 웹훅 트리거, cron 예약, 모든 표면의 테마를 지정하는 스킨, 데스크톱 UI 플러그인, TUI 위젯 및 반려동물 마스코트를 지원합니다.

**이 스킬은 허브입니다.** 본문에서는 정체성, 빠른 시작, 생성/오케스트레이션 및 핵심 불변 조건을 다룹니다. 그 외의 모든 내용은 참고 파일에 있습니다 — **답변하기 전에 아래에서 일치하는 참고 파일을 로드하세요**. 본문만으로 세부 질문에 답하지 마세요.

**문서:** https://hermes-agent.nousresearch.com/docs/

## 범위 및 검증

이 스킬은 간결한 운영 가이드이며, 모든 Hermes 기능에 대한 완전한 단일 진실 공급원이 아닙니다. Hermes 기능, 명령 또는 설정이 이 문서나 참고 파일에 언급되지 않았다고 해서 존재하지 않는다고 간주하지 마세요. 부정적인 답변을 하기 전에 실제 저장소와 공식 문서를 확인하세요.

검증에 적합한 대상:

- CLI 명령: `hermes --help`, `hermes <command> --help` 및 `hermes_cli/main.py`
- 사용자 문서: https://hermes-agent.nousresearch.com/docs/
- 소스 트리: https://github.com/NousResearch/hermes-agent

## 빠른 시작

```bash
# Install (shell installer — sets up uv, Python, the venv, and the launcher)
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# Interactive chat (default surface; set display.interface: tui to launch the Ink TUI instead)
hermes

# Single query
hermes chat -q "What is the capital of France?"

# Setup wizard  /  pick model+provider  /  health check
hermes setup
hermes model
hermes doctor

# Other surfaces
hermes desktop                 # launch the native desktop app (alias: hermes gui)
hermes dashboard               # web admin panel + embedded chat
hermes proxy                   # OpenAI-compatible local proxy backed by your OAuth provider
```

## 주요 경로

```
~/.hermes/config.yaml       Main configuration (settings — never secrets)
~/.hermes/.env              API keys and secrets ONLY (under $HERMES_HOME if set)
$HERMES_HOME/skills/        Installed skills
~/.hermes/skins/            Custom themes (see references/themes.md)
~/.hermes/desktop-plugins/  Desktop app UI plugins (see references/desktop-plugins.md)
~/.hermes/tui-widgets/      TUI widget apps (see references/tui-widgets.md)
~/.hermes/pets/             Installed pet mascots (see references/petdex.md)
~/.hermes/state.db          Canonical session store (SQLite + FTS5)
~/.hermes/sessions/         Gateway routing index, request dumps, *.jsonl transcripts
~/.hermes/logs/             Gateway and error logs
~/.hermes/auth.json         OAuth tokens and credential pools
~/.hermes/hermes-agent/     Source code (if git-installed)
```

프로필은 동일한 레이아웃으로 `~/.hermes/profiles/<name>/`을 사용합니다. 프로필이 활성화되면 `$HERMES_HOME`에서 실제 홈을 확인하세요 — `~/.hermes`를 절대 하드코딩하지 마세요.

## 라우팅 표 — 작업에 맞는 참고 파일 로드

| 사용자가 원하는 것... | 로드할 파일 |
|---|---|
| CLI 명령, 하위 명령, 플래그, "X를 어떻게 실행하나요" | `references/cli-reference.md` |
| 세션 내 슬래시 명령 | `references/slash-commands.md` |
| 제공자 설정, API 키, OAuth | `references/providers-and-models.md` |
| config.yaml 섹션, 도구 세트, 음성/STT/TTS | `references/configuration.md` |
| AGENTS.md / .hermes.md / CLAUDE.md 프로젝트 규칙 | `references/project-context-files.md` |
| 시크릿 마스킹, PII, 승인 모드, "권한 재설정" | `references/security-privacy.md` |
| 위임, cron, curator, kanban | `references/background-systems.md` |
| MCP 서버 (추가, 카탈로그, `hermes mcp`) | `references/native-mcp.md` |
| 웹훅 라우트 및 이벤트 기반 실행 | `references/webhooks.md` |
| 사용자 지정 테마/스킨 ("신스웨이브 테마", "금색 ● 변경") | `references/themes.md` + `templates/skin.yaml` |
| 데스크톱 앱 UI 요소 (창, 위젯, ⌘K 명령, 페이지) | `references/desktop-plugins.md` + `templates/plugin.js` |
| 라이브 TUI 패널 또는 모달 위젯 (티커, 시계, 대시보드) | `references/tui-widgets.md` + `templates/clock.mjs` |
| 반려동물 마스코트 — 설치, 선택, 크기 조정, 진단 | `references/petdex.md` |
| Windows 관련 문제 (키 바인딩, WinError 10106, BOM) | `references/windows-quirks.md` |
| 디버깅: 음성, 도구 누락, 게이트웨이, 보조 모델 | `references/troubleshooting.md` |
| 코드 기여: 도구, 슬래시 명령, 테스트 추가 | `references/contributor-guide.md` |
| delegate_task "N으로 제한됨" 보고 | `references/delegate-task-concurrency-diagnosis.md` |
| "앱 X가 Nous Portal 구독/OAuth를 사용할 수 있나요?" | `references/portal-auth-for-third-party-apps.md` |

참고 파일을 로드하지 않아도 적용되는 두 가지 테마 규칙: **스킨은 직접 적용하세요** (`hermes config set display.skin <name>` — 모든 표면이 약 1초 이내에 실시간으로 다시 표시됩니다. 사용자에게 `/skin`을 실행하라고 하지 마세요). 또한 **색상 하나를 조정하려면 ACTIVE 스킨을 편집하세요** (`hermes skin set <key> <hex>`) — 팔레트를 삭제하고 배경을 초기화하는 `default`를 절대 포크하지 마세요.

## 추가 Hermes 인스턴스 생성

추가 Hermes 프로세스는 완전히 독립적인 하위 프로세스로 실행하세요 — 세션, 도구 및 환경이 분리됩니다.

### delegate_task와 이것 중 무엇을 사용할까요

| | `delegate_task` | `hermes` 프로세스 생성 |
|-|-----------------|--------------------------|
| 격리 | 별도 대화, 공유 프로세스 | 완전히 독립적인 프로세스 |
| 실행 시간 | 수분 (부모 루프에 의해 제한) | 수시간/수일 |
| 도구 액세스 | 부모 도구의 하위 집합 | 전체 도구 액세스 |
| 대화형 | 아니요 | 예 (PTY 모드) |
| 사용 사례 | 빠른 병렬 하위 작업 | 장시간 자율 미션 |

### 일회성 모드

```
terminal(command="hermes chat -q 'Research GRPO papers and write summary to ~/research/grpo.md'", timeout=300)

# Background for long tasks:
terminal(command="hermes chat -q 'Set up CI/CD for ~/myapp'", background=true)
```

### 대화형 PTY 모드 (tmux 사용)

Hermes는 실제 터미널을 필요로 하는 prompt_toolkit을 사용합니다. 대화형 생성을 위해 tmux를 사용하세요:

```
# Start
terminal(command="tmux new-session -d -s agent1 -x 120 -y 40 'hermes'", timeout=10)

# Wait for startup, then send a message
terminal(command="sleep 8 && tmux send-keys -t agent1 'Build a FastAPI auth service' Enter", timeout=15)

# Read output
terminal(command="sleep 20 && tmux capture-pane -t agent1 -p", timeout=5)

# Send follow-up
terminal(command="tmux send-keys -t agent1 'Add rate limiting middleware' Enter", timeout=5)

# Exit
terminal(command="tmux send-keys -t agent1 '/exit' Enter && sleep 2 && tmux kill-session -t agent1", timeout=10)
```

### 멀티 에이전트 조정

```
# Agent A: backend
terminal(command="tmux new-session -d -s backend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t backend 'Build REST API for user management' Enter", timeout=15)

# Agent B: frontend
terminal(command="tmux new-session -d -s frontend -x 120 -y 40 'hermes -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t frontend 'Build React dashboard for user management' Enter", timeout=15)

# Check progress, relay context between them
terminal(command="tmux capture-pane -t backend -p | tail -30", timeout=5)
terminal(command="tmux send-keys -t frontend 'Here is the API schema from the backend agent: ...' Enter", timeout=5)
```

### 세션 재개

```
# Resume most recent session
terminal(command="tmux new-session -d -s resumed 'hermes --continue'", timeout=10)

# Resume specific session
terminal(command="tmux new-session -d -s resumed 'hermes --resume 20260225_143052_a1b2c3'", timeout=10)
```

### 팁

- 빠른 하위 작업에는 `delegate_task`를 우선 사용하세요 — 전체 Hermes 프로세스를 생성하는 것보다 오버헤드가 적습니다.
- 코드를 편집하는 에이전트를 생성할 때는 `-w` (worktree 모드)를 사용하세요 — git 충돌을 방지합니다.
- 일회성 모드에는 시간 제한을 설정하세요 — 복잡한 작업은 5~10분이 걸릴 수 있습니다.
- 실행 후 잊어도 되는 작업에는 `hermes chat -q`를 사용하세요 — PTY가 필요하지 않습니다.
- 대화형 생성에는 tmux를 사용하세요 — 원시 PTY 모드에서는 prompt_toolkit 때문에 `\r`과 `\n` 문제가 발생합니다.
- 예약 작업에는 생성 대신 `cronjob` 도구를 사용하세요 — 전달 및 재시도를 처리합니다.
- **"delegate_task가 N으로 제한됨" 보고** — Hermes의 세 가지 실제 제한 경로는 `references/delegate-task-concurrency-diagnosis.md`를 참조하세요. 아무 경로도 실행되지 않았다면 모델이 스스로 제한하면서 이를 "런타임이 제한한다"고 합리화하는 것입니다.
- **"$external_app이 Nous Portal 구독 / OAuth를 사용할 수 있나요?"** — 세 계층(플러그인 대 앱, Portal이 실제로 제공하는 것, 로컬 브로커 프록시 옵션)을 안내하려면 `references/portal-auth-for-third-party-apps.md`를 참조하세요.

## 표면 빠른 안내

- **데스크톱 앱** (`hermes desktop` / `hermes gui`) — macOS/Linux/Windows용 네이티브 Electron 앱: 스트리밍 채팅, 세션 목록, Cmd+K 팔레트, 파일 드래그 앤 드롭, 네이티브 알림, 프로필별 원격 게이트웨이 로그인. UI 플러그인으로 확장하세요 — `references/desktop-plugins.md`.
- **웹 대시보드** (`hermes dashboard`) — 메시징 채널, MCP 카탈로그, 웹훅, 메모리, 프로필 빌더 및 `hermes --tui` 채팅을 포함한 전체 관리 패널. OAuth/토큰 게이트로 보호됩니다.
- **Ink TUI** (`hermes --tui` 또는 `display.interface: tui`) — 도킹된 위젯 앱을 포함한 터미널 UI — `references/tui-widgets.md`.
- **OpenAI 호환 프록시** (`hermes proxy`) — 로그인한 OAuth 제공자를 기반으로 하는 로컬 OpenAI API입니다. Codex CLI, Aider, Cline 또는 모든 스크립트를 여기에 연결하면 API 키가 필요 없습니다.

## 핵심 불변 조건 (로드한 내용과 관계없이 절대 위반하지 마세요)

- **프롬프트 캐싱을 절대 깨뜨리지 마세요** — 대화 중간에 과거 컨텍스트, 도구 세트 또는 시스템 프롬프트를 변경하지 마세요. 유일한 예외는 컨텍스트 압축입니다.
- **메시지 역할 교대** — 두 개의 assistant 또는 두 개의 user 메시지를 연속으로 두지 마세요. `tool` 결과만 반복될 수 있습니다.
- **시크릿은 `.env`, 설정은 `config.yaml`** — 사용자에게 자격 증명이 아닌 설정을 `.env`에 넣으라고 절대 말하지 마세요.
- **프로필을 안전하게 처리하는 경로** — 코드에서는 `get_hermes_home()`, 세션에서 경로를 확인할 때는 `$HERMES_HOME`을 사용하세요.
- **사용자를 대신해 `config.yaml`을 직접 편집하지 마세요** — `hermes config set KEY VAL`을 사용하세요. 들여쓰기가 하나만 어긋나도 파일이 손상되어 실행 중인 게이트웨이가 중단될 수 있습니다.

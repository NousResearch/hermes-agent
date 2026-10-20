---
sidebar_position: 1
title: "CLI 인터페이스"
description: "Hermes Agent 터미널 인터페이스의 명령어, 키 바인딩, 퍼스낼리티 등을 익혀 보세요"
---

# CLI 인터페이스

Hermes Agent의 CLI는 웹 UI가 아닌 완전한 터미널 사용자 인터페이스(TUI)입니다. 여러 줄 편집, 슬래시 명령어 자동 완성, 대화 기록, 작업 중단 및 방향 전환, 도구 출력 스트리밍을 제공합니다. 터미널에서 주로 작업하는 사람을 위해 만들어졌습니다.

:::tip 처음 설정하기
`hermes setup --portal` 명령어 하나면 `hermes chat`을 사용할 준비가 끝납니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

:::tip
Hermes에는 모달 오버레이, 마우스 선택, 비차단 입력을 지원하는 최신 TUI도 포함되어 있습니다. `hermes --tui`로 실행하세요. 자세한 내용은 [TUI](tui.md) 가이드를 참조하세요.
:::

## CLI 실행하기

```bash
# Start an interactive session (default)
hermes

# Single query mode (non-interactive)
hermes chat -q "Hello"

# With a specific model
hermes chat --model "anthropic/claude-sonnet-4"

# With a specific provider
hermes chat --provider nous        # Use Nous Portal
hermes chat --provider openrouter  # Force OpenRouter

# With specific toolsets
hermes chat --toolsets "web,terminal,skills"

# Start with one or more skills preloaded
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# Resume previous sessions
hermes --continue             # Resume the most recent CLI session (-c)
hermes --resume <session_id>  # Resume a specific session by ID (-r)
hermes --resume latest        # Resume the most recent session (same as -c)
hermes --resume latest --in ./dir  # Resume ./dir's latest session, staying in ./dir

# Verbose mode (debug output)
hermes chat --verbose

# Isolated git worktree (for running multiple agents in parallel)
hermes -w                         # Interactive mode in worktree
hermes -w -z "Fix issue #123"     # Single query in worktree
```

### 플러그인 관리

`hermes plugins` 명령어는 동일한 옵트인 워크플로를 통해 네이티브 Hermes 플러그인과 이식 가능한 Agent Plugins v1 패키지를 관리합니다.

```bash
hermes plugins install owner/repository --no-enable
hermes plugins list
hermes plugins enable <plugin-name>
hermes plugins disable <plugin-name>
hermes plugins update <plugin-name>
hermes plugins remove <plugin-name>
```

이식 가능한 패키지는 명시적으로 활성화할 때까지 비활성화된 상태로 유지됩니다. Hermes는 현재 이식 가능한 Agent Skills와 stdio MCP 항목을 로드합니다. 정확히 지원되는 범위와 신뢰 경계는 [플러그인 개발자 가이드](/developer-guide/plugins#portable-agent-plugins-v1-packages)를 참조하세요.

## 인터페이스 구성

<img className="docs-terminal-figure" src="/docs/img/docs/cli-layout.svg" alt="배너, 대화 영역, 고정 입력 프롬프트를 보여 주는 Hermes CLI 레이아웃의 양식화된 미리 보기." />
<p className="docs-figure-caption">불안정한 텍스트 아트 대신 안정적인 문서 그림으로 렌더링한 Hermes CLI 배너, 대화 스트림, 고정 입력 프롬프트입니다.</p>

환영 배너에는 모델, 터미널 백엔드, 작업 디렉터리, 사용 가능한 도구, 설치된 스킬이 한눈에 표시됩니다.

### 상태 표시줄

입력 영역 위에는 다음과 같이 실시간으로 갱신되는 상태 표시줄이 있습니다.

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| 요소 | 설명 |
|---------|-------------|
| 모델 이름 | 현재 모델(26자를 초과하면 잘림) |
| 토큰 수 | 사용한 컨텍스트 토큰 / 최대 컨텍스트 창 |
| 컨텍스트 바 | 색상으로 임계값을 표시하는 시각적 채움 표시기 |
| 비용 | 예상 세션 비용(알 수 없거나 가격이 0인 모델은 `n/a`) |
| 🗜️ N | **컨텍스트 압축 횟수** — 실행 중인 세션이 자동으로 압축된 횟수입니다. 첫 압축이 실행된 후 표시됩니다. |
| ▶ N | **활성 백그라운드 작업** — 현재 세션에서 아직 실행 중인 `/background` 프롬프트의 수입니다. 하나 이상의 작업이 진행 중일 때 표시됩니다. |
| 소요 시간 | 경과한 세션 시간 |
| 세션 제목 | 세션에 제목이 지정되면 가장 오른쪽 끝에 금색 배지로 고정되어 표시됩니다. 긴 제목은 핵심 모델 및 컨텍스트 필드가 밀려나기 전에 잘립니다. |
| ⚠ YOLO | **YOLO 모드 경고** — 시작 시 `hermes --yolo`를 사용했거나 세션 중 `/yolo`를 전환하여 `HERMES_YOLO_MODE`가 켜져 있을 때 표시됩니다. 자동 승인 모드임을 잊지 않도록 배너 줄의 경고와 동일하게 표시됩니다. |

표시줄은 터미널 너비에 맞게 조정됩니다. 76열 이상에서는 전체 레이아웃, 52~75열에서는 간결한 레이아웃, 52열 미만에서는 모델과 소요 시간만 표시되며 활성화된 경우 YOLO 배지도 함께 표시됩니다.

**컨텍스트 색상 표시:**

| 색상 | 임계값 | 의미 |
|---------|-----------|---------|
| 녹색 | < 50% | 여유 공간이 충분함 |
| 노란색 | 50~80% | 가득 차는 중 |
| 주황색 | 80~95% | 한도에 가까워짐 |
| 빨간색 | ≥ 95% | 오버플로에 가까움 — `/compress`를 고려하세요 |

입력 토큰과 출력 토큰을 구분한 카테고리별 비용을 포함한 자세한 내역은 `/usage`를 사용하세요.

`openai-codex` 프로바이더에서는 `/usage`가 ChatGPT 계정에 저장된 사용량 한도 재설정도 표시합니다("You have N resets banked - use /usage reset to activate"). `/usage reset`은 저장된 재설정 하나를 사용하여 5시간 및 주간 한도를 완전히 복원합니다. Hermes는 한도가 소진되지 않은 동안에는 재설정을 사용하지 않습니다(저장된 재설정은 전체 허용량을 복원하므로 일찍 사용하면 낭비됩니다). 그래도 사용하려면 `/usage reset --force`를 전달하세요.

### 세션 재개 표시

이전 세션(`hermes -c` 또는 `hermes --resume <id>`)을 재개하면 배너와 입력 프롬프트 사이에 대화 기록을 간결하게 요약한 "이전 대화" 패널이 표시됩니다. 자세한 내용과 설정은 [세션 — 재개 시 대화 요약](sessions.md#conversation-recap-on-resume)을 참조하세요.

## 키 바인딩

| 키 | 동작 |
|-----|--------|
| `Enter` | 메시지 전송 |
| `Alt+Enter`, `Ctrl+J`, 또는 `Shift+Enter` | 새 줄(여러 줄 입력). `Shift+Enter`는 아래 설명처럼 터미널이 `Enter`와 구분되는 경우에 필요합니다. Windows Terminal에서는 `Alt+Enter`가 터미널에 가로채져 전체 화면 전환으로 동작하므로 대신 `Ctrl+Enter` 또는 `Ctrl+J`를 사용하세요. |
| `Alt+V` | 지원되는 터미널에서 클립보드의 이미지 붙여넣기 |
| `Ctrl+V` | 텍스트를 붙여넣고 클립보드 이미지를 기회가 있을 때 첨부 |
| `Ctrl+B` | 음성 모드가 활성화된 경우 음성 녹음 시작/중지(`voice.record_key`, 기본값: `ctrl+b`) |
| `Ctrl+G` | 현재 입력 버퍼를 `$EDITOR`(vim/nvim/nano/VS Code 등)에서 열기. 저장하고 종료하면 편집한 텍스트가 다음 프롬프트로 전송됩니다 — 길고 여러 단락으로 된 프롬프트에 적합합니다. |
| `Ctrl+X Ctrl+E` | 외부 편집기의 Emacs 스타일 대체 바인딩(`Ctrl+G`와 동일한 동작) |
| `Ctrl+S` | **프롬프트 보관.** 현재 초안을 보관하고 컴포저를 비워 먼저 다른 내용을 보낼 수 있습니다. 빈 컴포저에서 `Ctrl+S`를 다시 누르면 초안을 복원합니다(커서는 끝에 있고 첨부 이미지도 복원됨). 반복해서 누르면 덮어쓰지 않고 스택을 쌓으므로 이전 초안이 조용히 사라지지 않습니다. 두 개 이상 보관된 경우 `Ctrl+S`로 탐색 패널을 열 수 있습니다(`↑`/`↓`로 탐색, `Enter`로 복원, `D`로 삭제, `Esc` 또는 `Ctrl+S`로 닫기). 상태 표시줄의 `📌 N` 배지는 보관된 초안 수를 보여 줍니다. 여러 줄 초안은 빈 줄을 포함하여 정확히 원래 형태로 복원됩니다. 보관 내용은 세션 동안 메모리에만 존재하며 디스크에는 기록되지 않습니다. 초안에는 비밀 정보가 포함되는 경우가 많기 때문입니다. |
| `Ctrl+C` | 에이전트 중단(강제 종료하려면 2초 이내에 두 번 누름) |
| `Ctrl+D` | 종료 |
| `Ctrl+Z` | Hermes를 백그라운드로 일시 중지(Unix 전용). 셸에서 `fg`를 실행하여 재개합니다. |
| `Tab` | 자동 제안(고스트 텍스트) 수락 또는 슬래시 명령어 자동 완성 |
| `!<command>` | **셸 모드** — 모델 턴을 사용하지 않고 직접 셸 명령어 실행(예: `!git status`, `!pytest -x`). 아래를 참조하세요. |

**여러 줄 붙여넣기 미리 보기.** 여러 줄 블록을 붙여넣으면 CLI는 전체 내용을 스크롤백에 쏟아내는 대신 간결한 한 줄 미리 보기(`[pasted: 47 lines, 1,842 chars — press Enter to send]`)를 표시합니다. 전송되는 전체 내용은 그대로이며 이는 표시 방식만 다듬은 것입니다.

### `!` 셸 모드

줄을 `!`로 시작하면 에이전트에 보내는 대신 셸 명령어로 실행합니다.

```
> !git status
> !ls -la
> !pytest -x tests/cli
```

- **비용 없음.** 모델이 호출되지 않으므로 API 호출, 토큰, 지연 시간이 발생하지 않습니다.
- **대화에 들어가지 않음.** 명령어와 출력은 기록에 추가되지 않으므로 컨텍스트가 깨끗하게 유지되고 프롬프트 캐시에도 영향을 주지 않습니다.
- **에이전트의 `terminal` 도구가 실행되는 곳에서 실행됩니다.** 세션 작업 디렉터리를 사용하므로 `!pwd`는 에이전트가 확인하는 경로와 일치합니다.
- **승인은 계속 적용됩니다.** 위험한 명령어(`rm -rf`, `~/.hermes/config.yaml` 쓰기 등)는 에이전트의 `terminal` 도구와 동일한 승인 프롬프트를 거칩니다. `!`는 비용과 지연 시간을 줄이는 방법이지 보안 우회 수단이 아닙니다.
- **0이 아닌 종료 코드가 표시됩니다.** 명령어가 실패하면 출력 뒤에 `! exited <code>`가 표시됩니다.
- `!`만 입력하면 한 줄 사용 안내가 표시됩니다.

셸 모드는 CLI 전용입니다. 게이트웨이 플랫폼(Discord, Telegram, Slack)과 cron 실행에서는 이를 무시합니다. 해당 사용자에게는 이미 각자의 셸이 있기 때문입니다.

**최종 응답의 Markdown 제거.** CLI는 최종 에이전트 응답에서 가장 장황한 Markdown 펜스와 `**굵게**`/`*기울임꼴*` 래퍼를 제거하여 원본 소스가 아닌 읽기 쉬운 터미널 문장으로 표시합니다. 코드 블록과 목록은 유지됩니다. 이는 게이트웨이 플랫폼이나 도구 결과에는 영향을 주지 않으며, 이들은 네이티브 렌더링을 위해 Markdown을 그대로 유지합니다.

## 슬래시 명령어

`/`를 입력하면 자동 완성 드롭다운이 표시됩니다. Hermes는 다양한 CLI 슬래시 명령어, 동적 스킬 명령어, 사용자 정의 빠른 명령어를 지원합니다.

일반적인 예시는 다음과 같습니다.

| 명령어 | 설명 |
|---------|-------------|
| `/help` | 명령어 도움말 표시 |
| `/model` | 현재 모델 표시 또는 변경 |
| `/tools` | 현재 사용 가능한 도구 나열 |
| `/skills browse` | 스킬 허브와 공식 선택적 스킬 탐색 |
| `/background <prompt>` | 별도의 백그라운드 세션에서 프롬프트 실행 |
| `/skin` | 활성 CLI 스킨 표시 또는 전환 |
| `/voice on` | CLI 음성 모드 활성화(녹음하려면 `Ctrl+B` 누름) |
| `/voice tts` | Hermes 응답 음성 재생 전환 |
| `/reasoning high` | 추론 노력 수준 높이기 |
| `/title My Session` | 현재 세션 이름 지정 |
| `/status` | 세션 정보(모델/프로필/토큰/소요 시간)를 표시한 뒤 로컬 **세션 요약** 블록 표시(최근 턴 수, 가장 많이 사용한 도구, 변경한 파일, 최근 사용자 프롬프트 및 어시스턴트 응답). 순수 로컬 계산이며 LLM 호출은 없습니다. |
| `/context [all]` | 시각적 컨텍스트 사용량 분석 — 글리프 블록 그리드와 카테고리별 토큰 표(시스템 프롬프트 / 도구 / 스킬 / 메모리 / 대화 / 여유 공간). `/context all`은 스킬별 및 도구 세트별 비용도 추가합니다. |
| `/sessions` | 클래식 CLI 안에서 바로 대화형 세션 선택기 열기(TUI와 동일한 화면). 입력하여 필터링하고, 방향키로 이동한 뒤 Enter로 재개합니다. |

내장 CLI 및 메시징 명령어 전체 목록은 [슬래시 명령어 참조](../reference/slash-commands.md)를 참조하세요.

설정, 프로바이더, 무음 조정, 메시징/Discord 음성 사용법은 [음성 모드](features/voice-mode.md)를 참조하세요.

:::tip
명령어는 대소문자를 구분하지 않으므로 `/HELP`도 `/help`와 동일하게 동작합니다. 설치된 스킬도 자동으로 슬래시 명령어가 됩니다.
:::

## 빠른 명령어

LLM을 호출하지 않고 셸 명령어를 즉시 실행하는 사용자 정의 명령어를 정의할 수 있습니다. CLI와 메시징 플랫폼(Telegram, Discord 등)에서 모두 작동합니다.

```yaml
# ~/.hermes/config.yaml
quick_commands:
  status:
    type: exec
    command: systemctl status hermes-agent
  gpu:
    type: exec
    command: nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  restart:
    type: alias
    target: /gateway restart
```

그런 다음 어느 채팅에서든 `/status`, `/gpu`, `/restart`를 입력하세요. 더 많은 예시는 [설정 가이드](/user-guide/configuration#quick-commands)를 참조하세요.

## 시작 시 스킬 미리 불러오기

세션에서 사용할 스킬을 이미 알고 있다면 시작할 때 전달하세요.

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes는 첫 턴 전에 지정한 각 스킬을 세션 프롬프트에 로드합니다. 같은 플래그가 대화형 모드와 단일 쿼리 모드에서 모두 작동합니다.

## 스킬 슬래시 명령어

`~/.hermes/skills/`에 설치된 모든 스킬은 자동으로 슬래시 명령어로 등록됩니다. 스킬 이름이 명령어가 됩니다.

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# Just the skill name loads it and lets the agent ask what you need:
/excalidraw
```

## 퍼스낼리티

미리 정의된 퍼스낼리티를 설정하여 에이전트의 말투를 바꿀 수 있습니다.

```
/personality pirate
/personality kawaii
/personality concise
```

기본 제공 퍼스낼리티에는 `helpful`, `concise`, `technical`, `creative`, `teacher`, `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`, `noir`, `uwu`, `philosopher`, `hype`가 있습니다.

기본값(오버레이 없음)으로 돌아가려면 `/personality none`을 사용하세요. `default`와 `neutral`도 사용할 수 있습니다.

`~/.hermes/config.yaml`에서 사용자 정의 퍼스낼리티를 정의할 수도 있습니다.

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # Add your own!
```

## 여러 줄 입력

여러 줄 메시지를 입력하는 방법은 두 가지입니다.

1. **`Alt+Enter`, `Ctrl+J`, 또는 `Shift+Enter`** — 새 줄 삽입
2. **백슬래시 연결** — 줄 끝에 `\`를 입력하여 계속 입력:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

`Ctrl+J`와 백슬래시 연결은 기본적으로 활성화되어 있으며 Claude Code / Codex / OpenCode의 여러 줄 단축키와 일치합니다. iTerm2 같은 지원되는 터미널에서 Hermes는 `Shift+Enter`가 별도의 줄 바꿈 키로 전달되도록 확장 키 보고도 요청합니다. 터미널이 일반 `Enter`에 LF를 보내고 기존의 `Ctrl+J` 제출 방식으로 되돌리고 싶다면 이를 비활성화하세요.

```yaml
# ~/.hermes/config.yaml
display:
  cli_multiline_shortcuts: false
```

:::info
여러 줄 텍스트 붙여넣기도 지원됩니다. 위의 줄 바꿈 키를 사용하거나 내용을 바로 붙여넣으세요.
:::

### Shift+Enter 호환성

대부분의 터미널은 기본적으로 `Enter`와 `Shift+Enter`에 동일한 바이트 시퀀스를 보내므로 애플리케이션이 둘을 구분할 수 없습니다. Hermes는 터미널이 [Kitty 키보드 프로토콜](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) 또는 xterm의 `modifyOtherKeys` 모드를 통해 별도의 시퀀스를 보낼 때만 `Shift+Enter`를 인식합니다.

| 터미널 | 상태 |
|---|---|
| Kitty, foot, WezTerm, Ghostty | 별도의 `Shift+Enter`가 기본적으로 활성화됨 |
| iTerm2(최신 버전), Alacritty, VS Code 터미널, Warp | 설정에서 Kitty 프로토콜을 활성화하면 지원됨 |
| Windows Terminal Preview 1.25+ | 설정에서 Kitty 프로토콜을 활성화하면 지원됨 |
| macOS Terminal.app, 기본 Windows Terminal(안정 버전) | 지원되지 않음 — `Shift+Enter`를 `Enter`와 구분할 수 없음 |

터미널이 둘을 구분할 수 없는 경우에도 `Alt+Enter`와 `Ctrl+J`는 기본적으로 계속 작동합니다. **특히 Windows Terminal에서는 `Alt+Enter`가 터미널에 가로채져(전체 화면 전환) Hermes에 전달되지 않습니다. 줄 바꿈에는 `Ctrl+Enter`(`Ctrl+J`로 전달됨) 또는 `Ctrl+J`를 직접 사용하세요.**

## 턴 중 에이전트 방향 전환

에이전트가 작업하는 동안 새 턴을 시작하지 않고 수정 사항을 보낼 수 있습니다.

- **새 메시지를 입력하고 Enter 누르기** — 수정 사항을 사용하여 활성 턴의 방향 전환
- **`Ctrl+C`** — 현재 작업 중단(강제 종료하려면 2초 이내에 두 번 누름)
- 이미 표시된 도구 작업과 추론은 컨텍스트에 유지됨
- 실행 중인 도구는 수정 사항이 적용되기 전에 안전한 경계에 도달함

### 사용 중 입력 모드

`display.busy_input_mode` 설정 키는 에이전트가 작업하는 동안 Enter를 눌렀을 때의 동작을 제어합니다.

| 모드 | 동작 |
|------|----------|
| `"interrupt"`(기본값) | 메시지가 활성 턴의 방향을 전환합니다. 표시된 추론과 완료된 작업을 보존한 채 모델 생성이 다시 시작되며, 실행 중인 도구가 먼저 완료됩니다. |
| `"queue"` | 메시지가 조용히 대기열에 들어가고 에이전트가 완료된 후 다음 턴으로 전송됩니다. |
| `"steer"` | `/steer`를 통해 현재 실행에 메시지가 주입되며, 다음 도구 호출 후 에이전트에 도착합니다 — 중단이나 새 턴은 없습니다. |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # or "queue" or "interrupt" (default)
```

`"queue"` 모드는 별도의 후속 턴을 준비합니다. `"steer"`는 항상 다음 도구 결과 경계를 기다립니다. 기본 `"interrupt"` 모드는 실행 중인 도구를 취소하지 않으면서 모델 생성 중 더 빠르게 응답합니다. 턴과 포그라운드 작업을 취소하려면 `/stop`을 사용하세요. 알 수 없는 값은 `"interrupt"`로 대체됩니다.

`"steer"`에는 두 가지 자동 대체 동작이 있습니다. 에이전트가 아직 시작되지 않았거나 이미지가 첨부된 경우 메시지가 손실되지 않도록 `"queue"` 동작으로 대체됩니다.

CLI 안에서도 변경할 수 있습니다.

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip 최초 입력 안내
Hermes가 작업 중일 때 처음으로 Enter를 누르면 `/busy` 설정을 설명하는 한 줄 안내가 표시됩니다. 설치당 한 번만 표시되며 `config.yaml`의 `onboarding.seen.busy_input_prompt`가 표시 여부를 기록합니다. 이 팁을 다시 보려면 해당 키를 삭제하세요.
:::

### 백그라운드로 일시 중지

Unix 시스템에서는 **`Ctrl+Z`**를 눌러 다른 터미널 프로세스처럼 Hermes를 백그라운드로 일시 중지할 수 있습니다. 셸에 확인 메시지가 표시됩니다.

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

셸에 `fg`를 입력하면 중단했던 지점 그대로 세션이 재개됩니다. Windows에서는 지원되지 않습니다.

## 도구 진행 상황 표시

CLI는 에이전트가 작업하는 동안 애니메이션 피드백을 표시합니다.

**생각 중 애니메이션**(API 호출 중):
```
  ◜ (｡•́︿•̀｡) pondering... (1.2s)
  ◠ (⊙_⊙) contemplating... (2.4s)
  ✧٩(ˊᗜˋ*)و✧ got it! (3.1s)
```

**도구 실행 피드:**
```
  ┊ 💻 terminal `ls -la` (0.3s)
  ┊ 🔍 web_search (1.2s)
  ┊ 📄 web_extract (2.1s)
```

`/verbose`로 표시 모드를 순환할 수 있습니다: `off → new → all → verbose`. 이 명령어는 메시징 플랫폼에서도 활성화할 수 있습니다. [설정](/user-guide/configuration#display-settings)을 참조하세요.

### 도구 미리 보기 길이

`display.tool_preview_length` 설정 키는 도구 호출 미리 보기 줄(예: 파일 경로, 터미널 명령어)에 표시되는 최대 문자 수를 제어합니다. 기본값은 `0`이며 제한이 없음을 의미하므로 전체 경로와 명령어가 표시됩니다.

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # Truncate tool previews to 80 chars (0 = no limit)
```

좁은 터미널이나 도구 인수에 매우 긴 파일 경로가 포함된 경우 유용합니다.

## 세션 관리

### 세션 재개

CLI 세션을 종료하면 재개 명령어가 출력됩니다.

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

재개 옵션:

```bash
hermes --continue                          # Resume the most recent CLI session
hermes -c                                  # Short form
hermes -c "my project"                     # Resume a named session (latest in lineage)
hermes --resume 20260225_143052_a1b2c3     # Resume a specific session by ID
hermes --resume "refactoring auth"         # Resume by title
hermes --resume latest                     # Resume the most recent session (same as -c)
hermes --resume latest --in ./my-project   # Latest session for ./my-project's workspace
hermes -r 20260225_143052_a1b2c3           # Short form
```

재개하면 SQLite에서 전체 대화 기록을 복원합니다. 에이전트는 이전에 나가지 않은 것처럼 모든 메시지, 도구 호출, 응답을 확인합니다.

채팅 중 `/title My Session Name`을 사용하여 현재 세션 이름을 지정하거나 명령줄에서 `hermes sessions rename <id> <title>`을 사용하세요. 이전 세션을 찾아보려면 `hermes sessions list`를 사용하세요.

### 세션 저장소

CLI 세션은 `~/.hermes/state.db` 아래 Hermes의 SQLite 상태 데이터베이스에 저장됩니다. 데이터베이스에는 다음이 보관됩니다.

- 세션 메타데이터(ID, 제목, 타임스탬프, 토큰 카운터)
- 메시지 기록
- 압축 및 재개 세션 간 계보
- `session_search`에서 사용하는 전문 검색 인덱스

일부 메시징 어댑터는 데이터베이스와 함께 플랫폼별 대화 기록 파일도 보관하지만, CLI는 SQLite 세션 저장소에서 재개합니다.

### 컨텍스트 압축

컨텍스트 한도에 가까워지면 긴 대화가 자동으로 요약됩니다.

```yaml
# In ~/.hermes/config.yaml
compression:
  enabled: true
  threshold: 0.50    # Compress at 50% of context limit by default

# Summarization model configured under auxiliary:
auxiliary:
  compression:
    model: ""  # Leave empty to use the main chat model (default). Or pin a cheap fast model, e.g. "google/gemini-3-flash-preview".
```

압축이 실행되면 중간 턴이 요약되고 처음 3개 턴과 마지막 20개 턴은 항상 보존됩니다.

## 백그라운드 세션

CLI를 계속 사용하면서 별도의 백그라운드 세션에서 프롬프트를 실행할 수 있습니다.

```
/background Analyze the logs in /var/log and summarize any errors from today
```

Hermes는 즉시 작업을 확인하고 프롬프트를 다시 보여 줍니다.

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### 작동 방식

각 `/background` 프롬프트는 데몬 스레드에서 **완전히 별도의 에이전트 세션**을 생성합니다.

- **격리된 대화** — 백그라운드 에이전트는 현재 세션의 기록을 알지 못합니다. 사용자가 제공한 프롬프트만 받습니다.
- **동일한 설정** — 백그라운드 에이전트는 현재 세션의 모델, 프로바이더, 도구 세트, 추론 설정, 대체 모델을 상속합니다.
- **비차단** — 포그라운드 세션은 완전히 대화형으로 유지됩니다. 채팅하고 명령어를 실행하거나 백그라운드 작업을 더 시작할 수도 있습니다.
- **여러 작업** — 여러 백그라운드 작업을 동시에 실행할 수 있습니다. 각각 번호가 매겨진 ID를 받습니다.

### 결과

백그라운드 작업이 완료되면 터미널에 패널로 결과가 표시됩니다.

```
╭─ ⚕ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

작업이 실패하면 대신 오류 알림이 표시됩니다. 설정에서 `display.bell_on_complete`가 활성화되어 있으면 작업 완료 시 터미널 벨이 울립니다.

### 사용 사례

- **장시간 실행되는 조사** — 코드를 작업하는 동안 "/background research the latest developments in quantum error correction" 실행
- **파일 처리** — 계속 대화를 나누면서 "/background analyze all Python files in this repo and list any security issues" 실행
- **병렬 조사** — 여러 백그라운드 작업을 시작하여 서로 다른 관점을 탐색

:::info
백그라운드 세션은 기본 대화 기록에 표시되지 않습니다. 자체 작업 ID(예: `bg_143022_a1b2c3`)를 가진 독립 세션입니다.
:::

## 무음 모드

기본적으로 CLI는 다음과 같은 무음 모드로 실행됩니다.
- 도구의 자세한 로깅 억제
- kawaii 스타일의 애니메이션 피드백 활성화
- 출력물을 깔끔하고 사용자 친화적으로 유지

디버그 출력을 보려면:
```bash
hermes chat --verbose
```

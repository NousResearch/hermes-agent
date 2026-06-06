---
sidebar_position: 1
title: "CLI 인터페이스"
description: "Hermes 에이전트 터미널 인터페이스 마스터하기 — 명령어, 키 바인딩, 성격(personalities) 등"
---

# CLI 인터페이스 (CLI Interface)

Hermes 에이전트의 CLI는 웹 UI가 아닌 완전한 터미널 사용자 인터페이스(TUI)입니다. 여러 줄 편집(multiline editing), 슬래시 명령어 자동 완성, 대화 기록, 중단 및 리디렉션, 스트리밍 도구 출력을 지원합니다. 터미널 환경에 익숙한 사용자를 위해 제작되었습니다.

:::tip 첫 설정
명령어 하나 — `hermes setup --portal` — 이면 `hermes chat`을 실행할 준비가 완료됩니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

:::tip
Hermes는 모달 오버레이, 마우스 선택, 논블로킹(non-blocking) 입력이 포함된 최신 TUI도 함께 제공합니다. `hermes --tui`로 실행하세요 — [TUI](tui.md) 가이드를 참조하세요.
:::

## CLI 실행 (Running the CLI)

```bash
# 대화형 세션 시작 (기본값)
hermes

# 단일 쿼리 모드 (비대화형)
hermes chat -q "Hello"

# 특정 모델 사용
hermes chat --model "anthropic/claude-sonnet-4"

# 특정 제공자(provider) 사용
hermes chat --provider nous        # Nous Portal 사용
hermes chat --provider openrouter  # OpenRouter 강제 사용

# 특정 도구 세트(toolsets) 사용
hermes chat --toolsets "web,terminal,skills"

# 하나 이상의 스킬을 미리 로드하여 시작
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -q "open a draft PR"

# 이전 세션 재개
hermes --continue             # 가장 최근의 CLI 세션 재개 (-c)
hermes --resume <session_id>  # ID로 특정 세션 재개 (-r)

# 자세한 모드 (디버그 출력)
hermes chat --verbose

# 격리된 git 워크트리 (여러 에이전트를 병렬로 실행할 때 유용)
hermes -w                         # 워크트리에서 대화형 모드
hermes -w -q "Fix issue #123"     # 워크트리에서 단일 쿼리
```

## 인터페이스 레이아웃 (Interface Layout)

<img className="docs-terminal-figure" src="/docs/img/docs/cli-layout.svg" alt="Stylized preview of the Hermes CLI layout showing the banner, conversation area, and fixed input prompt." />
<p className="docs-figure-caption">깨지기 쉬운 텍스트 아트 대신 안정적인 문서 이미지로 렌더링된 Hermes CLI 배너, 대화 스트림, 고정 입력 프롬프트.</p>

환영 배너에는 현재 사용 중인 모델, 터미널 백엔드, 작업 디렉토리, 사용 가능한 도구 및 설치된 스킬이 한눈에 표시됩니다.

### 상태 표시줄 (Status Bar)

입력 영역 위에는 항상 표시되는 상태 표시줄이 있으며 실시간으로 업데이트됩니다:

```
 ⚕ claude-sonnet-4-20250514 │ 12.4K/200K │ [██████░░░░] 6% │ $0.06 │ 15m
```

| 요소 | 설명 |
|---------|-------------|
| 모델 이름 (Model name) | 현재 모델 (26자를 초과하면 잘림) |
| 토큰 수 (Token count) | 사용된 컨텍스트 토큰 / 최대 컨텍스트 창 |
| 컨텍스트 막대 (Context bar) | 색상으로 구분된 임계값이 있는 시각적 채우기 표시기 |
| 비용 (Cost) | 예상 세션 비용 (알 수 없거나 무료인 모델의 경우 `n/a`) |
| 🗜️ N | **컨텍스트 압축 횟수** — 실행 중인 세션이 자동 압축된 횟수. 첫 번째 압축이 시작되면 표시됩니다. |
| ▶ N | **활성화된 백그라운드 작업** — 현재 세션에서 아직 실행 중인 `/background` 프롬프트의 개수. 실행 중인 작업이 하나 이상 있을 때마다 나타납니다. |
| 경과 시간 (Duration) | 세션 경과 시간 |
| ⚠ YOLO | **YOLO 모드 경고** — `HERMES_YOLO_MODE`가 켜져 있을 때마다 표시됩니다 (실행 시 `hermes --yolo` 또는 세션 중 `/yolo` 토글). 자동 승인 모드에 있다는 것을 잊지 않도록 배너 줄 경고를 미러링합니다. |

상태 표시줄은 터미널 너비에 맞게 조정됩니다 — 너비가 76열 이상일 경우 전체 레이아웃, 52~75열일 경우 콤팩트 레이아웃, 52열 미만일 경우 최소 레이아웃(모델 + 소요 시간 + 켜져 있을 경우 YOLO 배지)으로 표시됩니다.

**컨텍스트 색상 구분:**

| 색상 | 임계값 | 의미 |
|-------|-----------|---------|
| 녹색 (Green) | < 50% | 충분한 공간 여유 |
| 노란색 (Yellow) | 50–80% | 채워지기 시작함 |
| 주황색 (Orange) | 80–95% | 한계에 근접함 |
| 빨간색 (Red) | ≥ 95% | 오버플로우 임박 — `/compress` 사용 고려 |

항목별 비용(입력 토큰과 출력 토큰 비율)을 포함한 자세한 내역을 보려면 `/usage` 명령어를 사용하세요.

### 세션 재개 디스플레이 (Session Resume Display)

이전 세션을 재개할 때(`hermes -c` 또는 `hermes --resume <id>`), 배너와 입력 프롬프트 사이에 "이전 대화(Previous Conversation)" 패널이 나타나 대화 기록의 간략한 요약을 보여줍니다. 자세한 내용과 구성 방법은 [세션 — 재개 시 대화 요약](sessions.md#conversation-recap-on-resume)을 참조하세요.

## 키 바인딩 (Keybindings)

| 키 | 동작 |
|-----|--------|
| `Enter` | 메시지 전송 |
| `Alt+Enter`, `Ctrl+J`, 또는 `Shift+Enter` | 새 줄 추가 (여러 줄 입력). `Shift+Enter`는 `Enter`와 구분할 수 있는 터미널이 필요합니다 (아래 내용 참조). Windows Terminal에서 `Alt+Enter`는 터미널(전체 화면 전환)에 캡처되므로, 대신 `Ctrl+Enter` 또는 `Ctrl+J`를 사용하세요. |
| `Alt+V` | 터미널에서 지원하는 경우 클립보드에서 이미지를 붙여넣기 |
| `Ctrl+V` | 텍스트를 붙여넣고 기회가 되면 클립보드 이미지를 첨부 |
| `Ctrl+B` | 음성 모드가 켜져 있을 때 음성 녹음 시작/중지 (`voice.record_key`, 기본값: `ctrl+b`) |
| `Ctrl+G` | 현재 입력 버퍼를 `$EDITOR` (vim/nvim/nano/VS Code/등)에서 열기. 저장 후 종료 시 편집한 텍스트를 다음 프롬프트로 전송 — 여러 단락의 긴 프롬프트 작성에 이상적입니다. |
| `Ctrl+X Ctrl+E` | 외부 편집기를 위한 Emacs 스타일의 대체 바인딩 (`Ctrl+G`와 동일한 동작). |
| `Ctrl+C` | 에이전트 작업 중단 (2초 안에 두 번 누르면 강제 종료) |
| `Ctrl+D` | 종료 |
| `Ctrl+Z` | 백그라운드로 Hermes 프로세스 일시 중단 (Unix 시스템 전용). 재개하려면 쉘에서 `fg`를 실행하세요. |
| `Tab` | 자동 제안(고스트 텍스트) 수락 또는 슬래시 명령어 자동 완성 |

**여러 줄 붙여넣기 미리보기 (Multiline paste preview).** 여러 줄 블록을 붙여넣을 때, CLI는 스크롤백에 내용 전체를 쏟아붓는 대신 콤팩트한 한 줄짜리 미리보기(`[pasted: 47 lines, 1,842 chars — press Enter to send]`)를 표시합니다. 실제로 전송되는 것은 전체 내용이며, 화면 표시를 다듬은 것에 불과합니다.

**최종 응답의 마크다운 제거 (Markdown stripping in final responses).** CLI는 에이전트의 *최종* 답변에서 가장 복잡한 마크다운 펜스(fences)와 `**bold**` / `*italic*` 래퍼(wrappers)를 제거하여 터미널에서 읽기 쉬운 산문 형태로 렌더링합니다. 코드 블록과 목록은 보존됩니다. 게이트웨이 플랫폼이나 도구 결과에는 영향을 미치지 않으며 — 이들은 네이티브 렌더링을 위해 마크다운을 유지합니다.

## 슬래시 명령어 (Slash Commands)

자동 완성 드롭다운을 보려면 `/`를 입력하세요. Hermes는 대규모 CLI 슬래시 명령어 세트, 동적 스킬 명령어 및 사용자 정의 빠른 명령어(quick commands)를 지원합니다.

일반적인 예시:

| 명령어 | 설명 |
|---------|-------------|
| `/help` | 명령어 도움말 표시 |
| `/model` | 현재 모델 표시 또는 변경 |
| `/tools` | 현재 사용 가능한 도구 목록 표시 |
| `/skills browse` | 스킬 허브 및 공식 선택 스킬 찾아보기 |
| `/background <prompt>` | 프롬프트를 별도의 백그라운드 세션에서 실행 |
| `/skin` | 활성 CLI 스킨 표시 또는 전환 |
| `/voice on` | CLI 음성 모드 켜기 (녹음하려면 `Ctrl+B` 누름) |
| `/voice tts` | Hermes 답변에 대한 음성 재생 토글 |
| `/reasoning high` | 추론 수준(reasoning effort) 높이기 |
| `/title My Session` | 현재 세션 이름 지정 |
| `/status` | 세션 정보(모델/프로필/토큰/지속 시간)와 함께 로컬 **세션 요약(Session recap)** 블록(최근 턴 수, 가장 많이 사용된 도구, 접근한 파일, 최신 사용자 프롬프트 + 어시스턴트 답변) 표시. 순수 로컬 컴퓨팅이며 LLM 호출 없음. |
| `/sessions` | 클래식 CLI 내에서 바로 인터랙티브 세션 선택기(TUI에서 사용하는 표면과 동일) 열기. 타이핑하여 필터링, 화살표 키로 탐색, Enter로 재개. |

전체 내장 CLI 및 메시징 명령어 목록은 [슬래시 명령어 레퍼런스](../reference/slash-commands.md)를 참조하세요.

설정, 제공자, 무음 튜닝(silence tuning), 메시징/Discord 음성 사용에 관한 내용은 [음성 모드](features/voice-mode.md)를 참조하세요.

:::tip
명령어는 대소문자를 구분하지 않습니다 — `/HELP`는 `/help`와 동일하게 동작합니다. 설치된 스킬 또한 자동으로 슬래시 명령어가 됩니다.
:::

## 빠른 명령어 (Quick Commands)

LLM을 호출하지 않고도 쉘 명령을 즉시 실행하는 사용자 지정 명령을 정의할 수 있습니다. 이는 CLI와 메시징 플랫폼(Telegram, Discord 등) 모두에서 동작합니다.

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

그런 다음 대화창에서 `/status`, `/gpu` 또는 `/restart`를 입력하세요. 자세한 예시는 [구성 가이드](/user-guide/configuration#quick-commands)를 참조하세요.

## 실행 시 스킬 미리 로드 (Preloading Skills at Launch)

세션에 어떤 스킬을 활성화할지 이미 알고 있다면 실행 시 인자로 전달하세요:

```bash
hermes -s hermes-agent-dev,github-auth
hermes chat -s github-pr-workflow -s github-auth
```

Hermes는 첫 턴 전에 이름이 지정된 각 스킬을 세션 프롬프트에 로드합니다. 동일한 플래그가 대화형 모드 및 단일 쿼리 모드 모두에서 작동합니다.

## 스킬 슬래시 명령어 (Skill Slash Commands)

`~/.hermes/skills/`에 설치된 모든 스킬은 자동으로 슬래시 명령어로 등록됩니다. 스킬 이름이 그대로 명령어가 됩니다:

```
/gif-search funny cats
/axolotl help me fine-tune Llama 3 on my dataset
/github-pr-workflow create a PR for the auth refactor

# 스킬 이름만 입력하면 스킬을 로드하고 에이전트가 어떤 도움이 필요한지 물어봅니다:
/excalidraw
```

## 성격 (Personalities)

에이전트의 말투를 변경하려면 미리 정의된 성격을 설정하세요:

```
/personality pirate
/personality kawaii
/personality concise
```

내장된 성격은 다음과 같습니다: `helpful`, `concise`, `technical`, `creative`, `teacher`, `kawaii`, `catgirl`, `pirate`, `shakespeare`, `surfer`, `noir`, `uwu`, `philosopher`, `hype`.

또한 `~/.hermes/config.yaml`에 사용자 지정 성격을 정의할 수도 있습니다:

```yaml
personalities:
  helpful: "You are a helpful, friendly AI assistant."
  kawaii: "You are a kawaii assistant! Use cute expressions..."
  pirate: "Arrr! Ye be talkin' to Captain Hermes..."
  # 직접 추가해 보세요!
```

## 여러 줄 입력 (Multi-line Input)

여러 줄의 메시지를 입력하는 방법은 두 가지가 있습니다:

1. **`Alt+Enter`, `Ctrl+J`, 또는 `Shift+Enter`** — 새 줄 추가
2. **백슬래시 연속(Backslash continuation)** — 줄 끝에 `\`를 입력하여 계속 이어나가기:

```
❯ Write a function that:\
  1. Takes a list of numbers\
  2. Returns the sum
```

:::info
여러 줄의 텍스트 붙여넣기를 지원합니다 — 위의 줄 바꿈 단축키를 사용하거나 내용을 그냥 그대로 붙여넣으세요.
:::

### Shift+Enter 호환성 (Shift+Enter compatibility)

대부분의 터미널은 기본적으로 `Enter`와 `Shift+Enter`에 대해 동일한 바이트 시퀀스를 전송하므로 애플리케이션은 두 키를 구분하지 못합니다. Hermes는 터미널이 [Kitty 키보드 프로토콜](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) 또는 xterm의 `modifyOtherKeys` 모드를 통해 고유한 시퀀스를 전송할 때만 `Shift+Enter`를 인식합니다.

| 터미널 | 상태 |
|---|---|
| Kitty, foot, WezTerm, Ghostty | 기본적으로 고유한 `Shift+Enter` 활성화 |
| iTerm2 (최신 버전), Alacritty, VS Code 터미널, Warp | 설정에서 Kitty 프로토콜 활성화 후 지원됨 |
| Windows Terminal Preview 1.25+ | 설정에서 Kitty 프로토콜 활성화 후 지원됨 |
| macOS Terminal.app, 기본 Windows Terminal (stable) | 미지원 — `Shift+Enter`를 `Enter`와 구분할 수 없음 |

터미널이 두 키를 구분할 수 없는 경우에도 `Alt+Enter`와 `Ctrl+J`는 모든 환경에서 계속 동작합니다. **특히 Windows Terminal에서는 `Alt+Enter`가 터미널에 캡처되어(전체 화면 토글) Hermes에 도달하지 않으므로 — 줄바꿈을 하려면 `Ctrl+Enter` (`Ctrl+J`로 전달됨) 또는 `Ctrl+J`를 직접 사용하세요.**

## 에이전트 중단 (Interrupting the Agent)

언제든지 에이전트 동작을 중단할 수 있습니다:

- **에이전트 작업 중 새 메시지 입력 + Enter** — 에이전트의 현재 동작을 멈추고 새로운 지시사항을 바로 처리합니다.
- **`Ctrl+C`** — 현재 동작 중단 (2초 내 2회 입력 시 강제 종료)
- 진행 중인 터미널 명령어는 즉시 강제 종료됩니다 (SIGTERM 후, 1초 뒤 SIGKILL)
- 중단 과정에 입력된 여러 메시지는 하나의 프롬프트로 병합됩니다.

### 동작 중 입력 모드 (Busy Input Mode)

`display.busy_input_mode` 설정 키는 에이전트가 작업하는 동안 Enter를 누를 때 어떻게 동작할지 결정합니다:

| 모드 | 동작 |
|------|----------|
| `"interrupt"` (기본값) | 메시지가 즉시 현재 작업을 중단하고 처리됩니다 |
| `"queue"` | 메시지가 대기열에 몰래 추가되며 에이전트가 현재 작업을 마친 후 다음 턴으로 전송됩니다 |
| `"steer"` | 메시지가 `/steer`를 통해 현재 실행 흐름에 주입되며 다음 도구 호출 후에 에이전트에게 전달됩니다 — 중단 없음, 새 턴 없음 |

```yaml
# ~/.hermes/config.yaml
display:
  busy_input_mode: "steer"   # 또는 "queue" 나 "interrupt" (기본값)
```

`"queue"` 모드는 진행 중인 작업을 실수로 취소하지 않고 다음 메시지를 준비하려 할 때 유용합니다. `"steer"` 모드는 에이전트의 작업을 멈추지 않고 방향만 조절하고자 할 때 — 예를 들어 코드를 편집하는 도중에 "참, 테스트 코드도 같이 확인해 줘"라고 말할 때 유용합니다. 알 수 없는 값은 `"interrupt"`로 폴백됩니다.

`"steer"`에는 두 가지 자동 폴백이 있습니다: 에이전트가 아직 시작되지 않았거나, 이미지가 첨부된 경우, 메시지를 잃어버리지 않도록 모드가 `"queue"`로 전환됩니다.

CLI 안에서도 모드를 변경할 수 있습니다:

```text
/busy queue
/busy steer
/busy interrupt
/busy status
```

:::tip 첫 터치 힌트 (First-touch hint)
에이전트가 작업 중일 때 처음으로 Enter를 누르면, Hermes는 `/busy` 설정을 상기시키는 한 줄 팁을 출력합니다 (`"(tip) Your message interrupted the current run…"`). 이는 설치 후 1회만 표시되며 — `config.yaml`의 `onboarding.seen.busy_input_prompt` 항목으로 기록됩니다. 이 항목을 지우면 팁을 다시 볼 수 있습니다.
:::

### 백그라운드로 일시 중단 (Suspending to Background)

Unix 시스템에서는 **`Ctrl+Z`**를 눌러 Hermes를 백그라운드로 일시 중단할 수 있습니다 — 다른 일반적인 터미널 프로세스와 같습니다. 쉘에서 다음과 같은 확인 메시지가 나타납니다:

```
Hermes Agent has been suspended. Run `fg` to bring Hermes Agent back.
```

세션을 중단했던 지점에서 정확히 재개하려면 쉘에서 `fg`를 입력하세요. 이는 Windows에서 지원되지 않습니다.

## 도구 진행 상태 표시 (Tool Progress Display)

CLI는 에이전트가 작업 중일 때 애니메이션 피드백을 보여줍니다:

**생각 중 애니메이션** (API 호출 중):
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

`/verbose`로 표시 모드를 전환할 수 있습니다: `off → new → all → verbose`. 이 명령어는 메시징 플랫폼에서도 활성화할 수 있습니다 — [구성](/user-guide/configuration#display-settings)을 참조하세요.

### 도구 미리보기 길이 (Tool Preview Length)

`display.tool_preview_length` 설정은 도구 호출 미리보기 줄(예: 파일 경로, 터미널 명령어)에 표시되는 최대 문자 수를 제어합니다. 기본값은 `0`으로 무제한을 의미하며 전체 경로와 명령어를 보여줍니다.

```yaml
# ~/.hermes/config.yaml
display:
  tool_preview_length: 80   # 도구 미리보기를 80자로 자름 (0 = 무제한)
```

터미널 화면이 좁거나 도구 인수에 매우 긴 파일 경로가 포함된 경우 유용합니다.

## 세션 관리 (Session Management)

### 세션 재개 (Resuming Sessions)

CLI 세션을 종료할 때 재개 명령어가 표시됩니다:

```
Resume this session with:
  hermes --resume 20260225_143052_a1b2c3

Session:        20260225_143052_a1b2c3
Duration:       12m 34s
Messages:       28 (5 user, 18 tool calls)
```

재개 옵션:

```bash
hermes --continue                          # 가장 최근의 CLI 세션 재개
hermes -c                                  # 짧은 형태
hermes -c "my project"                     # 이름이 지정된 세션(계통 내 최신 항목) 재개
hermes --resume 20260225_143052_a1b2c3     # ID로 특정 세션 재개
hermes --resume "refactoring auth"         # 제목으로 재개
hermes -r 20260225_143052_a1b2c3           # 짧은 형태
```

재개 시 SQLite에 보관된 전체 대화 기록이 복원됩니다. 에이전트는 마치 종료한 적이 없는 것처럼 이전의 모든 메시지, 도구 호출, 응답 내용을 똑같이 볼 수 있습니다.

대화창에서 `/title 내 세션 이름`을 입력해 현재 세션의 이름을 지정하거나, 명령줄에서 `hermes sessions rename <id> <title>`을 입력하세요. 이전 세션을 찾아보려면 `hermes sessions list`를 사용하세요.

### 세션 저장 (Session Storage)

CLI 세션은 `~/.hermes/state.db`에 있는 Hermes의 SQLite 상태 데이터베이스에 저장됩니다. 데이터베이스는 다음을 보관합니다:

- 세션 메타데이터 (ID, 제목, 타임스탬프, 토큰 카운터)
- 메시지 기록
- 압축/재개 세션에 걸친 계통(lineage)
- `session_search`에 사용되는 전체 텍스트 검색 색인

일부 메시징 어댑터는 데이터베이스 외에 플랫폼별 스크립트 파일을 함께 보관하기도 하지만, CLI 자체는 SQLite 세션 저장소에서 재개됩니다.

### 컨텍스트 압축 (Context Compression)

대화가 길어져 컨텍스트 한도에 도달하면 자동으로 요약됩니다:

```yaml
# ~/.hermes/config.yaml 파일
compression:
  enabled: true
  threshold: 0.50    # 기본적으로 컨텍스트 한도의 50% 지점에서 압축

# auxiliary 하위로 구성된 요약 모델 설정:
auxiliary:
  compression:
    model: ""  # 기본 채팅 모델 사용 시(기본값) 비워 둡니다. "google/gemini-3-flash-preview"와 같은 저렴하고 빠른 모델로 고정할 수도 있습니다.
```

압축이 트리거되면 처음 3번의 턴과 마지막 20번의 턴은 항상 보존되고 그 사이의 중간 턴들이 요약됩니다.

## 백그라운드 세션 (Background Sessions)

CLI로 다른 작업을 계속하는 동시에, 별도의 백그라운드 세션에서 프롬프트를 실행해 보세요:

```
/background Analyze the logs in /var/log and summarize any errors from today
```

Hermes는 즉시 작업을 확인하고 명령 프롬프트를 다시 넘겨줍니다:

```
🔄 Background task #1 started: "Analyze the logs in /var/log and summarize..."
   Task ID: bg_143022_a1b2c3
```

### 작동 방식 (How It Works)

각 `/background` 프롬프트는 데몬 스레드에 **완전히 독립된 에이전트 세션**을 생성합니다:

- **격리된 대화 (Isolated conversation)** — 백그라운드 에이전트는 현재 세션의 대화 내역에 대해 알지 못합니다. 오직 사용자가 입력한 프롬프트만 전달받습니다.
- **동일한 설정 (Same configuration)** — 백그라운드 에이전트는 현재 세션의 모델, 제공자, 도구 세트, 추론 설정 및 폴백 모델을 그대로 상속합니다.
- **논블로킹 (Non-blocking)** — 포그라운드 세션은 대화형 모드를 그대로 유지합니다. 채팅, 명령어 실행, 새로운 백그라운드 작업 시작을 계속 진행할 수 있습니다.
- **다중 작업 (Multiple tasks)** — 백그라운드 작업을 여러 개 동시에 실행할 수 있습니다. 각 작업에는 고유 번호 ID가 부여됩니다.

### 결과 (Results)

백그라운드 작업이 완료되면 그 결과가 터미널 패널에 출력됩니다:

```
╭─ ⚕ Hermes (background #1) ──────────────────────────────────╮
│ Found 3 errors in syslog from today:                         │
│ 1. OOM killer invoked at 03:22 — killed process nginx        │
│ 2. Disk I/O error on /dev/sda1 at 07:15                      │
│ 3. Failed SSH login attempts from 192.168.1.50 at 14:30      │
╰──────────────────────────────────────────────────────────────╯
```

작업에 실패할 경우에는 오류 알림이 뜹니다. 설정에서 `display.bell_on_complete`가 켜져 있으면 작업이 끝날 때 터미널에서 경고음이 울립니다.

### 활용 사례 (Use Cases)

- **장시간 리서치** — 코드 작업을 진행하는 동시에, "/background 양자 오류 수정의 최신 연구 결과를 리서치해 줘"
- **파일 처리** — 다른 대화를 진행하면서, "/background 이 저장소 내의 파이썬 파일을 모두 분석해서 보안 취약점을 나열해 줘"
- **병렬 조사** — 여러 백그라운드 작업을 시작해 여러 관점을 동시에 살펴보기

:::info
백그라운드 세션은 현재 진행 중인 주 대화 기록에 표시되지 않습니다. 이들은 각자의 작업 ID(예: `bg_143022_a1b2c3`)를 가진 독립 세션입니다.
:::

## 조용한 모드 (Quiet Mode)

기본적으로 CLI는 조용한 모드로 실행되며 다음과 같이 작동합니다:
- 도구의 자세한 로깅 표시 생략
- 카와이(kawaii) 스타일의 애니메이션 피드백 표시
- 출력을 깔끔하고 사용자 친화적인 형태로 유지

디버그 출력을 보려면 다음 명령어를 사용하세요:
```bash
hermes chat --verbose
```

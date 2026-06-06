---
sidebar_position: 2
title: "TUI"
description: "Hermes의 최신 터미널 UI를 실행하세요 — 마우스 친화적, 풍부한 오버레이, 그리고 논블로킹(non-blocking) 입력 지원."
---

# TUI

TUI는 Hermes의 최신 프론트엔드입니다 — [클래식 CLI](cli.md)와 동일한 Python 런타임을 백엔드로 사용하는 터미널 UI입니다. 동일한 에이전트, 동일한 세션, 동일한 슬래시 명령어를 사용하며, 상호작용하기 더 깔끔하고 반응성이 뛰어난 인터페이스를 제공합니다.

대화형으로 Hermes를 실행할 때 권장되는 방법입니다.

## 실행 (Launch)

```bash
# TUI 실행
hermes --tui

# 가장 최근의 TUI 세션 재개 (최근 클래식 세션으로 폴백될 수 있음)
hermes --tui -c
hermes --tui --continue

# ID 또는 제목으로 특정 세션 재개
hermes --tui -r 20260409_000000_aa11bb
hermes --tui --resume "my t0p session"

# 소스에서 직접 실행 — 사전 빌드(prebuild) 단계 건너뛰기 (TUI 기여자용)
hermes --tui --dev
```

환경 변수를 통해 활성화할 수도 있습니다:

```bash
export HERMES_TUI=1
hermes          # 이제 TUI를 사용합니다
hermes chat     # 마찬가지입니다
```

또는 `~/.hermes/config.yaml`에 영구적인 기본값으로 설정할 수 있습니다:

```yaml
display:
  interface: tui   # "cli" (기본값) 또는 "tui"
```

`display.interface: tui`로 설정하면, 인자 없는 `hermes` (및 `hermes chat`)가 TUI를 실행합니다. 명시적인 플래그가 항상 우선순위를 갖습니다 — `hermes --cli`를 실행하면 한 번의 호출에 대해 클래식 REPL로 돌아가며, 설정 기본값이 `cli`일 때는 `hermes --tui` / `HERMES_TUI=1`을 사용하여 TUI를 강제할 수 있습니다.

클래식 CLI는 여전히 배포되는 기본값입니다. [CLI 인터페이스](cli.md)에 문서화된 모든 내용 — 슬래시 명령어, 빠른 명령어, 스킬 사전 로딩, 성격(personalities), 여러 줄 입력, 인터럽트 — 은 TUI에서도 동일하게 작동합니다.

## TUI를 사용하는 이유 (Why the TUI)

- **즉각적인 첫 프레임** — 앱 로딩이 완료되기 전에 배너가 그려지므로 Hermes가 시작되는 동안 터미널이 멈춘 것처럼 느껴지지 않습니다.
- **논블로킹 입력** — 세션이 준비되기 전에 메시지를 입력하고 대기열에 넣을 수 있습니다. 첫 번째 프롬프트는 에이전트가 온라인 상태가 되는 즉시 전송됩니다.
- **풍부한 오버레이** — 모델 선택기, 세션 선택기, 승인 및 설명 요청 프롬프트 등이 인라인 흐름이 아닌 모달 패널로 렌더링됩니다.
- **라이브 세션 패널** — 도구와 스킬이 초기화되면서 점진적으로 채워집니다.
- **마우스 친화적인 선택** — SGR 반전 대신 균일한 배경으로 드래그하여 강조 표시합니다. 터미널의 일반적인 복사 제스처로 복사합니다.
- **대체 화면(Alternate-screen) 렌더링** — 차분 업데이트(differential updates)를 통해 스트리밍 시 깜박임이 없으며, 종료 후 스크롤백이 지저분해지지 않습니다.
- **컴포저 어포던스** — 긴 코드 조각을 위한 인라인 붙여넣기 접기, 클립보드 이미지 폴백이 포함된 `Cmd+V` / `Ctrl+V` 텍스트 붙여넣기, 안전한 괄호 붙여넣기(bracketed-paste safety), 이미지/파일 경로 첨부 정규화 등을 지원합니다.

동일한 [스킨(skins)](features/skins.md)과 [성격(personalities)](features/personality.md)이 적용됩니다. `/skin ares`, `/personality pirate` 명령어를 통해 세션 도중에 변경할 수 있으며 UI가 실시간으로 다시 그려집니다. 사용자 정의 가능한 키의 전체 목록과 클래식 및 TUI에 적용되는 키에 대해서는 [스킨 및 테마](features/skins.md)를 참조하세요 — TUI는 배너 팔레트, UI 색상, 프롬프트 글리프/색상, 세션 표시, 자동 완성 메뉴, 선택 배경, `tool_prefix`, `help_header`를 존중합니다.

### 접을 수 있는 배너 섹션 (Collapsible banner sections)

TUI 시작 배너는 런타임 정보를 4개의 접을 수 있는 섹션으로 그룹화하며, 각 섹션 제목 옆에는 `▸` / `▾` 쉐브론(chevron) 기호가 렌더링됩니다:

| 섹션 | 기본 상태 |
|---------|---------------|
| Tools (도구) | 열림 (Open) |
| Skills (스킬) | 접힘 (Collapsed) |
| System Prompt (시스템 프롬프트) | 접힘 (Collapsed) |
| MCP Servers (MCP 서버) | 접힘 (Collapsed) |

섹션 헤더(또는 쉐브론)의 아무 곳이나 클릭하여 토글할 수 있습니다. Tools 목록은 세션 시작 시 가장 많이 확인되는 섹션이므로 기본적으로 열려 있습니다. Skills, System Prompt, MCP Servers는 기본적으로 접혀 있어 수십 개의 스킬을 설치하거나 여러 MCP 서버를 연결하더라도 배너가 콤팩트하게 유지됩니다. 상태는 해당 배너 인스턴스에 국한되므로 다음 실행 시 기본값으로 초기화됩니다.

## 요구 사항 (Requirements)

- **Node.js** ≥ 20 — TUI는 Python CLI에서 시작된 하위 프로세스(subprocess)로 실행됩니다. `hermes doctor`가 이를 검증합니다.
- **TTY** — 클래식 CLI와 마찬가지로, stdin 파이프를 사용하거나 비대화형 환경에서 실행하면 단일 쿼리 모드로 폴백됩니다.

첫 실행 시 Hermes는 TUI의 Node 의존성을 `ui-tui/node_modules`에 설치합니다 (1회성, 몇 초 소요). 이후 실행은 빠릅니다. 새로운 Hermes 버전을 가져올 때 소스가 배포본(dist)보다 최신인 경우 TUI 번들이 자동으로 다시 빌드됩니다.

### 외부 사전 빌드 (External prebuild)

사전 빌드된 번들을 제공하는 배포판(Nix, 시스템 패키지)은 Hermes가 이를 가리키도록 설정할 수 있습니다:

```bash
export HERMES_TUI_DIR=/path/to/prebuilt/ui-tui
hermes --tui
```

해당 디렉토리에는 `dist/entry.js`가 있어야 합니다.

## 키 바인딩 (Keybindings)

키 바인딩은 [클래식 CLI](cli.md#keybindings)와 정확히 일치합니다. 유일한 동작 차이점은 다음과 같습니다:

- **마우스 드래그**는 균일한 선택 배경으로 텍스트를 강조 표시합니다.
- **`Cmd+V` / `Ctrl+V`**는 먼저 일반 텍스트 붙여넣기를 시도한 다음, OSC52/네이티브 클립보드 읽기로 폴백하고, 마지막으로 클립보드나 붙여넣은 페이로드가 이미지로 확인되면 이미지 첨부로 처리합니다.
- **`/terminal-setup`**은 macOS에서 더 나은 `Cmd+Enter` 및 실행 취소/다시 실행 패리티(parity)를 위해 로컬 VS Code / Cursor / Windsurf 터미널 바인딩을 설치합니다.
- **슬래시 자동 완성**은 인라인 드롭다운이 아닌 설명이 포함된 플로팅 패널로 열립니다.
- **`Ctrl+X`**는 라이브 세션 스위처를 엽니다. 대기열에 있는 메시지가 강조 표시된 경우(에이전트가 아직 실행 중일 때 전송된 경우), 스위처를 여는 대신 대기열에 있는 해당 메시지를 삭제합니다. **`Esc`**는 삭제하지 않고 편집을 취소하고 강조 표시를 해제합니다.
- **`Ctrl+G` / `Ctrl+X Ctrl+E`** — 멀티 라인 / 긴 프롬프트 작성을 위해 현재 입력 버퍼를 `$EDITOR`에서 엽니다; 저장 후 종료하면 내용이 프롬프트로 다시 전송됩니다.

## 슬래시 명령어 (Slash commands)

모든 슬래시 명령어는 변경 없이 작동합니다. 몇 가지는 TUI 전용이며 — 인라인 패널 대신 더 풍부한 출력을 생성하거나 오버레이로 렌더링됩니다:

| 명령어 | TUI 동작 |
|---------|--------------|
| `/help` | 화살표 키로 탐색 가능한 분류된 명령어가 포함된 오버레이 |
| `/sessions` (별칭 `/switch`) | 라이브 세션 스위처 — 열려 있는 TUI 세션 목록을 표시하고, 세션 간 전환, 닫기 또는 새 세션을 시작할 수 있습니다. |
| `/model` | 비용 힌트와 함께 제공자별로 그룹화된 모달 모델 선택기 |
| `/skin` | 라이브 미리보기 — 탐색하는 동안 테마 변경 사항이 즉시 적용됩니다. |
| `/details` | 상세한 도구 호출 정보(전역 또는 섹션별)를 토글합니다. |
| `/usage` | 풍부한 토큰 / 비용 / 컨텍스트 패널 |
| `/agents` (별칭 `/tasks`) | 관측 가능성(Observability) 오버레이 — 중지/일시 중지 제어, 브랜치별 비용 / 토큰 / 파일 요약, 턴(turn)별 히스토리가 포함된 라이브 서브에이전트 트리 |
| `/reload` | 재시작 없이 새로 추가된 API 키가 적용되도록 실행 중인 TUI 프로세스에 `~/.hermes/.env`를 다시 읽어옵니다. |
| `/mouse [on\|off\|toggle\|wheel\|buttons\|all]` | 런타임에 마우스 추적 사전 설정을 선택합니다 (`config.yaml`의 `display.mouse_tracking`에도 유지됨). `wheel` (1000+1006)은 tmux가 프롬프트 행 위에 "No image in clipboard"를 도배하게 만드는 호버(hover) 이벤트 없이 스크롤 휠 스크롤을 유지합니다. `buttons`는 드래그하여 선택 기능을 추가합니다. `all`은 호버 기반 UI를 사용하는 기본값입니다. |

설치된 스킬, 빠른 명령어, 성격 토글을 포함한 기타 모든 슬래시 명령어는 클래식 CLI와 동일하게 작동합니다. [슬래시 명령어 레퍼런스](../reference/slash-commands.md)를 참조하세요.

## 라이브 세션 스위처 (Live session switcher)

하나의 터미널이 여러 TUI 세션의 디스패처 역할을 하도록 하려면 라이브 세션 스위처를 사용하세요. 현재 TUI 프로세스에서 실행 중인 세션만 나열합니다. 닫힌 세션은 저장된 스크립트로 남아 있으며 `/resume` 또는 `hermes --tui --resume <id-or-title>` 명령을 사용하여 다시 열 수 있습니다.

다음 중 하나를 사용하여 엽니다:

- TUI에서 `Ctrl+X`
- `/sessions` 또는 `/switch`
- `/sessions new`를 사용하여 즉시 새로운 라이브 세션 생성
- 상태 줄에 있는 `N live sessions` 개수 클릭

<img alt="Hermes TUI Session Orchestrator with one live session and a +new row" src="/img/docs/tui-session-orchestrator/session-orchestrator.png" />

<video controls muted loop playsInline src="/img/docs/tui-session-orchestrator/session-orchestrator-demo.mp4" title="Hermes TUI Session Orchestrator demo" />

스위처 내에서:

- `↑` / `↓`는 선택 항목을 이동합니다. 마우스 클릭으로도 행을 선택할 수 있습니다.
- `Enter`는 선택한 라이브 세션으로 전환합니다.
- `Ctrl+D`는 선택한 라이브 세션을 닫습니다.
- `Ctrl+N`은 빈 라이브 세션을 시작합니다.
- `Ctrl+R`은 라이브 세션 목록을 새로 고칩니다.
- `Esc`는 스위처를 닫습니다.
- `+new`를 선택하고 프롬프트를 입력한 후 `Enter`를 누르면 새로운 라이브 세션이 시작됩니다. 해당 새 세션에 대해서만 모델을 선택하려면 먼저 `Tab`을 누르세요.

## LaTeX 수식 렌더링 (LaTeX math rendering)

TUI의 마크다운 파이프라인은 LaTeX 수식을 인라인으로 렌더링합니다: `$E = mc^2$` 및 `$$\frac{a}{b}$$`는 원시 TeX 소스 대신 유니코드 형식의 수식으로 렌더링됩니다. 인라인 및 블록 수식 모두에서 작동하며 지원되지 않는 구문은 복사할 수 있도록 코드 범위(code span)로 감싸진 리터럴 TeX로 폴백됩니다.

이 기능은 항상 켜져 있으며 구성할 필요가 없습니다. 클래식 CLI는 원시 TeX를 유지합니다.

## 밝은 테마 터미널 감지 (Light-terminal detection)

TUI는 밝은 터미널을 자동으로 감지하고 그에 따라 밝은 테마로 교체합니다. 감지는 세 가지 계층으로 작동합니다:

1. `HERMES_TUI_THEME` 환경 변수 — 최우선 순위입니다. 값: `light`, `dark` 또는 원시 6자리 배경 16진수 코드(예: `ffffff`, `1a1a2e`).
2. `COLORFGBG` 환경 변수 — xterm 파생 터미널에서 사용하는 클래식 "내 배경색은 무엇인가요?" 힌트입니다.
3. OSC 11을 통한 터미널 배경 프로브 — `COLORFGBG`를 설정하지 않는 최신 터미널(Ghostty, Warp, iTerm2, WezTerm, Kitty)에서 작동합니다.

터미널에 관계없이 영구적으로 밝은 테마를 원한다면:

```bash
export HERMES_TUI_THEME=light
```

## 작업 진행 표시줄 스타일 (Busy indicator styles)

상태 표시줄의 진행 상태 표시기(busy indicator)는 교체 가능합니다. 기본값은 에이전트 작업 중 2.5초마다 Hermes의 카와이(kawaii) 얼굴 팔레트를 회전시킵니다. 설정 또는 `/indicator` 슬래시 명령어를 통해 다른 스타일을 선택하세요:

```yaml
display:
  tui_status_indicator: kaomoji   # kaomoji | emoji | unicode | ascii
```

또는 세션 내에서: `/indicator emoji` (등). 스타일은 회전 시 상태 표시줄의 나머지 부분이 흔들리지 않도록 일치하는 글리프 너비로 제공됩니다.

## 자동 재개 (Auto-resume)

기본적으로 `hermes --tui`는 실행될 때마다 새 세션을 시작합니다. 자동으로 가장 최근의 TUI 세션에 다시 연결하려면(터미널이나 SSH 연결이 예기치 않게 끊어졌을 때 유용함) 옵트인하세요:

```bash
export HERMES_TUI_RESUME=1          # 가장 최근 TUI 세션
# 또는:
export HERMES_TUI_RESUME=<session-id>   # 특정 세션
```

변수를 해제하거나 `--resume <id>`를 명시적으로 전달하여 실행마다 재정의할 수 있습니다.

## 상태 줄 (Status line)

TUI의 상태 줄은 에이전트 상태를 실시간으로 추적합니다:

| 상태 | 의미 |
|--------|---------|
| `starting agent…` | 세션 ID가 활성화되었습니다; 도구와 스킬이 여전히 온라인 상태로 전환 중입니다. 입력 가능하며 메시지는 대기열에 추가되고 준비가 되면 전송됩니다. |
| `ready` | 에이전트가 유휴 상태이며, 입력을 대기 중입니다. |
| `thinking…` / `running…` | 에이전트가 추론 중이거나 도구를 실행 중입니다. |
| `interrupted` | 현재 턴이 취소되었습니다; 다시 전송하려면 Enter를 누르세요. |
| `forging session…` / `resuming…` | 초기 연결 또는 `--resume` 핸드셰이크. |

스킨별 상태 표시줄 색상과 임계값은 클래식 CLI와 공유됩니다. 사용자 지정에 대해서는 [스킨](features/skins.md)을 참조하세요.

상태 줄에는 다음 정보도 표시됩니다:

- **작업 디렉토리와 git 브랜치** — `~/projects/hermes-agent (docs/two-week-gap-sweep)`. 측면 터미널에서 `git checkout`을 실행할 때 브랜치 접미사가 업데이트되므로(mtime 캐시) TUI는 시작할 때가 아니라 실제 활성 브랜치를 반영합니다.
- **프롬프트당 경과 시간** — 턴이 실행되는 동안에는 `⏱ 12s/3m 45s` (라이브), 턴이 완료된 후에는 `⏲ 32s / 3m 45s`로 멈춥니다. 첫 번째 숫자는 사용자의 마지막 메시지 이후의 시간이고, 두 번째 숫자는 전체 세션 기간입니다. 새 프롬프트마다 재설정됩니다.
- **`🗜️ N`** — 실행 중인 세션이 자동 압축된 횟수입니다. 첫 번째 압축이 시작되면 나타납니다.
- **`▶ N`** — 이 세션에서 현재 실행 중인 `/background` 작업 수입니다. 비행 중(in-flight)인 작업이 하나 이상 있을 때 나타납니다.
- **`⚠ YOLO`** — YOLO 모드가 켜져 있을 때 항상 표시되는 경고입니다(`hermes --yolo`, `/yolo` 또는 `HERMES_YOLO_MODE=1`). 시작 배너에도 동일한 배지가 나타나므로 사용자가 알아채지 못하고 자동 승인 세션을 시작할 수 없습니다.

## 구성 (Configuration)

TUI는 `~/.hermes/config.yaml`, 프로필, 성격, 스킨, 빠른 명령어, 자격 증명 풀, 메모리 제공자, 도구/스킬 활성화 등 모든 표준 Hermes 구성을 존중합니다. TUI 전용 구성 파일은 없습니다.

몇 가지 키는 TUI 화면을 특별히 조정합니다:

```yaml
display:
  skin: default              # 내장 또는 사용자 정의 스킨
  personality: helpful
  details_mode: collapsed    # hidden | collapsed | expanded — 전역 아코디언 기본값
  sections:                  # 선택 사항: 섹션별 재정의 (일부만 지정 가능)
    thinking: expanded       # 항상 열림
    tools: expanded          # 항상 열림
    activity: collapsed      # 활동 패널 다시 활성화 (기본적으로 숨겨짐)
  mouse_tracking: all        # off | wheel | buttons | all (또는 하위 호환성을 위해 true/false)
                             #   wheel   — 1000+1006 (스크롤 + 클릭; 드래그 없음, 호버 없음 —
                             #             tmux 내부에서 호버 이벤트로 인한 프롬프트 행의
                             #             "No image in clipboard" 스팸을 차단하는 데 권장됨)
                             #   buttons — 터미널 쪽 드래그 선택을 위해 1002 추가
                             #   all     — 호버를 위해 1003 추가 (호버 시 스크롤바 페이지 넘김,
                             #             링크 mouseenter 등)
```

런타임 토글:

- `/details [hidden|collapsed|expanded|cycle]` — 전역 모드 설정
- `/details <section> [hidden|collapsed|expanded|reset]` — 특정 섹션 재정의
  (섹션: `thinking`, `tools`, `subagents`, `activity`)

**기본 가시성 (Default visibility)**

TUI는 의견이 반영된 섹션별 기본값을 제공하여, 쉐브론의 장벽이 아닌 라이브 스크립트 형태로 턴을 스트리밍합니다:

- `thinking` — **expanded**. 추론 내용이 모델에서 출력되는 즉시 인라인으로 스트리밍됩니다.
- `tools` — **expanded**. 도구 호출 및 그 결과가 열린 상태로 렌더링됩니다.
- `subagents` — 전역 `details_mode` 설정을 따릅니다 (기본적으로 쉐브론 아래 접혀 있으며 위임이 실제로 발생하기 전까지는 조용히 유지됩니다).
- `activity` — **hidden**. 주변 메타데이터(게이트웨이 힌트, 터미널 패리티 넛지, 백그라운드 알림)는 대부분의 일상적인 사용에 방해가 될 수 있습니다. 도구 실패 시 실패한 도구 행에는 계속 인라인으로 렌더링되며, 주변 오류/경고는 모든 패널이 숨겨졌을 때 플로팅 알림 백스톱을 통해 나타납니다.

섹션별 재정의는 섹션 기본값 및 전역 `details_mode`보다 우선합니다. 레이아웃을 변경하려면 다음과 같이 설정하세요:

- `display.sections.thinking: collapsed` — thinking(생각)을 다시 쉐브론 아래로 접기
- `display.sections.tools: collapsed` — 도구 호출을 다시 쉐브론 아래로 접기
- `display.sections.activity: collapsed` — 활동 패널 다시 활성화
- 런타임에서 `/details <section> <mode>` 사용

`display.sections`에 명시적으로 설정된 내용은 기본값보다 우선하므로 기존 구성은 변경 없이 계속 작동합니다.

## 세션 (Sessions)

세션은 TUI와 클래식 CLI 간에 공유되며, 두 인터페이스 모두 동일한 `~/.hermes/state.db`에 기록합니다. 한 인터페이스에서 세션을 시작하고 다른 인터페이스에서 재개할 수 있습니다. 세션 선택기에는 두 소스에서 생성된 세션이 태그와 함께 모두 표시됩니다.

수명 주기, 검색, 압축 및 내보내기에 대한 자세한 내용은 [세션](sessions.md)을 참조하세요.

## 실행 중인 게이트웨이에 연결 (Attaching to a running gateway)

기본적으로 TUI는 자체 프로세스 내 게이트웨이를 생성하므로 각 TUI 인스턴스는 자체적으로 완결됩니다. (예: tmux의 `hermes gateway run` 또는 systemd / launchd 서비스와 같이) 장기 실행 게이트웨이가 이미 실행 중인 경우 TUI가 해당 게이트웨이를 가리키도록 설정할 수 있습니다. 그런 다음 TUI는 씬 클라이언트(thin client)가 되어 동일한 게이트웨이에 연결된 모든 다른 표면(메시징 플랫폼, 웹 대시보드, 다른 TUI 세션)과 상태를 공유합니다.

시작하기 전에 환경 변수를 통해 웹소켓 URL을 설정하세요:

```bash
export HERMES_TUI_GATEWAY_URL="ws://localhost:8765/api/ws?token=<auth-token>"
hermes --tui
```

토큰은 게이트웨이의 API 인증 구성에서 가져옵니다 ( [API 서버](features/api-server.md) 참조). 환경 변수가 설정되면 TUI는 다음을 수행합니다:

- 로컬 게이트웨이 생성을 완전히 건너뜁니다 — 중복 플랫폼 어댑터가 없으며 포트 충돌이 없습니다.
- 모든 작업(슬래시 명령어, 이미지 첨부, 브라우저 진행 상황, 음성 이벤트 등)을 웹소켓을 통해 공유 게이트웨이로 라우팅합니다.
- 요청 간에 게이트웨이 URL이 회전하는(새 토큰 발급) 경우 자동으로 다시 연결합니다.

이는 웹 대시보드의 내장 TUI가 사용하는 채널과 동일합니다 ( [웹 대시보드](features/web-dashboard.md#chat) 참조) — 하나의 게이트웨이, 다수의 클라이언트.

## 클래식 CLI로 되돌리기 (Reverting to the classic CLI)

`hermes`를 시작하면 ( `--tui` 없이) 기본적으로 클래식 CLI가 유지됩니다. 시스템에서 TUI를 우선 사용하도록 하려면 `~/.hermes/config.yaml`에 `display.interface: tui`를 설정하거나 (영구적) 쉘 프로필에 `HERMES_TUI=1`을 설정하세요 (쉘 전용). 원래대로 돌아가려면 `interface: cli`를 설정하거나 환경 변수를 해제하거나 한 번만 `hermes --cli`를 전달하세요.

TUI 시작에 실패하는 경우(Node 없음, 누락된 번들, TTY 문제), Hermes는 오류 메시지를 출력하고 폴백합니다 — 사용자를 막힌 상태로 두지 않습니다.

## 같이 보기 (See also)

- [CLI 인터페이스](cli.md) — 전체 슬래시 명령어 및 키 바인딩 레퍼런스 (공유됨)
- [세션](sessions.md) — 재개, 브랜치 및 히스토리
- [스킨 및 테마](features/skins.md) — 배너, 상태 표시줄 및 오버레이 테마 지정
- [음성 모드(Voice Mode)](features/voice-mode.md) — 두 인터페이스에서 모두 작동
- [구성](configuration.md) — 모든 구성 키

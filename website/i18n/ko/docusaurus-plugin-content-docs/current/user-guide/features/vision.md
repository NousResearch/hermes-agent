---
title: 비전 및 이미지 붙여넣기
description: 클립보드의 이미지를 Hermes CLI에 붙여 넣어 멀티모달 비전 분석을 수행합니다.
sidebar_label: 비전 및 이미지 붙여넣기
sidebar_position: 7
---

# 비전 및 이미지 붙여넣기

Hermes Agent는 **멀티모달 비전**을 지원합니다. 클립보드의 이미지를 CLI에 직접 붙여 넣고 에이전트에게 이미지를 분석하거나, 설명하거나, 이미지로 작업하도록 요청할 수 있습니다. 이미지는 base64로 인코딩된 콘텐츠 블록으로 모델에 전송되므로 비전을 지원하는 모든 모델이 처리할 수 있습니다.

:::tip
Portal 구독자는 동일한 카탈로그에서 비전을 지원하는 모델(Claude, GPT-5, Gemini)을 사용할 수 있으며, 추가 자격 증명이 필요하지 않습니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 작동 방식

1. 이미지를 클립보드에 복사합니다(스크린샷, 브라우저 이미지 등).
2. 아래 방법 중 하나를 사용해 이미지를 첨부합니다.
3. 질문을 입력하고 Enter를 누릅니다.
4. 입력란 위에 `[📎 Image #1]` 배지가 표시됩니다.
5. 제출하면 이미지가 비전 콘텐츠 블록으로 모델에 전송됩니다.

전송하기 전에 여러 이미지를 첨부할 수 있으며, 각 이미지에는 고유한 배지가 표시됩니다. 첨부된 이미지를 모두 지우려면 `Ctrl+C`를 누릅니다.

이미지는 타임스탬프가 포함된 파일 이름의 PNG 파일로 `~/.hermes/images/`에 저장됩니다.

## 붙여넣기 방법

이미지를 첨부하는 방법은 터미널 환경에 따라 다릅니다. 모든 방법이 어디서나 작동하는 것은 아니므로, 전체 동작을 아래에 정리했습니다.

### `/paste` 명령

**명시적으로 이미지를 첨부할 때 가장 안정적인 대체 방법입니다.**

```
/paste
```

`/paste`를 입력하고 Enter를 누릅니다. Hermes가 클립보드에서 이미지를 확인하고 첨부합니다. 터미널이 `Cmd+V`/`Ctrl+V`를 다시 작성하거나, 이미지 하나만 복사해 대괄호 붙여넣기 텍스트 페이로드를 확인할 수 없을 때 가장 안전한 방법입니다.

### Ctrl+V / Cmd+V

이제 Hermes는 붙여넣기를 여러 단계의 흐름으로 처리합니다.
- 먼저 일반 텍스트를 붙여 넣습니다.
- 터미널이 텍스트를 제대로 전달하지 않은 경우 네이티브 클립보드 / OSC52 텍스트 대체 경로를 사용합니다.
- 클립보드 또는 붙여넣은 페이로드가 이미지나 이미지 경로로 확인되면 이미지를 첨부합니다.

따라서 macOS 스크린샷 임시 경로와 `file://...` 이미지 URI를 원시 텍스트로 입력란에 남기지 않고 즉시 첨부할 수 있습니다.

:::warning
클립보드에 **이미지만 있는 경우**(텍스트 없음)에도 터미널은 바이너리 이미지 바이트를 직접 보낼 수 없습니다. 명시적으로 이미지를 첨부하려면 `/paste`를 사용하세요.
:::

### VS Code / Cursor / Windsurf용 `/terminal-setup`

macOS의 로컬 VS Code 계열 통합 터미널에서 TUI를 실행하는 경우, Hermes는 더 나은 여러 줄 입력 및 실행 취소/다시 실행 동작을 위해 권장되는 `workbench.action.terminal.sendSequence` 바인딩을 설치할 수 있습니다.

```text
/terminal-setup
```

IDE가 `Cmd+Enter`, `Cmd+Z` 또는 `Shift+Cmd+Z`를 가로채는 경우에 특히 유용합니다. 로컬 컴퓨터에서만 이 명령을 실행하고 SSH 세션 내부에서는 실행하지 마세요.

## 플랫폼 호환성

| 환경 | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | 참고 |
|---|:---:|:---:|:---:|---|
| **macOS Terminal / iTerm2** | ✅ | ✅ | n/a | 최상의 경험 — 네이티브 클립보드 + 스크린샷 경로 복구 |
| **Apple Terminal** | ✅ | ✅ | n/a | Cmd+←/→/⌫가 다시 작성되면 Ctrl+A / Ctrl+E / Ctrl+U 대체 키를 사용하세요 |
| **Linux X11 데스크톱** | ✅ | ✅ | n/a | `xclip` 필요 (`apt install xclip`) |
| **Linux Wayland 데스크톱** | ✅ | ✅ | n/a | `wl-paste` 필요 (`apt install wl-clipboard`) |
| **WSL2 (Windows Terminal)** | ✅ | ✅ | n/a | `powershell.exe` 사용 — 추가 설치가 필요하지 않음 |
| **VS Code / Cursor / Windsurf (로컬)** | ✅ | ✅ | ✅ | 더 나은 Cmd+Enter / 실행 취소 / 다시 실행 동작에 권장 |
| **VS Code / Cursor / Windsurf (SSH)** | ❌² | ❌² | ❌³ | 로컬 컴퓨터에서 `/terminal-setup`을 실행하세요 |
| **SSH 터미널(모두)** | ❌² | ❌² | n/a | 원격 클립보드에 액세스할 수 없음 |

² 아래 [SSH 및 원격 세션](#ssh--remote-sessions) 참조
³ 이 명령은 로컬 IDE 키 바인딩을 작성하므로 원격 호스트에서 실행하면 안 됩니다.

## 플랫폼별 설정

### macOS

**설정이 필요하지 않습니다.** Hermes는 macOS에 기본으로 포함된 `osascript`를 사용해 클립보드를 읽습니다. 더 빠른 성능을 원하면 선택적으로 `pngpaste`를 설치하세요.

```bash
brew install pngpaste
```

### Linux (X11)

`xclip`을 설치합니다.

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux (Wayland)

최신 Linux 데스크톱(Ubuntu 22.04+, Fedora 34+)은 기본적으로 Wayland를 사용하는 경우가 많습니다. `wl-clipboard`을 설치합니다.

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip Wayland 사용 여부 확인
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland, "x11" = X11, "tty" = no display server
```
:::

### WSL2

**추가 설정이 필요하지 않습니다.** Hermes는(`/proc/version`을 통해) WSL2를 자동으로 감지하고, `powershell.exe`를 사용해 .NET의 `System.Windows.Forms.Clipboard`를 통해 Windows 클립보드에 액세스합니다. 이는 WSL2의 Windows 상호 운용 기능에 포함되어 있으며 `powershell.exe`는 기본적으로 사용할 수 있습니다.

클립보드 데이터는 stdout을 통해 base64로 인코딩된 PNG로 전송되므로 파일 경로 변환이나 임시 파일이 필요하지 않습니다.

:::info WSLg 참고
GUI를 지원하는 WSLg(WSL2)에서 실행하는 경우 Hermes는 먼저 PowerShell 경로를 시도한 다음 `wl-paste`로 대체합니다. WSLg의 클립보드 브리지는 이미지에 BMP 형식만 지원하므로 Hermes는 Pillow(설치된 경우) 또는 ImageMagick의 `convert` 명령을 사용해 BMP를 PNG로 자동 변환합니다.
:::

#### WSL2 클립보드 액세스 확인

```bash
# 1. Check WSL detection
grep -i microsoft /proc/version

# 2. Check PowerShell is accessible
which powershell.exe

# 3. Copy an image, then check
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# Should print "True"
```

## SSH 및 원격 세션

**SSH를 통한 클립보드 이미지 붙여넣기는 완전히 작동하지 않습니다.** 원격 컴퓨터에 SSH로 접속하면 Hermes CLI는 원격 호스트에서 실행됩니다. 클립보드 도구(`xclip`, `wl-paste`, `powershell.exe`, `osascript`)는 실행되는 컴퓨터의 클립보드, 즉 로컬 컴퓨터가 아닌 원격 서버의 클립보드를 읽습니다. 따라서 로컬 클립보드의 이미지에 원격 측에서 액세스할 수 없습니다.

텍스트는 터미널 붙여넣기 또는 OSC52를 통해 전달될 수 있지만, 이미지 클립보드 액세스와 로컬 스크린샷 임시 경로는 Hermes를 실행하는 컴퓨터에 묶여 있습니다.

### SSH의 대안

1. **이미지 파일 업로드** — 이미지를 로컬에 저장한 다음 `scp`, VSCode의 파일 탐색기(끌어다 놓기) 또는 다른 파일 전송 방법을 사용해 원격 서버에 업로드합니다. 그런 다음 경로로 참조합니다. *(`/attach <filepath>` 명령은 향후 릴리스에서 제공될 예정입니다.)*

2. **URL 사용** — 이미지가 온라인에서 액세스 가능하다면 메시지에 URL을 붙여 넣기만 하면 됩니다. 에이전트는 `vision_analyze`를 사용해 이미지 URL을 직접 확인할 수 있습니다.

3. **X11 포워딩** — `ssh -X`로 연결해 X11을 포워딩합니다. 그러면 원격 컴퓨터의 `xclip`이 로컬 X11 클립보드에 액세스할 수 있습니다. 로컬에서 X 서버가 실행 중이어야 합니다(macOS의 XQuartz, Linux X11 데스크톱에 기본 포함). 큰 이미지에서는 느립니다.

4. **메시징 플랫폼 사용** — Telegram, Discord, Slack 또는 WhatsApp을 통해 Hermes로 이미지를 보냅니다. 이러한 플랫폼은 이미지 업로드를 기본으로 처리하므로 클립보드/터미널의 제한을 받지 않습니다.

## 터미널에서 이미지를 붙여 넣을 수 없는 이유

이는 자주 혼동되는 부분이므로 기술적인 이유를 설명합니다.

터미널은 **텍스트 기반** 인터페이스입니다. Ctrl+V(또는 Cmd+V)를 누르면 터미널 에뮬레이터는 다음을 수행합니다.

1. 클립보드에서 **텍스트 콘텐츠**를 읽습니다.
2. 이를 [대괄호 붙여넣기](https://en.wikipedia.org/wiki/Bracketed-paste) 이스케이프 시퀀스로 감쌉니다.
3. 터미널의 텍스트 스트림을 통해 애플리케이션으로 보냅니다.

클립보드에 이미지만 있고 텍스트가 없으면 터미널이 보낼 것이 없습니다. 바이너리 이미지 데이터에 대한 표준 터미널 이스케이프 시퀀스는 없습니다. 터미널은 아무 작업도 하지 않습니다.

이 때문에 Hermes는 별도의 클립보드 확인을 사용합니다. 터미널 붙여넣기 이벤트를 통해 이미지 데이터를 받는 대신, 서브프로세스를 통해 OS 수준 도구(`osascript`, `powershell.exe`, `xclip`, `wl-paste`)를 직접 호출하여 클립보드를 독립적으로 읽습니다.

## 지원 모델

이미지 붙여넣기는 비전을 지원하는 모든 모델에서 작동합니다. 이미지는 OpenAI 비전 콘텐츠 형식의 base64 인코딩 데이터 URL로 전송됩니다.

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

대부분의 최신 모델은 이 형식을 지원하며, GPT-4 Vision, Claude(비전 지원), Gemini, OpenRouter를 통해 제공되는 오픈 소스 멀티모달 모델이 여기에 포함됩니다.

## 이미지 라우팅(비전 지원 모델과 텍스트 전용 모델)

사용자가 CLI 클립보드, 게이트웨이(Telegram/Discord 사진) 또는 다른 진입점에서 이미지를 첨부하면 Hermes는 현재 모델이 실제로 비전을 지원하는지에 따라 이미지를 라우팅합니다.

| 모델 | 이미지 처리 방식 |
|---|---|
| **비전 지원** (GPT-4V, 비전을 지원하는 Claude, Gemini, Qwen-VL, MiMo-VL 등) | 위에 설명한 공급자의 네이티브 이미지 콘텐츠 형식을 사용해 **실제 픽셀**로 전송됩니다. 텍스트 요약 계층이 없습니다. |
| **텍스트 전용** (DeepSeek V3, 소형 오픈 소스 모델, 이전 채팅 전용 엔드포인트) | `vision_analyze` 보조 도구를 통해 라우팅됩니다. 보조 비전 모델이 이미지를 설명하고 텍스트 설명이 대화에 주입됩니다. |

이를 직접 구성할 필요는 없습니다. Hermes는 공급자 메타데이터에서 현재 모델의 기능을 조회하고 적절한 경로를 자동으로 선택합니다. 실질적으로 비전 모델과 비전 미지원 모델 사이를 세션 중간에 전환해도 작업 흐름을 변경하지 않고 이미지 처리가 "그냥 작동"합니다. 텍스트 전용 모델은 거부할 수밖에 없는 잘못된 멀티모달 페이로드 대신 이미지에 대한 일관된 컨텍스트를 받습니다.

어떤 보조 모델이 텍스트 설명 경로를 처리하는지는 `auxiliary.vision`에서 구성할 수 있습니다. [보조 모델](/user-guide/configuration#auxiliary-models)을 참조하세요.

### `vision_analyze`도 동일한 이중 동작을 사용합니다

`vision_analyze` 도구 자체도 동일한 라우팅을 따릅니다. 활성 주 모델이 비전을 지원하고 해당 공급자가 도구 결과 내부의 이미지 콘텐츠를 지원하는 경우(현재 Anthropic, OpenAI, Azure-OpenAI 및 Gemini 3.x 스택), `vision_analyze`는 보조 설명 모델을 건너뛰고 원시 이미지 픽셀을 멀티모달 도구 결과 봉투로 반환합니다. 주 모델은 다음 턴에 이미지를 네이티브로 확인하므로 보조 호출, 텍스트 요약으로 인한 정보 손실, 추가 지연이 없습니다.

텍스트 전용 주 모델이거나 도구 결과 채널이 이미지를 전달하지 않는 공급자인 경우 `vision_analyze`는 기존 경로로 대체됩니다. 구성된 보조 비전 모델에 이미지 설명을 요청하고 설명을 일반 텍스트로 반환합니다. 어느 경우든 호출하는 도구의 시그니처는 동일하며, 도구는 활성 모델에 따라 런타임에 어느 경로를 사용할지 결정합니다.

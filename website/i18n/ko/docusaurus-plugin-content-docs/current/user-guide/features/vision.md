---
title: 비전 및 이미지 붙여넣기 (Vision & Image Paste)
description: 클립보드의 이미지를 Hermes CLI에 붙여넣어 멀티모달 비전 분석을 수행하세요.
sidebar_label: 비전 및 이미지 붙여넣기
sidebar_position: 7
---

# 비전 및 이미지 붙여넣기 (Vision & Image Paste)

Hermes Agent는 **멀티모달 비전(multimodal vision)**을 지원합니다. 클립보드의 이미지를 CLI에 직접 붙여넣고 에이전트에게 분석, 설명 또는 관련 작업을 요청할 수 있습니다. 이미지는 base64로 인코딩된 콘텐츠 블록으로 모델에 전송되므로, 비전 기능을 지원하는 모든 모델에서 처리할 수 있습니다.

:::tip
Portal 구독자는 추가 자격 증명 없이 동일한 카탈로그에서 비전 지원 모델(Claude, GPT-5, Gemini)을 사용할 수 있습니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 작동 방식

1. 이미지를 클립보드에 복사합니다 (스크린샷, 브라우저 이미지 등).
2. 아래 방법 중 하나를 사용하여 이미지를 첨부합니다.
3. 질문을 입력하고 Enter를 누릅니다.
4. 이미지가 입력창 위에 `[📎 Image #1]` 배지로 표시됩니다.
5. 제출 시 이미지가 비전 콘텐츠 블록으로 모델에 전송됩니다.

전송하기 전에 여러 이미지를 첨부할 수 있으며, 각각 고유한 배지가 부여됩니다. 연결된 모든 이미지를 지우려면 `Ctrl+C`를 누릅니다.

이미지는 `~/.hermes/images/`에 타임스탬프가 찍힌 파일명의 PNG 파일로 저장됩니다.

## 붙여넣기 방법

이미지를 첨부하는 방법은 터미널 환경에 따라 다릅니다. 모든 방법이 모든 곳에서 작동하는 것은 아닙니다. 전체 분류는 다음과 같습니다.

### `/paste` 명령어

**가장 안정적인 명시적 이미지 첨부 대체 수단입니다.**

```
/paste
```

`/paste`를 입력하고 Enter를 누르세요. Hermes가 클립보드에서 이미지를 확인하고 첨부합니다. 터미널이 `Cmd+V`/`Ctrl+V`를 덮어쓰거나, 이미지만 복사되어 검사할 텍스트 페이로드(bracketed-paste)가 없는 경우 가장 안전한 옵션입니다.

### Ctrl+V / Cmd+V

Hermes는 이제 붙여넣기를 계층화된 흐름으로 처리합니다.
- 일반 텍스트 붙여넣기를 먼저 시도
- 터미널이 텍스트를 깔끔하게 전달하지 못한 경우 네이티브 클립보드 / OSC52 텍스트로 폴백(fallback)
- 클립보드나 붙여넣은 페이로드가 이미지 또는 이미지 경로로 확인되는 경우 이미지 첨부

즉, 붙여넣은 macOS 스크린샷 임시 경로나 `file://...` 이미지 URI가 원시 텍스트로 작성기(composer)에 남아있는 대신 즉시 첨부될 수 있습니다.

:::warning
클립보드에 **이미지만** (텍스트 없음) 있는 경우, 터미널은 여전히 바이너리 이미지 데이터를 직접 전송할 수 없습니다. 명시적인 이미지 첨부 대안으로 `/paste`를 사용하세요.
:::

### VS Code / Cursor / Windsurf를 위한 `/terminal-setup`

macOS의 로컬 VS Code 제품군 통합 터미널 내에서 TUI를 실행하는 경우, Hermes는 향상된 다중 행 입력 및 실행 취소/다시 실행 패리티를 위해 권장되는 `workbench.action.terminal.sendSequence` 바인딩을 설치할 수 있습니다.

```text
/terminal-setup
```

IDE가 `Cmd+Enter`, `Cmd+Z` 또는 `Shift+Cmd+Z`를 가로채는 경우 특히 유용합니다. 로컬 머신에서만 실행하고 SSH 세션 내에서는 실행하지 마세요.

## 플랫폼 호환성

| 환경 | `/paste` | Cmd/Ctrl+V | `/terminal-setup` | 참고 |
|---|:---:|:---:|:---:|---|
| **macOS 터미널 / iTerm2** | ✅ | ✅ | 해당 없음 | 최상의 경험 — 네이티브 클립보드 + 스크린샷 경로 복구 |
| **Apple 터미널** | ✅ | ✅ | 해당 없음 | Cmd+←/→/⌫가 덮어써지는 경우 Ctrl+A / Ctrl+E / Ctrl+U 폴백 사용 |
| **Linux X11 데스크톱** | ✅ | ✅ | 해당 없음 | `xclip` 필요 (`apt install xclip`) |
| **Linux Wayland 데스크톱** | ✅ | ✅ | 해당 없음 | `wl-paste` 필요 (`apt install wl-clipboard`) |
| **WSL2 (Windows 터미널)** | ✅ | ✅ | 해당 없음 | `powershell.exe` 사용 — 추가 설치 필요 없음 |
| **VS Code / Cursor / Windsurf (로컬)** | ✅ | ✅ | ✅ | 향상된 Cmd+Enter / 실행 취소 / 다시 실행 패리티를 위해 권장 |
| **VS Code / Cursor / Windsurf (SSH)** | ❌² | ❌² | ❌³ | 로컬 머신에서 `/terminal-setup`을 실행하세요 |
| **SSH 터미널 (모두)** | ❌² | ❌² | 해당 없음 | 원격 클립보드에 접근할 수 없음 |

² 아래의 [SSH 및 원격 세션](#ssh--remote-sessions)을 참조하세요.
³ 이 명령어는 로컬 IDE 키바인딩을 작성하므로 원격 호스트에서 실행해서는 안 됩니다.

## 플랫폼별 설정

### macOS

**설정이 필요하지 않습니다.** Hermes는 클립보드를 읽기 위해 `osascript`(macOS에 내장됨)를 사용합니다. 성능을 높이려면 선택적으로 `pngpaste`를 설치하세요.

```bash
brew install pngpaste
```

### Linux (X11)

`xclip`을 설치하세요.

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux (Wayland)

최신 Linux 데스크톱(Ubuntu 22.04+, Fedora 34+)은 기본적으로 Wayland를 자주 사용합니다. `wl-clipboard`를 설치하세요.

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip Wayland 사용 여부 확인 방법
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland, "x11" = X11, "tty" = 디스플레이 서버 없음
```
:::

### WSL2

**추가 설정이 필요하지 않습니다.** Hermes는 ( `/proc/version`을 통해) 자동으로 WSL2를 감지하고 `powershell.exe`를 사용하여 .NET의 `System.Windows.Forms.Clipboard`를 통해 Windows 클립보드에 접근합니다. 이것은 WSL2의 Windows 상호 운용성에 내장되어 있으며 `powershell.exe`는 기본적으로 사용할 수 있습니다.

클립보드 데이터는 stdout을 통해 base64로 인코딩된 PNG로 전송되므로 파일 경로 변환이나 임시 파일이 필요하지 않습니다.

:::info WSLg 참고
WSLg(GUI를 지원하는 WSL2)를 실행하는 경우 Hermes는 먼저 PowerShell 경로를 시도한 다음 `wl-paste`로 폴백합니다. WSLg의 클립보드 브리지는 이미지용으로 BMP 형식만 지원합니다. Hermes는 Pillow(설치된 경우) 또는 ImageMagick의 `convert` 명령을 사용하여 BMP를 PNG로 자동 변환합니다.
:::

#### WSL2 클립보드 접근 권한 확인

```bash
# 1. WSL 감지 확인
grep -i microsoft /proc/version

# 2. PowerShell 접근 가능성 확인
which powershell.exe

# 3. 이미지를 복사한 후 확인
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# "True"가 출력되어야 합니다.
```

## SSH 및 원격 세션

**클립보드 이미지 붙여넣기는 SSH를 통해 완전히 작동하지 않습니다.** 원격 머신에 SSH로 접속하면 Hermes CLI가 원격 호스트에서 실행됩니다. 클립보드 도구(`xclip`, `wl-paste`, `powershell.exe`, `osascript`)는 자신이 실행되는 머신(로컬 머신이 아닌 원격 서버)의 클립보드를 읽습니다. 따라서 로컬 클립보드 이미지는 원격 측에서 접근할 수 없습니다.

터미널 붙여넣기나 OSC52를 통해 텍스트가 브리지(bridge)될 수는 있지만, 이미지 클립보드 접근 권한 및 로컬 스크린샷 임시 경로는 Hermes를 실행하는 머신에 여전히 종속됩니다.

### SSH를 위한 해결 방법

1. **이미지 파일 업로드** — 이미지를 로컬에 저장하고 `scp`, VSCode의 파일 탐색기(드래그 앤 드롭) 또는 다른 파일 전송 방법을 통해 원격 서버에 업로드합니다. 그런 다음 경로로 참조합니다. *(추후 릴리스에 `/attach <filepath>` 명령이 계획되어 있습니다.)*

2. **URL 사용** — 온라인에서 이미지에 접근할 수 있는 경우 메시지에 URL을 붙여넣으세요. 에이전트는 `vision_analyze`를 사용하여 모든 이미지 URL을 직접 볼 수 있습니다.

3. **X11 포워딩** — X11을 포워딩하기 위해 `ssh -X`로 접속합니다. 이렇게 하면 원격 머신의 `xclip`이 로컬 X11 클립보드에 접근할 수 있습니다. 로컬에서 X 서버가 실행 중이어야 합니다(macOS의 경우 XQuartz, Linux X11 데스크톱의 경우 내장). 큰 이미지의 경우 속도가 느립니다.

4. **메시징 플랫폼 사용** — Telegram, Discord, Slack 또는 WhatsApp을 통해 Hermes에 이미지를 보냅니다. 이러한 플랫폼은 이미지 업로드를 기본적으로 처리하며 클립보드/터미널 제한의 영향을 받지 않습니다.

## 터미널에서 이미지를 붙여넣을 수 없는 이유

이것은 혼동하기 쉬운 일반적인 원인이므로 기술적인 설명은 다음과 같습니다.

터미널은 **텍스트 기반** 인터페이스입니다. Ctrl+V (또는 Cmd+V)를 누르면 터미널 에뮬레이터는 다음을 수행합니다.

1. 클립보드에서 **텍스트 콘텐츠**를 읽습니다.
2. [bracketed paste](https://en.wikipedia.org/wiki/Bracketed-paste) 이스케이프 시퀀스로 래핑합니다.
3. 터미널의 텍스트 스트림을 통해 애플리케이션으로 보냅니다.

클립보드에 이미지(텍스트 없음)만 포함된 경우 터미널에서 보낼 내용이 없습니다. 바이너리 이미지 데이터에 대한 표준 터미널 이스케이프 시퀀스는 없습니다. 터미널은 아무 작업도 수행하지 않습니다.

이것이 Hermes가 별도의 클립보드 확인을 사용하는 이유입니다. 터미널 붙여넣기 이벤트를 통해 이미지 데이터를 받는 대신 서브프로세스를 통해 OS 수준 도구(`osascript`, `powershell.exe`, `xclip`, `wl-paste`)를 직접 호출하여 클립보드를 독립적으로 읽습니다.

## 지원되는 모델

이미지 붙여넣기는 비전 기능을 지원하는 모든 모델에서 작동합니다. 이미지는 OpenAI 비전 콘텐츠 형식의 base64 인코딩 데이터 URL로 전송됩니다.

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

GPT-4 Vision, Claude(비전 포함), Gemini 및 OpenRouter를 통해 제공되는 오픈 소스 멀티모달 모델을 포함하여 대부분의 최신 모델에서 이 형식을 지원합니다.

## 이미지 라우팅 (비전 지원 모델 vs 텍스트 전용 모델)

사용자가 CLI 클립보드, 게이트웨이(Telegram/Discord 사진) 또는 기타 진입점을 통해 이미지를 첨부하면 Hermes는 현재 모델이 비전을 실제로 지원하는지 여부에 따라 이미지를 라우팅합니다.

| 현재 모델 | 이미지 처리 방식 |
|---|---|
| **비전 지원 모델** (GPT-4V, 비전 지원 Claude, Gemini, Qwen-VL, MiMo-VL 등) | 위의 제공업체 네이티브 이미지 콘텐츠 형식을 사용하여 **실제 픽셀**로 전송됩니다. 텍스트 요약 계층이 없습니다. |
| **텍스트 전용 모델** (DeepSeek V3, 소규모 오픈 소스 모델, 이전 채팅 전용 엔드포인트) | `vision_analyze` 보조 도구를 통해 라우팅됩니다. 보조 비전 모델이 이미지를 설명하고, 텍스트 설명이 대화에 주입됩니다. |

사용자가 이를 구성할 필요는 없습니다. Hermes가 제공업체 메타데이터에서 현재 모델의 기능을 조회하고 적절한 경로를 자동으로 선택합니다. 실질적인 효과: 세션 도중에 비전 모델과 비 비전 모델 간에 전환할 수 있으며, 워크플로를 변경하지 않고도 이미지 처리가 "그냥 작동(just works)"합니다. 텍스트 전용 모델은 거부해야 할 깨진 멀티모달 페이로드 대신 이미지에 대한 일관된 컨텍스트를 얻습니다.

텍스트 설명 경로를 처리할 보조 모델은 `auxiliary.vision`에서 구성할 수 있습니다. [보조 모델(Auxiliary Models)](/user-guide/configuration#auxiliary-models)을 참조하세요.

### `vision_analyze`의 동일한 이중 동작

`vision_analyze` 도구 자체도 동일한 라우팅을 따릅니다. 활성 주 모델이 비전을 지원**하고** 해당 제공업체가 도구 결과 내에서 이미지 콘텐츠를 지원하는 경우(현재 Anthropic, OpenAI, Azure-OpenAI 및 Gemini 3.x 스택), `vision_analyze`는 보조 설명자를 건너뛰고 원시 이미지 픽셀을 멀티모달 도구 결과 봉투(envelope)로 반환합니다. 주 모델은 다음 차례에서 이미지를 기본적으로 봅니다. 보조 호출, 텍스트 요약 정보 손실 또는 추가 지연 시간이 없습니다.

텍스트 전용 주 모델(또는 도구 결과 채널이 이미지를 전달하지 않는 제공업체)의 경우 `vision_analyze`는 레거시 경로로 폴백합니다. 즉, 구성된 보조 비전 모델에 이미지를 설명하도록 요청하고 일반 텍스트로 설명을 반환합니다. 어느 경로든 호출 도구 서명은 동일하며, 활성 모델에 따라 실행 시에 어떤 경로를 취할지 도구가 결정합니다.

---
title: "Windows(네이티브) 가이드"
description: "Windows 10 / 11에서 Hermes Agent를 네이티브로 실행하는 방법 — 설치, 기능 매트릭스, UTF-8 콘솔, Git Bash, 예약 작업으로 게이트웨이 실행, 에디터 처리, PATH, 제거 및 일반적인 문제"
sidebar_label: "Windows(네이티브)"
sidebar_position: 3
---

# Windows(네이티브) 가이드

Hermes는 WSL, Cygwin, Docker 없이 Windows 10과 Windows 11에서 네이티브로 실행됩니다. 이 페이지에서는 네이티브 환경에서 작동하는 것, WSL 전용 기능, 설치 프로그램이 실제로 수행하는 작업, 그리고 조정해야 할 수 있는 Windows별 설정을 자세히 설명합니다.

설치만 필요하다면 [랜딩 페이지](/) 또는 [설치 페이지](../getting-started/installation#windows-native)의 한 줄 명령이면 충분합니다. 예상과 다르게 동작하는 부분이 있을 때 이 페이지로 돌아오세요.

:::tip WSL을 사용하시겠어요?
실제 POSIX 환경(대시보드의 내장 터미널, `fork` 의미론, Linux 스타일 파일 감시기 등)을 선호한다면 **[Windows(WSL2) 가이드](./windows-wsl-quickstart.md)**를 참조하세요. 두 환경은 깔끔하게 공존합니다. 네이티브 데이터는 `%LOCALAPPDATA%\hermes` 아래에, WSL 데이터는 `~/.hermes` 아래에 저장됩니다.
:::

## 빠른 설치

**PowerShell**(또는 Windows Terminal)을 열고 실행하세요.

```powershell
iex (irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1)
```

관리자 권한은 필요하지 않습니다. 설치 프로그램은 `%LOCALAPPDATA%\hermes\`에 설치하고 `hermes`를 **사용자 PATH**에 추가합니다. 설치가 끝나면 새 터미널을 여세요.

**설치 프로그램 옵션**(매개변수를 전달하려면 scriptblock 형식이 필요합니다):

```powershell
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| 매개변수 | 기본값 | 용도 |
|---|---|---|
| `-Branch` | `main` | 특정 브랜치를 클론합니다(PR 테스트에 유용). |
| `-Commit` | 설정되지 않음 | 특정 커밋 SHA로 설치를 고정합니다(`-Branch`보다 우선). |
| `-Tag` | 설정되지 않음 | 특정 git 태그로 설치를 고정합니다(예: `v0.14.0`). |
| `-NoVenv` | 끔 | venv 생성을 건너뜁니다(고급 — Python을 직접 관리해야 함). |
| `-SkipSetup` | 끔 | 설치 후 `hermes setup` 마법사를 건너뜁니다. |
| `-HermesHome` | `%LOCALAPPDATA%\hermes` | 데이터 디렉터리를 재정의합니다. |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | 코드 위치를 재정의합니다. |

설치 프로그램은 불안정한 git fetch를 자동으로 재시도하고, 다운로드한 `install.ps1` 페이로드에서 BOM을 제거합니다. 따라서 HTTP 전송 중 가져온 UTF-8 BOM 때문에 `[scriptblock]::Create((irm ...))` 형식이 더 이상 깨지지 않습니다.

### 데스크톱 설치 프로그램(대안)

PowerShell을 여는 대신 `.exe`를 두 번 클릭하고 싶다면 얇은 GUI 설치 프로그램을 사용할 수도 있습니다. Hermes Desktop을 다운로드하고 설치 프로그램을 실행하면, 최초 실행 시 GUI가 내부적으로 `install.ps1`을 호출하여 Python(`uv`를 통해), Node, PortableGit 및 아래에 설명된 나머지 의존성 부트스트랩을 준비합니다. 최초 실행 후에는 데스크톱 앱과 PowerShell로 설치한 `hermes` CLI가 동일한 `%LOCALAPPDATA%\hermes\hermes-agent` 설치 경로와 `%LOCALAPPDATA%\hermes` 데이터 디렉터리를 공유하므로 GUI와 CLI를 자유롭게 오갈 수 있습니다.

익숙한 Windows 설치 경험을 원하거나 Hermes를 개발자가 아닌 사용자에게 전달할 때는 데스크톱 설치 프로그램을 사용하세요. 이미 터미널을 사용 중이라면 PowerShell 한 줄 명령이 적합합니다.

### 의존성 부트스트랩(`dep_ensure`)

최초 실행 시(그리고 누락된 도구가 감지될 때 필요에 따라) Hermes는 필요한 Python 외 의존성을 확인하고 지연 설치하는 작은 Python 부트스트래퍼인 `hermes_cli/dep_ensure.py`를 실행합니다. Windows에서 관련된 의존성은 다음과 같습니다.

| 의존성 | Hermes가 필요한 이유 |
|---|---|
| **PortableGit** | 터미널 도구에 `bash.exe`를 제공하고 세션 내 클론에 `git`을 제공합니다. `dep_ensure`가 아니라 설치 시 준비됩니다. |
| **Node.js 26** | 브라우저 도구(`agent-browser`), TUI 웹 브리지 및 WhatsApp 브리지에 필요합니다. |
| **ffmpeg** | TTS/음성 메시지의 오디오 형식 변환에 사용됩니다. |
| **ripgrep** | 빠른 파일 검색에 사용됩니다. 사용할 수 없으면 `grep`으로 대체됩니다. |
| **npm 패키지** | `agent-browser`, Playwright Chromium 및 도구를 처음 사용할 때 설치되는 도구 세트별 Node 의존성입니다. |

각 의존성은 `shutil.which(...)` 방식으로 확인됩니다. 바이너리가 없고 실행이 대화형이면 `dep_ensure`가 설치 여부를 묻고, 실제 설치 로직은 `scripts\install.ps1 -ensure <dep>`에 위임합니다. 비대화형 실행(게이트웨이, cron, 헤드리스 데스크톱 실행)은 프롬프트를 건너뛰고 대신 `this feature needs <dep>`라는 명확한 오류를 표시합니다.

## 설치 프로그램이 실제로 하는 일

위에서 아래 순서로 실행됩니다.

1. **`uv`를 부트스트랩합니다** — Astral의 빠른 Python 관리자입니다. `%USERPROFILE%\.local\bin`에 설치됩니다.
2. **`uv`를 통해 Python 3.11을 설치합니다.** 기존 Python은 필요하지 않습니다.
3. **Node.js 26을 설치합니다**(사용할 수 있으면 winget, 그렇지 않으면 `%LOCALAPPDATA%\hermes\node` 아래에 압축을 푸는 포터블 Node tarball). 브라우저 도구와 WhatsApp 브리지에 사용됩니다.
4. **포터블 Git을 설치합니다** — `git`이 이미 PATH에 있으면 설치 프로그램이 이를 사용하고, 그렇지 않으면 공식 `git-for-windows` 릴리스에서 축소된 독립형 **PortableGit**(약 45MB)을 `%LOCALAPPDATA%\hermes\git`에 다운로드합니다. 관리자 권한, Windows 설치 프로그램 레지스트리, 다른 시스템 요소와의 간섭이 없습니다.
5. 저장소를 `%LOCALAPPDATA%\hermes\hermes-agent`에 **클론하고 그 안에 가상 환경을 생성합니다.**
6. **단계별 `uv pip install`** — 먼저 `.[all]`을 시도하고, GitHub의 rate limit으로 `git+https` 의존성이 불안정하면 점점 작은 세트(`[messaging,dashboard,ext]` → `[messaging]` → `.`)로 대체합니다. 하나의 일시적 오류 때문에 최소 설치로 떨어지는 문제를 방지합니다.
7. **`.env`에 따라 메시징 SDK를 자동 설치합니다** — `TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED` 중 하나가 있으면 `python -m ensurepip --upgrade`와 대상별 `pip install`을 실행하여 각 플랫폼의 SDK를 실제로 import할 수 있게 합니다.
8. 새 셸에서 Hermes가 `bash.exe`를 확실히 찾도록 **`HERMES_GIT_BASH_PATH`를 확인된 경로로 설정합니다.**
9. **`%LOCALAPPDATA%\hermes\hermes-agent\bin`을 사용자 PATH에 추가하고 `HERMES_HOME=%LOCALAPPDATA%\hermes`를 설정합니다** — 새 터미널을 연 후 `hermes` 명령을 사용할 수 있고 데이터 디렉터리를 가리키게 됩니다. `bin` 디렉터리에는 `hermes.exe`/`hermes-acp.exe` 실행 파일만 복사됩니다. 전체 `venv\Scripts`를 PATH에 넣지 않는 것은 Hermes가 사용자의 `python` 명령을 가리지 않게 하기 위해서입니다.
10. **`hermes setup`을 실행합니다** — 일반적인 최초 실행 마법사(모델, 공급자, 도구 세트)를 실행합니다. `-SkipSetup`으로 건너뛸 수 있습니다.

:::tip Windows에서 공급자 찾기를 건너뛰기
Windows에서 도구를 제대로 활용하기 위한 가장 번거로운 부분은 도구별 API 키(Firecrawl, FAL, Browser Use, OpenAI TTS)를 설정하는 일입니다. [Nous Portal](/user-guide/features/tool-gateway) 구독을 사용하면 하나의 OAuth 로그인으로 모델과 해당 도구를 모두 이용할 수 있습니다. 설치 프로그램이 끝난 뒤 `hermes setup --portal`을 실행하여 모든 항목을 연결하세요.
:::

## 기능 매트릭스

대시보드의 내장 터미널 창을 제외한 모든 기능은 Windows에서 네이티브로 실행됩니다.

| 기능 | 네이티브 Windows | WSL2 |
|---|---|---|
| CLI(`hermes chat`, `hermes setup`, `hermes gateway`, …) | ✓ | ✓ |
| 대화형 TUI(`hermes --tui`) | ✓ | ✓ |
| 메시징 게이트웨이(Telegram, Discord, Slack, WhatsApp, 15개 이상의 플랫폼) | ✓ | ✓ |
| Cron 스케줄러 | ✓ | ✓ |
| 브라우저 도구(Node를 통한 Chromium) | ✓ | ✓ |
| MCP 서버(stdio 및 HTTP) | ✓ | ✓ |
| 로컬 Ollama / LM Studio / llama-server | ✓ | ✓(WSL 네트워킹을 통해) |
| 웹 대시보드(세션, 작업, 메트릭, 설정) | ✓ | ✓ |
| 대시보드 `/chat` 내장 터미널 창 | ✗(POSIX PTY 필요) | ✓ |
| 로그인 시 자동 시작 | ✓(schtasks) | ✓(systemd) |

대시보드의 `/chat` 탭은 POSIX PTY(`ptyprocess`)를 통해 실제 터미널을 내장합니다. 네이티브 Windows에는 이에 상응하는 기본 요소가 없습니다. Python의 `pywinpty`/Windows ConPTY로 구현할 수는 있지만 별도의 구현이 필요하므로 향후 작업으로 취급합니다. **대시보드의 나머지 기능은 네이티브로 작동합니다**. 해당 탭 하나만 "이 기능에는 WSL2를 사용하세요" 배너를 표시합니다.

## Hermes가 Windows에서 셸 명령을 실행하는 방법

Hermes의 터미널 도구는 Claude Code와 동일한 방식으로 **Git Bash**를 통해 명령을 실행합니다. 이 방식은 모든 도구를 다시 작성하지 않고 POSIX와 Windows의 차이를 우회합니다.

`bash.exe` 검색 순서는 다음과 같습니다.

1. 설정되어 있으면 `HERMES_GIT_BASH_PATH` 환경 변수.
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe`(설치 프로그램이 관리하는 PortableGit).
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe`(이전 Git-for-Windows 레이아웃).
4. 시스템 Git-for-Windows 설치(`%ProgramFiles%\Git\bin\bash.exe` 등).
5. 마지막 수단으로 PATH에 있는 MSYS2, Cygwin 또는 모든 `bash.exe`.

설치 프로그램은 `HERMES_GIT_BASH_PATH`를 명시적으로 설정하므로 새 PowerShell 세션에서 다시 검색할 필요가 없습니다. Hermes가 특정 bash를 사용하도록 하려면 재정의하세요. 예를 들어 시스템 Git Bash나 심볼릭 링크를 통해 WSL 호스트의 bash를 사용할 수 있습니다.

**주의:** MinGit의 레이아웃은 전체 Git-for-Windows 설치와 다릅니다. bash는 `bin\bash.exe`가 아니라 `usr\bin\bash.exe` 아래에 있습니다. Hermes는 두 위치를 모두 확인합니다. MinGit zip을 수동으로 압축 해제한다면 **non-busybox** 변형(`MinGit-*-64-bit.zip`, `MinGit-*-busybox*.zip` 아님)을 선택하세요. busybox 빌드는 `bash` 대신 `ash`를 제공하며 대부분의 coreutils가 빠져 있습니다.

## Windows의 UTF-8 콘솔

Windows에서 Python의 기본 stdio는 콘솔의 활성 코드 페이지(대개 cp1252 또는 cp437)를 사용합니다. Hermes의 배너, 슬래시 명령 목록, 도구 피드, Rich 패널 및 스킬 설명에는 Unicode가 포함되어 있습니다. 별도 처리가 없으면 `UnicodeEncodeError: 'charmap' codec can't encode character…`가 발생합니다.

해결 방법은 모든 진입점(`cli.py::main`, `hermes_cli/main.py::main`, `gateway/run.py::main`)에서 초기에 호출되는 `hermes_cli/stdio.py::configure_windows_stdio()`에 구현되어 있습니다. 이 함수는 다음을 수행합니다.

1. `kernel32.SetConsoleCP`/`SetConsoleOutputCP`를 통해 콘솔 코드 페이지를 CP_UTF8(65001)로 전환합니다.
2. `sys.stdout`/`sys.stderr`/`sys.stdin`을 `errors='replace'`를 사용하는 UTF-8로 재구성합니다.
3. `PYTHONIOENCODING=utf-8` 및 `PYTHONUTF8=1`을 설정합니다(`setdefault`를 사용하므로 명시적인 사용자 값이 우선). 이에 따라 자식 Python 하위 프로세스가 UTF-8을 상속합니다.
4. `EDITOR`와 `VISUAL`이 모두 설정되지 않았으면 `EDITOR=notepad`를 설정합니다(아래 에디터 섹션 참조).

멱등적이며 Windows가 아닌 환경에서는 아무 작업도 하지 않습니다.

**비활성화:** 환경 변수에 `HERMES_DISABLE_WINDOWS_UTF8=1`을 설정하면 레거시 cp1252 stdio 경로로 돌아갑니다. 인코딩 문제를 이분 탐색할 때 유용하지만 일반적인 작업에서는 적절한 설정이 아닐 가능성이 큽니다.

## 에디터(`Ctrl-X Ctrl-E`, `/edit`)

#21561 이전에는 Windows에서 `Ctrl-X Ctrl-E`를 누르거나 `/edit`를 입력해도 아무 일도 일어나지 않았습니다. prompt_toolkit에는 POSIX 절대 경로로 된 대체 목록(`/usr/bin/nano`, `/usr/bin/pico`, `/usr/bin/vi` 등)이 하드코딩되어 있어, Git for Windows를 완전히 설치한 경우에도 Windows에서는 확인되지 않았습니다.

이제 Hermes의 Windows stdio shim은 기본값으로 `EDITOR=notepad`를 설정합니다. Notepad는 모든 Windows 설치에 포함되어 있고 블로킹 에디터로 작동하므로 `subprocess.call(["notepad", file])`은 창이 닫힐 때까지 대기합니다.

**사용자 재정의가 우선합니다**(setdefault로 설정하기 전에 확인됨).

| 에디터 | PowerShell 명령 |
|---|---|
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

PowerShell 프로필에 설정하여 영구적으로 적용할 수 있습니다.

```powershell
# In $PROFILE
$env:EDITOR = "code --wait"
```

또는 시스템 설정의 사용자 환경 변수로 설정하여 새 셸마다 적용되게 하세요.

## CLI에서 줄바꿈하는 `Ctrl+Enter`

Windows Terminal은 `Ctrl+Enter`를 전용 키 시퀀스로 전달합니다. Hermes는 이를 "줄바꿈 삽입"에 연결하므로 CLI에서 `Esc`를 누른 다음 `Enter`를 사용하는 방식으로 전환하지 않고 여러 줄 프롬프트를 작성할 수 있습니다. Windows Terminal, VS Code 통합 터미널 및 VT 이스케이프 시퀀스를 지원하는 최신 Windows 콘솔 호스트에서 작동합니다.

레거시 `cmd.exe` 콘솔에서는 `Ctrl+Enter`가 일반 `Enter`로 축약됩니다. 대신 `Esc Enter`를 사용하거나 Windows Terminal로 업그레이드하세요(무료이며 Windows 11에 기본 설치됨).

## Windows 로그인 시 게이트웨이 실행

Windows에서 `hermes gateway install`은 관리자 권한이 필요 없는 **예약 작업**과 Startup 폴더 대체 경로를 사용합니다.

### 설치

```powershell
hermes gateway install
```

내부적으로 다음 작업이 수행됩니다.

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` — 표준(권한 상승 없음) 권한으로 로그인 시 실행되는 작업을 등록합니다. UAC 프롬프트가 없습니다.
2. 그룹 정책으로 schtasks가 차단되면 `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`에 `start /min cmd.exe /d /c <wrapper>` 바로 가기를 기록하는 방식으로 대체합니다. 효과는 같지만 조금 더 단순합니다.
3. 게이트웨이를 **`pythonw.exe`를 통해 분리된 상태로** 실행합니다 — `python.exe`가 아닙니다. `pythonw.exe`에는 연결된 콘솔이 없으므로 같은 프로세스 그룹의 다른 프로세스에서 발생한 `CTRL_C_EVENT` 브로드캐스트로부터 안전합니다(같은 프로세스 그룹에서 Ctrl+C를 눌렀을 때 게이트웨이가 종료되던 실제 문제를 해결).

실행 시 사용하는 플래그: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`.

### 관리

```powershell
hermes gateway status      # Merged view: schtasks + Startup folder + running PID
hermes gateway start       # Starts the scheduled task now
hermes gateway stop        # Graceful SIGTERM equivalent (TerminateProcess via psutil)
hermes gateway restart
hermes gateway uninstall   # Removes schtasks entry, Startup shortcut, pid file
```

`hermes gateway status`는 멱등적입니다. 연속으로 천 번 호출해도 실수로 게이트웨이를 종료하지 않습니다. (PR #21561 이전에는 `os.kill(pid, 0)`이 C 수준에서 `CTRL_C_EVENT`와 충돌하여 조용히 종료시키기도 했습니다. 자세한 이야기는 아래 "프로세스 관리 내부"를 참조하세요.)

### 왜 Windows 서비스가 아닌가요?

서비스를 설치하려면 관리자 권한이 필요하고 게이트웨이 수명 주기가 사용자 로그인이 아니라 컴퓨터 부팅에 연결됩니다. 일반적인 Hermes 사용자가 원하는 것은 로그인하면 게이트웨이를 사용할 수 있고 로그아웃하면 사라지는 것입니다. 예약 작업은 권한 상승 없이 이를 정확히 수행합니다. 정말 서비스가 필요하다면 `nssm` 또는 `sc create`를 수동으로 사용하세요. 하지만 대부분의 경우 필요하지 않습니다.

## 데이터 레이아웃

| 경로 | 내용 |
|---|---|
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Git 체크아웃 + venv. `bin\hermes.exe` 실행 파일(`venv\Scripts\hermes.exe`에서 복사됨)은 사용자 PATH에 추가되는 명령입니다. `Remove-Item -Recurse`로 제거하고 재설치해도 안전합니다. |
| `%LOCALAPPDATA%\hermes\git\` | PortableGit(설치 프로그램이 준비한 경우에만). |
| `%LOCALAPPDATA%\hermes\node\` | 포터블 Node.js(설치 프로그램이 준비한 경우에만). |
| `%LOCALAPPDATA%\hermes\bin\` | Hermes가 업데이트에 사용하는 관리 대상 `uv.exe`(Python 관리자). |
| `%LOCALAPPDATA%\hermes\`(루트) | 사용자의 설정, 인증 정보, 스킬, 세션, 로그(`config.yaml`, `.env`, `skills\`, `sessions\`, `logs\` 등). **재설치 후에도 유지됩니다.** |

네이티브 Windows에서 설치 프로그램은 `HERMES_HOME=%LOCALAPPDATA%\hermes`를 설정하므로 데이터와 폐기 가능한 설치가 동일한 `%LOCALAPPDATA%\hermes` 루트 아래에 있습니다. 설치/런타임은 `hermes-agent\`, `git\`, `node\`, `bin\` 하위 디렉터리이고, 데이터 파일은 `%LOCALAPPDATA%\hermes`에 직접 저장됩니다. 재설치하면 `hermes-agent\` 체크아웃만 교체되므로 데이터는 유지됩니다. 다만 둘이 하나의 루트를 공유하므로 데이터를 유지하려면 `%LOCALAPPDATA%\hermes`에 `Remove-Item -Recurse`를 실행하지 말고 `hermes-agent\` 하위 디렉터리만 삭제하세요. 데이터 디렉터리 구조는 Linux의 `~/.hermes`와 동일하므로 컴퓨터 간에 미러링할 수 있습니다.

**`HERMES_HOME` 재정의:** 환경 변수를 다른 데이터 디렉터리(예: Linux/WSL 레이아웃에 맞추기 위한 `%USERPROFILE%\.hermes`)를 가리키도록 설정하세요. Linux에서와 동일하게 작동합니다.

## 브라우저 도구

브라우저 도구는 Node 헬퍼인 `agent-browser`를 사용하여 Chromium을 제어합니다. Windows에서는 다음과 같이 작동합니다.

- 설치 프로그램이 npm을 통해 `agent-browser`를 PATH에 추가합니다.
- `shutil.which("agent-browser", path=...)`는 `.cmd` shim을 자동으로 찾습니다. `CreateProcessW`는 확장자가 없는 shebang을 실행할 수 없으므로 Hermes는 항상 `.CMD` 래퍼를 확인합니다. shebang 스크립트를 직접 호출하지 말고 항상 `.cmd`를 사용하세요.
- Playwright Chromium은 최초 실행 시 자동으로 설치됩니다(`npx playwright install chromium`). 설치에 실패하면 `hermes doctor`가 수정 안내와 함께 알려줍니다.

## Windows에서 Hermes 실행 — 실용적인 참고 사항

### 설치 후 PATH

설치 프로그램은 `[Environment]::SetEnvironmentVariable`을 통해 `%LOCALAPPDATA%\hermes\hermes-agent\bin`을 **사용자 PATH**에 추가합니다. 이미 열려 있는 터미널은 이를 반영하지 않으므로 설치 후 새 PowerShell 창(또는 Windows Terminal 탭)을 여세요. 알고 있는 경우가 아니라면 직접 `$env:PATH += …`를 사용하지 말고 닫았다가 다시 여세요.

확인:

```powershell
Get-Command hermes        # should print C:\Users\<you>\AppData\Local\hermes\hermes-agent\bin\hermes.exe
hermes --version
```

### 환경 변수

Hermes는 `$env:X`(프로세스 범위)와 사용자 환경 변수(시스템 속성 → 환경 변수에서 영구적으로 설정)를 모두 지원합니다. `%LOCALAPPDATA%\hermes\.env`(`HERMES_HOME`)에 API 키를 설정하는 것이 일반적인 방법이며 Linux와 같습니다.

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

특별히 모든 Windows 프로세스가 API 키를 보게 하려는 것이 아니라면 사용자 환경 변수에 비밀을 넣지 마세요(원하는 방식이 아닙니다).

### Windows 전용 환경 변수

다음은 네이티브 Windows 설치에만 영향을 줍니다.

| 변수 | 효과 |
|---|---|
| `HERMES_GIT_BASH_PATH` | bash.exe 검색을 재정의합니다. 모든 bash(전체 Git-for-Windows, 심볼릭 링크를 통한 WSL bash, MSYS2, Cygwin)를 가리킬 수 있습니다. 설치 프로그램이 자동으로 설정합니다. |
| `HERMES_DISABLE_WINDOWS_UTF8` | `1`로 설정하면 UTF-8 stdio shim을 비활성화하고 로캘 코드 페이지로 돌아갑니다. 인코딩 문제를 이분 탐색할 때 유용합니다. |
| `EDITOR` / `VISUAL` | `/edit`와 `Ctrl-X Ctrl-E`에 사용할 에디터입니다. 둘 다 설정되지 않으면 Hermes는 `notepad`를 기본값으로 사용합니다. |

## 제거

PowerShell에서 실행하세요.

```powershell
hermes uninstall
```

이 방법이 가장 깔끔합니다. schtasks 항목, Startup 폴더 바로 가기, `hermes.cmd` shim을 제거하고 `%LOCALAPPDATA%\hermes\hermes-agent\`를 삭제하며 사용자 PATH를 정리합니다. 재설치할 경우를 위해 `%LOCALAPPDATA%\hermes\`의 나머지(설정, 인증 정보, 스킬, 세션, 로그)는 그대로 둡니다.

모두 삭제하려면 다음을 실행하세요.

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
# Also remove a legacy CLI/WSL data dir if you ever used one:
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
```

`hermes uninstall` CLI 하위 명령은 schtasks 항목이 다른 작업 이름으로 등록된 경우(이전 설치)도 처리합니다. 하드코딩된 작업 이름이 아니라 설치 경로로 검색합니다.

## 프로세스 관리 내부

이 내용은 배경 지식입니다. "스스로 종료되는" 이상한 현상을 디버깅할 때만 읽어도 됩니다.

Linux와 macOS에서 POSIX 관용구인 `os.kill(pid, 0)`은 권한을 확인하는 무작업 호출로, "이 PID가 살아 있고 신호를 보낼 수 있는가?"를 확인합니다. Windows에서 Python의 `os.kill`은 `sig=0`을 `CTRL_C_EVENT`로 매핑합니다. 둘은 정수 값 0에서 충돌하고 `GenerateConsoleCtrlEvent(0, pid)`를 통해 대상 PID가 속한 **전체 콘솔 프로세스 그룹**에 Ctrl+C를 브로드캐스트합니다. 이는 2012년부터 열려 있는 [bpo-14484](https://bugs.python.org/issue14484)입니다. 현재 동작에 의존하는 스크립트를 깨뜨리므로 수정되지 않을 예정입니다.

그 결과 Windows에서 `os.kill(pid, 0)`으로 "이 PID가 살아 있는지 확인"하던 모든 코드 경로가 대상 프로세스를 조용히 종료하고 있었습니다. Hermes는 이러한 모든 위치(11개 파일에 걸친 14곳)를 `psutil.pid_exists()`를 사용하는 `gateway.status._pid_exists()`로 이전했습니다(`psutil.pid_exists()`는 Windows에서 신호를 사용하지 않고 `OpenProcess + GetExitCodeProcess`를 사용). 플러그인이나 패치를 작성한다면 `psutil.pid_exists()` 또는 `gateway.status._pid_exists()`를 사용하세요. `os.kill(pid, 0)`은 절대 사용하지 마세요.

`scripts/check-windows-footguns.py`가 이를 CI에서 강제합니다. 새 `os.kill(pid, 0)` 호출은 `# windows-footgun: ok — <reason>` 마커가 해당 줄에 있지 않으면 `Windows footguns (blocking)` 검사를 통과하지 못합니다.

## 일반적인 문제

**설치 직후 `hermes: command not found`.**
새 PowerShell 창을 여세요. 설치 프로그램은 `%LOCALAPPDATA%\hermes\hermes-agent\bin`을 사용자 PATH에 추가했지만, 기존 셸은 이를 반영하려면 다시 시작해야 합니다. 그동안 `& "$env:LOCALAPPDATA\hermes\hermes-agent\bin\hermes.exe"`로 실행할 수 있습니다.

**도구 실행 시 `WinError 193: %1 is not a valid Win32 application`.**
`.cmd` shim을 우회하여 shebang 스크립트를 호출한 것입니다. Hermes는 `shutil.which(cmd, path=local_bin)`을 통해 명령을 확인하므로 PATHEXT가 `.CMD`를 찾습니다. 도구를 하드코딩된 경로로 호출하고 있다면 `.cmd` 변형으로 바꾸세요(예: `npx`가 아니라 `npx.cmd`).

**`[scriptblock]::Create(...)`가 `The assignment expression is not valid`와 함께 실패합니다.**
`install.ps1`를 다운로드할 때 UTF-8 BOM이 포함되었습니다. `irm | iex` 형식은 BOM을 자동으로 제거하지만 `[scriptblock]::Create((irm ...))`는 제거하지 않습니다. 간단한 `irm | iex` 형식으로 다시 실행하거나, 다음을 사용하여 스크립트를 수동으로 다운로드하고 BOM 없이 저장하세요: `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`.

**재시작 후 게이트웨이가 계속 실행되지 않습니다.**
`hermes gateway status`를 확인하세요. schtasks 항목, Startup 폴더 바로 가기(사용된 경우), 실행 중인 PID를 통합해서 보여줍니다. schtasks가 등록되어 있지만 실행되지 않는다면 그룹 정책이 `ONLOGON` 트리거를 차단하고 있을 수 있습니다. `schtasks /Query /TN HermesGateway /V /FO LIST`를 실행하여 작업의 실패 원인을 확인하거나, 제거한 다음 `HERMES_GATEWAY_FORCE_STARTUP=1`로 다시 설치하여 Startup 폴더 경로로 대체하세요.

**`$env:EDITOR`를 설정한 후에도 `/edit`가 아무 작업도 하지 않습니다.**
현재 프로세스에만 설정한 것입니다. 셸을 닫고 다시 열거나 시스템 속성 → 환경 변수에서 사용자 범위로 설정하세요. 새 PowerShell 창에서 `echo $env:EDITOR`로 확인합니다.

**브라우저 도구는 실행되지만 도구 시간이 초과됩니다.**
Chromium은 최초 실행 시 자동으로 설치됩니다. 설치가 실패했다면(GitHub rate limit 또는 Playwright CDN 문제) `hermes doctor`를 실행하세요. 누락된 Chromium을 알려주고 수정에 필요한 정확한 `npx playwright install chromium` 명령을 출력합니다.

**`agent-browser`가 이상한 Node 버전 오류와 함께 실패합니다.**
설치 프로그램은 `%LOCALAPPDATA%\hermes\node`에 Node 26을 준비하지만 PATH에 오래된 시스템 Node 18이 먼저 있을 수 있습니다. Hermes의 node 디렉터리를 PATH 앞쪽으로 옮기거나 다른 곳에서 Node를 사용하지 않는다면 시스템 설치를 삭제하세요.

**CLI에서 중국어/일본어/아랍어 문자가 `?`로 표시됩니다.**
UTF-8 stdio shim이 활성화되지 않았습니다. `HERMES_DISABLE_WINDOWS_UTF8`이 설정되어 있지 않은지 확인하세요(`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`). 비어 있는데도 계속 `?`가 표시되면 콘솔 호스트(매우 오래된 `cmd.exe`)가 UTF-8을 전혀 지원하지 않을 수 있습니다. Windows Terminal로 전환하세요.

**게이트웨이가 Telegram 사진을 보낼 수 없습니다 — "`BadRequest: payload contains invalid characters`".**
이는 Windows와 관련이 없지만 Windows에서 처음 드러나는 경우가 있습니다. 대개 JSON 본문에 파일 경로의 백슬래시가 이스케이프되지 않았다는 뜻입니다. Telegram에는 Hermes가 정규화한 경로가 전달되어야 하며, 원시 Windows 경로가 전달되어서는 안 됩니다. 커스텀 플러그인에서 이 오류가 보이면 사용자 입력의 `str(Path(...))`가 아니라 Hermes가 제공하는 경로를 전달하는지 확인하세요.

**`git pull` 후 다른 컴퓨터에서는 작동하던 인코딩이 이상합니다.**
Windows에서 Hermes 설정이나 스킬을 비 UTF-8 에디터(Notepad의 이전 Windows 버전, 일부 중국어 IME 등)로 편집했다면 파일이 BOM과 함께 저장되었을 수 있습니다. Hermes는 대부분의 설정 읽기에서 `utf-8-sig`를 허용하지만, 접힌 YAML 스칼라(`description: >`) 안의 BOM은 YAML 파싱을 조용히 깨뜨립니다. BOM 없는 일반 UTF-8로 파일을 다시 저장하세요.

## 다음 단계

- **[설치](../getting-started/installation.md)** — Linux/macOS/WSL2/Termux를 포함한 전체 설치 페이지.
- **[Windows(WSL2) 가이드](./windows-wsl-quickstart.md)** — POSIX 의미론이나 대시보드 터미널 창이 필요한 경우.
- **[CLI 레퍼런스](../reference/cli-commands.md)** — 모든 `hermes` 하위 명령.
- **[FAQ](../reference/faq.md)** — Windows에만 해당하지 않는 일반적인 질문.
- **[메시징 게이트웨이](./messaging/index.md)** — Windows에서 Telegram/Discord/Slack 실행.

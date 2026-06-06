---
title: "Windows (네이티브) 가이드"
description: "Windows 10 / 11에서 Hermes 에이전트를 네이티브로 실행하세요 — 설치, 기능 매트릭스, UTF-8 콘솔, Git Bash, 예약된 작업으로 게이트웨이 실행, 편집기 처리, PATH, 제거 및 일반적인 문제 해결"
sidebar_label: "Windows (네이티브)"
sidebar_position: 3
---

# Windows (네이티브) 가이드

Hermes는 WSL, Cygwin, Docker 없이 Windows 10 및 Windows 11에서 네이티브로 실행됩니다. 이 문서는 네이티브 환경에서 지원되는 것과 WSL 전용인 것, 설치 관리자가 실제로 수행하는 작업, 그리고 사용자가 건드려야 할 수 있는 Windows 전용 설정들에 대한 심층 가이드입니다.

단순히 설치만 하려는 경우, [랜딩 페이지](/) 또는 [설치 페이지](../getting-started/installation#windows-native-powershell)의 한 줄 명령어만 있으면 됩니다. 진행하다가 예상치 못한 문제에 부딪혔을 때 다시 이 문서를 참조하세요.

:::tip WSL을 선호하시나요?
(대시보드의 내장 터미널, `fork` 시맨틱, Linux 스타일의 파일 감시자 등을 위해) 실제 POSIX 환경을 선호한다면 **[Windows (WSL2) 가이드](./windows-wsl-quickstart.md)** 를 참조하세요. 두 환경은 깔끔하게 공존합니다: 네이티브 데이터는 `%LOCALAPPDATA%\hermes`에, WSL 데이터는 `~/.hermes`에 저장됩니다.
:::

## 빠른 설치 (Quick install)

저희 웹사이트에서 [Hermes Desktop 설치 관리자를 다운로드](https://hermes-agent.nousresearch.com/desktop)하고 실행하세요.

또는 명령줄로만 설치하려면 **PowerShell** (또는 Windows Terminal)을 열고 다음을 실행하세요:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

관리자 권한은 필요하지 않습니다. 설치 관리자는 `%LOCALAPPDATA%\hermes\` 경로에 설치를 진행하며, **사용자 PATH**에 `hermes`를 추가합니다. 설치가 완료된 후 새 터미널을 여세요.

**설치 관리자 옵션** (매개변수를 전달하려면 스크립트 블록 형태가 필요합니다):

```powershell
& ([scriptblock]::Create((irm https://hermes-agent.nousresearch.com/install.ps1))) -NoVenv -SkipSetup -Branch main
```

| 매개변수 | 기본값 | 목적 |
| ------------- | ------------------------------------ | ---------------------------------------------------------- |
| `-Branch` | `main` | 특정 브랜치 복제 (PR 테스트에 유용) |
| `-Commit` | 미설정 | 특정 git 태그(예: `v0.14.0`)로 설치 고정 |
| `-Tag` | 미설정 | 특정 git 태그(예: `v0.14.0`)로 설치 고정 |
| `-NoVenv` | 꺼짐 | venv 생성 건너뛰기 (고급 — 직접 Python을 관리하는 경우) |
| `-SkipSetup` | 꺼짐 | 설치 후 `hermes setup` 마법사 건너뛰기 |
| `-HermesHome` | `%LOCALAPPDATA%\hermes` | 데이터 디렉토리 재정의 |
| `-InstallDir` | `%LOCALAPPDATA%\hermes\hermes-agent` | 코드 위치 재정의 |

설치 관리자는 불안정한 git 가져오기(fetch)를 자동으로 재시도하며, 다운로드된 `install.ps1` 페이로드에서 BOM을 제거합니다. 따라서 HTTP 전송 중 추가된 UTF-8 BOM이 `[scriptblock]::Create((irm ...))` 형태의 실행을 방해하지 않습니다.

### 의존성 부트스트랩 (`dep_ensure`)

첫 실행 시(그리고 누락된 도구가 감지되어 필요할 때마다), Hermes는 필요한 비-Python 의존성을 확인하고 지연 설치하는 작은 Python 부트스트래퍼(`hermes_cli/dep_ensure.py`)를 실행합니다. Windows에서 관련된 의존성은 다음과 같습니다:

| 의존성 | Hermes에 필요한 이유 |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **PortableGit** | 터미널 도구를 위한 `bash.exe` 및 세션 내 클론을 위한 `git` 제공. `dep_ensure`가 아닌 설치 시점에 프로비저닝됩니다. |
| **Node.js 22** | 브라우저 도구(`agent-browser`), TUI의 웹 브릿지, WhatsApp 브릿지에 필요합니다. |
| **ffmpeg** | TTS / 음성 메시지를 위한 오디오 형식 변환. |
| **ripgrep** | 빠른 파일 검색 — 사용할 수 없는 경우 `grep`으로 폴백합니다. |
| **npm packages** | `agent-browser`, Playwright Chromium 및 도구 세트별 Node 의존성은 브라우저 도구를 처음 사용할 때 한 번 설치됩니다. |

각 의존성은 `shutil.which(...)` 스타일의 검사를 갖습니다. 바이너리가 누락되어 있고 대화형 모드로 실행 중인 경우, `dep_ensure`는 설치를 제안합니다(실제 설치 로직은 `scripts\install.ps1 -ensure <dep>`로 위임). 비대화형 실행(게이트웨이, 크론, 헤드리스 데스크톱 실행)은 프롬프트를 건너뛰고 대신 명확한 `this feature needs <dep>` 오류를 표시합니다.

## 설치 관리자가 실제로 수행하는 작업 (What the installer actually does)

위에서 아래 순서대로 수행합니다:

1. **`uv` 부트스트랩** — Astral의 빠른 Python 패키지 관리자입니다. `%USERPROFILE%\.local\bin`에 설치됩니다.
2. **`uv`를 통한 Python 3.11 설치** — 기존에 설치된 Python이 필요하지 않습니다.
3. **Node.js 22 설치** (winget이 가능한 경우 사용, 없으면 휴대용 Node 압축 파일을 `%LOCALAPPDATA%\hermes\node`에 해제). 브라우저 도구와 WhatsApp 브릿지에 사용됩니다.
4. **휴대용 Git 설치** — `git`이 이미 PATH에 있다면 그것을 사용하고, 그렇지 않으면 공식 `git-for-windows` 릴리스에서 경량화된 독립형 **PortableGit**(~45 MB)을 다운로드하여 `%LOCALAPPDATA%\hermes\git`에 설치합니다. 관리자 권한, Windows 설치 관리자 레지스트리, 시스템 내 다른 요소와의 간섭이 전혀 없습니다.
5. **저장소 클론** — `%LOCALAPPDATA%\hermes\hermes-agent` 경로로 복제하고 내부에 virtualenv를 생성합니다.
6. **단계별 `uv pip install`** — 먼저 `.[all]`을 시도하고, 속도 제한이 걸린 GitHub에서 `git+https` 의존성 설치가 실패할 경우 점진적으로 더 작은 집합(`[messaging,dashboard,ext]` → `[messaging]` → `.`)으로 축소하며 재시도합니다. "단일 실패로 인해 최소 설치로 떨어지는" 상황을 방지합니다.
7. **`.env` 키 기반 메시징 SDK 자동 설치** — `TELEGRAM_BOT_TOKEN` / `DISCORD_BOT_TOKEN` / `SLACK_BOT_TOKEN` / `SLACK_APP_TOKEN` / `WHATSAPP_ENABLED`가 존재하는 경우, `python -m ensurepip --upgrade` 및 대상화된 `pip install` 호출을 실행하여 각 플랫폼의 SDK를 실제로 가져올 수 있도록(importable) 합니다.
8. **`HERMES_GIT_BASH_PATH` 설정** — 해결된 `bash.exe`의 경로를 지정하여 새로운 쉘에서 Hermes가 이를 결정적으로(deterministically) 찾을 수 있게 합니다.
9. **사용자 PATH에 `%LOCALAPPDATA%\hermes\bin` 추가** — 새 터미널을 열었을 때 `hermes` 명령어를 사용할 수 있게 합니다.
10. **`hermes setup` 실행** — 일반적인 첫 실행 마법사(모델, 제공자, 도구 세트)를 실행합니다. `-SkipSetup`으로 건너뛸 수 있습니다.

:::tip Windows에서 제공자(provider) 설정 과정 건너뛰기
Windows에서, 도구별 API 키 설정(Firecrawl, FAL, Browser Use, OpenAI TTS)은 유용한 에이전트를 구축하는 데 있어 가장 큰 마찰 요소입니다. [Nous Portal](/user-guide/features/tool-gateway) 구독은 하나의 OAuth 로그인으로 모델**과** 해당 모든 도구를 처리합니다. 설치 관리자가 완료된 후, `hermes setup --portal`을 실행하여 모든 것을 한 번에 연결하세요.
:::

## 기능 매트릭스 (Feature matrix)

대시보드의 내장 터미널 창을 제외한 모든 기능은 Windows에서 네이티브로 실행됩니다.

| 기능 | 네이티브 Windows | WSL2 |
| --------------------------------------------------------------------- | ------------------- | ---------------------- |
| CLI (`hermes chat`, `hermes setup`, `hermes gateway`, …) | ✓ | ✓ |
| 인터랙티브 TUI (`hermes --tui`) | ✓ | ✓ |
| 메시징 게이트웨이 (Telegram, Discord, Slack, WhatsApp, 15개 이상 플랫폼) | ✓ | ✓ |
| 크론(Cron) 스케줄러 | ✓ | ✓ |
| 브라우저 도구 (Node 기반 Chromium) | ✓ | ✓ |
| MCP 서버 (stdio 및 HTTP) | ✓ | ✓ |
| 로컬 Ollama / LM Studio / llama-server | ✓ | ✓ (WSL 네트워킹을 통해) |
| 웹 대시보드 (세션, 작업, 지표, 구성) | ✓ | ✓ |
| 대시보드 `/chat` 내장 터미널 창 | ✗ (POSIX PTY 필요) | ✓ |
| 로그인 시 자동 시작 | ✓ (schtasks) | ✓ (systemd) |

대시보드의 `/chat` 탭은 POSIX PTY(`ptyprocess`)를 통해 실제 터미널을 내장합니다. 네이티브 Windows에는 이와 동등한 기본 요소가 없습니다; Python의 `pywinpty` / Windows ConPTY가 작동하겠지만 이는 별도의 구현입니다 — 미래의 과제로 남겨둡니다. **나머지 대시보드 기능은 네이티브로 작동합니다** — 오직 해당 탭 하나만 "이를 위해 WSL2를 사용하세요" 배너를 표시합니다.

## Hermes가 Windows에서 쉘 명령어를 실행하는 방법 (How Hermes runs shell commands on Windows)

Hermes의 터미널 도구는 Claude Code와 동일한 전략인 **Git Bash**를 통해 명령어를 실행합니다. 이는 모든 도구를 재작성하지 않고도 POSIX와 Windows 간의 격차를 우회합니다.

`bash.exe` 확인 순서:

1. 설정된 경우 `HERMES_GIT_BASH_PATH` 환경 변수.
2. `%LOCALAPPDATA%\hermes\git\usr\bin\bash.exe` (설치 관리자가 관리하는 PortableGit).
3. `%LOCALAPPDATA%\hermes\git\bin\bash.exe` (이전 Git-for-Windows 레이아웃).
4. 시스템 Git-for-Windows 설치 경로 (`%ProgramFiles%\Git\bin\bash.exe` 등).
5. MSYS2, Cygwin 또는 PATH에 있는 모든 `bash.exe` (최후의 수단).

설치 관리자는 `HERMES_GIT_BASH_PATH`를 명시적으로 설정하므로 새 PowerShell 세션에서 다시 검색할 필요가 없습니다. Hermes가 특정 bash(예: 시스템 Git Bash 또는 심볼릭 링크를 통한 WSL 호스팅 bash)를 사용하도록 하려면 이 변수를 재정의하세요.

**함정:** MinGit의 레이아웃은 전체 Git-for-Windows 설치 관리자와 다릅니다 — bash가 `bin\bash.exe`가 아닌 `usr\bin\bash.exe`에 위치합니다. Hermes는 두 곳을 모두 확인합니다. MinGit 압축 파일을 수동으로 푸는 경우 반드시 **non-busybox** 변형(`MinGit-*-busybox*.zip`이 아닌 `MinGit-*-64-bit.zip`)을 선택하세요 — busybox 빌드는 `bash` 대신 `ash`를 포함하며 대부분의 coreutils가 누락되어 있습니다.

## Windows의 UTF-8 콘솔 (UTF-8 console on Windows)

Windows에서 Python의 기본 stdio는 콘솔의 활성 코드 페이지(일반적으로 cp1252 또는 cp437)를 사용합니다. Hermes의 배너, 슬래시 명령어 목록, 도구 피드, Rich 패널, 스킬 설명 등에는 모두 유니코드가 포함되어 있습니다. 별도의 조치 없이는 이 모든 것들이 `UnicodeEncodeError: 'charmap' codec can't encode character…` 오류와 함께 충돌합니다.

이에 대한 수정 사항은 모든 진입점(`cli.py::main`, `hermes_cli/main.py::main`, `gateway/run.py::main`)의 초기에 호출되는 `hermes_cli/stdio.py::configure_windows_stdio()`에 포함되어 있습니다. 이는 다음과 같이 작동합니다:

1. `kernel32.SetConsoleCP` / `SetConsoleOutputCP`를 통해 콘솔 코드 페이지를 CP_UTF8(65001)로 전환합니다.
2. `errors='replace'` 옵션과 함께 `sys.stdout` / `sys.stderr` / `sys.stdin`을 UTF-8로 재구성합니다.
3. `PYTHONIOENCODING=utf-8` 및 `PYTHONUTF8=1`을 설정하여(`setdefault`를 사용하므로 사용자의 명시적 설정이 우선함) 하위 Python 프로세스가 UTF-8을 상속하도록 합니다.
4. `EDITOR`나 `VISUAL`이 설정되지 않은 경우 `EDITOR=notepad`로 설정합니다 (아래 편집기 섹션 참조).

이 동작은 멱등성(idempotent)을 가지며, Windows가 아닌 환경에서는 아무 작업도 수행하지 않습니다.

**비활성화:** 환경 변수에 `HERMES_DISABLE_WINDOWS_UTF8=1`을 설정하면 레거시 cp1252 stdio 경로로 폴백합니다. 인코딩 버그의 원인을 찾기 위해 이분 탐색(bisecting)할 때 유용하지만, 정상적인 운영 환경에서는 올바른 설정이 아닐 가능성이 높습니다.

## 편집기 (`Ctrl-X Ctrl-E`, `/edit`)

#21561 패치 이전에는, Windows에서 `Ctrl-X Ctrl-E`를 누르거나 `/edit`를 입력해도 아무 일도 일어나지 않았습니다. prompt_toolkit은 하드코딩된 POSIX 절대 경로 폴백 목록(`/usr/bin/nano`, `/usr/bin/pico`, `/usr/bin/vi` 등)을 가지고 있는데, 이는 완전한 Git for Windows가 설치되어 있더라도 Windows에서는 절대 해석될 수 없는 경로들입니다.

Hermes의 Windows stdio shim은 이제 `EDITOR=notepad`를 기본값으로 설정합니다. 메모장(Notepad)은 모든 Windows 설치에 포함되어 있으며 블로킹(blocking) 편집기로 작동합니다 — 즉 `subprocess.call(["notepad", file])`은 창이 닫힐 때까지 대기합니다.

**사용자 재정의가 우선합니다** (기본값을 설정하기 전에 먼저 확인합니다):

| 편집기 | PowerShell 명령어 |
| --------- | ---------------------------------------------------------------------------------- |
| VS Code | `$env:EDITOR = "code --wait"` |
| Notepad++ | `$env:EDITOR = "'C:\Program Files\Notepad++\notepad++.exe' -multiInst -nosession"` |
| Neovim | `$env:EDITOR = "nvim"` |
| Helix | `$env:EDITOR = "hx"` |

VS Code의 경우 `--wait` 플래그가 매우 중요합니다 — 이 플래그가 없으면 편집기가 즉시 제어권을 반환하여 Hermes가 빈 버퍼를 받게 됩니다.

PowerShell 프로필에 영구적으로 설정하세요:

```powershell
# $PROFILE 내부에
$env:EDITOR = "code --wait"
```

또는 시스템 설정에서 사용자 환경 변수로 설정하여 모든 새 쉘에 적용되도록 할 수 있습니다.

## CLI에서 새 줄 입력을 위한 `Ctrl+Enter` (`Ctrl+Enter` for newline in the CLI)

Windows Terminal은 `Ctrl+Enter`를 전용 키 시퀀스로 전달합니다. Hermes는 이를 "새 줄 삽입"으로 바인딩하여, `Esc` 후 `Enter`로 돌아갈 필요 없이 CLI에서 여러 줄의 프롬프트를 작성할 수 있게 합니다. 이 기능은 Windows Terminal, VS Code 통합 터미널 및 VT 이스케이프 시퀀스를 지원하는 모든 최신 Windows 콘솔 호스트에서 작동합니다.

레거시 `cmd.exe` 콘솔에서는 `Ctrl+Enter`가 단순 `Enter`로 처리됩니다 — 대신 `Esc Enter`를 사용하거나 Windows Terminal로 업그레이드하세요 (무료이며 Windows 11에는 기본 설치되어 있습니다).

## Windows 로그인 시 게이트웨이 실행 (Running the gateway at Windows login)

Windows에서 `hermes gateway install`은 관리자 권한 없이 **예약된 작업(Scheduled Tasks)** 과 시작프로그램(Startup) 폴더 폴백을 사용합니다.

### 설치 (Install)

```powershell
hermes gateway install
```

내부적으로 일어나는 일:

1. `schtasks /Create /SC ONLOGON /RL LIMITED /TN HermesGateway` — 관리자 권한 없이(표준 권한으로) 로그인 시 실행되는 작업을 등록합니다. UAC 프롬프트가 표시되지 않습니다.
2. 그룹 정책에 의해 schtasks가 차단된 경우, `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`에 `start /min cmd.exe /d /c <wrapper>` 단축 아이콘을 생성하는 방식으로 폴백합니다. 효과는 동일하지만 약간 투박한 방식입니다.
3. 게이트웨이를 **`pythonw.exe`를 통해 분리된(detached) 상태로 생성**합니다 — `python.exe`가 아닙니다. `pythonw.exe`는 콘솔이 연결되어 있지 않아 동위 프로세스의 `CTRL_C_EVENT` 브로드캐스트로부터 안전합니다 (과거에 같은 프로세스 그룹 내에서 Ctrl+C를 누르면 게이트웨이가 죽던 실제 문제를 해결함).

프로세스 생성 시 사용되는 플래그: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW | CREATE_BREAKAWAY_FROM_JOB`.

### 관리 (Manage)

```powershell
hermes gateway status      # 병합된 뷰: schtasks + 시작프로그램 폴더 + 실행 중인 PID
hermes gateway start       # 즉시 예약된 작업 시작
hermes gateway stop        # 정상적인 SIGTERM과 동일 (psutil을 통한 TerminateProcess)
hermes gateway restart
hermes gateway uninstall   # schtasks 항목, 시작프로그램 단축 아이콘, pid 파일 제거
```

`hermes gateway status`는 멱등성(idempotent)을 가집니다 — 연속해서 천 번을 호출해도 실수로 게이트웨이를 죽이지 않습니다. (PR #21561 이전에는 C 레벨에서 `os.kill(pid, 0)`이 `CTRL_C_EVENT`와 충돌하여 조용히 게이트웨이를 죽이곤 했습니다 — 자세한 내용은 아래의 "프로세스 관리 내부 작동 방식" 참조).

### 왜 Windows 서비스가 아닌가요? (Why not a Windows Service?)

서비스(Service)를 설치하려면 관리자 권한이 필요하며, 게이트웨이의 수명 주기가 사용자 로그인이 아닌 머신 부팅에 묶이게 됩니다. 일반적인 Hermes 사용자는 '로그인 → 게이트웨이 사용 가능', '로그아웃 → 게이트웨이 종료'의 흐름을 원합니다. 예약된 작업(Scheduled Tasks)은 권한 상승 없이 이 역할을 정확히 수행합니다. 정말로 서비스를 원한다면 수동으로 `nssm` 또는 `sc create`를 사용하세요 — 하지만 아마도 필요하지 않을 것입니다.

## 데이터 레이아웃 (Data layout)

| 경로 | 내용 |
| ------------------------------------- | ------------------------------------------------------------------- |
| `%LOCALAPPDATA%\hermes\hermes-agent\` | Git 체크아웃 + venv. `Remove-Item -Recurse`로 삭제하고 재설치해도 안전합니다. |
| `%LOCALAPPDATA%\hermes\git\` | PortableGit (설치 관리자가 프로비저닝한 경우에만 존재). |
| `%LOCALAPPDATA%\hermes\node\` | 휴대용 Node.js (설치 관리자가 프로비저닝한 경우에만 존재). |
| `%LOCALAPPDATA%\hermes\bin\` | 사용자 PATH에 추가된 `hermes.cmd` shim. |
| `%USERPROFILE%\.hermes\` | 사용자의 구성, 인증, 스킬, 세션, 로그. **재설치 후에도 유지됩니다.** |

이러한 분리는 의도적입니다: `%LOCALAPPDATA%\hermes`는 일회성 인프라입니다 (날려버려도 설치 명령어 한 줄로 복구할 수 있습니다). 반면 `%USERPROFILE%\.hermes`는 사용자의 데이터(구성, 메모리, 스킬, 세션 기록)이며, Linux 설치 환경과 완벽히 동일한 구조를 가집니다. 머신 간에 이 디렉토리를 복사하면 Hermes 환경이 그대로 이동합니다.

**`HERMES_HOME` 재정의:** 다른 데이터 디렉토리를 가리키도록 환경 변수를 설정할 수 있습니다. Linux 환경과 동일하게 동작합니다.

## 브라우저 도구 (Browser tool)

브라우저 도구는 `agent-browser` (Node 헬퍼)를 사용하여 Chromium을 제어합니다. Windows에서는:

- 설치 관리자가 npm을 통해 PATH에 `agent-browser`를 배치합니다.
- `shutil.which("agent-browser", path=...)`가 `.cmd` shim을 자동으로 인식합니다 — `CreateProcessW`는 확장자가 없는 shebang을 실행할 수 없으므로, Hermes는 항상 `.CMD` 래퍼로 경로를 확인합니다. shebang 스크립트를 수동으로 호출하지 말고 항상 `.cmd`를 통해 실행하세요.
- Playwright Chromium은 첫 실행 시 자동 설치됩니다 (`npx playwright install chromium`). 설치 실패 시 `hermes doctor`가 해결 힌트와 함께 이를 안내합니다.

## Windows에서 Hermes 실행 — 실용적 참고 사항

### 설치 후 PATH (PATH after install)

설치 관리자는 `[Environment]::SetEnvironmentVariable`을 통해 **사용자 PATH**에 `%LOCALAPPDATA%\hermes\bin`을 추가합니다. 하지만 기존 터미널은 이를 즉시 인식하지 못하므로, 설치 후에는 새로운 PowerShell 창(또는 Windows Terminal 탭)을 열어야 합니다. 창을 닫고 다시 열기만 하세요. 정확히 이해하고 있지 않다면 수동으로 `$env:PATH += …`를 실행하지 마세요.

확인하기:

```powershell
Get-Command hermes        # C:\Users\<you>\AppData\Local\hermes\bin\hermes.cmd 가 출력되어야 합니다.
hermes --version
```

### 환경 변수 (Environment variables)

Hermes는 `$env:X` (프로세스 범위)와 사용자 환경 변수 (시스템 속성 → 환경 변수에서 설정, 영구적)를 모두 존중합니다. `%USERPROFILE%\.hermes\.env`에 API 키를 설정하는 것이 일반적인 방법이며, 이는 Linux와 같습니다:

```
OPENROUTER_API_KEY=sk-or-...
TELEGRAM_BOT_TOKEN=...
```

모든 Windows 프로세스가 볼 수 있게 하려는 명확한 의도가 아니라면, 사용자 환경 변수에 비밀 키를 넣지 마세요 (대부분 원치 않는 결과일 것입니다).

### Windows 전용 환경 변수

이는 네이티브 Windows 설치 환경에만 영향을 미칩니다:

| 변수 | 효과 |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `HERMES_GIT_BASH_PATH` | bash.exe 검색 경로를 재정의합니다. Git-for-Windows, 심볼릭 링크를 통한 WSL bash, MSYS2, Cygwin 등 어떤 bash든 지정할 수 있습니다. 설치 관리자가 이를 자동으로 설정합니다. |
| `HERMES_DISABLE_WINDOWS_UTF8` | `1`로 설정하면 UTF-8 stdio shim을 비활성화하고 로캘 코드 페이지로 폴백합니다. 인코딩 버그 원인을 이분 탐색(bisecting)할 때 유용합니다. |
| `EDITOR` / `VISUAL` | `/edit` 및 `Ctrl-X Ctrl-E`를 위한 편집기입니다. 둘 다 설정되지 않은 경우 Hermes는 기본적으로 `notepad`를 사용합니다. |

## 제거 (Uninstall)

PowerShell에서 실행하세요:

```powershell
hermes uninstall
```

이것이 깔끔한 방법입니다 — schtasks 항목, 시작프로그램 폴더 단축 아이콘, `hermes.cmd` shim을 제거하고 `%LOCALAPPDATA%\hermes\hermes-agent\`를 삭제하며, 사용자 PATH를 정리합니다. 재설치하는 경우를 대비해 `%USERPROFILE%\.hermes\` (구성, 인증, 스킬, 세션, 로그)는 그대로 남겨둡니다.

모든 것을 완전히 삭제하려면:

```powershell
hermes uninstall
Remove-Item -Recurse -Force "$env:USERPROFILE\.hermes"
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\hermes"
```

또한 `hermes uninstall` CLI 하위 명령어는 예약된 작업(schtasks) 항목이 다른 작업 이름으로 등록되었던 과거 버전의 경우도 처리합니다 — 하드코딩된 작업 이름 대신 설치 경로로 검색합니다.

## 프로세스 관리 내부 작동 방식 (Process management internals)

이 부분은 배경 지식입니다 — "자동으로 종료되는" 이상 현상을 디버깅하는 상황이 아니라면 건너뛰어도 됩니다.

Linux 및 macOS에서 `os.kill(pid, 0)`이라는 POSIX 관용구는 작동하지 않는 권한 확인(no-op permission check)입니다: "이 PID가 살아있고 신호를 보낼 수 있는가?"를 확인합니다. 하지만 Windows에서 Python의 `os.kill`은 `sig=0`을 `CTRL_C_EVENT`와 매핑하며 — 0이라는 정수값에서 충돌 발생 — 대상 PID가 포함된 **전체 콘솔 프로세스 그룹**에 Ctrl+C를 브로드캐스트하는 `GenerateConsoleCtrlEvent(0, pid)`를 통해 전달됩니다. 이 문제는 2012년부터 열려 있는 [bpo-14484](https://bugs.python.org/issue14484) 버그입니다. 이를 수정하면 현재 동작에 의존하는 스크립트들이 손상될 수 있으므로 고쳐지지 않을 것입니다.

결과적으로, Windows에서 `os.kill(pid, 0)`을 통해 "이 PID가 살아있는지 확인"하려던 모든 코드 경로가 대상을 조용히 죽이고 있었습니다. Hermes는 (11개 파일의 14개 부분을) 이러한 검사가 포함된 모든 부분을 `psutil.pid_exists()`(내부적으로 Windows에서 `OpenProcess + GetExitCodeProcess`를 사용하며 신호를 보내지 않음)를 사용하는 `gateway.status._pid_exists()`로 마이그레이션했습니다. 플러그인이나 패치를 작성 중이라면 직접 `psutil.pid_exists()` 또는 `gateway.status._pid_exists()`를 사용하고 — 절대 `os.kill(pid, 0)`을 사용하지 마세요.

`scripts/check-windows-footguns.py`는 CI에서 이 규칙을 강제합니다: 새로운 `os.kill(pid, 0)` 호출은 해당 줄에 `# windows-footgun: ok — <이유>` 마커가 없는 한 `Windows footguns (blocking)` 검사에서 실패합니다.

## 일반적인 문제 해결 (Common pitfalls)

**설치 직후 `hermes: command not found` 오류가 발생하는 경우.**
새 PowerShell 창을 여세요. 설치 관리자가 사용자 PATH에 `%LOCALAPPDATA%\hermes\bin`을 추가했지만, 기존 쉘은 이를 인식하기 위해 재시작해야 합니다.

**도구 실행 시 `WinError 193: %1 is not a valid Win32 application` 오류.**
`.cmd` shim을 우회하는 shebang-script 호출을 실행하셨군요. Hermes는 `shutil.which(cmd, path=local_bin)`를 통해 명령을 확인하므로 PATHEXT가 `.CMD`를 선택합니다 — 도구를 하드코딩된 경로를 통해 실행하는 중이라면, `.cmd` 변형(예: `npx`가 아닌 `npx.cmd`)으로 변경하세요.

**`[scriptblock]::Create(...)`가 `The assignment expression is not valid` 오류와 함께 실패하는 경우.**
다운로드한 `install.ps1`에 UTF-8 BOM이 포함되어 있습니다. `irm | iex` 형태는 BOM을 자동으로 제거하지만, `[scriptblock]::Create((irm ...))` 형태는 그렇지 않습니다. 단순한 `irm | iex` 형태로 다시 실행하거나, 스크립트를 직접 다운로드하여 `[IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))`를 통해 BOM 없이 저장하세요.

**재시작 후 게이트웨이가 계속 실행되지 않는 경우.**
`hermes gateway status`를 확인하세요 — schtasks 항목, 시작프로그램 폴더 단축 아이콘(사용된 경우) 및 활성 PID가 병합되어 표시됩니다. schtasks는 등록되었지만 실행 중이 아니라면 그룹 정책이 `ONLOGON` 트리거를 차단하고 있을 수 있습니다. `schtasks /Query /TN HermesGateway /V /FO LIST`를 실행하여 작업의 실패 이유를 확인하거나, `HERMES_GATEWAY_FORCE_STARTUP=1` 설정과 함께 재설치하여 시작프로그램 폴더 경로로 폴백하세요.

**`$env:EDITOR` 설정 후에도 `/edit`가 아무 작업도 수행하지 않는 경우.**
현재 프로세스에만 설정된 것입니다. 쉘을 닫았다 다시 열거나, 시스템 속성 → 환경 변수에서 사용자 범위(User scope)에 설정하세요. 새 PowerShell 창에서 `echo $env:EDITOR`로 확인하세요.

**브라우저 도구는 실행되지만 도구 시간이 초과되는 경우.**
Chromium은 첫 실행 시 자동으로 설치됩니다. 설치에 실패한 경우(GitHub 속도 제한, Playwright CDN 오류 등), `hermes doctor`를 실행하세요 — 누락된 Chromium을 알리고 이를 해결하기 위한 정확한 `npx playwright install chromium` 명령어를 출력합니다.

**이상한 Node 버전 오류와 함께 `agent-browser`가 실패하는 경우.**
설치 관리자는 `%LOCALAPPDATA%\hermes\node`에 Node 22를 프로비저닝하지만, 시스템의 PATH 맨 앞에 오래된 시스템 Node 18이 있을 수 있습니다. Hermes의 Node 디렉토리를 PATH의 앞쪽으로 이동시키거나, 다른 곳에서 Node를 사용하지 않는다면 시스템에 설치된 Node를 삭제하세요.

**CLI에서 중국어 / 일본어 / 아랍어 문자가 `?`로 표시되는 경우.**
UTF-8 stdio shim이 활성화되지 않았습니다. `HERMES_DISABLE_WINDOWS_UTF8` 환경 변수가 설정되어 있지 않은지(`Get-ChildItem env:HERMES_DISABLE_WINDOWS_UTF8`) 확인하세요. 변수가 비어 있는데도 계속 `?`가 보인다면, 콘솔 호스트(아주 오래된 `cmd.exe`)가 UTF-8을 전혀 지원하지 않을 수 있습니다 — Windows Terminal로 전환하세요.

**게이트웨이가 Telegram으로 사진을 보낼 수 없는 경우 — "`BadRequest: payload contains invalid characters`".**
이 오류는 Windows와 직접적인 관련은 없지만, 종종 Windows 환경에서 처음 발견됩니다. 대개 JSON 본문 내부의 파일 경로에 이스케이프되지 않은 백슬래시(\)가 포함되어 있다는 의미입니다. Telegram은 가공되지 않은 Windows 경로가 아닌 Hermes가 정규화한 경로를 받아야 합니다. 맞춤형 플러그인 내부에서 이 오류를 발견했다면 사용자 입력으로 받은 `str(Path(...))` 대신 Hermes가 제공한 경로를 올바르게 전달하고 있는지 확인하세요.

**`git pull` 이후 발생하는 "다른 컴퓨터에서는 잘 작동하는데 인코딩이 이상해요" 문제.**
Windows의 UTF-8을 지원하지 않는 편집기(구형 Windows 버전의 메모장, 일부 중국어 IME)를 사용하여 Hermes 구성 파일이나 스킬을 수정했다면 파일이 BOM과 함께 저장되었을 수 있습니다. Hermes는 대부분의 구성 읽기 작업에서 `utf-8-sig`를 허용하지만, 접혀 있는 YAML 스칼라(`description: >`) 내부의 BOM은 소리 없이 YAML 파싱을 망가뜨립니다. 해당 파일을 BOM 없는 순수 UTF-8로 다시 저장하세요.

## 다음으로 이동 (Where to go next)

- **[설치(Installation)](../getting-started/installation.md)** — Linux/macOS/WSL2/Termux를 포함한 전체 설치 안내.
- **[Windows (WSL2) 가이드](./windows-wsl-quickstart.md)** — POSIX 시맨틱 또는 대시보드 터미널 창을 원할 경우.
- **[CLI 레퍼런스](../reference/cli-commands.md)** — 모든 `hermes` 하위 명령어.
- **[FAQ](../reference/faq.md)** — Windows 환경이 아닌 일반적인 질문들.
- **[메시징 게이트웨이](./messaging/index.md)** — Windows에서 Telegram/Discord/Slack 실행.

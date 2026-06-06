---
title: "Windows (WSL2) 가이드"
description: "WSL2를 통해 Windows에서 Hermes Agent 실행하기 — 설정, Windows와 Linux 간의 파일 시스템 액세스, 네트워킹 및 일반적인 문제 해결"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Windows (WSL2) 가이드

Hermes Agent는 이제 기본 Windows와 WSL2를 **모두** 지원합니다. 이 페이지에서는 WSL2 경로를 다루며, 기본 PowerShell 설치에 대해서는 전용 **[Windows (Native) 가이드](./windows-native.md)**를 참조하세요.

**네이티브(Native) 대신 WSL2를 선택해야 하는 경우:**
- 대시보드의 내장 터미널(`/chat` 탭)을 사용하고 싶은 경우 — 이 창은 POSIX PTY를 요구하며 WSL2에서만 작동합니다.
- POSIX 위주의 개발 작업을 하고 있으며 Hermes 세션이 개발 도구와 동일한 파일 시스템 / 경로를 공유하기를 원하는 경우.
- 이미 WSL2 환경이 구축되어 있으며 두 번째 설치를 유지하고 싶지 않은 경우.

**네이티브 환경이 충분하거나 더 나은 경우:**
- 대화형 채팅, 게이트웨이(Telegram/Discord 등), 크론 스케줄러, 브라우저 도구, MCP 서버 및 대부분의 Hermes 기능은 Windows에서 네이티브로 실행됩니다.
- 파일을 참조하거나 URL을 열 때마다 WSL과 Windows 경계를 넘나드는 것에 대해 신경 쓰고 싶지 않은 경우.

WSL2에는 사실상 두 대의 컴퓨터가 실행 중인 것과 같습니다. Windows 호스트와 WSL이 관리하는 Linux VM입니다. 대부분의 혼란은 현재 어느 환경에 있는지 확실하지 않을 때 발생합니다.

이 가이드에서는 Hermes에 구체적으로 영향을 미치는 이러한 분할 부분(WSL2 설치, Windows와 Linux 간에 파일 이동, 양방향 네트워킹, 사용자들이 실제로 겪는 문제점)을 다룹니다.

:::info 简体中文
최소 설치 경로에 대한 중국어 설명은 이 페이지에 유지됩니다 — 오른쪽 상단의 **언어(language)** 메뉴를 통해 **简体中文**을 선택하세요.
:::

## 왜 WSL2인가 (네이티브 Windows와 비교)

네이티브 Windows 설치는 Windows에서 직접 실행됩니다. Windows 터미널(PowerShell, Windows Terminal 등), Windows 파일 시스템 경로(`C:\Users\…`), Windows 프로세스를 사용합니다. Hermes는 쉘 명령을 실행하기 위해 Git Bash를 사용하며, 이것이 Claude Code 및 다른 에이전트들이 오늘날 Windows를 다루는 방식입니다. 이는 완전한 재작성 없이 POSIX 대 Windows 격차를 우회합니다.

WSL2는 가벼운 VM에서 실제 Linux 커널을 실행하므로, 그 안의 Hermes는 Ubuntu에서 실행하는 것과 본질적으로 동일합니다. 이는 `fork`, `/tmp`, UNIX 소켓, 신호 체계(signal semantics), PTY 기반 터미널, `bash`/`zsh`와 같은 쉘, Linux에서처럼 동작하는 `rg`, `git`, `ffmpeg`와 같은 실제 POSIX 환경을 원할 때 유용합니다.

WSL2의 실질적인 결과:

- Hermes CLI, 게이트웨이, 세션, 메모리, 스킬 및 도구 런타임은 모두 Linux VM 내부에 존재합니다.
- Windows 프로그램(브라우저, 네이티브 앱, 로그인된 프로필이 있는 Chrome)은 외부에 존재합니다.
- 두 시스템이 통신할 때마다 — 파일을 공유하거나, URL을 열거나, Chrome을 제어하거나, 로컬 모델 서버에 연결하거나, 전화기에 Hermes 게이트웨이를 노출할 때마다 — 경계를 넘게 됩니다. 이 경계들이 바로 이 가이드의 주제입니다.

## WSL2 설치

**관리자 권한 PowerShell** 또는 Windows 터미널에서 다음을 실행합니다:

```powershell
wsl --install
```

새로운 Windows 10 22H2+ 또는 Windows 11 시스템에서 이는 WSL2 커널, Virtual Machine Platform 기능 및 기본 Ubuntu 배포판을 설치합니다. 메시지가 나타나면 재부팅하세요. 재부팅 후 Ubuntu가 열리고 Linux 사용자 이름과 암호를 묻습니다. 이것은 Windows 계정과 관련 없는 **새로운 Linux 사용자**입니다.

실제로 WSL2(기존 WSL1 아님)에 있는지 확인하세요:

```powershell
wsl --list --verbose
```

`VERSION 2`가 표시되어야 합니다. 배포판이 `VERSION 1`로 표시되면 변환하세요:

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

Hermes는 WSL1에서 안정적으로 작동하지 않습니다. WSL1은 즉석에서 Linux 시스템 호출을 변환하며 일부 동작(procfs, 신호, 네트워크)이 실제 Linux와 다릅니다.

### 배포판 선택

우리는 Ubuntu(LTS)를 기준으로 테스트합니다. Debian도 작동합니다. Arch와 NixOS도 원하는 사용자에게는 작동하지만, 한 줄짜리 설치 프로그램은 Debian 기반 `apt` 시스템을 가정합니다. 해당 경로는 [Nix 설정 가이드](/getting-started/nix-setup)를 참조하세요.

### systemd 활성화 (권장)

hermes 게이트웨이(및 계속 실행하려는 기타 모든 항목)는 systemd를 사용하면 관리하기가 더 쉽습니다. 최신 WSL에서는 배포판 내부에 들어가 한 번 활성화하세요:

```bash
sudo tee /etc/wsl.conf >/dev/null <<'EOF'
[boot]
systemd=true

[interop]
enabled=true
appendWindowsPath=true

[automount]
options = "metadata,umask=22,fmask=11"
EOF
```

그런 다음 PowerShell에서 다음을 실행합니다:

```powershell
wsl --shutdown
```

WSL 터미널을 다시 엽니다. `ps -p 1 -o comm=` 명령은 `systemd`를 인쇄해야 합니다.

위의 `metadata` 마운트 옵션은 중요합니다. 이 옵션이 없으면 `/mnt/c/...`의 파일이 실제 Linux 권한 비트를 저장할 수 없으며, 이는 Windows 경로 아래의 스크립트에서 `chmod +x`와 같은 명령을 손상시킵니다.

### WSL 내부에 Hermes 설치

WSL2 쉘이 열리면 다음을 실행합니다:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
hermes
```

설치 프로그램은 WSL2를 일반 Linux로 취급합니다 — WSL 관련 특별한 조치가 필요하지 않습니다. 전체 레이아웃은 [설치](/getting-started/installation)를 참조하세요.

## 파일 시스템: Windows ↔ WSL2 경계 넘기

이 부분은 많은 사람들이 가장 헷갈려하는 부분입니다. **두 개의 파일 시스템**이 존재하며, 파일을 어디에 두는지가 성능, 정확성 및 도구의 가시성에 영향을 미칩니다.

### 두 가지 방향

| 방향 | 내부 경로 | 사용하는 경로 |
|---|---|---|
| WSL에서 바라본 Windows 디스크 | `C:\Users\you\Documents` | `/mnt/c/Users/you/Documents` |
| Windows에서 바라본 WSL 디스크 | `/home/you/code` | `\\wsl$\Ubuntu\home\you\code` (최신 빌드에서는 `\\wsl.localhost\Ubuntu\...`) |

둘 다 실제이며 모두 작동하지만 **동일한 파일 시스템이 아닙니다** — 내부적으로 9P 네트워크 프로토콜로 연결되어 있습니다. 이로 인해 실질적인 성능 및 의미론적 결과가 발생합니다.

### Hermes와 프로젝트를 보관할 위치

**경험 법칙: Linux와 관련된 모든 것을 Linux 파일 시스템 내부에 보관하세요.**

- Hermes 설치(`~/.hermes/`) — Linux 측. 설치 프로그램이 이미 이렇게 합니다.
- WSL에서 작업하는 git 저장소 — Linux 측 (`~/code/...`, `~/projects/...`).
- 모델, 데이터 세트, venvs — Linux 측.

이 규칙을 따를 때 얻는 이점:

- **빠른 I/O.** `/mnt/c/...`의 작업은 9P를 통과하며 네이티브 ext4보다 10–100배 느립니다. `~/code`에서는 순식간에 끝나는 1만 개 파일 저장소의 `git status`가 `/mnt/c`에서는 15초 이상 걸릴 수 있습니다.
- **정확한 권한.** Linux 권한 비트는 `/mnt/c`에서 최선을 다해 에뮬레이션될 뿐입니다. `ssh`가 "나쁜 권한(bad permissions)"으로 키를 거부하거나 `chmod +x`가 조용히 실패하는 일은 흔합니다.
- **안정적인 파일 감시자.** 9P를 통한 inotify는 불안정합니다. 파일 감시자(개발 서버, 테스트 실행기)가 `/mnt/c`의 변경 사항을 일상적으로 놓칩니다.
- **대소문자 구분 문제 방지.** Windows 경로는 기본적으로 대소문자를 구분하지 않지만 Linux는 구분합니다. `Readme.md`와 `README.md`가 모두 있는 프로젝트는 어느 쪽에 있느냐에 따라 다르게 작동합니다.

Windows 앱에서 열고 싶거나, Windows Chrome의 DevTools MCP에서 현재 디렉토리가 Windows가 도달할 수 있는 경로여야 하는 등 Windows 측에 파일이 **반드시** 있어야 할 때만 `/mnt/c`에 보관하세요.

### 앞뒤로 파일 가져오기

**Windows → WSL 안으로:** 가장 쉬운 방법은 탐색기를 열고 주소 표시줄에 `\\wsl.localhost\Ubuntu`를 입력하는 것입니다. 그런 다음 `\home\<you>\...`로 끌어서 놓을 수 있습니다. 또는 PowerShell에서:

```powershell
wsl cp /mnt/c/Users/you/Downloads/file.pdf ~/incoming/
```

**WSL → Windows 안으로:** `/mnt/c/Users/<you>/...`에 복사하면 즉시 Windows 탐색기에 나타납니다:

```bash
cp ~/reports/output.pdf /mnt/c/Users/you/Desktop/
```

**Windows 앱에서 WSL 파일 열기** (GUI 편집기, 브라우저 등): `explorer.exe` 또는 `wslview`를 사용하세요:

```bash
sudo apt install wslu     # 한 번 설치 — wslview, wslpath, wslopen 등을 제공
wslview ~/reports/output.pdf    # Windows 기본 핸들러로 엽니다
explorer.exe .                  # 현재 WSL 디렉토리를 Windows 탐색기에서 엽니다
```

**두 세계 간의 경로 변환:**

```bash
wslpath -w ~/code/project        # → \\wsl.localhost\Ubuntu\home\you\code\project
wslpath -u 'C:\Users\you'        # → /mnt/c/Users/you
```

### 줄 바꿈(Line endings), BOM 및 git

Windows 편집기로 Windows 측에서 파일을 편집하면 `CRLF` 줄 바꿈이 생길 수 있습니다. Linux 측의 `bash` 또는 Python이 이를 읽을 때 쉘 스크립트가 `bad interpreter: /bin/bash^M`으로 중단되고 Python이 BOM이 있는 `.env` 파일에서 실패할 수 있습니다.

수정 방법은 WSL(Windows 아님) 내부에 올바른 git 구성을 설정하는 것입니다:

```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

이미 CRLF가 있는 파일의 경우:

```bash
sudo apt install dos2unix
dos2unix path/to/script.sh
```

### "WSL 내부에 클론(Clone)할 것인가 아니면 `/mnt/c`에 클론할 것인가?"

WSL 내부에 클론하세요. 특별한 이유가 없는 한 항상 그렇게 하세요. 전형적인 Hermes 워크플로우(`hermes chat`, 저장소를 `rg`/`ripgrep`하는 도구 호출, 파일 감시자, 백그라운드 게이트웨이)는 `/mnt/c/Users/you/myrepo`보다 `~/code/myrepo`에서 극적으로 빠르고 안정적입니다.

한 가지 예외: **Windows 바이너리를 실행하는 MCP 브릿지.** `cmd.exe`를 통해 `chrome-devtools-mcp`를 사용하는 경우 ([MCP 가이드: WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome) 참조), Hermes의 현재 작업 디렉토리가 `~`이면 Windows가 `UNC` 경고와 함께 불평할 수 있습니다. 이 경우 Windows 프로세스에 드라이브 문자가 있는 작업 디렉토리가 있도록 `/mnt/c/` 아래 어딘가에서 Hermes를 시작하세요.

## 네트워킹: WSL ↔ Windows

WSL2는 자체 네트워크 스택이 있는 가벼운 VM에서 실행됩니다. 즉, WSL 내부의 `localhost`는 Windows의 `localhost`와 **같지 않습니다** — 네트워크 관점에서 볼 때 두 개의 분리된 호스트입니다. 각 서비스에 대해 트래픽이 흐르는 방향을 결정하고 올바른 브릿지를 선택해야 합니다.

가장 흔하게 발생하는 두 가지 경우가 있습니다.

### 사례 1 — WSL의 Hermes가 Windows의 서비스와 통신

가장 일반적인 사례: Windows에서 **Ollama, LM Studio 또는 llama-server**를 실행 중이고 WSL 내부의 Hermes가 거기에 접근해야 합니다.

이에 대한 정식 사용법은 제공자 가이드에 있습니다: **[로컬 모델을 위한 WSL2 네트워킹 →](/integrations/providers#wsl2-networking-windows-users)**

요약:

- **Windows 11 22H2 이상:** 미러링된 네트워킹 모드를 켭니다 (`%USERPROFILE%\.wslconfig`에 `networkingMode=mirrored` 추가 후 `wsl --shutdown`). 그러면 양방향으로 `localhost`가 작동합니다.
- **Windows 10 또는 이전 빌드:** Windows 호스트 IP(WSL 가상 네트워크의 기본 게이트웨이)를 사용하고 Windows의 서버가 `127.0.0.1`뿐만 아니라 `0.0.0.0`에도 바인딩되어 있는지 확인합니다. 일반적으로 포트에 대한 Windows 방화벽 규칙도 필요합니다.

전체 표(Ollama / LM Studio / vLLM / SGLang 바인딩 주소, 방화벽 규칙 한 줄 명령어, 동적 IP 도우미, Hyper-V 방화벽 우회 방법)는 위의 링크를 따르세요 — 중복하지 마세요.

### 사례 2 — Windows(또는 LAN)의 무언가가 WSL의 Hermes와 통신

이것은 반대 방향이며 문서화가 덜 되어 있지만 다음과 같은 경우에 필요합니다:

- Windows 브라우저에서 Hermes **웹 대시보드** 사용.
- Windows 측 도구에서 **OpenAI 호환 API 서버**(`API_SERVER_ENABLED=true`일 때 `hermes gateway`에 의해 노출됨) 사용. [API 서버 기능 페이지](/user-guide/features/api-server) 참조.
- **메시징 게이트웨이**(Telegram, Discord 등) 테스트 — 플랫폼이 로컬 웹훅 URL에 핑을 보낼 때 일반적으로 단순한 포트 포워딩보다 `cloudflared`/`ngrok`을 사용합니다.

#### 하위 사례 2a: Windows 호스트 자체에서

**미러링된 모드가 활성화된 Windows 11 22H2 이상**에서는 아무것도 할 필요가 없습니다. `0.0.0.0:8080`(또는 `127.0.0.1:8080`)에 바인딩된 WSL의 프로세스는 Windows 브라우저에서 `http://localhost:8080`으로 접근할 수 있습니다. WSL은 바인딩을 호스트에 자동으로 게시합니다.

**NAT 모드**(Windows 10 / 이전 Windows 11)에서 WSL2의 기본 "localhost 포워딩"은 일반적으로 Linux 측 `127.0.0.1` 바인딩을 Windows `localhost`로 포워딩하므로 `--host 127.0.0.1`로 시작된 Hermes 서비스는 대개 Windows에서 `http://localhost:PORT`로 접근 가능합니다. 그렇지 않은 경우:

- WSL 내부에서 `0.0.0.0`에 명시적으로 바인딩합니다.
- `ip -4 addr show eth0 | grep inet`으로 WSL VM의 IP를 찾고 Windows에서 거기에 연결합니다.

#### 하위 사례 2b: LAN에 있는 다른 기기(전화기, 태블릿, 다른 PC)에서

이것은 정말 고통스러운 부분입니다. 트래픽은 **LAN 기기 → Windows 호스트 → WSL VM**으로 흐르며 두 홉을 모두 설정해야 합니다:

1. **WSL 내부의 모든 인터페이스에 바인딩.** `127.0.0.1`에서 수신 대기하는 프로세스는 VM 외부에서 절대 접근할 수 없습니다. `0.0.0.0`을 사용하세요.

2. **Windows → WSL VM 포트 포워딩.** 미러링된 모드에서는 이것이 자동입니다. NAT 모드에서는 관리자 PowerShell에서 포트별로 직접 수행해야 합니다:

   ```powershell
   # WSL VM의 현재 IP 가져오기 (NAT에서 WSL이 다시 시작될 때마다 변경됨)
   $wslIp = (wsl hostname -I).Trim().Split(' ')[0]

   # Windows 포트 8080을 WSL:8080으로 포워딩
   netsh interface portproxy add v4tov4 `
     listenaddress=0.0.0.0 listenport=8080 `
     connectaddress=$wslIp connectport=8080

   # Windows 방화벽 허용
   New-NetFirewallRule -DisplayName "Hermes WSL 8080" `
     -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```

   나중에 `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080`으로 제거할 수 있습니다.

3. **LAN 기기에서 `http://<windows-lan-ip>:8080`으로 가리키기.**

NAT 모드에서는 다시 시작할 때마다 WSL VM IP가 변경되므로 단발성 규칙은 다음 `wsl --shutdown` 때까지만 유지됩니다. 지속적인 설정을 원한다면 미러링된 모드를 사용하거나 Windows 로그인 시 실행되는 스크립트에 포트 프록시 단계를 넣으세요.

클라우드 메시징 제공업체(Telegram `setWebhook`, Slack 이벤트 등)의 웹훅의 경우, 포트 포워딩과 씨름하지 마세요 — `cloudflared` 터널을 사용하세요. [웹훅 가이드](/user-guide/messaging/webhooks)를 참조하세요.

## Windows에서 Hermes 서비스를 장기간 실행하기

Hermes [도구 게이트웨이(Tool Gateway)](/user-guide/features/tool-gateway)와 API 서버는 오래 지속되는 프로세스입니다. WSL2에는 프로세스를 유지할 수 있는 몇 가지 옵션이 있습니다.

### Hermes를 빠르게 열 수 있는 바탕 화면 바로 가기

대화형 Hermes 쉘을 위한 더블클릭 런처만 원한다면 Windows 쪽에 생성하여 자동으로 WSL로 진입하게 하세요:

1. Windows 바탕 화면을 마우스 오른쪽 버튼으로 클릭하고 **새로 만들기 -> 바로 가기**를 선택합니다.
2. 대상의 경우 배포판 이름을 사용합니다 (필요한 경우 `Ubuntu` 바꾸기):

   ```text
   wt.exe -w 0 -p "Ubuntu" wsl.exe -d Ubuntu --cd ~ -- bash -ic "hermes"
   ```

3. `Hermes`와 같이 명확한 이름을 지정합니다.

이것은 Windows 터미널을 열고, WSL 배포판을 시작하며, Linux 홈 디렉토리에 떨어뜨리고, Hermes를 실행합니다. `hermes`가 아직 PATH에 없으면 수동으로 WSL을 한 번 열고 `source ~/.bashrc`를 실행하거나, 프로젝트 체크아웃 내에서 명령을 `uv run hermes`로 바꿉니다.

선택적 다듬기:

- **사용자 지정 아이콘:** **속성 -> 아이콘 변경**을 열고 저장소의 Hermes 파비콘과 같은 `.ico` 파일을 가리킵니다.
- **고정된 런처:** 바로 가기가 작동하면 시작 메뉴 또는 작업 표시줄에 고정하여 다시 찾아볼 필요가 없게 합니다.

### WSL 내부에서 systemd 사용 (권장)

위의 설정 섹션에 따라 systemd를 활성화한 경우, `hermes gateway` 및 API 서버는 여느 Linux 머신에서와 같은 방식으로 작동합니다. 게이트웨이 설정 마법사를 사용하세요:

```bash
hermes gateway setup
```

그러면 WSL이 시작될 때 게이트웨이가 자동으로 켜지도록 systemd 사용자 유닛을 설치할 것을 제안합니다.

### Windows 로그인 시 WSL 자체가 시작되도록 만들기

WSL의 VM은 무언가가 사용할 때만 유지됩니다. 터미널 창을 열지 않고도 게이트웨이에 접근할 수 있게 유지하려면 작업 스케줄러를 통해 Windows 로그인 시 WSL 프로세스를 부팅합니다:

- **트리거:** 로그온할 때 (해당 사용자).
- **작업:** 프로그램 시작
  - 프로그램: `C:\Windows\System32\wsl.exe`
  - 인수: `-d Ubuntu --exec /bin/sh -c "sleep infinity"`

그러면 VM이 활성 상태로 유지되어 systemd로 관리되는 게이트웨이가 계속 실행됩니다. Windows 11에서는 최신 `wsl --install --no-launch` + 자동 시작 흐름도 작동합니다. `sleep infinity` 트릭은 이식 가능한 버전입니다.

## GPU 패스스루 (로컬 모델)

WSL2는 WSL 커널 5.10.43 이상부터 기본적으로 **NVIDIA** GPU를 지원합니다 — Windows에 표준 NVIDIA 드라이버를 설치하면(WSL 내부에 Linux NVIDIA 드라이버를 설치하지 **마세요**) WSL 내부의 `nvidia-smi`가 GPU를 봅니다. 거기에서 일반적인 방식대로 CUDA 툴킷, `torch`, `vllm`, `sglang` 및 `llama-server`가 실제 GPU에 기반하여 빌드됩니다.

WSL2 내부의 AMD ROCm 및 Intel Arc 지원은 여전히 진화 중이며 Hermes의 테스트 매트릭스를 벗어납니다. 현재 드라이버에서 작동할 수는 있지만 권장할 만한 방법이 없습니다.

이미 Windows 드라이버를 통해 GPU를 사용하는 **Windows-네이티브** 로컬 모델 서버(Windows용 Ollama, LM Studio)를 실행 중인 경우 WSL GPU 패스스루가 전혀 필요하지 않습니다 — 그냥 위의 사례 1을 따르고 WSL에서 네트워크를 통해 연결하세요.

## 일반적인 문제 해결

**Windows에 호스팅된 Ollama / LM Studio에 대한 "Connection refused"(연결 거부).**
[WSL2 네트워킹](/integrations/providers#wsl2-networking-windows-users)을 참조하세요. 열에 아홉은 서버가 `127.0.0.1`에 바인딩되어 `0.0.0.0`이 필요하거나(Ollama: `OLLAMA_HOST=0.0.0.0`), 방화벽 규칙이 누락된 것입니다.

**저장소의 `git status` / `hermes chat`에서 엄청난 지연.**
아마도 `/mnt/c/...` 아래에서 작업하고 있을 것입니다. 저장소를 `~/code/...` (Linux 측)으로 이동하세요. 속도가 훨씬 빠릅니다.

**스크립트의 `bad interpreter: /bin/bash^M`.**
Windows 편집기에서 생성된 CRLF 줄 바꿈. `dos2unix script.sh`를 실행하고 WSL git 구성에서 `core.autocrlf input`을 설정하세요.

**MCP를 통해 시작된 Windows 바이너리에서 "UNC paths are not supported" 경고.**
Hermes의 작업 디렉토리가 Linux 파일 시스템 안에 있으며 Windows `cmd.exe`는 이를 어떻게 해야 할지 모릅니다. 해당 세션에 대해 `/mnt/c/...`에서 Hermes를 시작하거나, Windows 실행 파일을 호출하기 전에 Windows가 도달할 수 있는 경로로 `cd`하는 래퍼를 사용하세요.

**절전/최대 절전 모드 이후의 시계 오차(Clock drift).**
호스트가 절전 모드에서 다시 시작된 후 WSL2의 시계가 몇 분 지연될 수 있으며, 이는 인증 기반 서비스(OAuth, HTTPS API)를 망가뜨립니다. 필요시 다음과 같이 수정하세요:

```bash
sudo hwclock -s
```

또는 `ntpdate`를 설치하고 로그인 시 실행되게 하세요.

**미러링된 모드를 활성화한 후, 또는 VPN이 연결되었을 때 DNS가 작동 중지됨.**
미러링된 모드는 호스트 네트워크 설정을 WSL로 프록시합니다 — Windows DNS가 이상한 경우(VPN 스플릿 터널, 회사 리졸버), WSL은 이를 상속합니다. 해결 방법: `resolv.conf`를 수동으로 재정의하세요 (`/etc/wsl.conf`에 `generateResolvConf=false`를 설정한 다음, 직접 만든 `/etc/resolv.conf`에 `1.1.1.1` 또는 VPN의 DNS를 작성).

**설치 프로그램을 실행한 후 `hermes`를 찾을 수 없음.**
설치 프로그램은 `~/.bashrc`를 통해 쉘의 PATH에 `~/.local/bin`을 추가합니다. 현재 세션에 적용하려면 `source ~/.bashrc`를 실행하거나(또는 새 터미널 열기) 해야 합니다.

**Windows Defender가 WSL 파일에서 느리게 작동함.**
Windows에서 액세스할 때 Defender는 9P 브릿지를 통해 파일을 스캔하며, 이는 `/mnt/c` 스타일의 교차 경계 액세스의 느림을 증폭시킵니다. WSL 내부에서만 WSL 파일을 다루면 아무 문제가 없습니다. `\\wsl$\...`에 대해 Windows 도구를 자주 사용하는 경우, 실시간 스캔에서 WSL 배포판 경로를 제외하는 것을 고려하세요.

**디스크 공간 부족.**
WSL2는 VM 디스크를 `%LOCALAPPDATA%\Packages\...` 아래에 스파스 VHDX(sparse VHDX)로 저장합니다. 파일이 삭제되더라도 커지기만 하고 자동으로 축소되지는 않습니다. 공간을 되찾으려면: `wsl --shutdown` 후 관리자 PowerShell에서 `Optimize-VHD -Path <path-to-ext4.vhdx> -Mode Full` (Hyper-V 도구 필요) — 또는 WSL 문서에 기록된 더 간단한 `diskpart` 방법을 사용하세요.

## 다음 단계

- **[설치](/getting-started/installation)** — 실제 설치 단계 (Linux/WSL2/Termux는 모두 동일한 설치 프로그램을 사용합니다).
- **[통합(Integrations) → 제공자(Providers) → WSL2 네트워킹](/integrations/providers#wsl2-networking-windows-users)** — 로컬 모델 서버를 위한 정식 네트워킹 심층 분석.
- **[MCP 가이드 → WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)** — WSL의 Hermes에서 로그인한 Windows Chrome 제어.
- **[도구 게이트웨이](/user-guide/features/tool-gateway)** 및 **[웹 대시보드](/user-guide/features/web-dashboard)** — WSL에서 네트워크의 다른 곳으로 노출하고 싶을 가장 흔한 장기 실행 서비스들.

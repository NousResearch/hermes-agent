---
title: "Windows (WSL2) 가이드"
description: "WSL2를 통해 Windows에서 Hermes Agent 실행하기 — 설정, Windows와 Linux 간 파일 시스템 접근, 네트워킹 및 흔한 문제"
sidebar_label: "Windows (WSL2)"
sidebar_position: 2
---

# Windows (WSL2) 가이드

Hermes Agent는 이제 **기본 Windows와 WSL2를 모두** 지원합니다. 이 페이지에서는 WSL2 경로를 다룹니다. 기본 PowerShell 설치 방법은 전용 **[Windows (Native) Guide](./windows-native.md)**를 참고하세요.

**기본 Windows 대신 WSL2를 선택할 때:**
- 대시보드의 내장 터미널(`/chat` 탭)을 사용하려는 경우 — 이 창은 POSIX PTY가 필요하며 WSL2에서만 사용할 수 있습니다.
- POSIX 중심의 개발 작업을 하며 Hermes 세션이 개발 도구와 같은 파일 시스템 / 경로를 공유하기를 원하는 경우.
- 이미 WSL2 환경이 있고 두 번째 설치를 관리하고 싶지 않은 경우.

**기본 Windows로도 충분한 경우(또는 기본 Windows가 더 나은 경우):**
- 대화형 채팅, 게이트웨이(Telegram/Discord/기타), cron 스케줄러, 브라우저 도구, MCP 서버 및 대부분의 Hermes 기능은 모두 Windows에서 기본적으로 실행됩니다.
- 파일을 참조하거나 URL을 열 때마다 WSL↔Windows 경계를 신경 쓰고 싶지 않은 경우.

WSL2에서는 사실상 두 대의 컴퓨터가 작동합니다. Windows 호스트와 WSL이 관리하는 Linux VM입니다. 대부분의 혼란은 매 순간 자신이 어느 쪽에 있는지 확신하지 못해서 생깁니다.

이 가이드에서는 Hermes에 직접 영향을 주는 분할 환경의 부분을 다룹니다. WSL2 설치, Windows와 Linux 간 파일을 주고받는 방법, 양방향 네트워킹, 그리고 사용자가 실제로 자주 겪는 문제를 설명합니다.

:::info 简体中文
최소 설치 경로를 설명하는 중국어 안내가 이 페이지에 함께 제공됩니다 — 오른쪽 위의 **언어** 메뉴에서 전환한 다음 **简体中文**을 선택하세요.
:::

## WSL2를 사용하는 이유 (기본 Windows와 비교)

기본 Windows 설치는 Windows에서 직접 실행됩니다. Windows 터미널(PowerShell, Windows Terminal 등), Windows 파일 시스템 경로(`C:\Users\…`), Windows 프로세스를 사용합니다. Hermes는 Git Bash를 사용해 셸 명령을 실행합니다. 이는 현재 Claude Code와 다른 에이전트가 Windows를 처리하는 방식이며, 전면적인 재작성 없이 POSIX와 Windows의 차이를 우회합니다.

WSL2는 경량 VM에서 실제 Linux 커널을 실행하므로, 그 안의 Hermes는 Ubuntu에서 실행하는 것과 사실상 같습니다. 실제 POSIX 환경이 필요할 때 유용합니다. `fork`, `/tmp`, UNIX 소켓, 시그널 의미 체계, PTY 기반 터미널, `bash`/`zsh` 같은 셸, 그리고 Linux에서와 동일하게 동작하는 `rg`, `git`, `ffmpeg` 같은 도구를 사용할 수 있습니다.

WSL2를 사용할 때의 실제 결과는 다음과 같습니다.

- Hermes CLI, 게이트웨이, 세션, 메모리, 스킬 및 도구 런타임은 모두 Linux VM 안에 있습니다.
- Windows 프로그램(브라우저, 기본 앱, 로그인된 프로필이 있는 Chrome)은 VM 밖에 있습니다.
- 파일을 공유하거나, URL을 열거나, Chrome을 제어하거나, 로컬 모델 서버에 접속하거나, Hermes 게이트웨이를 휴대폰에 노출하는 등 두 환경이 통신할 때마다 경계를 넘습니다. 이 가이드는 바로 그 경계를 다룹니다.

## WSL2 설치

**관리자 PowerShell** 또는 Windows Terminal에서 실행합니다.

```powershell
wsl --install
```

새로 설치한 Windows 10 22H2+ 또는 Windows 11에서는 WSL2 커널, Virtual Machine Platform 기능, 기본 Ubuntu 배포판이 설치됩니다. 재부팅하라는 메시지가 나타나면 재부팅하세요. 재부팅 후 Ubuntu가 열리고 Linux 사용자 이름과 비밀번호를 묻습니다. 이는 Windows 계정과 무관한 **새 Linux 사용자**입니다.

실제로 레거시 WSL1이 아니라 WSL2를 사용 중인지 확인합니다.

```powershell
wsl --list --verbose
```

`VERSION  2`가 표시되어야 합니다. 배포판에 `VERSION  1`이 표시되면 다음과 같이 변환합니다.

```powershell
wsl --set-version Ubuntu 2
wsl --set-default-version 2
```

Hermes는 WSL1에서 안정적으로 작동하지 않습니다. WSL1은 Linux 시스템 호출을 즉석에서 변환하므로 procfs, 시그널, 네트워크 등의 동작이 실제 Linux와 다릅니다.

### 배포판 선택

Ubuntu(LTS)를 기준으로 테스트합니다. Debian도 작동합니다. Arch와 NixOS도 원하는 사용자는 사용할 수 있지만, 한 줄 설치 프로그램은 Debian 계열의 `apt` 시스템을 전제로 합니다. 해당 경로는 [Nix 설정 가이드](/getting-started/nix-setup)를 참고하세요.

### systemd 활성화(권장)

hermes 게이트웨이(및 계속 실행해 두려는 다른 항목)는 systemd를 사용하면 관리하기 쉽습니다. 최신 WSL에서는 배포판 안에서 한 번만 활성화하면 됩니다.

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

그런 다음 PowerShell에서 실행합니다.

```powershell
wsl --shutdown
```

WSL 터미널을 다시 엽니다. `ps -p 1 -o comm=`을 실행하면 `systemd`가 출력되어야 합니다.

위의 `metadata` 마운트 옵션은 중요합니다. 이 옵션이 없으면 `/mnt/c/...`의 파일에 실제 Linux 권한 비트를 저장할 수 없어 Windows 경로 아래 스크립트에서 `chmod +x`를 실행하는 등의 작업이 깨집니다.

### WSL 안에 Hermes 설치

WSL2 셸을 열었으면 다음을 실행합니다.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
hermes
```

설치 프로그램은 WSL2를 일반 Linux로 취급하므로 WSL 전용 설정은 필요하지 않습니다. 전체 구성은 [설치](/getting-started/installation)를 참고하세요.

## 파일 시스템: Windows ↔ WSL2 경계 넘기

이 부분에서 가장 많은 사용자가 어려움을 겪습니다. **두 개의 파일 시스템**이 있으며, 파일을 어디에 두는지는 성능, 정확성, 도구에서 파일을 볼 수 있는지에 영향을 줍니다.

### 두 방향

| 방향 | 내부 경로 | 사용하는 경로 |
|---|---|---|
| WSL에서 본 Windows 디스크 | `C:\Users\you\Documents` | `/mnt/c/Users/you/Documents` |
| Windows에서 본 WSL 디스크 | `/home/you/code` | `\\wsl$\Ubuntu\home\you\code` (또는 최신 빌드에서는 `\\wsl.localhost\Ubuntu\...`) |

둘 다 실제 파일 시스템이며 모두 작동하지만, **같은 파일 시스템은 아닙니다**. 내부적으로 9P 네트워크 프로토콜을 통해 연결됩니다. 따라서 성능과 의미 체계에 실제 차이가 생깁니다.

### Hermes와 프로젝트를 둘 위치

**일반적인 원칙: Linux에 가까운 것은 모두 Linux 파일 시스템 안에 둡니다.**

- Hermes 설치(`~/.hermes/`) — Linux 쪽. 설치 프로그램이 이미 이 위치에 설치합니다.
- WSL에서 작업하는 git 저장소 — Linux 쪽(`~/code/...`, `~/projects/...`).
- 모델, 데이터셋, venv — Linux 쪽.

이 원칙을 따르면 다음과 같은 이점이 있습니다.

- **빠른 I/O.** `/mnt/c/...`에서의 작업은 9P를 거치므로 기본 ext4보다 10~100배 느립니다. `~/code`에서는 즉시 끝나는 1만 개 파일 저장소의 `git status`가 `/mnt/c`에서는 15초 이상 걸릴 수 있습니다.
- **정확한 권한.** Linux 권한 비트는 `/mnt/c`에서 최선의 방식으로 에뮬레이트됩니다. SSH가 "bad permissions" 때문에 키를 거부하거나 `chmod +x`가 조용히 실패하는 일이 흔합니다.
- **신뢰할 수 있는 파일 감시.** 9P를 통한 inotify는 불안정합니다. `/mnt/c`에서 실행하는 개발 서버와 테스트 실행기 같은 파일 감시 도구가 변경 사항을 자주 놓칩니다.
- **대소문자 구분 문제 없음.** Windows 경로는 기본적으로 대소문자를 구분하지 않지만 Linux는 구분합니다. `Readme.md`와 `README.md`가 모두 있는 프로젝트는 어느 쪽에서 작업하는지에 따라 다르게 동작합니다.

Windows 쪽에 파일이 있어야 **할 때만** `/mnt/c`에 두세요. 예를 들어 Windows GUI 앱에서 열고 싶거나, Windows Chrome의 DevTools MCP가 현재 디렉터리를 Windows에서 접근 가능한 경로로 요구하는 경우입니다.

### 파일을 주고받는 방법

**Windows에서 WSL로:** 가장 쉬운 방법은 탐색기를 열고 주소 표시줄에 `\\wsl.localhost\Ubuntu`를 입력하는 것입니다. 그런 다음 `\home\<you>\...`로 끌어다 놓을 수 있습니다. 또는 PowerShell에서 다음을 실행합니다.

```powershell
wsl cp /mnt/c/Users/you/Downloads/file.pdf ~/incoming/
```

**WSL에서 Windows로:** `/mnt/c/Users/<you>/...`에 복사하면 Windows 탐색기에 즉시 표시됩니다.

```bash
cp ~/reports/output.pdf /mnt/c/Users/you/Desktop/
```

**Windows 앱(GUI 편집기, 브라우저 등)에서 WSL 파일 열기:** `explorer.exe` 또는 `wslview`를 사용합니다.

```bash
sudo apt install wslu     # once — gives you wslview, wslpath, wslopen, etc.
wslview ~/reports/output.pdf    # opens with the Windows default handler
explorer.exe .                  # opens the current WSL dir in Windows Explorer
```

**두 환경 사이에서 경로 변환하기:**

```bash
wslpath -w ~/code/project        # → \\wsl.localhost\Ubuntu\home\you\code\project
wslpath -u 'C:\Users\you'        # → /mnt/c/Users/you
```

### 줄바꿈, BOM 및 git

Windows 편집기로 Windows 쪽에서 파일을 편집하면 `CRLF` 줄바꿈이 들어갈 수 있습니다. Linux 쪽의 `bash` 또는 Python이 이를 읽으면 셸 스크립트가 `bad interpreter: /bin/bash^M` 오류와 함께 깨지고, BOM이 있는 `.env` 파일 때문에 Python이 실패할 수 있습니다.

해결 방법은 WSL 안에서(Windows가 아닌) 적절한 git 설정을 적용하는 것입니다.

```bash
git config --global core.autocrlf input
git config --global core.eol lf
```

이미 CRLF인 파일은 다음과 같이 처리합니다.

```bash
sudo apt install dos2unix
dos2unix path/to/script.sh
```

### "WSL 안에 복제할까, `/mnt/c`에 복제할까?"

WSL 안에 복제하세요. 특별한 이유가 없는 한 항상 그렇습니다. 일반적인 Hermes 작업(`hermes chat`, 저장소에서 `rg`/`ripgrep`를 실행하는 도구 호출, 파일 감시, 백그라운드 게이트웨이)은 `/mnt/c/Users/you/myrepo`보다 `~/code/myrepo`에서 훨씬 빠르고 안정적입니다.

한 가지 예외는 **Windows 바이너리를 실행하는 MCP 브리지**입니다. `cmd.exe`를 통해 `chrome-devtools-mcp`를 사용한다면([MCP 가이드: WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome) 참고), Hermes의 현재 작업 디렉터리가 `~`일 때 Windows가 `UNC` 경고를 표시할 수 있습니다. 이 경우 해당 세션에서는 `/mnt/c/` 아래에서 Hermes를 시작하거나, Windows 프로세스를 실행하기 전에 Windows에서 접근 가능한 경로로 `cd`하는 래퍼를 사용하세요.

## 네트워킹: WSL ↔ Windows

WSL2는 자체 네트워크 스택을 가진 경량 VM에서 실행됩니다. 따라서 WSL 안의 `localhost`는 Windows의 `localhost`와 같지 않습니다. 네트워크 관점에서는 서로 다른 호스트입니다. 각 서비스마다 트래픽 방향을 정하고 올바른 브리지를 선택해야 합니다.

두 가지 경우가 계속해서 발생합니다.

### 사례 1 — WSL의 Hermes가 Windows의 서비스와 통신

가장 흔한 경우입니다. **Windows에서 Ollama, LM Studio 또는 llama-server를 실행하고** Hermes(WSL 안)가 여기에 접속해야 하는 경우입니다.

표준 방법은 providers 가이드에 있습니다. **[WSL2 Networking for Local Models →](/integrations/providers#wsl2-networking-windows-users)**

요약하면 다음과 같습니다.

- **Windows 11 22H2+:** 미러링 네트워킹 모드를 켭니다(`%USERPROFILE%\.wslconfig`에서 `networkingMode=mirrored` 설정 후 `wsl --shutdown`). 그러면 양방향에서 `localhost`가 작동합니다.
- **Windows 10 또는 이전 빌드:** Windows 호스트 IP(WSL 가상 네트워크의 기본 게이트웨이)를 사용하고 Windows의 서버가 `127.0.0.1`이 아니라 `0.0.0.0`에 바인딩되도록 합니다. 대개 Windows 방화벽에 포트 규칙도 필요합니다.

전체 표(Ollama / LM Studio / vLLM / SGLang 바인드 주소, 방화벽 규칙 한 줄 명령, 동적 IP 도우미, Hyper-V 방화벽 우회)는 위 링크를 참고하세요. 중복해서 작성하지 않습니다.

### 사례 2 — Windows(또는 LAN)의 무언가가 WSL의 Hermes와 통신

이 반대 방향은 다른 곳에서 문서화된 내용이 적지만, 다음과 같은 경우에 필요합니다.

- Windows 브라우저에서 Hermes **웹 대시보드** 사용.
- Windows 쪽 도구에서 **OpenAI 호환 API 서버** 사용(`API_SERVER_ENABLED=true`일 때 `hermes gateway`가 노출). [API Server 기능 페이지](/user-guide/features/api-server)를 참고하세요.
- 메시징 게이트웨이(Telegram, Discord 등) 테스트. 플랫폼이 로컬 웹훅 URL을 호출하는 경우이며, 일반적으로 직접 포트 포워딩하는 대신 `cloudflared`/`ngrok`를 사용합니다.

#### 하위 사례 2a: Windows 호스트 자체에서

**미러링 모드가 활성화된 Windows 11 22H2+**에서는 할 일이 없습니다. WSL의 프로세스가 `0.0.0.0:8080`(또는 `127.0.0.1:8080`)에 바인딩하면 WSL이 바인딩을 호스트로 자동 게시하므로 Windows 브라우저에서 `http://localhost:8080`으로 접근할 수 있습니다.

**NAT 모드**(Windows 10 / 이전 Windows 11)에서는 WSL2의 기본 "localhost forwarding"이 일반적으로 Linux 쪽 `127.0.0.1` 바인딩을 Windows의 `localhost`로 전달하므로, `--host 127.0.0.1`로 시작한 Hermes 서비스는 보통 Windows에서 `http://localhost:PORT`로 접근할 수 있습니다. 접근할 수 없다면 다음을 시도하세요.

- WSL 안에서 명시적으로 `0.0.0.0`에 바인딩합니다.
- `ip -4 addr show eth0 | grep inet`으로 WSL VM의 IP를 찾고 Windows에서 해당 IP로 접속합니다.

#### 하위 사례 2b: LAN의 다른 장치(휴대폰, 태블릿, 다른 PC)에서

이 경우가 실제로 가장 어렵습니다. 트래픽은 **LAN 장치 → Windows 호스트 → WSL VM**으로 흐르며, 두 홉을 모두 설정해야 합니다.

1. **WSL 안에서 모든 인터페이스에 바인딩합니다.** `127.0.0.1`에서 수신하는 프로세스는 VM 외부에서 절대 접근할 수 없습니다. `0.0.0.0`을 사용하세요.

2. **Windows에서 WSL VM으로 포트 포워딩합니다.** 미러링 모드에서는 자동입니다. NAT 모드에서는 관리자 PowerShell에서 포트마다 직접 설정해야 합니다.

   ```powershell
   # Grab the WSL VM's current IP (it changes on every WSL restart under NAT)
   $wslIp = (wsl hostname -I).Trim().Split(' ')[0]

   # Forward Windows port 8080 → WSL:8080
   netsh interface portproxy add v4tov4 `
     listenaddress=0.0.0.0 listenport=8080 `
     connectaddress=$wslIp connectport=8080

   # Allow it through Windows Firewall
   New-NetFirewallRule -DisplayName "Hermes WSL 8080" `
     -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
   ```

   나중에 다음 명령으로 제거합니다. `netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=8080`.

3. **LAN 장치에서 `http://<windows-lan-ip>:8080`을 대상으로 지정합니다.**

NAT 모드에서는 WSL VM IP가 재시작할 때마다 바뀌므로 한 번 설정한 규칙은 다음 `wsl --shutdown`까지만 유지됩니다. 지속적으로 사용하려면 미러링 모드를 사용하거나 Windows 로그인 시 실행되는 스크립트에 포트 프록시 단계를 넣으세요.

클라우드 메시징 제공업체의 웹훅(Telegram `setWebhook`, Slack 이벤트 등)은 포트 포워딩과 씨름하지 말고 `cloudflared` 터널을 사용하세요. [웹훅 가이드](/user-guide/messaging/webhooks)를 참고하세요.

## Windows에서 Hermes 서비스를 장기간 실행하기

Hermes [Tool Gateway](/user-guide/features/tool-gateway)와 API 서버는 장시간 실행되는 프로세스입니다. WSL2에서는 계속 실행하기 위한 몇 가지 방법이 있습니다.

### Hermes를 빠르게 여는 바탕 화면 바로 가기

대화형 Hermes 셸을 더블클릭으로 실행하는 바로 가기만 필요하다면 Windows 쪽에서 만들고 WSL로 진입하게 하세요.

1. Windows 바탕 화면을 마우스 오른쪽 버튼으로 클릭하고 **새로 만들기 -> 바로 가기**를 선택합니다.
2. 대상에는 배포판 이름을 사용합니다(필요하면 `Ubuntu`를 바꾸세요).

   ```text
   wt.exe -w 0 -p "Ubuntu" wsl.exe -d Ubuntu --cd ~ -- bash -ic "hermes"
   ```

3. `Hermes`처럼 알아보기 쉬운 이름을 지정합니다.

그러면 Windows Terminal이 열리고, WSL 배포판이 시작되며, Linux 홈 디렉터리로 이동한 뒤 Hermes가 실행됩니다. 아직 `hermes`가 PATH에 없다면 WSL을 한 번 수동으로 열고 `source ~/.bashrc`를 실행하거나, 명령을 프로젝트 체크아웃 안에서 `uv run hermes`로 바꾸세요.

선택적으로 다음을 다듬을 수 있습니다.

- **사용자 지정 아이콘:** **속성 -> 아이콘 변경**을 열고 저장소의 Hermes 파비콘 같은 `.ico` 파일을 지정합니다.
- **고정 실행기:** 바로 가기가 작동하면 시작 메뉴 또는 작업 표시줄에 고정해 다시 찾지 않아도 되게 합니다.

### systemd를 사용하는 WSL 안에서(권장)

위 설정 섹션에 따라 systemd를 활성화했다면 `hermes gateway`와 API 서버는 모든 Linux 컴퓨터에서처럼 작동합니다. 게이트웨이 설정 마법사를 사용하세요.

```bash
hermes gateway setup
```

WSL이 시작될 때 게이트웨이가 자동으로 올라오도록 systemd 사용자 유닛을 설치할지 묻습니다.

### Windows 로그인 시 WSL 자체 시작

WSL의 VM은 무언가가 사용 중일 때만 계속 실행됩니다. 터미널 창을 열어 두지 않고 게이트웨이에 접근할 수 있게 하려면 작업 스케줄러를 통해 Windows 로그인 시 WSL 프로세스를 부팅하세요.

- **트리거:** 로그온할 때(사용자 계정).
- **동작:** 프로그램 시작
  - 프로그램: `C:\Windows\System32\wsl.exe`
  - 인수: `-d Ubuntu --exec /bin/sh -c "sleep infinity"`

이렇게 하면 VM이 계속 살아 있어 systemd가 관리하는 게이트웨이가 실행 상태로 유지됩니다. Windows 11에서는 더 새로운 `wsl --install --no-launch` + 자동 시작 흐름도 작동합니다. `sleep infinity` 방식은 이식성 있는 방법입니다.

## GPU 패스스루(로컬 모델)

WSL2는 WSL 커널 5.10.43+부터 **NVIDIA** GPU를 기본적으로 지원합니다. Windows에 표준 NVIDIA 드라이버를 설치하고(WSL 안에는 Linux NVIDIA 드라이버를 설치하지 **마세요**), WSL 안에서 `nvidia-smi`를 실행하면 GPU가 표시됩니다. 이후 CUDA 툴킷, `torch`, `vllm`, `sglang`, `llama-server`는 평소처럼 실제 GPU에 맞춰 빌드됩니다.

WSL2 안의 AMD ROCm 및 Intel Arc 지원은 아직 발전 중이며 Hermes 테스트 범위 밖입니다. 최신 드라이버에서 작동할 수는 있지만 권장할 수 있는 절차는 없습니다.

Windows 드라이버를 통해 이미 GPU를 사용하는 **Windows 기본 로컬 모델 서버**(Windows용 Ollama, LM Studio)를 실행 중이라면 WSL GPU 패스스루가 전혀 필요하지 않습니다. 위의 사례 1을 따라 WSL에서 네트워크를 통해 접속하세요.

## 흔한 문제

**Windows에서 실행한 Ollama / LM Studio에 "Connection refused"가 표시됩니다.**
[WSL2 Networking](/integrations/providers#wsl2-networking-windows-users)을 참고하세요. 90%의 경우 서버가 `127.0.0.1`에 바인딩되어 있어 `0.0.0.0`이 필요하거나(Ollama: `OLLAMA_HOST=0.0.0.0`), 방화벽 규칙이 빠진 상태입니다.

**저장소에서 `git status` / `hermes chat`가 매우 느립니다.**
아마 `/mnt/c/...` 아래에서 작업 중일 것입니다. 저장소를 `~/code/...`(Linux 쪽)로 옮기세요. 한 자릿수 차이로 빨라집니다.

**스크립트에서 `bad interpreter: /bin/bash^M`이 표시됩니다.**
Windows 편집기에서 생긴 CRLF 줄바꿈입니다. `dos2unix script.sh`를 실행하고 WSL git 설정에서 `core.autocrlf input`을 지정하세요.

**MCP를 통해 실행한 Windows 바이너리에서 "UNC paths are not supported" 경고가 표시됩니다.**
Hermes의 현재 작업 디렉터리가 Linux 파일 시스템 안에 있고 Windows `cmd.exe`가 이를 처리하지 못하는 것입니다. 해당 세션에서는 `/mnt/c/...`에서 Hermes를 시작하거나, Windows 실행 파일을 호출하기 전에 Windows에서 접근 가능한 경로로 `cd`하는 래퍼를 사용하세요.

**절전/최대 절전 모드 후 시간이 어긋납니다.**
호스트가 절전 모드에서 복귀한 후 WSL2의 시계가 몇 분 늦어질 수 있어 인증서 기반 기능(OAuth, HTTPS API)이 깨집니다. 필요할 때 다음으로 수정합니다.

```bash
sudo hwclock -s
```

또는 `ntpdate`를 설치하고 로그인 시 실행하세요.

**미러링 모드를 활성화한 후 또는 VPN 연결 시 DNS가 작동하지 않습니다.**
미러링 모드는 호스트 네트워크 설정을 WSL로 전달합니다. Windows DNS가 이상하면(VPN 분할 터널, 기업용 확인자) WSL도 이를 상속합니다. 해결 방법은 `resolv.conf`를 수동으로 덮어쓰는 것입니다(`/etc/wsl.conf`에서 `generateResolvConf=false`를 설정한 뒤 `1.1.1.1` 또는 VPN의 DNS를 사용해 자체 `/etc/resolv.conf`를 작성).

**설치 프로그램을 실행한 후 `hermes`를 찾을 수 없습니다.**
설치 프로그램은 `~/.bashrc`를 통해 셸의 PATH에 `~/.local/bin`을 추가합니다. 현재 세션에 적용하려면 `source ~/.bashrc`를 실행하거나 새 터미널을 열어야 합니다.

**Windows Defender가 WSL 파일에서 느립니다.**
Windows에서 접근할 때 Defender가 9P 브리지를 통해 파일을 검사하므로 `/mnt/c` 방식의 경계 통과 접근이 더 느려집니다. WSL 안에서만 WSL 파일을 다룬다면 문제가 되지 않습니다. Windows 도구로 `\\wsl$\...`를 자주 사용한다면 WSL 배포판 경로를 실시간 검사에서 제외하는 것을 고려하세요.

**디스크 공간이 부족합니다.**
WSL2는 `%LOCALAPPDATA%\Packages\...` 아래에 희소 VHDX로 VM 디스크를 저장합니다. 디스크는 커지지만 파일을 삭제해도 자동으로 줄어들지 않습니다. 공간을 회수하려면 `wsl --shutdown`을 실행한 다음 관리자 PowerShell에서 `Optimize-VHD -Path <path-to-ext4.vhdx> -Mode Full`을 실행하세요(Hyper-V 도구 필요). 또는 WSL 문서에 설명된 더 간단한 `diskpart` 방법을 사용하세요.

## 다음 단계

- **[설치](/getting-started/installation)** — 실제 설치 단계(Linux/WSL2/Termux는 모두 같은 설치 프로그램 사용).
- **[통합 → Providers → WSL2 Networking](/integrations/providers#wsl2-networking-windows-users)** — 로컬 모델 서버를 위한 네트워킹 상세 설명.
- **[MCP 가이드 → WSL → Windows Chrome](/guides/use-mcp-with-hermes#wsl2-bridge-hermes-in-wsl-to-windows-chrome)** — WSL의 Hermes에서 로그인된 Windows Chrome 제어.
- **[Tool Gateway](/user-guide/features/tool-gateway)** 및 **[Web Dashboard](/user-guide/features/web-dashboard)** — WSL에서 네트워크의 나머지 환경으로 노출하려는 경우가 가장 많은 장기 실행 서비스.

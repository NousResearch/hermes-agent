---
sidebar_position: 2
title: "설치"
description: "Linux, macOS, WSL2, native Windows 또는 Termux를 통한 Android에 Hermes Agent 설치"
---

# 설치

Hermes Agent를 2분 안에 실행해 보세요!

:::tip 플랫폼 지원
전체 플랫폼 지원 매트릭스(OS, 배포 방법, 플랫폼별 기능)를 보려면 **[플랫폼 지원](./platform-support.md)**을 참고하세요.
:::

## 빠른 설치
### macOS 또는 Windows에서 Hermes Desktop 설치 프로그램 사용(권장)
명령줄 앱과 데스크톱 앱을 쉽게 설치하려면 웹사이트에서 [Hermes Desktop 설치 프로그램](https://hermes-agent.nousresearch.com/)을 내려받아 실행하세요.

### Hermes Desktop 없이 설치
Hermes Desktop 없이 명령줄만 설치하려면 다음을 실행하세요.

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (native)

PowerShell에서 실행하세요.
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

명령줄만 설치한 뒤 Hermes Desktop도 설치하고 실행하려면 다음만 실행하면 됩니다.
```bash
hermes desktop
```

### 설치 프로그램이 하는 일

설치 프로그램은 모든 작업을 자동으로 처리합니다. 모든 의존성(Python, Node.js, ripgrep, ffmpeg), 저장소 복제, 가상 환경, 전역 `hermes` 명령 설정, LLM 제공자 설정까지 처리하므로 설치가 끝나면 바로 대화할 수 있습니다.

#### 설치 구조

일반 사용자로 설치하는지 root로 설치하는지에 따라 설치 위치가 달라집니다.

| 설치 프로그램 | 코드 위치 | `hermes` 바이너리 | 데이터 디렉터리 |
| --- | --- | --- | --- |
| 사용자별 설치(git installer) | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` (symlink) | `~/.hermes/` |
| root 모드(`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/` (또는 `$HERMES_HOME`) |

root 모드의 **FHS 구조**(`/usr/local/lib/…`, `/usr/local/bin/hermes`)는 Linux에서 다른 시스템 전체 개발 도구가 설치되는 위치와 같습니다. 하나의 시스템 설치로 모든 사용자를 지원해야 하는 공유 머신에 유용합니다. 사용자별 설정(auth, skills, sessions)은 각 사용자의 `~/.hermes/` 또는 명시적인 `HERMES_HOME` 아래에 계속 저장됩니다.

### 설치 후

셸을 다시 불러오고 대화를 시작하세요.

```bash
source ~/.bashrc   # or: source ~/.zshrc
hermes             # Start chatting!
```

나중에 개별 설정을 다시 구성하려면 전용 명령을 사용하세요.

```bash
hermes model          # Choose your LLM provider and model
hermes tools          # Configure which tools are enabled
hermes gateway setup  # Set up messaging platforms
hermes config set     # Set individual config values
hermes config get     # Inspect individual config values
hermes setup          # Or run the full setup wizard to configure everything at once
```

:::tip 가장 빠른 경로: Nous Portal
하나의 구독으로 300개 이상의 모델과 [Tool Gateway](/user-guide/features/tool-gateway)(웹 검색, 이미지 생성, TTS, 클라우드 브라우저)를 사용할 수 있습니다. 도구별 키를 따로 관리할 필요가 없습니다.

```bash
hermes setup --portal
```

이 명령은 로그인하고 Nous를 제공자로 설정하며 한 번에 Tool Gateway를 활성화합니다.
:::

---

## 사전 요구 사항

**설치 프로그램:** Windows가 아닌 플랫폼에서는 **Git**만 있으면 됩니다. Linux에서는 `curl`과 `xz-utils`도 사용할 수 있어야 합니다(설치 프로그램이 Node.js를 `.tar.xz` 아카이브로 내려받습니다). 데스크톱 앱은 네이티브 모듈을 컴파일하기 위해 `g++`(Debian/Ubuntu에서는 `build-essential`)도 필요합니다. 나머지는 설치 프로그램이 자동으로 처리합니다.

- **uv** (빠른 Python 패키지 관리자)
- **Python 3.11** (uv를 통해, sudo 불필요)
- **Node.js v22** (브라우저 자동화 및 WhatsApp 브리지용)
- **ripgrep** (빠른 파일 검색)
- **ffmpeg** (TTS용 오디오 형식 변환)

:::info
Python, Node.js, ripgrep, ffmpeg를 직접 설치할 필요는 없습니다. 설치 프로그램이 없는 항목을 찾아 설치합니다. `git`만 사용할 수 있는지 확인하세요(`git --version`). Linux에서는 `curl`과 `xz-utils`가 설치되어 있는지 확인하세요(Debian/Ubuntu: `sudo apt install curl xz-utils`). 데스크톱 앱에는 `build-essential`도 설치하세요(`sudo apt install build-essential`).
:::

:::tip Nix 사용자
Nix는 **더 이상 명시적으로 지원되는 설치 경로가 아닙니다**(최선의 노력으로만 지원). 이미 Nix(NixOS, macOS 또는 Linux)를 사용한다면 Nix flake, 선언적 NixOS 모듈, 선택적 컨테이너 모드를 제공하는 전용 설정 경로가 있습니다. **[Nix 및 NixOS 설정](./nix-setup.md)** 가이드를 참고하세요.
:::

---

## 수동 / 개발자 설치

저장소를 복제하고 소스에서 설치하려면(기여, 특정 브랜치 실행, 가상 환경 완전 제어 등) 기여 가이드의 [개발 환경 설정](../developer-guide/contributing.md#development-setup) 절을 참고하세요.

---

## sudo 없이 / 시스템 서비스 사용자로 설치

전용 권한 없는 사용자(예: `hermes` systemd 서비스 계정 또는 `sudo` 권한이 없는 사용자)로 Hermes를 실행할 수 있습니다. 설치 과정에서 실제로 root가 필요한 부분은 Playwright의 `--with-deps` 단계뿐이며, 이 단계는 Chromium이 사용하는 공유 라이브러리(`libnss3`, `libxkbcommon` 등)를 `apt`로 설치합니다. 설치 프로그램은 sudo 사용 가능 여부를 감지해 sudo가 없으면 적절히 축소하여 진행합니다. Chromium 바이너리를 서비스 사용자 자신의 Playwright 캐시에 설치하고 관리자가 별도로 실행해야 할 정확한 명령을 출력합니다.

**권장 분리 방식(Debian/Ubuntu):**

1. **관리자 권한이 있는 sudo 사용자로 한 번**, Chromium에 필요한 시스템 라이브러리를 설치합니다.
   ```bash
   sudo npx playwright install-deps chromium
   ```
   (어디서나 실행할 수 있으며 `npx`가 Playwright를 즉시 가져옵니다.)

2. **권한 없는 서비스 사용자로** 일반 설치 프로그램을 실행합니다. sudo가 없음을 감지하고 `--with-deps`를 건너뛴 뒤 Chromium을 사용자의 로컬 Playwright 캐시에 설치합니다.
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   Playwright 단계를 완전히 건너뛰려면(예: headless로 실행하며 브라우저 자동화가 필요 없는 경우) `--skip-browser`를 전달하세요.
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

   설치 프로그램은 [`cua-driver`](../user-guide/features/computer-use.md)도 미리 설치하므로 Computer Use 도구셋을 활성화하는 즉시 사용할 수 있습니다. 선택하지 않으려면 `--skip-computer-use`를 전달하세요(이 경우 도구를 활성화할 때 필요에 따라 설치됩니다).

3. **서비스 사용자 셸에서 `hermes`를 사용할 수 있게 합니다.** 설치 프로그램은 런처를 `~/.local/bin/hermes`에 기록합니다. 시스템 서비스 계정은 `~/.local/bin`이 없는 최소 PATH를 사용하는 경우가 많습니다. 사용자 환경에 추가하거나 런처를 시스템 위치에 symlink하세요.
   ```bash
   # Option A — add to the service user's profile
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # Option B — symlink system-wide (run as an admin)
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **확인:** 이제 `hermes doctor`가 문제없이 실행되어야 합니다. `ModuleNotFoundError: No module named 'dotenv'`가 표시되면 venv 런처(`~/.hermes/hermes-agent/venv/bin/hermes`)가 아니라 시스템 Python으로 저장소 소스의 `hermes` 파일(`~/.hermes/hermes-agent/hermes`)을 호출하고 있는 것입니다. 3단계를 수정하세요.

5. **이 계정에서 메시징 게이트웨이를 실행하나요?** 사용자 서비스는 로그아웃하면 중지되며 서비스 사용자의 lingering을 활성화하기 전에는 부팅 시 시작되지 않습니다.

   ```bash
   sudo loginctl enable-linger <service-user>
   ```

   서비스 자체 설정은 [메시징 게이트웨이](/user-guide/messaging/)를 참고하세요.

같은 패턴은 Arch(설치 프로그램이 동일한 sudo 감지 로직으로 pacman 사용), Fedora/RHEL, openSUSE에서도 작동합니다. 이 배포판은 `--with-deps`를 전혀 지원하지 않으므로 관리자가 항상 시스템 라이브러리를 별도로 설치해야 합니다. 관련 `dnf`/`zypper` 명령은 설치 프로그램이 출력합니다.

---

## 문제 해결

| 문제 | 해결 방법 |
| --- | --- |
| `hermes: command not found` | 셸을 다시 불러오거나(`source ~/.bashrc`) PATH를 확인하세요. |
| `API key not set` | `hermes model`로 제공자를 구성하거나 `hermes config set OPENROUTER_API_KEY your_key`를 실행하세요. |
| 업데이트 후 설정이 없음 | `hermes config check`를 실행한 다음 `hermes config migrate`를 실행하세요. |

더 자세한 진단은 `hermes doctor`를 실행하세요. 누락된 항목과 해결 방법을 정확히 알려줍니다.

## 설치 방법 자동 감지

Hermes는 git installer, Docker 또는 NixOS를 통해 설치되었는지 자동으로 감지하고 `hermes update`에서 해당 경로에 맞는 업데이트 명령을 출력합니다. 설정할 환경 변수는 없습니다. 설치 구조(`~/.hermes/hermes-agent/` checkout, Docker image stamp 또는 Nix store path)를 기준으로 감지합니다. `hermes doctor`의 환경 요약에도 감지된 방법이 표시됩니다.

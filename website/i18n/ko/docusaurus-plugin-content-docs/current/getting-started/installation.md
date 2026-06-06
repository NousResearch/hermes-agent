---
sidebar_position: 2
title: "설치"
description: "Linux, macOS, WSL2, 기본 Windows 또는 Termux를 통한 Android에 Hermes Agent 설치하기"
---

# 설치

2분 안에 Hermes Agent를 설치하고 바로 사용해 보세요!

## 빠른 설치
### macOS 또는 Windows에서 Hermes Desktop 설치 프로그램 사용 (권장)
커맨드라인 및 데스크톱 애플리케이션을 쉽게 설치하려면 웹사이트에서 [Hermes Desktop 설치 프로그램 다운로드](https://hermes-agent.nousresearch.com/desktop) 후 실행하세요.

### Hermes Desktop 없이 설치:
Hermes Desktop 없이 커맨드라인 전용으로 설치하려면 다음을 실행하세요.

#### Linux / macOS / WSL2 / Android (Termux)
```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

#### Windows (기본)

PowerShell에서 실행하세요:
```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1) 
```

커맨드라인 전용으로 설치한 후 Hermes Desktop을 설치하고 실행하려면 다음 명령어를 실행하면 됩니다.
```bash
hermes desktop
```

### 설치 프로그램이 수행하는 작업

설치 프로그램은 모든 의존성(Python, Node.js, ripgrep, ffmpeg), 저장소 복제(repo clone), 가상 환경, 전역 `hermes` 명령어 설정 및 LLM 프로바이더 구성을 자동으로 처리합니다. 완료되고 나면 바로 대화를 시작할 수 있습니다.

#### 설치 레이아웃

설치 프로그램이 항목을 배치하는 위치는 일반 사용자로 설치하는지 또는 root 권한으로 설치하는지에 따라 다릅니다.

| 설치 방법 | 코드 위치 | `hermes` 바이너리 | 데이터 디렉터리 |
|---|---|---|---|
| pip install | Python site-packages | `~/.local/bin/hermes` (console_scripts) | `~/.hermes/` |
| 사용자별 (git 설치 프로그램) | `~/.hermes/hermes-agent/` | `~/.local/bin/hermes` (심볼릭 링크) | `~/.hermes/` |
| Root 모드 (`sudo curl … \| sudo bash`) | `/usr/local/lib/hermes-agent/` | `/usr/local/bin/hermes` | `/root/.hermes/` (또는 `$HERMES_HOME`) |

Root 모드 **FHS 레이아웃** (`/usr/local/lib/…`, `/usr/local/bin/hermes`)은 Linux에서 다른 시스템 전역 개발자 도구가 저장되는 위치와 일치합니다. 이는 하나의 시스템 설치로 모든 사용자에게 서비스를 제공하려는 공유 머신 배포에 유용합니다. 사용자별 설정(인증, skills, 세션)은 각 사용자의 `~/.hermes/` 또는 명시적으로 지정된 `HERMES_HOME` 디렉터리에 계속 유지됩니다.

### 설치 후 작업

셸을 다시 로드하고 대화를 시작하세요.

```bash
source ~/.bashrc   # 또는: source ~/.zshrc
hermes             # 대화 시작하기!
```

나중에 개별 설정을 다시 구성하려면 전용 명령어를 사용하세요.

```bash
hermes model          # LLM 프로바이더 및 모델 선택
hermes tools          # 활성화할 도구 설정
hermes gateway setup  # 메시징 플랫폼 설정
hermes config set     # 개별 설정 값 구성
hermes setup          # 또는 모든 항목을 한 번에 구성하는 전체 설정 마법사 실행
```

:::tip 가장 빠른 방법: Nous Portal
구독 하나로 300개 이상의 모델과 [Tool Gateway](/user-guide/features/tool-gateway)(웹 검색, 이미지 생성, TTS, 클라우드 브라우저)를 모두 이용할 수 있습니다. 도구별 API 키를 번거롭게 관리할 필요가 없습니다.

```bash
hermes setup --portal
```

이 명령어 하나로 로그인, Nous를 프로바이더로 설정, Tool Gateway 활성화가 동시에 수행됩니다.
:::

---

## 사전 요구 사항

**설치 프로그램:** Windows가 아닌 플랫폼의 경우 유일한 사전 요구 사항은 **Git**입니다. 설치 프로그램이 나머지 모든 항목을 자동으로 처리합니다.

- **uv** (빠른 Python 패키지 관리자)
- **Python 3.11** (uv를 통해 설치되며, sudo 권한 불필요)
- **Node.js v22** (브라우저 자동화 및 WhatsApp 브리지용)
- **ripgrep** (빠른 파일 검색)
- **ffmpeg** (TTS용 오디오 형식 변환)

:::info
Python, Node.js, ripgrep 또는 ffmpeg를 수동으로 설치할 필요가 **없습니다**. 설치 프로그램이 누락된 항목을 감지하여 자동으로 설치해 줍니다. `git`이 준비되어 있는지 확인해 주세요 (`git --version`).
:::

:::tip Nix 사용자
NixOS, macOS 또는 Linux에서 Nix를 사용하는 경우, Nix flake, 선언적 NixOS 모듈 및 옵션 컨테이너 모드를 제공하는 전용 설치 경로가 있습니다. **[Nix & NixOS 설정](./nix-setup.md)** 가이드를 참조하세요.
:::

---

## 수동 / 개발자 설치

기여를 하거나, 특정 브랜치에서 실행하거나, 가상 환경을 완전히 제어하기 위해 저장소를 복제하고 소스에서 직접 설치하려면 기여 가이드의 [개발 환경 설정](../developer-guide/contributing.md#development-setup) 섹션을 참조하세요.

---

## 비 Sudo / 시스템 서비스 사용자 설치

Hermes를 권한이 없는 전용 사용자(예: `hermes` systemd 서비스 계정 또는 `sudo` 권한이 없는 모든 사용자)로 실행하는 것을 지원합니다. 설치 경로에서 루트 권한이 정말로 필요한 유일한 단계는 Chromium에서 사용하는 공유 라이브러리(`libnss3`, `libxkbcommon` 등)를 `apt`로 설치하는 Playwright의 `--with-deps` 단계입니다. 설치 프로그램은 sudo 사용이 가능한지 감지하고, 불가능한 경우 정상적으로 기능을 축소하여(gracefully degrade) 작동합니다. 즉, 서비스 사용자의 자체 Playwright 캐시에 Chromium 바이너리를 설치하고 관리자가 별도로 실행해야 하는 정확한 명령어를 출력합니다.

**권한 분할 권장 단계 (Debian/Ubuntu):**

1. **sudo 권한을 가진 관리자 계정으로 한 번**, Chromium에 필요한 시스템 라이브러리를 설치합니다.
   ```bash
   sudo npx playwright install-deps chromium
   ```
   (이 명령어는 어디서든 실행할 수 있습니다. `npx`가 즉석에서 Playwright를 가져와 실행합니다.)

2. **비권한 서비스 사용자로 전환하여** 일반 설치 프로그램을 실행합니다. 프로그램이 sudo 권한이 없음을 감지하여 `--with-deps` 단계를 건너뛰고 사용자의 로컬 Playwright 캐시에 Chromium을 설치합니다.
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
   ```

   브라우저 자동화가 필요 없고 백그라운드(headless)로만 실행하려는 등의 이유로 Playwright 단계를 완전히 건너뛰려면 `--skip-browser`를 전달하세요.
   ```bash
   curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash -s -- --skip-browser
   ```

3. **서비스 사용자의 셸에서 `hermes`를 사용할 수 있도록 설정합니다.** 설치 프로그램은 실행기를 `~/.local/bin/hermes`에 작성합니다. 시스템 서비스 계정은 종종 `~/.local/bin`을 포함하지 않는 최소한의 PATH를 가집니다. 사용자 환경 변수에 추가하거나 시스템 전역 경로에 실행기 심볼릭 링크를 생성하세요.
   ```bash
   # 옵션 A — 서비스 사용자의 프로필에 추가
   echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

   # 옵션 B — 시스템 전역 심볼릭 링크 생성 (관리자로 실행)
   sudo ln -s /home/hermes/.hermes/hermes-agent/venv/bin/hermes /usr/local/bin/hermes
   ```

4. **검증:** 이제 `hermes doctor`를 정상적으로 실행할 수 있어야 합니다. 만약 `ModuleNotFoundError: No module named 'dotenv'` 오류가 발생한다면, 가상환경 실행기(`~/.hermes/hermes-agent/venv/bin/hermes`) 대신 시스템 Python을 통해 저장소 소스 `hermes` 파일(`~/.hermes/hermes-agent/hermes`)을 직접 호출한 것입니다. 3단계를 수정하세요.

동일한 패턴이 Arch(설치 프로그램이 동일한 sudo 감지 논리를 사용해 pacman을 작동함), Fedora/RHEL, openSUSE에서도 적용됩니다. 해당 배포판들은 `--with-deps`를 전혀 지원하지 않으므로 관리자가 항상 시스템 라이브러리를 별도로 설치해야 합니다. 관련 `dnf`/`zypper` 명령어는 설치 프로그램에 의해 출력됩니다.

---

## 문제 해결

| 문제 | 해결 방법 |
|---------|----------|
| `hermes: command not found` | 셸을 다시 로드(`source ~/.bashrc`)하거나 PATH 설정을 확인하세요. |
| `API key not set` | `hermes model`을 실행하여 프로바이더를 설정하거나, `hermes config set OPENROUTER_API_KEY your_key`를 실행하세요. |
| 업데이트 후 설정 유실 | `hermes config check`를 실행한 후 `hermes config migrate`를 실행하세요. |

더 많은 진단이 필요하면 `hermes doctor`를 실행하세요. 무엇이 누락되었고 어떻게 수정해야 하는지 정확히 알려줍니다.

## 설치 방법 자동 감지

Hermes는 `pip`, git 설치 프로그램, Homebrew 또는 NixOS 중 어떤 것을 통해 설치되었는지 자동으로 감지하며, `hermes update`를 실행하면 해당 설치 경로에 맞는 업데이트 명령어를 출력합니다. 별도로 설정해야 하는 환경 변수는 없으며, 감지는 설치 레이아웃(Python site-packages, `~/.hermes/hermes-agent/`, Homebrew 접두사 또는 Nix store 경로)을 기반으로 수행됩니다. `hermes doctor`도 환경 요약에 감지된 설치 방법을 표시합니다.

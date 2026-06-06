---
sidebar_position: 3
title: "데스크톱 앱"
description: "기본(native) Hermes 데스크톱 앱 — 스트리밍 도구 출력, 나란히 보는 미리보기, 파일 브라우저, 음성, 크론, 프로필, 스킬 및 설정을 갖춘 세련된 Hermes 채팅 경험. macOS, Windows 및 Linux 지원."
---

# 데스크톱 앱 (Desktop App)

Hermes 데스크톱 앱은 CLI 및 게이트웨이에서 사용하는 것과 **동일한** 에이전트를 기반으로 구축된 네이티브 앱입니다 — 동일한 구성, 동일한 API 키, 동일한 세션, 동일한 스킬, 동일한 기억(memory)을 사용합니다. 별도의 제품이거나 가벼운 클론이 아닙니다. 동일한 Hermes Agent 코어 및 설정을 사용하며, 현대적이고 세심하게 디자인된 UI를 통해 이를 구동합니다. 터미널에서 `hermes`를 사용해 본 적이 있다면, 그곳에서 설정한 모든 것이 이미 이곳에 있으며, 여기서 한 모든 작업이 그곳에도 나타납니다.

**macOS, Windows, Linux**에서 실행됩니다.

:::tip 어떤 인터페이스인가요?
Hermes에는 모두 동일한 에이전트와 통신하는 여러 프론트엔드가 있습니다:

- **데스크톱 앱** (현재 문서) — 채팅, 구성 및 관리를 위한 목적에 맞게 구축된 UI가 있는 네이티브 애플리케이션입니다.
- **CLI** (`hermes`) 및 **[TUI](./tui.md)** (`hermes --tui`) — 터미널 인터페이스입니다.
- **[웹 대시보드](./features/web-dashboard.md)** (`hermes dashboard`) — 브라우저 관리자 패널입니다. 선택 사항인 **Chat** 탭은 가상 터미널(pseudo-terminal)을 통해 TUI를 포함합니다.

상황에 맞는 것을 선택하세요. 상태를 공유하므로 한 곳에서 세션을 시작하고 다른 곳에서 다시 시작할 수 있습니다.
:::

## 설치 (Install)

[Hermes Desktop 설치 지침](../getting-started/installation.md)을 따르세요.

이미 Hermes가 설치되어 있는 경우, 다음을 실행하기만 하면 됩니다:

```bash
hermes desktop
```

그러면 현재 구성, 키, 세션 및 스킬이 사용됩니다.

## 앱의 구성 요소 (What's in the app)

데스크톱 앱은 채팅 우선 창과 탐색을 위한 왼쪽 사이드바로 구성되어 있습니다. 동시에 여러 에이전트 대화를 관리하고, 메시징 제공자를 구성하며, 아티팩트를 생성하고, 프로젝트의 폴더 구조를 탐색하며, 한 번에 여러 프로젝트에서 작업할 수 있도록 구축되었습니다.

### 채팅 (Chat)

앱의 중심입니다. 다음을 제공합니다:

- 에이전트가 작업하는 동안 라이브 도구 활동 및 구조화된 도구 호출 요약이 포함된 **스트리밍 응답**.
- 다른 모든 Hermes 표면과 **동일한 대화 기록** — 여기서 시작된 세션은 CLI/TUI에서 다시 시작할 수 있으며 그 반대의 경우도 마찬가지입니다.
- 다음 메시지에 첨부하기 위해 채팅 영역 아무 곳에나 파일을 **드래그 앤 드롭**.
- **우측 미리보기 레일** — 채팅을 계속하는 동안 웹 페이지, 파일 및 도구 출력을 나란히 렌더링합니다.

번들로 제공되는 로컬 백엔드가 아닌 다른 머신의 Hermes 인스턴스와 채팅 중이신가요? 아래의 [원격 백엔드에 연결하기](#connecting-to-a-remote-backend)를 참조하세요. — 원격 호스팅된 대시보드 연결의 작동 방식(인증 게이트, `/api/ws` 채팅 소켓 및 WebSocket 닫기 코드 분류)에 대한 전체 내용을 보려면 [웹 대시보드 → 원격 백엔드에 Hermes Desktop 연결](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)을 참조하세요.

### 파일 브라우저 (File browser)

앱을 떠나지 않고 작업 디렉토리를 탐색하고 미리 볼 수 있습니다 — 에이전트가 파일을 읽고, 쓰고, 편집하는 것을 따라가는 데 유용합니다. `hermes desktop --cwd <경로>` (또는 `HERMES_DESKTOP_CWD` 환경 변수)를 사용하여 초기 프로젝트 디렉토리를 설정할 수 있습니다.

### 음성 (Voice)

Hermes에게 말하고 대답을 들으세요. 다른 곳에서 사용할 수 있는 [음성 모드](./features/voice-mode.md)와 동일합니다. macOS에서는 OS가 마이크 접근 권한을 한 번 묻습니다.

### 설정 및 온보딩 (Settings & onboarding)

YAML을 편집하는 대신 실제 UI에서 제공자, 모델, 도구 및 자격 증명을 관리하세요. 첫 실행 온보딩을 통해 몇 초 만에 첫 번째 메시지를 보낼 수 있습니다. 설정 창은 제공자/키, 모델 선택, 도구 세트 구성, MCP 서버, 게이트웨이 및 세션 관리를 다룹니다.

### 관리 창 (Management panes)

또한 터미널로 떨어지지 않아도 되도록 더 넓은 범위의 Hermes 관리 표면을 제공합니다:

- **Skills** — [스킬(skills)](./features/skills.md)을 탐색, 설치 및 관리합니다.
- **Cron** — [예약된 작업(scheduled jobs)](../reference/cli-commands.md#hermes-cron)을 보고 관리합니다.
- **Profiles** — [Hermes 프로필](./profiles.md) (격리된 구성/스킬/세션) 간에 전환합니다.
- **Messaging** — 게이트웨이 채널을 설정합니다.
- **Agents** 및 **Command Center** — 다중 에이전트 작업을 위한 오케스트레이션 표면입니다.

## 업데이트 (Updating)

앱은 백그라운드에서 업데이트를 확인하고 준비가 되면 클릭 한 번으로 업데이트할 수 있는 옵션을 제공합니다.

[수동 업데이트 프로세스](https://hermes-agent.nousresearch.com/docs/getting-started/updating) 역시 GUI에서 작동합니다.

## CLI 참조: `hermes desktop`

CLI를 통해 시작하려면 단순히 `hermes desktop`을 실행하세요. 기본적으로 워크스페이스 Node 의존성을 설치하고, 현재 OS의 압축되지 않은 Electron 앱을 빌드한 다음, 해당 패키지 아티팩트를 실행합니다.

| 플래그 | 설명 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| `--skip-build` | npm 설치/패키지를 건너뛰고 `apps/desktop/release`에서 기존의 압축되지 않은 앱을 실행합니다 |
| `--force-build` | 콘텐츠 스탬프가 일치하더라도 전체 재빌드를 강제합니다 |
| `--build-only` | 데스크톱 앱을 빌드하지만 실행하지는 않습니다 (`hermes update`에서 사용됨) |
| `--source` | 패키지된 앱 대신 `apps/desktop/dist`에 대해 `electron .`를 통해 실행합니다 |
| `--cwd PATH` | 데스크톱 채팅 세션의 초기 프로젝트 디렉토리 (`HERMES_DESKTOP_CWD` 설정) |
| `--hermes-root PATH` | 앱이 사용하는 Hermes 소스 루트를 재정의합니다 (`HERMES_DESKTOP_HERMES_ROOT` 설정) |
| `--ignore-existing` | 백엔드 확인 중에 이미 `PATH`에 있는 `hermes` CLI를 무시하도록 앱에 강제합니다 |
| `--fake-boot` | 시작 UI의 유효성을 검사하기 위해 결정론적 부팅 지연을 활성화합니다 |

## 작동 방식 (How it works)

패키지된 앱은 Electron 쉘만 포함하여 제공됩니다. 첫 실행 시 Hermes Agent 런타임을 `HERMES_HOME` (`~/.hermes`, 또는 Windows의 경우 `%LOCALAPPDATA%\hermes`)에 설치합니다 — **CLI 설치에서 사용하는 것과 동일한 레이아웃**이며, 그렇기 때문에 이 둘은 상호 교환이 가능합니다. React 렌더러는 표준 게이트웨이 API를 통해 `hermes dashboard` 백엔드와 대화하며, 에이전트를 다시 구현하는 대신 재사용합니다. 설치, 백엔드 확인 및 자체 업데이트 로직은 Electron 메인 프로세스에 존재합니다.

## 원격 백엔드에 연결하기 (Connecting to a remote backend)

기본적으로 앱은 자체 **로컬** 백엔드를 시작하고 관리합니다. 대신 VPS, 홈 서버 또는 Tailscale 뒤에 있는 Mini 등 다른 기기에서 실행 중인 Hermes 백엔드를 가리키도록 할 수도 있습니다.

:::info 원격 백엔드는 실행 중인 `hermes dashboard` 프로세스입니다
"원격 백엔드"란 원격 머신에서 실행 중인 **`hermes dashboard`** 서버를 의미합니다 — 이 프로세스에 데스크톱 앱이 연결됩니다. 이 섹션의 내용은 해당 대시보드가 실제로 실행 중이고 도달할 수 없다면 작동하지 않습니다. 데스크톱 앱이 이를 대신 시작해 주지 않습니다. 사용자(또는 `systemd` 서비스)가 원격 호스트에서 `hermes dashboard`를 계속 실행 상태로 유지하고, 앱이 여기에 접속하는 것입니다. 메시징 채널(Telegram, Discord 등)도 사용하는 경우, **게이트웨이**는 독립적으로 시작하는 *별도의* 장기 실행 프로세스입니다 — 설정 단계 후의 참고 사항을 확인하세요.
:::

연결은 두 부분으로 나뉩니다: 백엔드에서는 **인증 제공자(auth provider)** 로 대시보드를 보호하고, 앱에서는 백엔드의 URL을 입력하고 로그인합니다. 대시보드를 루프백 주소가 아닌 주소에 바인딩하면 인증 게이트가 자동으로 작동하며, 여기서 구성한 제공자가 데스크톱 앱의 통과를 허용합니다.

**백엔드가 위치한 곳에 따라 제공자를 선택하세요:**

- **OAuth (Nous Portal) — 자신의 머신을 넘어 접속할 수 있는 모든 경우에 선호됩니다.** 로그인이 Nous 계정에 대해 확인되므로 VPS, 공개 호스트 또는 모든 원격 백엔드에 적합한 옵션입니다. `hermes dashboard register` (또는 Portal의 [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) 페이지)를 사용하여 대시보드를 등록하여 OAuth 클라이언트를 프로비저닝한 다음, 앱에서 **Sign in with Nous Research** 로 로그인하세요. 자체 자격 증명 제공자(identity provider)를 운영하는 경우 자체 호스팅 OIDC 제공자도 같은 방식으로 작동합니다.
- **사용자 이름/비밀번호 — 로컬 / 신뢰할 수 있는 네트워크에서만 사용.** 백엔드가 동일하게 신뢰할 수 있는 LAN에 있거나 VPN(예: Tailscale)을 통해서만 접속할 수 있을 때 가장 간단한 옵션입니다. 외부 자격 증명 제공자 없이 단일 공유 자격 증명을 보호하므로, **공용 인터넷에 노출된 대시보드에는 사용하지 마세요** — 대신 OAuth를 사용하세요.

이 섹션의 나머지 부분에서는 신뢰할 수 있는 네트워크에서 가장 빠르게 구축할 수 있는 사용자 이름/비밀번호 경로를 보여줍니다. OAuth 경로의 경우 [웹 대시보드 → 기본 제공자: Nous Research](./features/web-dashboard.md#default-provider-nous-research)를 참조하세요.

### 백엔드 측 (원격 머신)

사용자 이름과 비밀번호를 설정한 다음, 접근 가능한 주소에 바인딩하여 대시보드를 시작합니다. 자격 증명은 `~/.hermes/.env` (비밀키 파일, 권한 0600)에 보관됩니다:

```bash
# 1. 대시보드 로그인 자격 증명을 설정합니다.
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# 권장: 세션이 재시작 후에도 유지되도록 안정적인 서명용 비밀키.
# 이 값이 없으면 부팅 시마다 무작위 키가 생성되어 재시작할 때마다 로그아웃됩니다.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. 도달 가능한 주소에 바인딩된 대시보드를 실행합니다. 루프백이 아닌
#    바인딩은 인증 게이트를 활성화하며, 사용자 이름/비밀번호 제공자가 로그인을 처리합니다.
hermes dashboard --no-open --host 0.0.0.0 --port 9119
```

데스크톱 앱이 연결할 수 있기를 원하는 한 이 `hermes dashboard` 프로세스를 계속 실행 상태로 유지하세요 — 이 프로세스가 중지되면 앱이 더 이상 백엔드에 연결할 수 없습니다. `systemd`, `tmux` 또는 선호하는 프로세스 관리자 하에서 실행하여 로그아웃과 재부팅 후에도 유지되도록 하세요.

메시징 채널을 이용한다면 이와 별도로 원격 호스트에서 **게이트웨이가 실행 중인지** 확인하세요 — 데스크톱 앱이 대화하는 대상은 대시보드 백엔드이지만, Telegram/Discord/Slack 게이트웨이 세션은 별도로 시작하고 유지해야 하는 다른 프로세스입니다. 게이트웨이 설정은 [메시징(Messaging)](./messaging/index.md)을 참조하세요.

평문 비밀번호를 보관하고 싶지 않으신가요? `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH`를 scrypt 해시로 설정하세요 — `python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"`로 해시를 계산할 수 있습니다. 전체 구성 표면(config.yaml 키, 모든 환경 변수, 속도 제한기): [웹 대시보드 → 사용자 이름/비밀번호 제공자](./features/web-dashboard.md#usernamepassword-provider-no-oauth-idp).

대시보드를 systemd 서비스로 실행하시나요? 유닛에 `EnvironmentFile=%h/.hermes/.env`를 추가하여 부팅 시 환경에 자격 증명이 포함되도록 하세요.

:::warning
대시보드는 `.env` (API 키, 비밀키)를 읽고 쓰며 에이전트 명령을 실행할 수 있습니다. 위에서 보여준 **사용자 이름/비밀번호** 설정은 신뢰할 수 있는 네트워크용입니다 — 비밀번호로 보호된 대시보드를 공개 인터넷에 직접 노출하지 마세요; VPN 뒤에 배치하세요. [Tailscale](https://tailscale.com/)이 깔끔한 옵션입니다: 머신의 tailscale IP(`--host <tailscale-ip>`)에 바인딩하고 원격 URL로 `http://<tailscale-ip>:9119`를 사용하여 tailnet만 접근할 수 있게 하세요. 공용 인터넷을 통해 백엔드에 연결하려면 **OAuth (Nous Portal)** 제공자를 사용하세요.
:::

### 앱 측

**Settings(설정) → Gateway(게이트웨이) → Remote gateway(원격 게이트웨이):**

1. **Remote URL(원격 URL)** — `http://<backend-host>:9119` (리버스 프록시를 사용하는 경우 `/hermes`와 같은 경로 접두사도 작동함)
2. **Sign in(로그인)** — 앱은 백엔드가 알리는 제공자가 무엇인지 감지하여 버튼을 알맞게 조정합니다. 사용자 이름/비밀번호 백엔드의 경우 자격 증명 양식을 여는 **Sign in** 버튼을 표시합니다(1단계의 자격 증명 입력). OAuth 백엔드의 경우 제공자의 브라우저 로그인을 실행하는 **Sign in with `<provider>`** (예: *Sign in with Nous Research*)를 표시합니다. 어느 쪽이든 앱은 결국 백엔드에 대해 인증된 세션을 갖게 됩니다.
3. **Save and reconnect(저장 및 다시 연결)** — 데스크톱 쉘을 원격 백엔드로 전환합니다. 세션은 자동으로 새로 고쳐집니다. `HERMES_DASHBOARD_BASIC_AUTH_SECRET`이 설정된 경우 재시작 후에도 로그인 상태가 유지됩니다.

앱을 시작하기 전에 `HERMES_DESKTOP_REMOTE_URL` 환경 변수를 사용하여 UI 없이 백엔드 URL을 설정할 수도 있습니다(앱 내 설정을 재정의함). 하지만 로그인 과정은 Gateway 설정 패널에서 여전히 진행해야 합니다.

### 문제 해결 (Troubleshooting)

- **로그인이 401 / "Invalid credentials" 오류로 실패함** — 사용자 이름 또는 비밀번호가 백엔드의 `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`와 일치하지 않습니다. 백엔드는 알 수 없는 사용자와 잘못된 비밀번호 모두에 대해 동일한 일반적인 오류를 반환하므로(열거 공격 방지), 둘 다 다시 확인하세요. `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'` 명령을 통해 게이트가 켜져 있는지 확인하세요 — `true`를 보고하고 `"basic"`을 포함해야 합니다.
- **"Sign in" 버튼이 없음 — 대신 세션 토큰을 요구함** — 백엔드의 사용자 이름/비밀번호 제공자가 활성화되지 않았습니다. `/api/status`의 `auth_providers`에 `"basic"`이 나열되지 않습니다. `~/.hermes/.env`에 사용자 이름과 비밀번호(또는 비밀번호 해시)가 모두 설정되어 있고 대시보드 프로세스가 이를 실제로 로드했는지 확인하세요.
- **재시작할 때마다 로그아웃됨** — `HERMES_DASHBOARD_BASIC_AUTH_SECRET`을 안정적인 값으로 설정하세요. 이 값이 없으면 토큰 서명 키가 부팅될 때마다 재생성되어 모든 세션이 무효화됩니다.
- **연결 거부됨 / 시간 초과(Connection refused / times out)** — 백엔드가 `127.0.0.1`(기본값)에 바인딩되어 있거나 방화벽/VPN이 포트를 차단하고 있습니다. `0.0.0.0` 또는 tailscale IP에 바인딩하고 신뢰할 수 있는 네트워크에 포트를 개방하세요.

웹 대시보드 관점에서의 동일한 설정에 대해서는 [웹 대시보드 → 원격 백엔드에 Hermes Desktop 연결](./features/web-dashboard.md#connecting-hermes-desktop-to-a-remote-backend)을 참조하세요. 환경 변수는 [환경 변수 → 웹 대시보드 및 Hermes Desktop](../reference/environment-variables.md#web-dashboard--hermes-desktop)에 분류되어 있습니다.

## 문제 해결 (Troubleshooting)

부팅 로그는 `HERMES_HOME/logs/desktop.log` (백엔드 출력 및 최근 Python 트레이스백 포함)에 기록됩니다. — 앱에서 부팅 실패가 보고되면 먼저 이를 확인하세요. CLI에서 tail 명령어를 통해 확인할 수도 있습니다:

```bash
hermes logs gui -f
```

일반적인 초기화 방법:

```bash
# 깨끗한 첫 실행 설정 강제 (macOS/Linux)
rm "$HOME/.hermes/hermes-agent/.hermes-bootstrap-complete"

# 깨진 Python venv 재빌드 (macOS/Linux)
rm -rf "$HOME/.hermes/hermes-agent/venv"

# 멈춰버린 macOS 마이크 프롬프트 초기화
tccutil reset Microphone com.nousresearch.hermes
```

## 소스에서 빌드하기 (Building from source)

앱 자체를 해킹하고 싶다면, 저장소 루트에서 워크스페이스 의존성을 한 번 설치한 다음 `apps/desktop`에서 개발 서버를 실행하세요:

```bash
npm install          # 저장소 루트에서 — apps/desktop, web, apps/shared 링크 연결
cd apps/desktop
npm run dev          # Vite 렌더러 + Electron (Python 백엔드를 부팅함)
```

앱이 특정 체크아웃을 가리키게 하거나 실제 구성과 격리된 샌드박스로 실행하세요:

```bash
HERMES_DESKTOP_HERMES_ROOT=/path/to/clone npm run dev
HERMES_HOME=/tmp/throwaway npm run dev
npm run dev:fake-boot   # 결정론적 지연 시간으로 시작 오버레이를 테스트합니다
```

설치 관리자 빌드:

```bash
npm run dist:mac     # DMG + zip
npm run dist:win     # NSIS + MSI
npm run dist:linux   # AppImage + deb + rpm
npm run pack         # release/ 하위에 압축되지 않은 앱 빌드 (설치 관리자 없음)
```

환경에 관련 자격 증명이 존재하는 경우(`CSC_LINK` / `CSC_KEY_PASSWORD` / `APPLE_*`는 macOS, `WIN_CSC_*`는 Windows) macOS/Windows 서명 및 공증이 자동으로 실행됩니다.

## 같이 보기 (See also)

- [CLI 가이드](./cli.md) — 터미널 인터페이스
- [TUI](./tui.md) — 데스크톱 백엔드에서 재사용되는 현대적인 터미널 UI
- [웹 대시보드](./features/web-dashboard.md) — 채팅 탭이 포함된 브라우저 관리자 패널
- [구성](./configuration.md) — 데스크톱 앱이 읽고 쓰는 구성 설정
- [Windows (네이티브)](./windows-native.md) — 네이티브 Windows 설치 가이드

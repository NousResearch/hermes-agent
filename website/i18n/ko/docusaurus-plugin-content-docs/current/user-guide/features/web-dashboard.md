---
sidebar_position: 15
title: "웹 대시보드"
description: "구성, API 키, MCP 서버, 메시징 페어링, 웹훅, 게이트웨이, 메모리, 자격 증명, 세션, 로그, 분석, cron 작업 및 스킬을 관리하기 위한 브라우저 기반 관리 패널"
---

# 웹 대시보드

웹 대시보드는 Hermes Agent 설치를 관리하기 위한 브라우저 기반 UI입니다. YAML 파일을 직접 편집하거나 CLI 명령을 실행하는 대신, 깔끔한 웹 인터페이스에서 설정을 구성하고 API 키를 관리하며 세션을 모니터링할 수 있습니다.

:::tip
호스팅 모드 인증은 Nous Portal OAuth를 사용합니다. 대시보드가 실제 백엔드와도 통신하도록 하려면 `hermes setup --portal`이 모델과 도구 게이트웨이도 설정합니다. [Nous Portal](/integrations/nous-portal)을 참고하세요.
:::

## 빠른 시작

```bash
hermes dashboard
```

이 명령은 로컬 웹 서버를 시작하고 브라우저에서 `http://127.0.0.1:9119`을 엽니다. 대시보드는 전적으로 컴퓨터에서 실행되며, localhost 밖으로 데이터가 전송되지 않습니다.

### 옵션

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--port` | `9119` | 웹 서버를 실행할 포트 |
| `--host` | `127.0.0.1` | 바인딩 주소 |
| `--no-open` | — | 브라우저를 자동으로 열지 않음 |
| `--insecure` | 꺼짐 | **지원 중단됨 / 동작하지 않음.** 이전에는 loopback이 아닌 바인딩에서 인증을 우회했지만, 더 이상 인증을 비활성화하지 않습니다. 공개 바인딩에는 항상 인증 제공자(비밀번호 또는 OAuth)가 필요합니다 |
| `--isolated` | 꺼짐 | 이름이 지정된 프로필(`worker dashboard`)에서 시작할 때 머신 대시보드로 라우팅하지 않고 프로필별 전용 서버를 실행 |

```bash
# Custom port
hermes dashboard --port 8080

# Bind to all interfaces (use with caution on shared networks)
hermes dashboard --host 0.0.0.0

# Start without opening browser
hermes dashboard --no-open
```

## 여러 프로필 관리

대시보드는 **머신 수준** 관리 화면입니다. 하나의 서버가 해당 머신의 모든 [프로필](../profiles.md)을 관리합니다. 사이드바의 프로필 전환기(프로필이 둘 이상 있을 때 표시됨)에서 관리 페이지가 읽고 쓸 프로필을 결정하며, Config, API Keys, Skills, MCP, Models 및 Chat 탭이 모두 이를 따릅니다. 대시보드 자체가 아닌 다른 프로필을 선택하면 관리 대상 프로필을 표시하는 황색 배너가 나타나므로 쓰기 대상이 모호해지지 않습니다.

선택 항목은 URL(`?profile=<name>`)에 저장됩니다. 따라서 `http://127.0.0.1:9119/skills?profile=worker` 같은 딥 링크를 열면 전환기가 미리 선택된 상태로 표시되고 새로 고침 후에도 유지됩니다.

프로필 별칭에서 대시보드를 실행하면 두 번째 서버를 시작하는 대신 머신 대시보드로 라우팅됩니다.

```bash
worker dashboard
# → already running: opens the browser at ?profile=worker
# → not running:     starts the machine dashboard with "worker" preselected
```

`--isolated`를 전달하면 이 동작을 사용하지 않고 해당 프로필 범위의 전용 서버를 실행할 수 있습니다(통합 전 동작 방식이며, 서로 다른 프로필의 대시보드를 서로 다른 인증으로 의도적으로 공개할 때 유용합니다).

**Chat** 탭도 전환기를 따릅니다. 범위가 지정된 채팅은 선택한 프로필의 `HERMES_HOME`으로 PTY 자식 프로세스를 실행하므로, 대화가 해당 프로필의 모델, 스킬, 메모리 및 세션 기록과 함께 실행됩니다. 프로필을 전환하면 새 터미널 세션이 시작됩니다.

전환기에 흡수되지 않고 프로필별로 유지되는 항목은 다음과 같습니다. 게이트웨이 프로세스(`hermes -p <name> gateway …`로 관리), 각 프로필의 세션 데이터베이스, cron 스케줄러(Cron 페이지는 자체 필터로 이미 여러 프로필을 집계함)입니다.

## 사전 요구 사항

기본 `hermes-agent` 설치에는 HTTP 스택이나 PTY 도우미가 포함되지 않으며, 이들은 선택적 추가 기능입니다. **웹 대시보드**에는 FastAPI와 Uvicorn(`web` extra)이 필요합니다. **Chat** 탭에는 의사 터미널 뒤에 포함된 TUI를 실행할 `ptyprocess`도 필요합니다(POSIX에서는 `pty` extra). 다음 명령으로 둘 다 설치하세요.

```bash
cd ~/.hermes/hermes-agent && uv pip install -e ".[web,pty]"
```

`web` extra는 FastAPI/Uvicorn을 설치하고, `pty`는 `ptyprocess`(POSIX) 또는 `pywinpty`(네이티브 Windows — 포함된 TUI 자체는 여전히 WSL이 필요함)를 설치합니다. `cd ~/.hermes/hermes-agent && uv pip install -e "[all]"`은 두 extra를 모두 포함하므로 메시징/음성 등도 사용하려면 가장 간단한 방법입니다.

필수 의존성이 없는 상태에서 `hermes dashboard`를 실행하면 설치해야 할 항목을 안내합니다. 프런트엔드가 아직 빌드되지 않았고 `npm`을 사용할 수 있다면 첫 실행 시 자동으로 빌드됩니다.

Chat 탭은 모든 `hermes dashboard` 실행에 포함됩니다. PTY/WebSocket을 통해 TUI를 실행하는 포함된 브라우저 채팅 창은 별도의 플래그 없이 항상 사용할 수 있습니다.

## 페이지

### 상태

랜딩 페이지에는 설치 상태에 대한 실시간 개요가 표시됩니다.

- **에이전트 버전** 및 출시일
- **게이트웨이 상태** — 실행 중/중지됨, PID, 연결된 플랫폼 및 각 플랫폼의 상태
- **활성 세션** — 최근 5분 내 활성화된 세션 수
- **최근 세션** — 모델, 메시지 수, 토큰 사용량 및 대화 미리보기가 포함된 최근 세션 20개의 목록

상태 페이지는 5초마다 자동으로 새로 고쳐집니다.

#### 리소스 부족 배너

호스트의 메모리나 디스크가 부족해지면 대시보드 상단에 배너가 표시됩니다(추가 요청 없이 동일한 상태 폴링으로 제공됨).

- **"에이전트의 메모리가 거의 부족하여 재시작될 수 있습니다"** — 게이트웨이의 30초 하트비트가 샘플링한 시스템 가용 메모리가 *상승* 수준(< 128 MiB 또는 < 15%) 또는 *위험* 수준(< 64 MiB 또는 < 5%)으로 감소한 경우
- **"에이전트가 예기치 않게 재시작되었습니다. 메모리 부족으로 인한 것일 가능성이 높습니다"** — 이전 부팅에서 메모리 압박 중 비정상 종료가 발생한 사실이 수명 주기 원장에 기록된 경우(의심되는 OOM kill)
- **디스크 경고** — `~/.hermes`가 위치한 볼륨이 거의 가득 찬 경우(*상승*: 여유 공간 512MB 미만, *위험*: 256MB 미만)

한 번에 가장 심각한 활성 경고만 표시됩니다(디스크 위험 > 메모리 위험 > OOM 재시작 > 디스크 상승 > 메모리 상승). 닫기 처리는 현재 게이트웨이 부팅에 한정됩니다. 경고를 닫으면 다음 활성 경고가 표시되고, 게이트웨이를 재시작하거나 단계가 상승하면(상승 → 위험) 다시 열리며, 오래된 하트비트에서는 잘못된 경고 대신 아무것도 표시되지 않습니다.

### Chat

**Chat** 탭은 전체 Hermes TUI(`hermes --tui`로 실행할 때와 동일한 인터페이스)를 브라우저에 직접 포함합니다. 대시보드가 실제 TUI 바이너리를 실행하고 그 ANSI 출력을 [xterm.js](https://xtermjs.org/)의 WebGL 렌더러로 표시하므로, 터미널 TUI에서 할 수 있는 모든 작업(슬래시 명령, 모델 선택기, 도구 호출 카드, Markdown 스트리밍, 명확화/ sudo/승인 프롬프트, 스킨 테마)을 여기서도 동일하게 사용할 수 있습니다.

**작동 방식:**

- `/api/pty`가 대시보드의 세션 토큰으로 인증된 WebSocket을 엽니다.
- 서버가 POSIX 의사 터미널 뒤에서 `hermes --tui`를 실행합니다.
- 키 입력은 PTY로 전달되고, ANSI 출력은 브라우저로 스트리밍됩니다.
- xterm.js의 WebGL 렌더러가 각 셀을 정수 픽셀 격자에 그립니다. 마우스 추적(SGR 1006), 넓은 문자(Unicode 11), 상자 그리기 글리프가 모두 네이티브로 렌더링됩니다.
- 브라우저 창 크기를 조정하면 `@xterm/addon-fit` addon을 통해 TUI 크기도 조정됩니다.

**기존 세션 재개:** **Sessions** 탭에서 세션 옆의 재생 아이콘(▶)을 클릭하세요. 그러면 `/chat?resume=<id>`로 이동하고 `--resume`과 함께 TUI를 실행하여 전체 기록을 불러옵니다.

**세션 전환기(오른쪽 레일):** Chat 탭에는 터미널 옆의 좁은 오른쪽 레일에 ChatGPT 스타일의 자체 대화 목록이 있어 페이지를 벗어나지 않고 대화를 전환할 수 있습니다. 레일의 위쪽에는 모델 선택기가, 바로 아래에는 세션 목록이 배치되며 터미널이 화면 대부분을 차지합니다. 목록에는 활성 프로필의 최근 세션이 표시됩니다. 제목(없으면 메시지 미리보기로 대체), 상대적인 마지막 활성화 시간, 메시지 수, CLI가 아닌 세션의 소스 채널이 포함됩니다. 행을 클릭하면 해당 위치에서 세션이 재개되고(터미널이 해당 대화의 기록과 함께 다시 생성됨) 활성 세션이 강조 표시됩니다. **새 채팅**은 새 세션을 시작하고, 새로 고침 컨트롤은 목록을 다시 불러옵니다. 레일은 전환만 지원하며 삭제, 이름 변경, 내보내기 및 일괄 정리는 여전히 **Sessions** 탭에서 수행합니다. 좁은 화면에서는 슬라이드 오버 패널로 접힙니다.

**사전 요구 사항:**

- Node.js(`hermes --tui`와 동일한 요구 사항이며, 첫 실행 시 TUI 번들이 빌드됨)
- `ptyprocess` — `pty` extra로 설치(`cd ~/.hermes/hermes-agent && uv pip install -e "[web,pty]"`, 또는 `[all]`로 둘 다 설치 가능)
- POSIX 커널(Linux, macOS 또는 WSL2). `/chat` 터미널 창에는 특히 POSIX PTY가 필요합니다. 네이티브 Windows Python에는 이에 상응하는 기능이 없으므로, 네이티브 Windows 설치에서는 대시보드의 나머지 기능(세션, 작업, 메트릭, 구성 편집기)은 작동하지만 `/chat` 탭에는 이 기능을 사용하려면 WSL2를 사용하라는 배너가 표시됩니다.

브라우저 탭을 닫으면 서버에서 PTY가 정상적으로 정리됩니다. 다시 열면 새 세션이 생성됩니다.

[Hermes Desktop](#connecting-hermes-desktop-to-a-remote-backend)을 자체 번들 백엔드 대신 다른 머신에서 실행 중인 대시보드에 연결하려면 아래 원격 백엔드 섹션을 참고하세요.

### Hermes Desktop을 원격 백엔드에 연결

Hermes Desktop은 일반적으로 자체 로컬 백엔드를 실행하지만, **Settings → Gateway → Remote gateway**를 통해 원격 머신(가상 머신, 홈랩 장비 등)에서 실행 중인 대시보드에 연결할 수도 있습니다. 이는 "Desktop에는 백엔드가 준비되었다고 나오지만 채팅이 작동하지 않는다"는 보고가 발생하는 가장 흔한 원인입니다. Desktop의 준비 상태 확인은 실제 채팅 연결에 필요한 것보다 적은 항목만 확인하기 때문입니다.

:::info 사전 요구 사항: 원격 호스트에서 `hermes dashboard`가 실행 중이어야 합니다
Desktop이 연결하는 "원격 백엔드"는 원격 머신에서 실행 중인 `hermes dashboard` 프로세스, 즉 이 페이지에서 설명하는 동일한 서버입니다. 아래 단계가 의미를 가지려면 이 서버가 실행 중이고 연결 가능해야 합니다. Desktop이 서버를 시작해 주는 것이 아니므로, 로그아웃과 재부팅 후에도 유지되도록 `systemd`/`tmux`/기타 방식으로 실행하세요. **게이트웨이**(Telegram/Discord/Slack 등)는 별도의 장기 실행 프로세스입니다. 메시징 채널을 사용한다면 독립적으로 시작해야 하며, 데스크톱 앱이 연결하는 대상이 아닙니다.
:::

Desktop의 "remote backend is ready" 프로브는 공개 엔드포인트인 `GET /api/status`만 호출하므로 호스트에서 대시보드가 실행 중이기만 하면 응답합니다. 실제 채팅 연결은 `/api/ws`(및 `/api/pty`)에 대한 별도의 **WebSocket**이며, 상태 프로브가 확인하지 않는 두 가지 검사를 추가로 통과해야 합니다.

1. **인증되어 있어야 합니다.** 대시보드가 loopback이 아닌 주소에 바인딩되면 인증 게이트가 활성화됩니다. 사용자 이름과 비밀번호로 보호하세요(번들로 제공되는 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp)). Desktop은 한 번 로그인한 뒤 일회용 티켓을 통해 WebSocket에 사용할 세션을 재사용합니다. 구성된 제공자가 없으면 loopback이 아닌 대시보드는 **시작 시 연결을 거부합니다**.
2. **바인딩 호스트가 클라이언트를 허용하고 Host 헤더와 일치해야 합니다.** loopback 바인딩(`127.0.0.1`)은 loopback 클라이언트만 허용하므로 자격 증명이 있어도 원격 머신은 소켓 계층에서 거부됩니다. 피어 IP 가드가 원격 클라이언트를 통과시키도록 loopback이 아닌 주소(`--host 0.0.0.0`)에 바인딩하세요. Desktop에 입력하는 원격 URL은 대시보드가 바인딩된 동일한 호스트를 통해 대시보드에 연결되어야 합니다. DNS 재바인딩 가드가 Host 헤더의 일치를 요구하기 때문입니다.

#### 원격 대시보드 설정

사용자 이름과 비밀번호를 설정한 다음 연결 가능한 주소에 바인딩하여 대시보드를 실행하세요. `systemd` 서비스의 경우:

```ini
[Service]
EnvironmentFile=%h/.hermes/.env
ExecStart=/path/to/venv/bin/python -m hermes_cli.main dashboard \
    --host 0.0.0.0 --port 9119 --no-open
```

`~/.hermes/.env`의 내용은 다음과 같습니다.

```bash
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
HERMES_DASHBOARD_BASIC_AUTH_SECRET=<32+ random bytes; openssl rand -base64 32>
```

그런 다음 Desktop에서 **Remote URL**(예: `http://VM_IP:9119`)을 입력하고 해당 사용자 이름과 비밀번호로 **Sign in**하세요. 전체 구성 항목은 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp) 섹션을 참고하세요.

:::tip Desktop을 다시 시도하기 전에 게이트가 켜져 있는지 확인하세요
어떤 머신에서든 다음 명령으로 대시보드가 사용자 이름/비밀번호 제공자를 알리는지 확인할 수 있습니다.

```bash
curl -s http://VM_IP:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

- `auth_required: true`이고 제공자 목록에 `"basic"`이 있음 → Desktop의 **Sign in** 흐름이 작동합니다.
- `auth_required: false` → 바인딩이 loopback이거나 게이트가 활성화되지 않은 것입니다. loopback이 아닌 주소에 바인딩하세요.
- `auth_required: true`이지만 `"basic"` 제공자가 없음 → 사용자 이름/비밀번호 환경 변수가 로드되지 않은 것입니다. 먼저 이를 수정하세요.
:::

`/api/status`에 `"basic"` 제공자와 함께 게이트가 켜진 것으로 표시되는데도 로그인 후 Desktop이 계속 연결되지 않는다면, 문제는 기본 설정 이후 단계에 있습니다. 새 `desktop.log`(Settings → Gateway → Open logs)와 동일한 재시도 시간대의 대시보드 로그를 확보하고 `/api/ws` 종료 코드를 확인하세요(4403 = 요청 가드가 채팅 WS를 거부함(예: Host/피어 불일치), 4401 = WS 티켓이 인증되지 않음).
### 설정

`config.yaml`용 폼 기반 편집기입니다. 150개가 넘는 모든 설정 필드를 `DEFAULT_CONFIG`에서 자동으로 검색해 탭으로 구분된 카테고리로 정리합니다:

![설정 관리 페이지 — 왼쪽의 섹션 필터와 오른쪽의 자동 검색 필드](/img/dashboard/admin-config.png)


- **model** — 기본 모델, 제공자, 기본 URL, 추론 설정
- **terminal** — 백엔드(local/docker/ssh/modal), 타임아웃, 셸 환경설정
- **display** — 스킨, 도구 진행 상황, 재개 표시, 스피너 설정
- **agent** — 최대 반복 횟수, 게이트웨이 타임아웃, 서비스 등급
- **delegation** — 서브에이전트 제한, 추론 작업량
- **memory** — 제공자 선택, 컨텍스트 주입 설정
- **approvals** — 위험한 명령 승인 모드(smart/manual/off)
- 그 외 — `config.yaml`의 모든 섹션에 해당하는 폼 필드

유효한 값이 알려진 필드(터미널 백엔드, 스킨, 승인 모드 등)는 드롭다운으로 표시됩니다. 불리언은 토글로 표시됩니다. 나머지는 모두 텍스트 입력란입니다.

**작업:**

- **Save** — 변경 사항을 즉시 `config.yaml`에 기록합니다
- **Reset to defaults** — 모든 필드를 기본값으로 되돌립니다(Save를 클릭하기 전에는 저장되지 않음)
- **Export** — 현재 설정을 JSON으로 다운로드합니다
- **Import** — JSON 설정 파일을 업로드해 현재 값을 대체합니다

:::tip
설정 변경 사항은 다음 에이전트 세션을 시작하거나 게이트웨이를 재시작할 때 적용됩니다. 웹 대시보드는 `hermes config set`과 게이트웨이가 읽는 동일한 `config.yaml` 파일을 편집합니다.
:::

### API 키

API 키와 자격 증명이 저장된 `.env` 파일을 관리합니다. 키는 카테고리별로 그룹화됩니다:

- **LLM Providers** — OpenRouter, Anthropic, OpenAI, DeepSeek 등
- **Tool API Keys** — Browserbase, Firecrawl, Tavily, ElevenLabs 등
- **Messaging Platforms** — Telegram, Discord, Slack 봇 토큰 등
- **Agent Settings** — `API_SERVER_ENABLED` 같은 비밀이 아닌 환경 변수

각 키에는 다음 정보가 표시됩니다:
- 현재 설정 여부(값을 일부 가린 미리보기 포함)
- 용도 설명
- 제공자의 가입/키 페이지 링크
- 값을 설정하거나 업데이트하는 입력란
- 값을 삭제하는 삭제 버튼

고급/거의 사용하지 않는 키는 기본적으로 토글 뒤에 숨겨집니다.

### 세션

모든 에이전트 세션을 탐색하고 확인합니다. 각 행에는 세션 제목, 출처 플랫폼 아이콘(CLI, Telegram, Discord, Slack, cron), 모델 이름, 메시지 수, 도구 호출 수, 마지막 활성 시점부터의 시간이 표시됩니다. 활성 세션에는 맥동하는 배지가 표시됩니다.

- **Filter** — **Chats / Automation / All** 탭으로 목록 범위를 지정합니다. *Chats*(기본값)는 사람과의 대화를 표시하고 자동화 노이즈(cron, tool, API, ACP 세션)를 숨깁니다. *Automation*은 자동화 세션만 표시합니다. *All*은 전부 표시합니다. 정확한 출처 드롭다운으로 특정 채널 하나(예: Telegram만)로 더 좁힐 수 있습니다. 검색은 현재 필터를 따릅니다.
- **Search** — FTS5를 사용해 모든 메시지 내용에서 전문 검색을 수행합니다. 결과에는 일치 부분이 강조된 발췌문이 표시되며, 펼쳤을 때 처음 일치하는 메시지로 자동 스크롤됩니다.
- **Stats** — 요약 막대에 전체 세션 수, 저장소에서 활성 상태인 세션 수, 보관된 세션 수, 전체 메시지 수, 출처별 내역이 표시됩니다.
- **Expand** — 세션을 클릭해 전체 메시지 기록을 불러옵니다. 메시지는 역할(user, assistant, system, tool)에 따라 색상이 구분되며 구문 강조가 적용된 Markdown으로 렌더링됩니다.
- **Tool calls** — 도구 호출이 포함된 assistant 메시지에는 함수 이름과 JSON 인수가 있는 접을 수 있는 블록이 표시됩니다.
- **Rename** — 세션 제목을 인라인으로 설정하거나 지웁니다(연필 아이콘).
- **Export** — 세션(메타데이터 + 전체 메시지 기록)을 JSON으로 다운로드합니다(다운로드 아이콘).
- **Prune** — 헤더의 "Prune old sessions" 버튼은 N일보다 오래된 종료 세션을 삭제합니다.
- **Delete** — 휴지통 아이콘으로 세션과 메시지 기록을 삭제합니다.

![세션 관리 페이지 — 통계 막대, 정리, 행별 이름 변경 / 내보내기 / 삭제](/img/dashboard/admin-sessions.png)

### 로그

필터링과 실시간 추적 기능으로 에이전트, 게이트웨이, 오류 로그 파일을 확인합니다.

- **File** — `agent`, `errors`, `gateway` 로그 파일 간 전환
- **Level** — 로그 수준별 필터링: ALL, DEBUG, INFO, WARNING, ERROR
- **Component** — 소스 구성 요소별 필터링: all, gateway, agent, tools, cli, cron
- **Lines** — 표시할 줄 수 선택(50, 100, 200, 500)
- **Auto-refresh** — 5초마다 새 로그 줄을 폴링하는 실시간 추적 토글
- **Color-coded** — 심각도에 따라 로그 줄 색상 표시(오류는 빨간색, 경고는 노란색, 디버그는 흐리게)

### 분석

세션 기록에서 계산한 사용량 및 비용 분석입니다. 기간(7일, 30일, 90일)을 선택하면 다음을 확인할 수 있습니다:

- **Summary cards** — 전체 토큰(입력/출력), 캐시 적중률, 전체 예상 또는 실제 비용, 일일 평균과 함께 표시되는 전체 세션 수
- **Daily token chart** — 일별 입력 및 출력 토큰 사용량을 보여주는 누적 막대 차트. 마우스를 올리면 내역과 비용이 툴팁으로 표시됩니다.
- **Daily breakdown table** — 각 날짜의 날짜, 세션 수, 입력 토큰, 출력 토큰, 캐시 적중률, 비용
- **Per-model breakdown** — 사용한 각 모델, 세션 수, 토큰 사용량, 예상 비용을 보여주는 표

### Cron

반복 일정에 따라 에이전트 프롬프트를 실행하는 예약 cron 작업을 생성하고 관리합니다.

- **Create** — 이름(선택 사항), 프롬프트, cron 표현식(예: `0 9 * * *`), 전달 대상(local, Telegram, Discord, Slack, email)을 입력합니다.
- **Job list** — 각 작업에 이름, 프롬프트 미리보기, 일정 표현식, 상태 배지(enabled/paused/error), 전달 대상, 마지막 실행 시각, 다음 실행 시각이 표시됩니다.
- **Pause / Resume** — 작업을 활성 상태와 일시 중지 상태 사이에서 전환합니다.
- **Edit** — 미리 채워진 모달을 열어 작업의 프롬프트, 일정, 이름, 전달 대상을 변경합니다.
- **Trigger now** — 정상 일정과 관계없이 작업을 즉시 실행합니다.
- **Delete** — cron 작업을 영구적으로 삭제합니다.

### 프로필

[프로필](../profiles.md)을 생성하고 관리합니다. 프로필은 각각 고유한 설정, 스킬, 세션을 가진 격리된 Hermes 인스턴스입니다.

- **Profile cards** — 모델/제공자, 스킬 수, 게이트웨이 상태, 설명, 배지(active, default, alias)를 표시합니다.
- **Create** — 이름과 선택 사항인 기본 프로필에서 복제 / 전체 복제 / 번들 스킬 제외, 설명, 모델을 설정합니다. 전용 Profile Builder 페이지(`/profiles/new`)에서는 전체 흐름(모델, MCP, 스킬)을 제공합니다.
- **Manage skills & tools** — 해당 프로필 범위의 Skills 페이지로 이동합니다(사이드바 프로필 전환기가 설정됨).
- **Set as active** — **향후 CLI/gateway 실행**에서 사용할 고정 기본값을 변경합니다(`hermes profile use`와 동일). 대시보드가 관리하는 대상은 변경하지 않으며, 이는 프로필 전환기의 역할입니다.
- **Edit model / description / SOUL** — 해당 프로필에 기록되는 인라인 편집기
- **Rename / Delete** — 이름이 지정된 프로필에만 적용됩니다.

### 스킬

설치된 스킬과 도구 모음을 탐색·검색·토글하고 허브에서 새 항목을 설치합니다. 스킬은 `~/.hermes/skills/`에서 불러오며 카테고리별로 그룹화됩니다.

- **Search** — 이름, 설명, 카테고리로 설치된 스킬과 도구 모음을 필터링합니다.
- **Category filter** — 카테고리 필을 클릭해 목록을 좁힙니다(예: MLOps, MCP, Red Teaming, AI).
- **Toggle** — 스위치로 개별 스킬을 활성화하거나 비활성화합니다. 변경 사항은 다음 세션부터 적용됩니다.
- **Toolsets** — 별도 보기에서 기본 제공 도구 모음(파일 작업, 웹 브라우징 등)의 활성/비활성 상태, 설정 요구 사항, 포함된 도구 목록을 표시합니다.
- **Browse hub** — 세 번째 보기에서 모든 소스에 걸쳐 스킬 허브를 검색하고(`hermes skills search`와 동일), 식별자로 검색 결과를 설치하며 실시간 설치 로그를 제공합니다. 설치된 스킬을 새로 고치는 "Update all" 버튼도 있습니다.

![스킬 관리 페이지 — Browse hub 보기: 검색, 설치, 업데이트](/img/dashboard/admin-skills-hub.png)

### MCP

CLI 없이 [MCP](./mcp) 서버를 관리합니다. `hermes mcp`가 읽는 `config.yaml`의 동일한 `mcp_servers` 블록을 사용합니다.

**내 MCP 서버:**

- **Add** — HTTP/SSE 서버(URL) 또는 stdio 서버(명령 + 인수)를 등록하며, stdio 서버에는 선택 사항인 `KEY=VALUE` 환경 변수를 지정할 수 있습니다.
- **Enable / disable** — 서버를 삭제하지 않고 켜거나 끕니다. 비활성화된 서버는 설정에 남아 있으므로 나중에 다시 활성화할 수 있습니다. 다음 게이트웨이 재시작부터 적용됩니다.
- **Test** — 서버에 연결하고 도구를 나열한 후 연결을 끊어 에이전트가 의존하기 전에 연결을 확인합니다.
- **Remove** — 설정에서 서버를 삭제합니다.
- 목록 보기에서는 비밀처럼 보이는 환경 변수 값이 가려집니다.

**Catalog:** Nous가 승인한 MCP 서버(`optional-mcps/` 번들 카탈로그)를 탐색하고 한 번의 클릭으로 원하는 서버를 설치합니다. API 키가 필요한 항목은 인라인으로 API 키를 입력하도록 요청하며, 값은 `.env`에 저장됩니다. `hermes mcp catalog` / `hermes mcp install`에서 사용하는 것과 동일한 카탈로그입니다.

![MCP 관리 페이지 — 활성화/비활성화 토글이 있는 내 서버와 설치 카탈로그](/img/dashboard/admin-mcp.png)

### 웹훅

동적 [웹훅 구독](/user-guide/messaging/webhooks)을 관리합니다. 먼저 메시징 설정에서 webhook 플랫폼을 활성화해야 하며, 활성화되지 않은 경우 페이지에 안내가 표시됩니다.

- **Create** — 이름, 설명, 이벤트 필터, 전달 대상, 선택 사항인 직접 전달 모드, 에이전트 프롬프트를 설정합니다. 생성하면 페이지에 복사할 경로 URL과 한 번만 표시되는 HMAC 비밀이 나타납니다.
- **Enable / disable** — 구독을 켜거나 끕니다. 비활성화된 경로는 구독 파일에 남지만 게이트웨이는 수신 이벤트를 거부합니다(403). 게이트웨이가 파일을 핫 리로드하므로 변경 사항은 다음 이벤트부터 적용되며 재시작이 필요하지 않습니다.
- **List** — 각 구독에 URL, 이벤트, 전달 대상이 표시됩니다.
- **Delete** — 구독을 삭제합니다.

![웹훅 관리 페이지 — 활성화/비활성화 토글이 있는 구독](/img/dashboard/admin-webhooks.png)

### 페어링

CLI 없이 메시징 사용자를 승인하고 취소합니다. 원격 관리자가 Telegram/Discord 등의 사용자를 페어링된 게이트웨이에 등록할 수 있습니다. `hermes pairing`과 완전히 동일한 기능을 제공합니다.

- **Pending requests** — 각 요청에 플랫폼, 코드, 사용자, 경과 시간이 표시되며 Approve 버튼이 제공됩니다.
- **Approved users** — 각 사용자에 플랫폼과 사용자가 표시되며 Revoke 버튼이 제공됩니다.
- **Clear pending** — 보류 중인 모든 페어링 코드를 삭제합니다.

![페어링 관리 페이지](/img/dashboard/admin-pairing.png)

### 채널

브라우저에서 Hermes를 모든 메시징 플랫폼에 연결합니다. `hermes setup gateway`와 완전히 동일한 기능을 제공합니다. 페이지에는 지원되는 모든 채널(Telegram, Discord, Slack, Matrix, Mattermost, WhatsApp, Signal, BlueBubbles/iMessage, Email, SMS/Twilio, DingTalk, Feishu/Lark, WeCom, WeChat, QQ Bot, Yuanbao와 API 서버 및 webhook 엔드포인트)이 실시간 연결 상태와 함께 나열됩니다.

- **Configure** — 채널에 필요한 필드만 정확히 포함한 플랫폼별 폼을 엽니다(봇 토큰, 앱 토큰, 서버 URL, 허용 목록 등). 비밀 값은 비밀번호 입력란으로 표시되고 가려진 상태로 저장됩니다. 필드를 비워 두면 기존 값이 유지됩니다. 필수 필드는 표시되고 검증됩니다. "Setup guide" 링크는 플랫폼 자격 증명 문서로 연결됩니다.
- **Enable / disable** — 채널을 켜거나 끕니다. 자격 증명은 디스크에 유지되고 활성 상태만 변경됩니다.
- **Test** — 채널이 설정되고 활성화되었는지, 게이트웨이에서 실시간 연결을 보고하고 있는지 확인합니다.
- **Restart gateway** — 자격 증명은 `~/.hermes/.env`에, 활성화 플래그는 `config.yaml`에 기록됩니다. 게이트웨이는 다음 재시작 시 활성화된 각 채널에 연결하며, 페이지에서 바로 재시작할 수 있습니다.

![채널 관리 페이지 — 상태, 활성화 토글, 플랫폼별 설정 폼이 있는 모든 메시징 플랫폼](/img/dashboard/admin-channels.png)

### 시스템

설치 전체에 적용되는 작업을 위한 통합 관리 패널입니다:

- **Host** — 실시간 시스템 통계: OS / 커널, 아키텍처, 호스트 이름, Python 및 Hermes 버전, CPU 코어 수 + 사용률, 메모리, Hermes 홈 디렉터리의 디스크 사용량, 가동 시간, 평균 부하. (CPU/메모리는 설치된 경우 `psutil`에서 가져오며, 식별 정보 필드는 항상 표시됩니다.) Hermes 버전에는 **update-status badge**(최신 상태 / N개 커밋 뒤처짐)와 **Check for updates** 버튼이 표시됩니다. git 설치에서 업데이트가 가능하면 **Update now** 버튼을 통해 확인 대화상자가 열립니다. 이 대화상자에는 백그라운드에서 `hermes update`를 실행하기 전에 가져올 커밋 수가 표시됩니다. Docker/Nix 설치에서는 대시보드가 해당 위치에 직접 업데이트를 적용할 수 없으므로 올바른 별도 명령을 표시합니다.
- **Nous Portal** — 로그인 상태, 활성 추론 제공자, Tool Gateway 라우팅 표(Portal을 통해 실행되는 도구와 로컬에서 실행되는 도구), 구독 관리 링크를 표시합니다. `hermes portal`의 읽기 전용 미러입니다.
- **Skill curator** — 백그라운드 스킬 유지 관리 상태(active / paused, 간격, 마지막 실행 시각)를 표시하고 일시 중지/재개 및 지금 실행 버튼을 제공합니다. `hermes curator`의 미러입니다.
- **Gateway** — 실시간 상태(running/stopped, PID, state)와 함께 메시징 게이트웨이를 시작, 중지, 재시작합니다.
- **Memory** — 외부 메모리 제공자(또는 내장 기능만)를 선택하고 내장 `MEMORY.md` / `USER.md` 저장소를 초기화합니다.
- **Credential pool** — 에이전트가 순환 사용하도록 할 API 키를 제공자별로 추가·삭제합니다. 목록에서는 키가 가려지며 원시 값은 에이전트에만 전달됩니다.
- **Operations** — `doctor` 실행, 보안 감사, 백업 생성, 백업 아카이브에서 복원, 스킬 업데이트, 시스템 프롬프트 크기 내역 표시, 지원 덤프 생성, 폐기된 설정 마이그레이션을 수행합니다. 각 작업은 백그라운드 작업으로 실행되며 실시간 로그가 페이지에 스트리밍됩니다.
- **Checkpoints** — `/rollback` 섀도 저장소 크기를 확인하고 정리합니다.
- **Shell hooks** — 동의 + 실행 가능 상태와 함께 설정된 훅을 나열하고, **훅을 생성**(이벤트, 명령, 매처, 타임아웃, 선택 동의 승인 포함)하거나 삭제합니다. 훅은 임의의 명령을 실행하므로 생성 폼에 보안 경고가 표시되며 동의를 승인한 후에만 훅이 실행됩니다.

![시스템 관리 페이지 — 호스트 통계와 Nous Portal 상태](/img/dashboard/admin-system-top.png)

![시스템 관리 페이지 — 스킬 큐레이터, 게이트웨이, 메모리, 자격 증명 풀](/img/dashboard/admin-system-curator.png)

![시스템 관리 페이지 — 작업, 체크포인트, 셸 훅](/img/dashboard/admin-system-ops.png)

셸 훅 생성(동의 확인란과 임의 명령 실행 경고 참고):

![새 셸 훅 모달](/img/dashboard/admin-hook-create.png)

:::warning Security
웹 대시보드는 API 키와 비밀이 포함된 `.env` 파일을 읽고 씁니다. 기본적으로 `127.0.0.1`에 바인딩되므로 로그인 없이 로컬 컴퓨터에서만 접근할 수 있습니다. 루프백이 아닌 주소(예: `0.0.0.0`)에 바인딩하면 [인증 게이트](#authentication-gated-mode)가 활성화됩니다. 인증 제공자(사용자 이름/비밀번호 또는 OAuth)를 설정할 때까지 서버가 시작되지 않습니다.
:::
## `/reload` 슬래시 명령

대시보드 PR에는 대화형 CLI를 위한 `/reload` 슬래시 명령도 추가되어 있습니다. 웹 대시보드에서 API 키를 변경한 후(또는 `.env`를 직접 편집한 후), 활성 CLI 세션에서 `/reload`를 사용하면 변경 사항을 적용하기 위해 재시작할 필요가 없습니다.

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

이 명령은 실행 중인 프로세스의 환경에 `~/.hermes/.env`를 다시 읽어들입니다. 대시보드를 통해 새 provider 키를 추가한 후 즉시 사용하고 싶을 때 유용합니다.

## REST API

웹 대시보드는 프런트엔드가 사용하는 REST API를 제공합니다. 자동화를 위해 이 엔드포인트를 직접 호출할 수도 있습니다.

:::tip 프로필 범위 엔드포인트
관리 엔드포인트 그룹인 `/api/config`, `/api/env`, `/api/skills`,
`/api/tools/toolsets`, `/api/mcp`, `/api/model/{info,options,auxiliary,set}`은
선택적 `?profile=<name>` 쿼리 매개변수(또는 쓰기 요청의 JSON 본문에 포함하는 `"profile"`)를
허용하며, 해당 프로필의 `HERMES_HOME`을 대상으로 읽기/쓰기를 수행합니다. 생략하면 대시보드
자체 프로필이 사용됩니다. 알 수 없는 프로필 이름은 `404`를 반환합니다. `/api/pty` WebSocket도
동일한 매개변수를 받아 선택한 프로필에서 채팅을 실행합니다.
:::

### GET /api/status

에이전트 버전, 게이트웨이 상태, 플랫폼 상태 및 활성 세션 수를 반환합니다.

응답에는 `components`/`overall` 상태 판정에 절대 영향을 주지 않는 두 개의 참고용 리소스 블록도 포함됩니다.

- **`memory`** — 게이트웨이의 30초 heartbeat와 수명 주기 원장에서 요약합니다. 필드: `pressure` (`ok` / `elevated` / `critical` / `unknown`), `gateway_rss_mb`, `system_total_mb`, `system_available_mb`, `swap_used_mb`, `sampled_at`, `boot_id`, `last_boot_unclean`, `last_boot_suspected_oom`. 사용 가능한 시스템 메모리가 128 MiB(또는 15%) 미만이면 `pressure`는 `elevated`, 64 MiB(또는 5%) 미만이면 `critical`입니다. 이는 이후 비정상 종료가 의심되는 OOM kill로 표시되는 것과 동일한 수준입니다. 150초보다 오래된 heartbeat(또는 미래 시각이 기록된 heartbeat)는 수치를 유지하지만 `pressure`를 `unknown`으로 낮춥니다. 따라서 중단된 게이트웨이의 마지막 샘플이 현재 읽기인 것처럼 보일 수 없습니다.
- **`disk`** — `~/.hermes`가 있는 볼륨의 실시간 `shutil.disk_usage()` 샘플입니다. 필드: `pressure`, `free_mb`, `total_mb`, `used_percent`, `sampled_at`. 여유 공간이 512 MB 미만이거나(또는 여유 헤드룸이 4 GB 미만인 상태에서 사용률이 85% 이상이면) `elevated`, 여유 공간이 256 MB 미만이거나(또는 여유 헤드룸이 1 GB 미만인 상태에서 사용률이 95% 이상이면) `critical`입니다.

두 수집기는 모두 장애 안전(fail-safe) 방식으로 동작합니다. 샘플링 오류가 발생하면 상태 엔드포인트 자체를 실패시키는 대신 블록을 `{"pressure": "unknown"}`으로 낮춥니다. `/api/status`는 공개 API이므로 수치는 대략적인 값(정수 MB, 정수 퍼센트)입니다.

### GET /api/sessions

메타데이터(모델, 토큰 수, 타임스탬프, 미리보기)와 함께 최근 세션 20개를 반환합니다.

### GET /api/config

현재 `config.yaml` 내용을 JSON으로 반환합니다.

### GET /api/config/defaults

기본 구성 값을 반환합니다.

### GET /api/config/schema

모든 구성 필드를 설명하는 스키마를 반환합니다. 각 필드에는 해당되는 경우 타입, 설명, 카테고리 및 선택 옵션이 포함됩니다. 프런트엔드는 이를 사용해 각 필드에 맞는 입력 위젯을 렌더링합니다.

### PUT /api/config

새 구성을 저장합니다. 본문: `{"config": {...}}`.

### GET /api/env

알려진 모든 환경 변수의 설정/미설정 상태, 마스킹된 값, 설명 및 카테고리를 반환합니다.

### PUT /api/env

환경 변수를 설정합니다. 본문: `{"key": "VAR_NAME", "value": "secret"}`.

### DELETE /api/env

환경 변수를 제거합니다. 본문: `{"key": "VAR_NAME"}`.

### GET /api/sessions/\{session_id\}

단일 세션의 메타데이터를 반환합니다.

### GET /api/sessions/\{session_id\}/messages

도구 호출과 타임스탬프를 포함한 메시지 기록의 제한된 페이지를 반환합니다. 기본적으로 시간순으로 정렬된 최신 메시지 500개를 반환합니다. 명시적으로 페이지를 지정하려면 `limit`(최대 500), `offset` 및 `order=oldest|latest`를 사용합니다.

### GET /api/sessions/search

메시지 콘텐츠 전체를 대상으로 전문 검색을 수행합니다. 쿼리 매개변수: `q`. 강조 표시된 발췌문과 일치하는 세션 ID를 반환합니다.

### DELETE /api/sessions/\{session_id\}

세션과 해당 메시지 기록을 삭제합니다.

### GET /api/logs

로그 행을 반환합니다. 쿼리 매개변수: `file` (agent/errors/gateway), `lines` (개수), `level`, `component`.

### GET /api/analytics/usage

토큰 사용량, 비용 및 세션 분석을 반환합니다. 쿼리 매개변수: `days` (기본값 30). 응답에는 일별 세부 내역과 모델별 집계가 포함됩니다.

### GET /api/cron/jobs

구성된 모든 cron 작업의 상태, 일정 및 실행 기록을 반환합니다.

### POST /api/cron/jobs

새 cron 작업을 생성합니다. 본문: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`.

### POST /api/cron/jobs/\{job_id\}/pause

cron 작업을 일시 중지합니다.

### POST /api/cron/jobs/\{job_id\}/resume

일시 중지된 cron 작업을 재개합니다.

### POST /api/cron/jobs/\{job_id\}/trigger

예약된 시간 외에 cron 작업을 즉시 실행합니다.

### DELETE /api/cron/jobs/\{job_id\}

cron 작업을 삭제합니다.

### GET /api/skills

이름, 설명, 카테고리 및 활성화 상태와 함께 모든 skill을 반환합니다.

### PUT /api/skills/toggle

skill을 활성화하거나 비활성화합니다. 본문: `{"name": "skill-name", "enabled": true}`.

### GET /api/tools/toolsets

레이블, 설명, 도구 목록 및 활성/구성 상태와 함께 모든 toolset을 반환합니다.

### 관리 엔드포인트

이 엔드포인트들은 MCP, Channels, Webhooks, Pairing 및 System 페이지를 구동합니다. 모두 `/api/`의 나머지 엔드포인트와 동일한 인증 게이트 뒤에 있습니다.

| 메서드 및 경로 | 용도 |
|---------------|---------|
| `GET /api/mcp/servers` | 구성된 MCP 서버 목록(환경 변수 값은 마스킹) |
| `POST /api/mcp/servers` | 서버 추가. 본문: `{name, url?, command?, args?, env?, auth?}` |
| `POST /api/mcp/servers/{name}/test` | 연결, 도구 목록 조회, 연결 해제 |
| `PUT /api/mcp/servers/{name}/enabled` | 서버 활성화 / 비활성화 |
| `DELETE /api/mcp/servers/{name}` | 서버 제거 |
| `GET /api/mcp/catalog` | Nous 승인 MCP 카탈로그 탐색 |
| `POST /api/mcp/catalog/install` | 카탈로그 항목 설치(필수 환경 변수 포함) |
| `GET /api/messaging/platforms` | 모든 메시징 채널과 상태 및 플랫폼별 설정 필드 목록 조회 |
| `PUT /api/messaging/platforms/{id}` | 채널 구성. 본문: `{enabled?, env?, clear_env?}` (`env`는 `.env`에, enabled는 `config.yaml`에 기록) |
| `POST /api/messaging/platforms/{id}/test` | 채널의 구성, 활성화 및 연결 여부 보고 |
| `GET /api/pairing` | 대기 중인 메시징 사용자 및 승인된 메시징 사용자 목록 조회 |
| `POST /api/pairing/approve` | 코드 승인. 본문: `{platform, code}` |
| `POST /api/pairing/revoke` | 사용자 승인 취소. 본문: `{platform, user_id}` |
| `POST /api/pairing/clear-pending` | 대기 중인 모든 코드 삭제 |
| `GET /api/webhooks` | 구독 및 플랫폼 활성화 상태 목록 조회 |
| `POST /api/webhooks` | 구독 생성(일회성 secret 반환) |
| `DELETE /api/webhooks/{name}` | 구독 제거 |
| `GET /api/credentials/pool` | 풀에 저장된 순환 키 목록(마스킹) |
| `POST /api/credentials/pool` | 키 추가. 본문: `{provider, api_key, label?}` |
| `DELETE /api/credentials/pool/{provider}/{index}` | 키 제거(1부터 시작하는 인덱스) |
| `GET /api/memory` | 활성 provider, 사용 가능한 provider 및 내장 파일 크기 |
| `PUT /api/memory/provider` | provider 선택(빈 값 = 내장 기능만 사용) |
| `POST /api/memory/reset` | 내장 메모리 초기화. 본문: `{target: all\|memory\|user}` |
| `POST /api/gateway/start` · `/stop` · `/restart` | 게이트웨이 수명 주기(백그라운드 실행) |
| `POST /api/ops/doctor` · `/security-audit` · `/backup` · `/import` | 진단 및 유지 관리(백그라운드 실행; `/api/actions/{name}/status`로 추적) |
| `GET /api/ops/hooks` | 구성된 셸 hook 및 허용 목록 상태 |
| `GET /api/ops/checkpoints` · `POST .../prune` | `/rollback` 저장소 검사 / 정리 |
| `POST /api/ops/hooks` · `DELETE /api/ops/hooks` | 셸 hook 생성 / 제거(동의 필요) |
| `GET /api/system/stats` | 호스트 통계 — OS, CPU, 메모리, 디스크, 가동 시간 |
| `GET /api/hermes/update/check` | 업데이트 가능 여부(뒤처진 커밋 수, 설치 방법)를 적용하지 않고 보고합니다. Git 설치가 뒤처진 경우 변경된 항목의 `commits` 목록(`sha`, `summary`, `author`, `at`)도 반환합니다. `?force=1`은 6시간 캐시를 무효화합니다. |
| `GET /api/curator` · `PUT .../paused` · `POST .../run` | Skill curator 상태 + 일시 중지/재개 + 실행 |
| `GET /api/portal` | Nous Portal 인증 + Tool Gateway 라우팅(읽기 전용) |
| `POST /api/ops/prompt-size` · `/dump` · `/config-migrate` | 진단(백그라운드 실행) |
| `PUT /api/webhooks/{name}/enabled` | webhook 경로 활성화 / 비활성화 |
| `POST /api/skills/hub/install` · `/uninstall` · `/update` | Skills hub 작업(백그라운드 실행) |
| `GET /api/skills/hub/search` | 모든 소스에서 skill hub 검색 |
| `GET /api/sessions/stats` | 세션 저장소 통계 |
| `PATCH /api/sessions/{id}` | 세션 이름 변경 / 보관 |
| `GET /api/sessions/{id}/export` | 세션을 JSON으로 내보내기(메타데이터 + 메시지) |
| `POST /api/sessions/prune` | N일보다 오래된 종료 세션 삭제 |
| `PUT /api/cron/jobs/{id}` | cron 작업의 prompt / schedule / name / deliver 편집 |

## 인증(게이트 모드)

대시보드가 공개 또는 루프백이 아닌 주소(즉, `127.0.0.1` / `localhost` 이외의 주소)에 바인딩되면 Hermes Agent가 인증 게이트를 활성화합니다. 모든 요청에는 검증된 세션 쿠키가 포함되어야 하며, 그렇지 않으면 로그인 페이지로 이동합니다. 기본 제공되는 provider는 세 가지입니다.

- **[사용자 이름/비밀번호](#usernamepassword-provider-no-oauth-idp)** — 셀프 호스팅 / 온프레미스 / 홈랩 대시보드에 인증을 추가하는 가장 간단한 방법입니다. 외부 identity provider가 필요 없습니다. **신뢰할 수 있는 네트워크 또는 VPN 뒤에서만 사용하세요. 공개 인터넷에 노출해서는 안 됩니다.**
- **[OAuth (Nous Portal)](#default-provider-nous-research)** — 호스팅 배포 및 공개 인터넷에서 접근할 수 있는 모든 대시보드에 적합하며, [원격 Hermes Desktop 연결](#connecting-hermes-desktop-to-a-remote-backend)을 위한 권장 경로입니다. 모든 로그인은 Nous 계정에 대해 검증되므로 인터넷에 공개되는 환경에 적합한 provider입니다.
- **[셀프 호스팅 OIDC](#self-hosted-oidc-provider)** — 표준 OpenID Connect(Keycloak, Auth0, Okta, Google, OIDC 브리지를 통한 GitHub 등)를 사용해 자체 identity provider를 연결합니다. Nous Portal은 관여하지 않으며, 규격을 준수하는 OIDC 서버 앞에 배치하면 공개 인터넷 노출에도 적합합니다.

루프백에 바인딩된 운영자 소유 대시보드는 영향을 받지 않습니다. 인증도, 로그인 페이지도 없습니다.

### 게이트가 활성화되는 경우

| 플래그 | 인증 게이트 | 사용 사례 |
|-------|-----------|----------|
| `hermes dashboard` (기본값 — `127.0.0.1`에 바인딩) | OFF | 로컬 개발 |
| `hermes dashboard --host 0.0.0.0` | **ON** | 원격 / 운영 — 사용자 이름/비밀번호 provider 또는 OAuth로 보호 |

바인딩 호스트가 `127.0.0.1`, `::1` 또는 `localhost`가 아닌 경우에만 게이트가 활성화됩니다. `0.0.0.0`(또는 RFC1918 / LAN 주소)에 바인딩하면 게이트가 활성화됩니다. 기존 `--insecure` 플래그는 **더 이상 게이트를 비활성화하지 않습니다** — 하위 호환성을 위해 허용되지만 경고와 함께 무시됩니다.

:::danger `--insecure`는 아무 작업도 하지 않습니다 — 인증을 비활성화하지 않습니다
2026년 6월 보안 강화 이후 `--insecure`는 더 이상 대시보드 인증을 우회하지 않습니다. 루프백이 아닌 바인딩에는 항상 인증 provider(사용자 이름/비밀번호 provider 또는 OAuth)가 필요합니다. 인증 없는 대시보드를 사용하려면 `127.0.0.1`에 바인딩하고 SSH 터널 또는 Tailscale을 통해 접근하세요.
:::

### Fail-closed 의미

게이트가 활성화되어야 하지만 등록된 `DashboardAuthProvider`가 없으면(Nous plugin도, 사용자 지정 plugin도 없으면) `hermes dashboard`는 명시적인 오류 메시지와 함께 바인딩을 거부합니다. "기본 거부이지만 모든 요청을 허용"하는 대체 동작은 없습니다. 잘못 구성된 게이트 대시보드는 시작되지 않습니다.

대화형으로(실제 터미널에서) `hermes dashboard --host 0.0.0.0`을 실행하고 아직 provider가 구성되지 않은 경우, Hermes는 단순히 실패하는 대신 즉시 설정할 수 있도록 제안합니다. **사용자 이름 및 비밀번호**를 선택하면(`dashboard.basic_auth`를 `config.yaml`에 기록하고 몇 초 안에 실행) 되고, **OAuth**를 선택하면 `hermes dashboard register`로 안내됩니다. 비대화형 호출자(Docker/s6, CI, 파이프로 연결된 실행)는 프롬프트를 건너뛰고 위의 fail-closed 오류를 표시하므로, 인증 없이는 무인 배포도 시작되지 않습니다.
### 기본 제공자: Nous Research

번들로 제공되는 `plugins/dashboard_auth/nous` 플러그인은 **항상 설치되어 있으며** 클라이언트 ID가 구성되면 자동으로 `nous`라는 이름의 `DashboardAuthProvider`를 등록합니다.

모든 로그인이 Nous Portal에 대해 검증되고 Nous 계정으로 보호되므로, **대시보드를 퍼블릭 인터넷에 노출할 때 적합한 제공자는 Nous 제공자입니다.**

#### 대시보드 등록

Nous 제공자를 사용하려면 OAuth 클라이언트 ID(`agent:{id}` 형식)가 필요합니다. 다음 두 가지 방법으로 얻을 수 있습니다.

- **CLI — `hermes dashboard register`.** 대시보드가 실행될 호스트에서 실행합니다. 기존 Nous 로그인을 확인하고(아직 로그인하지 않았다면 먼저 `hermes setup` 실행), Portal에 셀프 호스팅 OAuth 클라이언트를 등록한 다음 `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 `~/.hermes/.env`에 대신 기록합니다. 선택적 플래그로는 `--name`(사람이 읽을 수 있는 레이블이며, 지정하지 않으면 자동 생성)과 `--redirect-uri`(인터넷에 연결된 호스트의 퍼블릭 HTTPS 콜백 URL)가 있습니다.

  ```bash
  hermes dashboard register
  # ✓ Registered dashboard "swift_falcon"
  # …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
  ```

- **GUI — Local Dashboards 페이지.** Nous Portal에서 [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards)를 열어 브라우저에서 셀프 호스팅 대시보드를 등록하고, 이름을 지정하고, 관리하고, 폐기할 수 있습니다. 생성된 `agent:{id}` 클라이언트 ID를 `HERMES_DASHBOARD_OAUTH_CLIENT_ID`(env) 또는 `dashboard.oauth.client_id`(config.yaml)에 복사합니다. CLI로 등록한 대시보드를 폐기하는 곳도 여기입니다.

#### 구성

플러그인은 두 가지 설정 경로에서 읽으며, 비어 있지 않은 환경 변수가 설정되어 있으면 환경 변수가 우선합니다.

**`config.yaml`** — 표준 설정 경로:

```yaml
dashboard:
  oauth:
    client_id: agent:01HXYZ…             # required to engage the gate
```

**환경 변수** — 운영자 재정의:

| 환경 변수 | 재정의 대상 | 형식 | 설정 주체 |
|---------|-----------|--------|----------------|
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | `dashboard.oauth.client_id` | `agent:{instance_id}` | `hermes dashboard register` |

Hermes Agent 규칙(`~/.hermes/.env`는 API 키 / 시크릿만을 위한 파일)에 따라 **로컬 개발, 온프레미스 및 직접 제어하는 모든 배포에서는 이 값을 `config.yaml`에 설정하는 것을 권장합니다.** 환경 변수 경로는 호스팅 플랫폼의 시크릿 주입 기능을 통해 배포별 `client_id`를 전달할 수 있도록 존재합니다. 이미지 내부의 `config.yaml`을 편집할 필요가 없다는 것이 이 경로의 주된 목적입니다.

비어 있는 환경 변수 값은 설정되지 않은 것으로 처리되므로, 값이 채워지지 않은 상태로 프로비저닝된 플랫폼 시크릿이 유효한 `config.yaml` 항목을 실수로 가리는 일이 없습니다.

어느 소스에서도 client_id를 제공하지 않으면 플러그인이 구체적인 이유를 보고하고, 대시보드의 fail-closed 바인딩 오류가 정확히 수정할 항목을 알려 줍니다.

```
Refusing to bind dashboard to 0.0.0.0 — the auth gate engages on
non-loopback binds, but no auth providers are registered.

Bundled providers reported these issues:
  • nous: HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and
    dashboard.oauth.client_id in config.yaml is empty). …

Configure an auth provider before exposing the dashboard:
  • Password: set dashboard.basic_auth.username + password_hash in config.yaml
  • OAuth: run `hermes dashboard register` (Nous Portal) or install a
    DashboardAuthProvider plugin.
There is no unauthenticated public-bind option — to keep it local, bind
127.0.0.1 and tunnel in (SSH / Tailscale).
```

#### 사용 예: Nous Research

로그인된 Hermes 설치 상태에서 Nous로 보호되는 대시보드까지 세 단계로 진행합니다.

**1. 로그인하고 대시보드를 등록합니다.** `hermes dashboard register`는 기존 Nous 로그인을 사용해 OAuth 클라이언트를 프로비저닝하고 `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 `~/.hermes/.env`에 대신 기록합니다.

```bash
hermes setup            # if you're not already logged into Nous Portal
hermes dashboard register
# ✓ Registered dashboard "swift_falcon"
# …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
```

**2. 연결 가능한 주소에서 대시보드를 실행합니다.** 루프백이 아닌 주소에 바인딩하면 OAuth 게이트가 활성화되고, 방금 기록한 `client_id`가 `nous` 제공자를 활성화합니다.

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. 로그인합니다.** `http://<host>:9119/`를 열면 `/login`으로 이동합니다. **Sign in with Nous Research**를 클릭하고 → Portal에서 인증한 뒤 → 인증된 대시보드로 돌아옵니다. 다음 명령으로 어느 컴퓨터에서든 게이트를 확인할 수 있습니다.

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

이후 `GET /api/auth/me`는 검증된 세션(`provider: nous`)을 반환합니다. 인터넷에 연결된 호스트에서는 `--redirect-uri https://hermes.example.com/auth/callback`으로 등록하고 `HERMES_DASHBOARD_PUBLIC_URL`을 설정해 OAuth 콜백이 퍼블릭 URL로 확인되도록 합니다([퍼블릭 URL 재정의](#public-url-override) 참조).

### 사용자 이름/비밀번호 제공자(OAuth IDP 없음)

OAuth IDP를 연결하지 않고 셀프 호스팅 환경에서 “대시보드에 비밀번호만 설정”하려는 경우, 번들 `plugins/dashboard_auth/basic` 플러그인이 OAuth 리디렉션 대신 **사용자 이름과 비밀번호**로 인증하는 `basic`이라는 이름의 `DashboardAuthProvider`를 등록합니다.

이 플러그인은 OAuth 제공자와 동일한 게이트에 연결됩니다. 루프백이 아닌 바인딩에서 게이트가 활성화되고, 로그인 페이지에는 이 제공자를 위한 자격 증명 양식(“X로 로그인” 버튼 대신)이 표시됩니다. 로그인 이후의 모든 기능(세션 쿠키, 투명한 갱신, WS 티켓, 로그아웃, 감사 로그)은 OAuth 경로와 동일합니다. 세션은 제공자가 직접 발급하는 상태 비저장 HMAC 서명 토큰이므로 **데이터베이스와 외부 IDP가 필요 없습니다.** 비밀번호 해싱에는 표준 라이브러리 `scrypt`(타사 의존성 없음)를 사용합니다.

:::warning 신뢰할 수 있는 네트워크에서만 사용 — 퍼블릭 인터넷에서는 사용하지 마세요
사용자 이름/비밀번호 제공자는 **신뢰할 수 있는 네트워크**의 셀프 호스팅 / 온프레미스 / 홈랩 대시보드 또는 **VPN**을 통해서만 연결할 수 있는 대시보드를 위한 것입니다. 외부 IDP, MFA 또는 사용자별 계정 없이 하나의 공유 자격 증명만 보호하므로 **대시보드를 퍼블릭 인터넷에 직접 노출하는 데 적합하지 않습니다.** 인터넷에 연결된 대시보드에는 [Nous Research 제공자](#default-provider-nous-research)(또는 자체 [셀프 호스팅 OIDC](#self-hosted-oidc-provider) / [사용자 지정 OAuth](#custom-providers) 제공자)를 사용하세요.
:::

#### 구성

Nous 제공자와 마찬가지로 `config.yaml`(표준 설정 경로)에서 읽으며, 비어 있지 않은 환경 변수가 있으면 환경 변수가 우선합니다. `username`과 `password_hash`(권장) 또는 `password` 중 하나가 함께 구성된 경우에만 활성화됩니다. 그렇지 않으면 아무 작업도 하지 않으므로 OAuth 사용자와 루프백 운영자는 영향을 받지 않습니다.

**`config.yaml`:**

```yaml
dashboard:
  basic_auth:
    username: admin
    # Preferred — no plaintext at rest. Compute with:
    #   python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"
    password_hash: "scrypt$16384$8$1$…$…"
    # ...or a plaintext password (hashed in-memory at load; less safe at rest):
    # password: "s3cret"
    secret: "<32+ random bytes, base64 or hex>"  # token-signing key
    session_ttl_seconds: 43200                    # optional; access-token lifetime (default 12h)
```

**환경 변수 재정의:**

| 환경 변수 | 재정의 대상 | 비고 |
|---------|-----------|-------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | `dashboard.basic_auth.username` | 활성화에 필요 |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | `dashboard.basic_auth.password_hash` | 권장(평문을 저장하지 않음) |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | `dashboard.basic_auth.password` | 평문; **config의 `password_hash`보다 우선**하므로 env를 통해 교체할 수 있음 |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | `dashboard.basic_auth.secret` | 토큰 서명 키 |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | `dashboard.basic_auth.session_ttl_seconds` | 액세스 토큰 수명 |

:::caution 안정적인 세션을 위해 명시적인 `secret`을 설정하세요
`secret`이 비어 있으면 프로세스별 무작위 서명 키가 생성됩니다. 단일 프로세스에서는 괜찮지만, **재시작할 때마다 모든 세션이 무효화되고** 세션이 **여러 워커에 걸쳐 유지되지 않습니다.** 재시작 후에도 유지되거나 여러 워커에서 사용하는 배포에서는 명시적인 `secret`을 설정하세요.
:::

`/auth/password-login` 엔드포인트는 클라이언트 IP별로 속도 제한됩니다(기본값: 분당 10회 → HTTP 429). 또한 존재하지 않는 사용자와 잘못된 비밀번호 모두에 대해 단일한 일반 `401 Invalid credentials`를 반환하므로 사용자 이름 열거 오라클로 사용할 수 없습니다.

#### 사용 예: 사용자 이름/비밀번호

신뢰할 수 있는 네트워크에서 비밀번호로 보호되는 대시보드를 처음부터 세 단계로 구성합니다.

**1. `~/.hermes/.env`에 자격 증명을 설정합니다.** 평문이 저장되지 않도록 비밀번호를 해시하고, 재시작 후에도 세션이 유지되도록 안정적인 서명 시크릿을 설정합니다.

```bash
# Compute a scrypt hash of your chosen password:
HASH=$(python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('choose-a-strong-password'))")

cat >> ~/.hermes/.env <<EOF
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=$HASH
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env
```

**2. 연결 가능한 주소에서 대시보드를 실행합니다.** 루프백이 아닌 주소에 바인딩하면 게이트가 활성화되고, 사용자 이름과 해시가 `basic` 제공자를 활성화합니다.

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. 로그인합니다.** `http://<host>:9119/`를 열면 `/login`으로 이동합니다. **자격 증명 양식**(“X로 로그인” 버튼이 아님)이 표시됩니다. `admin` / 비밀번호를 입력하면 인증된 대시보드로 이동합니다. 다음 명령으로 어느 컴퓨터에서든 게이트를 확인할 수 있습니다.

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

이후 `GET /api/auth/me`는 검증된 세션(`provider: basic`)을 반환합니다. 위 경고를 참고해 VPN 뒤에서 사용하세요. 퍼블릭 호스트에는 대신 [Nous Research](#default-provider-nous-research) 또는 [셀프 호스팅 OIDC](#self-hosted-oidc-provider) 제공자를 사용하세요.

#### 직접 비밀번호 제공자 작성

`basic`은 확장 지점의 한 구현일 뿐입니다. 모든 플러그인은 `DashboardAuthProvider` 서브클래스에 `supports_password = True`를 설정하고 `complete_password_login(*, username, password) -> Session`을 구현하여 비밀번호 제공자를 등록할 수 있습니다(거부 시 `InvalidCredentialsError`, 백업 저장소가 중단된 경우 `ProviderError` 발생). 순수 비밀번호 제공자라면 OAuth의 `start_login` / `complete_login` 메서드는 `NotImplementedError` 스텁으로 남겨 둘 수 있습니다. LDAP 바인드, 자격 증명 데이터베이스 또는 리디렉션을 사용하지 않는 다른 인증 방식에는 이 경로를 사용하세요. 프레임워크가 양식, 라우트, 쿠키 및 갱신을 대신 처리합니다.

### 셀프 호스팅 OIDC 제공자

자체 IDP를 운영한다면 번들 `plugins/dashboard_auth/self_hosted` 플러그인이 **표준 OpenID Connect**를 사용해 대시보드를 인증합니다. IDP별 코드가 필요 없고 Nous Portal도 사용하지 않습니다. 이 플러그인은 호환되는 모든 OIDC 서버에서 검증되며 작동합니다.

> **Authentik · Keycloak · Zitadel · Authelia · Auth0 · Okta · Google · …**

Nous 제공자와 마찬가지로 자동으로 로드되며, 구성된 경우에만 자신을 등록하므로 루프백 대시보드에서는 아무 작업도 하지 않습니다.

#### 구성

**issuer**와 **client_id**(클라이언트 시크릿이 없는 퍼블릭 PKCE 클라이언트)를 구성합니다. 플러그인은 `{issuer}/.well-known/openid-configuration`에서 IDP의 `authorization_endpoint`, `token_endpoint`, `jwks_uri`를 가져오므로 엔드포인트 URL을 직접 하드코딩할 필요가 없습니다.

**`config.yaml`** — 표준 설정 경로:

```yaml
dashboard:
  oauth:
    provider: self-hosted
    self_hosted:
      issuer: https://auth.example.com/application/o/hermes/   # required
      client_id: hermes-dashboard                              # required
      scopes: "openid profile email"                           # optional (this is the default)
```

**환경 변수** — 운영자 재정의(비어 있지 않은 환경 변수가 설정되면 env가 `config.yaml`보다 우선하며, 빈 값은 설정되지 않은 것으로 처리됨):

| 환경 변수 | 재정의 대상 | 비고 |
|---------|-----------|-------|
| `HERMES_DASHBOARD_OIDC_ISSUER` | `dashboard.oauth.self_hosted.issuer` | OIDC issuer URL — 필수 |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | `dashboard.oauth.self_hosted.client_id` | 퍼블릭 client id — 필수 |
| `HERMES_DASHBOARD_OIDC_SCOPES` | `dashboard.oauth.self_hosted.scopes` | 기본값은 `openid profile email` |

IDP에서 authorization-code + PKCE(S256) grant를 사용하는 **퍼블릭** 애플리케이션/클라이언트를 등록하고 대시보드의 콜백을 허용된 리디렉션 URI로 추가합니다. 콜백은 `<dashboard public URL>/auth/callback`입니다(프록시 뒤에서 대시보드가 퍼블릭 URL을 확인하는 방법은 [퍼블릭 URL 재정의](#public-url-override) 참조).
#### 검증되는 항목

Provider는 검색된 `jwks_uri`에 대해 OpenID Connect **ID token**(RS256/ES256)을 검증하며, `iss` 및 `aud` 클레임은 구성된 `issuer`와 `client_id`에 고정됩니다. 표준 OIDC 클레임은 대시보드 세션에 다음과 같이 매핑됩니다.

| 세션 필드 | 클레임(들) |
|---------------|----------|
| `user_id` | `sub` (필수) |
| `email` | `email` |
| `display_name` | `name` → `preferred_username` → `nickname` → `email` |
| `org_id` | `org_id` / `organization`, 없으면 가입된 `groups` |

ID token이 신원을 설정하는 데 사용되며, access token은 불투명 값으로 취급됩니다(OIDC 사양에서는 access token이 JWT여야 한다고 요구하지 않습니다). 엔드포인트 URL은 HTTPS여야 합니다(로컬 개발용 IDP에는 loopback `http://` 허용). 또한 discovery 문서에 명시된 `issuer`는 구성된 값과 일치해야 합니다(끝 슬래시의 차이는 허용). IDP가 refresh token을 발급하는 경우 표준 `refresh_token` grant를 사용한 자동 재인증에 사용하며, 로그아웃 시 IDP가 RFC 7009 `revocation_endpoint`를 알리는 경우 이를 호출합니다.

> **Confidential client**(`client_secret`이 있는 클라이언트)는 아직 지원되지 않습니다. 브라우저 기반 대시보드에서 일반적으로 사용하는 public + PKCE 클라이언트를 구성하세요.

#### 예제: Keycloak

[Keycloak](https://www.keycloak.org/)은 로컬 테스트용으로 구성하기 가장 쉬운 self-hosted OIDC 서버 중 하나입니다. 개발 모드에서 단일 컨테이너로 실행되고(인메모리 DB), 표준적인 OIDC discovery를 제공합니다. 이 안내를 따르면 몇 분 안에 대시보드 로그인을 사용할 수 있습니다.

**1. 미리 구성된 realm으로 Keycloak을 실행합니다.** 다음 realm export를 `realm-hermes.json`으로 저장하세요. 이 파일은 `hermes` realm, **public PKCE client**(`hermes-dashboard`), 테스트 사용자를 정의하며, 부팅 시 모두 가져오므로 admin UI에서 클릭할 항목이 없습니다.

```json
{
  "realm": "hermes",
  "enabled": true,
  "clients": [
    {
      "clientId": "hermes-dashboard",
      "name": "Hermes Agent Dashboard",
      "enabled": true,
      "publicClient": true,
      "standardFlowEnabled": true,
      "protocol": "openid-connect",
      "redirectUris": ["http://localhost:9119/auth/callback"],
      "webOrigins": ["http://localhost:9119"],
      "attributes": { "pkce.code.challenge.method": "S256" }
    }
  ],
  "users": [
    {
      "username": "testuser",
      "enabled": true,
      "emailVerified": true,
      "email": "testuser@example.com",
      "firstName": "Test",
      "lastName": "User",
      "credentials": [
        { "type": "password", "value": "testpassword", "temporary": false }
      ]
    }
  ]
}
```

해당 파일을 import 디렉터리에 마운트하여 실행합니다(Keycloak 26 이상).

```bash
docker run --rm -p 8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  -v "$PWD/realm-hermes.json:/opt/keycloak/data/import/realm-hermes.json:ro" \
  quay.io/keycloak/keycloak:26.0 \
  start-dev --import-realm
```

실행되면 realm은
`http://localhost:8080/realms/hermes/.well-known/openid-configuration`에서 표준 OIDC discovery를 알립니다(`http://localhost:8080/realms/hermes`가 issuer). admin console은
`http://localhost:8080/`에 있습니다(`admin` / `admin`).

**2. 대시보드가 해당 서버를 가리키도록 합니다.** self-hosted plugin은 loopback `http://` issuer를 허용하므로(그 외 issuer에는 HTTPS 필요) 로컬 Keycloak을 그대로 사용할 수 있습니다.

```bash
export HERMES_DASHBOARD_OIDC_ISSUER="http://localhost:8080/realms/hermes"
export HERMES_DASHBOARD_OIDC_CLIENT_ID="hermes-dashboard"
export HERMES_DASHBOARD_PUBLIC_URL="http://localhost:9119"
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

`HERMES_DASHBOARD_PUBLIC_URL`은 대시보드에 OAuth callback이
`http://localhost:9119/auth/callback`임을 알립니다. 이는 위 realm에 등록한 redirect URI입니다.
`0.0.0.0`에 바인딩하는 것(비-loopback 바인딩)이 OAuth gate를 활성화합니다.

**3. 로그인합니다.** `http://localhost:9119/`을 열면 `/login`으로 이동합니다. **Sign in with Self-Hosted OIDC**를 클릭하고 Keycloak에서 `testuser` / `testpassword`로 인증한 뒤 인증된 대시보드로 돌아옵니다. 사이드바에는 `Logged in as Test User via self-hosted`가 표시되고, `GET /api/auth/me`는 검증된 세션(`provider: self-hosted`, `email: testuser@example.com`)을 반환합니다.

> 다른 호스트/포트에 바인딩하거나 해당 환경에서 브라우즈하는 경우, Keycloak
> admin console의 클라이언트 **Valid redirect URIs**(Clients → hermes-dashboard → Settings)에 해당 origin의
> `…/auth/callback`을 추가하세요. 같은 방식이 Authentik, Zitadel, Authelia 및 기타 OIDC 서버에도 적용되며, issuer
> URL과 클라이언트 등록 UI만 다릅니다.

### Public URL 재정의

기본적으로 대시보드는 요청에서 OAuth callback URL을 재구성합니다. 즉 `X-Forwarded-Host` + `X-Forwarded-Proto` + `X-Forwarded-Prefix`를 사용합니다(`start_server`가 gate 활성화 시 `proxy_headers=True`로 uvicorn을 구성합니다). 세 헤더를 모두 올바르게 설정하는 reverse proxy 뒤에서는 별도 설정 없이 동작합니다.

이 헤더를 안정적으로 전달하지 않는 reverse proxy 뒤에 배포하는 경우(수동 nginx 설정, 온프레미스 ingress, 일부 proxy chain이 있는 custom-domain 배포), `dashboard.public_url`(또는 `HERMES_DASHBOARD_PUBLIC_URL`)을 대시보드에 접속하는 **완전한 public URL**로 설정하세요.

```yaml
dashboard:
  public_url: "https://dashboard.example.com/hermes"
```

설정하면 OAuth callback URL은 `<public_url>/auth/callback`이 그대로 됩니다. 이 경로에서는 운영자가 public URL을 명시적으로 선언했으므로 `X-Forwarded-Prefix`가 무시됩니다. 이미 `public_url`에 prefix가 포함된 일반적인 경우 prefix를 다시 붙이면 이중 prefix가 되기 때문이며, 이는 의도된 동작입니다.

다른 대시보드 설정과 우선순위가 같습니다. env가 `config.yaml`보다 우선합니다.

| 표면 | 재정의 경로 | 사용 시점 |
|---------|---------------|-------------|
| `config.yaml`의 `dashboard.public_url` | `HERMES_DASHBOARD_PUBLIC_URL` | 로컬 개발 / 온프레미스(권장) |
| `HERMES_DASHBOARD_PUBLIC_URL` env var | — | 호스팅 플랫폼 secret / CI |
| (설정되지 않음) | — | 기본값 — `X-Forwarded-*` 헤더에서 재구성 |

검증에서는 `http://` / `https://` scheme이 없거나, host가 없거나, quote / angle / whitespace / control character가 포함된 값을 거부합니다. 잘못된 값은 로그인 흐름이 중단되어 사용자가 악성 URL로 이동하는 대신 헤더 재구성으로 조용히 대체됩니다.

> **참고:** `public_url`은 OAuth callback URL만 재정의합니다. `Secure` cookie flag는 여전히 `request.url.scheme`으로 제어됩니다(proxy_headers 사용 시 X-Forwarded-Proto). 따라서 TLS가 종료되는 public 배포에서 `http://` `public_url`을 사용하면 Secure가 아닌 cookie가 생성됩니다. 이는 운영상의 함정이므로 `public_url`을 upstream의 적절한 TLS 종료와 함께 사용하세요.

### OAuth flow

Provider는 [Nous Portal OAuth contract v1](https://github.com/NousResearch/nous-account-service/blob/main/docs/agent-dashboard-oauth-contract.md), 즉 PKCE(S256)를 사용하는 authorization-code grant를 구현합니다.

1. 세션 cookie 없이 `/`에 접속한 사용자는 gate에 의해 `/login`으로 리디렉션됩니다.
2. 로그인 페이지에 "Continue with Nous Research" 버튼이 표시되고 → `/auth/login?provider=nous`로 이동합니다.
3. 서버가 단기간 유효한 cookie에 PKCE state를 저장하고 사용자를 `https://portal.nousresearch.com/oauth/authorize?…`로 리디렉션합니다.
4. 사용자가 Portal에서 인증하면 `/auth/callback?code=…&state=…`에 도착합니다.
5. 서버가 `POST /api/oauth/token`에서 code를 access token으로 교환하고, Portal의 JWKS(`/.well-known/jwks.json`)에 대해 JWT signature를 검증한 다음 `hermes_session_at` cookie를 설정합니다.
6. 사용자가 `/`로 리디렉션됩니다(또는 `next=` query parameter를 통해 원래의 deep-link path로 이동합니다).

Access token의 TTL은 15분입니다. **contract v1에는 refresh token이 없습니다.** token이 만료되면 SPA의 fetch wrapper가 401 envelope를 감지하고 전체 페이지를 `/login`으로 이동시켜 흐름을 다시 실행합니다.

### 설정되는 Cookies

| 이름 | 수명 | 비고 |
|------|----------|-------|
| `hermes_session_at` | Token TTL (15 min) | HttpOnly, SameSite=Lax, Secure-when-HTTPS |
| `hermes_session_pkce` | 10 min | HttpOnly; 왕복 과정에서 PKCE verifier + provider hint를 보관 |
| `hermes_session_rt` | v1에서 사용하지 않음 | 향후 호환성을 위해 예약됨; `refresh_token`이 비어 있으면 기록되지 않음 |

세 cookie 모두 `Path=/` 및 `SameSite=Lax`입니다. 대시보드에 HTTPS로 접속할 때 `Secure` flag가 설정됩니다(request URL scheme으로 감지하며, `proxy_headers=True`일 때 upstream TLS terminator의 `X-Forwarded-Proto`를 따릅니다).

### 로그아웃

사이드바 widget에는 `Logged in as <user_id…> via nous`와 로그아웃 아이콘이 표시됩니다. 이를 클릭하면 `/auth/logout`에 POST하여 모든 대시보드 인증 cookie를 삭제하고 `/login`으로 리디렉션합니다.

### Audit log

모든 로그인 시작, 성공, 실패 및 세션 검증 실패는 `$HERMES_HOME/logs/dashboard-auth.log`에 JSON line으로 기록됩니다. 민감한 필드(`access_token`, `refresh_token`, `code`, `code_verifier`, `state`, `Authorization` header)는 기록 전에 삭제됩니다.

### Custom providers

Nous가 아닌 OAuth provider(예: Google, GitHub, custom OIDC)를 연결하려면 `DashboardAuthProvider`를 등록하는 plugin을 만드세요.

```python
# ~/.hermes/plugins/dashboard-auth-myidp/__init__.py
from hermes_cli.dashboard_auth import DashboardAuthProvider, Session, LoginStart

class MyIdPProvider(DashboardAuthProvider):
    name = "myidp"
    display_name = "My Identity Provider"

    def start_login(self, *, redirect_uri): ...
    def complete_login(self, *, code, state, code_verifier, redirect_uri): ...
    def verify_session(self, *, access_token): ...
    def refresh_session(self, *, refresh_token): ...
    def revoke_session(self, *, refresh_token): ...

def register(ctx):
    ctx.register_dashboard_auth_provider(MyIdPProvider())
```

로그인 페이지에는 등록된 모든 provider가 나열되며, 여러 provider를 함께 구성할 수 있습니다. 사용자는 `/login`에서 하나를 선택합니다.

### 비대화형(bearer-token) 인증

대화형 human login(session cookie + refresh)과 함께 `DashboardAuthProvider` ABC는 `supports_token = True` + `verify_token(token=...)`을 통한 **비대화형 service-to-service** 기능을 지원합니다. Provider가 이를 선택하면 수신한 `Authorization: Bearer <token>`을 검증하고, 성공 시 해당 provider가 token-authable로 표시한 endpoint에 대해 `TokenPrincipal`을 요청에 연결합니다(`request.state.token_principal`). cookie, redirect, refresh는 사용하지 않습니다.

번들에 포함된 첫 번째 consumer는 **drain** provider(`plugins/dashboard_auth/drain`)입니다. `nous-account-service`가 `HERMES_DASHBOARD_DRAIN_SECRET`을 통해 agent별 secret을 provision하면, provider는 constant-time compare로 수신 bearer token을 검증하고 `/api/gateway/drain`을 token-authable로 등록합니다. 이 기능은 **fail closed** 방식으로 동작합니다. 약하거나 짧은 secret(< 256 bits)은 등록 시 거부되고 endpoint는 비활성 상태로 유지되며, env var가 설정되지 않으면 아무 작업도 하지 않습니다. 동작 설정(`scope`, `min_secret_chars`)은 `config.yaml`의 `dashboard.drain_auth` 아래에 있습니다.

Custom provider도 같은 방식으로 `supports_token`/`verify_token`을 구현하여 자체 machine-authable endpoint를 노출할 수 있습니다.

### Gate가 켜져 있는지 확인하기

```bash
# Quick env-var path.
HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:test \
  hermes dashboard --host 0.0.0.0

# Or the equivalent via config.yaml (recommended for local dev / on-prem):
#
#   dashboard:
#     oauth:
#       client_id: agent:test
#
# then just:
hermes dashboard --host 0.0.0.0

# Hit /api/status to see the gate state:
curl -s http://127.0.0.1:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

대시보드의 React StatusPage는 "Web server" 아래에 같은 필드를 표시합니다. 로그인하면 사이드바 AuthWidget에 현재 identity가 표시됩니다.

## Hermes Desktop을 원격 backend에 연결하기

Hermes Desktop은 다른 머신에서 실행 중인 Hermes backend(VPS, home server, Tailscale 뒤의 Mini)를 제어할 수 있습니다. 앱에서는 **Settings → Gateway → Remote gateway**에 있으며, 여기서 **Remote URL**과 **Sign in** 방법을 요청합니다. (Desktop 앱 자체의 설치, 설정, chat은 [Hermes Desktop](/user-guide/desktop) 페이지를 참고하세요.)

번들 auth provider 중 하나로 원격 대시보드를 보호하면 Desktop 앱은 backend가 알리는 provider를 사용해 로그인합니다. 자신의 머신 밖에서 접근할 수 있는 backend(VPS, public host, 인터넷에 노출된 모든 대상)의 경우 권장 provider는 **OAuth (Nous Portal)**입니다. ([`hermes dashboard register`](#registering-a-dashboard)로 등록한 뒤 *Sign in with Nous Research*로 로그인하세요.) 번들로 제공되는 [username/password provider](#usernamepassword-provider-no-oauth-idp)는 backend가 신뢰할 수 있는 LAN에 있거나 VPN으로만 접근 가능한 경우 가장 빠른 선택이지만, **public internet에 직접 노출하기에는 적합하지 않습니다**. 대시보드를 non-loopback address에 바인딩하면 auth gate가 활성화되며, 로그인 후 Desktop은 chat WebSocket에 세션을 자동으로 재사용합니다. 복사하거나 붙여 넣을 token은 없습니다.

아래 레시피는 신뢰할 수 있는 네트워크에서 가장 빠르게 구성할 수 있는 username/password 경로를 사용합니다. OAuth 경로는 [Default provider: Nous Research](#default-provider-nous-research)를 참고하세요.
### 백엔드에서(원격 머신)

```bash
# 1. Set the dashboard login credentials in ~/.hermes/.env (secrets file, 0600).
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# Recommended: a stable signing secret so sessions survive restarts.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. Run the dashboard bound to a reachable address. The non-loopback bind
#    engages the auth gate; the username/password provider handles login.
hermes dashboard --no-open --host 0.0.0.0 --port 9119
```

평문으로 저장하지 않으려면 scrypt 해시와 함께 `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH`를 사용하세요. 전체 설정은 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp)를 참조하세요.

대시보드를 systemd 서비스로 실행하는 경우, 해당 유닛에 `EnvironmentFile=%h/.hermes/.env`가 있으면 `~/.hermes/.env`가 자동으로 로드되므로 부팅 시 자격 증명이 환경에 설정됩니다.

:::warning
대시보드는 사용자의 `.env`(API 키, 비밀)를 읽고 쓸 수 있으며 에이전트 명령을 실행할 수 있습니다. 여기서 설명하는 **사용자 이름/비밀번호** 설정은 신뢰할 수 있는 네트워크를 위한 것입니다. 비밀번호로 보호된 대시보드를 개방된 인터넷에 직접 노출하지 마세요. VPN 뒤에 배치하세요. [Tailscale](https://tailscale.com/)이 깔끔한 선택입니다. 머신의 tailscale IP(`--host <tailscale-ip>`)에 바인딩하고 Remote URL에 `http://<tailscale-ip>:9119`를 사용하세요. tailnet에 있는 기기만 연결할 수 있습니다. 공용 인터넷을 통해 백엔드에 연결하려면 대신 **OAuth (Nous Portal)** 제공자를 사용하세요.
:::

### Hermes Desktop에서

**설정 → Gateway → 원격 게이트웨이:**

- **Remote URL** — `http://<backend-host>:9119` (리버스 프록시 앞에 배치하는 경우 `/hermes` 같은 경로 접두사를 지원합니다.)
- **Sign in** — 앱이 사용자 이름/비밀번호 게이트웨이를 감지하고 **Sign in** 버튼을 표시합니다. 버튼을 클릭한 뒤 1단계에서 설정한 자격 증명을 입력하세요.
- **Save and reconnect** — 데스크톱 셸을 원격 백엔드로 전환합니다.

백엔드에 `HERMES_DASHBOARD_BASIC_AUTH_SECRET`가 설정되어 있으면 세션이 자동으로 갱신되며 재시작 후에도 유지됩니다.

### 환경 변수 재정의

앱 내 설정 대신, 실행 전에 환경 변수를 설정해 데스크톱이 백엔드를 가리키도록 할 수 있습니다. `HERMES_DESKTOP_REMOTE_URL`이 설정되면 저장된 앱 내 URL보다 우선하며, Gateway 설정 패널에 "env override" 배지가 표시되고 편집이 비활성화됩니다. 패널에서 사용자 이름과 비밀번호로 계속 **Sign in**해야 합니다.

| 환경 변수 | 값 |
|---------|-------|
| `HERMES_DESKTOP_REMOTE_URL` | `http://<backend-host>:9119` |

### 문제 해결

- **"Remote gateway incomplete"** — 원격 URL을 입력하지 않았습니다.
- **401 / "Invalid credentials"와 함께 로그인이 실패함** — 사용자 이름 또는 비밀번호가 백엔드의 `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`와 일치하지 않습니다. 백엔드는 존재하지 않는 사용자와 잘못된 비밀번호에 동일한 일반 오류를 반환하므로 둘 다 확인하세요. `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'`로 게이트를 확인하세요. `true`가 보고되고 `"basic"`이 포함되어야 합니다.
- **"Sign in" 버튼이 없고 대신 세션 토큰을 요청함** — 사용자 이름/비밀번호 제공자가 활성화되지 않았습니다(`/api/status`에 `"basic"`이 표시되지 않음). 사용자 이름과 비밀번호(또는 비밀번호 해시)가 설정되어 있고 대시보드 프로세스가 이를 로드했는지 확인하세요.
- **재시작할 때마다 로그아웃됨** — `HERMES_DASHBOARD_BASIC_AUTH_SECRET`을 안정적인 값으로 설정하세요. 그렇지 않으면 부팅할 때마다 서명 키가 새로 생성됩니다.
- **연결이 거부되거나 시간 초과됨** — 백엔드가 연결 가능한 주소가 아니라 기본값인 `127.0.0.1`에 바인딩되었거나, 방화벽/VPN이 포트를 차단하고 있습니다. `0.0.0.0` 또는 tailscale IP에 바인딩하고 신뢰할 수 있는 네트워크에 포트를 개방하세요.

## CORS

웹 서버는 CORS를 localhost 오리진으로만 제한합니다.

- `http://localhost:9119` / `http://127.0.0.1:9119` (프로덕션)
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173` (Vite 개발 서버)

사용자 지정 포트에서 서버를 실행하면 해당 오리진이 자동으로 추가됩니다.

## 개발

웹 대시보드 프런트엔드에 기여하는 경우:

```bash
# Terminal 1: start the backend API
hermes dashboard --no-open

# Terminal 2: start the Vite dev server with HMR
cd web/
npm install
npm run dev
```

`http://localhost:5173`의 Vite 개발 서버는 `/api` 요청을 FastAPI 백엔드 `http://127.0.0.1:9119`로 프록시합니다.

프런트엔드는 React 19, TypeScript, Tailwind CSS v4, shadcn/ui 스타일 컴포넌트로 구축되었습니다. 프로덕션 빌드 결과물은 FastAPI 서버가 정적 SPA로 제공하는 `hermes_cli/web_dist/`에 출력됩니다.

## 업데이트 시 자동 빌드

`hermes update`를 실행하면 `npm`을 사용할 수 있는 경우 웹 프런트엔드가 자동으로 다시 빌드됩니다. 이를 통해 대시보드가 코드 업데이트와 동기화됩니다. `npm`이 설치되어 있지 않으면 업데이트 시 프런트엔드 빌드를 건너뛰며, `hermes dashboard`가 처음 실행될 때 빌드합니다.

## 테마 및 플러그인

대시보드에는 기본 제공 테마 8개가 포함되어 있으며, 사용자 정의 테마, 플러그인 탭, 백엔드 API 라우트를 추가해 확장할 수 있습니다. 저장소를 복제할 필요 없이 모두 바로 사용할 수 있습니다.

**헤더 바에서 테마를 실시간으로 전환**하려면 언어 전환기 옆의 팔레트 아이콘을 클릭하세요. 선택한 테마는 `config.yaml`의 `dashboard.theme`에 저장되며 페이지를 불러올 때 복원됩니다.

**같은 선택기에서 글꼴을 독립적으로 변경**할 수도 있습니다. 테마 목록 아래의 **Font** 섹션은 현재 활성화된 테마의 UI 글꼴을 재정의합니다. 선택 사항은 테마를 전환해도 유지됩니다(`config.yaml` → `dashboard.font`). 이를 지우고 활성 테마 자체의 글꼴로 돌아가려면 **Theme default**를 선택하세요.

기본 제공 테마:

| 테마 | 특징 |
|-------|-----------|
| **Hermes Teal** (`default`) | 어두운 틸색 + 크림색, 시스템 글꼴, 편안한 간격 |
| **Hermes Teal (Large)** (`default-large`) | 18px 텍스트와 더 넉넉한 간격을 적용한 기본 테마 |
| **Nous Blue** (`nous-blue`) | 여유 있는 간격과 Nous 브랜드 파란색 강조 |
| **Midnight** (`midnight`) | 짙은 청색-보라색, Inter + JetBrains Mono |
| **Ember** (`ember`) | 따뜻한 진홍색 + 청동색, Spectral 세리프 + IBM Plex Mono |
| **Mono** (`mono`) | 회색조, IBM Plex, 간결한 구성 |
| **Cyberpunk** (`cyberpunk`) | 검은색 바탕의 네온 그린, Share Tech Mono |
| **Rosé** (`rose`) | 분홍색 + 아이보리색, Fraunces 세리프, 넉넉한 공간 |

직접 테마를 만들거나 플러그인 탭을 추가하거나 셸 슬롯에 삽입하거나 플러그인 전용 REST 엔드포인트를 노출하려면 **[대시보드 확장](./extending-the-dashboard)**을 참조하세요. 이 완전한 가이드에서는 다음을 다룹니다.

- 테마 YAML 스키마 — palette, typography, layout, assets, componentStyles, colorOverrides, customCSS
- 레이아웃 변형 — `standard`, `cockpit`, `tiled`
- 플러그인 매니페스트, SDK, 셸 슬롯, 페이지 범위 슬롯(기본 제공 페이지를 재정의하지 않고 위젯 삽입), 백엔드 FastAPI 라우트
- 테마와 플러그인을 결합한 전체 단계별 안내(Strike Freedom cockpit 데모)
- 검색, 다시 불러오기, 문제 해결

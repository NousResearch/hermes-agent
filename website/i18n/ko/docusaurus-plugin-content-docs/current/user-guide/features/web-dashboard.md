---
sidebar_position: 15
title: "웹 대시보드"
description: "구성, API 키, MCP 서버, 메시징 페어링, 웹훅, 게이트웨이, 메모리, 자격 증명, 세션, 로그, 분석, 크론 작업 및 스킬 관리를 위한 브라우저 기반 관리 패널"
---

# 웹 대시보드 (Web Dashboard)

웹 대시보드는 Hermes 에이전트 설치 환경을 관리하기 위한 브라우저 기반 UI입니다. YAML 파일을 편집하거나 CLI 명령어를 실행하는 대신, 깔끔한 웹 인터페이스에서 설정을 구성하고 API 키를 관리하며 세션을 모니터링할 수 있습니다.

:::tip
호스팅 모드 인증은 Nous Portal OAuth를 사용합니다. 대시보드가 실제 백엔드와 통신하도록 하려면, `hermes setup --portal`을 실행하여 모델과 도구 게이트웨이도 함께 연결하세요. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 빠른 시작 (Quick Start)

```bash
hermes dashboard
```

이 명령어는 로컬 웹 서버를 시작하고 브라우저에서 `http://127.0.0.1:9119`를 엽니다. 대시보드는 전적으로 로컬 컴퓨터에서만 실행되며 — 데이터가 localhost 밖으로 나가지 않습니다.

### 옵션 (Options)

| 플래그 | 기본값 | 설명 |
|------|---------|-------------|
| `--port` | `9119` | 웹 서버를 실행할 포트 |
| `--host` | `127.0.0.1` | 바인딩할 주소 |
| `--no-open` | — | 브라우저를 자동으로 열지 않음 |
| `--insecure` | 꺼짐 | localhost 이외의 호스트에 바인딩 허용 (**위험** — 네트워크에 API 키가 노출됩니다; 방화벽 및 강력한 인증과 함께 사용하세요) |

```bash
# 사용자 지정 포트
hermes dashboard --port 8080

# 모든 인터페이스에 바인딩 (공유 네트워크에서는 주의해서 사용하세요)
hermes dashboard --host 0.0.0.0

# 브라우저 열지 않고 시작
hermes dashboard --no-open
```

## 요구 사항 (Prerequisites)

기본 `hermes-agent` 설치에는 HTTP 스택이나 PTY 헬퍼가 포함되어 있지 않으며 — 이는 선택적 추가 기능입니다. **웹 대시보드**에는 FastAPI와 Uvicorn(`web` 옵션)이 필요합니다. **Chat** 탭 또한 가상 터미널(pseudo-terminal) 뒤에 내장된 TUI를 생성하기 위해 `ptyprocess`가 필요합니다(POSIX의 `pty` 옵션). 다음 명령어로 둘 다 설치하세요:

```bash
pip install 'hermes-agent[web,pty]'
```

`web` 옵션은 FastAPI/Uvicorn을 가져오고; `pty` 옵션은 `ptyprocess`(POSIX) 또는 `pywinpty`(네이티브 Windows — 내장된 TUI 자체는 여전히 WSL이 필요함에 유의)를 가져옵니다. `pip install hermes-agent[all]`은 두 옵션을 모두 포함하며 메시징/음성 등을 원할 때 가장 쉬운 경로입니다.

의존성 없이 `hermes dashboard`를 실행하면 설치해야 할 내용을 알려줍니다. 프론트엔드가 아직 빌드되지 않았고 `npm`을 사용할 수 있는 경우 첫 실행 시 자동으로 빌드됩니다.

Chat 탭은 모든 `hermes dashboard` 실행에 포함되어 있습니다 — (PTY/WebSocket을 통해 TUI를 실행하는) 내장된 브라우저 채팅 창은 추가 플래그 없이 항상 사용할 수 있습니다.

## 페이지 안내 (Pages)

### 상태 (Status)

랜딩 페이지는 설치 환경의 실시간 개요를 보여줍니다:

- **에이전트 버전** 및 릴리스 날짜
- **게이트웨이 상태** — 실행/중지됨, PID, 연결된 플랫폼 및 해당 상태
- **활성 세션** — 지난 5분 동안 활성화된 세션 수
- **최근 세션** — 모델, 메시지 수, 토큰 사용량 및 대화 미리보기가 포함된 가장 최근 세션 20개 목록

상태 페이지는 5초마다 자동 새로 고침됩니다.

### 채팅 (Chat)

**Chat** 탭은 전체 Hermes TUI(`hermes --tui`에서 얻는 것과 동일한 인터페이스)를 브라우저에 직접 포함합니다. 터미널 TUI에서 할 수 있는 모든 작업 — 슬래시 명령어, 모델 선택기, 도구 호출 카드, 마크다운 스트리밍, 명확화(clarify)/sudo/승인 프롬프트, 스킨 테마 설정 — 이 여기서도 동일하게 작동합니다. 왜냐하면 대시보드가 실제 TUI 바이너리를 실행하고 WebGL 렌더러가 포함된 [xterm.js](https://xtermjs.org/)를 통해 그 ANSI 출력을 픽셀 수준의 완벽한 셀 레이아웃으로 렌더링하기 때문입니다.

**작동 방식:**

- `/api/pty`는 대시보드의 세션 토큰으로 인증된 WebSocket을 엽니다.
- 서버는 POSIX 가상 터미널 뒤에서 `hermes --tui`를 생성합니다.
- 키보드 입력이 PTY로 전달되고, ANSI 출력이 브라우저로 스트리밍되어 돌아옵니다.
- xterm.js의 WebGL 렌더러는 각 셀을 정수 픽셀 그리드에 그립니다; 마우스 추적(SGR 1006), 넓은 문자(Unicode 11) 및 상자 그리기 문자(box-drawing glyphs)가 모두 네이티브로 렌더링됩니다.
- 브라우저 창의 크기를 조정하면 `@xterm/addon-fit` 애드온을 통해 TUI 크기가 조정됩니다.

**기존 세션 재개:** **Sessions** 탭에서 세션 옆의 재생 아이콘(▶)을 클릭하세요. 그러면 `/chat?resume=<id>`로 이동하고 `--resume`과 함께 TUI를 시작하여 전체 기록을 로드합니다.

**요구 사항:**

- Node.js (`hermes --tui`와 동일한 요구 사항; 첫 실행 시 TUI 번들이 빌드됨)
- `ptyprocess` — `pty` 옵션으로 설치됨 (`pip install 'hermes-agent[web,pty]'` 또는 `[all]`이 둘 다 포함함)
- POSIX 커널(Linux, macOS 또는 WSL2). `/chat` 터미널 창은 특별히 POSIX PTY를 필요로 합니다 — 네이티브 Windows Python에는 동등한 기능이 없으므로, 네이티브 Windows 설치의 경우 대시보드의 나머지(세션, 작업, 지표, 구성 편집기)는 작동하지만 `/chat` 탭에는 해당 기능을 위해 WSL2를 사용하라는 배너가 표시됩니다.

브라우저 탭을 닫으면 서버에서 PTY가 깨끗하게 수거(reap)됩니다. 다시 열면 새 세션이 생성됩니다.

[Hermes Desktop](#connecting-hermes-desktop-to-a-remote-backend)이 자체 내장 백엔드 대신 다른 기기에서 실행 중인 대시보드를 가리키도록 하려면 아래의 원격 백엔드 섹션을 참조하세요.

### 원격 백엔드에 Hermes Desktop 연결 (Connecting Hermes Desktop to a remote backend)

Hermes Desktop은 일반적으로 자체 로컬 백엔드를 실행하지만, **Settings → Gateway → Remote gateway**를 통해 원격 머신(VM, 홈랩 서버 등)에서 실행 중인 대시보드에 연결할 수도 있습니다. 이것은 "Desktop은 백엔드가 준비되었다고 하는데 채팅이 전혀 작동하지 않아요"라는 보고가 나오는 가장 흔한 원인입니다. 왜냐하면 Desktop의 준비 상태 확인(readiness check)은 라이브 채팅 연결에 실제로 필요한 것보다 더 적은 항목만 검증하기 때문입니다.

:::info 사전 요구 사항: 원격 호스트에서 `hermes dashboard`가 실행 중이어야 합니다
Desktop이 연결되는 "원격 백엔드"는 이 문서에서 설명하는 것과 동일한 서버, 즉 원격 머신에서 실행 중인 **`hermes dashboard`** 프로세스입니다. 아래의 모든 단계를 진행하기 전에 대시보드가 먼저 실행 중이고 연결 가능해야 합니다; Desktop은 대시보드에 연결하는 것이지 대신 시작해 주지 않습니다. 로그아웃과 재부팅 후에도 계속 작동할 수 있도록 `systemd`/`tmux` 등의 관리 도구를 이용해 백엔드를 실행 상태로 유지하세요. **게이트웨이**(Telegram/Discord/Slack 등)는 이와 *별개인* 장기 실행 프로세스입니다 — 메시징 채널을 사용하신다면 독립적으로 시작하세요; 게이트웨이는 데스크톱 앱이 연결하는 대상이 아닙니다.
:::

Desktop의 "원격 백엔드가 준비됨" 확인용 프로브는 공개 엔드포인트인 `GET /api/status`만 호출합니다. 이 엔드포인트는 호스트에서 대시보드가 *어떻게든* 실행되고 있기만 하면 바로 응답합니다. 하지만 라이브 채팅 연결은 `/api/ws`(및 `/api/pty`)에 대한 **별도의** WebSocket 연결이며, 이 소켓은 상태 프로브가 확인하지 않는 두 가지 추가 확인 절차를 거칩니다:

1. **인증이 필요합니다.** 대시보드가 루프백이 아닌 주소에 바인딩되면 인증 게이트가 작동합니다. 사용자 이름과 비밀번호로 대시보드를 보호하세요(내장된 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp) 사용); Desktop은 한 번 로그인한 뒤, 1회용 티켓을 통해 얻은 세션을 WebSocket용으로 재사용합니다. 구성된 제공자가 없으면 루프백이 아닌 대시보드는 **시작 시 안전하게 닫힌 상태로 실패(fail closed)** 합니다.
2. **바인딩 호스트는 클라이언트를 허용하고 Host 헤더와 일치해야 합니다.** 루프백 바인딩(`127.0.0.1`)은 루프백 클라이언트만 수용하므로 자격 증명에 관계없이 소켓 계층에서 원격 머신의 접근을 거부합니다. 다른 클라이언트(피어 IP)가 접속할 수 있도록 루프백이 아닌 주소(`--host 0.0.0.0`)에 바인딩하세요. 또한 Desktop에 입력하는 원격 URL은 대시보드가 바인딩된 호스트와 동일한 경로로 접근해야 합니다 — DNS 리바인딩 방어 정책으로 인해 Host 헤더가 일치해야 하기 때문입니다.

#### 원격 대시보드 설정

사용자 이름과 비밀번호를 설정한 다음, 도달 가능한 주소에 바인딩된 대시보드를 실행하세요. `systemd` 서비스의 경우 다음과 같이 작성합니다:

```ini
[Service]
EnvironmentFile=%h/.hermes/.env
ExecStart=/path/to/venv/bin/python -m hermes_cli.main dashboard \
    --host 0.0.0.0 --port 9119 --no-open
```

`~/.hermes/.env` 파일 내용:

```bash
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
HERMES_DASHBOARD_BASIC_AUTH_SECRET=<32바이트 이상의 무작위 값; openssl rand -base64 32>
```

그런 다음 Desktop에서 **Remote URL**(예: `http://VM_IP:9119`)을 입력하고 앞서 설정한 사용자 이름과 비밀번호로 **Sign in**하세요. 자세한 설정 항목은 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp) 섹션을 참조하세요.

:::tip Desktop에서 다시 시도하기 전에 게이트가 켜져 있는지 확인하세요
아무 머신에서나 대시보드가 사용자 이름/비밀번호 제공자를 알리고 있는지 확인하세요:

```bash
curl -s http://VM_IP:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

- `auth_required: true` 이고 제공자 목록에 `"basic"`이 있음 → Desktop의 **Sign in** 절차가 정상적으로 작동합니다.
- `auth_required: false` → 바인딩이 루프백이거나 게이트가 작동하지 않았습니다. 루프백이 아닌 주소에 바인딩하세요.
- `auth_required: true` 이지만 `"basic"` 제공자가 없음 → 사용자 이름/비밀번호 환경 변수가 로드되지 않았습니다. 이것부터 먼저 해결하세요.
:::

`/api/status`에서 `"basic"` 제공자와 함께 게이트가 켜져 있음을 보여주고, Desktop에서 로그인한 후에도 *여전히* 연결에 실패한다면 문제는 기본 설정을 넘어선 것입니다 — 해당 재시도 기간의 새로운 `desktop.log` (Settings → Gateway → Open logs)와 대시보드 로그를 확보하고 `/api/ws` 종료 코드(close code)를 확인하세요 (4403 = 요청 가드가 채팅 WS 거부, 예: 호스트/피어 불일치; 4401 = WS 티켓이 인증되지 않음).

### 구성 (Config)

`config.yaml`을 위한 양식(form) 기반 편집기입니다. 150개가 넘는 모든 구성 필드가 `DEFAULT_CONFIG`에서 자동 발견되어 탭 형식의 카테고리로 구성됩니다:

- **model** — 기본 모델, 제공자, 기본 URL, 추론 설정
- **terminal** — 백엔드 (local/docker/ssh/modal), 시간 초과, 쉘 기본 설정
- **display** — 스킨, 도구 진행 상황, 재개 표시, 스피너 설정
- **agent** — 최대 반복 횟수, 게이트웨이 시간 초과, 서비스 티어
- **delegation** — 서브에이전트 제한, 추론 수준(reasoning effort)
- **memory** — 제공자 선택, 컨텍스트 주입 설정
- **approvals** — 위험한 명령어 승인 모드 (ask/yolo/deny)
- 기타 — config.yaml의 모든 섹션에 해당하는 양식 필드가 있습니다.

알려진 유효한 값(터미널 백엔드, 스킨, 승인 모드 등)이 있는 필드는 드롭다운으로 렌더링됩니다. 부울(Boolean)은 토글 버튼으로 렌더링됩니다. 그 외의 모든 것은 텍스트 입력입니다.

**동작:**

- **Save (저장)** — 변경 사항을 `config.yaml`에 즉시 기록합니다.
- **Reset to defaults (기본값으로 재설정)** — 모든 필드를 기본값으로 되돌립니다. (Save를 클릭하기 전까지는 저장되지 않음)
- **Export (내보내기)** — 현재 구성을 JSON으로 다운로드합니다.
- **Import (가져오기)** — JSON 구성 파일을 업로드하여 현재 값을 바꿉니다.

:::tip
구성 변경 사항은 다음 에이전트 세션이나 게이트웨이 재시작 시에 적용됩니다. 웹 대시보드는 `hermes config set` 및 게이트웨이가 읽어 들이는 것과 동일한 `config.yaml` 파일을 편집합니다.
:::

### API 키 (API Keys)

API 키 및 자격 증명이 저장되는 `.env` 파일을 관리합니다. 키는 카테고리별로 그룹화됩니다:

- **LLM 제공자** — OpenRouter, Anthropic, OpenAI, DeepSeek 등.
- **도구 API 키** — Browserbase, Firecrawl, Tavily, ElevenLabs 등.
- **메시징 플랫폼** — Telegram, Discord, Slack 봇 토큰 등.
- **에이전트 설정** — `API_SERVER_ENABLED`와 같은 비밀이 아닌 환경 변수.

각 키는 다음 내용을 표시합니다:
- 현재 설정되어 있는지 여부 (가려진 값 미리보기와 함께)
- 이 키의 용도에 대한 설명
- 제공자의 가입/키 페이지 링크
- 값을 설정하거나 업데이트하기 위한 입력 필드
- 값을 제거하기 위한 삭제 버튼

고급/자주 사용되지 않는 키는 기본적으로 토글 뒤에 숨겨져 있습니다.

### 세션 (Sessions)

모든 에이전트 세션을 찾아보고 검사합니다. 각 행에는 세션 제목, 소스 플랫폼 아이콘(CLI, Telegram, Discord, Slack, cron), 모델 이름, 메시지 수, 도구 호출 수 및 마지막으로 활성화된 시간이 표시됩니다. 라이브 세션은 깜박이는 배지로 표시됩니다.

- **Search (검색)** — FTS5를 사용하여 모든 메시지 내용에 대한 전체 텍스트 검색을 지원합니다. 결과에는 강조 표시된 스니펫이 표시되며 확장 시 첫 번째로 일치하는 메시지로 자동 스크롤됩니다.
- **Stats (통계)** — 요약 표시줄에 총 세션 수, 저장소에서 활성화된 수, 보관된 수, 총 메시지 수 및 소스별 내역이 표시됩니다.
- **Expand (펼치기)** — 세션을 클릭하면 전체 메시지 기록이 로드됩니다. 메시지는 역할(user, assistant, system, tool)에 따라 색상으로 구분되며 구문 강조가 적용된 Markdown으로 렌더링됩니다.
- **Tool calls (도구 호출)** — 도구 호출이 포함된 assistant 메시지는 함수 이름과 JSON 인수가 포함된 접을 수 있는 블록으로 표시됩니다.
- **Rename (이름 바꾸기)** — 연필 아이콘을 클릭하여 세션 제목을 바로 설정하거나 지울 수 있습니다.
- **Export (내보내기)** — 세션(메타데이터 + 전체 메시지 기록)을 JSON으로 다운로드합니다(다운로드 아이콘).
- **Prune (정리)** — 헤더의 "Prune old sessions" 버튼을 사용하여 N일 이상 된 종료된 세션을 삭제합니다.
- **Delete (삭제)** — 휴지통 아이콘을 클릭하여 세션과 해당 메시지 기록을 제거합니다.

![Sessions admin page — stats bar, prune, and per-row rename / export / delete](/img/dashboard/admin-sessions.png)

### 로그 (Logs)

필터링 및 실시간 테일링(live tailing) 기능과 함께 에이전트, 게이트웨이 및 오류 로그 파일을 확인합니다.

- **File (파일)** — `agent`, `errors`, `gateway` 로그 파일 간에 전환합니다.
- **Level (수준)** — 로그 수준(ALL, DEBUG, INFO, WARNING 또는 ERROR)별로 필터링합니다.
- **Component (컴포넌트)** — 소스 컴포넌트(all, gateway, agent, tools, cli 또는 cron)별로 필터링합니다.
- **Lines (줄 수)** — 표시할 줄 수(50, 100, 200 또는 500)를 선택합니다.
- **Auto-refresh (자동 새로고침)** — 5초마다 새 로그 라인을 폴링하는 라이브 테일링을 토글합니다.
- **Color-coded (색상 구분)** — 심각도에 따라 로그 라인에 색상이 지정됩니다(오류는 빨간색, 경고는 노란색, 디버그는 희미한 색).

### 분석 (Analytics)

세션 기록에서 계산된 사용량 및 비용 분석입니다. 기간(7일, 30일 또는 90일)을 선택하여 다음을 확인할 수 있습니다:

- **요약 카드** — 총 토큰(입력/출력), 캐시 적중률, 총 예상/실제 비용, 일일 평균을 포함한 총 세션 수.
- **일일 토큰 차트** — 일별 입력 및 출력 토큰 사용량을 보여주는 누적 막대 차트. 마우스오버 시 세부 정보와 비용이 툴팁으로 표시됩니다.
- **일별 내역 표** — 날짜, 세션 수, 입력 토큰, 출력 토큰, 캐시 적중률, 각 날짜의 비용.
- **모델별 내역** — 사용된 각 모델, 세션 수, 토큰 사용량 및 예상 비용을 보여주는 표.

### 크론 (Cron)

정해진 일정에 따라 에이전트 프롬프트를 반복적으로 실행하는 예약된 크론 작업을 생성하고 관리합니다.

- **Create (생성)** — 이름(선택 사항), 프롬프트, 크론 표현식(예: `0 9 * * *`) 및 전달 대상(local, Telegram, Discord, Slack 또는 이메일)을 입력합니다.
- **작업 목록** — 각 작업에는 이름, 프롬프트 미리보기, 일정 표현식, 상태 배지(활성화됨/일시 중지됨/오류), 전달 대상, 마지막 실행 시간 및 다음 실행 시간이 표시됩니다.
- **Pause / Resume (일시 중지 / 재개)** — 작업을 활성 상태와 일시 중지 상태 간에 전환합니다.
- **Edit (편집)** — 미리 채워진 모달을 열어 작업의 프롬프트, 일정, 이름 또는 전달 대상을 변경합니다.
- **Trigger now (지금 실행)** — 정상적인 일정을 벗어나 작업을 즉시 실행합니다.
- **Delete (삭제)** — 크론 작업을 영구적으로 제거합니다.

### 스킬 (Skills)

설치된 스킬과 도구 세트를 찾아보고, 검색 및 토글하며, 허브에서 새로운 스킬을 설치합니다. 스킬은 `~/.hermes/skills/`에서 로드되며 카테고리별로 그룹화됩니다.

- **Search (검색)** — 이름, 설명 또는 카테고리별로 설치된 스킬과 도구 세트를 필터링합니다.
- **Category filter (카테고리 필터)** — 카테고리 알약 버튼(pill)을 클릭하여 목록을 좁힙니다(예: MLOps, MCP, Red Teaming, AI).
- **Toggle (토글)** — 스위치로 개별 스킬을 활성화 또는 비활성화합니다. 변경 사항은 다음 세션에서 적용됩니다.
- **Toolsets (도구 세트)** — 별도의 뷰에서 내장 도구 세트(파일 작업, 웹 브라우징 등)를 활성/비활성 상태, 설정 요구 사항 및 포함된 도구 목록과 함께 보여줍니다.
- **Browse hub (허브 탐색)** — 세 번째 뷰에서는 모든 소스에 걸쳐 스킬 허브를 검색하고(`hermes skills search`와 동일), 식별자(identifier)를 통해 실시간 설치 로그와 함께 결과를 설치하며, 설치된 스킬을 새로 고치는 "Update all(모두 업데이트)" 버튼을 제공합니다.

![Skills admin page — the Browse hub view: search, install, and update](/img/dashboard/admin-skills-hub.png)

### MCP

CLI 없이 [MCP](/integrations/mcp) 서버를 관리합니다. `hermes mcp`가 읽는 것과 동일한 `config.yaml`의 `mcp_servers` 블록을 사용합니다.

**사용자의 MCP 서버:**

- **Add (추가)** — HTTP/SSE 서버(URL) 또는 stdio 서버(명령 + 인수)를 등록합니다. stdio 서버의 경우 선택적 `KEY=VALUE` 환경 변수를 추가할 수 있습니다.
- **Enable / disable (활성화 / 비활성화)** — 서버를 삭제하지 않고 켜거나 끕니다. 비활성화된 서버는 구성 파일에 유지되므로 나중에 다시 활성화할 수 있습니다. 다음 게이트웨이 재시작 시 적용됩니다.
- **Test (테스트)** — 에이전트가 의존하기 전에 서버에 연결하고, 도구를 나열한 후 연결을 해제하여 연결 상태를 확인합니다.
- **Remove (제거)** — 구성 파일에서 서버를 삭제합니다.
- 목록 뷰에서 비밀값 형태의 환경 변수는 가려집니다.

**카탈로그 (Catalog):** Nous에서 승인한 MCP 서버(번들로 제공되는 `optional-mcps/` 카탈로그)를 탐색하고 클릭 한 번으로 설치하세요. API 키가 필요한 항목은 인라인으로 키를 묻고, 그 값은 `.env`로 저장됩니다. 이는 `hermes mcp catalog` / `hermes mcp install`에서 사용하는 것과 동일한 카탈로그입니다.

![MCP admin page — your servers with enable/disable toggles, plus the install catalog](/img/dashboard/admin-mcp.png)

### 웹훅 (Webhooks)

동적 [웹훅 구독](/user-guide/messaging/webhooks)을 관리합니다. 이 기능을 사용하려면 먼저 메시징 설정에서 웹훅 플랫폼을 활성화해야 하며, 활성화되지 않은 경우 페이지에 힌트가 표시됩니다.

- **Create (생성)** — 이름, 설명, 이벤트 필터, 전송 대상, 선택적 직접 전송 모드, 그리고 에이전트 프롬프트를 지정합니다. 생성 시 경로 URL과 복사할 일회성 HMAC 비밀키(secret)를 표시합니다.
- **Enable / disable (활성화 / 비활성화)** — 구독을 켜거나 끕니다. 비활성화된 경로는 구독 파일에 그대로 유지되지만, 게이트웨이는 수신되는 이벤트를 거부(403)합니다. 게이트웨이가 파일을 핫 리로드(hot-reload)하므로 재시작 없이 다음 이벤트부터 즉시 적용됩니다.
- **List (목록)** — 각 구독의 URL, 이벤트 및 전송 대상이 표시됩니다.
- **Delete (삭제)** — 구독을 제거합니다.

![Webhooks admin page — subscriptions with enable/disable toggles](/img/dashboard/admin-webhooks.png)

### 페어링 (Pairing)

원격 관리자가 CLI를 통하지 않고 웹 브라우저에서 Telegram/Discord 등의 사용자를 페어링된 게이트웨이에 온보딩할 때, 메시징 사용자를 승인하거나 취소할 수 있습니다. `hermes pairing` 명령어와 완전히 동일한 기능을 제공합니다.

- **Pending requests (대기 중인 요청)** — 플랫폼, 코드, 사용자 및 요청 기간을 표시하며 승인(Approve) 버튼을 제공합니다.
- **Approved users (승인된 사용자)** — 플랫폼과 사용자를 표시하며 취소(Revoke) 버튼을 제공합니다.
- **Clear pending (대기 중인 항목 지우기)** — 처리되지 않은 모든 페어링 코드를 삭제합니다.

![Pairing admin page](/img/dashboard/admin-pairing.png)

### 채널 (Channels)

웹 브라우저에서 Hermes를 모든 메시징 플랫폼에 연결합니다 — `hermes setup gateway` 명령어와 완전히 동일한 기능을 제공합니다. 지원되는 모든 채널(Telegram, Discord, Slack, Matrix, Mattermost, WhatsApp, Signal, BlueBubbles/iMessage, Email, SMS/Twilio, DingTalk, Feishu/Lark, WeCom, WeChat, QQ Bot, Yuanbao, 그리고 API 서버 및 웹훅 엔드포인트)과 실시간 연결 상태가 목록으로 표시됩니다.

- **Configure (구성)** — 각 채널에 꼭 필요한 필드(봇 토큰, 앱 토큰, 서버 URL, 허용 목록 등)만 포함된 플랫폼별 양식을 엽니다. 비밀키는 비밀번호 입력 형태로 렌더링되며 가려진 상태로 저장됩니다; 필드를 비워두면 기존 값이 유지됩니다. 필수 필드는 표시되고 유효성 검사가 진행됩니다. "Setup guide(설정 가이드)" 링크를 클릭하면 해당 플랫폼의 자격 증명 관련 문서로 이동합니다.
- **Enable / disable (활성화 / 비활성화)** — 채널을 켜거나 끕니다. 자격 증명은 디스크에 그대로 유지되며 활성화 상태만 변경됩니다.
- **Test (테스트)** — 채널이 구성되고, 활성화되었으며, 게이트웨이에서 정상적으로 실시간 연결을 보고하는지 확인합니다.
- **Restart gateway (게이트웨이 재시작)** — 자격 증명은 `~/.hermes/.env`에 기록되고 활성화 플래그는 `config.yaml`에 기록됩니다; 다음번 재시작 시 활성화된 각 채널이 연결되며, 페이지 내에서 바로 게이트웨이 재시작을 트리거할 수 있습니다.

![Channels admin page — every messaging platform with status, enable toggles, and per-platform setup forms](/img/dashboard/admin-channels.png)

### 시스템 (System)

설치 환경 전체의 작업을 통합 관리하는 패널입니다:

- **Host (호스트)** — 실시간 시스템 상태 정보: OS / 커널, 아키텍처, 호스트 이름, Python 및 Hermes 버전, CPU 코어 수 + 사용률, 메모리, Hermes 홈 디렉토리의 디스크 사용량, 시스템 가동 시간(uptime), 부하 평균(load average). (CPU/메모리/디스크 정보는 `psutil`이 설치된 경우 수집되며, 식별 필드는 항상 표시됩니다.) Hermes 버전 옆에는 **업데이트 상태 배지**(최신 버전 / N개의 커밋 뒤쳐짐)와 **Check for updates(업데이트 확인)** 버튼이 있습니다. Git 또는 Pip 기반 설치 환경에서 업데이트가 가능한 경우, **Update now(지금 업데이트)** 버튼을 누르면 가져올 커밋 수를 보여주는 확인 창이 열리고 백엔드에서 `hermes update`를 실행합니다. Docker/Nix/Homebrew 기반 설치 환경에서는 대시보드가 자체적으로 인플레이스(in-place) 업데이트를 수행할 수 없으므로, 대신 별도로 실행해야 할 알맞은 명령어를 안내해 줍니다.
- **Nous Portal** — 로그인 상태, 활성화된 추론(inference) 제공자, Tool Gateway 라우팅 테이블(어떤 도구가 Portal을 통해 실행되고 어떤 도구가 로컬에서 실행되는지)을 표시하며, 구독 관리 링크를 제공합니다. `hermes portal` 명령어의 읽기 전용 미러(mirror) 역할을 합니다.
- **Skill curator (스킬 큐레이터)** — 백그라운드 스킬 유지보수 상태(활성 / 일시 중지, 실행 주기, 마지막 실행 시간)를 표시하며 일시 중지/재개 및 즉시 실행 버튼을 제공합니다. `hermes curator` 기능을 미러링합니다.
- **Gateway (게이트웨이)** — 메시징 게이트웨이를 시작, 중지, 재시작할 수 있으며 실시간 상태(실행 중/중지됨, PID, 상태)를 보여줍니다.
- **Memory (메모리)** — 외부 메모리 제공자(또는 내장 메모리만 사용)를 선택하고, 내장된 `MEMORY.md` / `USER.md` 저장소를 초기화할 수 있습니다.
- **Credential pool (자격 증명 풀)** — 에이전트가 각 제공자(provider)에 대해 라운드 로빈(round-robin) 방식으로 순환하며 사용할 API 키를 추가하거나 제거할 수 있습니다. 목록에서 키는 가려진 상태로 표시되며, 원시(raw) 값은 에이전트에만 전달됩니다.
- **Operations (작업)** — 진단(doctor) 실행, 보안 감사, 백업 생성, 백업 아카이브에서 복원, 스킬 업데이트, 시스템 프롬프트 크기 세부 정보 표시, 지원 덤프(support dump) 생성, 만료된 설정 마이그레이션을 수행할 수 있습니다. 각 항목은 백그라운드 작업을 생성하고 그 실시간 로그 스트림을 페이지에 보여줍니다.
- **Checkpoints (체크포인트)** — `/rollback` 섀도(shadow) 저장소의 크기를 확인하고 불필요한 데이터를 정리(prune)합니다.
- **Shell hooks (쉘 훅)** — 허용 여부(consent) 및 실행 가능 상태와 함께 구성된 훅 목록을 보여줍니다. 이벤트, 명령어, 매처(matcher), 시간 초과(timeout), 명시적 허용 권한 부여 기능을 포함해 새로운 훅을 **Create(생성)** 하거나 제거할 수 있습니다. 훅은 임의의 명령어를 실행할 수 있으므로, 생성 양식에는 보안 경고가 포함되어 있으며 사용자가 권한을 허용한 후에만 훅이 작동합니다.

![System admin page — host stats and Nous Portal status](/img/dashboard/admin-system-top.png)

![System admin page — skill curator, gateway, memory, and credential pool](/img/dashboard/admin-system-curator.png)

![System admin page — operations, checkpoints, and shell hooks](/img/dashboard/admin-system-ops.png)

쉘 훅 생성 (동의 확인 체크박스와 임의 명령어 실행 경고에 유의하세요):

![New shell hook modal](/img/dashboard/admin-hook-create.png)

:::warning 보안
웹 대시보드는 API 키와 비밀키가 포함된 `.env` 파일을 읽고 씁니다. 기본적으로 `127.0.0.1`에 바인딩되므로 로컬 컴퓨터에서만 접근할 수 있습니다. 만약 `0.0.0.0`으로 바인딩할 경우, 해당 네트워크의 누구나 자격 증명을 열람하고 수정할 수 있게 됩니다. 대시보드 자체에는 인증 기능이 없습니다.
:::

## `/reload` 슬래시 명령어

대시보드 PR은 인터랙티브 CLI에 `/reload` 슬래시 명령어를 추가합니다. 웹 대시보드를 통해 (또는 `.env`를 직접 편집하여) API 키를 변경한 후, 활성화된 CLI 세션에서 `/reload`를 사용하면 프로세스를 재시작하지 않고도 변경 사항을 적용할 수 있습니다:

```
You → /reload
  Reloaded .env (3 var(s) updated)
```

이 명령어는 실행 중인 프로세스의 환경에 `~/.hermes/.env`를 다시 읽어 들입니다. 대시보드를 통해 새 제공자의 키를 추가하고 즉시 사용하고 싶을 때 매우 유용합니다.

## REST API

웹 대시보드는 프론트엔드가 사용하는 REST API를 노출합니다. 자동화를 위해 이러한 엔드포인트를 직접 호출할 수도 있습니다:

### GET /api/status

에이전트 버전, 게이트웨이 상태, 플랫폼 상태, 활성 세션 수를 반환합니다.

### GET /api/sessions

메타데이터(모델, 토큰 수, 타임스탬프, 미리보기)가 포함된 최근 20개의 세션을 반환합니다.

### GET /api/config

현재의 `config.yaml` 내용을 JSON 형태로 반환합니다.

### GET /api/config/defaults

기본 구성 값을 반환합니다.

### GET /api/config/schema

모든 구성 필드(유형, 설명, 카테고리, 해당되는 경우 선택 옵션)를 설명하는 스키마를 반환합니다. 프론트엔드는 이를 바탕으로 각 필드에 적합한 입력 위젯을 렌더링합니다.

### PUT /api/config

새 구성을 저장합니다. 본문: `{"config": {...}}`.

### GET /api/env

알려진 모든 환경 변수의 설정/해제 상태, 가려진 값, 설명 및 카테고리를 반환합니다.

### PUT /api/env

환경 변수를 설정합니다. 본문: `{"key": "VAR_NAME", "value": "secret"}`.

### DELETE /api/env

환경 변수를 제거합니다. 본문: `{"key": "VAR_NAME"}`.

### GET /api/sessions/\{session_id\}

단일 세션의 메타데이터를 반환합니다.

### GET /api/sessions/\{session_id\}/messages

도구 호출 및 타임스탬프를 포함한 세션의 전체 메시지 기록을 반환합니다.

### GET /api/sessions/search

메시지 콘텐츠 전체를 대상으로 텍스트 검색을 수행합니다. 쿼리 매개변수: `q`. 일치하는 세션 ID와 강조 표시된 스니펫을 반환합니다.

### DELETE /api/sessions/\{session_id\}

세션 및 그 메시지 기록을 삭제합니다.

### GET /api/logs

로그 기록을 줄 단위로 반환합니다. 쿼리 매개변수: `file` (agent/errors/gateway), `lines` (가져올 개수), `level`, `component`.

### GET /api/analytics/usage

토큰 사용량, 비용 및 세션 분석 결과를 반환합니다. 쿼리 매개변수: `days` (기본값 30). 응답에는 일별 세부 내역 및 모델별 집계가 포함됩니다.

### GET /api/cron/jobs

상태, 일정 및 실행 기록을 포함하여 구성된 모든 크론 작업을 반환합니다.

### POST /api/cron/jobs

새 크론 작업을 생성합니다. 본문: `{"prompt": "...", "schedule": "0 9 * * *", "name": "...", "deliver": "local"}`.

### POST /api/cron/jobs/\{job_id\}/pause

크론 작업을 일시 중지합니다.

### POST /api/cron/jobs/\{job_id\}/resume

일시 중지된 크론 작업을 재개합니다.

### POST /api/cron/jobs/\{job_id\}/trigger

정상적인 일정을 벗어나 크론 작업을 즉시 실행합니다.

### DELETE /api/cron/jobs/\{job_id\}

크론 작업을 삭제합니다.

### GET /api/skills

이름, 설명, 카테고리 및 활성화 상태를 포함하여 모든 스킬을 반환합니다.

### PUT /api/skills/toggle

스킬을 활성화 또는 비활성화합니다. 본문: `{"name": "skill-name", "enabled": true}`.

### GET /api/tools/toolsets

레이블, 설명, 도구 목록, 활성화/구성 상태를 포함하여 모든 도구 세트를 반환합니다.

### 관리 엔드포인트 (Admin endpoints)

MCP, Channels, Webhooks, Pairing, System 페이지에 필요한 기능을 제공합니다. 이들은 `/api/`의 다른 모든 엔드포인트와 동일한 인증 게이트의 보호를 받습니다.

| 메서드 및 경로 | 목적 |
|---------------|---------|
| `GET /api/mcp/servers` | 설정된 MCP 서버 목록 조회 (환경 변수 값은 가려짐) |
| `POST /api/mcp/servers` | 서버 추가. 본문: `{name, url?, command?, args?, env?, auth?}` |
| `POST /api/mcp/servers/{name}/test` | 연결, 도구 나열, 연결 해제 테스트 |
| `PUT /api/mcp/servers/{name}/enabled` | 서버 활성화 / 비활성화 |
| `DELETE /api/mcp/servers/{name}` | 서버 제거 |
| `GET /api/mcp/catalog` | Nous 승인 MCP 카탈로그 살펴보기 |
| `POST /api/mcp/catalog/install` | 카탈로그 항목 설치 (필수 환경 변수 포함) |
| `GET /api/messaging/platforms` | 모든 메시징 채널의 상태와 플랫폼별 설정 필드 나열 |
| `PUT /api/messaging/platforms/{id}` | 채널 설정. 본문: `{enabled?, env?, clear_env?}` (환경 변수는 `.env`에, 활성화 여부는 `config.yaml`에 기록) |
| `POST /api/messaging/platforms/{id}/test` | 채널 설정, 활성화, 및 게이트웨이 연결 상태 검사 |
| `GET /api/pairing` | 대기 중 및 승인된 메시징 사용자 목록 조회 |
| `POST /api/pairing/approve` | 인증 코드 승인. 본문: `{platform, code}` |
| `POST /api/pairing/revoke` | 사용자 권한 취소. 본문: `{platform, user_id}` |
| `POST /api/pairing/clear-pending` | 대기 중인 모든 코드 삭제 |
| `GET /api/webhooks` | 구독 목록 및 플랫폼 활성화 상태 조회 |
| `POST /api/webhooks` | 구독 생성 (일회성 비밀키 반환) |
| `DELETE /api/webhooks/{name}` | 구독 제거 |
| `GET /api/credentials/pool` | 풀링된 회전식(rotation) 키 나열 (값 가려짐) |
| `POST /api/credentials/pool` | 키 추가. 본문: `{provider, api_key, label?}` |
| `DELETE /api/credentials/pool/{provider}/{index}` | 키 제거 (1부터 시작하는 인덱스) |
| `GET /api/memory` | 활성화된 제공자 + 사용 가능한 제공자 + 내장 메모리 파일 크기 정보 |
| `PUT /api/memory/provider` | 제공자 선택 (빈 값이면 내장 메모리만 사용) |
| `POST /api/memory/reset` | 내장 메모리 재설정. 본문: `{target: all\|memory\|user}` |
| `POST /api/gateway/start` · `/stop` · `/restart` | 게이트웨이 수명 주기 제어 (백그라운드 처리) |
| `POST /api/ops/doctor` · `/security-audit` · `/backup` · `/import` | 진단 및 유지 보수 (백그라운드 처리; `/api/actions/{name}/status`로 꼬리표 붙임) |
| `GET /api/ops/hooks` | 구성된 쉘 훅 및 허용 목록(allowlist) 상태 조회 |
| `GET /api/ops/checkpoints` · `POST .../prune` | `/rollback` 저장소 점검 및 정리 |
| `POST /api/ops/hooks` · `DELETE /api/ops/hooks` | 쉘 훅 생성 / 제거 (동의 기반) |
| `GET /api/system/stats` | 호스트 상태 — OS, CPU, 메모리, 디스크, 업타임 정보 |
| `GET /api/hermes/update/check` | 업데이트 적용 없이 가용성(뒤쳐진 커밋 수, 설치 방식) 확인. `?force=1`은 6시간 캐시를 무효화함 |
| `GET /api/curator` · `PUT .../paused` · `POST .../run` | 스킬 큐레이터 상태 조회 + 일시 중지/재개 + 수동 실행 |
| `GET /api/portal` | Nous Portal 인증 정보 + Tool Gateway 라우팅 (읽기 전용) |
| `POST /api/ops/prompt-size` · `/dump` · `/config-migrate` | 진단 관련 명령 (백그라운드 처리) |
| `PUT /api/webhooks/{name}/enabled` | 웹훅 라우팅 켜기 / 끄기 |
| `POST /api/skills/hub/install` · `/uninstall` · `/update` | 스킬 허브 동작 (백그라운드 처리) |
| `GET /api/skills/hub/search` | 모든 스킬 허브 데이터 소스 검색 |
| `GET /api/sessions/stats` | 세션 저장소 통계 확인 |
| `PATCH /api/sessions/{id}` | 세션 이름 변경 / 보관 처리 |
| `GET /api/sessions/{id}/export` | 메타데이터와 메시지가 포함된 세션을 JSON으로 내보내기 |
| `POST /api/sessions/prune` | N일 이상 지난 종료된 세션을 삭제 |
| `PUT /api/cron/jobs/{id}` | 크론 작업의 프롬프트 / 스케줄 / 이름 / 전송 설정 등 편집 |

## 인증 (게이트 모드) (Authentication (gated mode))

대시보드가 `127.0.0.1` / `localhost`를 제외한 공개 주소나 루프백이 아닌 주소에 바인딩될 때 Hermes 에이전트는 인증 게이트를 활성화합니다. 모든 요청은 검증된 세션 쿠키를 포함해야 하며, 그렇지 않으면 로그인 페이지로 튕겨집니다. 기본으로 3가지 제공자(provider)가 탑재되어 있습니다:

- **[사용자 이름/비밀번호](#usernamepassword-provider-no-oauth-idp)** — 자체 호스팅 / 온프레미스 / 홈랩 대시보드에 인증을 추가하는 가장 간단한 방법입니다. 외부 자격 증명 제공자(IDP)가 필요하지 않습니다. **신뢰할 수 있는 네트워크나 VPN 뒤에서만 사용하세요 — 공개 인터넷에 노출하지 마세요.**
- **[OAuth (Nous Portal)](#default-provider-nous-research)** — 호스팅된 배포, 공개 인터넷을 통해 접근 가능한 모든 대시보드, 그리고 [원격 Hermes Desktop 연결](#connecting-hermes-desktop-to-a-remote-backend)을 권장하는 방법입니다. 모든 로그인은 Nous 계정으로 검증되므로 인터넷에 노출된 환경에 가장 적합한 제공자입니다.
- **[자체 호스팅 OIDC](#self-hosted-oidc-provider)** — 표준 OpenID Connect(Keycloak, Auth0, Okta, Google, OIDC 브릿지를 통한 GitHub 등)를 이용해 보유 중인 자격 증명 제공자를 연결할 때 사용합니다. Nous Portal이 전혀 관여하지 않으며, 호환되는 OIDC 서버를 앞단에 둘 경우 공개 인터넷 노출에 적합합니다.

루프백에 바인딩한 대시보드는 이 인증 기능의 영향을 받지 않으며, 인증이나 로그인 페이지가 없습니다.

### 언제 게이트가 켜지는가 (When the gate engages)

| 플래그 | 인증 게이트 | 사용 사례 |
|-------|-----------|----------|
| `hermes dashboard` (기본값 — `127.0.0.1`에 바인딩) | OFF | 로컬 개발 |
| `hermes dashboard --host 0.0.0.0` | **ON** | 원격 / 프로덕션 — 사용자 이름/비밀번호 제공자나 OAuth로 보호하세요 |

다음 두 조건을 모두 만족할 때만 게이트가 작동합니다:

1. 바인딩 호스트가 `127.0.0.1`, `::1`, `localhost` 또는 `0.0.0.0`이 아니며 (의역: 루프백이 아니며) AND
2. `--insecure` 플래그가 설정되어 있지 **않을** 때.

:::danger `--insecure`는 인증을 완전히 비활성화합니다
`--insecure` 플래그를 사용하면 게이트를 건너뛰고, 사용자의 `.env`(API 키, 비밀키)를 읽고 쓸 수 있으며 에이전트 명령어를 실행할 수 있는 인증되지 않은 대시보드를 서비스합니다. **이 옵션을 원격 연결에 사용하지 마세요.** 다른 머신에 대시보드를 노출하려면 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp) (또는 OAuth)를 구성하고 `--insecure` 옵션을 뺀 채로 둬야 합니다. 이 플래그는 완전히 신뢰할 수 있고 방화벽으로 보호된 단일 호스트 네트워크에서 최후의 수단으로만 존재합니다.
:::

### 시작 실패 동작 구조 (Fail-closed semantics)

게이트가 작동해야 할 상황에서 어떠한 `DashboardAuthProvider`도 등록되지 않았다면(Nous 플러그인이나 커스텀 플러그인 등) `hermes dashboard`는 명시적인 오류 메시지와 함께 바인딩을 거부합니다. "기본적으로는 거부하되, (설정이 없다면) 모두 허용한다"와 같은 폴백 기능은 존재하지 않습니다 — 구성이 잘못된(gated) 대시보드는 결코 시작되지 않습니다.

### 기본 제공자: Nous Research (Default provider: Nous Research)

기본 번들된 `plugins/dashboard_auth/nous` 플러그인은 **항상 설치**되고 자동 로드됩니다. 클라이언트 ID가 설정되면 `nous`라는 이름의 `DashboardAuthProvider`를 자동으로 등록합니다.

모든 로그인이 Nous Portal에 대해 확인되고 사용자의 Nous 계정으로 보호되므로, **Nous 제공자는 공개 인터넷에 대시보드를 노출할 때 사용하기 적합합니다.**

#### 대시보드 등록하기 (Registering a dashboard)

Nous 제공자를 사용하려면 OAuth 클라이언트 ID(형식: `agent:{id}`)가 필요합니다. 얻는 방법에는 두 가지가 있습니다:

- **CLI — `hermes dashboard register`.** 대시보드가 상주할 호스트에서 이 명령을 실행하세요. 이는 기존 Nous 로그인 상태를 확인하고(로그인되어 있지 않다면 먼저 `hermes setup`을 실행하세요), 자체 호스팅된 OAuth 클라이언트를 Portal에 등록한 다음, `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 `~/.hermes/.env`에 기록해 줍니다. 선택적 플래그: `--name` (사람이 읽을 수 있는 레이블, 없으면 자동 생성됨), `--redirect-uri` (인터넷 연결 호스트를 위한 공개 HTTPS 콜백 URL).

  ```bash
  hermes dashboard register
  # ✓ Registered dashboard "swift_falcon"
  # …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
  ```

- **GUI — 로컬 대시보드 페이지.** 브라우저에서 직접 자체 호스팅 대시보드를 등록하고, 이름을 지정하고, 관리 및 취소하려면 Nous Portal에서 [`/local-dashboards`](https://portal.nousresearch.com/local-dashboards) 페이지를 여세요. 생성된 `agent:{id}` 클라이언트 ID를 복사하여 환경 변수 `HERMES_DASHBOARD_OAUTH_CLIENT_ID` 또는 `config.yaml`의 `dashboard.oauth.client_id`로 설정하세요. 이곳은 CLI를 통해 등록한 대시보드 권한을 취소할 때 사용하는 곳이기도 합니다.

#### 구성 (Configuration)

플러그인은 두 가지 영역을 읽으며, 값이 비어 있지 않은 경우 환경 변수가 더 우선합니다:

**`config.yaml`** — 기준이 되는 표면:

```yaml
dashboard:
  oauth:
    client_id: agent:01HXYZ…             # 게이트를 활성화하기 위해 필요
```

**환경 변수** — 운영자 오버라이드:

| 환경 변수 | 덮어쓰는 항목 | 형식 | 프로비저닝 수단 |
|---------|-----------|--------|----------------|
| `HERMES_DASHBOARD_OAUTH_CLIENT_ID` | `dashboard.oauth.client_id` | `agent:{instance_id}` | `hermes dashboard register` |

Hermes 에이전트의 관례(`~/.hermes/.env`는 오직 API 키와 비밀키용)에 따라, 로컬 개발, 온프레미스 및 직접 제어하는 모든 배포에서는 **이 값들을 `config.yaml`에 설정하는 것을 권장**합니다. 환경 변수 방식은 이미지 내의 `config.yaml`을 수정할 필요 없이 호스팅 플랫폼의 비밀키 주입 시스템을 통해 배포별로 `client_id`를 푸시할 수 있게 하려는 주된 목적으로 만들어졌습니다.

비어 있는 환경 변수는 설정되지 않은 것으로 처리되므로, 값 없이 선언만 된(provisioned-but-not-populated) 플랫폼 비밀키가 유효한 `config.yaml` 항목을 실수로 가리는(shadow) 일은 발생하지 않습니다.

두 경로 모두에서 client_id를 제공하지 않으면 플러그인은 구체적인 이유를 보고하며 대시보드의 fail-closed 바인딩 오류를 통해 무엇을 고쳐야 할지 정확히 알려줍니다:

```
Refusing to bind dashboard to 0.0.0.0 — the OAuth auth gate engages on
non-loopback binds, but no auth providers are registered.

Bundled providers reported these issues:
  • nous: HERMES_DASHBOARD_OAUTH_CLIENT_ID is not set (and
    dashboard.oauth.client_id in config.yaml is empty). The Nous Portal
    provisions this env var (shape 'agent:{instance_id}') when it
    deploys a Hermes Agent instance — set it to your provisioned
    client id (either as an env var or under dashboard.oauth.client_id
    in config.yaml), or pass --insecure to skip the OAuth gate entirely.

Or pass --insecure to skip the auth gate (NOT recommended on untrusted
networks).
```

#### 실습 예제: Nous Research (Worked example: Nous Research)

로그인된 Hermes 환경에서 3단계로 Nous 게이트 보호 대시보드를 구성합니다.

**1. 로그인하고 대시보드를 등록합니다.** `hermes dashboard register`는 기존 Nous 로그인을 사용하여 OAuth 클라이언트를 프로비저닝하고 `HERMES_DASHBOARD_OAUTH_CLIENT_ID`를 `~/.hermes/.env`에 기록합니다:

```bash
hermes setup            # 아직 Nous Portal에 로그인하지 않은 경우
hermes dashboard register
# ✓ Registered dashboard "swift_falcon"
# …writes HERMES_DASHBOARD_OAUTH_CLIENT_ID to ~/.hermes/.env
```

**2. 대시보드를 도달 가능한 주소로 실행합니다.** `--insecure` 옵션 없는 루프백 외 바인딩은 OAuth 게이트를 활성화하며, 방금 기록된 `client_id`가 `nous` 제공자를 작동시킵니다:

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. 로그인합니다.** `http://<host>:9119/`에 접속하면 `/login`으로 튕겨집니다. **Sign in with Nous Research**를 클릭 → Portal에서 인증 → 인증된 대시보드로 돌아옵니다. 아무 기기에서나 게이트 상태를 확인해 보세요:

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

그러면 `GET /api/auth/me`가 인증된 세션(`provider: nous`)을 반환합니다. 인터넷 연결 호스트의 경우 `--redirect-uri https://hermes.example.com/auth/callback` 옵션으로 등록하고, OAuth 콜백이 퍼블릭 URL로 확인되도록 `HERMES_DASHBOARD_PUBLIC_URL`을 설정하세요 ([퍼블릭 URL 재정의](#public-url-override) 참조).

### 사용자 이름/비밀번호 제공자 (OAuth IDP 없음) (Username/password provider (no OAuth IDP))

자체 호스팅된 "내 대시보드에 그냥 비밀번호 하나 걸고 싶어"라는 식의 배포를 위해, OAuth 신원 제공자(identity provider)를 구성하고 싶지 않다면 번들된 `plugins/dashboard_auth/basic` 플러그인이 `basic`이라는 이름의 `DashboardAuthProvider`를 등록합니다. 이 제공자는 OAuth 리디렉트 대신 **사용자 이름과 비밀번호**로 인증합니다.

이는 OAuth 제공자와 동일한 게이트에 연결됩니다: `--insecure` 옵션이 없는 루프백 외 바인딩에서 게이트가 활성화되고, 로그인 페이지는 ("Sign in with X" 버튼 대신) 이 제공자를 위한 자격 증명 양식을 렌더링합니다. 로그인 이후의 과정 — 세션 쿠키, 투명한 리프레시(transparent refresh), WS 티켓, 로그아웃, 감사 로그 등 — 은 모두 OAuth 경로와 동일합니다. 세션은 제공자 자신이 발행한 무상태(stateless) HMAC 서명 토큰이므로 **데이터베이스와 외부 IDP가 전혀 없습니다**. 비밀번호 해싱에는 (서드 파티 의존성 없이) 표준 라이브러리의 `scrypt`가 사용됩니다.

:::warning 신뢰할 수 있는 네트워크에서만 사용하세요 — 공개 인터넷은 피하세요
사용자 이름/비밀번호 제공자는 **신뢰할 수 있는 네트워크** 상의 자체 호스팅 / 온프레미스 / 홈랩 대시보드, 또는 **VPN**을 통해서만 도달할 수 있는 환경을 위해 만들어졌습니다. 이는 외부 신원 제공자나 MFA, 다중 계정 지원 없이 공유되는 단일 자격 증명만 보호하므로, **대시보드를 공개 인터넷에 직접 노출하는 용도로는 적합하지 않습니다**. 인터넷 연결 대시보드의 경우 대신 [Nous Research 제공자](#default-provider-nous-research) (또는 [자체 호스팅 OIDC](#self-hosted-oidc-provider) / [사용자 지정 OAuth](#custom-providers) 제공자)를 사용하세요.
:::

#### 구성 (Configuration)

Nous 제공자와 마찬가지로 `config.yaml` (기준점)을 먼저 읽으며, 비어 있지 않은 환경 변수가 설정되면 그것이 더 우선 적용됩니다. `username`과 함께 `password_hash`(권장) 또는 `password` 중 하나가 구성될 때만 활성화됩니다 — 그렇지 않으면 작동하지 않으므로 OAuth 사용자나 루프백/`--insecure` 운영자에게는 영향을 미치지 않습니다.

**`config.yaml`:**

```yaml
dashboard:
  basic_auth:
    username: admin
    # 권장 사항 — 평문으로 보관하지 않습니다. 다음 명령어로 생성하세요:
    #   python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('PW'))"
    password_hash: "scrypt$16384$8$1$…$…"
    # ...또는 평문 비밀번호 (로드할 때 인메모리에서 해싱됨; 보관상 안전성이 떨어짐):
    # password: "s3cret"
    secret: "<32바이트 이상 임의값, base64 또는 hex>"  # 토큰 서명 키
    session_ttl_seconds: 43200                    # 선택 사항; 액세스 토큰 유효 기간 (기본 12시간)
```

**환경 변수 오버라이드:**

| 환경 변수 | 덮어쓰는 항목 | 참고 |
|---------|-----------|-------|
| `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` | `dashboard.basic_auth.username` | 활성화를 위해 필수 |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` | `dashboard.basic_auth.password_hash` | 권장 (평문 저장 방지) |
| `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` | `dashboard.basic_auth.password` | 평문 비밀번호; **`config.yaml`의 `password_hash`보다 우선**하므로 환경 변수로 교체 가능 |
| `HERMES_DASHBOARD_BASIC_AUTH_SECRET` | `dashboard.basic_auth.secret` | 토큰 서명 키 |
| `HERMES_DASHBOARD_BASIC_AUTH_TTL_SECONDS` | `dashboard.basic_auth.session_ttl_seconds` | 액세스 토큰 유효 기간 |

:::caution 안정적인 세션을 위해 명시적인 `secret`을 설정하세요
`secret` 값이 비어 있으면 프로세스 단위의 임의 서명 키가 생성됩니다. 단일 프로세스일 땐 괜찮지만, 이는 **재시작 시마다 모든 세션이 무효화**되고 세션이 **여러 워커(worker) 프로세스 간에 공유되지 않는다**는 것을 의미합니다. 재시작 시 세션을 유지하거나 다중 워커 배포 시엔 명시적인 `secret`을 설정하세요.
:::

`/auth/password-login` 엔드포인트는 클라이언트 IP당 속도 제한(기본 분당 10회 → HTTP 429)이 적용되며, 알 수 없는 사용자와 잘못된 비밀번호 모두에 대해 단일 형태의 일반적인 `401 Invalid credentials` 오류를 반환하므로 사용자 이름 열거 공격(enumeration oracle)에 사용할 수 없습니다.

#### 실습 예제: 사용자 이름/비밀번호 (Worked example: username/password)

아무런 설정이 없는 상태에서 신뢰할 수 있는 네트워크 상의 비밀번호 보호 대시보드 구축까지 3단계로 진행합니다.

**1. `~/.hermes/.env`에 자격 증명을 설정합니다.** 평문이 디스크에 남아있지 않도록 비밀번호를 해싱하고, 재시작 시 세션 유지를 위해 안정적인 서명 키를 설정하세요:

```bash
# 선택한 비밀번호의 scrypt 해시를 계산합니다:
HASH=$(python -c "from plugins.dashboard_auth.basic import hash_password; print(hash_password('choose-a-strong-password'))")

cat >> ~/.hermes/.env <<EOF
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH=$HASH
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env
```

**2. 대시보드를 도달 가능한 주소로 실행합니다.** `--insecure` 옵션 없는 루프백 외 바인딩은 인증 게이트를 활성화하며, 사용자 이름 + 해시 조합은 `basic` 제공자를 작동시킵니다:

```bash
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

**3. 로그인합니다.** `http://<host>:9119/`에 접속하면 `/login` 페이지로 튕겨집니다 — ("Sign in with X" 버튼 대신) **자격 증명 폼**이 표시됩니다. `admin` / 비밀번호를 입력 → 인증된 대시보드로 접속합니다. 모든 컴퓨터에서 게이트 상태를 확인해 보세요:

```bash
curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["basic"]
```

그러면 `GET /api/auth/me`는 인증된 세션(`provider: basic`)을 반환합니다. 앞의 경고문처럼 이 기능은 VPN 뒤에 숨겨 사용하세요; 공개 호스트 환경이라면 [Nous Research](#default-provider-nous-research) 또는 [자체 호스팅 OIDC](#self-hosted-oidc-provider) 제공자를 대신 사용하세요.

#### 고유한 비밀번호 제공자 작성하기 (Writing your own password provider)

`basic` 제공자는 플러그인 확장성의 한 구현 예시일 뿐입니다. 모든 커스텀 플러그인에서 자체 비밀번호 제공자를 등록할 수 있습니다: `DashboardAuthProvider`를 상속한 클래스에서 `supports_password = True`를 설정하고, `complete_password_login(*, username, password) -> Session` 메서드를 구현하세요 (로그인을 거부할 경우 `InvalidCredentialsError`를 반환하고, 연결된 DB 등에 장애가 났다면 `ProviderError`를 반환). 순수 비밀번호만 사용하는 제공자라면, OAuth용 `start_login` 및 `complete_login` 메서드는 `NotImplementedError`를 발생시키는 스텁(stub) 상태로 둬도 됩니다. 이 방식은 LDAP 바인딩, 별도 자격 증명 데이터베이스 등 리디렉션을 쓰지 않는 어떠한 인증 스키마에도 적용할 수 있으며 — 프레임워크가 양식 렌더링, 라우팅, 쿠키 처리 및 갱신(refresh)을 모두 알아서 처리해 줍니다.

### 자체 호스팅 OIDC 제공자 (Self-hosted OIDC provider)

자체적으로 식별 정보 제공자(IDP)를 운영한다면, 번들로 포함된 `plugins/dashboard_auth/self_hosted` 플러그인이 **표준 OpenID Connect**를 사용해 대시보드 인증을 처리합니다. IDP별 전용 코드나 Nous Portal을 통할 필요가 없습니다. 이는 모든 OIDC 호환 서버에서 검증되었고 정상 작동합니다:

> **Authentik · Keycloak · Zitadel · Authelia · Auth0 · Okta · Google · …**

Nous 제공자와 마찬가지로 이 플러그인은 자동 로드되며, 올바르게 설정되었을 때만 자신을 활성화합니다. 따라서 루프백이나 `--insecure` 상태의 대시보드에는 아무런 영향도 주지 않습니다.

#### 구성 (Configuration)

**발급자(issuer)** 와 **클라이언트 ID(client_id)** 를 설정하세요. 여기서 클라이언트 ID는 클라이언트 시크릿(client_secret)이 없는 공개 PKCE 클라이언트를 의미합니다. 플러그인은 `{issuer}/.well-known/openid-configuration`에서 `authorization_endpoint`, `token_endpoint`, `jwks_uri`를 자체적으로 가져오므로, 사용자가 엔드포인트 URL을 직접 하드코딩할 필요가 없습니다.

**`config.yaml`** — 기준 설정 파일:

```yaml
dashboard:
  oauth:
    provider: self-hosted
    self_hosted:
      issuer: https://auth.example.com/application/o/hermes/   # 필수
      client_id: hermes-dashboard                              # 필수
      scopes: "openid profile email"                           # 선택 사항 (기본값)
```

**환경 변수** — 운영 환경의 오버라이드. 값이 비어 있지 않으면 `config.yaml` 설정보다 우선 적용됩니다. (값이 비어 있다면 설정되지 않은 것으로 간주함)

| 환경 변수 | 덮어쓰는 항목 | 비고 |
|---------|-----------|-------|
| `HERMES_DASHBOARD_OIDC_ISSUER` | `dashboard.oauth.self_hosted.issuer` | OIDC 발급자 URL — 필수 |
| `HERMES_DASHBOARD_OIDC_CLIENT_ID` | `dashboard.oauth.self_hosted.client_id` | 퍼블릭 클라이언트 ID — 필수 |
| `HERMES_DASHBOARD_OIDC_SCOPES` | `dashboard.oauth.self_hosted.scopes` | 기본값 `openid profile email` |

IDP 설정 화면에서 승인 코드(authorization-code)와 PKCE(S256) 권한 부여 방식이 적용된 **퍼블릭(public)** 애플리케이션/클라이언트를 등록하고, 허용된 리디렉트 URI에 대시보드의 콜백을 추가하세요. 콜백 URL은 `<dashboard public URL>/auth/callback` 형식을 갖습니다. (프록시 뒤에서 대시보드가 퍼블릭 URL을 알아내는 방법은 [퍼블릭 URL 재정의](#public-url-override)를 참고하세요.)

#### 검증 절차 (What it verifies)

제공자는 확인된 `jwks_uri`에 대해 OpenID Connect의 **ID 토큰**(RS256/ES256)을 검증합니다. 이 과정에서 `iss` 및 `aud` 클레임(claim)이 직접 구성한 `issuer` 및 `client_id`와 일치하는지 꼼꼼히 대조합니다. 표준 OIDC 클레임은 다음과 같이 대시보드 세션에 맵핑됩니다:

| 세션 필드 | 클레임(들) |
|---------------|----------|
| `user_id` | `sub` (필수) |
| `email` | `email` |
| `display_name` | `name` → `preferred_username` → `nickname` → `email` 순 |
| `org_id` | `org_id` / `organization`, 없을 경우 조인된 `groups` |

신원을 확립하는 것은 ID 토큰입니다 — 액세스 토큰은 단순히 의미를 알 수 없는 문자열(opaque)로 처리됩니다 (OIDC 스펙은 액세스 토큰이 JWT 형태일 것을 요구하지 않기 때문입니다). 엔드포인트 URL은 무조건 HTTPS여야 하며 (단, 로컬 개발용 IDP의 경우 루프백에서만 `http://` 허용), 발견된(discovery) 문서가 나타내는 `issuer`는 사용자가 구성한 `issuer` 값과 반드시 일치해야 합니다 (맨 끝 슬래시(/) 차이 정도는 허용됨). IDP가 리프레시 토큰(refresh token)을 발급하는 경우 이를 표준 `refresh_token` 권한 부여 흐름에서 소리 없는 재인증(silent re-auth)에 사용하며, 로그아웃 시엔 IDP가 알린 RFC 7009 `revocation_endpoint`를 호출합니다.

> **기밀 클라이언트(Confidential clients)** (즉 `client_secret`을 가진 클라이언트)는 아직 지원하지 않습니다 — 브라우저 대상 대시보드의 일반적인 선택인 퍼블릭(public) + PKCE 클라이언트를 구성해 사용하세요.

#### 실습 예제: Keycloak (Worked example: Keycloak)

[Keycloak](https://www.keycloak.org/)은 로컬 테스트 환경을 구축하기에 가장 쉬운 자체 호스팅 OIDC 서버 중 하나입니다 — 개발 모드(인메모리 DB) 기반의 단일 컨테이너로 실행되며 표준 OIDC 검색 엔드포인트를 노출합니다. 이 튜토리얼을 따라 하면 텅 빈 상태에서 작동하는 대시보드 로그인 화면까지 몇 분 내에 도달할 수 있습니다.

**1. 미리 구성된 렐름(realm)과 함께 Keycloak 실행하기.** 다음 렐름 추출 데이터를 `realm-hermes.json`으로 저장하세요 — 이 JSON 데이터는 `hermes` 렐름, **퍼블릭 PKCE 클라이언트**(`hermes-dashboard`), 그리고 테스트 사용자를 모두 포함하며, 부팅 시 곧바로 불러와지므로 관리자 화면을 클릭할 필요가 전혀 없습니다:

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

앞서 만든 JSON 파일을 가져오기 경로에 마운트한 상태로 Keycloak(버전 26 이상)을 시작합니다:

```bash
docker run --rm -p 8080:8080 \
  -e KC_BOOTSTRAP_ADMIN_USERNAME=admin \
  -e KC_BOOTSTRAP_ADMIN_PASSWORD=admin \
  -v "$PWD/realm-hermes.json:/opt/keycloak/data/import/realm-hermes.json:ro" \
  quay.io/keycloak/keycloak:26.0 \
  start-dev --import-realm
```

구동이 끝나면 렐름은 `http://localhost:8080/realms/hermes/.well-known/openid-configuration`에서 표준 OIDC 엔드포인트를 노출합니다 (발급자(issuer)는 `http://localhost:8080/realms/hermes`). 관리자 화면은 `http://localhost:8080/`(`admin` / `admin`)에 위치합니다.

**2. 대시보드 연결하기.** 자체 호스팅 플러그인은 루프백 환경에서는 특별히 `http://` 발급자를 허용하므로 (루프백이 아닌 모든 발급자는 HTTPS 필수), 방금 띄운 로컬 Keycloak을 그대로 쓸 수 있습니다:

```bash
export HERMES_DASHBOARD_OIDC_ISSUER="http://localhost:8080/realms/hermes"
export HERMES_DASHBOARD_OIDC_CLIENT_ID="hermes-dashboard"
export HERMES_DASHBOARD_PUBLIC_URL="http://localhost:9119"
hermes dashboard --host 0.0.0.0 --port 9119 --no-open
```

`HERMES_DASHBOARD_PUBLIC_URL` 설정은 대시보드에게 자신의 OAuth 콜백이 `http://localhost:9119/auth/callback`임을 명시합니다 — 이는 아까 전 렐름에 등록한 리디렉트 URI와 같습니다. `--insecure`를 빼고 `0.0.0.0`(루프백 아님)에 바인딩하는 것이 곧 OAuth 게이트를 여는 스위치가 됩니다.

**3. 로그인하기.** `http://localhost:9119/`에 접속하면 `/login`으로 튕겨집니다. **Sign in with Self-Hosted OIDC**를 클릭 → Keycloak 화면에서 `testuser` / `testpassword`로 인증 → 성공적으로 인증된 대시보드 화면으로 진입합니다. 사이드바에는 `Logged in as Test User via self-hosted`라고 표시되며, `GET /api/auth/me`는 검증된 세션을 응답으로 보내줍니다 (`provider: self-hosted`, `email: testuser@example.com`).

> 만약 다른 호스트/포트에서 바인딩이나 열람을 원하신다면, Keycloak 관리자 콘솔(Clients → hermes-dashboard → Settings)에서 클라이언트의 **Valid redirect URIs**에 해당 출처(origin)의 `…/auth/callback`을 꼭 추가해 주세요. 이 방식은 Authentik, Zitadel, Authelia를 포함한 다른 OIDC 서버들에도 동일하게 적용됩니다 — 오직 발급자 URL과 클라이언트 등록 화면의 생김새만이 다를 뿐입니다.

### 퍼블릭 URL 재정의 (Public URL override)

기본적으로, 대시보드는 수신된 요청의 헤더를 재구성하여 OAuth 콜백 URL을 파악합니다 — `X-Forwarded-Host` + `X-Forwarded-Proto` + `X-Forwarded-Prefix` (대시보드가 활성화되었을 때 `start_server`가 `proxy_headers=True`로 uvicorn을 켤 경우). 만약 리버스 프록시가 이 세 가지 헤더를 모두 올바르게 전달해 준다면, 대시보드는 별도 설정 없이 바로 동작할 수 있습니다.

하지만 저 헤더들을 완벽하게 포워딩하지 못하는 리버스 프록시 환경(수동 설정한 nginx, 온프레미스 인그레스(ingress), 복잡한 프록시 체인을 거치는 커스텀 도메인 배포)의 경우, 대시보드에 실제로 접속할 수 있는 **완전한 퍼블릭 URL**을 `dashboard.public_url` (또는 `HERMES_DASHBOARD_PUBLIC_URL`) 속성에 지정해야 합니다:

```yaml
dashboard:
  public_url: "https://dashboard.example.com/hermes"
```

이 값이 설정되면 OAuth 콜백 URL은 어떠한 가공 없이 곧바로 `<public_url>/auth/callback`이 됩니다 — 운영자가 퍼블릭 URL을 명시적으로 확정 지었기 때문에, 이 경로를 거칠 땐 `X-Forwarded-Prefix`가 완전히 무시됩니다. 이는 의도된 것입니다: `public_url` 내부에 이미 접두사가 들어있는 상황에서 헤더를 다시 덧붙이면 접두사가 두 번 반복(double-prefix)되는 흔한 오류가 생기기 때문입니다.

다른 대시보드 설정과 마찬가지로, `config.yaml`보다 환경 변수가 높은 우선순위를 가집니다:

| 설정 표면 | 오버라이드 경로 | 언제 써야 하나 |
|---------|---------------|-------------|
| `config.yaml` 내 `dashboard.public_url` | `HERMES_DASHBOARD_PUBLIC_URL` | 로컬 개발 / 온프레미스 (기준점) |
| `HERMES_DASHBOARD_PUBLIC_URL` 환경 변수 | — | 호스팅 플랫폼의 시크릿(secret) 주입 / CI |
| (미설정) | — | 기본 동작 — `X-Forwarded-*` 헤더로 조립 |

입력값 검증 기능은 `http://` / `https://` 스킴(scheme)이 없거나, 호스트가 없거나, 따옴표 / 괄호 / 공백 문자 / 제어 문자 등이 섞여 있는 값을 모조리 튕겨냅니다. 형태가 기괴한 값이 들어오면 사용자를 적대적인(hostile) URL로 날려버리는 대신 헤더 재구성 모드로 자연스럽게 폴백(fallback)하여 로그인 흐름이 중단되지 않게 막습니다.

> **주의:** `public_url`은 오로지 OAuth 콜백 URL에만 덮어쓰기 효과를 가집니다. 브라우저의 쿠키 `Secure` 플래그는 여전히 `request.url.scheme` (프록시 헤더 설정 하에선 `X-Forwarded-Proto`)의 통제를 받습니다. 다시 말해, HTTPS를 끊고 넘겨주는(TLS-terminated) 퍼블릭 배포 환경에서 `public_url`에 `http://`를 적어 넣으면 Secure가 아닌(비보안) 쿠키가 구워집니다. 이는 운영자의 중대한 실수(footgun)입니다 — `public_url`은 언제나 업스트림의 적절한 TLS 종료 환경과 세트로 맞춰 사용하세요.

### OAuth 흐름 (OAuth flow)

이 제공자는 [Nous Portal OAuth contract v1](https://github.com/NousResearch/nous-account-service/blob/main/docs/agent-dashboard-oauth-contract.md) 명세를 구현했습니다 — PKCE (S256)가 포함된 승인 코드(authorization-code) 부여 방식입니다:

1. 사용자가 세션 쿠키 없이 `/`에 접속 → 게이트가 `/login`으로 리디렉트.
2. 로그인 페이지에서 "Continue with Nous Research" 버튼 노출 → `/auth/login?provider=nous`.
3. 서버가 짧은 유효 기간의 쿠키 안에 PKCE 상태(state)를 쑤셔 넣고, 사용자를 `https://portal.nousresearch.com/oauth/authorize?…`로 리디렉트.
4. 사용자가 Portal에서 인증 후 `/auth/callback?code=…&state=…`로 되돌아옴.
5. 서버는 `POST /api/oauth/token`을 이용해 이 코드를 액세스 토큰으로 교환하고, Portal의 JWKS(`/.well-known/jwks.json`)에 대해 JWT 서명을 검증한 뒤, 마침내 `hermes_session_at` 쿠키를 발급.
6. 사용자는 다시 `/`(또는 `next=` 쿼리 파라미터가 가리키던 처음 방문 목적지)로 리디렉트.

액세스 토큰의 TTL(유효 기간)은 15분입니다. **v1 계약에는 리프레시 토큰이 없습니다** — 토큰이 만료되면 SPA(Single Page App)의 fetch 래퍼(wrapper)가 HTTP 401 봉투(envelope)를 감지하고, 전체 페이지를 `/login`으로 강제로 돌려보내 이 과정을 다시 실행하도록 합니다.

### 설정되는 쿠키 목록 (Cookies set)

| 쿠키 이름 | 유효 기간 | 설명 |
|------|----------|-------|
| `hermes_session_at` | 토큰 TTL (15분) | HttpOnly, SameSite=Lax, HTTPS일 땐 Secure 추가 |
| `hermes_session_pkce` | 10분 | HttpOnly; 리디렉션 왕복 동안 PKCE 검증자(verifier) 및 제공자 힌트 보유 |
| `hermes_session_rt` | v1에선 미사용 | 미래 하위 호환성을 위해 예약됨; `refresh_token`이 빈 값이면 기록하지 않음 |

이 세 가지 쿠키 모두 `Path=/` 및 `SameSite=Lax` 속성을 가집니다. `Secure` 플래그는 대시보드가 HTTPS를 통해 도달되었을 때만 켜집니다 (요청 URL의 스킴을 감지하여 동작하며 — `proxy_headers=True`일 땐 업스트림 TLS 종료 장비에서 온 `X-Forwarded-Proto` 헤더를 존중합니다).

### 로그아웃 (Logout)

사이드바 위젯은 `Logged in as <user_id…> via nous` 메시지와 함께 로그아웃 아이콘을 띄워 줍니다. 이를 클릭하면 `/auth/logout` 경로로 POST 요청이 날아가 대시보드 인증 쿠키를 몽땅 지운 다음, 사용자를 다시 `/login`으로 보냅니다.

### 감사 로그 (Audit log)

로그인 시도, 성공, 실패 및 세션 검증 오류는 전부 하나의 JSON 형태 기록으로 묶여 `$HERMES_HOME/logs/dashboard-auth.log`에 저장됩니다. 민감한 내용물(`access_token`, `refresh_token`, `code`, `code_verifier`, `state`, `Authorization` 헤더 등)은 로그에 적히기 전에 알아서 검열(redacted)됩니다.

### 커스텀 제공자 (Custom providers)

Nous와 무관한 OAuth 제공자(예: Google, GitHub, 직접 만든 OIDC 등)를 연결하고 싶다면 `DashboardAuthProvider`를 상속하는 플러그인을 하나 만드시면 됩니다:

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

로그인 페이지에는 서버에 등록된 모든 제공자가 나열됩니다; 제공자를 층층이(stacked) 여러 개 올려놓으면, 사용자가 `/login` 화면에서 어떤 수단으로 로그인할지 선택할 수 있습니다.

### 인증 게이트 작동 여부 확인하기 (Verifying the gate is on)

```bash
# 환경 변수를 이용한 간단한 경로
HERMES_DASHBOARD_OAUTH_CLIENT_ID=agent:test \
  hermes dashboard --host 0.0.0.0

# config.yaml을 이용한 동치 방식 (로컬 개발 / 온프레미스에서 권장):
#
#   dashboard:
#     oauth:
#       client_id: agent:test
#
# 그런 다음 켜주기만 하면 됨:
hermes dashboard --host 0.0.0.0

# 게이트 상태 확인을 위해 /api/status 조회:
curl -s http://127.0.0.1:9119/api/status | jq '.auth_required, .auth_providers'
# true
# ["nous"]
```

대시보드의 React StatusPage 안 "Web server" 항목에서도 위와 같은 내용을 확인할 수 있습니다. 성공적으로 로그인하고 나면 사이드바의 AuthWidget이 당신의 현재 신원(identity)을 띄워 줍니다.

## 원격 백엔드에 Hermes Desktop 연결하기 (Connecting Hermes Desktop to a remote backend)

Hermes Desktop 앱은 (VPS, 홈 서버, Tailscale망 안의 Mac Mini 등) 다른 기기에서 도는 Hermes 백엔드를 직접 제어할 수 있습니다. 앱 내비게이션의 **Settings(설정) → Gateway(게이트웨이) → Remote gateway(원격 게이트웨이)** 항목에 가보면, 이 백엔드에 연결하기 위해 **Remote URL(원격 URL)** 과 인증용 **Sign in(로그인)** 버튼을 요구합니다. (데스크톱 앱 그 자체의 설치법이나 설정, 채팅에 대한 정보는 [Hermes Desktop](/user-guide/desktop) 페이지를 참고해 주세요.)

여러분은 원격 대시보드를 번들 제공자 중 하나로 막아두고(protect), 데스크톱 앱은 백엔드가 보내주는 해당 제공자의 인증 수단에 맞춰 로그인을 진행하게 됩니다. 본인의 PC 범위를 넘어선 백엔드 — VPS, 퍼블릭 호스트, 인터넷에 열려 있는 모든 것 — 의 경우 가장 권장되는 인증 제공자는 **OAuth (Nous Portal)** 입니다 ([`hermes dashboard register`](#registering-a-dashboard)로 등록하고, *Sign in with Nous Research*로 로그인하세요). 번들로 들어있는 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp)는 백엔드가 신뢰할 수 있는 LAN 환경이나 오로지 VPN을 통해서만 도달할 수 있을 때 가장 빠르게 올릴 수 있는 세팅이지만, **공개 인터넷에 직접 드러내기엔 매우 부적합합니다**. 대시보드를 루프백 밖으로 바인딩하는 행위 자체가 그 즉시 인증 게이트의 잠금장치를 작동시킵니다; 한 번 로그인에 성공하고 나면 Desktop 앱이 WebSocket 기반의 채팅을 띄우기 위해 기존 세션을 알아서 계속 재활용합니다 — 토큰을 복사해서 붙여넣고 하는 번거로운 작업은 아예 없습니다.

아래 소개하는 절차는 신뢰할 수 있는 네트워크에서 가장 빠르게 구성할 수 있는 사용자 이름/비밀번호 방식을 보여줍니다; OAuth 연동 방식을 원하신다면 [기본 제공자: Nous Research](#default-provider-nous-research)를 살펴보세요.

### 백엔드에서 (원격 머신) (On the backend (the remote machine))

```bash
# 1. ~/.hermes/.env 파일에 대시보드 로그인용 자격 증명을 설정합니다 (비밀키 파일, 권한 0600).
cat >> ~/.hermes/.env <<'EOF'
HERMES_DASHBOARD_BASIC_AUTH_USERNAME=admin
HERMES_DASHBOARD_BASIC_AUTH_PASSWORD=choose-a-strong-password
# 권장 사항: 세션이 재시작 시에도 끊기지 않도록 안정적인 서명 키를 설정해 줍니다.
HERMES_DASHBOARD_BASIC_AUTH_SECRET=$(openssl rand -base64 32)
EOF
chmod 600 ~/.hermes/.env

# 2. 도달 가능한 범위로 대시보드 데몬을 바인딩해 시작합니다. 이처럼 루프백이 아닌
#    바인딩은 인증 게이트를 걸어 잠그게 만들며, 사용자 이름/비밀번호 제공자가 로그인을 주관합니다.
hermes dashboard --no-open --host 0.0.0.0 --port 9119
```

평문을 저장소에 남기고 싶지 않으신가요? 비밀번호 대신 scrypt 해시를 넣어 `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH` 항목을 사용해 보세요 — 모든 옵션에 대해서는 [사용자 이름/비밀번호 제공자](#usernamepassword-provider-no-oauth-idp) 항목을 참고하시면 됩니다.

만약 대시보드를 systemd 서비스로 띄우고 계신다면, 서비스 유닛(unit) 설정에 `EnvironmentFile=%h/.hermes/.env`를 추가해 주세요. 그러면 부팅 시 알아서 이 자격 증명 환경 변수를 싹 다 읽어 들입니다.

:::warning
대시보드는 여러분의 `.env`(API 키, 비밀 정보들)를 마음껏 읽고 쓰며, 심지어 에이전트 명령어까지 실행할 수 있는 막강한 권한을 갖고 있습니다. 여기서 예시로 보여드린 **사용자 이름/비밀번호** 방식은 완전히 신뢰할 수 있는 네트워크용입니다 — 비밀번호로 막아뒀다고 해서 이런 통짜 대시보드를 바깥 인터넷에 그대로 노출하는 짓은 절대 하지 마세요. 꼭 VPN을 한 겹 씌워주세요. [Tailscale](https://tailscale.com/)이 가장 깔끔한 대안이 됩니다: 머신의 tailscale IP(`--host <tailscale-ip>`)로 바인딩하고 리모트 URL에 `http://<tailscale-ip>:9119`를 박아두면 오로지 자신의 tailnet(사설망)에 접속된 기기들만 여기에 닿을 수 있습니다. 공개 인터넷 환경 너머의 백엔드를 통제하고 싶으시다면 무조건 **OAuth (Nous Portal)** 제공자를 쓰시기 바랍니다.
:::

### Hermes Desktop 앱에서 (In Hermes Desktop)

**Settings(설정) → Gateway(게이트웨이) → Remote gateway(원격 게이트웨이):**

- **Remote URL(원격 URL)** — `http://<backend-host>:9119` (리버스 프록시를 사용할 경우 `/hermes`와 같은 하위 경로(path prefix)까지 입력해도 잘 작동합니다)
- **Sign in(로그인)** — 앱은 사용자 이름/비밀번호 기반 게이트웨이를 알아채고 **Sign in** 버튼을 띄웁니다; 버튼을 클릭하고 1단계에서 설정한 자격 증명을 입력하세요
- **Save and reconnect(저장 및 재연결)** — 데스크톱 앱의 내부 통신 창구가 즉시 원격 백엔드 쪽으로 전환됩니다

세션은 백그라운드에서 자체적으로 새로고침(refresh)되며, 백엔드에 `HERMES_DASHBOARD_BASIC_AUTH_SECRET`을 설정해 두었을 경우 백엔드가 재부팅되어도 로그인이 풀리지 않습니다.

### 환경 변수로 덮어쓰기 (Environment-variable override)

앱 내 설정을 켜서 URL을 박아두는 대신, 데스크톱 앱을 실행하기 전에 환경 변수 하나로 백엔드의 위치를 강제 지정해버릴 수 있습니다. 이 `HERMES_DESKTOP_REMOTE_URL` 변수를 세팅하고 앱을 켜면 앱의 자체 저장 설정값이 무시되며 (Gateway 세팅 창에는 "env override" 뱃지가 뜨고 입력란이 비활성화됨); 단, **Sign in(로그인)** 버튼을 눌러 사용자 이름과 비밀번호를 넣고 들어가는 절차는 UI상에서 그대로 진행하셔야 합니다.

| 환경 변수 | 값 |
|---------|-------|
| `HERMES_DESKTOP_REMOTE_URL` | `http://<backend-host>:9119` |

### 문제 해결 (Troubleshooting)

- **"Remote gateway incomplete"** — 원격 URL을 아직 입력하지 않으셨습니다.
- **로그인 시 401 / "Invalid credentials" (잘못된 자격 증명) 에러** — 사용자 이름이나 비밀번호가 백엔드에 기록된 `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` / `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`와 맞지 않습니다. 백엔드 데몬은 모르는 아이디를 넣었을 때나 비밀번호를 틀렸을 때나 언제나 똑같은 뭉뚱그려진(generic) 에러만을 뱉기 때문에 양쪽 다 점검하셔야 합니다. 게이트 문이 잘 닫혀있는지 `curl -s http://<host>:9119/api/status | jq '.auth_required, .auth_providers'`로 찔러보세요 — `true`라고 나오면서 목록 안에 `"basic"`이 들어있어야 정상입니다.
- **"Sign in" 버튼이 없고 뜬금없이 세션 토큰부터 달라고 하는 경우** — 사용자 이름/비밀번호 제공자가 꺼져있는 겁니다 (`/api/status` 결과에 `"basic"`이 없으면 100%입니다). 환경 변수 안에 사용자 이름과 비밀번호(혹은 비밀번호 해시값)가 다 잘 적혀 있는지, 대시보드 데몬이 켜질 때 저걸 제대로 빨아들였는지 다시 확인하세요.
- **재시작할 때마다 자꾸 로그아웃이 됨** — `HERMES_DASHBOARD_BASIC_AUTH_SECRET` 항목에 한 번 고정된(stable) 값을 박아 넣어두세요; 저게 없으면 백엔드 데몬이 켜질 때마다 매번 새로운 서명용 키를 스스로 파내기 때문에 과거 발급한 세션이 다 날아가 버립니다.
- **Connection refused (연결이 거부됨) / 시간 초과(timeout)** — 백엔드가 도달 가능한 주소가 아닌 `127.0.0.1` (기본값)에 바인딩되어 있거나, 방화벽/VPN이 해당 포트를 꽉 막고 있는 상황입니다. `0.0.0.0` 또는 Tailscale용 IP로 바인딩 주소를 바꾸고 본인의 신뢰할 수 있는 네트워크 대역에 한해 그 포트를 활짝 열어주세요.

## CORS

웹 서버는 CORS를 localhost origin에만 엄격하게 한정 짓습니다:

- `http://localhost:9119` / `http://127.0.0.1:9119` (운영 환경)
- `http://localhost:3000` / `http://127.0.0.1:3000`
- `http://localhost:5173` / `http://127.0.0.1:5173` (Vite 개발 서버)

사용자 지정 포트로 서버를 띄울 경우, 그 출처(origin)는 화이트리스트에 자동으로 추가됩니다.

## 개발 (Development)

웹 대시보드 프론트엔드 작업에 기여(contribute)하시려는 경우:

```bash
# 터미널 1: 백엔드 API 기동
hermes dashboard --no-open

# 터미널 2: HMR을 지원하는 Vite 개발 서버 기동
cd web/
npm install
npm run dev
```

`http://localhost:5173`에 올라간 Vite 개발 서버는 프론트에서 날아오는 `/api` 요청을 `http://127.0.0.1:9119`에 올라간 FastAPI 백엔드로 고스란히 프록시 처리(proxy)합니다.

프론트엔드 소스는 React 19, TypeScript, Tailwind CSS v4, 그리고 shadcn/ui 스타일의 컴포넌트로 짜여 있습니다. 프로덕션 빌드(production build)는 `hermes_cli/web_dist/` 폴더에 최종 산출물을 내뱉으며, 이렇게 모인 SPA 정적 파일들을 FastAPI 서버가 서빙(serve)하게 됩니다.

## 업데이트 시 자동 빌드 (Automatic Build on Update)

`hermes update` 명령어를 실행할 때 운영체제에 `npm` 명령어가 깔려 있다면 시스템은 웹 프론트엔드를 자동으로 다시 빌드합니다. 이를 통해 소스 코드가 업데이트되었을 때 대시보드 역시 그 보폭을 맞출 수 있습니다. 만일 `npm`이 안 깔려 있다면 프론트엔드 빌드를 건너뛰며, 나중에 `hermes dashboard`를 처음 구동할 때 알아서 빌드를 시도하게 될 것입니다.

## 테마 및 플러그인 (Themes & plugins)

대시보드에는 6가지 기본 테마가 내장되어 있으며, 사용자 정의 테마, 플러그인 탭 및 백엔드 API 라우트 기능을 통해 확장 가능합니다 — 저장소를 직접 클론할 필요 없이 단순히 끼워 넣기(drop-in)만 하면 됩니다.

**실시간 테마 변경** 헤더 바(header bar)에서 언어 변경 버튼 옆에 있는 팔레트 아이콘을 클릭하세요. 선택한 테마는 `config.yaml`의 `dashboard.theme` 아래에 저장되며 페이지를 새로고침해도 그대로 유지됩니다.

내장 테마 목록:

| 테마 | 특징 |
|-------|-----------|
| **Hermes Teal** (`default`) | 짙은 청록색(Dark teal) + 크림색, 시스템 글꼴, 편안한 여백 |
| **Hermes Teal (Large)** (`default-large`) | 기본과 동일하나 18px 텍스트와 더 넉넉한 여백 제공 |
| **Midnight** (`midnight`) | 깊은 청자색(Deep blue-violet), Inter + JetBrains Mono 글꼴 |
| **Ember** (`ember`) | 따뜻한 크림슨 + 브론즈, Spectral 세리프 + IBM Plex Mono 글꼴 |
| **Mono** (`mono`) | 흑백(Grayscale) 테마, IBM Plex 글꼴, 콤팩트한 구성 |
| **Cyberpunk** (`cyberpunk`) | 검정 바탕에 네온 그린 포인트, Share Tech Mono 글꼴 |
| **Rosé** (`rose`) | 분홍 + 상아색(ivory), Fraunces 세리프, 넓은 여백 |

자신만의 테마를 만들거나 플러그인 탭 추가, 쉘 슬롯(shell slots)에 위젯 주입, 플러그인 전용 REST 엔드포인트 등을 구현하려면 **[대시보드 확장 (Extending the Dashboard)](./extending-the-dashboard)** 를 참조하세요 — 이 전체 가이드에서 다루는 내용은 다음과 같습니다:

- 테마 YAML 스키마 — 색상 팔레트, 타이포그래피(typography), 레이아웃, 에셋(assets), componentStyles, colorOverrides, customCSS
- 레이아웃 종류 — `standard`, `cockpit`, `tiled`
- 플러그인 매니페스트(manifest), SDK, 쉘 슬롯(shell slots), 페이지 범위(page-scoped) 슬롯(기존 페이지를 덮어쓰지 않고 위젯만 주입), 백엔드 FastAPI 라우트
- 테마와 플러그인이 결합된 종합 튜토리얼 (Strike Freedom 콕핏(cockpit) 데모)
- 탐색(Discovery), 리로드(reload) 및 문제 해결(troubleshooting)

---
sidebar_position: 7
title: "Docker"
description: "Docker에서 Hermes Agent 실행 및 Docker를 터미널 백엔드로 사용하기"
---

# Hermes Agent — Docker

Docker가 Hermes Agent와 연동되는 방식은 서로 다른 두 가지입니다.

1. **Docker 내부에서 Hermes 실행** — 에이전트 자체가 컨테이너 안에서 실행됩니다(이 페이지의 주요 내용).
2. **Docker를 터미널 백엔드로 사용** — 에이전트는 호스트에서 실행되지만 모든 명령을 단일 영구 Docker 샌드박스 컨테이너 안에서 실행합니다. 이 컨테이너는 Hermes 프로세스가 살아 있는 동안 도구 호출, `/new`, 서브에이전트 사이에서도 유지됩니다([설정 → Docker 백엔드](./configuration.md#docker-backend) 참조).

이 페이지에서는 방법 1을 다룹니다. 컨테이너는 호스트에서 `/opt/data`로 마운트한 단일 디렉터리에 모든 사용자 데이터(설정, API 키, 세션, 스킬, 메모리)를 저장합니다. 이미지 자체에는 상태가 없으므로 설정을 잃지 않고 새 버전을 가져와 업그레이드할 수 있습니다.

## 빠른 시작

Hermes Agent를 처음 실행한다면 호스트에 데이터 디렉터리를 만들고 컨테이너를 대화형으로 시작하여 설정 마법사를 실행하세요.

:::caution 브라우저 기반 VPS 콘솔에서 설치 명령을 실행하지 마세요
일부 VPS 제공업체(Hetzner Cloud 및 기타 여러 업체)는 호스트 관리용 브라우저 기반 콘솔을 제공합니다. 이러한 콘솔은 특수 문자를 잘못 전송합니다. `:`가 `;`로 입력되거나 `@`가 잘못 표시될 수 있고, 영어가 아닌 키보드 레이아웃에서는 문제가 더 심해집니다. 이로 인해 `-v ~/.hermes:/opt/data`, `-e KEY=value`, 붙여 넣은 API 키/토큰과 같은 `docker run` 인수가 조용히 손상될 수 있습니다.

명령을 안전하게 복사해 붙여 넣으려면 대신 SSH로 접속하세요(**`ssh root@<host>`**). 브라우저 콘솔을 사용해야 한다면 명령을 붙여 넣지 말고 직접 입력하세요. Enter를 누르기 전에 결과의 모든 `:`, `@`, `=`, `/`를 다시 확인하세요.
:::

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

설정 마법사가 시작되고 API 키를 입력하면 `~/.hermes/.env`에 기록됩니다. 이 작업은 한 번만 하면 됩니다. 이 시점에 게이트웨이가 연결할 채팅 시스템을 설정하는 것을 적극 권장합니다.

:::tip
컨테이너 안에서 `hermes setup --portal`을 한 번 실행하세요. 새로 고침 토큰은 마운트된 `~/.hermes` 볼륨에 유지됩니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 게이트웨이 모드로 실행

설정을 마쳤다면 컨테이너를 영구 게이트웨이(Telegram, Discord, Slack, WhatsApp 등)로 백그라운드에서 실행하세요.

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

포트 8642는 게이트웨이의 [OpenAI 호환 API 서버](./features/api-server.md)와 상태 확인 엔드포인트를 공개합니다. 채팅 플랫폼(Telegram, Discord 등)만 사용한다면 선택 사항이지만, 대시보드나 외부 도구가 게이트웨이에 연결되게 하려면 필요합니다.

:::tip 게이트웨이는 감독됩니다
공식 Docker 이미지에서는 `gateway run`이 **s6-overlay에 의해 자동으로 감독됩니다**. 게이트웨이 프로세스가 충돌하면 컨테이너를 잃지 않고 몇 초 안에 재시작되며, `HERMES_DASHBOARD=1`이 설정된 경우 대시보드도 함께 감독됩니다. `gateway run` CMD 프로세스 자체는 컨테이너를 계속 실행시키는 `sleep infinity` 하트비트입니다. 실제 게이트웨이 프로세스는 s6이 관리하므로 `docker stop`은 모든 프로세스를 깔끔하게 종료하고, `docker logs`에는 감독되는 게이트웨이의 출력이 표시됩니다.

`docker logs`에서 업그레이드를 확인하는 한 줄짜리 흔적을 볼 수 있습니다. 이 동작을 끄고 예전처럼 "게이트웨이가 컨테이너의 주 프로세스이며 컨테이너 종료 = 게이트웨이 종료" 의미를 사용하려면 `--no-supervise`를 전달하거나 `HERMES_GATEWAY_NO_SUPERVISE=1`을 설정하세요. 이 옵션은 컨테이너가 게이트웨이 상태 코드로 종료되기를 원하는 CI 스모크 테스트에 유용합니다. 프로덕션 배포에서는 기본 감독 방식이 훨씬 낫습니다.

이 동작은 s6 기반 이미지만 해당합니다. 이전(tini 기반) 이미지는 여전히 포그라운드 주 프로세스로 `gateway run`을 실행합니다.
:::

:::note 게이트웨이 로그가 저장되는 위치
전체 라우팅 표는 아래의 [로그가 저장되는 위치](#where-the-logs-go) 섹션을 참조하세요(프로필별 게이트웨이, 대시보드, 부팅 조정기, 컨테이너 전체 `docker logs`).
:::

:::note 무인 게이트웨이의 도구 루프 강제 중단
`tool_loop_guardrails.hard_stop_enabled` 설정의 기본값은 `false`입니다. 반복되는 도구 호출 경고를 사람이 볼 수 있는 대화형 CLI와 TUI 세션에서는 합리적인 값입니다. 하지만 무인 게이트웨이나 서버 배포에서는 경고만으로 반복 도구 호출 루프에 빠진 에이전트를 멈추지 못할 수 있습니다. 회로 차단기 동작을 원한다면 프로필의 `config.yaml`에서 강제 중단을 명시적으로 활성화하세요.

```yaml
tool_loop_guardrails:
  hard_stop_enabled: true
  hard_stop_after:
    exact_failure: 5
    idempotent_no_progress: 5
```
:::

참고: API 서버는 `API_SERVER_ENABLED=true`일 때 활성화됩니다. 컨테이너 내부에서 `127.0.0.1` 외부로 공개하려면 `API_SERVER_HOST=0.0.0.0`과 `API_SERVER_KEY`도 설정하세요(최소 8자이며 `openssl rand -hex 32`로 생성할 수 있습니다). 예시는 다음과 같습니다.

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -e API_SERVER_ENABLED=true \
  -e API_SERVER_HOST=0.0.0.0 \
  -e API_SERVER_KEY="$(openssl rand -hex 32)" \
  -e API_SERVER_CORS_ORIGINS='*' \
  nousresearch/hermes-agent gateway run
```

인터넷에 연결된 시스템에서 포트를 여는 것은 보안 위험입니다. 위험을 이해하지 못한다면 실행하지 마세요.

## 대시보드 실행

내장 웹 대시보드는 게이트웨이와 같은 컨테이너에서 감독되는 s6-rc 서비스로 실행됩니다. 대시보드를 시작하려면 `HERMES_DASHBOARD=1`을 설정하세요.

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  -p 9119:9119 \
  -e HERMES_DASHBOARD=1 \
  nousresearch/hermes-agent gateway run
```

대시보드는 s6이 감독하므로 충돌하면 짧은 대기 후 `s6-supervise`가 자동으로 재시작합니다. 대시보드의 stdout/stderr는 `docker logs <container>`로 전달됩니다(접두사 없음). 게이트웨이 자체의 출력은 이제 프로필별 s6-log 파일에 저장되므로 두 스트림이 섞이지 않습니다. 자세한 내용은 아래 [로그가 저장되는 위치](#where-the-logs-go)를 참조하세요.

| 환경 변수 | 설명 | 기본값 |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | 감독되는 대시보드 서비스를 활성화하려면 `1`(또는 `true` / `yes`)로 설정 | *(설정되지 않음 — 서비스는 등록되지만 중지 상태로 유지됨)* |
| `HERMES_DASHBOARD_HOST` | 대시보드 HTTP 서버의 바인드 주소 | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | 대시보드 HTTP 서버의 포트 | `9119` |
| `HERMES_DASHBOARD_INSECURE` | **지원 중단됨 / 동작하지 않음.** 이전에는 인증 게이트를 우회했지만, 2026년 6월 보안 강화 이후 인증을 비활성화하지 않습니다. 루프백이 아닌 바인드에는 항상 인증 제공자가 필요합니다 | *(무시됨 — 대신 제공자를 설정하세요)* |

컨테이너 내부의 대시보드는 기본적으로 `0.0.0.0`에 바인드됩니다. 그렇지 않으면 게시된 `-p 9119:9119` 포트에 호스트에서 연결할 수 없습니다. 사이드카/리버스 프록시 설정을 위해 바인드를 컨테이너 루프백으로 제한하려면 `HERMES_DASHBOARD_HOST=127.0.0.1`을 설정하세요.

대시보드의 인증 게이트는 다음 두 조건이 모두 참일 때 자동으로 작동합니다.

1. 바인드 호스트가 루프백이 아닙니다(예: 컨테이너 내부의 기본값 `0.0.0.0`).
2. `DashboardAuthProvider` 플러그인이 등록되어 있습니다.

두 번째 조건을 충족하는 기본 제공 방법은 세 가지입니다.

- **사용자 이름/비밀번호** — 신뢰할 수 있는 네트워크나 VPN 뒤의 셀프 호스팅/온프레미스/홈랩 컨테이너에 가장 간단한 방법입니다. `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD`를 설정하고, 재시작 후에도 세션을 유지하려면 `HERMES_DASHBOARD_BASIC_AUTH_SECRET`도 설정하세요. 인터넷에 직접 공개하기에는 적합하지 않습니다.
- **OAuth(Nous Portal)** — 호스팅/공개 배포용입니다. `HERMES_DASHBOARD_OAUTH_CLIENT_ID`가 설정되면 `dashboard_auth/nous` 제공자가 활성화됩니다.
- **셀프 호스팅 OIDC** — 표준 OpenID Connect를 통해 자체 ID 제공자로 인증합니다. `HERMES_DASHBOARD_OIDC_ISSUER` + `HERMES_DASHBOARD_OIDC_CLIENT_ID`가 설정되면 `dashboard_auth/self_hosted` 제공자가 활성화됩니다.

어떤 방법을 선택하든 게이트는 보호된 경로에 도달하기 전에 호출자를 로그인 페이지로 리디렉션합니다. 세 제공자에 대한 자세한 내용은 [웹 대시보드 → 인증](features/web-dashboard.md#authentication-gated-mode)을 참조하세요.

제공자가 등록되지 않았고 바인드가 루프백이 아니면 대시보드는 시작 시 누락된 환경 변수를 알려 주는 특정 오류와 함께 **안전하게 실패합니다**. 공개 바인드에서 인증 없이 대시보드를 제공하는 우회 방법은 더 이상 없습니다. `HERMES_DASHBOARD_INSECURE=1`은 이제 지원 중단된 동작 없음(no-op)이며(경고를 기록하고 무시됨), 제공자를 설정하거나 `HERMES_DASHBOARD_HOST=127.0.0.1`로 바인드한 뒤 SSH 터널/Tailscale을 통해 대시보드에 접속하세요.

:::warning `--insecure`가 제거된 이유
인증되지 않은 공개 대시보드는 2026년 6월 MCP 설정 지속성 공격의 진입점이었습니다. 인터넷 스캐너가 노출된 대시보드(및 OpenAI API 서버)에 접근한 뒤 에이전트가 SSH 키 백도어를 심도록 유도했습니다. 이제 루프백이 아닌 모든 바인드에서는 인증 게이트가 필수입니다. 신뢰할 수 있는 LAN/홈랩 장비에서는 기본 제공 사용자 이름/비밀번호 제공자(`HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `_PASSWORD`)가 이를 충족하는 인프라 없는 방법입니다.
:::

대시보드를 별도 컨테이너로 실행하는 것도 지원됩니다. 단, 해당 컨테이너가 호스트 PID 및 네트워크 네임스페이스를 공유해야 합니다(예: 저장소 자체의 `docker-compose.yml`처럼 `network_mode: host` 사용 — 해당 파일의 `dashboard` 서비스를 참조하세요). 게이트웨이 활성 상태 감지는 게이트웨이 프로세스와 공유 PID 네임스페이스를 필요로 하므로, 이 제한은 공유 PID 네임스페이스가 없는 격리된 브리지 네트워크 컨테이너에서 실행하는 대시보드에만 적용됩니다.

## 대화형 실행(CLI 채팅)

실행 중인 데이터 디렉터리에 연결된 대화형 채팅 세션을 열려면 다음을 실행하세요.

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

또는 이미 실행 중인 컨테이너에서 터미널을 열었다면(예: Docker Desktop 사용) 다음만 실행하세요.

```sh
/opt/hermes/.venv/bin/hermes
```

## 영구 볼륨

`/opt/data` 볼륨은 모든 Hermes 상태의 단일 기준 원본입니다. 호스트의 `~/.hermes/` 디렉터리에 매핑되며 다음을 포함합니다.

| 경로 | 내용 |
|------|----------|
| `.env` | API 키와 비밀값 |
| `config.yaml` | 모든 Hermes 설정 |
| `SOUL.md` | 에이전트 성격/정체성 |
| `sessions/` | 대화 기록 |
| `memories/` | 영구 메모리 저장소 |
| `skills/` | 설치된 스킬 |
| `home/` | Hermes 도구 하위 프로세스(`git`, `ssh`, `gh`, `npm`, 스킬 CLI)의 프로필별 HOME |
| `cron/` | 예약된 작업 정의 |
| `hooks/` | 이벤트 훅 |
| `logs/` | 런타임 로그 |
| `skins/` | 사용자 지정 CLI 스킨 |

### 변경할 수 없는 설치 트리

호스팅 및 게시된 Docker 이미지에서 `/opt/hermes`는 설치된 애플리케이션 트리입니다. 런타임 `hermes` 사용자가 소유하지 않으며 읽기 전용이므로 에이전트 턴, 게이트웨이 세션, 대시보드 작업, 일반적인 `docker exec hermes hermes ...` 명령으로 핵심 소스, 번들된 `.venv`, `node_modules`, TUI 번들을 직접 수정할 수 없습니다.

변경 가능한 Hermes 상태는 모두 `/opt/data` 아래에 있어야 합니다. 여기에는 설정, `.env`, 프로필, 스킬, 메모리, 세션, 로그, 대시보드 업로드, 플러그인 및 기타 사용자가 관리하는 파일이 포함됩니다. 이미지에서는 런타임 `.pyc` 쓰기와 Hermes의 `/opt/hermes` 내부 지연 의존성 설치도 비활성화합니다. 게시된 이미지에 필요한 선택적 플랫폼 의존성은 이미지에 미리 포함하거나 새 이미지 빌드를 통해 설치해야 합니다.

호스팅/게시된 이미지에서 에이전트의 자기 개선 범위는 `/opt/data` 아래의 스킬, 메모리, 플러그인, 설정으로 제한됩니다. `/opt/hermes`의 설치된 핵심 소스는 변경할 수 없으며, 핵심 변경은 실행 중인 설치를 직접 수정하는 대신 저장소에 PR을 보내고 이미지를 업데이트하여 반영합니다.

`/opt/data` 외부의 파일을 복구하거나 검사해야 한다면 의도적으로 root 셸을 사용하세요. `hermes` 셸은 일반적으로 `docker exec hermes hermes ...`를 런타임 사용자로 되돌립니다. 일회성 root 호출이 명시적으로 필요할 때는 `HERMES_DOCKER_EXEC_AS_ROOT=1`을 설정하세요.

`~` 아래에 자격 증명을 저장하는 스킬 CLI는 데이터 볼륨 루트가 아니라 하위 프로세스 HOME을 대상으로 초기화해야 합니다. 예를 들어 [xurl 스킬](./skills/bundled/social-media/social-media-xurl.md)은 OAuth 상태를 `~/.xurl`에 저장합니다. 공식 Docker 구성에서는 Hermes 도구 호출이 이를 `/opt/data/home/.xurl`로 읽으므로 `HOME=/opt/data/home`으로 수동 xurl 인증을 실행하고 `HOME=/opt/data/home xurl auth status`로 확인하세요.

:::warning
동일한 데이터 디렉터리에서 두 개의 Hermes **게이트웨이** 컨테이너를 동시에 실행하지 마세요. 세션 파일과 메모리 저장소는 동시 쓰기를 지원하도록 설계되지 않았습니다.
:::

## 다중 프로필 지원

Hermes는 [여러 프로필](../reference/profile-commands.md)을 지원합니다. 프로필은 하나의 설치에서 독립 에이전트(서로 다른 SOUL, 스킬, 메모리, 세션, 자격 증명)를 실행할 수 있게 해 주는 별도의 `~/.hermes/` 하위 디렉터리입니다. **공식 Docker 이미지 내부에서는 s6 감독 트리가 각 프로필을 독립적인 감독 서비스로 취급하므로 모든 프로필을 하나의 컨테이너에서 호스팅하는 것이 권장 배포 방식입니다.**

`hermes profile create <name>`으로 만든 각 프로필에는 다음이 제공됩니다.

- 런타임에 동적으로 등록되는 `/run/service/gateway-<name>/` 전용 s6 서비스 슬롯. 컨테이너를 다시 빌드할 필요가 없습니다.
- `s6-supervise`가 관리하는 백오프와 충돌 시 자동 재시작.
- `${HERMES_HOME}/logs/gateways/<name>/current`에 저장되는 프로필별 순환 로그(1MB 파일 10개).
- 컨테이너 재시작 후에도 상태가 유지됩니다. 부팅 시 조정기는 각 프로필 디렉터리의 `gateway_state.json`을 읽고 마지막 기록 상태가 `running`인 프로필만 슬롯을 다시 시작합니다. 명시적으로 중지한 게이트웨이(`hermes gateway stop`)만 재시작 후에도 중지 상태로 유지됩니다. 컨테이너 재시작, 이미지 업그레이드 또는 예기치 않은 종료는 기록된 상태를 `running`으로 남기므로 다음 부팅 시 게이트웨이가 자동으로 시작됩니다.

호스트에서 실행하는 수명 주기 명령은 컨테이너 안에서도 동일하게 동작합니다.

```sh
# Create a profile — registers the gateway-<name> s6 slot.
docker exec hermes hermes profile create coder

# Start / stop / restart — dispatches s6-svc; the gateway lifecycle survives docker restart.
docker exec hermes hermes -p coder gateway start
docker exec hermes hermes -p coder gateway stop
docker exec hermes hermes -p coder gateway restart

# Status — reports `Manager: s6 (container supervisor)` inside the container.
docker exec hermes hermes -p coder gateway status

# Remove a profile — tears down the s6 slot too.
docker exec hermes hermes profile delete coder
```

내부적으로 컨테이너 안의 `hermes gateway start/stop/restart`는 가로채어 올바른 서비스 디렉터리에 대해 `s6-svc`로 전달됩니다. s6 명령을 직접 배울 필요는 없습니다. 원시 감독기 상태를 확인하려면 `/command/s6-svstat /run/service/gateway-<name>`을 사용하세요(`/command/`는 감독 트리가 생성한 프로세스에만 PATH로 제공되므로 `docker exec`에서 호출할 때는 절대 경로를 전달해야 합니다).

### 컨테이너 외부에서 둘 이상의 프로필에 접근하기

외부에서 프로필의 게이트웨이에 접근하는 표면은 두 가지이며 서로 다르게 동작합니다. 둘을 혼동하지 마세요.

**Hermes Desktop(및 웹 대시보드).** Desktop 앱의 **Remote Gateway** 연결은 `hermes dashboard` 백엔드(기본 **포트 9119**, `HERMES_DASHBOARD=1`로 활성화됨)와 통신하며 OpenAI API 서버가 아닙니다. 하나의 대시보드 백엔드가 같은 위치에 있는 **모든** 프로필을 제공합니다. 앱의 프로필 전환기가 각 요청에 대상 프로필을 보내고 백엔드는 디스크에서 해당 프로필의 `HERMES_HOME`을 엽니다. 따라서 Desktop에서는 프로필마다 두 번째 포트나 두 번째 연결이 필요하지 않습니다. 하나의 `:9119` 연결로 전환기를 통해 모든 프로필에 접근할 수 있습니다.

**OpenAI 호환 API 클라이언트(Open WebUI, LobeChat, `/v1/...`).** 이 클라이언트는 각 프로필의 **API 서버**에 연결하며, 모든 프로필에서 **포트 8642**에 바인드됩니다(`API_SERVER_PORT` / `platforms.api_server.extra.port`에서 확인되며 자동 할당도 없고 `config.yaml`/`gateway.port` 키도 없습니다). 특정 두 번째 프로필에 연결하려면 해당 프로필의 **자체** `.env`에 다른 `API_SERVER_PORT`를 지정하세요. 그렇지 않으면 해당 게이트웨이도 8642에 바인드하려고 하여 기본 프로필과 충돌합니다.

```sh
# Create the profile (registers its gateway-<name> s6 slot)
docker exec hermes hermes profile create work

# Point its API server at a free port (write to the profile's own .env)
cat >> /opt/data/profiles/work/.env <<'EOF'
API_SERVER_ENABLED=true
API_SERVER_PORT=8643
EOF

docker exec hermes hermes -p work gateway restart
```

각 프로필의 **자체** `.env`에 `API_SERVER_PORT`를 유지하고, 컨테이너 전체 `environment:` 블록에는 절대 넣지 마세요. 전역 값은 모든 프로필을 같은 포트로 강제하여 충돌을 일으킵니다. 브리지 네트워킹에서는 `docker-compose.yml`에 추가 포트를 게시하세요(`- "8643:8643"`). `network_mode: host`에서는 이미 호스트에서 접근할 수 있습니다. 기본 프로필의 8642 연결에는 영향이 없습니다.

### 프로필 여러 개를 여러 컨테이너가 아닌 하나의 컨테이너에서 실행하는 이유

s6로 마이그레이션하기 전에는 컨테이너 내부에 여러 게이트웨이를 관리하는 감독기가 없었기 때문에 "프로필당 컨테이너 하나"가 권장 패턴이었습니다. PID 1로 s6을 사용하면서 더 이상 그럴 필요가 없고, 거의 모든 면에서 단일 컨테이너 구성이 더 간단합니다.

| | 하나의 컨테이너, 여러 프로필 | 프로필당 컨테이너 하나 |
|---|---|---|
| 디스크 오버헤드 | 이미지 하나, 번들된 venv 하나, Playwright 캐시 하나 | 이미지 N개 / 캐시 N개 |
| 메모리 오버헤드 | 공유 Python 인터프리터 캐시, 공유 node_modules | 컨테이너마다 중복 |
| 프로필 생성 | `docker exec ... hermes profile create <name>`(수 초) | 새 `docker run` 호출 + 포트 할당 + 바인드 마운트 설정 |
| 프로필별 충돌 복구 | `s6-supervise` 자동 재시작 | Docker의 `--restart unless-stopped`(느리고 형제 작업도 종료) |
| 로그 | `s6-log`를 통한 프로필별 순환 파일과 컨테이너 부팅 감사 로그 | 컨테이너마다 `docker logs <name>` — 기본 순환 없음 |
| 백업 | `~/.hermes` 디렉터리 하나 | 조정해야 할 디렉터리 N개 |

기본 프로필(`default`)은 첫 부팅 시 항상 등록되므로 새 컨테이너에는 감독되는 게이트웨이가 하나 기본으로 제공됩니다. 추가 프로필은 순수한 런타임 추가입니다.

### 별도 컨테이너가 필요한 경우

프로필을 컨테이너 안에서 실행하는 것이 기본입니다. 다음처럼 특별한 이유가 있을 때만 프로필마다 별도 컨테이너를 실행하세요.

- **작업별 리소스 격리** — 예를 들어 프로필 A의 폭주한 브라우저 도구 세션이 프로필 B를 OOM 상태로 만들지 못하게 합니다. 컨테이너에서는 프로필별로 `--memory` / `--cpus`를 지정할 수 있습니다.
- **독립적인 이미지 고정** — 작업마다 서로 다른 업스트림 이미지 태그를 사용합니다.
- **네트워크 분할** — 프로필마다 별도의 Docker 네트워크를 사용합니다(예: 고객용 네트워크와 내부 네트워크).
- **규정 준수 / 영향 범위** — 서로 다른 자격 증명이 OS 수준의 프로세스 트리를 공유하지 않습니다.

이 경우 프로필마다 서로 다른 `container_name`, `volumes`, `ports`로 서비스를 하나씩 선언하세요.

```yaml
services:
  hermes-work:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-work
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes-work:/opt/data

  hermes-personal:
    image: nousresearch/hermes-agent:latest
    container_name: hermes-personal
    restart: unless-stopped
    command: gateway run
    ports:
      - "8643:8642"
    volumes:
      - ~/.hermes-personal:/opt/data
```

[영구 볼륨](#persistent-volumes)의 경고는 여전히 적용됩니다. 두 컨테이너가 동시에 같은 `~/.hermes` 디렉터리를 가리키지 않도록 하세요. 각 컨테이너 안의 s6 감독기는 자체 프로필 집합을 관리하며, 컨테이너 간 데이터 볼륨을 공유하면 세션 파일과 메모리 저장소가 손상됩니다.

## 로그가 저장되는 위치

s6 컨테이너에는 네 가지 서로 다른 로그 표면이 있습니다. "왜 내 게이트웨이 로그가 `docker logs`에 표시되지 않지?"라는 혼란이 흔히 발생합니다. 요약하면 다음과 같습니다.

| 출처 | 저장 위치 | 읽는 방법 |
|---|---|---|
| **프로필별 게이트웨이**(`hermes gateway run` 및 s6 아래의 프로필별 게이트웨이) | 두 곳에 Tee됨: `docker logs <container>`(실시간, 추가 접두사 없음) 및 `${HERMES_HOME}/logs/gateways/<profile>/current`(순환, ISO-8601 타임스탬프, 1MB 파일 10개) | `docker logs -f hermes` 또는 호스트에서 `tail -F ~/.hermes/logs/gateways/default/current` |
| **대시보드**(`HERMES_DASHBOARD=1`인 경우) | `docker logs <container>`(접두사 없음) | 게이트웨이 줄과 섞여 있는 `docker logs -f hermes` |
| **부팅 조정기**(각 컨테이너 시작 시 복원된 프로필 게이트웨이를 기록) | `${HERMES_HOME}/logs/container-boot.log`(추가 전용 감사 로그) | `tail -F ~/.hermes/logs/container-boot.log` |
| **일반 Hermes 로그**(`agent.log`, `errors.log`) | `${HERMES_HOME}/logs/`(프로필별) | `docker exec hermes hermes logs --follow [--level WARNING] [--session <id>]` |

알아 두면 좋은 실질적인 결과는 두 가지입니다.

- `logs/gateways/<profile>/current`의 파일 사본은 컨테이너 재시작 후에도 유지됩니다. `docker logs`는 현재 컨테이너의 수명 동안 출력만 보존하며(`docker rm` 시 삭제됨), 순환 파일은 바인드 마운트된 볼륨에 남습니다.
- 부팅 조정기의 감사 줄 형식은 `<iso-timestamp> profile=<name> prior_state=<state> action=<registered|started>`입니다. `grep profile=coder ~/.hermes/logs/container-boot.log`를 실행하면 특정 프로필이 마지막으로 복원된 시점과 s6이 자동 시작했는지 빠르게 확인할 수 있습니다.

## 환경 변수 전달

API 키는 컨테이너 내부의 `/opt/data/.env`에서 읽습니다. 환경 변수를 직접 전달할 수도 있습니다.

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

직접 지정한 `-e` 플래그는 `.env`의 값을 덮어씁니다. 디스크에 키를 저장하고 싶지 않은 CI/CD 또는 시크릿 관리자 연동에 유용합니다.

:::note Docker를 **터미널 백엔드**로 사용하려고 하나요?
이 페이지는 Hermes 자체를 Docker 안에서 실행하는 방법을 다룹니다. Hermes가 에이전트의 `terminal` / `execute_code` 호출을 Docker 샌드박스 컨테이너 안에서 실행하게 하려면(모든 Hermes 프로세스가 공유하는 하나의 장기 실행 컨테이너 — 이슈 #20561 참조) 별도의 설정 블록이 필요합니다. `terminal.backend: docker`와 `terminal.docker_image`, `terminal.docker_volumes`, `terminal.docker_forward_env`, `terminal.docker_env`, `terminal.docker_run_as_host_user`, `terminal.docker_extra_args`, `terminal.docker_persist_across_processes`, `terminal.docker_orphan_reaper`를 설정하세요. 컨테이너 수명 주기 규칙을 포함한 전체 설정은 [설정 → Docker 백엔드](configuration.md#docker-backend)를 참조하세요.
:::

## Docker Compose 예시

게이트웨이와 대시보드를 함께 영구 배포하려면 `docker-compose.yaml`이 편리합니다.

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # gateway API
      - "9119:9119"   # dashboard (only reached when HERMES_DASHBOARD=1)
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # Uncomment to forward specific env vars instead of using .env file:
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

`docker compose up -d`로 시작하고 `docker compose logs -f`로 로그를 확인하세요. 감독되는 게이트웨이의 stdout도 볼륨의 `${HERMES_HOME}/logs/gateways/<profile>/current`에 Tee됩니다. 전체 라우팅 표는 [로그가 저장되는 위치](#where-the-logs-go)를 참조하세요.

## 선택 사항: Linux 데스크톱 오디오 브리지

Docker에서 음성 모드를 사용하려면 두 가지가 별도로 필요합니다. Hermes가 컨테이너 내부의 오디오 장치를 검사할 수 있어야 하고, 컨테이너가 호스트 오디오 서버에 연결할 수 있어야 합니다. 아래 설정은 PulseAudio 호환 소켓을 제공하는 Linux 데스크톱(많은 PipeWire 설정 포함)의 호스트 오디오 연결을 다룹니다.

:::caution
이는 일반적인 Docker Desktop 기능이 아니라 Linux 데스크톱용 우회 방법입니다. 호스트 오디오가 이미 작동하고 있으며 Hermes 컨테이너 안에서 CLI 음성 모드를 사용하려는 경우에 유용합니다. Hermes가 계속 `Running inside Docker container -- no audio devices`를 보고한다면 `PULSE_SERVER` / `PIPEWIRE_REMOTE`에 대한 Docker 오디오 탐색 지원이 포함된 빌드를 사용하세요.
:::

먼저 Compose 파일 옆에 ALSA 설정을 만드세요.

```conf title="asound.conf"
pcm.!default {
    type pulse
    hint {
        show on
        description "Default ALSA Output (PulseAudio)"
    }
}

pcm.pulse {
    type pulse
}

ctl.!default {
    type pulse
}
```

그런 다음 ALSA PulseAudio 플러그인이 설치된 작은 파생 이미지를 빌드하세요.

```dockerfile title="Dockerfile.audio"
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*
```

Compose에서 해당 이미지를 사용하고 호스트 사용자의 PulseAudio 소켓과 쿠키를 전달하세요.

```yaml
services:
  hermes:
    build:
      context: .
      dockerfile: Dockerfile.audio
    image: hermes-agent-audio
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    volumes:
      - ~/.hermes:/opt/data
      - /run/user/${HERMES_UID}/pulse:/run/user/${HERMES_UID}/pulse
      - ~/.config/pulse/cookie:/tmp/pulse-cookie:ro
      - ./asound.conf:/etc/asound.conf:ro
    environment:
      - HERMES_UID=${HERMES_UID}
      - HERMES_GID=${HERMES_GID}
      - XDG_RUNTIME_DIR=/run/user/${HERMES_UID}
      - PULSE_SERVER=unix:/run/user/${HERMES_UID}/pulse/native
      - PULSE_COOKIE=/tmp/pulse-cookie
```

컨테이너 프로세스가 사용자별 오디오 소켓에 접근할 수 있도록 호스트 UID/GID로 시작하세요.

```sh
export HERMES_UID="$(id -u)"
export HERMES_GID="$(id -g)"
docker compose up -d --build
```

컨테이너 내부에서 PortAudio가 인식하는 장치를 확인하려면 다음을 실행하세요.

```sh
docker exec hermes /opt/hermes/.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 리소스 제한

Hermes 컨테이너에는 중간 정도의 리소스가 필요합니다. 권장 최소값은 다음과 같습니다.

| 리소스 | 최소 | 권장 |
|----------|---------|-------------|
| 메모리 | 1 GB | 2–4 GB |
| CPU | 1 코어 | 2 코어 |
| 디스크(데이터 볼륨) | 500 MB | 2GB 이상(세션/스킬에 따라 증가) |

브라우저 자동화(Playwright/Chromium)는 가장 많은 메모리를 사용하는 기능입니다. 브라우저 도구가 필요하지 않다면 1GB로 충분합니다. 브라우저 도구를 활성화할 때는 최소 2GB를 할당하세요.

Docker에서 제한을 설정하세요.

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  --memory=4g --cpus=2 \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

## Dockerfile이 하는 일

공식 이미지는 `debian:13.4`를 기반으로 하며 다음을 포함합니다.

- 잠금 파일에서 `uv sync --frozen --no-install-project`를 사용해 번들된 추가 기능(`all`, `messaging`, Anthropic/Bedrock/Azure identity, Hindsight, Matrix)의 의존성을 동기화한 Python 3.13 및 Hermes 자체의 의존성 없는 editable 설치.
- Node.js 26 + npm(브라우저 자동화, WhatsApp 브리지, TUI/Desktop 번들 및 워크스페이스 빌드 도구용)
- Chromium이 포함된 Playwright(`npx playwright install --with-deps chromium --only-shell`)
- 시스템 유틸리티로 ripgrep, ffmpeg, git, `xz-utils`
- **`docker-cli`** — 컨테이너 안에서 실행되는 에이전트가 호스트의 Docker 데몬을 사용해 `docker build`, `docker run`, 컨테이너 검사 등을 수행할 수 있습니다(선택하려면 `/var/run/docker.sock`을 바인드 마운트).
- **`openssh-client`** — 컨테이너 내부에서 [SSH 터미널 백엔드](/user-guide/configuration#ssh-backend)를 사용할 수 있습니다. SSH 백엔드는 시스템 `ssh` 바이너리를 실행하므로 컨테이너 설치에 이것이 없으면 조용히 실패합니다.
- WhatsApp 브리지(`scripts/whatsapp-bridge/`)
- PID 1로 동작하는 **[`s6-overlay`](https://github.com/just-containers/s6-overlay) v3**(기존 `tini` 대체) — 대시보드와 프로필별 게이트웨이를 충돌 시 자동 재시작으로 감독하고, 좀비 하위 프로세스를 수거하며, 시그널을 전달합니다.

이미지는 `/opt/hermes`를 런타임에 변경할 수 없는 설치 트리로 취급합니다. Docker 내부에서 사용 가능해야 하는 선택적 Python 추가 기능, Node 워크스페이스, TUI 자산은 이미지 빌드 중에 포함해야 합니다. 런타임 지연 설치는 비활성화되므로 감독되는 게이트웨이와 `docker exec hermes …` 명령이 읽기 전용 소스 트리에 의존성 산출물을 쓰려고 하지 않습니다.

컨테이너의 `ENTRYPOINT`는 작은 디스패처(`docker/entrypoint-dispatch.sh`)입니다. 컨테이너가 PID 1을 소유할 때(일반적인 Docker / Podman) s6-overlay의 `/init`을 실행하고 아래에 설명된 전체 감독 트리를 사용합니다. 플랫폼이 자체 PID-1 init 아래에서 이미지 진입점을 감싸는 경우(Fly.io Machines, `docker run --init`, 일부 Nomad/Kubernetes 설정), `/init`은 `s6-overlay-suexec: fatal: can only run as pid 1`로 중단됩니다. 이때 디스패처는 대신 stage2 부트스트랩을 직접 실행하고 s6 없이 주 래퍼를 exec합니다. 이 대체 경로에서도 요청한 명령은 실행되지만 감독되는 서비스(대시보드, 프로필별 게이트웨이)는 사용할 수 없습니다.

PID 1 경로에서 `/init`은 다음을 수행합니다.
1. root로 `/etc/cont-init.d/01-hermes-setup`(=`docker/stage2-hook.sh`)을 실행합니다. 선택적 UID/GID 매핑, 볼륨 소유권 수정, 첫 부팅 시 `.env` / `config.yaml` / `SOUL.md` 초기값 생성, `HERMES_SKIP_CONFIG_MIGRATION=1`이 아닌 경우 비대화형 설정 스키마 마이그레이션, 번들 스킬 동기화를 수행합니다.
2. `/etc/cont-init.d/02-reconcile-profiles`(=`hermes_cli.container_boot`)를 실행합니다. `$HERMES_HOME/profiles/<name>/`을 순회하고 `/run/service/gateway-<profile>/` 아래에 프로필별 게이트웨이 s6 서비스 슬롯을 다시 만들며 마지막 기록 상태가 `running`인 프로필만 자동 시작합니다([프로필별 게이트웨이 감독](#per-profile-gateway-supervision) 참조).
3. 정적 `main-hermes` 및 `dashboard` s6-rc 서비스를 시작합니다.
4. 컨테이너의 CMD를 주 프로그램으로 exec합니다(`/opt/hermes/docker/main-wrapper.sh`). 이 프로그램은 사용자가 `docker run`에 전달한 인수를 다음과 같이 라우팅합니다.
   - 인수가 없으면 → `hermes`(기본값)
   - 첫 번째 인수가 PATH에 있는 실행 파일이면(예: `sleep`, `bash`) → 직접 exec
   - 그 외에는 → `hermes <args>`(하위 명령 전달)
   이 주 프로그램이 종료되면 해당 종료 코드와 함께 컨테이너도 종료됩니다.

:::warning s6 이전 이미지와 비교한 호환성 변경
이제 컨테이너 ENTRYPOINT는 `/usr/bin/tini`가 아니라 `entrypoint-dispatch.sh` 디스패처입니다(PID 1에서 s6-overlay의 `/init`으로 위임). 문서에 설명된 다섯 가지 `docker run` 호출 패턴(인수 없음, `chat -q "…"`, `sleep infinity`, `bash`, `--tui`)은 tini 기반 이미지와 동일하게 동작합니다. tini 특유의 시그널 동작에 의존하거나 `/usr/bin/tini --` 호출을 하드코딩한 다운스트림 래퍼가 있다면 이전 이미지 태그로 고정하세요.
:::

:::warning 권한 모델
명령 체인에 `/init`(또는 stage2 훅으로 전달하는 기존 `docker/entrypoint.sh` shim)을 유지하지 않는 한 이미지 진입점을 재정의하지 마세요. s6-overlay의 `/init`은 첫 부팅 시 볼륨에 chown할 수 있도록 root로 실행된 다음, 모든 감독 서비스와 주 프로그램을 `s6-setuidgid`를 통해 `hermes` 사용자로 낮춥니다. 공식 이미지에서 `hermes gateway run`을 root로 시작하면 `/opt/data`에 root 소유 파일이 남아 이후 대시보드나 게이트웨이 시작이 중단될 수 있으므로 기본적으로 거부됩니다. 의도적으로 그 위험을 수용할 때만 `HERMES_ALLOW_ROOT_GATEWAY=1`을 설정하세요.
:::

### `docker exec`는 자동으로 `hermes` 사용자로 권한을 낮춥니다

`docker exec hermes <cmd>`는 기본적으로 컨테이너 안에서 root로 실행되지만, 이미지에는 PATH의 가장 앞에 있는 얇은 shim `/opt/hermes/bin/hermes`가 포함되어 있습니다. 이 shim은 root 호출자를 감지하고 `s6-setuidgid hermes`를 통해 투명하게 다시 exec합니다. 따라서 `docker exec hermes login`, `docker exec hermes profile create …`, `docker exec hermes setup` 등은 모두 UID 10000이 소유한 파일을 작성하며, 감독되는 게이트웨이가 별도 `--user` 플래그 없이 읽을 수 있습니다. root가 아닌 호출자(감독되는 프로세스 자체, `docker exec --user hermes`, 컨테이너 내부의 kanban 서브에이전트)는 직접 venv 바이너리를 exec하는 단락 경로를 사용하므로 빈번한 경로에 오버헤드가 없습니다.

root 의미를 유지하는 `docker exec`가 정말 필요하다면 호출마다 다음처럼 선택 해제하세요.

```sh
docker exec -e HERMES_DOCKER_EXEC_AS_ROOT=1 hermes <cmd>
```

shim은 대소문자를 구분하지 않고 `1` / `true` / `yes`를 허용합니다. 그 외 값(예: `=0` 같은 오타)은 모두 권한을 낮추는 경로로 전달되므로 조용한 선택 해제가 불가능합니다. `s6-setuidgid`를 사용할 수 없다면(사용자 지정 빌드에서 s6-overlay를 제거한 경우) shim은 root로 실행하지 않고 종료 코드 126으로 종료합니다. 이를 통해 `docker exec hermes login`이 `auth.json`을 `root:root`로 작성해 모든 채팅 플랫폼 메시지에서 감독되는 게이트웨이 인증을 망가뜨리던 과거의 위험한 동작으로 돌아가지 않고, 깨진 권한 모델을 명확히 드러냅니다.

### 프로필별 게이트웨이 감독

`hermes profile create <name>`으로 만든 각 프로필은 자동으로 `/run/service/gateway-<name>/`에 상태가 유지되는 s6 감독 게이트웨이 서비스로 등록됩니다. 사용자 관점의 작업 흐름과 수명 주기 명령은 위의 [다중 프로필 지원](#multi-profile-support)을 참조하세요.

**s6 이전 이미지와 비교한 감독의 이점:**

- 게이트웨이 충돌은 약 1초의 백오프 후 `s6-supervise`가 자동으로 재시작합니다.
- `HERMES_DASHBOARD=1`로 활성화한 대시보드도 같은 감독 트리에서 감독되며 동일하게 자동 재시작됩니다.
- `docker restart`, 이미지 업그레이드(`docker compose up -d --force-recreate`), 예기치 않은 종료 후에도 실행 중인 게이트웨이가 유지됩니다. cont-init 조정기가 `$HERMES_HOME/profiles/<name>/gateway_state.json`을 읽고 마지막 기록이 `running`이면 슬롯을 다시 시작합니다. 명시적인 `hermes gateway stop`만 `stopped`를 기록해 재시작 후에도 게이트웨이를 중지 상태로 유지합니다. 재시작 또는 업그레이드 중 컨테이너/s6에 전송되는 SIGTERM은 "계속 실행 중"으로 처리되므로 자동 시작됩니다.
- 프로필별 게이트웨이 로그는 `$HERMES_HOME/logs/gateways/<profile>/current`에 유지되며(`s6-log`가 순환), 조정기의 작업은 부팅마다 `$HERMES_HOME/logs/container-boot.log`에 추가됩니다. 전체 라우팅 표는 [로그가 저장되는 위치](#where-the-logs-go)를 참조하세요.

컨테이너 내부에서 `hermes status`를 실행하면 `Manager: s6 (container supervisor)`가 보고됩니다. 원시 감독기 화면을 확인하려면 `/command/s6-svstat /run/service/gateway-<name>`을 사용하세요(`/command/`는 감독 트리 프로세스에만 PATH로 제공되므로 `docker exec`에서 호출할 때는 절대 경로를 전달해야 합니다).

## 업그레이드

최신 이미지를 가져온 뒤 컨테이너를 다시 만드세요. 데이터 디렉터리는 보존되며, 컨테이너는 게이트웨이를 시작하기 전에 마운트된 `$HERMES_HOME/config.yaml`에 대해 비대화형 설정 스키마 마이그레이션을 실행합니다. 마이그레이션이 필요하면 Hermes가 먼저 `config.yaml`과 `.env` 옆에 타임스탬프가 포함된 백업을 작성합니다.

```sh
docker pull nousresearch/hermes-agent:latest
docker rm -f hermes
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

또는 Docker Compose를 사용하세요.

```sh
docker compose pull
docker compose up -d
```

새 이미지가 다시 쓰기 전에 영구 설정을 수동으로 검사하거나 마이그레이션해야 할 때만 `HERMES_SKIP_CONFIG_MIGRATION=1`을 설정하세요.

## 스킬 및 자격 증명 파일

Docker를 실행 환경으로 사용할 때(위의 방법이 아니라 에이전트가 Docker 샌드박스 안에서 명령을 실행하는 경우 — [설정 → Docker 백엔드](./configuration.md#docker-backend) 참조), Hermes는 모든 도구 호출에 하나의 장기 실행 컨테이너를 재사용하고 스킬 디렉터리(`~/.hermes/skills/`)와 스킬이 선언한 모든 자격 증명 파일을 읽기 전용 볼륨으로 해당 컨테이너에 자동 바인드 마운트합니다. 스킬 스크립트, 템플릿, 참조 자료는 수동 설정 없이 샌드박스에서 사용할 수 있습니다. 컨테이너가 Hermes 프로세스의 수명 동안 유지되므로 설치한 의존성이나 작성한 파일도 다음 도구 호출에 남아 있습니다.

SSH 및 Modal 백엔드에서도 동일하게 동기화됩니다. 스킬과 자격 증명 파일은 각 명령 전에 rsync 또는 Modal 마운트 API를 통해 업로드됩니다.

## 컨테이너에 도구 더 설치하기

공식 이미지에는 엄선된 유틸리티 집합이 포함되어 있지만([Dockerfile이 하는 일](#what-the-dockerfile-does) 참조), 에이전트가 원하는 모든 도구가 사전 설치되어 있는 것은 아닙니다. 권장 방법은 노력과 지속성이 증가하는 순서로 다섯 가지입니다.

### npm 또는 Python 도구 — `npx` 또는 `uvx` 사용

npm 또는 PyPI에 게시된 도구는 Hermes에게 `npx`(npm) 또는 `uvx`(Python)를 통해 실행하도록 지시하고, 해당 명령을 영구 메모리에 기억하도록 하세요. 도구에 설정 파일이나 자격 증명이 필요하면 `/opt/data` 아래에 기록하도록 지시하세요(예: `/opt/data/<tool>/config.yaml`).

의존성은 필요할 때 가져와 컨테이너 수명 동안 캐시됩니다. `/opt/data` 아래에 작성한 설정은 호스트의 바인드 마운트 디렉터리에 있으므로 컨테이너 재시작 후에도 유지됩니다. 패키지 캐시 자체는 `docker rm` 후 다시 만들어지지만, 다음 도구 실행 시 `npx`와 `uvx`가 투명하게 다시 가져옵니다.

### 기타 도구(apt 패키지, 바이너리) — 설치하고 기억하기

npm이나 PyPI 외부의 도구( `apt` 패키지, 사전 빌드된 바이너리, 이미지에 아직 없는 언어 런타임)는 Hermes에게 설치 방법(예: `apt-get update && apt-get install -y <package>`)을 지시하고 설치 명령을 기억하도록 하세요. 도구는 컨테이너의 남은 수명 동안 유지되며, 컨테이너가 재시작되면 다음에 필요할 때 Hermes가 설치 명령을 다시 실행합니다.

빠르게 설치하고 가끔 사용하는 도구에 적합합니다. 계속 사용하는 도구에는 다음 방법을 권장합니다.

### 영구 설치 — 파생 이미지 빌드

컨테이너가 시작될 때마다 재설치 지연 없이 도구를 즉시 사용할 수 있어야 한다면 `nousresearch/hermes-agent`를 상속하고 도구를 레이어에 설치하는 새 이미지를 빌드하세요.

```dockerfile
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends <your-package> \
    && rm -rf /var/lib/apt/lists/*
USER hermes
```

빌드한 뒤 공식 이미지 대신 사용하세요.

```sh
docker build -t my-hermes:latest .
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  my-hermes:latest gateway run
```

진입점 스크립트와 `/opt/data` 의미는 변경 없이 상속되므로 이 페이지의 나머지 내용도 그대로 적용됩니다. 최신 업스트림 `nousresearch/hermes-agent`를 가져올 때 이미지를 다시 빌드해야 한다는 점을 기억하세요.

### 복잡한 도구 또는 다중 서비스 스택 — 사이드카 컨테이너 실행

자체 서비스(데이터베이스, 웹 서버, 큐, 헤드리스 브라우저 팜)를 포함하거나 Hermes 컨테이너 안에 넣기에는 너무 무거운 도구는 공유 Docker 네트워크에서 별도 컨테이너로 실행하세요. Hermes는 로컬 추론 서버에 연결하는 것과 같은 방식으로 컨테이너 이름을 사용해 사이드카에 접근합니다([로컬 추론 서버에 연결하기](#connecting-to-local-inference-servers-vllm-ollama-etc) 참조).

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes:/opt/data
    networks:
      - hermes-net

  my-tool:
    image: example/my-tool:latest
    container_name: my-tool
    restart: unless-stopped
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

Hermes 컨테이너 내부에서 사이드카는 `http://my-tool:<port>`(또는 해당 서비스가 제공하는 프로토콜)로 접근할 수 있습니다. 이 패턴은 각 서비스의 수명 주기, 리소스 제한, 업그레이드 주기를 독립적으로 유지하고 하나의 도구에만 필요한 의존성으로 Hermes 이미지를 불필요하게 키우지 않습니다.

### 널리 유용한 도구 — 이슈 또는 풀 리퀘스트 열기

대부분의 Hermes Agent 사용자에게 유용할 도구라면 비공개 파생 이미지로 유지하기보다 업스트림에 기여하는 것을 고려하세요. 도구와 사용 사례를 설명하는 이슈 또는 풀 리퀘스트를 [hermes-agent 저장소](https://github.com/NousResearch/hermes-agent)에 열어 주세요. 공식 이미지에 번들된 도구는 모든 사용자에게 혜택을 주고 다운스트림 포크의 유지 관리 부담을 피할 수 있습니다.

## 로컬 추론 서버에 연결하기(vLLM, Ollama 등)

Hermes를 Docker에서 실행하면서 추론 서버(vLLM, Ollama, text-generation-inference 등)를 호스트나 다른 컨테이너에서도 실행한다면 네트워킹에 특별한 주의가 필요합니다.

### Docker Compose(권장)

두 서비스를 같은 Docker 네트워크에 배치하세요. 이것이 가장 안정적인 방법입니다.

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    container_name: vllm
    command: >
      --model Qwen/Qwen2.5-7B-Instruct
      --served-model-name my-model
      --host 0.0.0.0
      --port 8000
    ports:
      - "8000:8000"
    networks:
      - hermes-net
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"
    volumes:
      - ~/.hermes:/opt/data
    networks:
      - hermes-net

networks:
  hermes-net:
    driver: bridge
```

그런 다음 `~/.hermes/config.yaml`에서 **컨테이너 이름**을 호스트 이름으로 사용하세요.

```yaml
model:
  provider: custom
  model: my-model
  base_url: http://vllm:8000/v1
  api_key: "none"
```

:::tip 핵심 사항
- 호스트 이름으로 **컨테이너 이름**(`vllm`)을 사용하세요. Hermes 컨테이너 자체를 가리키는 `localhost`나 `127.0.0.1`은 사용하지 마세요.
- `model` 값은 vLLM에 전달한 `--served-model-name`과 일치해야 합니다.
- `api_key`에는 비어 있지 않은 문자열을 지정하세요(vLLM은 헤더를 요구하지만 기본적으로 검증하지 않습니다).
- `base_url` 끝에 슬래시를 넣지 마세요.
:::

### 독립 Docker run(Compose 없음)

추론 서버가 호스트에서 직접 실행되고 Docker 안에서 실행되지 않는다면 macOS/Windows에서는 `host.docker.internal`을 사용하고, Linux에서는 `--network host`를 사용하세요.

**macOS / Windows:**

```sh
docker run -d \
  --name hermes \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://host.docker.internal:8000/v1
  api_key: "none"
```

**Linux(호스트 네트워킹):**

```sh
docker run -d \
  --name hermes \
  --network host \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

```yaml
# config.yaml
model:
  provider: custom
  model: my-model
  base_url: http://127.0.0.1:8000/v1
  api_key: "none"
```

:::warning `--network host`를 사용하면 `-p` 플래그가 무시되며 모든 컨테이너 포트가 호스트에 직접 공개됩니다.
:::

### 연결 확인

Hermes 컨테이너 내부에서 추론 서버에 연결할 수 있는지 확인하세요.

```sh
docker exec hermes curl -s http://vllm:8000/v1/models
```

제공된 모델이 나열된 JSON 응답이 표시되어야 합니다. 실패한다면 다음을 확인하세요.

1. 두 컨테이너가 같은 Docker 네트워크에 있는지(`docker network inspect hermes-net`)
2. 추론 서버가 `127.0.0.1`이 아니라 `0.0.0.0`에서 수신 대기 중인지
3. 포트 번호가 일치하는지

### Ollama

Ollama도 같은 방식으로 작동합니다. Ollama가 호스트에서 실행 중이면 macOS/Windows에서는 `host.docker.internal:11434`를 사용하고, Linux에서 `--network host`를 사용할 때는 `127.0.0.1:11434`를 사용하세요. Ollama가 같은 Docker 네트워크의 자체 컨테이너에서 실행 중이면 다음과 같이 설정합니다.

```yaml
model:
  provider: custom
  model: llama3
  base_url: http://ollama:11434/v1
  api_key: "none"
```

## 문제 해결

### 컨테이너가 즉시 종료됨

로그를 확인하세요: `docker logs hermes`. 일반적인 원인은 다음과 같습니다.
- `.env` 파일이 없거나 유효하지 않음 — 먼저 대화형으로 실행하여 설정을 완료하세요.
- 공개된 포트를 사용하는 경우 포트 충돌.

### "Permission denied" 오류

컨테이너의 stage2 훅은 각 감독 서비스 내부에서 `s6-setuidgid`를 통해 root가 아닌 `hermes` 사용자(UID 10000)로 권한을 낮춥니다. 호스트의 `~/.hermes/`가 다른 UID의 소유라면 `HERMES_UID`/`HERMES_GID` — 또는 LinuxServer.io와 NAS 이미지 호환을 위한 `PUID`/`PGID` 별칭 — 를 호스트 사용자와 일치하도록 설정하거나 데이터 디렉터리를 쓰기 가능하게 만드세요.

```sh
chmod -R 755 ~/.hermes
```

NAS(UGOS, Synology, unRAID)에서는 일반적으로 데이터 디렉터리가 컨테이너에서 `chown`할 수 없는 호스트 UID가 소유한 **바인드 마운트**입니다. `PUID`/`PGID`(또는 `HERMES_UID`/`HERMES_GID`)를 해당 호스트 사용자로 설정하여 런타임이 UID 10000 대신 마운트 소유자로 실행되게 하세요.

```sh
docker run -d \
  --name hermes \
  -e PUID=1000 -e PGID=10 \
  -v /volume1/docker/hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

`docker exec hermes <cmd>`도 자동으로 UID 10000으로 권한을 낮춥니다. 자세한 내용과 호출별 선택 해제 방법은 [`docker exec`는 자동으로 `hermes` 사용자로 권한을 낮춥니다](#docker-exec-automatically-drops-to-the-hermes-user)를 참조하세요.

### 브라우저 도구가 작동하지 않음

Playwright에는 공유 메모리가 필요합니다. Docker run 명령에 `--shm-size=1g`를 추가하세요.

```sh
docker run -d \
  --name hermes \
  --shm-size=1g \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

### 네트워크 문제 후 게이트웨이가 재연결되지 않음

`--restart unless-stopped` 플래그가 대부분의 일시적인 장애를 처리합니다. 게이트웨이가 멈춘 상태라면 컨테이너를 재시작하세요.

```sh
docker restart hermes
```

### 컨테이너 상태 확인

```sh
docker logs --tail 50 hermes          # Recent logs
docker run -it --rm nousresearch/hermes-agent:latest version     # Verify version
docker stats hermes                    # Resource usage
```

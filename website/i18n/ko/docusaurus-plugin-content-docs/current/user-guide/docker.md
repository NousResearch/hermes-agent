---
sidebar_position: 7
title: "Docker"
description: "Hermes Agent를 Docker에서 실행하고 Docker를 터미널 백엔드로 사용하기"
---

# Hermes Agent — Docker

Docker가 Hermes Agent와 상호 작용하는 방식에는 두 가지 명확한 방법이 있습니다:

1. **Docker 내에서 Hermes 실행** — 에이전트 자체가 컨테이너 내부에서 실행됩니다 (이 페이지의 주요 내용)
2. **Docker를 터미널 백엔드로 사용** — 에이전트는 호스트에서 실행되지만 모든 명령어를 툴 호출, `/new`, 하위 에이전트 전반에 걸쳐 지속되는 단일 샌드박스 컨테이너 내에서 실행합니다 (자세한 내용은 [구성 → Docker 백엔드](./configuration.md#docker-backend) 참조)

이 페이지에서는 첫 번째 옵션에 대해 다룹니다. 컨테이너는 모든 사용자 데이터(설정, API 키, 세션, 스킬, 메모리)를 호스트에서 마운트된 `/opt/data` 디렉토리에 저장합니다. 이미지 자체는 상태를 저장하지 않으므로, 설정을 잃지 않고 새 버전을 풀(pull)하여 업그레이드할 수 있습니다.

## 빠른 시작

Hermes Agent를 처음 실행하는 경우, 호스트에 데이터 디렉토리를 만들고 대화형으로 컨테이너를 시작하여 설정 마법사를 실행하세요:

:::caution 웹 브라우저 기반 VPS 콘솔 사용 시 주의사항
일부 VPS 제공업체(Hetzner Cloud 등)는 호스트 관리를 위한 웹 브라우저 기반 콘솔을 제공합니다. 이러한 콘솔은 특수 문자를 잘못 전송할 수 있습니다. 예를 들어 `:`가 `;`로 전달되거나 `@`가 잘못 렌더링될 수 있으며, 영어가 아닌 키보드 레이아웃의 경우 더 심각할 수 있습니다. 이는 `-v ~/.hermes:/opt/data`, `-e KEY=value` 및 붙여넣은 API 키/토큰과 같은 `docker run` 인수를 조용히 손상시킵니다.

복사-붙여넣기를 안전하게 하려면 **SSH를 통해 접속**(`ssh root@<host>`)하세요. 브라우저 콘솔을 사용해야 한다면 붙여넣기 대신 명령어를 수동으로 입력하고 Enter 키를 누르기 전에 모든 `:`, `@`, `=`, `/`를 다시 확인하세요.
:::

```sh
mkdir -p ~/.hermes
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent setup
```

그러면 설정 마법사가 시작되어 API 키를 묻고 `~/.hermes/.env`에 기록합니다. 이 작업은 한 번만 수행하면 됩니다. 이 시점에서 게이트웨이가 작동할 채팅 시스템을 설정하는 것이 좋습니다.

:::tip
컨테이너 내부에서 한 번 `hermes setup --portal`을 실행하세요. 새로 고침 토큰은 마운트된 `~/.hermes` 볼륨에 유지됩니다. [Nous Portal](/integrations/nous-portal)을 참조하세요.
:::

## 게이트웨이 모드로 실행

구성이 완료되면 백그라운드에서 컨테이너를 영구적인 게이트웨이(Telegram, Discord, Slack, WhatsApp 등)로 실행합니다:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  nousresearch/hermes-agent gateway run
```

포트 8642는 게이트웨이의 [OpenAI 호환 API 서버](./features/api-server.md) 및 상태(health) 엔드포인트를 노출합니다. 채팅 플랫폼(Telegram, Discord 등)만 사용하는 경우 선택 사항이지만, 대시보드나 외부 도구가 게이트웨이에 접근하려면 필수입니다.

:::tip 게이트웨이는 감독하에 실행됩니다
공식 Docker 이미지 내에서 `gateway run`은 **자동으로 s6-overlay에 의해 감독**됩니다. 게이트웨이 프로세스가 충돌하면 컨테이너를 잃지 않고 몇 초 내에 재시작되며, 대시보드(`HERMES_DASHBOARD=1` 설정 시)도 함께 감독됩니다. `gateway run` CMD 프로세스 자체는 컨테이너를 유지하는 `sleep infinity` 하트비트이며, 실제 게이트웨이 프로세스는 s6가 관리합니다. 따라서 `docker stop`은 여전히 모든 것을 깔끔하게 종료하지만 `docker logs`에는 감독된 게이트웨이의 출력이 표시됩니다.

`docker logs`에서 업그레이드를 확인하는 한 줄의 로그를 볼 수 있습니다. 이전처럼 "게이트웨이가 컨테이너의 메인 프로세스이며, 컨테이너 종료 = 게이트웨이 종료"라는 동작 방식을 선택하려면 `--no-supervise`를 전달하거나 `HERMES_GATEWAY_NO_SUPERVISE=1`을 설정하세요. 이 설정은 컨테이너가 게이트웨이의 상태 코드로 종료되기를 원하는 CI 연기 테스트(smoke test)에 유용합니다. 프로덕션 배포의 경우에는 기본 제공되는 감독 기능이 훨씬 낫습니다.

이 동작은 s6 기반 이미지에만 적용됩니다. 이전(tini 기반) 이미지는 여전히 `gateway run`을 포그라운드 메인 프로세스로 실행합니다.
:::

:::note 게이트웨이 로그 위치
프로필별 게이트웨이, 대시보드, 부팅 조정기, 컨테이너 전체 `docker logs` 등 전체 라우팅 맵은 아래의 [로그 위치](#where-the-logs-go) 섹션을 참조하세요.
:::

참고: API 서버는 `API_SERVER_ENABLED=true`에 의해 제어됩니다. 컨테이너 내부의 `127.0.0.1`을 넘어 노출하려면 `API_SERVER_HOST=0.0.0.0`과 `API_SERVER_KEY`(최소 8자 이상, `openssl rand -hex 32`로 생성 가능)도 설정해야 합니다. 예시:

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

인터넷에 직접 연결된 머신에서 포트를 여는 것은 보안 위험이 따릅니다. 위험을 이해하지 않는 한 포트를 열지 마세요.

## 대시보드 실행

내장된 웹 대시보드는 게이트웨이와 동일한 컨테이너에서 감독되는 s6-rc 서비스로 실행됩니다. `HERMES_DASHBOARD=1`을 설정하여 대시보드를 켭니다:

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

대시보드는 s6에 의해 감독됩니다. 충돌이 발생하면 `s6-supervise`가 짧은 백오프 후에 자동으로 다시 시작합니다. 대시보드의 stdout/stderr은 접두사 없이 `docker logs <container>`로 전달됩니다 (게이트웨이 자체의 출력은 이제 프로필별 s6-log 파일에 있으므로, 이 두 스트림은 서로 충돌하지 않습니다. 자세한 내용은 아래의 [로그 위치](#where-the-logs-go) 참조).

| 환경 변수 | 설명 | 기본값 |
|---------------------|-------------|---------|
| `HERMES_DASHBOARD` | 감독되는 대시보드 서비스를 활성화하려면 `1` (또는 `true` / `yes`)로 설정 | *(설정되지 않음 — 서비스는 등록되지만 다운 상태로 유지됨)* |
| `HERMES_DASHBOARD_HOST` | 대시보드 HTTP 서버의 바인드 주소 | `0.0.0.0` |
| `HERMES_DASHBOARD_PORT` | 대시보드 HTTP 서버의 포트 | `9119` |
| `HERMES_DASHBOARD_INSECURE` | OAuth 인증 게이트 없이 바인드하려면 `1` (또는 `true` / `yes`)로 설정. 대시보드가 API 키와 세션 데이터를 노출하므로, OAuth 계약이 없는 리버스 프록시 뒤의 신뢰할 수 있는 네트워크에서만 사용하세요. | *(설정되지 않음 — `DashboardAuthProvider`가 등록된 경우 게이트 적용됨)* |

컨테이너 내부의 대시보드는 기본적으로 `0.0.0.0`에 바인딩됩니다. 그렇지 않으면 호스트에서 게시된 `-p 9119:9119` 포트에 연결할 수 없습니다. 바인딩을 컨테이너 루프백으로 제한하려면(사이드카/리버스 프록시 설정의 경우) `HERMES_DASHBOARD_HOST=127.0.0.1`을 설정하세요.

대시보드의 인증 게이트는 다음 두 가지 조건이 모두 참일 때 자동으로 활성화됩니다:

1. 바인드 호스트가 루프백이 아님 (예: 컨테이너 내부의 기본값 `0.0.0.0`), **그리고**
2. `DashboardAuthProvider` 플러그인이 등록되어 있음.

두 번째 조건을 충족하는 세 가지 번들 방법이 있습니다:

- **사용자 이름/비밀번호** — 신뢰할 수 있는 네트워크나 VPN 뒤에 있는 자체 호스팅 / 온프레미스 / 홈랩 컨테이너에 가장 적합: `HERMES_DASHBOARD_BASIC_AUTH_USERNAME` + `HERMES_DASHBOARD_BASIC_AUTH_PASSWORD` (및 재시작에도 안정적인 세션을 위해 `HERMES_DASHBOARD_BASIC_AUTH_SECRET`)를 설정합니다. 공용 인터넷에 직접 노출하기에는 적합하지 않습니다.
- **OAuth (Nous Portal)** — 호스팅/퍼블릭 배포용: `HERMES_DASHBOARD_OAUTH_CLIENT_ID`가 설정될 때마다 `dashboard_auth/nous` 제공자가 활성화됩니다.
- **자체 호스팅 OIDC** — 표준 OpenID Connect를 통해 자체 ID 제공자에 인증: `HERMES_DASHBOARD_OIDC_ISSUER` + `HERMES_DASHBOARD_OIDC_CLIENT_ID`가 설정되면 `dashboard_auth/self_hosted` 제공자가 활성화됩니다.

어떤 것을 선택하든, 게이트는 호출자가 보호된 라우트에 도달하기 전에 로그인 페이지로 리디렉션합니다. 세 제공자 모두에 대해서는 [웹 대시보드 → 인증](features/web-dashboard.md#authentication-gated-mode)을 참조하세요.

제공자가 등록되지 않았고 바인딩이 루프백이 아닌 경우, 대시보드는 시작 시 누락된 환경 변수를 가리키는 특정 오류와 함께 **작동을 실패(fail closed)**합니다. `HERMES_DASHBOARD_INSECURE=1` 탈출 해치는 게이트를 완전히 비활성화하지만(바인딩 호스트만으로는 절대 `--insecure`를 의미하지 않음), 인증되지 않은 대시보드를 제공하게 됩니다. 앞에 고유한 인증 계층이 없는 한 제공자를 구성하세요.

:::warning `HERMES_DASHBOARD_INSECURE=1`은 API 키를 노출합니다
OAuth 게이트를 비활성화하면 게시된 포트에 접근할 수 있는 모든 사람에게 대시보드의 API 영역(모델 키 및 세션 데이터 포함)이 제공됩니다. 앞에 자체 인증 계층이 있거나 완전히 통제하는 신뢰할 수 있는 LAN 환경인 경우에만 이 기능을 활성화하세요.
:::

대시보드를 별도의 컨테이너로 실행하는 것은 지원되지 않습니다. 게이트웨이 활성 상태 감지를 위해서는 게이트웨이 프로세스와 동일한 PID 네임스페이스를 공유해야 합니다.

## 대화형 실행 (CLI 채팅)

실행 중인 데이터 디렉토리에 대해 대화형 채팅 세션을 열려면:

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent
```

또는 (Docker Desktop 등을 통해) 이미 실행 중인 컨테이너에서 터미널을 연 경우 다음 명령어를 실행하면 됩니다:

```sh
/opt/hermes/.venv/bin/hermes
```

## 영구 볼륨

`/opt/data` 볼륨은 모든 Hermes 상태의 단일 진실 공급원입니다. 이 볼륨은 호스트의 `~/.hermes/` 디렉토리에 매핑되며 다음을 포함합니다:

| 경로 | 내용 |
|------|----------|
| `.env` | API 키와 비밀 정보 |
| `config.yaml` | 모든 Hermes 구성 정보 |
| `SOUL.md` | 에이전트 페르소나/정체성 |
| `sessions/` | 대화 기록 |
| `memories/` | 영구 메모리 저장소 |
| `skills/` | 설치된 스킬들 |
| `home/` | Hermes 툴 하위 프로세스(`git`, `ssh`, `gh`, `npm` 및 스킬 CLI)를 위한 프로필별 HOME |
| `cron/` | 예약된 작업 정의 |
| `hooks/` | 이벤트 훅 |
| `logs/` | 런타임 로그 |
| `skins/` | 사용자 정의 CLI 스킨 |

자격 증명을 `~` 아래에 저장하는 스킬 CLI는 데이터 볼륨 루트만이 아니라 하위 프로세스 HOME에 대해 초기화되어야 합니다. 예를 들어, [xurl 스킬](./skills/bundled/social-media/social-media-xurl.md)은 OAuth 상태를 `~/.xurl`에 저장합니다. 공식 Docker 레이아웃에서 Hermes 도구 호출은 이를 `/opt/data/home/.xurl`로 읽으므로, `HOME=/opt/data/home`으로 수동 xurl 인증을 실행하고 `HOME=/opt/data/home xurl auth status`로 확인하세요.

:::warning
동일한 데이터 디렉토리에 대해 두 개의 Hermes **게이트웨이** 컨테이너를 동시에 실행하지 마세요. 세션 파일과 메모리 저장소는 동시 쓰기 액세스를 위해 설계되지 않았습니다.
:::

## 다중 프로필 지원

Hermes는 단일 설치에서 독립적인 에이전트(서로 다른 SOUL, 스킬, 메모리, 세션, 자격 증명)를 실행할 수 있게 해주는 [다중 프로필](../reference/profile-commands.md) (별도의 `~/.hermes/` 하위 디렉토리)을 지원합니다. **공식 Docker 이미지 내부에서 s6 감독 트리는 각 프로필을 일급 감독 서비스로 취급하므로, 권장되는 배포 방식은 모든 프로필을 호스팅하는 하나의 컨테이너**를 사용하는 것입니다.

`hermes profile create <name>`으로 생성된 각 프로필에는 다음이 제공됩니다:

- 런타임에 동적으로 등록되는 `/run/service/gateway-<name>/`에 전용 s6 서비스 슬롯 할당 (컨테이너 재빌드 불필요)
- `s6-supervise`에서 관리하는 충돌 시 자동 재시작 및 백오프 메커니즘
- `${HERMES_HOME}/logs/gateways/<name>/current` 에 저장되는 프로필별 로테이션 로그 (각 1MB 크기의 10개 아카이브)
- 컨테이너 재시작 시 상태 유지: 부팅 시 조정기(reconciler)가 각 프로필 디렉토리에서 `gateway_state.json`을 읽어와서 마지막으로 기록된 상태가 `running`인 프로필에 대해서만 슬롯을 다시 시작시킵니다. 중지된 프로필은 중지된 상태로 유지됩니다.

호스트에서 실행하던 수명 주기 명령어가 컨테이너 내부에서도 동일하게 작동합니다:

```sh
# 프로필 생성 — gateway-<name> s6 슬롯을 등록합니다.
docker exec hermes hermes profile create coder

# 시작 / 중지 / 재시작 — s6-svc를 디스패치합니다. 게이트웨이 수명 주기는 docker 재시작 후에도 유지됩니다.
docker exec hermes hermes -p coder gateway start
docker exec hermes hermes -p coder gateway stop
docker exec hermes hermes -p coder gateway restart

# 상태 — 컨테이너 내부에서 `Manager: s6 (container supervisor)`를 보고합니다.
docker exec hermes hermes -p coder gateway status

# 프로필 제거 — s6 슬롯도 함께 해제합니다.
docker exec hermes hermes profile delete coder
```

내부적으로, 컨테이너 내의 `hermes gateway start/stop/restart`는 인터셉트되어 올바른 서비스 디렉토리로의 `s6-svc`로 라우팅됩니다. s6 명령어를 직접 배울 필요는 없습니다. 원시 관리자 상태의 경우 `/command/s6-svstat /run/service/gateway-<name>`을 사용하세요 (참고로 `/command/`는 감독 트리에 의해 생성된 프로세스의 PATH에만 존재합니다. `docker exec`에서 호출할 때는 절대 경로를 전달하세요).

### 여러 컨테이너 대신 단일 컨테이너에서 여러 프로필을 사용하는 이유

s6 마이그레이션 이전에는 "프로필당 하나의 컨테이너"가 권장 패턴이었습니다. 여러 게이트웨이를 관리할 컨테이너 내 감독자가 없었기 때문입니다. PID 1로 동작하는 s6를 통해 더 이상 그럴 필요가 없어졌으며, 단일 컨테이너 레이아웃이 거의 모든 측면에서 더 간단합니다:

| | 단일 컨테이너, 다중 프로필 | 프로필당 1 컨테이너 |
|---|---|---|
| 디스크 오버헤드 | 1 이미지, 1 번들 venv, 1 Playwright 캐시 | N 이미지 / N 캐시 |
| 메모리 오버헤드 | 공유 Python 인터프리터 캐시, 공유 node_modules | 컨테이너별로 중복됨 |
| 프로필 생성 | `docker exec ... hermes profile create <name>` (수초 소요) | 새로운 `docker run` 실행 + 포트 할당 + 바인드 마운트 설정 |
| 프로필별 충돌 복구 | `s6-supervise` 자동 재시작 | Docker의 `--restart unless-stopped` (느리고, 형제 작업을 종료시킴) |
| 로그 | `s6-log`를 통한 프로필별 순환 파일 및 컨테이너 부팅 감사 로그 | 컨테이너별 `docker logs <name>` — 내장된 순환 없음 |
| 백업 | 1개의 `~/.hermes` 디렉토리 | 조정이 필요한 N개의 디렉토리 |

기본 프로필(`default`)은 첫 부팅 시 항상 등록되므로, 새 컨테이너는 기본적으로 하나의 감독 게이트웨이와 함께 제공됩니다. 추가 프로필은 런타임에 추가됩니다.

### 별도의 컨테이너가 필요한 경우

단일 컨테이너 다중 프로필이 기본값입니다. 다음과 같은 특별한 이유가 있을 때만 프로필당 별도의 컨테이너를 실행하세요:

- **워크로드별 리소스 격리** — 예: 프로필 A의 통제 불능 브라우저 툴 세션이 프로필 B에 OOM(메모리 부족)을 유발해서는 안 되는 경우. 컨테이너를 사용하면 프로필별로 `--memory` / `--cpus` 제한을 둘 수 있습니다.
- **독립적인 이미지 고정(pinning)** — 워크로드별로 다른 업스트림 이미지 태그를 사용하는 경우.
- **네트워크 분할** — 프로필별로 별개의 Docker 네트워크를 구성 (예: 하나는 고객 대면, 하나는 내부용).
- **컴플라이언스 / 장애 범위 제한** — 서로 다른 자격 증명이 OS 수준 프로세스 트리를 절대 공유하지 않도록 하는 경우.

이런 경우에는 서로 다른 `container_name`, `volumes`, `ports`를 사용하여 프로필당 하나의 서비스를 선언하세요:

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

[영구 볼륨](#persistent-volumes)의 경고는 여전히 적용됩니다. 두 컨테이너가 동일한 `~/.hermes` 디렉토리를 동시에 가리키지 않도록 하세요. 각 컨테이너 내부의 s6 감독자는 자신의 프로필 집합을 관리하며, 컨테이너 간에 데이터 볼륨을 공유하면 세션 파일과 메모리 저장소가 손상됩니다.

## 로그 위치

s6 컨테이너에는 네 가지 별개의 로그 영역이 있으며, `docker logs`에 게이트웨이 내용이 표시되지 않는 것은 흔한 혼란의 원인입니다. 치트시트:

| 출처 | 저장 위치 | 읽는 방법 |
|---|---|---|
| **프로필별 게이트웨이** (`hermes gateway run` 및 s6 하의 프로필별 게이트웨이) | 두 곳에 분할(Tee): `docker logs <container>` (실시간, 추가 접두사 없음) **및** `${HERMES_HOME}/logs/gateways/<profile>/current` (순환, ISO-8601 타임스탬프, 1MB 아카이브 10개) | 호스트에서 `docker logs -f hermes` 또는 `tail -F ~/.hermes/logs/gateways/default/current` |
| **대시보드** (`HERMES_DASHBOARD=1`일 때) | `docker logs <container>` (접두사 없음) | `docker logs -f hermes` — 게이트웨이 라인과 교차되어 나타남 |
| **부팅 조정기** (각 컨테이너 시작 시 복원된 프로필 게이트웨이 기록) | `${HERMES_HOME}/logs/container-boot.log` (추가 전용 감사 로그) | `tail -F ~/.hermes/logs/container-boot.log` |
| **일반 Hermes 로그** (`agent.log`, `errors.log`) | `${HERMES_HOME}/logs/` (프로필 인식) | `docker exec hermes hermes logs --follow [--level WARNING] [--session <id>]` |

알아두면 좋은 두 가지 실제 결과:

- 컨테이너 재시작 후에도 유지되는 것은 `logs/gateways/<profile>/current` 에 있는 파일 복사본입니다. `docker logs`는 현재 컨테이너의 수명 동안의 출력만 보존하며 (`docker rm` 시 삭제됨), 순환 파일들은 바인드 마운트된 볼륨에 지속됩니다.
- 부팅 조정기의 감사 라인 형태는 `<iso-timestamp> profile=<name> prior_state=<state> action=<registered|started>` 이므로, `grep profile=coder ~/.hermes/logs/container-boot.log` 를 통해 특정 프로필이 마지막으로 언제 복원되었고 s6가 그것을 자동으로 시작했는지 확인할 수 있습니다.

## 환경 변수 전달

API 키는 컨테이너 내부의 `/opt/data/.env`에서 읽어옵니다. 환경 변수를 직접 전달할 수도 있습니다:

```sh
docker run -it --rm \
  -v ~/.hermes:/opt/data \
  -e ANTHROPIC_API_KEY="sk-ant-..." \
  -e OPENAI_API_KEY="sk-..." \
  nousresearch/hermes-agent
```

직접 지정한 `-e` 플래그는 `.env` 파일의 값을 재정의합니다. 이는 CI/CD 또는 디스크에 키를 저장하고 싶지 않은 비밀 관리자 통합에 유용합니다.

:::note Docker를 **터미널 백엔드**로 찾고 계신가요?
이 페이지는 Hermes 자체를 Docker 내부에서 실행하는 방법을 다룹니다. Hermes가 에이전트의 `terminal` / `execute_code` 호출을 Docker 샌드박스 컨테이너 내부에서 실행하도록 하려면 (Hermes 프로세스 전체에서 하나의 오래 지속되는 컨테이너 공유 — 이슈 #20561 참조), 별도의 설정 블록이 필요합니다. `terminal.backend: docker`와 `terminal.docker_image`, `terminal.docker_volumes`, `terminal.docker_forward_env`, `terminal.docker_env`, `terminal.docker_run_as_host_user`, `terminal.docker_extra_args`, `terminal.docker_persist_across_processes`, `terminal.docker_orphan_reaper` 등이 있습니다. 컨테이너 수명 주기 규칙을 포함한 전체 설정은 [구성 → Docker 백엔드](configuration.md#docker-backend)를 참조하세요.
:::

## Docker Compose 예시

게이트웨이와 대시보드를 모두 포함하는 영구 배포의 경우 `docker-compose.yaml`을 사용하는 것이 편리합니다:

```yaml
services:
  hermes:
    image: nousresearch/hermes-agent:latest
    container_name: hermes
    restart: unless-stopped
    command: gateway run
    ports:
      - "8642:8642"   # 게이트웨이 API
      - "9119:9119"   # 대시보드 (HERMES_DASHBOARD=1일 때만 접근 가능)
    volumes:
      - ~/.hermes:/opt/data
    environment:
      - HERMES_DASHBOARD=1
      # .env 파일을 사용하는 대신 특정 환경 변수를 전달하려면 주석을 해제하세요:
      # - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      # - OPENAI_API_KEY=${OPENAI_API_KEY}
      # - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

`docker compose up -d`로 시작하고 `docker compose logs -f`로 로그를 봅니다. 감독되는 게이트웨이의 stdout도 볼륨의 `${HERMES_HOME}/logs/gateways/<profile>/current`에 분할(tee) 저장됩니다. 전체 라우팅 맵은 [로그 위치](#where-the-logs-go)를 참조하세요.

## 선택 사항: 리눅스 데스크톱 오디오 브릿지

Docker 내 음성 모드가 작동하려면 두 가지 조건이 충족되어야 합니다. Hermes가 컨테이너 내부의 오디오 장치를 탐색하도록 허용되어야 하고, 컨테이너가 호스트 오디오 서버에 연결할 수 있어야 합니다. 아래 설정은 호스트 오디오 설정을 다루며, PulseAudio 호환 소켓을 노출하는 수많은 PipeWire 환경 등 리눅스 데스크톱 환경을 지원합니다.

:::caution
이것은 일반적인 Docker Desktop 기능이 아닌 리눅스 데스크톱을 위한 해결 방법(workaround)입니다. 호스트 오디오가 이미 작동하고 Hermes 컨테이너 내에서 CLI 음성 모드를 사용하고자 할 때 유용합니다. 만약 Hermes가 계속해서 `Running inside Docker container -- no audio devices`라는 오류를 보고한다면, `PULSE_SERVER` / `PIPEWIRE_REMOTE`에 대한 Docker 오디오 검색 지원이 포함된 빌드를 사용하세요.
:::

먼저, Compose 파일 옆에 ALSA 설정 파일을 생성합니다:

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

그런 다음 ALSA PulseAudio 플러그인이 설치된 파생 이미지를 빌드합니다:

```dockerfile title="Dockerfile.audio"
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libasound2-plugins \
    && rm -rf /var/lib/apt/lists/*
```

Compose에서 해당 이미지를 사용하고 호스트 사용자의 PulseAudio 소켓과 쿠키를 통과시킵니다:

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

컨테이너 프로세스가 사용자별 오디오 소켓에 접근할 수 있도록 호스트 UID/GID로 시작합니다:

```sh
export HERMES_UID="$(id -u)"
export HERMES_GID="$(id -g)"
docker compose up -d --build
```

컨테이너 내부에서 PortAudio가 인식하는 내용을 확인하려면 다음을 실행합니다:

```sh
docker exec hermes /opt/hermes/.venv/bin/python -c "import sounddevice as sd; print(sd.query_devices())"
```

## 리소스 한도

Hermes 컨테이너는 적당한 리소스가 필요합니다. 권장되는 최소 사양:

| 리소스 | 최소 | 권장 |
|----------|---------|-------------|
| 메모리 | 1 GB | 2–4 GB |
| CPU | 1 코어 | 2 코어 |
| 디스크 (데이터 볼륨) | 500 MB | 2+ GB (세션/스킬에 따라 증가) |

브라우저 자동화(Playwright/Chromium)는 가장 메모리를 많이 소모하는 기능입니다. 브라우저 도구가 필요하지 않은 경우 1GB로 충분합니다. 브라우저 도구를 활성화하는 경우에는 최소 2GB를 할당하세요.

Docker에서 한도를 설정합니다:

```sh
docker run -d \
  --name hermes \
  --restart unless-stopped \
  --memory=4g --cpus=2 \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

## Dockerfile의 역할

공식 이미지는 `debian:13.4`를 기반으로 하며 다음을 포함합니다:

- 모든 Hermes 종속성을 갖춘 Python 3 (`uv pip install -e ".[all]"`)
- Node.js + npm (브라우저 자동화 및 WhatsApp 브릿지용)
- Chromium을 사용하는 Playwright (`npx playwright install --with-deps chromium --only-shell`)
- 시스템 유틸리티로서의 ripgrep, ffmpeg, git, `xz-utils`
- **`docker-cli`** — 이를 통해 컨테이너 내부에서 실행되는 에이전트가 호스트의 Docker 데몬을 구동하여 `docker build`, `docker run`, 컨테이너 검사 등을 수행할 수 있습니다 (`/var/run/docker.sock` 바인드 마운트 필요).
- **`openssh-client`** — 컨테이너 내부에서 [SSH 터미널 백엔드](/user-guide/configuration#ssh-backend)를 사용할 수 있게 해줍니다. SSH 백엔드는 시스템의 `ssh` 바이너리를 호출하므로, 이게 없으면 컨테이너 환경에서 조용히 실패하게 됩니다.
- WhatsApp 브릿지 (`scripts/whatsapp-bridge/`)
- PID 1로서의 **[`s6-overlay`](https://github.com/just-containers/s6-overlay) v3** (이전의 `tini` 대체) — 대시보드 및 프로필별 게이트웨이를 감독하고, 충돌 시 자동 재시작하며, 좀비 하위 프로세스를 정리하고 신호를 전달합니다.

컨테이너의 `ENTRYPOINT`는 s6-overlay의 `/init`입니다. 부팅 시 다음을 수행합니다:
1. 루트 권한으로 `/etc/cont-init.d/01-hermes-setup` ( = `docker/stage2-hook.sh`) 실행: 선택적 UID/GID 재매핑, 볼륨 소유권 수정, 첫 부팅 시 `.env` / `config.yaml` / `SOUL.md` 시딩, `HERMES_SKIP_CONFIG_MIGRATION=1`이 아닌 이상 비대화형 설정 스키마 마이그레이션 실행, 번들 스킬 동기화.
2. `/etc/cont-init.d/02-reconcile-profiles` ( = `hermes_cli.container_boot`) 실행: `$HERMES_HOME/profiles/<name>/` 을 순회하며 `/run/service/gateway-<profile>/` 아래에 프로필별 게이트웨이 s6 서비스 슬롯을 재생성하고, 마지막 기록 상태가 `running`인 것만 자동으로 시작합니다 ([프로필별 게이트웨이 감독](#per-profile-gateway-supervision) 참조).
3. 정적인 `main-hermes`와 `dashboard` s6-rc 서비스를 시작합니다.
4. 사용자가 `docker run`에 전달한 인수를 라우팅하는 컨테이너의 CMD 메인 프로그램(`/opt/hermes/docker/main-wrapper.sh`)을 실행(Exec)합니다:
   - 인수 없음 → `hermes` (기본값)
   - 첫 번째 인수가 PATH에 있는 실행 파일 (예: `sleep`, `bash`) → 직접 실행(exec)
   - 그 외 → `hermes <args>` (서브커맨드 통과)
   이 메인 프로그램이 종료되면 컨테이너도 해당 종료 코드와 함께 종료됩니다.

:::warning s6 이전 이미지와 호환되지 않는 변경 사항
컨테이너의 ENTRYPOINT가 이제 `/usr/bin/tini`가 아니라 `/init` (s6-overlay)입니다. 문서화된 5가지 `docker run` 호출 패턴(인수 없음, `chat -q "…"`, `sleep infinity`, `bash`, `--tui`)은 모두 tini 기반 이미지와 동일하게 동작합니다. 하지만 tini 특유의 시그널 동작에 의존하거나 하드코딩된 `/usr/bin/tini --` 호출을 사용하던 다운스트림 래퍼가 있는 경우 이전 이미지 태그에 고정하세요.
:::

:::warning 권한 모델
명령 체인에 `/init` (또는 과거 `docker/entrypoint.sh` 호환 래퍼)을 유지하지 않는 한 이미지의 진입점(entrypoint)을 재정의하지 마세요. s6-overlay의 `/init`는 첫 부팅 시 볼륨의 권한을 변경(chown)할 수 있도록 루트로 실행된 다음, 메인 프로그램뿐만 아니라 감독되는 모든 서비스에 대해 `s6-setuidgid`를 통해 `hermes` 사용자 계정으로 권한을 낮춥니다. 공식 이미지 내부에서 루트로 `hermes gateway run`을 시작하는 것은 `/opt/data` 내에 루트 소유 파일을 남겨 차후 대시보드나 게이트웨이 시작을 망가뜨릴 수 있으므로 기본적으로 거부됩니다. 그 위험을 의도적으로 수용할 경우에만 `HERMES_ALLOW_ROOT_GATEWAY=1`을 설정하세요.
:::

### `docker exec`는 `hermes` 사용자로 자동 전환됩니다

`docker exec hermes <cmd>`는 기본적으로 컨테이너 내부에서 root로 실행되지만, 이 이미지는 `s6-setuidgid hermes`를 통해 투명하게 재실행(re-execs)하여 루트 호출자를 감지하는 얇은 심(shim)을 `/opt/hermes/bin/hermes`에 (PATH에서 가장 먼저) 제공합니다. 따라서 `docker exec hermes login`, `docker exec hermes profile create …`, `docker exec hermes setup` 등은 모두 추가 `--user` 플래그 없이도 UID 10000(감독되는 게이트웨이가 읽을 수 있음) 소유의 파일을 작성합니다. 비 루트 호출자 (감독되는 프로세스 자체, `docker exec --user hermes`, 컨테이너 내의 칸반 서브에이전트)는 venv 바이너리를 직접 실행하는 숏서킷(short-circuit)에 도달하므로 핫 패스에 오버헤드가 없습니다.

루트 의미(semantics)를 유지하는 `docker exec`가 명시적으로 필요한 경우 (진단 세션, 루트 전용 상태 검사, 루트가 소유한 `/opt/data` 외부 파일) 실행 시 제외 처리할 수 있습니다:

```sh
docker exec -e HERMES_DOCKER_EXEC_AS_ROOT=1 hermes <cmd>
```

이 심(shim)은 대소문자를 구분하지 않고 `1` / `true` / `yes`를 허용합니다. 그 외의 값 — `=0`과 같은 오타 포함 — 은 모두 권한 강등을 거치게 되므로, 사용자 모르게 강등 없이 실행되는 경우는 불가능합니다. 만약 `s6-setuidgid`를 사용할 수 없는 경우(s6-overlay가 제거된 커스텀 빌드 등), 이 심(shim)은 루트 권한으로 실행되는 것을 거부하고 상태 코드 126과 함께 종료됩니다. 이는 `docker exec hermes login`이 `auth.json`을 `root:root`로 작성하여 이후 통신 플랫폼 메시지가 올 때마다 감독되는 게이트웨이의 인증이 망가지는 역사적인 오류를 반복하는 대신, 망가진 권한 모델을 눈에 띄게 드러내게 합니다.

### 프로필별 게이트웨이 감독

`hermes profile create <name>`으로 생성된 각 프로필은 `/run/service/gateway-<name>/`에 등록된 s6 감독형 게이트웨이 서비스를 자동으로 얻게 되며, 컨테이너가 다시 시작되어도 상태가 유지되는 자동 재시작 기능을 지원합니다. 사용자 전용 워크플로우와 수명 주기 명령어는 위의 [다중 프로필 지원](#multi-profile-support) 섹션을 참고하세요.

**s6 이전 이미지 대비 감독 기능의 이점:**

- 게이트웨이 충돌 시 `s6-supervise`가 약 1초의 대기 시간(backoff) 후 자동으로 재시작합니다.
- `HERMES_DASHBOARD=1`로 활성화된 경우, 대시보드 또한 동일한 감독 트리에서 관리되며 동일한 자동 재시작 처리를 받게 됩니다.
- `docker restart`는 실행 중이던 게이트웨이를 유지합니다: 컨테이너 초기화 시 조정기(reconciler)가 `$HERMES_HOME/profiles/<name>/gateway_state.json`을 읽고 마지막 상태가 `running`인 슬롯만 다시 시작합니다. 정지된 게이트웨이는 정지 상태로 유지됩니다.
- 프로필별 게이트웨이 로그는 `$HERMES_HOME/logs/gateways/<profile>/current` 에 유지되며 (`s6-log`에 의해 순환됨), 조정기의 작업 내용 역시 부팅 시마다 `$HERMES_HOME/logs/container-boot.log`에 추가됩니다. 전체 로그 경로는 [로그 위치](#where-the-logs-go)를 참고하세요.

컨테이너 내부에서 `hermes status`를 실행하면 `Manager: s6 (container supervisor)`를 보고합니다. 로우(raw) 레벨의 관리자 상태를 확인하려면 `/command/s6-svstat /run/service/gateway-<name>`을 사용하세요 (`/command/`는 감독 트리 프로세스의 경로에만 포함되어 있습니다; `docker exec`에서 호출할 때는 절대 경로를 지정하세요).

## 업그레이드

최신 이미지를 당겨오고(pull) 컨테이너를 다시 생성합니다. 데이터 디렉토리는 보존되며, 컨테이너는 게이트웨이를 시작하기 전에 마운트된 `$HERMES_HOME/config.yaml`에 대해 비대화형 구성 스키마 마이그레이션을 실행합니다. 마이그레이션이 필요한 경우, Hermes는 `config.yaml`과 `.env` 파일 옆에 타임스탬프가 찍힌 백업본을 먼저 작성합니다.

```sh
docker pull nousresearch/hermes-agent:latest
docker rm -f hermes
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

또는 Docker Compose를 사용하는 경우:

```sh
docker compose pull
docker compose up -d
```

새 이미지가 구성을 다시 쓰게 하기 전에 수동으로 설정값을 검사하거나 마이그레이션해야 할 때만 `HERMES_SKIP_CONFIG_MIGRATION=1`을 설정하세요.

## 스킬과 자격 증명 파일

Docker를 실행 환경으로 사용할 때 (위의 방법들이 아닌 에이전트가 Docker 샌드박스 내부에서 명령을 실행할 때 — [구성 → Docker 백엔드](./configuration.md#docker-backend) 참고), Hermes는 모든 툴 호출에 대해 단일 장기 유지 컨테이너를 재사용하며, 읽기 전용 볼륨으로 스킬 디렉토리(`~/.hermes/skills/`) 및 스킬이 선언한 자격 증명 파일을 컨테이너에 자동 마운트합니다. 수동 설정 없이 샌드박스 내부에서 스킬 스크립트, 템플릿, 참고자료를 이용할 수 있으며, 이 컨테이너는 Hermes 프로세스가 살아있는 동안 지속되므로 한 번 설치한 의존성이나 작성한 파일이 다음 툴 호출 때까지 남아있습니다.

SSH 및 Modal 백엔드 환경에서도 같은 방식이 적용됩니다 — 각 명령 전에 rsync 또는 Modal 마운트 API를 통해 스킬과 자격 증명 파일이 업로드됩니다.

## 컨테이너에 도구 추가 설치하기

공식 이미지는 선별된 유틸리티 세트를 포함하고 있지만 ([Dockerfile의 역할](#what-the-dockerfile-does) 참조), 에이전트가 필요로 할 수 있는 모든 도구가 미리 설치되어 있지는 않습니다. 권장되는 5가지 접근 방식이 있으며, 지속성과 필요한 작업량 순으로 정리되어 있습니다.

### npm 또는 Python 툴 — `npx` 또는 `uvx` 사용

npm이나 PyPI에 등록된 도구의 경우, Hermes에게 `npx` (npm) 또는 `uvx` (Python)를 통해 실행하도록 지시하고, 이 명령을 영구 메모리에 기억하게 하세요. 도구에 구성 파일이나 자격 증명이 필요한 경우 `/opt/data` 아래 (예: `/opt/data/<tool>/config.yaml`)에 파일을 저장하도록 지시하세요.

의존성은 필요할 때 즉시 받아오며, 컨테이너가 유지되는 동안 캐싱됩니다. `/opt/data` 아래 쓰인 설정값들은 바인드 마운트된 호스트 디렉토리에 존재하기 때문에 컨테이너 재시작 후에도 살아남습니다. `docker rm` 후에는 패키지 캐시가 지워지지만 도구가 다음 번에 실행될 때 `npx` 및 `uvx`가 백그라운드에서 투명하게 다시 다운로드합니다.

### 기타 툴 (apt 패키지, 바이너리) — 설치 후 기억하기

npm이나 PyPI에 없는 도구의 경우(예: `apt` 패키지, 사전 빌드된 바이너리, 이미지에 포함되지 않은 런타임 언어 등) Hermes에게 설치 방법을 지시하고 (예: `apt-get update && apt-get install -y <package>`) 해당 설치 명령을 기억하도록 지시하세요. 도구는 남은 컨테이너의 수명 동안 유지되며, 컨테이너가 다시 시작된 후 도구가 다시 필요해지면 Hermes가 이 설치 명령을 다시 실행합니다.

이 방식은 설치가 빠르고 이따금씩 사용되는 도구에 적합합니다. 지속적으로 사용하는 도구라면 다음 방식을 고려하세요.

### 영구적 설치 — 파생 이미지 빌드하기

재설치 지연 없이 컨테이너 시작 즉시 모든 도구를 사용해야 하는 경우 `nousresearch/hermes-agent`에서 파생되어 레이어에 도구를 설치하는 새로운 이미지를 빌드하세요:

```dockerfile
FROM nousresearch/hermes-agent:latest

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends <your-package> \
    && rm -rf /var/lib/apt/lists/*
USER hermes
```

빌드하고 공식 이미지 대신 사용하세요:

```sh
docker build -t my-hermes:latest .
docker run -d \
  --name hermes \
  --restart unless-stopped \
  -v ~/.hermes:/opt/data \
  -p 8642:8642 \
  my-hermes:latest gateway run
```

진입점 스크립트와 `/opt/data` 동작 방식은 그대로 상속되므로 이 페이지의 나머지 부분도 동일하게 적용됩니다. 상위 `nousresearch/hermes-agent` 새 버전을 당겨올 때마다 파생 이미지도 잊지 말고 새로 빌드하세요.

### 복잡한 툴이나 다중 서비스 스택 — 사이드카 컨테이너 실행하기

도구가 자체 서비스를 가져오거나 (데이터베이스, 웹 서버, 큐(queue), 헤드리스 브라우저 팜(farm)), Hermes 컨테이너 내부에 설치하기엔 너무 무거운 도구의 경우 공유 Docker 네트워크 위에서 별도의 컨테이너로 실행하세요. Hermes는 로컬 추론 서버에 연결하는 방식과 동일하게 컨테이너 이름을 통해 사이드카에 접근할 수 있습니다 ([로컬 추론 서버에 연결하기](#connecting-to-local-inference-servers-vllm-ollama-etc) 참조).

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

Hermes 컨테이너 내부에서 사이드카에 접속하려면 `http://my-tool:<port>` (또는 사용되는 프로토콜)로 접근할 수 있습니다. 이 패턴은 각 서비스의 생명 주기, 자원 할당량, 업데이트 주기를 독립적으로 가져갈 수 있게 해주며 Hermes 이미지에 다른 도구들에선 쓰이지 않는 불필요한 의존성을 넣지 않도록 방지해 줍니다.

### 널리 유용한 도구들 — 이슈 및 풀 리퀘스트 열기

해당 도구가 대부분의 Hermes Agent 사용자에게 유용할 것 같다면, 비공개 파생 이미지에만 적용하지 말고 메인 프로젝트에 기여하는 것을 고려해 보세요. 해당 도구와 그 사용 예시를 설명하는 내용으로 [hermes-agent 저장소](https://github.com/NousResearch/hermes-agent)에 이슈를 제기하거나 풀 리퀘스트를 여세요. 공식 이미지에 번들로 포함된 도구는 모든 사용자에게 이점을 제공하고 개별 포크(fork) 프로젝트를 유지보수하는 부담을 줄여 줍니다.

## 로컬 추론 서버(vLLM, Ollama 등)에 연결하기

Hermes를 Docker에서 실행하고 추론 서버(vLLM, Ollama, text-generation-inference 등)가 호스트나 다른 컨테이너에서 실행 중인 경우 네트워크에 세심한 주의를 기울여야 합니다.

### Docker Compose (권장)

두 서비스를 같은 Docker 네트워크상에 배치하는 것이 가장 안정적인 방법입니다:

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

그런 다음 `~/.hermes/config.yaml` 에서 **컨테이너 이름**을 호스트 이름으로 사용합니다:

```yaml
model:
  provider: custom
  model: my-model
  base_url: http://vllm:8000/v1
  api_key: "none"
```

:::tip 주요 사항
- Hermes 컨테이너 자기 자신을 가리키는 `localhost` 또는 `127.0.0.1` 대신 호스트명으로 **컨테이너 이름** (`vllm`)을 사용하세요.
- `model`의 값은 vLLM에 넘겨준 `--served-model-name`과 반드시 일치해야 합니다.
- vLLM은 기본적으로 헤더가 필수지만 유효성을 검증하지는 않으므로, 빈칸이 아닌 어떤 값이든 `api_key`에 설정해 주면 됩니다.
- `base_url`의 마지막에 후행 슬래시를 **포함하지 마세요.**
:::

### 단독 Docker run (Compose 미사용)

추론 서버가 Docker 컨테이너가 아닌 호스트 상에서 직접 돌아가고 있다면, macOS/Windows에서는 `host.docker.internal`을 사용하고, Linux에서는 `--network host`를 사용하세요:

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

**Linux (호스트 네트워킹):**

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

:::warning `--network host`를 사용하면 `-p` 플래그는 무시됩니다 — 모든 컨테이너의 포트가 호스트에 직접 노출됩니다.
:::

### 연결 확인

Hermes 컨테이너 내부에서 추론 서버에 연결할 수 있는지 확인하세요:

```sh
docker exec hermes curl -s http://vllm:8000/v1/models
```

제공 중인 모델 목록이 포함된 JSON 응답을 볼 수 있어야 합니다. 연결되지 않는 경우 다음을 확인하세요:

1. 두 컨테이너가 동일한 Docker 네트워크 안에 있는지 (`docker network inspect hermes-net`)
2. 추론 서버가 `127.0.0.1`이 아닌 `0.0.0.0`에서 수신 대기 중인지
3. 포트 번호가 올바르게 일치하는지

### Ollama

Ollama도 동일한 방식으로 동작합니다. 호스트에서 직접 실행하는 경우, `host.docker.internal:11434` (macOS/Windows) 또는 `127.0.0.1:11434` (Linux `--network host` 포함)를 사용하세요. Ollama가 동일한 Docker 네트워크의 자체 컨테이너에 있는 경우:

```yaml
model:
  provider: custom
  model: llama3
  base_url: http://ollama:11434/v1
  api_key: "none"
```

## 문제 해결

### 컨테이너가 즉시 종료됨

로그를 확인하세요: `docker logs hermes`. 일반적인 원인:
- `.env` 파일이 없거나 잘못됨 — 설정을 완료하려면 먼저 대화형(interactive)으로 실행하세요
- 포트가 노출된 상태로 실행되는 경우의 포트 충돌

### "권한 거부(Permission denied)" 오류

컨테이너의 2단계 훅은 각 감독되는 서비스 내의 `s6-setuidgid`를 통해 루트가 아닌 `hermes` 사용자(UID 10000)로 권한을 떨어뜨립니다. 호스트 `~/.hermes/`의 소유자가 다른 UID인 경우 호스트 사용자와 일치하도록 `HERMES_UID`/`HERMES_GID` — 또는 LinuxServer.io 및 NAS 이미지와의 호환을 위한 `PUID`/`PGID` 별칭 — 를 설정하거나 데이터 디렉토리에 쓰기 권한이 있는지 확인하세요:

```sh
chmod -R 755 ~/.hermes
```

NAS (UGOS, Synology, unRAID)에서 데이터 디렉토리는 대개 호스트 UID가 소유한 **바인드 마운트**이므로 컨테이너에서 권한을 변경(`chown`)할 수 없습니다. 런타임이 UID 10000 대신 마운트 소유자로 실행되게 하려면 호스트 사용자에 맞게 `PUID`/`PGID` (또는 `HERMES_UID`/`HERMES_GID`)를 지정하세요:

```sh
docker run -d \
  --name hermes \
  -e PUID=1000 -e PGID=10 \
  -v /volume1/docker/hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

`docker exec hermes <cmd>` 또한 자동으로 UID 10000으로 떨어집니다. 자세한 내용과 실행 시마다 제외하는 방법은 [`docker exec`는 `hermes` 사용자로 자동 전환됩니다](#docker-exec-automatically-drops-to-the-hermes-user) 섹션을 참고하세요.

### 브라우저 툴 작동 불가

Playwright는 공유 메모리를 필요로 합니다. Docker 실행 시 `--shm-size=1g` 옵션을 덧붙여 주세요:

```sh
docker run -d \
  --name hermes \
  --shm-size=1g \
  -v ~/.hermes:/opt/data \
  nousresearch/hermes-agent gateway run
```

### 네트워크 장애 이후 게이트웨이 재연결 안 됨

`--restart unless-stopped` 플래그는 대부분의 일시적인 실패에 대응할 수 있습니다. 그래도 게이트웨이가 멈춰 있는 경우 컨테이너를 재시작하세요:

```sh
docker restart hermes
```

### 컨테이너 상태 확인

```sh
docker logs --tail 50 hermes          # 최근 로그
docker run -it --rm nousresearch/hermes-agent:latest version     # 버전 확인
docker stats hermes                    # 리소스 사용량
```

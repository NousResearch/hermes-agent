---
title: 브라우저 자동화
description: 여러 제공자, CDP를 통한 로컬 Chromium 계열 브라우저 또는 클라우드 브라우저를 제어하여 웹 상호 작용, 양식 작성, 스크래핑 등을 수행합니다.
sidebar_label: 브라우저
sidebar_position: 5
---

# 브라우저 자동화 (Browser Automation)

Hermes Agent는 여러 백엔드 옵션이 포함된 전체 브라우저 자동화 도구 세트를 제공합니다:

- **Browserbase 클라우드 모드**: 관리형 클라우드 브라우저 및 봇 방지 도구를 제공하는 [Browserbase](https://browserbase.com) 활용
- **Browser Use 클라우드 모드**: 대체 클라우드 브라우저 제공자인 [Browser Use](https://browser-use.com) 활용
- **Firecrawl 클라우드 모드**: 스크래핑 기능이 내장된 클라우드 브라우저를 제공하는 [Firecrawl](https://firecrawl.dev) 활용
- **Camofox 로컬 모드**: 클라우드 종속성 없이 로컬 스푸핑(탐지 방지) 브라우징을 제공하는 [Camofox](https://github.com/jo-inc/camofox-browser) 활용 (Firefox 기반)
- **로컬 Chromium 계열 CDP**: `/browser connect`를 사용하여 브라우저 도구를 자신의 Chrome, Brave, Chromium 또는 Edge 인스턴스에 연결
- **로컬 브라우저 모드**: `agent-browser` CLI 및 로컬 Chromium 설치 활용

모든 모드에서 에이전트는 웹사이트를 탐색하고, 페이지 요소와 상호 작용하며, 양식을 작성하고, 정보를 추출할 수 있습니다.

## 개요

페이지는 **접근성 트리(accessibility trees)** (텍스트 기반 스냅샷)로 표현되므로 LLM 에이전트에 이상적입니다. 대화형 요소에는 에이전트가 클릭 및 타이핑에 사용하는 참조 ID(`@e1`, `@e2` 등)가 부여됩니다.

주요 기능:

- **다중 제공자 클라우드 실행** — Browserbase, Browser Use 또는 Firecrawl — 로컬 브라우저 불필요
- **로컬 Chromium 계열 통합** — CDP를 통해 실행 중인 Chrome, Brave, Chromium 또는 Edge 브라우저에 연결하여 직접 브라우징 수행
- **기본 스텔스 기능** — 무작위 지문(fingerprints), CAPTCHA 해결, 주거용 프록시(Browserbase)
- **세션 격리** — 각 작업은 고유한 브라우저 세션을 갖습니다.
- **자동 정리** — 비활성 세션은 시간 초과 후 닫힙니다.
- **비전 분석** — 시각적 이해를 위한 스크린샷 + AI 분석

## 설정

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있는 경우 별도의 API 키 없이 **[Tool Gateway](tool-gateway.md)**를 통해 브라우저 자동화를 사용할 수 있습니다. 신규 설치자는 `hermes setup --portal`을 실행하여 로그인하고 모든 게이트웨이 도구를 한 번에 켤 수 있으며, 기존 설치자는 `hermes model` 또는 `hermes tools`를 통해 브라우저 제공자로 **Nous Subscription**을 선택할 수 있습니다.
:::

### Browserbase 클라우드 모드

Browserbase 관리형 클라우드 브라우저를 사용하려면 다음을 추가하세요:

```bash
# ~/.hermes/.env 에 추가
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

자격 증명은 [browserbase.com](https://browserbase.com)에서 얻을 수 있습니다.

### Browser Use 클라우드 모드

Browser Use를 클라우드 브라우저 제공자로 사용하려면 다음을 추가하세요:

```bash
# ~/.hermes/.env 에 추가
BROWSER_USE_API_KEY=***
```

API 키는 [browser-use.com](https://browser-use.com)에서 얻을 수 있습니다. Browser Use는 REST API를 통해 클라우드 브라우저를 제공합니다. Browserbase와 Browser Use 자격 증명이 모두 설정된 경우 Browserbase가 우선합니다.

### Firecrawl 클라우드 모드

Firecrawl을 클라우드 브라우저 제공자로 사용하려면 다음을 추가하세요:

```bash
# ~/.hermes/.env 에 추가
FIRECRAWL_API_KEY=fc-***
```

API 키는 [firecrawl.dev](https://firecrawl.dev)에서 얻을 수 있습니다. 그런 다음 Firecrawl을 브라우저 제공자로 선택하세요:

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

선택적 설정:

```bash
# 자체 호스팅 Firecrawl 인스턴스 (기본값: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# 세션 TTL(초 단위) (기본값: 300)
FIRECRAWL_BROWSER_TTL=600
```

### 하이브리드 라우팅: 공개 URL은 클라우드로, LAN/localhost는 로컬로

클라우드 제공자가 구성된 경우, Hermes는 사설/루프백/LAN 주소(`localhost`, `127.0.0.1`, `192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`, `*.local`, `*.lan`, `*.internal`, IPv6 루프백 `::1`, 링크-로컬 `169.254.x.x`)로 확인되는 URL에 대해 **로컬 Chromium 사이드카(sidecar)**를 자동으로 생성합니다. 퍼블릭 URL은 동일한 대화 내에서 계속해서 클라우드 제공자를 사용합니다.

이것은 일반적인 "로컬에서 개발 중이지만 Browserbase를 사용하고 있습니다" 워크플로우를 해결합니다. 에이전트는 제공자를 전환하거나 SSRF 가드를 비활성화하지 않고도 `http://localhost:3000`의 대시보드를 스크린샷으로 찍으면서 동시에 `https://github.com`을 스크랩할 수 있습니다. 클라우드 제공자는 절대 사설 URL을 보지 못합니다.

이 기능은 **기본적으로 켜져 있습니다**. 이를 비활성화하려면 (모든 URL이 이전처럼 구성된 클라우드 제공자로 이동합니다):

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

자동 라우팅이 비활성화된 상태에서 `browser.allow_private_urls: true`를 함께 설정하지 않으면 사설 URL은 `"Blocked: URL targets a private or internal address"`로 거부됩니다 (이 설정을 켜면 클라우드 제공자가 사설 URL 접근을 시도하지만 Browserbase 등은 사용자의 LAN에 도달할 수 없으므로 대개 작동하지 않습니다).

요구 사항: 로컬 사이드카는 순수 로컬 모드와 동일한 `agent-browser` CLI를 사용하므로 해당 항목이 설치되어 있어야 합니다(`hermes setup tools → Browser Automation`을 통해 자동 설치됨). 공개 URL에서 사설 주소로의 탐색 후 리디렉션은 계속 차단됩니다 (공개 경로를 통한 내부 리디렉션 꼼수로 LAN에 도달할 수 없습니다).

### Camofox 로컬 모드

[Camofox](https://github.com/jo-inc/camofox-browser)는 Camoufox (C++ 핑거프린트 스푸핑을 지원하는 Firefox 포크)를 래핑하는 자체 호스팅 Node.js 서버입니다. 클라우드 종속성 없이 로컬 스푸핑(탐지 방지) 브라우징을 제공합니다.

```bash
# 먼저 Camofox 브라우저 서버를 클론합니다.
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# 기본 컨테이너 설정으로 Docker를 사용해 빌드 및 시작합니다.
# (아키텍처 자동 감지: M1/M2는 aarch64, Intel은 x86_64)
make up

# 기본 컨테이너를 중지하고 제거합니다.
make down

# 클린 빌드 강제 실행 (예: VERSION/RELEASE 업그레이드 후)
make reset

# 빌드 없이 바이너리만 다운로드
make fetch

# 아키텍처 또는 버전을 명시적으로 재정의
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up`은 즉시 기본 컨테이너를 시작합니다. 더 큰 Node 힙, VNC 또는 영구 프로필 디렉토리와 같은 사용자 지정 런타임 설정이 필요한 경우, 이미지를 먼저 빌드한 다음 직접 실행하세요:

```bash
# 기본 컨테이너 시작 없이 이미지 빌드
make build

# 영구 저장소, VNC 라이브 보기, 확장된 Node 힙과 함께 시작
mkdir -p ~/.camofox-docker
docker run -d \
  --name camofox-browser \
  --restart unless-stopped \
  -p 9377:9377 \
  -p 6080:6080 \
  -p 5901:5900 \
  -e CAMOFOX_PORT=9377 \
  -e ENABLE_VNC=1 \
  -e VNC_BIND=0.0.0.0 \
  -e VNC_RESOLUTION=1920x1080 \
  -e MAX_OLD_SPACE_SIZE=2048 \
  -v ~/.camofox-docker:/root/.camofox \
  camofox-browser:135.0.1-aarch64
```

VNC가 활성화된 상태에서 브라우저는 창(headed mode)으로 실행되며 브라우저(`http://localhost:6080` (noVNC))에서 실시간으로 볼 수 있습니다. 또한 기본 VNC 클라이언트를 `localhost:5901`에 연결할 수도 있습니다.

이미 `make up`을 실행한 경우 사용자 지정 컨테이너를 시작하기 전에 해당 기본 컨테이너를 중지하고 제거하십시오.

```bash
make down
# 그런 다음 위의 사용자 지정 docker run 명령을 실행합니다.
```

그런 다음 `~/.hermes/.env`에 다음을 설정합니다:

```bash
CAMOFOX_URL=http://localhost:9377
```

Camofox가 Docker에서 실행 중이고 호스트 머신에서 서비스되는 웹 앱을 열고 싶다면, 루프백 재작성을 활성화하세요. `CAMOFOX_URL`은 여전히 호스트의 노출된 제어 API를 가리켜야 하지만, `http://127.0.0.1:3000`과 같은 페이지 URL은 컨테이너 내부에서 `http://host.docker.internal:3000`으로 열려야 합니다.

```yaml
# ~/.hermes/config.yaml
browser:
  camofox:
    rewrite_loopback_urls: true
    loopback_host_alias: host.docker.internal  # 기본값; 필요한 경우 LAN IP를 사용하세요
```

이에 해당하는 환경 변수:

```bash
CAMOFOX_REWRITE_LOOPBACK_URLS=true
CAMOFOX_LOOPBACK_HOST_ALIAS=host.docker.internal
```

재작성은 루프백 호스트(`localhost`, `127.0.0.1`, `::1`)가 있는 페이지 탐색 URL에만 적용됩니다. 이는 `CAMOFOX_URL`을 변경하지 않습니다. 브라우저가 이미 호스트에서 실행되고 루프백 URL이 올바른 비-Docker Camofox 설치에서는 비활성화된 상태로 두십시오.

또는 `hermes tools` → Browser Automation → Camofox를 통해 구성할 수 있습니다.

`CAMOFOX_URL`이 설정되면 모든 브라우저 도구는 Browserbase 또는 agent-browser 대신 Camofox를 통해 자동으로 라우팅됩니다.

#### 영구적인 브라우저 세션 (Persistent browser sessions)

기본적으로 각 Camofox 세션은 무작위 ID를 부여받으므로, 쿠키 및 로그인 정보는 에이전트를 재시작할 때마다 유지되지 않습니다. 영구적인 브라우저 세션을 활성화하려면 `~/.hermes/config.yaml`에 다음을 추가하세요:

```yaml
browser:
  camofox:
    managed_persistence: true
```

그런 다음 Hermes를 완전히 다시 시작하여 새 구성이 선택되도록 합니다.

:::warning 중첩 경로가 중요합니다
Hermes는 `browser.camofox.managed_persistence`를 읽으며, **최상위** `managed_persistence`는 읽지 않습니다. 일반적인 실수는 다음과 같이 작성하는 것입니다:

```yaml
# ❌ 잘못됨 — Hermes가 무시합니다
managed_persistence: true
```

플래그가 잘못된 경로에 배치되면 Hermes는 기본 무작위 임시 `userId`로 조용히 대체하며, 모든 세션마다 로그인 상태가 손실됩니다.
:::

##### Hermes가 하는 일
- 프로필에 따라 결정론적인 `userId`를 Camofox에 전송하여 서버가 여러 세션에서 동일한 Firefox 프로필을 재사용할 수 있도록 합니다.
- 정리(cleanup) 중 서버 측 컨텍스트 파괴를 건너뛰어, 에이전트 작업 사이에 쿠키와 로그인이 살아남도록 합니다.
- 활성화된 Hermes 프로필로 `userId` 범위를 지정하여 다른 Hermes 프로필이 서로 다른 브라우저 프로필을 갖도록 합니다(프로필 격리).

##### Hermes가 하지 않는 일
- Camofox 서버에서 영구성을 강제하지 않습니다. Hermes는 안정적인 `userId`만 보낼 뿐이며, 서버는 해당 `userId`를 영구적인 Firefox 프로필 디렉토리에 매핑하여 이를 준수해야 합니다.
- 사용 중인 Camofox 서버 빌드가 모든 요청을 일회성(ephemeral)으로 취급하는 경우(예: 항상 저장된 프로필을 로드하지 않고 `browser.newContext()`를 호출하는 경우), Hermes는 해당 세션을 지속시킬 수 없습니다. 사용자별 프로필 영구성을 지원하는 Camofox 빌드를 실행 중인지 확인하십시오.

##### 작동 여부 확인

1. Hermes 및 Camofox 서버를 시작합니다.
2. 브라우저 작업에서 Google (또는 기타 로그인 사이트)을 열고 수동으로 로그인합니다.
3. 브라우저 작업을 정상적으로 종료합니다.
4. 새로운 브라우저 작업을 시작합니다.
5. 동일한 사이트를 다시 엽니다 — 여전히 로그인되어 있어야 합니다.

5단계에서 로그아웃된 경우, Camofox 서버가 안정적인 `userId`를 존중하지 않는 것입니다. 구성 경로를 다시 확인하고, `config.yaml` 편집 후 Hermes를 완전히 다시 시작했는지 확인하며, 사용 중인 Camofox 서버 버전이 영구적인 사용자별 프로필을 지원하는지 확인하십시오.

##### 상태가 저장되는 위치

Hermes는 프로필 범위 디렉토리 `~/.hermes/browser_auth/camofox/` (또는 기본 프로필이 아닌 경우 `$HERMES_HOME` 아래의 해당 디렉토리)에서 안정적인 `userId`를 파생합니다. 실제 브라우저 프로필 데이터는 해당 `userId`를 키로 하여 Camofox 서버 측에 저장됩니다. 영구 프로필을 완전히 초기화하려면 Camofox 서버에서 이를 지우고 해당하는 Hermes 프로필의 상태 디렉토리를 제거하십시오.

#### 외부에서 관리되는 Camofox 세션

다른 앱(데스크탑 어시스턴트, 커스텀 통합 앱, 다른 에이전트 등)이 보이는 Camofox 브라우저를 구동하는 경우, Hermes가 자체적으로 격리된 프로필을 생성하는 대신 동일한 ID 내부에서 작동하도록 구성할 수 있습니다.

세 가지 옵션이 동작을 제어합니다:

| 설정 | 환경 변수 | 효과 |
|---------|---------|--------|
| `browser.camofox.user_id` | `CAMOFOX_USER_ID` | 탭을 생성할 때 Hermes가 사용하는 Camofox `userId`입니다. 이를 설정하면 세션이 "외부 관리" 모드로 전환됩니다. |
| `browser.camofox.session_key` | `CAMOFOX_SESSION_KEY` | 탭 생성 시 전송되는 `sessionKey` (`listItemId` 라고도 함)입니다. 기존 탭을 채택(adopt)할 때 탭을 일치시키기 위해 사용됩니다. 설정하지 않으면 작업별(per-task) 기본값으로 대체됩니다. |
| `browser.camofox.adopt_existing_tab` | `CAMOFOX_ADOPT_EXISTING_TAB` | true일 경우, Hermes는 처음 사용할 때 `GET /tabs?userId=<user_id>`를 호출하여 새 탭을 만들기 전에 기존 탭을 재사용합니다. |

환경 변수가 `config.yaml`보다 우선합니다. 두 형태 모두 가능합니다:

```yaml
browser:
  camofox:
    user_id: shared-camofox
    session_key: visible-tab
    adopt_existing_tab: true
```

```bash
CAMOFOX_USER_ID=shared-camofox
CAMOFOX_SESSION_KEY=visible-tab
CAMOFOX_ADOPT_EXISTING_TAB=true
```

**`user_id`가 설정되었을 때 변경되는 사항:**

- Hermes는 작업 종료 시 파괴적인 정리를 건너뜁니다(`managed_persistence: true`와 동일). 외부 앱의 탭/쿠키/프로필이 그대로 유지됩니다.
- Hermes는 `DELETE /sessions/<user_id>`를 호출하지 **않습니다** — 이 엔드포인트는 모든 사용자 데이터를 삭제하므로, 이를 실행할 경우 외부 앱의 세션까지 삭제될 수 있습니다.

**탭 채택(Tab adoption) 작동 방식 (`adopt_existing_tab: true`일 때):**

1. 프로세스 시작 후 첫 번째 브라우저 도구 호출 시, Hermes는 `GET /tabs?userId=<user_id>`를 요청합니다(시간 초과 5초).
2. 응답에 포함된 탭 중에서 `listItemId == session_key`인 탭이 있다면, Hermes는 해당 그룹에서 가장 최근에 생성된 탭을 채택합니다.
3. 그렇지 않다면, Hermes는 해당 사용자를 위해 생성된 가장 최신 탭을 채택합니다(`listItemId`에 관계없이).
4. 탭이 존재하지 않거나 요청이 실패하면 Hermes는 다음 작업에서 새로운 탭을 생성하도록 폴백(fallback)합니다.

채택은 세션에 대해 `tab_id`가 채워질 때까지만 한 번만 실행됩니다. 외부 앱이 채택된 탭을 도중에 닫으면 다음 브라우저 도구 호출 시 Camofox 오류가 발생합니다 — Hermes는 호출할 때마다 새로운 탭을 찾기 위해 다시 폴링하지 않습니다.

**`session_key` 선택하기:** Hermes가 신뢰할 수 있게 *특정* 기존 탭에 연결되게 하려면 `session_key`를 외부 앱이 생성할 때 사용한 `listItemId`로 설정하세요. `session_key`를 설정하지 않고 `user_id`만 설정할 경우, Hermes는 작업별 고유한 `session_key` (`task_<id>`)를 생성합니다 — 이 경우 쿠키와 프로필은 외부 앱과 공유되지만, 탭을 재사용하지 않고 나란히 새 탭을 열게 됩니다.

**동시성 관련 참고:** 외부 앱과 Hermes는 동시에 동일한 Camofox `userId`를 구동할 수 있지만, Camofox는 클라이언트 간의 탭별 포커스를 조정하지 않습니다. 애플리케이션 계층에서 소유권을 조정하세요(예: Hermes가 실행되는 동안 외부 앱이 일시 정지).

#### VNC 라이브 뷰

Camofox가 브라우저 창이 보이는 헤디드(headed) 모드로 실행될 때 상태 확인 응답에 VNC 포트를 노출합니다. Hermes는 자동으로 이를 감지하여 탐색 응답에 VNC URL을 포함시키므로, 에이전트가 브라우저 작동 화면을 실시간으로 볼 수 있는 링크를 사용자와 공유할 수 있습니다.

### 로컬 Chromium 계열 브라우저 (`/browser connect` 및 CDP 활용)

클라우드 제공자 대신 Chrome DevTools Protocol(CDP)을 통해 내 컴퓨터에서 실행 중인 Chrome, Brave, Chromium 또는 Edge 인스턴스에 Hermes 브라우저 도구를 연결할 수 있습니다. 이는 에이전트가 수행 중인 작업을 실시간으로 확인하고 싶을 때, 사용자의 쿠키/세션이 필요한 페이지와 상호 작용해야 할 때, 또는 클라우드 브라우저 비용을 절약하고 싶을 때 유용합니다.

:::note
`/browser connect`는 **대화형 CLI 슬래시 명령**입니다 — 게이트웨이를 통해 디스패치되지 않습니다. WebUI, Telegram, Discord 또는 기타 게이트웨이 채팅 내에서 실행하려 하면 메시지가 일반 텍스트로 에이전트에게 전송되며 명령이 실행되지 않습니다. 터미널(`hermes` 또는 `hermes chat`)에서 Hermes를 시작하고 거기서 `/browser connect`를 입력하세요.
:::

CLI에서 다음을 사용합니다:

```
/browser connect                 # http://127.0.0.1:9222에 있는 로컬 Chromium 계열 브라우저를 자동 실행/연결합니다
/browser connect ws://host:port  # 특정 CDP 엔드포인트에 연결합니다
/browser status                  # 현재 연결 상태를 확인합니다
/browser disconnect              # 연결을 분리하고 클라우드/로컬 모드로 돌아갑니다
```

원격 디버깅이 활성화된 브라우저가 아직 실행 중이지 않은 경우, Hermes는 `--remote-debugging-port=9222` 옵션을 사용하여 지원되는 Chromium 계열 브라우저의 자동 시작을 시도합니다. Brave, Google Chrome, Chromium 및 Microsoft Edge가 감지 대상이며, `/opt/brave-bin/brave` 및 `/snap/bin/brave`와 같은 일반적인 Linux 설치 경로를 포함합니다.

:::tip
CDP가 활성화된 상태에서 Chromium 계열 브라우저를 수동으로 시작하려면 일반 프로필로 이미 실행 중인 경우에도 디버그 포트가 열릴 수 있도록 전용 user-data-dir을 사용하세요:

```bash
# Linux — Brave
brave-browser \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# Linux — Google Chrome
google-chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=$HOME/.hermes/chrome-debug \
  --no-first-run \
  --no-default-browser-check &

# macOS — Brave
"/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &

# macOS — Google Chrome
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --remote-debugging-port=9222 \
  --user-data-dir="$HOME/.hermes/chrome-debug" \
  --no-first-run \
  --no-default-browser-check &
```

그런 다음 Hermes CLI를 시작하고 `/browser connect`를 실행합니다.

**왜 `--user-data-dir`을 사용해야 하나요?** 이를 지정하지 않으면 일반 브라우저 인스턴스가 이미 실행 중인 상태에서 Chromium 계열 브라우저를 시작할 경우, 일반적으로 기존 프로세스에 새 창을 열 뿐입니다. 그런데 그 기존 프로세스는 `--remote-debugging-port` 옵션 없이 시작되었기 때문에 9222 포트가 열리지 않습니다. 전용 user-data-dir을 설정하면 디버그 포트가 실제로 수신 대기하는 새로운 브라우저 프로세스를 강제로 실행합니다. `--no-first-run --no-default-browser-check` 옵션은 새로운 프로필에 대한 첫 실행 마법사를 건너뜁니다.
:::

CDP를 통해 연결되면 모든 브라우저 도구(`browser_navigate`, `browser_click` 등)는 클라우드 세션을 새로 띄우지 않고 로컬 브라우저 인스턴스에서 작동합니다.

### WSL2 + Windows Chrome: `/browser connect` 보다 MCP를 우선 사용하세요

Hermes가 WSL2 내에서 실행되지만 제어하려는 Chrome 창이 Windows 호스트에서 실행 중인 경우 `/browser connect`는 최선의 경로가 아닌 경우가 많습니다.

그 이유:

- `/browser connect`는 Hermes 자체가 사용 가능한 CDP 엔드포인트에 연결할 수 있다고 기대합니다.
- 최신 Chrome 실시간 디버깅 세션은 종종 기존의 `9222` 포트처럼 WSL에서 직접 접근할 수 없는 호스트-로컬 엔드포인트를 노출합니다.
- Windows Chrome이 디버깅 가능하더라도 가장 깔끔한 통합 방법은 종종 Windows 쪽 브라우저 MCP 서버를 Chrome에 연결한 후, Hermes가 그 MCP 서버와 통신하도록 하는 것입니다.

이러한 설정의 경우 Hermes MCP 지원을 통해 `chrome-devtools-mcp`를 사용하는 것이 좋습니다.

실제 설정 방법은 MCP 가이드를 참조하세요:

- [Hermes에서 MCP 사용하기](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### 로컬 브라우저 모드

클라우드 자격 증명을 전혀 설정하지 **않고** `/browser connect`도 사용하지 않는 경우, Hermes는 `agent-browser`로 구동되는 로컬 Chromium 설치를 통해 계속해서 브라우저 도구를 사용할 수 있습니다.

### 선택적 환경 변수

```bash
# 더 나은 CAPTCHA 해결을 위한 주거용 프록시 (기본값: "true")
BROWSERBASE_PROXIES=true

# 커스텀 Chromium을 이용한 고급 스텔스 기능 — Scale 요금제 필요 (기본값: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# 네트워크 연결이 끊긴 후 세션 재연결 — 유료 요금제 필요 (기본값: "true")
BROWSERBASE_KEEP_ALIVE=true

# 커스텀 세션 시간 초과 설정(초 단위, 최대 21600 = 6시간) (기본값: 프로젝트 기본값)
# 예: 600(10분), 1800(30분), 21600(6시간)
BROWSERBASE_SESSION_TIMEOUT=1800

# 자동 정리 전 비활동 시간 제한(초 단위) (기본값: 120)
BROWSER_INACTIVITY_TIMEOUT=120

# 추가 Chromium 실행 플래그(쉼표 또는 줄바꿈으로 구분).
# Hermes는 root 또는 AppArmor로 제한된 비특권 사용자 네임스페이스(Ubuntu 23.10+, DGX Spark, 많은 컨테이너 이미지)
# 를 감지하면 자동으로 `--no-sandbox,--disable-dev-shm-usage`를 주입합니다.
# 따라서 대부분의 사용자는 이를 설정할 필요가 없습니다.
# Hermes가 자동으로 추가하지 않는 플래그가 필요할 때만 수동으로 설정하세요. 
# 이를 수동으로 설정하면 자동 주입 기능은 비활성화됩니다.
AGENT_BROWSER_ARGS=--no-sandbox
```

### agent-browser CLI 설치

```bash
npm install -g agent-browser
# 또는 저장소에 로컬로 설치:
npm install
```

:::info
`browser` 도구 세트는 config의 `toolsets` 목록에 포함되어 있거나 `hermes config set toolsets '["hermes-cli", "browser"]'`를 통해 활성화되어야 합니다.
:::

## 사용 가능한 도구 (Available Tools)

### `browser_navigate`

URL로 이동합니다. 다른 모든 브라우저 도구보다 먼저 호출해야 합니다. Browserbase 세션을 초기화합니다.

```
Navigate to https://github.com/NousResearch
```

:::tip
단순한 정보 검색의 경우 `web_search`나 `web_extract`가 빠르고 비용이 적게 듭니다. 페이지와 **상호 작용**(버튼 클릭, 양식 작성, 동적 콘텐츠 처리)해야 할 때 브라우저 도구를 사용하세요.
:::

### `browser_snapshot`

현재 페이지의 접근성 트리에 대한 텍스트 기반 스냅샷을 가져옵니다. `browser_click` 및 `browser_type`에서 사용할 수 있도록 대화형 요소에 `@e1`, `@e2`와 같은 참조 ID를 제공합니다.

- **`full=false`** (기본값): 대화형 요소만 표시하는 압축된 보기
- **`full=true`**: 전체 페이지 콘텐츠

8000자를 초과하는 스냅샷은 LLM에 의해 자동으로 요약됩니다.

### `browser_click`

스냅샷의 참조 ID로 식별된 요소를 클릭합니다.

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

입력 필드에 텍스트를 입력합니다. 먼저 필드를 지운 다음 새 텍스트를 입력합니다.

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

페이지를 위나 아래로 스크롤하여 더 많은 콘텐츠를 표시합니다.

```
Scroll down to see more results
```

### `browser_press`

키보드 키를 누릅니다. 양식 제출이나 탐색에 유용합니다.

```
Press Enter to submit the form
```

지원되는 키: `Enter`, `Tab`, `Escape`, `ArrowDown`, `ArrowUp` 등

### `browser_back`

브라우저 기록의 이전 페이지로 돌아갑니다.

### `browser_get_images`

현재 페이지의 모든 이미지와 해당 URL 및 대체 텍스트(alt text)를 나열합니다. 분석할 이미지를 찾을 때 유용합니다.

### `browser_vision`

스크린샷을 찍고 비전 AI로 분석합니다. 텍스트 스냅샷이 중요한 시각적 정보를 포착하지 못할 때 사용하십시오. CAPTCHA, 복잡한 레이아웃 또는 시각적 확인 과제에 특히 유용합니다.

스크린샷은 영구적으로 저장되며 AI 분석 내용과 함께 파일 경로가 반환됩니다. 메시징 플랫폼(Telegram, Discord, Slack, WhatsApp)에서는 에이전트에게 스린샷을 공유하도록 요청할 수 있습니다 — `MEDIA:` 메커니즘을 통해 기본 사진 첨부 파일로 전송됩니다.

```
What does the chart on this page show?
```

스크린샷은 `~/.hermes/cache/screenshots/`에 저장되며 24시간 후에 자동으로 정리됩니다.

### `browser_console`

현재 페이지에서 브라우저 콘솔 출력(log/warn/error 메시지) 및 포착되지 않은 JavaScript 예외를 가져옵니다. 접근성 트리에 나타나지 않는 조용한 JS 오류를 감지하는 데 필수적입니다.

```
Check the browser console for any JavaScript errors
```

읽은 후 콘솔을 지우려면 `clear=True`를 사용하십시오. 그러면 후속 호출은 새로운 메시지만 표시합니다.

`browser_console`은 `expression` 인수와 함께 호출될 때 JavaScript도 평가합니다 — DevTools 콘솔과 동일한 형태이며, 결과는 파싱되어 돌아옵니다(JSON 직렬화된 객체는 딕셔너리가 되고 원시 값은 원시 상태로 유지됨).

```
browser_console(expression="document.querySelector('h1').textContent")
browser_console(expression="JSON.stringify(performance.timing)")
```

현재 세션에 대해 CDP 수퍼바이저(supervisor)가 활성화되어 있는 경우(CDP 지원 백엔드에 대해 `browser_navigate`를 실행한 세션의 일반적인 경우), 평가는 수퍼바이저의 영구적인 WebSocket을 통해 실행되므로 하위 프로세스 시작 비용이 발생하지 않습니다. 그렇지 않으면 표준 agent-browser CLI 경로로 폴백합니다. 동작은 어느 쪽이든 동일하며 지연 시간만 변경됩니다.

### `browser_cdp`

Raw Chrome DevTools Protocol 패스스루 — 다른 도구에서 다루지 않는 브라우저 작업들을 위한 탈출구(escape hatch)입니다. 네이티브 대화 상자 처리, iframe 범위 평가, 쿠키/네트워크 제어 또는 에이전트에 필요한 모든 CDP 동사를 사용할 수 있습니다.

**세션 시작 시 CDP 엔드포인트에 도달할 수 있는 경우에만 사용할 수 있습니다** — 즉, `/browser connect`가 실행 중인 Chrome, Brave, Chromium 또는 Edge에 연결되었거나 `config.yaml`에 `browser.cdp_url`이 설정되어 있어야 합니다. 기본 로컬 agent-browser 모드, Camofox 및 클라우드 제공자(Browserbase, Browser Use, Firecrawl)는 현재 이 도구에 CDP를 노출하지 않습니다 — 클라우드 제공자는 세션별 CDP URL을 가지고 있지만 실시간 세션 라우팅은 후속 작업으로 제공될 예정입니다.

**CDP 메서드 참조:** https://chromedevtools.github.io/devtools-protocol/ — 에이전트는 특정 메서드의 페이지에 `web_extract`를 사용하여 매개변수와 반환 형태를 조회할 수 있습니다.

일반적인 패턴:

```
# 탭 나열 (브라우저 레벨, target_id 없음)
browser_cdp(method="Target.getTargets")

# 탭의 기본 JS 대화 상자 처리
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# 특정 탭에서 JS 평가
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# 모든 쿠키 가져오기
browser_cdp(method="Network.getAllCookies")
```

브라우저 레벨 메서드(`Target.*`, `Browser.*`, `Storage.*`)는 `target_id`를 생략합니다. 페이지 레벨 메서드(`Page.*`, `Runtime.*`, `DOM.*`, `Emulation.*`)는 `Target.getTargets`에서 얻은 `target_id`가 필요합니다. 각각의 상태 비저장 호출은 독립적입니다 — 세션은 호출 간에 유지되지 않습니다.

**교차 출처(Cross-origin) iframe:** 해당 iframe의 활성 수퍼바이저 세션을 통해 CDP 호출을 라우팅하려면 `frame_id` (`is_oopif=true`인 `browser_snapshot.frame_tree.children[]`에서 가져옴)를 전달하세요. 이것이 브라우저베이스에서 교차 출처 iframe 내부의 `Runtime.evaluate`가 작동하는 방식이며, 상태 비저장 CDP 연결은 서명된 URL 만료 문제에 부딪힐 수 있습니다. 예:

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<frame_id from browser_snapshot>",
)
```

동일 출처 iframe에는 `frame_id`가 필요하지 않습니다 — 대신 최상위 `Runtime.evaluate`에서 `document.querySelector('iframe').contentDocument`를 사용하세요.

### `browser_dialog`

네이티브 JS 대화 상자(`alert` / `confirm` / `prompt` / `beforeunload`)에 응답합니다. 이 도구가 존재하기 전에는 대화 상자가 페이지의 JavaScript 스레드를 조용히 차단하여 후속 `browser_*` 호출이 중단되거나(hang) 오류가 발생했지만, 이제 에이전트는 `browser_snapshot` 출력에서 보류 중인 대화 상자를 확인하고 명시적으로 응답할 수 있습니다.

**워크플로우:**
1. `browser_snapshot`을 호출합니다. 대화 상자가 페이지를 차단하고 있으면 `pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]` 형태로 표시됩니다.
2. `browser_dialog(action="accept")` 또는 `browser_dialog(action="dismiss")`를 호출합니다. `prompt()` 대화 상자의 경우 응답을 제공하려면 `prompt_text="..."`를 전달하세요.
3. 스냅샷 다시 캡처 — `pending_dialogs`가 비워지고 페이지의 JS 스레드가 다시 시작됩니다.

Page/Runtime/Target 이벤트를 구독하는 작업당 하나의 영구 WebSocket인 CDP 수퍼바이저를 통해 **감지가 자동으로 이루어집니다**. 수퍼바이저는 스냅샷에 `frame_tree` 필드도 채워 에이전트가 교차 출처(OOPIF) iframe을 포함하여 현재 페이지의 iframe 구조를 볼 수 있게 합니다.

**가용성 매트릭스:**

| 백엔드 | `pending_dialogs`를 통한 감지 | 응답 (`browser_dialog` 도구) |
|---|---|---|
| `/browser connect` 또는 `browser.cdp_url`을 통한 로컬 Chrome | ✓ | ✓ 전체 워크플로우 지원 |
| Browserbase | ✓ | ✓ 전체 워크플로우 지원 (주입된 XHR 브리지를 통해) |
| Camofox / 기본 로컬 agent-browser | ✗ | ✗ (CDP 엔드포인트 없음) |

**Browserbase에서 작동하는 방식.** Browserbase의 CDP 프록시는 약 10ms 내에 서버 측에서 실제 기본 대화 상자를 자동으로 닫아버리므로 `Page.handleJavaScriptDialog`를 사용할 수 없습니다. 수퍼바이저는 `window.alert`/`confirm`/`prompt`를 동기식 XHR로 재정의하는 작은 스크립트를 `Page.addScriptToEvaluateOnNewDocument`를 통해 삽입합니다. 우리는 `Fetch.enable`을 통해 해당 XHR을 가로챕니다 — 에이전트의 응답과 함께 `Fetch.fulfillRequest`를 호출할 때까지 페이지의 JS 스레드는 XHR에서 차단된 상태로 유지됩니다. `prompt()` 반환 값은 수정되지 않고 다시 페이지 JS로 전달됩니다.

**대화 상자 정책(Dialog policy)**은 `config.yaml`의 `browser.dialog_policy`에 구성되어 있습니다:

| 정책 | 동작 |
|--------|----------|
| `must_respond` (기본값) | 캡처, 스냅샷에 표면화하고, 명시적인 `browser_dialog()` 호출을 기다립니다. 버그가 있는 에이전트가 영원히 지연되지 않도록 안전 장치로서 `browser.dialog_timeout_s`(기본값 300초) 후에 자동으로 닫힙니다(auto-dismiss). |
| `auto_dismiss` | 캡처 후 즉시 닫습니다. 에이전트는 여전히 `browser_state` 기록에서 대화 상자를 보지만 조치를 취할 필요는 없습니다. |
| `auto_accept` | 캡처 후 즉시 수락합니다. 공격적인 `beforeunload` 프롬프트가 있는 페이지를 탐색할 때 유용합니다. |

광고가 많은 페이지에서 페이로드 크기를 제한하기 위해 `browser_snapshot.frame_tree` 내부의 **프레임 트리**는 30개의 프레임과 OOPIF 깊이 2로 제한됩니다. 한도에 도달하면 `truncated: true` 플래그가 나타나며, 전체 트리가 필요한 에이전트는 `Page.getFrameTree`와 함께 `browser_cdp`를 사용할 수 있습니다.

## 실용적인 예제

### 웹 양식 채우기

```
사용자: example.com에서 내 이메일 john@example.com으로 계정을 가입해 줘.

에이전트 워크플로우:
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → 참조가 있는 양식 필드 확인
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → "Create Account" 클릭
6. browser_snapshot()  → 성공 확인
```

### 동적 콘텐츠 조사

```
사용자: 지금 GitHub에서 트렌딩 중인 인기 저장소는 뭐야?

에이전트 워크플로우:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → 트렌딩 저장소 목록 읽기
3. 포맷팅된 결과 반환
```

## 세션 녹화 (Session Recording)

브라우저 세션을 WebM 비디오 파일로 자동 녹화합니다:

```yaml
browser:
  record_sessions: true  # 기본값: false
```

활성화되면 첫 번째 `browser_navigate`에서 녹화가 자동으로 시작되고 세션이 닫힐 때 `~/.hermes/browser_recordings/`에 저장됩니다. 로컬 모드 및 클라우드 모드(Browserbase) 모두에서 작동합니다. 72시간이 지난 녹화물은 자동으로 정리됩니다.

## 스텔스 기능 (Stealth Features)

Browserbase는 자동 스텔스 기능을 제공합니다:

| 기능 | 기본값 | 비고 |
|---------|---------|-------|
| 기본 스텔스 | 항상 켜짐 | 무작위 지문, 뷰포트 무작위화, CAPTCHA 해결 |
| 주거용 프록시 | 켜짐 | 더 나은 접근을 위해 주거용 IP를 통해 라우팅 |
| 고급 스텔스 | 꺼짐 | 사용자 지정 Chromium 빌드, Scale 요금제 필요 |
| 연결 유지 (Keep Alive) | 켜짐 | 네트워크 끊김 후 세션 재연결 |

:::note
요금제에서 유료 기능을 사용할 수 없는 경우 Hermes가 자동으로 대체 수단을 적용하여(먼저 `keepAlive` 비활성화, 그 다음 프록시 비활성화), 무료 요금제에서도 브라우징이 계속 작동하도록 합니다.
:::

## 세션 관리 (Session Management)

- 각 작업은 Browserbase를 통해 분리된 브라우저 세션을 갖습니다.
- 비활성 상태가 지나면 세션이 자동으로 정리됩니다(기본값: 2분).
- 백그라운드 스레드가 30초마다 오래된(stale) 세션을 확인합니다.
- 분리된 세션이 고립되지 않도록 프로세스 종료 시 비상 정리(emergency cleanup)가 실행됩니다.
- 세션은 Browserbase API(`REQUEST_RELEASE` 상태)를 통해 릴리스됩니다.

## 제한 사항 (Limitations)

- **텍스트 기반 상호 작용** — 픽셀 좌표가 아닌 접근성 트리에 의존합니다.
- **스냅샷 크기** — 큰 페이지는 8000자에서 잘리거나 LLM을 통해 요약될 수 있습니다.
- **세션 타임아웃** — 클라우드 세션은 제공자의 요금제 설정에 따라 만료됩니다.
- **비용** — 클라우드 세션은 제공자의 크레딧을 소비합니다. 세션은 대화가 종료되거나 비활성화되면 자동으로 정리됩니다. 무료 로컬 브라우징의 경우 `/browser connect`를 사용하십시오.
- **파일 다운로드 불가** — 브라우저에서 파일을 다운로드할 수 없습니다.

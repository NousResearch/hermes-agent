---
title: 브라우저 자동화
description: 여러 제공업체, CDP를 통한 로컬 Chromium 계열 브라우저 또는 웹 상호작용, 양식 입력, 스크래핑 등에 사용할 클라우드 브라우저를 제어합니다.
sidebar_label: 브라우저
sidebar_position: 5
---

# 브라우저 자동화

Hermes Agent에는 여러 백엔드 옵션을 갖춘 완전한 브라우저 자동화 도구 모음이 포함되어 있습니다.

- **Browserbase 클라우드 모드** — 관리형 클라우드 브라우저와 봇 방지 도구를 제공하는 [Browserbase](https://browserbase.com)
- **Browser Use 클라우드 모드** — 대안으로 사용할 수 있는 클라우드 브라우저 제공업체 [Browser Use](https://browser-use.com)
- **Browser Use 모드** — [Browser Use CLI 3.0](https://github.com/browser-use/browser-use)을 사용하는 모드로, 웹 작업에서 SOTA인 새로운 브라우저 하네스이며 로컬 Chrome 또는 Browser Use 클라우드 브라우저를 자동화합니다.
- **Firecrawl 클라우드 모드** — 스크래핑 기능이 내장된 클라우드 브라우저를 제공하는 [Firecrawl](https://firecrawl.dev)
- **Camofox 로컬 모드** — 로컬 봇 탐지 회피 브라우징을 제공하는 [Camofox](https://github.com/jo-inc/camofox-browser)(Firefox 기반 지문 스푸핑)
- **Lightpanda 로컬 엔진** — 컴퓨터용 Zig 기반 헤드리스 브라우저인 [Lightpanda](https://lightpanda.io)는 처음부터 제작되었으며, 즉시 시작되고 Chrome보다 메모리를 16배 적게 사용하며 9배 빠릅니다. 아직 지원하지 않는 작업에는 자동으로 Chrome을 사용합니다.
- **로컬 Chromium 계열 CDP** — `/browser connect`를 사용해 자체 Chrome, Brave, Chromium 또는 Edge 인스턴스에 브라우저 도구 연결
- **로컬 브라우저 모드** — `agent-browser` CLI와 로컬 Chromium 설치를 사용하는 모드

모든 모드에서 에이전트는 웹사이트를 탐색하고, 페이지 요소와 상호작용하고, 양식을 작성하고, 정보를 추출할 수 있습니다.

## 개요

페이지는 **접근성 트리**(텍스트 기반 스냅샷)로 표현되므로 LLM 에이전트에 적합합니다. 상호작용 가능한 요소에는 에이전트가 클릭과 입력에 사용하는 ref ID(예: `@e1`, `@e2`)가 부여됩니다.

주요 기능:

- **다중 제공업체 클라우드 실행** — Browserbase, Browser Use 또는 Firecrawl을 사용하며 로컬 브라우저가 필요하지 않습니다.
- **로컬 Chromium 계열 통합** — CDP를 통해 실행 중인 Chrome, Brave, Chromium 또는 Edge 브라우저에 연결하여 직접 브라우징합니다.
- **내장 스텔스 기능** — 무작위 지문, CAPTCHA 해결, 주거용 프록시(Browserbase)
- **세션 격리** — 각 작업에 자체 브라우저 세션이 할당됩니다.
- **자동 정리** — 비활성 세션은 시간 초과 후 닫힙니다.
- **비전 분석** — 시각적 이해를 위한 스크린샷 및 AI 분석

## 설정

:::tip Nous 구독자
유료 [Nous Portal](https://portal.nousresearch.com) 구독이 있다면 별도의 API 키 없이 **[Tool Gateway](tool-gateway.md)**를 통해 브라우저 자동화를 사용할 수 있습니다. 새로 설치한 경우 `hermes setup --portal`을 실행해 로그인하고 모든 게이트웨이 도구를 한 번에 켤 수 있습니다. 기존 설치에서는 `hermes model` 또는 `hermes tools`에서 브라우저 제공업체로 **Nous Subscription**을 선택하면 됩니다.
:::

### Browserbase 클라우드 모드

Browserbase 관리형 클라우드 브라우저를 사용하려면 다음을 추가합니다.

```bash
# Add to ~/.hermes/.env
BROWSERBASE_API_KEY=***
BROWSERBASE_PROJECT_ID=your-project-id-here
```

자격 증명은 [browserbase.com](https://browserbase.com)에서 발급받습니다.

### Browser Use 클라우드 모드

Browser Use를 클라우드 브라우저 제공업체로 사용하려면 다음을 추가합니다.

```bash
# Add to ~/.hermes/.env
BROWSER_USE_API_KEY=***
```

API 키는 [browser-use.com](https://browser-use.com)에서 발급받습니다.

### Browser Use 모드(기본값)

Browser Use 모드는 내장 브라우저 도구 대신 [Browser Use CLI 3.0](https://github.com/browser-use/browser-use)을 사용합니다. 이 CLI는 웹 작업에서 최첨단 성능을 제공하는 새로운 브라우저 하네스입니다. 에이전트는 브라우저에서 Python을 작성하고 실행하여 웹페이지를 클릭하고, 입력하고, 드래그하고, 스크래핑하고, 상호작용합니다.

**기본 브라우저 모드입니다**. `browser.backend`가 설정되지 않았고 `browser-use` CLI를 실행할 수 있으면(설치되어 있거나 `uvx`를 통해 사용 가능할 때) 에이전트에는 단일 `browser_exec` 도구가 제공됩니다. CLI를 실행할 수 없으면 Hermes는 자동으로 내장 브라우저 도구로 대체합니다.

이 모드는 설정된 브라우저 백엔드와 결합되는 **드라이버**입니다. 즉 로컬 Chrome, Nous 구독 클라우드 브라우저, Browserbase, Firecrawl 또는 Browser Use 클라우드 브라우저 중 `hermes tools` → Browser Automation에서 선택한 브라우저 소스를 구동합니다. 단, Camofox는 하네스가 연결할 CDP 엔드포인트가 없으므로 예외입니다. Camofox 설정에서는 내장 브라우저 도구가 자동으로 유지됩니다.

**동시 세션:** `browser_exec`는 모든 백엔드에서 이름별로 브라우저 작업을 격리하는 `session=<name>` 인자를 받습니다. 이름마다 자체 하네스 데몬(자체 IPC 소켓, 로그, 상태)과 클라우드 백엔드의 자체 브라우저가 할당되므로 병렬 하위 에이전트나 동시에 진행되는 채팅이 하나의 공유 연결을 더 이상 망가뜨리지 않습니다. `session`을 생략하면 공유 기본 데몬을 사용하며, 한 번에 하나씩 브라우징할 때 적합합니다.

내장 브라우저 도구를 사용하도록 옵트아웃하려면 `/browser use off`를 사용하거나 다음을 설정합니다.

```yaml
# Add to ~/.hermes/config.yaml
browser:
  backend: "off"
```

(`backend: "browser-use"`를 사용하면 모드를 명시적으로 강제할 수 있습니다.)

Browser Use 자체의 클라우드 브라우저에는 `browser-use auth login` 또는 `BROWSER_USE_API_KEY`가 필요합니다. 다른 브라우저 소스는 기존 자격 증명을 그대로 사용합니다.

:::note
Browser Use 모드는 컴퓨터에서 모델이 작성한 Python을 실행하므로, 터미널 액세스도 가능한 세션에만 `browser_exec` 도구가 제공됩니다. 터미널 도구 모음 없이 설정된 플랫폼(예: 접근이 제한된 메시징 화면)은 대신 기본 브라우저 도구를 사용합니다.
:::

### Firecrawl 클라우드 모드

Firecrawl을 클라우드 브라우저 제공업체로 사용하려면 다음을 추가합니다.

```bash
# Add to ~/.hermes/.env
FIRECRAWL_API_KEY=fc-***
```

API 키는 [firecrawl.dev](https://firecrawl.dev)에서 발급받습니다. 그런 다음 브라우저 제공업체로 Firecrawl을 선택합니다.

```bash
hermes setup tools
# → Browser Automation → Firecrawl
```

선택적 설정:

```bash
# Self-hosted Firecrawl instance (default: https://api.firecrawl.dev)
FIRECRAWL_API_URL=http://localhost:3002

# Session TTL in seconds (default: 300)
FIRECRAWL_BROWSER_TTL=600
```

### 하이브리드 라우팅: 공개 URL은 클라우드로, LAN/localhost는 로컬로

클라우드 제공업체가 설정되어 있으면 Hermes는 비공개/루프백/LAN 주소(`localhost`, `127.0.0.1`, `192.168.x.x`, `10.x.x.x`, `172.16-31.x.x`, `*.local`, `*.lan`, `*.internal`, IPv6 루프백 `::1`, 링크 로컬 `169.254.x.x`)로 확인되는 URL에 **로컬 Chromium 사이드카**를 자동으로 실행합니다. 공개 URL은 같은 대화에서 계속 클라우드 제공업체를 사용합니다.

이 기능은 "Browserbase를 사용하면서 로컬에서 개발하는" 일반적인 작업 흐름을 해결합니다. 제공업체를 바꾸거나 SSRF 보호를 끄지 않고도 에이전트가 `http://localhost:3000`에서 대시보드를 스크린샷으로 캡처하고 `https://github.com`을 스크래핑할 수 있습니다. 클라우드 제공업체에는 비공개 URL이 전달되지 않습니다.

이 기능은 **기본적으로 켜져 있습니다**. 비활성화하여 모든 URL이 기존처럼 설정된 클라우드 제공업체로 가게 하려면 다음을 설정합니다.

```yaml
# ~/.hermes/config.yaml
browser:
  cloud_provider: browserbase
  auto_local_for_private_urls: false
```

자동 라우팅을 비활성화하면 `browser.allow_private_urls: true`도 설정하지 않는 한 비공개 URL은 `"Blocked: URL targets a private or internal address"`로 거부됩니다. 이 옵션을 설정하면 클라우드 제공업체가 해당 URL에 접속을 시도하지만, Browserbase 등은 LAN에 접근할 수 없으므로 대개 작동하지 않습니다.

요구 사항: 로컬 사이드카는 순수 로컬 모드와 동일한 `agent-browser` CLI를 사용하므로 설치해야 합니다(`hermes setup tools → Browser Automation`이 자동으로 설치합니다). 공개 URL에서 비공개 주소로 이동하는 탐색 후 리디렉션은 여전히 차단됩니다. 공개 경로를 통해 LAN에 접근하기 위해 내부 주소로 리디렉션하는 우회는 사용할 수 없습니다.

### Camofox 로컬 모드

[Camofox](https://github.com/jo-inc/camofox-browser)는 C++ 지문 스푸핑 기능을 갖춘 Firefox 포크인 Camoufox를 감싸는 자체 호스팅 Node.js 서버입니다. 클라우드 의존성 없이 로컬 봇 탐지 회피 브라우징을 제공합니다.

```bash
# Clone the Camofox browser server first
git clone https://github.com/jo-inc/camofox-browser
cd camofox-browser

# Build and start with Docker using the default container settings
# (auto-detects arch: aarch64 on M1/M2, x86_64 on Intel)
make up

# Stop and remove the default container
make down

# Force a clean rebuild (for example, after upgrading VERSION/RELEASE)
make reset

# Just download binaries without building
make fetch

# Override arch or version explicitly
make up ARCH=x86_64
make up VERSION=135.0.1 RELEASE=beta.24
```

`make up`은 기본 컨테이너를 즉시 시작합니다. 더 큰 Node 힙, VNC 또는 영구 프로필 디렉터리 같은 사용자 지정 런타임 설정을 사용하려면 먼저 이미지를 빌드한 다음 직접 실행합니다.

```bash
# Build the image without starting the default container
make build

# Start with persistence, VNC live view, and a larger Node heap
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

VNC를 활성화하면 브라우저가 창이 표시되는 모드로 실행되며 `http://localhost:6080`(noVNC)에서 브라우저로 실시간 확인할 수 있습니다. 네이티브 VNC 클라이언트를 사용해 `localhost:5901`에 연결할 수도 있습니다.

이미 `make up`을 실행했다면 사용자 지정 컨테이너를 시작하기 전에 기본 컨테이너를 중지하고 제거합니다.

```bash
make down
# then run the custom docker run command above
```

그런 다음 `~/.hermes/.env`에 설정합니다.

```bash
CAMOFOX_URL=http://localhost:9377
```

Camofox가 Docker에서 실행 중이고 호스트 컴퓨터에서 제공하는 웹 앱을 열려면 루프백 재작성을 활성화합니다. `CAMOFOX_URL`은 호스트에 게시된 제어 API를 계속 가리켜야 하지만, `http://127.0.0.1:3000` 같은 페이지 URL은 컨테이너 내부에서 `http://host.docker.internal:3000`으로 열어야 합니다.

```yaml
# ~/.hermes/config.yaml
browser:
  camofox:
    rewrite_loopback_urls: true
    loopback_host_alias: host.docker.internal  # default; use a LAN IP if needed
```

동등한 환경 변수:

```bash
CAMOFOX_REWRITE_LOOPBACK_URLS=true
CAMOFOX_LOOPBACK_HOST_ALIAS=host.docker.internal
```

이 재작성은 루프백 호스트(`localhost`, `127.0.0.1`, `::1`)가 포함된 페이지 탐색 URL에만 적용됩니다. `CAMOFOX_URL`은 변경하지 않습니다. 브라우저가 호스트에서 직접 실행되어 루프백 URL이 올바른 비 Docker Camofox 설치에서는 비활성화된 상태로 두십시오.

또는 `hermes tools` → Browser Automation → Camofox를 통해 설정합니다.

`CAMOFOX_URL`이 설정되면 모든 브라우저 도구는 Browserbase 또는 agent-browser 대신 자동으로 Camofox를 통해 라우팅됩니다.

#### 영구 브라우저 세션

기본적으로 각 Camofox 세션에는 무작위 정체성이 할당되므로 에이전트를 다시 시작하면 쿠키와 로그인이 유지되지 않습니다. 영구 브라우저 세션을 활성화하려면 `~/.hermes/config.yaml`에 다음을 추가합니다.

```yaml
browser:
  camofox:
    managed_persistence: true
```

새 설정을 적용하려면 Hermes를 완전히 다시 시작합니다.

:::warning 중첩 경로가 중요합니다
Hermes는 최상위 `managed_persistence`가 아니라 `browser.camofox.managed_persistence`를 읽습니다. 다음처럼 작성하는 것은 흔한 실수입니다.

```yaml
# ❌ Wrong — Hermes ignores this
managed_persistence: true
```

플래그를 잘못된 경로에 두면 Hermes는 조용히 무작위 임시 `userId`로 대체하므로 매 세션마다 로그인 상태가 사라집니다.
:::

##### Hermes가 하는 일
- Camofox가 세션 간 동일한 Firefox 프로필을 재사용할 수 있도록 결정론적인 프로필 범위 `userId`를 보냅니다.
- 정리할 때 서버 측 컨텍스트 삭제를 건너뛰므로 에이전트 작업 사이에도 쿠키와 로그인이 유지됩니다.
- `userId`를 활성 Hermes 프로필에 한정하므로 서로 다른 Hermes 프로필은 서로 다른 브라우저 프로필을 사용합니다(프로필 격리).

##### Hermes가 하지 않는 일
- Camofox 서버에 영구 저장을 강제하지 않습니다. Hermes는 안정적인 `userId`만 보내며, 서버가 해당 `userId`를 영구 Firefox 프로필 디렉터리에 매핑하여 이를 준수해야 합니다.
- Camofox 서버 빌드가 저장된 프로필을 불러오지 않고 모든 요청을 임시로 처리한다면(예: 항상 저장된 프로필을 로드하지 않은 채 `browser.newContext()` 호출), Hermes가 세션을 영구화할 수 없습니다. userId 기반 프로필 영구 저장을 구현한 Camofox 빌드를 사용해야 합니다.

##### 작동 여부 확인

1. Hermes와 Camofox 서버를 시작합니다.
2. 브라우저 작업에서 Google(또는 로그인 사이트)을 열고 수동으로 로그인합니다.
3. 브라우저 작업을 정상적으로 종료합니다.
4. 새 브라우저 작업을 시작합니다.
5. 같은 사이트를 다시 엽니다. 여전히 로그인되어 있어야 합니다.

5단계에서 로그아웃되어 있다면 Camofox 서버가 안정적인 `userId`를 준수하지 않는 것입니다. 설정 경로를 다시 확인하고, `config.yaml`을 편집한 뒤 Hermes를 완전히 다시 시작했는지 확인하며, Camofox 서버 버전이 사용자별 영구 프로필을 지원하는지 검증합니다.

##### 상태 저장 위치

Hermes는 프로필 범위 디렉터리 `~/.hermes/browser_auth/camofox/`(또는 기본 프로필이 아닌 경우 `$HERMES_HOME` 아래의 동등한 경로)에서 안정적인 `userId`를 파생합니다. 실제 브라우저 프로필 데이터는 Camofox 서버 측에서 해당 `userId`를 키로 저장됩니다. 영구 프로필을 완전히 초기화하려면 Camofox 서버에서 프로필을 지우고 해당 Hermes 프로필의 상태 디렉터리를 제거합니다.

#### 외부에서 관리하는 Camofox 세션

다른 앱(데스크톱 도우미, 사용자 지정 통합 또는 다른 에이전트)이 표시되는 Camofox 브라우저를 구동하는 경우, 자체적으로 격리된 프로필을 생성하는 대신 동일한 정체성 안에서 작동하도록 Hermes를 설정합니다.

세 가지 설정이 동작을 제어합니다.

| 설정 | 환경 변수 | 효과 |
|---------|---------|--------|
| `browser.camofox.user_id` | `CAMOFOX_USER_ID` | 탭 생성 시 Hermes가 사용하는 Camofox `userId`입니다. 설정하면 세션이 "외부 관리" 모드로 전환됩니다. |
| `browser.camofox.session_key` | `CAMOFOX_SESSION_KEY` | 탭 생성 시 보내는 `sessionKey`(일명 `listItemId`)입니다. 기존 탭을 채택할 때 일치하는 탭을 찾는 데 사용됩니다. 설정하지 않으면 작업별 값이 기본으로 사용됩니다. |
| `browser.camofox.adopt_existing_tab` | `CAMOFOX_ADOPT_EXISTING_TAB` | true이면 Hermes가 처음 사용할 때 `GET /tabs?userId=<user_id>`를 호출하고 새 탭을 만들기 전에 기존 탭을 재사용합니다. |

환경 변수는 `config.yaml`보다 우선합니다. 어느 형식이든 사용할 수 있습니다.

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

**`user_id`를 설정하면 달라지는 점:**

- Hermes는 작업 종료 시 파괴적인 정리를 건너뜁니다(`managed_persistence: true`와 동일). 다른 앱의 탭/쿠키/프로필이 유지됩니다.
- Hermes는 `DELETE /sessions/<user_id>`를 호출하지 않습니다. 이 엔드포인트는 모든 사용자 데이터를 삭제하므로 호출하면 외부 앱의 세션을 없애기 때문입니다.

**탭 채택 방식(`adopt_existing_tab: true`일 때):**

1. 프로세스가 시작된 후 첫 브라우저 도구 호출에서 Hermes는 `GET /tabs?userId=<user_id>`를 실행합니다(시간 초과 5초).
2. 응답의 탭 중 `listItemId == session_key`인 탭이 있으면 Hermes는 해당 그룹에서 가장 최근에 생성된 탭을 채택합니다.
3. 그렇지 않으면 사용자에 해당하는 가장 최근 생성 탭(`listItemId`가 무엇이든)을 채택합니다.
4. 탭이 없거나 요청이 실패하면 Hermes는 다음 작업에서 새 탭을 생성합니다.

채택은 세션에 `tab_id`가 채워질 때까지만 실행됩니다. 외부 앱이 실행 중에 채택된 탭을 닫으면 다음 브라우저 도구 호출에서 Camofox 오류가 표시됩니다. Hermes는 호출마다 새 탭을 찾기 위해 다시 조회하지 않습니다.

**`session_key` 선택:** 특정 기존 탭에 안정적으로 연결하려면 외부 앱이 탭을 만들 때 사용한 `listItemId`를 `session_key`로 설정합니다. `session_key`를 설정하지 않고 `user_id`만 설정하면 Hermes는 작업별 `session_key`(`task_<id>`)를 생성합니다. 이 경우 외부 앱과 쿠키 및 프로필은 공유하지만, 기존 탭을 재사용하는 대신 옆에 자체 탭을 엽니다.

**동시성 참고:** 외부 앱과 Hermes는 동일한 Camofox `userId`를 동시에 구동할 수 있지만 Camofox는 클라이언트 간 탭 포커스를 조정하지 않습니다. 애플리케이션 계층에서 소유권을 조정하십시오(예: Hermes가 실행되는 동안 외부 앱을 일시 중지).

#### VNC 실시간 보기

Camofox가 창이 표시되는 모드로 실행되면 상태 확인 응답에 VNC 포트를 노출합니다. Hermes는 이를 자동으로 검색하고 탐색 응답에 VNC URL을 포함하므로 에이전트가 브라우저를 실시간으로 볼 수 있는 링크를 공유할 수 있습니다.

### Lightpanda 로컬 엔진

[Lightpanda](https://lightpanda.io)는 처음부터 작성된 오픈 소스 헤드리스 브라우저입니다. 즉시 시작되고 Chrome보다 9배 빠르며 메모리를 16배 적게 사용하므로, 장시간 소형 VM에서 실행되는 에이전트에 유용합니다.

Lightpanda는 **로컬 엔진**이며 로컬 `agent-browser` 경로(클라우드 제공업체가 아님)에서 선택합니다. 바이너리를 설치하고 `PATH`에 추가한 다음([Lightpanda 설치 안내](https://lightpanda.io/docs) 참고), 다음을 설정합니다.

```yaml
# Add to ~/.hermes/config.yaml
browser:
  engine: lightpanda
```

또는 환경 변수를 사용합니다.

```bash
AGENT_BROWSER_ENGINE=lightpanda
```

Hermes는 로컬 Chrome을 구동하는 것과 같은 방식으로 `agent-browser`를 통해 CDP로 Lightpanda를 구동합니다.

**자동 Chrome 대체.** Lightpanda는 아직 Chrome의 모든 기능을 지원하지 않으므로 통합은 작업을 방해하지 않습니다. Lightpanda가 지원하는 작업은 Lightpanda가 처리하고, 지원하지 않는 작업은 Hermes가 투명하게 Chrome에서 재시도합니다. 지원 범위에는 에이전트 핵심 작업인 navigate, snapshot, click, type, scroll, back, press, eval이 포함됩니다. Lightpanda에는 그래픽 렌더러가 없으므로 스크린샷도 Chrome으로 대체되며, 같은 이유로 `browser_vision`은 Chrome으로 사전 라우팅됩니다.

### CDP를 통한 로컬 Chromium 계열 브라우저(`/browser connect`)

클라우드 제공업체 대신 Chrome DevTools Protocol(CDP)을 통해 Hermes 브라우저 도구를 실행 중인 자체 Chrome, Brave, Chromium 또는 Edge 인스턴스에 연결할 수 있습니다. 에이전트가 하는 일을 실시간으로 보고 싶거나, 자체 쿠키/세션이 필요한 페이지와 상호작용하거나, 클라우드 브라우저 비용을 피하고 싶을 때 유용합니다.

:::note
`/browser connect`는 **대화형 CLI 슬래시 명령**이며 게이트웨이에서 전달되지 않습니다. WebUI, Telegram, Discord 또는 다른 게이트웨이 채팅에서 실행하면 메시지가 일반 텍스트로 에이전트에 전송되고 명령은 실행되지 않습니다. 터미널에서 Hermes(`hermes` 또는 `hermes chat`)를 시작한 뒤 `/browser connect`를 입력하십시오.
:::

CLI에서 다음을 사용합니다.

```
/browser connect                 # Auto-launch/connect to a local Chromium-family browser at http://127.0.0.1:9222
/browser connect ws://host:port  # Connect to a specific CDP endpoint
/browser status                  # Check current connection
/browser disconnect              # Detach and return to cloud/local mode
```

원격 디버깅이 활성화된 브라우저가 이미 실행 중이 아니라면 Hermes는 `--remote-debugging-port=9222`로 지원되는 Chromium 계열 브라우저를 자동 실행하려고 시도합니다. Brave, Google Chrome, Chromium, Microsoft Edge를 검색하며 `/opt/brave-bin/brave`, `/snap/bin/brave` 같은 일반적인 Linux 설치 경로도 포함됩니다.

:::tip
CDP가 활성화된 Chromium 계열 브라우저를 수동으로 시작하려면 전용 사용자 데이터 디렉터리를 사용하십시오. 그래야 브라우저가 이미 일반 프로필로 실행 중이어도 디버그 포트가 실제로 열립니다.

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

**`--user-data-dir`를 사용하는 이유는 무엇인가요?** 일반 인스턴스가 이미 실행 중일 때 원격 디버깅 없이 시작된 기존 프로세스에서 새 창이 열리는 경우가 많습니다. 그러면 해당 프로세스에는 `--remote-debugging-port`가 없으므로 포트 9222가 열리지 않습니다. 전용 사용자 데이터 디렉터리는 디버그 포트가 실제로 수신 대기하는 새 브라우저 프로세스를 강제로 생성합니다. `--no-first-run --no-default-browser-check`는 새 프로필의 최초 실행 마법사를 건너뜁니다.

**Chrome 136 이상에서는 전용 프로필이 필수입니다.** 보안 강화 조치로 Chrome 136 이상은 기본 사용자 데이터 디렉터리와 `--remote-debugging-port`를 함께 사용하면, 다른 Chrome이 실행 중이지 않은 완전한 콜드 스타트에서도 원격 디버깅 포트를 조용히 열지 않습니다. 브라우저는 정상적으로 실행되지만 9222에서 아무것도 수신하지 않으므로 `/browser connect`와 수동 `curl http://127.0.0.1:9222/json/version`이 모두 연결 거부로 실패합니다. 오류 메시지도 없습니다. 해결 방법은 위 명령처럼 항상 기본 프로필 디렉터리가 아닌 곳(예: `$HOME/.hermes/chrome-debug`)을 가리키는 `--user-data-dir`을 전달하는 것입니다. 이 변경을 적용한 Chrome, Chromium, Edge, Brave 빌드에 해당합니다.
:::

CDP로 연결하면 모든 브라우저 도구(`browser_navigate`, `browser_click` 등)가 클라우드 세션을 생성하는 대신 라이브 브라우저 인스턴스에서 작동합니다.

### WSL2 + Windows Chrome: `/browser connect`보다 MCP 권장

Hermes가 WSL2 내부에서 실행되고 제어하려는 Chrome 창이 Windows 호스트에서 실행 중이라면 `/browser connect`가 최선의 경로가 아닐 수 있습니다.

이유:

- `/browser connect`는 Hermes 자체가 사용 가능한 CDP 엔드포인트에 접근할 수 있어야 합니다.
- 최신 Chrome 라이브 디버깅 세션은 호스트 로컬 엔드포인트를 노출하는 경우가 많으며, 이 엔드포인트는 일반적인 `9222` 포트와 같은 방식으로 WSL에서 직접 접근할 수 없습니다.
- Windows 측 브라우저 MCP 서버가 Chrome에 연결하고 Hermes가 해당 MCP 서버와 통신하도록 하는 방식이 통합 면에서 더 깔끔한 경우가 많습니다.

이 설정에는 Hermes MCP 지원을 통한 `chrome-devtools-mcp`를 권장합니다.

실제 설정 방법은 MCP 안내를 참고하십시오.

- [Hermes에서 MCP 사용](../../guides/use-mcp-with-hermes.md#wsl2-bridge-hermes-in-wsl-to-windows-chrome)

### 로컬 브라우저 모드

클라우드 자격 증명을 설정하지 않고 `/browser connect`도 사용하지 않는 경우에도 Hermes는 `agent-browser`가 구동하는 로컬 Chromium 설치를 통해 브라우저 도구를 사용할 수 있습니다.

### 선택적 환경 변수

```bash
# Residential proxies for better CAPTCHA solving (default: "true")
BROWSERBASE_PROXIES=true

# Advanced stealth with custom Chromium — requires Scale Plan (default: "false")
BROWSERBASE_ADVANCED_STEALTH=false

# Session reconnection after disconnects — requires paid plan (default: "true")
BROWSERBASE_KEEP_ALIVE=true

# Custom session timeout in seconds (max 21600 = 6 hours) (default: project default)
# Examples: 600 (10min), 1800 (30min), 21600 (6h max)
BROWSERBASE_SESSION_TIMEOUT=1800

# Inactivity timeout before auto-cleanup in seconds (default: 120)
BROWSER_INACTIVITY_TIMEOUT=120

# Local browser engine. Applies to the built-in browser tools
# (agent-browser path). Equivalent to browser.engine in config.yaml.
#   auto       — agent-browser's default (currently Chrome)
#   lightpanda — Lightpanda
#   chrome     — force Chrome explicitly
AGENT_BROWSER_ENGINE=auto

# Extra Chromium launch flags (comma- or newline-separated). Hermes auto-injects
# `--no-sandbox,--disable-dev-shm-usage` when it detects root or AppArmor-restricted
# unprivileged user namespaces (Ubuntu 23.10+, DGX Spark, many container images),
# so most users don't need to set this. Set it manually only if you need a flag
# Hermes doesn't add automatically; setting it disables the auto-injection.
AGENT_BROWSER_ARGS=--no-sandbox
```

### agent-browser CLI 설치

아무것도 설치할 필요가 없습니다. 첫 브라우저 도구 사용 시 `agent-browser`가 `npx agent-browser`를 통해 자동으로 확인됩니다. 최초의 npx 다운로드를 피하려면 미리 전역으로 설치할 수 있습니다(선택 사항).

```bash
npm install -g agent-browser
```

:::info
`browser` 도구 모음은 설정의 `toolsets` 목록에 포함되어 있거나 `hermes config set toolsets '["hermes-cli", "browser"]'`를 통해 활성화되어 있어야 합니다.
:::

## 사용 가능한 도구

### `browser_navigate`

URL로 이동합니다. 다른 브라우저 도구보다 먼저 호출해야 하며 Browserbase 세션을 초기화합니다.

```
Navigate to https://github.com/NousResearch
```

:::tip
단순한 정보 검색에는 `web_search` 또는 `web_extract`를 우선 사용하십시오. 더 빠르고 저렴합니다. 페이지와 상호작용하거나 버튼을 클릭하고, 양식을 작성하고, 동적 콘텐츠를 처리해야 할 때 브라우저 도구를 사용합니다.
:::

### `browser_snapshot`

현재 페이지의 접근성 트리를 텍스트 기반 스냅샷으로 가져옵니다. `browser_click`과 `browser_type`에 사용할 수 있도록 ref ID(예: `@e1`)가 있는 상호작용 요소를 반환합니다.

- **`full=false`**(기본값): 상호작용 요소만 표시하는 간결한 보기
- **`full=true`**: 페이지 전체 콘텐츠

15,000자를 초과하는 스냅샷은 자동으로 잘리거나 LLM으로 요약됩니다(`web_extract`와 동일한 페이지별 예산). 이때 완전한 스냅샷은 `~/.hermes/cache/web/`에 저장되며, 도구 출력에는 파일 경로와 전체 접근성 트리를 다시 스냅샷하지 않고 페이지 단위로 읽을 수 있는 즉시 사용 가능한 `read_file` 호출이 포함됩니다. 여기에는 잘린 이후의 요소 ref도 포함됩니다.

### `browser_click`

스냅샷의 ref ID로 식별한 요소를 클릭합니다.

```
Click @e5 to press the "Sign In" button
```

### `browser_type`

입력 필드에 텍스트를 입력합니다. 먼저 필드를 지운 다음 입력합니다.

```
Type "hermes agent" into the search field @e3
```

### `browser_scroll`

더 많은 콘텐츠를 표시하도록 페이지를 위나 아래로 스크롤합니다.

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

브라우저 기록에서 이전 페이지로 이동합니다.

### `browser_get_images`

현재 페이지의 모든 이미지와 URL 및 대체 텍스트를 나열합니다. 분석할 이미지를 찾을 때 유용합니다.

### `browser_vision`

스크린샷을 촬영하고 비전 AI로 분석합니다. 텍스트 스냅샷에 중요한 시각 정보가 담기지 않을 때, 특히 CAPTCHA, 복잡한 레이아웃 또는 시각적 검증 과제에 유용합니다.

스크린샷은 영구 저장되며 AI 분석과 함께 파일 경로가 반환됩니다. 메시징 플랫폼(Telegram, Discord, Slack, WhatsApp)에서는 에이전트에게 스크린샷 공유를 요청할 수 있으며 `MEDIA:` 메커니즘을 통해 네이티브 사진 첨부로 전송됩니다.

```
What does the chart on this page show?
```

스크린샷은 `~/.hermes/cache/screenshots/`에 저장되며 24시간 후 자동으로 정리됩니다.

### `browser_console`

현재 페이지의 브라우저 콘솔 출력(로그/경고/오류)과 포착되지 않은 JavaScript 예외를 가져옵니다. 접근성 트리에 표시되지 않는 조용한 JavaScript 오류를 찾는 데 필수적입니다.

```
Check the browser console for any JavaScript errors
```

`clear=True`를 사용하면 콘솔을 읽은 뒤 지울 수 있으므로 이후 호출에서는 새 출력만 확인할 수 있습니다.

`browser_console`은 `expression` 인자를 사용해 JavaScript도 평가합니다. DevTools 콘솔과 같은 형태이며 결과는 파싱되어 반환됩니다(JSON으로 직렬화되는 객체는 dict가 되고 원시 값은 그대로 유지됩니다).

```
browser_console(expression="document.querySelector('h1').textContent")
browser_console(expression="JSON.stringify(performance.timing)")
```

현재 세션에서 CDP supervisor가 활성화되어 있으면(일반적으로 CDP 지원 백엔드에서 `browser_navigate`를 실행한 모든 세션) 평가가 supervisor의 영구 WebSocket을 통해 실행되므로 하위 프로세스 시작 비용이 없습니다. 그렇지 않으면 표준 agent-browser CLI 경로를 사용합니다. 동작은 동일하고 지연 시간만 달라집니다.

평가는 기본적으로 제한되지 않으므로 `fetch`를 사용하고, 저장소를 읽고, 양식 값을 조회하고, DOM을 추출할 수 있습니다. 비공개/내부 주소를 대상으로 하는 요청은 비로컬 백엔드에서 여전히 차단됩니다(SSRF 보호는 이 설정과 독립적입니다). 로그인된 프로필로 악성 페이지를 탐색하면서 민감한 JavaScript 기본 요소(쿠키, 저장소, 클립보드, 네트워크 호출, 양식 값)에 엄격한 거부 목록을 적용하려면 `config.yaml`에서 `browser.restrict_evaluate: true`를 설정하십시오. 거부 목록은 기본 요소 이름과 일치하므로 `fetch`나 `cookie`라는 단어를 단순히 포함하는 정상적인 표현식도 차단한다는 점에 유의하십시오.

### `browser_cdp`

다른 도구가 다루지 않는 브라우저 작업을 위한 원시 Chrome DevTools Protocol 전달 기능입니다. 네이티브 대화 상자 처리, iframe 범위 평가, 쿠키/네트워크 제어 또는 에이전트에 필요한 모든 CDP 동사에 사용합니다.

**세션 시작 시 CDP 엔드포인트에 접근할 수 있을 때만 사용할 수 있습니다.** 즉 `/browser connect`가 실행 중인 Chrome, Brave, Chromium 또는 Edge 브라우저에 연결했거나 `config.yaml`에 `browser.cdp_url`이 설정되어 있어야 합니다. 기본 로컬 agent-browser 모드, Camofox 및 클라우드 제공업체(Browserbase, Browser Use, Firecrawl)는 현재 이 도구에 CDP를 노출하지 않습니다. 클라우드 제공업체에는 세션별 CDP URL이 있지만 라이브 세션 라우팅은 후속 기능입니다.

**CDP 메서드 참고:** https://chromedevtools.github.io/devtools-protocol/ — 에이전트는 `web_extract`로 특정 메서드 페이지를 조회하여 매개변수와 반환 형태를 확인할 수 있습니다.

일반적인 패턴:

```
# List tabs (browser-level, no target_id)
browser_cdp(method="Target.getTargets")

# Handle a native JS dialog on a tab
browser_cdp(method="Page.handleJavaScriptDialog",
            params={"accept": true, "promptText": ""},
            target_id="<tabId>")

# Evaluate JS in a specific tab
browser_cdp(method="Runtime.evaluate",
            params={"expression": "document.title", "returnByValue": true},
            target_id="<tabId>")

# Get all cookies
browser_cdp(method="Network.getAllCookies")
```

브라우저 수준 메서드(`Target.*`, `Browser.*`, `Storage.*`)는 `target_id`를 생략합니다. 페이지 수준 메서드(`Page.*`, `Runtime.*`, `DOM.*`, `Emulation.*`)는 `Target.getTargets`에서 가져온 `target_id`가 필요합니다. 각 상태 비저장 호출은 독립적이며 세션은 호출 간에 유지되지 않습니다.

**교차 출처 iframe:** `browser_snapshot.frame_tree.children[]`에서 `is_oopif=true`인 프레임의 `frame_id`를 전달하면 해당 iframe에 대한 CDP 호출이 supervisor의 라이브 세션을 통해 라우팅됩니다. 이것이 Browserbase에서 상태 비저장 CDP 연결이 서명된 URL 만료를 겪지 않고 교차 출처 iframe 내부에서 `Runtime.evaluate`를 실행하는 방법입니다. 예:

```
browser_cdp(
  method="Runtime.evaluate",
  params={"expression": "document.title", "returnByValue": True},
  frame_id="<frame_id from browser_snapshot>",
)
```

동일 출처 iframe에는 `frame_id`가 필요하지 않습니다. 최상위 `Runtime.evaluate`에서 `document.querySelector('iframe').contentDocument`를 사용합니다.

### `browser_dialog`

네이티브 JS 대화 상자(`alert` / `confirm` / `prompt` / `beforeunload`)에 응답합니다. 이 도구가 없을 때는 대화 상자가 페이지의 JavaScript 스레드를 조용히 차단하여 이후 `browser_*` 호출이 멈추거나 오류를 냈습니다. 이제 에이전트는 `browser_snapshot` 출력에서 보류 중인 대화 상자를 확인하고 명시적으로 응답할 수 있습니다.

**작업 흐름:**
1. `browser_snapshot`을 호출합니다. 페이지를 차단하는 대화 상자가 있으면 `pending_dialogs: [{"id": "d-1", "type": "alert", "message": "..."}]`로 표시됩니다.
2. `browser_dialog(action="accept")` 또는 `browser_dialog(action="dismiss")`를 호출합니다. `prompt()` 대화 상자에는 응답을 제공하기 위해 `prompt_text="..."`를 전달합니다.
3. 다시 스냅샷을 생성합니다. `pending_dialogs`가 비어 있고 페이지의 JS 스레드가 재개됩니다.

**감지는 자동으로 실행됩니다.** 영구 CDP supervisor가 작업별 하나의 WebSocket에서 Page/Runtime/Target 이벤트를 구독합니다. supervisor는 현재 페이지의 iframe 구조를 에이전트가 확인할 수 있도록 스냅샷에 `frame_tree` 필드도 채웁니다. 여기에는 교차 출처(OOPIF) iframe도 포함됩니다.

**사용 가능성 표:**

| 백엔드 | `pending_dialogs`를 통한 감지 | 응답(`browser_dialog` 도구) |
|---|---|---|
| `/browser connect` 또는 `browser.cdp_url`을 통한 로컬 Chrome | ✓ | ✓ 전체 작업 흐름 |
| Browserbase | ✓ | ✓ 전체 작업 흐름(주입된 XHR 브리지 사용) |
| Camofox / 기본 로컬 agent-browser | ✗ | ✗(CDP 엔드포인트 없음) |

**Browserbase에서 작동하는 방식.** Browserbase의 CDP 프록시는 실제 네이티브 대화 상자를 서버 측에서 약 10ms 이내에 자동으로 닫으므로 `Page.handleJavaScriptDialog`를 사용할 수 없습니다. supervisor는 `Page.addScriptToEvaluateOnNewDocument`를 통해 `window.alert`/`confirm`/`prompt`를 동기 XHR로 덮어씁니다. `Fetch.enable`을 통해 이 XHR을 가로채고, 에이전트의 응답으로 `Fetch.fulfillRequest`를 호출할 때까지 페이지의 JS 스레드는 XHR에서 차단된 상태로 유지됩니다. `prompt()` 반환 값은 변경 없이 페이지 JS로 왕복합니다.

**대화 상자 정책**은 `config.yaml`의 `browser.dialog_policy`에서 설정합니다.

| 정책 | 동작 |
|--------|----------|
| `must_respond` (기본값) | 캡처하고 스냅샷에 표시한 뒤 명시적인 `browser_dialog()` 호출을 기다립니다. 안전을 위해 `browser.dialog_timeout_s`(기본값 300초) 후 자동으로 닫으므로 버그가 있는 에이전트가 영원히 멈추지 않습니다. |
| `auto_dismiss` | 캡처하고 즉시 닫습니다. 에이전트는 여전히 `browser_state` 기록에서 대화 상자를 볼 수 있지만 조작할 필요는 없습니다. |
| `auto_accept` | 캡처하고 즉시 수락합니다. 공격적인 `beforeunload` 프롬프트가 있는 페이지를 탐색할 때 유용합니다. |

**`browser_snapshot.frame_tree` 내부의 프레임 트리**는 광고가 많은 페이지에서 페이로드 크기를 제한하기 위해 프레임 30개, OOPIF 깊이 2로 제한됩니다. 제한에 도달하면 `truncated: true` 플래그가 표시됩니다. 전체 트리가 필요한 에이전트는 `Page.getFrameTree`와 함께 `browser_cdp`를 사용할 수 있습니다.

## 실제 예시

### 웹 양식 작성

```
User: Sign up for an account on example.com with my email john@example.com

Agent workflow:
1. browser_navigate("https://example.com/signup")
2. browser_snapshot()  → sees form fields with refs
3. browser_type(ref="@e3", text="john@example.com")
4. browser_type(ref="@e5", text="SecurePass123")
5. browser_click(ref="@e8")  → clicks "Create Account"
6. browser_snapshot()  → confirms success
```

### 동적 콘텐츠 조사

```
User: What are the top trending repos on GitHub right now?

Agent workflow:
1. browser_navigate("https://github.com/trending")
2. browser_snapshot(full=true)  → reads trending repo list
3. Returns formatted results
```

## 세션 녹화

브라우저 세션을 WebM 동영상 파일로 자동 녹화합니다.

```yaml
browser:
  record_sessions: true  # default: false
```

활성화하면 첫 `browser_navigate`에서 녹화가 자동으로 시작되고 세션이 닫힐 때 `~/.hermes/browser_recordings/`에 저장됩니다. 로컬 및 클라우드(Browserbase) 모드에서 모두 작동합니다. 72시간이 지난 녹화는 자동으로 정리됩니다.

## 창이 표시되는 모드(Visible Browser Window)

기본적으로 로컬 브라우저는 헤드리스로 실행됩니다. 보고 상호작용할 수 있는 Chromium 창을 표시하려면 창이 표시되는 모드를 활성화합니다.

```yaml
browser:
  headed: true  # default: false
```

또는 환경 변수를 사용합니다: `AGENT_BROWSER_HEADED=1`.

창이 표시되는 모드는 두 가지 일을 합니다.

1. **Chromium을 표시되는 창으로 실행합니다**(로컬 모드에서 `agent-browser`에 `--headed` 전달).
2. **턴 사이에 창을 열어 둡니다.** 일반적으로 브라우저 세션은 에이전트의 각 응답 후 정리되지만, 창이 표시되는 모드에서는 턴별 정리를 건너뛰므로 에이전트가 작업하는 모습을 보고 수동으로 개입하고(로그인 문제, CAPTCHA) 대화 중 로그인 상태를 유지할 수 있습니다.

유휴 세션은 여전히 `browser.inactivity_timeout`(브라우저 활동이 없는 기본 120초) 후 회수되며 모든 세션은 종료 시 닫힙니다. 창이 표시되는 모드는 로컬 브라우저에만 영향을 주며 클라우드 세션에는 영향을 주지 않습니다.

## 스텔스 기능

Browserbase는 자동 스텔스 기능을 제공합니다.

| 기능 | 기본값 | 참고 |
|---------|---------|-------|
| 기본 스텔스 | 항상 켜짐 | 무작위 지문, 뷰포트 무작위화, CAPTCHA 해결 |
| 주거용 프록시 | 켜짐 | 더 나은 접근성을 위해 주거용 IP를 통해 라우팅 |
| 고급 스텔스 | 꺼짐 | 사용자 지정 Chromium 빌드, Scale Plan 필요 |
| 연결 유지 | 켜짐 | 네트워크 문제가 발생한 뒤 세션 재연결 |

:::note
요금제에서 유료 기능을 사용할 수 없으면 Hermes가 자동으로 대체합니다. 먼저 `keepAlive`를 비활성화하고 그 다음 프록시를 비활성화하므로 무료 요금제에서도 브라우징이 계속 작동합니다.
:::

## 세션 관리

- 각 작업은 Browserbase를 통해 격리된 브라우저 세션을 할당받습니다.
- 세션은 비활성 상태가 되면 자동으로 정리됩니다(기본값: 2분).
- 백그라운드 스레드는 30초마다 오래된 세션을 확인합니다.
- 고아 세션을 방지하기 위해 프로세스 종료 시 긴급 정리가 실행됩니다.
- 세션은 Browserbase API(`REQUEST_RELEASE` 상태)를 통해 해제됩니다.

## 제한 사항

- **텍스트 기반 상호작용** — 픽셀 좌표가 아니라 접근성 트리에 의존합니다.
- **스냅샷 크기** — 큰 페이지는 15,000자에서 잘리거나 LLM으로 요약될 수 있습니다(`web_extract`와 동일). 완전한 스냅샷은 `~/.hermes/cache/web/`에 저장되고 출력에서 `read_file` 페이징 경로를 안내합니다.
- **세션 시간 초과** — 클라우드 세션은 제공업체 요금제 설정에 따라 만료됩니다.
- **비용** — 클라우드 세션은 제공업체 크레딧을 사용합니다. 세션은 대화가 끝나거나 비활성 상태가 되면 자동으로 정리됩니다. 무료 로컬 브라우징에는 `/browser connect`를 사용하십시오.
- **파일 다운로드 불가** — 브라우저에서 파일을 다운로드할 수 없습니다.

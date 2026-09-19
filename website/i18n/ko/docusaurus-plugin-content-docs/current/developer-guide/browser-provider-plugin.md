---
sidebar_position: 13
title: "브라우저 제공자 플러그인"
description: "Hermes Agent용 클라우드 브라우저 백엔드 플러그인을 만드는 방법"
---

# 브라우저 제공자 플러그인 만들기

브라우저 제공자 플러그인은 클라우드 모드의 `browser_*` 도구 호출(탐색, 클릭, 스크린샷 등)을 처리하는 **클라우드 브라우저 백엔드**를 등록합니다. 기본 제공자 Browserbase, Browser Use, Firecrawl은 모두 `plugins/browser/<name>/` 아래 플러그인으로 제공됩니다. 새 제공자를 추가하거나 번들된 제공자를 재정의하려면 옆에 디렉터리를 추가하면 됩니다.

:::tip
브라우저 백엔드는 Hermes가 지원하는 여러 **백엔드 플러그인** 중 하나입니다. 나머지는 [웹 검색 제공자 플러그인](/developer-guide/web-search-provider-plugin)(이 ABC를 의도적으로 그대로 따릅니다), [이미지 생성](/developer-guide/image-gen-provider-plugin), [동영상 생성](/developer-guide/video-gen-provider-plugin), [메모리 제공자](/developer-guide/memory-provider-plugin), [컨텍스트 엔진](/developer-guide/context-engine-plugin), [시크릿 소스](/developer-guide/secret-source-plugin), [모델 제공자](/developer-guide/model-provider-plugin)입니다. 일반 도구/훅/CLI 플러그인은 [Hermes 플러그인 만들기](/developer-guide/plugins)에 설명되어 있습니다.
:::

## 전체 구조

브라우저 제공자는 브라우징을 구현하지 **않습니다**. 제공자가 구현하는 것은 **세션 수명 주기**입니다. 즉, 원격 브라우저 세션을 만들고 CDP 웹소켓 URL을 반환한 뒤 세션을 종료합니다. Hermes 자체의 브라우저 스택(`agent-browser` + `tools/browser_tool.py`)은 반환된 CDP URL에 연결해 그곳에서 페이지를 제어하므로, 모든 제공자는 `browser_*` 도구 세트를 추가 작업 없이 그대로 사용할 수 있습니다.

활성 제공자는 `browser.cloud_provider`를 `config.yaml`에서 설정해 선택합니다. `tools/browser_tool.py`의 디스패처는 제공자별 조건문이 없는 순수 레지스트리 조회입니다.

## 검색

Hermes는 다음 세 위치에서 브라우저 백엔드를 검색합니다.

1. **번들** — `<repo>/plugins/browser/<name>/` (`kind: backend`로 자동 로드)
2. **사용자** — `~/.hermes/plugins/browser/<name>/` (`plugins.enabled`를 통해 또는 `hermes plugins enable <name>`으로 선택)
3. **Pip** — `hermes_agent.plugins` 진입점을 선언하는 패키지

각 플러그인의 `register(ctx)`는 `ctx.register_browser_provider(...)`를 호출해 인스턴스를 `agent/browser_registry.py`의 레지스트리에 추가합니다.

## 디렉터리 구조

```
plugins/browser/my-backend/
├── __init__.py     # register() entry point
├── provider.py     # BrowserProvider subclass
└── plugin.yaml     # Manifest with kind: backend and provides_browser_providers
```

`plugin.yaml`:

```yaml
name: browser-my-backend
version: 1.0.0
description: "My cloud browser backend. Requires MY_BACKEND_API_KEY."
author: you
kind: backend
provides_browser_providers:
  - my-backend
```

`__init__.py`:

```python
from plugins.browser.my_backend.provider import MyBackendProvider


def register(ctx) -> None:
    ctx.register_browser_provider(MyBackendProvider())
```

## BrowserProvider ABC

`agent.browser_provider.BrowserProvider`를 구현합니다. 수명 주기 메서드 세 개와 식별 정보를 정의합니다.

```python
from agent.browser_provider import BrowserProvider


class MyBackendProvider(BrowserProvider):
    @property
    def name(self) -> str:
        return "my-backend"          # the browser.cloud_provider config value

    @property
    def display_name(self) -> str:
        return "My Backend"          # shown in `hermes tools`

    def is_available(self) -> bool:
        """Cheap check only — env var present, dep importable.
        NO network calls: runs at tool-registration time and on every
        `hermes tools` paint."""
        return bool(os.environ.get("MY_BACKEND_API_KEY"))

    def create_session(self, task_id: str) -> dict:
        """Create a remote browser session; return the session-metadata contract."""
        session = my_api.create_browser(...)
        return {
            "session_name": f"my-backend-{task_id}",  # unique agent-browser session name
            "bb_session_id": session.id,              # provider session ID (for cleanup)
            "cdp_url": session.cdp_ws_url,            # CDP websocket URL
            "features": {"stealth": True},            # feature flags you enabled
        }

    def close_session(self, session_id: str) -> bool:
        """Terminate by provider session ID. Log-and-return-False on error —
        never raise, so the dispatcher's cleanup loop keeps moving."""
        ...

    def emergency_cleanup(self, session_id: str) -> None:
        """Best-effort teardown from atexit/signal handlers. Must not raise."""
        ...
```

### 세션 메타데이터 계약

`create_session()`은 최소한 `session_name`, `bb_session_id`, `cdp_url`, `features`를 반환해야 합니다. 알아두어야 할 특징은 두 가지입니다.

- **`bb_session_id`는 레거시 키 이름**으로, `tools/browser_tool.py`와의 하위 호환성을 위해 그대로 유지됩니다. 이 값은 공급업체와 관계없이 **여러분의** 제공자 세션 ID를 담습니다. 이름을 바꾸지 마세요.
- `create_session()`은 **예외를 발생시킬 수 있습니다**. 자격 증명 누락에는 `ValueError`, 네트워크/API 오류에는 `RuntimeError`를 사용할 수 있습니다. 디스패처는 이 오류를 사용자에게 전달합니다. 반면 `close_session`/`emergency_cleanup`은 절대로 예외를 발생시키면 안 됩니다.

선택적 `external_call_id` 키는 관리형 게이트웨이 청구를 지원합니다.

### `get_setup_schema()` — `hermes tools` 선택기 행

API 키 프롬프트와 설치 훅이 있는 브라우저 자동화 선택기의 정식 옵션으로 표시되도록 이 메서드를 재정의합니다.

```python
def get_setup_schema(self) -> dict:
    return {
        "name": "My Backend",
        "badge": "paid",
        "tag": "Cloud browser with stealth and proxies",
        "env_vars": [
            {"key": "MY_BACKEND_API_KEY",
             "prompt": "My Backend API key",
             "url": "https://mybackend.example"},
        ],
        "post_setup": "agent_browser",   # ensures local Chromium is installed (agent-browser itself resolves via npx)
    }
```

도구 백엔드에 대한 프로젝트 표준에 따르면, 백엔드를 `hermes tools`를 통해 선택하고 구성할 수 없다면 완료된 것이 아닙니다. “이 환경 변수를 수동으로 설정하세요”는 통합이 아닙니다.

## 사용자가 구성하기

```yaml
browser:
  cloud_provider: my-backend
```

## 참조 구현

`plugins/browser/` 아래 번들된 세 제공자는 복잡도가 낮은 순서대로 정식 예제입니다. `firecrawl`(가장 단순함), `browser_use`, `browserbase`(유료 기능을 사용할 수 없을 때 우아하게 대체하는 스텔스/프록시/킵얼라이브 기능 플래그) 순입니다. 가장 비슷한 것을 복사하세요.

## 체크리스트

- [ ] `name`은 소문자이며 안정적이어야 합니다(사용자가 설정값으로 입력합니다).
- [ ] `is_available()`은 네트워크 호출을 전혀 하지 않습니다.
- [ ] `create_session()`은 전체 메타데이터 계약을 반환합니다(`bb_session_id` 키 이름 유지).
- [ ] `close_session()`/`emergency_cleanup()`은 절대로 예외를 발생시키지 않습니다.
- [ ] `get_setup_schema()`는 `hermes tools`가 백엔드를 구성할 수 있도록 환경 변수를 노출합니다.
- [ ] `plugin.yaml`은 `kind: backend`와 `provides_browser_providers`를 선언합니다.

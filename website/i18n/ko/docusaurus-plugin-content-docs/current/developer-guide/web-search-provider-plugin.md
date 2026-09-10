---
sidebar_position: 12
title: "웹 검색 프로바이더 플러그인"
description: "Hermes Agent용 웹 검색/추출/크롤 백엔드 플러그인 구축 방법"
---

# 웹 검색 프로바이더 플러그인 구축

웹 검색 프로바이더 플러그인은 `web_search`, `web_extract`, 그리고 (선택적으로) 심층 크롤 도구 호출을 처리하는 백엔드를 등록합니다. 기본 제공 프로바이더인 Firecrawl, SearXNG, Tavily, Exa, Parallel, Brave Search(무료 티어), xAI, DDGS는 모두 `plugins/web/<name>/` 아래 플러그인으로 제공됩니다. 이들과 나란히 디렉터리를 추가하여 새 프로바이더를 추가하거나 번들된 프로바이더를 재정의할 수 있습니다.

:::tip
웹 검색은 Hermes가 지원하는 여러 **백엔드 플러그인** 중 하나입니다. 나머지는 [이미지 생성 프로바이더 플러그인](/developer-guide/image-gen-provider-plugin), [비디오 생성 프로바이더 플러그인](/developer-guide/video-gen-provider-plugin), [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin), [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin), [모델 프로바이더 플러그인](/developer-guide/model-provider-plugin)입니다. 일반 도구/훅/CLI 플러그인은 [Hermes 플러그인 구축](/developer-guide/plugins)에 설명되어 있습니다.
:::

## 검색 방식

Hermes는 웹 검색 백엔드를 세 곳에서 검색합니다.

1. **번들** — `<repo>/plugins/web/<name>/`(`kind: backend`로 자동 로드되며 항상 사용 가능)
2. **사용자** — `~/.hermes/plugins/web/<name>/`(`plugins.enabled` 또는 `hermes plugins enable <name>`을 통해 선택적으로 활성화)
3. **Pip** — `hermes_agent.plugins` 진입점을 선언하는 패키지

각 플러그인의 `register(ctx)` 함수는 `ctx.register_web_search_provider(...)`를 호출하며, 이 호출은 인스턴스를 `agent/web_search_registry.py`의 레지스트리에 넣습니다. 각 기능의 활성 프로바이더는 설정으로 선택됩니다.

| 기능 | 설정 키 | 대체 값 |
|---|---|---|
| `web_search` | `web.search_backend` | `web.backend` |
| `web_extract` | `web.extract_backend` | `web.backend` |
| `web_extract` 내부의 심층 크롤 모드 | `web.extract_backend` | `web.backend` |

두 키가 모두 설정되지 않으면 Hermes는 환경에 있는 API 키/URL을 바탕으로 백엔드를 자동 감지합니다. `hermes tools`가 선택 과정을 안내합니다.

## 디렉터리 구조

```
plugins/web/my-backend/
├── __init__.py     # register() entry point
├── provider.py     # WebSearchProvider subclass
└── plugin.yaml     # Manifest with kind: backend and provides_web_providers
```

`brave_free/`와 `ddgs/`는 트리 내부의 가장 작은 참고 구현입니다. `brave_free`는 API 키로 게이트되는 검색 전용 프로바이더이고, `ddgs`는 SDK를 지연 설치하는 키 없는 프로바이더입니다.

## WebSearchProvider ABC

`agent.web_search_provider.WebSearchProvider`를 상속하세요. 필수 멤버는 `name`, `is_available()`, 그리고 구현할 `search()` / `extract()` 중 하나뿐입니다. (심층 크롤링은 별도 메서드가 아니라 `extract()`의 모드입니다.)

```python
# plugins/web/my-backend/provider.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider


class MyBackendWebSearchProvider(WebSearchProvider):
    """Minimal search-only provider against the My Backend HTTP API."""

    @property
    def name(self) -> str:
        # Stable id used in web.search_backend / web.extract_backend / web.backend
        # config keys. Lowercase, no spaces; hyphens permitted.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # Human label shown in `hermes tools`. Defaults to `name`.
        return "My Backend"

    def is_available(self) -> bool:
        # Cheap check — env var present, optional dep importable, etc.
        # MUST NOT make network calls (runs on every `hermes tools` paint).
        return bool(os.getenv("MY_BACKEND_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.environ["MY_BACKEND_API_KEY"]
        try:
            resp = httpx.get(
                "https://api.example.com/search",
                params={"q": query, "count": max(1, min(int(limit), 20))},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            return {"success": False, "error": str(exc)}

        # Response shape is fixed — see "Response shape" below.
        return {
            "success": True,
            "data": {
                "web": [
                    {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "description": item.get("snippet", ""),
                        "position": idx + 1,
                    }
                    for idx, item in enumerate(data.get("results", []))
                ],
            },
        }
```

```python
# plugins/web/my-backend/__init__.py
from plugins.web.my_backend.provider import MyBackendWebSearchProvider


def register(ctx) -> None:
    """Plugin entry point — called once at load time."""
    ctx.register_web_search_provider(MyBackendWebSearchProvider())
```

## plugin.yaml

```yaml
name: web-my-backend
version: 1.0.0
description: "My Backend web search — Bearer-auth REST API"
author: Your Name
kind: backend
provides_web_providers:
  - my-backend
requires_env:
  - MY_BACKEND_API_KEY
```

| 키 | 목적 |
|---|---|
| `kind: backend` | 플러그인을 백엔드 로딩 경로로 전달합니다 |
| `provides_web_providers` | 이 플러그인이 등록하는 프로바이더 `name` 목록입니다. `register()`가 실행되기 전에도 로더가 `hermes tools`에서 플러그인을 알릴 때 사용합니다 |
| `requires_env` | `hermes plugins install` 중 대화형 자격 증명 프롬프트입니다(자세한 형식은 [Hermes 플러그인 구축](/developer-guide/plugins#gate-on-environment-variables) 참조) |

## ABC 참조

전체 계약은 `agent/web_search_provider.py`에 있습니다. 재정의할 수 있는 메서드:

| 멤버 | 필수 여부 | 기본값 | 목적 |
|---|---|---|---|
| `name` | ✅ | — | `web.*_backend` 설정에서 사용하는 안정적인 ID |
| `display_name` | — | `name` | `hermes tools`에 표시되는 레이블 |
| `is_available()` | ✅ | — | 환경 변수와 선택적 의존성 등을 확인하는 저비용 가용성 게이트 |
| `supports_search()` | — | `True` | `web_search` 라우팅을 위한 기능 플래그 |
| `supports_extract()` | — | `False` | `web_extract` 라우팅을 위한 기능 플래그 |
| `search(query, limit)` | 조건부 | raises | `supports_search()`가 `True`를 반환할 때 필요 |
| `extract(urls, **kwargs)` | 조건부 | raises | `supports_extract()`가 `True`를 반환할 때 필요 |

프로바이더는 하나의 클래스에서 여러 기능을 알릴 수 있습니다. Firecrawl, Tavily, Exa, Parallel은 모두 검색과 추출을 구현합니다. Brave Search와 DDGS는 검색 전용이고, SearXNG는 검색 전용이며 문서화된 "추출 프로바이더와 짝지어 사용" 흐름을 제공합니다.

## 응답 형식

도구 래퍼는 백엔드 간 변환이 필요하지 않도록 고정된 봉투 형식을 기대합니다.

**검색 성공:**

```python
{
    "success": True,
    "data": {
        "web": [
            {"title": str, "url": str, "description": str, "position": int},
            ...
        ],
    },
}
```

**추출 성공:**

```python
{
    "success": True,
    "data": [
        {
            "url": str,
            "title": str,
            "content": str,
            "raw_content": str,
            "metadata": dict,    # optional
            "error": str,        # optional, only on per-URL failure
        },
        ...
    ],
}
```

**어느 기능이든 실패 시:**

```python
{"success": False, "error": "human-readable message"}
```

`search()`와 `extract()`는 모두 `async def`일 수 있습니다. 디스패처는 `inspect.iscoroutinefunction`으로 코루틴 함수를 감지하고 그에 맞게 await합니다. 블로킹 I/O(HTTP, SDK 호출)를 수행하는 동기 구현도 작은 백엔드에서는 괜찮으며, 디스패처가 스레딩을 처리합니다.

## 기능 플래그

Hermes는 `supports_*` 플래그를 기준으로 호출을 올바른 프로바이더에 라우팅합니다. 일반적인 멀티 프로바이더 설정은 다음과 같습니다.

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "brave-free"     # search-only, fast, free 2k/mo
  extract_backend: "firecrawl"     # extract + crawl, paid quota
```

`web.search_backend` 또는 `web.extract_backend`가 설정되지 않으면 둘 다 `web.backend`로 대체됩니다. 이것도 설정되지 않으면 Hermes는 환경 변수 존재 여부를 기준으로 요청된 기능을 지원하는 첫 번째 가용 프로바이더를 선택합니다.

프로바이더가 한 기능만 지원한다면 다른 플래그는 기본값(`False`)으로 두세요. 그러면 레지스트리가 해당 도구에서 이를 건너뛰므로, 사용자가 검색에만 X를 사용하면서 에이전트에 추출을 요청했을 때 오해를 부르는 "프로바이더 X 실패" 오류가 표시되지 않습니다.

## Hermes가 도구에 연결하는 방식

`web_search` 및 `web_extract` 도구는 `tools/web_tools.py`에 있습니다. 호출 시 다음을 수행합니다.

1. 관련 설정 키를 읽습니다(`web_search`에는 `web.search_backend`, `web_extract`에는 `web.extract_backend`).
2. 레지스트리에 해당 `name`의 프로바이더를 요청합니다.
3. `is_available()` 및 일치하는 `supports_*()` 플래그를 확인합니다.
4. `search()` / `extract()`로 디스패치합니다(심층 크롤은 `extract()` 내부의 모드로 실행). 메서드가 코루틴이면 await합니다.
5. 응답 봉투를 JSON으로 직렬화하여 LLM에 반환합니다.

오류는 도구 결과로 노출되며, 이를 어떻게 설명할지는 LLM이 결정합니다. 프로바이더가 등록되지 않았거나 가용한 모든 프로바이더가 기능 게이트를 통과하지 못하면 도구는 `hermes tools`를 안내하는 유용한 오류를 반환합니다.

## 선택적 의존성 지연 설치

프로바이더가 DDGS의 `ddgs` 패키지처럼 서드파티 SDK를 감싼다면 모듈 최상위에서 `import`하지 마세요. `is_available()` 또는 `search()` 내부에서 `tools.lazy_deps.ensure(...)`를 사용하세요. Hermes는 `security.allow_lazy_installs`로 보호되는 첫 사용 시 패키지를 설치합니다. 보안 모델은 [Hermes 플러그인 구축 → 지연 설치](/developer-guide/plugins#lazy-install-optional-python-dependencies)를 참조하세요.

## 참고 구현

- **`plugins/web/brave_free/`** — 작고 API 키로 게이트되는 검색 전용 HTTP 프로바이더입니다. 좋은 시작 템플릿입니다.
- **`plugins/web/ddgs/`** — SDK를 지연 설치하는 키 없는 프로바이더입니다. Python 패키지를 감싸는 백엔드에 유용한 패턴입니다.
- **`plugins/web/firecrawl/`** — 여러 형식 모드를 지원하는 완전한 다기능 프로바이더(검색 + 추출 + 크롤)입니다.
- **`plugins/web/searxng/`** — 인증 없이 URL로 설정하는 자체 호스팅 백엔드입니다.
- **`plugins/web/xai/`** — Grok의 서버 측 `web_search` 도구를 통한 LLM 기반 검색입니다. 새 환경 변수를 추가하지 않고 기존 OAuth/환경 변수 자격 증명 표면(`tools/xai_http.py`)을 재사용하는 방법과 네트워크를 사용하지 않는 계약을 준수하는 저비용 `is_available()` 작성 방법을 보여줍니다.

## pip으로 배포

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-web = "my_backend_web_package"
```

`my_backend_web_package`는 최상위 `register` 함수를 노출해야 합니다. 전체 설정은 일반 플러그인 가이드의 [pip으로 배포](/developer-guide/plugins#distribute-via-pip)를 참조하세요.

## 관련 페이지

- [웹 검색](/user-guide/features/web-search) — 사용자 대상 기능 문서 및 백엔드별 설정
- [플러그인 개요](/user-guide/features/plugins) — 모든 플러그인 유형을 한눈에 보기
- [Hermes 플러그인 구축](/developer-guide/plugins) — 일반 도구/훅/슬래시 명령 가이드


---
sidebar_position: 12
title: "Web Search Provider Plugins"
description: "Hermes 에이전트용 웹 검색/추출/크롤링 백엔드 플러그인을 빌드하는 방법"
---

# 웹 검색 프로바이더 플러그인 빌드하기

웹 검색 프로바이더 플러그인은 `web_search`, `web_extract` 및 (선택적으로) 심층 크롤링(deep-crawl) 도구 호출을 서비스하는 백엔드를 등록합니다. 내장 프로바이더 — Firecrawl, SearXNG, Tavily, Exa, Parallel, Brave Search(무료 티어), xAI 및 DDGS — 모두 `plugins/web/<name>/` 하위의 플러그인으로 제공됩니다. 같은 위치에 디렉토리를 놓아 새 프로바이더를 추가하거나 내장된 프로바이더를 덮어쓸 수 있습니다.

:::tip
웹 검색은 Hermes가 지원하는 여러 **백엔드 플러그인** 중 하나입니다. 자체 ABC(추상 기본 클래스)를 갖춘 다른 플러그인으로는 [이미지 생성 프로바이더 플러그인](/developer-guide/image-gen-provider-plugin), [비디오 생성 프로바이더 플러그인](/developer-guide/video-gen-provider-plugin), [메모리 프로바이더 플러그인](/developer-guide/memory-provider-plugin), [컨텍스트 엔진 플러그인](/developer-guide/context-engine-plugin) 및 [모델 프로바이더 플러그인](/developer-guide/model-provider-plugin)이 있습니다. 일반 도구/훅/CLI 플러그인은 [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin)를 참고하세요.
:::

## 검색(Discovery) 작동 방식

Hermes는 세 곳에서 웹 검색 백엔드를 스캔합니다:

1. **번들(Bundled)** — `<repo>/plugins/web/<name>/` (`kind: backend`와 함께 자동 로드되며 항상 사용 가능)
2. **사용자(User)** — `~/.hermes/plugins/web/<name>/` (`plugins.enabled` 또는 `hermes plugins enable <name>`을 통해 선택적으로 사용)
3. **Pip** — `hermes_agent.plugins` 진입점(entry point)을 선언하는 패키지

각 플러그인의 `register(ctx)` 함수는 `ctx.register_web_search_provider(...)`를 호출하여 인스턴스를 `agent/web_search_registry.py`의 레지스트리에 넣습니다. 각 기능의 활성 프로바이더는 구성을 통해 선택됩니다:

| 기능 | 구성 키 | 폴백(Fall back) |
|---|---|---|
| `web_search` | `web.search_backend` | `web.backend` |
| `web_extract` | `web.extract_backend` | `web.backend` |
| `web_extract` 내의 심층 크롤링 모드 | `web.extract_backend` | `web.backend` |

두 키 모두 설정되지 않은 경우 Hermes는 환경(environment)에 존재하는 API 키나 URL을 기준으로 백엔드를 자동 감지합니다. `hermes tools`를 통해 사용자가 선택할 수 있도록 안내합니다.

## 디렉토리 구조

```
plugins/web/my-backend/
├── __init__.py     # register() 진입점
├── provider.py     # WebSearchProvider 하위 클래스
└── plugin.yaml     # kind: backend 및 provides_web_providers가 포함된 매니페스트
```

트리 내부의 참고 자료로는 `brave_free/`와 `ddgs/`가 가장 작습니다 — `brave_free`는 API 키가 제한된 검색 전용 프로바이더를 위해, `ddgs`는 SDK를 지연 설치(lazy-install)하는 키가 없는 프로바이더를 위함입니다.

## WebSearchProvider ABC

`agent.web_search_provider.WebSearchProvider`의 하위 클래스를 만듭니다. 유일한 필수 멤버는 `name`, `is_available()`, 그리고 `search()` / `extract()` / `crawl()` 중 여러분이 구현한 항목입니다.

```python
# plugins/web/my-backend/provider.py
from __future__ import annotations

import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider


class MyBackendWebSearchProvider(WebSearchProvider):
    """My Backend HTTP API에 대한 최소한의 검색 전용 프로바이더입니다."""

    @property
    def name(self) -> str:
        # web.search_backend / web.extract_backend / web.backend
        # 구성 키에서 사용되는 고정 ID. 소문자, 공백 없음; 하이픈 허용.
        return "my-backend"

    @property
    def display_name(self) -> str:
        # `hermes tools`에 표시되는 사람이 읽을 수 있는 레이블. 기본값은 `name`입니다.
        return "My Backend"

    def is_available(self) -> bool:
        # 저렴한 비용의 검사 — 환경 변수 존재 여부, 선택적 종속성 가져오기 가능 여부 등.
        # 절대로 네트워크 호출을 해서는 안 됩니다 (모든 `hermes tools` 실행 때마다 호출됨).
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

        # 응답 형태가 고정되어 있습니다 — 아래 "응답 형태"를 참조하세요.
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
    """플러그인 진입점 — 로드 시 한 번 호출됩니다."""
    ctx.register_web_search_provider(MyBackendWebSearchProvider())
```

## plugin.yaml

```yaml
name: web-my-backend
version: 1.0.0
description: "My Backend 웹 검색 — Bearer-auth REST API"
author: Your Name
kind: backend
provides_web_providers:
  - my-backend
requires_env:
  - MY_BACKEND_API_KEY
```

| 키 | 목적 |
|---|---|
| `kind: backend` | 플러그인을 백엔드 로드 경로로 라우팅합니다 |
| `provides_web_providers` | 이 플러그인이 등록하는 프로바이더 `name` 목록 — `register()`가 실행되기 전이라도 로더가 `hermes tools`에서 이 플러그인을 광고할 때 사용됩니다. |
| `requires_env` | `hermes plugins install` 중의 대화형 자격 증명 프롬프트 (다양한 형식을 위해 [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin#gate-on-environment-variables) 참조) |

## ABC 참조

전체 계약(contract)은 `agent/web_search_provider.py`에 있습니다. 재정의(override)할 수 있는 메서드:

| 멤버 | 필수 | 기본값 | 목적 |
|---|---|---|---|
| `name` | ✅ | — | `web.*_backend` 설정에서 사용되는 고정 ID |
| `display_name` | — | `name` | `hermes tools`에 표시되는 레이블 |
| `is_available()` | ✅ | — | 저렴한 가용성 게이트 — 환경 변수, 선택적 종속성 |
| `supports_search()` | — | `True` | `web_search` 라우팅을 위한 기능 플래그 |
| `supports_extract()` | — | `False` | `web_extract` 라우팅을 위한 기능 플래그 |
| `search(query, limit)` | 조건부 | 발생(raises) | `supports_search()`가 `True`를 반환할 때 필수 |
| `extract(urls, **kwargs)` | 조건부 | 발생(raises) | `supports_extract()`가 `True`를 반환할 때 필수 |

프로바이더는 단일 클래스에서 여러 기능을 광고할 수 있습니다. Firecrawl, Tavily, Exa, 그리고 Parallel은 모두 검색 및 추출을 구현합니다. Brave Search와 DDGS는 검색 전용입니다; SearXNG는 문서화된 "추출 프로바이더와 페어링" 워크플로우를 갖춘 검색 전용입니다.

## 응답 형태 (Response shape)

도구 래퍼는 백엔드 간에 번역할 필요가 없도록 고정된 봉투(envelope)를 예상합니다.

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
            "metadata": dict,    # 선택 사항
            "error": str,        # 선택 사항, URL별 실패 시에만
        },
        ...
    ],
}
```

**둘 중 하나의 기능, 실패 시:**

```python
{"success": False, "error": "사람이 읽을 수 있는 메시지"}
```

`search()`와 `extract()` 둘 다 `async def`일 수 있습니다. 디스패처는 `inspect.iscoroutinefunction`을 통해 코루틴 함수를 감지하고 그에 따라 `await`합니다. I/O(HTTP, SDK 호출)를 차단하는 동기식(Sync) 구현은 작은 백엔드의 경우 괜찮습니다. 디스패처가 스레딩을 처리합니다.

## 기능 플래그 (Capability flags)

Hermes는 `supports_*` 플래그를 기반으로 올바른 프로바이더로 호출을 라우팅합니다. 일반적인 다중 프로바이더 설정은 다음과 같습니다:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "brave-free"     # 검색 전용, 빠름, 무료 2k/월
  extract_backend: "firecrawl"     # 추출 + 크롤링, 유료 할당량
```

`web.search_backend` 또는 `web.extract_backend`가 설정되지 않으면 둘 다 `web.backend`로 떨어집니다(fall through). 이 역시 설정되지 않으면 Hermes는 환경 변수 존재 여부를 기반으로 요청된 기능을 지원하는 첫 번째 사용 가능한 프로바이더를 선택합니다.

프로바이더가 단 하나의 기능만 지원한다면, 다른 플래그는 기본값(`False`)으로 두고 레지스트리가 해당 도구에 대해 이 프로바이더를 건너뛰게 하세요. 그러면 사용자가 X를 검색 전용으로 사용할 때 에이전트에게 추출을 요청하는 과정에서 "프로바이더 X가 실패했습니다"라는 오해를 부르는 오류를 보지 않게 될 것입니다.

## Hermes가 도구에 연결하는 방식

`web_search` 및 `web_extract` 도구는 `tools/web_tools.py`에 있습니다. 호출 시 다음과 같이 동작합니다:

1. 관련된 구성 키를 읽습니다 (`web_search`의 경우 `web.search_backend`, `web_extract`의 경우 `web.extract_backend`)
2. 레지스트리에 해당 `name`을 가진 프로바이더를 요청합니다
3. `is_available()` 및 일치하는 `supports_*()` 플래그를 확인합니다
4. `search()` / `extract()` / `crawl()`로 디스패치하며 메서드가 코루틴인 경우 `await`합니다
5. 응답 봉투(envelope)를 JSON 직렬화하고 LLM에 다시 넘겨줍니다

오류는 도구 결과로 표출되며, 사용자에 대한 설명 방식은 LLM이 결정합니다. 등록된 프로바이더가 없거나(또는 사용 가능한 모든 프로바이더가 기능 게이트를 통과하지 못한 경우), 도구는 `hermes tools`를 가리키는 유용한 오류를 반환합니다.

## 선택적 종속성의 지연 설치 (Lazy-installing)

프로바이더가 (DDGS가 `ddgs` 패키지에 대해 수행하는 것처럼) 타사 SDK를 래핑하는 경우, 최상위 모듈에서 패키지를 `import`하지 마세요. 대신 `is_available()` 또는 `search()` 내부에서 `tools.lazy_deps.ensure(...)`를 사용하세요 — Hermes는 첫 사용 시 `security.allow_lazy_installs`의 제어 하에 패키지를 설치할 것입니다. 보안 모델에 대해서는 [Hermes 플러그인 빌드하기 → 지연 설치](/guides/build-a-hermes-plugin#lazy-install-optional-python-dependencies)를 참고하세요.

## 참조 구현

- **`plugins/web/brave_free/`** — 작고, API 키가 제한된 검색 전용 HTTP 프로바이더. 좋은 시작 템플릿입니다.
- **`plugins/web/ddgs/`** — SDK를 지연 설치하는 키 없는 프로바이더. Python 패키지를 래핑하는 백엔드에 유용한 패턴입니다.
- **`plugins/web/firecrawl/`** — 여러 형식의 모드를 지원하는 완전한 다중 기능(검색 + 추출 + 크롤링) 프로바이더입니다.
- **`plugins/web/searxng/`** — 인증 없이 URL로 구성된 자체 호스팅 백엔드입니다.
- **`plugins/web/xai/`** — Grok 서버 측의 `web_search` 도구를 통한 LLM 기반 검색입니다. 새로운 환경 변수를 추가하지 않고 기존의 OAuth/환경 변수 자격 증명 환경(`tools/xai_http.py`)을 재사용하는 방법과 네트워크 연결이 없는(no-network) 계약을 준수하는 저렴한 `is_available()` 작성 방법을 보여줍니다.

## pip를 통한 배포

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-web = "my_backend_web_package"
```

`my_backend_web_package`는 최상위 수준의 `register` 함수를 노출해야 합니다. 전체 설정은 일반 플러그인 가이드의 [pip를 통한 배포](/guides/build-a-hermes-plugin#distribute-via-pip)를 참조하세요.

## 관련 문서

- [웹 검색](/user-guide/features/web-search) — 사용자 지향 기능 문서 및 백엔드별 구성
- [플러그인 개요](/user-guide/features/plugins) — 모든 플러그인 유형 한눈에 보기
- [Hermes 플러그인 빌드하기](/guides/build-a-hermes-plugin) — 일반 도구/훅/슬래시 명령어 가이드

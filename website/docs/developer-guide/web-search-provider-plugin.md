---
sidebar_position: 12
title: "Web Search Provider Plugins"
description: "How to build a web-search/extract/crawl backend plugin for Hermes Agent"
---

# Building a Web Search Provider Plugin

Web-search provider plugins register a backend that services `web_search`, `web_extract`, and (optionally) deep-crawl tool calls. Built-in providers — Firecrawl, SearXNG, Tavily, Exa, Parallel, Brave Search (free tier), and DDGS — all ship as plugins under `plugins/web/<name>/`. You can add a new one, or override a bundled one, by dropping a directory next to them.

:::tip
Web search is one of several **backend plugins** Hermes supports. The others (with their own ABCs) are [Image Generation Provider Plugins](/docs/developer-guide/image-gen-provider-plugin), [Video Generation Provider Plugins](/docs/developer-guide/video-gen-provider-plugin), [Memory Provider Plugins](/docs/developer-guide/memory-provider-plugin), [Context Engine Plugins](/docs/developer-guide/context-engine-plugin), and [Model Provider Plugins](/docs/developer-guide/model-provider-plugin). General tool/hook/CLI plugins live in [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin).
:::

## How discovery works

Hermes scans for web-search backends in three places:

1. **Bundled** — `<repo>/plugins/web/<name>/` (auto-loaded with `kind: backend`, always available)
2. **User** — `~/.hermes/plugins/web/<name>/` (opt-in via `plugins.enabled` or `hermes plugins enable <name>`)
3. **Pip** — packages declaring a `hermes_agent.plugins` entry point

Each plugin's `register(ctx)` function calls `ctx.register_web_search_provider(...)` — that puts the instance into the registry in `agent/web_search_registry.py`. The active provider for each capability is picked by config:

| Capability | Config key | Falls back to |
|---|---|---|
| `web_search` | `web.search_backend` | `web.backend` |
| `web_extract` | `web.extract_backend` | `web.backend` |
| Deep crawl modes inside `web_extract` | `web.extract_backend` | `web.backend` |

When neither key is set, Hermes auto-detects the backend from whichever API key/URL is present in the environment. `hermes tools` walks users through selection.

## Directory structure

```
plugins/web/my-backend/
├── __init__.py     # register() entry point
├── provider.py     # WebSearchProvider subclass
└── plugin.yaml     # Manifest with kind: backend and provides_web_providers
```

`brave_free/` and `ddgs/` are the smallest in-tree references — `brave_free` for an API-key-gated search-only provider, `ddgs` for a no-key provider that lazy-installs its SDK.

## The WebSearchProvider ABC

Subclass `agent.web_search_provider.WebSearchProvider`. The only required members are `name`, `is_available()`, and whichever of `search()` / `extract()` / `crawl()` you implement.

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

    def supports_crawl(self) -> bool:
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

| Key | Purpose |
|---|---|
| `kind: backend` | Routes the plugin through the backend-loading path |
| `provides_web_providers` | List of provider `name`s this plugin registers — used by the loader to advertise the plugin in `hermes tools` even before `register()` runs |
| `requires_env` | Interactive credential prompt during `hermes plugins install` (see [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin#gate-on-environment-variables) for the rich format) |

## ABC reference

Full contract in `agent/web_search_provider.py`. Methods you may override:

| Member | Required | Default | Purpose |
|---|---|---|---|
| `name` | ✅ | — | Stable id used in `web.*_backend` config |
| `display_name` | — | `name` | Label shown in `hermes tools` |
| `is_available()` | ✅ | — | Cheap availability gate — env vars, optional deps |
| `supports_search()` | — | `True` | Capability flag for `web_search` routing |
| `supports_extract()` | — | `False` | Capability flag for `web_extract` routing |
| `supports_crawl()` | — | `False` | Capability flag for deep-crawl modes |
| `search(query, limit)` | conditional | raises | Required when `supports_search()` returns `True` |
| `extract(urls, **kwargs)` | conditional | raises | Required when `supports_extract()` returns `True` |
| `crawl(url, **kwargs)` | conditional | raises | Required when `supports_crawl()` returns `True` |

Providers can advertise multiple capabilities from a single class — Firecrawl, Tavily, Exa, and Parallel all implement all three of search/extract/crawl. Brave Search and DDGS are search-only; SearXNG is search-only with a documented "pair me with an extract provider" workflow.

## Response shape

The tool wrapper expects a fixed envelope so it doesn't have to translate between backends.

**Search success:**

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

**Extract success:**

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

**Either capability, on failure:**

```python
{"success": False, "error": "human-readable message"}
```

Both `search()` and `extract()` may be `async def` — the dispatcher detects coroutine functions via `inspect.iscoroutinefunction` and awaits accordingly. Sync implementations that do blocking I/O (HTTP, SDK calls) are fine for small backends; the dispatcher handles threading.

## Capability flags

Hermes routes calls to the right provider based on the `supports_*` flags. A common multi-provider setup:

```yaml
# ~/.hermes/config.yaml
web:
  search_backend: "brave-free"     # search-only, fast, free 2k/mo
  extract_backend: "firecrawl"     # extract + crawl, paid quota
```

When `web.search_backend` or `web.extract_backend` aren't set, both fall through to `web.backend`. When that's also unset, Hermes picks the first available provider that supports the requested capability based on env-var presence.

If your provider only supports one capability, leave the other flags at their default (`False`) and the registry will skip it for that tool — users won't see misleading "provider X failed" errors when they're using X only for search and asking the agent to extract.

## How Hermes wires it into the tools

The `web_search` and `web_extract` tools live in `tools/web_tools.py`. At call time they:

1. Read the relevant config key (`web.search_backend` for `web_search`, `web.extract_backend` for `web_extract`)
2. Ask the registry for the provider with that `name`
3. Check `is_available()` and the matching `supports_*()` flag
4. Dispatch to `search()` / `extract()` / `crawl()`, awaiting if the method is a coroutine
5. JSON-serialize the response envelope and hand it back to the LLM

Errors surface as the tool result; the LLM decides how to explain them. If no provider is registered (or every available one fails the capability gate), the tool returns a helpful error pointing at `hermes tools`.

## Lazy-installing optional dependencies

If your provider wraps a third-party SDK (like DDGS does with the `ddgs` package), don't `import` it at module top level. Use `tools.lazy_deps.ensure(...)` inside `is_available()` or `search()` — Hermes will install the package on first use, gated by `security.allow_lazy_installs`. See [Build a Hermes Plugin → Lazy-install](/docs/guides/build-a-hermes-plugin#lazy-install-optional-python-dependencies) for the security model.

## Reference implementations

- **`plugins/web/brave_free/`** — small, API-key-gated, search-only HTTP provider. Good starting template.
- **`plugins/web/ddgs/`** — no-key provider that lazy-installs its SDK. Useful pattern for backends that wrap a Python package.
- **`plugins/web/firecrawl/`** — full multi-capability provider (search + extract + crawl) with multiple format modes.
- **`plugins/web/searxng/`** — self-hosted, URL-configured backend with no auth.

## Distribute via pip

```toml
# pyproject.toml
[project.entry-points."hermes_agent.plugins"]
my-backend-web = "my_backend_web_package"
```

`my_backend_web_package` must expose a top-level `register` function. See [Distribute via pip](/docs/guides/build-a-hermes-plugin#distribute-via-pip) in the general plugin guide for the full setup.

## Related pages

- [Web Search](/docs/user-guide/features/web-search) — user-facing feature documentation and per-backend configuration
- [Plugins overview](/docs/user-guide/features/plugins) — all plugin types at a glance
- [Build a Hermes Plugin](/docs/guides/build-a-hermes-plugin) — general tools/hooks/slash commands guide

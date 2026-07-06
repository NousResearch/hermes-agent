---
sidebar_position: 13
title: "Browser Provider Plugins"
description: "How to build a cloud browser backend plugin for Hermes Agent"
---

# Building a Browser Provider Plugin

Browser provider plugins register a **cloud browser backend** that services cloud-mode `browser_*` tool calls (navigate, click, screenshot, …). Built-in providers — Browserbase, Browser Use, and Firecrawl — all ship as plugins under `plugins/browser/<name>/`. You can add a new one, or override a bundled one, by dropping a directory next to them.

:::tip
Browser backends are one of several **backend plugins** Hermes supports. The others (with their own ABCs) are [Web Search Provider Plugins](/developer-guide/web-search-provider-plugin) (which this ABC deliberately mirrors), [Image Generation](/developer-guide/image-gen-provider-plugin), [Video Generation](/developer-guide/video-gen-provider-plugin), [Memory Providers](/developer-guide/memory-provider-plugin), [Context Engines](/developer-guide/context-engine-plugin), [Secret Sources](/developer-guide/secret-source-plugin), and [Model Providers](/developer-guide/model-provider-plugin). General tool/hook/CLI plugins live in [Build a Hermes Plugin](/developer-guide/plugins).
:::

## How it fits together

A browser provider does **not** implement browsing. It implements **session lifecycle**: create a remote browser session, hand back a CDP websocket URL, and tear the session down. Hermes' own browser stack (`agent-browser` + `tools/browser_tool.py`) connects to whatever CDP URL you return and drives the page from there — every provider gets the full `browser_*` toolset for free.

The active provider is selected by `browser.cloud_provider` in `config.yaml`; the dispatcher in `tools/browser_tool.py` is a pure registry lookup with no per-provider conditionals.

## Discovery

Hermes scans for browser backends in three places:

1. **Bundled** — `<repo>/plugins/browser/<name>/` (auto-loaded with `kind: backend`)
2. **User** — `~/.hermes/plugins/browser/<name>/` (opt-in via `plugins.enabled` or `hermes plugins enable <name>`)
3. **Pip** — packages declaring a `hermes_agent.plugins` entry point

Each plugin's `register(ctx)` calls `ctx.register_browser_provider(...)`, which puts the instance into the registry in `agent/browser_registry.py`.

## Directory structure

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

## The BrowserProvider ABC

Implement `agent.browser_provider.BrowserProvider`. Three lifecycle methods plus identity:

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

### The session-metadata contract

`create_session()` must return at least `session_name`, `bb_session_id`, `cdp_url`, and `features`. Two quirks worth knowing:

- **`bb_session_id` is a legacy key name** kept verbatim for backward compatibility with `tools/browser_tool.py` — it holds *your* provider's session ID regardless of vendor. Don't rename it.
- `create_session()` **may raise** — `ValueError` for missing credentials, `RuntimeError` for network/API failures. The dispatcher surfaces these to the user. This differs from `close_session`/`emergency_cleanup`, which must never raise.

An optional `external_call_id` key supports managed-gateway billing.

### `get_setup_schema()` — the `hermes tools` picker row

Override this to appear as a first-class option in the Browser Automation picker with API-key prompts and an install hook:

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
        "post_setup": "agent_browser",   # auto-installs the agent-browser npm dep
    }
```

Per the project standard for tool backends: if a backend can't be selected and configured through `hermes tools`, it isn't done — "set this env var manually" is not an integration.

## Users configure it

```yaml
browser:
  cloud_provider: my-backend
```

## Reference implementations

The three bundled providers under `plugins/browser/` are the canonical examples, in ascending complexity: `firecrawl` (simplest), `browser_use`, and `browserbase` (stealth/proxy/keep-alive feature flags with graceful fallback when paid features are unavailable). Copy the closest one.

## Checklist

- [ ] `name` is lowercase and stable (it's a config value users write)
- [ ] `is_available()` makes zero network calls
- [ ] `create_session()` returns the full metadata contract (`bb_session_id` key name intact)
- [ ] `close_session()` / `emergency_cleanup()` never raise
- [ ] `get_setup_schema()` exposes your env vars so `hermes tools` can configure the backend
- [ ] `plugin.yaml` declares `kind: backend` + `provides_browser_providers`

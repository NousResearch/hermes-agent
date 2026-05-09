# Web Tools Provider Architecture

## Overview

Web tools (`web_search`, `web_extract`) use a **per-capability backend selection** system that allows different providers for search and extract independently.

## Config Keys

```yaml
web:
  backend: "firecrawl"       # Shared fallback — applies to both if specific keys not set
  search_backend: ""         # Per-capability override for web_search
  extract_backend: ""        # Per-capability override for web_extract
```

**Selection priority (per capability):**
1. `web.search_backend` / `web.extract_backend` (explicit per-capability)
2. `web.backend` (shared fallback)
3. Auto-detect from environment variables

When per-capability keys are empty (default), behavior is identical to the legacy single-backend selection.

## Architecture

```
web_search_tool()
    └─ _get_search_backend()
         ├─ web.search_backend (if set + available)
         └─ _get_backend() fallback

web_extract_tool()
    └─ _get_extract_backend()
         ├─ web.extract_backend (if set + available)
         └─ _get_backend() fallback
```

## Provider ABCs

New providers implement these interfaces in `tools/web_providers/`:

```python
from tools.web_providers.base import WebSearchProvider, WebExtractProvider

class MySearchProvider(WebSearchProvider):
    def provider_name(self) -> str: ...
    def is_configured(self) -> bool: ...
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]: ...

class MyExtractProvider(WebExtractProvider):
    def provider_name(self) -> str: ...
    def is_configured(self) -> bool: ...
    def extract(self, urls: List[str], **kwargs) -> Dict[str, Any]: ...
```

## Adding a New Search Provider

1. Create `tools/web_providers/your_provider.py` implementing `WebSearchProvider`
2. Add availability check to `_is_backend_available()` in `web_tools.py`
3. Add dispatch branch in `web_search_tool()` 
4. Add provider to `hermes tools` picker in `tools_config.py`
5. Add env var to `OPTIONAL_ENV_VARS` in `config.py` (if needed)
6. Write tests in `tests/tools/`

Search-only providers (like SearXNG) don't need to implement `WebExtractProvider`.
Extract-only providers don't need to implement `WebSearchProvider`.

## hermes tools UX

The provider picker uses **progressive disclosure**:
- **Default path** (90% of users): Pick one provider → sets `web.backend` for both. One selection, done.
- **Advanced path**: "Configure separately" option at bottom → two-step sub-picker for search + extract independently.

See `.hermes/plans/2026-05-03-web-tools-provider-architecture.md` for the full UX flow diagram.

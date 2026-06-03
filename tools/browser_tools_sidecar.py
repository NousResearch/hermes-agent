"""Shared browser/scraping sidecar tools.

This module exposes a small Hermes tool wrapper around an internal
``browser-tools`` HTTP sidecar. The sidecar can combine multiple runtimes such
as CloakBrowser for stealth/visual rendering and Scrapling for structured
fetching, extraction, and crawling. Keeping those heavy browser dependencies in
one service lets many agents share them without installing Playwright/Patchright
inside every Hermes container.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from tools.registry import registry


DEFAULT_BROWSER_TOOLS_URL = "http://browser-tools:8790"
MAX_TIMEOUT_MS = 120_000
MAX_TEXT_LIMIT = 50_000


def _base_url() -> str:
    return os.getenv("BROWSER_TOOLS_URL", "").rstrip("/")


def check_browser_tools_sidecar_requirements() -> bool:
    """Expose tools only when the caller configured a sidecar URL."""
    return bool(_base_url())


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _post_json(path: str, payload: dict[str, Any], timeout_ms: int) -> str:
    base = _base_url()
    if not base:
        return json.dumps({
            "success": False,
            "error": "BROWSER_TOOLS_URL is not configured",
            "hint": f"Set BROWSER_TOOLS_URL={DEFAULT_BROWSER_TOOLS_URL} in the agent/container environment.",
        })

    timeout_ms = _clamp_int(timeout_ms, 30_000, 1_000, MAX_TIMEOUT_MS)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=(timeout_ms / 1000.0) + 5) as resp:
            raw = resp.read(MAX_TEXT_LIMIT + 100_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read(20_000).decode("utf-8", errors="replace")
        return json.dumps({"success": False, "error": f"sidecar HTTP {e.code}", "detail": detail})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "error_type": type(e).__name__})

    try:
        data = json.loads(raw)
    except Exception:
        return json.dumps({"success": False, "error": "invalid JSON from browser-tools sidecar", "body": raw[:20_000]})
    return json.dumps(data, ensure_ascii=False)


def browser_tools_fetch(
    url: str,
    mode: str = "auto",
    timeout_ms: int = 30_000,
    text_limit: int = 12_000,
    screenshot: bool = False,
    humanize: bool = False,
    wait_until: str = "domcontentloaded",
    task_id: str | None = None,
) -> str:
    """Fetch a webpage through the shared browser-tools sidecar."""
    if not url:
        return json.dumps({"success": False, "error": "url is required"})
    timeout_ms = _clamp_int(timeout_ms, 30_000, 1_000, MAX_TIMEOUT_MS)
    text_limit = _clamp_int(text_limit, 12_000, 0, MAX_TEXT_LIMIT)
    payload = {
        "url": url,
        "mode": mode or "auto",
        "timeout_ms": timeout_ms,
        "text_limit": text_limit,
        "screenshot": bool(screenshot),
        "humanize": bool(humanize),
        "wait_until": wait_until or "domcontentloaded",
    }
    return _post_json("/fetch", payload, timeout_ms)


def browser_tools_extract(
    url: str,
    selectors: dict[str, str] | None = None,
    mode: str = "scrapling_http",
    timeout_ms: int = 30_000,
    text_limit: int = 12_000,
    include_links: bool = True,
    task_id: str | None = None,
) -> str:
    """Run structured selector extraction through Scrapling in the sidecar."""
    if not url:
        return json.dumps({"success": False, "error": "url is required"})
    timeout_ms = _clamp_int(timeout_ms, 30_000, 1_000, MAX_TIMEOUT_MS)
    text_limit = _clamp_int(text_limit, 12_000, 0, MAX_TEXT_LIMIT)
    payload = {
        "url": url,
        "mode": mode or "scrapling_http",
        "selectors": selectors or {},
        "timeout_ms": timeout_ms,
        "text_limit": text_limit,
        "include_links": bool(include_links),
    }
    return _post_json("/extract", payload, timeout_ms)


registry.register(
    name="browser_tools_fetch",
    toolset="browser",
    description="Fetch a page through a shared browser/scraping sidecar that can route to CloakBrowser or Scrapling.",
    emoji="🌐",
    schema={
        "name": "browser_tools_fetch",
        "description": "Fetch a webpage using the shared browser-tools sidecar. Use this when BROWSER_TOOLS_URL is configured and agents should share one container that combines CloakBrowser stealth rendering with Scrapling scraping/extraction.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch."},
                "mode": {"type": "string", "description": "Routing mode: auto, cloak, scrapling_http, scrapling_dynamic, scrapling_stealth, or plain_http.", "default": "auto"},
                "timeout_ms": {"type": "integer", "description": "Request/navigation timeout in milliseconds, capped at 120000.", "default": 30000},
                "text_limit": {"type": "integer", "description": "Maximum visible text characters to return, capped at 50000.", "default": 12000},
                "screenshot": {"type": "boolean", "description": "Capture a screenshot when supported by the chosen mode.", "default": False},
                "humanize": {"type": "boolean", "description": "Ask the sidecar to use humanized browser behavior when supported.", "default": False},
                "wait_until": {"type": "string", "description": "Browser wait condition for rendered modes.", "default": "domcontentloaded"},
            },
            "required": ["url"],
        },
    },
    handler=lambda args, **kwargs: browser_tools_fetch(
        url=args.get("url", ""),
        mode=args.get("mode", "auto"),
        timeout_ms=args.get("timeout_ms", 30_000),
        text_limit=args.get("text_limit", 12_000),
        screenshot=bool(args.get("screenshot", False)),
        humanize=bool(args.get("humanize", False)),
        wait_until=args.get("wait_until", "domcontentloaded"),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_browser_tools_sidecar_requirements,
    max_result_size_chars=100_000,
)

registry.register(
    name="browser_tools_extract",
    toolset="browser",
    description="Extract structured fields from a webpage through the shared Scrapling-enabled browser-tools sidecar.",
    emoji="🧩",
    schema={
        "name": "browser_tools_extract",
        "description": "Extract structured webpage fields through the shared browser-tools sidecar. Best for CSS/XPath extraction, links, headings, product/docs pages, and tool-enrichment workflows.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to extract from."},
                "selectors": {"type": "object", "description": "Mapping of output field name to Scrapling-compatible CSS/XPath selector. CSS is default; prefix values with xpath: for XPath.", "additionalProperties": {"type": "string"}, "default": {}},
                "mode": {"type": "string", "description": "Extraction mode: scrapling_http, scrapling_dynamic, scrapling_stealth, or auto.", "default": "scrapling_http"},
                "timeout_ms": {"type": "integer", "description": "Request/navigation timeout in milliseconds, capped at 120000.", "default": 30000},
                "text_limit": {"type": "integer", "description": "Maximum page text characters to include, capped at 50000.", "default": 12000},
                "include_links": {"type": "boolean", "description": "Include discovered links in the response.", "default": True},
            },
            "required": ["url"],
        },
    },
    handler=lambda args, **kwargs: browser_tools_extract(
        url=args.get("url", ""),
        selectors=args.get("selectors") if isinstance(args.get("selectors"), dict) else {},
        mode=args.get("mode", "scrapling_http"),
        timeout_ms=args.get("timeout_ms", 30_000),
        text_limit=args.get("text_limit", 12_000),
        include_links=bool(args.get("include_links", True)),
        task_id=kwargs.get("task_id"),
    ),
    check_fn=check_browser_tools_sidecar_requirements,
    max_result_size_chars=100_000,
)

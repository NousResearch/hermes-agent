#!/usr/bin/env python3
"""
Standalone Web Tools Module

This module provides generic web tools that work with multiple backend providers.
Backend is selected during ``hermes tools`` setup (web.backend in config.yaml).
When available, Hermes can route Firecrawl calls through a Nous-hosted tool-gateway
for Nous Subscribers only.

Available tools:
- web_search_tool: Search the web for information
- web_extract_tool: Extract content from specific web pages

Backend compatibility:
- Exa: https://exa.ai (search, extract)
- Firecrawl: https://docs.firecrawl.dev/introduction (search, extract; direct or derived firecrawl-gateway.<domain> for Nous Subscribers)
- Parallel: https://docs.parallel.ai (search, extract)
- Tavily: https://tavily.com (search, extract)

LLM Processing:
- Uses OpenRouter API with Gemini 3 Flash Preview for intelligent content extraction
- Extracts key excerpts and creates markdown summaries to reduce token usage

Debug Mode:
- Set WEB_TOOLS_DEBUG=true to enable detailed logging
- Creates web_tools_debug_UUID.json in ./logs directory
- Captures all tool calls, results, and compression metrics

Usage:
    from web_tools import web_search_tool, web_extract_tool, web_crawl_tool

    # Search the web
    results = web_search_tool("Python machine learning libraries", limit=3)

    # Extract content from URLs
    content = web_extract_tool(["https://example.com"], format="markdown")

    # Crawl a website
    crawl_data = web_crawl_tool("example.com", "Find contact information")
"""

import json
import logging
import os
import re
import asyncio
import hashlib
from html.parser import HTMLParser
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Tuple
import httpx  # noqa: F401 — kept at module top so tests can patch tools.web_tools.httpx
# After the web-provider plugin migration (PR #25182), the Firecrawl SDK
# proxy, client construction, and response-shape normalizers all live in
# plugins.web.firecrawl.provider. We re-export the names that external
# code, integration tests, and unit-test patches reach for so the public
# surface stays stable.
if TYPE_CHECKING:
    from firecrawl import Firecrawl  # noqa: F401 — type hints only
from plugins.web.firecrawl.provider import (
    Firecrawl,  # noqa: F401  # re-exported for tests that mock.patch("tools.web_tools.Firecrawl")
    _firecrawl_backend_help_suffix,
    _get_firecrawl_client,  # noqa: F401  # re-exported for tests that `from tools.web_tools import _get_firecrawl_client`
    _get_firecrawl_gateway_url,
    _is_tool_gateway_ready,
    check_firecrawl_api_key,
)
# Tavily helpers re-exported for backward-compat with existing unit tests
# (tests/tools/test_web_tools_tavily.py imports these names directly).
from plugins.web.tavily.provider import (  # noqa: F401 — backward-compat names
    _normalize_tavily_documents,
    _normalize_tavily_search_results,
    _tavily_request,
)
# Parallel + Exa clients re-exported for backward-compat with existing
# unit tests (tests/tools/test_web_tools_config.py imports _get_parallel_client
# / _get_async_parallel_client / _get_exa_client directly).
from plugins.web.parallel.provider import (  # noqa: F401 — backward-compat names
    _get_async_parallel_client,
    _get_parallel_client,
)
from plugins.web.exa.provider import _get_exa_client  # noqa: F401

# Module-level cache slots for the per-vendor clients. The plugins read/write
# these via tools.web_tools so unit tests that reset
# ``tools.web_tools._<vendor>_client = None`` between cases keep working.
_firecrawl_client: Optional[Any] = None
_firecrawl_client_config: Optional[Any] = None
_parallel_client: Optional[Any] = None
_async_parallel_client: Optional[Any] = None
_exa_client: Optional[Any] = None

from agent.auxiliary_client import (
    async_call_llm,
    extract_content_or_reasoning,
    get_async_text_auxiliary_client,
)
from tools.debug_helpers import DebugSession
# Imported solely so unit tests can monkeypatch these names on
# tools.web_tools (the firecrawl plugin reads them via its own import chain).
from tools.managed_tool_gateway import (  # noqa: F401 — backward-compat names for tests
    build_vendor_gateway_url,
    peek_nous_access_token as _peek_nous_access_token,
    read_nous_access_token as _read_nous_access_token,
    resolve_managed_tool_gateway,
)
from tools.tool_backend_helpers import (  # noqa: F401
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    prefers_gateway,
)
from tools.url_safety import async_is_safe_url, is_safe_url, normalize_url_for_request, sensitive_query_param_name
from tools.website_policy import check_website_access
import sys

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _env_value(name: str) -> str:
    """Resolve ``name`` via Hermes config-aware env, falling back to process env.

    Mirrors the SearXNG provider's ``_searxng_url()`` so that values set
    through Hermes' config/.env layer (``hermes config set``, ``hermes tools``)
    are honored here too — not just raw process-env exports. Without this,
    a config-only ``SEARXNG_URL`` (or any provider key) leaves the backend
    auto-detect cascade and ``check_web_api_key()`` blind to it. See #34290.
    """
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:
        val = None
    if val is None:
        val = os.getenv(name, "")
    return (val or "").strip()


def _has_env(name: str) -> bool:
    return bool(_env_value(name))

def _load_web_config() -> dict:
    """Load the ``web:`` section from ~/.hermes/config.yaml."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web", {})
    except (ImportError, Exception):
        return {}


# The built-in web backends whose availability is driven by hardcoded
# env-var / package / OAuth probes below. Any name NOT in this set is a
# candidate plugin-registered provider and must be resolved through the
# web_search_registry (``is_available()``) instead. Kept as a single named
# constant so the whitelist early-returns and the availability chokepoint
# stay in sync.
#
# NOTE: this intentionally includes ``xai``, which the registry's
# ``_LEGACY_PREFERENCE`` does NOT — xai availability is probed via
# ``has_xai_credentials()`` (env var OR auth.json OAuth), not a registered
# WebSearchProvider. Keep the two sets aligned by hand: if xai ever ships as
# a registered provider, drop it here so the registry path takes over.
_LEGACY_WEB_BACKENDS = frozenset(
    {"parallel", "firecrawl", "tavily", "exa", "searxng", "brave-free", "ddgs", "xai"}
)


def _registered_web_provider(backend: str):
    """Return a plugin-registered web provider by name, or ``None``.

    Consults ``agent.web_search_registry`` so backends contributed by the
    plugin system (which are absent from :data:`_LEGACY_WEB_BACKENDS`) are
    discoverable during availability/selection resolution. Returns ``None``
    on any lookup failure so callers can fall through to legacy checks.
    """
    if not backend:
        return None
    try:
        from agent.web_search_registry import get_provider

        return get_provider(backend)
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry lookup failed for %r: %s", backend, exc)
        return None


def _registered_web_provider_available(backend: str):
    """Availability of a *registered* web provider, or ``None`` if unregistered.

    Returns ``True``/``False`` when *backend* names a registered provider
    (calling its ``is_available()``), or ``None`` when it isn't registered —
    letting the caller fall through to the legacy built-in probes.
    """
    provider = _registered_web_provider(backend)
    if provider is None:
        return None
    try:
        return bool(provider.is_available())
    except Exception as exc:  # noqa: BLE001 — a broken provider is "unavailable"
        logger.debug("web provider %r.is_available() raised: %s", backend, exc)
        return False


def _list_registered_web_providers():
    """Return all plugin-registered web providers (empty list on failure)."""
    try:
        from agent.web_search_registry import list_providers

        return list_providers()
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry list failed: %s", exc)
        return []


def _get_backend() -> str:
    """Determine which web backend to use (shared fallback).

    Reads ``web.backend`` from config.yaml (set by ``hermes tools``).
    Falls back to whichever API key is present for users who configured
    keys manually without running setup.
    """
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured in _LEGACY_WEB_BACKENDS or _registered_web_provider(configured) is not None:
        return configured

    # Fallback for manual / legacy config — pick the highest-priority
    # available backend. Explicit user credentials (TAVILY_API_KEY etc.)
    # beat the managed-tool-gateway probe so a deliberate setup is not
    # pre-empted by a Nous OAuth token whose subscription tier may not
    # actually grant web-search access (the gateway then fails at runtime
    # with "no subscription" and the tool returns an error to the agent
    # without falling back). Free-tier backends trail the paid ones.
    backend_candidates = (
        ("tavily", _has_env("TAVILY_API_KEY")),
        ("exa", _has_env("EXA_API_KEY")),
        ("parallel", _has_env("PARALLEL_API_KEY")),
        ("firecrawl", _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL")),
        ("firecrawl", _is_tool_gateway_ready()),
        ("searxng", _has_env("SEARXNG_URL")),
        ("brave-free", _has_env("BRAVE_SEARCH_API_KEY")),
        ("ddgs", _ddgs_package_importable()),
    )
    for backend, available in backend_candidates:
        if available:
            return backend

    # Final fallback: walk plugin-registered providers so a custom backend
    # (with no built-in creds present) still resolves. Built-in names are
    # already covered above, so this only surfaces plugin-contributed
    # providers via their own is_available() gate. We hold the provider
    # object already, so probe it directly rather than round-tripping through
    # _is_backend_available() (which would re-do the registry lookup).
    for provider in _list_registered_web_providers():
        if provider.name in _LEGACY_WEB_BACKENDS:
            continue
        try:
            if provider.is_available():
                return provider.name
        except Exception as exc:  # noqa: BLE001 — a broken provider is skipped
            logger.debug("web provider %r.is_available() raised: %s", provider.name, exc)

    return "firecrawl"  # default (backward compat)


def _get_search_backend() -> str:
    """Determine which backend to use for web_search specifically.

    Selection priority:
    1. ``web.search_backend`` (per-capability override)
    2. ``web.backend`` (shared fallback — existing behavior)
    3. Auto-detect from env vars

    This enables using different providers for search vs extract
    (e.g. SearXNG for search + Firecrawl for extract).
    """
    return _get_capability_backend("search")


def _get_extract_backend() -> str:
    """Determine which backend to use for web_extract specifically.

    Selection priority:
    1. ``web.extract_backend`` (per-capability override)
    2. ``web.backend`` (shared fallback — existing behavior)
    3. Auto-detect from env vars
    """
    return _get_capability_backend("extract")


def _get_capability_backend(capability: str) -> str:
    """Shared helper for per-capability backend selection.

    Reads ``web.{capability}_backend`` from config; if set and available,
    uses it. Otherwise falls through to the shared ``_get_backend()``.
    """
    cfg = _load_web_config()
    specific = (cfg.get(f"{capability}_backend") or "").lower().strip()
    if specific and _is_backend_available(specific):
        return specific
    return _get_backend()


def _is_backend_available(backend: str) -> bool:
    """Return True when the selected backend is currently usable.

    For plugin-registered backends (any name outside
    :data:`_LEGACY_WEB_BACKENDS`), availability is delegated to the
    provider's ``is_available()`` via the web_search_registry. This is the
    single chokepoint through which ``_get_backend``,
    ``_get_capability_backend``, and ``check_web_api_key`` all resolve
    availability — fixing custom-provider discovery for every caller at once
    (issues #28651, #31873, #32698). Built-in backends keep their cheap
    hardcoded probes below.
    """
    backend = (backend or "").lower().strip()
    if backend not in _LEGACY_WEB_BACKENDS:
        registered = _registered_web_provider_available(backend)
        if registered is not None:
            return registered
    if backend == "exa":
        return _has_env("EXA_API_KEY")
    if backend == "parallel":
        return _has_env("PARALLEL_API_KEY")
    if backend == "firecrawl":
        return check_firecrawl_api_key()
    if backend == "tavily":
        return _has_env("TAVILY_API_KEY")
    if backend == "searxng":
        return _has_env("SEARXNG_URL")
    if backend == "brave-free":
        return _has_env("BRAVE_SEARCH_API_KEY")
    if backend == "ddgs":
        return _ddgs_package_importable()
    if backend == "xai":
        # Cheap probe — env var OR auth.json has OAuth tokens. Must not
        # call resolve_xai_http_credentials() here because the OAuth path
        # can trigger a network token refresh, and _is_backend_available
        # runs on every web_search dispatch + every `hermes tools` repaint.
        try:
            from tools.xai_http import has_xai_credentials
            return has_xai_credentials()
        except Exception:
            return False
    return False


def _web_provider_fallback_candidates(primary_name: str, capability: str) -> List[Any]:
    """Return available providers to try after a configured provider fails."""
    _ensure_web_search_plugins_registered()
    try:
        from agent.web_search_registry import get_provider, list_providers
    except Exception:
        return []

    primary_name = str(primary_name or "").strip()
    registered = {str(provider.name): provider for provider in list_providers()}
    seen_names = {primary_name}
    ordered_names = []
    for name in _MATRIX_PROVIDER_ORDER:
        if name not in seen_names:
            ordered_names.append(name)
            seen_names.add(name)
    ordered_names.extend(sorted(name for name in registered if name not in seen_names))

    capability_check = {
        "search": "supports_search",
        "extract": "supports_extract",
        "crawl": "supports_crawl",
    }.get(capability)
    if capability_check is None:
        return []

    candidates = []
    for name in ordered_names:
        provider = get_provider(name)
        if provider is None:
            continue
        supports = getattr(provider, capability_check, None)
        try:
            provider_supported = bool(callable(supports) and supports())
        except Exception:
            provider_supported = False
        if not provider_supported:
            continue
        try:
            available = bool(provider.is_available())
        except Exception:
            available = False
        if available:
            candidates.append(provider)
    return candidates


def _web_search_error_message(response_data: Any, default: str = "web search failed") -> str:
    """Return a compact provider error message for fallback metadata."""
    if isinstance(response_data, dict):
        error = response_data.get("error") or response_data.get("message")
        if error:
            return str(error)
    return default


def _web_search_should_try_fallback(response_data: Any) -> bool:
    if not isinstance(response_data, dict):
        return True
    if response_data.get("success", False):
        return False
    error = str(response_data.get("error") or "").lower()
    if not error:
        return True
    retryable_markers = (
        "payment required",
        "insufficient credits",
        "quota",
        "rate limit",
        "429",
        "402",
        "432",
        "timeout",
        "temporarily",
        "unavailable",
    )
    return any(marker in error for marker in retryable_markers)


def _ddgs_package_importable() -> bool:
    """Return True when the ``ddgs`` Python package can be imported.

    ddgs is the only backend whose availability is driven by a package
    presence rather than an env var / config entry.  Wrapped in a helper
    so auto-detect and ``_is_backend_available`` share the same check
    (and tests can monkeypatch a single symbol).
    """
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        return False

# ─── Firecrawl Client ────────────────────────────────────────────────────────

# ─── Firecrawl Client ────────────────────────────────────────────────────────
# After PR #25182, the firecrawl client, lazy SDK proxy, dual-auth config
# resolution, response normalizers, and check_firecrawl_api_key() all live
# in plugins.web.firecrawl.provider and are re-exported at the top of this
# module so external callers (integration tests, tool-registry gating) and
# unit tests that patch tools.web_tools.<name> continue to work.


def _web_requires_env() -> list[str]:
    """Return tool metadata env vars for the currently enabled web backends.

    The gateway env vars are always reported — they're metadata strings
    used by the tool registry to light up the tool when the variable is
    set.  Gating them on ``managed_nous_tools_enabled()`` only saved
    string noise in the metadata list, but cost a synchronous HTTP
    refresh against the Nous portal on every CLI startup (invoked at
    tool-registration time).  The behavioral contract is: if the env var
    is set, the tool sees it; if not, it doesn't.  Not-logged-in users
    simply don't have the vars set, so the extra entries are harmless.
    """
    return [
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
    ]


# ─── Parallel / Tavily / Firecrawl helpers — moved into plugins ──────────────
# After PR #25182, the per-vendor client construction, request helpers, and
# response normalizers all live in plugins.web.<vendor>.provider:
#   - parallel: plugins/web/parallel/provider.py
#   - tavily:   plugins/web/tavily/provider.py
#   - firecrawl: plugins/web/firecrawl/provider.py
# The names from the firecrawl plugin (Firecrawl proxy, _get_firecrawl_client,
# _to_plain_object, _normalize_result_list, _extract_web_search_results,
# _extract_scrape_payload, _is_tool_gateway_ready, etc.) are re-exported at
# the top of this module for backward-compat with integration tests and
# unit-test patches.


DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION = 5000
_DIRECT_HTTP_FALLBACK_MAX_TEXT_CHARS = 1_000_000


def _classify_web_extract_failure(error: Any, provider_name: str = "") -> Optional[str]:
    """Return a stable machine-readable category for web extraction failures."""
    text = str(error or "").lower()
    provider = str(provider_name or "").lower()
    if not text:
        return None
    if provider == "firecrawl" and (
        "payment required" in text
        or "insufficient credits" in text
        or "credits have been exhausted" in text
        or "top up" in text
        or "402" in text
    ):
        return "provider_credit_exhaustion"
    if (
        "api key" in text
        or "not configured" in text
        or "missing direct config" in text
        or "missing credentials" in text
    ):
        return "provider_credentials_absent"
    return None


class _HTMLTextExtractor(HTMLParser):
    """Small stdlib HTML-to-text extractor for direct HTTP fallback."""

    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "dl", "fieldset",
        "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
        "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data:
            self._parts.append(data)

    def text(self) -> str:
        text = "".join(self._parts)
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()


def _html_to_text(content: str) -> str:
    parser = _HTMLTextExtractor()
    try:
        parser.feed(content)
        parser.close()
        text = parser.text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", content)
        text = re.sub(r"\s+", " ", text).strip()
    return text or content


def _extract_html_title(content: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    title = re.sub(r"\s+", " ", match.group(1)).strip()
    return title


def _truncate_direct_http_content(content: str) -> str:
    if len(content) <= _DIRECT_HTTP_FALLBACK_MAX_TEXT_CHARS:
        return content
    return (
        content[:_DIRECT_HTTP_FALLBACK_MAX_TEXT_CHARS]
        + "\n\n[Direct HTTP fallback content truncated after "
        + f"{_DIRECT_HTTP_FALLBACK_MAX_TEXT_CHARS:,} characters.]"
    )


async def _direct_http_extract_one(url: str, format: Optional[str] = None) -> Dict[str, Any]:
    """Fetch one URL without a scrape provider for credit-exhaustion fallback."""
    blocked = check_website_access(url)
    if blocked:
        return {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": blocked["message"],
            "blocked_by_policy": {
                "host": blocked["host"],
                "rule": blocked["rule"],
                "source": blocked["source"],
            },
        }

    headers = {
        "User-Agent": (
            "Hermes web_extract direct-http fallback "
            "(provider credit exhaustion recovery)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain,*/*;q=0.8",
    }
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0, headers=headers) as client:
        response = await client.get(url)
        response.raise_for_status()

    final_url = str(response.url)
    if not is_safe_url(final_url):
        return {
            "url": final_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": "Blocked: URL redirected to a private or internal network address",
        }

    final_blocked = check_website_access(final_url)
    if final_blocked:
        return {
            "url": final_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": final_blocked["message"],
            "blocked_by_policy": {
                "host": final_blocked["host"],
                "rule": final_blocked["rule"],
                "source": final_blocked["source"],
            },
        }

    body = response.content or b""
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    metadata = {
        "source": "direct_http",
        "content_type": content_type or None,
        "byte_length": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }

    is_pdf = content_type == "application/pdf" or final_url.lower().split("?", 1)[0].endswith(".pdf")
    if is_pdf:
        content = (
            "[Direct HTTP fallback fetched PDF source bytes; text extraction was not "
            "available in the scraper fallback. "
            f"byte_length={metadata['byte_length']} sha256={metadata['sha256']}]"
        )
        return {
            "url": final_url,
            "title": "",
            "content": content,
            "raw_content": content,
            "metadata": metadata,
        }

    text = response.text or body.decode(response.encoding or "utf-8", errors="replace")
    title = _extract_html_title(text)
    if format != "html" and ("html" in content_type or re.search(r"<html[\s>]", text, re.IGNORECASE)):
        text = _html_to_text(text)
    text = _truncate_direct_http_content(text)
    return {
        "url": final_url,
        "title": title,
        "content": text,
        "raw_content": text,
        "metadata": metadata,
    }


async def _apply_web_extract_degradation_fallback(
    *,
    provider_name: str,
    results: List[Dict[str, Any]],
    format: Optional[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply deterministic fallback for known provider degradation categories."""
    if str(provider_name or "").lower() != "firecrawl":
        return results, []

    updated: List[Dict[str, Any]] = []
    degradations: List[Dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            updated.append(result)
            continue

        error = result.get("error")
        category = _classify_web_extract_failure(error, provider_name)
        if category != "provider_credit_exhaustion":
            updated.append(result)
            continue

        url = str(result.get("url") or "").strip()
        degradation = {
            "category": category,
            "primary_provider": "firecrawl",
            "primary_operation": "web_extract",
            "primary_status": "failed",
            "primary_error": str(error or ""),
            "fallback_provider": "direct_http",
            "fallback_status": "not_attempted",
        }

        if not url or not is_safe_url(url):
            degradation["fallback_status"] = "blocked"
            degradation["fallback_error"] = "URL was empty or failed safety checks"
            retained = dict(result)
            retained["degradation"] = degradation
            updated.append(retained)
            degradations.append(degradation)
            continue

        try:
            fallback = await _direct_http_extract_one(url, format=format)
        except Exception as exc:  # noqa: BLE001 - preserve primary failure
            fallback = {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": str(exc),
            }

        retained: Dict[str, Any]
        if isinstance(fallback, dict) and fallback.get("content") and not fallback.get("error"):
            degradation["fallback_status"] = "succeeded"
            retained = dict(fallback)
            retained["degradation"] = degradation
        else:
            degradation["fallback_status"] = "failed"
            degradation["fallback_error"] = str((fallback or {}).get("error") or "direct HTTP fallback failed")
            retained = dict(result)
            retained["degradation"] = degradation

        updated.append(retained)
        degradations.append(degradation)

    return updated, degradations


# Default budget (characters) of clean page text sent to the model. Pages at
# or under this size are returned whole; larger pages are head+tail truncated
# and the full text is stored on disk (see _store_full_text). Spending context,
# not API dollars — so this is generous relative to the old 5k summary cap.
# Override via web.extract_char_limit in config.yaml.
DEFAULT_EXTRACT_CHAR_LIMIT = 15000

# Hard ceiling on the full-text file written to cache/web. The truncate-store
# path otherwise calls path.write_text(content) with no upper bound, so a
# multi-MB page (some backends return very large markdown) writes unbounded
# bytes to disk on every extract. Cap the stored copy; the model only ever
# sees char_limit anyway, and a 2MB page is already far more than any single
# read_file paging session needs. Mirrors the pre-truncate-store era's 2MB
# refusal ceiling, but stores (capped) instead of refusing.
MAX_STORED_TEXT_CHARS = 2_000_000

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


def _is_nous_auxiliary_client(client: Any) -> bool:
    """Return True when the resolved auxiliary backend is Nous Portal."""
    from urllib.parse import urlparse

    base_url = str(getattr(client, "base_url", "") or "")
    host = (urlparse(base_url).hostname or "").lower()
    return host == "nousresearch.com" or host.endswith(".nousresearch.com")


def _resolve_web_extract_auxiliary(model: Optional[str] = None) -> tuple[Optional[Any], Optional[str], Dict[str, Any]]:
    """Resolve the current web-extract auxiliary client, model, and extra body."""
    client, default_model = get_async_text_auxiliary_client("web_extract")
    configured_model = os.getenv("AUXILIARY_WEB_EXTRACT_MODEL", "").strip()
    effective_model = model or configured_model or default_model

    extra_body: Dict[str, Any] = {}
    if client is not None and _is_nous_auxiliary_client(client):
        from agent.auxiliary_client import get_auxiliary_extra_body
        from agent.portal_tags import nous_portal_tags
        extra_body = get_auxiliary_extra_body() or {"tags": nous_portal_tags()}

    return client, effective_model, extra_body


def _get_default_summarizer_model() -> Optional[str]:
    """Return the current default model for web extraction summarization."""
    _, model, _ = _resolve_web_extract_auxiliary()
    return model


async def process_content_with_llm(
    content: str,
    url: str = "",
    title: str = "",
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> Optional[str]:
    """
    Process web content using LLM to create intelligent summaries with key excerpts.

    This function uses Gemini 3 Flash Preview (or specified model) via OpenRouter API
    to intelligently extract key information and create markdown summaries,
    significantly reducing token usage while preserving all important information.

    For very large content (>500k chars), uses chunked processing with synthesis.
    For extremely large content (>2M chars), refuses to process entirely.

    Args:
        content (str): The raw content to process
        url (str): The source URL (for context, optional)
        title (str): The page title (for context, optional)
        model (str): The model to use for processing (default: google/gemini-3-flash-preview)
        min_length (int): Minimum content length to trigger processing (default: 5000)

    Returns:
        Optional[str]: Processed markdown content, or None if content too short or processing fails
    """
    # Size thresholds
    MAX_CONTENT_SIZE = 2_000_000  # 2M chars - refuse entirely above this
    CHUNK_THRESHOLD = 500_000     # 500k chars - use chunked processing above this
    CHUNK_SIZE = 100_000          # 100k chars per chunk
    MAX_OUTPUT_SIZE = 5000        # Hard cap on final output size

    try:
        content_len = len(content)

        # Refuse if content is absurdly large
        if content_len > MAX_CONTENT_SIZE:
            size_mb = content_len / 1_000_000
            logger.warning("Content too large (%.1fMB > 2MB limit). Refusing to process.", size_mb)
            return f"[Content too large to process: {size_mb:.1f}MB. Try using web_crawl with specific extraction instructions, or search for a more focused source.]"

        # Skip processing if content is too short
        if content_len < min_length:
            logger.debug("Content too short (%d < %d chars), skipping LLM processing", content_len, min_length)
            return None

        # Create context information
        context_info = []
        if title:
            context_info.append(f"Title: {title}")
        if url:
            context_info.append(f"Source: {url}")
        context_str = "\n".join(context_info) + "\n\n" if context_info else ""

        # Check if we need chunked processing
        if content_len > CHUNK_THRESHOLD:
            logger.info("Content large (%d chars). Using chunked processing...", content_len)
            return await _process_large_content_chunked(
                content, context_str, model, CHUNK_SIZE, MAX_OUTPUT_SIZE
            )

        # Standard single-pass processing for normal content
        logger.info("Processing content with LLM (%d characters)", content_len)

        processed_content = await _call_summarizer_llm(content, context_str, model)

        if processed_content:
            # Enforce output cap
            if len(processed_content) > MAX_OUTPUT_SIZE:
                processed_content = processed_content[:MAX_OUTPUT_SIZE] + "\n\n[... summary truncated for context management ...]"

            # Log compression metrics
            processed_length = len(processed_content)
            compression_ratio = processed_length / content_len if content_len > 0 else 1.0
            logger.info("Content processed: %d -> %d chars (%.1f%%)", content_len, processed_length, compression_ratio * 100)

        return processed_content

    except Exception as e:
        logger.warning(
            "web_extract LLM summarization failed (%s). "
            "Tip: increase auxiliary.web_extract.timeout in config.yaml "
            "or switch to a faster auxiliary model.",
            str(e)[:120],
        )
        # Fall back to truncated raw content instead of returning a useless
        # error message.  The first ~5000 chars are almost always more useful
        # to the model than "[Failed to process content: ...]".
        truncated = content[:MAX_OUTPUT_SIZE]
        if len(content) > MAX_OUTPUT_SIZE:
            truncated += (
                f"\n\n[Content truncated — showing first {MAX_OUTPUT_SIZE:,} of "
                f"{len(content):,} chars. LLM summarization timed out. "
                f"To fix: increase auxiliary.web_extract.timeout in config.yaml, "
                f"or use a faster auxiliary model. Use browser_navigate for the full page.]"
            )
        return truncated


async def _call_summarizer_llm(
    content: str,
    context_str: str,
    model: Optional[str],
    max_tokens: int = 20000,
    is_chunk: bool = False,
    chunk_info: str = ""
) -> Optional[str]:
    """
    Make a single LLM call to summarize content.

    Args:
        content: The content to summarize
        context_str: Context information (title, URL)
        model: Model to use
        max_tokens: Maximum output tokens
        is_chunk: Whether this is a chunk of a larger document
        chunk_info: Information about chunk position (e.g., "Chunk 2/5")

    Returns:
        Summarized content or None on failure
    """
    if is_chunk:
        # Chunk-specific prompt - aware that this is partial content
        system_prompt = """You are an expert content analyst processing a SECTION of a larger document. Your job is to extract and summarize the key information from THIS SECTION ONLY.

Important guidelines for chunk processing:
1. Do NOT write introductions or conclusions - this is a partial document
2. Focus on extracting ALL key facts, figures, data points, and insights from this section
3. Preserve important quotes, code snippets, and specific details verbatim
4. Use bullet points and structured formatting for easy synthesis later
5. Note any references to other sections (e.g., "as mentioned earlier", "see below") without trying to resolve them

Your output will be combined with summaries of other sections, so focus on thorough extraction rather than narrative flow."""

        user_prompt = f"""Extract key information from this SECTION of a larger document:

{context_str}{chunk_info}

SECTION CONTENT:
{content}

Extract all important information from this section in a structured format. Focus on facts, data, insights, and key details. Do not add introductions or conclusions."""

    else:
        # Standard full-document prompt
        system_prompt = """You are an expert content analyst. Your job is to process web content and create a comprehensive yet concise summary that preserves all important information while dramatically reducing bulk.

Create a well-structured markdown summary that includes:
1. Key excerpts (quotes, code snippets, important facts) in their original format
2. Comprehensive summary of all other important information
3. Proper markdown formatting with headers, bullets, and emphasis

Your goal is to preserve ALL important information while reducing length. Never lose key facts, figures, insights, or actionable information. Make it scannable and well-organized."""

        user_prompt = f"""Please process this web content and create a comprehensive markdown summary:

{context_str}CONTENT TO PROCESS:
{content}

Create a markdown summary that captures all key information in a well-organized, scannable format. Include important quotes and code snippets in their original formatting. Focus on actionable information, specific details, and unique insights."""

    # Call the LLM with retry logic — keep retries low since summarization
    # is a nice-to-have; the caller falls back to truncated content on failure.
    max_retries = 2
    retry_delay = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
            if aux_client is None or not effective_model:
                logger.warning("No auxiliary model available for web content processing")
                return None
            call_kwargs = {
                "task": "web_extract",
                "model": effective_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": max_tokens,
                # No explicit timeout — async_call_llm reads auxiliary.web_extract.timeout
                # from config.yaml. Fresh configs ship with 360s; if the key is absent
                # the runtime default is 30s (_DEFAULT_AUX_TIMEOUT in
                # agent/auxiliary_client.py). Users with slow local models should set
                # or increase auxiliary.web_extract.timeout in config.yaml.
            }
            if extra_body:
                call_kwargs["extra_body"] = extra_body
            response = await async_call_llm(**call_kwargs)
            content = extract_content_or_reasoning(response)
            if content:
                return content
            # Reasoning-only / empty response — let the retry loop handle it
            logger.warning("LLM returned empty content (attempt %d/%d), retrying", attempt + 1, max_retries)
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                continue
            return content  # Return whatever we got after exhausting retries
        except RuntimeError:
            logger.warning("No auxiliary model available for web content processing")
            return None
        except Exception as api_error:
            last_error = api_error
            if attempt < max_retries - 1:
                logger.warning("LLM API call failed (attempt %d/%d): %s", attempt + 1, max_retries, str(api_error)[:100])
                logger.warning("Retrying in %ds...", retry_delay)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
            else:
                raise last_error

    return None


async def _process_large_content_chunked(
    content: str,
    context_str: str,
    model: Optional[str],
    chunk_size: int,
    max_output_size: int
) -> Optional[str]:
    """
    Process large content by chunking, summarizing each chunk in parallel,
    then synthesizing the summaries.

    Args:
        content: The large content to process
        context_str: Context information
        model: Model to use
        chunk_size: Size of each chunk in characters
        max_output_size: Maximum final output size

    Returns:
        Synthesized summary or None on failure
    """
    # Split content into chunks
    chunks = []
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)

    logger.info("Split into %d chunks of ~%d chars each", len(chunks), chunk_size)

    # Summarize each chunk in parallel
    async def summarize_chunk(chunk_idx: int, chunk_content: str) -> tuple[int, Optional[str]]:
        """Summarize a single chunk."""
        try:
            chunk_info = f"[Processing chunk {chunk_idx + 1} of {len(chunks)}]"
            summary = await _call_summarizer_llm(
                chunk_content,
                context_str,
                model,
                max_tokens=10000,
                is_chunk=True,
                chunk_info=chunk_info
            )
            if summary:
                logger.info("Chunk %d/%d summarized: %d -> %d chars", chunk_idx + 1, len(chunks), len(chunk_content), len(summary))
            return chunk_idx, summary
        except Exception as e:
            logger.warning("Chunk %d/%d failed: %s", chunk_idx + 1, len(chunks), str(e)[:50])
            return chunk_idx, None

    # Run all chunk summarizations in parallel
    tasks = [summarize_chunk(i, chunk) for i, chunk in enumerate(chunks)]
    # Use return_exceptions=True so a single task failure does not discard
    # all other successfully summarized chunks.
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions, then collect successful summaries in order
    successful_results = []
    for result_item in results:
        if isinstance(result_item, BaseException):
            logger.warning("Chunk summarization task failed: %s", result_item)
            continue
        successful_results.append(result_item)

    summaries = []
    for chunk_idx, summary in sorted(successful_results, key=lambda x: x[0]):
        if summary:
            summaries.append(f"## Section {chunk_idx + 1}\n{summary}")

    if not summaries:
        logger.debug("All chunk summarizations failed")
        return "[Failed to process large content: all chunk summarizations failed]"

    logger.info("Got %d/%d chunk summaries", len(summaries), len(chunks))

    # If only one chunk succeeded, just return it (with cap)
    if len(summaries) == 1:
        result = summaries[0]
        if len(result) > max_output_size:
            result = result[:max_output_size] + "\n\n[... truncated ...]"
        return result

    # Synthesize the summaries into a final summary
    logger.info("Synthesizing %d summaries...", len(summaries))

    combined_summaries = "\n\n---\n\n".join(summaries)

    synthesis_prompt = f"""You have been given summaries of different sections of a large document.
Synthesize these into ONE cohesive, comprehensive summary that:
1. Removes redundancy between sections
2. Preserves all key facts, figures, and actionable information
3. Is well-organized with clear structure
4. Is under {max_output_size} characters

{context_str}SECTION SUMMARIES:
{combined_summaries}

Create a single, unified markdown summary."""

    try:
        aux_client, effective_model, extra_body = _resolve_web_extract_auxiliary(model)
        if aux_client is None or not effective_model:
            logger.warning("No auxiliary model for synthesis, concatenating summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        call_kwargs = {
            "task": "web_extract",
            "model": effective_model,
            "messages": [
                {"role": "system", "content": "You synthesize multiple summaries into one cohesive, comprehensive summary. Be thorough but concise."},
                {"role": "user", "content": synthesis_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 20000,
        }
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        response = await async_call_llm(**call_kwargs)
        final_summary = extract_content_or_reasoning(response)

        # Retry once on empty content (reasoning-only response)
        if not final_summary:
            logger.warning("Synthesis LLM returned empty content, retrying once")
            response = await async_call_llm(**call_kwargs)
            final_summary = extract_content_or_reasoning(response)

        # If still None after retry, fall back to concatenated summaries
        if not final_summary:
            logger.warning("Synthesis failed after retry — concatenating chunk summaries")
            fallback = "\n\n".join(summaries)
            if len(fallback) > max_output_size:
                fallback = fallback[:max_output_size] + "\n\n[... truncated ...]"
            return fallback

        # Enforce hard cap
        if len(final_summary) > max_output_size:
            final_summary = final_summary[:max_output_size] + "\n\n[... summary truncated for context management ...]"

        original_len = len(content)
        final_len = len(final_summary)
        compression = final_len / original_len if original_len > 0 else 1.0

        logger.info("Synthesis complete: %d -> %d chars (%.2f%%)", original_len, final_len, compression * 100)
        return final_summary

    except Exception as e:
        logger.warning("Synthesis failed: %s", str(e)[:100])
        # Fall back to concatenated summaries with truncation
        fallback = "\n\n".join(summaries)
        if len(fallback) > max_output_size:
            fallback = fallback[:max_output_size] + "\n\n[... truncated due to synthesis failure ...]"
        return fallback


def clean_base64_images(text: str) -> str:
    """
    Remove base64 encoded images from text to reduce token count and clutter.

    This function finds and removes base64 encoded images in various formats:
    - (data:image/png;base64,...)
    - (data:image/jpeg;base64,...)
    - (data:image/svg+xml;base64,...)
    - data:image/[type];base64,... (without parentheses)

    Args:
        text: The text content to clean

    Returns:
        Cleaned text with base64 images replaced with placeholders
    """
    # Pattern to match base64 encoded images wrapped in parentheses
    # Matches: (data:image/[type];base64,[base64-string])
    base64_with_parens_pattern = r'\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)'

    # Pattern to match base64 encoded images without parentheses
    # Matches: data:image/[type];base64,[base64-string]
    base64_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'

    # Replace parentheses-wrapped images first
    cleaned_text = re.sub(base64_with_parens_pattern, '[BASE64_IMAGE_REMOVED]', text)

    # Then replace any remaining non-parentheses images
    cleaned_text = re.sub(base64_pattern, '[BASE64_IMAGE_REMOVED]', cleaned_text)

    return cleaned_text


def _get_extract_char_limit() -> int:
    """Resolve the per-page char budget from config, clamped to a sane range."""
    try:
        configured = _load_web_config().get("extract_char_limit")
        if configured is not None:
            value = int(configured)
            # Floor at 2k (below that the footer dominates), no hard ceiling
            # beyond a generous guard so a typo can't blow up context.
            return max(2000, min(value, 500_000))
    except (TypeError, ValueError):
        pass
    return DEFAULT_EXTRACT_CHAR_LIMIT


def convert_base64_images_to_links(text: str) -> str:
    """Replace inline base64 image blobs with labeled markdown links.

    base64 image payloads are token bombs (a single inline PNG can be tens of
    thousands of characters), so we never send the raw bytes to the model. But
    we preserve the fact that an image was there, and its alt text, as an
    inspectable placeholder. Real (http/https) markdown image links are left
    untouched so the agent can ``web_extract`` / ``vision_analyze`` them.

    Transformations:
      ``![alt](data:image/png;base64,AAAA...)``  -> ``[IMAGE: alt](base64 image omitted)``
      ``(data:image/png;base64,AAAA...)``        -> ``[IMAGE]``
      bare ``data:image/...;base64,AAAA...``     -> ``[IMAGE]``
    """
    # 1. Markdown image with base64 source -> keep alt text, drop the blob.
    def _md_repl(m: "re.Match[str]") -> str:
        alt = (m.group("alt") or "").strip()
        return f"[IMAGE: {alt}]" if alt else "[IMAGE]"

    md_b64 = re.compile(
        r"!\[(?P<alt>[^\]]*)\]\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)"
    )
    out = md_b64.sub(_md_repl, text)

    # 2. Parenthesised base64 (non-markdown) and 3. bare base64 -> [IMAGE].
    out = re.sub(r"\(\s*data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+\)", "[IMAGE]", out)
    out = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE]", out)
    return out


def _store_full_text(url: str, content: str) -> Optional[str]:
    """Write the full extracted page to cache/web and return its absolute path.

    The file is mounted read-only into remote backends (Docker/Modal/SSH) via
    credential_files._CACHE_DIRS, so the agent's terminal/read_file tools can
    page through the complete text on any backend. Returns None on failure
    (storage is best-effort; truncated content is still returned to the model).
    """
    try:
        import hashlib
        from urllib.parse import urlparse
        from hermes_constants import get_hermes_dir

        cache_dir = get_hermes_dir("cache/web", "web_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        host = (urlparse(url).hostname or "page").replace(":", "_")
        slug = re.sub(r"[^A-Za-z0-9._-]", "-", host)[:60].strip("-") or "page"
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
        path = cache_dir / f"{slug}-{digest}.md"
        # Bound the stored copy so a pathologically large page can't write
        # unbounded bytes to disk. If capped, append a marker so a reader of
        # the file knows it isn't the literal complete page.
        if len(content) > MAX_STORED_TEXT_CHARS:
            content = (
                content[:MAX_STORED_TEXT_CHARS]
                + f"\n\n[... stored copy truncated at {MAX_STORED_TEXT_CHARS:,} chars "
                f"of {len(content):,}; re-extract a more specific URL for the rest ...]"
            )
        path.write_text(content, encoding="utf-8")
        return str(path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to store full web_extract text for %s: %s", url, exc)
        return None


def _truncate_with_footer(
    content: str,
    url: str,
    char_limit: int,
) -> tuple[str, bool]:
    """Return (model_text, was_truncated) for one page's clean content.

    Pages at or under ``char_limit`` are returned whole. Larger pages get a
    head+tail window (~75% head / ~25% tail) cut on a markdown line boundary
    where possible, plus an explicit footer telling the model exactly how much
    it is seeing, where the full text is stored, and which read_file call pages
    in the omitted middle. Deterministic — no model involvement.
    """
    if len(content) <= char_limit:
        return content, False

    head_budget = int(char_limit * 0.75)
    tail_budget = char_limit - head_budget

    head = content[:head_budget]
    tail = content[-tail_budget:]
    # Snap the head cut back to the last newline so we don't slice mid-line.
    nl = head.rfind("\n")
    if nl > head_budget * 0.5:
        head = head[:nl]
    # Snap the tail cut forward to the next newline for the same reason.
    nl = tail.find("\n")
    if 0 <= nl < tail_budget * 0.5:
        tail = tail[nl + 1:]

    total = len(content)
    stored_path = _store_full_text(url, content)
    shown = len(head) + len(tail)

    footer_lines = [
        "",
        "─" * 8 + " [TRUNCATED] " + "─" * 8,
        f"Showing {len(head):,} chars (head) + {len(tail):,} chars (tail) "
        f"of {total:,} total clean characters.",
    ]
    if stored_path:
        # The omitted middle begins right after the head we're showing. Give
        # the model a concrete starting line (head line count + 1) so its first
        # read_file lands in the gap instead of guessing <line>. read_file is
        # 1-indexed; +1 moves past the last head line we already showed.
        middle_start_line = head.count("\n") + 2
        footer_lines.append(f"Full text saved to: {stored_path}")
        footer_lines.append(
            f'To read the omitted middle: read_file path="{stored_path}" '
            f"offset={middle_start_line} limit=200  (the file is the complete page; "
            f"raise/lower offset to page through it)."
        )
    else:
        footer_lines.append(
            "Full text could not be stored; re-run web_extract on a more "
            "specific URL or use browser_navigate for the complete page."
        )
    footer_lines.append("─" * 29)

    model_text = head + "\n\n[... middle omitted — see footer ...]\n\n" + tail
    model_text += "\n" + "\n".join(footer_lines)
    return model_text, True


# ─── Exa / Parallel inline helpers — moved into plugins ──────────────────────
# After PR #25182, the exa client + search/extract and parallel client +
# search/extract helpers all live in their respective plugins:
#   - plugins/web/exa/provider.py
#   - plugins/web/parallel/provider.py
# Both plugins register through agent.web_search_registry and the
# dispatchers in this file resolve them via get_active_*_provider().


def _ensure_web_plugins_loaded() -> None:
    """Idempotently trigger plugin discovery so the web registry is populated.

    Every bundled web provider (brave-free, ddgs, searxng, exa, parallel,
    tavily, firecrawl) registers itself via ``plugins/web/<vendor>/__init__.py``
    during plugin discovery. Tool dispatch can be reached from contexts that
    haven't already triggered discovery — subprocess agent runs, delegate
    children, standalone scripts, certain test paths — and without it the
    registry is empty and ``get_provider('firecrawl')`` returns ``None`` even
    when the user has ``web.extract_backend: firecrawl`` configured and
    ``FIRECRAWL_API_KEY`` set. The symptom is a misleading "No web extract
    provider configured" error (issue #27580).

    Mirrors :func:`tools.browser_tool._ensure_browser_plugins_loaded` exactly:
    the underlying discovery call is idempotent and cheap on subsequent
    invocations.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001
        # Warning, not debug: if a plugin import is genuinely broken the
        # user otherwise hits the misleading "No web extract provider
        # configured" error this helper is meant to eliminate, with no
        # clue in normal logs about the real cause.
        logger.warning("Web plugin discovery failed (non-fatal): %s", exc)


def web_search_tool(query: str, limit: int = 5) -> str:
    """
    Search the web for information using available search API backend.

    This function provides a generic interface for web search that can work
    with multiple backends (Parallel or Firecrawl).

    Note: This function returns search result metadata only (URLs, titles, descriptions).
    Use web_extract_tool to get full content from specific URLs.

    Args:
        query (str): The search query to look up
        limit (int): Maximum number of results to return (default: 5)

    Returns:
        str: JSON string containing search results with the following structure:
             {
                 "success": bool,
                 "data": {
                     "web": [
                         {
                             "title": str,
                             "url": str,
                             "description": str,
                             "position": int
                         },
                         ...
                     ]
                 }
             }

    Raises:
        Exception: If search fails or API key is not set
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 100)

    debug_call_data = {
        "parameters": {
            "query": query,
            "limit": limit
        },
        "error": None,
        "results_count": 0,
        "original_response_size": 0,
        "final_response_size": 0
    }

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)

        # Dispatch through the web search registry. All 7 providers
        # (brave-free, ddgs, searxng, exa, parallel, tavily, firecrawl)
        # now live as plugins; the dispatcher is just a registry lookup +
        # delegation. Sync only — every provider's search() is sync.
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import (
            get_active_search_provider,
            get_provider as _wsp_get_provider,
        )

        backend = _get_search_backend()
        provider = _wsp_get_provider(backend) if backend else None
        if provider is None or not provider.supports_search():
            # Fall back to availability-walked active provider when the
            # configured backend isn't a registered search provider (typo,
            # uninstalled plugin, or capability mismatch).
            provider = get_active_search_provider()

        if provider is None:
            response_data = {
                "success": False,
                "error": (
                    "No web search provider configured. "
                    "Run `hermes tools` to set one up."
                ),
            }
        else:
            logger.info(
                "Web search via %s: '%s' (limit: %d)",
                provider.name, query, limit,
            )
            primary_exception = None
            try:
                response_data = provider.search(query, limit)
            except Exception as exc:  # noqa: BLE001 - fallback may recover quota outages
                primary_exception = exc
                response_data = {"success": False, "error": str(exc)}

            if _web_search_should_try_fallback(response_data):
                attempted = [provider.name]
                fallback_errors = {}
                fallback_succeeded = False
                first_error = _web_search_error_message(response_data)
                for fallback_provider in _web_provider_fallback_candidates(
                    provider.name, "search"
                ):
                    if fallback_provider.name in attempted:
                        continue
                    attempted.append(fallback_provider.name)
                    logger.info(
                        "Web search fallback via %s after %s failed",
                        fallback_provider.name,
                        provider.name,
                    )
                    try:
                        fallback_response = fallback_provider.search(query, limit)
                    except Exception as exc:  # noqa: BLE001 - keep walking retryable fallbacks
                        fallback_response = {"success": False, "error": str(exc)}

                    if isinstance(fallback_response, dict) and fallback_response.get(
                        "success", False
                    ):
                        fallback_response.setdefault("meta", {})
                        if isinstance(fallback_response["meta"], dict):
                            fallback_response["meta"].update(
                                {
                                    "primary_provider": provider.name,
                                    "provider": fallback_provider.name,
                                    "fallback_from": provider.name,
                                    "fallback_reason": first_error,
                                    "providers_attempted": attempted,
                                }
                            )
                            if fallback_errors:
                                fallback_response["meta"]["fallback_errors"] = fallback_errors
                        response_data = fallback_response
                        fallback_succeeded = True
                        break

                    fallback_errors[fallback_provider.name] = _web_search_error_message(
                        fallback_response
                    )
                    if not _web_search_should_try_fallback(fallback_response):
                        break

                if primary_exception is not None and not fallback_succeeded:
                    raise primary_exception

                if (
                    len(attempted) > 1
                    and isinstance(response_data, dict)
                    and not response_data.get("success", False)
                ):
                    response_data.setdefault("meta", {})
                    if isinstance(response_data["meta"], dict):
                        response_data["meta"].update(
                            {
                                "primary_provider": provider.name,
                                "provider": provider.name,
                                "providers_attempted": attempted,
                                "fallback_errors": fallback_errors,
                            }
                        )
            elif primary_exception is not None:
                raise primary_exception

        if not isinstance(response_data, dict):
            response_data = {
                "success": False,
                "error": _web_search_error_message(
                    response_data,
                    default="web search returned an invalid response",
                ),
            }

        debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        debug_call_data["final_response_size"] = len(result_json)
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()
        return result_json

    except Exception as e:
        error_msg = f"Error searching web: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("web_search_tool", debug_call_data)
        _debug.save()

        return tool_error(error_msg)


def _normalize_matrix_queries(queries: Any) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """Normalize matrix-search query input into labeled query objects."""
    if isinstance(queries, str):
        raw_items = [line.strip() for line in queries.splitlines() if line.strip()]
    elif isinstance(queries, list):
        raw_items = queries
    else:
        return [], "queries must be a list of strings/objects or newline-delimited text"

    normalized: List[Dict[str, str]] = []
    for item in raw_items:
        if isinstance(item, str):
            query = item.strip()
            label = query
        elif isinstance(item, dict):
            query = str(item.get("query") or item.get("q") or "").strip()
            label = str(item.get("label") or item.get("id") or query).strip()
        else:
            return [], "each query must be a string or an object with a query field"
        if query:
            normalized.append({"label": label or query, "query": query})

    if not normalized:
        return [], "at least one non-empty query is required"
    return normalized[:12], None


def _coerce_matrix_provider_list(value: Any) -> List[str]:
    """Normalize caller-supplied provider names without logging secrets."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = value
    else:
        return []

    normalized: List[str] = []
    for item in raw_items:
        name = str(item or "").strip().lower()
        if name and name not in normalized:
            normalized.append(name)
    return normalized[:12]


_MATRIX_PROVIDER_ENV_KEYS = {
    "exa": ("EXA_API_KEY",),
    "parallel": ("PARALLEL_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
    "firecrawl": ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "FIRECRAWL_GATEWAY_URL"),
    "searxng": ("SEARXNG_URL",),
    "brave-free": ("BRAVE_SEARCH_API_KEY",),
    "ddgs": (),
}
_MATRIX_PROVIDER_ORDER = (
    "firecrawl",
    "parallel",
    "tavily",
    "exa",
    "searxng",
    "brave-free",
    "ddgs",
)


def _ensure_web_search_plugins_registered() -> None:
    """Best-effort load of bundled web provider plugins for direct callers."""
    try:
        from agent.web_search_registry import list_providers

        if list_providers():
            return
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception as exc:  # noqa: BLE001 - matrix can still report no providers
        logger.debug("Could not ensure web provider plugins are registered: %s", exc)


def _provider_present_env_keys(provider: str) -> List[str]:
    return [key for key in _MATRIX_PROVIDER_ENV_KEYS.get(provider, ()) if _has_env(key)]


def _provider_matrix_status() -> Dict[str, Any]:
    """Return registry-backed provider availability without network calls."""
    _ensure_web_search_plugins_registered()
    try:
        from agent.web_search_registry import get_provider, list_providers
    except Exception:
        return {
            "available_providers": [],
            "missing_providers": list(_MATRIX_PROVIDER_ORDER),
            "providers": {},
        }

    registered = {provider.name for provider in list_providers()}
    ordered = list(_MATRIX_PROVIDER_ORDER)
    ordered.extend(sorted(registered - set(ordered)))

    available: List[str] = []
    missing: List[str] = []
    providers: Dict[str, Dict[str, Any]] = {}
    for name in ordered:
        provider = get_provider(name)
        supports_search = bool(provider and provider.supports_search())
        try:
            provider_available = bool(provider and supports_search and provider.is_available())
        except Exception as exc:  # noqa: BLE001
            provider_available = False
            availability_error = str(exc)
        else:
            availability_error = None
        entry: Dict[str, Any] = {
            "registered": provider is not None,
            "supports_search": supports_search,
            "available": provider_available,
            "required_env_keys": list(_MATRIX_PROVIDER_ENV_KEYS.get(name, ())),
            "present_env_keys": _provider_present_env_keys(name),
        }
        if availability_error:
            entry["availability_error"] = availability_error
        providers[name] = entry
        if provider_available:
            available.append(name)
        else:
            missing.append(name)

    return {
        "available_providers": available,
        "missing_providers": missing,
        "providers": providers,
    }


def _canonicalize_matrix_result_url(url: str) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url.strip())
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))
    except Exception:
        return url.strip()


def _normalize_provider_matrix_results(
    provider: str, response_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    results = (
        response_data.get("data", {}).get("web", [])
        if isinstance(response_data, dict)
        else []
    )
    normalized: List[Dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        normalized.append(
            {
                "provider": provider,
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "position": int(result.get("position") or index + 1),
            }
        )
    return normalized


def _web_provider_matrix_search(
    query: str,
    limit: int = 5,
    providers: Any = None,
    required_providers: Any = None,
) -> Dict[str, Any]:
    """Search one query across all requested registered web providers."""
    query = str(query or "").strip()
    if not query:
        return {"success": False, "error": "query is required for provider matrix search"}
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 10)

    status = _provider_matrix_status()
    requested = _coerce_matrix_provider_list(providers)
    if not requested or "all" in requested:
        requested = [
            name
            for name, entry in status.get("providers", {}).items()
            if entry.get("registered")
        ]

    required = set(_coerce_matrix_provider_list(required_providers))
    if not requested:
        return {
            "success": False,
            "coverage_status": "failed",
            "degraded_coverage": True,
            "error": "No configured web search providers are available for provider matrix search.",
            "provider_status": status,
            "requested_providers": requested,
            "providers": [],
        }

    try:
        from agent.web_search_registry import get_provider
        from tools.interrupt import is_interrupted
    except Exception as exc:
        return {"success": False, "error": f"Web provider registry unavailable: {exc}"}

    provider_results: Dict[str, Dict[str, Any]] = {}
    provider_reports: List[Dict[str, Any]] = []
    fused_index: Dict[str, Dict[str, Any]] = {}
    required_failures: List[str] = []

    for provider_name in requested:
        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        provider = get_provider(provider_name)
        if provider is None:
            provider_results[provider_name] = {
                "success": False,
                "result_count": 0,
                "error": f"Web search provider {provider_name!r} is not registered.",
                "results": [],
            }
            provider_reports.append({
                "provider": provider_name,
                "status": "unavailable",
                "error": "provider is not registered",
                "results": [],
            })
            if provider_name in required:
                required_failures.append(provider_name)
            continue

        display_name = getattr(provider, "display_name", provider_name)
        try:
            supports_search = bool(provider.supports_search())
        except Exception:
            supports_search = False
        if not supports_search:
            provider_results[provider_name] = {
                "success": False,
                "result_count": 0,
                "error": "provider does not support search",
                "results": [],
            }
            provider_reports.append({
                "provider": provider_name,
                "display_name": display_name,
                "status": "unsupported",
                "error": "provider does not support search",
                "results": [],
            })
            if provider_name in required:
                required_failures.append(provider_name)
            continue

        provider_status = status.get("providers", {}).get(provider_name, {})
        if not provider_status.get("available"):
            provider_results[provider_name] = {
                "success": False,
                "result_count": 0,
                "error": "provider credentials or endpoint are unavailable",
                "results": [],
            }
            provider_reports.append({
                "provider": provider_name,
                "display_name": display_name,
                "status": "unavailable",
                "error": "provider credentials or endpoint are unavailable",
                "results": [],
            })
            if provider_name in required:
                required_failures.append(provider_name)
            continue

        try:
            response_data = provider.search(query, limit=limit)
            provider_success = bool(
                isinstance(response_data, dict) and response_data.get("success", False)
            )
            normalized = _normalize_provider_matrix_results(provider_name, response_data)
            provider_results[provider_name] = {
                "success": provider_success,
                "result_count": len(normalized),
                "results": normalized,
            }
            if not provider_success:
                error = (
                    response_data.get("error") if isinstance(response_data, dict) else "provider search failed"
                )
                provider_results[provider_name]["error"] = error
                provider_reports.append({
                    "provider": provider_name,
                    "display_name": display_name,
                    "status": "error",
                    "error": str(error or "search failed"),
                    "results": [],
                })
                if provider_name in required:
                    required_failures.append(provider_name)
                continue
            provider_reports.append({
                "provider": provider_name,
                "display_name": display_name,
                "status": "ok",
                "result_count": len(normalized),
                "results": normalized,
            })
            for item in normalized:
                key = (
                    _canonicalize_matrix_result_url(item["url"])
                    or item["url"]
                    or f"{provider_name}:{item['position']}"
                )
                entry = fused_index.setdefault(
                    key,
                    {
                        "url": item["url"],
                        "title": item["title"],
                        "description": item["description"],
                        "providers": [],
                        "positions": {},
                    },
                )
                if item["provider"] not in entry["providers"]:
                    entry["providers"].append(item["provider"])
                entry["positions"][item["provider"]] = item["position"]
                if not entry.get("title") and item["title"]:
                    entry["title"] = item["title"]
                if not entry.get("description") and item["description"]:
                    entry["description"] = item["description"]
        except Exception as exc:
            provider_results[provider_name] = {
                "success": False,
                "result_count": 0,
                "error": str(exc),
                "results": [],
            }
            provider_reports.append({
                "provider": provider_name,
                "display_name": display_name,
                "status": "error",
                "error": f"search failed: {type(exc).__name__}: {exc}",
                "results": [],
            })
            if provider_name in required:
                required_failures.append(provider_name)

    fused_results = []
    for entry in fused_index.values():
        provider_hits = len(entry["providers"])
        avg_position = sum(entry["positions"].values()) / max(1, provider_hits)
        fused_results.append(
            {
                "url": entry["url"],
                "title": entry["title"],
                "description": entry["description"],
                "providers": sorted(entry["providers"]),
                "provider_hits": provider_hits,
                "positions": entry["positions"],
                "position": min(entry["positions"].values()) if entry["positions"] else None,
                "_avg_position": avg_position,
            }
        )
    fused_results.sort(
        key=lambda item: (
            -int(item.get("provider_hits") or 0),
            float(item.get("_avg_position") or 999.0),
            str(item.get("title") or ""),
        )
    )

    trimmed_results = []
    for item in fused_results[:limit]:
        payload = dict(item)
        payload.pop("_avg_position", None)
        trimmed_results.append(payload)

    successful = [
        provider
        for provider, result in provider_results.items()
        if result.get("success")
    ]
    failed_or_unavailable = [
        report for report in provider_reports if report.get("status") != "ok"
    ]
    coverage_status = "complete"
    if not successful:
        coverage_status = "failed"
    elif failed_or_unavailable:
        coverage_status = "degraded"

    response: Dict[str, Any] = {
        "success": bool(successful) and not required_failures,
        "query": query,
        "strategy": "provider_matrix",
        "coverage_status": coverage_status,
        "degraded_coverage": coverage_status != "complete",
        "data": {"web": trimmed_results},
        "provider_status": status,
        "providers_requested": len(provider_reports),
        "providers_succeeded": len(successful),
        "providers_used": successful,
        "providers_missing": [item["provider"] for item in failed_or_unavailable],
        "providers": provider_reports,
        "provider_results": provider_results,
    }
    if required:
        response["required_providers"] = sorted(required)
    if not successful:
        response["error"] = "All requested web search providers failed."
    if required_failures:
        response["error"] = (
            "Required search providers unavailable or failed: "
            + ", ".join(sorted(set(required_failures)))
        )
    return response


def web_search_matrix_tool(
    queries: Any = None,
    limit_per_query: int = 3,
    *,
    query: Any = None,
    limit: Any = None,
    providers: Any = None,
    required_providers: Any = None,
) -> str:
    """Run a bounded matrix of web searches and return per-query + deduped results."""
    if query is None and isinstance(queries, str):
        query = queries
        queries = None

    if query is not None or providers or required_providers:
        response_data = _web_provider_matrix_search(
            query=query,
            limit=limit if limit is not None else limit_per_query,
            providers=providers,
            required_providers=required_providers,
        )
        return json.dumps(response_data, indent=2, ensure_ascii=False)

    normalized, error = _normalize_matrix_queries(queries)
    if error:
        return tool_error(error, success=False)

    try:
        limit = int(limit_per_query)
    except (TypeError, ValueError):
        limit = 3
    limit = min(max(limit, 1), 10)

    matrix = []
    deduped = []
    seen_urls = set()
    failures = []

    for item in normalized:
        query = item["query"]
        raw = web_search_tool(query, limit=limit)
        try:
            payload = json.loads(raw)
        except Exception as exc:
            payload = {"success": False, "error": f"Invalid web_search response: {exc}"}

        results = []
        if isinstance(payload, dict):
            results = ((payload.get("data") or {}).get("web") or [])
            if not payload.get("success", False):
                failures.append({
                    "label": item["label"],
                    "query": query,
                    "error": payload.get("error") or "web_search failed",
                })

        for result in results:
            url = str(result.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append({
                "query_label": item["label"],
                "query": query,
                **result,
            })

        matrix.append({
            "label": item["label"],
            "query": query,
            "success": bool(isinstance(payload, dict) and payload.get("success", False)),
            "results_count": len(results),
            "results": results,
            "error": payload.get("error") if isinstance(payload, dict) else None,
        })

    response_data = {
        "success": not failures,
        "data": {
            "matrix": matrix,
            "deduped_web": deduped,
        },
        "meta": {
            "query_count": len(normalized),
            "limit_per_query": limit,
            "configured_backend": _get_search_backend(),
        },
    }
    if failures:
        response_data["errors"] = failures
    return json.dumps(response_data, indent=2, ensure_ascii=False)


async def web_extract_tool(
    urls: List[str],
    format: str = None,
    char_limit: Optional[int] = None,
) -> str:
    """
    Extract content from specific web pages using available extraction API backend.

    Returns clean page content (markdown/text) with NO LLM summarization. The
    extract backends (Firecrawl, Tavily, Exa, Parallel) already return clean,
    boilerplate-stripped content, so we return it directly and fast. Pages over
    ``char_limit`` are head+tail truncated with an explicit footer; the full
    text is stored under cache/web and the footer tells the model how to
    read_file the omitted middle. Inline base64 images are replaced with
    ``[IMAGE: alt]`` placeholders (real image URLs are preserved as links).

    Args:
        urls (List[str]): List of URLs to extract content from
        format (str): Desired output format ("markdown" or "html", optional)
        char_limit (Optional[int]): Per-page char budget sent to the model
            (default: web.extract_char_limit or 15000). Larger pages truncate.

    Security: URLs are checked for embedded secrets before fetching.

    Returns:
        str: JSON string with a ``results`` list; each entry has
             ``url``, ``title``, ``content``, ``error``. ``content`` is the
             (possibly truncated) clean page text.

    Raises:
        Exception: If extraction fails or API key is not set
    """
    # Block URLs containing embedded secrets (exfiltration prevention).
    # URL-decode first so percent-encoded secrets (%73k- = sk-) are caught.
    from agent.redact import _PREFIX_RE
    from urllib.parse import unquote
    normalized_urls: List[str] = []
    for _url in urls:
        normalized_url = normalize_url_for_request(_url)
        if (
            _PREFIX_RE.search(_url)
            or _PREFIX_RE.search(unquote(_url))
            or _PREFIX_RE.search(normalized_url)
            or _PREFIX_RE.search(unquote(normalized_url))
        ):
            return json.dumps({
                "success": False,
                "error": "Blocked: URL contains what appears to be an API key or token. "
                         "Secrets must not be sent in URLs.",
            })
        sensitive_query_key = sensitive_query_param_name(normalized_url)
        if sensitive_query_key:
            return json.dumps({
                "success": False,
                "error": (
                    "Blocked: URL contains a credential-like query parameter "
                    f"({sensitive_query_key}). Web extract backends are third-party "
                    "readers; remove the sensitive query parameter or use a local "
                    "browser session when this access is explicitly required."
                ),
            })
        normalized_urls.append(normalized_url)

    debug_call_data = {
        "parameters": {
            "urls": normalized_urls,
            "format": format,
            "char_limit": char_limit,
        },
        "error": None,
        "pages_extracted": 0,
        "pages_truncated": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "truncation_metrics": [],
        "processing_applied": []
    }

    try:
        logger.info("Extracting content from %d URL(s)", len(normalized_urls))

        # ── SSRF protection — filter out private/internal URLs before any backend ──
        safe_urls = []
        ssrf_blocked: List[Dict[str, Any]] = []
        for url in normalized_urls:
            if not await async_is_safe_url(url):
                ssrf_blocked.append({
                    "url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
            else:
                safe_urls.append(url)

        extract_provider_name = None
        extract_degradations: List[Dict[str, Any]] = []

        # Dispatch only safe URLs to the configured backend
        if not safe_urls:
            results = []
        else:
            backend = _get_extract_backend()

            # All seven providers (brave-free, ddgs, searxng, exa, parallel,
            # tavily, firecrawl) now live as plugins. The dispatcher is a
            # registry lookup + delegation. Some providers' extract() is
            # async (parallel, firecrawl), others sync (exa, tavily) — we
            # detect coroutine functions and await; sync functions run
            # inline (the policy gate, SSRF re-check, etc. live inside the
            # provider itself for the firecrawl per-URL loop).
            _ensure_web_plugins_loaded()
            from agent.web_search_registry import (
                get_active_extract_provider,
                get_provider as _wsp_get_provider,
            )

            provider = _wsp_get_provider(backend) if backend else None
            if provider is None or not provider.supports_extract():
                # When the configured name IS registered but doesn't support
                # extract (search-only providers like brave-free / ddgs /
                # searxng), surface that as a typed "search-only" error
                # rather than silently switching backends. When the name
                # isn't registered at all (typo / uninstalled plugin), fall
                # through to the active-provider walk.
                if provider is not None and not provider.supports_extract():
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"{provider.display_name} is a search-only "
                                "backend and cannot extract URL content. "
                                "Set web.extract_backend to firecrawl, "
                                "tavily, exa, or parallel."
                            ),
                        },
                        ensure_ascii=False,
                    )
                provider = get_active_extract_provider()
                if provider is None:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                "No web extract provider configured. "
                                "Set web.extract_backend to firecrawl, "
                                "tavily, exa, or parallel."
                            ),
                        },
                        ensure_ascii=False,
                    )

            logger.info(
                "Web extract via %s: %d URL(s)", provider.name, len(safe_urls)
            )
            extract_provider_name = provider.name

            # Async-or-sync dispatch: parallel + firecrawl have async
            # extract(); exa + tavily are sync.
            import inspect
            if inspect.iscoroutinefunction(provider.extract):
                results = await provider.extract(safe_urls, format=format)
            else:
                # Run sync extract() in a thread so we don't block the
                # event loop on network I/O.
                results = await asyncio.to_thread(
                    provider.extract, safe_urls, format=format
                )

            if not isinstance(results, list):
                results = []
            results, extract_degradations = await _apply_web_extract_degradation_fallback(
                provider_name=extract_provider_name or "",
                results=results,
                format=format,
            )

        # Merge any SSRF-blocked results back in
        if ssrf_blocked:
            results = ssrf_blocked + results

        response = {"results": results}

        pages_extracted = len(response.get('results', []))
        logger.info("Extracted content from %d pages", pages_extracted)

        debug_call_data["pages_extracted"] = pages_extracted
        debug_call_data["original_response_size"] = len(json.dumps(response))

        effective_char_limit = char_limit if char_limit is not None else _get_extract_char_limit()
        try:
            effective_char_limit = max(2000, min(int(effective_char_limit), 500_000))
        except (TypeError, ValueError):
            effective_char_limit = DEFAULT_EXTRACT_CHAR_LIMIT

        # Truncate-and-store: no LLM. For each result, convert inline base64
        # images to labeled placeholders (keeping alt text + real image URLs),
        # then return the clean content directly if within budget, or a
        # head+tail window plus a footer pointing at the stored full text.
        debug_call_data["processing_applied"].append("truncate_and_store")
        for result in response.get("results", []):
            if result.get("error"):
                continue
            url = result.get("url", "")
            raw_content = result.get("raw_content", "") or result.get("content", "")
            if not raw_content:
                continue
            clean = convert_base64_images_to_links(raw_content)
            model_text, truncated = _truncate_with_footer(clean, url, effective_char_limit)
            result["content"] = model_text
            if truncated:
                debug_call_data["pages_truncated"] += 1
                debug_call_data["truncation_metrics"].append({
                    "url": url,
                    "original_size": len(clean),
                    "sent_size": len(model_text),
                })
                logger.info("%s (truncated %d -> %d chars)", url, len(clean), len(model_text))
            else:
                logger.info("%s (%d chars, whole)", url, len(clean))

        # Trim output to minimal fields per entry: title, content, error
        trimmed_results = []
        for r in response.get("results", []):
            trimmed = {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
            }
            if "provider_status" in r:
                trimmed["provider_status"] = r["provider_status"]
            if "blocked_by_policy" in r:
                trimmed["blocked_by_policy"] = r["blocked_by_policy"]
            if "degradation" in r:
                trimmed["degradation"] = r["degradation"]
            trimmed_results.append(trimmed)
        trimmed_response = {"results": trimmed_results}
        provider_status = next(
            (
                r.get("provider_status")
                for r in response.get("results", [])
                if isinstance(r.get("provider_status"), dict)
            ),
            None,
        )
        if provider_status:
            trimmed_response["provider_status"] = provider_status
        if extract_provider_name or extract_degradations:
            trimmed_response["meta"] = {
                "extract_provider": extract_provider_name,
            }
            if extract_degradations:
                trimmed_response["meta"]["degradations"] = extract_degradations

        if trimmed_response.get("results") == []:
            result_json = tool_error("Content was inaccessible or not found")
        else:
            result_json = json.dumps(trimmed_response, indent=2, ensure_ascii=False)

        # base64 images were already converted to placeholders per-result above;
        # this is a belt-and-suspenders sweep over the serialized JSON in case a
        # provider tucked a blob somewhere unexpected (e.g. metadata).
        cleaned_result = convert_base64_images_to_links(result_json)

        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_conversion")

        # Log debug information
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()

        return cleaned_result

    except Exception as e:
        error_msg = f"Error extracting content: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("web_extract_tool", debug_call_data)
        _debug.save()

        return tool_error(error_msg)


async def web_crawl_tool(
    url: str,
    instructions: str = None,
    depth: str = "basic",
    use_llm_processing: bool = True,
    model: Optional[str] = None,
    min_length: int = DEFAULT_MIN_LENGTH_FOR_SUMMARIZATION
) -> str:
    """
    Crawl a website with specific instructions using available crawling API backend.

    This function provides a generic interface for web crawling that can work
    with multiple backends. Currently uses Firecrawl.

    Args:
        url (str): The base URL to crawl (can include or exclude https://)
        instructions (str): Instructions for what to crawl/extract using LLM intelligence (optional)
        depth (str): Depth of extraction ("basic" or "advanced", default: "basic")
        use_llm_processing (bool): Whether to process content with LLM for summarization (default: True)
        model (Optional[str]): The model to use for LLM processing (defaults to current auxiliary backend model)
        min_length (int): Minimum content length to trigger LLM processing (default: 5000)

    Returns:
        str: JSON string containing crawled content. If LLM processing is enabled and successful,
             the 'content' field will contain the processed markdown summary instead of raw content.
             Each page is processed individually.

    Raises:
        Exception: If crawling fails or API key is not set
    """
    debug_call_data = {
        "parameters": {
            "url": url,
            "instructions": instructions,
            "depth": depth,
            "use_llm_processing": use_llm_processing,
            "model": model,
            "min_length": min_length
        },
        "error": None,
        "pages_crawled": 0,
        "pages_processed_with_llm": 0,
        "original_response_size": 0,
        "final_response_size": 0,
        "compression_metrics": [],
        "processing_applied": []
    }

    try:
        effective_model = model or _get_default_summarizer_model()
        auxiliary_available = check_auxiliary_model()
        backend = _get_backend()

        # Tavily (and any future plugin advertising supports_crawl=True)
        # dispatches through agent.web_search_registry. The crawl response
        # shape — {"results": [{"url", "title", "content", ...}]} — is then
        # post-processed by the shared LLM-summarization path below.
        from agent.web_search_registry import (
            get_active_crawl_provider,
            get_provider as _wsp_get_provider,
        )

        crawl_provider = _wsp_get_provider(backend) if backend else None
        if crawl_provider is not None and not crawl_provider.supports_crawl():
            # When the configured provider is search-only AND cannot
            # extract URLs either (brave-free / ddgs / searxng), surface a
            # typed "search-only" error rather than silently switching to
            # a different crawl backend. When the provider supports extract
            # but not crawl (e.g. firecrawl), fall through to the legacy
            # firecrawl-via-extract path below.
            if not crawl_provider.supports_extract():
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"{crawl_provider.display_name} is a search-only "
                            "backend and cannot crawl URLs. "
                            "Set FIRECRAWL_API_KEY for crawling, or use "
                            "web_search instead."
                        ),
                    },
                    ensure_ascii=False,
                )
            crawl_provider = None  # let legacy firecrawl path handle it
        if crawl_provider is None:
            crawl_provider = get_active_crawl_provider()

        # Mirror main's upstream availability gate: when the resolved
        # provider is configured-but-unavailable (e.g. firecrawl without
        # FIRECRAWL_API_KEY), short-circuit BEFORE we dispatch so the
        # error envelope matches the legacy top-level shape
        # ``{"success": False, "error": "..."}`` rather than burying the
        # configuration message inside a per-page ``results[]`` entry.
        if crawl_provider is not None and not crawl_provider.is_available():
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "web_crawl requires Firecrawl. Set FIRECRAWL_API_KEY, "
                        f"FIRECRAWL_API_URL{_firecrawl_backend_help_suffix()}, "
                        "or use web_search + web_extract instead."
                    ),
                },
                ensure_ascii=False,
            )

        if crawl_provider is not None:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = f'https://{url}'

            # SSRF protection — block private/internal addresses
            if not is_safe_url(url):
                return json.dumps({"results": [{"url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address"}]}, ensure_ascii=False)

            # Website policy check
            blocked = check_website_access(url)
            if blocked:
                logger.info("Blocked web_crawl for %s by rule %s", blocked["host"], blocked["rule"])
                return json.dumps({"results": [{"url": url, "title": "", "content": "", "error": blocked["message"],
                    "blocked_by_policy": {"host": blocked["host"], "rule": blocked["rule"], "source": blocked["source"]}}]}, ensure_ascii=False)

            from tools.interrupt import is_interrupted as _is_int
            if _is_int():
                return tool_error("Interrupted", success=False)

            logger.info("Web crawl via %s: %s", crawl_provider.name, url)

            # Async-or-sync dispatch — Tavily's crawl is sync, but a future
            # async-crawl provider works transparently.
            import inspect
            crawl_kwargs = {"depth": depth, "limit": 20}
            if instructions:
                crawl_kwargs["instructions"] = instructions

            if inspect.iscoroutinefunction(crawl_provider.crawl):
                response = await crawl_provider.crawl(url, **crawl_kwargs)
            else:
                response = await asyncio.to_thread(
                    crawl_provider.crawl, url, **crawl_kwargs
                )

            # Provider returns {"results": [...]} matching what the shared
            # LLM post-processing below expects.
            if not isinstance(response, dict):
                response = {"results": []}
            response.setdefault("results", [])

            # Fall through to the shared LLM processing and trimming below
            # (skip the Firecrawl-specific crawl logic)
            pages_crawled = len(response.get('results', []))
            logger.info("Crawled %d pages", pages_crawled)
            debug_call_data["pages_crawled"] = pages_crawled
            debug_call_data["original_response_size"] = len(json.dumps(response))

            # Process each result with LLM if enabled
            if use_llm_processing and auxiliary_available:
                logger.info("Processing crawled content with LLM (parallel)...")
                debug_call_data["processing_applied"].append("llm_processing")

                async def _process_tavily_crawl(result):
                    page_url = result.get('url', 'Unknown URL')
                    title = result.get('title', '')
                    content = result.get('content', '')
                    if not content:
                        return result, None, "no_content"
                    original_size = len(content)
                    processed = await process_content_with_llm(content, page_url, title, effective_model, min_length)
                    if processed:
                        result['raw_content'] = content
                        result['content'] = processed
                        metrics = {"url": page_url, "original_size": original_size, "processed_size": len(processed),
                                   "compression_ratio": len(processed) / original_size if original_size else 1.0, "model_used": effective_model}
                        return result, metrics, "processed"
                    metrics = {"url": page_url, "original_size": original_size, "processed_size": original_size,
                               "compression_ratio": 1.0, "model_used": None, "reason": "content_too_short"}
                    return result, metrics, "too_short"

                tasks = [_process_tavily_crawl(r) for r in response.get('results', [])]
                # Use return_exceptions=True so a single task failure does not
                # discard all other successfully processed crawl results.
                processed_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result_item in processed_results:
                    if isinstance(result_item, BaseException):
                        logger.warning("Tavily crawl processing task failed: %s", result_item)
                        continue
                    result, metrics, status = result_item
                    if status == "processed":
                        debug_call_data["compression_metrics"].append(metrics)
                        debug_call_data["pages_processed_with_llm"] += 1

            if use_llm_processing and not auxiliary_available:
                logger.warning("LLM processing requested but no auxiliary model available, returning raw content")
                debug_call_data["processing_applied"].append("llm_processing_unavailable")

            trimmed_results = [{"url": r.get("url", ""), "title": r.get("title", ""), "content": r.get("content", ""), "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {})} for r in response.get("results", [])]
            result_json = json.dumps({"results": trimmed_results}, indent=2, ensure_ascii=False)
            cleaned_result = clean_base64_images(result_json)
            debug_call_data["final_response_size"] = len(cleaned_result)
            _debug.log_call("web_crawl_tool", debug_call_data)
            _debug.save()
            return cleaned_result

        # No registered provider supports crawl AND no crawl-capable plugin
        # is available. Surface a typed error pointing the user at the two
        # crawl-capable providers (Firecrawl + Tavily).
        return json.dumps(
            {
                "success": False,
                "error": (
                    "web_crawl has no available backend. "
                    "Set FIRECRAWL_API_KEY (or FIRECRAWL_API_URL for "
                    f"self-hosted){_firecrawl_backend_help_suffix()}, "
                    "or set TAVILY_API_KEY for Tavily. "
                    "Alternatively use web_search + web_extract instead."
                ),
            },
            ensure_ascii=False,
        )

    except Exception as e:
        error_msg = f"Error crawling website: {str(e)}"
        logger.debug("%s", error_msg)

        debug_call_data["error"] = error_msg
        _debug.log_call("web_crawl_tool", debug_call_data)
        _debug.save()

        return tool_error(error_msg)


# Convenience function to check Firecrawl credentials
def check_web_api_key() -> bool:
    """Check whether the configured web backend is available.

    Used as the ``check_fn`` gate for the ``web_search`` and ``web_extract``
    tool registry entries — so a plugin-registered provider that reports
    ``is_available()`` must light the tools up even when no built-in backend
    has credentials (issues #28651, #31873). Resolution funnels through
    :func:`_is_backend_available`, which delegates non-legacy names to the
    registry.
    """
    configured = _load_web_config().get("backend", "").lower().strip()
    if configured and _is_backend_available(configured):
        return True
    # Any built-in backend with credentials present. This is a boolean OR, so
    # unlike _get_backend() the probe order is irrelevant.
    if any(_is_backend_available(backend) for backend in _LEGACY_WEB_BACKENDS):
        return True
    # Any plugin-registered provider the registry considers active for either
    # capability. Delegating to the registry's own availability-filtered
    # resolvers keeps a single authority for "is a custom provider usable"
    # rather than re-implementing the walk here.
    try:
        from agent.web_search_registry import (
            get_active_search_provider,
            get_active_extract_provider,
        )

        return (
            get_active_search_provider() is not None
            or get_active_extract_provider() is not None
        )
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry availability check failed: %s", exc)
        return False


def check_auxiliary_model() -> bool:
    """Check if an auxiliary text model is available for LLM content processing."""
    client, _, _ = _resolve_web_extract_auxiliary()
    return client is not None


_WEB_PROVIDER_CONFIG_VARS = {
    "brave-free": ("BRAVE_SEARCH_API_KEY",),
    "ddgs": (),
    "exa": ("EXA_API_KEY",),
    "firecrawl": (
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_USER_TOKEN",
    ),
    "parallel": ("PARALLEL_API_KEY",),
    "searxng": ("SEARXNG_URL",),
    "tavily": ("TAVILY_API_KEY",),
}
_WEB_MATRIX_CAPABILITIES = ("search", "extract", "crawl")
_WEB_MATRIX_SEARCH_PROVIDERS = (
    "all",
    "brave-free",
    "ddgs",
    "exa",
    "firecrawl",
    "parallel",
    "searxng",
    "tavily",
)


def _normalize_matrix_items(value: Any, *, allowed: Optional[set[str]] = None) -> list[str]:
    """Normalize tool args that may arrive as a string or list."""
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(part).strip() for part in value]
    else:
        raw_items = [str(value).strip()]

    normalized: list[str] = []
    for item in raw_items:
        lowered = item.lower()
        if not lowered:
            continue
        if allowed is not None and lowered not in allowed:
            continue
        if lowered not in normalized:
            normalized.append(lowered)
    return normalized


def _safe_provider_bool(provider: Any, method_name: str) -> bool:
    try:
        method = getattr(provider, method_name)
        return bool(method())
    except Exception:
        return False


def _provider_config_presence(provider_name: str) -> dict[str, list[str]]:
    env_names = _WEB_PROVIDER_CONFIG_VARS.get(provider_name, ())
    present = [name for name in env_names if _has_env(name)]
    missing = [name for name in env_names if name not in present]
    return {"present": present, "missing": missing}


def _provider_matrix_row(provider: Any) -> dict[str, Any]:
    name = str(getattr(provider, "name", "") or "").strip().lower()
    capabilities = {
        "search": _safe_provider_bool(provider, "supports_search"),
        "extract": _safe_provider_bool(provider, "supports_extract"),
        "crawl": _safe_provider_bool(provider, "supports_crawl"),
    }
    available = _safe_provider_bool(provider, "is_available")
    config = _provider_config_presence(name)
    return {
        "name": name,
        "display_name": str(getattr(provider, "display_name", name) or name),
        "available": available,
        "configured": bool(config["present"]) or available,
        "capabilities": capabilities,
        "config": config,
    }


def _configured_browser_cloud_provider() -> str:
    try:
        from hermes_cli.config import load_config

        browser_cfg = (load_config() or {}).get("browser", {})
        if isinstance(browser_cfg, dict):
            return str(browser_cfg.get("cloud_provider") or "").strip().lower()
    except Exception as exc:
        logger.debug("Could not read browser cloud provider config: %s", exc)
    return ""


def _firecrawl_surface_status(providers_by_name: dict[str, dict[str, Any]]) -> dict[str, Any]:
    provider = providers_by_name.get("firecrawl") or {}
    capabilities = provider.get("capabilities") if isinstance(provider, dict) else {}
    available = bool(provider.get("available")) if isinstance(provider, dict) else False
    browser_provider = _configured_browser_cloud_provider()
    browser_key_present = _has_env("FIRECRAWL_API_KEY")
    return {
        "search": available and bool(capabilities.get("search")),
        "scrape": available and bool(capabilities.get("extract")),
        "extract": available and bool(capabilities.get("extract")),
        "crawl": available and bool(capabilities.get("crawl")),
        "interact": browser_provider == "firecrawl" and browser_key_present,
        "browser_cloud_provider": browser_provider or None,
        "config": {
            "present": [name for name in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL") if _has_env(name)],
            "missing": [name for name in ("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL") if not _has_env(name)],
        },
    }


def _matrix_provider_status(providers: list[dict[str, Any]]) -> dict[str, Any]:
    available = sorted(
        row["name"]
        for row in providers
        if row.get("available")
    )
    providers_by_name = {
        row["name"]: {
            "available": row.get("available", False),
            "capabilities": row.get("capabilities", {}),
            "required_env_keys": list(_WEB_PROVIDER_CONFIG_VARS.get(row["name"], ())),
            "present_env_keys": row.get("config", {}).get("present", []),
            "missing_env_keys": row.get("config", {}).get("missing", []),
        }
        for row in providers
    }
    return {
        "available_providers": available,
        "missing_providers": sorted(
            row["name"]
            for row in providers
            if not row.get("available")
        ),
        "providers": providers_by_name,
    }


def _canonicalize_matrix_result_url(url: str) -> str:
    if not url:
        return ""
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url.strip())
    except Exception:
        return url.strip()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, parts.query, "")
    )


def _normalize_matrix_search_results(
    provider: str,
    response_data: dict[str, Any],
) -> list[dict[str, Any]]:
    results = (
        response_data.get("data", {}).get("web", [])
        if isinstance(response_data, dict)
        else []
    )
    normalized: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            continue
        normalized.append(
            {
                "provider": provider,
                "url": str(result.get("url") or ""),
                "title": str(result.get("title") or ""),
                "description": str(result.get("description") or ""),
                "position": int(result.get("position") or index + 1),
            }
        )
    return normalized


def _run_web_search_matrix_query(
    query: str,
    limit: int,
    providers: list[dict[str, Any]],
    requested_providers: list[str],
) -> dict[str, Any]:
    from agent.web_search_registry import get_provider

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 5
    limit = min(max(limit, 1), 10)

    providers_by_name = {row["name"]: row for row in providers}
    if not requested_providers or "all" in requested_providers:
        requested = [
            row["name"]
            for row in providers
            if row.get("available") and row.get("capabilities", {}).get("search")
        ]
    else:
        requested = [
            provider
            for provider in requested_providers
            if provider in providers_by_name and provider != "all"
        ]

    available_requested = [
        provider
        for provider in requested
        if providers_by_name.get(provider, {}).get("available")
        and providers_by_name.get(provider, {}).get("capabilities", {}).get("search")
    ]
    missing_requested = [
        provider for provider in requested if provider not in available_requested
    ]

    provider_results: dict[str, dict[str, Any]] = {}
    fused_index: dict[str, dict[str, Any]] = {}

    for provider_name in available_requested:
        provider = get_provider(provider_name)
        if provider is None:
            provider_results[provider_name] = {
                "success": False,
                "result_count": 0,
                "error": f"Web search provider {provider_name!r} is not registered.",
                "results": [],
            }
            continue
        try:
            response_data = provider.search(query, limit=limit)
            normalized = _normalize_matrix_search_results(provider_name, response_data)
            provider_results[provider_name] = {
                "success": bool(response_data.get("success", True)),
                "result_count": len(normalized),
                "results": normalized,
            }
            if response_data.get("error"):
                provider_results[provider_name]["error"] = str(response_data["error"])
            for item in normalized:
                key = (
                    _canonicalize_matrix_result_url(item["url"])
                    or item["url"]
                    or f"{provider_name}:{item['position']}"
                )
                entry = fused_index.setdefault(
                    key,
                    {
                        "url": item["url"],
                        "title": item["title"],
                        "description": item["description"],
                        "providers": [],
                        "positions": {},
                    },
                )
                if item["provider"] not in entry["providers"]:
                    entry["providers"].append(item["provider"])
                entry["positions"][item["provider"]] = item["position"]
                if not entry.get("title") and item["title"]:
                    entry["title"] = item["title"]
                if not entry.get("description") and item["description"]:
                    entry["description"] = item["description"]
        except Exception as exc:
            provider_results[provider_name] = {
                "success": False,
                "result_count": 0,
                "error": str(exc),
                "results": [],
            }

    fused_results = []
    for entry in fused_index.values():
        provider_hits = len(entry["providers"])
        avg_position = sum(entry["positions"].values()) / max(1, provider_hits)
        fused_results.append(
            {
                "url": entry["url"],
                "title": entry["title"],
                "description": entry["description"],
                "providers": sorted(entry["providers"]),
                "provider_hits": provider_hits,
                "positions": entry["positions"],
                "position": (
                    min(entry["positions"].values()) if entry["positions"] else None
                ),
                "_avg_position": avg_position,
            }
        )

    fused_results.sort(
        key=lambda item: (
            -int(item.get("provider_hits") or 0),
            float(item.get("_avg_position") or 999.0),
            str(item.get("title") or ""),
        )
    )
    trimmed_results = []
    for item in fused_results[:limit]:
        payload = dict(item)
        payload.pop("_avg_position", None)
        trimmed_results.append(payload)

    if not available_requested:
        return {
            "query": query,
            "strategy": "all_available" if not requested_providers or "all" in requested_providers else "requested",
            "data": {"web": []},
            "providers_used": [],
            "providers_missing": missing_requested,
            "provider_results": provider_results,
            "search_error": "No configured web search providers are available for web_search_matrix.",
        }

    return {
        "query": query,
        "strategy": "all_available" if not requested_providers or "all" in requested_providers else "requested",
        "data": {"web": trimmed_results},
        "providers_used": available_requested,
        "providers_missing": missing_requested,
        "provider_results": provider_results,
    }


def web_search_matrix(
    query: Optional[str] = None,
    limit: int = 5,
    providers: Optional[List[str]] = None,
    require_capabilities: Optional[List[str]] = None,
    require_providers: Optional[List[str]] = None,
) -> str:
    """Return provider status, or run a fused multi-provider search.

    The result reports provider names, capability flags, availability, active
    providers, and env-var presence by variable name only. It never includes
    env values. When ``query`` is supplied, the same payload also includes
    fused search results across the available/requested search providers.
    """
    required_caps = _normalize_matrix_items(
        require_capabilities,
        allowed=set(_WEB_MATRIX_CAPABILITIES),
    )
    required_providers = _normalize_matrix_items(require_providers)
    requested_search_providers = _normalize_matrix_items(
        providers,
        allowed=set(_WEB_MATRIX_SEARCH_PROVIDERS),
    )

    try:
        from hermes_cli.plugins import discover_plugins

        discover_plugins()
    except Exception as exc:
        logger.debug("Could not discover web provider plugins: %s", exc)

    from agent.web_search_registry import (
        get_active_crawl_provider,
        get_active_extract_provider,
        get_active_search_provider,
        list_providers,
    )

    providers = [_provider_matrix_row(provider) for provider in list_providers()]
    providers_by_name = {row["name"]: row for row in providers}

    def _active_row(provider: Any, capability: str) -> Optional[dict[str, Any]]:
        if provider is None:
            return None
        row = _provider_matrix_row(provider)
        return {
            "name": row["name"],
            "display_name": row["display_name"],
            "available": row["available"],
            "capability": capability,
        }

    active = {
        "search": _active_row(get_active_search_provider(), "search"),
        "extract": _active_row(get_active_extract_provider(), "extract"),
        "crawl": _active_row(get_active_crawl_provider(), "crawl"),
    }

    blocked_reasons: list[str] = []
    if not providers:
        blocked_reasons.append("no web providers are registered")

    for name in required_providers:
        provider = providers_by_name.get(name)
        if provider is None:
            blocked_reasons.append(f"required provider '{name}' is not registered")
            continue
        if not provider["available"]:
            blocked_reasons.append(f"required provider '{name}' is not available")

    for capability in required_caps:
        if not any(
            row["available"] and row["capabilities"].get(capability)
            for row in providers
        ):
            blocked_reasons.append(
                f"no available web provider supports {capability}"
            )

    status = "dependency_blocked" if blocked_reasons else "ok"
    payload: dict[str, Any] = {
        "success": not blocked_reasons,
        "status": status,
        "requirements": {
            "capabilities": required_caps,
            "providers": required_providers,
        },
        "active": active,
        "providers": providers,
        "provider_status": _matrix_provider_status(providers),
        "firecrawl_surfaces": _firecrawl_surface_status(providers_by_name),
        "blocked_reasons": blocked_reasons,
    }

    query_text = str(query or "").strip()
    if query_text and not blocked_reasons:
        search_payload = _run_web_search_matrix_query(
            query_text,
            limit,
            providers,
            requested_search_providers,
        )
        payload.update(search_payload)
        if search_payload.get("search_error"):
            payload["success"] = False
            payload["status"] = "dependency_blocked"
            payload["blocked_reasons"] = [
                *payload["blocked_reasons"],
                str(search_payload["search_error"]),
            ]
    elif query_text:
        payload.update(
            {
                "query": query_text,
                "strategy": "skipped_dependency_blocked",
                "data": {"web": []},
                "providers_used": [],
                "providers_missing": requested_search_providers,
                "provider_results": {},
                "search_error": "web_search_matrix requirements are unavailable; search skipped.",
            }
        )

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)

    # Check if API keys are available
    web_available = check_web_api_key()
    tool_gateway_available = _is_tool_gateway_ready()
    firecrawl_key_available = bool(os.getenv("FIRECRAWL_API_KEY", "").strip())
    firecrawl_url_available = bool(os.getenv("FIRECRAWL_API_URL", "").strip())

    if web_available:
        backend = _get_backend()
        print(f"✅ Web backend: {backend}")
        if backend == "exa":
            print("   Using Exa API (https://exa.ai)")
        elif backend == "parallel":
            print("   Using Parallel API (https://parallel.ai)")
        elif backend == "tavily":
            print("   Using Tavily API (https://tavily.com)")
        elif backend == "searxng":
            print(f"   Using SearXNG (search only): {_env_value('SEARXNG_URL')}")
        elif backend == "brave-free":
            print("   Using Brave Search free tier (search only)")
        elif backend == "ddgs":
            print("   Using DuckDuckGo via ddgs package (search only)")
        elif firecrawl_url_available:
            print(f"   Using self-hosted Firecrawl: {os.getenv('FIRECRAWL_API_URL').strip().rstrip('/')}")
        elif firecrawl_key_available:
            print("   Using direct Firecrawl cloud API")
        elif tool_gateway_available:
            print(f"   Using Firecrawl tool-gateway: {_get_firecrawl_gateway_url()}")
        else:
            print("   Firecrawl backend selected but not configured")
    else:
        print("❌ No web search backend configured")
        print(
            "Set EXA_API_KEY, PARALLEL_API_KEY, TAVILY_API_KEY, FIRECRAWL_API_KEY, FIRECRAWL_API_URL"
            f"{_firecrawl_backend_help_suffix()}"
        )

    if not web_available:
        sys.exit(1)

    print("🛠️  Web tools ready for use!")
    print(f"   Extract char limit: {_get_extract_char_limit()} chars "
          "(pages over this are truncated; full text stored in cache/web)")

    # Show debug mode status
    if _debug.active:
        print(f"🐛 Debug mode ENABLED - Session ID: {_debug.session_id}")
        print(f"   Debug logs will be saved to: {_debug.log_dir}/web_tools_debug_{_debug.session_id}.json")
    else:
        print("🐛 Debug mode disabled (set WEB_TOOLS_DEBUG=true to enable)")

    print("\nBasic usage:")
    print("  from web_tools import web_search_tool, web_extract_tool")
    print("  import asyncio")
    print("")
    print("  # Search (synchronous)")
    print("  results = web_search_tool('Python tutorials')")
    print("")
    print("  # Extract (asynchronous, no LLM — truncate-and-store)")
    print("  async def main():")
    print("      content = await web_extract_tool(['https://example.com'])")
    print("      # bigger budget for one call:")
    print("      content = await web_extract_tool(['https://docs.python.org'], char_limit=40000)")
    print("  asyncio.run(main())")

    print("\nDebug mode:")
    print("  export WEB_TOOLS_DEBUG=true")
    print("  # Logs saved to: ./logs/web_tools_debug_UUID.json")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from tools.registry import registry, tool_error

WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": "Search the web for information. Returns up to 5 results by default with titles, URLs, and descriptions. The query is passed through to the configured backend, so operators such as site:domain, filetype:pdf, intitle:word, -term, and \"exact phrase\" may work when the backend supports them.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to look up on the web. You may include backend-supported operators such as site:example.com, filetype:pdf, intitle:word, -term, or \"exact phrase\"."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 5.",
                "minimum": 1,
                "maximum": 100,
                "default": 5
            }
        },
        "required": ["query"]
    }
}

WEB_EXTRACT_SCHEMA = {
    "name": "web_extract",
    "description": "Extract content from web page URLs. Returns clean page content in markdown/text (no LLM summarization — fast). Also works with PDF URLs (arxiv papers, documents) — pass the PDF link directly. Pages within the char budget (default 15000) return whole; larger pages return a head+tail window with a footer telling you the full text's saved file path and the read_file call to page through the omitted middle. Inline images appear as [IMAGE: alt] placeholders; real image URLs are kept as links. If a URL fails or times out, use the browser tool instead.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of URLs to extract content from (max 5 URLs per call)",
                "maxItems": 5
            },
            "char_limit": {
                "type": "integer",
                "description": "Optional per-page character budget sent back (default 15000). Pages larger than this are head+tail truncated with the full text stored to disk. Raise it when you need more of a long page inline.",
                "minimum": 2000
            }
        },
        "required": ["urls"]
    }
}

WEB_SEARCH_MATRIX_SCHEMA = {
    "name": "web_search_matrix",
    "description": (
        "Search across configured web providers, run labeled query matrices, "
        "and report provider capability status. Returns fused results when "
        "query is supplied, plus provider availability, active "
        "search/extract/crawl providers, Firecrawl surface status, and env-var "
        "presence by variable name only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Optional search query. Omit this to report provider status "
                    "without making external search calls."
                ),
            },
            "queries": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "query": {"type": "string"},
                            },
                            "required": ["query"],
                        },
                    ]
                },
                "description": "One to twelve source-discovery queries, as strings or {label, query} objects.",
                "minItems": 1,
                "maxItems": 12,
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of fused search results to return.",
                "minimum": 1,
                "maximum": 10,
            },
            "limit_per_query": {
                "type": "integer",
                "description": "Maximum results per labeled query. Defaults to 3.",
                "minimum": 1,
                "maximum": 10,
                "default": 3,
            },
            "providers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(_WEB_MATRIX_SEARCH_PROVIDERS),
                },
                "description": (
                    "Optional search provider subset. Defaults to all available "
                    "search providers when query is supplied."
                ),
            },
            "required_providers": {
                "type": "array",
                "description": (
                    "Providers that must succeed in source-discovery query mode. "
                    "Use when a research protocol explicitly requires a named provider."
                ),
                "items": {
                    "type": "string",
                    "enum": [
                        "brave-free",
                        "ddgs",
                        "exa",
                        "firecrawl",
                        "parallel",
                        "searxng",
                        "tavily",
                    ],
                },
            },
            "require_capabilities": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["search", "extract", "crawl"],
                },
                "description": (
                    "Optional capabilities that must have at least one available "
                    "provider. If unmet, the result status is dependency_blocked."
                ),
            },
            "require_providers": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "brave-free",
                        "ddgs",
                        "exa",
                        "firecrawl",
                        "parallel",
                        "searxng",
                        "tavily",
                    ],
                },
                "description": (
                    "Optional provider names that must be registered and available. "
                    "Use ['firecrawl'] for jobs that require Firecrawl-backed capture."
                ),
            },
        },
        "required": [],
    },
}


def _handle_web_search_matrix_registry(args, **kw):
    if args.get("queries") is not None or args.get("required_providers") is not None:
        return web_search_matrix_tool(
            args.get("queries"),
            limit_per_query=args.get("limit_per_query", 3),
            query=args.get("query"),
            limit=args.get("limit"),
            providers=args.get("providers"),
            required_providers=args.get("required_providers"),
        )
    return web_search_matrix(
        query=args.get("query"),
        limit=args.get("limit") or 5,
        providers=args.get("providers"),
        require_capabilities=args.get("require_capabilities"),
        require_providers=args.get("require_providers"),
    )


registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(args.get("query", ""), limit=args.get("limit", 5)),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    emoji="🔍",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_extract",
    toolset="web",
    schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [],
        "markdown",
        char_limit=args.get("char_limit"),
    ),
    check_fn=check_web_api_key,
    requires_env=_web_requires_env(),
    is_async=True,
    emoji="📄",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_search_matrix",
    toolset="web",
    schema=WEB_SEARCH_MATRIX_SCHEMA,
    handler=_handle_web_search_matrix_registry,
    check_fn=lambda: True,
    emoji="🔎",
    max_result_size_chars=100_000,
)

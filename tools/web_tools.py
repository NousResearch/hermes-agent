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
    from web_tools import web_search_tool, web_extract_tool
    
    # Search the web
    results = web_search_tool("Python machine learning libraries", limit=3)
    
    # Extract content from URLs  
    content = web_extract_tool(["https://example.com"], format="markdown")
"""

import json
import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING
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
from tools.url_safety import async_is_safe_url, normalize_url_for_request
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

def _get_backend() -> str:
    """Determine which web backend to use (shared fallback).

    Reads ``web.backend`` from config.yaml (set by ``hermes tools``).
    Falls back to whichever API key is present for users who configured
    keys manually without running setup.
    """
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured in {"parallel", "firecrawl", "tavily", "exa", "searxng", "brave-free", "ddgs", "xai"}:
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
    """Return True when the selected backend is currently usable."""
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
            response_data = provider.search(query, limit)

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
        policy_blocked: List[Dict[str, Any]] = []
        try:
            from tools.website_policy import check_website_access
        except Exception:
            check_website_access = lambda _url: None  # noqa: E731
        for url in normalized_urls:
            if not await async_is_safe_url(url):
                ssrf_blocked.append({
                    "url": url, "title": "", "content": "",
                    "error": "Blocked: URL targets a private or internal network address",
                })
                continue
            blocked = check_website_access(url)
            if blocked:
                policy_blocked.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": blocked["message"],
                    "blocked_by_policy": {
                        "host": blocked["host"],
                        "rule": blocked["rule"],
                        "source": blocked["source"],
                    },
                })
                continue
            safe_urls.append(url)

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

        # Re-check final provider URLs so redirects/canonicalization cannot
        # bypass the central website policy on providers that do not enforce it
        # internally.
        if results:
            filtered_results: List[Dict[str, Any]] = []
            for result in results:
                final_url = result.get("url", "")
                blocked = check_website_access(final_url) if final_url else None
                if blocked:
                    filtered_results.append({
                        "url": final_url,
                        "title": result.get("title", ""),
                        "content": "",
                        "raw_content": "",
                        "error": blocked["message"],
                        "blocked_by_policy": {
                            "host": blocked["host"],
                            "rule": blocked["rule"],
                            "source": blocked["source"],
                        },
                    })
                else:
                    filtered_results.append(result)
            results = filtered_results

        # Merge any blocked results back in
        if ssrf_blocked or policy_blocked:
            results = ssrf_blocked + policy_blocked + results

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
        trimmed_results = [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "error": r.get("error"),
                **({  "blocked_by_policy": r["blocked_by_policy"]} if "blocked_by_policy" in r else {}),
            }
            for r in response.get("results", [])
        ]
        trimmed_response = {"results": trimmed_results}

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


# Convenience function to check Firecrawl credentials
def check_web_api_key() -> bool:
    """Check whether the configured web backend is available."""
    configured = _load_web_config().get("backend", "").lower().strip()
    if configured in {"exa", "parallel", "firecrawl", "tavily", "searxng", "brave-free", "ddgs", "xai"}:
        return _is_backend_available(configured)
    return any(
        _is_backend_available(backend)
        for backend in ("exa", "parallel", "firecrawl", "tavily", "searxng", "brave-free", "ddgs", "xai")
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

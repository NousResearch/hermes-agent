#!/usr/bin/env python3
"""Generic web_search / web_extract tools over pluggable backends.

Backend is selected during ``hermes tools`` (``web.backend`` in config.yaml; per
capability via ``web.search_backend`` / ``web.extract_backend``). Every vendor
implementation lives in ``plugins/web/<vendor>/provider.py`` and registers with
``agent.web_search_registry``; this module owns selection, safety gates,
caching, keyless rescue, and the truncate-and-store result pipeline.
Debug: ``WEB_TOOLS_DEBUG=true`` writes ``logs/web_tools_debug_<UUID>.json``.
"""

import json
import logging
import os
from typing import List, Any, Optional
# Per-vendor client cache slots; plugins read/write these via tools.web_tools (tests reset them to None).
_firecrawl_client = _firecrawl_client_config = _parallel_client = _async_parallel_client = _exa_client = None

from plugins.web.firecrawl.provider import _is_tool_gateway_ready, check_firecrawl_api_key
from tools.debug_helpers import DebugSession
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, selection_exists
from tools.url_safety import async_is_safe_url
from tools.web_tools_rescue import _rescue_eligible, _rescue_search
from tools.web_tools_truncate import _effective_char_limit, _trim_results, _truncate_results, convert_base64_images_to_links
from tools.web_tools_extract import (
    _extract_safe_urls, _merge_in_order, _no_provider_error, _resolve_extract_provider, _result_entry,
    _strict_selection_error, _validate_extract_urls,
)

logger = logging.getLogger(__name__)


# ─── Backend Selection ────────────────────────────────────────────────────────

def _env_value(name: str) -> str:
    """Resolve ``name`` via the config-aware env layer (``hermes config set`` values), then process env.

    Mirrors the SearXNG provider's ``_searxng_url()`` so that values set through Hermes' config/.env layer
    (``hermes config set``, ``hermes tools``) are honored here too — not just raw process-env exports.
    Without this, a config-only ``SEARXNG_URL`` (or any provider key) leaves the backend auto-detect cascade
    and ``check_web_api_key()`` blind to it. See #34290.
    """
    try:
        from hermes_cli.config import get_env_value
        val = get_env_value(name)
    except Exception:
        val = None
    return ((os.getenv(name, "") if val is None else val) or "").strip()


def _has_env(name: str) -> bool:
    return bool(_env_value(name))


def _load_web_config() -> dict:
    """Load the ``web:`` section from config.yaml; always a dict (a null section yields ``{}``)."""
    try:
        from hermes_cli.config import load_config
        return load_config().get("web") or {}
    except Exception:
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
    {"parallel", "firecrawl", "tavily", "exa", "searxng", "brave-free", "ddgs", "xai", "keenable"}
)


def _registered_web_provider(backend: str):
    """Plugin-registered web provider by name, or ``None``."""
    return _registry_call("get_provider", None, backend) if backend else None


def _list_registered_web_providers():
    """All plugin-registered web providers (empty list on failure)."""
    return _registry_call("list_providers", [])


def _probe(provider, method: str, context: str = "") -> Optional[bool]:
    """``bool(provider.<method>())``, or ``None`` if it raised (a broken provider is unavailable; *context* is
    appended to the debug log line, e.g. " during readiness check")."""
    try:
        return bool(getattr(provider, method)())
    except Exception as exc:  # noqa: BLE001 — a broken provider is "unavailable"
        name = getattr(provider, "name", provider)
        logger.debug("web provider %r.%s() raised%s: %s", name, method, context, exc)
        return None


def _get_backend() -> str:
    """Shared web backend name. A stored ``web.backend`` is returned as-is — no availability probe, no
    fallback — so a broken selection surfaces the vendor's honest error rather than silently rerouting.
    Autodetect runs ONLY when no web selection has ever been stored."""
    configured = _configured_backend()
    if configured:
        # "nous" (managed subscription) is serviced by firecrawl, routed through the managed Tool Gateway.
        return "firecrawl" if configured == NOUS_MANAGED_PROVIDER else configured
    if selection_exists("web"):
        # Selection exists (use_gateway / per-capability keys) but no shared name: firecrawl, no ladder.
        return "firecrawl"

    Reads ``web.backend`` from config.yaml (set by ``hermes tools``). A
    stored backend name is returned as-is — no availability probe, no
    fallback — so the vendor path can raise its own honest error when the
    selection is broken. The credential/entitlement autodetect ladder runs
    ONLY when no web selection has ever been stored.
    """
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured:
        # Strict: the stored selection is final, known name or not — an
        # unknown/typoed name surfaces as the vendor path's honest error
        # rather than silently rerouting through the credential ladder.
        # The managed "Nous Subscription" selection ("nous") is serviced by
        # the firecrawl provider, whose client resolver routes it through
        # the managed Tool Gateway.
        from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER

        if configured == NOUS_MANAGED_PROVIDER:
            return "firecrawl"
        return configured

    from tools.tool_backend_helpers import selection_exists

    if selection_exists("web"):
        # A web selection exists (e.g. use_gateway key or per-capability
        # backends) but the shared backend name is empty — keep the
        # firecrawl default rather than credential-laddering.
        return "firecrawl"

    # Never-configured install — pick the highest-priority available
    # backend. Explicit user credentials (TAVILY_API_KEY etc.)
    # beat the managed-tool-gateway probe so a deliberate setup is not
    # pre-empted by a Nous OAuth token whose subscription tier may not
    # actually grant web-search access (the gateway then fails at runtime
    # with "no subscription" and the tool returns an error to the agent
    # without falling back). Free-tier backends trail the paid ones.
    backend_candidates = (
        ("tavily", _has_env("TAVILY_API_KEY")), ("perplexity", _has_env("PERPLEXITY_API_KEY")),
        ("exa", _has_env("EXA_API_KEY")),
        ("parallel", _has_env("PARALLEL_API_KEY")),
        ("keenable", _has_env("KEENABLE_API_KEY")),
        ("firecrawl", _has_env("FIRECRAWL_API_KEY") or _has_env("FIRECRAWL_API_URL")),
        ("firecrawl", _is_tool_gateway_ready()), ("searxng", _has_env("SEARXNG_URL")),
        ("brave-free", _has_env("BRAVE_SEARCH_API_KEY")), ("ddgs", _ddgs_package_importable()),
    )
    for backend, available in backend_candidates:
        if available:
            return backend

    # Plugin-contributed providers (built-ins are covered above); probe the held object directly.
    for provider in _list_registered_web_providers():
        if provider.name not in _LEGACY_WEB_BACKENDS and _probe(provider, "is_available"):
            return provider.name

    # Keyless free tier — strictly last so it never pre-empts a keyed backend. Discovery must run
    # first: reachable from contexts that haven't loaded plugins (subprocess runs, delegate children).
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import _keyless_preference, _keyless_tier_enabled
        if _keyless_tier_enabled():
            for name in _keyless_preference():
                provider = _registered_web_provider(name)
                if provider is not None and _probe(provider, "is_keyless_available"):
                    return name
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("keyless fallback walk failed: %s", exc)

    # Keyless free-tier walk — zero credentials anywhere. Providers with a
    # public anonymous endpoint (Parallel, Exa — see
    # plugins/web/keyless_mcp.py) can still serve, unless the user disabled
    # the tier via ``web.keyless_fallback: false``. Strictly last so it
    # never pre-empts any keyed/importable backend above. Discovery must
    # run first — this path is reachable from contexts that haven't loaded
    # plugins yet (subprocess agent runs, delegate children, scripts).
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import _keyless_preference, _keyless_tier_enabled

        if _keyless_tier_enabled():
            for name in _keyless_preference():
                provider = _registered_web_provider(name)
                if provider is None:
                    continue
                try:
                    if provider.is_keyless_available():
                        return name
                except Exception as exc:  # noqa: BLE001 — skip broken provider
                    logger.debug(
                        "web provider %r.is_keyless_available() raised: %s", name, exc
                    )
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("keyless fallback walk failed: %s", exc)

    return "firecrawl"  # default (backward compat)


def _get_search_backend() -> str:
    """Backend for web_search: ``web.search_backend`` (strict, no probe) > ``web.backend`` > autodetect."""
    return _configured_backend("search_backend") or _get_backend()


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

    Reads ``web.{capability}_backend`` from config; a stored value is
    returned unconditionally (strict selection — no availability probe).
    A selected-but-broken backend surfaces the vendor path's honest error
    instead of being silently replaced by whatever the credential ladder
    finds. Falls through to the shared ``_get_backend()`` only when no
    per-capability override is stored.
    """
    cfg = _load_web_config()
    specific = (cfg.get(f"{capability}_backend") or "").lower().strip()
    if specific:
        return specific
    return _get_backend()


def _tavily_explicitly_configured() -> bool:
    cfg = _load_web_config()
    return any(
        (cfg.get(key) or "").lower().strip() == "tavily"
        for key in ("backend", "search_backend", "extract_backend")
    )


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
    if backend == "keenable":
        return _has_env("KEENABLE_API_KEY")
    if backend == "firecrawl":
        return check_firecrawl_api_key()
    if backend == "tavily":
        return _has_env("TAVILY_API_KEY") or _tavily_explicitly_configured()
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
    """ddgs is the only backend gated on package presence; single symbol so tests can patch it."""
    try:
        import ddgs  # noqa: F401
        return True
    except ImportError:
        return False


# ─── One-shot keyless rescue (keyed/configured backend failed) ───────────────

def _keyless_rescue_enabled() -> bool:
    """Read ``web.keyless_rescue`` from config (default: enabled).

    Also implicitly off whenever the keyless tier itself is disabled
    (``web.keyless_fallback: false``).
    """
    cfg = _load_web_config()
    if not cfg.get("keyless_rescue", True):
        return False
    try:
        from agent.web_search_registry import _keyless_tier_enabled

        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — registry optional
        logger.debug("keyless rescue tier check failed: %s", exc)
        return False


def _rescue_eligible(provider) -> bool:
    """True when a failed call on *provider* should get a one-shot rescue.

    Eligible: the call ran a keyed/configured path — either a non-ring
    backend (searxng, brave-free, xai, custom plugins, managed gateway) or
    a ring vendor operating in keyed mode. NOT eligible: the call already
    went through the keyless ring (its failure means the ring was walked;
    re-walking would just repeat it).
    """
    if not _keyless_rescue_enabled():
        return False
    if provider is None:
        return False
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING, use_keyless

        name = getattr(provider, "name", "")
        if name in _KEYLESS_RING:
            key_var = {
                "exa": "EXA_API_KEY",
                "parallel": "PARALLEL_API_KEY",
                "tavily": "TAVILY_API_KEY",
                "firecrawl": "FIRECRAWL_API_KEY",
                "keenable": "KEENABLE_API_KEY",
            }.get(name, "")
            from agent.web_search_provider import get_provider_env

            api_key = get_provider_env(key_var) if key_var else ""
            # Keyless-mode ring vendors already walked the ring on failure.
            return not use_keyless(name, api_key)
        return True
    except Exception as exc:  # noqa: BLE001 — rescue is best-effort
        logger.debug("rescue eligibility check failed: %s", exc)
        return False


def _rescue_search(provider_name: str, original_error: str, query: str, limit: int) -> dict:
    """One-shot keyless-ring rescue for a failed keyed/configured search.

    Stateless by design: this call alone routes to the free-tier ring; the
    NEXT web_search call attempts the chosen backend again. The result is
    annotated with the original backend failure so the model (and the
    user) can see the configured backend needs attention.
    """
    from plugins.web.keyless_mcp import search_with_failover

    logger.warning(
        "web_search backend '%s' failed (%s); one-shot keyless rescue",
        provider_name, (original_error or "")[:200],
    )
    rescued = search_with_failover(provider_name, query, limit)
    if rescued.get("success"):
        data = rescued.setdefault("data", {})
        data["rescued_from"] = provider_name
        data["backend_error"] = (
            f"Configured backend '{provider_name}' failed this call "
            f"({(original_error or 'unknown error')[:300]}); result served "
            "by the keyless free tier. The next call will use "
            f"'{provider_name}' again."
        )
        return rescued
    # Ring also failed: surface the ORIGINAL backend error (it names the
    # user's configured setup) with the rescue note appended.
    return {
        "success": False,
        "error": (
            f"{original_error or 'search failed'} "
            f"(keyless rescue also failed: {rescued.get('error', 'unknown')})"
        ),
    }


def _policy_blocked_result(result: dict) -> bool:
    """True when an extract result failed because of the user's website
    policy — an intentional refusal, never a backend outage. Policy blocks
    must NOT be rescued: routing the same URL through the keyless ring
    would fetch content the user explicitly blocked."""
    if result.get("blocked_by_policy"):
        return True
    return "blocked by website policy" in str(result.get("error") or "").lower()


def _rescue_extract(provider_name: str, urls: list, results: list) -> list:
    """One-shot keyless-ring rescue for a failed keyed/configured extract.

    Fires only when EVERY url failed (whole-backend failure); partial
    results are page problems and pass through untouched. Stateless —
    the next web_extract call attempts the chosen backend again.

    Website-policy refusals are intentional, not failures: entries flagged
    by ``_policy_blocked_result`` are never re-fetched through the ring and
    their original (blocked) results are preserved verbatim.
    """
    from plugins.web.keyless_mcp import extract_with_failover

    # Partition out policy blocks. Rescue only genuine backend failures.
    if len(results) == len(urls):
        rescue_idx = [i for i, r in enumerate(results) if not _policy_blocked_result(r)]
    else:  # defensive: provider broke order parity — treat all as rescueable
        rescue_idx = list(range(len(results)))
    if not rescue_idx:
        return results  # every failure is an intentional policy block

    rescue_urls = [urls[i] for i in rescue_idx] if len(results) == len(urls) else list(urls)
    original_error = next(
        (results[i].get("error") for i in rescue_idx if results[i].get("error")),
        "extract failed",
    )
    logger.warning(
        "web_extract backend '%s' failed all %d URL(s) (%s); one-shot keyless rescue",
        provider_name, len(rescue_urls), (original_error or "")[:200],
    )
    rescued = extract_with_failover(provider_name, list(rescue_urls))
    rescued_errors = [r.get("error", "") for r in rescued]
    if rescued and all(e for e in rescued_errors):
        return results  # rescue also failed everywhere: keep original errors
    for r in rescued:
        if not r.get("error"):
            meta = r.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["rescued_from"] = provider_name
                meta["backend_error"] = (original_error or "")[:300]
    if len(rescued) == len(rescue_idx) and len(results) == len(urls):
        merged = list(results)
        for pos, i in enumerate(rescue_idx):
            merged[i] = rescued[pos]
        return merged
    return rescued


# ─── Firecrawl Client ────────────────────────────────────────────────────────

def _xai_available() -> bool:
    # Cheap probe only (env var OR auth.json OAuth): resolve_xai_http_credentials() may hit the network.
    try:
        from tools.xai_http import has_xai_credentials
        return has_xai_credentials()
    except Exception:
        return False


# Built-in backends -> cheap availability probes; any other name is a plugin provider resolved via the
# registry's ``is_available()``. Lambdas so test patches of module-level helpers (_ddgs_package_importable,
# check_firecrawl_api_key) are honored at call time. ``xai`` is probed via has_xai_credentials(), not a
# registered provider, though the registry's _LEGACY_PREFERENCE omits it — drop it if xai ever registers.
_BUILTIN_AVAILABILITY = {
    "exa": lambda: _has_env("EXA_API_KEY"),
    "parallel": lambda: _has_env("PARALLEL_API_KEY"),
    "keenable": lambda: _has_env("KEENABLE_API_KEY"),
    "firecrawl": lambda: check_firecrawl_api_key(),
    "tavily": lambda: _has_env("TAVILY_API_KEY")
    or any(_configured_backend(k) == "tavily" for k in ("backend", "search_backend", "extract_backend")),
    "perplexity": lambda: _has_env("PERPLEXITY_API_KEY"),
    "searxng": lambda: _has_env("SEARXNG_URL"),
    "brave-free": lambda: _has_env("BRAVE_SEARCH_API_KEY"),
    "ddgs": lambda: _ddgs_package_importable(),
    "xai": _xai_available,
}
_LEGACY_WEB_BACKENDS = frozenset(_BUILTIN_AVAILABILITY)


def _is_backend_available(backend: str) -> bool:
    """True when *backend* is usable — the single availability chokepoint. Non-legacy names delegate to the
    registered provider's ``is_available()`` (unregistered names fall through); built-ins use cheap probes.

    For plugin-registered backends (any name outside :data:`_LEGACY_WEB_BACKENDS`), availability is
    delegated to the provider's ``is_available()`` via the web_search_registry. This is the single
    chokepoint through which ``_get_backend``, ``_get_capability_backend``, and ``check_web_api_key`` all
    resolve availability — fixing custom-provider discovery for every caller at once (issues #28651, #31873,
    #32698). Built-in backends keep their cheap hardcoded probes below.
    """
    backend = (backend or "").lower().strip()
    provider = None if backend in _LEGACY_WEB_BACKENDS else _registered_web_provider(backend)
    if provider is not None:
        return _probe(provider, "is_available") or False
    probe = _BUILTIN_AVAILABILITY.get(backend)
    return probe() if probe else False


# ─── Firecrawl Client ──────────────────────────────────────────────────────── After PR #25182, the
# firecrawl client, lazy SDK proxy, dual-auth config resolution, response normalizers, and
# check_firecrawl_api_key() all live in plugins.web.firecrawl.provider.
def _web_requires_env() -> list[str]:
    """Tool-registry metadata env vars for the web backends. Gateway vars are always listed: gating them
    on ``managed_nous_tools_enabled()`` cost a synchronous portal HTTP refresh at every CLI startup.
    Contract: set var -> tool sees it; extras are harmless for the not-logged-in."""
    return [
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "KEENABLE_API_KEY",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "FIRECRAWL_GATEWAY_URL",
        "TOOL_GATEWAY_DOMAIN",
        "TOOL_GATEWAY_SCHEME",
        "TOOL_GATEWAY_USER_TOKEN",
    ]

_debug = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")


# ─── Dispatch ─────────────────────────────────────────────────────────────────

# ─── Exa / Parallel inline helpers — moved into plugins ────────────────────── After PR #25182, the exa
# client + search/extract and parallel client + search/extract helpers all live in their respective plugins:
# - plugins/web/exa/provider.py - plugins/web/parallel/provider.py Both plugins register through
# agent.web_search_registry and the dispatchers in this file resolve them via get_active_*_provider().
def _ensure_web_plugins_loaded() -> None:
    """Idempotently run plugin discovery so the web registry is populated. Dispatch is reachable from contexts
    that never triggered discovery (subprocess agent runs, delegate children, scripts); without it a
    configured backend yields a misleading "No web ... provider" error.

    Every bundled web provider (brave-free, ddgs, searxng, exa, parallel, tavily, firecrawl, keenable)
    registers itself via ``plugins/web/<vendor>/__init__.py`` during plugin discovery. Tool dispatch can be
    reached from contexts that haven't already triggered discovery — subprocess agent runs, delegate
    children, standalone scripts, certain test paths — and without it the registry is empty and
    ``get_provider('firecrawl')`` returns ``None`` even when the user has ``web.extract_backend: firecrawl``
    configured and ``FIRECRAWL_API_KEY`` set. See #27580.
    """
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered
        _ensure_plugins_discovered()
    except Exception as exc:  # noqa: BLE001
        # Warning, not debug: a broken plugin import is otherwise invisible.
        logger.warning("Web plugin discovery failed (non-fatal): %s", exc)


def _finish_debug(call_name: str, debug_call_data: dict, error_msg: Optional[str] = None) -> Optional[str]:
    """Log the call into the debug session; with *error_msg*, record it and return its ``tool_error`` envelope."""
    if error_msg is not None:
        logger.debug("%s", error_msg)
        debug_call_data["error"] = error_msg
    _debug.log_call(call_name, debug_call_data)
    _debug.save()
    return None if error_msg is None else tool_error(error_msg)


def web_search_tool(query: str, limit: int = 5) -> str:
    """Search the web via the configured backend.

    Returns a JSON string ``{"success": bool, "data": {"web": [{"title", "url", "description", "position"},
    ...]}}`` (metadata only — use web_extract_tool for page content) or ``{"success": false, "error": ...}``.
    """
    try:
        limit = min(max(int(limit), 1), 100)
    except (TypeError, ValueError):
        limit = 5
    debug_call_data = {
        "parameters": {"query": query, "limit": limit}, "error": None, "results_count": 0,
        "original_response_size": 0, "final_response_size": 0,
    }

    try:
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return tool_error("Interrupted", success=False)
        # Sync only — every provider's search() is sync.
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import get_active_search_provider, get_provider as _wsp_get_provider
        backend = _get_search_backend()
        provider = _wsp_get_provider(backend) if backend else None
        if provider is None or not provider.supports_search():
            from tools.tool_backend_helpers import (
                selection_error,
                selection_exists,
            )

            if provider is None and backend and selection_exists("web"):
                disabled_key = _disabled_web_plugin_for(capability="search")
                if disabled_key:
                    _vendor = disabled_key.split("/", 1)[-1]
                    error_text = (
                        f"web.search_backend is set to '{_vendor}', but its "
                        f"plugin ('{disabled_key}') is disabled in config. "
                        f"Re-enable it with `hermes plugins enable {disabled_key}` "
                        "(or remove it from plugins.disabled)."
                    )
                else:
                    error_text = selection_error(
                        "web",
                        f"'{backend}'",
                        "no registered web search provider has that name",
                    )
                response_data = {"success": False, "error": error_text}
                result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
                debug_call_data["error"] = error_text
                _debug.log_call("web_search_tool", debug_call_data)
                _debug.save()
                return result_json
            # Never-configured install: fall back to the availability-walked
            # active provider (legacy autodetect behavior).
            provider = get_active_search_provider()

        if provider is None:
            fallback = "No web search provider configured. Run `hermes tools` to set one up."
            response_data = {"success": False, "error": _no_provider_error("search", fallback)}
        else:
            logger.info(
                "Web search via %s: '%s' (limit: %d)",
                provider.name, query, limit,
            )
            # ── TTL memo + single-flight (tools/web_result_cache.py) ──
            # Sits after every safety/config check and directly around the
            # paid vendor call. Identical queries within the TTL (subagent
            # fan-outs, repeat lookups) are served from memory; concurrent
            # identical queries share one request via the flight lock. The
            # provider is asked for the BUCKETED count (10/20/50/100) so
            # near-identical limits share an entry; the caller's requested
            # count is sliced out below. Only successful responses cache.
            from tools.web_result_cache import (
                bucket_limit as _bucket_limit,
                search_memo as _search_memo,
                slice_search_response as _slice_search_response,
            )

            def _paid_search() -> tuple[dict, bool]:
                _fetch_limit = _bucket_limit(limit)
                _rescued = False
                try:
                    _resp = provider.search(query, _fetch_limit)
                except Exception as exc:  # noqa: BLE001 — candidate for rescue
                    if _rescue_eligible(provider):
                        _rescued = True
                        _resp = _rescue_search(
                            provider.name, str(exc), query, _fetch_limit
                        )
                    else:
                        raise
                else:
                    if not _resp.get("success") and _rescue_eligible(provider):
                        # One-shot keyless rescue: THIS call rides the
                        # free-tier ring; the next call attempts the chosen
                        # backend again.
                        _rescued = True
                        _resp = _rescue_search(
                            provider.name,
                            str(_resp.get("error", "")),
                            query,
                            _fetch_limit,
                        )
                return _resp, _rescued

            response_data = _search_memo.lookup(provider.name, query, limit)
            if response_data is None:
                with _search_memo.flight_lock(provider.name, query, limit):
                    # Re-check inside the lock: a concurrent identical call
                    # may have stored while this one waited.
                    response_data = _search_memo.lookup(
                        provider.name, query, limit
                    )
                    if response_data is None:
                        response_data, _was_rescued = _paid_search()
                        # Never cache a rescue-served response: it came from
                        # a ring vendor, not the chosen backend (wrong key),
                        # and caching it would make the one-shot rescue
                        # sticky for this query for a whole TTL — the next
                        # call must attempt the chosen backend again.
                        if not _was_rescued:
                            _search_memo.store(
                                provider.name, query, limit, response_data
                            )
            response_data = _slice_search_response(response_data, limit)

        debug_call_data["results_count"] = len(response_data.get("data", {}).get("web", []))
        result_json = json.dumps(response_data, indent=2, ensure_ascii=False)
        debug_call_data["final_response_size"] = len(result_json)
        _finish_debug("web_search_tool", debug_call_data)
        return result_json
    except Exception as e:
        return _finish_debug("web_search_tool", debug_call_data, f"Error searching web: {str(e)}")


def _memoized_search(provider, query: str, limit: int) -> dict:
    """TTL memo + single-flight around the paid vendor call (tools/web_result_cache.py); sits after every
    safety/config check. The provider is asked for the BUCKETED count so near-identical limits share an entry;
    the caller's count is sliced out. Only successful, non-rescued responses are cached — caching a rescue
    would make the one-shot ring fallback sticky for a whole TTL."""
    from tools.web_result_cache import bucket_limit, search_memo, slice_search_response

    def _paid_search() -> tuple[dict, bool]:
        fetch_limit = bucket_limit(limit)
        try:
            resp = provider.search(query, fetch_limit)
        except Exception as exc:  # noqa: BLE001 — candidate for rescue
            if not _rescue_eligible(provider):
                raise
            return _rescue_search(provider.name, str(exc), query, fetch_limit), True
        if not resp.get("success") and _rescue_eligible(provider):
            return _rescue_search(provider.name, str(resp.get("error", "")), query, fetch_limit), True
        return resp, False

    response_data = search_memo.lookup(provider.name, query, limit)
    if response_data is None:
        with search_memo.flight_lock(provider.name, query, limit):
            # Re-check inside the lock: a concurrent identical call may have stored.
            response_data = search_memo.lookup(provider.name, query, limit)
            if response_data is None:
                response_data, was_rescued = _paid_search()
                if not was_rescued:
                    search_memo.store(provider.name, query, limit, response_data)
    return slice_search_response(response_data, limit)


async def web_extract_tool(urls: List[Any], format: str = None, char_limit: Optional[int] = None) -> str:
    """Extract clean page content (no LLM) from URLs via the configured backend.

    Pages over ``char_limit`` (default web.extract_char_limit or 15000) are head+tail truncated with a footer
    pointing at the stored full text; inline base64 images become ``[IMAGE: alt]``. URLs carrying secrets are
    refused before any fetch; private-network URLs are blocked per entry. Returns JSON ``{"results": [...]}``.
    """
    normalized_urls, normalized_indices, invalid_urls, blocked = _validate_extract_urls(urls)
    if blocked is not None:
        return blocked
    debug_call_data = {
        "parameters": {"urls": normalized_urls, "format": format, "char_limit": char_limit}, "error": None,
        "pages_extracted": 0, "pages_truncated": 0, "original_response_size": 0, "final_response_size": 0,
        "truncation_metrics": [], "processing_applied": [],
    }

    try:
        logger.info("Extracting content from %d URL(s)", len(normalized_urls))
        # SSRF protection — filter private/internal URLs before any backend.
        safe_urls, safe_indices, ssrf_blocked = [], [], {}
        for index, url in zip(normalized_indices, normalized_urls):
            if await async_is_safe_url(url):
                safe_urls.append(url)
                safe_indices.append(index)
            else:
                ssrf_blocked[index] = _result_entry(
                    url, "Blocked: URL targets a private or internal network address"
                )

        results = []
        if safe_urls:
            backend = _get_extract_backend()
            _ensure_web_plugins_loaded()
            from agent.web_search_registry import (
                get_active_extract_provider,
                get_provider as _wsp_get_provider,
                _disabled_web_plugin_for,
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
                from tools.tool_backend_helpers import (
                    selection_error,
                    selection_exists,
                )

                if backend and selection_exists("web"):
                    # Strict selection: a stored-but-unregistered backend
                    # errors by name instead of silently switching to
                    # whatever the availability walk finds.
                    disabled_key = _disabled_web_plugin_for(capability="extract")
                    if disabled_key:
                        _vendor = disabled_key.split("/", 1)[-1]
                        error_text = (
                            f"web.extract_backend is set to '{_vendor}', but "
                            f"its plugin ('{disabled_key}') is disabled in "
                            f"config. Re-enable it with `hermes plugins "
                            f"enable {disabled_key}` (or remove it from "
                            "plugins.disabled)."
                        )
                    else:
                        error_text = selection_error(
                            "web",
                            f"'{backend}'",
                            "no registered web extract provider has that name",
                        )
                    return json.dumps(
                        {"success": False, "error": error_text},
                        ensure_ascii=False,
                    )
                provider = get_active_extract_provider()
                if provider is None:
                    # If the configured backend is a bundled web plugin the
                    # user explicitly disabled, the backend is set correctly
                    # and the real fix is to re-enable the plugin — say so
                    # instead of telling them to set web.extract_backend
                    # (which they already did). #40190 follow-up.
                    disabled_key = _disabled_web_plugin_for(capability="extract")
                    if disabled_key:
                        _vendor = disabled_key.split("/", 1)[-1]
                        return json.dumps(
                            {
                                "success": False,
                                "error": (
                                    f"web.extract_backend is set to '{_vendor}', "
                                    f"but its plugin ('{disabled_key}') is disabled "
                                    "in config. Re-enable it with "
                                    f"`hermes plugins enable {disabled_key}` "
                                    "(or remove it from plugins.disabled)."
                                ),
                            },
                            ensure_ascii=False,
                        )
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


            # ── Extract cache (tools/web_result_cache.py) ─────────────────
            # Disk-backed via cache/web: a URL extracted within the TTL is
            # served from disk instead of re-scraped. Deliberately placed
            # AFTER the secret-URL gate, SSRF gate, provider resolution, and
            # strict-selection validation, and gated per-URL on the website
            # blocklist policy — a hit skips only the vendor call, never a
            # control. Policy-blocked URLs are treated as cache misses so
            # dispatch handles them exactly as it would without a cache.
            # Keys include the provider and format, so switching backends or
            # formats within the TTL never serves the other's content.
            from tools.web_result_cache import (
                extract_cache_get as _extract_cache_get,
                extract_cache_put as _extract_cache_put,
            )
            from tools.website_policy import check_website_access as _check_site
            cached_results: Dict[int, Dict[str, Any]] = {}
            fetch_urls: List[str] = []
            fetch_positions: List[int] = []
            for position, url in enumerate(safe_urls):
                hit = None
                try:
                    _policy_block = _check_site(url)
                except Exception:  # noqa: BLE001 — policy errors fail open like dispatch
                    _policy_block = None
                if _policy_block is None:
                    hit = _extract_cache_get(
                        url, format=format, provider=provider.name
                    )
                if hit is not None:
                    cached_results[position] = hit
                else:
                    fetch_urls.append(url)
                    fetch_positions.append(position)

            if not fetch_urls:
                results = [cached_results[i] for i in range(len(safe_urls))]
            else:
                logger.info(
                    "Web extract via %s: %d URL(s)", provider.name, len(fetch_urls)
                )

                # Async-or-sync dispatch: parallel + firecrawl have async
                # extract(); exa + tavily are sync.
                import inspect
                _extract_rescued = False
                try:
                    if inspect.iscoroutinefunction(provider.extract):
                        results = await provider.extract(fetch_urls, format=format)
                    else:
                        # Run sync extract() in a thread so we don't block the
                        # event loop on network I/O.
                        results = await asyncio.to_thread(
                            provider.extract, fetch_urls, format=format
                        )
                except Exception as exc:  # noqa: BLE001 — candidate for rescue
                    if _rescue_eligible(provider):
                        _extract_rescued = True
                        failed = [
                            {"url": u, "title": "", "content": "", "error": str(exc)}
                            for u in fetch_urls
                        ]
                        results = await asyncio.to_thread(
                            _rescue_extract, provider.name, fetch_urls, failed
                        )
                    else:
                        raise
                else:
                    # One-shot keyless rescue when the WHOLE batch failed
                    # (backend-level outage, not per-page problems). Stateless:
                    # the next web_extract call uses the chosen backend again.
                    if (
                        results
                        and all(r.get("error") for r in results)
                        and _rescue_eligible(provider)
                    ):
                        _extract_rescued = True
                        results = await asyncio.to_thread(
                            _rescue_extract, provider.name, fetch_urls, results
                        )

                # Cache each successful fetch's full clean text for TTL reuse
                # (best-effort; oversized pages are skipped by the cache).
                # NEVER cache a rescue-served batch: it came from a ring
                # vendor, not the chosen backend, and caching it would make
                # the one-shot rescue sticky for a whole TTL — the next call
                # must attempt the chosen backend again.
                if not _extract_rescued:
                    for fetched_pos, fetched in enumerate(results):
                        if fetched_pos >= len(fetch_urls):
                            break
                        if fetched.get("error"):
                            continue
                        _content = (
                            fetched.get("raw_content", "") or fetched.get("content", "")
                        )
                        if _content:
                            _extract_cache_put(
                                fetch_urls[fetched_pos],
                                _content,
                                title=fetched.get("title", ""),
                                format=format,
                                provider=provider.name,
                            )

                # Merge fetched results back with cache hits, restoring the
                # safe_urls order the downstream reconstruction expects.
                if cached_results:
                    merged: List[Dict[str, Any]] = [None] * len(safe_urls)  # type: ignore[list-item]
                    for position, hit in cached_results.items():
                        merged[position] = hit
                    for fetched_pos, position in enumerate(fetch_positions):
                        merged[position] = (
                            results[fetched_pos]
                            if fetched_pos < len(results)
                            else {
                                "url": safe_urls[position],
                                "title": "",
                                "content": "",
                                "error": "Extract backend returned no result for this URL",
                            }
                        )
                    results = merged

        # Reconstruct the original input order across invalid, blocked, and
        # provider-processed entries. Providers are expected to preserve the
        # order of the safe URL list they receive.
        if invalid_urls or ssrf_blocked:
            fixed = {**ssrf_blocked, **invalid_urls}
            results = _merge_in_order(len(urls), fixed, safe_indices, safe_urls, results)

        logger.info("Extracted content from %d pages", len(results))
        debug_call_data["pages_extracted"] = len(results)
        debug_call_data["original_response_size"] = len(json.dumps({"results": results}))
        debug_call_data["processing_applied"].append("truncate_and_store")
        _truncate_results(results, _effective_char_limit(char_limit), debug_call_data)
        trimmed = _trim_results(results)
        result_json = (
            json.dumps({"results": trimmed}, indent=2, ensure_ascii=False) if trimmed
            else tool_error("Content was inaccessible or not found")
        )
        # Belt-and-suspenders sweep of the serialized JSON: a provider may tuck a base64 blob in metadata.
        cleaned_result = convert_base64_images_to_links(result_json)
        debug_call_data["final_response_size"] = len(cleaned_result)
        debug_call_data["processing_applied"].append("base64_image_conversion")
        _finish_debug("web_extract_tool", debug_call_data)
        return cleaned_result
    except Exception as e:
        return _finish_debug("web_extract_tool", debug_call_data, f"Error extracting content: {str(e)}")


# Convenience function to check Firecrawl credentials
def _provider_is_ready(provider) -> bool:
    """Return True when *provider* reports readiness without raising.

    ``get_active_*_provider()`` intentionally returns an explicitly configured
    backend even when ``is_available()`` is False so the dispatcher can emit a
    precise missing-credential error. Tool/doctor readiness gates must still
    require a true availability probe — otherwise ``hermes doctor`` paints a
    green ✓ for a backend that cannot run (issue #78412).

    A provider that can serve anonymously (``is_keyless_available()`` — the
    Exa/Parallel free tier) IS ready: keyless mode is a working state, not a
    misconfiguration.
    """
    if provider is None:
        return False
    try:
        if provider.is_available():
            return True
    except Exception as exc:  # noqa: BLE001 — broken provider == not ready
        logger.debug(
            "web provider %r.is_available() raised during readiness check: %s",
            getattr(provider, "name", provider),
            exc,
        )
        return False
    try:
        return bool(provider.is_keyless_available())
    except Exception as exc:  # noqa: BLE001 — broken provider == not ready
        logger.debug(
            "web provider %r.is_keyless_available() raised during readiness check: %s",
            getattr(provider, "name", provider),
            exc,
        )
        return False


def check_web_api_key() -> bool:
    """Check whether the configured web backend is available.

    ``get_active_*_provider()`` returns an explicitly configured backend even when ``is_available()`` is
    False (so dispatch can emit a precise error), so readiness gates (tool check_fn, ``hermes doctor``)
    must probe for real. Keyless mode (Exa/Parallel free tier) is a working state, not a misconfig.

    See #78412.
    """
    # ``or ""``: a null ``web.backend`` value yields None from ``.get``, and
    # ``None.lower()`` would raise. Mirrors ``_get_backend``.
    configured = (_load_web_config().get("backend") or "").lower().strip()
    if configured and _is_backend_available(configured):
        return True
    # Any built-in backend with credentials present. This is a boolean OR, so
    # unlike _get_backend() the probe order is irrelevant.
    if any(_is_backend_available(backend) for backend in _LEGACY_WEB_BACKENDS):
        return True
    # Plugin-registered path: the active-provider resolvers return an explicit
    # config hit even when credentials are missing (so the tool can print a
    # precise "set FOO_API_KEY" error). Readiness still requires a true
    # availability probe — keyed (is_available) OR keyless-capable
    # (is_keyless_available; the Exa/Parallel anonymous free tier serves
    # zero-credential installs, so those count as ready). Discovery must run
    # first — check_fn fires at tool-registration time, before any dispatch
    # has populated the registry.
    try:
        _ensure_web_plugins_loaded()
        from agent.web_search_registry import (
            get_active_search_provider,
            get_active_extract_provider,
        )

        return (
            _provider_is_ready(get_active_search_provider())
            or _provider_is_ready(get_active_extract_provider())
        )
    except Exception as exc:  # noqa: BLE001 — registry optional; never fatal
        logger.debug("web provider registry availability check failed: %s", exc)
        return False

if __name__ == "__main__":
    """
    Simple test/demo when run directly
    """
    print("🌐 Standalone Web Tools Module")
    print("=" * 40)

    # Check if API keys are available
    web_available = check_web_api_key()
    tool_gateway_available = _is_tool_gateway_ready()
    from hermes_cli.config import get_env_value as _gev
    firecrawl_key_available = bool((_gev("FIRECRAWL_API_KEY") or "").strip())
    firecrawl_url_available = bool((_gev("FIRECRAWL_API_URL") or "").strip())

    if web_available:
        backend = _get_backend()
        print(f"✅ Web backend: {backend}")
        if backend == "exa":
            print("   Using Exa API (https://exa.ai)")
        elif backend == "parallel":
            print("   Using Parallel API (https://parallel.ai)")
        elif backend == "tavily":
            if _has_env("TAVILY_API_KEY"):
                print("   Using Tavily API (https://tavily.com)")
            else:
                print("   Using Tavily keyless (https://docs.tavily.com/documentation/keyless)")
        elif backend == "searxng":
            print(f"   Using SearXNG (search only): {_env_value('SEARXNG_URL')}")
        elif backend == "brave-free":
            print("   Using Brave Search free tier (search only)")
        elif backend == "ddgs":
            print("   Using DuckDuckGo via ddgs package (search only)")
        elif firecrawl_url_available:
            print(f"   Using self-hosted Firecrawl: {(_gev('FIRECRAWL_API_URL') or '').strip().rstrip('/')}")
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
    name="web_search", toolset="web", schema=WEB_SEARCH_SCHEMA,
    handler=lambda args, **kw: web_search_tool(args.get("query", ""), limit=args.get("limit", 5)),
    check_fn=check_web_api_key, requires_env=_web_requires_env(), emoji="🔍",
    max_result_size_chars=100_000,
)
registry.register(
    name="web_extract", toolset="web", schema=WEB_EXTRACT_SCHEMA,
    handler=lambda args, **kw: web_extract_tool(
        args.get("urls", [])[:5] if isinstance(args.get("urls"), list) else [], "markdown",
        char_limit=args.get("char_limit"),
    ),
    check_fn=check_web_api_key, requires_env=_web_requires_env(), is_async=True, emoji="📄",
    max_result_size_chars=100_000,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Dict  # noqa: F401,E402
from typing import TYPE_CHECKING  # noqa: F401,E402
import asyncio  # noqa: F401,E402
import httpx  # noqa: F401,E402
import re  # noqa: F401,E402
import sys  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_EXTRACT_CHAR_LIMIT': ('tools.web_tools_truncate', 'DEFAULT_EXTRACT_CHAR_LIMIT'),
    'Firecrawl': ('plugins.web.firecrawl.provider', 'Firecrawl'),
    'MAX_STORED_TEXT_CHARS': ('tools.web_tools_truncate', 'MAX_STORED_TEXT_CHARS'),
    'build_vendor_gateway_url': ('tools.managed_tool_gateway', 'build_vendor_gateway_url'),
    'managed_nous_tools_enabled': ('tools.tool_backend_helpers', 'managed_nous_tools_enabled'),
    'normalize_url_for_request': ('tools.url_safety', 'normalize_url_for_request'),
    'nous_tool_gateway_unavailable_message': ('tools.tool_backend_helpers', 'nous_tool_gateway_unavailable_message'),
    'prefers_gateway': ('tools.tool_backend_helpers', 'prefers_gateway'),
    'resolve_managed_tool_gateway': ('tools.managed_tool_gateway', 'resolve_managed_tool_gateway'),
    'sensitive_query_param_name': ('tools.url_safety', 'sensitive_query_param_name'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----

"""web_extract helpers: URL validation, provider resolution, cache-aware dispatch, fallback chain.

Order of controls (each is a gate, never skipped by a cache hit): secret-URL
refusal -> SSRF filter (in web_tools.web_extract_tool) -> provider resolution
(strict selection; per entry of the ``web.extract_backends`` chain) -> per-URL
website policy -> disk cache -> vendor call with one-shot keyless rescue (final
chain entry only) -> website policy + SSRF on every page reported under a URL
that was not requested (redirect targets). Logs under the origin
(tools.web_tools) logger.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from tools.tool_backend_helpers import selection_error, selection_exists
from tools.url_safety import normalize_url_for_request
from tools.web_tools_rescue import _policy_blocked_result, _rescue_eligible, _rescue_extract

logger = logging.getLogger("tools.web_tools")

_NO_RESULT_ERROR = "Extract backend returned no result for this URL"
_DEFAULT_EXTRACT_TIMEOUT_S = 120.0
_EXTRACT_BACKENDS_HINT = "firecrawl, tavily, keenable, exa, or parallel."
_INVALID_ITEM_ERROR = (
    "Invalid URL item at index {}: expected a URL string or an object with a string 'url' or 'href' field"
)


def _web_extract_url(value: Any) -> Optional[str]:
    """URL from a model-supplied extract item (str, or dict with ``url``/``href``); None if unusable.

    Models sometimes forward a whole search result instead of its URL, hence the dict form. Never
    stringify arbitrary objects into misleading fetch targets.
    """
    if isinstance(value, dict):
        value = value.get("url") or value.get("href")
    return (value.strip() or None) if isinstance(value, str) else None


def _disabled_plugin_error(capability: str, disabled_key: str, key: Optional[str] = None) -> str:
    """Error text when the configured backend's bundled plugin is disabled in config. *key* names the
    config key that selected it (default ``web.<capability>_backend``; chain entries pass
    ``web.extract_backends``)."""
    vendor = disabled_key.split("/", 1)[-1]
    return (
        f"{key or f'web.{capability}_backend'} is set to '{vendor}', but its plugin ('{disabled_key}') is "
        f"disabled in config. Re-enable it with `hermes plugins enable {disabled_key}` "
        "(or remove it from plugins.disabled)."
    )


def _no_provider_error(capability: str, fallback: str) -> str:
    """Error when no provider resolved: point at a disabled bundled plugin if that is the real cause."""
    from agent.web_search_registry import _disabled_web_plugin_for
    disabled_key = _disabled_web_plugin_for(capability=capability)
    return _disabled_plugin_error(capability, disabled_key) if disabled_key else fallback


def _strict_selection_error(capability: str, backend: str) -> str:
    """Error for a stored-but-unregistered backend: name the disabled plugin, else the bad selection.
    Strict selection never silently switches to whatever the availability walk finds."""
    failure = f"no registered web {capability} provider has that name"
    return _no_provider_error(capability, selection_error("web", f"'{backend}'", failure))


def _result_entry(url: str, error: Optional[str]) -> dict[str, Any]:
    return {"url": url, "title": "", "content": "", "error": error}


def _extract_error_json(error: str) -> str:
    return json.dumps({"success": False, "error": error}, ensure_ascii=False)


def _refuse_all(error: str):
    """Whole-call refusal tuple for ``_validate_extract_urls`` (exfiltration prevention)."""
    return None, None, None, json.dumps({"success": False, "error": error})


def _merge_in_order(
    total: int, fixed: dict[int, dict], fetch_positions: list[int], fetch_urls: list[str], results: list[dict]
) -> list[dict]:
    """Rebuild a ``total``-long result list: *fixed* entries by position, fetched *results* at
    *fetch_positions* (a short provider list yields ``_NO_RESULT_ERROR`` entries for the rest)."""
    merged = dict(fixed)
    for pos, position in enumerate(fetch_positions):
        missing = _result_entry(fetch_urls[pos], _NO_RESULT_ERROR)
        merged[position] = results[pos] if pos < len(results) else missing
    return [merged[i] for i in range(total)]


def _validate_extract_urls(urls: list[Any]):
    """Normalize model-supplied items and block URLs carrying secrets (percent-encoded forms are unquoted
    and checked too). Returns ``(normalized_urls, normalized_indices, invalid_urls, blocked_json)``;
    ``blocked_json`` is a whole-call refusal (exfiltration prevention) or None."""
    from agent.redact import _PREFIX_RE
    from urllib.parse import unquote

    normalized_urls, normalized_indices, invalid_urls = [], [], {}
    for index, item in enumerate(urls):
        _url = _web_extract_url(item)
        if _url is None:
            invalid_urls[index] = _result_entry("", _INVALID_ITEM_ERROR.format(index))
            continue
        normalized_url = normalize_url_for_request(_url)
        if any(_PREFIX_RE.search(c) for c in (_url, unquote(_url), normalized_url, unquote(normalized_url))):
            return _refuse_all(
                "Blocked: URL contains what appears to be an API key or token. "
                "Secrets must not be sent in URLs."
            )
        normalized_urls.append(normalized_url)
        normalized_indices.append(index)
    return normalized_urls, normalized_indices, invalid_urls, None


def _resolve_extract_provider(backend: str):
    """Resolve the extract provider for *backend*; returns ``(provider, error_json)``.

    A registered search-only backend is a typed error (never a silent switch). An unregistered name with
    a stored web selection is a strict-selection error; with no selection, fall through to the walk.
    """
    from agent.web_search_registry import get_active_extract_provider, get_provider as _wsp_get_provider
    provider = _wsp_get_provider(backend) if backend else None
    if provider is not None and provider.supports_extract():
        return provider, None
    if provider is not None:
        return None, _extract_error_json(
            f"{provider.display_name} is a search-only backend and cannot extract URL content. "
            "Set web.extract_backend to " + _EXTRACT_BACKENDS_HINT
        )
    if backend and selection_exists("web"):
        return None, _extract_error_json(_strict_selection_error("extract", backend))
    provider = get_active_extract_provider()
    if provider is None:
        fallback = "No web extract provider configured. Set web.extract_backend to " + _EXTRACT_BACKENDS_HINT
        return None, _extract_error_json(_no_provider_error("extract", fallback))
    return provider, None


def _extract_timeout_seconds() -> float:
    """Wall-clock cap for one provider ``extract()`` dispatch (``web.extract_timeout``, default 120s).

    A hanging backend (server keeps the response open without finishing) otherwise stalls the
    tool call indefinitely. 0 or a negative value disables the cap.
    """
    from tools.web_tools import _load_web_config
    try:
        return float(_load_web_config().get("extract_timeout", _DEFAULT_EXTRACT_TIMEOUT_S))
    except (TypeError, ValueError):
        return _DEFAULT_EXTRACT_TIMEOUT_S


def _backend_failed_whole_batch(results: list[dict]) -> bool:
    from plugins.web._common import is_origin_http_error
    errors = [r.get("error") for r in results]
    return bool(errors) and all(errors) and not any(is_origin_http_error(e) for e in errors)


async def _fetch_with_rescue(
    provider, fetch_urls: list[str], format: Optional[str], *, rescue: bool = True
) -> tuple[list[dict], bool]:
    """Call ``provider.extract`` (async or sync-in-thread), with one-shot keyless rescue.

    Rescue fires on a raised exception — including a dispatch timeout — or when the WHOLE batch
    failed (backend outage, not per-page problems): an origin HTTP 4xx/5xx is the page's own answer,
    so a batch with one never triggers it. Returns ``(results, rescued)``. ``rescue=False`` withholds
    it (non-final ``web.extract_backends`` entries: the configured chain, not the free ring, is what
    such a failure falls through to) — the failure then surfaces raw: the exception propagates, a
    timeout/all-error batch is returned as-is.
    """
    import inspect
    timeout = _extract_timeout_seconds()
    try:
        if inspect.iscoroutinefunction(provider.extract):
            coro = provider.extract(fetch_urls, format=format)
        else:  # sync extract() runs in a thread so network I/O never blocks the loop
            coro = asyncio.to_thread(provider.extract, fetch_urls, format=format)
        if timeout > 0:
            results = await asyncio.wait_for(coro, timeout=timeout)
        else:
            results = await coro
    except TimeoutError:  # hanging backend — bounded, never a stalled tool call
        logger.warning("web_extract provider '%s' timed out after %.0fs for %d URL(s)",
                       provider.name, timeout, len(fetch_urls))
        failed = [_result_entry(u, f"Extract timed out after {timeout:.0f}s via {provider.name}")
                  for u in fetch_urls]
        if not (rescue and _rescue_eligible(provider)):
            return failed, False
        return await asyncio.to_thread(_rescue_extract, provider.name, fetch_urls, failed), True
    except Exception as exc:
        if not (rescue and _rescue_eligible(provider)):
            raise
        failed = [_result_entry(u, str(exc)) for u in fetch_urls]
        return await asyncio.to_thread(_rescue_extract, provider.name, fetch_urls, failed), True
    if _backend_failed_whole_batch(results) and rescue and _rescue_eligible(provider):
        return await asyncio.to_thread(_rescue_extract, provider.name, fetch_urls, results), True
    return results, False


async def _final_url_refusal(fetched: dict) -> Optional[dict]:
    """Refusal entry when a page's reported (post-redirect) URL fails the website policy or SSRF gate,
    in the shape Firecrawl's own redirect re-check returns; None when it passes. A policy check that
    raises refuses (fail closed): the requested URL passed, but nothing vouches for this one."""
    from tools.url_safety import async_is_safe_url
    from tools.website_policy import check_website_access
    final_url, title = fetched["url"], fetched.get("title", "")
    refusal = {"url": final_url, "title": title, "content": "", "raw_content": ""}
    try:
        blocked = check_website_access(final_url)
    except Exception as exc:
        logger.warning("Website policy check failed for redirect target %s", final_url, exc_info=True)
        return {**refusal, "error": f"Website policy check failed for redirect target: {exc}"}
    if blocked:
        logger.info("Blocked redirected web_extract for %s by rule %s", blocked["host"], blocked["rule"])
        policy = {k: blocked[k] for k in ("host", "rule", "source")}
        return {**refusal, "error": blocked["message"], "blocked_by_policy": policy}
    if not await async_is_safe_url(final_url):
        logger.info("Blocked redirected web_extract for unsafe final URL: %s", final_url)
        return {**refusal, "error": "Blocked: URL targets a private or internal network address"}
    return None


async def _gate_final_urls(results: list[dict], requested: set[str]) -> list[dict]:
    """Re-gate every page a provider reports under a URL other than one we asked for (a redirect
    target, a canonicalised host): the SSRF and website-policy checks only ever saw the requested URL.
    Covers every provider shape — Keenable reports the target in ``url`` with ``sourceURL`` = requested,
    Exa/Parallel/Tavily report the vendor's URL in both — and rescued batches alike."""
    gated = []
    for fetched in results:
        url = fetched.get("url")
        if url and url not in requested and not fetched.get("error"):
            fetched = await _final_url_refusal(fetched) or fetched
        gated.append(fetched)
    return gated


async def _dispatch_extract(
    provider, fetch_urls: list[str], format: Optional[str], *, rescue: bool = True
) -> list[dict]:
    """Fetch *fetch_urls* (keyless rescue unless withheld), re-gate redirect targets, then cache the
    passes. Rescued batches are never cached."""
    from tools.web_result_cache import extract_cache_put, extract_cache_record_final_url
    results, rescued = await _fetch_with_rescue(provider, fetch_urls, format, rescue=rescue)
    requested = set(fetch_urls)
    results = await _gate_final_urls(results, requested)
    if rescued:
        return results

    # Cache each successful fetch under the REQUESTED url it reports as its own — never by list
    # position: providers omit failed URLs or return successes out of request order, and a positional
    # write filed one page's text under another URL's key for the whole TTL. ``metadata.sourceURL``
    # counts because Keenable/Firecrawl put the requested URL there when ``url`` is the redirect target.
    # An entry naming no requested URL is served but not cached (a miss re-fetches; a mis-key poisons).
    for fetched in results:
        meta = fetched.get("metadata")
        source = meta.get("sourceURL") if isinstance(meta, dict) else None
        url = next((u for u in (fetched.get("url"), source) if u in requested), None)
        _content = fetched.get("raw_content", "") or fetched.get("content", "")
        if url and _content and not fetched.get("error"):
            stored = extract_cache_put(url, _content, fetched.get("title", ""), format=format, provider=provider.name)
            if stored and fetched.get("url") and fetched["url"] != url:
                extract_cache_record_final_url(url, fetched["url"], _content, format=format, provider=provider.name)
    return results


async def _cached_final_url_allowed(hit: dict, requested: str) -> bool:
    """Gate a cache hit on the URL its content came from: a redirector cached before its target was
    blocklisted (or turned private) must not keep serving that target. Consumes ``final_url``."""
    final_url = hit.pop("final_url", requested)
    if final_url == requested:  # the requested URL already passed both gates
        return True
    from tools.url_safety import async_is_safe_url
    from tools.website_policy import check_website_access
    try:
        if check_website_access(final_url) is not None:
            return False
    except Exception:  # fail closed: a miss, and the live fetch re-gates the redirect
        logger.warning("Website policy check failed for cached final URL %s", final_url, exc_info=True)
        return False
    return await async_is_safe_url(final_url)


async def _extract_safe_urls(
    provider, safe_urls: list[str], format: Optional[str], *, rescue: bool = True
) -> list[dict]:
    """Serve cache hits, fetch the rest, and merge back in ``safe_urls`` order.

    The disk cache (tools/web_result_cache.py) sits AFTER the secret-URL gate, SSRF gate, and provider
    resolution, and is gated per-URL on the website policy — a hit skips only the vendor call, never a
    control; policy-blocked URLs are cache misses, judged on both the requested and the cached final
    URL. Keys include provider and format, so switching either within the TTL never serves the other's
    content. ``rescue`` is forwarded to :func:`_dispatch_extract`."""
    from tools.web_result_cache import extract_cache_get
    from tools.website_policy import check_website_access as _check_site
    cached_results, fetch_urls, fetch_positions = {}, [], []
    for position, url in enumerate(safe_urls):
        try:
            _policy_block = _check_site(url)
        except Exception:
            _policy_block = None
        hit = extract_cache_get(url, format=format, provider=provider.name) if _policy_block is None else None
        if hit is not None and not await _cached_final_url_allowed(hit, url):
            hit = None  # a miss: the live fetch re-checks the redirect and refuses with blocked_by_policy
        if hit is not None:
            cached_results[position] = hit
        else:
            fetch_urls.append(url)
            fetch_positions.append(position)

    if not fetch_urls:
        return [cached_results[i] for i in range(len(safe_urls))]
    logger.info("Web extract via %s: %d URL(s)", provider.name, len(fetch_urls))
    results = await _dispatch_extract(provider, fetch_urls, format, rescue=rescue)
    if not cached_results:
        return results
    return _merge_in_order(len(safe_urls), cached_results, fetch_positions, fetch_urls, results)


# ─── Fallback chain (web.extract_backends) ────────────────────────────────────

def _contentless(result: Any) -> bool:
    """True when a result row carries no usable page text — neither ``content`` nor ``raw_content`` has
    any non-whitespace. Backends answer an unhydrated SPA, a soft bot wall, or a payload with neither
    markdown nor HTML with HTTP 200 and an empty body and NO ``error`` (firecrawl's ``_scrape_one`` is
    one such shape); for the fallback chain that row is as retryable as an explicit failure."""
    if not isinstance(result, dict):
        return True
    return not any(
        isinstance(result.get(key), str) and result[key].strip() for key in ("content", "raw_content")
    )


def _failed_row(result: Any) -> bool:
    """A failure-shaped result row: not a dict, carries an ``error``, or is contentless."""
    return not isinstance(result, dict) or bool(result.get("error")) or _contentless(result)


def _batch_failed(results: list[dict]) -> bool:
    """True when the batch is empty or EVERY row is failure-shaped — the retryable outcome for the chain.
    Partial success (some pages usable) is a final answer: the batch is not shopped to the next backend."""
    return not results or all(_failed_row(r) for r in results)


def _policy_blocked_batch(results: list[dict]) -> bool:
    """True when any row is a website-policy refusal — a terminal decision, never re-dispatched."""
    return any(_policy_blocked_result(r) for r in results if isinstance(r, dict))


def _error_text(error_json: Optional[str]) -> str:
    """The ``error`` field of a ``_extract_error_json`` envelope, for log lines."""
    try:
        return str(json.loads(error_json or "{}").get("error") or error_json or "")
    except (TypeError, ValueError):
        return str(error_json or "")


def _resolve_chain_entry(backend: str):
    """Resolve one explicit ``web.extract_backends`` entry EXACTLY; returns ``(provider, error_json)``.

    An entry that is unregistered, or registered but search-only, is a typed error — never a silent
    substitute. In particular the scalar active provider (``get_active_extract_provider`` resolves
    ``web.extract_backend`` / ``web.backend``, a different key) is never dispatched in an entry's place:
    that could hit a backend the user never listed in this slot. A disabled bundled plugin is named as
    the cause when it is one.
    """
    from agent.web_search_registry import _disabled_web_plugin_for, get_provider as _wsp_get_provider
    provider = _wsp_get_provider(backend)
    if provider is not None and provider.supports_extract():
        return provider, None
    if provider is not None:
        return None, _extract_error_json(
            f"{provider.display_name} is a search-only backend and cannot extract URL content. "
            "Set web.extract_backends entries to " + _EXTRACT_BACKENDS_HINT
        )
    disabled_key = _disabled_web_plugin_for(configured=backend, capability="extract")
    if disabled_key:
        return None, _extract_error_json(_disabled_plugin_error("extract", disabled_key, key="web.extract_backends"))
    return None, _extract_error_json(
        f"web.extract_backends entry '{backend}' does not match any registered web extract provider. "
        "Set it to " + _EXTRACT_BACKENDS_HINT
    )


async def _extract_with_fallback(
    backends: list[str], safe_urls: list[str], format: Optional[str], *, explicit: bool
) -> tuple[list[dict], Optional[str]]:
    """Walk the extract backends in order; returns ``(results, error_json)`` (at most one is set).

    Each entry is resolved and dispatched through :func:`_extract_safe_urls` (policy, cache, timeout).
    A retryable outcome hands the batch to the next entry with a logged warning: the entry does not
    resolve, ``extract()`` raises, or the batch is empty or every row is failure-shaped — an ``error``,
    OR a contentless body (HTTP 200 + empty content, no error). A ``blocked_by_policy`` row is a
    terminal decision: the batch is returned as-is and the blocked URL is never re-dispatched. The
    one-shot keyless rescue is withheld from every entry but the last, so a failing entry falls through
    to the configured chain rather than the free ring.

    The FINAL entry's outcome is surfaced exactly as a single backend's would be — its rows (per-URL
    errors included), its raised exception, or ``[]``. Should the final entry fail to *resolve*, the most
    recent per-URL rows from an earlier entry are returned when there are any (a bad trailing entry must
    not erase a real fetch outcome), else the resolution error.

    ``explicit=False`` is the pre-chain path unchanged: the single scalar backend resolved via
    :func:`_resolve_extract_provider` (strict selection, active-provider walk for never-configured
    installs, rescue on).
    """
    resolve = _resolve_chain_entry if explicit else _resolve_extract_provider
    backends = backends or [""]  # no scalar resolved: let _resolve_extract_provider walk / report
    results: list[dict] = []  # most recent per-URL rows from an entry that was actually invoked
    last_error_json: Optional[str] = None
    for idx, backend in enumerate(backends):
        final = idx == len(backends) - 1
        provider, error_json = resolve(backend)
        if provider is None:
            last_error_json = error_json
            if not final or results:  # otherwise the error itself is the tool's answer — no log needed
                logger.warning("web_extract backend '%s' skipped: %s", backend, _error_text(error_json))
            continue
        try:
            attempt = await _extract_safe_urls(provider, safe_urls, format, rescue=final)
        except Exception as exc:
            if final:
                raise
            logger.warning(
                "web_extract via %s raised (%s); trying the next configured backend", provider.name, exc
            )
            continue
        if attempt:
            results = attempt
        if final or _policy_blocked_batch(attempt) or not _batch_failed(attempt):
            return attempt, None
        first = next((r.get("error") for r in attempt if isinstance(r, dict) and r.get("error")), None)
        logger.warning(
            "web_extract via %s returned no usable content for %d URL(s) (%s); trying the next configured backend",
            provider.name, len(safe_urls), first or ("empty response" if not attempt else "contentless rows"),
        )
    if results:
        return results, None
    return [], last_error_json

"""Opt-in ordered fallback chain for web_search / web_extract.

``web.search_fallback_chain`` / ``web.extract_fallback_chain`` name backends tried IN ORDER when the
selected backend fails a call (error result, exception, timeout, unresolvable selection), BEFORE the
one-shot keyless rescue. Explicit by design: the empty default keeps a stored selection's honest
error. A fallback-served result is annotated (``fallback_from`` / ``served_by`` / ``chain_errors`` /
``backend_error``) and never cached, so the next call attempts the selected backend first again.
Backends without credentials or without the capability are skipped and named in ``chain_errors``.
Logs under the origin (tools.web_tools) logger.
"""

import asyncio
import inspect
import logging
from typing import Any, Optional

logger = logging.getLogger("tools.web_tools")

_ERROR_SNIPPET = 300


def _snippet(error: Any) -> str:
    return (str(error or "") or "unknown error")[:_ERROR_SNIPPET]


def fallback_chain(capability: str, primary: str) -> list[str]:
    """Ordered, de-duplicated ``web.<capability>_fallback_chain`` names, the primary excluded.
    A bare string is a one-element chain."""
    from tools.web_tools import _load_web_config
    raw = _load_web_config().get(f"{capability}_fallback_chain")
    names = [raw] if isinstance(raw, str) else raw if isinstance(raw, (list, tuple)) else []
    primary = (primary or "").strip().lower()
    chain: list[str] = []
    for name in names:
        name = str(name or "").strip().lower()
        if name and name != primary and name not in chain:
            chain.append(name)
    return chain


def chain_summary(chain_errors: list[dict]) -> str:
    """``exa: <error>; tavily: <error>`` for error texts."""
    return "; ".join(f"{e['backend']}: {e['error']}" for e in chain_errors)


def _chain_provider(name: str, capability: str):
    """``(provider, None)`` when *name* can serve *capability* now, else ``(None, skip_reason)``."""
    from agent.web_search_registry import get_provider
    from tools.web_tools import _provider_is_ready
    provider = get_provider(name)
    if provider is None:
        return None, "skipped: no registered web provider has that name"
    supports = provider.supports_search() if capability == "search" else provider.supports_extract()
    if not supports:
        return None, f"skipped: {provider.display_name} cannot {capability}"
    if not _provider_is_ready(provider):
        return None, "skipped: not configured (no credentials)"
    return provider, None


def _fallback_note(capability: str, primary: str, served_by: str, primary_error: str) -> str:
    return (
        f"Configured backend '{primary}' failed this call ({_snippet(primary_error)}); result served by "
        f"fallback '{served_by}' (web.{capability}_fallback_chain). The next call will use '{primary}' again."
    )


def search_via_chain(primary: str, primary_error: str, query: str, limit: int) -> tuple[Optional[dict], list[dict]]:
    """Walk the search chain after *primary* failed. Returns ``(served_response | None, chain_errors)``;
    ``chain_errors`` is empty when no chain is configured. An empty result list is a success."""
    chain = fallback_chain("search", primary)
    if not chain:
        return None, []
    chain_errors = [{"backend": primary, "error": _snippet(primary_error)}]
    for name in chain:
        provider, skip = _chain_provider(name, "search")
        if provider is None:
            chain_errors.append({"backend": name, "error": skip})
            continue
        try:
            resp = provider.search(query, limit)
        except Exception as exc:  # any vendor/SDK error moves the call on; it is named in chain_errors
            logger.debug("web_search fallback '%s' raised", name, exc_info=True)
            resp = {"success": False, "error": str(exc)}
        if resp.get("success"):
            logger.warning("web_search backend '%s' failed (%s); served by fallback '%s'",
                           primary, _snippet(primary_error)[:200], name)
            resp.setdefault("data", {}).update(
                fallback_from=primary, served_by=name, chain_errors=chain_errors,
                backend_error=_fallback_note("search", primary, name, primary_error),
            )
            return resp, chain_errors
        chain_errors.append({"backend": name, "error": _snippet(resp.get("error"))})
    return None, chain_errors


def search_unresolved_primary(primary: str, selection_error: str, query: str, limit: int) -> Optional[dict]:
    """A stored selection that resolves to no provider: serve from the chain when one is configured,
    else None (the strict-selection error stands)."""
    served, chain_errors = search_via_chain(primary, selection_error, query, limit)
    if served is not None or not chain_errors:
        return served
    return {"success": False, "error": f"{selection_error} (fallback chain also failed: {chain_summary(chain_errors[1:])})",
            "data": {"chain_errors": chain_errors}}


async def call_extract(provider, urls: list[str], format: Optional[str], timeout: float) -> list[dict]:
    """``provider.extract`` (async, or sync in a thread so network I/O never blocks the loop), capped at
    *timeout* seconds (0 or negative = no cap). Raises ``TimeoutError`` on the cap."""
    if inspect.iscoroutinefunction(provider.extract):
        coro = provider.extract(urls, format=format)
    else:
        coro = asyncio.to_thread(provider.extract, urls, format=format)
    return await (asyncio.wait_for(coro, timeout=timeout) if timeout > 0 else coro)


def _retryable(entry: dict) -> bool:
    """A failed entry another backend might serve: not the page's own HTTP answer, not a policy block."""
    from plugins.web._common import is_origin_http_error
    from tools.web_tools_rescue import _policy_blocked_result
    error = entry.get("error")
    return bool(error) and not is_origin_http_error(error) and not _policy_blocked_result(entry)


def _by_requested_url(results: list[dict], urls: list[str]) -> list[Optional[dict]]:
    """Align *results* to *urls*: positional when the provider kept parity, else matched on ``url`` or
    ``metadata.sourceURL`` (None where the provider returned nothing for a URL)."""
    if len(results) == len(urls):
        return list(results)
    aligned: dict[str, dict] = {}
    for entry in results:
        meta = entry.get("metadata")
        source = meta.get("sourceURL") if isinstance(meta, dict) else None
        url = next((u for u in (entry.get("url"), source) if u in urls), None)
        if url and url not in aligned:
            aligned[url] = entry
    return [aligned.get(u) for u in urls]


async def _try_extract_backend(name: str, urls: list[str], format: Optional[str], timeout: float):
    """``(aligned_results | None, error | None)`` for one chain element on *urls*."""
    provider, skip = _chain_provider(name, "extract")
    if provider is None:
        return None, skip
    try:
        got = await call_extract(provider, urls, format, timeout)
    except TimeoutError:
        return None, f"timed out after {timeout:.0f}s"
    except Exception as exc:  # any vendor/SDK error moves the URLs on; it is named in chain_errors
        logger.debug("web_extract fallback '%s' raised", name, exc_info=True)
        return None, str(exc)
    return _by_requested_url(got, urls), None


def _mark_served(entry: dict, primary: str, served_by: str, primary_error: str) -> dict:
    meta = entry.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta.update(fallback_from=primary, served_by=served_by,
                    backend_error=_fallback_note("extract", primary, served_by, primary_error))
    return entry


async def extract_via_chain(primary: str, urls: list[str], results: list[dict], format: Optional[str],
                            timeout: float) -> tuple[list[dict], bool]:
    """Re-send only the URLs *primary* failed to the chain backends, in order; partial success stays.
    Returns ``(results aligned to urls, any_served)``; unchanged when no chain is configured."""
    from tools.web_tools_extract import _NO_RESULT_ERROR, _result_entry
    chain = fallback_chain("extract", primary)
    if not chain:
        return results, False
    aligned = [r if r is not None else _result_entry(u, _NO_RESULT_ERROR)
               for u, r in zip(urls, _by_requested_url(results, urls))]
    pending = [i for i, r in enumerate(aligned) if _retryable(r)]
    if not pending:
        return aligned, False
    primary_error = aligned[pending[0]].get("error")
    chain_errors = [{"backend": primary, "error": _snippet(primary_error)}]
    failed_count, served_any = len(pending), False
    for name in chain:
        if not pending:
            break
        got, error = await _try_extract_backend(name, [urls[i] for i in pending], format, timeout)
        still = []
        for i, entry in zip(pending, got or [None] * len(pending)):
            if entry is not None and not entry.get("error") and (entry.get("content") or entry.get("raw_content")):
                aligned[i] = _mark_served(entry, primary, name, primary_error)
                served_any = True
            else:
                still.append(i)
                error = error or (entry or {}).get("error") or "no result for this URL"
        if still:
            chain_errors.append({"backend": name, "error": _snippet(error)})
        pending = still
    summary = chain_summary(chain_errors[1:])
    for i in pending:
        aligned[i] = {**aligned[i], "error": f"{aligned[i].get('error')} (fallback chain also failed: {summary})"}
    logger.warning("web_extract backend '%s' failed %d URL(s); fallback chain served %d",
                   primary, failed_count, failed_count - len(pending))
    return aligned, served_any


def is_chain_served(entry: dict) -> bool:
    """True for an entry a fallback backend served — never cached under the primary's key."""
    meta = entry.get("metadata")
    return isinstance(meta, dict) and bool(meta.get("served_by"))

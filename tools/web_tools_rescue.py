"""One-shot keyless-ring rescue for failed keyed/configured web calls.

Stateless by design: a rescue routes THIS call through the free-tier ring (plugins/web/keyless_mcp.py);
the next web_search/web_extract call attempts the chosen backend again. Callers must never cache a
rescue-served response, or the one-shot rescue becomes sticky for a whole TTL. Logs under the origin
(tools.web_tools) logger.
"""

import asyncio
import logging

logger = logging.getLogger("tools.web_tools")

# Ring vendor -> env var holding its paid key (keyed mode ⇒ eligible for rescue).
_RING_KEY_VARS = {
    "exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY",
    "firecrawl": "FIRECRAWL_API_KEY", "keenable": "KEENABLE_API_KEY",
}


def _keyless_rescue_enabled() -> bool:
    """``web.keyless_rescue`` (default on), implicitly off when the keyless tier is disabled."""
    from tools.web_tools import _load_web_config
    if not _load_web_config().get("keyless_rescue", True):
        return False
    try:
        from agent.web_search_registry import _keyless_tier_enabled
        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — registry optional
        logger.debug("keyless rescue tier check failed: %s", exc)
        return False


def _rescue_eligible(provider) -> bool:
    """True when a failed call on *provider* should get a one-shot rescue.

    Eligible: a keyed/configured path — any non-ring backend, or a ring vendor in keyed mode. A ring
    vendor already in keyless mode is NOT eligible: its failure means the ring was already walked.
    """
    if not _keyless_rescue_enabled() or provider is None:
        return False
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING, use_keyless
        name = getattr(provider, "name", "")
        if name not in _KEYLESS_RING:
            return True
        from agent.web_search_provider import get_provider_env
        key_var = _RING_KEY_VARS.get(name, "")
        return not use_keyless(name, get_provider_env(key_var) if key_var else "")
    except Exception as exc:  # noqa: BLE001 — rescue is best-effort
        logger.debug("rescue eligibility check failed: %s", exc)
        return False


def _rescue_search(provider_name: str, original_error: str, query: str, limit: int) -> dict:
    """Rescue a failed search via the ring; annotate the result with the original failure."""
    from plugins.web.keyless_mcp import search_with_failover
    logger.warning(
        "web_search backend '%s' failed (%s); one-shot keyless rescue",
        provider_name, (original_error or "")[:200],
    )
    rescued = search_with_failover(provider_name, query, limit)
    if rescued.get("success"):
        rescued.setdefault("data", {}).update(
            rescued_from=provider_name,
            backend_error=(
                f"Configured backend '{provider_name}' failed this call "
                f"({(original_error or 'unknown error')[:300]}); result served by the keyless free tier. "
                f"The next call will use '{provider_name}' again."
            ),
        )
        return rescued
    # Ring also failed: the ORIGINAL error names the user's setup, so lead with it.
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
    if not isinstance(result, dict):
        return False
    if result.get("blocked_by_policy"):
        return True
    return "blocked by website policy" in str(result.get("error") or "").lower()


def _map_extract_rows_by_url(
    urls: list, indices: list[int], rows: object
) -> dict[int, dict]:
    """Map well-formed extract rows to requested positions by exact URL.

    Provider batches are untrusted at this boundary: they may be short,
    reordered, or malformed. Positional merging can therefore attach content to
    the wrong URL. Missing and malformed rows are ignored so callers can retain
    the corresponding original failure.
    """
    if not isinstance(rows, list):
        return {}

    pending: dict[str, list[int]] = {}
    for index in indices:
        if index >= len(urls) or not isinstance(urls[index], str):
            continue
        pending.setdefault(urls[index], []).append(index)

    mapped: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_url = row.get("url")
        if not isinstance(row_url, str):
            continue
        candidates = pending.get(row_url)
        if not candidates:
            continue
        mapped[candidates.pop(0)] = row
    return mapped


def _ordered_extract_batch(urls: list, results: object) -> list | None:
    """A complete, exact-URL primary batch, or None (never assume positional parity)."""
    if not isinstance(results, list) or len(results) != len(urls):
        return None
    mapped = _map_extract_rows_by_url(urls, list(range(len(urls))), results)
    return [mapped[i] for i in range(len(urls))] if len(mapped) == len(urls) else None


def _retry_extract_indices(urls: list, results: object) -> tuple[list | None, list[int]]:
    """Partition a trustworthy batch; an incomplete policy batch is fail-closed."""
    ordered = _ordered_extract_batch(urls, results)
    if ordered is not None:
        return ordered, [i for i, row in enumerate(ordered) if row.get("error") and not _policy_blocked_result(row)]
    rows = results if isinstance(results, list) else []
    if any(_policy_blocked_result(row) for row in rows):
        return None, []
    return None, list(range(len(urls)))


def _extract_failure_message(results: object) -> str:
    rows = results if isinstance(results, list) else []
    return next((str(r["error"]) for r in rows if isinstance(r, dict) and r.get("error")), "extract failed")


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

    ordered, rescue_idx = _retry_extract_indices(urls, results)
    if not rescue_idx:
        return ordered if ordered is not None else results
    rescue_urls = [urls[i] for i in rescue_idx]
    original_error = _extract_failure_message(results)
    logger.warning(
        "web_extract backend '%s' failed all %d URL(s) (%s); one-shot keyless rescue",
        provider_name, len(rescue_urls), (original_error or "")[:200],
    )
    rescued = extract_with_failover(provider_name, list(rescue_urls))
    mapped = _map_extract_rows_by_url(urls, rescue_idx, rescued)
    replacements = {
        index: result
        for index, result in mapped.items()
        if _policy_blocked_result(result) or not result.get("error")
    }
    if not replacements:
        return results

    for result in replacements.values():
        if not result.get("error"):
            meta = result.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta["rescued_from"] = provider_name
                meta["backend_error"] = (original_error or "")[:300]

    if ordered is not None:
        # Preserve policy rows, ordering, and ordinary failures. Replace only
        # exact-URL rows that rescue successfully or add a new policy refusal.
        merged = list(ordered)
        for index, result in replacements.items():
            merged[index] = result
        return merged

    # A malformed original batch has no trustworthy positional identity. Only
    # return rescued rows when every input URL was mapped exactly; otherwise
    # preserve the original response rather than collapsing or reordering it.
    if len(mapped) == len(urls):
        return [mapped[index] for index in range(len(urls))]
    return results


def _get_fallback_backend(capability: str) -> str:
    """Return the configured request-level fallback for *capability*.

    A capability-specific value wins over the shared fallback. Unlike
    ``web.backend`` (a selection default), this backend is attempted only
    after the selected primary provider fails a request.
    """
    from tools.web_tools import _load_web_config
    cfg = _load_web_config()
    return (
        (cfg.get(f"{capability}_fallback_backend") or cfg.get("fallback_backend") or "")
        .lower()
        .strip()
    )


def _get_secondary_provider(primary_name: str, capability: str):
    """Resolve a configured, distinct provider that supports *capability*."""
    name = _get_fallback_backend(capability)
    if not name or name == primary_name:
        return None
    try:
        from agent.web_search_registry import get_provider

        provider = get_provider(name)
        supports = getattr(provider, f"supports_{capability}", None)
        if provider is None or not callable(supports) or not supports():
            logger.warning(
                "web.%s_fallback_backend '%s' is unavailable or unsupported",
                capability,
                name,
            )
            return None
        try:
            from plugins.web.keyless_mcp import provider_tier

            if provider_tier(name) == "free":
                logger.warning(
                    "Configured web.%s fallback '%s' is pinned to the keyless tier; skipping it",
                    capability,
                    name,
                )
                return None
            if not provider.is_available():
                logger.warning(
                    "Configured web.%s fallback '%s' is unavailable in its configured mode; skipping it",
                    capability,
                    name,
                )
                return None
        except Exception as exc:  # noqa: BLE001 — unavailable secondary is non-fatal
            logger.warning(
                "Configured web.%s fallback '%s' availability check failed: %s",
                capability,
                name,
                exc,
            )
            return None
        return provider
    except Exception as exc:  # noqa: BLE001 — fallback is best-effort
        logger.warning("web %s fallback '%s' could not load: %s", capability, name, exc)
        return None


def _try_fallback_search(
    primary_name: str, original_error: str, query: str, limit: int
) -> tuple[dict | None, str]:
    """Try the configured secondary search provider once.

    Returns ``(successful_result, fallback_error)``. A missing or invalid
    secondary is represented by ``(None, "")`` so the existing keyless rescue
    can still run unchanged.
    """
    provider = _get_secondary_provider(primary_name, "search")
    if provider is None:
        return None, ""
    logger.warning(
        "web_search backend '%s' failed (%s); trying configured fallback '%s'",
        primary_name,
        (original_error or "")[:200],
        provider.name,
    )
    try:
        result = provider.search(query, limit)
    except Exception as exc:  # noqa: BLE001 — continue to keyless rescue
        return None, str(exc)
    if not isinstance(result, dict):
        return None, "search returned an invalid response"
    if not result.get("success"):
        return None, str(result.get("error", "search failed"))
    data = result.setdefault("data", {})
    if not isinstance(data, dict):
        return None, "search returned invalid data"
    data["served_by"] = provider.name
    data["fallback_from"] = primary_name
    data["backend_error"] = (original_error or "unknown error")[:300]
    return result, ""


async def _try_fallback_extract(
    primary_name: str, urls: list, results: list, *, format: str | None = None
) -> tuple[list | None, str]:
    """Try the configured secondary extract provider for genuine failures.

    Policy-blocked URLs are never sent to another provider. A secondary that
    fails the whole retry returns ``None`` so the existing keyless rescue can
    still run.
    """
    provider = _get_secondary_provider(primary_name, "extract")
    if provider is None:
        return None, ""

    ordered, retry_indices = _retry_extract_indices(urls, results)
    if not retry_indices:
        return None, ""
    retry_urls = [urls[i] for i in retry_indices]
    original_error = _extract_failure_message(results)
    logger.warning(
        "web_extract backend '%s' failed (%s); trying configured fallback '%s'",
        primary_name,
        (original_error or "")[:200],
        provider.name,
    )

    import inspect

    try:
        if inspect.iscoroutinefunction(provider.extract):
            retried = await provider.extract(retry_urls, format=format)
        else:
            retried = await asyncio.to_thread(
                provider.extract, retry_urls, format=format
            )
    except Exception as exc:  # noqa: BLE001 — continue to keyless rescue
        return None, str(exc)

    mapped = _map_extract_rows_by_url(urls, retry_indices, retried)
    for result in mapped.values():
        if not result.get("error"):
            metadata = result.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["served_by"] = provider.name
                metadata["fallback_from"] = primary_name
                metadata["backend_error"] = (original_error or "unknown error")[:300]

    if ordered is not None:
        # Preserve the complete original batch and replace only rows that can be
        # associated with an exact requested URL. This keeps policy signals,
        # ordering, and unreturned failures intact for later rescue partitioning.
        merged = list(ordered)
        for index, result in mapped.items():
            if _policy_blocked_result(result) or not result.get("error"):
                merged[index] = result
        if any(_policy_blocked_result(result) for result in mapped.values()):
            return merged, ""
        if any(not result.get("error") for result in mapped.values()):
            return merged, ""
    else:
        # Without a complete original batch, accept a fallback only when it
        # supplies one safely mapped row for every requested URL.
        if len(mapped) == len(urls):
            ordered = [mapped[index] for index in range(len(urls))]
            if any(_policy_blocked_result(result) for result in ordered):
                return ordered, ""
            if any(not result.get("error") for result in ordered):
                return ordered, ""

    fallback_error = next(
        (
            result.get("error")
            for result in mapped.values()
            if result.get("error")
        ),
        "extract returned no valid results",
    )
    return None, str(fallback_error)


def _fallback_extract_needs_rescue(urls: list, results: list) -> bool:
    """True when every non-policy fallback row is still a genuine failure.

    A secondary can discover a redirect-time policy block for one URL while
    failing another URL for an ordinary provider reason. Preserve the blocked
    row, but allow keyless rescue for the remaining failures. Partial fallback
    success remains terminal and is never expanded into per-page rescue.
    """
    if len(results) != len(urls):
        return False
    rescueable = [result for result in results if not _policy_blocked_result(result)]
    return bool(rescueable) and all(result.get("error") for result in rescueable)

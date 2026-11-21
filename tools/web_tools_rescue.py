"""One-shot keyless-ring rescue for failed keyed/configured web calls.

Stateless by design: a rescue routes THIS call through the free-tier ring (plugins/web/keyless_mcp.py);
the next web_search/web_extract call attempts the chosen backend again. Callers must never cache a
rescue-served response, or the one-shot rescue becomes sticky for a whole TTL. Logs under the origin
(tools.web_tools) logger.
"""

import logging
from typing import Any

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
    except Exception as exc:
        logger.debug("keyless rescue tier check failed: %s", exc)
        return False


def annotate_pre_keyless_rescue_refusal(backend_name: str, reason: str, result) -> None:
    """Surface a refused rescue (plugin block, denied or unanswered approval) so it is visible, not silent."""
    if result is None:
        return
    logger.warning("pre_keyless_rescue kept backend '%s' failure off the keyless ring: %s",
                   backend_name, (reason or "")[:200])
    note = f" [keyless cloud rescue not used: {reason}]"
    if isinstance(result, dict) and result.get("error"):
        result["error"] = f"{result['error']}{note}"
    elif isinstance(result, list):
        for entry in result:
            if isinstance(entry, dict) and entry.get("error"):
                entry["error"] = f"{entry['error']}{note}"


def _pre_keyless_rescue_refusal(*, kind: str, backend_name: str, error: str, query: str | None = None,
                                urls: list[str] | None = None) -> str | None:
    """Fire ``pre_keyless_rescue``; return None to let the rescue run, else why it does not.

    Same directives and the same human gate as ``pre_tool_call``: ``{"action": "block", "message"}``
    vetoes, ``{"action": "approve", "message", "rule_key"?}`` asks the user through
    ``request_tool_approval`` (deny, timeout, or nobody to answer keep the data home), and
    block > approve > none. ``None`` and irrelevant returns are no opinion; a bare ``False`` is a veto.
    A callback that raises or times out is a veto (fail closed, like ``pre_tool_call``): the call
    then keeps its own error, which is what it returns without the rescue anyway.
    """
    try:
        from hermes_cli.lifecycle import invoke_hook
        from hermes_cli.plugins import _PreToolCallDirective, _resolve_block_from_details
    except ImportError:
        return None
    payload: dict[str, Any] = {"kind": kind, "backend_name": backend_name, "error": (error or "")[:500]}
    if query is not None:
        payload["query"] = query
    if urls is not None:
        payload["urls"] = list(urls)
    try:
        results = invoke_hook("pre_keyless_rescue", **payload)
    except Exception as exc:
        logger.warning("pre_keyless_rescue hook dispatch failed; keeping the failure local", exc_info=True)
        return f"pre_keyless_rescue dispatch failed: {type(exc).__name__}"
    if any(result is False for result in results):
        return "vetoed by a pre_keyless_rescue plugin"
    approve = None
    for result in results:
        if not isinstance(result, dict):
            continue
        action, message = result.get("action"), result.get("message")
        message = message if isinstance(message, str) and message else None
        if action == "block" and message:
            return message
        if action == "approve" and approve is None:
            rule_key = result.get("rule_key")
            approve = _PreToolCallDirective(action="approve", message=message,
                                            rule_key=rule_key.strip() or None if isinstance(rule_key, str) else None)
    if approve is None:
        return None
    return _resolve_block_from_details(approve, "web_search" if kind == "search" else "web_extract")


def _ring_vendor_keyless(name: str) -> bool:
    """Did *name*'s own provider route this call through the anonymous keyless ring?

    Mirrors the predicate each ring provider evaluates before calling, so eligibility reflects what
    actually happened. Firecrawl owns extra routes that bypass the ring without a key — the managed
    Nous Tool Gateway (persisted ``nous`` selection, or the legacy never-configured fallback when the
    gateway is ready) and a self-hosted ``FIRECRAWL_API_URL`` — so it is asked directly.
    """
    if name == "firecrawl":
        from plugins.web.firecrawl.provider import _use_keyless_ring
        return _use_keyless_ring()
    from agent.web_search_provider import get_provider_env
    from plugins.web.keyless_mcp import use_keyless
    key_var = _RING_KEY_VARS.get(name, "")
    return use_keyless(name, get_provider_env(key_var) if key_var else "")


def _managed_search_fallback(provider, original_error: str, query: str, limit: int):
    """Try managed Firecrawl for this call only; None leaves the original error for keyless rescue.
    Managed Firecrawl is billed, so a caller on free fast search alone never reaches it."""
    from agent.web_search_provider import get_provider_env
    from tools import web_tools as _wt
    if (getattr(provider, "name", "") != "perplexity"
            or get_provider_env("PERPLEXITY_API_KEY") or not _wt._managed_web_search() or not _wt._is_tool_gateway_ready()):
        return None
    logger.warning("web_search managed Perplexity failed (%s); serving this call from managed Firecrawl", (original_error or "")[:200])
    try:
        from agent.web_search_registry import get_provider
        resp = get_provider("firecrawl").search(query, limit)
    except Exception as exc:
        resp = {"success": False, "error": str(exc)}
    if not resp.get("success"):
        logger.warning("managed Firecrawl fallback failed too: %s", str(resp.get("error", ""))[:200])
        return None
    resp.setdefault("data", {}).update(
        fallback_from="managed_primary",
        backend_error=f"Primary managed search failed this call ({(original_error or 'unknown error')[:300]}); "
                      "result served by the managed fallback. The next call will use the primary again.",
    )
    return resp


def _rescue_eligible(provider) -> bool:
    """True when a failed call on *provider* should get a one-shot rescue.

    Eligible: any call that did NOT go through the keyless ring — a non-ring backend, a ring vendor
    in keyed mode, or a ring vendor routed through the managed gateway / a self-hosted instance. A
    ring vendor that walked the ring is NOT eligible: its failure means the ring already failed.
    """
    if not _keyless_rescue_enabled() or provider is None:
        return False
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING
        name = getattr(provider, "name", "")
        return name not in _KEYLESS_RING or not _ring_vendor_keyless(name)
    except Exception as exc:
        logger.debug("rescue eligibility check failed: %s", exc)
        return False


def _rescue_search(provider_name: str, original_error: str, query: str, limit: int) -> dict:
    """Rescue a failed search via the ring; annotate the result with the original failure."""
    from plugins.web.keyless_mcp import search_with_failover
    refusal = _pre_keyless_rescue_refusal(
        kind="search", backend_name=provider_name, error=original_error, query=query,
    )
    if refusal:
        failed = {"success": False, "error": original_error or "search failed"}
        annotate_pre_keyless_rescue_refusal(provider_name, refusal, failed)
        return failed
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
    """True for a website-policy refusal — intentional, never rescued (it would fetch blocked content)."""
    error = str(result.get("error") or "").lower()
    return bool(result.get("blocked_by_policy")) or "blocked by website policy" in error


def _rescue_extract(provider_name: str, urls: list, results: list) -> list:
    """Rescue a whole-batch extract failure via the ring.

    Only genuine failures are re-fetched; policy-blocked entries are preserved verbatim. If the provider
    broke url/result order parity, every entry is treated as rescueable and the ring's list replaces the
    batch wholesale.
    """
    from plugins.web.keyless_mcp import extract_with_failover

    parity = len(results) == len(urls)
    rescue_idx = [i for i, r in enumerate(results) if not parity or not _policy_blocked_result(r)]
    if not rescue_idx:
        return results  # every failure is an intentional policy block

    rescue_urls = [urls[i] for i in rescue_idx] if parity else list(urls)
    errors = (results[i].get("error") for i in rescue_idx if results[i].get("error"))
    original_error = next(errors, "extract failed")
    refusal = _pre_keyless_rescue_refusal(
        kind="extract", backend_name=provider_name, error=original_error, urls=rescue_urls,
    )
    if refusal:
        annotate_pre_keyless_rescue_refusal(provider_name, refusal, results)
        return results
    logger.warning(
        "web_extract backend '%s' failed all %d URL(s) (%s); one-shot keyless rescue",
        provider_name, len(rescue_urls), (original_error or "")[:200],
    )
    rescued = extract_with_failover(provider_name, list(rescue_urls))
    if rescued and all(r.get("error", "") for r in rescued):
        return results  # rescue also failed everywhere: keep original errors
    for r in rescued:
        meta = None if r.get("error") else r.setdefault("metadata", {})
        if isinstance(meta, dict):
            meta["rescued_from"] = provider_name
            meta["backend_error"] = (original_error or "")[:300]
    if parity and len(rescued) == len(rescue_idx):
        replacements = dict(zip(rescue_idx, rescued))
        return [replacements.get(i, r) for i, r in enumerate(results)]
    return rescued

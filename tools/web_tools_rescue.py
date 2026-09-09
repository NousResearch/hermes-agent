"""One-shot keyless-ring rescue for failed keyed/configured web calls.

Stateless by design: a rescue routes THIS call through the free-tier ring (plugins/web/keyless_mcp.py);
the next web_search/web_extract call attempts the chosen backend again. Callers must never cache a
rescue-served response, or the one-shot rescue becomes sticky for a whole TTL. Logs under the origin
(tools.web_tools) logger.
"""

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


# ─── One-shot keyed backstop (keyless ring exhausted) ────────────────────────
#
# Exact mirror of the rescue above, in the opposite direction: a rescue sends a
# failed KEYED call to the free ring; a backstop sends an exhausted KEYLESS call
# to the keyed path. Complementary by construction — _rescue_eligible() is true
# iff the call ran keyed, _backstop_eligible() iff it ran keyless and the whole
# ring was throttled — so no single failure can trigger both.


def _keyed_backstop_enabled() -> bool:
    """``web.keyed_backstop`` (default OFF), implicitly off when the keyless tier is disabled.

    Opt-in rather than opt-out because this is the one path in the web stack that can spend an API
    key the user did not explicitly route through, so enabling it is the operator's decision.
    """
    from tools.web_tools import _load_web_config
    if not _load_web_config().get("keyed_backstop", False):
        return False
    try:
        from agent.web_search_registry import _keyless_tier_enabled
        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — backstop is best-effort
        logger.debug("keyed backstop tier check failed: %s", exc)
        return False


def _backstop_key_for(provider_name: str) -> str:
    """The vendor API key for *provider_name*, or ``""``.

    Only ring vendors have a keyed counterpart worth falling back to; a non-ring backend never rode
    the ring in the first place. ``tavily`` is opt-in keyless via `hermes tools` rather than a ring
    member, so it is keyed here but absent from _RING_KEY_VARS.
    """
    try:
        from plugins.web.keyless_mcp import _KEYLESS_RING
        if provider_name not in _KEYLESS_RING:
            return ""
        key_var = {**_RING_KEY_VARS, "tavily": "TAVILY_API_KEY"}.get(provider_name, "")
        if not key_var:
            return ""
        from agent.web_search_provider import get_provider_env
        return get_provider_env(key_var) or ""
    except Exception as exc:  # noqa: BLE001 — backstop is best-effort
        logger.debug("backstop key lookup failed for %r: %s", provider_name, exc)
        return ""


def _backstop_eligible(provider, error: str, *, exhausted: bool | None = None) -> bool:
    """True when an exhausted keyless call should retry on the keyed path.

    All must hold: the backstop is enabled (and the keyless tier with it); the free tier is
    genuinely exhausted, not merely erroring once — read from *error* via the search ring's marker,
    or passed as *exhausted* by callers (extract) whose ring reports throttling per-URL instead of
    in one string; and the vendor is a ring member with a real API key to retry with.

    Reachability — why a ``free`` pin is the TARGET case, not an exclusion: the ring only runs when
    use_keyless() is true, and with a key on file that happens exactly under a ``free`` pin (tier
    ``auto`` with a key routes keyed, so no ring, so no exhaustion). A ``free`` pin therefore means
    "prefer the free endpoint", and the backstop makes that preference survive a dry free tier.
    That single reachable population is also why the feature ships opt-in.
    """
    if provider is None or not _keyed_backstop_enabled():
        return False
    try:
        from plugins.web.keyless_mcp import ring_exhausted
        if exhausted is None:
            exhausted = ring_exhausted(error)
        if not exhausted:
            return False
        return bool(_backstop_key_for(getattr(provider, "name", "")))
    except Exception as exc:  # noqa: BLE001 — backstop is best-effort
        logger.debug("backstop eligibility check failed: %s", exc)
        return False


def _backstop_search(provider, original_error: str, query: str, limit: int) -> dict:
    """One-shot keyed retry after the keyless ring came back exhausted.

    Stateless, exactly like the rescue: this call alone spends the API key; the next starts on the
    free ring again. Re-runs the provider's own ``search`` under force_keyed() so the vendor SDK
    logic is reused rather than duplicated here.
    """
    name = getattr(provider, "name", "")
    logger.warning(
        "keyless ring exhausted for '%s' (%s); one-shot keyed backstop",
        name, (original_error or "")[:200],
    )
    try:
        from plugins.web.keyless_mcp import force_keyed
        with force_keyed(name):
            keyed = provider.search(query, limit)
    except Exception as exc:  # noqa: BLE001 — backstop must never mask the ring
        logger.warning("keyed backstop for '%s' raised: %s", name, exc)
        return {
            "success": False,
            "error": f"{original_error or 'search failed'} (keyed backstop also failed: {exc})",
        }
    if keyed.get("success"):
        keyed.setdefault("data", {}).update(
            backstopped_from="keyless",
            backend_error=(
                "The keyless free tier was exhausted for this call "
                f"({(original_error or 'all vendors throttled')[:300]}); result served by keyed "
                f"'{name}'. The next call will try the free ring again."
            ),
        )
        return keyed
    return {
        "success": False,
        "error": (
            f"{original_error or 'search failed'} "
            f"(keyed backstop also failed: {keyed.get('error', 'unknown')})"
        ),
    }


def _backstop_extract(provider, urls: list, results: list) -> list:
    """One-shot keyed retry for an extract batch the ring could not serve."""
    name = getattr(provider, "name", "")
    logger.warning(
        "keyless ring exhausted for '%s' extract (%d url(s)); keyed backstop", name, len(urls),
    )
    try:
        from plugins.web.keyless_mcp import force_keyed
        with force_keyed(name):
            keyed = provider.extract(list(urls))
    except Exception as exc:  # noqa: BLE001 — keep the ring's results
        logger.warning("keyed extract backstop for '%s' raised: %s", name, exc)
        return results
    # Only prefer the keyed answer when it actually improved on the ring.
    if any(not r.get("error") for r in (keyed or [])):
        return keyed
    return results

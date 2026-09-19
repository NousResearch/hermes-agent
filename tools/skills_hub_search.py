"""Skills Hub discovery: the centralized Hermes index fetch (cached, stale-
fallback), the source router, and parallel/unified search across source
adapters.

Split out of ``tools/skills_hub.py``; hub state (cache dir, ``TapsManager``, JSON
cache reads) is still read from there at call time.
"""

from __future__ import annotations

import logging
import httpx
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tools.skills_hub_clawhub import ClawHubSource
from tools.skills_hub_github import GitHubAuth, GitHubSource, _filter_results_by_provider, _provider_filter_of
from tools.skills_hub_models import SkillMeta, SkillSource, TRUST_RANK, _dedupe_by_trust
from tools.skills_hub_official import HermesIndexSource, OptionalSkillSource
from tools.skills_hub_skillssh import SkillsShSource
from tools.skills_hub_sources import BrowseShSource, LobeHubSource, UrlSource, WellKnownSkillSource

# Log-record parity with the origin module.
logger = logging.getLogger("tools.skills_hub")

HERMES_INDEX_URL = "https://hermes-agent.nousresearch.com/docs/api/skills-index.json"
HERMES_INDEX_TTL = 6 * 3600  # 6 hours


def _hermes_index_cache_file() -> Path:
    from tools.skills_hub import _index_cache_dir
    return _index_cache_dir() / "hermes-index.json"


def _load_hermes_index() -> Optional[dict]:
    """Fetch the centralized skills index (docs site, rebuilt daily), cached
    locally for HERMES_INDEX_TTL; on any failure serve the stale cache.

    Brotli is deliberately NOT negotiated: the index is tens of MB and httpx's
    streaming Brotli decoder (brotlicffi, pinned for Discord attachments) raises
    DecodingError on payloads this size — which surfaced as a silently empty
    Skills Hub. gzip/deflate first; the identity retry covers proxies that
    ignore the header and return Brotli anyway.
    """
    from tools.skills_hub import _read_json_if_fresh
    cache_file = _hermes_index_cache_file()
    cached = _read_json_if_fresh(cache_file, HERMES_INDEX_TTL)
    if cached is not None:
        return cached
    data = None
    for accept_encoding in ("gzip, deflate", "identity"):
        try:
            resp = httpx.get(HERMES_INDEX_URL, timeout=15, follow_redirects=True,
                             headers={"Accept-Encoding": accept_encoding})
            if resp.status_code != 200:
                logger.debug("Hermes index fetch returned %d", resp.status_code)
                return _load_stale_index_cache()
            data = resp.json()
            break
        except httpx.DecodingError as e:
            logger.debug("Hermes index decode failed (Accept-Encoding=%s): %s", accept_encoding, e)
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            logger.debug("Hermes index fetch failed: %s", e)
            return _load_stale_index_cache()
    if not isinstance(data, dict) or "skills" not in data:
        return _load_stale_index_cache()
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass
    return data


def _load_stale_index_cache() -> Optional[dict]:
    """Fall back to the cache regardless of age when the network fetch fails."""
    from tools.skills_hub import _read_json_if_fresh
    return _read_json_if_fresh(_hermes_index_cache_file(), float("inf"))


# External API sources the centralized index already covers; skipped when the
# index is available and no source filter is active (~70 GitHub calls/search
# for unauthenticated users otherwise).
_API_SOURCE_IDS = frozenset({"github", "skills-sh", "clawhub", "lobehub", "well-known"})
# Consulted only when the index answered a non-empty query with nothing: the
# index is rebuilt asynchronously and lags the registries, so a skill that is
# live on skills.sh may not be in it yet. GitHub stays out — one miss (a typo)
# would burn an unauthenticated user's whole hourly GitHub budget.
_INDEX_MISS_FALLBACK_IDS = _API_SOURCE_IDS - {"github"}
# Cap on the fallback pass. ClawHub can take minutes; without its own budget
# every unfiltered miss (a typo) would stall CLI/TUI/dashboard for the callers'
# full 30 s ``overall_timeout`` where the index alone answered instantly.
_INDEX_MISS_FALLBACK_BUDGET = 8.0


def create_source_router(auth: Optional[GitHubAuth] = None) -> List[SkillSource]:
    """All configured source adapters, in priority order."""
    from tools.skills_hub import TapsManager
    if auth is None:
        auth = GitHubAuth()
    return [
        OptionalSkillSource(auth=auth),   # official optional skills (highest priority)
        HermesIndexSource(auth=auth),     # centralized index (search + resolved install paths)
        SkillsShSource(auth=auth),
        WellKnownSkillSource(),
        UrlSource(),                      # direct HTTP(S) URL to a SKILL.md
        GitHubSource(auth=auth, extra_taps=TapsManager().list_taps()),
        ClawHubSource(),
        LobeHubSource(),
        BrowseShSource(),                 # browse.sh site-specific browser skills
    ]


def _search_one_source(
    src: SkillSource, query: str, limit: int, provider_filter: str = "",
) -> Tuple[str, List[SkillMeta]]:
    """Search a single source.  Runs in a thread for parallelism."""
    try:
        # These sources mix providers in one catalog. Narrow before their top-N
        # cut so another provider cannot crowd every requested match out.
        if provider_filter and isinstance(src, (HermesIndexSource, GitHubSource)):
            return src.source_id(), src.search(query, limit=limit, provider_filter=provider_filter)
        return src.source_id(), src.search(query, limit=limit)
    except Exception as e:
        logger.debug("Search failed for %s: %s", src.source_id(), e)
        return src.source_id(), []


def _select_active_sources(sources: List[SkillSource], source_filter: str) -> List[SkillSource]:
    """Sources to query for ``source_filter``.

    A provider filter (nvidia/openai/...) is not a source id — the data lives
    in the index/github source under ``extra.provider`` — so it selects like
    "all". Mixed-provider sources narrow before their top-N cut; the walker
    cuts every source's results. "official" is always queried alongside an
    explicit source filter.
    """
    effective = "all" if _provider_filter_of(source_filter) else source_filter
    index_available = effective == "all" and any(
        src.source_id() == "hermes-index" and getattr(src, "is_available", False) for src in sources
    )
    active: List[SkillSource] = []
    for src in sources:
        sid = src.source_id()
        if effective != "all" and sid != effective and sid != "official":
            continue
        if index_available and sid in _API_SOURCE_IDS:
            continue
        active.append(src)
    return active


def _index_miss_fallback_sources(
    sources: List[SkillSource], active: List[SkillSource], query: str, source_counts: Dict[str, int],
    provider_filter: str = "",
) -> List[SkillSource]:
    """Registries to consult after the index stood in for them and found nothing.

    Empty for a browse (no query), when the index was not consulted (no skip
    happened), when it returned matches, or when it did not answer at all
    (timed out: no budget is left and the registries would only be blamed as
    late without being asked). Also empty under a provider filter
    (``--source nvidia``): the fallback registries carry no ``extra.provider``,
    so their results would all be cut and the calls would only burn budget.
    """
    if not query.strip() or provider_filter or not any(src.source_id() == "hermes-index" for src in active):
        return []
    if source_counts.get("hermes-index") != 0:
        return []
    return [src for src in sources if src.source_id() in _INDEX_MISS_FALLBACK_IDS and src not in active]


def _fan_out(
    active: List[SkillSource], query: str, per_source_limits: Dict[str, int], provider_filter: str,
    deadline: float, on_source_done: Optional[Any], all_results: List[SkillMeta],
    source_counts: Dict[str, int], timed_out_ids: List[str],
) -> None:
    """Query ``active`` in parallel until ``deadline`` (monotonic), merging into the accumulators."""
    from concurrent.futures import as_completed
    from tools.daemon_pool import DaemonThreadPoolExecutor

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        timed_out_ids.extend(src.source_id() for src in active)
        return
    # Not a ``with`` block: its shutdown(wait=True) would block on a slow source
    # (ClawHub) for minutes and defeat ``overall_timeout``. Daemon workers so an
    # abandoned source cannot block interpreter exit either.
    pool = DaemonThreadPoolExecutor(max_workers=min(len(active), 8))
    futures = {
        pool.submit(
            _search_one_source, src, query, per_source_limits.get(src.source_id(), 50), provider_filter,
        ): src.source_id()
        for src in active
    }
    try:
        for fut in as_completed(futures, timeout=remaining):
            try:
                sid, results = fut.result(timeout=0)
                if provider_filter:
                    # One owner for the merged provider cut: sources that cannot
                    # filter per-tap (official, url, ...) are narrowed here, so
                    # every caller (CLI, TUI gateway, dashboard) sees one rule.
                    results = _filter_results_by_provider(results, provider_filter)
                source_counts[sid] = len(results)
                all_results.extend(results)
                if on_source_done:
                    on_source_done(sid, len(results))
            except Exception:
                pass
    except TimeoutError:
        late = [futures[f] for f in futures if not f.done()]
        timed_out_ids.extend(late)
        if late:
            logger.debug("Skills browse timed out waiting for: %s", ", ".join(late))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def parallel_search_sources(
    sources: List[SkillSource], query: str = "", per_source_limits: Optional[Dict[str, int]] = None,
    source_filter: str = "all", overall_timeout: float = 30, on_source_done: Optional[Any] = None,
) -> Tuple[List[SkillMeta], Dict[str, int], List[str]]:
    """Search all sources in parallel with an overall timeout.

    Returns ``(all_results, source_counts, timed_out_ids)``. *on_source_done*
    is an optional ``(source_id, count) -> None`` progress callback. Under a
    provider filter every source's results are narrowed before they are
    counted and merged, so callers need no provider logic of their own.

    When the centralized index stood in for the external registries and found
    nothing for a non-empty query, those registries are queried within the
    same ``overall_timeout`` — capped at ``_INDEX_MISS_FALLBACK_BUDGET`` so a
    slow registry cannot turn an instant index miss into a 30 s stall — so
    every caller (CLI, TUI gateway, dashboard) still finds skills the index
    has not picked up yet.
    """
    per_source_limits = per_source_limits or {}
    active = _select_active_sources(sources, source_filter)
    provider_filter = _provider_filter_of(source_filter)
    all_results: List[SkillMeta] = []
    source_counts: Dict[str, int] = {}
    timed_out_ids: List[str] = []
    if not active:
        return all_results, source_counts, timed_out_ids

    deadline = time.monotonic() + overall_timeout
    _fan_out(active, query, per_source_limits, provider_filter, deadline, on_source_done,
             all_results, source_counts, timed_out_ids)
    fallback = _index_miss_fallback_sources(sources, active, query, source_counts, provider_filter)
    if fallback:
        fallback_deadline = min(deadline, time.monotonic() + _INDEX_MISS_FALLBACK_BUDGET)
        _fan_out(fallback, query, per_source_limits, provider_filter, fallback_deadline, on_source_done,
                 all_results, source_counts, timed_out_ids)
    return all_results, source_counts, timed_out_ids


def _rerank_setting() -> dict:
    """``skills.hub_relevance_rerank`` mapping from config.yaml ({} when absent/malformed)."""
    try:
        from agent.skill_utils import _load_raw_config
        section = _load_raw_config().get("skills", {}).get("hub_relevance_rerank", {})
    except Exception:
        return {}
    return section if isinstance(section, dict) else {}


def unified_search(query: str, sources: List[SkillSource],
                   source_filter: str = "all", limit: int = 10,
                   relevance_scorer: Optional[Any] = None) -> List[SkillMeta]:
    """Search all sources (in parallel) and merge results.

    When ``skills.hub_relevance_rerank.enabled`` is true in config.yaml, the
    trust-sorted merge is re-ordered by query relevance (Jev Score; opt-in,
    disabled by default) before the limit cut. ``relevance_scorer`` injects
    the ``(query, candidates) -> {identifier: score}`` callable (tests/CI use
    a stub so no network is needed); any scorer failure fails open to trust
    order.
    """
    all_results, _, _ = parallel_search_sources(sources, query=query, source_filter=source_filter, overall_timeout=30)
    deduped = _dedupe_by_trust(all_results)
    # Stable-sort by trust before truncating so the limit cut never drops a
    # builtin/official entry because a high-volume community source finished
    # first; insertion order is preserved within each rank.
    deduped.sort(key=lambda r: -TRUST_RANK.get(r.trust_level, 0))
    if relevance_scorer is not None or _rerank_setting().get("enabled"):
        from tools.skills_hub_rerank import rerank_by_relevance
        deduped = rerank_by_relevance(query, deduped, scorer=relevance_scorer)
    return deduped[:limit]

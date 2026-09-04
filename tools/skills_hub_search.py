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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tools.skills_hub_clawhub import ClawHubSource
from tools.skills_hub_github import GitHubAuth, GitHubSource, _PROVIDER_FILTER_VALUES, _filter_results_by_provider
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


def _search_one_source(src: SkillSource, query: str, limit: int) -> Tuple[str, List[SkillMeta]]:
    """Search a single source.  Runs in a thread for parallelism."""
    try:
        return src.source_id(), src.search(query, limit=limit)
    except Exception as e:
        logger.debug("Search failed for %s: %s", src.source_id(), e)
        return src.source_id(), []


def _select_active_sources(sources: List[SkillSource], source_filter: str) -> List[SkillSource]:
    """Sources to query for ``source_filter``.

    A provider filter (nvidia/openai/...) is not a source id — the data lives
    in the index/github source under ``extra.provider`` — so it selects like
    "all"; the narrowing happens later on the merged results. "official" is
    always included alongside an explicit source filter.
    """
    effective = "all" if source_filter.strip().lower() in _PROVIDER_FILTER_VALUES else source_filter
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


def parallel_search_sources(
    sources: List[SkillSource], query: str = "", per_source_limits: Optional[Dict[str, int]] = None,
    source_filter: str = "all", overall_timeout: float = 30, on_source_done: Optional[Any] = None,
) -> Tuple[List[SkillMeta], Dict[str, int], List[str]]:
    """Search all sources in parallel with an overall timeout.

    Returns ``(all_results, source_counts, timed_out_ids)``. *on_source_done*
    is an optional ``(source_id, count) -> None`` progress callback.
    """
    from concurrent.futures import as_completed

    per_source_limits = per_source_limits or {}
    active = _select_active_sources(sources, source_filter)
    all_results: List[SkillMeta] = []
    source_counts: Dict[str, int] = {}
    timed_out_ids: List[str] = []
    if not active:
        return all_results, source_counts, timed_out_ids

    # Not a ``with`` block: its shutdown(wait=True) would block on a slow source
    # (ClawHub) for minutes and defeat ``overall_timeout``. Daemon workers so an
    # abandoned source cannot block interpreter exit either.
    from tools.daemon_pool import DaemonThreadPoolExecutor
    pool = DaemonThreadPoolExecutor(max_workers=min(len(active), 8))
    futures = {
        pool.submit(_search_one_source, src, query, per_source_limits.get(src.source_id(), 50)): src.source_id()
        for src in active
    }
    try:
        for fut in as_completed(futures, timeout=overall_timeout):
            try:
                sid, results = fut.result(timeout=0)
                source_counts[sid] = len(results)
                all_results.extend(results)
                if on_source_done:
                    on_source_done(sid, len(results))
            except Exception:
                pass
    except TimeoutError:
        timed_out_ids = [futures[f] for f in futures if not f.done()]
        if timed_out_ids:
            logger.debug("Skills browse timed out waiting for: %s", ", ".join(timed_out_ids))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return all_results, source_counts, timed_out_ids


def unified_search(query: str, sources: List[SkillSource],
                   source_filter: str = "all", limit: int = 10) -> List[SkillMeta]:
    """Search all sources (in parallel) and merge results."""
    all_results, _, _ = parallel_search_sources(sources, query=query, source_filter=source_filter, overall_timeout=30)
    # Provider filters target ``extra.provider`` on the merged set, not a source id.
    if source_filter.strip().lower() in _PROVIDER_FILTER_VALUES:
        all_results = _filter_results_by_provider(all_results, source_filter)
    deduped = _dedupe_by_trust(all_results)
    # Stable-sort by trust before truncating so the limit cut never drops a
    # builtin/official entry because a high-volume community source finished
    # first; insertion order is preserved within each rank.
    deduped.sort(key=lambda r: -TRUST_RANK.get(r.trust_level, 0))
    return deduped[:limit]

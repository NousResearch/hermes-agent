"""Suggest skills the user has not loaded yet based on usage patterns.

Builds a lightweight recommender on top of existing infrastructure:
* tools.skills_tool._find_all_skills - full catalog (name + description + category).
* tools.skill_usage.load_usage - per-skill use_count, view_count, last_used_at.
* tools.skill_usage.read_suppressed_names - skills the user has dismissed or pinned.

The recommender is opt-in (a CLI/agent call), deterministic (no model inference),
and cheap (single pass over the catalog + usage map). It does NOT touch any
persisted state - dismiss() writes through to the existing suppression registry
so the rest of the system stays aware.

Design constraints:
* Pure functions only - no class state, no async. Tests stay synchronous.
* Never raise into the agent loop - return [] on any internal failure.
* No new config knobs - all knobs are function kwargs with sane defaults.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Word-boundary regex for keyword tokenisation. Strips punctuation, lower-cases.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")

# Stopwords that should not count toward similarity. Small, English-only; expand
# later if the use cases demand multilingual awareness.
_STOPWORDS: frozenset = frozenset(
    {
        "a", "an", "and", "or", "the", "to", "of", "for", "with", "without",
        "in", "on", "at", "by", "from", "as", "is", "are", "was", "were",
        "be", "been", "being", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "you", "your", "yours",
        "i", "we", "our", "us", "me", "my", "mine",
        "use", "used", "uses", "using", "load", "loaded", "loads",
        "skill", "skills", "tool", "tools",
        "when", "if", "or", "but", "not", "no",
    }
)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp into a timezone-aware datetime. None on failure."""
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _tokenise(text: str) -> List[str]:
    """Lowercase token list with stopwords and very short tokens removed."""
    if not text:
        return []
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if len(tok) < 3:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _safe_catalog_loader() -> Optional[Callable[[], List[Dict[str, Any]]]]:
    """Return _find_all_skills if it can be imported, else None.

    The recommender must never raise into the agent loop; missing imports
    degrade to empty recommendations rather than blowing up a session.
    """
    try:
        from tools.skills_tool import _find_all_skills  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("skill_recommender: _find_all_skills unavailable: %s", e)
        return None
    return _find_all_skills


def _safe_usage_loader() -> Dict[str, Dict[str, Any]]:
    """Load the per-skill usage map from disk, returning empty on any failure.

    Never raises; missing/corrupt store is treated as no usage data.
    """
    try:
        from tools.skill_usage import load_usage  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("skill_recommender: load_usage unavailable: %s", e)
        return {}
    try:
        data = load_usage()
        return data if isinstance(data, dict) else {}
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("skill_recommender: load_usage failed: %s", e)
        return {}


def _safe_suppressed_loader() -> Set[str]:
    try:
        from tools.skill_usage import read_suppressed_names  # type: ignore

        names = read_suppressed_names()
        return set(names) if isinstance(names, (set, frozenset, list, tuple)) else set(names or ())
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("skill_recommender: read_suppressed_names failed: %s", e)
        return set()


def _describe_index(
    catalog: List[Dict[str, Any]],
    usage_loader: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Build {name: {description, category, tokens, used_recently, use_count}}.

    ``usage_loader`` is a 0-arg callable returning the per-skill usage map.
    Defaults to the on-disk store via ``_safe_usage_loader``. Tests pass a
    stub; production callers leave it None.
    """
    idx: Dict[str, Dict[str, Any]] = {}
    now = datetime.now(timezone.utc)
    usage = usage_loader() if usage_loader else _safe_usage_loader()
    for entry in catalog:
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        description = str(entry.get("description") or "")
        category = str(entry.get("category") or "")
        tokens = _tokenise(" ".join([description, category]))
        record = usage.get(name, {}) or {}
        last_used = _parse_iso(record.get("last_used_at"))
        days_since = (now - last_used).days if last_used else None
        idx[name] = {
            "description": description,
            "category": category,
            "tokens": tokens,
            "use_count": int(record.get("use_count") or 0),
            "view_count": int(record.get("view_count") or 0),
            "last_used_at": record.get("last_used_at"),
            "days_since_used": days_since,
        }
    return idx


def _safe_call(fn, *, default):
    """Invoke fn(), returning default on any exception. Never raises."""
    try:
        result = fn()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("skill_recommender: loader raised, using default: %s", e)
        return set(default) if not isinstance(default, (set, frozenset)) else set(default)
    if isinstance(result, (set, frozenset, list, tuple)):
        return set(result)
    return set(default) if not isinstance(default, (set, frozenset)) else set(default)


def _score_query(
    query_tokens: Iterable[str],
    catalog_index: Dict[str, Dict[str, Any]],
    suppressed: Set[str],
    *,
    history_weight: float,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Score every non-suppressed skill against the query tokens.

    Score = token-overlap Jaccard * 2 + recency_bonus. Returns a list sorted
    by descending score. Suppressed (dismissed or pinned) skills are skipped.
    """
    qt = list(query_tokens)
    if not qt:
        return []
    qt_set = set(qt)
    scored: List[Tuple[str, float, Dict[str, Any]]] = []
    for name, meta in catalog_index.items():
        if name in suppressed:
            continue
        tokens = meta.get("tokens") or []
        if not tokens:
            continue
        ts = set(tokens)
        inter = qt_set & ts
        if not inter:
            continue
        # Weighted score: count of matched query tokens (so a 3-token query
        # matching 2 tokens beats a 3-token query matching 1), normalised by
        # query length. This avoids the Jaccard ambiguity when two skills
        # match the same number of distinct tokens but one has more total
        # tokens in its description.
        match_score = len(inter) / len(qt_set)
        # Penalty for very long descriptions that pad with unrelated terms.
        length_penalty = 1.0 / (1.0 + max(len(ts) - len(inter), 0) / 10.0)
        score = match_score * length_penalty
        # Recency bonus: weight by use_count * decay(days_since_used)
        uc = meta.get("use_count") or 0
        if uc and meta.get("days_since_used") is not None:
            decay = 1.0 / (1.0 + max(meta["days_since_used"], 0) / 30.0)
            score += history_weight * min(uc / 10.0, 1.0) * decay
        scored.append((name, score, meta))
    scored.sort(key=lambda row: row[1], reverse=True)
    return scored


def recommend_skills(
    query: str,
    *,
    top_k: int = 5,
    history_weight: float = 0.3,
    catalog_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    suppressed_loader: Optional[Callable[[], Set[str]]] = None,
    usage_loader: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Return up to top_k skill recommendations matching *query*.

    Each result is {name, description, category, score, why}. Excludes skills
    already in the suppressed set (dismissed or pinned).

    Pure function; all heavy deps are loaded lazily and degrade to empty on
    import failure so a broken recommender never breaks an agent loop.
    """
    query_tokens = _tokenise(query or "")
    if not query_tokens:
        return []
    catalog = (catalog_loader or _safe_catalog_loader() or (lambda: []))()
    if not catalog:
        return []
    suppressed = _safe_call(suppressed_loader or _safe_suppressed_loader, default=set())
    index = _describe_index(catalog, usage_loader=usage_loader)
    scored = _score_query(query_tokens, index, suppressed, history_weight=history_weight)
    out: List[Dict[str, Any]] = []
    for name, score, meta in scored[: max(0, int(top_k))]:
        out.append(
            {
                "name": name,
                "description": meta.get("description", ""),
                "category": meta.get("category", ""),
                "score": round(float(score), 4),
                "why": _why_for(name, meta, query_tokens),
            }
        )
    return out


def _why_for(name: str, meta: Dict[str, Any], query_tokens: List[str]) -> str:
    """Build a short human-readable rationale string for the recommendation."""
    bits: List[str] = []
    matched = sorted(set(query_tokens) & set(meta.get("tokens") or []))
    if matched:
        bits.append("matches: " + ", ".join(matched[:5]))
    uc = meta.get("use_count") or 0
    if uc:
        bits.append(f"used {uc}x")
    days = meta.get("days_since_used")
    if days is not None:
        bits.append(f"last {days}d ago")
    return "; ".join(bits) or "matched by token overlap"


def recommend_for_recent_activity(
    *,
    top_k: int = 5,
    lookback_days: int = 7,
    min_use_count: int = 1,
    max_seed_skills: int = 5,
    catalog_loader: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    suppressed_loader: Optional[Callable[[], Set[str]]] = None,
    usage_loader: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Recommend skills based on what the user has used recently.

    Steps:
      1. Pick the most-recently-used skills (up to max_seed_skills).
      2. Build a synthetic query from their combined token bag.
      3. Run the same Jaccard recommender, excluding seeds.
      4. Suppress anything in the suppressed set.

    Useful when the agent wants to surface "skills related to what you
    already do" without needing a free-text query.
    """
    catalog = (catalog_loader or _safe_catalog_loader() or (lambda: []))()
    if not catalog:
        return []
    suppressed = _safe_call(suppressed_loader or _safe_suppressed_loader, default=set())
    usage = usage_loader() if usage_loader else _safe_usage_loader()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=int(lookback_days))

    # Rank skills by recent use, descending.
    candidates: List[Tuple[str, datetime, int]] = []
    for name, record in usage.items():
        if name in suppressed:
            continue
        last_used = _parse_iso((record or {}).get("last_used_at"))
        use_count = int((record or {}).get("use_count") or 0)
        if use_count < min_use_count or last_used is None or last_used < cutoff:
            continue
        candidates.append((name, last_used, use_count))
    candidates.sort(key=lambda row: (row[1], row[2]), reverse=True)
    seeds = [name for name, _, _ in candidates[: max(0, int(max_seed_skills))]]
    if not seeds:
        return []

    index = _describe_index(catalog, usage_loader=usage_loader)
    seed_tokens: List[str] = []
    for s in seeds:
        seed_tokens.extend(index.get(s, {}).get("tokens") or [])
    if not seed_tokens:
        return []
    scored = _score_query(seed_tokens, index, suppressed, history_weight=0.0)
    out: List[Dict[str, Any]] = []
    for name, score, meta in scored[: max(0, int(top_k))]:
        if name in seeds:
            continue
        out.append(
            {
                "name": name,
                "description": meta.get("description", ""),
                "category": meta.get("category", ""),
                "score": round(float(score), 4),
                "why": "related to recent: " + ", ".join(seeds[:3]),
            }
        )
    return out


def dismiss(skill_name: str) -> Tuple[bool, str]:
    """Mark a skill as dismissed so it stops being recommended.

    Delegates to ``tools.skill_usage.add_suppressed_name`` so the rest of
    the system (curator, web UI, etc.) sees the same suppressed set.
    Returns (True, "") on success, (False, reason) on failure.
    """
    name = str(skill_name or "").strip()
    if not name:
        return False, "empty skill name"
    try:
        from tools.skill_usage import add_suppressed_name  # type: ignore

        add_suppressed_name(name)
        return True, ""
    except Exception as e:
        return False, str(e) or type(e).__name__

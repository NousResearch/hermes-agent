"""Description-aware fuzzy scoring for slash-menu completions (ported from superagent-ai/grok-cli
``src/ui/slash-menu.ts``; mirrored in ``ui-tui/src/app/slash/fuzzyScore.ts``). Tiers: exact command
token (0), prefix (1), substring (2); the DESCRIPTION is tokenized and matched at +3 (3/4/5), so
``/summary`` surfaces a command whose description mentions summaries. Lower wins; ``math.inf`` = miss."""

from __future__ import annotations

import math
import re
from typing import Callable

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# (tier bump, field predicate) in priority order: exact, prefix, substring.
_TIERS = (
    (0, lambda field, q: field == q or f"/{field}" == q),
    (1, lambda field, q: field.startswith(q) or f"/{field}".startswith(q)),
    (2, lambda field, q: q in field),
)


def tokenize_search_text(value: str) -> list[str]:
    """Lowercase ``value`` and return it alongside its alphanumeric words."""
    normalized = value.lower()
    return [normalized, *[t for t in _TOKEN_SPLIT.split(normalized) if t]]


def normalize_slash_search_query(query: str) -> str:
    """Trim, drop leading slashes, lowercase — ``/Model`` and ``model`` alike."""
    return query.strip().lstrip("/").lower()


def _score_fields(fields: list[str], query: str, offset: int) -> float:
    return next((offset + bump for bump, pred in _TIERS if any(pred(f, query) for f in fields)), math.inf)


def score_slash_completion_item(item: dict, query: str) -> float:
    """Score one completion item dict against ``query``: ``text`` is the replacement token (may carry
    a leading slash or trailing space); ``meta`` is the human description. Lower is better."""
    name = str(item.get("text", "")).strip().lstrip("/")
    return min(
        _score_fields(tokenize_search_text(name), query, 0),
        _score_fields(tokenize_search_text(str(item.get("meta", ""))), query, 3),
    )


def fuzzy_rank_slash_items(
    items: list[dict], catalog: list[dict], query: str
) -> tuple[list[dict], Callable[[dict], float]]:
    """Merge fuzzy-matched ``catalog`` entries the prefix filter missed into ``items`` (which keep their
    identity) and sort by score (stable within a tier). Also returns a ``score_of`` lookup for downstream
    rankers to use as a leading sort key."""
    seen = {str(item.get("text", "")).strip() for item in items}
    merged = list(items) + [
        item for item in catalog
        if str(item.get("text", "")).strip() not in seen and not math.isinf(score_slash_completion_item(item, query))
    ]
    scores = {id(item): score_slash_completion_item(item, query) for item in merged}
    ranked = sorted((item for item in merged if not math.isinf(scores[id(item)])), key=lambda item: scores[id(item)])
    return ranked, lambda item: scores.get(id(item), math.inf)

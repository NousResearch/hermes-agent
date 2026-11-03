"""Description-aware fuzzy scoring for slash-menu completions (ported from superagent-ai/grok-cli
``src/ui/slash-menu.ts``; mirrored in ``ui-tui/src/app/slash/fuzzyScore.ts``). Tiers: exact command
token (0), prefix (1), substring (2); the DESCRIPTION is tokenized and matched at +3 (3/4/5), so
``/summary`` surfaces a command whose description mentions summaries. Lower wins; ``math.inf`` = miss."""

from __future__ import annotations

import math
import re
from typing import Callable

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")

# Tokenization memo: the completion path re-scores the same name/meta strings on every keystroke
# (identical item lists, evolving queries), so the split cost is paid once per distinct string.
# Bounded and cleared whole when full — pure cache of a pure function.
_TOKEN_CACHE: dict[str, list[str]] = {}
_TOKEN_CACHE_LIMIT = 8192

# Same memo as ``(field, "/" + field)``: the tiers compare both forms, and building that string per
# item per keystroke was the scorer's largest allocation.
_FIELD_CACHE: dict[str, list[tuple[str, str]]] = {}
_FIELD_CACHE_LIMIT = 8192


def tokenize_search_text(value: str) -> list[str]:
    """Lowercase ``value`` and return it alongside its alphanumeric words."""
    cached = _TOKEN_CACHE.get(value)
    if cached is None:
        normalized = value.lower()
        cached = [normalized, *[t for t in _TOKEN_SPLIT.split(normalized) if t]]
        if len(_TOKEN_CACHE) >= _TOKEN_CACHE_LIMIT:
            _TOKEN_CACHE.clear()
        _TOKEN_CACHE[value] = cached
    return cached


def _search_fields(value: str) -> list[tuple[str, str]]:
    """``(field, "/field")`` for each tokenized field of *value* — the forms every tier compares."""
    tokens = tokenize_search_text(value)  # memo hit: keeps the public token memo the single source
    cached = _FIELD_CACHE.get(value)
    if cached is None:
        cached = [(field, f"/{field}") for field in tokens]
        if len(_FIELD_CACHE) >= _FIELD_CACHE_LIMIT:
            _FIELD_CACHE.clear()
        _FIELD_CACHE[value] = cached
    return cached


def normalize_slash_search_query(query: str) -> str:
    """Trim, drop leading slashes, lowercase — ``/Model`` and ``model`` alike."""
    return query.strip().lstrip("/").lower()


def _score_fields(fields: list[tuple[str, str]], query: str, offset: int) -> float:
    """First tier that hits wins — exact token (offset), prefix (+1), substring (+2); ``inf`` = miss."""
    for field, slashed in fields:
        if field == query or slashed == query:
            return offset
    for field, slashed in fields:
        if field.startswith(query) or slashed.startswith(query):
            return offset + 1
    for field, _slashed in fields:
        if query in field:
            return offset + 2
    return math.inf


def score_slash_completion_item(item: dict, query: str) -> float:
    """Score one completion item dict against ``query``: ``text`` is the replacement token (may carry
    a leading slash or trailing space); ``meta`` is the human description. Lower is better.

    A name hit ends the scoring: name tiers score 0-2 and every description tier is offset by 3, so
    the description can only matter when the name does not match at all — and it is tokenized only
    then."""
    name_score = _score_fields(_search_fields(str(item.get("text", "")).strip().lstrip("/")), query, 0)
    if name_score < 3:
        return name_score
    return _score_fields(_search_fields(str(item.get("meta", ""))), query, 3)


def fuzzy_rank_slash_items(
    items: list[dict], catalog: list[dict], query: str
) -> tuple[list[dict], Callable[[dict], float]]:
    """Merge fuzzy-matched ``catalog`` entries the prefix filter missed into ``items`` (which keep their
    identity) and sort by score (stable within a tier). Also returns a ``score_of`` lookup for downstream
    rankers to use as a leading sort key."""
    seen = {str(item.get("text", "")).strip() for item in items}
    merged = list(items)
    # Score exactly once per item: the merge filter and the score map share one pass, so an
    # 850-skill catalog costs ~850 scorings per keystroke, not ~1700.
    scores = {id(item): score_slash_completion_item(item, query) for item in items}
    for item in catalog:
        if str(item.get("text", "")).strip() in seen:
            continue
        score = score_slash_completion_item(item, query)
        if not math.isinf(score):
            merged.append(item)
            scores[id(item)] = score
    ranked = sorted((item for item in merged if not math.isinf(scores[id(item)])), key=lambda item: scores[id(item)])
    return ranked, lambda item: scores.get(id(item), math.inf)

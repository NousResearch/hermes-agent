"""Query-relevance rerank for Skills Hub merged results (TypeSafe Jev Score).

Optional, disabled by default. When ``skills.hub_relevance_rerank.enabled``
is true in config.yaml, ``rerank_by_relevance`` scores each merged
``SkillMeta`` against the query with a Jev Score question and re-orders
*within* each trust rank so the limit cut still never drops a
builtin/official entry for a community one: relevance decides order inside a
rank, trust decides across ranks. The scorer is dependency-injected
(defaults to the Jev ``/v1/systemone`` Score client below); any scorer
failure fails open to the input order. No network happens when disabled or
when no scorer resolves.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("tools.skills_hub")

JEV_SYSTEMONE_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"

#: Score-question rubric levels (ascending relevance) for the Jev Score call.
RELEVANCE_LEVELS = ["Irrelevant", "Somewhat relevant", "Highly relevant"]

_Scorer = Callable[[str, Sequence[Any]], Dict[str, float]]


def _candidate_text(meta: Any) -> str:
    """Compact ``name + description + tags`` text a scorer can judge."""
    tags = getattr(meta, "tags", None)
    tag_text = (
        " ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags or "")
    )
    parts = [
        str(getattr(meta, "name", "") or ""),
        str(getattr(meta, "description", "") or ""),
    ]
    if tag_text.strip():
        parts.append(tag_text)
    return "\n".join(parts)


def jev_score_relevance(
    query: str, candidates: Sequence[Any], *, timeout: float = 8.0
) -> Dict[str, float]:
    """Score each candidate's query relevance via Jev ``Score`` (0-10 scale).

    One ``/v1/systemone`` call with one Score question per candidate; the
    answer ``score`` is the probability-weighted level index, scaled to 0-10
    for the shared ``RELEVANCE_LEVELS`` rubric. The key comes from
    ``TYPESAFE_API_KEY`` via the profile-scoped secret scope. Raises on any
    failure — callers fail open to trust order.
    """
    from agent.secret_scope import get_secret

    api_key = get_secret("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError("TYPESAFE_API_KEY is not set")
    questions = {
        f"{i}:{meta.identifier}": {
            "type": "score",
            "instructions": "How relevant is this skill to the search query?",
            "criteria": RELEVANCE_LEVELS,
        }
        for i, meta in enumerate(candidates)
    }
    body = json.dumps({
        "model": JEV_MODEL,
        "state": f"Query: {query}\n\n"
        + "\n---\n".join(_candidate_text(m) for m in candidates),
        "questions": questions,
    }).encode("utf-8")
    req = urllib.request.Request(
        JEV_SYSTEMONE_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    answers = payload.get("answers", {}) if isinstance(payload, dict) else {}
    if not isinstance(answers, dict):
        answers = {}
    scores: Dict[str, float] = {}
    for i, meta in enumerate(candidates):
        answer = answers.get(f"{i}:{meta.identifier}", {})
        if not isinstance(answer, dict):
            answer = {}
        try:
            scores[meta.identifier] = (
                float(answer.get("score", 0.0)) / (len(RELEVANCE_LEVELS) - 1) * 10.0
            )
        except (TypeError, ValueError):
            scores[meta.identifier] = 0.0
    return scores


def rerank_by_relevance(
    query: str,
    results: List[Any],
    *,
    scorer: Optional[_Scorer] = None,
    timeout: float = 8.0,
) -> List[Any]:
    """Stable re-order of trust-sorted ``results`` by query relevance.

    Rerank runs *within* each trust rank: trust decides across ranks (the
    trust-before-truncate invariant holds), relevance decides inside a rank.
    ``scorer`` maps ``(query, candidates)`` to ``{identifier: 0-10 score}``;
    the default is the Jev Score client. Stable: ties keep trust order.
    Any scorer failure fails open to the input order (logged at debug).
    """
    if len(results) < 2:
        return list(results)
    score_fn = scorer or (
        lambda q, cands: jev_score_relevance(q, cands, timeout=timeout)
    )
    try:
        from tools.skills_hub_models import TRUST_RANK as _TRUST_RANK
    except Exception:
        _TRUST_RANK = {}
    try:
        scores = score_fn(query, results)
    except Exception as e:
        logger.debug("Skills Hub relevance rerank failed; keeping trust order: %s", e)
        return list(results)
    try:
        ranked = sorted(
            range(len(results)),
            key=lambda i: (
                -_TRUST_RANK.get(results[i].trust_level, 0),
                -float(scores.get(results[i].identifier, 0.0)),
                i,
            ),
        )
    except (TypeError, ValueError, AttributeError) as e:
        logger.debug("Skills Hub relevance rerank failed; keeping trust order: %s", e)
        return list(results)
    return [results[i] for i in ranked]

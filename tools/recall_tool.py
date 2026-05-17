"""Unified recall tool for graph memory + session history.

This is a routing layer, not a new memory store.  It lets the model ask for
manual recall with explicit depth/source/budget/provenance while keeping the
heavy parts opt-in.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from agent.memory_router import stable_recall_key
from tools.registry import registry, tool_error

_DEPTH_ORDER = {"light": 1, "standard": 2, "deep": 3, "evidence": 4}
_BUDGET_LIMITS = {"tiny": 1, "small": 2, "medium": 3, "large": 5}
_VALID_SOURCES = {"graph", "session_fts", "session_summary"}


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, Iterable):
        return [str(p).strip() for p in value if str(p).strip()]
    return []


def _normalize_depth(depth: str) -> str:
    depth = (depth or "standard").strip().lower()
    return depth if depth in _DEPTH_ORDER else "standard"


def _normalize_budget(budget: str) -> str:
    budget = (budget or "medium").strip().lower()
    return budget if budget in _BUDGET_LIMITS else "medium"


def _default_sources(depth: str, *, has_memory_manager: bool) -> List[str]:
    sources: List[str] = []
    if has_memory_manager:
        sources.append("graph")
    if _DEPTH_ORDER.get(depth, 2) >= _DEPTH_ORDER["standard"]:
        sources.append("session_summary" if depth in {"deep", "evidence"} else "session_fts")
    return sources or (["session_summary"] if depth in {"deep", "evidence"} else ["session_fts"])


def recall(
    query: str,
    *,
    mode: str = "manual",
    depth: str = "standard",
    sources: Optional[List[str]] = None,
    budget: str = "medium",
    provenance: str = "ids",
    role_filter: Optional[str] = None,
    limit: Optional[int] = None,
    memory_manager=None,
    db=None,
    current_session_id: Optional[str] = None,
) -> str:
    """Route a recall request across provider memory and session history."""
    query = (query or "").strip()
    if not query:
        return tool_error("recall requires a non-empty query", success=False)

    depth = _normalize_depth(depth)
    budget = _normalize_budget(budget)
    mode = (mode or "manual").strip().lower()
    provenance = (provenance or "ids").strip().lower()

    normalized_sources = [s for s in _as_list(sources) if s in _VALID_SOURCES]
    if not normalized_sources:
        normalized_sources = _default_sources(depth, has_memory_manager=bool(memory_manager))

    max_limit = _BUDGET_LIMITS[budget]
    if limit is None:
        limit = max_limit
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = max_limit
    limit = max(1, min(limit, max_limit, 5))

    results: Dict[str, Any] = {}
    errors: List[str] = []

    if "graph" in normalized_sources:
        if memory_manager is None:
            errors.append("graph source requested but no external memory provider is active")
        else:
            try:
                graph_text = memory_manager.recall_now_all(
                    query,
                    mode=mode,
                    depth=depth,
                    sources=normalized_sources,
                    budget=budget,
                    provenance=provenance,
                    session_id=current_session_id or "",
                )
                if graph_text and graph_text.strip():
                    results["graph"] = graph_text
                else:
                    results["graph"] = ""
            except Exception as exc:  # pragma: no cover - defensive fail-open
                errors.append(f"graph recall failed: {exc}")

    wants_session = any(s in normalized_sources for s in ("session_fts", "session_summary"))
    if wants_session:
        if db is None:
            errors.append("session source requested but session database is not available")
        else:
            try:
                from tools.session_search_tool import session_search

                # session_search currently returns summaries for queried history.
                # `session_fts` is treated as the standard, bounded session-history
                # backend until a raw metadata-only backend lands.
                raw = session_search(
                    query=query,
                    role_filter=role_filter,
                    limit=limit,
                    db=db,
                    current_session_id=current_session_id,
                )
                try:
                    session_payload = json.loads(raw)
                    results["sessions"] = session_payload
                    if isinstance(session_payload, dict) and session_payload.get("success") is False:
                        errors.append("session recall returned unsuccessful result")
                except Exception:
                    results["sessions"] = raw
            except Exception as exc:  # pragma: no cover - defensive fail-open
                errors.append(f"session recall failed: {exc}")

    recall_key = stable_recall_key(query, mode=mode, depth=depth, sources=normalized_sources)
    auto_recall_key = stable_recall_key(query, mode="auto", depth=depth, sources=normalized_sources)
    return json.dumps(
        {
            "success": bool(results) and not errors,
            "query": query,
            "mode": mode,
            "depth": depth,
            "sources": normalized_sources,
            "budget": budget,
            "provenance": provenance,
            "recall_key": recall_key,
            "auto_recall_key": auto_recall_key,
            "results": results,
            "errors": errors,
        },
        ensure_ascii=False,
    )


def check_recall_requirements() -> bool:
    # Keep the session_search toolset availability aligned with the concrete
    # session_search backend.  recall can still use graph-only paths at runtime
    # through AIAgent's special dispatch, but its public tool schema lives in
    # the session_search toolset and should not make that toolset appear
    # available earlier than session_search itself.
    try:
        from tools.session_search_tool import check_session_search_requirements
        return check_session_search_requirements()
    except Exception:
        return False


RECALL_SCHEMA = {
    "name": "recall",
    "description": (
        "Unified explicit recall across external memory providers and past sessions. "
        "Use when you need more context than the automatic lightweight memory prefetch provided. "
        "Choose depth='light' for graph/background facts, 'standard' for graph + bounded session hits, "
        "'deep' for session_search-style historical summaries, and 'evidence' when the user asks for source/provenance. "
        "Manual recall is never duplicate-suppressed; it also marks the recall key so automatic recall does not immediately repeat it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Recall query. Use OR between keywords for broad session history search."},
            "mode": {"type": "string", "enum": ["manual", "user", "auto"], "default": "manual"},
            "depth": {"type": "string", "enum": ["light", "standard", "deep", "evidence"], "default": "standard"},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["graph", "session_fts", "session_summary"]},
                "description": "Sources to query. Omit for depth-based defaults.",
            },
            "budget": {"type": "string", "enum": ["tiny", "small", "medium", "large"], "default": "medium"},
            "provenance": {"type": "string", "enum": ["none", "ids", "links", "verbatim"], "default": "ids"},
            "role_filter": {"type": "string", "description": "Optional session role filter, e.g. 'user,assistant'."},
            "limit": {"type": "integer", "description": "Max session results; clamped by budget and hard-capped at 5."},
        },
        "required": ["query"],
    },
}


registry.register(
    name="recall",
    toolset="session_search",
    schema=RECALL_SCHEMA,
    handler=lambda args, **kw: recall(
        query=args.get("query") or "",
        mode=args.get("mode", "manual"),
        depth=args.get("depth", "standard"),
        sources=args.get("sources"),
        budget=args.get("budget", "medium"),
        provenance=args.get("provenance", "ids"),
        role_filter=args.get("role_filter"),
        limit=args.get("limit"),
        memory_manager=kw.get("memory_manager"),
        db=kw.get("db"),
        current_session_id=kw.get("current_session_id"),
    ),
    check_fn=check_recall_requirements,
    emoji="🧠",
)

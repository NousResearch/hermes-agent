"""Hermes-native local-first research search tools."""

from __future__ import annotations

import json
from typing import Any

from tools.registry import registry


def _research_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        rs = (cfg or {}).get("research_search") or {}
        if isinstance(rs, dict):
            return bool(rs.get("enabled", True))
    except Exception:
        pass
    return True


def _json_result(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


RESEARCH_GATHER_SCHEMA = {
    "name": "research_gather",
    "description": (
        "Plan and run a local-first web research pass. Classifies the topic, "
        "generates typed query fan-out, searches local research memory, uses "
        "web_search/web_extract for fresh evidence, and returns a structured "
        "evidence bundle with sources, gaps, conflicts, and usage. Use this "
        "as the default entrypoint for search, research, report, latest, current, "
        "or as-of-date tasks. Final reports should cite source_table or "
        "citation_metadata from the returned bundle."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user question or research task.",
            },
            "topic_type": {
                "type": "string",
                "description": (
                    "Search profile: auto, current_events, technical, academic, "
                    "medical_pharma, finance, engineering, technology, gaming, "
                    "social_trends, music, shopping, domestic_news, geopolitics, "
                    "sports, legal_regulatory, company_market, product, local, "
                    "obscure_lookup, or general."
                ),
                "default": "auto",
            },
            "freshness": {
                "type": "string",
                "description": "Freshness target: auto, latest, recent, stable, or historical.",
                "default": "auto",
            },
            "depth": {
                "type": "string",
                "description": "Research depth: fast, balanced, or thorough.",
                "default": "thorough",
            },
            "max_queries": {
                "type": "integer",
                "description": "Maximum fan-out search queries. Defaults to config.",
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum pages/sources to extract and return. Defaults to config.",
            },
        },
        "required": ["question"],
    },
}

RESEARCH_LOCAL_SEARCH_SCHEMA = {
    "name": "research_local_search",
    "description": (
        "Search the local DuckDB research memory with FTS. Returns cached "
        "documents/evidence only; use research_gather for live web fallback."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "vertical": {
                "type": "string",
                "description": "Local vertical/corpus to search, or auto.",
                "default": "auto",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum local results.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

RESEARCH_INDEX_URL_SCHEMA = {
    "name": "research_index_url",
    "description": (
        "Fetch URLs with web_extract and index them into the local DuckDB "
        "research memory for future local-first searches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URLs to fetch and index, maximum 5 per call.",
                "maxItems": 5,
            },
            "vertical": {
                "type": "string",
                "description": "Corpus/vertical label for indexed documents.",
                "default": "web",
            },
            "force_refresh": {
                "type": "boolean",
                "description": "Reserved for cache refresh behavior.",
                "default": False,
            },
            "render_mode": {
                "type": "string",
                "description": "Reserved renderer preference: auto, extract, or browser.",
                "default": "auto",
            },
        },
        "required": ["urls"],
    },
}

RESEARCH_STATUS_SCHEMA = {
    "name": "research_status",
    "description": (
        "Return status for the local research-search stack, including DuckDB "
        "availability, database path, FTS/chunk/vector status, discovery backend, "
        "crawler policy, and whether research_gather auto-indexing is enabled."
    ),
    "parameters": {"type": "object", "properties": {}},
}

RESEARCH_PLAN_SCHEMA = {
    "name": "research_plan",
    "description": "Plan typed research query lanes and source requirements without fetching.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Research question."},
            "topic_type": {"type": "string", "default": "auto"},
            "freshness": {"type": "string", "default": "auto"},
            "depth": {"type": "string", "default": "thorough"},
        },
        "required": ["question"],
    },
}

RESEARCH_SEARCH_CANDIDATES_SCHEMA = {
    "name": "research_search_candidates",
    "description": "Discovery only: search local research memory and live web, returning candidate URLs without extraction.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Research question."},
            "query": {"type": "string", "description": "Direct search query override."},
            "topic_type": {"type": "string", "default": "auto"},
            "freshness": {"type": "string", "default": "auto"},
            "depth": {"type": "string", "default": "balanced"},
            "max_queries": {"type": "integer", "description": "Maximum fan-out queries."},
            "limit": {"type": "integer", "default": 20},
        },
    },
}

RESEARCH_EXTRACT_EVIDENCE_SCHEMA = {
    "name": "research_extract_evidence",
    "description": "Extract selected URLs and normalize them as evidence/source records.",
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
            "question": {"type": "string", "description": "Question used for relevance context."},
            "max_sources": {"type": "integer", "default": 5},
        },
        "required": ["urls"],
    },
}

RESEARCH_RERANK_SCHEMA = {
    "name": "research_rerank",
    "description": "Rank already gathered research source dictionaries for quality and diversity.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Research question."},
            "sources": {"type": "array", "items": {"type": "object"}},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["question", "sources"],
    },
}

RESEARCH_GAP_ANALYZE_SCHEMA = {
    "name": "research_gap_analyze",
    "description": "Analyze gathered sources for extraction, diversity, official/primary, freshness, and adversarial gaps.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Research question."},
            "sources": {"type": "array", "items": {"type": "object"}},
            "plan": {"type": "object", "description": "Optional research_plan output plan."},
        },
        "required": ["question"],
    },
}

RESEARCH_HELP_SCHEMA = {
    "name": "research_help",
    "description": "Return compact guidance for choosing research_search tools and workflows.",
    "parameters": {
        "type": "object",
        "properties": {
            "topic_type": {"type": "string", "default": "auto"},
        },
    },
}


def _handle_research_gather(args: dict, **_kw) -> str:
    from agent.research_search import research_gather

    return _json_result(
        research_gather(
            question=args.get("question", ""),
            topic_type=args.get("topic_type", "auto"),
            freshness=args.get("freshness", "auto"),
            depth=args.get("depth", "thorough"),
            max_queries=args.get("max_queries"),
            max_pages=args.get("max_pages"),
        )
    )


def _handle_research_local_search(args: dict, **_kw) -> str:
    from agent.research_search import research_local_search

    return _json_result(
        research_local_search(
            query=args.get("query", ""),
            vertical=args.get("vertical", "auto"),
            limit=args.get("limit", 10),
        )
    )


def _handle_research_index_url(args: dict, **_kw) -> str:
    from agent.research_search import research_index_url

    raw_urls = args.get("urls", [])
    urls = raw_urls[:5] if isinstance(raw_urls, list) else []
    return _json_result(
        research_index_url(
            urls=urls,
            vertical=args.get("vertical", "web"),
            force_refresh=bool(args.get("force_refresh", False)),
            render_mode=args.get("render_mode", "auto"),
        )
    )


def _handle_research_status(_args: dict, **_kw) -> str:
    from agent.research_search import research_status

    return _json_result(research_status())


def _handle_research_plan(args: dict, **_kw) -> str:
    from agent.research_search import research_plan

    return _json_result(
        research_plan(
            question=args.get("question", ""),
            topic_type=args.get("topic_type", "auto"),
            freshness=args.get("freshness", "auto"),
            depth=args.get("depth", "thorough"),
        )
    )


def _handle_research_search_candidates(args: dict, **_kw) -> str:
    from agent.research_search import research_search_candidates

    return _json_result(
        research_search_candidates(
            question=args.get("question", ""),
            query=args.get("query", ""),
            topic_type=args.get("topic_type", "auto"),
            freshness=args.get("freshness", "auto"),
            depth=args.get("depth", "balanced"),
            max_queries=args.get("max_queries"),
            limit=args.get("limit", 20),
        )
    )


def _handle_research_extract_evidence(args: dict, **_kw) -> str:
    from agent.research_search import research_extract_evidence

    raw_urls = args.get("urls", [])
    urls = raw_urls[:5] if isinstance(raw_urls, list) else []
    return _json_result(
        research_extract_evidence(
            urls=urls,
            question=args.get("question", ""),
            max_sources=args.get("max_sources", 5),
        )
    )


def _handle_research_rerank(args: dict, **_kw) -> str:
    from agent.research_search import research_rerank

    return _json_result(
        research_rerank(
            question=args.get("question", ""),
            sources=args.get("sources", []),
            limit=args.get("limit", 10),
        )
    )


def _handle_research_gap_analyze(args: dict, **_kw) -> str:
    from agent.research_search import research_gap_analyze

    return _json_result(
        research_gap_analyze(
            question=args.get("question", ""),
            sources=args.get("sources", []),
            plan=args.get("plan"),
        )
    )


def _handle_research_help(args: dict, **_kw) -> str:
    from agent.research_search import research_help

    return _json_result(research_help(topic_type=args.get("topic_type", "auto")))


registry.register(
    name="research_gather",
    toolset="research_search",
    schema=RESEARCH_GATHER_SCHEMA,
    handler=_handle_research_gather,
    check_fn=_research_enabled,
    emoji="🧭",
    max_result_size_chars=150_000,
)
registry.register(
    name="research_local_search",
    toolset="research_search",
    schema=RESEARCH_LOCAL_SEARCH_SCHEMA,
    handler=_handle_research_local_search,
    check_fn=_research_enabled,
    emoji="🗂️",
    max_result_size_chars=100_000,
)
registry.register(
    name="research_index_url",
    toolset="research_search",
    schema=RESEARCH_INDEX_URL_SCHEMA,
    handler=_handle_research_index_url,
    check_fn=_research_enabled,
    emoji="📚",
    max_result_size_chars=100_000,
)
registry.register(
    name="research_status",
    toolset="research_search",
    schema=RESEARCH_STATUS_SCHEMA,
    handler=_handle_research_status,
    check_fn=_research_enabled,
    emoji="📊",
    max_result_size_chars=20_000,
)

for _name, _schema, _handler, _emoji in [
    ("research_plan", RESEARCH_PLAN_SCHEMA, _handle_research_plan, "🗺️"),
    (
        "research_search_candidates",
        RESEARCH_SEARCH_CANDIDATES_SCHEMA,
        _handle_research_search_candidates,
        "🔎",
    ),
    (
        "research_extract_evidence",
        RESEARCH_EXTRACT_EVIDENCE_SCHEMA,
        _handle_research_extract_evidence,
        "🧾",
    ),
    ("research_rerank", RESEARCH_RERANK_SCHEMA, _handle_research_rerank, "📌"),
    (
        "research_gap_analyze",
        RESEARCH_GAP_ANALYZE_SCHEMA,
        _handle_research_gap_analyze,
        "🕳️",
    ),
    ("research_help", RESEARCH_HELP_SCHEMA, _handle_research_help, "❔"),
]:
    registry.register(
        name=_name,
        toolset="research_search",
        schema=_schema,
        handler=_handler,
        check_fn=_research_enabled,
        emoji=_emoji,
        max_result_size_chars=80_000,
    )

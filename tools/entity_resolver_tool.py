#!/usr/bin/env python3
"""
Entity Resolver Tool — fuzzy + phonetic search across all registry sheets.

Searches contacts, projects, entities, land_proposals, and topics in one call.
Uses the NounResolver's in-memory index (built on startup, refreshed every 5 min)
so queries are sub-millisecond — no API calls at search time.

Matching tiers (applied per word window):
  1. Exact       — normalized string equality
  2. Fuzzy       — Jaro-Winkler + token_sort_ratio (rapidfuzz)
  3. Phonetic    — Metaphone code match (jellyfish; falls back to vowel-stripped prefix)

Multi-word queries are split into 1–3 word windows so "sunny amber project"
simultaneously matches the contact "Sunny Sadhwani" and the project "Ranka Amber".

For project / entity / land results, associated_contacts from the sheet are returned
so the agent can immediately know who is linked to that project without a second call.

Registered as: entity_resolver
Toolset: google_workspace
"""

import json
import logging

logger = logging.getLogger(__name__)


# ── Handler ────────────────────────────────────────────────────────────────────

def _handle(params: dict, **_kwargs) -> str:
    query   = str(params.get("query", "")).strip()
    limit   = int(params.get("limit", 10))
    types   = params.get("types") or None          # list[str] or None → all sheets
    context = params.get("context") or None        # list[str] of recent messages

    if not query:
        return json.dumps({"error": "query is required"})

    if limit < 1 or limit > 50:
        limit = 10

    try:
        from tools.noun_resolver import get_resolver
        resolver = get_resolver()
    except Exception as e:
        return json.dumps({"error": f"resolver unavailable: {e}"})

    results = resolver.search(query, limit=limit, types=types, context=context)

    return json.dumps({
        "query":   query,
        "count":   len(results),
        "results": results,
    }, indent=2, ensure_ascii=False)


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = {
    "name": "entity_resolver",
    "description": (
        "Search for people, projects, entities, land proposals, or topics by name. "
        "Handles voice transcription errors, partial names, phonetic variants, and aliases "
        "— e.g. 'ragoo' → 'Raghu Iyer', 'amber project' → 'Ranka Amber', "
        "'narsem' → 'Narasimha Raju'. "
        "Multi-word queries match across all entity types simultaneously: "
        "'sunny amber' returns both the contact 'Sunny Sadhwani' and project 'Ranka Amber'. "
        "For project/entity/land matches, associated_contacts are returned automatically. "
        "Use this before calling contact_resolver when you are unsure whether the user "
        "is referring to a person, a project, or a land parcel."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Name or phrase to search. Can be a full name, first name only, "
                    "phonetic misspelling, voice error, alias, or multi-word phrase. "
                    "Examples: 'ragoo', 'bhuvanesh', 'narsem raju', 'amber project', "
                    "'sunny sadhwani', 'RO', 'riverstun farms'."
                ),
            },
            "types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["contacts", "projects", "entities", "land_proposals", "topics"],
                },
                "description": (
                    "Optional: restrict results to specific sheet types. "
                    "Omit to search all sheets. "
                    "Example: ['contacts'] to search people only."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return. Default 10, max 50.",
            },
            "context": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional: list of recent message strings. "
                    "Entities mentioned in context are ranked higher (+0.05 score). "
                    "Useful for disambiguating common first names."
                ),
            },
        },
        "required": ["query"],
    },
}


# ── Availability check ─────────────────────────────────────────────────────────

def _check_available() -> bool:
    try:
        from tools.noun_resolver import get_resolver  # noqa: F401
        return True
    except ImportError:
        return False


# ── Registration ───────────────────────────────────────────────────────────────

from tools.registry import registry  # noqa: E402

registry.register(
    name="entity_resolver",
    toolset="general",
    schema=_SCHEMA,
    handler=_handle,
    check_fn=_check_available,
    requires_env=[],
    is_async=False,
)

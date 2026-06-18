"""Agent-facing Skills-Hub DISCOVERY (Phase 6 — the "employee model").

Lets an agent find and preview skills it does NOT already have, when a job
needs an approach outside its mastered set. This is the *discover* half of the
employee model (identify a limit -> discover -> acquire OR delegate); it is
strictly **READ-ONLY**: it searches/inspects remote skill registries but never
installs. Acquisition (install) is a separate, trust-gated step.

Wraps the existing Skills Hub library (``tools.skills_hub`` +
``hermes_cli.skills_hub``) that backs the ``hermes skills`` CLI, so it inherits
the same source router, trust ranking, and index-aware search.

Registered into the ``skills`` toolset (auto-imported by tools/registry.py).
"""
import json

from tools.registry import registry, tool_error
from tools.skills_tool import check_skills_requirements

_MAX_LIMIT = 20
_TRUST_RANK = {"builtin": 3, "trusted": 2, "community": 1}


def _search(query: str, limit: int):
    from tools.skills_hub import (
        GitHubAuth, create_source_router, parallel_search_sources,
    )

    sources = create_source_router(GitHubAuth())
    results, _counts, _timed_out = parallel_search_sources(
        sources, query=query, source_filter="all", overall_timeout=20,
    )
    # Dedupe by identifier, keeping the most-trusted variant.
    seen: dict = {}
    for r in results:
        rank = _TRUST_RANK.get(r.trust_level, 0)
        if r.identifier not in seen or rank > _TRUST_RANK.get(seen[r.identifier].trust_level, 0):
            seen[r.identifier] = r
    ranked = sorted(
        seen.values(),
        key=lambda r: (-_TRUST_RANK.get(r.trust_level, 0), r.name.lower()),
    )[:limit]
    return [
        {"name": r.name, "description": r.description, "source": r.source,
         "trust": r.trust_level, "identifier": r.identifier}
        for r in ranked
    ]


def skill_discover(query=None, identifier=None, limit=10, task_id=None):
    try:
        limit = max(1, min(int(limit or 10), _MAX_LIMIT))
    except (TypeError, ValueError):
        limit = 10

    try:
        if identifier:
            from hermes_cli.skills_hub import inspect_skill
            meta = inspect_skill(str(identifier))
            if not meta:
                return json.dumps({"success": False, "error": f"no skill found for '{identifier}'"})
            return json.dumps({"success": True, "skill": meta})

        if not query or not str(query).strip():
            return json.dumps({
                "success": False,
                "error": "provide 'query' to search the hub, or 'identifier' to inspect a skill",
            })

        results = _search(str(query).strip(), limit)
        return json.dumps({
            "success": True,
            "query": query,
            "count": len(results),
            "results": results,
            "note": "Discovery only — this does NOT install. To use a found skill, "
                    "request acquisition (a separate, approved step) or delegate the task.",
        })
    except Exception as exc:  # noqa: BLE001 - never crash the agent loop on a hub hiccup
        return tool_error(f"skill_discover failed: {exc}")


SKILL_DISCOVER_SCHEMA = {
    "name": "skill_discover",
    "description": (
        "Search the Skills Hub for skills you don't already have — use when a task "
        "needs an approach outside your mastered skills. Search by need (e.g. "
        "'parse PDF bank statements') to see candidates, then pass an `identifier` "
        "to preview that skill's details + SKILL.md. READ-ONLY: this does not "
        "install anything; acquiring a skill is a separate, approved step. Always "
        "check your own skills_list first — only reach for the hub when you're "
        "genuinely missing a capability the job needs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language description of the capability you need."},
            "identifier": {"type": "string", "description": "A skill identifier from a prior search result, to preview its details."},
            "limit": {"type": "integer", "description": f"Max results (default 10, max {_MAX_LIMIT})."},
        },
        "required": [],
    },
}


registry.register(
    name="skill_discover",
    toolset="skills",
    schema=SKILL_DISCOVER_SCHEMA,
    handler=lambda args, **kw: skill_discover(
        query=args.get("query"), identifier=args.get("identifier"),
        limit=args.get("limit", 10), task_id=kw.get("task_id"),
    ),
    check_fn=check_skills_requirements,
    emoji="🔎",
)

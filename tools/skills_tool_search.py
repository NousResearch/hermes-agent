"""``skill_search``: query installed skills and the Skills Hub without loading SKILL.md bodies.

A model-facing wrapper over the installed-skill index ``skills_list`` reads and the hub search
stack ``hermes skills search`` drives (``tools.skills_hub_search``). Results are compact
(identifier, description, provenance, tags); loading a body stays ``skill_view`` and installing
stays ``hermes skills install``.
"""

import json
from typing import Any, Dict, List

from tools.registry import tool_error
from tools.skills_tool_plugin import MAX_DESCRIPTION_LENGTH, MAX_NAME_LENGTH

# "installed" plus every hub SOURCE_ID ``create_source_router`` builds except "url", which
# resolves a direct link rather than answering a query.
SEARCH_SOURCES = (
    "all", "installed", "official", "hermes-index", "skills-sh", "well-known",
    "github", "clawhub", "lobehub", "browse-sh",
)
MAX_RESULTS = 50

SKILL_SEARCH_SCHEMA = {
    "name": "skill_search",
    "description": (
        "Search installed skills and the Hermes Skills Hub by query without loading full SKILL.md "
        "bodies. Returns compact identifiers and descriptions only; use skill_view for installed "
        "matches, or tell the user to run `hermes skills install <identifier>` for hub matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, e.g. 'flutter qa', 'github review', or 'kubernetes'.",
            },
            "source": {
                "type": "string",
                "enum": list(SEARCH_SOURCES),
                "description": "Optional source filter. Default 'all'. Use 'installed' to skip the remote hub.",
            },
            "limit": {"type": "integer", "description": f"Maximum results to return; capped at {MAX_RESULTS}."},
            "include_installed": {
                "type": "boolean",
                "description": "When true (default), list matching installed skills before hub results.",
            },
        },
        "required": ["query"],
    },
}


def _clip(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _clip_tags(value: Any, *, max_tags: int = 20, max_len: int = 64) -> List[str]:
    return [_clip(tag, max_len) for tag in value[:max_tags]] if isinstance(value, list) else []


def _installed_matches(skill: Dict[str, Any], query: str) -> bool:
    fields = (skill.get("name"), skill.get("description"), skill.get("category"),
              " ".join(skill.get("tags") or []))
    return query.lower() in " ".join(str(f or "") for f in fields).lower()


def _installed_row(skill: Dict[str, Any]) -> Dict[str, Any]:
    name = _clip(skill.get("name"), MAX_NAME_LENGTH)
    # The bare name is the handle skills_list already tells the model to pass to skill_view.
    return {"name": name, "identifier": name, "source": "installed",
            "description": _clip(skill.get("description"), MAX_DESCRIPTION_LENGTH),
            "installed": True, "category": skill.get("category"),
            "tags": _clip_tags(skill.get("tags") or [])}


def _hub_row(meta: Any) -> Dict[str, Any]:
    row = {"name": _clip(getattr(meta, "name", ""), MAX_NAME_LENGTH),
           "identifier": _clip(getattr(meta, "identifier", ""), 256),
           "source": _clip(getattr(meta, "source", ""), 64),
           "description": _clip(getattr(meta, "description", ""), MAX_DESCRIPTION_LENGTH),
           "installed": False,
           "trust_level": _clip(getattr(meta, "trust_level", "community"), 64),
           "tags": _clip_tags(getattr(meta, "tags", []) or [])}
    for key in ("repo", "path"):
        if value := getattr(meta, key, None):
            row[key] = _clip(value, 256)
    return row


def skill_search(query: str, source: str = "all", limit: int = 10,
                 include_installed: bool = True, task_id: str = None) -> str:
    """Installed matches first (``skills_list`` order), then hub matches, deduped by identifier
    and capped at ``limit`` (max ``MAX_RESULTS``). ``task_id`` is handler parity."""
    try:
        query = str(query or "").strip()
        if not query:
            return tool_error("skill_search requires a non-empty query", success=False)
        source = str(source or "all").strip() or "all"
        if source not in SEARCH_SOURCES:
            return tool_error(f"Unsupported skill search source '{source}'. "
                              f"Use one of: {', '.join(SEARCH_SOURCES)}", success=False)
        capped = max(1, min(int(limit or 10), MAX_RESULTS))
        results: List[Dict[str, Any]] = []
        seen: set = set()

        def _add(row: Dict[str, Any]) -> bool:
            key = row.get("identifier") or row.get("name")
            if key and key not in seen:
                seen.add(key)
                results.append(row)
            return len(results) >= capped

        if include_installed or source == "installed":
            from tools.skills_tool import _find_all_skills, _sort_skills
            for skill in _sort_skills([s for s in _find_all_skills() if _installed_matches(s, query)]):
                if _add(_installed_row(skill)):
                    break
        if source != "installed" and len(results) < capped:
            from tools.skills_hub_search import create_source_router, unified_search
            for meta in unified_search(query, create_source_router(), source_filter=source,
                                       limit=capped - len(results)):
                if _add(_hub_row(meta)):
                    break
        return json.dumps({
            "success": True, "query": query, "source": source,
            "include_installed": include_installed, "limit": capped,
            "results": results, "count": len(results),
            "hint": ("Use skill_view(name=<identifier>) for installed results; external hub results "
                     "install with `hermes skills install <identifier>`."),
        }, ensure_ascii=False)
    except Exception as e:
        return tool_error(str(e), success=False)

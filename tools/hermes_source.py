"""Hermes Source Accelerator tool wrappers.

These wrappers expose the deterministic source accelerator hot path to agents.
They deliberately do not call an LLM, Graphify rebuild, or GitNexus full reindex.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import json

from tools.registry import registry


def _as_json(value: Any) -> str:
    """Coerce handler payloads to a JSON string.

    Every tool handler in Hermes is contracted to return a JSON string
    (see ``tools.registry.tool_result``). The source-accelerator query
    functions return dicts, which previously fell through to
    ``HermesState._encode_content`` and got persisted with the
    ``\\x00json:`` multimodal-content sentinel. That sentinel embeds a
    literal NUL byte in the conversation history, which Copilot's
    Anthropic-side proxy rejects with HTTP 400 ``invalid_request_body``
    because JSON strings may not contain ``\\u0000``.

    Returning a plain JSON string here keeps the tool result on the
    scalar path: ``append_message`` stores it verbatim, no sentinel,
    no NUL byte, no provider tripwire.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"error": "non-serializable tool result", "repr": repr(value)[:500]})


def _unwrap_args(args: Any) -> dict[str, Any]:
    """Normalize tool arguments across provider quirks.

    Vertex/Gemini emits ``{"parameters": {"query": ...}}`` while OpenAI
    and Anthropic-flavored callers emit ``{"query": ...}`` directly. The
    accelerator's handler reads ``args.get("query", "")`` and previously
    fell through to an empty string on the Vertex shape, which then
    triggered a match-all slow path on every scope (observed: 9.2s
    wall on prompts like ``MEMORY.md`` and ``XyZyXyZyXyZ_does_not_exist``
    under copilot/gemini-2.5-pro and google-gemini-cli/gemini-3-flash-preview).

    If the dict has exactly one key ``parameters`` whose value is a
    dict, unwrap one level.
    """
    if isinstance(args, dict) and set(args.keys()) == {"parameters"} and isinstance(args["parameters"], dict):
        return args["parameters"]
    return args if isinstance(args, dict) else {}


_ACCELERATOR_ROOT = Path(__file__).resolve().parents[2] / "workspace" / "scripts" / "source_accelerator"
if str(_ACCELERATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACCELERATOR_ROOT))


def _check_source_accelerator() -> bool:
    return (_ACCELERATOR_ROOT / "source_accelerator" / "query.py").exists()


def _search(args: dict[str, Any], **kw):
    from source_accelerator.query import search
    args = _unwrap_args(args)
    query = str(args.get("query", "")).strip()
    if not query:
        return _as_json({
            "error": "empty query",
            "hint": "pass query=<file path, symbol name, config key, skill name, tool name, cron name, log keyword, memory keyword, or session text>",
            "examples": [
                {"query": "kanban_tools.py"},
                {"query": "SessionManager"},
                {"query": "kanban.max_in_progress"},
                {"query": "Bitwarden", "scope": "skills"},
                {"query": "telegram", "scope": "src"},
                {"query": "ERROR", "scope": "logs"},
                {"query": "routing", "scope": "memory"},
            ],
            "valid_scopes": ["auto", "src", "tests", "skills", "workspace", "scripts", "plugins", "profiles", "configs", "docs", "obsidian", "logs", "memory", "sessions"],
            "valid_modes": ["auto", "path", "symbol", "text", "logs", "memory_session", "relationship", "architecture"],
        })
    return _as_json(search(
        query=query,
        scope=str(args.get("scope", "auto")),
        mode=str(args.get("mode", "auto")),
        limit=int(args.get("limit", 20)),
    ))


def _open(args: dict[str, Any], **kw):
    from source_accelerator.query import open_result
    args = _unwrap_args(args)
    rid = str(args.get("result_id", "")).strip()
    if not rid:
        return _as_json({
            "error": "empty result_id",
            "hint": "pass result_id=<id returned by hermes_source_search results[].result_id>",
        })
    return _as_json(open_result(
        result_id=rid,
        context_lines=int(args.get("context_lines", args.get("context", 80))),
    ))


def _status(args: dict[str, Any], **kw):
    from source_accelerator.status import status
    return _as_json(status())


def _refresh(args: dict[str, Any], **kw):
    from source_accelerator.indexer import refresh_fast
    args = _unwrap_args(args)
    kind=str(args.get("kind", "fast"))
    if kind not in {"fast", "manual"}:
        return _as_json({"status":"error","error":"Only deterministic fast/manual refresh is supported by this tool. Heavy Graphify/GitNexus rebuilds are intentionally out of hot path."})
    return _as_json(refresh_fast(kind=kind, force=bool(args.get("force", False)), write_obsidian=True))


# --- Discoverability: rich descriptions so the model's tool selector reaches
# for the right tool on natural-language prompts WITHOUT needing to load a skill
# first. Empirically (evidence sweep 2026-06-01, 175 runs across 7 model lanes)
# every model that direct-answered Group 1b/1c/1d prompts did so because the
# default tool descriptions were too terse to compete with the model's "I
# know this from training data" reflex. Spelling out the trigger patterns in
# the description itself was the cheapest fix.

SEARCH_DESCRIPTION = (
    "Blazing-fast deterministic recognition for ANYTHING inside the Hermes "
    "source tree, workspace, skills, plugins, configs, docs, Obsidian vault, "
    "runtime logs, memory mirror, or session metadata. Backed by a SQLite "
    "FTS5/trigram index plus ctags: sub-second on cold cache, low-ms warmed. "
    "ALWAYS try this BEFORE search_files / read_file / grep / find / terminal / "
    "skill_view when the question is one of: "
    "(a) WHERE is file/path X (`query: kanban_tools.py`); "
    "(b) WHERE is symbol/class/function X defined (`query: SessionManager`); "
    "(c) WHICH config key controls X (`query: kanban.max_in_progress`); "
    "(d) WHICH tool / skill / plugin handles X (`query: telegram` or `query: Bitwarden, scope: skills`); "
    "(e) FIND the cron / profile / MCP for X (`query: project-fast intelligence index`); "
    "(f) WHAT'S in the Obsidian note about X (`query: Fast Project Intelligence`); "
    "(g) Search the runtime / gateway / agent / error LOGS for X (`query: kanban dispatcher, scope: logs`); "
    "(h) WHAT did I save in my local MEMORY.md / USER.md about X (`query: routing, scope: memory`). "
    "Scopes: auto (default), src, tests, skills, workspace, scripts, plugins, profiles, configs, docs, obsidian, logs, memory, sessions. "
    "For cross-session durable recall ('what do I remember about X', preferences, decisions, identity facts) prefer `mnemosyne_recall` / `mnemosyne_shared_recall` instead. "
    "For 'who calls X' / 'what would break if I change X' prefer `mcp_gitnexus_context` / `mcp_gitnexus_impact`."
)

OPEN_DESCRIPTION = (
    "Open a specific result returned by hermes_source_search with full file "
    "context around the matched line. Use this AFTER a search hit when you "
    "need exact line numbers and surrounding code to answer 'what's the "
    "value of X' or 'what line is X on'. Pass result_id from the search "
    "result. Reads from disk, not from the index; always fresh content."
)

STATUS_DESCRIPTION = (
    "Show index health: scope coverage, row counts, last refresh time, "
    "staleness. Use when search returns unexpectedly empty results or "
    "before deciding whether to call hermes_source_refresh."
)

REFRESH_DESCRIPTION = (
    "Deterministic fast/manual refresh of the source accelerator index "
    "(SQLite FTS5 + trigram). No LLM, no Graphify rebuild, no GitNexus "
    "reindex; those are heavy and intentionally out of the hot path. "
    "Use after writing significant new files or when status reports the "
    "index is stale."
)


SEARCH_SCHEMA={
    "type":"object",
    "description": SEARCH_DESCRIPTION,
    "properties":{
        "query":{"type":"string","description":"REQUIRED. File path, symbol/class/function name, config key, skill name, tool name, cron name, log keyword, memory keyword, or session text. Empty/whitespace queries are rejected fast."},
        "scope":{"type":"string","default":"auto","description":"Comma-separated scopes or 'auto' (default). Valid: src, tests, skills, workspace, scripts, plugins, profiles, configs, docs, obsidian, logs, memory, sessions. Use 'logs' for log searches, 'memory' for MEMORY.md/USER.md, 'sessions' for past session metadata."},
        "mode":{"type":"string","default":"auto","description":"auto, path, symbol, text, logs, memory_session, relationship, architecture. Leave as 'auto' unless you need to force one."},
        "limit":{"type":"integer","default":20,"minimum":1,"maximum":100},
    },
    "required":["query"],
}
OPEN_SCHEMA={
    "type":"object",
    "description": OPEN_DESCRIPTION,
    "properties":{
        "result_id":{"type":"string","description":"REQUIRED. result_id returned by hermes_source_search results[].result_id."},
        "context_lines":{"type":"integer","default":80,"minimum":0,"maximum":500},
    },
    "required":["result_id"],
}
STATUS_SCHEMA={"type":"object","description": STATUS_DESCRIPTION,"properties":{}}
REFRESH_SCHEMA={
    "type":"object",
    "description": REFRESH_DESCRIPTION,
    "properties":{
        "kind":{"type":"string","default":"fast","enum":["fast","manual"]},
        "force":{"type":"boolean","default":False},
    },
}

registry.register(name="hermes_source_search", toolset="source_intel", schema=SEARCH_SCHEMA, handler=_search, check_fn=_check_source_accelerator, description=SEARCH_DESCRIPTION, emoji="⚡", max_result_size_chars=80_000)
registry.register(name="hermes_source_open", toolset="source_intel", schema=OPEN_SCHEMA, handler=_open, check_fn=_check_source_accelerator, description=OPEN_DESCRIPTION, emoji="📖", max_result_size_chars=100_000)
registry.register(name="hermes_source_status", toolset="source_intel", schema=STATUS_SCHEMA, handler=_status, check_fn=_check_source_accelerator, description=STATUS_DESCRIPTION, emoji="🩺", max_result_size_chars=40_000)
registry.register(name="hermes_source_refresh", toolset="source_intel", schema=REFRESH_SCHEMA, handler=_refresh, check_fn=_check_source_accelerator, description=REFRESH_DESCRIPTION, emoji="🔄", max_result_size_chars=80_000)

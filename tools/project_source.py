"""Project Source Accelerator tool wrappers.

These mirror ``hermes_source_*`` but are parameterized by a project root,
allowing the same blazing-fast deterministic recognition layer (SQLite
FTS5/trigram + ctags) to work on ANY user project, not just the Hermes
source tree, without an LLM, Graphify rebuild, or GitNexus full reindex.

Per-project layout (created by ``pintel init``):

    <PROJECT_ROOT>/
      .planning/intelligence/
        indexes/project-index.sqlite     # hot index
        graphify/                        # cold concept graph
        gitnexus/INDEX.md                # corpus pointer
        BENCHMARKS.md / FEATURE_MAP.md / ... # human-readable maps

The wrappers route via the ``project_intelligence`` package shipped with
Hermes (workspace/scripts/project_intelligence/). They accept a
``project_root`` argument or fall back to ``$PINTEL_PROJECT_ROOT`` /
``$HERMES_WORKSPACE_PROJECT``. They share the ``_unwrap_args`` and
``_as_json`` helpers with hermes_source for consistent Vertex-envelope
and JSON-string handling, and they reject empty queries fast.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import json

from tools.registry import registry


# Re-use the helpers from hermes_source so behavior is identical.
def _as_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps({"error": "non-serializable tool result", "repr": repr(value)[:500]})


def _unwrap_args(args: Any) -> dict[str, Any]:
    if isinstance(args, dict) and set(args.keys()) == {"parameters"} and isinstance(args["parameters"], dict):
        return args["parameters"]
    return args if isinstance(args, dict) else {}


_PI_ROOT = Path(__file__).resolve().parents[2] / "workspace" / "scripts" / "project_intelligence"
if str(_PI_ROOT) not in sys.path:
    sys.path.insert(0, str(_PI_ROOT))


def _check_project_intelligence() -> bool:
    return (_PI_ROOT / "project_intelligence" / "query.py").exists()


def _resolve_project_root(args: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve the project root from explicit arg or environment.

    Returns ``(project_root, error_message)``. If ``error_message`` is
    not None, the handler should return it immediately.
    """
    root = (
        args.get("project_root")
        or args.get("project")
        or os.environ.get("PINTEL_PROJECT_ROOT")
        or os.environ.get("HERMES_WORKSPACE_PROJECT")
    )
    if not root:
        return None, _as_json({
            "error": "no project_root",
            "hint": (
                "Pass project_root=<absolute path to project> OR set "
                "$PINTEL_PROJECT_ROOT in the agent's environment. The "
                "project must already have been bootstrapped with "
                "`pintel --project <root> init` (creates "
                ".planning/intelligence/ with the hot index)."
            ),
            "examples": [
                {"project_root": "/home/ndsadmin/dev/dashboard", "query": "AccountController"},
                {"project_root": "/mnt/devvm/projects/<svc>", "query": "init_project"},
            ],
        })
    root = str(Path(root).expanduser().resolve())
    if not Path(root).is_dir():
        return None, _as_json({"error": "project_root does not exist", "project_root": root})
    return root, None


def _cfg(root: str, slug: str | None = None):
    """Load the per-project ProjectConfig (cheap; just reads filesystem)."""
    from project_intelligence.config import ProjectConfig
    return ProjectConfig.discover(root, slug=slug)


def _search(args: dict[str, Any], **kw):
    from project_intelligence.query import search
    args = _unwrap_args(args)
    query = str(args.get("query", "")).strip()
    if not query:
        return _as_json({
            "error": "empty query",
            "hint": (
                "Pass query=<file path | symbol | config key | route | "
                "feature keyword | doc heading>. For empty/whitespace "
                "queries we fast-fail so you don't accidentally trigger "
                "a match-all scan."
            ),
            "examples": [
                {"project_root": "/home/ndsadmin/dev/dashboard", "query": "AccountController"},
                {"project_root": "/home/ndsadmin/dev/dashboard", "query": "signin", "scope": "src"},
            ],
        })
    root, err = _resolve_project_root(args)
    if err:
        return err
    cfg = _cfg(root, slug=args.get("slug"))
    return _as_json(search(
        cfg,
        query=query,
        scope=str(args.get("scope", "auto")),
        limit=int(args.get("limit", 20)),
    ))


def _open(args: dict[str, Any], **kw):
    from project_intelligence.query import open_result
    args = _unwrap_args(args)
    rid = str(args.get("result_id", "")).strip()
    if not rid:
        return _as_json({"error": "empty result_id", "hint": "pass result_id=<id from project_source_search>"})
    root, err = _resolve_project_root(args)
    if err:
        return err
    cfg = _cfg(root, slug=args.get("slug"))
    return _as_json(open_result(
        cfg,
        result_id=rid,
        context_lines=int(args.get("context_lines", args.get("context", 80))),
    ))


def _status(args: dict[str, Any], **kw):
    from project_intelligence.status import status
    args = _unwrap_args(args)
    root, err = _resolve_project_root(args)
    if err:
        return err
    cfg = _cfg(root, slug=args.get("slug"))
    return _as_json(status(cfg))


def _refresh(args: dict[str, Any], **kw):
    from project_intelligence.indexer import refresh_fast
    args = _unwrap_args(args)
    root, err = _resolve_project_root(args)
    if err:
        return err
    cfg = _cfg(root, slug=args.get("slug"))
    return _as_json(refresh_fast(cfg, force=bool(args.get("force", False))))


# ---- Descriptions ----------------------------------------------------------

SEARCH_DESCRIPTION = (
    "Blazing-fast deterministic recognition for ANYTHING inside a user "
    "project (NOT the Hermes source; for that use hermes_source_search). "
    "Backed by a per-project SQLite FTS5/trigram hot index under "
    "<PROJECT_ROOT>/.planning/intelligence/indexes/project-index.sqlite. "
    "ALWAYS try this BEFORE search_files / read_file / grep / find / "
    "terminal when the question is about a project file, symbol, config "
    "key, route, feature, doc, log, or session. Pass project_root=<abs "
    "path> (or set $PINTEL_PROJECT_ROOT). Examples: "
    "(a) `project_root: /home/ndsadmin/dev/dashboard, query: AccountController` (find a class); "
    "(b) `project_root: /mnt/devvm/projects/<svc>, query: signin, scope: src` (find a feature); "
    "(c) `project_root: <root>, query: API_KEY, scope: configs` (find a config key); "
    "(d) `project_root: <root>, query: ERROR, scope: logs` (search runtime logs). "
    "Scopes: auto (default), src, tests, scripts, configs, docs, routes, "
    "logs, sessions. For cross-project / Hermes-internal lookups use "
    "hermes_source_search. For 'who calls X' / 'what would break' use "
    "mcp_gitnexus_context / mcp_gitnexus_impact with --repo <project-slug>. "
    "Project must be bootstrapped via `pintel --project <root> init` first."
)

OPEN_DESCRIPTION = (
    "Open a specific result returned by project_source_search with full "
    "file context around the matched line. Pass project_root + result_id."
)

STATUS_DESCRIPTION = (
    "Show per-project hot-index health: scope coverage, row counts, "
    "last refresh time, staleness. Pass project_root."
)

REFRESH_DESCRIPTION = (
    "Deterministic incremental refresh of the per-project hot index. "
    "No LLM, no Graphify rebuild, no GitNexus reindex. Pass project_root."
)


# ---- Schemas ---------------------------------------------------------------

_PROJECT_ROOT_PROP = {
    "type": "string",
    "description": (
        "REQUIRED unless $PINTEL_PROJECT_ROOT is set. Absolute path to "
        "the project root (the directory that contains .planning/intelligence/ "
        "after `pintel init`)."
    ),
}

SEARCH_SCHEMA = {
    "type": "object",
    "description": SEARCH_DESCRIPTION,
    "properties": {
        "project_root": _PROJECT_ROOT_PROP,
        "query": {"type": "string", "description": "REQUIRED. File path, symbol, config key, route, feature keyword, doc heading, log keyword, or session text."},
        "scope": {"type": "string", "default": "auto", "description": "Comma-separated scopes or 'auto'. Valid: src, tests, scripts, configs, docs, routes, logs, sessions."},
        "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
        "slug": {"type": "string", "description": "Optional project slug (defaults to directory name)."},
    },
    "required": ["query"],
}

OPEN_SCHEMA = {
    "type": "object",
    "description": OPEN_DESCRIPTION,
    "properties": {
        "project_root": _PROJECT_ROOT_PROP,
        "result_id": {"type": "string", "description": "REQUIRED. result_id returned by project_source_search."},
        "context_lines": {"type": "integer", "default": 80, "minimum": 0, "maximum": 500},
        "slug": {"type": "string"},
    },
    "required": ["result_id"],
}

STATUS_SCHEMA = {
    "type": "object",
    "description": STATUS_DESCRIPTION,
    "properties": {
        "project_root": _PROJECT_ROOT_PROP,
        "slug": {"type": "string"},
    },
}

REFRESH_SCHEMA = {
    "type": "object",
    "description": REFRESH_DESCRIPTION,
    "properties": {
        "project_root": _PROJECT_ROOT_PROP,
        "force": {"type": "boolean", "default": False},
        "slug": {"type": "string"},
    },
}

registry.register(name="project_source_search", toolset="source_intel", schema=SEARCH_SCHEMA, handler=_search, check_fn=_check_project_intelligence, description=SEARCH_DESCRIPTION, emoji="🛰️", max_result_size_chars=80_000)
registry.register(name="project_source_open", toolset="source_intel", schema=OPEN_SCHEMA, handler=_open, check_fn=_check_project_intelligence, description=OPEN_DESCRIPTION, emoji="📂", max_result_size_chars=100_000)
registry.register(name="project_source_status", toolset="source_intel", schema=STATUS_SCHEMA, handler=_status, check_fn=_check_project_intelligence, description=STATUS_DESCRIPTION, emoji="📡", max_result_size_chars=40_000)
registry.register(name="project_source_refresh", toolset="source_intel", schema=REFRESH_SCHEMA, handler=_refresh, check_fn=_check_project_intelligence, description=REFRESH_DESCRIPTION, emoji="🔁", max_result_size_chars=80_000)

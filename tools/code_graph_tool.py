from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.code_graph.indexer import index_repo
from tools.code_graph.query import (
    context_for_goal,
    graph_status,
    impact_for_paths,
    neighbors_for_symbol,
    search_symbols,
    symbol_detail,
)
from tools.registry import registry


def _json(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _root(root: str | None) -> Path:
    path = Path(root or ".").resolve()
    if not path.exists():
        raise ValueError(f"root does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"root is not a directory: {path}")
    return path


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def code_graph_index(
    root: str | None = None,
    force: bool = False,
    max_file_size_bytes: int = 512_000,
    task_id: str | None = None,
) -> str:
    try:
        return _json(
            index_repo(
                _root(root),
                force=bool(force),
                max_file_size_bytes=int(max_file_size_bytes or 512_000),
            )
        )
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def code_graph_status(root: str | None = None, task_id: str | None = None) -> str:
    try:
        return _json(graph_status(_root(root)))
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def code_graph_search(
    query: str,
    root: str | None = None,
    kind: str = "symbol",
    limit: int = 20,
    task_id: str | None = None,
) -> str:
    try:
        if kind not in {"symbol", "all"}:
            return _json({"success": False, "error": f"unsupported search kind for MVP: {kind}"})
        return _json(search_symbols(_root(root), query, limit=int(limit or 20)))
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def code_graph_symbol(
    symbol: str,
    root: str | None = None,
    task_id: str | None = None,
) -> str:
    try:
        return _json(symbol_detail(_root(root), symbol))
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def code_graph_neighbors(
    symbol: str,
    root: str | None = None,
    limit: int = 50,
    task_id: str | None = None,
) -> str:
    try:
        return _json(neighbors_for_symbol(_root(root), symbol, limit=int(limit or 50)))
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def code_graph_impact(
    paths: list[str] | str,
    root: str | None = None,
    limit: int = 50,
    task_id: str | None = None,
) -> str:
    try:
        return _json(impact_for_paths(_root(root), _as_list(paths), limit=int(limit or 50)))
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


def code_graph_context(
    goal: str,
    root: str | None = None,
    budget_chars: int = 20_000,
    task_id: str | None = None,
) -> str:
    try:
        return _json(context_for_goal(_root(root), goal, budget_chars=int(budget_chars or 20_000)))
    except Exception as exc:
        return _json({"success": False, "error": str(exc)})


_ROOT_PARAM = {
    "type": "string",
    "description": "Repository root. Defaults to the current workspace.",
}


registry.register(
    name="code_graph_index",
    toolset="code_graph",
    schema={
        "name": "code_graph_index",
        "description": (
            "Index a repository into Hermes' local read-only code graph cache. "
            "This reads repository files and writes only the profile cache, not the repository."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "root": _ROOT_PARAM,
                "force": {
                    "type": "boolean",
                    "description": "Reindex files even when their hashes are unchanged.",
                    "default": False,
                },
                "max_file_size_bytes": {
                    "type": "integer",
                    "description": "Skip source files larger than this many bytes.",
                    "default": 512000,
                },
            },
        },
    },
    handler=lambda args, **kw: code_graph_index(
        args.get("root"),
        bool(args.get("force", False)),
        int(args.get("max_file_size_bytes", 512_000) or 512_000),
        kw.get("task_id"),
    ),
    max_result_size_chars=50_000,
)

registry.register(
    name="code_graph_status",
    toolset="code_graph",
    schema={
        "name": "code_graph_status",
        "description": "Report whether the local code graph is missing, fresh, or stale for a repository.",
        "parameters": {
            "type": "object",
            "properties": {"root": _ROOT_PARAM},
        },
    },
    handler=lambda args, **kw: code_graph_status(args.get("root"), kw.get("task_id")),
    max_result_size_chars=20_000,
)

registry.register(
    name="code_graph_search",
    toolset="code_graph",
    schema={
        "name": "code_graph_search",
        "description": "Search indexed code graph symbols by name or qualified name.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symbol query."},
                "root": _ROOT_PARAM,
                "kind": {
                    "type": "string",
                    "enum": ["symbol", "all"],
                    "description": "MVP supports symbol search; all is accepted as an alias.",
                    "default": "symbol",
                },
                "limit": {"type": "integer", "description": "Maximum matches.", "default": 20},
            },
            "required": ["query"],
        },
    },
    handler=lambda args, **kw: code_graph_search(
        args.get("query", ""),
        args.get("root"),
        args.get("kind", "symbol"),
        int(args.get("limit", 20) or 20),
        kw.get("task_id"),
    ),
    max_result_size_chars=50_000,
)

registry.register(
    name="code_graph_symbol",
    toolset="code_graph",
    schema={
        "name": "code_graph_symbol",
        "description": "Inspect one indexed symbol's definition location and metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name or qualified name."},
                "root": _ROOT_PARAM,
            },
            "required": ["symbol"],
        },
    },
    handler=lambda args, **kw: code_graph_symbol(
        args.get("symbol", ""),
        args.get("root"),
        kw.get("task_id"),
    ),
    max_result_size_chars=30_000,
)

registry.register(
    name="code_graph_neighbors",
    toolset="code_graph",
    schema={
        "name": "code_graph_neighbors",
        "description": "Return imports and textual references near an indexed symbol.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Symbol name or qualified name."},
                "root": _ROOT_PARAM,
                "limit": {"type": "integer", "description": "Maximum neighbor entries.", "default": 50},
            },
            "required": ["symbol"],
        },
    },
    handler=lambda args, **kw: code_graph_neighbors(
        args.get("symbol", ""),
        args.get("root"),
        int(args.get("limit", 50) or 50),
        kw.get("task_id"),
    ),
    max_result_size_chars=50_000,
)

registry.register(
    name="code_graph_impact",
    toolset="code_graph",
    schema={
        "name": "code_graph_impact",
        "description": "Estimate directly impacted symbols, imports, references, and likely tests for paths.",
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Repository-relative paths to inspect.",
                },
                "root": _ROOT_PARAM,
                "limit": {"type": "integer", "description": "Maximum entries per section.", "default": 50},
            },
            "required": ["paths"],
        },
    },
    handler=lambda args, **kw: code_graph_impact(
        args.get("paths", []),
        args.get("root"),
        int(args.get("limit", 50) or 50),
        kw.get("task_id"),
    ),
    max_result_size_chars=50_000,
)

registry.register(
    name="code_graph_context",
    toolset="code_graph",
    schema={
        "name": "code_graph_context",
        "description": "Build a compact, ranked code graph context bundle for an implementation goal.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "Task or implementation goal."},
                "root": _ROOT_PARAM,
                "budget_chars": {
                    "type": "integer",
                    "description": "Approximate maximum JSON payload size.",
                    "default": 20000,
                },
            },
            "required": ["goal"],
        },
    },
    handler=lambda args, **kw: code_graph_context(
        args.get("goal", ""),
        args.get("root"),
        int(args.get("budget_chars", 20_000) or 20_000),
        kw.get("task_id"),
    ),
    max_result_size_chars=80_000,
)


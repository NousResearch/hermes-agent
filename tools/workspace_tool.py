#!/usr/bin/env python3
"""Workspace tool — inspect and search the Hermes workspace."""

from __future__ import annotations

import json
from typing import Any

from agent.workspace import (
    index_workspace_knowledgebase,
    workspace_list,
    workspace_retrieve,
    workspace_search,
    workspace_status,
)
from hermes_cli.config import load_config
from tools.registry import registry


WORKSPACE_SCHEMA = {
    "name": "workspace",
    "description": "Manage the Hermes workspace under HERMES_HOME. Use this to inspect workspace status, rebuild the workspace manifest, list files, or search within workspace documents without relying on the terminal environment.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "index", "list", "search", "retrieve"],
                "description": "What to do: status shows roots and counts, index rebuilds the manifest and chunk index, list enumerates files, search searches text lines, retrieve returns ranked chunk-level retrieval results.",
            },
            "query": {
                "type": "string",
                "description": "Regex query to search for when action='search'.",
            },
            "path": {
                "type": "string",
                "description": "Optional subpath within the workspace to scope list/search operations.",
            },
            "file_glob": {
                "type": "string",
                "description": "Optional filename glob filter for search, e.g. '*.md'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries or matches to return.",
                "default": 20,
            },
            "offset": {
                "type": "integer",
                "description": "Skip the first N entries or matches.",
                "default": 0,
            },
            "recursive": {
                "type": "boolean",
                "description": "When action='list', recurse through subdirectories (default true).",
                "default": True,
            },
        },
        "required": ["action"],
    },
}


def workspace_tool(
    action: str,
    query: str = "",
    path: str = "",
    file_glob: str | None = None,
    limit: int = 20,
    offset: int = 0,
    recursive: bool = True,
) -> str:
    try:
        config = load_config()
        if action == "status":
            result: dict[str, Any] = workspace_status(config)
        elif action == "index":
            result = index_workspace_knowledgebase(config)
        elif action == "list":
            result = workspace_list(
                config=config,
                relative_path=path,
                recursive=recursive,
                limit=limit,
                offset=offset,
            )
        elif action == "search":
            result = workspace_search(
                query=query,
                config=config,
                relative_path=path,
                file_glob=file_glob,
                limit=limit,
                offset=offset,
            )
        elif action == "retrieve":
            result = workspace_retrieve(
                query=query,
                config=config,
                limit=limit,
            )
        else:
            result = {"success": False, "error": f"Unknown action: {action}"}
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:  # pragma: no cover - defensive wrapper
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False)


registry.register(
    name="workspace",
    toolset="workspace",
    schema=WORKSPACE_SCHEMA,
    handler=lambda args, **kw: workspace_tool(
        action=args.get("action", ""),
        query=args.get("query", ""),
        path=args.get("path", ""),
        file_glob=args.get("file_glob"),
        limit=args.get("limit", 20),
        offset=args.get("offset", 0),
        recursive=args.get("recursive", True),
    ),
    check_fn=lambda: True,
)

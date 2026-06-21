#!/usr/bin/env python3
"""Change the agent's working directory (workspace) in the Hermes desktop/TUI.

The desktop left-sidebar groups sessions by their working directory (cwd).
The ``set_workspace`` tool rebinds the RUNNING session's cwd so it moves
under a different workspace folder in the sidebar.
"""

import json
import os
from typing import Callable, Optional

from tools.registry import registry, tool_error


def set_workspace_tool(
    cwd: Optional[str] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Change the working directory (workspace) for this agent session.

    The desktop left-sidebar groups sessions by cwd. This tool rebinds the
    session's cwd so the desktop renderer re-groups it under the new workspace.
    Requires an absolute or ~-relative path to an existing directory.
    """
    if callback is None:
        return tool_error("set_workspace is only available in the Hermes desktop/TUI app.")

    if not cwd or not cwd.strip():
        return tool_error("cwd (target directory) is required.")

    try:
        result = callback(cwd=cwd.strip())
    except Exception as exc:
        return tool_error(f"Failed to change workspace: {exc}")

    if not result:
        return tool_error("Failed to change workspace, or it timed out.")

    if isinstance(result, dict) and result.get("error"):
        return tool_error(result["error"])

    # Success: result is {cwd: resolved_path, branch: git_branch_or_empty}.
    return json.dumps(result, ensure_ascii=False)


def check_workspace_requirements() -> bool:
    """Desktop GUI / TUI only — HERMES_DESKTOP is set on the gateway."""
    return (os.getenv("HERMES_DESKTOP") or "").strip().lower() in ("1", "true", "yes")


SET_WORKSPACE_SCHEMA = {
    "name": "set_workspace",
    "description": (
        "Change THIS session's working directory / workspace. The desktop left "
        "sidebar re-groups the session under the new workspace folder. Pass an "
        "absolute or ~ path to an existing directory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Absolute or ~-relative path to an existing directory to bind this session to.",
            },
        },
        "required": ["cwd"],
    },
}


registry.register(
    name="set_workspace",
    toolset="file",
    schema=SET_WORKSPACE_SCHEMA,
    handler=lambda args, **kw: set_workspace_tool(
        cwd=args.get("cwd"),
        callback=kw.get("callback"),
    ),
    check_fn=check_workspace_requirements,
    emoji="📁",
)

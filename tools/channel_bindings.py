"""Channel-to-repo binding manager for Hermes Agent.

Provides a JSON-file store mapping channel IDs to GitHub repo configurations.
Used by the /bind slash command in the gateway and as a tool for the agent
to inspect or modify bindings during conversations.

Storage location: {HERMES_HOME}/channel_bindings.json
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home


_BINDINGS_FILE: Optional[Path] = None
_lock = threading.Lock()


def _get_bindings_path() -> Path:
    global _BINDINGS_FILE
    if _BINDINGS_FILE is None:
        _BINDINGS_FILE = get_hermes_home() / "channel_bindings.json"
    return _BINDINGS_FILE


def _load_bindings() -> Dict[str, Any]:
    path = _get_bindings_path()
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"bindings": {}}
    return {"bindings": {}}


def _save_bindings(data: Dict[str, Any]) -> None:
    path = _get_bindings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(path)


def get_binding(channel_id: str) -> Optional[Dict[str, Any]]:
    data = _load_bindings()
    return data.get("bindings", {}).get(channel_id)


def set_binding(
    channel_id: str,
    repo_url: str,
    repo_name: str,
    workspace_path: str,
    branch: str = "main",
) -> Dict[str, Any]:
    with _lock:
        data = _load_bindings()
        now = datetime.now(timezone.utc).isoformat()
        bindings = data.setdefault("bindings", {})

        if channel_id in bindings:
            bindings[channel_id].update(
                {
                    "repo_url": repo_url,
                    "repo_name": repo_name,
                    "workspace_path": workspace_path,
                    "branch": branch,
                    "updated_at": now,
                }
            )
        else:
            bindings[channel_id] = {
                "repo_url": repo_url,
                "repo_name": repo_name,
                "workspace_path": workspace_path,
                "branch": branch,
                "created_at": now,
                "updated_at": now,
            }

        _save_bindings(data)
        return bindings[channel_id]


def remove_binding(channel_id: str) -> bool:
    with _lock:
        data = _load_bindings()
        bindings = data.get("bindings", {})
        if channel_id in bindings:
            del bindings[channel_id]
            _save_bindings(data)
            return True
        return False


def list_all_bindings() -> Dict[str, Any]:
    return _load_bindings()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------
from tools.registry import registry


def _tool_get_channel_binding(channel_id: str = None, task_id: str = None) -> str:
    if not channel_id:
        return json.dumps({"error": "channel_id is required"})
    binding = get_binding(channel_id)
    if binding:
        return json.dumps({"found": True, "binding": binding})
    return json.dumps({"found": False, "message": "No binding for this channel"})


def _tool_set_channel_binding(
    channel_id: str = None,
    repo_url: str = None,
    repo_name: str = None,
    workspace_path: str = None,
    branch: str = "main",
    task_id: str = None,
) -> str:
    if not all([channel_id, repo_url, repo_name, workspace_path]):
        return json.dumps(
            {
                "error": "channel_id, repo_url, repo_name, and workspace_path are required"
            }
        )
    binding = set_binding(channel_id, repo_url, repo_name, workspace_path, branch)
    return json.dumps({"success": True, "binding": binding})


def _tool_remove_channel_binding(channel_id: str = None, task_id: str = None) -> str:
    if not channel_id:
        return json.dumps({"error": "channel_id is required"})
    if remove_binding(channel_id):
        return json.dumps({"success": True, "message": "Binding removed"})
    return json.dumps(
        {"success": False, "message": "No binding found for this channel"}
    )


def _tool_list_channel_bindings(task_id: str = None) -> str:
    data = list_all_bindings()
    bindings = data.get("bindings", {})
    return json.dumps({"bindings": bindings, "count": len(bindings)})


registry.register(
    name="get_channel_binding",
    toolset="channel-bindings",
    schema={
        "name": "get_channel_binding",
        "description": "Get the GitHub repo binding for a Discord channel. Returns the repo URL, name, workspace path, and branch if a binding exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The Discord channel ID to look up",
                },
            },
            "required": ["channel_id"],
        },
    },
    handler=lambda args, **kw: _tool_get_channel_binding(
        channel_id=args.get("channel_id"),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="set_channel_binding",
    toolset="channel-bindings",
    schema={
        "name": "set_channel_binding",
        "description": "Bind a Discord channel to a GitHub repo. Stores the repo URL, name, workspace path, and branch for future reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The Discord channel ID",
                },
                "repo_url": {"type": "string", "description": "Full GitHub repo URL"},
                "repo_name": {
                    "type": "string",
                    "description": "Repo name in owner/repo format",
                },
                "workspace_path": {
                    "type": "string",
                    "description": "Local path where the repo is/will be cloned",
                },
                "branch": {
                    "type": "string",
                    "description": "Git branch to use (default: main)",
                },
            },
            "required": ["channel_id", "repo_url", "repo_name", "workspace_path"],
        },
    },
    handler=lambda args, **kw: _tool_set_channel_binding(
        channel_id=args.get("channel_id"),
        repo_url=args.get("repo_url"),
        repo_name=args.get("repo_name"),
        workspace_path=args.get("workspace_path"),
        branch=args.get("branch", "main"),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="remove_channel_binding",
    toolset="channel-bindings",
    schema={
        "name": "remove_channel_binding",
        "description": "Remove the GitHub repo binding for a Discord channel.",
        "parameters": {
            "type": "object",
            "properties": {
                "channel_id": {
                    "type": "string",
                    "description": "The Discord channel ID",
                },
            },
            "required": ["channel_id"],
        },
    },
    handler=lambda args, **kw: _tool_remove_channel_binding(
        channel_id=args.get("channel_id"),
        task_id=kw.get("task_id"),
    ),
)

registry.register(
    name="list_channel_bindings",
    toolset="channel-bindings",
    schema={
        "name": "list_channel_bindings",
        "description": "List all channel-to-GitHub-repo bindings.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    handler=lambda args, **kw: _tool_list_channel_bindings(
        task_id=kw.get("task_id"),
    ),
)

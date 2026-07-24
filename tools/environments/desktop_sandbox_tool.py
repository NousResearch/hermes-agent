"""Status and lifecycle tool for the task-scoped desktop sandbox."""

from __future__ import annotations

import json
from typing import Any

from tools.environments.desktop_lease import get_desktop_sandbox_manager
from tools.registry import registry


DESKTOP_SANDBOX_SCHEMA = {
    "name": "desktop_sandbox",
    "description": "Inspect or release the shared desktop sandbox for this task.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "release"],
                "description": "Use status to inspect the current task lease or release to end it.",
            },
        },
        "required": ["action"],
    },
}


def handle_desktop_sandbox(args: dict[str, Any], **kwargs: Any) -> str:
    task_id = str(kwargs.get("task_id") or "default")
    action = str(args.get("action") or "status")
    manager = get_desktop_sandbox_manager()
    if action == "release":
        manager.release(task_id)
        return json.dumps({"released": task_id})
    if action == "status":
        return json.dumps(manager.status(task_id))
    return json.dumps({"error": f"unknown desktop_sandbox action: {action}"})


registry.register(
    name="desktop_sandbox",
    toolset="desktop_sandbox",
    schema=DESKTOP_SANDBOX_SCHEMA,
    handler=handle_desktop_sandbox,
    check_fn=lambda: True,
    emoji="🖥️",
)

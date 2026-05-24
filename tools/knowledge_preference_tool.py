"""Knowledge preference management tool.

Tool name: manage_knowledge_preference
"""

from __future__ import annotations

import json
from typing import Optional

from agent.knowledge_preferences import KnowledgePreferenceManager
from tools.registry import registry


def manage_knowledge_preference(
    action: str,
    domain: Optional[str] = None,
    project: Optional[str] = None,
    pattern: Optional[str] = None,
    allow: Optional[bool] = None,
    reason: Optional[str] = None,
    pref_id: Optional[str] = None,
    task_id: str = None,
) -> str:
    """Manage knowledge promotion preferences.

    Args:
        action: One of 'save', 'list', 'delete'
        domain: Domain slug (for save)
        project: Project slug (for save), '*' for any project
        pattern: Keyword pattern to match (for save)
        allow: True to auto-promote, False to auto-deny (for save)
        reason: Optional reason (for save)
        pref_id: Preference ID (for delete)
        task_id: Optional task identifier

    Returns:
        JSON string with result
    """
    mgr = KnowledgePreferenceManager()

    if action == "save":
        if not all([domain, project, pattern, allow is not None]):
            return json.dumps({
                "success": False,
                "error": "save requires domain, project, pattern, allow",
            })
        pref_id = mgr.save_preference(
            domain=domain,
            project=project,
            pattern=pattern,
            allow=allow,
            reason=reason or "",
        )
        return json.dumps({
            "success": True,
            "action": "save",
            "pref_id": pref_id,
        })

    elif action == "list":
        prefs = mgr.list_preferences()
        return json.dumps({
            "success": True,
            "action": "list",
            "count": len(prefs),
            "preferences": prefs,
        })

    elif action == "delete":
        if not pref_id:
            return json.dumps({
                "success": False,
                "error": "delete requires pref_id",
            })
        deleted = mgr.delete_preference(pref_id)
        if deleted:
            return json.dumps({
                "success": True,
                "action": "delete",
                "pref_id": pref_id,
            })
        return json.dumps({
            "success": False,
            "error": f"Preference ID {pref_id} not found",
        })

    else:
        return json.dumps({
            "success": False,
            "error": f"Invalid action: {action}. Valid: save, list, delete",
        })


def check_requirements() -> bool:
    """Always available — creates preference file on first save."""
    return True


# Register the tool
registry.register(
    name="manage_knowledge_preference",
    toolset="knowledge",
    schema={
        "name": "manage_knowledge_preference",
        "description": (
            "Manage knowledge promotion preferences. "
            "Save preferences to auto-promote or auto-deny knowledge based on domain/pattern. "
            "List all preferences. Delete by ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["save", "list", "delete"],
                    "description": "Action to perform",
                },
                "domain": {"type": "string", "description": "Domain slug (for save)"},
                "project": {"type": "string", "description": "Project slug or '*' for any (for save)"},
                "pattern": {"type": "string", "description": "Keyword pattern (for save)"},
                "allow": {"type": "boolean", "description": "Auto-promote (true) or auto-deny (false)"},
                "reason": {"type": "string", "description": "Optional reason"},
                "pref_id": {"type": "string", "description": "Preference ID (for delete)"},
            },
            "required": ["action"],
        },
    },
    handler=lambda args, **kw: manage_knowledge_preference(
        action=args.get("action", "list"),
        domain=args.get("domain"),
        project=args.get("project"),
        pattern=args.get("pattern"),
        allow=args.get("allow"),
        reason=args.get("reason"),
        pref_id=args.get("pref_id"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_requirements,
)

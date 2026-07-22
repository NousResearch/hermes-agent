"""UX launcher/shortcut tool.

Provides a compact menu of safe, high-value Hermes entry points for
productivity/UX so an agent/human can quickly choose the next action.

Registered tool name: ``ux_launcher``

Success envelope: zodern:relay-style ``{kind: success, data: {...}}``
Failure envelope: zodern:relay-style ``{kind: failure, error, code}``
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.validation_helper import relay_failure, relay_success

NAME = "ux_launcher"

MENU_ITEMS = [
    {
        "id": "hermes_tools",
        "command": "hermes tools",
        "category": "dev",
        "guidance": "Open the Hermes tool surface so you can inspect or run tooling.",
    },
    {
        "id": "hermes_tools_list",
        "command": "hermes tools list",
        "category": "dev",
        "guidance": "List available tools; use when onboarding or checking for a capability signature.",
    },
    {
        "id": "cron_list",
        "command": "hermes cron list",
        "category": "ops",
        "guidance": "Inspect scheduled jobs; use when validating background automation health.",
    },
    {
        "id": "platform_connectivity_check",
        "command": "platform connectivity check",
        "category": "messaging",
        "guidance": "Verify messaging/platform route availability before sending or polling.",
    },
    {
        "id": "repo_health",
        "command": "repo health",
        "category": "workspace",
        "guidance": "Run repository-wide hygiene checks: dirty trees, large files, branch state.",
    },
    {
        "id": "test_health",
        "command": "test health",
        "category": "workspace",
        "guidance": "Run a lightweight tests smoke check; catches breakage before root-cause work.",
    },
    {
        "id": "docker_health",
        "command": "docker health",
        "category": "ops",
        "guidance": "Check local Docker containers/services; useful before platform-dependent tasks.",
    },
    {
        "id": "config_validation",
        "command": "config validation",
        "category": "docs",
        "guidance": "Validate Hermes/plugin config files so bad YAML/JSON does not break runtime.",
    },
    {
        "id": "session_search",
        "command": "session search",
        "category": "docs",
        "guidance": "Search local session history for prior decisions, issues, or references.",
    },
    {
        "id": "todo_list_status",
        "command": "todo list/status",
        "category": "workspace",
        "guidance": "Surface an in-session lightweight todo status when shifting context.",
    },
]


def _parse_filter(args: Dict[str, Any]) -> List[str]:
    """Return normalized lowercased filter keywords.

    Accepts ``filter``/``filters``/``category`` as a list or comma string.
    """
    raw = (
        args.get("filters")
        or args.get("filter")
        or args.get("category")
        or ""
    )
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, str) and raw.strip():
        items = [segment.strip() for segment in raw.split(",")]
    else:
        items = []

    allowed = {"workspace", "dev", "ops", "docs", "messaging"}
    normalized = []
    for item in items:
        token = item.lower()
        if token and token in allowed:
            normalized.append(token)
    return list(dict.fromkeys(normalized))


def check_fn(args: Dict[str, Any]) -> bool:
    """Always return True; this is a UI/UX guidance surface."""
    return True


def run(args: Dict[str, Any]) -> Dict[str, Any]:
    """Build the UX launcher menu and return a relay-style envelope."""
    if not isinstance(args, dict):
        return relay_failure("args must be a mapping", code="invalid_args")

    keywords = _parse_filter(args)

    full_menu = [
        {
            "id": item["id"],
            "command": item["command"],
            "category": item["category"],
            "guidance": item["guidance"],
        }
        for item in MENU_ITEMS
    ]

    if not keywords:
        menu = full_menu
    else:
        seen = set()
        menu = []
        for item in full_menu:
            if item["category"] in keywords or item["id"] in keywords:
                if item["id"] not in seen:
                    menu.append(item)
                    seen.add(item["id"])

    payload = {
        "tool": NAME,
        "filter_keywords": keywords,
        "total_entries": len(MENU_ITEMS),
        "returned_entries": len(menu),
        "menu": menu,
        "full_menu": full_menu,
    }
    return relay_success(payload)


__all__ = ["NAME", "MENU_ITEMS", "check_fn", "run"]

"""Slack usergroup management tools for Hermes gateway/local runtime.

Uses the same SLACK_BOT_TOKEN that the Hermes Slack gateway uses. The update
endpoint replaces the complete usergroup membership, so helpers reject empty
membership and expose read tools to inspect current state first.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, List

from tools.registry import registry

SLACK_API_BASE = "https://slack.com/api"


def check_slack_usergroup_requirements() -> bool:
    return bool(os.getenv("SLACK_BOT_TOKEN", "").strip())


def _token() -> str:
    token = os.getenv("SLACK_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("SLACK_BOT_TOKEN is not set")
    return token


def _normalize_users(users: Any) -> List[str]:
    if isinstance(users, str):
        parts = users.split(",")
    elif isinstance(users, Iterable):
        parts = [str(user) for user in users]
    else:
        parts = []

    normalized: List[str] = []
    seen = set()
    for part in parts:
        user_id = str(part).strip()
        if not user_id or user_id in seen:
            continue
        normalized.append(user_id)
        seen.add(user_id)

    if not normalized:
        raise ValueError("At least one Slack user ID is required; refusing to empty a usergroup")
    return normalized


def _slack_api(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    body = urllib.parse.urlencode(
        {key: value for key, value in params.items() if value is not None}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{SLACK_API_BASE}/{method}",
        data=body,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(f"Slack API error on {method}: {payload.get('error', 'unknown')}")
    return payload


def slack_list_usergroups(include_users: bool = False, include_disabled: bool = False) -> str:
    data = _slack_api(
        "usergroups.list",
        {
            "include_users": "true" if include_users else "false",
            "include_disabled": "true" if include_disabled else "false",
        },
    )
    groups = []
    for group in data.get("usergroups", []):
        groups.append(
            {
                "id": group.get("id"),
                "handle": group.get("handle"),
                "name": group.get("name"),
                "description": group.get("description"),
                "is_disabled": group.get("date_delete") not in (None, 0),
                "users": group.get("users") if include_users else None,
            }
        )
    return json.dumps({"ok": True, "usergroups": groups}, ensure_ascii=False)


def slack_list_usergroup_users(usergroup: str) -> str:
    data = _slack_api("usergroups.users.list", {"usergroup": usergroup.strip()})
    return json.dumps(
        {"ok": True, "usergroup": usergroup.strip(), "users": data.get("users", [])},
        ensure_ascii=False,
    )


def slack_update_usergroup_users(usergroup: str, users: Any) -> str:
    usergroup = usergroup.strip()
    normalized_users = _normalize_users(users)
    data = _slack_api(
        "usergroups.users.update",
        {"usergroup": usergroup, "users": ",".join(normalized_users)},
    )
    return json.dumps(
        {
            "ok": True,
            "usergroup": usergroup,
            "users": normalized_users,
            "slack_response": {"usergroup": data.get("usergroup", {})},
        },
        ensure_ascii=False,
    )


SLACK_LIST_USERGROUPS_SCHEMA = {
    "name": "slack_list_usergroups",
    "description": "List Slack usergroups/subteams using Hermes' SLACK_BOT_TOKEN. Use this to find a usergroup ID/handle before updating membership.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_users": {"type": "boolean", "default": False},
            "include_disabled": {"type": "boolean", "default": False},
        },
    },
}

SLACK_LIST_USERGROUP_USERS_SCHEMA = {
    "name": "slack_list_usergroup_users",
    "description": "List current member user IDs of a Slack usergroup/subteam. Always call this before slack_update_usergroup_users because Slack update replaces complete membership.",
    "parameters": {
        "type": "object",
        "properties": {
            "usergroup": {"type": "string", "description": "Slack usergroup ID, e.g. S0123456789"},
        },
        "required": ["usergroup"],
    },
}

SLACK_UPDATE_USERGROUP_USERS_SCHEMA = {
    "name": "slack_update_usergroup_users",
    "description": "Replace the complete membership of a Slack usergroup/subteam using Hermes' SLACK_BOT_TOKEN. Requires usergroups:write. Provide the full final member list; empty lists are rejected.",
    "parameters": {
        "type": "object",
        "properties": {
            "usergroup": {"type": "string", "description": "Slack usergroup ID, e.g. S0123456789"},
            "users": {
                "oneOf": [
                    {"type": "string", "description": "Comma-separated Slack user IDs"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Complete final membership as Slack user IDs. Do not pass only the delta.",
            },
        },
        "required": ["usergroup", "users"],
    },
}

registry.register(
    name="slack_list_usergroups",
    toolset="slack",
    schema=SLACK_LIST_USERGROUPS_SCHEMA,
    handler=lambda args, **kw: slack_list_usergroups(
        bool(args.get("include_users", False)), bool(args.get("include_disabled", False))
    ),
    check_fn=check_slack_usergroup_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="💬",
)

registry.register(
    name="slack_list_usergroup_users",
    toolset="slack",
    schema=SLACK_LIST_USERGROUP_USERS_SCHEMA,
    handler=lambda args, **kw: slack_list_usergroup_users(args.get("usergroup", "")),
    check_fn=check_slack_usergroup_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="💬",
)

registry.register(
    name="slack_update_usergroup_users",
    toolset="slack",
    schema=SLACK_UPDATE_USERGROUP_USERS_SCHEMA,
    handler=lambda args, **kw: slack_update_usergroup_users(
        args.get("usergroup", ""), args.get("users", [])
    ),
    check_fn=check_slack_usergroup_requirements,
    requires_env=["SLACK_BOT_TOKEN"],
    emoji="💬",
)

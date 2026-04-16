"""Shared schema helpers for group-oriented control-plane tools."""

from __future__ import annotations

from typing import Any, Mapping


def build_group_control_properties(
    *,
    platform_label: str,
    target_description: str,
    target_group_description: str,
    file_path_description: str,
    extra_properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "target": {
            "type": "string",
            "description": target_description,
        },
        "message": {
            "type": "string",
            "description": "Message body for send_message or optional social request decision note.",
        },
        "file_path": {
            "type": "string",
            "description": file_path_description,
        },
        "worker_name": {
            "type": "string",
            "description": "Intel worker name such as 钢镚 or 二狗.",
        },
        "target_group": {
            "type": "string",
            "description": target_group_description,
        },
        "objective": {
            "type": "string",
            "description": "Mission objective for an intel worker.",
        },
        "request_key": {
            "type": "string",
            "description": f"{platform_label} social request key such as friend:<flag> or group:<flag>.",
        },
        "request_type": {
            "type": "string",
            "enum": ["friend", "group"],
            "description": "Optional request type filter for list_requests.",
        },
        "status": {
            "type": "string",
            "enum": ["pending", "approved", "rejected", "ignored"],
            "description": "Optional status filter for list_requests.",
        },
        "user_id": {
            "type": "string",
            "description": f"{platform_label} user id for profile lookup or moderation.",
        },
        "user_query": {
            "type": "string",
            "description": f"{platform_label} user selector text for moderation when user_id is unknown.",
        },
        "reason": {
            "type": "string",
            "description": "Reason for moderation actions.",
        },
        "duration_seconds": {
            "type": "integer",
            "description": "Mute duration in seconds for mute_user.",
        },
        "mode": {
            "type": "string",
            "enum": ["default", "collect_only", "project_mode", "disabled"],
            "description": "Group policy mode when action=set_policy.",
        },
        "archive_enabled": {
            "type": "boolean",
            "description": "Archive toggle for group policy updates.",
        },
        "daily_report_enabled": {
            "type": "boolean",
            "description": "Daily report toggle for group policy or intel reporting updates.",
        },
        "daily_report_target": {
            "type": "string",
            "description": "Automatic daily report delivery target.",
        },
        "manual_report_target": {
            "type": "string",
            "description": "Manual report delivery target.",
        },
        "notify_target": {
            "type": "string",
            "description": "Intel mission status update target.",
        },
        "purge_raw_after_rollup": {
            "type": "boolean",
            "description": "Whether raw collected messages should be deleted after rollup.",
        },
        "group_name": {
            "type": "string",
            "description": f"Optional human-friendly {platform_label} group label.",
        },
        "report_date": {
            "type": "string",
            "description": "Daily report date in YYYY-MM-DD format.",
        },
        "query": {
            "type": "string",
            "description": "Archive text search query.",
        },
        "delivery_target": {
            "type": "string",
            "description": "Explicit delivery target for deliver_report.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of rows or requests to return.",
        },
        "auto_approve_friend_requests": {
            "type": "boolean",
            "description": f"Whether inbound {platform_label} friend requests should be approved automatically.",
        },
        "auto_approve_group_add_requests": {
            "type": "boolean",
            "description": f"Whether inbound {platform_label} group join requests should be approved automatically.",
        },
        "auto_approve_group_invites": {
            "type": "boolean",
            "description": f"Whether inbound {platform_label} group invitation requests should be approved automatically.",
        },
        "notes": {
            "type": "string",
            "description": "Operator notes for group policy, social policy, or intel worker updates.",
        },
        "no_cache": {
            "type": "boolean",
            "description": "Whether get_user_profile should bypass cache.",
        },
    }
    if extra_properties:
        properties.update(dict(extra_properties))
    return properties

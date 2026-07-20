"""Unified model-facing messaging control plane for QQ/NapCat and Weixin."""

from __future__ import annotations

from typing import Any

from tools.qq_control_tool import QQ_CONTROL_SPEC, qq_control_tool
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message
from tools.weixin_control_tool import WEIXIN_CONTROL_SPEC, weixin_control_tool


_QQ_PLATFORM_ALIASES = {
    "qq": "qq_napcat",
    "qq_napcat": "qq_napcat",
    "qq-napcat": "qq_napcat",
    "napcat": "qq_napcat",
}

_WEIXIN_PLATFORM_ALIASES = {
    "weixin": "weixin",
    "wechat": "weixin",
    "wx": "weixin",
}

_QQ_ONLY_ACTIONS = {
    "list_requests",
    "get_request",
    "approve_request",
    "reject_request",
    "list_friends",
    "get_user_profile",
    "get_social_policy",
    "set_social_policy",
    "clear_social_policy",
    "list_workers",
    "get_worker",
    "hire_worker",
    "pause_worker",
    "resume_worker",
    "stop_worker",
    "set_reporting",
    "run_report_now",
    "reconcile_workers",
    "list_joined_groups",
    "enable_collect_only",
    "list_files",
    "upload_file",
    "delete_file",
    "create_folder",
    "delete_folder",
    "group_file_system_info",
    "get_file_url",
    "move_file",
    "rename_file",
    "forward_file",
    "find_file",
    "resolve_file",
    "resolve_folder",
    "get_file_url_resolved",
    "delete_file_resolved",
    "forward_file_resolved",
    "move_file_resolved",
    "rename_file_resolved",
    "delete_folder_resolved",
    "pause",
    "stop",
}


def _copy_properties(properties: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in properties.items():
        copied[key] = dict(value) if isinstance(value, dict) else value
    return copied


def _collect_actions(*specs) -> list[str]:
    actions: set[str] = set()
    for spec in specs:
        actions.update(spec.extra_actions)
        for route in spec.route_specs_factory():
            actions.update(route.actions)
    return sorted(actions)


_MESSAGING_CONTROL_PROPERTIES = _copy_properties(QQ_CONTROL_SPEC.properties)
for key, value in WEIXIN_CONTROL_SPEC.properties.items():
    _MESSAGING_CONTROL_PROPERTIES.setdefault(key, dict(value) if isinstance(value, dict) else value)

_MESSAGING_CONTROL_PROPERTIES["platform"] = {
    "type": "string",
    "description": (
        "Target platform for ambiguous actions or actions without a target. "
        "Use qq_napcat (aliases: qq, napcat) or weixin (aliases: wx, wechat)."
    ),
}
_MESSAGING_CONTROL_PROPERTIES["target"] = {
    "type": "string",
    "description": (
        "Messaging target. QQ examples: group:123456, qq_napcat:group:123456, qq_napcat:dm:179033731, or a numeric group id. "
        "Weixin examples: weixin:wxid_xxx, weixin:filehelper, weixin:project@chatroom, or project@chatroom."
    ),
}
_MESSAGING_CONTROL_PROPERTIES["target_group"] = {
    "type": "string",
    "description": (
        "Group reference for worker or policy actions. QQ examples: group:123456 or qq_napcat:group:123456. "
        "Weixin examples: project@chatroom or weixin:project@chatroom."
    ),
}
_MESSAGING_CONTROL_PROPERTIES["file_path"] = {
    "type": "string",
    "description": "Optional local attachment path for supported send_message and QQ group-file actions.",
}


MESSAGING_CONTROL_SCHEMA = {
    "name": "messaging_control",
    "description": (
        "Unified messaging control plane for QQ/NapCat and Weixin. "
        "Prefer this as the default entry point for supported messaging-admin work: sending messages, "
        "group listening/report policy, archive/report inspection, employee route management, QQ social requests, "
        "QQ intel workers, QQ group files, and QQ moderation. "
        "Provide platform when the action is ambiguous or when there is no QQ/Weixin target to infer from. "
        "Do not write scripts yourself, and do not use terminal or execute_code when messaging_control supports the action."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": _collect_actions(QQ_CONTROL_SPEC, WEIXIN_CONTROL_SPEC),
                "description": "messaging_control action to perform.",
            },
            **_MESSAGING_CONTROL_PROPERTIES,
        },
        "required": ["action"],
    },
}


def _normalize_platform(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw in _QQ_PLATFORM_ALIASES:
        return "qq_napcat"
    if raw in _WEIXIN_PLATFORM_ALIASES:
        return "weixin"
    return ""


def _infer_platform_from_target(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("weixin:") or "@chatroom" in raw or raw == "filehelper" or raw.startswith("wxid_"):
        return "weixin"
    if raw.startswith("qq_napcat:") or raw.startswith("group:") or raw.isdigit():
        return "qq_napcat"
    return ""


def _resolve_platform(args: dict[str, Any]) -> str:
    explicit = _normalize_platform(args.get("platform"))
    if explicit:
        return explicit

    for key in ("target", "target_group"):
        inferred = _infer_platform_from_target(args.get(key))
        if inferred:
            return inferred

    action = str(args.get("action") or "").strip().lower()
    if action in _QQ_ONLY_ACTIONS:
        return "qq_napcat"
    if action == "resume" and str(args.get("worker_name") or "").strip() and not str(args.get("target") or "").strip():
        return "qq_napcat"
    return ""


def _platform_resolution_error(args: dict[str, Any]) -> str:
    action = str(args.get("action") or "").strip().lower()
    if action in {"list_employee_routes", "set_employee_route", "clear_employee_route"}:
        return (
            "Cannot determine platform for employee-route action. "
            "Set platform to qq_napcat or weixin."
        )
    return (
        "Cannot determine platform for messaging_control. "
        "Set platform to qq_napcat or weixin, or use a QQ/Weixin target such as "
        "group:123456, qq_napcat:group:123456, or weixin:project@chatroom."
    )


def messaging_control_tool(args, **kw):
    del kw
    action = str(args.get("action") or "").strip().lower()
    if not action:
        return tool_error("'action' is required.")

    platform = _resolve_platform(dict(args))
    if not platform:
        return tool_error(_platform_resolution_error(dict(args)), success=False)

    payload = dict(args)
    payload.pop("platform", None)
    if platform == "qq_napcat":
        return qq_control_tool(payload)
    if platform == "weixin":
        return weixin_control_tool(payload)
    return tool_error(f"Unsupported messaging_control platform: {platform}")


registry.register(
    name="messaging_control",
    toolset="messaging",
    schema=MESSAGING_CONTROL_SCHEMA,
    handler=messaging_control_tool,
    check_fn=_check_send_message,
    emoji="🧩",
)

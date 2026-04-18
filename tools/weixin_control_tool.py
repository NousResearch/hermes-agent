"""Unified top-level Weixin control plane for model-facing use."""

from __future__ import annotations

from tools.control_plane_helpers import (
    ControlPlaneSpec,
    ControlRouteSpec,
    build_control_schema_from_spec,
    run_control_plane,
)
from tools.employee_route_tool import EMPLOYEE_ROUTE_CONTROL_PROPERTIES, employee_route_tool
from tools.group_control_schema import build_group_control_properties
from tools.registry import registry, tool_error, tool_result
from tools.send_message_tool import _check_send_message, send_message_tool
from tools.weixin_group_archive_tool import weixin_group_archive_tool
from tools.weixin_group_policy_tool import weixin_group_policy_tool


_GROUP_POLICY_ACTIONS = {
    "list_policies",
    "get_policy",
    "set_policy",
    "clear_policy",
    "collect_only",
    "resume_chat",
    "disable_group",
}

_GROUP_ARCHIVE_ACTIONS = {
    "list_recent",
    "search",
    "list_reports",
    "get_report",
    "rollup_daily",
    "rollup_due",
    "snapshot_report",
    "deliver_report",
}

_EMPLOYEE_ROUTE_ACTIONS = {
    "list_employee_routes",
    "set_employee_route",
    "clear_employee_route",
}

_GROUP_MODERATION_ACTIONS = {
    "mute_user",
    "kick_user",
}

_ACTION_ALIASES = {
    "send": "send_message",
    "mute": "mute_user",
    "mute_member": "mute_user",
    "kick": "kick_user",
    "kick_member": "kick_user",
    "collect-only": "set_policy",
    "collect_only": "set_policy",
    "no_reply": "set_policy",
    "no-reply": "set_policy",
    "resume": "resume_chat",
    "disable": "disable_group",
    "report": "report_now",
}


def _weixin_control_route_specs() -> list[ControlRouteSpec]:
    return [
        ControlRouteSpec(
            actions={"send_message"},
            transform=lambda payload: {**payload, "action": "send"},
            handler=send_message_tool,
        ),
        ControlRouteSpec(
            actions=_EMPLOYEE_ROUTE_ACTIONS,
            transform=lambda payload: {
                **payload,
                "platform": "weixin",
                "action": {
                    "list_employee_routes": "list_routes",
                    "set_employee_route": "set_route",
                    "clear_employee_route": "clear_route",
                }[str(payload.get("action") or "")],
            },
            handler=employee_route_tool,
        ),
        ControlRouteSpec(actions=_GROUP_POLICY_ACTIONS, handler=weixin_group_policy_tool),
        ControlRouteSpec(
            actions={"report_now"},
            transform=lambda payload: {**payload, "action": "deliver_report"},
            handler=weixin_group_archive_tool,
        ),
        ControlRouteSpec(actions=_GROUP_ARCHIVE_ACTIONS, handler=weixin_group_archive_tool),
    ]


_WEIXIN_CONTROL_PROPERTIES = build_group_control_properties(
    platform_label="Weixin",
    target_description=(
        "Weixin target such as weixin:wxid_xxx, weixin:filehelper, or weixin:group@chatroom. "
        "For group policy/archive actions, use group@chatroom or weixin:group@chatroom."
    ),
    target_group_description="Weixin target group reference such as group@chatroom.",
    file_path_description="Optional local attachment path for send_message.",
    extra_properties=EMPLOYEE_ROUTE_CONTROL_PROPERTIES,
)
_WEIXIN_CONTROL_PROPERTIES["mode"] = {
    "type": "string",
    "enum": ["default", "collect_only", "disabled"],
    "description": "Weixin group policy mode when action=set_policy.",
}


def _normalize_control_args(args: dict) -> dict:
    normalized = dict(args)
    original_action = str(args.get("action") or "").strip().lower()
    action = str(_ACTION_ALIASES.get(original_action, original_action)).strip().lower()
    normalized["action"] = action

    if original_action in {"collect_only", "collect-only", "no_reply", "no-reply"}:
        normalized.setdefault("mode", "collect_only")
        normalized.setdefault("archive_enabled", True)

    if original_action in {"report_now", "report"} and normalized.get("target"):
        normalized["action"] = "report_now"

    return normalized


WEIXIN_CONTROL_SPEC = ControlPlaneSpec(
    name="weixin_control",
    description=(
        "Unified Weixin control plane. "
        "Prefer this tool for Weixin operations such as sending messages, "
        "configuring group listening policy, managing employee routes, and inspecting archive/report state. "
        "Use this instead of terminal or execute_code for supported Weixin admin work. "
        "Choose one action and provide the related fields."
    ),
    aliases=_ACTION_ALIASES,
    route_specs_factory=_weixin_control_route_specs,
    properties=_WEIXIN_CONTROL_PROPERTIES,
    extra_actions={
        "send",
        "collect_only",
        "no_reply",
        "report_now",
        "mute",
        "mute_member",
        "mute_user",
        "kick",
        "kick_member",
        "kick_user",
    },
    normalize_args=_normalize_control_args,
)

WEIXIN_CONTROL_SCHEMA = build_control_schema_from_spec(WEIXIN_CONTROL_SPEC)


def _unsupported_weixin_control_action(payload: dict) -> str:
    action = str(payload.get("action") or "").strip().lower()
    if action in _GROUP_MODERATION_ACTIONS:
        return tool_result(
            success=False,
            platform="weixin",
            action=action,
            capability="not_capable",
            target=str(payload.get("target") or "").strip(),
            detail="微信群暂不支持禁言/踢人。",
        )
    return tool_error(
        "Unsupported Weixin control action. Use send_message, employee-route, group policy, or archive/report actions."
    )


def weixin_control_tool(args, **kw):
    del kw
    action = str(args.get("action") or "").strip().lower()
    if not action:
        return tool_error("'action' is required.")
    return run_control_plane(
        dict(args),
        spec=WEIXIN_CONTROL_SPEC,
        unsupported=_unsupported_weixin_control_action,
    )


registry.register(
    name="weixin_control",
    toolset="messaging",
    schema=WEIXIN_CONTROL_SCHEMA,
    handler=weixin_control_tool,
    check_fn=_check_send_message,
    emoji="🧩",
)

"""Unified top-level QQ/NapCat control plane for model-facing use."""

from __future__ import annotations

import json

from gateway.qq_group_archive import QqGroupArchiveStore
from tools.control_plane_helpers import (
    ControlPlaneSpec,
    ControlRouteSpec,
    build_control_schema_from_spec,
    run_control_plane,
)
from tools.employee_route_tool import EMPLOYEE_ROUTE_CONTROL_PROPERTIES, employee_route_tool
from tools.group_control_schema import build_group_control_properties
from tools.qq_group_archive_tool import qq_group_archive_tool
from tools.qq_group_file_tool import qq_group_file_tool
from tools.qq_group_moderation_tool import qq_group_moderation_tool
from tools.qq_group_policy_tool import qq_group_policy_tool
from tools.qq_intel_tool import qq_intel_tool
from tools.qq_social_tool import qq_social_tool
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, send_message_tool


_SOCIAL_ACTIONS = {
    "list_requests",
    "get_request",
    "approve_request",
    "reject_request",
    "list_friends",
    "get_user_profile",
    "get_social_policy",
    "set_social_policy",
    "clear_social_policy",
}

_INTEL_ACTIONS = {
    "list_workers",
    "get_worker",
    "hire_worker",
    "pause_worker",
    "resume_worker",
    "stop_worker",
    "set_reporting",
    "run_report_now",
    "reconcile_workers",
}

_EMPLOYEE_ROUTE_ACTIONS = {
    "list_employee_routes",
    "set_employee_route",
    "clear_employee_route",
}

_GROUP_POLICY_ACTIONS = {
    "list_joined_groups",
    "list_policies",
    "get_policy",
    "set_policy",
    "clear_policy",
    "enable_collect_only",
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

_GROUP_MODERATION_ACTIONS = {
    "mute_user",
    "kick_user",
}

_GROUP_FILE_ACTIONS = {
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
}

_GROUP_FILE_ACTION_MAP = {
    "list_files": "list",
    "upload_file": "upload",
    "delete_file": "delete",
    "create_folder": "create_folder",
    "delete_folder": "delete_folder",
    "group_file_system_info": "system_info",
    "get_file_url": "get_url",
    "move_file": "move",
    "rename_file": "rename",
    "forward_file": "forward",
    "find_file": "find",
    "resolve_file": "resolve",
    "resolve_folder": "resolve_folder",
    "get_file_url_resolved": "get_url_resolved",
    "delete_file_resolved": "delete_resolved",
    "forward_file_resolved": "forward_resolved",
    "move_file_resolved": "move_resolved",
    "rename_file_resolved": "rename_resolved",
    "delete_folder_resolved": "delete_folder_resolved",
}

_ACTION_ALIASES = {
    "send": "send_message",
    "mute": "mute_user",
    "mute_member": "mute_user",
    "ban": "mute_user",
    "ban_user": "mute_user",
    "ban_member": "mute_user",
    "kick": "kick_user",
    "kick_member": "kick_user",
    "remove_user": "kick_user",
    "remove_member": "kick_user",
    "list_groups": "list_joined_groups",
    "list_group_policies": "list_policies",
    "get_group_policy": "get_policy",
    "set_group_policy": "set_policy",
    "clear_group_policy": "clear_policy",
    "enable_collect_only": "enable_collect_only",
    "resume_chat": "resume_chat",
    "disable_group": "disable_group",
    "collect_only": "set_policy",
    "collect-only": "set_policy",
    "listen_only": "set_policy",
    "listen-only": "set_policy",
    "no_reply": "set_policy",
    "no-reply": "set_policy",
    "pause": "pause_worker",
    "resume": "resume_worker",
    "stop": "stop_worker",
    "report": "report_now",
}

def _policy_has_runtime_override(policy: dict) -> bool:
    mode = str(policy.get("mode") or "").strip().lower()
    return bool(
        mode not in {"", "default"}
        or bool(policy.get("archive_enabled"))
        or bool(policy.get("daily_report_enabled"))
        or str(policy.get("daily_report_target") or "").strip()
        or str(policy.get("manual_report_target") or "").strip()
        or not bool(policy.get("purge_raw_after_rollup", True))
    )


def _augment_policy_with_reporting(policy: dict) -> dict:
    group_id = str(policy.get("group_id") or "").strip()
    if not group_id:
        return dict(policy)

    reporting = QqGroupArchiveStore().describe_group_reporting(group_id=group_id)
    augmented = dict(policy)
    augmented["reporting"] = reporting

    if reporting["overlay"]["active"] and not _policy_has_runtime_override(policy):
        effective_policy = dict(policy)
        effective_policy["mode"] = "collect_only"
        effective_policy["archive_enabled"] = True
        effective_policy["daily_report_enabled"] = bool(reporting["overlay"]["daily_report_enabled"])
        daily_targets = list(reporting["effective_targets"]["daily_report_targets"])
        manual_targets = list(reporting["effective_targets"]["manual_report_targets"])
        effective_policy["daily_report_target"] = daily_targets[0] if len(daily_targets) == 1 else None
        effective_policy["manual_report_target"] = manual_targets[0] if len(manual_targets) == 1 else None
        effective_policy["daily_report_targets"] = daily_targets
        effective_policy["manual_report_targets"] = manual_targets
        augmented["effective_policy"] = effective_policy
    else:
        augmented["effective_policy"] = dict(policy)
    return augmented


def _maybe_augment_group_policy_result(action: str, result_json: str) -> str:
    if action not in {"get_policy", "list_policies"}:
        return result_json
    try:
        payload = json.loads(result_json)
    except Exception:
        return result_json
    if payload.get("error") or not payload.get("success"):
        return result_json

    if action == "get_policy" and isinstance(payload.get("policy"), dict):
        payload["policy"] = _augment_policy_with_reporting(payload["policy"])
        return json.dumps(payload, ensure_ascii=False)

    if action == "list_policies" and isinstance(payload.get("policies"), list):
        payload["policies"] = [
            _augment_policy_with_reporting(policy)
            if isinstance(policy, dict) else policy
            for policy in payload["policies"]
        ]
        return json.dumps(payload, ensure_ascii=False)

    return result_json


def _postprocess_qq_control_result(args: dict, result):
    return _maybe_augment_group_policy_result(str(args.get("action") or ""), result)


def _qq_control_route_specs() -> list[ControlRouteSpec]:
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
                "platform": "qq_napcat",
                "action": {
                    "list_employee_routes": "list_routes",
                    "set_employee_route": "set_route",
                    "clear_employee_route": "clear_route",
                }[str(payload.get("action") or "")],
            },
            handler=employee_route_tool,
        ),
        ControlRouteSpec(actions=_SOCIAL_ACTIONS, handler=qq_social_tool),
        ControlRouteSpec(actions=_INTEL_ACTIONS, handler=qq_intel_tool),
        ControlRouteSpec(
            actions=_GROUP_POLICY_ACTIONS,
            handler=qq_group_policy_tool,
        ),
        ControlRouteSpec(
            actions={"report_now"},
            transform=lambda payload: {**payload, "action": "deliver_report"},
            handler=qq_group_archive_tool,
        ),
        ControlRouteSpec(actions=_GROUP_ARCHIVE_ACTIONS, handler=qq_group_archive_tool),
        ControlRouteSpec(
            actions=_GROUP_FILE_ACTIONS,
            transform=lambda payload: {
                **payload,
                "action": _GROUP_FILE_ACTION_MAP[str(payload.get("action") or "")],
            },
            handler=qq_group_file_tool,
        ),
        ControlRouteSpec(actions=_GROUP_MODERATION_ACTIONS, handler=qq_group_moderation_tool),
    ]

_QQ_CONTROL_PROPERTIES = build_group_control_properties(
    platform_label="QQ",
    target_description=(
        "QQ target such as group:<id>, qq_napcat:group:<id>, qq_napcat:dm:<id>, or a numeric group id."
    ),
    target_group_description="QQ target group reference for hiring an intel worker.",
    file_path_description="Optional local attachment path for send_message or QQ group file upload.",
    extra_properties={
    **EMPLOYEE_ROUTE_CONTROL_PROPERTIES,
    "folder_id": {
        "type": "string",
        "description": "QQ group file folder id for list/upload/delete_folder actions.",
    },
    "folder_name": {
        "type": "string",
        "description": "Folder name when creating a QQ group file folder.",
    },
    "parent_id": {
        "type": "string",
        "description": "Parent folder id when creating a QQ group file folder.",
    },
    "file_name": {
        "type": "string",
        "description": "Optional display name when uploading a QQ group file.",
    },
    "file_id": {
        "type": "string",
        "description": "QQ group file id for file actions.",
    },
    "busid": {
        "type": "integer",
        "description": "NapCat busid for group file delete/get_url actions.",
    },
    "target_dir": {
        "type": "string",
        "description": "Target folder id/path when moving a QQ group file.",
    },
    "new_name": {
        "type": "string",
        "description": "New file name when renaming a QQ group file.",
    },
    "current_parent_directory": {
        "type": "string",
        "description": "Current parent directory when renaming a QQ group file.",
    },
    "target_group_id": {
        "type": "string",
        "description": "Destination QQ group when forwarding a QQ group file.",
    },
    "include_folders": {
        "type": "boolean",
        "description": "Whether QQ group file search should include folders.",
    },
    "recursive": {
        "type": "boolean",
        "description": "Whether QQ group file search should recurse into subfolders.",
    },
    "exact": {
        "type": "boolean",
        "description": "Whether QQ group file search should require exact name matches.",
    },
    "max_results": {
        "type": "integer",
        "description": "Maximum number of QQ group file matches to return.",
    },
    },
)

QQ_CONTROL_SPEC = ControlPlaneSpec(
    name="qq_control",
    description=(
        "Unified QQ/NapCat control plane. "
        "Prefer this tool for QQ operations such as social requests, group listening policy, "
        "archive/report inspection, intel worker missions, employee route management, group file operations, and group moderation. "
        "Use this instead of shell scripts, terminal commands, or execute_code for supported QQ admin work. "
        "Group moderation routed through qq_control still preserves approval and protected-user guardrails. "
        "Choose one action and provide the related fields."
    ),
    aliases=_ACTION_ALIASES,
    route_specs_factory=_qq_control_route_specs,
    properties=_QQ_CONTROL_PROPERTIES,
    extra_actions={"send", "collect_only", "no_reply", "pause", "resume", "stop", "report_now"},
    normalize_args=lambda args: _normalize_control_args(args),
    postprocess_result=_postprocess_qq_control_result,
)

QQ_CONTROL_SCHEMA = build_control_schema_from_spec(QQ_CONTROL_SPEC)


def qq_control_tool(args, **kw):
    del kw
    action = str(args.get("action") or "").strip().lower()
    if not action:
        return tool_error("'action' is required.")
    return run_control_plane(
        dict(args),
        spec=QQ_CONTROL_SPEC,
        unsupported=lambda _payload: tool_error(
            "Unsupported QQ control action. Use a social, intel, employee-route, group policy, archive, group file, or moderation action."
        ),
    )


def _normalize_control_args(args: dict) -> dict:
    normalized = dict(args)
    original_action = str(args.get("action") or "").strip().lower()
    action = str(_ACTION_ALIASES.get(original_action, original_action)).strip().lower()
    normalized["action"] = action

    if original_action in {"collect_only", "collect-only", "listen_only", "listen-only", "no_reply", "no-reply"}:
        normalized.setdefault("mode", "collect_only")
        normalized.setdefault("archive_enabled", True)

    if original_action in {"report_now", "report"}:
        if normalized.get("worker_name"):
            normalized["action"] = "run_report_now"
        elif normalized.get("target"):
            normalized["action"] = "report_now"

    return normalized


registry.register(
    name="qq_control",
    toolset="messaging",
    schema=QQ_CONTROL_SCHEMA,
    handler=qq_control_tool,
    check_fn=_check_send_message,
    emoji="🧩",
)

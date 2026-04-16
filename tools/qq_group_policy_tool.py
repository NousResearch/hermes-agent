"""QQ/NapCat per-group policy management tool."""

from __future__ import annotations

import json

from gateway.config import Platform
from gateway.qq_group_archive import QqGroupArchiveStore
from gateway.qq_group_policies import (
    clear_group_policy,
    get_group_policy,
    list_group_policies,
    normalize_group_policy_mode,
    summarize_group_policy_state,
    set_group_policy,
)
from tools.registry import registry, tool_error
from tools.qq_group_tool_common import (
    require_admin_session,
    resolve_delivery_target,
    resolve_group_target,
    session_actor_label,
)
from tools.send_message_tool import _check_send_message, _error, _qq_napcat_call

_ACTION_ALIASES = {
    "collect_only": "set_policy",
    "collect-only": "set_policy",
    "enable_collect_only": "set_policy",
    "disable_group": "set_policy",
    "resume_chat": "set_policy",
    "listen_only": "set_policy",
    "listen-only": "set_policy",
    "no_reply": "set_policy",
    "no-reply": "set_policy",
}


QQ_GROUP_POLICY_SCHEMA = {
    "name": "qq_group_policy",
    "description": (
        "Inspect or change QQ/NapCat inbound group policy. "
        "Use this to list joined groups, inspect current policy, or set a group to "
        "'collect_only', 'project_mode', 'disabled', or 'default'. "
        "This is the correct tool for configuring groups that should only be listened to and archived without hitting the main model."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_joined_groups",
                    "list_policies",
                    "get_policy",
                    "set_policy",
                    "clear_policy",
                    "collect_only",
                    "enable_collect_only",
                    "disable_group",
                    "resume_chat",
                    "no_reply",
                ],
                "description": "Policy operation to perform.",
            },
            "target": {
                "type": "string",
                "description": (
                    "QQ group target. Accepts 'group:123456', 'qq_napcat:group:123456', or a numeric group id. "
                    "If omitted for get/set/clear, Hermes uses the current QQ group session."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["default", "collect_only", "project_mode", "disabled"],
                "description": "Effective policy mode when action='set_policy'. Optional: if omitted, Hermes keeps the current mode and only updates the provided switches/notes.",
            },
            "archive_enabled": {
                "type": "boolean",
                "description": "Optional explicit archive toggle when action='set_policy'. Defaults to true for collect_only, false otherwise.",
            },
            "daily_report_enabled": {
                "type": "boolean",
                "description": "Optional explicit daily-report toggle when action='set_policy'. Defaults to false unless already enabled.",
            },
            "daily_report_target": {
                "type": "string",
                "description": "Optional automatic daily-report delivery target. Use current_chat, current_user_dm, none, qq_napcat:group:<id>, or qq_napcat:dm:<id>.",
            },
            "manual_report_target": {
                "type": "string",
                "description": "Optional default delivery target for immediate/manual reports. Use current_chat, current_user_dm, none, qq_napcat:group:<id>, or qq_napcat:dm:<id>.",
            },
            "purge_raw_after_rollup": {
                "type": "boolean",
                "description": "Whether raw collected rows should be deleted after a daily report is stored. Defaults to true.",
            },
            "group_name": {
                "type": "string",
                "description": "Optional human-friendly group label for reports and policy listings.",
            },
            "notes": {
                "type": "string",
                "description": "Optional operator note saved with the policy.",
            },
            "reply_mode": {
                "type": "string",
                "enum": ["no_reply", "collect_only"],
                "description": "Shortcut-friendly reply behavior alias. 'no_reply' maps to collect_only archiving mode.",
            },
        },
        "required": ["action"],
    },
}


def qq_group_policy_tool(args, **kw):
    del kw

    normalized_args = _normalize_policy_args(args)
    action = str(normalized_args.get("action") or "").strip().lower()
    if action not in {"list_joined_groups", "list_policies", "get_policy", "set_policy", "clear_policy"}:
        return tool_error(
            "Unsupported action. Use 'list_joined_groups', 'list_policies', 'get_policy', 'set_policy', or 'clear_policy'."
        )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config

        config = load_gateway_config()
    except Exception as exc:
        return json.dumps(_error(f"Failed to load gateway config: {exc}"), ensure_ascii=False)

    pconfig = config.platforms.get(Platform.QQ_NAPCAT)
    if not pconfig or not pconfig.enabled:
        return tool_error(
            "Platform 'qq_napcat' is not configured. Set up NapCat credentials in ~/.hermes/config.yaml or environment variables."
        )

    try:
        if action == "list_joined_groups":
            from model_tools import _run_async

            result = _run_async(_list_joined_groups(pconfig.extra))
            result = _augment_joined_groups_payload(result)
            result["action"] = action
            return json.dumps(result, ensure_ascii=False)

        if action == "list_policies":
            payload = _augment_policy_payload({"success": True, "action": action, "policies": list_group_policies()})
            return json.dumps(payload, ensure_ascii=False)

        group_id = resolve_group_target(normalized_args.get("target"))
        if action == "get_policy":
            payload = _augment_policy_payload(
                {
                    "success": True,
                    "action": action,
                    "target": str(normalized_args.get("target") or f"group:{group_id}"),
                    "group_id": group_id,
                    "policy": get_group_policy(group_id),
                }
            )
            return json.dumps(payload, ensure_ascii=False)

        admin_error = require_admin_session(
            "修改 QQ 群监听/采集策略",
        )
        if admin_error:
            return json.dumps(_error(admin_error), ensure_ascii=False)

        if action == "clear_policy":
            payload = _augment_policy_payload(
                {
                    "success": True,
                    "action": action,
                    "target": str(normalized_args.get("target") or f"group:{group_id}"),
                    "group_id": group_id,
                    "policy": clear_group_policy(group_id),
                }
            )
            return json.dumps(payload, ensure_ascii=False)

        current_policy = get_group_policy(group_id)
        mode_arg = normalized_args.get("mode")
        if mode_arg is None or str(mode_arg).strip() == "":
            mode = current_policy["mode"]
        else:
            mode = normalize_group_policy_mode(mode_arg)
        policy = set_group_policy(
            group_id,
            mode=mode,
            archive_enabled=normalized_args.get("archive_enabled"),
            daily_report_enabled=normalized_args.get("daily_report_enabled"),
            daily_report_target=resolve_delivery_target(normalized_args.get("daily_report_target")),
            manual_report_target=resolve_delivery_target(normalized_args.get("manual_report_target")),
            purge_raw_after_rollup=normalized_args.get("purge_raw_after_rollup"),
            group_name=str(normalized_args.get("group_name") or "").strip() or None,
            notes=str(normalized_args.get("notes") or "").strip() or None,
            updated_by=session_actor_label(),
        )
        payload = _augment_policy_payload(
            {
                "success": True,
                "action": action,
                "target": str(normalized_args.get("target") or f"group:{group_id}"),
                "group_id": group_id,
                "policy": policy,
            }
        )
        reply_behavior = _policy_reply_behavior(payload.get("policy") or {})
        if reply_behavior:
            payload["reply_behavior"] = reply_behavior
        return json.dumps(payload, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps(_error(str(exc)), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"QQ group policy action failed: {exc}"), ensure_ascii=False)


def _normalize_policy_action(value) -> str:
    action = str(value or "").strip().lower()
    return _ACTION_ALIASES.get(action, action)


def _normalize_policy_args(args: dict) -> dict:
    normalized = dict(args)
    original_action = str(args.get("action") or "").strip().lower()
    action = _normalize_policy_action(original_action)
    normalized["action"] = action

    if original_action in {"collect_only", "collect-only", "listen_only", "listen-only", "no_reply", "no-reply", "enable_collect_only"}:
        normalized.setdefault("mode", "collect_only")
        normalized.setdefault("archive_enabled", True)

    if original_action == "resume_chat":
        normalized.setdefault("mode", "default")
        normalized.setdefault("archive_enabled", False)
        normalized.setdefault("daily_report_enabled", False)

    if original_action == "disable_group":
        normalized.setdefault("mode", "disabled")
        normalized.setdefault("archive_enabled", False)
        normalized.setdefault("daily_report_enabled", False)

    if str(normalized.get("reply_mode") or "").strip().lower() in {"no_reply", "no-reply", "collect_only", "collect-only"}:
        normalized["action"] = "set_policy"
        normalized.setdefault("mode", "collect_only")
        normalized.setdefault("archive_enabled", True)

    return normalized


def _policy_reply_behavior(policy: dict) -> str | None:
    behavior = summarize_group_policy_state(policy).get("reply_behavior")
    if behavior == "respond":
        return None
    return str(behavior or "").strip() or None


async def _list_joined_groups(extra: dict) -> dict:
    data, error = await _qq_napcat_call(extra, "get_group_list", {})
    if error:
        return error

    groups = data if isinstance(data, list) else []
    merged = []
    for item in groups:
        if not isinstance(item, dict):
            continue
        group_id = str(item.get("group_id") or item.get("groupCode") or "").strip()
        if not group_id:
            continue
        merged.append(
            {
                "group_id": group_id,
                "group_name": str(item.get("group_name") or item.get("groupName") or group_id),
                "member_count": item.get("member_count"),
                "max_member_count": item.get("max_member_count"),
                "policy": get_group_policy(group_id),
            }
        )
    merged.sort(key=lambda item: (str(item.get("group_name") or ""), item["group_id"]))
    return {
        "success": True,
        "platform": "qq_napcat",
        "groups": merged,
        "count": len(merged),
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
    policy_summary = summarize_group_policy_state(policy)
    effective_policy = dict(policy)
    daily_targets = list(reporting["delivery_targets"]["daily_report_targets"])
    manual_targets = list(reporting["delivery_targets"]["manual_report_targets"])
    effective_policy["daily_report_targets"] = daily_targets
    effective_policy["manual_report_targets"] = manual_targets

    if reporting["overlay"]["active"] and not _policy_has_runtime_override(policy):
        effective_policy["mode"] = "collect_only"
        effective_policy["archive_enabled"] = True
        effective_policy["daily_report_enabled"] = bool(reporting["overlay"]["daily_report_enabled"])
        effective_policy["daily_report_target"] = daily_targets[0] if len(daily_targets) == 1 else None
        effective_policy["manual_report_target"] = manual_targets[0] if len(manual_targets) == 1 else None

    augmented = dict(policy)
    augmented["policy_summary"] = policy_summary
    augmented["reporting"] = reporting
    augmented["effective_policy"] = effective_policy
    augmented["collect_only"] = bool(reporting.get("collect_only"))
    augmented["replies_disabled"] = bool(reporting.get("replies_disabled"))
    augmented["reply_behavior"] = str(reporting.get("reply_behavior") or policy_summary["reply_behavior"])
    augmented["report_control"] = dict(reporting.get("report_control") or {})
    return augmented


def _augment_policy_payload(payload: dict) -> dict:
    if payload.get("error") or not payload.get("success"):
        return payload

    if isinstance(payload.get("policy"), dict):
        updated = dict(payload)
        updated["policy"] = _augment_policy_with_reporting(payload["policy"])
        policy = updated["policy"]
        updated["policy_summary"] = dict(policy.get("policy_summary") or {})
        updated["collect_only"] = bool(policy.get("collect_only"))
        updated["replies_disabled"] = bool(policy.get("replies_disabled"))
        updated["reply_behavior"] = str(policy.get("reply_behavior") or "")
        updated["report_control"] = dict(policy.get("report_control") or {})
        updated["delivery_targets"] = dict(
            ((policy.get("reporting") or {}).get("delivery_targets") or {})
        )
        return updated

    if isinstance(payload.get("policies"), list):
        updated = dict(payload)
        updated["policies"] = [
            _augment_policy_with_reporting(policy)
            if isinstance(policy, dict) else policy
            for policy in payload["policies"]
        ]
        return updated

    return payload


def _augment_joined_groups_payload(payload: dict) -> dict:
    if payload.get("error") or not payload.get("success") or not isinstance(payload.get("groups"), list):
        return payload

    updated = dict(payload)
    groups: list[dict] = []
    for item in payload["groups"]:
        if not isinstance(item, dict):
            groups.append(item)
            continue
        merged = dict(item)
        policy = merged.get("policy")
        if isinstance(policy, dict):
            merged["policy"] = _augment_policy_with_reporting(policy)
            merged["policy_summary"] = dict(merged["policy"].get("policy_summary") or {})
        groups.append(merged)
    updated["groups"] = groups
    return updated


registry.register(
    name="qq_group_policy",
    toolset="messaging",
    schema=QQ_GROUP_POLICY_SCHEMA,
    handler=qq_group_policy_tool,
    check_fn=_check_send_message,
    emoji="🧭",
)

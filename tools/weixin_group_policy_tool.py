"""Weixin per-group policy management tool."""

from __future__ import annotations

import json

from gateway.config import Platform
from gateway.weixin_group_archive import WeixinGroupArchiveStore
from gateway.weixin_group_policies import (
    clear_group_policy,
    get_group_policy,
    list_group_policies,
    normalize_group_policy_mode,
    summarize_group_policy_state,
    set_group_policy,
)
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, _error
from tools.weixin_group_tool_common import (
    require_admin_session,
    resolve_delivery_target,
    resolve_group_target,
    session_actor_label,
)

_ACTION_ALIASES = {
    "collect_only": "set_policy",
    "collect-only": "set_policy",
    "disable_group": "set_policy",
    "resume_chat": "set_policy",
    "no_reply": "set_policy",
    "no-reply": "set_policy",
}


WEIXIN_GROUP_POLICY_SCHEMA = {
    "name": "weixin_group_policy",
    "description": (
        "Inspect or change Weixin inbound group policy. "
        "Use this to inspect or set a Weixin group to 'collect_only', 'disabled', or 'default'. "
        "This is the correct tool for configuring Weixin groups that should be listened to and archived without hitting the main model."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_policies",
                    "get_policy",
                    "set_policy",
                    "clear_policy",
                    "collect_only",
                    "disable_group",
                    "resume_chat",
                    "no_reply",
                ],
                "description": "Policy operation to perform.",
            },
            "target": {
                "type": "string",
                "description": (
                    "Weixin group target. Accepts 'group@chatroom', 'group:group@chatroom', or 'weixin:group@chatroom'. "
                    "If omitted for get/set/clear, Hermes uses the current Weixin group session."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["default", "collect_only", "disabled"],
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
                "description": "Optional automatic daily-report delivery target. Use current_chat, current_user_dm, none, weixin:group@chatroom, or weixin:wxid_xxx.",
            },
            "manual_report_target": {
                "type": "string",
                "description": "Optional default delivery target for immediate/manual reports. Use current_chat, current_user_dm, none, weixin:group@chatroom, or weixin:wxid_xxx.",
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


def weixin_group_policy_tool(args, **kw):
    del kw

    normalized_args = _normalize_policy_args(args)
    action = str(normalized_args.get("action") or "").strip().lower()
    if action not in {"list_policies", "get_policy", "set_policy", "clear_policy"}:
        return tool_error(
            "Unsupported action. Use 'list_policies', 'get_policy', 'set_policy', or 'clear_policy'."
        )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted")

    try:
        from gateway.config import load_gateway_config

        config = load_gateway_config()
    except Exception as exc:
        return json.dumps(_error(f"Failed to load gateway config: {exc}"), ensure_ascii=False)

    pconfig = config.platforms.get(Platform.WEIXIN)
    if not pconfig or not pconfig.enabled:
        return tool_error(
            "Platform 'weixin' is not configured. Set up Weixin credentials in ~/.hermes/config.yaml or environment variables."
        )

    try:
        if action == "list_policies":
            payload = _augment_policy_payload({"success": True, "action": action, "policies": list_group_policies()})
            return json.dumps(payload, ensure_ascii=False)

        chat_id = resolve_group_target(normalized_args.get("target"))
        if action == "get_policy":
            payload = _augment_policy_payload(
                {
                    "success": True,
                    "action": action,
                    "target": str(normalized_args.get("target") or f"weixin:{chat_id}"),
                    "chat_id": chat_id,
                    "policy": get_group_policy(chat_id),
                }
            )
            return json.dumps(payload, ensure_ascii=False)

        admin_error = require_admin_session("修改微信群监听/采集策略")
        if admin_error:
            return json.dumps(_error(admin_error), ensure_ascii=False)

        if action == "clear_policy":
            payload = _augment_policy_payload(
                {
                    "success": True,
                    "action": action,
                    "target": str(normalized_args.get("target") or f"weixin:{chat_id}"),
                    "chat_id": chat_id,
                    "policy": clear_group_policy(chat_id),
                }
            )
            return json.dumps(payload, ensure_ascii=False)

        current_policy = get_group_policy(chat_id)
        mode_arg = normalized_args.get("mode")
        if mode_arg is None or str(mode_arg).strip() == "":
            mode = current_policy["mode"]
        else:
            mode = normalize_group_policy_mode(mode_arg)
            if mode == "project_mode":
                raise ValueError("Weixin group policy currently does not support project_mode.")
        policy = set_group_policy(
            chat_id,
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
                "target": str(normalized_args.get("target") or f"weixin:{chat_id}"),
                "chat_id": chat_id,
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
        return json.dumps(_error(f"Weixin group policy action failed: {exc}"), ensure_ascii=False)


def _normalize_policy_action(value) -> str:
    action = str(value or "").strip().lower()
    return _ACTION_ALIASES.get(action, action)


def _normalize_policy_args(args: dict) -> dict:
    normalized = dict(args)
    original_action = str(args.get("action") or "").strip().lower()
    action = _normalize_policy_action(original_action)
    normalized["action"] = action

    if original_action in {"collect_only", "collect-only", "no_reply", "no-reply"}:
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


def _augment_policy_payload(payload: dict) -> dict:
    result = dict(payload)
    store = WeixinGroupArchiveStore()
    if isinstance(result.get("policy"), dict):
        policy = dict(result["policy"])
        policy["reporting"] = store.describe_group_reporting(chat_id=str(policy.get("chat_id") or ""))
        result["policy"] = policy
    if isinstance(result.get("policies"), list):
        policies = []
        for item in result["policies"]:
            if not isinstance(item, dict):
                policies.append(item)
                continue
            policy = dict(item)
            policy["reporting"] = store.describe_group_reporting(chat_id=str(policy.get("chat_id") or ""))
            policies.append(policy)
        result["policies"] = policies
    return result


registry.register(
    name="weixin_group_policy",
    toolset="messaging",
    schema=WEIXIN_GROUP_POLICY_SCHEMA,
    handler=weixin_group_policy_tool,
    check_fn=_check_send_message,
    emoji="🧭",
)

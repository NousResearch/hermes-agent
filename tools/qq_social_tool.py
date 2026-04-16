"""QQ/NapCat social request and profile control tool."""

from __future__ import annotations

import json

from gateway.config import Platform
from gateway.qq_social_policy import (
    clear_social_policy,
    describe_social_policy_state,
    get_social_policy,
    set_social_policy,
)
from gateway.qq_social_requests import (
    describe_social_request_state,
    get_social_request,
    list_social_requests,
    parse_social_request_key,
    summarize_social_requests,
    update_social_request_status,
)
from tools.qq_group_tool_common import (
    require_admin_session,
    resolve_delivery_target,
    session_actor_label,
)
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, _error, _qq_napcat_call


QQ_SOCIAL_TOOL_SCHEMA = {
    "name": "qq_social_control",
    "description": (
        "Manage QQ/NapCat social requests and contact lookups. "
        "Use this to inspect pending friend/group requests, approve or reject them, "
        "list friends, query a QQ user profile via get_stranger_info, or configure "
        "automatic handling for inbound friend/group requests."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_requests",
                    "get_request",
                    "approve_request",
                    "reject_request",
                    "list_friends",
                    "get_user_profile",
                    "get_social_policy",
                    "set_social_policy",
                    "clear_social_policy",
                ],
                "description": "QQ social control action to perform.",
            },
            "request_key": {
                "type": "string",
                "description": "Stored QQ social request key in the form 'friend:<flag>' or 'group:<flag>'.",
            },
            "request_type": {
                "type": "string",
                "enum": ["friend", "group"],
                "description": "Optional filter when action='list_requests'.",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "approved", "rejected", "ignored"],
                "description": "Optional request status filter when action='list_requests'.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of social requests or friends to return. Defaults to 20.",
            },
            "user_id": {
                "type": "string",
                "description": "QQ user id for action='get_user_profile'.",
            },
            "no_cache": {
                "type": "boolean",
                "description": "Whether get_user_profile should bypass cached profile data.",
            },
            "message": {
                "type": "string",
                "description": "Optional decision note. Used as group-reject reason or friend-approve remark when supported.",
            },
            "auto_approve_friend_requests": {
                "type": "boolean",
                "description": "Whether new inbound QQ friend requests should be approved automatically.",
            },
            "auto_approve_group_add_requests": {
                "type": "boolean",
                "description": "Whether inbound QQ group join requests should be approved automatically.",
            },
            "auto_approve_group_invites": {
                "type": "boolean",
                "description": "Whether inbound QQ group invitation requests should be approved automatically.",
            },
            "notify_target": {
                "type": "string",
                "description": "Optional target for automatic social-request handling notices. Use current_chat, current_user_dm, none, qq_napcat:group:<id>, or qq_napcat:dm:<id>.",
            },
            "notes": {
                "type": "string",
                "description": "Operator note stored alongside the social auto-handling policy.",
            },
        },
        "required": ["action"],
    },
}


def qq_social_tool(args, **kw):
    del kw

    action = str(args.get("action") or "").strip().lower()
    if action not in {
        "list_requests",
        "get_request",
        "approve_request",
        "reject_request",
        "list_friends",
        "get_user_profile",
        "get_social_policy",
        "set_social_policy",
        "clear_social_policy",
    }:
        return tool_error(
            "Unsupported action. Use 'list_requests', 'get_request', 'approve_request', "
            "'reject_request', 'list_friends', 'get_user_profile', "
            "'get_social_policy', 'set_social_policy', or 'clear_social_policy'."
        )

    from tools.interrupt import is_interrupted

    if is_interrupted():
        return tool_error("Interrupted")

    admin_error = require_admin_session("处理 QQ 社交请求或查询社交资料")
    if admin_error:
        return json.dumps(_error(admin_error), ensure_ascii=False)

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
        if action == "list_requests":
            normalized_limit = _normalize_limit(args.get("limit"))
            requests = list_social_requests(
                status=args.get("status"),
                request_type=args.get("request_type"),
                limit=normalized_limit,
            )
            enriched_requests = [_with_request_state(item) for item in requests]
            return json.dumps(
                {
                    "success": True,
                    "requests": enriched_requests,
                    "filters": {
                        "status": _normalize_optional_filter(args.get("status")),
                        "request_type": _normalize_optional_filter(args.get("request_type")),
                        "limit": normalized_limit,
                    },
                    "summary": summarize_social_requests(enriched_requests),
                },
                ensure_ascii=False,
            )

        if action == "get_request":
            request_key = _require_request_key(args.get("request_key"))
            request = get_social_request(request_key)
            if request is None:
                raise ValueError(f"QQ social request '{request_key}' does not exist.")
            return json.dumps({"success": True, "request": _with_request_state(request)}, ensure_ascii=False)

        if action == "get_social_policy":
            policy = get_social_policy()
            return json.dumps(
                {
                    "success": True,
                    "policy": policy,
                    "policy_state": describe_social_policy_state(policy),
                },
                ensure_ascii=False,
            )

        if action == "list_friends":
            result = _run_async_napcat_call(pconfig.extra, "get_friend_list", {})
            rows = result if isinstance(result, list) else []
            friends = []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                user_id = str(item.get("user_id") or item.get("uin") or "").strip()
                if not user_id:
                    continue
                friends.append(
                    {
                        "user_id": user_id,
                        "nickname": str(item.get("nickname") or item.get("nick") or "").strip() or user_id,
                        "remark": str(item.get("remark") or item.get("remark_name") or "").strip() or None,
                    }
                )
            return json.dumps({"success": True, "friends": friends[: _normalize_limit(args.get("limit"))]}, ensure_ascii=False)

        if action == "get_user_profile":
            user_id = _normalize_user_id(args.get("user_id"))
            profile = _run_async_napcat_call(
                pconfig.extra,
                "get_stranger_info",
                {"user_id": int(user_id), "no_cache": bool(args.get("no_cache", False))},
            )
            if not isinstance(profile, dict):
                raise ValueError("QQ NapCat did not return a valid user profile.")
            return json.dumps(
                {
                    "success": True,
                    "profile": {
                        "user_id": str(profile.get("user_id") or user_id),
                        "nickname": str(profile.get("nickname") or profile.get("nick") or user_id).strip(),
                        "sex": str(profile.get("sex") or "").strip() or None,
                        "age": profile.get("age"),
                        "level": profile.get("level"),
                        "raw_profile": profile,
                    },
                },
                ensure_ascii=False,
            )

        if action == "clear_social_policy":
            policy = clear_social_policy(updated_by=session_actor_label())
            return json.dumps(
                {
                    "success": True,
                    "policy": policy,
                    "policy_state": describe_social_policy_state(policy),
                },
                ensure_ascii=False,
            )

        if action == "set_social_policy":
            policy = set_social_policy(
                auto_approve_friend_requests=args.get("auto_approve_friend_requests"),
                auto_approve_group_add_requests=args.get("auto_approve_group_add_requests"),
                auto_approve_group_invites=args.get("auto_approve_group_invites"),
                notify_target=resolve_delivery_target(args.get("notify_target")),
                notes=str(args.get("notes") or "").strip() or None,
                updated_by=session_actor_label(),
            )
            return json.dumps(
                {
                    "success": True,
                    "policy": policy,
                    "policy_state": describe_social_policy_state(policy),
                },
                ensure_ascii=False,
            )

        request_key = _require_request_key(args.get("request_key"))
        request = get_social_request(request_key)
        if request is None:
            raise ValueError(f"QQ social request '{request_key}' does not exist.")
        if str(request.get("status") or "pending").strip().lower() != "pending":
            raise ValueError(f"QQ social request '{request_key}' has already been handled.")
        request_type, flag = parse_social_request_key(request_key)
        decision_message = str(args.get("message") or "").strip() or None
        approve = action == "approve_request"

        if request_type == "group":
            params = {
                "flag": flag,
                "sub_type": str(request.get("sub_type") or "add").strip() or "add",
                "approve": approve,
            }
            if not approve and decision_message:
                params["reason"] = decision_message
            raw = _run_async_napcat_call(pconfig.extra, "set_group_add_request", params)
        else:
            params = {
                "flag": flag,
                "approve": approve,
            }
            if approve and decision_message:
                params["remark"] = decision_message
            raw = _run_async_napcat_call(pconfig.extra, "set_friend_add_request", params)

        updated = update_social_request_status(
            request_key,
            status="approved" if approve else "rejected",
            handled_by=session_actor_label(),
            handled_via="manual_tool",
            note=decision_message,
        )
        return json.dumps(
            {
                "success": True,
                "request": _with_request_state(updated),
                "raw_response": raw or {},
            },
            ensure_ascii=False,
        )
    except ValueError as exc:
        return json.dumps(_error(str(exc)), ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"QQ social action failed: {exc}"), ensure_ascii=False)


def _normalize_limit(value) -> int:
    try:
        parsed = int(value or 20)
    except (TypeError, ValueError):
        raise ValueError("'limit' must be an integer.") from None
    return max(1, min(parsed, 200))


def _normalize_optional_filter(value) -> str | None:
    text = str(value or "").strip().lower()
    return text or None


def _require_request_key(value) -> str:
    request_key = str(value or "").strip()
    if not request_key:
        raise ValueError("'request_key' is required.")
    return request_key


def _normalize_user_id(value) -> str:
    user_id = str(value or "").strip()
    if not user_id or not user_id.lstrip("-").isdigit():
        raise ValueError("'user_id' must be a numeric QQ user id.")
    return user_id


def _run_async_napcat_call(extra: dict, action: str, params: dict):
    from model_tools import _run_async

    data, error = _run_async(_qq_napcat_call(extra, action, params))
    if error:
        raise ValueError(str(error.get("error") or error))
    return data


def _with_request_state(request: dict) -> dict:
    enriched = dict(request or {})
    enriched["request_state"] = describe_social_request_state(enriched)
    return enriched


registry.register(
    name="qq_social_control",
    toolset="messaging",
    schema=QQ_SOCIAL_TOOL_SCHEMA,
    handler=qq_social_tool,
    check_fn=_check_send_message,
    emoji="🤝",
)

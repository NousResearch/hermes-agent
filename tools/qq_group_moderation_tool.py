"""QQ/NapCat group moderation tool."""

from __future__ import annotations

import json
import os

from gateway.config import Platform
from tools.approval import request_dangerous_action_approval
from tools.qq_group_tool_common import resolve_group_target as _resolve_group_target
from tools.registry import registry, tool_error
from tools.send_message_tool import _check_send_message, _error, _qq_napcat_call


_MAX_MUTE_SECONDS_DEFAULT = 24 * 60 * 60
_ACTION_ALIASES = {
    "mute": "mute_user",
    "mute_member": "mute_user",
    "ban": "mute_user",
    "ban_user": "mute_user",
    "ban_member": "mute_user",
    "kick": "kick_user",
    "kick_member": "kick_user",
    "remove_user": "kick_user",
    "remove_member": "kick_user",
}


QQ_GROUP_MODERATION_SCHEMA = {
    "name": "qq_group_moderation",
    "description": (
        "Moderate a QQ/NapCat group by muting or kicking a member. "
        "Prefer routing model-facing QQ moderation requests through qq_control, which dispatches here "
        "while preserving the same approval and protected-user guardrails. "
        "Use this tool directly for QQ group mute/kick requests instead of writing scripts, "
        "shell commands, or raw NapCat API calls. "
        "Only works for QQ groups, requires a reason, and every action must "
        "be explicitly approved by the configured administrator before execution. "
        "The tool refuses to act on group owners, group admins, protected users, or the bot itself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["mute_user", "kick_user", "mute", "kick"],
                "description": "Moderation action to perform.",
            },
            "target": {
                "type": "string",
                "description": (
                    "QQ group target. Accepts 'group:123456', 'qq_napcat:group:123456', or a numeric group id. "
                    "If omitted, Hermes will only use the current QQ group session."
                ),
            },
            "user_id": {
                "type": "string",
                "description": "Numeric QQ user ID of the member to moderate.",
            },
            "user_query": {
                "type": "string",
                "description": "QQ member selector. Accepts a numeric QQ user ID, group card, or nickname when the exact QQ user ID is not known.",
            },
            "duration_seconds": {
                "type": "integer",
                "description": "Mute duration in seconds when action='mute_user'. Required for mute_user.",
            },
            "duration_minutes": {
                "type": "integer",
                "description": "Mute duration in minutes when action='mute_user'. Shortcut-friendly alias for duration_seconds.",
            },
            "reason": {
                "type": "string",
                "description": "Short moderator reason explaining why the action is needed. Required.",
            },
        },
        "required": ["action", "reason"],
    },
}


def qq_group_moderation_tool(args, **kw):
    """Handle QQ/NapCat group moderation operations."""
    del kw

    action = _normalize_action(args.get("action"))
    if action not in {"mute_user", "kick_user"}:
        return tool_error("Unsupported action. Use 'mute_user' or 'kick_user'.")

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
        group_id = _resolve_group_target(args.get("target"))
        user_id, user_query = _normalize_user_selector(
            args.get("user_id"),
            args.get("user_query"),
        )
        reason = _normalize_reason(args.get("reason"))
        duration_seconds = _normalize_duration_seconds(
            _coerce_duration_seconds(args),
            action=action,
            max_mute_seconds=_get_max_mute_seconds(pconfig.extra),
        )
    except ValueError as exc:
        return json.dumps(_error(str(exc)), ensure_ascii=False)

    try:
        from model_tools import _run_async

        result = _run_async(
            _dispatch_group_moderation_action(
                action=action,
                extra=pconfig.extra,
                group_id=group_id,
                user_id=user_id,
                user_query=user_query,
                duration_seconds=duration_seconds,
                reason=reason,
            )
        )
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(_error(f"QQ group moderation failed: {exc}"), ensure_ascii=False)


def _normalize_action(value) -> str:
    action = str(value or "").strip().lower()
    return _ACTION_ALIASES.get(action, action)


def _normalize_user_id(value, *, arg_name: str) -> str:
    user_id = str(value or "").strip()
    if not user_id:
        raise ValueError(f"'{arg_name}' is required.")
    if not user_id.isdigit():
        raise ValueError(f"'{arg_name}' must be a numeric QQ user ID.")
    return user_id


def _normalize_user_selector(user_id_value, user_query_value) -> tuple[str | None, str | None]:
    user_id_text = str(user_id_value or "").strip()
    user_query_text = str(user_query_value or "").strip()
    if not user_id_text and not user_query_text:
        raise ValueError("'user_id' or 'user_query' is required.")
    user_id = _normalize_user_id(user_id_text, arg_name="user_id") if user_id_text else None
    user_query = user_query_text[:100] if user_query_text else None
    return user_id, user_query


def _normalize_reason(value) -> str:
    reason = str(value or "").strip()
    if not reason:
        raise ValueError("'reason' is required.")
    return reason[:200]


def _coerce_duration_seconds(args: dict) -> int | str | None:
    if args.get("duration_seconds") is not None and str(args.get("duration_seconds")).strip() != "":
        return args.get("duration_seconds")

    minutes = args.get("duration_minutes")
    if minutes is None or str(minutes).strip() == "":
        return None
    try:
        return int(minutes) * 60
    except (TypeError, ValueError):
        return minutes


def _get_max_mute_seconds(extra: dict) -> int:
    raw = extra.get("max_mute_seconds", extra.get("moderation_max_mute_seconds", _MAX_MUTE_SECONDS_DEFAULT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = _MAX_MUTE_SECONDS_DEFAULT
    return max(60, value)


def _normalize_duration_seconds(value, *, action: str, max_mute_seconds: int) -> int | None:
    if action != "mute_user":
        return None
    if value is None or str(value).strip() == "":
        raise ValueError("'duration_seconds' is required when action='mute_user'.")
    try:
        duration_seconds = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("'duration_seconds' must be a positive integer.") from exc
    if duration_seconds <= 0:
        raise ValueError("'duration_seconds' must be greater than zero.")
    if duration_seconds > max_mute_seconds:
        raise ValueError(f"'duration_seconds' exceeds the configured maximum of {max_mute_seconds} seconds.")
    return duration_seconds


async def _dispatch_group_moderation_action(
    *,
    action: str,
    extra: dict,
    group_id: str,
    user_id: str | None,
    user_query: str | None,
    duration_seconds: int | None,
    reason: str,
) -> dict:
    numeric_group_id = int(group_id)
    resolved_user_id, resolve_error = await _resolve_user_id(
        extra=extra,
        group_id=numeric_group_id,
        user_id=user_id,
        user_query=user_query,
    )
    if resolve_error:
        return _error(resolve_error)
    if not resolved_user_id:
        return _error("Failed to resolve the QQ user to moderate.")

    numeric_user_id = int(resolved_user_id)

    member_data, member_error = await _qq_napcat_call(
        extra,
        "get_group_member_info",
        {"group_id": numeric_group_id, "user_id": numeric_user_id, "no_cache": True},
    )
    if member_error:
        return member_error
    if not isinstance(member_data, dict) or not member_data:
        return _error("QQ NapCat did not return member information for the requested user.")

    protection_error = await _check_target_safety(extra, numeric_group_id, resolved_user_id, member_data)
    if protection_error:
        return _error(protection_error)

    approval = request_dangerous_action_approval(
        action_preview=_build_action_preview(
            action=action,
            group_id=group_id,
            user_id=resolved_user_id,
            duration_seconds=duration_seconds,
            reason=reason,
        ),
        description=_build_action_description(
            action=action,
            group_id=group_id,
            user_id=resolved_user_id,
            duration_seconds=duration_seconds,
            reason=reason,
        ),
        prompt_title="这事得先请董事长拍板",
        approver_name="董事长",
    )
    if not approval.get("approved"):
        return _error(str(approval.get("message") or "这事没拿到董事长授权，先不执行。"))

    if action == "mute_user":
        data, error = await _qq_napcat_call(
            extra,
            "set_group_ban",
            {"group_id": numeric_group_id, "user_id": numeric_user_id, "duration": int(duration_seconds or 0)},
        )
        if error:
            return error
        return {
            "success": True,
            "approval_required": True,
            "approved": True,
            "platform": "qq_napcat",
            "action": action,
            "capability": "supported",
            "target": f"group:{group_id}",
            "group_id": group_id,
            "subject_id": resolved_user_id,
            "subject_name": _member_display_name(member_data),
            "user_id": resolved_user_id,
            "duration_seconds": int(duration_seconds or 0),
            "duration_minutes": max(1, int(duration_seconds or 0) // 60) if duration_seconds else None,
            "reason": reason,
            "member_role": str(member_data.get("role") or "member"),
            "member_name": _member_display_name(member_data),
            "raw_response": data or {},
        }

    data, error = await _qq_napcat_call(
        extra,
        "set_group_kick",
        {"group_id": numeric_group_id, "user_id": numeric_user_id, "reject_add_request": False},
    )
    if error:
        return error
    return {
        "success": True,
        "approval_required": True,
        "approved": True,
        "platform": "qq_napcat",
        "action": action,
        "capability": "supported",
        "target": f"group:{group_id}",
        "group_id": group_id,
        "subject_id": resolved_user_id,
        "subject_name": _member_display_name(member_data),
        "user_id": resolved_user_id,
        "reason": reason,
        "member_role": str(member_data.get("role") or "member"),
        "member_name": _member_display_name(member_data),
        "raw_response": data or {},
    }


async def _resolve_user_id(
    *,
    extra: dict,
    group_id: int,
    user_id: str | None,
    user_query: str | None,
) -> tuple[str | None, str | None]:
    if user_id:
        return user_id, None

    query = str(user_query or "").strip()
    if not query:
        return None, "'user_id' or 'user_query' is required."
    if query.isdigit():
        return query, None

    member_list, member_error = await _qq_napcat_call(
        extra,
        "get_group_member_list",
        {"group_id": group_id},
    )
    if member_error:
        return None, str(member_error.get("error") or "Failed to resolve QQ group member.")
    if not isinstance(member_list, list):
        return None, "QQ NapCat returned an invalid group member list."

    matches = _match_group_members(member_list, query)
    if not matches:
        return None, f"没有在这个 QQ 群里精确找到“{query}”，请改用明确的 QQ 号。"
    if len(matches) > 1:
        candidates = "、".join(_member_candidate_label(member) for member in matches[:5])
        return None, f"群里有多个成员匹配“{query}”：{candidates}。请改用明确的 QQ 号。"

    resolved = str((matches[0] or {}).get("user_id") or "").strip()
    if not resolved.isdigit():
        return None, "QQ NapCat returned a matched member without a numeric QQ user ID."
    return resolved, None


def _match_group_members(member_list: list[dict], query: str) -> list[dict]:
    normalized = _normalize_member_query(query)
    matches: list[dict] = []
    for member in member_list:
        if not isinstance(member, dict):
            continue
        member_values = {
            _normalize_member_query(member.get("user_id")),
            _normalize_member_query(member.get("card")),
            _normalize_member_query(member.get("nickname")),
        }
        member_values.discard("")
        if normalized in member_values:
            matches.append(member)
    return matches


def _normalize_member_query(value) -> str:
    text = str(value or "").strip()
    if text.startswith("@"):
        text = text[1:].strip()
    return text.casefold()


def _member_candidate_label(member: dict) -> str:
    name = _member_display_name(member)
    user_id = str(member.get("user_id") or "").strip()
    if name and user_id and name != user_id:
        return f"{name}({user_id})"
    return user_id or name or "unknown"


async def _check_target_safety(extra: dict, group_id: int, user_id: str, member_data: dict) -> str | None:
    role = str(member_data.get("role") or "member").strip().lower()
    if role in {"owner", "admin"}:
        return f"Refusing to moderate QQ group {role} user {user_id}."

    protected_ids = _protected_user_ids(extra)
    if user_id in protected_ids:
        return f"Refusing to moderate protected QQ user {user_id}."

    login_data, login_error = await _qq_napcat_call(extra, "get_login_info", {})
    if login_error is not None:
        return "Refusing to moderate QQ users because the bot identity could not be verified."
    if not isinstance(login_data, dict):
        return "Refusing to moderate QQ users because the bot identity response was invalid."

    bot_user_id = str(login_data.get("user_id") or "").strip()
    if not bot_user_id:
        return "Refusing to moderate QQ users because the bot identity response was incomplete."
    if bot_user_id == user_id:
        return "Refusing to moderate the bot's own QQ account."

    if int(member_data.get("group_id") or group_id) != group_id:
        return "QQ NapCat returned member information for a different group."

    return None


def _protected_user_ids(extra: dict) -> set[str]:
    protected: set[str] = set()

    def _merge(value) -> None:
        if isinstance(value, str):
            protected.update(item.strip() for item in value.split(",") if item.strip())
        elif isinstance(value, (list, tuple, set)):
            protected.update(str(item).strip() for item in value if str(item).strip())

    for key in ("admin_users", "protected_users"):
        _merge(extra.get(key))

    for env_key in (
        "HERMES_SESSION_ADMIN_USER_IDS",
        "GATEWAY_ADMIN_USERS",
        "QQ_NAPCAT_ADMIN_USERS",
    ):
        _merge(os.getenv(env_key, ""))

    return protected


def _member_display_name(member_data: dict) -> str:
    return (
        str(member_data.get("card") or "").strip()
        or str(member_data.get("nickname") or "").strip()
        or str(member_data.get("user_id") or "").strip()
    )


def _build_action_preview(*, action: str, group_id: str, user_id: str, duration_seconds: int | None, reason: str) -> str:
    parts = [f"qq_group_moderation {action}", f"group:{group_id}", f"user:{user_id}"]
    if duration_seconds is not None:
        parts.append(f"duration:{duration_seconds}")
    parts.append(f"reason:{reason}")
    return " ".join(parts)


def _build_action_description(*, action: str, group_id: str, user_id: str, duration_seconds: int | None, reason: str) -> str:
    if action == "mute_user":
        return (
            f"将 QQ 群 {group_id} 的成员 {user_id} 禁言 {int(duration_seconds or 0)} 秒。原因：{reason}"
        )
    return f"将 QQ 群 {group_id} 的成员 {user_id} 移出群聊。原因：{reason}"


registry.register(
    name="qq_group_moderation",
    toolset="messaging",
    schema=QQ_GROUP_MODERATION_SCHEMA,
    handler=qq_group_moderation_tool,
    check_fn=_check_send_message,
    emoji="🛡️",
)

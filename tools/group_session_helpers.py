"""Platform-neutral helpers for session-derived delivery targets."""

from __future__ import annotations

import os
from typing import Callable


def require_admin_session(action: str) -> str | None:
    platform = str(os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    if not platform:
        return None

    flag = str(os.getenv("HERMES_SESSION_IS_ADMIN") or "").strip().lower()
    if flag in {"true", "1", "yes", "on"}:
        return None
    return f"这类操作得先请董事长拍板，当前会话没有管理员授权，不能{action}。"


def session_actor_label() -> str:
    user_id = str(os.getenv("HERMES_SESSION_USER_ID") or "").strip()
    user_name = str(os.getenv("HERMES_SESSION_USER_NAME") or "").strip()
    if user_name and user_id:
        return f"{user_name}({user_id})"
    return user_name or user_id or "unknown"


def current_chat_delivery_target() -> str:
    platform = str(os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    chat_id = str(os.getenv("HERMES_SESSION_CHAT_ID") or "").strip()
    chat_type = str(os.getenv("HERMES_SESSION_CHAT_TYPE") or "").strip().lower()
    thread_id = str(os.getenv("HERMES_SESSION_THREAD_ID") or "").strip()

    if not platform or not chat_id:
        raise ValueError("Current session is missing a concrete chat target.")

    if platform == "qq_napcat":
        prefix = "group" if chat_type == "group" else "dm"
        return f"qq_napcat:{prefix}:{chat_id}"

    if thread_id:
        return f"{platform}:{chat_id}:{thread_id}"
    return f"{platform}:{chat_id}"


def current_session_target_parts() -> dict[str, str | None] | None:
    platform = str(os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    chat_id = str(os.getenv("HERMES_SESSION_CHAT_ID") or "").strip()
    chat_type = str(os.getenv("HERMES_SESSION_CHAT_TYPE") or "").strip().lower() or None
    thread_id = str(os.getenv("HERMES_SESSION_THREAD_ID") or "").strip() or None

    if not platform or platform == "local" or not chat_id:
        return None
    return {
        "platform_name": platform,
        "chat_id": chat_id,
        "chat_type": chat_type,
        "thread_id": thread_id,
    }


def current_user_dm_delivery_target(
    *,
    custom_resolver: Callable[[str, str], str] | None = None,
) -> str:
    platform = str(os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    user_id = str(os.getenv("HERMES_SESSION_USER_ID") or "").strip()
    if not platform:
        raise ValueError("Current session is missing HERMES_SESSION_PLATFORM.")
    if not user_id:
        raise ValueError("Current session is missing HERMES_SESSION_USER_ID, cannot resolve current_user_dm.")

    if custom_resolver is not None:
        return custom_resolver(platform, user_id)
    if platform == "qq_napcat":
        return f"qq_napcat:dm:{user_id}"
    return f"{platform}:{user_id}"


def resolve_delivery_target(
    target,
    *,
    allow_none: bool = True,
    shorthand_platform: str | None = None,
    current_user_dm_resolver: Callable[[str, str], str] | None = None,
) -> str | None:
    explicit = str(target or "").strip()
    if not explicit:
        return None if allow_none else current_chat_delivery_target()

    lowered = explicit.lower()
    if lowered in {"none", "off", "disabled", "disable", "clear"}:
        return ""
    if lowered in {"current_chat", "this_chat", "current_group", "this_group"}:
        return current_chat_delivery_target()
    if lowered in {"current_user_dm", "dm_me", "private_me", "my_dm"}:
        return current_user_dm_delivery_target(custom_resolver=current_user_dm_resolver)

    if explicit.startswith(("group:", "dm:")):
        if not shorthand_platform:
            raise ValueError(
                "Unsupported shorthand delivery target without a platform context. "
                "Use an explicit target like platform:group:<id>."
            )
        return f"{shorthand_platform}:{explicit}"
    if ":" in explicit:
        return explicit

    raise ValueError(
        "Unsupported delivery target. Use current_chat, current_user_dm, none, "
        "or an explicit target like qq_napcat:group:<id> / qq_napcat:dm:<id>."
    )


def tool_error_json(message: str) -> dict[str, str]:
    return {"error": str(message)}

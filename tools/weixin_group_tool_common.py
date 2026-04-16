"""Shared helpers for Weixin group control tools."""

from __future__ import annotations

import os

from tools.group_scope_helpers import (
    resolve_group_chat_id,
    resolve_optional_group_chat_id,
)
from tools.group_session_helpers import (
    current_chat_delivery_target,
    current_user_dm_delivery_target,
    require_admin_session,
    resolve_delivery_target as _resolve_delivery_target,
    session_actor_label,
    tool_error_json as _tool_error_json,
)


WEIXIN_GROUP_PLATFORM = "weixin"


def _normalize_explicit_weixin_group_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Weixin group target is required.")
    if text.startswith("weixin:"):
        text = text.split(":", 1)[1].strip()
    if text.startswith("group:"):
        text = text.split(":", 1)[1].strip()
    if not text.endswith("@chatroom"):
        raise ValueError("Weixin group target must look like '<chatroom>@chatroom'.")
    return text


def resolve_optional_group_target(target) -> str | None:
    return resolve_optional_group_chat_id(target, explicit_resolver=_normalize_explicit_weixin_group_id)


def resolve_group_target(target) -> str:
    return resolve_group_chat_id(
        target,
        expected_platform=WEIXIN_GROUP_PLATFORM,
        explicit_resolver=_normalize_explicit_weixin_group_id,
        missing_target_error=(
            "No Weixin group target specified. This tool only defaults to the current Weixin group session."
            if os.getenv("HERMES_SESSION_PLATFORM", "").strip().lower() == WEIXIN_GROUP_PLATFORM
            else "No Weixin group target specified. Use target='group@chatroom' or 'weixin:group@chatroom'."
        ),
    )


def resolve_delivery_target(target, *, allow_none: bool = True) -> str | None:
    return _resolve_delivery_target(
        target,
        allow_none=allow_none,
        shorthand_platform=WEIXIN_GROUP_PLATFORM,
    )


def tool_error_json(message: str) -> str:
    return _tool_error_json(message)


__all__ = [
    "WEIXIN_GROUP_PLATFORM",
    "current_chat_delivery_target",
    "current_user_dm_delivery_target",
    "require_admin_session",
    "resolve_delivery_target",
    "resolve_group_target",
    "resolve_optional_group_target",
    "session_actor_label",
    "tool_error_json",
]

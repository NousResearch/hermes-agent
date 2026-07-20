"""Shared helpers for QQ/NapCat group control tools."""

from __future__ import annotations

import os

from gateway.platforms.qq_napcat import resolve_qq_napcat_group_id
from tools.group_scope_helpers import (
    current_group_chat_id,
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

QQ_GROUP_PLATFORM = "qq_napcat"


def resolve_optional_group_target(target) -> str | None:
    return resolve_optional_group_chat_id(
        target,
        explicit_resolver=lambda value: str(resolve_qq_napcat_group_id(value)),
    )


def resolve_group_target(target) -> str:
    return resolve_group_chat_id(
        target,
        expected_platform=QQ_GROUP_PLATFORM,
        explicit_resolver=lambda value: str(resolve_qq_napcat_group_id(value)),
        missing_target_error=(
            "No QQ group target specified. This tool only defaults to the current QQ group session."
            if os.getenv("HERMES_SESSION_PLATFORM", "").strip().lower() == QQ_GROUP_PLATFORM
            else "No QQ group target specified. Use target='group:<id>'."
        ),
    )


def resolve_delivery_target(target, *, allow_none: bool = True) -> str | None:
    return _resolve_delivery_target(
        target,
        allow_none=allow_none,
        shorthand_platform=QQ_GROUP_PLATFORM,
    )


def tool_error_json(message: str) -> str:
    return _tool_error_json(message)

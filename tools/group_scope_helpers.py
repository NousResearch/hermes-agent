"""Helpers for resolving group chat targets from session or explicit input."""

from __future__ import annotations

import os
from typing import Callable

from gateway.group_policy_store import normalize_group_scope_key, split_group_scope_key


def current_group_scope_key(*, expected_platform: str | None = None) -> str:
    platform = str(os.getenv("HERMES_SESSION_PLATFORM") or "").strip().lower()
    chat_type = str(os.getenv("HERMES_SESSION_CHAT_TYPE") or "").strip().lower()
    chat_id = str(os.getenv("HERMES_SESSION_CHAT_ID") or "").strip()

    if not platform:
        raise ValueError("Current session is missing HERMES_SESSION_PLATFORM.")
    if expected_platform and platform != str(expected_platform).strip().lower():
        raise ValueError(f"Current session is not a {expected_platform} group session.")
    if chat_type != "group":
        raise ValueError("Current session is not a group session.")
    if not chat_id:
        raise ValueError("Current group session is missing HERMES_SESSION_CHAT_ID.")

    return normalize_group_scope_key(platform, chat_id)


def current_group_chat_id(*, expected_platform: str | None = None) -> str:
    _platform, chat_id = split_group_scope_key(current_group_scope_key(expected_platform=expected_platform))
    return chat_id


def resolve_optional_group_chat_id(
    target,
    *,
    explicit_resolver: Callable[[str], str],
) -> str | None:
    explicit = str(target or "").strip()
    if not explicit:
        return None
    resolved = str(explicit_resolver(explicit) or "").strip()
    if not resolved:
        raise ValueError("Failed to resolve group target.")
    return resolved


def resolve_group_chat_id(
    target,
    *,
    expected_platform: str,
    explicit_resolver: Callable[[str], str],
    home_chat_id: str | None = None,
    missing_target_error: str | None = None,
) -> str:
    resolved = resolve_optional_group_chat_id(target, explicit_resolver=explicit_resolver)
    if resolved:
        return resolved

    if str(home_chat_id or "").strip():
        return str(explicit_resolver(str(home_chat_id).strip()) or "").strip()

    try:
        return current_group_chat_id(expected_platform=expected_platform)
    except ValueError as exc:
        if missing_target_error:
            raise ValueError(missing_target_error) from exc
        raise

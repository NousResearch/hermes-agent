"""Discord REST v10 guild-settings request builder (approved scalars only).

This module is deliberately transport-free. It validates the scalar guild
settings Hermes exposes and returns a request descriptor for the request-owned
``discord_guild_settings`` consumer.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Final

__all__ = [
    "AFK_TIMEOUT_VALUES",
    "DESCRIPTION_MAX",
    "EDITABLE_GUILD_SETTINGS",
    "GuildSettingsError",
    "NAME_MAX",
    "edit_guild_request",
]

NAME_MIN: Final = 2
NAME_MAX: Final = 100
DESCRIPTION_MAX: Final = 1024
VERIFICATION_LEVEL_VALUES: Final = frozenset(range(5))
DEFAULT_MESSAGE_NOTIFICATION_VALUES: Final = frozenset(range(2))
EXPLICIT_CONTENT_FILTER_VALUES: Final = frozenset(range(3))
AFK_TIMEOUT_VALUES: Final = frozenset({60, 300, 900, 1800, 3600})

_SNOWFLAKE_MAX: Final = (1 << 64) - 1
_SNOWFLAKE_RE: Final = re.compile(r"^[0-9]{1,64}$")


class GuildSettingsError(ValueError):
    """Raised when a guild-settings request violates the exposed contract."""


def _validate_snowflake(value: Any, field: str) -> str:
    """Return a positive Discord snowflake in canonical decimal-string form."""
    if isinstance(value, bool):
        raise GuildSettingsError(f"{field!r} must be a snowflake, not a bool")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and _SNOWFLAKE_RE.fullmatch(value):
        parsed = int(value, 10)
    else:
        raise GuildSettingsError(
            f"{field!r} must be a snowflake (int or decimal str), "
            f"got {type(value).__name__}"
        )

    if not 1 <= parsed <= _SNOWFLAKE_MAX:
        raise GuildSettingsError(f"{field!r} snowflake out of range: {value!r}")

    # Discord's wire contract represents snowflakes as decimal strings. This
    # also removes ambiguous zero padding from otherwise equivalent identities.
    return str(parsed)


def _validate_name(value: Any) -> str:
    if not isinstance(value, str):
        raise GuildSettingsError("'name' must be a string")
    if value != value.strip():
        raise GuildSettingsError("'name' cannot have leading or trailing whitespace")
    if not NAME_MIN <= len(value) <= NAME_MAX:
        raise GuildSettingsError(
            f"'name' must be between {NAME_MIN} and {NAME_MAX} characters"
        )
    return value


def _validate_str(
    value: Any,
    field: str,
    max_len: int,
    *,
    allow_none: bool = False,
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise GuildSettingsError(f"{field!r} must be a string")
    if len(value) > max_len:
        raise GuildSettingsError(f"{field!r} exceeds {max_len} characters")
    return value


def _validate_int_enum(
    value: Any,
    field: str,
    allowed: frozenset[int],
    *,
    allow_none: bool = False,
) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuildSettingsError(f"{field!r} must be an integer")
    if value not in allowed:
        allowed_text = ", ".join(str(item) for item in sorted(allowed))
        raise GuildSettingsError(f"{field!r} must be one of: {allowed_text}")
    return value


def _validate_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise GuildSettingsError(f"{field!r} must be a boolean")
    return value


def _validate_optional_snowflake(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _validate_snowflake(value, field)


_FIELD_VALIDATORS: Dict[str, Callable[[Any], Any]] = {
    "name": _validate_name,
    "description": lambda value: _validate_str(
        value,
        "description",
        DESCRIPTION_MAX,
        allow_none=True,
    ),
    "verification_level": lambda value: _validate_int_enum(
        value,
        "verification_level",
        VERIFICATION_LEVEL_VALUES,
        allow_none=True,
    ),
    "default_message_notifications": lambda value: _validate_int_enum(
        value,
        "default_message_notifications",
        DEFAULT_MESSAGE_NOTIFICATION_VALUES,
        allow_none=True,
    ),
    "explicit_content_filter": lambda value: _validate_int_enum(
        value,
        "explicit_content_filter",
        EXPLICIT_CONTENT_FILTER_VALUES,
        allow_none=True,
    ),
    "premium_progress_bar_enabled": lambda value: _validate_bool(
        value,
        "premium_progress_bar_enabled",
    ),
    "afk_channel_id": lambda value: _validate_optional_snowflake(
        value,
        "afk_channel_id",
    ),
    "system_channel_id": lambda value: _validate_optional_snowflake(
        value,
        "system_channel_id",
    ),
    "rules_channel_id": lambda value: _validate_optional_snowflake(
        value,
        "rules_channel_id",
    ),
    "public_updates_channel_id": lambda value: _validate_optional_snowflake(
        value,
        "public_updates_channel_id",
    ),
    "safety_alerts_channel_id": lambda value: _validate_optional_snowflake(
        value,
        "safety_alerts_channel_id",
    ),
    "afk_timeout": lambda value: _validate_int_enum(
        value,
        "afk_timeout",
        AFK_TIMEOUT_VALUES,
    ),
}

EDITABLE_GUILD_SETTINGS: Final = frozenset(_FIELD_VALIDATORS)


def edit_guild_request(guild_id: Any, **fields: Any) -> Dict[str, Any]:
    """Build a validated ``PATCH /guilds/{guild_id}`` request descriptor.

    Snowflakes are normalized to canonical decimal strings. At least one
    editable field is required; an empty PATCH is never emitted.
    """
    canonical_guild_id = _validate_snowflake(guild_id, "guild_id")

    payload: Dict[str, Any] = {}
    for key, value in fields.items():
        validator = _FIELD_VALIDATORS.get(key)
        if validator is None:
            raise GuildSettingsError(f"unsupported guild setting: {key!r}")
        payload[key] = validator(value)

    if not payload:
        raise GuildSettingsError("no guild settings provided")

    return {
        "method": "PATCH",
        "path": f"/guilds/{canonical_guild_id}",
        "json": payload,
    }

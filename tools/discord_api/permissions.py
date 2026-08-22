"""Discord channel-permission overwrite REST request builders (API v10).

Discord serializes snowflakes and permission bitfields as decimal strings in
HTTP payloads. These helpers validate and canonicalize that wire contract while
remaining transport-free.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Union

__all__ = [
    "MAX_SNOWFLAKE",
    "PermissionOverwriteError",
    "TYPE_MEMBER",
    "TYPE_ROLE",
    "delete_channel_permission_request",
    "set_channel_permission_request",
]

MAX_SNOWFLAKE: int = (1 << 64) - 1
MAX_PERMISSION_BITS: int = (1 << 64) - 1

# Discord REST v10: 0 = role, 1 = member.
TYPE_ROLE: int = 0
TYPE_MEMBER: int = 1
_VALID_TYPES = (TYPE_ROLE, TYPE_MEMBER)
_DECIMAL_RE = re.compile(r"^\d{1,64}$")


class PermissionOverwriteError(ValueError):
    """Raised when a permission-overwrite request cannot be built."""


def _canonical_decimal(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> str:
    if isinstance(value, bool):
        raise PermissionOverwriteError(f"{name} must be a decimal integer, got {value!r}")
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and _DECIMAL_RE.fullmatch(value):
        number = int(value)
    else:
        raise PermissionOverwriteError(f"{name} must be decimal digits, got {value!r}")
    if number < minimum or number > maximum:
        raise PermissionOverwriteError(
            f"{name} must be within [{minimum}, {maximum}], got {value!r}"
        )
    return str(number)


def _validate_snowflake(value: Any, name: str) -> str:
    return _canonical_decimal(
        value,
        name,
        minimum=1,
        maximum=MAX_SNOWFLAKE,
    )


def _validate_bitfield(value: Any, name: str) -> str:
    return _canonical_decimal(
        value,
        name,
        minimum=0,
        maximum=MAX_PERMISSION_BITS,
    )


def _validate_type(type_: Any) -> int:
    if isinstance(type_, bool) or not isinstance(type_, int) or type_ not in _VALID_TYPES:
        raise PermissionOverwriteError(
            f"type_ must be 0 (role) or 1 (member), got {type_!r}"
        )
    return type_


def set_channel_permission_request(
    channel_id: Union[int, str],
    overwrite_id: Union[int, str],
    *,
    allow: Union[int, str] = 0,
    deny: Union[int, str] = 0,
    type_: int,
) -> Dict[str, Any]:
    """Build ``PUT /channels/{channel}/permissions/{overwrite}``."""
    channel = _validate_snowflake(channel_id, "channel_id")
    overwrite = _validate_snowflake(overwrite_id, "overwrite_id")
    return {
        "method": "PUT",
        "path": f"/channels/{channel}/permissions/{overwrite}",
        "payload": {
            "allow": _validate_bitfield(allow, "allow"),
            "deny": _validate_bitfield(deny, "deny"),
            "type": _validate_type(type_),
        },
    }


def delete_channel_permission_request(
    channel_id: Union[int, str],
    overwrite_id: Union[int, str],
) -> Dict[str, Any]:
    """Build ``DELETE /channels/{channel}/permissions/{overwrite}``."""
    channel = _validate_snowflake(channel_id, "channel_id")
    overwrite = _validate_snowflake(overwrite_id, "overwrite_id")
    return {
        "method": "DELETE",
        "path": f"/channels/{channel}/permissions/{overwrite}",
        "payload": None,
    }

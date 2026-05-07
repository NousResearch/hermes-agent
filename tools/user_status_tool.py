#!/usr/bin/env python3
"""User Status Tool — cross-bot user-state read/write.

Thin tool layer over ``agent/user_status.py``. Lets the agent inspect and
update a small shared snapshot of user state (device mode, AFK status,
focus project, quiet hours, location) so multiple gateway profiles
(Telegram, Discord, Slack, ...) stay coherent.

Two operations exposed via a single ``user_status`` tool with an
``action`` parameter:

* ``get`` — return the current state as JSON.
* ``set`` — write one field via ``user_status.save_field()`` and return
  the post-write state. Field name is validated against the
  :class:`UserStatus` dataclass; unknown fields are rejected with a
  clear error.

The ``writer`` argument identifies which profile/platform performed the
write. If the caller doesn't pass one explicitly, the tool falls back to
``$HERMES_PROFILE`` (the same env var ``kanban_tools.py`` uses) and then
to ``"agent"`` so writes are always attributed.

Issue #21122 / kanban t_315c0bfc (Task B).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import fields as dc_fields
from typing import Any, Optional

from agent import user_status as _user_status_mod
from agent.user_status import UserStatus, load, save_field
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------

# Metadata fields that the storage layer manages itself; the tool never
# lets the agent overwrite these directly.
_RESERVED_FIELDS = {"per_field_updated_at", "updated_by"}


def _allowed_fields() -> tuple[str, ...]:
    """Return the user-writable field names from the UserStatus dataclass."""
    return tuple(
        f.name for f in dc_fields(UserStatus) if f.name not in _RESERVED_FIELDS
    )


# ---------------------------------------------------------------------------
# Writer attribution
# ---------------------------------------------------------------------------


def _resolve_writer(explicit: Optional[str]) -> str:
    """Pick a writer tag: explicit arg → $HERMES_PROFILE → 'agent'."""
    if explicit:
        w = str(explicit).strip()
        if w:
            return w
    env = os.environ.get("HERMES_PROFILE", "").strip()
    if env:
        return env
    return "agent"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def get_user_status() -> str:
    """Return the current cross-bot user-status snapshot as a JSON string."""
    try:
        status = load()
        return json.dumps(status.to_dict(), ensure_ascii=False, sort_keys=True)
    except Exception as e:
        logger.exception("get_user_status failed: %s", e)
        return tool_error(f"failed to read user_status: {type(e).__name__}: {e}")


def set_user_status(field: str, value: Any, writer: Optional[str] = None) -> str:
    """Write a single field via :func:`agent.user_status.save_field`.

    Returns the updated snapshot as a JSON string, or a JSON error.
    """
    if not field or not isinstance(field, str):
        return tool_error("'field' is required and must be a string.")

    allowed = _allowed_fields()
    if field in _RESERVED_FIELDS:
        return tool_error(
            f"field '{field}' is reserved metadata and cannot be set directly. "
            f"Allowed fields: {list(allowed)}"
        )
    if field not in allowed:
        return tool_error(
            f"unknown user_status field '{field}'. Allowed fields: {list(allowed)}"
        )

    writer_tag = _resolve_writer(writer)
    try:
        status = save_field(field, value, writer_tag)
    except ValueError as e:
        # Defensive: storage layer may add new validation in future.
        return tool_error(str(e))
    except Exception as e:
        logger.exception("set_user_status failed: %s", e)
        return tool_error(f"failed to write user_status: {type(e).__name__}: {e}")

    return json.dumps(status.to_dict(), ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Tool entry point + schema
# ---------------------------------------------------------------------------


def user_status_tool(
    action: str,
    field: Optional[str] = None,
    value: Any = None,
    writer: Optional[str] = None,
) -> str:
    """Single dispatcher for the ``user_status`` tool."""
    if action == "get":
        return get_user_status()
    if action == "set":
        if not field:
            return tool_error("'field' is required for action='set'.")
        return set_user_status(field, value, writer=writer)
    return tool_error(
        f"unknown action '{action}'. Use 'get' or 'set'.",
        allowed_actions=["get", "set"],
    )


def check_user_status_requirements() -> bool:
    """No third-party deps — always available."""
    return True


USER_STATUS_SCHEMA = {
    "name": "user_status",
    "description": (
        "Read or write a small cross-bot snapshot of the user's current "
        "state (device mode, AFK, focus project, quiet hours, location). "
        "Persisted in a single JSON file shared by every gateway profile "
        "so Telegram / Discord / Slack / etc. stay coherent.\n\n"
        "Actions:\n"
        "- 'get': return the current snapshot, including per-field "
        "timestamps and the last writer.\n"
        "- 'set': update ONE field. Requires 'field' (one of: "
        "device_mode, afk_status, focus_project, quiet_hours_until, "
        "location) and 'value'. The active profile is recorded as the "
        "writer automatically; other fields are preserved.\n\n"
        "Use this when the user tells you something durable about their "
        "current context ('I'm afk till 5', 'switching to mobile', "
        "'focusing on project X') so other bots can react accordingly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set"],
                "description": "Operation to perform.",
            },
            "field": {
                "type": "string",
                "enum": list(_allowed_fields()),
                "description": "Field name (required for action='set').",
            },
            "value": {
                "description": (
                    "New value for the field (required for action='set'). "
                    "Typically a string; pass null to clear."
                ),
            },
            "writer": {
                "type": "string",
                "description": (
                    "Optional override for the writer tag stored as "
                    "'updated_by'. Defaults to the active profile name."
                ),
            },
        },
        "required": ["action"],
    },
}


# --- Registry ---
registry.register(
    name="user_status",
    toolset="user_status",
    schema=USER_STATUS_SCHEMA,
    handler=lambda args, **kw: user_status_tool(
        action=args.get("action", ""),
        field=args.get("field"),
        value=args.get("value"),
        writer=args.get("writer"),
    ),
    check_fn=check_user_status_requirements,
    emoji="📡",
)


__all__ = [
    "get_user_status",
    "set_user_status",
    "user_status_tool",
    "USER_STATUS_SCHEMA",
]

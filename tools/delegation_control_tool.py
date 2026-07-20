#!/usr/bin/env python3
"""LLM-facing control surface for background delegation lifecycles."""

from __future__ import annotations

import json
from typing import Optional

from tools.async_delegation import (
    _DEFAULT_STALLED_AFTER_SECONDS,
    get_async_delegation,
    interrupt_async_delegation,
    list_async_delegations,
)
from tools.approval import get_current_session_key
from tools.delegate_tool import is_spawn_paused, set_spawn_paused
from tools.registry import registry


def _configured_stalled_after_seconds() -> float:
    """Read delegation.stalled_after_seconds with a safe positive fallback."""
    try:
        from tools.delegate_tool import _load_config

        value = _load_config().get("stalled_after_seconds")
        if value is not None:
            parsed = float(value)
            if parsed > 0:
                return parsed
    except (TypeError, ValueError):
        pass
    except Exception:
        pass
    return _DEFAULT_STALLED_AFTER_SECONDS


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def delegation_control(action: str, delegation_id: Optional[str] = None) -> str:
    """List, inspect, cancel, pause, or resume async delegations."""
    normalized_action = str(action or "").strip().lower()
    threshold = _configured_stalled_after_seconds()
    owner_session_key = get_current_session_key(default="")

    if normalized_action == "list":
        return _json(
            {
                "action": "list",
                "spawn_paused": is_spawn_paused(owner_session_key),
                "delegations": list_async_delegations(
                    owner_session_key=owner_session_key,
                    stalled_after_seconds=threshold,
                ),
            }
        )

    if normalized_action in {"status", "cancel"} and not str(
        delegation_id or ""
    ).strip():
        return _json(
            {
                "error": f"delegation_id is required for action='{normalized_action}'.",
                "action": normalized_action,
            }
        )

    if normalized_action == "status":
        identifier = str(delegation_id).strip()
        record = get_async_delegation(
            identifier,
            owner_session_key=owner_session_key,
            stalled_after_seconds=threshold,
        )
        if record is None:
            return _json(
                {
                    "error": f"Unknown delegation_id: {identifier}",
                    "status": "not_found",
                    "delegation_id": identifier,
                }
            )
        return _json({"action": "status", "delegation": record})

    if normalized_action == "cancel":
        identifier = str(delegation_id).strip()
        result = interrupt_async_delegation(
            identifier,
            reason="delegation_control tool",
            owner_session_key=owner_session_key,
        )
        if result.get("status") == "not_found":
            result = {
                **result,
                "error": f"Unknown delegation_id: {identifier}",
            }
        return _json(result)

    if normalized_action == "pause":
        return _json(
            {
                "action": "pause",
                "spawn_paused": set_spawn_paused(True, owner_session_key),
            }
        )

    if normalized_action == "resume":
        return _json(
            {
                "action": "resume",
                "spawn_paused": set_spawn_paused(False, owner_session_key),
            }
        )

    return _json(
        {
            "error": (
                f"Unsupported action: {action!r}. Use one of: "
                "list, status, cancel, pause, resume."
            )
        }
    )


DELEGATION_CONTROL_SCHEMA = {
    "name": "delegation_control",
    "description": (
        "Inspect and control background delegations by handle. List running and "
        "recent tasks, inspect liveness/workspace state, precisely cancel one "
        "delegation, or pause/resume new delegation spawns."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "status", "cancel", "pause", "resume"],
                "description": "Lifecycle control action.",
            },
            "delegation_id": {
                "type": "string",
                "description": (
                    "Delegation handle required for status/cancel, for example "
                    "deleg_ab12cd34."
                ),
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="delegation_control",
    toolset="delegation_control",
    schema=DELEGATION_CONTROL_SCHEMA,
    handler=lambda args, **kw: delegation_control(
        action=args.get("action", ""),
        delegation_id=args.get("delegation_id"),
    ),
    emoji="🎛️",
)

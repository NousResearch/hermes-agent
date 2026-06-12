"""Model-facing control-plane tools.

These wrappers expose the durable local control-plane DB without asking the
model to shell out through the terminal tool. Mutations are intentionally gated:
callers must pass an explicit root or run in a control-plane worker context.
"""
from __future__ import annotations

import json
from typing import Any

from hermes_cli import control_db as cp
from tools.registry import registry, tool_error


def _check_control_plane_mode() -> bool:
    # Opt-in by profile toolset or by worker/PM environment. This keeps normal
    # chat sessions from seeing an extra control surface unless configured.
    import os
    if os.environ.get("HERMES_CONTROL_INSTANCE_ID") or os.environ.get("HERMES_CONTROL_PROFILE_ID"):
        return True
    try:
        from hermes_cli.config import load_config
        return "control_plane" in (load_config().get("toolsets") or [])
    except Exception:
        return False


def _connect(args: dict[str, Any]):
    root = args.get("root")
    from pathlib import Path
    return cp.connect(root=Path(root) if root else None)


def _handle_control_plane_status(args: dict[str, Any], **_: Any) -> str:
    try:
        conn = _connect(args)
        try:
            if args.get("action") == "emit":
                event_id = cp.emit_status(
                    conn,
                    instance_id=args["instance_id"],
                    dispatch_id=args.get("dispatch_id"),
                    status=args["status"],
                    summary=args["summary"],
                    details=args.get("details") or {},
                )
                return json.dumps({"event_id": event_id})
            return json.dumps({"events": cp.list_status_events(conn, dispatch_id=args.get("dispatch_id"), profile_id=args.get("profile_id"), limit=int(args.get("limit") or 50))})
        finally:
            conn.close()
    except Exception as exc:
        return tool_error(str(exc), tool_name="control_plane_status")


def _handle_control_plane_blocker(args: dict[str, Any], **_: Any) -> str:
    try:
        conn = _connect(args)
        try:
            action = args.get("action")
            if action == "open":
                blocker_id = cp.open_blocker(
                    conn,
                    dispatch_id=args["dispatch_id"],
                    instance_id=args["instance_id"],
                    severity=args.get("severity") or "blocked",
                    kind=args.get("kind") or "other",
                    summary=args["summary"],
                    details=args.get("details") or {},
                    response_profile=args.get("response_profile"),
                )
                return json.dumps({"blocker_id": blocker_id})
            if action == "resolve":
                ok = cp.resolve_blocker(conn, args["blocker_id"], resolver_instance_id=args["resolver_instance_id"], resolution=args.get("resolution") or {})
                return json.dumps({"resolved": ok})
            return json.dumps({"blockers": cp.list_blockers(conn, dispatch_id=args.get("dispatch_id"), status=args.get("status"), response_profile=args.get("response_profile"), limit=int(args.get("limit") or 50))})
        finally:
            conn.close()
    except Exception as exc:
        return tool_error(str(exc), tool_name="control_plane_blocker")


def _handle_control_plane_message(args: dict[str, Any], **_: Any) -> str:
    try:
        conn = _connect(args)
        try:
            action = args.get("action") or "list"
            if action == "list":
                clauses = []
                params: list[Any] = []
                if args.get("receiver"):
                    clauses.append("receiver_profile=?")
                    params.append(args["receiver"])
                if args.get("status"):
                    clauses.append("status=?")
                    params.append(args["status"])
                if args.get("kind"):
                    clauses.append("kind=?")
                    params.append(args["kind"])
                where = "WHERE " + " AND ".join(clauses) if clauses else ""
                limit = int(args.get("limit") or 50)
                rows = conn.execute(f"SELECT * FROM cp_messages {where} ORDER BY created_at_ms DESC LIMIT ?", (*params, limit)).fetchall()
                return json.dumps({"messages": [dict(r) for r in rows]})
            status_map = {"ack": "acknowledged", "resolve": "resolved", "supersede": "superseded", "cancel": "cancelled"}
            if action not in status_map:
                raise ValueError(f"unknown message action: {action}")
            if not args.get("actor_instance_id"):
                raise PermissionError("message mutation requires actor_instance_id")
            if not args.get("message_id"):
                raise ValueError("message mutation requires message_id")
            result = cp.transition_message_status(
                conn,
                args["message_id"],
                status=status_map[action],
                actor_instance_id=args.get("actor_instance_id"),
                actor_profile=args.get("actor_profile"),
                actor_type=args.get("actor_type") or "receiver",
                reason=args.get("reason"),
                metadata=args.get("metadata") or {},
            )
            return json.dumps(result)
        finally:
            conn.close()
    except Exception as exc:
        return tool_error(str(exc), tool_name="control_plane_message")


CONTROL_PLANE_STATUS_SCHEMA = {
    "name": "control_plane_status",
    "description": "Emit or list structured Hermes control-plane status events in the local DB.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["emit", "list"], "default": "list"},
            "root": {"type": "string", "description": "Optional Hermes root for an offline/temp control DB."},
            "instance_id": {"type": "string"},
            "dispatch_id": {"type": "string"},
            "profile_id": {"type": "string"},
            "status": {"type": "string"},
            "summary": {"type": "string"},
            "details": {"type": "object"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["action"],
    },
}

CONTROL_PLANE_BLOCKER_SCHEMA = {
    "name": "control_plane_blocker",
    "description": "Open, resolve, or list Hermes control-plane blockers in the local DB.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["open", "resolve", "list"], "default": "list"},
            "root": {"type": "string"},
            "dispatch_id": {"type": "string"},
            "instance_id": {"type": "string"},
            "blocker_id": {"type": "string"},
            "resolver_instance_id": {"type": "string"},
            "severity": {"type": "string", "enum": ["info", "warning", "blocked", "critical"]},
            "kind": {"type": "string", "enum": ["approval_needed", "missing_context", "test_failure", "review_failure", "dependency", "auth", "policy", "runtime", "other"]},
            "summary": {"type": "string"},
            "details": {"type": "object"},
            "resolution": {"type": "object"},
            "response_profile": {"type": "string"},
            "status": {"type": "string"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["action"],
    },
}

CONTROL_PLANE_MESSAGE_SCHEMA = {
    "name": "control_plane_message",
    "description": "List or terminally close Hermes control-plane messages with audited receiver/admin authorization.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "ack", "resolve", "supersede", "cancel"], "default": "list"},
            "root": {"type": "string"},
            "message_id": {"type": "string"},
            "receiver": {"type": "string"},
            "kind": {"type": "string"},
            "status": {"type": "string"},
            "actor_instance_id": {"type": "string"},
            "actor_profile": {"type": "string"},
            "actor_type": {"type": "string", "enum": ["receiver", "admin", "bootstrap"], "default": "receiver"},
            "reason": {"type": "string"},
            "metadata": {"type": "object"},
            "limit": {"type": "integer", "default": 50},
        },
        "required": ["action"],
    },
}

registry.register(name="control_plane_status", toolset="control_plane", schema=CONTROL_PLANE_STATUS_SCHEMA, handler=_handle_control_plane_status, check_fn=_check_control_plane_mode, emoji="🛂")
registry.register(name="control_plane_blocker", toolset="control_plane", schema=CONTROL_PLANE_BLOCKER_SCHEMA, handler=_handle_control_plane_blocker, check_fn=_check_control_plane_mode, emoji="🚧")
registry.register(name="control_plane_message", toolset="control_plane", schema=CONTROL_PLANE_MESSAGE_SCHEMA, handler=_handle_control_plane_message, check_fn=_check_control_plane_mode, emoji="✉️")

"""Interactive Discord approval boxes for outbound deliverables.

The tool creates a durable approval record under ``HERMES_HOME/approval_boxes``
and asks the live Discord adapter to render an Approve / Needs Work / Reject
component view. A click is authorized by the adapter's normal user/role gate,
then atomically finalizes the record and disables the controls.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error, tool_result

_RECORD_LOCK = Lock()
_MAX_TITLE = 180
_MAX_BODY = 6_000


DISCORD_APPROVAL_BOX_SCHEMA = {
    "name": "discord_approval_box",
    "description": (
        "Create an interactive Discord approval box for an external-facing "
        "deliverable. The box has Approve, Needs Work, and Reject controls. "
        "Use it before releasing public posts, client communications, or files. "
        "For deliverables, provide the Google Drive URL so the reviewer can open it."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short description of the action awaiting approval."},
            "body": {"type": "string", "description": "What will be released, scope, and any material risk or limitation."},
            "drive_url": {"type": "string", "description": "Google Drive URL for the deliverable. Required when a file is being reviewed."},
            "channel_id": {"type": "string", "description": "Discord channel or thread ID. Omit to use the current Discord conversation."},
        },
        "required": ["title", "body"],
    },
}


def _approval_dir() -> Path:
    return get_hermes_home() / "approval_boxes"


def _record_path(approval_id: str) -> Path:
    return _approval_dir() / f"{approval_id}.json"


def _safe_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _write_record(record: Dict[str, Any]) -> None:
    directory = _approval_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = _record_path(record["id"])
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def create_approval_record(*, title: str, body: str, drive_url: str, channel_id: str) -> Dict[str, Any]:
    """Persist a new pending approval record and return it."""
    record = {
        "id": uuid.uuid4().hex[:12],
        "status": "pending",
        "title": _safe_text(title, _MAX_TITLE),
        "body": _safe_text(body, _MAX_BODY),
        "drive_url": _safe_text(drive_url, 2_000),
        "channel_id": str(channel_id),
        "created_at": time.time(),
        "resolved_at": None,
        "resolved_by": None,
    }
    with _RECORD_LOCK:
        _write_record(record)
    return record


def get_approval_record(approval_id: str) -> Optional[Dict[str, Any]]:
    path = _record_path(str(approval_id))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def resolve_approval(approval_id: str, decision: str, resolved_by: str) -> Optional[Dict[str, Any]]:
    """Finalize a pending approval. First authorized decision wins."""
    if decision not in {"approved", "needs_work", "rejected"}:
        return None
    with _RECORD_LOCK:
        record = get_approval_record(approval_id)
        if not record or record.get("status") != "pending":
            return None
        record["status"] = decision
        record["resolved_at"] = time.time()
        record["resolved_by"] = _safe_text(resolved_by, 200) or "authorized user"
        _write_record(record)
        return record


def _current_target(args: Dict[str, Any]) -> str:
    explicit = str(args.get("channel_id") or "").strip()
    if explicit:
        return explicit
    try:
        from gateway.session_context import get_session_env
        thread_id = get_session_env("HERMES_SESSION_THREAD_ID", "").strip()
        chat_id = get_session_env("HERMES_SESSION_CHAT_ID", "").strip()
        return thread_id or chat_id
    except Exception:
        return ""


def check_discord_approval_box_requirements() -> bool:
    """Expose only when this profile is configured with a Discord bot."""
    return bool(os.getenv("DISCORD_BOT_TOKEN", "").strip())


def discord_approval_box_tool(args: Dict[str, Any], **_kw: Any) -> str:
    title = _safe_text(args.get("title"), _MAX_TITLE)
    body = _safe_text(args.get("body"), _MAX_BODY)
    drive_url = _safe_text(args.get("drive_url"), 2_000)
    channel_id = _current_target(args)
    if not title or not body:
        return tool_error("Both title and body are required.")
    if not channel_id:
        return tool_error("No Discord channel could be resolved. Pass channel_id explicitly.")

    try:
        from gateway.run import _gateway_runner_ref
        from gateway.config import Platform
        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform("discord")) if runner is not None else None
    except Exception:
        adapter = None
    if adapter is None or not callable(getattr(adapter, "send_deliverable_approval", None)):
        return tool_error(
            "Interactive Discord approval boxes require the live Discord gateway adapter. "
            "The current gateway has not loaded that capability."
        )

    record = create_approval_record(
        title=title, body=body, drive_url=drive_url, channel_id=channel_id,
    )
    try:
        # discord.py owns an aiohttp session bound to the gateway's event
        # loop. _run_async creates a private loop, which breaks channel.send.
        from agent.async_utils import safe_schedule_threadsafe
        future = safe_schedule_threadsafe(adapter.send_deliverable_approval(
            chat_id=channel_id,
            title=record["title"],
            body=record["body"],
            drive_url=record["drive_url"],
            approval_id=record["id"],
        ), getattr(adapter, "_event_loop", None))
        if future is None:
            raise RuntimeError("Discord gateway event loop is unavailable")
        result = future.result(timeout=30)
    except Exception as exc:
        try:
            _record_path(record["id"]).unlink(missing_ok=True)
        except OSError:
            pass
        return tool_error(f"Could not deliver approval box: {exc}")
    if not getattr(result, "success", False):
        try:
            _record_path(record["id"]).unlink(missing_ok=True)
        except OSError:
            pass
        return tool_error(f"Could not deliver approval box: {getattr(result, 'error', 'unknown error')}")
    return tool_result(
        success=True,
        approval_id=record["id"],
        message_id=getattr(result, "message_id", None),
        status="pending",
    )


registry.register(
    name="discord_approval_box",
    toolset="discord",
    schema=DISCORD_APPROVAL_BOX_SCHEMA,
    handler=discord_approval_box_tool,
    check_fn=check_discord_approval_box_requirements,
    requires_env=["DISCORD_BOT_TOKEN"],
    description="Interactive Discord approval controls for outbound deliverables",
    emoji="✅",
)

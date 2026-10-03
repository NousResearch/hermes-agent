"""The server half of an MCP Apps host (stable spec 2026-01-26): the client extension Hermes
advertises, a tool's ``_meta.ui`` and its visibility, and the view record.

A view record is what an MCP App view of one model tool call needs from Hermes: the server and the
tool the call ran on, the arguments sent and the raw ``CallToolResult``. The MCP handler opens it
while the call runs, keyed by the ``(session_id, tool_call_id)`` the tool executor binds;
``agent/tool_executor.py::_commit_tool_result`` moves it onto the tool row as
``display_metadata["mcp_app"]``, where ``tui_gateway/methods_mcp_apps.py`` reads it back."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Optional

# A record whose call never reaches the commit point (an interrupted turn) is evicted, not kept.
_MAX_RECORDS = 256
_records: "OrderedDict[tuple[str, str], dict]" = OrderedDict()
_lock = threading.Lock()


def client_extensions() -> dict:
    """``ClientSession(extensions=...)``: Hermes hosts ``text/html;profile=mcp-app`` views (spec 1498-1522)."""
    from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID

    return {EXTENSION_ID: {"mimeTypes": [APP_MIME_TYPE]}}


def tool_ui(tool: Any) -> Optional[dict]:
    """``_meta.ui`` of a live SDK ``Tool`` or a schema-cache stand-in, the deprecated flat
    ``_meta["ui/resourceUri"]`` folded in (spec 325-347); None when the tool declares neither."""
    meta = getattr(tool, "meta", None)
    if not isinstance(meta, dict):
        return None
    ui = dict(meta["ui"]) if isinstance(meta.get("ui"), dict) else {}
    if "resourceUri" not in ui and isinstance(meta.get("ui/resourceUri"), str):
        ui["resourceUri"] = meta["ui/resourceUri"]
    return ui or None


def visible_to(tool: Any, audience: str) -> bool:
    """Whether ``_meta.ui.visibility`` admits *audience* (``"model"`` or ``"app"``); it defaults to
    both (spec 397)."""
    visibility = (tool_ui(tool) or {}).get("visibility")
    return audience in visibility if isinstance(visibility, list) else True


def live_tool(server: Any, tool_name: str) -> Any:
    """The server's live ``Tool`` named *tool_name*, or None."""
    return next((t for t in getattr(server, "_tools", ()) if t.name == tool_name), None)


def wire(model: Any) -> dict:
    """An SDK model as it travelled: the fields the server sent, under their wire names."""
    return model.model_dump(by_alias=True, mode="json", exclude_unset=True)


def open_record(server_name: str, server: Any, tool_name: str, arguments: dict) -> Optional[tuple[str, str]]:
    """Open the view record of the model call about to send *arguments*, when its live tool
    declares a UI resource; returns the record key, or None (no view, or no executor-bound call:
    a view's own ``tools/call`` has no row to land on)."""
    from tools.approval_context import _approval_session_id, _approval_tool_call_id

    key = (_approval_session_id.get(), _approval_tool_call_id.get())
    if not all(key):
        return None
    if not isinstance((tool_ui(live_tool(server, tool_name)) or {}).get("resourceUri"), str):
        return None
    with _lock:
        _records[key] = {"server": server_name, "tool": tool_name, "arguments": arguments}
        _records.move_to_end(key)
        while len(_records) > _MAX_RECORDS:
            _records.popitem(last=False)
    return key


def record_result(key: Optional[tuple[str, str]], result: Any) -> None:
    """Keep the raw ``CallToolResult`` on the record; one over the MCP hard cap is not stored, so
    its view ends cancelled rather than carrying megabytes on the row."""
    if key is None:
        return
    from tools.mcp_tool_content import _MCP_HARD_RESULT_CAP_CHARS

    payload = wire(result)
    if len(json.dumps(payload, ensure_ascii=False)) > _MCP_HARD_RESULT_CAP_CHARS:
        return
    with _lock:
        if key in _records:
            _records[key]["result"] = payload


def running_record(session_id: str, tool_call_id: str) -> Optional[dict]:
    """The record of a call that has not reached its commit point yet, else None."""
    with _lock:
        record = _records.get((session_id, tool_call_id))
        return dict(record) if record is not None else None


def forget_record(session_id: str, tool_call_id: str) -> None:
    """Drop the in-memory record once its tool row is durable."""
    with _lock:
        _records.pop((session_id, tool_call_id), None)

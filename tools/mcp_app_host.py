"""The server half of an MCP Apps host (stable spec 2026-01-26): the client extension Hermes
advertises, a tool's ``_meta.ui`` and its visibility, and the view slot.

A view is a function of one model tool call's server, tool, arguments and raw ``CallToolResult``;
of these the tool row lacks only the raw result. The tool executor binds a fresh slot around each
call's dispatch (``agent/tool_executor.py::_ToolCallRef.view``), the MCP handler fills it when the
call's live tool declares a UI resource, and ``_commit_tool_result`` copies it onto the tool row as
``display_metadata["mcp_app"]``, where ``tui_gateway/methods_mcp_apps.py`` reads it back."""

from __future__ import annotations

import contextlib
import json
from contextvars import ContextVar
from typing import Any, Iterator, Optional

# (the dispatched call's registry name, its view slot): a tool another tool dispatches in-process
# (``execute_code``) runs in the same context but under another name, so it never fills the slot.
_slot: ContextVar[Optional[tuple[str, dict]]] = ContextVar("mcp_app_view_slot", default=None)


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


@contextlib.contextmanager
def bind_slot(tool_name: str, slot: dict) -> Iterator[None]:
    """Bind *slot* as the view slot of the call to *tool_name* (registry name) for one dispatch."""
    token = _slot.set((tool_name, slot))
    try:
        yield
    finally:
        _slot.reset(token)


def open_record(server_name: str, server: Any, tool_name: str, arguments: dict) -> Optional[dict]:
    """Fill the bound view slot of the model call about to send *arguments* when its live tool
    declares a UI resource; returns the slot, or None (no view, or no slot for this call: a view's
    own ``tools/call`` runs outside the executor and has no row to land on)."""
    from tools.mcp_tool_schema import mcp_prefixed_tool_name

    bound = _slot.get()
    if bound is None or bound[0] != mcp_prefixed_tool_name(server_name, tool_name):
        return None
    if not isinstance((tool_ui(live_tool(server, tool_name)) or {}).get("resourceUri"), str):
        return None
    slot = bound[1]
    slot.update(server=server_name, tool=tool_name, arguments=arguments)
    return slot


def record_result(slot: Optional[dict], result: Any) -> None:
    """Keep the raw ``CallToolResult`` in the slot; one over the MCP hard cap is not stored, so its
    view ends cancelled rather than carrying megabytes on the row."""
    if slot is None:
        return
    from tools.mcp_tool_content import _MCP_HARD_RESULT_CAP_CHARS

    payload = wire(result)
    if len(json.dumps(payload, ensure_ascii=False)) <= _MCP_HARD_RESULT_CAP_CHARS:
        slot["result"] = payload

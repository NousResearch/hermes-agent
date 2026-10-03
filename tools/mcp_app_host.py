"""The server half of an MCP Apps host (stable spec 2026-01-26): the client extension Hermes
advertises, and a tool's ``_meta.ui`` and its visibility."""

from __future__ import annotations

from typing import Any, Optional


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

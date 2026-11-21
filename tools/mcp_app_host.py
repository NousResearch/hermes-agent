"""MCP Apps tool metadata (stable spec 2026-01-26): a tool's ``_meta.ui`` and its visibility."""

from __future__ import annotations

from typing import Any, Optional


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

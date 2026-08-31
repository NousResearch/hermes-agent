#!/usr/bin/env python3
"""The `desktop_preview` tool — the preview pane beside the chat, as ONE tool.

Consolidation (#95681, maintainer-directed): open_preview, close_preview,
and read_preview each re-taught "the preview pane beside the chat" world;
one action enum states it once (576 -> ~210 tok). The read action keeps
its agent-callback dispatch (agent_runtime_helpers routes action=read
through agent.read_preview_callback, same as read_preview did).

Lives in the ``desktop_ui`` toolset — desktop-app sessions only.
"""

import json

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target, open_preview_tool
from tools.registry import registry, tool_error


def preview_open(url: str, label: str = "") -> str:
    return open_preview_tool(url=url, label=label)


def preview_close(url: str = "") -> str:
    target = _normalize_target(url or "")
    try:
        ok = desktop_ui.emit("preview.close", {"url": target})
    except Exception as exc:  # noqa: BLE001
        return tool_error(f"Failed to close the preview: {exc}")
    if not ok:
        return tool_error("The preview pane is only available in the Hermes desktop app.")
    return json.dumps({"success": True, "closed": target or "all"}, ensure_ascii=False)


def _handle_preview(args, **kw):
    """Non-read actions only: action=read is dispatched at the agent level
    (needs the GUI callback), mirroring the old read_preview special path."""
    action = (args.get("action") or "").strip()
    if action == "open":
        return preview_open(url=args.get("url", ""), label=args.get("label", ""))
    if action == "close":
        return preview_close(url=args.get("url", ""))
    if action == "read":
        return tool_error(
            "preview read must run inside a desktop session (no GUI callback here)."
        )
    return tool_error("action must be one of: open, close, read.")


PREVIEW_SCHEMA = {
    "name": "desktop_preview",
    "description": (
        "The preview pane beside the chat in the Hermes desktop app. open: show "
        "a web URL (bare domains fine), a localhost dev server, or a file path "
        "(HTML renders live) — opens for the current window only. close: dismiss "
        "the whole pane, or one tab via url. read: what the pane currently shows "
        "— returns {kind, url, title, text, start, end, total_chars}; a Browser "
        "tab's text is the rendered page's visible text, paged with start/count "
        "(char offsets); a file tab answers identity only (read the file with "
        "read_file)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["open", "close", "read"]},
            "url": {
                "type": "string",
                "description": "open: the target. close: one tab (omit for the whole pane).",
            },
            "label": {"type": "string", "description": "open: optional tab label."},
            "start": {"type": "integer", "description": "read: 0-indexed char offset."},
            "count": {"type": "integer", "description": "read: chars to return (capped per read)."},
        },
        "required": ["action"],
    },
}


registry.register(
    name="desktop_preview",
    toolset="desktop_ui",
    schema=PREVIEW_SCHEMA,
    handler=_handle_preview,
    emoji="🖼️",
)

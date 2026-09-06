#!/usr/bin/env python3
"""The `desktop_preview` tool — the preview pane beside the chat, as ONE tool.

open/close/read used to be three tools that each re-taught "the preview pane" world;
one action enum states it once (~576 -> ~210 schema tokens). action=read keeps its
agent-level callback dispatch (agent_runtime_helpers -> agent.read_preview_callback).
Lives in the ``desktop_ui`` toolset — desktop-app sessions only.
"""

from tools import desktop_ui
from tools.open_preview_tool import _normalize_target, open_preview_tool
from tools.registry import registry, tool_error


def preview_close(url: str = "") -> str:
    target = _normalize_target(url or "")
    return desktop_ui.emit_or_error(
        "preview.close",
        {"url": target},
        "Failed to close the preview: ",
        "The preview pane is only available in the Hermes desktop app.",
        {"success": True, "closed": target or "all"},
    )


_ACTIONS = {
    "open": lambda args: open_preview_tool(url=args.get("url", ""), label=args.get("label", "")),
    "close": lambda args: preview_close(url=args.get("url", "")),
    # read needs the GUI callback and is dispatched at the agent level.
    "read": lambda args: tool_error(
        "preview read must run inside a desktop session (no GUI callback here)."),
}


def _handle_preview(args, **kw):
    """Non-read actions only: action=read is dispatched at the agent level."""
    fn = _ACTIONS.get((args.get("action") or "").strip())
    if fn is None:
        return tool_error("action must be one of: open, close, read.")
    return fn(args)


PREVIEW_SCHEMA = {
    "name": "desktop_preview",
    "description": (
        "Open, close, or read the preview pane beside the chat. open: show "
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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import json  # noqa: F401,E402

def preview_open(url: str, label: str = "") -> str:
    return open_preview_tool(url=url, label=label)
# ---- END PLUGIN-COMPAT ----

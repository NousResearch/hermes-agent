#!/usr/bin/env python3
"""Propose an MCP server to the user as an inline card in the desktop chat.

The card (install / enable / authorize + decline) lives in the desktop renderer, so this
tool round-trips through the gateway's blocking-prompt bridge (the one ``clarify`` uses):
tui_gateway emits ``mcp.setup.request``, the renderer walks the user through the existing
REST flows (catalog install, enable, OAuth) and answers with ``mcp.setup.respond``. Lives in
the ``desktop_ui`` toolset, which the GUI gateway enables only for desktop-sourced sessions;
elsewhere the agent falls back to ``hermes mcp install <name>`` in the terminal.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

_ACTIONS = ("install", "enable", "authorize")


def setup_mcp_tool(server: str = "", action: str = "install", reason: str = "", callback: Optional[Callable] = None) -> str:
    """Ask the desktop GUI to run an MCP setup flow; return its JSON outcome."""
    if callback is None:
        # Still down — the server task is reconnecting, or it has exhausted its retry budget and parked
        # (e.g. a dead stdio subprocess). Probing here would write into a dead/absent transport and re-arm
        # the breaker forever (#16788). Instead, ask the (always-present) server task to rebuild the
        # transport — which respawns a dead stdio subprocess — and return a clean "reconnecting" error so
        # the model backs off without burning iterations. The breaker resets once the fresh session
        # initializes (_run_stdio/_run_http call _reset_server_error).
        return tool_error(
            "setup_mcp is only available in the Hermes desktop app. Use the "
            "terminal instead: `hermes mcp install <name>` for catalog entries, "
            "`hermes mcp login <name>` for OAuth.")

    name = (server or "").strip()
    if not name:
        return tool_error("server is required — the catalog or config name of the MCP server.")

    action = (action or "install").strip().lower()
    if action not in _ACTIONS:
        return tool_error(f"action must be one of {', '.join(_ACTIONS)}.")

    try:
        raw = callback(name, action, (reason or "").strip())
    except Exception as exc:
        return tool_error(f"MCP setup flow failed: {exc}")

    if not raw:
        # The renderer never answered (timeout / closed window). Distinct from an explicit
        # decline, which arrives as {"status": "declined"}.
        return json.dumps({
            "status": "unanswered",
            "server": name,
            "note": ("The user did not respond to the setup card. Do not retry "
                     "immediately; continue without the server or ask in chat."),
        }, ensure_ascii=False)

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"status": "error", "detail": str(raw)}, ensure_ascii=False)


SETUP_MCP_SCHEMA = {
    "name": "setup_mcp",
    "description": (
        "Propose an MCP server as an inline consent card (install a catalog "
        "entry, re-enable a disabled server, or run OAuth); blocks until the "
        "user acts. Use when they ask to add an MCP or a task clearly needs "
        "a missing one. Never hand-edit mcp_servers config for them — always "
        "use this tool. Never re-ask after a decline — on declined/"
        "unanswered, continue without it. Catalog names: `hermes mcp "
        "catalog` in the terminal."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": "Catalog name (install) or mcp_servers config name (enable/authorize).",
            },
            "action": {
                "type": "string",
                "enum": ["install", "enable", "authorize"],
                "description": "Defaults to install.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence on the card: why this helps right now.",
            },
        },
        "required": ["server"],
    },
}


registry.register(
    name="setup_mcp",
    toolset="desktop_ui",
    schema=SETUP_MCP_SCHEMA,
    handler=lambda args, **kw: setup_mcp_tool(
        server=args.get("server", ""),
        action=args.get("action", "install"),
        reason=args.get("reason", ""),
        callback=kw.get("callback"),
    ),
    emoji="🔌",
)

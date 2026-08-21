#!/usr/bin/env python3
"""Propose an MCP server to the user as an inline card in the desktop chat.

The card (install / enable / authorize + decline) lives in the desktop
renderer, so this tool round-trips through the gateway's blocking-prompt
bridge — the same one ``clarify`` uses: tui_gateway emits
``mcp.setup.request``, the renderer walks the user through the flow via the
existing REST endpoints (catalog install, enable, OAuth), and answers with
``mcp.setup.respond`` once the flow settles. This module is just schema + a
thin dispatcher over the platform-injected callback.

Lives in the ``desktop_ui`` toolset, which the GUI gateway enables only for
desktop-sourced sessions — on every other surface the agent falls back to
``hermes mcp install <name>`` in the terminal.
"""

import json
from typing import Callable, Optional

from tools.registry import registry, tool_error

_ACTIONS = ("install", "enable", "authorize")


def setup_mcp_tool(
    server: str = "",
    action: str = "install",
    reason: str = "",
    callback: Optional[Callable] = None,
) -> str:
    """Ask the desktop GUI to run an MCP setup flow; return its JSON outcome."""
    if callback is None:
        return tool_error(
            "setup_mcp is only available in the Hermes desktop app. Use the "
            "terminal instead: `hermes mcp install <name>` for catalog entries, "
            "`hermes mcp login <name>` for OAuth."
        )

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
        # The renderer never answered (timeout / closed window). Distinct from
        # an explicit decline, which arrives as {"status": "declined"}.
        return json.dumps(
            {
                "status": "unanswered",
                "server": name,
                "note": (
                    "The user did not respond to the setup card. Do not retry "
                    "immediately; continue without the server or ask in chat."
                ),
            },
            ensure_ascii=False,
        )

    # Desktop answers with a JSON object; pass it through, else wrap the raw text.
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"status": "error", "detail": str(raw)}, ensure_ascii=False)


SETUP_MCP_SCHEMA = {
    "name": "setup_mcp",
    "description": (
        "Propose an MCP server to the user as an inline consent card in the "
        "Hermes desktop chat. The card lets them install a catalog entry, "
        "re-enable a disabled server, or run an OAuth login — right there, "
        "without opening the Capabilities tab — and blocks until they act or "
        "decline. Use when the user asks to add/set up an MCP (e.g. \"add the "
        "linear mcp\"), or when a task clearly needs one that is missing or "
        "unauthorized. Never call it twice for the same server after a "
        "decline. Returns JSON {status: installed|enabled|authorized|declined|"
        "unanswered|error, server, detail?, tools?}. On declined/unanswered, "
        "continue without the server. Catalog names: run `hermes mcp catalog` "
        "in the terminal to list them."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "server": {
                "type": "string",
                "description": (
                    "The server's catalog name (for install) or its name in "
                    "mcp_servers config (for enable/authorize)."
                ),
            },
            "action": {
                "type": "string",
                "enum": ["install", "enable", "authorize"],
                "description": (
                    "install: add a catalog entry (prompts for any required "
                    "keys). enable: re-enable a disabled configured server. "
                    "authorize: run the OAuth browser flow for a configured "
                    "server. Defaults to install."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One short sentence shown on the card: why this server "
                    "helps right now (e.g. \"To read the JIRA ticket you "
                    "linked\")."
                ),
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

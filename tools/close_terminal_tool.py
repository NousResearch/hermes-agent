#!/usr/bin/env python3
"""Close a read-only agent terminal tab in the HT AI Agent desktop GUI.

Each ``terminal(background=true)`` process is mirrored as a read-only tab in the
desktop's terminal pane. This tool lets the agent drop a tab it no longer needs
to show — WITHOUT killing the process (use ``process(action='kill')`` for that).
The output keeps buffering and the user can reopen the tab from the status stack.

It routes through the process registry's ``on_close`` sink, which the desktop
gateway wires to emit a ``terminal.close`` event the renderer handles. Like
``read_terminal`` it is gated on ``HERMES_DESKTOP`` so it never appears outside
the GUI.
"""

import json
import os

from utils import env_var_enabled

from tools.process_registry import process_registry
from tools.registry import registry, tool_error


def close_terminal_tool(process_id: str) -> str:
    """Ask the desktop GUI to close a background process's read-only tab."""
    pid = (process_id or "").strip()
    if not pid:
        return tool_error("process_id is required (the background process whose tab to close).")

    return json.dumps(process_registry.request_close_terminal(pid), ensure_ascii=False)


def check_close_terminal_requirements() -> bool:
    """Desktop GUI only — HERMES_DESKTOP is set on the gateway the app spawns."""
    return env_var_enabled("HERMES_DESKTOP")


CLOSE_TERMINAL_SCHEMA = {
    "name": "close_terminal",
    "description": (
        "Close the read-only terminal tab for one of your background processes in "
        "the HT AI Agent desktop GUI (the tabs mirroring terminal(background=true) runs). "
        "This does NOT kill the process — it only drops the tab/view; the output "
        "keeps buffering and the user can reopen it from the status stack. Use it "
        "to tidy up when a background process's live terminal is no longer worth "
        "showing. To actually stop the process, use process(action='kill') instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "process_id": {
                "type": "string",
                "description": (
                    "The background process's session id (from terminal(background=true) "
                    "output or process(action='list')) whose tab should be closed."
                ),
            },
        },
        "required": ["process_id"],
    },
}


registry.register(
    name="close_terminal",
    toolset="terminal",
    schema=CLOSE_TERMINAL_SCHEMA,
    handler=lambda args, **kw: close_terminal_tool(process_id=args.get("process_id", "")),
    check_fn=check_close_terminal_requirements,
    emoji="🖥️",
)

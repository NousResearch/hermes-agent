"""Close a read-only agent terminal tab in the Hermes desktop GUI WITHOUT killing
the mirrored ``terminal(background=true)`` process (output keeps buffering; the user
can reopen it). Routes through the process registry's ``on_close`` sink, which the
desktop gateway wires to a ``terminal.close`` event. ``desktop_ui`` toolset only."""

import json

from tools.process_registry import process_registry
from tools.registry import registry, tool_error


def close_terminal_tool(process_id: str) -> str:
    """Ask the desktop GUI to close a background process's read-only tab."""
    pid = (process_id or "").strip()
    if not pid:
        return tool_error("process_id is required (the background process whose tab to close).")
    return json.dumps(process_registry.request_close_terminal(pid), ensure_ascii=False)


CLOSE_TERMINAL_SCHEMA = {
    "name": "close_terminal",
    "description": (
        "Hide a background process's terminal tab (process keeps running) in "
        "the Hermes desktop GUI (the tabs mirroring terminal(background=true) runs). "
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
    toolset="desktop_ui",
    schema=CLOSE_TERMINAL_SCHEMA,
    handler=lambda args, **kw: close_terminal_tool(process_id=args.get("process_id", "")),
    emoji="🖥️",
)

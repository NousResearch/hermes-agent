"""``type: prompt`` quick commands on the TUI/Desktop gateway (slash.exec + commands.catalog)."""

import threading
import types

from tui_gateway import server


def test_prompt_quick_command_reaches_the_agent_and_marks_review_button(monkeypatch):
    """slash.exec answers a send directive (never the slash worker's dead CLI queue); the catalog flags
    ``review_button`` so the Desktop Review pane can render it."""
    cfg = {"quick_commands": {"rn": {"type": "prompt", "prompt": "Notes for {args}", "review_button": "Release notes"}}}
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    server._sessions["sid"] = {"agent": types.SimpleNamespace(), "session_key": "session-key", "history": [],
                               "history_lock": threading.Lock(), "slash_worker": None, "cols": 80}
    try:
        resp = server.handle_request(
            {"id": "1", "method": "slash.exec", "params": {"session_id": "sid", "command": "rn v2.3"}})
        assert resp["result"] == {"type": "send", "message": "Notes for v2.3"}
        catalog = server.handle_request({"id": "2", "method": "commands.catalog", "params": {}})["result"]
        assert catalog["commands"]["/rn"]["review_button"] == "Release notes"
    finally:
        server._sessions.pop("sid", None)

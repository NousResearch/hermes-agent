"""The Ink TUI / Desktop sudo card fires the human-input hook pair without the typed password (#132333)."""

import sys
import threading
from unittest.mock import patch

PASSWORD = "hunter2-tui-do-not-leak"


def test_tui_sudo_request_fires_human_input_hooks(monkeypatch):
    from hermes_cli import banner

    # The real server binds callbacks on import; isolate only its process-wide side effects.
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    from tui_gateway import server, server_requests
    from agent.vault_backends import unlock
    from tools import project_tools, skills_tool, terminal_tool, terminal_tool_sudo

    monkeypatch.setattr(terminal_tool, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_current_session_tls", threading.local())
    monkeypatch.setattr(project_tools, "_workspace_callback", None)
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", None)
    monkeypatch.setattr(terminal_tool_sudo, "_sudo_password_cache", {})

    sid, session_key = "human-input-tui", "human-input-conversation"
    monkeypatch.setitem(server._sessions, sid, {"session_key": session_key, "source": "desktop"})

    def answer(frame):
        assert server_requests.resolve_response(
            {"jsonrpc": "2.0", "id": frame["id"], "result": {"value": PASSWORD}}
        )
        return True

    monkeypatch.setattr(server, "write_json", answer)
    calls = []

    def capture(name, **kwargs):
        if name.startswith("on_human_input_"):
            calls.append((name, kwargs))
        return []

    tokens = server._set_session_context(session_key, ui_session_id=sid)
    try:
        server._wire_callbacks(sid)
        with patch("hermes_cli.plugins.invoke_hook", side_effect=capture):
            assert terminal_tool_sudo._prompt_for_sudo_password(command="sudo systemctl restart nginx") == PASSWORD
    finally:
        server._clear_session_context(tokens)
        server_requests.reset_for_tests()

    assert [name for name, _ in calls] == ["on_human_input_request", "on_human_input_resolved"]
    request, resolved = calls[0][1], calls[1][1]
    assert request["kind"] == "sudo" and request["request_id"] == resolved["request_id"]
    assert request["platform"] == "desktop" and request["session_key"] == session_key
    assert "systemctl restart nginx" in request["prompt"]
    assert resolved["outcome"] == "provided"
    assert PASSWORD not in repr(calls)

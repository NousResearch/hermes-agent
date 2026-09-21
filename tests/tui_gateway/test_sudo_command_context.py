"""Sudo questions carry the original, redacted command through the real request path."""

from contextvars import ContextVar
import sys
import threading


def test_sudo_request_preserves_command_without_leaking_prompt_context(monkeypatch):
    from hermes_cli import banner

    # The real server binds callbacks on import; isolate only its process-wide side effects.
    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    from tui_gateway import server, server_requests
    from tui_gateway.contracts.registry import SERVER_REQUESTS
    from gateway.run import _redact_approval_command
    from agent import redact
    from agent.vault_backends import unlock
    from tools import project_tools, skills_tool, terminal_tool, terminal_tool_sudo

    # Restore the real registrations, including unrelated callbacks wired by the gateway.
    monkeypatch.setattr(terminal_tool, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_current_session", ContextVar("test_vault_current_session", default=None))
    monkeypatch.setattr(project_tools, "_workspace_callback", None)
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", None)
    monkeypatch.setattr(terminal_tool_sudo, "_sudo_password_cache", {})
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)

    sid, session_key = "sudo-dialog", "sudo-conversation"
    monkeypatch.setitem(server._sessions, sid, {"session_key": session_key, "source": "desktop"})
    command = (
        "printf '%s\\n' '" + "full context Ω " * 1000 + "' &&\n"
        "sudo env API_TOKEN=ghp_" + "X" * 36 + " first-command | sort;\n"
        "sudo second-command --keep-final-argument"
    )
    expected = _redact_approval_command(command)
    assert expected != command
    frames, replays = [], []

    def refuse(frame):
        frames.append(frame)
        replays.append(server_requests.open_requests(sid))
        assert server_requests.resolve_response(
            {"jsonrpc": "2.0", "id": frame["id"], "result": {"value": ""}}
        )
        return True

    monkeypatch.setattr(server, "write_json", refuse)
    tokens = server._set_session_context(session_key, ui_session_id=sid)
    try:
        server._wire_callbacks(sid)
        assert terminal_tool_sudo._transform_sudo_command(command) == (command, None)
        assert len(frames) == 1
        frame = frames[0]
        assert frame["method"] == "sudo"
        assert frame["params"] == {"session_id": sid, "command": expected}
        assert replays == [[{key: frame[key] for key in ("id", "method", "params")}]]
        assert server_requests.open_requests(sid) == []
        assert terminal_tool_sudo._get_cached_sudo_password() == ""
        assert terminal_tool_sudo.get_sudo_prompt_command() == ""

        # A legacy zero-argument caller must not inherit the preceding command.
        assert terminal_tool._get_sudo_password_callback()() == ""
        assert frames[-1]["params"] == {"session_id": sid, "command": ""}
        assert terminal_tool_sudo._prompt_for_sudo_password() == ""
        assert frames[-1]["params"] == {"session_id": sid, "command": ""}
        assert SERVER_REQUESTS["sudo"].params(session_id=sid).command == ""

        def failed_legacy_callback():
            assert terminal_tool_sudo.get_sudo_prompt_command() == command
            raise RuntimeError("prompt unavailable")

        terminal_tool.set_sudo_password_callback(failed_legacy_callback)
        assert terminal_tool_sudo._prompt_for_sudo_password(command=command) == ""
        assert terminal_tool_sudo.get_sudo_prompt_command() == ""
    finally:
        server._clear_session_context(tokens)
        server_requests.reset_for_tests()

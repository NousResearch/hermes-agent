"""Skill credential prompts follow the turn that asked, not the last wired session.

Regression for the process-global secret callback: wiring a newer session replaced
the closure, so a prompt raised during an older turn was delivered — and its
submitted value continued setup — under the newer session's id.
"""

import sys
import threading


def _gateway(monkeypatch):
    """Import the real gateway after neutralizing process-wide import side effects."""
    from hermes_cli import banner

    monkeypatch.setattr(banner, "prefetch_update_check", lambda: None)
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    from agent.vault_backends import unlock
    from tools import project_tools, skills_tool, terminal_tool, terminal_tool_sudo
    from tui_gateway import server, server_requests

    monkeypatch.setattr(terminal_tool, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_callback_tls", threading.local())
    monkeypatch.setattr(unlock, "_current_session_tls", threading.local())
    monkeypatch.setattr(project_tools, "_workspace_callback", None)
    monkeypatch.setattr(skills_tool, "_secret_capture_callback", None)
    monkeypatch.setattr(terminal_tool_sudo, "_sudo_password_cache", {})
    monkeypatch.delenv("HERMES_UI_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    return server, server_requests, skills_tool


def _capture(server, server_requests, skills_tool, monkeypatch, *, values):
    """Drive the real secret ask and record which session received it and what was stored."""
    frames, stored = [], []

    def answer(frame):
        frames.append(frame)
        sid = frame["params"]["session_id"]
        assert server_requests.resolve_response(
            {"jsonrpc": "2.0", "id": frame["id"], "result": {"value": values[sid]}}
        )
        return True

    def save(key, value):
        stored.append((key, value))
        return {"success": True, "stored_as": key, "validated": False}

    monkeypatch.setattr(server, "write_json", answer)
    monkeypatch.setattr("hermes_cli.config.save_env_value_secure", save)
    result = skills_tool._capture_required_environment_variables(
        "demo-skill", [{"name": "DEMO_TOKEN", "prompt": "Token"}]
    )
    return frames, stored, result


def test_secret_prompt_goes_to_active_turn_not_last_wired_session(monkeypatch):
    """While session A's turn is active, a prompt must not land on the closure sid."""
    from gateway.session_context import get_session_env

    server, server_requests, skills_tool = _gateway(monkeypatch)
    server._wire_callbacks("session-A")
    server._wire_callbacks("session-B")  # replaces the process-global callback

    tokens = server._set_session_context("turn-A", ui_session_id="session-A")
    try:
        assert get_session_env("HERMES_UI_SESSION_ID") == "session-A"
        frames, stored, result = _capture(
            server, server_requests, skills_tool, monkeypatch,
            values={"session-A": "owner-secret", "session-B": "closure-secret"},
        )
    finally:
        server._clear_session_context(tokens)
        server_requests.reset_for_tests()

    assert [frame["method"] for frame in frames] == ["secret"]
    assert frames[0]["params"]["session_id"] == "session-A"
    assert stored == [("DEMO_TOKEN", "owner-secret")]
    assert result == {"missing_names": [], "setup_skipped": False, "gateway_setup_hint": None}
    assert server_requests.open_requests("session-A") == []
    assert server_requests.open_requests("session-B") == []


def test_secret_prompt_falls_back_to_wired_sid_without_turn_context(monkeypatch):
    """With no turn context bound, the prompt still reaches the session that wired it."""
    from gateway.session_context import get_session_env, reset_session_vars

    server, server_requests, skills_tool = _gateway(monkeypatch)
    reset_session_vars()
    assert get_session_env("HERMES_UI_SESSION_ID") == ""

    server._wire_callbacks("only-session")
    try:
        frames, stored, result = _capture(
            server, server_requests, skills_tool, monkeypatch,
            values={"only-session": "wired-secret"},
        )
    finally:
        server_requests.reset_for_tests()

    assert [frame["method"] for frame in frames] == ["secret"]
    assert frames[0]["params"]["session_id"] == "only-session"
    assert stored == [("DEMO_TOKEN", "wired-secret")]
    assert result["setup_skipped"] is False

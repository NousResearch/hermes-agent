"""Protected-instruction approval fails closed at once, without asking anyone, where nobody can answer.

``hermes chat -q`` (how kanban spawns every worker), cron and unattended platforms can have the CLI approval
callback registered on the agent thread. The gate used to ask it, block for the full approvals timeout, then
report "timed out" — which reads as a human ignoring a prompt nobody saw. The interactive approve/deny paths
are covered in test_file_write_safety.py; the real CLI modal in tests/hermes_cli/test_cli_approval_ui.py.
"""

import json

import pytest


@pytest.fixture(autouse=True)
def _gate_on(monkeypatch):
    import tools.file_tools_write_guards as guards
    from tools.terminal_tool import set_approval_callback
    monkeypatch.setattr(guards, "_protected_instruction_config", lambda: (True, []))
    for name in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_CRON_SESSION", "HERMES_SESSION_PLATFORM"):
        monkeypatch.delenv(name, raising=False)
    set_approval_callback(None)
    yield
    set_approval_callback(None)


@pytest.mark.parametrize("env", [
    {"HERMES_SINGLE_QUERY_SESSION": "1"},
    {"HERMES_CRON_SESSION": "1"},
    {"HERMES_SESSION_PLATFORM": "webhook"},
], ids=["single_query", "cron", "webhook"])
def test_registered_callback_is_never_asked_without_a_user(tmp_path, monkeypatch, env):
    from tools.terminal_tool import set_approval_callback
    from tools.file_tools import write_file_tool

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    asked = []
    set_approval_callback(lambda command, description, **kw: asked.append(command) or "once")
    target = tmp_path / "SOUL.md"
    res = json.loads(write_file_tool(str(target), "content"))

    assert asked == []
    assert "no interactive user or gateway is present" in res["error"]
    assert "timed out" not in res["error"]
    assert not target.exists()


@pytest.mark.parametrize("mode", ["approve", "deny"])
def test_single_query_mode_never_auto_approves_this_gate(tmp_path, monkeypatch, mode):
    """``approvals.single_query_mode`` governs the dangerous-command gate only: a protected-instruction write
    always needs a live human."""
    from tools import approval_context
    from tools.file_tools import write_file_tool
    from tools.terminal_tool import set_approval_callback

    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setattr(approval_context, "_get_single_query_approval_mode", lambda: mode)
    asked = []
    set_approval_callback(lambda command, description, **kw: asked.append(command) or "once")
    target = tmp_path / "SOUL.md"
    res = json.loads(write_file_tool(str(target), "content"))

    assert asked == []
    assert "BLOCKED" in res["error"]
    assert not target.exists()


def test_finite_worker_with_local_notifier_fails_before_gateway_wait(tmp_path, monkeypatch):
    """A registered local worker callback is not a reachable human transport.

    Kanban workers are finite and always register this callback. Reaching the gateway wait here used to
    park the protected write for the whole approval timeout even though no external prompt existed.
    """
    from gateway.session_finite import finite_turn_scope
    from tools import approval as approval_module
    from tools import approval_gateway_wait
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.file_tools import write_file_tool

    notified = []
    polled = []
    monkeypatch.setattr(approval_gateway_wait, "_poll_event", lambda *args, **kwargs: polled.append(1) or "timeout")
    session_key = "agent:implementer:local:dm:worker"
    approval_module.register_gateway_notify(session_key, notified.append)
    target = tmp_path / "AGENTS.md"
    session_token = set_current_session_key(session_key)
    try:
        with finite_turn_scope(True):
            res = json.loads(write_file_tool(str(target), "content"))
    finally:
        approval_module.unregister_gateway_notify(session_key)
        reset_current_session_key(session_token)

    assert notified == []
    assert polled == []
    assert "no interactive user or gateway is present" in res["error"]
    assert "timed out" not in res["error"]
    assert not target.exists()


def test_broker_marked_finite_worker_keeps_explicit_human_gate(tmp_path):
    """The finite exception still requires an exact human grant; it never auto-approves the write."""
    from gateway.session_finite import finite_turn_scope
    from tools import approval as approval_module
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.file_tools import write_file_tool

    session_key = "agent:implementer:local:dm:brokered-worker"
    notified = []

    def answer(data):
        notified.append(data)
        approval_module.resolve_gateway_approval(session_key, "once", request_id=data["request_id"])

    approval_module.register_gateway_notify(session_key, answer)
    target = tmp_path / "AGENTS.md"
    session_token = set_current_session_key(session_key)
    try:
        with finite_turn_scope(True, approval_broker=True):
            result = json.loads(write_file_tool(str(target), "content"))
    finally:
        approval_module.unregister_gateway_notify(session_key)
        reset_current_session_key(session_token)

    assert result.get("verified") is True
    assert target.read_text() == "content"
    assert len(notified) == 1
    assert notified[0]["allow_session"] is False and notified[0]["allow_permanent"] is False

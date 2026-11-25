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

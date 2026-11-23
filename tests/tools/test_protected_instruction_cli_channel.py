"""Who may answer, and who may switch the gate off.

Two channels, one rule: nobody but a person at a terminal may grant what the protected-instruction gate
requires.

* Fail closed **at once** where nobody can answer. ``hermes chat -q`` (how kanban spawns every worker),
  cron and unattended platforms can carry a registered CLI callback *or* a gateway notifier for their
  session; the gate used to ask either one, block for the full approvals timeout, then report "timed out",
  which reads as a human ignoring a prompt nobody saw. The interactive approve/deny paths are covered in
  test_file_write_safety.py; the real CLI modal in tests/hermes_cli/test_cli_approval_ui.py.
* Refuse the **self-grant**: the switch itself (``security.protected_instruction_files``) is settable only
  from a terminal, because ``hermes config set`` is an agent-available command — a lane that could run it
  would be granting the approval the gate exists to require.
"""

import json
import sys

import pytest

import tools.file_tools_write_guards as _guards

# Captured before the autouse fixture swaps the reader for ``(True, [])``: the self-grant test has to
# exercise the real one, since its whole point is which config the gate ends up reading.
_REAL_GATE_CONFIG = _guards._protected_instruction_config


class _TTY:
    """A stdin a person is at, for the positive half of the self-grant test."""

    def isatty(self) -> bool:
        return True


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


def test_a_registered_gateway_notifier_is_never_asked_without_a_user(tmp_path, monkeypatch):
    """The CLI panel is not the only surface a worker can carry: a gateway notifier registered for the
    session used to take the gateway round-trip and sit out the full approval timeout, then report
    "timed out" — 20 minutes for three blocked writes and nothing to show for it."""
    from tools import approval, approval_gateway_wait
    from tools.approval_context import get_current_session_key
    from tools.file_tools import write_file_tool

    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    waited = []
    monkeypatch.setattr(approval_gateway_wait, "_await_gateway_decision",
                        lambda *a, **kw: waited.append(a) or {"resolved": True, "choice": "once"})
    session_key = get_current_session_key()
    approval.register_gateway_notify(session_key, lambda approval_data: None)
    try:
        target = tmp_path / "SOUL.md"
        res = json.loads(write_file_tool(str(target), "content"))
    finally:
        approval.unregister_gateway_notify(session_key)

    assert waited == []
    assert "no interactive user or gateway is present" in res["error"]
    assert "timed out" not in res["error"]
    assert not target.exists()


@pytest.mark.parametrize("unattended_env", [
    {"HERMES_SINGLE_QUERY_SESSION": "1"},
    {},  # an agent's own subprocess: no marker, but also no TTY on stdin
], ids=["single_query", "no_terminal"])
def test_config_set_refuses_to_disable_the_gate_without_a_terminal(tmp_path, monkeypatch, capsys,
                                                                   unattended_env):
    """``hermes -p <lane> config set security.protected_instruction_files false`` — what a dispatched lane
    did once its write was correctly BLOCKED — is refused, and the gate stays on."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name, value in unattended_env.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit) as refused:
        cli_config.set_config_value("security.protected_instruction_files", "false")

    assert refused.value.code == 1
    assert "Refusing to set 'security.protected_instruction_files'" in capsys.readouterr().err
    assert _REAL_GATE_CONFIG() == (True, [])
    written = (tmp_path / "config.yaml")
    assert not written.exists() or "protected_instruction_files: false" not in written.read_text()


def test_a_person_at_a_terminal_may_still_turn_the_gate_off(tmp_path, monkeypatch):
    """The knob stays real for its owner: refusing *everyone* would be a different bug."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(sys, "stdin", _TTY())

    cli_config.set_config_value("security.protected_instruction_files", "false")

    assert _REAL_GATE_CONFIG() == (False, [])


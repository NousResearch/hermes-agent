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


# The gate's switch, spelled so the raw text never contains the leaf: YAML double-quoted scalars decode
# ``\uXXXX`` and ``\xXX``, and ``set_config_value`` parses the argument after the guard used to look at it.
_ESCAPED_DISABLE_PAYLOADS = [
    ('{"protected_instruction_files": false}', "plain"),
    ('{"protected_instruction_\\u0066iles": false}', "u0066"),
    ('{"protected_instruction_\\x66iles": false}', "x66"),
    ('{"\\u0070rotected_instruction_files": false}', "u0070"),
]


def _flipped_home(home, *, extra_patterns: str = ""):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "security:\n  protected_instruction_files: false\n" + extra_patterns)


def _gate_after_write(home):
    """The gate's real reading of *home*, with the config caches dropped (no stale parse)."""
    from hermes_cli import config as cli_config

    cli_config._LOAD_CONFIG_CACHE.clear()
    cli_config._RAW_CONFIG_CACHE.clear()
    return _REAL_GATE_CONFIG()


@pytest.mark.parametrize("payload,case", _ESCAPED_DISABLE_PAYLOADS, ids=[c for _, c in _ESCAPED_DISABLE_PAYLOADS])
@pytest.mark.parametrize("force", [False, True], ids=["plain", "force"])
def test_config_set_cannot_hide_the_switch_behind_a_yaml_escape(tmp_path, monkeypatch, capsys,
                                                               payload, case, force):
    """``config set security '{"protected_instruction_\\u0066iles": false}'`` wrote the real leaf and the
    gate read off: the guard substring-matched the raw argument while the writer parsed it afterwards."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")

    with pytest.raises(SystemExit) as refused:
        cli_config.set_config_value("security", payload, force=force)

    assert refused.value.code == 1
    assert "Refusing to set 'security'" in capsys.readouterr().err
    written = tmp_path / "config.yaml"
    assert not written.exists() or "protected_instruction_files" not in written.read_text()
    assert _gate_after_write(tmp_path) == (True, [])


@pytest.mark.parametrize("key,value,force", [
    ("security.protected_instruction_files", "true", False),
    # replacing the section itself needs --force once it has sub-keys; the guard must not be the wall then
    ("security", '{"protected_instruction_files": true}', True),
], ids=["leaf", "section"])
def test_the_watchdog_repair_write_is_allowed_without_a_terminal(tmp_path, monkeypatch, key, value,
                                                                force):
    """``protected-instruction-guard.py`` restores a flipped home with ``config set ... true`` from a cron
    tick. Refusing that direction too left the only automated mitigation failing open."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _flipped_home(tmp_path)
    assert _gate_after_write(tmp_path) == (False, [])

    cli_config.set_config_value(key, value, force=force)

    assert "protected_instruction_files: true" in (tmp_path / "config.yaml").read_text()
    assert _gate_after_write(tmp_path) == (True, [])


def test_a_repair_keeps_the_gates_extra_patterns(tmp_path, monkeypatch):
    """Turning the gate back on is not a section rewrite: what it covered has to survive."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _flipped_home(tmp_path, extra_patterns="  protected_instruction_extra_patterns:\n    - RULES.md\n")
    assert _gate_after_write(tmp_path) == (False, ["RULES.md"])

    cli_config.set_config_value("security.protected_instruction_files", "true")

    written = (tmp_path / "config.yaml").read_text()
    assert "protected_instruction_files: true" in written
    assert "RULES.md" in written
    assert _gate_after_write(tmp_path) == (True, ["RULES.md"])


def test_config_set_cannot_narrow_the_gates_coverage_without_a_terminal(tmp_path, monkeypatch, capsys):
    """Narrowing what the gate covers is the same self-grant as switching it off."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _flipped_home(tmp_path)
    (tmp_path / "config.yaml").write_text(
        "security:\n  protected_instruction_extra_patterns:\n    - RULES.md\n")

    with pytest.raises(SystemExit) as refused:
        cli_config.set_config_value("security.protected_instruction_extra_patterns", "[]")

    assert refused.value.code == 1
    assert "Refusing to set 'security.protected_instruction_extra_patterns'" in capsys.readouterr().err
    assert "- RULES.md" in (tmp_path / "config.yaml").read_text()


def test_an_unrelated_config_write_is_not_caught_by_the_guard(tmp_path, monkeypatch):
    """A home whose gate is off is not frozen: the refusal covers the switch, not every setting."""
    from hermes_cli import config as cli_config

    monkeypatch.setattr(_guards, "_protected_instruction_config", _REAL_GATE_CONFIG)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    _flipped_home(tmp_path)

    cli_config.set_config_value("display.skin", "ares")

    assert "ares" in (tmp_path / "config.yaml").read_text()


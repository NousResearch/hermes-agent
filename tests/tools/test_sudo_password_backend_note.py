"""One-time WARNING when a sudo password is sent to a terminal backend other than the local one.

Every case goes through a real environment's ``execute`` or ``_prepare_command`` with a cached
password; the returned command text and sudo stdin must be the same as without the note.
"""

import logging
import sys

import pytest

from tools import terminal_tool, terminal_tool_sudo
from tools.environments.base import BaseEnvironment
from tools.environments.docker import DockerEnvironment
from tools.environments.local import LocalEnvironment
from tools.environments.ssh import SSHEnvironment

PW = "synthetic-note-pw-3"
NOTE_LOGGER = "tools.environments.sudo_password_note"


class _RecordingSandboxEnv(BaseEnvironment):
    """Non-local backend that records ``_run_bash`` calls instead of running them."""

    def __init__(self):
        super().__init__(cwd="/", timeout=10)
        self.calls = []

    def _run_bash(self, cmd_string, *, login=False, timeout=120, stdin_data=None):
        self.calls.append((cmd_string, stdin_data))
        return cmd_string

    def _wait_for_process(self, proc, timeout=120, **_kwargs):
        return {"output": "", "returncode": 0}

    def cleanup(self):
        pass


def _bare(env_cls):
    """An instance of a real backend class without connecting it; only ``_prepare_command`` runs.
    The askpass answer is pinned to False so no capability probe reaches a backend."""
    env = object.__new__(env_cls)
    env._sudo_askpass_ok = False
    env.cleanup = lambda: None
    return env


def _forget_noted_backends():
    """Clear the once-only state if the note module is loaded (base.py imports it lazily)."""
    module = sys.modules.get(NOTE_LOGGER)
    if module is not None:
        module._reset_noted_backends()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool.set_sudo_password_callback(None)
    terminal_tool_sudo._reset_cached_sudo_passwords()
    _forget_noted_backends()
    yield
    terminal_tool_sudo._reset_cached_sudo_passwords()
    _forget_noted_backends()


def _notes(caplog):
    return [r for r in caplog.records if r.name == NOTE_LOGGER and r.levelno == logging.WARNING]


def test_non_local_backend_logs_one_warning_across_two_commands(caplog):
    terminal_tool_sudo._set_cached_sudo_password(PW)
    env = _RecordingSandboxEnv()
    env._sudo_askpass_ok = False
    with caplog.at_level(logging.WARNING, logger=NOTE_LOGGER):
        env.execute("sudo true")
        env.execute("sudo id -u")

    notes = _notes(caplog)
    assert len(notes) == 1
    message = notes[0].getMessage()
    assert "_recording_sandbox_env terminal backend" in message
    assert "only valid there" in message
    assert PW not in message
    # The password still reached the backend on stdin for both commands.
    assert [stdin for _cmd, stdin in env.calls] == [PW + "\n", PW + "\n"]


def test_plugin_backend_note_uses_its_registered_name(caplog):
    """A plugin backend is named as registered, not after its Python class."""
    import types

    from tools.terminal_tool_backends import _stamp_plugin_env

    terminal_tool_sudo._set_cached_sudo_password(PW)
    env = _RecordingSandboxEnv()
    env._sudo_askpass_ok = False
    provider = types.SimpleNamespace(name="Acme-Box", default_cwd=None)
    assert _stamp_plugin_env(env, provider, "/") is env
    with caplog.at_level(logging.WARNING, logger=NOTE_LOGGER):
        env.execute("sudo true")

    messages = [r.getMessage() for r in _notes(caplog)]
    assert len(messages) == 1
    assert "the acme-box terminal backend" in messages[0]
    assert "_recording_sandbox_env" not in messages[0]
    assert PW not in messages[0]


def test_each_backend_type_logs_its_own_note_once(caplog):
    terminal_tool_sudo._set_cached_sudo_password(PW)
    docker, ssh = _bare(DockerEnvironment), _bare(SSHEnvironment)
    with caplog.at_level(logging.WARNING, logger=NOTE_LOGGER):
        docker._prepare_command("sudo true")
        ssh._prepare_command("sudo true")
        docker._prepare_command("sudo ls")
        _bare(DockerEnvironment)._prepare_command("sudo ls")
        ssh._prepare_command("sudo ls")

    messages = [r.getMessage() for r in _notes(caplog)]
    assert len(messages) == 2
    assert "docker terminal backend" in messages[0]
    assert "ssh terminal backend" in messages[1]
    assert all(PW not in m for m in messages)


def test_local_backend_logs_no_note(caplog):
    terminal_tool_sudo._set_cached_sudo_password(PW)
    local = _bare(LocalEnvironment)
    with caplog.at_level(logging.DEBUG, logger=NOTE_LOGGER):
        result = local._prepare_command("sudo true")
    assert result == ("sudo -S -p '' true", PW + "\n")
    assert _notes(caplog) == []


@pytest.mark.parametrize("password", [None, ""])
def test_no_password_logs_no_note(caplog, monkeypatch, password):
    # None: nothing configured or cached; "": SUDO_PASSWORD set to an empty value.
    if password is not None:
        monkeypatch.setenv("SUDO_PASSWORD", password)
    with caplog.at_level(logging.DEBUG, logger=NOTE_LOGGER):
        _bare(DockerEnvironment)._prepare_command("sudo true")
    assert _notes(caplog) == []


def test_command_without_sudo_logs_no_note(caplog):
    terminal_tool_sudo._set_cached_sudo_password(PW)
    with caplog.at_level(logging.DEBUG, logger=NOTE_LOGGER):
        result = _bare(DockerEnvironment)._prepare_command("echo sudo-less && ls")
    assert result == ("echo sudo-less && ls", None)
    assert _notes(caplog) == []


@pytest.mark.parametrize("env_cls", [DockerEnvironment, SSHEnvironment, LocalEnvironment])
@pytest.mark.parametrize(
    ("command", "cached", "expected"),
    [
        ("sudo true", True, ("sudo -S -p '' true", PW + "\n")),
        ("sudo a && sudo b", True, ("sudo -S -p '' a && sudo -S -p '' b", (PW + "\n") * 2)),
        ("sudo true", False, ("sudo true", None)),
        ("ls -la", True, ("ls -la", None)),
    ],
)
def test_prepare_command_output_is_unchanged(env_cls, command, cached, expected):
    """Control: passes on base too. The note never alters command text or sudo stdin."""
    if cached:
        terminal_tool_sudo._set_cached_sudo_password(PW)
    env = _bare(env_cls)
    assert env._prepare_command(command) == expected
    # Same answer on a second call, after the note has been logged.
    assert env._prepare_command(command) == expected

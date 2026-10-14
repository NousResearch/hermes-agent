"""Background sudo spawns must reach a password instead of burning PAM attempts (#133622).

Covers:
  - spawn_local pipes the SUDO_PASSWORD lines into the rewritten ``sudo -S`` stdin
  - spawn_local refuses a background sudo with no password source (fail-closed, no spawn)
  - spawn_local still runs a background sudo unchanged when NOPASSWD applies
  - spawn_via_env embeds the password on the backgrounded subshell's stdin, not the wrapper's
"""

import stat
import textwrap

import pytest

from tools.process_registry import ProcessRegistry
from tools import terminal_tool_sudo


@pytest.fixture(autouse=True)
def _headless_without_cached_password(monkeypatch):
    """No prompt callback, no configured password, no session cache: every test starts
    from the headless state where a background sudo has no way to authenticate."""
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    terminal_tool_sudo._reset_cached_sudo_passwords()


def _write_fake_sudo(bin_dir, record_path):
    """Stand-in sudo: records argv (one line per arg) plus the first stdin line, which is
    what ``sudo -S`` consumes, then exits 0 like a successful authenticated command.
    Invoked by absolute path — a login shell's rc files reorder PATH, so PATH shadowing
    would race the real host sudo."""
    sudo_path = bin_dir / "sudo"
    sudo_path.write_text(
        textwrap.dedent(f"""\
        #!/usr/bin/env bash
        printf 'ARG:%s\\n' "$@" > {record_path}
        IFS= read -r line
        printf 'STDIN:%s\\n' "$line" >> {record_path}
        exit 0
    """)
    )
    sudo_path.chmod(
        sudo_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    return sudo_path


def _wait_until_exited(registry, session, timeout=20):
    result = registry.wait(session.id, timeout=timeout)
    assert result.get("status") == "exited", f"session never completed: {result}"
    return result


def test_spawn_local_background_sudo_receives_password_on_stdin(monkeypatch, tmp_path):
    record = tmp_path / "sudo_record.txt"
    sudo_path = _write_fake_sudo(tmp_path, record)
    monkeypatch.setenv("SUDO_PASSWORD", "hunter2")

    registry = ProcessRegistry()
    session = registry.spawn_local(
        command=f"{sudo_path} whoami",
        cwd=str(tmp_path),
        task_id="t-sudo-bg",
        session_key="s1",
    )

    _wait_until_exited(registry, session)
    recorded = record.read_text()
    assert "ARG:-S" in recorded  # rewritten to the piped-password form
    assert "ARG:whoami" in recorded
    assert "STDIN:hunter2" in recorded  # the password line reached sudo's stdin


def test_spawn_local_background_sudo_without_password_fails_closed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        ProcessRegistry, "_sudo_nopasswd_locally", staticmethod(lambda: False)
    )

    registry = ProcessRegistry()
    with pytest.raises(RuntimeError, match="SUDO_PASSWORD"):
        registry.spawn_local(
            command="sudo whoami",
            cwd=str(tmp_path),
            task_id="t-sudo-closed",
            session_key="s1",
        )


def test_spawn_local_background_sudo_nopasswd_runs_unchanged(monkeypatch, tmp_path):
    record = tmp_path / "sudo_record.txt"
    sudo_path = _write_fake_sudo(tmp_path, record)
    monkeypatch.setattr(
        ProcessRegistry, "_sudo_nopasswd_locally", staticmethod(lambda: True)
    )

    registry = ProcessRegistry()
    session = registry.spawn_local(
        command=f"{sudo_path} whoami",
        cwd=str(tmp_path),
        task_id="t-sudo-nopw",
        session_key="s1",
    )

    _wait_until_exited(registry, session)
    recorded = record.read_text()
    assert "ARG:whoami" in recorded
    assert "ARG:-S" not in recorded  # NOPASSWD: no rewrite, no password pipe


class _CapturingEnv:
    """Minimal non-local backend: records the wrapper command spawn_via_env builds."""

    def __init__(self):
        self.commands = []

    def execute(
        self, command, timeout=None, rewrite_compound_background=None, **kwargs
    ):
        self.commands.append(command)
        return {"output": "4242\n", "returncode": 0}


def test_spawn_via_env_sudo_password_reaches_subshell_stdin(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "hunter2")
    env = _CapturingEnv()
    registry = ProcessRegistry()
    # Keep the poller from spinning on the fake env; the wrapper build is what is asserted.
    monkeypatch.setattr(registry, "_track_started", lambda *a, **k: None)

    registry.spawn_via_env(
        env, command="sudo whoami", task_id="t-sudo-env", session_key="s1"
    )

    bg = env.commands[0]
    assert bg.startswith("mkdir -p ")  # wrapper shape unchanged
    assert "hunter2 | nohup bash -lc" in bg  # password pipes onto the subshell
    assert "sudo -S -p" in bg  # rewrite landed inside the payload


def test_spawn_via_env_compound_sudo_gets_one_line_per_invocation(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "hunter2")
    env = _CapturingEnv()
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_track_started", lambda *a, **k: None)

    registry.spawn_via_env(
        env,
        command="sudo true && sudo whoami",
        task_id="t-sudo-env-2",
        session_key="s1",
    )

    bg = env.commands[0]
    # Two sudo invocations must produce two password lines (sudo -S reads one each),
    # same prepend-to-stdin semantics the foreground execute() path guarantees.
    assert bg.count("sudo -S -p") == 2
    assert "hunter2\nhunter2" in bg


def test_spawn_via_env_without_password_leaves_wrapper_unchanged(monkeypatch, tmp_path):
    env = _CapturingEnv()
    registry = ProcessRegistry()
    monkeypatch.setattr(registry, "_track_started", lambda *a, **k: None)

    registry.spawn_via_env(
        env, command="sudo whoami", task_id="t-sudo-env-3", session_key="s1"
    )

    bg = env.commands[0]
    assert bg.startswith("mkdir -p ")
    assert "sudo -S" not in bg  # no rewrite: no password source resolved
    assert "hunter2" not in bg  # nothing to embed

"""An interrupted command must not adopt the shared environment's cwd.

The command wrapper prints the ``__HERMES_CWD_*`` marker AFTER the command
returns, so a killed / timed-out command emits none and ``env.cwd`` still holds
whatever the last command to FINISH left there. One local environment is shared
by every session (``_resolve_container_task_id`` collapses cwd-only overrides to
``"default"``), so that leftover is routinely ANOTHER session's directory.

Without the ``cwd_observed`` gate, the post-command dual-write stamped that
foreign directory onto the interrupted session's durable record, and every later
command in that session ran there — a silent re-home into a directory the user
never opened.
"""

import os
import tempfile

import pytest

import tools.terminal_tool as tt
from tools.environments.local import LocalEnvironment


@pytest.fixture(autouse=True)
def _clean_store(monkeypatch):
    monkeypatch.setattr(tt, "_session_cwd", {})
    monkeypatch.setattr(tt, "_task_env_overrides", {})
    monkeypatch.delenv("TERMINAL_ENV", raising=False)


@pytest.fixture
def env(tmp_path):
    environment = LocalEnvironment(cwd=str(tmp_path), timeout=30)
    environment.init_session()
    yield environment
    environment.cleanup()


def _run(env, session_key, command, timeout=30):
    """One turn of the terminal tool's per-session cwd handling.

    Mirrors terminal_tool.py: resolve this session's cwd, run, then dual-write
    the record only when the command reported where it finished.
    """
    command_cwd = tt._resolve_command_cwd(
        workdir=None, default_cwd=env.cwd, session_key=session_key, env_type="local"
    )
    result = env.execute(command, cwd=command_cwd, timeout=timeout)
    if result.get("cwd_observed"):
        tt.record_session_cwd(session_key, getattr(env, "cwd", None))
    return result, command_cwd


class TestCwdObservedFlag:
    def test_completed_command_reports_its_cwd(self, env, tmp_path):
        target = tmp_path / "done"
        target.mkdir()
        result, _ = _run(env, "sess", f"cd {target} && pwd")
        assert result["cwd_observed"] is True

    def test_interrupted_command_reports_no_cwd(self, env, tmp_path):
        target = tmp_path / "slow"
        target.mkdir()
        result, _ = _run(env, "sess", f"cd {target} && sleep 20", timeout=2)
        # Killed before the wrapper could print the marker.
        assert not result.get("cwd_observed")


class TestInterruptDoesNotStealAnotherSessionsCwd:
    def test_record_survives_an_interrupt(self, env, tmp_path):
        mine = tmp_path / "mine"
        theirs = tmp_path / "theirs"
        mine.mkdir()
        theirs.mkdir()

        _run(env, "mine", f"cd {mine} && pwd")
        assert tt.get_session_cwd("mine") == str(mine)

        # Another chat finishes a command; the shared env now points at it.
        _run(env, "theirs", f"cd {theirs} && pwd")
        assert os.path.realpath(env.cwd) == os.path.realpath(str(theirs))

        # My command is interrupted. My record must not adopt their directory.
        _run(env, "mine", f"cd {mine} && sleep 20", timeout=2)
        assert tt.get_session_cwd("mine") == str(mine)

    def test_next_command_still_runs_in_my_directory(self, env, tmp_path):
        mine = tmp_path / "mine"
        theirs = tmp_path / "theirs"
        mine.mkdir()
        theirs.mkdir()

        _run(env, "mine", f"cd {mine} && pwd")
        _run(env, "theirs", f"cd {theirs} && pwd")
        _run(env, "mine", f"cd {mine} && sleep 20", timeout=2)

        result, _ = _run(env, "mine", "pwd")
        assert os.path.realpath(result["output"].strip()) == os.path.realpath(str(mine))

    def test_single_session_keeps_its_own_prior_directory(self, env, tmp_path):
        """No second session needed: a lone session must not re-home either."""
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()

        _run(env, "solo", f"cd {first} && pwd")
        # Move the shared env elsewhere the way any other consumer would.
        env.execute(f"cd {second} && pwd", cwd=str(second))

        _run(env, "solo", f"cd {first} && sleep 20", timeout=2)
        assert tt.get_session_cwd("solo") == str(first)


class TestEchoIsGatedToo:
    def test_interrupted_command_does_not_echo_a_foreign_cwd(self, env, tmp_path):
        """The echo tells the model where it ended up; it must not lie."""
        mine = tmp_path / "mine"
        theirs = tmp_path / "theirs"
        mine.mkdir()
        theirs.mkdir()

        _run(env, "mine", f"cd {mine} && pwd")
        _run(env, "theirs", f"cd {theirs} && pwd")

        result, command_cwd = _run(env, "mine", f"cd {mine} && sleep 20", timeout=2)

        # The echo block in terminal_tool reads env.cwd only when observed.
        post_cwd = getattr(env, "cwd", None) if result.get("cwd_observed") else None
        echoed = (
            str(post_cwd)
            if post_cwd
            and command_cwd
            and os.path.realpath(str(post_cwd)) != os.path.realpath(str(command_cwd))
            else None
        )
        assert echoed is None


class TestTerminalToolReadsTheFlag:
    """Drive ``tt.terminal_tool`` itself, not a reimplementation of its gate.

    The classes above prove that the ENVIRONMENT produces ``cwd_observed``
    correctly. These tests prove that the terminal tool CONSUMES it: the
    record write and the cwd echo. Without them, reverting either call site
    to the ungated read passes every other test in this file (verified by
    mutation during review).
    """

    def _tool(self, monkeypatch, env, command, task_id, timeout=None):
        import json

        # One shared env for every session, like the real local backend
        # (_resolve_container_task_id collapses cwd-only sessions to "default").
        monkeypatch.setattr(tt, "_active_environments", {"default": env})
        monkeypatch.setattr(tt, "_last_activity", {})
        monkeypatch.setattr(
            tt, "_get_env_config",
            lambda: {"env_type": "local", "cwd": env.cwd, "timeout": 60,
                     "lifetime_seconds": 3600},
        )
        monkeypatch.setattr(
            tt, "_check_all_guards",
            lambda command, env_type, **kwargs: {"approved": True},
        )
        return json.loads(
            tt.terminal_tool(command=command, task_id=task_id, timeout=timeout)
        )

    def test_interrupt_keeps_record_and_echoes_no_cwd(self, env, tmp_path, monkeypatch):
        mine = tmp_path / "mine"
        theirs = tmp_path / "theirs"
        mine.mkdir()
        theirs.mkdir()

        # My session establishes its directory through the real tool.
        result = self._tool(monkeypatch, env, f"cd {mine} && pwd", "mine")
        assert result["exit_code"] == 0
        assert tt.get_session_cwd("mine") == str(mine)

        # Another session finishes a command; the shared env moves to it.
        result = self._tool(monkeypatch, env, f"cd {theirs} && pwd", "theirs")
        assert result["exit_code"] == 0
        assert os.path.realpath(env.cwd) == os.path.realpath(str(theirs))

        # My command is interrupted. The tool must not write the record
        # (terminal_tool.py record write) ...
        result = self._tool(
            monkeypatch, env, f"cd {mine} && sleep 20", "mine", timeout=2
        )
        assert tt.get_session_cwd("mine") == str(mine)
        # ... and must not echo the foreign directory to the model
        # (terminal_tool.py cwd echo). Ungated, env.cwd (theirs) differs from
        # my command_cwd (mine), so the echo would fire with THEIR directory.
        assert "cwd" not in result or result.get("cwd") is None

    def test_completed_command_still_records_and_echoes(self, env, tmp_path, monkeypatch):
        """The gate must not break the observed path: cd still round-trips."""
        target = tmp_path / "target"
        target.mkdir()

        result = self._tool(monkeypatch, env, f"cd {target} && pwd", "sess")
        assert result["exit_code"] == 0
        assert tt.get_session_cwd("sess") == str(target)
        assert os.path.realpath(result["cwd"]) == os.path.realpath(str(target))

"""terminal_tool() must refuse commands referencing a denied secret path
before doing any environment setup — the guard sits ahead of
``_get_env_config()`` so it can't be skipped by an env-selection quirk, and
it is NOT gated by ``force=True`` (force only pre-confirms the separate
dangerous-command check; it isn't a secret-access override).
"""

from __future__ import annotations

import json

import tools.terminal_tool as terminal_tool


def test_env_file_read_is_blocked_before_env_setup(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=live-secret-value", encoding="utf-8")

    def _fail_if_called():
        raise AssertionError("_get_env_config must not run for a blocked command")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _fail_if_called)

    out = json.loads(
        terminal_tool.terminal_tool(command=f"cat {env_file}", workdir=str(tmp_path))
    )
    assert out["status"] == "error"
    assert "secret-bearing environment file" in out["error"]
    assert "live-secret-value" not in json.dumps(out)


def test_force_does_not_bypass_secret_guard(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=live-secret-value", encoding="utf-8")

    def _fail_if_called():
        raise AssertionError("_get_env_config must not run for a blocked command")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _fail_if_called)

    out = json.loads(
        terminal_tool.terminal_tool(
            command=f"cat {env_file}", workdir=str(tmp_path), force=True
        )
    )
    assert out["status"] == "error"
    assert "secret-bearing environment file" in out["error"]


def test_ordinary_command_reaches_env_setup(tmp_path, monkeypatch):
    """Sanity check: an unrelated command still reaches _get_env_config —
    proves the guard isn't accidentally blocking everything. terminal_tool()
    wraps its body in a broad except-Exception, so a raise from the patched
    helper surfaces as an error-status JSON rather than propagating."""
    calls = []

    def _record():
        calls.append(1)
        raise RuntimeError("stop here, we only care that we got this far")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _record)

    out = json.loads(
        terminal_tool.terminal_tool(command="ls -la /tmp", workdir=str(tmp_path))
    )
    assert calls == [1]
    assert out["status"] == "error"


class TestGuardUsesFinalizedCwdNotJustWorkdir:
    """Regression: the guard used to check only the raw workdir= argument,
    which is None for the common case (model didn't pass an explicit
    workdir). The real command later resolves its cwd from a task override,
    a live session cwd (a prior `cd`), or TERMINAL_CWD -- omitted workdir
    must still be checked against THAT cwd, not the Python process's own.
    """

    def test_omitted_workdir_uses_task_override_cwd(self, tmp_path, monkeypatch):
        """Uses auth.json (exact-path credential-store match), not .env
        (blocked by bare basename regardless of which directory it resolves
        into) -- only auth.json actually proves the guard resolved the
        RIGHT directory: it's blocked solely because it resolves to
        <mocked HERMES_HOME>/auth.json, which only happens when the task
        override cwd is honored, not the process's real cwd."""
        import agent.file_safety as fs

        monkeypatch.setattr(fs, "_hermes_home_path", lambda: tmp_path)
        (tmp_path / "auth.json").write_text("{}", encoding="utf-8")

        terminal_tool.register_task_env_overrides("guard-test-task", {"cwd": str(tmp_path)})
        try:
            out = json.loads(
                terminal_tool.terminal_tool(command="cat auth.json", task_id="guard-test-task")
            )
        finally:
            terminal_tool._task_env_overrides.pop("guard-test-task", None)

        assert out["status"] == "error"
        assert "credential store" in out["error"]

    def test_omitted_workdir_uses_live_session_cwd(self, tmp_path, monkeypatch):
        """A prior `cd` in this session set env.cwd; a later bare `cat
        auth.json` (no explicit workdir) must be checked against THAT
        directory, not the task/config default or the process cwd. Uses
        auth.json for the same reason as the task-override test above."""
        import agent.file_safety as fs

        monkeypatch.setattr(fs, "_hermes_home_path", lambda: tmp_path)
        (tmp_path / "auth.json").write_text("{}", encoding="utf-8")

        fake_env = type("FakeEnv", (), {"cwd": str(tmp_path)})()
        monkeypatch.setitem(
            terminal_tool._active_environments, "guard-live-cwd-task", fake_env
        )
        try:
            out = json.loads(
                terminal_tool.terminal_tool(
                    command="cat auth.json", task_id="guard-live-cwd-task"
                )
            )
        finally:
            terminal_tool._active_environments.pop("guard-live-cwd-task", None)

        assert out["status"] == "error"
        assert "credential store" in out["error"]

    def test_explicit_workdir_still_wins_over_live_session_cwd(self, tmp_path, monkeypatch):
        """explicit workdir= must still take priority over a live session
        cwd, matching _resolve_command_cwd's own precedence. HERMES_HOME is
        mocked to decoy_dir: if the guard wrongly preferred the live session
        cwd (decoy_dir) over the explicit workdir (real_dir), a bare
        `cat auth.json` would resolve to decoy_dir/auth.json == mocked
        HERMES_HOME/auth.json and get blocked; with the correct precedence
        it resolves to real_dir/auth.json instead, which isn't HERMES_HOME
        and isn't blocked. Stops the call at _get_env_config (same technique
        as test_ordinary_command_reaches_env_setup) so this doesn't need a
        real environment to execute against -- only that the guard let it
        through."""
        import agent.file_safety as fs

        decoy_dir = tmp_path / "decoy"
        decoy_dir.mkdir()
        monkeypatch.setattr(fs, "_hermes_home_path", lambda: decoy_dir)
        real_dir = tmp_path / "real"
        real_dir.mkdir()

        fake_env = type("FakeEnv", (), {"cwd": str(decoy_dir)})()
        monkeypatch.setitem(
            terminal_tool._active_environments, "guard-precedence-task", fake_env
        )

        def _stop_here():
            raise RuntimeError("stop here, we only care whether the guard blocked it")

        monkeypatch.setattr(terminal_tool, "_get_env_config", _stop_here)

        try:
            out = json.loads(
                terminal_tool.terminal_tool(
                    command="cat auth.json",
                    task_id="guard-precedence-task",
                    workdir=str(real_dir),
                )
            )
        finally:
            terminal_tool._active_environments.pop("guard-precedence-task", None)

        # real_dir isn't HERMES_HOME (decoy_dir is) -- must not be blocked,
        # so execution reaches (and fails at) _get_env_config instead.
        assert "credential store" not in out.get("error", "")
        assert "stop here" in out.get("error", "")

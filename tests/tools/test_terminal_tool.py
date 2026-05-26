"""Regression tests for sudo detection and sudo password handling."""

import json

import tools.terminal_tool as terminal_tool


def setup_function():
    terminal_tool._reset_cached_sudo_passwords()


def teardown_function():
    terminal_tool._reset_cached_sudo_passwords()


def test_searching_for_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "rg --line-number --no-heading --with-filename 'sudo' . | head -n 20"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_printf_literal_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "printf '%s\\n' sudo"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_non_command_argument_named_sudo_does_not_trigger_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    command = "grep -n sudo README.md"
    transformed, sudo_stdin = terminal_tool._transform_sudo_command(command)

    assert transformed == command
    assert sudo_stdin is None


def test_actual_sudo_command_uses_configured_password(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")

    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


def test_actual_sudo_after_leading_env_assignment_is_rewritten(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("DEBUG=1 sudo whoami")

    assert transformed == "DEBUG=1 sudo -S -p '' whoami"
    assert sudo_stdin == "testpass\n"


def test_explicit_empty_sudo_password_tries_empty_without_prompt(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError("interactive sudo prompt should not run for explicit empty password")

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo true")

    assert transformed == "sudo -S -p '' true"
    assert sudo_stdin == "\n"


def test_cached_sudo_password_is_used_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    terminal_tool._set_cached_sudo_password("cached-pass")

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("echo ok && sudo whoami")

    assert transformed == "echo ok && sudo -S -p '' whoami"
    assert sudo_stdin == "cached-pass\n"


def test_cached_sudo_password_isolated_by_session_key(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    terminal_tool._set_cached_sudo_password("alpha-pass")

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-b")
    assert terminal_tool._get_cached_sudo_password() == ""

    monkeypatch.setenv("HERMES_SESSION_KEY", "session-a")
    assert terminal_tool._get_cached_sudo_password() == "alpha-pass"


def test_passwordless_sudo_skips_interactive_prompt_and_rewrite(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")

    def _fail_prompt(*_args, **_kwargs):
        raise AssertionError(
            "interactive sudo prompt should not run when sudo -n already works"
        )

    monkeypatch.setattr(terminal_tool, "_prompt_for_sudo_password", _fail_prompt)
    monkeypatch.setattr(terminal_tool, "_sudo_nopasswd_works", lambda: True, raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo whoami")

    assert transformed == "sudo whoami"
    assert sudo_stdin is None


def test_passwordless_sudo_probe_rechecks_local_terminal(monkeypatch):
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    calls = []

    class Result:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return Result(0 if len(calls) == 1 else 1)

    monkeypatch.setattr(terminal_tool.subprocess, "run", fake_run)

    assert terminal_tool._sudo_nopasswd_works() is True
    assert terminal_tool._sudo_nopasswd_works() is False
    assert len(calls) == 2
    assert calls[0][0] == ["sudo", "-n", "true"]
    assert calls[1][0] == ["sudo", "-n", "true"]


def test_passwordless_sudo_probe_is_disabled_for_nonlocal_terminal_env(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    def _fail_run(*_args, **_kwargs):
        raise AssertionError("host sudo probe must not run for non-local terminal envs")

    monkeypatch.setattr(terminal_tool.subprocess, "run", _fail_run)

    assert terminal_tool._sudo_nopasswd_works() is False


def test_force_path_still_runs_unbypassable_guards(monkeypatch):
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    result = terminal_tool._check_unbypassable_guards("echo x > ~/.hermes/config.yaml", "local")
    assert result["approved"] is False
    assert result.get("protected_config") is True


def test_force_terminal_tool_blocks_protected_config_before_execution(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    repo_dir = hermes_home / "hermes-agent"
    repo_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.chdir(repo_dir)

    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(hermes_home),
            "timeout": 30,
        },
    )
    executed = []

    class FakeEnv:
        env = {}
        cwd = str(hermes_home)

        def execute(self, *_args, **_kwargs):
            executed.append(True)
            raise AssertionError("protected config command should not execute")

    old_envs = dict(terminal_tool._active_environments)
    old_activity = dict(terminal_tool._last_activity)
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()
    terminal_tool._active_environments["default"] = FakeEnv()
    try:
        commands = [
            ("echo x > ~/.hermes/config.yaml", None),
            ("echo x > config.yaml", None),
            ("echo x > config.yaml", str(hermes_home)),
        ]
        for command, workdir in commands:
            if workdir is None:
                raw = terminal_tool.terminal_tool(command, force=True)
            else:
                raw = terminal_tool.terminal_tool(command, force=True, workdir=workdir)
            result = json.loads(raw)
            assert result["status"] == "blocked"
            assert result["exit_code"] == -1
            assert "protected Hermes config/env" in result["error"]
    finally:
        terminal_tool._active_environments.clear()
        terminal_tool._active_environments.update(old_envs)
        terminal_tool._last_activity.clear()
        terminal_tool._last_activity.update(old_activity)

    assert executed == []




def test_force_terminal_tool_resolves_relative_workdir_like_execution(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes-home"
    repo_dir = hermes_home / "hermes-agent"
    repo_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": str(hermes_home),
            "timeout": 30,
        },
    )

    executed = []

    class FakeEnv:
        env = {}
        cwd = str(repo_dir)

        def execute(self, *_args, **_kwargs):
            executed.append(True)
            raise AssertionError("protected config command should not execute")

    old_envs = dict(terminal_tool._active_environments)
    old_activity = dict(terminal_tool._last_activity)
    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()
    terminal_tool._active_environments["default"] = FakeEnv()
    try:
        raw = terminal_tool.terminal_tool("echo x > config.yaml", force=True, workdir="..")
        result = json.loads(raw)
        assert result["status"] == "blocked"
        assert result["exit_code"] == -1
        assert "protected Hermes config/env" in result["error"]
    finally:
        terminal_tool._active_environments.clear()
        terminal_tool._active_environments.update(old_envs)
        terminal_tool._last_activity.clear()
        terminal_tool._last_activity.update(old_activity)

    assert executed == []

def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def test_process_stdin_to_shell_runs_unbypassable_guards(tmp_path):
    from tools.process_registry import ProcessRegistry, ProcessSession

    registry = ProcessRegistry()
    session = ProcessSession(
        id="proc_test",
        command="bash",
        cwd=str(tmp_path),
        started_at=0,
        env_type="local",
    )

    class FakePty:
        def write(self, _data):
            raise AssertionError("blocked stdin must not be written")

    session._pty = FakePty()
    registry._running[session.id] = session

    result = registry.submit_stdin(session.id, "echo x > ~/.hermes/config.yaml")
    assert result["status"] == "blocked"
    assert result["exit_code"] == -1
    assert "protected Hermes config/env" in result["error"]

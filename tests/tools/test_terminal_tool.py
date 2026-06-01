"""Regression tests for sudo detection and sudo password handling."""

import json
from types import SimpleNamespace

import tools.terminal_tool as terminal_tool
from tools import process_registry as process_registry_module


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


def test_validate_workdir_allows_windows_drive_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project") is None
    assert terminal_tool._validate_workdir("C:/Users/Alice/project") is None


def test_validate_workdir_allows_windows_unc_paths():
    assert terminal_tool._validate_workdir(r"\\server\share\project") is None


def test_validate_workdir_blocks_shell_metacharacters_in_windows_paths():
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project; rm -rf /")
    assert terminal_tool._validate_workdir(r"C:\Users\Alice\project$(whoami)")
    assert terminal_tool._validate_workdir("C:\\Users\\Alice\\project\nwhoami")


def _base_terminal_config(tmp_path):
    return {
        "env_type": "local",
        "modal_mode": "auto",
        "docker_image": "",
        "singularity_image": "",
        "modal_image": "",
        "daytona_image": "",
        "cwd": str(tmp_path),
        "timeout": 30,
    }


def test_codex_exec_detection_handles_paths_and_env_prefixes():
    commands = [
        "codex-yuna exec fix tests",
        "codex exec fix tests",
        "/usr/local/bin/codex-yuna exec fix tests",
        "./bin/codex exec fix tests",
        "FOO=bar OPENAI_API_KEY=x codex-yuna exec fix tests",
        "env FOO=bar codex exec fix tests",
        "env -u FOO codex exec fix tests",
        "env -uFOO codex exec fix tests",
        "env -i FOO=bar ./codex-yuna exec fix tests",
    ]

    for command in commands:
        assert terminal_tool._codex_launch_info(command)["is_codex_exec"], command


def test_codex_exec_detection_ignores_non_executable_mentions():
    commands = [
        "echo codex-yuna exec fix tests",
        "python -c 'print(\"codex-yuna exec\")'",
        "python - <<'PY'\ncodex exec sample\nPY",
        "printf '%s\\n' 'codex exec fix tests'",
    ]

    for command in commands:
        assert not terminal_tool._codex_launch_info(command)["is_codex_exec"], command


def test_codex_help_and_version_are_harmless():
    commands = [
        "codex-yuna --version",
        "codex --version",
        "codex-yuna exec --help",
        "codex exec help",
        "codex exec -h",
        "codex exec --version",
    ]

    for command in commands:
        info = terminal_tool._codex_launch_info(command)
        assert info["is_harmless"], command
        assert not info["is_codex_exec"], command


def test_terminal_blocks_foreground_codex_exec_before_launch(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_ALLOW_FOREGROUND_CODEX_EXEC", raising=False)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_terminal_config(tmp_path))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    result = json.loads(
        terminal_tool.terminal_tool(
            "codex-yuna exec implement the task",
            background=False,
            pty=True,
        )
    )

    assert result["status"] == "error"
    assert result["codex_exec_policy"] == "blocked"
    assert "background=true" in result["error"]
    assert "notify_on_complete=true" in result["error"]
    assert "pty=true" in result["error"]
    assert "HERMES_ALLOW_FOREGROUND_CODEX_EXEC" in result["error"]


def test_terminal_blocks_codex_exec_without_pty(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_terminal_config(tmp_path))
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)

    result = json.loads(
        terminal_tool.terminal_tool(
            "codex exec implement the task",
            background=True,
            notify_on_complete=True,
            pty=False,
        )
    )

    assert result["status"] == "error"
    assert result["codex_exec_policy"] == "blocked"
    assert "pty=true" in result["error"]


def test_terminal_allows_codex_help_foreground(monkeypatch, tmp_path):
    config = _base_terminal_config(tmp_path)
    executed = {}

    class DummyEnv:
        def execute(self, command, **kwargs):
            executed["command"] = command
            executed["kwargs"] = kwargs
            return {"output": "usage: codex exec", "returncode": 0}

    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setitem(terminal_tool._active_environments, "default", DummyEnv())
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    try:
        result = json.loads(terminal_tool.terminal_tool("codex exec --help"))
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert result["exit_code"] == 0
    assert executed["command"] == "codex exec --help"


def test_background_codex_exec_without_notify_defaults_notify_on_complete(monkeypatch, tmp_path):
    config = _base_terminal_config(tmp_path)
    dummy_env = SimpleNamespace(env={})
    spawned = {}

    def fake_spawn_local(**_kwargs):
        spawned.update(_kwargs)
        return SimpleNamespace(
            id="proc_codex",
            pid=1234,
            notify_on_complete=False,
            watcher_platform="",
        )

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                "codex exec implement the task",
                background=True,
                pty=True,
                notify_on_complete=False,
            )
        )
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert spawned["use_pty"] is True
    assert result["session_id"] == "proc_codex"
    assert result["codex_exec_policy"] == "notify_on_complete_defaulted"
    assert result["notify_on_complete_defaulted"] is True
    assert result["notify_on_complete"] is True
    assert "Hermes defaulted notify_on_complete=true" in result["hint"]


def test_background_codex_exec_with_watch_patterns_prefers_completion(monkeypatch, tmp_path):
    config = _base_terminal_config(tmp_path)
    dummy_env = SimpleNamespace(env={})

    def fake_spawn_local(**_kwargs):
        return SimpleNamespace(
            id="proc_codex_watch",
            pid=1234,
            notify_on_complete=False,
            watcher_platform="",
        )

    monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                "codex exec implement the task",
                background=True,
                pty=True,
                notify_on_complete=False,
                watch_patterns=["DONE"],
            )
        )
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert result["codex_exec_policy"] == "notify_on_complete_defaulted"
    assert result["notify_on_complete"] is True
    assert "watch_patterns ignored" in result["watch_patterns_ignored"]


def test_background_codex_exec_default_notify_registers_gateway_watcher(monkeypatch, tmp_path):
    config = _base_terminal_config(tmp_path)
    dummy_env = SimpleNamespace(env={})
    original_watchers = list(process_registry_module.process_registry.pending_watchers)

    def fake_spawn_local(**_kwargs):
        return SimpleNamespace(
            id="proc_codex_gateway",
            pid=1234,
            notify_on_complete=False,
            watcher_platform="",
        )

    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "qqbot")
    monkeypatch.setenv("HERMES_SESSION_CHAT_ID", "chat-1")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "user-1")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setattr(process_registry_module.process_registry, "spawn_local", fake_spawn_local)
    monkeypatch.setitem(terminal_tool._active_environments, "default", dummy_env)
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)
    process_registry_module.process_registry.pending_watchers.clear()

    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                "codex exec implement the task",
                background=True,
                pty=True,
                notify_on_complete=False,
            )
        )
        watchers = list(process_registry_module.process_registry.pending_watchers)
    finally:
        process_registry_module.process_registry.pending_watchers[:] = original_watchers
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert result["notify_on_complete"] is True
    assert watchers
    assert watchers[-1]["session_id"] == "proc_codex_gateway"
    assert watchers[-1]["platform"] == "qqbot"
    assert watchers[-1]["notify_on_complete"] is True


def test_foreground_codex_exec_escape_hatch_bypasses_only_foreground_guard(monkeypatch, tmp_path):
    config = _base_terminal_config(tmp_path)
    executed = {}

    class DummyEnv:
        def execute(self, command, **kwargs):
            executed["command"] = command
            executed["kwargs"] = kwargs
            return {"output": "ok", "returncode": 0}

    monkeypatch.setenv("HERMES_ALLOW_FOREGROUND_CODEX_EXEC", "1")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *_args, **_kwargs: {"approved": True})
    monkeypatch.setitem(terminal_tool._active_environments, "default", DummyEnv())
    monkeypatch.setitem(terminal_tool._last_activity, "default", 0.0)

    try:
        result = json.loads(
            terminal_tool.terminal_tool(
                "codex exec implement the task",
                background=False,
                pty=True,
            )
        )
    finally:
        terminal_tool._active_environments.pop("default", None)
        terminal_tool._last_activity.pop("default", None)

    assert result["exit_code"] == 0
    assert executed["command"] == "codex exec implement the task"

"""Tests for the terminal_tool main path and support functions.

Focus areas:
  * check_terminal_requirements()  -- backend requirement gates
  * terminal_tool()                -- the tool entry point (validation, sudo,
                                      foreground execution, background spawn,
                                      signal interpretation)
  * _handle_terminal()             -- tool handler wrapper (notify mapping,
                                      argument recovery)
  * _check_vercel_sandbox_requirements() -- Vercel auth/runtime gates
"""

import json
from types import SimpleNamespace, ModuleType

import pytest

import tools.terminal_tool as terminal_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_config(**overrides):
    """A complete local-backend config dict, mirroring _get_env_config output."""
    config = {
        "env_type": "local",
        "modal_mode": "auto",
        "docker_image": "img",
        "docker_forward_env": [],
        "singularity_image": "simg",
        "modal_image": "mimg",
        "daytona_image": "dimg",
        "vercel_runtime": "",
        "cwd": "/tmp",
        "host_cwd": None,
        "docker_mount_cwd_to_workspace": False,
        "timeout": 30,
        "lifetime_seconds": 300,
        "ssh_host": "",
        "ssh_user": "",
        "ssh_port": 22,
        "ssh_key": "",
        "ssh_persistent": False,
        "local_persistent": False,
        "container_cpu": 1,
        "container_memory": 5120,
        "container_disk": 51200,
        "container_persistent": True,
        "docker_volumes": [],
        "docker_env": {},
        "docker_run_as_host_user": False,
        "docker_network": True,
        "docker_extra_args": [],
        "docker_shm_size": "1g",
        "docker_persist_across_processes": True,
        "docker_shared_container_key": "",
        "docker_orphan_reaper": True,
    }
    config.update(overrides)
    return config


class MockEnv:
    """Stand-in for a base-environment object returned by _create_environment."""

    def __init__(self, execute_result=None, cwd=None):
        self.execute_result = execute_result or {"output": "hello", "returncode": 0, "error": ""}
        self.cwd = cwd
        self.persistent = False
        self.env = {}  # background local path reads env.env
        self.executed = []

    def execute(self, command, **kwargs):
        self.executed.append((command, kwargs))
        return dict(self.execute_result)

    def cleanup(self):
        pass


def _patch_guard_headroom(monkeypatch, config=None):
    """Mock everything needed to get *past* config resolution and into execution.

    Leaves env creation and guards to the caller where relevant.
    """
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config or _base_config())
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: False)
    # Fresh module state so no previously-created env leaks into this test.
    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})


def _patch_foreground_success(monkeypatch, env, config=None):
    """Full mock set for a foreground command that reaches env.execute()."""
    _patch_guard_headroom(monkeypatch, config)
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})


def _patch_vercel_secrets(monkeypatch, mapping):
    monkeypatch.setattr(
        "agent.secret_scope.get_secret", lambda name: mapping.get(name)
    )


# ---------------------------------------------------------------------------
# check_terminal_requirements
# ---------------------------------------------------------------------------


def test_requirements_local_backend(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_config(env_type="local"))
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_docker_found_and_version_works(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_config(env_type="docker"))
    monkeypatch.setattr("tools.environments.docker.find_docker", lambda: "/usr/bin/docker")
    monkeypatch.setattr(terminal_tool.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_docker_missing(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_config(env_type="docker"))
    monkeypatch.setattr("tools.environments.docker.find_docker", lambda: None)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_singularity_found_and_version_works(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_config(env_type="singularity"))
    monkeypatch.setattr(terminal_tool.shutil, "which", lambda name: "/usr/bin/apptainer")
    monkeypatch.setattr(terminal_tool.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_ssh_ok(monkeypatch):
    config = _base_config(env_type="ssh", ssh_host="h", ssh_user="u")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_ssh_missing_user(monkeypatch):
    config = _base_config(env_type="ssh", ssh_host="h", ssh_user="")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_managed_backend(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="managed")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_get_modal_backend_state", lambda mode: {"selected_backend": "managed"})
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_modal_direct_no_credentials(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="direct")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(
        terminal_tool, "_get_modal_backend_state",
        lambda mode: {"selected_backend": "direct", "mode": "direct"},
    )
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: None)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_vercel_delegates_to_checker(monkeypatch):
    config = _base_config(env_type="vercel_sandbox")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_check_vercel_sandbox_requirements", lambda cfg: True)
    assert terminal_tool.check_terminal_requirements() is True


def _patch_daytona_import(monkeypatch):
    """Make ``from daytona import Daytona`` succeed even without the SDK."""
    fake = ModuleType("daytona")
    fake.Daytona = object()
    monkeypatch.setitem(__import__("sys").modules, "daytona", fake)


def test_requirements_daytona_with_api_key(monkeypatch):
    _patch_daytona_import(monkeypatch)
    config = _base_config(env_type="daytona")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name: "key" if name == "DAYTONA_API_KEY" else None)
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_daytona_without_api_key(monkeypatch):
    _patch_daytona_import(monkeypatch)
    config = _base_config(env_type="daytona")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr("agent.secret_scope.get_secret", lambda name: None)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_unknown_env_with_plugin_provider(monkeypatch):
    config = _base_config(env_type="custom-env")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    provider = SimpleNamespace(check_requirements=lambda cfg: True)
    monkeypatch.setattr(terminal_tool, "_get_plugin_env_provider", lambda env_type: provider)
    assert terminal_tool.check_terminal_requirements() is True


def test_requirements_unknown_env_without_plugin(monkeypatch):
    config = _base_config(env_type="custom-env")
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_get_plugin_env_provider", lambda env_type: None)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_exception_returns_false(monkeypatch):
    def _boom():
        raise RuntimeError("config exploded")

    monkeypatch.setattr(terminal_tool, "_get_env_config", _boom)
    assert terminal_tool.check_terminal_requirements() is False


# ---------------------------------------------------------------------------
# _check_vercel_sandbox_requirements
# ---------------------------------------------------------------------------

_VERCEL_OK_SPEC = object()


def _vercel_config(**overrides):
    config = {"vercel_runtime": "", "container_disk": 51200}
    config.update(overrides)
    return config


def test_vercel_unsupported_runtime(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: _VERCEL_OK_SPEC)
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config(vercel_runtime="node20")) is False


def test_vercel_invalid_disk(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: _VERCEL_OK_SPEC)
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config(container_disk=8192)) is False


def test_vercel_sdk_not_installed(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: None)
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config()) is False


def test_vercel_oidc_token_auth(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: _VERCEL_OK_SPEC)
    _patch_vercel_secrets(monkeypatch, {"VERCEL_OIDC_TOKEN": "tok"})
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config()) is True


def test_vercel_full_token_auth(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: _VERCEL_OK_SPEC)
    _patch_vercel_secrets(
        monkeypatch,
        {"VERCEL_TOKEN": "t", "VERCEL_PROJECT_ID": "p", "VERCEL_TEAM_ID": "m"},
    )
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config()) is True


def test_vercel_partial_token_auth(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: _VERCEL_OK_SPEC)
    _patch_vercel_secrets(monkeypatch, {"VERCEL_TOKEN": "t", "VERCEL_PROJECT_ID": "p"})
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config()) is False


def test_vercel_no_auth_at_all(monkeypatch):
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: _VERCEL_OK_SPEC)
    _patch_vercel_secrets(monkeypatch, {})
    assert terminal_tool._check_vercel_sandbox_requirements(_vercel_config()) is False


# ---------------------------------------------------------------------------
# terminal_tool -- entry-point validation & guidance
# ---------------------------------------------------------------------------


def test_non_string_command_returns_error_json():
    result = json.loads(terminal_tool.terminal_tool(1234))
    assert result["exit_code"] == -1
    assert result["status"] == "error"
    assert "expected string, got int" in result["error"]


def test_invalid_timeout_rejected(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    result = json.loads(terminal_tool.terminal_tool("echo hi", timeout=0))
    assert "timeout must be a positive number of seconds" in result["error"]


def test_foreground_timeout_over_cap_rejected(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    over_cap = terminal_tool.FOREGROUND_MAX_TIMEOUT + 1
    result = json.loads(terminal_tool.terminal_tool("echo hi", timeout=over_cap))
    assert "exceeds the maximum" in result["error"]
    assert str(over_cap) in result["error"]


def test_foreground_long_lived_command_flagged(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    result = json.loads(terminal_tool.terminal_tool("nohup sleep 100"))
    assert result["exit_code"] == -1
    assert "nohup/disown/setsid" in result["error"]


def test_workdir_validation_failure(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    result = json.loads(terminal_tool.terminal_tool("echo hi", workdir="/tmp; rm -rf /"))
    assert result["status"] == "blocked"
    assert "workdir contains disallowed character" in result["error"]


# ---------------------------------------------------------------------------
# terminal_tool -- foreground execution
# ---------------------------------------------------------------------------


def test_foreground_success_returns_result_json(monkeypatch):
    env = MockEnv(execute_result={"output": "hello world", "returncode": 0, "error": ""})
    _patch_foreground_success(monkeypatch, env)

    result = json.loads(terminal_tool.terminal_tool("echo hello world"))
    assert result["output"] == "hello world"
    assert result["exit_code"] == 0
    assert result["error"] is None
    # The command must be forwarded to the environment, with bounded capture on.
    assert env.executed[0][0] == "echo hello world"
    assert env.executed[0][1].get("bounded_capture") is True


def test_signal_killed_command_includes_signal_note(monkeypatch):
    env = MockEnv(execute_result={"output": "killed", "returncode": -9, "error": ""})
    _patch_foreground_success(monkeypatch, env)

    result = json.loads(terminal_tool.terminal_tool("run-something"))
    assert result["exit_code"] == -9
    assert "signal 9" in result["exit_code_meaning"]


def test_sudo_command_forwards_to_env(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "secret")
    try:
        env = MockEnv(execute_result={"output": "done", "returncode": 0, "error": ""})
        _patch_foreground_success(monkeypatch, env)

        result = json.loads(terminal_tool.terminal_tool("sudo apt install -y ripgrep"))
        assert result["exit_code"] == 0
        assert env.executed[0][0] == "sudo apt install -y ripgrep"
    finally:
        monkeypatch.delenv("SUDO_PASSWORD", raising=False)


def test_transform_sudo_command_rewrites_and_returns_stdin(monkeypatch):
    monkeypatch.setenv("SUDO_PASSWORD", "testpass")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)

    transformed, sudo_stdin = terminal_tool._transform_sudo_command("sudo apt install -y ripgrep")
    assert transformed == "sudo -S -p '' apt install -y ripgrep"
    assert sudo_stdin == "testpass\n"


# ---------------------------------------------------------------------------
# terminal_tool -- background spawn & notify/watch conflict
# ---------------------------------------------------------------------------


class FakeProcSession:
    id = "proc-123"
    pid = 9999
    watcher_platform = ""
    watcher_interval = 5
    notify_on_complete = False
    watch_patterns = None
    parent_session_id = ""


class FakeProcessRegistry:
    def __init__(self):
        self.pending_watchers = []

    def spawn_local(self, **kwargs):
        return FakeProcSession()

    def spawn_via_env(self, **kwargs):
        return FakeProcSession()


def test_background_notify_watch_conflict(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})

    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())

    import gateway.session_context as sc
    monkeypatch.setattr(sc, "async_delivery_supported", lambda: True)
    monkeypatch.setattr(sc, "get_session_env", lambda key, default="": default)

    result = json.loads(
        terminal_tool.terminal_tool(
            "run-server",
            background=True,
            notify_on_complete=True,
            watch_patterns=["ready"],
        )
    )
    assert result["exit_code"] == 0
    # notify_on_complete wins over watch_patterns; watch_patterns is dropped.
    assert result["notify_on_complete"] is True
    assert "watch_patterns_ignored" in result
    assert "duplicate notifications" in result["watch_patterns_ignored"]
    assert "watch_patterns" not in result


# ---------------------------------------------------------------------------
# _handle_sudo_failure
# ---------------------------------------------------------------------------


def test_handle_sudo_failure_gateway_context_tip(monkeypatch):
    monkeypatch.setattr(
        terminal_tool, "env_var_enabled",
        lambda name, default="": name == "HERMES_GATEWAY_SESSION",
    )
    monkeypatch.setattr(terminal_tool, "_in_delegated_child_context", lambda: False)
    out = terminal_tool._handle_sudo_failure("sudo: a password is required", "local")
    assert "To enable sudo over messaging" in out


def test_handle_sudo_failure_delegated_child_tip(monkeypatch):
    monkeypatch.setattr(terminal_tool, "env_var_enabled", lambda name, default="": False)
    monkeypatch.setattr(terminal_tool, "_in_delegated_child_context", lambda: True)
    out = terminal_tool._handle_sudo_failure("sudo: a password is required", "local")
    assert "Subagents cannot prompt for a sudo password" in out


def test_handle_sudo_failure_headless_passthrough(monkeypatch):
    monkeypatch.setattr(terminal_tool, "env_var_enabled", lambda name, default="": False)
    monkeypatch.setattr(terminal_tool, "_in_delegated_child_context", lambda: False)
    out = terminal_tool._handle_sudo_failure("sudo: a password is required", "local")
    assert out == "sudo: a password is required"


def test_handle_sudo_failure_ignores_other_output(monkeypatch):
    monkeypatch.setattr(
        terminal_tool, "env_var_enabled",
        lambda name, default="": name == "HERMES_GATEWAY_SESSION",
    )
    monkeypatch.setattr(terminal_tool, "_in_delegated_child_context", lambda: False)
    out = terminal_tool._handle_sudo_failure("all good here", "local")
    assert out == "all good here"


# ---------------------------------------------------------------------------
# _handle_terminal
# ---------------------------------------------------------------------------


def test_handle_terminal_delegates_to_terminal_tool(monkeypatch):
    captured = {}

    def _fake_terminal_tool(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    monkeypatch.setattr(terminal_tool, "terminal_tool", _fake_terminal_tool)
    res = terminal_tool._handle_terminal(
        {"command": "ls", "background": True, "workdir": "/tmp", "timeout": 5}
    )
    assert res == '{"ok": true}'
    assert captured["command"] == "ls"
    assert captured["background"] is True
    assert captured["workdir"] == "/tmp"
    assert captured["timeout"] == 5


def test_handle_terminal_rejects_code_misrouting():
    res = terminal_tool._handle_terminal({"code": "print(1)"})
    assert "terminal received a 'code' parameter" in res


def test_handle_terminal_rejects_notify_on_foreground():
    res = terminal_tool._handle_terminal({"command": "ls", "notify": True})
    assert "notify only applies to background commands" in res


def test_handle_terminal_rejects_pty_on_foreground():
    res = terminal_tool._handle_terminal({"command": "ls", "pty": True})
    assert "pty requires background=true" in res


def test_handle_terminal_notify_bool_maps_to_notify_on_complete(monkeypatch):
    captured = {}
    monkeypatch.setattr(terminal_tool, "terminal_tool", lambda **kw: captured.update(kw) or "{}")
    terminal_tool._handle_terminal({"command": "job", "background": True, "notify": True})
    assert captured["notify_on_complete"] is True
    assert captured["watch_patterns"] is None


def test_handle_terminal_notify_list_maps_to_watch_patterns(monkeypatch):
    captured = {}
    monkeypatch.setattr(terminal_tool, "terminal_tool", lambda **kw: captured.update(kw) or "{}")
    terminal_tool._handle_terminal({"command": "server", "background": True, "notify": ["ready"]})
    assert captured["watch_patterns"] == ["ready"]
    assert captured["notify_on_complete"] is False


def test_handle_terminal_rejects_bad_notify_type():
    res = terminal_tool._handle_terminal({"command": "ls", "background": True, "notify": "boom"})
    assert "notify must be true/false" in res


# ---------------------------------------------------------------------------
# terminal_tool -- gateway lifecycle guard (supervised gateway process)
# ---------------------------------------------------------------------------


class _RaiseEnv(MockEnv):
    """A MockEnv whose execute() raises a given exception each call."""

    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def execute(self, command, **kwargs):
        self.executed.append((command, kwargs))
        raise self.exc


def _patch_gateway_env(monkeypatch, env=None):
    """Reach the gateway lifecycle guard: supervised gateway probe -> True."""
    _patch_guard_headroom(monkeypatch)
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env or MockEnv())
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "_is_supervised_gateway_process", lambda: True)


def test_gateway_launchctl_submit_blocked(monkeypatch):
    _patch_gateway_env(monkeypatch)
    import cron.lifecycle_guard as clg
    monkeypatch.setattr(clg, "contains_launchctl_submit_command", lambda command: True)
    monkeypatch.setattr(
        clg, "contains_gateway_lifecycle_command_or_referenced_script", lambda *a, **k: False
    )
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["exit_code"] == 1
    assert result["status"] == "error"
    assert "launchctl submit/bootstrap registers a persistent" in result["error"]


def test_gateway_lifecycle_command_blocked(monkeypatch):
    _patch_gateway_env(monkeypatch)
    import cron.lifecycle_guard as clg
    monkeypatch.setattr(clg, "contains_launchctl_submit_command", lambda command: False)
    monkeypatch.setattr(
        clg, "contains_gateway_lifecycle_command_or_referenced_script", lambda *a, **k: True
    )
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["exit_code"] == 1
    assert result["status"] == "error"
    assert "cannot restart, stop, or uninstall the gateway" in result["error"]


def test_gateway_supervised_normal_command_passes(monkeypatch):
    env = MockEnv(execute_result={"output": "ok", "returncode": 0, "error": ""})
    _patch_gateway_env(monkeypatch, env)
    import cron.lifecycle_guard as clg
    monkeypatch.setattr(clg, "contains_launchctl_submit_command", lambda command: False)
    monkeypatch.setattr(
        clg, "contains_gateway_lifecycle_command_or_referenced_script", lambda *a, **k: False
    )
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["exit_code"] == 0
    assert result["output"] == "ok"
    assert env.executed, "env.execute should have been called for a normal command"


# ---------------------------------------------------------------------------
# terminal_tool -- guard blocking / pending / approval notes
# ---------------------------------------------------------------------------


def test_guard_blocked_returns_error(monkeypatch):
    env = MockEnv()
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: {"approved": False, "message": "Explicitly denied", "description": "flagged as dangerous"},
    )
    result = json.loads(terminal_tool.terminal_tool("rm -rf /tmp/foo"))
    assert result["status"] == "blocked"
    assert result["exit_code"] == -1
    assert "Explicitly denied" in result["error"]


def test_guard_blocked_fallback_message(monkeypatch):
    env = MockEnv()
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: {"approved": False, "description": "flagged as dangerous"},
    )
    result = json.loads(terminal_tool.terminal_tool("rm -rf /tmp/foo"))
    assert result["status"] == "blocked"
    assert "Command denied" in result["error"]


def test_guard_pending_approval_returns_pending(monkeypatch):
    env = MockEnv()
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: {
            "approved": False, "status": "pending_approval", "command": "cmd",
            "description": "desc", "pattern_key": "pk", "smart_denied": True,
            "allow_permanent": True,
        },
    )
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["status"] == "pending_approval"
    assert result["approval_pending"] is True
    assert result["command"] == "cmd"
    assert result["pattern_key"] == "pk"


def test_force_skips_guard_check(monkeypatch):
    monkeypatch.setattr("tools.interrupt.clear_current_thread_interrupt", lambda: None)
    env = MockEnv(execute_result={"output": "ran", "returncode": 0, "error": ""})
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: pytest.fail("guard must be skipped when force=True"),
    )
    result = json.loads(terminal_tool.terminal_tool("echo hi", force=True))
    assert result["exit_code"] == 0
    assert env.executed


def test_user_approved_command_gets_foreground_note(monkeypatch):
    monkeypatch.setattr("tools.interrupt.clear_current_thread_interrupt", lambda: None)
    env = MockEnv(execute_result={"output": "out", "returncode": 0, "error": ""})
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: {"approved": True, "user_approved": True, "description": "flagged as dangerous"},
    )
    result = json.loads(terminal_tool.terminal_tool("run-cmd"))
    assert result["exit_code"] == 0
    assert "approved by the user" in result["approval"]


def test_smart_approved_command_gets_foreground_note(monkeypatch):
    env = MockEnv(execute_result={"output": "out", "returncode": 0, "error": ""})
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: {"approved": True, "smart_approved": True, "description": "flagged"},
    )
    result = json.loads(terminal_tool.terminal_tool("run-cmd"))
    assert result["exit_code"] == 0
    assert "auto-approved by smart approval" in result["approval"]


# ---------------------------------------------------------------------------
# terminal_tool -- pty / background execution path variants
# ---------------------------------------------------------------------------


def test_pty_disabled_for_pipe_stdin_background(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    monkeypatch.setattr(terminal_tool, "_command_requires_pipe_stdin", lambda command: True)
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    result = json.loads(terminal_tool.terminal_tool("gh auth login --with-token", background=True, pty=True))
    assert "PTY disabled" in result["pty_note"]


def test_background_spawn_via_env_nonlocal(monkeypatch):
    config = _base_config(env_type="ssh", ssh_host="h", ssh_user="u")
    _patch_guard_headroom(monkeypatch, config)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    result = json.loads(terminal_tool.terminal_tool("ls", background=True))
    assert result["exit_code"] == 0
    assert result["session_id"] == "proc-123"


def test_background_user_approved_adds_note(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(
        terminal_tool, "_check_all_guards",
        lambda *a, **k: {"approved": True, "user_approved": True, "description": "flagged as dangerous"},
    )
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    result = json.loads(terminal_tool.terminal_tool("run-server", background=True))
    assert result["exit_code"] == 0
    assert "approved by the user" in result["approval"]


def test_background_silent_process_hint(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    result = json.loads(terminal_tool.terminal_tool("sleep 999", background=True))
    assert "runs SILENTLY" in result["hint"]


def test_background_ci_poller_hint(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    result = json.loads(
        terminal_tool.terminal_tool("gh pr view 1 --json statusCheckRollup --jq .x", background=True)
    )
    assert "homebrewed CI poller" in result["hint"]


def test_background_notify_unsupported_async(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    import gateway.session_context as sc
    monkeypatch.setattr(sc, "async_delivery_supported", lambda: False)
    result = json.loads(terminal_tool.terminal_tool("run", background=True, notify_on_complete=True))
    assert result["notify_on_complete"] is False
    assert "notify_unsupported" in result


def test_background_watcher_platform_registers_watcher(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    reg = FakeProcessRegistry()
    monkeypatch.setattr(pr, "process_registry", reg)
    import gateway.session_context as sc
    monkeypatch.setattr(sc, "async_delivery_supported", lambda: True)

    def _gse(key, default=""):
        return {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_CHAT_ID": "c123",
            "HERMES_SESSION_ID": "sess-1",
        }.get(key, default)

    monkeypatch.setattr(sc, "get_session_env", _gse)
    result = json.loads(terminal_tool.terminal_tool("run", background=True, notify_on_complete=True))
    assert result["notify_on_complete"] is True
    assert len(reg.pending_watchers) == 1
    assert reg.pending_watchers[0]["platform"] == "telegram"
    assert reg.pending_watchers[0]["chat_id"] == "c123"


def test_background_watch_patterns_set(monkeypatch):
    _patch_guard_headroom(monkeypatch)
    env = MockEnv()
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kw: env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda *a, **k: {"approved": True})
    import tools.process_registry as pr
    monkeypatch.setattr(pr, "process_registry", FakeProcessRegistry())
    import gateway.session_context as sc
    monkeypatch.setattr(sc, "async_delivery_supported", lambda: True)
    monkeypatch.setattr(sc, "get_session_env", lambda key, default="": default)
    result = json.loads(terminal_tool.terminal_tool("run-srv", background=True, watch_patterns=["ready"]))
    assert result["watch_patterns"] == ["ready"]


# ---------------------------------------------------------------------------
# terminal_tool -- foreground result processing
# ---------------------------------------------------------------------------


def test_foreground_timeout_exception_returns_124(monkeypatch):
    env = _RaiseEnv(TimeoutError("command timeout"))
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda *a: None)
    result = json.loads(terminal_tool.terminal_tool("sleep 100"))
    assert result["exit_code"] == 124
    assert "timed out after" in result["error"]


def test_foreground_transient_error_retries_then_fails(monkeypatch):
    env = _RaiseEnv(ValueError("boom"))
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(terminal_tool.time, "sleep", lambda *a: None)
    result = json.loads(terminal_tool.terminal_tool("flakey-cmd"))
    assert result["exit_code"] == -1
    assert "Command execution failed" in result["error"]
    assert len(env.executed) == 4  # initial attempt + 3 retries


def test_cwd_observed_records_session_cwd(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    env = MockEnv(
        execute_result={"output": "moved", "returncode": 0, "error": "", "cwd_observed": True},
        cwd="/tmp/newdir",
    )
    _patch_foreground_success(monkeypatch, env)
    result = json.loads(terminal_tool.terminal_tool("cd /tmp/newdir"))
    assert result["exit_code"] == 0
    assert result["cwd"] == "/tmp/newdir"
    assert terminal_tool.get_session_cwd("") == "/tmp/newdir"


def test_spill_path_reports_truncation_metadata(monkeypatch, tmp_path):
    spill = tmp_path / "spill.log"
    spill.write_text("full-spill-content")
    env = MockEnv(
        execute_result={
            "output": "out", "returncode": 0, "error": "",
            "output_total_chars": 100, "full_output_path": str(spill),
        }
    )
    _patch_foreground_success(monkeypatch, env)
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["full_output_path"] == str(spill)
    assert result["output_total_chars"] == 100
    assert "truncation_note" in result


def test_verification_evidence_recorded(monkeypatch):
    env = MockEnv(execute_result={"output": "o", "returncode": 0, "error": ""})
    _patch_foreground_success(monkeypatch, env)
    monkeypatch.setattr(
        "agent.verification_evidence.record_terminal_result",
        lambda **kw: {"status": "verified", "kind": "k", "scope": "s", "canonical_command": "c"},
    )
    result = json.loads(terminal_tool.terminal_tool("echo hi"))
    assert result["verification_evidence"]["status"] == "verified"


def test_sudo_auth_failure_invalidates_cache_and_reprompts(monkeypatch):
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(terminal_tool, "_sudo_password_cache", {})
    # A STABLE callback object so the sudo-password cache scope
    # (``callback:{id(cb)}``) is identical across set and get calls.
    _pw_cb = lambda: "pw"  # noqa: E731
    monkeypatch.setattr(terminal_tool, "_get_sudo_password_callback", lambda: _pw_cb)
    terminal_tool._set_cached_sudo_password("secret")
    monkeypatch.setattr(terminal_tool, "_in_delegated_child_context", lambda: False)
    monkeypatch.setattr(terminal_tool, "env_var_enabled", lambda name, default="": False)
    env = MockEnv(execute_result={"output": "sudo: authentication failed\nbad", "returncode": 1, "error": ""})
    _patch_foreground_success(monkeypatch, env)
    result = json.loads(terminal_tool.terminal_tool("sudo apt install -y x"))
    assert result["exit_code"] == 1
    assert result["sudo_auth_failed"] is True
    assert result["sudo_cache_cleared"] is True
    assert "Sudo authentication failed" in result["output"]


# ---------------------------------------------------------------------------
# check_terminal_requirements -- singularity missing & modal branches
# ---------------------------------------------------------------------------


def test_requirements_singularity_missing(monkeypatch):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: _base_config(env_type="singularity"))
    monkeypatch.setattr(terminal_tool.shutil, "which", lambda name: None)
    assert terminal_tool.check_terminal_requirements() is False


def _patch_modal_state(monkeypatch, config, state, managed_enabled=False):
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: config)
    monkeypatch.setattr(terminal_tool, "_get_modal_backend_state", lambda mode: state)
    monkeypatch.setattr(terminal_tool, "managed_nous_tools_enabled", lambda: managed_enabled)
    monkeypatch.setattr(
        terminal_tool, "nous_tool_gateway_unavailable_message", lambda *a, **k: "gateway unavailable"
    )


def test_requirements_modal_managed_blocked(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="managed")
    state = {"selected_backend": None, "managed_mode_blocked": True, "mode": "managed"}
    _patch_modal_state(monkeypatch, config, state)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_managed_unavailable(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="managed")
    state = {"selected_backend": None, "managed_mode_blocked": False, "mode": "managed"}
    _patch_modal_state(monkeypatch, config, state)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_direct_nocreds_managed_enabled(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="direct")
    state = {"selected_backend": None, "managed_mode_blocked": False, "mode": "direct"}
    _patch_modal_state(monkeypatch, config, state, managed_enabled=True)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_direct_nocreds_managed_disabled(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="direct")
    state = {"selected_backend": None, "managed_mode_blocked": False, "mode": "direct"}
    _patch_modal_state(monkeypatch, config, state, managed_enabled=False)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_auto_nocreds_managed_enabled(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="auto")
    state = {"selected_backend": None, "managed_mode_blocked": False, "mode": "auto"}
    _patch_modal_state(monkeypatch, config, state, managed_enabled=True)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_auto_nocreds_managed_disabled(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="auto")
    state = {"selected_backend": None, "managed_mode_blocked": False, "mode": "auto"}
    _patch_modal_state(monkeypatch, config, state, managed_enabled=False)
    assert terminal_tool.check_terminal_requirements() is False


def test_requirements_modal_direct_with_sdk_present(monkeypatch):
    config = _base_config(env_type="modal", modal_mode="direct")
    state = {"selected_backend": "direct", "managed_mode_blocked": False, "mode": "direct"}
    _patch_modal_state(monkeypatch, config, state)
    monkeypatch.setattr(terminal_tool.importlib.util, "find_spec", lambda name: object())
    assert terminal_tool.check_terminal_requirements() is True

"""Unit tests for the Novita AI cloud sandbox environment backend."""

import threading
import types as _types
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Novita SDK objects
# ---------------------------------------------------------------------------


@dataclass
class _CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None


@dataclass
class _CommandExitException(Exception):
    """Minimal stand-in for novita_sandbox CommandExitException."""
    stdout: str
    stderr: str
    exit_code: int
    error: Optional[str] = None


def _make_run_response(stdout="", stderr="", exit_code=0):
    return _CommandResult(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _make_exit_exception(stdout="", stderr="", exit_code=1):
    return _CommandExitException(stdout=stdout, stderr=stderr, exit_code=exit_code)


def _make_sandbox(sandbox_id="sb-novita-123"):
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    sb.commands.run.return_value = _make_run_response()
    return sb


def _make_sandbox_info(sandbox_id="sb-novita-existing"):
    info = MagicMock()
    info.sandbox_id = sandbox_id
    return info


def _make_paginator(items=None):
    paginator = MagicMock()
    if items:
        paginator.has_next = True
        paginator.next_items.return_value = items
    else:
        paginator.has_next = False
    return paginator


def _patch_novita_imports(monkeypatch):
    """Patch the novita_sandbox SDK so NovitaEnvironment can be imported without it."""
    novita_mod = _types.ModuleType("novita_sandbox")
    novita_core_mod = _types.ModuleType("novita_sandbox.core")
    novita_mod.core = novita_core_mod
    novita_core_mod.Sandbox = MagicMock
    novita_core_mod.SandboxQuery = MagicMock

    sandbox_mod = _types.ModuleType("novita_sandbox.core.sandbox")
    sandbox_fs_mod = _types.ModuleType("novita_sandbox.core.sandbox.filesystem")
    sandbox_fs_fs_mod = _types.ModuleType("novita_sandbox.core.sandbox.filesystem.filesystem")
    sandbox_fs_fs_mod.WriteEntry = dict

    novita_core_mod.CommandExitException = _CommandExitException

    monkeypatch.setitem(__import__("sys").modules, "novita_sandbox", novita_mod)
    monkeypatch.setitem(__import__("sys").modules, "novita_sandbox.core", novita_core_mod)
    monkeypatch.setitem(__import__("sys").modules, "novita_sandbox.core.sandbox", sandbox_mod)
    monkeypatch.setitem(__import__("sys").modules, "novita_sandbox.core.sandbox.filesystem", sandbox_fs_mod)
    monkeypatch.setitem(__import__("sys").modules, "novita_sandbox.core.sandbox.filesystem.filesystem", sandbox_fs_fs_mod)

    return novita_core_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def novita_sdk(monkeypatch):
    """Provide a mock novita_sandbox.core module and return it for assertions."""
    return _patch_novita_imports(monkeypatch)


@pytest.fixture()
def make_env(novita_sdk, monkeypatch):
    """Factory that creates a NovitaEnvironment with a mocked SDK."""
    lazy_ensure = MagicMock()
    monkeypatch.setattr("tools.lazy_deps.ensure", lazy_ensure)
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])

    def _factory(
        sandbox=None,
        paginator_items=None,
        home_dir="/home/user",
        persistent=True,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        sandbox.commands.run.return_value = _make_run_response(stdout=home_dir)

        novita_sdk.Sandbox.create = MagicMock(return_value=sandbox)
        novita_sdk.Sandbox.connect = MagicMock(return_value=sandbox)
        novita_sdk.Sandbox.list = MagicMock(
            return_value=_make_paginator(paginator_items)
        )

        from tools.environments.novita import NovitaEnvironment

        env = NovitaEnvironment(persistent_filesystem=persistent, **kwargs)
        env._lazy_ensure = lazy_ensure
        return env

    return _factory


class TestLazyDependencies:
    def test_constructor_ensures_novita_terminal_dependency(self, make_env):
        env = make_env(persistent=False)

        env._lazy_ensure.assert_called_once_with("terminal.novita", prompt=False)


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------


class TestCwdResolution:
    def test_default_cwd_resolves_home(self, make_env):
        env = make_env(home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_tilde_cwd_resolves_home(self, make_env):
        env = make_env(cwd="~", home_dir="/home/testuser")
        assert env.cwd == "/home/testuser"

    def test_explicit_cwd_not_overridden(self, make_env):
        env = make_env(cwd="/workspace", home_dir="/root")
        assert env.cwd == "/workspace"

    def test_home_detection_failure_keeps_default_cwd(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = RuntimeError("exec failed")
        env = make_env(sandbox=sb)
        assert env.cwd == "/home/user"  # keeps constructor default

    def test_empty_home_keeps_default_cwd(self, make_env):
        env = make_env(home_dir="")
        assert env.cwd == "/home/user"  # keeps constructor default


# ---------------------------------------------------------------------------
# Sandbox persistence / resume
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persistent_resumes_via_list(self, make_env, novita_sdk):
        existing_info = _make_sandbox_info("sb-existing")
        existing_sb = _make_sandbox("sb-existing")
        existing_sb.commands.run.return_value = _make_run_response(stdout="/root")

        novita_sdk.Sandbox.connect = MagicMock(return_value=existing_sb)

        env = make_env(paginator_items=[existing_info], persistent=True, task_id="mytask")

        novita_sdk.Sandbox.list.assert_called_once()
        call_kwargs = novita_sdk.Sandbox.list.call_args
        assert call_kwargs.kwargs.get("limit") == 1 or call_kwargs[1].get("limit") == 1
        novita_sdk.Sandbox.connect.assert_called_once_with("sb-existing")
        novita_sdk.Sandbox.create.assert_not_called()

    def test_persistent_creates_new_when_none_found(self, make_env, novita_sdk):
        env = make_env(paginator_items=None, persistent=True, task_id="mytask")

        novita_sdk.Sandbox.list.assert_called_once()
        novita_sdk.Sandbox.connect.assert_not_called()
        novita_sdk.Sandbox.create.assert_called_once()

    def test_persistent_lookup_failure_falls_back_to_create(self, make_env, novita_sdk):
        novita_sdk.Sandbox.list = MagicMock(side_effect=RuntimeError("api error"))

        env = make_env(persistent=True, task_id="mytask")

        novita_sdk.Sandbox.create.assert_called_once()

    def test_non_persistent_skips_lookup(self, make_env, novita_sdk):
        env = make_env(persistent=False)

        novita_sdk.Sandbox.list.assert_not_called()
        novita_sdk.Sandbox.connect.assert_not_called()
        novita_sdk.Sandbox.create.assert_called_once()

    def test_create_passes_metadata_with_task_id(self, make_env, novita_sdk):
        make_env(persistent=False, task_id="my-task")

        call_kwargs = novita_sdk.Sandbox.create.call_args
        metadata = call_kwargs.kwargs.get("metadata") or call_kwargs[1].get("metadata")
        assert metadata == {"hermes_task_id": "my-task"}

    def test_create_passes_template_when_provided(self, make_env, novita_sdk):
        make_env(persistent=False, template="my-template-id")

        call_kwargs = novita_sdk.Sandbox.create.call_args
        template = call_kwargs.kwargs.get("template") or call_kwargs[1].get("template")
        assert template == "my-template-id"

    def test_create_uses_secure_sandbox_for_task_isolation(self, make_env, novita_sdk):
        make_env(persistent=False)

        call_kwargs = novita_sdk.Sandbox.create.call_args
        secure = call_kwargs.kwargs.get("secure")
        assert secure is True

    def test_create_sets_30_minute_sandbox_timeout(self, make_env, novita_sdk):
        make_env(persistent=False)

        call_kwargs = novita_sdk.Sandbox.create.call_args
        timeout = call_kwargs.kwargs.get("timeout")
        assert timeout == 30 * 60

    def test_create_passes_none_template_when_empty(self, make_env, novita_sdk):
        make_env(persistent=False, template="")

        call_kwargs = novita_sdk.Sandbox.create.call_args
        template = call_kwargs.kwargs.get("template") or call_kwargs[1].get("template")
        assert template is None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_persistent_cleanup_pauses_sandbox(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        env.cleanup()
        sb.beta_pause.assert_called_once()
        sb.kill.assert_not_called()

    def test_non_persistent_cleanup_kills_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        sb.kill.assert_called_once()
        sb.beta_pause.assert_not_called()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._sandbox.beta_pause.side_effect = RuntimeError("pause failed")
        env.cleanup()  # should not raise
        assert env._sandbox is None

    def test_cleanup_does_not_block_on_slow_sync_back(self, make_env):
        event = threading.Event()
        env = make_env(persistent=False)
        sb = env._sandbox
        env._cleanup_sync_timeout = 0.01

        def slow_sync_back():
            event.wait(timeout=5)

        env._sync_manager.sync_back = slow_sync_back

        try:
            env.cleanup()
            sb.kill.assert_called_once()
            assert env._sandbox is None
        finally:
            event.set()

    def test_cleanup_does_not_block_on_slow_kill(self, make_env):
        event = threading.Event()
        env = make_env(persistent=False)
        env._cleanup_lifecycle_timeout = 0.01

        def slow_kill():
            event.wait(timeout=5)

        env._sandbox.kill.side_effect = slow_kill

        try:
            env.cleanup()
            assert env._sandbox is None
        finally:
            event.set()


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


class TestExecute:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),       # $HOME detection
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_run_response(stdout="hello", exit_code=0),  # actual cmd
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_stderr_merged_into_output(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_run_response(stdout="out", stderr="err", exit_code=0),
        ]
        env = make_env(sandbox=sb)

        result = env.execute("cmd_with_stderr")
        assert "out" in result["output"]
        assert "err" in result["output"]

    def test_sdk_timeout_passed_to_run(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_run_response(stdout="ok", exit_code=0),
        ]
        env = make_env(sandbox=sb, timeout=42)

        env.execute("echo hello")
        call_args = sb.commands.run.call_args_list[-1]
        assert call_args[1].get("timeout") == 42.0

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_run_response(stdout="not found", exit_code=127),
        ]
        env = make_env(sandbox=sb)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127

    def test_stdin_data_wraps_heredoc(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_run_response(stdout="ok", exit_code=0),
        ]
        env = make_env(sandbox=sb)

        env.execute("python3", stdin_data="print('hi')")
        call_args = sb.commands.run.call_args_list[-1]
        cmd = call_args[0][0]
        assert "HERMES_STDIN_" in cmd
        assert "print" in cmd
        assert "hi" in cmd

    def test_custom_cwd_in_command_wrapper(self, make_env):
        """CWD is embedded in the command string via _wrap_command, not as a kwarg."""
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_run_response(stdout="/tmp", exit_code=0),
        ]
        env = make_env(sandbox=sb)

        env.execute("pwd", cwd="/tmp")
        call_args = sb.commands.run.call_args_list[-1]
        cmd = call_args[0][0]
        assert "cd" in cmd
        assert "/tmp" in cmd

    def test_nonzero_exit_via_command_exit_exception(self, make_env):
        """CommandExitException (non-zero exit) must return real exit code, not 1."""
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            _make_exit_exception(stdout="err output", exit_code=127),
        ]
        env = make_env(sandbox=sb)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127
        assert "err output" in result["output"]

    def test_sdk_error_surfaces_as_returncode_1(self, make_env):
        sb = _make_sandbox()
        sb.commands.run.side_effect = [
            _make_run_response(stdout="/root"),
            _make_run_response(stdout="", exit_code=0),  # init_session
            RuntimeError("sdk error"),
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo x")
        assert result["returncode"] == 1


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupt_kills_sandbox_and_returns_130(self, make_env, monkeypatch):
        sb = _make_sandbox()
        event = threading.Event()
        calls = {"n": 0}

        def run_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _make_run_response(stdout="/root")  # $HOME detection
            if calls["n"] == 2:
                return _make_run_response(stdout="", exit_code=0)  # init_session
            event.wait(timeout=5)  # simulate long-running command
            return _make_run_response(stdout="done", exit_code=0)

        sb.commands.run.side_effect = run_side_effect
        env = make_env(sandbox=sb)

        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: True)
        try:
            result = env.execute("sleep 10")
            assert result["returncode"] == 130
            sb.kill.assert_called()
        finally:
            event.set()

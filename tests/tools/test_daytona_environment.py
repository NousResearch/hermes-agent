"""Unit tests for the Daytona cloud sandbox environment backend."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Daytona SDK objects
# ---------------------------------------------------------------------------

def _make_exec_response(result="", exit_code=0):
    return SimpleNamespace(result=result, exit_code=exit_code)


def _make_sandbox(sandbox_id="sb-123", state="started"):
    sb = MagicMock()
    sb.id = sandbox_id
    sb.state = state
    sb.process.exec.return_value = _make_exec_response()
    return sb


def _patch_daytona_imports(monkeypatch):
    """Patch the daytona SDK so DaytonaEnvironment can be imported without it."""
    import types as _types

    import enum

    class _SandboxState(str, enum.Enum):
        STARTED = "started"
        STOPPED = "stopped"
        ARCHIVED = "archived"
        ERROR = "error"

    daytona_mod = _types.ModuleType("daytona")
    daytona_mod.Daytona = MagicMock
    daytona_mod.CreateSandboxFromImageParams = MagicMock
    daytona_mod.DaytonaError = type("DaytonaError", (Exception,), {})
    daytona_mod.Resources = MagicMock(name="Resources")
    daytona_mod.SandboxState = _SandboxState

    monkeypatch.setitem(__import__("sys").modules, "daytona", daytona_mod)
    return daytona_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def daytona_sdk(monkeypatch):
    """Provide a mock daytona SDK module and return it for assertions."""
    return _patch_daytona_imports(monkeypatch)


@pytest.fixture()
def make_env(daytona_sdk, monkeypatch):
    """Factory that creates a DaytonaEnvironment with a mocked SDK."""
    # Prevent is_interrupted from interfering
    monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False)

    def _factory(
        sandbox=None,
        find_one_side_effect=None,
        home_dir="/root",
        persistent=True,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        # Mock the $HOME detection
        sandbox.process.exec.return_value = _make_exec_response(result=home_dir)

        mock_client = MagicMock()
        mock_client.create.return_value = sandbox

        if find_one_side_effect is not None:
            mock_client.find_one.side_effect = find_one_side_effect
        else:
            # Default: no existing sandbox found
            mock_client.find_one.side_effect = daytona_sdk.DaytonaError("not found")

        daytona_sdk.Daytona = MagicMock(return_value=mock_client)

        from tools.environments.daytona import DaytonaEnvironment

        kwargs.setdefault("disk", 10240)
        env = DaytonaEnvironment(
            image="test-image:latest",
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_client = mock_client  # expose for assertions
        return env

    return _factory


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
        sb.process.exec.side_effect = RuntimeError("exec failed")
        env = make_env(sandbox=sb)
        assert env.cwd == "/home/daytona"  # keeps constructor default

    def test_empty_home_keeps_default_cwd(self, make_env):
        env = make_env(home_dir="")
        assert env.cwd == "/home/daytona"  # keeps constructor default


# ---------------------------------------------------------------------------
# Sandbox persistence / resume
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistent_resumes_existing_sandbox(self, make_env):
        existing = _make_sandbox(sandbox_id="sb-existing")
        existing.process.exec.return_value = _make_exec_response(result="/root")
        env = make_env(find_one_side_effect=lambda **kw: existing, persistent=True)
        existing.start.assert_called_once()
        # Should NOT have called create since find_one succeeded
        env._mock_client.create.assert_not_called()

    def test_persistent_creates_new_when_none_found(self, make_env, daytona_sdk):
        env = make_env(
            find_one_side_effect=daytona_sdk.DaytonaError("not found"),
            persistent=True,
        )
        env._mock_client.create.assert_called_once()

    def test_non_persistent_skips_find_one(self, make_env):
        env = make_env(persistent=False)
        env._mock_client.find_one.assert_not_called()
        env._mock_client.create.assert_called_once()


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_persistent_cleanup_stops_sandbox(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        env.cleanup()
        sb.stop.assert_called_once()

    def test_non_persistent_cleanup_deletes_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        env._mock_client.delete.assert_called_once_with(sb)

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=True)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_errors(self, make_env):
        env = make_env(persistent=True)
        env._sandbox.stop.side_effect = RuntimeError("stop failed")
        env.cleanup()  # should not raise
        assert env._sandbox is None


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

class TestExecute:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        # First call: $HOME detection; subsequent calls: actual commands
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),       # $HOME
            _make_exec_response(result="hello", exit_code=0),  # actual cmd
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        result = env.execute("echo hello")
        assert result["output"] == "hello"
        assert result["returncode"] == 0

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="not found", exit_code=127),
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127

    def test_stdin_data_wraps_heredoc(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="ok", exit_code=0),
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        env.execute("python3", stdin_data="print('hi')")
        # Check that the command passed to exec contains heredoc markers
        call_args = sb.process.exec.call_args_list[-1]
        cmd = call_args[0][0]
        assert "HERMES_EOF_" in cmd
        assert "print('hi')" in cmd

    def test_custom_cwd_passed_through(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),
            _make_exec_response(result="/tmp", exit_code=0),
        ]
        sb.state = "started"
        env = make_env(sandbox=sb)

        env.execute("pwd", cwd="/tmp")
        call_kwargs = sb.process.exec.call_args_list[-1][1]
        assert call_kwargs["cwd"] == "/tmp"

    def test_daytona_error_triggers_retry(self, make_env, daytona_sdk):
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),  # $HOME
            daytona_sdk.DaytonaError("transient"),  # first attempt fails
            _make_exec_response(result="ok", exit_code=0),  # retry succeeds
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo retry")
        assert result["output"] == "ok"
        assert result["returncode"] == 0


# ---------------------------------------------------------------------------
# Resource conversion
# ---------------------------------------------------------------------------

class TestResourceConversion:
    def _get_resources_kwargs(self, daytona_sdk):
        return daytona_sdk.Resources.call_args.kwargs

    def test_memory_converted_to_gib(self, make_env, daytona_sdk):
        env = make_env(memory=5120)
        assert self._get_resources_kwargs(daytona_sdk)["memory"] == 5

    def test_disk_converted_to_gib(self, make_env, daytona_sdk):
        env = make_env(disk=10240)
        assert self._get_resources_kwargs(daytona_sdk)["disk"] == 10

    def test_small_values_clamped_to_1(self, make_env, daytona_sdk):
        env = make_env(memory=100, disk=100)
        kw = self._get_resources_kwargs(daytona_sdk)
        assert kw["memory"] == 1
        assert kw["disk"] == 1


# ---------------------------------------------------------------------------
# Ensure sandbox ready
# ---------------------------------------------------------------------------

class TestInterrupt:
    def test_interrupt_stops_sandbox_and_returns_130(self, make_env, monkeypatch):
        sb = _make_sandbox()
        sb.state = "started"
        event = threading.Event()
        calls = {"n": 0}

        def exec_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _make_exec_response(result="/root")  # $HOME detection
            event.wait(timeout=5)  # simulate long-running command
            return _make_exec_response(result="done", exit_code=0)

        sb.process.exec.side_effect = exec_side_effect
        env = make_env(sandbox=sb)

        monkeypatch.setattr(
            "tools.environments.daytona.is_interrupted", lambda: True
        )
        try:
            result = env.execute("sleep 10")
            assert result["returncode"] == 130
            sb.stop.assert_called()
        finally:
            event.set()


# ---------------------------------------------------------------------------
# Retry exhaustion
# ---------------------------------------------------------------------------

class TestRetryExhausted:
    def test_both_attempts_fail(self, make_env, daytona_sdk):
        sb = _make_sandbox()
        sb.state = "started"
        sb.process.exec.side_effect = [
            _make_exec_response(result="/root"),       # $HOME
            daytona_sdk.DaytonaError("fail1"),         # first attempt
            daytona_sdk.DaytonaError("fail2"),         # retry
        ]
        env = make_env(sandbox=sb)

        result = env.execute("echo x")
        assert result["returncode"] == 1
        assert "Daytona execution error" in result["output"]


# ---------------------------------------------------------------------------
# Ensure sandbox ready
# ---------------------------------------------------------------------------

class TestEnsureSandboxReady:
    def test_restarts_stopped_sandbox(self, make_env):
        env = make_env()
        env._sandbox.state = "stopped"
        env._ensure_sandbox_ready()
        env._sandbox.start.assert_called()

    def test_no_restart_when_running(self, make_env):
        env = make_env()
        env._sandbox.state = "started"
        env._ensure_sandbox_ready()
        env._sandbox.start.assert_not_called()

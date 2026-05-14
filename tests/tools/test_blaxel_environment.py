"""Unit tests for the Blaxel cloud sandbox environment backend."""

import threading
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock Blaxel SDK objects
# ---------------------------------------------------------------------------


def _make_exec_response(stdout="", stderr="", exit_code=0, logs=None, status="completed"):
    return SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        logs=logs if logs is not None else stdout,
        exit_code=exit_code,
        status=status,
    )


def _make_sandbox():
    sb = MagicMock()
    sb.process.exec.return_value = _make_exec_response()
    return sb


def _patch_blaxel_imports(monkeypatch):
    """Inject mock blaxel.core / blaxel.core.sandbox / blaxel.core.volume
    modules so the BlaxelEnvironment import path works without the real
    SDK installed."""
    import sys
    import types as _types

    core_mod = _types.ModuleType("blaxel.core")
    sandbox_mod = _types.ModuleType("blaxel.core.sandbox")
    volume_mod = _types.ModuleType("blaxel.core.volume")

    class _SandboxAPIError(Exception):
        def __init__(self, msg="", status_code=None):
            super().__init__(msg)
            self.status_code = status_code

    class _VolumeAPIError(Exception):
        def __init__(self, msg="", status_code=None):
            super().__init__(msg)
            self.status_code = status_code

    core_mod.SyncSandboxInstance = MagicMock()
    core_mod.SyncVolumeInstance = MagicMock()
    sandbox_mod.SandboxAPIError = _SandboxAPIError
    volume_mod.VolumeAPIError = _VolumeAPIError

    blaxel_mod = _types.ModuleType("blaxel")
    blaxel_mod.core = core_mod

    monkeypatch.setitem(sys.modules, "blaxel", blaxel_mod)
    monkeypatch.setitem(sys.modules, "blaxel.core", core_mod)
    monkeypatch.setitem(sys.modules, "blaxel.core.sandbox", sandbox_mod)
    monkeypatch.setitem(sys.modules, "blaxel.core.volume", volume_mod)
    return SimpleNamespace(
        SyncSandboxInstance=core_mod.SyncSandboxInstance,
        SyncVolumeInstance=core_mod.SyncVolumeInstance,
        SandboxAPIError=_SandboxAPIError,
        VolumeAPIError=_VolumeAPIError,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def blaxel_sdk(monkeypatch):
    return _patch_blaxel_imports(monkeypatch)


@pytest.fixture()
def make_env(blaxel_sdk, monkeypatch):
    """Factory that creates a BlaxelEnvironment with a mocked SDK."""
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])
    # Default region — keep tests deterministic regardless of host env.
    monkeypatch.setenv("BL_REGION", "us-pdx-1")
    # Skip the readiness probe during unit tests; it consumes mock exec
    # calls and is exercised by integration tests instead.
    monkeypatch.setattr(
        "tools.environments.blaxel.BlaxelEnvironment._wait_for_sandbox_ready",
        lambda self, max_wait_seconds=60.0: None,
    )

    def _factory(
        sandbox=None,
        home_dir="/root",
        persistent=True,
        **kwargs,
    ):
        sandbox = sandbox or _make_sandbox()
        sandbox.process.exec.return_value = _make_exec_response(stdout=home_dir)

        blaxel_sdk.SyncSandboxInstance.create.return_value = sandbox

        from tools.environments.blaxel import BlaxelEnvironment

        env = BlaxelEnvironment(
            image="blaxel/base-image:latest",
            persistent_filesystem=persistent,
            **kwargs,
        )
        env._mock_sdk = blaxel_sdk
        return env

    return _factory


# ---------------------------------------------------------------------------
# Constructor / cwd resolution
# ---------------------------------------------------------------------------


class TestCwdResolution:
    """When no volume is mounted (non-persistent), cwd falls back to $HOME."""

    def test_default_cwd_resolves_home(self, make_env):
        env = make_env(persistent=False, home_dir="/home/blaxel")
        assert env.cwd == "/home/blaxel"

    def test_tilde_cwd_resolves_home(self, make_env):
        env = make_env(persistent=False, cwd="~", home_dir="/home/blaxel")
        assert env.cwd == "/home/blaxel"

    def test_explicit_cwd_not_overridden(self, make_env):
        env = make_env(persistent=False, cwd="/workspace", home_dir="/root")
        assert env.cwd == "/workspace"

    def test_home_detection_failure_keeps_default_cwd(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = RuntimeError("exec failed")
        env = make_env(persistent=False, sandbox=sb)
        # /root is the default home (set by self._remote_home before detection).
        assert env.cwd == "/root"


# ---------------------------------------------------------------------------
# Sandbox persistence / reattach
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persistent_creates_sandbox_with_volume(self, make_env, blaxel_sdk):
        env = make_env(persistent=True, task_id="mytask")
        blaxel_sdk.SyncSandboxInstance.create.assert_called_once()
        sandbox_config = blaxel_sdk.SyncSandboxInstance.create.call_args[0][0]
        assert sandbox_config["name"] == "hermes-mytask"
        assert sandbox_config["labels"] == {"hermes_task_id": "mytask"}
        assert "volumes" in sandbox_config

    def test_resource_names_are_sanitized_and_capped(self, make_env, blaxel_sdk):
        task_id = "ABC/Some Very Long Task ID With Spaces and Symbols!" + ("x" * 80)
        env = make_env(persistent=True, task_id=task_id)

        sandbox_config = blaxel_sdk.SyncSandboxInstance.create.call_args[0][0]
        volume_config = (
            blaxel_sdk.SyncVolumeInstance.create_if_not_exists.call_args[0][0]
        )
        assert sandbox_config["name"].startswith("hermes-")
        assert len(sandbox_config["name"]) <= 40
        assert re.fullmatch(r"[a-z0-9-]+", sandbox_config["name"])
        assert sandbox_config["name"] != f"hermes-{task_id}"
        assert re.search(r"-[0-9a-f]{8}$", sandbox_config["name"])

        assert volume_config["name"].startswith("hermes-")
        assert volume_config["name"].endswith("-data")
        assert len(volume_config["name"]) <= 40
        assert re.fullmatch(r"[a-z0-9-]+", volume_config["name"])
        assert re.search(r"-[0-9a-f]{8}-data$", volume_config["name"])

        label_value = sandbox_config["labels"]["hermes_task_id"]
        assert len(label_value) <= 63
        assert re.fullmatch(r"[a-z0-9-]+", label_value)
        assert label_value != task_id

    def test_non_persistent_creates_sandbox_without_volume(
        self, make_env, blaxel_sdk,
    ):
        env = make_env(persistent=False)
        blaxel_sdk.SyncSandboxInstance.create.assert_called_once()
        sandbox_config = blaxel_sdk.SyncSandboxInstance.create.call_args[0][0]
        assert "volumes" not in sandbox_config

    def test_name_conflict_reuses_healthy_existing(self, make_env, blaxel_sdk):
        existing = _make_sandbox()
        existing.process.exec.return_value = _make_exec_response(stdout="/root")
        blaxel_sdk.SyncSandboxInstance.create.side_effect = (
            blaxel_sdk.SandboxAPIError("ALREADY_EXISTS", status_code=409)
        )
        blaxel_sdk.SyncSandboxInstance.get.return_value = existing
        from tools.environments.blaxel import BlaxelEnvironment
        env = BlaxelEnvironment(
            image="blaxel/base-image:latest",
            persistent_filesystem=True,
            task_id="conflict",
        )
        # The existing sandbox is responsive, so create() is NOT retried.
        assert blaxel_sdk.SyncSandboxInstance.create.call_count == 1
        blaxel_sdk.SyncSandboxInstance.delete.assert_not_called()
        assert env._sandbox is existing

    def test_name_conflict_recreates_when_existing_unresponsive(
        self, make_env, blaxel_sdk, monkeypatch,
    ):
        # First create() raises 409, then the second create() succeeds.
        fresh = _make_sandbox()
        fresh.process.exec.return_value = _make_exec_response(stdout="/root")
        blaxel_sdk.SyncSandboxInstance.create.side_effect = [
            blaxel_sdk.SandboxAPIError("ALREADY_EXISTS", status_code=409),
            fresh,
        ]
        # The "existing" one is unresponsive (its exec raises forever).
        zombie = _make_sandbox()
        zombie.process.exec.side_effect = RuntimeError("WORKLOAD_UNAVAILABLE")
        blaxel_sdk.SyncSandboxInstance.get.return_value = zombie
        # Skip the real time.sleep so the test stays fast.
        monkeypatch.setattr("tools.environments.blaxel.time.sleep", lambda _s: None)
        # Fast-forward _sandbox_is_responsive's deadline.
        clock = {"t": 0.0}
        def fake_monotonic():
            clock["t"] += 100
            return clock["t"]
        monkeypatch.setattr(
            "tools.environments.blaxel.time.monotonic", fake_monotonic,
        )

        from tools.environments.blaxel import BlaxelEnvironment
        env = BlaxelEnvironment(
            image="blaxel/base-image:latest",
            persistent_filesystem=True,
            task_id="zombie",
        )
        assert blaxel_sdk.SyncSandboxInstance.create.call_count == 2
        blaxel_sdk.SyncSandboxInstance.delete.assert_called_with("hermes-zombie")
        assert env._sandbox is fresh


# ---------------------------------------------------------------------------
# Volume-backed persistence
# ---------------------------------------------------------------------------


class TestVolume:
    def test_persistent_creates_volume(self, make_env, blaxel_sdk):
        env = make_env(persistent=True, task_id="mytask")
        blaxel_sdk.SyncVolumeInstance.create_if_not_exists.assert_called_once()
        vol_config = (
            blaxel_sdk.SyncVolumeInstance.create_if_not_exists.call_args[0][0]
        )
        assert vol_config["name"] == "hermes-mytask-data"
        assert vol_config["region"] == "us-pdx-1"
        assert vol_config["labels"] == {"hermes_task_id": "mytask"}
        assert env._volume_name == "hermes-mytask-data"

    def test_persistent_mounts_volume_on_sandbox(self, make_env, blaxel_sdk):
        env = make_env(persistent=True, task_id="mytask")
        sandbox_config = blaxel_sdk.SyncSandboxInstance.create.call_args[0][0]
        assert "volumes" in sandbox_config
        assert sandbox_config["volumes"] == [{
            "name": "hermes-mytask-data",
            "mount_path": "/blaxel/persistent",
            "read_only": False,
        }]

    def test_persistent_cwd_points_at_volume(self, make_env):
        env = make_env(persistent=True, task_id="mytask")
        assert env.cwd == "/blaxel/persistent"

    def test_non_persistent_skips_volume(self, make_env, blaxel_sdk):
        env = make_env(persistent=False)
        blaxel_sdk.SyncVolumeInstance.create_if_not_exists.assert_not_called()
        sandbox_config = blaxel_sdk.SyncSandboxInstance.create.call_args[0][0]
        assert "volumes" not in sandbox_config
        assert env._volume_name is None

    def test_volume_creation_failure_raises_before_sandbox_create(
        self, make_env, blaxel_sdk,
    ):
        blaxel_sdk.SyncVolumeInstance.create_if_not_exists.side_effect = (
            blaxel_sdk.VolumeAPIError("quota exceeded", status_code=403)
        )
        with pytest.raises(RuntimeError, match="persistent filesystem requested"):
            make_env(persistent=True, task_id="mytask")
        blaxel_sdk.SyncSandboxInstance.create.assert_not_called()

    def test_persistent_without_volume_sdk_raises(self, blaxel_sdk, monkeypatch):
        import sys

        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
        monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])
        monkeypatch.setenv("BL_REGION", "us-pdx-1")
        monkeypatch.delattr(sys.modules["blaxel.core"], "SyncVolumeInstance")

        from tools.environments.blaxel import BlaxelEnvironment
        with pytest.raises(RuntimeError, match="does not expose volume support"):
            BlaxelEnvironment(
                image="blaxel/base-image:latest",
                persistent_filesystem=True,
                task_id="mytask",
            )


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_persistent_cleanup_deletes_sandbox_and_preserves_volume(self, make_env):
        env = make_env(persistent=True)
        sb = env._sandbox
        volume_name = env._volume_name
        env.cleanup()
        sb.delete.assert_called_once()
        assert volume_name is not None

    def test_non_persistent_cleanup_deletes_sandbox(self, make_env):
        env = make_env(persistent=False)
        sb = env._sandbox
        env.cleanup()
        sb.delete.assert_called_once()

    def test_cleanup_idempotent(self, make_env):
        env = make_env(persistent=False)
        env.cleanup()
        env.cleanup()  # should not raise

    def test_cleanup_swallows_404(self, make_env, blaxel_sdk):
        env = make_env(persistent=False)
        env._sandbox.delete.side_effect = blaxel_sdk.SandboxAPIError(
            "gone", status_code=404,
        )
        env.cleanup()  # should not raise

    def test_cleanup_swallows_generic_errors(self, make_env):
        env = make_env(persistent=False)
        env._sandbox.delete.side_effect = RuntimeError("boom")
        env.cleanup()  # should not raise
        assert env._sandbox is None


# ---------------------------------------------------------------------------
# Execute (short timeout — blocking path)
# ---------------------------------------------------------------------------


class TestExecuteShort:
    def test_basic_command(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(stdout="/root"),       # $HOME
            _make_exec_response(exit_code=0),          # init_session
            _make_exec_response(stdout="hello", exit_code=0),  # actual cmd
        ]
        env = make_env(persistent=False, sandbox=sb, timeout=30)

        result = env.execute("echo hello")
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_blocking_path_passes_timeout_in_ms(self, make_env):
        """When timeout <= 50s, the SDK's blocking path is used with timeout in ms."""
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(stdout="/root"),
            _make_exec_response(exit_code=0),
            _make_exec_response(stdout="ok", exit_code=0),
        ]
        env = make_env(persistent=False, sandbox=sb, timeout=42)

        env.execute("echo hello")
        last_call = sb.process.exec.call_args_list[-1]
        config = last_call[0][0]
        assert config["wait_for_completion"] is True
        assert config["timeout"] == 42_000

    def test_nonzero_exit_code(self, make_env):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(stdout="/root"),
            _make_exec_response(exit_code=0),
            _make_exec_response(stdout="not found", exit_code=127),
        ]
        env = make_env(persistent=False, sandbox=sb, timeout=30)

        result = env.execute("bad_cmd")
        assert result["returncode"] == 127


# ---------------------------------------------------------------------------
# Execute (long timeout — async polling path)
# ---------------------------------------------------------------------------


class TestExecuteLong:
    def test_long_timeout_uses_async_polling(self, make_env, monkeypatch):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(stdout="/root"),  # $HOME
            _make_exec_response(exit_code=0),     # init_session
            _make_exec_response(),                # async start (return ignored)
        ]

        # process.get returns "completed" on the first poll
        sb.process.get.return_value = SimpleNamespace(
            status="completed", exit_code=0, stdout="hello", stderr="", logs="hello",
        )

        # No real sleeps in tests
        monkeypatch.setattr("tools.environments.blaxel.time.sleep", lambda _s: None)

        env = make_env(persistent=False, sandbox=sb, timeout=120)
        result = env.execute("sleep 1 && echo hello")

        # The third exec call should have used wait_for_completion=False
        async_call = sb.process.exec.call_args_list[-1]
        assert async_call[0][0]["wait_for_completion"] is False
        assert "name" in async_call[0][0]

        assert result["returncode"] == 0
        assert "hello" in result["output"]

    def test_long_timeout_preserves_nonzero_exit_code(self, make_env, monkeypatch):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(stdout="/root"),
            _make_exec_response(exit_code=0),
            _make_exec_response(),
        ]
        sb.process.get.return_value = SimpleNamespace(
            status="completed", exit_code=7, stdout="bad", stderr="", logs="bad",
        )
        monkeypatch.setattr("tools.environments.blaxel.time.sleep", lambda _s: None)

        env = make_env(persistent=False, sandbox=sb, timeout=120)
        result = env.execute("exit 7")

        assert result["returncode"] == 7
        assert "bad" in result["output"]

    def test_long_timeout_kills_on_timeout(self, make_env, monkeypatch):
        sb = _make_sandbox()
        sb.process.exec.side_effect = [
            _make_exec_response(stdout="/root"),
            _make_exec_response(exit_code=0),
            _make_exec_response(),  # async start
        ]
        # Always return "running" so the polling loop hits its deadline
        sb.process.get.return_value = SimpleNamespace(
            status="running", exit_code=None, stdout="", stderr="", logs="",
        )

        # Fake monotonic clock that jumps forward past the deadline
        clock = {"t": 0.0}

        def fake_monotonic():
            clock["t"] += 100
            return clock["t"]

        monkeypatch.setattr("tools.environments.blaxel.time.monotonic", fake_monotonic)
        monkeypatch.setattr("tools.environments.blaxel.time.sleep", lambda _s: None)

        env = make_env(persistent=False, sandbox=sb, timeout=120)
        result = env.execute("sleep 9999")

        sb.process.kill.assert_called_once()
        assert result["returncode"] == 124


# ---------------------------------------------------------------------------
# Resource handling
# ---------------------------------------------------------------------------


class TestResourceHandling:
    @staticmethod
    def _last_sandbox_config(blaxel_sdk):
        for mock_method in (
            blaxel_sdk.SyncSandboxInstance.create_if_not_exists,
            blaxel_sdk.SyncSandboxInstance.create,
        ):
            if mock_method.call_args is not None:
                return mock_method.call_args[0][0]
        raise AssertionError("Sandbox creation was never called")

    def test_memory_passed_to_create(self, make_env, blaxel_sdk):
        env = make_env(memory=4096)
        config = self._last_sandbox_config(blaxel_sdk)
        assert config["memory"] == 4096

    def test_region_uses_env_var(self, make_env, blaxel_sdk, monkeypatch):
        monkeypatch.setenv("BL_REGION", "eu-lon-1")
        env = make_env()
        config = self._last_sandbox_config(blaxel_sdk)
        assert config["region"] == "eu-lon-1"

    def test_region_defaults_when_env_unset(self, make_env, blaxel_sdk, monkeypatch):
        monkeypatch.delenv("BL_REGION", raising=False)
        env = make_env()
        config = self._last_sandbox_config(blaxel_sdk)
        assert config["region"] == "us-pdx-1"

    def test_ttl_passed_to_create(self, make_env, blaxel_sdk):
        env = make_env(ttl="1h")
        config = self._last_sandbox_config(blaxel_sdk)
        assert config["ttl"] == "1h"

    def test_lazy_dep_registration(self):
        from tools.lazy_deps import LAZY_DEPS
        assert LAZY_DEPS["terminal.blaxel"] == ("blaxel==0.2.52",)

    def test_wait_for_sandbox_ready_raises_on_timeout(
        self, blaxel_sdk, monkeypatch,
    ):
        monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
        monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
        monkeypatch.setattr("tools.credential_files.get_skills_directory_mount", lambda **kw: None)
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kw: [])
        monkeypatch.setenv("BL_REGION", "us-pdx-1")

        sb = _make_sandbox()
        sb.process.exec.side_effect = RuntimeError("WORKLOAD_UNAVAILABLE")
        blaxel_sdk.SyncSandboxInstance.create.return_value = sb

        clock = {"t": 0.0}

        def fake_monotonic():
            clock["t"] += 100
            return clock["t"]

        monkeypatch.setattr("tools.environments.blaxel.time.monotonic", fake_monotonic)
        monkeypatch.setattr("tools.environments.blaxel.time.sleep", lambda _s: None)

        from tools.environments.blaxel import BlaxelEnvironment
        with pytest.raises(RuntimeError, match="did not become ready"):
            BlaxelEnvironment(
                image="blaxel/base-image:latest",
                persistent_filesystem=False,
                task_id="timeout",
            )


# ---------------------------------------------------------------------------
# Interrupt
# ---------------------------------------------------------------------------


class TestInterrupt:
    def test_interrupt_kills_process_and_returns_130(self, make_env, monkeypatch):
        sb = _make_sandbox()
        event = threading.Event()
        calls = {"n": 0}

        def exec_side_effect(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _make_exec_response(stdout="/root")
            if calls["n"] == 2:
                return _make_exec_response(exit_code=0)
            event.wait(timeout=5)
            return _make_exec_response(stdout="done", exit_code=0)

        sb.process.exec.side_effect = exec_side_effect
        env = make_env(persistent=False, sandbox=sb, timeout=30)

        monkeypatch.setattr(
            "tools.environments.base.is_interrupted", lambda: True
        )
        try:
            result = env.execute("sleep 10")
            assert result["returncode"] == 130
            sb.process.kill.assert_called()
        finally:
            event.set()

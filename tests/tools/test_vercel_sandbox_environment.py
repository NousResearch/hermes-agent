"""Unit tests for the Vercel Sandbox terminal backend."""

from __future__ import annotations

import importlib
import io
import re
import sys
import tarfile
import threading
import types
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeRunResult:
    def __init__(self, output: str | bytes = "", exit_code: int = 0):
        self._output = output
        self.exit_code = exit_code

    def output(self) -> str | bytes:
        return self._output


class _FakeSandboxStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    ABORTED = "aborted"
    SNAPSHOTTING = "snapshotting"


@dataclass(frozen=True)
class _FakeSnapshot:
    snapshot_id: str


class _FakeSandbox:
    def __init__(
        self,
        *,
        cwd: str = "/vercel/sandbox",
        home: str = "/home/vercel",
        status: _FakeSandboxStatus = _FakeSandboxStatus.RUNNING,
    ):
        self.sandbox = SimpleNamespace(cwd=cwd, id="sb-123")
        self.status = status
        self.home = home
        self.closed = 0
        self.client = SimpleNamespace(close=self._close)
        self.run_command_calls: list[tuple[str, list[str], dict]] = []
        self.run_command_side_effects: list[object] = []
        self.write_files_calls: list[list[dict[str, object]]] = []
        self.write_files_side_effects: list[object] = []
        self.download_file_calls: list[tuple[str, Path]] = []
        self.download_file_side_effects: list[object] = []
        self.download_file_content = b""
        self.stop_calls: list[tuple[tuple, dict]] = []
        self.snapshot_calls: list[tuple[tuple, dict]] = []
        self.snapshot_side_effects: list[object] = []
        self.snapshot_id = "snap_default"
        self.refresh_calls = 0
        self.wait_for_status_calls: list[tuple[object, object, object]] = []
        self.wait_for_status_side_effects: list[object] = []

    def _close(self) -> None:
        self.closed += 1

    def refresh(self) -> None:
        self.refresh_calls += 1

    def wait_for_status(self, status: _FakeSandboxStatus | str, *, timeout, poll_interval) -> None:
        self.wait_for_status_calls.append((status, timeout, poll_interval))
        if self.wait_for_status_side_effects:
            effect = self.wait_for_status_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                effect(status, timeout, poll_interval)
                return
        self.status = _FakeSandboxStatus(status)

    def run_command(self, cmd: str, args: list[str] | None = None, **kwargs):
        args = list(args or [])
        self.run_command_calls.append((cmd, args, kwargs))
        if self.run_command_side_effects:
            effect = self.run_command_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(cmd, args, kwargs)
            return effect
        script = args[1] if len(args) > 1 else ""
        if 'printf %s "$HOME"' in script:
            return _FakeRunResult(self.home)
        return _FakeRunResult("")

    def write_files(self, files: list[dict[str, object]]) -> None:
        self.write_files_calls.append(files)
        if self.write_files_side_effects:
            effect = self.write_files_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                effect(files)

    def download_file(self, remote_path: str, local_path) -> str:
        destination = Path(local_path)
        self.download_file_calls.append((remote_path, destination))
        if self.download_file_side_effects:
            effect = self.download_file_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(remote_path, destination)
        destination.write_bytes(self.download_file_content)
        return str(destination.resolve())

    def stop(self, *args, **kwargs) -> None:
        self.stop_calls.append((args, kwargs))

    def snapshot(self, *args, **kwargs):
        self.snapshot_calls.append((args, kwargs))
        if self.snapshot_side_effects:
            effect = self.snapshot_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if callable(effect):
                return effect(*args, **kwargs)
            if isinstance(effect, str):
                return _FakeSnapshot(effect)
            return effect
        return _FakeSnapshot(self.snapshot_id)


@dataclass(frozen=True)
class _FakeResources:
    vcpus: float | None = None
    memory: int | None = None


@dataclass(frozen=True)
class _FakeWriteFile:
    path: str
    content: bytes


class _FakeSDK:
    def __init__(self):
        self.create_kwargs: list[dict[str, object]] = []
        self.create_side_effects: list[object] = []
        self.sandboxes: list[_FakeSandbox] = []

    @property
    def current(self) -> _FakeSandbox:
        return self.sandboxes[-1]

    def create(self, **kwargs):
        self.create_kwargs.append(kwargs)
        if self.create_side_effects:
            effect = self.create_side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            if isinstance(effect, _FakeSandbox):
                self.sandboxes.append(effect)
                return effect
        sandbox = _FakeSandbox()
        self.sandboxes.append(sandbox)
        return sandbox


def _cwd_result(body: str = "", *, cwd: str = "/vercel/sandbox", exit_code: int = 0):
    def _result(_cmd: str, args: list[str], _kwargs: dict):
        script = args[1] if len(args) > 1 else ""
        match = re.search(r"__HERMES_CWD_[A-Za-z0-9]+__", script)
        marker = match.group(0) if match else "__HERMES_CWD_MISSING__"
        prefix = f"{body}\n\n" if body else "\n"
        return _FakeRunResult(f"{prefix}{marker}{cwd}{marker}\n", exit_code)

    return _result


def _tar_bytes(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture()
def vercel_sdk(monkeypatch):
    fake_sdk = _FakeSDK()
    sandbox_mod = types.ModuleType("vercel.sandbox")
    sandbox_mod.Sandbox = types.SimpleNamespace(create=fake_sdk.create)
    sandbox_mod.Resources = _FakeResources
    sandbox_mod.WriteFile = _FakeWriteFile
    sandbox_mod.SandboxStatus = _FakeSandboxStatus

    vercel_mod = types.ModuleType("vercel")
    vercel_mod.sandbox = sandbox_mod

    monkeypatch.setitem(sys.modules, "vercel", vercel_mod)
    monkeypatch.setitem(sys.modules, "vercel.sandbox", sandbox_mod)
    return fake_sdk


@pytest.fixture()
def vercel_module(vercel_sdk, monkeypatch):
    monkeypatch.setattr("tools.environments.base.is_interrupted", lambda: False)
    monkeypatch.setattr("tools.credential_files.get_credential_file_mounts", lambda: [])
    monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
    monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

    module = importlib.import_module("tools.environments.vercel_sandbox")
    return importlib.reload(module)


@pytest.fixture()
def make_env(vercel_module, request):
    envs = []

    def _cleanup_envs():
        for env in envs:
            env._sync_manager = None
            env.cleanup()

    request.addfinalizer(_cleanup_envs)

    def _factory(**kwargs):
        kwargs.setdefault("runtime", "node22")
        kwargs.setdefault("cwd", vercel_module.DEFAULT_VERCEL_CWD)
        kwargs.setdefault("timeout", 30)
        kwargs.setdefault("task_id", "task-123")
        env = vercel_module.VercelSandboxEnvironment(**kwargs)
        envs.append(env)
        return env

    return _factory


class TestStartup:
    def test_default_cwd_tracks_remote_workspace_root(self, make_env, vercel_sdk):
        sandbox = _FakeSandbox(cwd="/workspace")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env()

        assert env.cwd == "/workspace"

    def test_tilde_cwd_resolves_against_remote_home(self, make_env, vercel_sdk):
        sandbox = _FakeSandbox(home="/home/custom")
        vercel_sdk.create_side_effects.append(sandbox)

        env = make_env(cwd="~")

        assert env.cwd == "/home/custom"

    def test_pending_sandbox_timeout_raises_descriptive_error(
        self, make_env, vercel_sdk
    ):
        sandbox = _FakeSandbox(status=_FakeSandboxStatus.PENDING)
        sandbox.wait_for_status_side_effects.append(TimeoutError("still pending"))
        vercel_sdk.create_side_effects.append(sandbox)

        with pytest.raises(RuntimeError, match="Sandbox did not reach running state"):
            make_env()


class TestFileSync:
    def test_initial_sync_uploads_managed_files_under_remote_home(
        self, make_env, vercel_sdk, monkeypatch, tmp_path
    ):
        src = tmp_path / "token.txt"
        src.write_text("secret-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
        monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

        make_env()

        uploaded = vercel_sdk.current.write_files_calls[0]
        assert uploaded == [
            {
                "path": "/home/vercel/.hermes/credentials/token.txt",
                "content": b"secret-token",
            }
        ]

    def test_execute_resyncs_changed_managed_files(
        self, make_env, vercel_sdk, monkeypatch, tmp_path
    ):
        src = tmp_path / "token.txt"
        src.write_text("secret-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr("tools.credential_files.iter_skills_files", lambda **kwargs: [])
        monkeypatch.setattr("tools.credential_files.iter_cache_files", lambda **kwargs: [])

        env = make_env()
        src.write_text("updated-secret-token")
        monkeypatch.setenv("HERMES_FORCE_FILE_SYNC", "1")
        vercel_sdk.current.run_command_side_effects.append(_cwd_result("hello"))

        result = env.execute("echo hello")

        assert result == {"output": "hello\n", "returncode": 0}
        assert vercel_sdk.current.write_files_calls[-1] == [
            {
                "path": "/home/vercel/.hermes/credentials/token.txt",
                "content": b"updated-secret-token",
            }
        ]

    def test_cleanup_syncs_back_snapshots_closes_and_is_idempotent(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        src = tmp_path / "token.txt"
        src.write_text("host-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda **kwargs: [],
        )
        env = make_env()
        sandbox = vercel_sdk.current
        sandbox.snapshot_id = "snap_cleanup"
        vercel_sdk.current.download_file_content = _tar_bytes(
            {
                "home/vercel/.hermes/credentials/token.txt": b"remote-token",
                "home/vercel/.hermes/credentials/new.txt": b"new-remote",
                "home/vercel/.hermes/unmapped/skip.txt": b"skip",
            }
        )

        env.cleanup()
        env.cleanup()

        assert src.read_text() == "remote-token"
        assert (tmp_path / "new.txt").read_text() == "new-remote"
        assert not (tmp_path / "skip.txt").exists()
        assert len(sandbox.snapshot_calls) == 1
        assert len(sandbox.stop_calls) == 1  # always stop after snapshot to avoid resource leaks
        assert sandbox.closed == 1
        assert vercel_module._load_snapshots() == {"task-123": "snap_cleanup"}

    def test_cleanup_sync_back_failure_from_download_does_not_block_snapshot(
        self, make_env, vercel_sdk, monkeypatch, tmp_path
    ):
        src = tmp_path / "token.txt"
        src.write_text("host-token")
        monkeypatch.setattr(
            "tools.credential_files.get_credential_file_mounts",
            lambda: [
                {
                    "host_path": str(src),
                    "container_path": "/root/.hermes/credentials/token.txt",
                }
            ],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_skills_files",
            lambda **kwargs: [],
        )
        monkeypatch.setattr(
            "tools.credential_files.iter_cache_files",
            lambda **kwargs: [],
        )
        env = make_env()
        sandbox = vercel_sdk.current
        sandbox.run_command_side_effects.extend(
            [
                _FakeRunResult("tar failed", exit_code=2),
                _FakeRunResult(""),
                _FakeRunResult("tar failed", exit_code=2),
                _FakeRunResult(""),
                _FakeRunResult("tar failed", exit_code=2),
                _FakeRunResult(""),
            ]
        )
        monkeypatch.setattr("tools.environments.file_sync.time.sleep", lambda _delay: None)

        env.cleanup()

        assert src.read_text() == "host-token"
        assert len(sandbox.snapshot_calls) == 1
        assert sandbox.closed == 1
        assert len(sandbox.download_file_calls) == 0


class TestExecute:

    @pytest.mark.parametrize(
        ("make_unhealthy", "label"),
        [
            (
                lambda sandbox: setattr(
                    sandbox, "status", _FakeSandboxStatus.STOPPED
                ),
                "terminal state",
            ),
            (
                lambda sandbox: setattr(
                    sandbox,
                    "refresh",
                    lambda: (_ for _ in ()).throw(RuntimeError("refresh failed")),
                ),
                "refresh failure",
            ),
        ],
        ids=["terminal-state", "refresh-failure"],
    )
    def test_execute_recreates_unhealthy_sandbox_before_running_command(
        self, make_env, vercel_sdk, make_unhealthy, label
    ):
        env = make_env()
        original = vercel_sdk.current
        make_unhealthy(original)

        replacement = _FakeSandbox()
        replacement.run_command_side_effects.extend(
            [
                _FakeRunResult(replacement.home),
                _cwd_result("hello"),
            ]
        )
        vercel_sdk.create_side_effects.append(replacement)

        result = env.execute("echo hello")

        assert result == {"output": "hello\n", "returncode": 0}, label
        assert original.closed == 1
        assert vercel_sdk.current is replacement

    def test_run_bash_handle_uses_captured_sandbox_for_exec_and_cancel(
        self, make_env
    ):
        env = make_env()
        original = env._sandbox
        assert original is not None
        replacement = _FakeSandbox()
        started = threading.Event()
        release = threading.Event()

        def blocking_command(_cmd: str, _args: list[str], _kwargs: dict):
            started.set()
            release.wait(timeout=5)
            return _FakeRunResult("done")

        original.run_command_side_effects.append(blocking_command)

        handle = env._run_bash("echo done")
        assert started.wait(timeout=1)

        env._sandbox = replacement
        handle.kill()
        release.set()

        assert handle.wait(timeout=2) == 0
        assert len(original.stop_calls) == 1
        assert replacement.stop_calls == []
        cmd, args, kwargs = original.run_command_calls[-1]
        assert cmd == "bash"
        assert args == ["-c", "echo done"]
        assert kwargs["cwd"] == "/vercel/sandbox"


class TestSnapshotPersistence:
    def test_create_restores_from_saved_snapshot(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        vercel_module._store_snapshot("task-123", "snap_saved")
        restored = _FakeSandbox(cwd="/restored")
        vercel_sdk.create_side_effects.append(restored)

        env = make_env()

        assert env.cwd == "/restored"
        assert vercel_sdk.create_kwargs[0]["source"] == {
            "type": "snapshot",
            "snapshot_id": "snap_saved",
        }
        assert vercel_module._load_snapshots() == {"task-123": "snap_saved"}

    def test_restore_failure_prunes_snapshot_and_falls_back_to_fresh_sandbox(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        vercel_module._store_snapshot("task-123", "snap_stale")
        fresh = _FakeSandbox(cwd="/fresh")
        vercel_sdk.create_side_effects.extend(
            [RuntimeError("snapshot missing"), fresh]
        )

        env = make_env()

        assert env.cwd == "/fresh"
        assert vercel_sdk.create_kwargs[0]["source"] == {
            "type": "snapshot",
            "snapshot_id": "snap_stale",
        }
        assert "source" not in vercel_sdk.create_kwargs[1]
        assert vercel_module._load_snapshots() == {}

    def test_cleanup_stops_when_snapshot_fails_without_storing_metadata(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = make_env()
        sandbox = vercel_sdk.current
        sandbox.snapshot_side_effects.append(RuntimeError("snapshot failed"))

        env.cleanup()

        assert len(sandbox.snapshot_calls) == 1
        assert len(sandbox.stop_calls) == 1
        assert sandbox.closed == 1
        assert vercel_module._load_snapshots() == {}

    def test_non_persistent_cleanup_stops_without_snapshot(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = make_env(persistent_filesystem=False)
        sandbox = vercel_sdk.current

        env.cleanup()

        assert sandbox.snapshot_calls == []
        assert len(sandbox.stop_calls) == 1
        assert sandbox.closed == 1
        assert vercel_module._load_snapshots() == {}

    def test_persistent_cleanup_without_task_id_stops_without_snapshot(
        self, make_env, vercel_module, vercel_sdk, monkeypatch, tmp_path
    ):
        hermes_home = tmp_path / ".hermes"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        env = make_env(task_id="")
        sandbox = vercel_sdk.current

        env.cleanup()

        assert sandbox.snapshot_calls == []
        assert len(sandbox.stop_calls) == 1
        assert sandbox.closed == 1
        assert vercel_module._load_snapshots() == {}


class TestCleanup:
    def test_cleanup_continues_when_sync_back_raises(self, make_env, vercel_sdk):
        env = make_env()
        sandbox = vercel_sdk.current

        class FailingSyncManager:
            def sync_back(self):
                raise RuntimeError("download failed")

        env._sync_manager = FailingSyncManager()

        env.cleanup()

        assert len(sandbox.snapshot_calls) == 1
        assert sandbox.closed == 1

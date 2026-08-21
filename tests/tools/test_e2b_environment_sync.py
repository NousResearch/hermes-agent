"""Regression tests for the E2B backend's file-sync contract.

These tests deliberately use the real :class:`FileSyncManager`.  The remote
filesystem double rejects writes to missing parent directories even though the
current E2B service creates them, so Hermes owns that portability guarantee.
"""

from __future__ import annotations

import io
import posixpath
import shlex
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from tools.environments import e2b as backend
from tools.environments.file_sync import FileSyncManager, iter_sync_files


class SandboxMissing(Exception):
    pass


class RemoteFileMissing(Exception):
    pass


class StrictRemoteFiles:
    """Minimal E2B file API that requires parents to exist before writes."""

    def __init__(self):
        self.directories = {"/", "/home", "/home/user"}
        self.files: dict[str, bytes] = {}
        self.fail_writes = False
        self.download_payload = _tar_bytes({})

    def mkdir_p(self, path: str) -> None:
        current = ""
        for part in path.strip("/").split("/"):
            current += f"/{part}"
            self.directories.add(current)

    def write(self, path: str, data: bytes) -> None:
        self.write_files([{"path": path, "data": data}])

    def write_files(self, payload: list[dict]) -> None:
        if self.fail_writes:
            raise RuntimeError("E2B upload unavailable")
        for entry in payload:
            parent = posixpath.dirname(entry["path"])
            if parent not in self.directories:
                raise RuntimeError(f"missing remote directory: {parent}")
        for entry in payload:
            self.files[entry["path"]] = entry["data"]

    def read(self, _path: str, *, format: str):
        assert format == "bytes"
        return bytearray(self.download_payload)

    def remove(self, path: str) -> None:
        self.files.pop(path, None)


class RemoteCommands:
    def __init__(self, files: StrictRemoteFiles):
        self.files = files
        self.calls: list[tuple[str, dict]] = []

    def run(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        parts = shlex.split(command)
        if parts[:2] == ["mkdir", "-p"]:
            for path in parts[2:]:
                self.files.mkdir_p(path)
        return SimpleNamespace(stdout="", stderr="", exit_code=0)


class SandboxSession:
    def __init__(self, sandbox_id: str, *, fail_writes: bool = False):
        self.sandbox_id = sandbox_id
        self.files = StrictRemoteFiles()
        self.files.fail_writes = fail_writes
        self.commands = RemoteCommands(self.files)
        self.connect_calls: list[dict] = []
        self.pause_calls: list[dict] = []
        self.kill_calls: list[dict] = []
        self.reconnect_effect: Exception | None = None

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.reconnect_effect is not None:
            raise self.reconnect_effect
        return self

    def pause(self, **kwargs):
        self.pause_calls.append(kwargs)
        return True

    def kill(self, **kwargs):
        self.kill_calls.append(kwargs)
        return True


class SandboxService:
    def __init__(self):
        self.sessions: dict[str, SandboxSession] = {}
        self.fail_writes = False

    def create(self, **_kwargs):
        session = SandboxSession(
            f"sb-{len(self.sessions) + 1}",
            fail_writes=self.fail_writes,
        )
        self.sessions[session.sandbox_id] = session
        return session

    def connect(self, sandbox_id: str, **_kwargs):
        if sandbox_id not in self.sessions:
            raise SandboxMissing(sandbox_id)
        return self.sessions[sandbox_id]


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture()
def e2b_service(monkeypatch):
    service = SandboxService()

    e2b_module = types.ModuleType("e2b")
    e2b_module.Sandbox = SimpleNamespace(
        create=service.create,
        connect=service.connect,
    )
    exceptions = types.ModuleType("e2b.exceptions")
    exceptions.SandboxNotFoundException = SandboxMissing
    exceptions.FileNotFoundException = RemoteFileMissing

    monkeypatch.setitem(sys.modules, "e2b", e2b_module)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setattr(backend, "_ensure_e2b_sdk", lambda: None)
    monkeypatch.setattr(backend.E2BEnvironment, "init_session", lambda self: None)
    return service


def _environment(**kwargs) -> backend.E2BEnvironment:
    return backend.E2BEnvironment(
        api_key="profile-key",
        task_id="sync-contract",
        timeout=30,
        lifetime_seconds=90,
        **kwargs,
    )


def test_real_sync_inventory_excludes_profile_env_and_config():
    hermes_home = get_hermes_home()
    profile_env = hermes_home / ".env"
    profile_config = hermes_home / "config.yaml"
    skill = hermes_home / "skills" / "incident-triage" / "SKILL.md"
    profile_env.write_text("E2B_API_KEY=must-not-upload\n", encoding="utf-8")
    profile_config.write_text("terminal:\n  backend: e2b\n", encoding="utf-8")
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text("triage", encoding="utf-8")

    inventory = iter_sync_files("/home/user/.hermes")
    host_paths = {host for host, _remote in inventory}

    assert (str(skill), "/home/user/.hermes/skills/incident-triage/SKILL.md") in inventory
    assert str(profile_env) not in host_paths
    assert str(profile_config) not in host_paths


def test_real_sync_manager_creates_nested_parents_before_upload(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    mappings = []
    for relative in (
        "skills/research/arxiv/SKILL.md",
        "skills/github/SKILL.md",
        "cache/models/catalog.json",
    ):
        host = tmp_path / relative
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_text(relative, encoding="utf-8")
        mappings.append((str(host), f"/home/user/.hermes/{relative}"))
    monkeypatch.setattr(backend, "iter_sync_files", lambda _base: mappings)

    env = _environment(persistent_filesystem=False)
    session = env._sandbox

    assert session.files.files == {
        remote: Path(host).read_bytes() for host, remote in mappings
    }
    env.cleanup()


def test_failed_initial_sync_fails_startup_and_cleans_up_new_sandbox(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("triage", encoding="utf-8")
    monkeypatch.setattr(
        backend,
        "iter_sync_files",
        lambda _base: [(str(host), "/home/user/.hermes/skills/incident-triage/SKILL.md")],
    )
    e2b_service.fail_writes = True

    with pytest.raises(backend.EnvironmentConnectionError, match="initial state sync"):
        _environment(persistent_filesystem=True)

    session = e2b_service.sessions["sb-1"]
    assert session.kill_calls == [{"api_key": "profile-key"}]
    assert backend._load_sandbox_record("sync-contract", "base") is None


def test_failed_initial_sync_preserves_resumed_sandbox(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    host = tmp_path / "skills" / "existing" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        backend,
        "iter_sync_files",
        lambda _base: [(str(host), "/home/user/.hermes/skills/existing/SKILL.md")],
    )

    first = _environment(persistent_filesystem=True)
    sandbox_id = first._sandbox_id
    session = first._sandbox
    first.cleanup()
    session.files.fail_writes = True

    with pytest.raises(backend.EnvironmentConnectionError, match="initial state sync"):
        _environment(persistent_filesystem=True)

    assert session.kill_calls == []
    assert session.pause_calls == [
        {"keep_memory": False, "api_key": "profile-key"},
        {"keep_memory": False, "api_key": "profile-key"},
    ]
    assert backend._load_sandbox_record("sync-contract", "base") == {
        "sandbox_id": sandbox_id,
        "template": "base",
    }


def test_successful_empty_initial_sync_pulls_new_skill_and_memory_directories(
    e2b_service,
    monkeypatch,
):
    monkeypatch.setattr(backend, "iter_sync_files", lambda _base: [])
    env = _environment(persistent_filesystem=True)
    env._sandbox.files.download_payload = _tar_bytes(
        {
            "home/user/.hermes/skills/incident-triage/SKILL.md": b"triage skill",
            "home/user/.hermes/memories/MEMORY.md": b"incident lesson",
        }
    )

    env.cleanup()

    hermes_home = get_hermes_home()
    assert (hermes_home / "skills/incident-triage/SKILL.md").read_bytes() == b"triage skill"
    assert (hermes_home / "memories/MEMORY.md").read_bytes() == b"incident lesson"


def test_missing_active_sandbox_is_recreated_and_state_is_reuploaded(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("triage", encoding="utf-8")
    remote = "/home/user/.hermes/skills/incident-triage/SKILL.md"
    monkeypatch.setattr(
        backend,
        "iter_sync_files",
        lambda _base: [(str(host), remote)],
    )
    env = _environment(persistent_filesystem=True)
    stale = env._sandbox
    stale.reconnect_effect = SandboxMissing("removed outside Hermes")

    env._before_execute()

    replacement = env._sandbox
    assert replacement is not stale
    assert replacement.files.files[remote] == b"triage"
    assert backend._load_sandbox_record("sync-contract", "base")["sandbox_id"] == (
        replacement.sandbox_id
    )
    env.cleanup()


def test_cleanup_does_not_recreate_missing_sandbox(
    e2b_service,
    monkeypatch,
):
    monkeypatch.setattr(backend, "iter_sync_files", lambda _base: [])
    env = _environment(persistent_filesystem=True)
    stale = env._sandbox
    stale.reconnect_effect = SandboxMissing("removed outside Hermes")

    env.cleanup()

    assert len(e2b_service.sessions) == 1
    assert stale.pause_calls == []
    assert backend._load_sandbox_record("sync-contract", "base") is None


def test_sync_reports_transaction_failure_to_caller(tmp_path):
    host = tmp_path / "skill.md"
    host.write_text("state", encoding="utf-8")

    def fail_upload(_host: str, _remote: str) -> None:
        raise RuntimeError("offline")

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host), "/home/user/.hermes/skills/skill.md")],
        upload_fn=fail_upload,
        delete_fn=lambda _paths: None,
    )

    assert manager.sync(force=True) is False

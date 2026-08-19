"""Focused tests for the E2B terminal backend."""

from __future__ import annotations

import json
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


class SandboxFailure(Exception):
    pass


class SandboxMissing(SandboxFailure):
    pass


class ApiRateLimited(SandboxFailure):
    pass


class AuthenticationFailed(Exception):
    pass


class RemoteFileMissing(Exception):
    pass


class CommandFailed(Exception):
    def __init__(self, *, stdout="", stderr="", exit_code=1, error=None):
        super().__init__(stderr)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.error = error


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class RecordingCommandHandle:
    def __init__(self, result: CommandResult | None = None):
        self.result = result or CommandResult()
        self.kill_calls = 0

    def wait(self, on_stdout=None, on_stderr=None):
        if self.result.stdout and on_stdout:
            on_stdout(self.result.stdout)
        if self.result.stderr and on_stderr:
            on_stderr(self.result.stderr)
        if self.result.exit_code:
            raise CommandFailed(
                stdout=self.result.stdout,
                stderr=self.result.stderr,
                exit_code=self.result.exit_code,
                error=self.result.error,
            )
        return self.result

    def kill(self):
        self.kill_calls += 1
        return True


class RecordingCommands:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.run_hook = None

    def run(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        if self.run_hook is not None:
            return self.run_hook(command, kwargs)
        result = CommandResult()
        return RecordingCommandHandle(result) if kwargs.get("background") else result


class RecordingFiles:
    def __init__(self):
        self.write_calls: list[tuple[str, bytes]] = []
        self.write_files_calls: list[list[dict]] = []
        self.remove_calls: list[str] = []
        self.remove_effects: dict[str, Exception] = {}
        self.read_payload = b"archive"

    def write(self, path: str, data: bytes):
        self.write_calls.append((path, data))

    def write_files(self, files: list[dict]):
        self.write_files_calls.append(files)

    def remove(self, path: str):
        self.remove_calls.append(path)
        effect = self.remove_effects.get(path)
        if effect:
            raise effect

    def read(self, path: str, *, format: str):
        assert format == "bytes"
        return bytearray(self.read_payload)


class SandboxSession:
    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        self.commands = RecordingCommands()
        self.files = RecordingFiles()
        self.connect_calls: list[dict] = []
        self.pause_calls: list[dict] = []
        self.kill_calls: list[dict] = []
        self.reconnect_effect: Exception | None = None
        self.pause_effect: Exception | None = None

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.reconnect_effect:
            raise self.reconnect_effect
        return self

    def pause(self, **kwargs):
        self.pause_calls.append(kwargs)
        if self.pause_effect:
            raise self.pause_effect
        return True

    def kill(self, **kwargs):
        self.kill_calls.append(kwargs)
        return True


class SandboxService:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.connect_calls: list[tuple[str, dict]] = []
        self.sessions: dict[str, SandboxSession] = {}
        self.connect_effect: Exception | None = None

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        session = SandboxSession(f"sb-{len(self.create_calls)}")
        self.sessions[session.sandbox_id] = session
        return session

    def connect(self, sandbox_id: str, **kwargs):
        self.connect_calls.append((sandbox_id, kwargs))
        if self.connect_effect:
            raise self.connect_effect
        return self.sessions.setdefault(sandbox_id, SandboxSession(sandbox_id))


class RecordingSyncManager:
    instances: list["RecordingSyncManager"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sync_calls: list[bool] = []
        self.sync_back_calls = 0
        self.instances.append(self)

    def sync(self, *, force=False):
        self.sync_calls.append(force)

    def sync_back(self):
        self.sync_back_calls += 1


@pytest.fixture()
def e2b_backend(monkeypatch):
    service = SandboxService()

    e2b_root = types.ModuleType("e2b")
    e2b_root.Sandbox = SimpleNamespace(create=service.create, connect=service.connect)
    exceptions = types.ModuleType("e2b.exceptions")
    exceptions.SandboxNotFoundException = SandboxMissing
    exceptions.RateLimitException = ApiRateLimited
    exceptions.FileNotFoundException = RemoteFileMissing
    exceptions.SandboxException = SandboxFailure
    exceptions.AuthenticationException = AuthenticationFailed
    command_handle = types.ModuleType("e2b.sandbox.commands.command_handle")
    command_handle.CommandExitException = CommandFailed

    monkeypatch.setitem(sys.modules, "e2b", e2b_root)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "e2b.sandbox", types.ModuleType("e2b.sandbox"))
    monkeypatch.setitem(
        sys.modules,
        "e2b.sandbox.commands",
        types.ModuleType("e2b.sandbox.commands"),
    )
    monkeypatch.setitem(
        sys.modules,
        "e2b.sandbox.commands.command_handle",
        command_handle,
    )

    from tools.environments import e2b as backend

    RecordingSyncManager.instances.clear()
    monkeypatch.setattr(backend, "_ensure_e2b_sdk", lambda: None)
    monkeypatch.setattr(backend, "FileSyncManager", RecordingSyncManager)
    monkeypatch.setattr(backend.E2BEnvironment, "init_session", lambda self: None)
    return backend, service


def make_environment(backend, **kwargs):
    return backend.E2BEnvironment(
        api_key="profile-key",
        task_id="task-1",
        timeout=30,
        lifetime_seconds=90,
        **kwargs,
    )


def test_persistent_create_resume_and_pause_contract(e2b_backend):
    backend, service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)

    create = service.create_calls[0]
    assert create["template"] == "base"
    assert create["timeout"] == 90
    assert create["api_key"] == "profile-key"
    assert create["lifecycle"] == {
        "on_timeout": {"action": "pause", "keep_memory": False},
        "auto_resume": False,
    }
    sandbox_id = env._sandbox_id
    assert service.sessions[sandbox_id].commands.calls[0][0] == (
        "mkdir -p /home/user/.hermes"
    )
    env.cleanup()
    assert service.sessions[sandbox_id].pause_calls == [
        {"keep_memory": False, "api_key": "profile-key"}
    ]

    resumed = make_environment(backend, persistent_filesystem=True)
    assert service.connect_calls == [
        (sandbox_id, {"timeout": 90, "api_key": "profile-key"})
    ]
    resumed.cleanup()


def test_ephemeral_cleanup_kills_sandbox(e2b_backend):
    backend, service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    assert service.create_calls[0]["lifecycle"] == {
        "on_timeout": "kill",
        "auto_resume": False,
    }

    env.cleanup()

    assert session.kill_calls == [{"api_key": "profile-key"}]
    assert session.pause_calls == []


def test_missing_saved_sandbox_creates_fresh_but_transient_resume_does_not(
    e2b_backend,
):
    backend, service = e2b_backend
    backend._store_sandbox_record("task-1", "sb-stale", "base")
    service.connect_effect = SandboxMissing("gone")

    env = make_environment(backend, persistent_filesystem=True)
    assert env._sandbox_id == "sb-1"
    assert len(service.create_calls) == 1

    env.cleanup()
    backend._store_sandbox_record("task-1", "sb-rate-limited", "base")
    service.connect_effect = ApiRateLimited("try later")
    service.create_calls.clear()

    with pytest.raises(backend.EnvironmentConnectionError, match="try later"):
        make_environment(backend, persistent_filesystem=True)

    assert service.create_calls == []
    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == "sb-rate-limited"


def test_failed_pause_preserves_persistence_pointer(e2b_backend, caplog):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)
    sandbox_id = env._sandbox_id
    env._sandbox.pause_effect = RuntimeError("pause unavailable")

    env.cleanup()

    assert "pause unavailable" in caplog.text
    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == sandbox_id


def test_command_cancellation_is_pid_scoped_and_race_safe(e2b_backend):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    run_entered = threading.Event()
    allow_handle = threading.Event()
    command_killed = threading.Event()

    class BlockingCommandHandle(RecordingCommandHandle):
        def wait(self, on_stdout=None, on_stderr=None):
            assert command_killed.wait(timeout=2)
            raise CommandFailed(stderr="killed", exit_code=137)

        def kill(self):
            super().kill()
            command_killed.set()
            return True

    handle = BlockingCommandHandle()

    def delayed_run(_command, kwargs):
        assert kwargs["background"] is True
        run_entered.set()
        assert allow_handle.wait(timeout=2)
        return handle

    session.commands.run_hook = delayed_run
    process = env._run_bash("sleep 30", timeout=1)
    assert run_entered.wait(timeout=2)
    process.kill()
    allow_handle.set()

    assert process.wait(timeout=2) == 137
    assert handle.kill_calls == 1
    assert session.kill_calls == []


def test_command_transport_failure_is_reported_as_backend_degradation(e2b_backend):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)

    def unavailable(_command, _kwargs):
        raise ApiRateLimited("capacity unavailable")

    env._sandbox.commands.run_hook = unavailable

    with pytest.raises(backend.EnvironmentConnectionError, match="capacity unavailable"):
        env.execute("echo hello", timeout=2)


def test_e2b_file_transport_uses_bulk_bytes_and_idempotent_delete(
    e2b_backend,
    tmp_path,
):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.bin"
    first.write_text("hello", encoding="utf-8")
    second.write_bytes(b"\x00\x01")

    env._e2b_bulk_upload(
        [(str(first), "/home/user/.hermes/first.txt"), (str(second), "/tmp/second.bin")]
    )
    payload = env._sandbox.files.write_files_calls[0]
    assert payload == [
        {"path": "/home/user/.hermes/first.txt", "data": b"hello"},
        {"path": "/tmp/second.bin", "data": b"\x00\x01"},
    ]

    env._sandbox.files.remove_effects["/already-gone"] = RemoteFileMissing()
    env._e2b_delete(["/already-gone", "/present"])
    assert env._sandbox.files.remove_calls == ["/already-gone", "/present"]

    destination = tmp_path / "sync.tar"
    env._sandbox.files.read_payload = b"tar-bytes"
    env._e2b_bulk_download(destination)
    assert destination.read_bytes() == b"tar-bytes"


def test_config_bridge_and_profile_scoped_cache_keys(monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import terminal_tool

    home = Path(__import__("os").environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "terminal:\n  backend: e2b\n  e2b_template: team-template\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_E2B_TEMPLATE", raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    config = terminal_tool._get_env_config()
    assert config["env_type"] == "e2b"
    assert config["e2b_template"] == "team-template"
    assert config["cwd"] == "/home/user"

    token_a = set_hermes_home_override(home / "profile-a")
    try:
        key_a = terminal_tool._resolve_container_task_id(None)
    finally:
        reset_hermes_home_override(token_a)
    token_b = set_hermes_home_override(home / "profile-b")
    try:
        key_b = terminal_tool._resolve_container_task_id(None)
    finally:
        reset_hermes_home_override(token_b)
    assert key_a != key_b
    assert key_a.endswith(":default")
    assert key_b.endswith(":default")


def test_factory_uses_scoped_key_and_never_foreign_process_secret(
    e2b_backend,
    monkeypatch,
):
    _backend, service = e2b_backend
    from agent import secret_scope
    from tools import terminal_tool
    from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

    assert "E2B_API_KEY" in _HERMES_PROVIDER_ENV_BLOCKLIST
    monkeypatch.setenv("E2B_API_KEY", "foreign-process-key")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"E2B_API_KEY": "profile-a-key"})
    try:
        env = terminal_tool._create_environment(
            env_type="e2b",
            image="",
            cwd="/home/user",
            timeout=30,
            container_config={
                "e2b_template": "base",
                "lifetime_seconds": 90,
                "container_persistent": False,
            },
            task_id="profile-a",
        )
        assert service.create_calls[-1]["api_key"] == "profile-a-key"
        env.cleanup()
    finally:
        secret_scope.reset_secret_scope(token)

    empty = secret_scope.set_secret_scope({})
    try:
        with pytest.raises(ValueError, match="E2B_API_KEY"):
            terminal_tool._create_environment(
                env_type="e2b",
                image="",
                cwd="/home/user",
                timeout=30,
                container_config={"container_persistent": False},
                task_id="profile-b",
            )
    finally:
        secret_scope.reset_secret_scope(empty)
        secret_scope.set_multiplex_active(False)


def test_persistence_store_is_structured_and_template_scoped(e2b_backend):
    backend, _service = e2b_backend
    backend._store_sandbox_record("task-1", "sb-base", "base")
    backend._store_sandbox_record("task-1", "sb-custom", "custom")

    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == "sb-base"
    assert backend._load_sandbox_record("task-1", "custom")["sandbox_id"] == "sb-custom"
    raw = json.loads(backend._sandbox_store_path().read_text(encoding="utf-8"))
    assert len(raw) == 2


def test_setup_wizard_persists_e2b_backend_template_and_key(monkeypatch):
    from hermes_cli import setup

    saved_env: dict[str, str] = {}
    monkeypatch.setattr(setup, "prompt_choice", lambda *_args: 5)
    monkeypatch.setattr(setup, "get_env_value", lambda _name: None)
    monkeypatch.setattr(
        setup,
        "prompt",
        lambda _label, default=None, password=False: (
            "profile-api-key" if password else "team-template"
        ),
    )
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *_args: True)
    monkeypatch.setattr(setup, "save_env_value", saved_env.__setitem__)
    monkeypatch.setattr(setup, "save_config", lambda _config: None)
    monkeypatch.setattr(setup, "print_header", lambda *_args: None)
    monkeypatch.setattr(setup, "print_info", lambda *_args: None)
    monkeypatch.setattr(setup, "print_success", lambda *_args: None)

    config = {"terminal": {"backend": "local"}}
    setup.setup_terminal_backend(config)

    assert config["terminal"] == {
        "backend": "e2b",
        "e2b_template": "team-template",
        "container_persistent": True,
    }
    assert saved_env == {
        "E2B_API_KEY": "profile-api-key",
        "TERMINAL_ENV": "e2b",
        "TERMINAL_E2B_TEMPLATE": "team-template",
    }

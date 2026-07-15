from __future__ import annotations

import base64
from pathlib import Path

import pytest

import tools.environments.isolated_worker as environment_module
import tools.terminal_tool as terminal_module
from gateway.isolated_worker import canonical_lease_id
from hermes_cli import config as config_module
from tools.environments.base import BaseEnvironment
from tools.environments.isolated_worker import IsolatedWorkerEnvironment


class _FakeClient:
    instances: list["_FakeClient"] = []

    def __init__(self, socket_path: Path, **kwargs):
        self.socket_path = socket_path
        self.kwargs = kwargs
        self.started: list[dict] = []
        self.cancelled: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def start(self, command, *, cwd, timeout_seconds, stdin=b""):
        session_id = f"job-{len(self.started)}"
        self.started.append(
            {
                "session_id": session_id,
                "command": command,
                "cwd": cwd,
                "timeout_seconds": timeout_seconds,
                "stdin": stdin,
            }
        )
        return session_id

    def poll(self, session_id, *, wait_milliseconds=100):
        del wait_milliseconds
        output = b"" if session_id == "job-0" else b"worker-output"
        return {
            "session_id": session_id,
            "state": "exited",
            "returncode": 0,
            "stdout_b64": base64.b64encode(output).decode("ascii"),
            "stderr_b64": "",
            "drained": True,
            "complete": True,
        }

    def cancel(self, session_id):
        self.cancelled.append(session_id)
        return {"session_id": session_id, "state": "cancelled"}

    def close(self):
        self.closed = True


def test_environment_derives_lease_from_exact_session_and_never_falls_back(
    monkeypatch,
) -> None:
    _FakeClient.instances.clear()
    monkeypatch.setattr(environment_module, "IsolatedWorkerClient", _FakeClient)
    monkeypatch.setattr(
        BaseEnvironment,
        "init_session",
        lambda self: setattr(self, "_snapshot_ready", True),
    )

    environment = IsolatedWorkerEnvironment(
        socket_path=Path("/run/muncho-isolated-worker/worker.sock"),
        expected_server_uid=101,
        expected_server_gid=102,
        expected_socket_uid=0,
        expected_socket_gid=103,
        task_id="conversation-42",
        timeout=180,
    )
    client = _FakeClient.instances[-1]
    assert environment.lease_id == canonical_lease_id("conversation-42")
    assert client.kwargs["lease_id"] == canonical_lease_id("conversation-42")
    assert client.started[0]["cwd"] == Path("/workspace")
    assert "/workspace/.hermes-runtime" in client.started[0]["command"]

    process = environment._run_bash(
        "printf worker-output",
        timeout=999,
        stdin_data="input",
    )
    result = environment._wait_for_process(process, timeout=2)
    assert result == {"output": "worker-output", "returncode": 0}
    assert client.started[1]["cwd"] == Path("/workspace")
    assert client.started[1]["timeout_seconds"] == 300
    assert client.started[1]["stdin"] == b"input"

    environment.cleanup()
    assert client.closed is True
    with pytest.raises(RuntimeError, match="isolated_worker_environment_closed"):
        environment._execute_worker(
            "true",
            cwd=Path("/workspace"),
            timeout=1,
        )

    with pytest.raises(ValueError, match="requires_exact_session_id"):
        IsolatedWorkerEnvironment(
            socket_path=Path("/run/muncho-isolated-worker/worker.sock"),
            expected_server_uid=101,
            expected_server_gid=102,
            expected_socket_uid=0,
            expected_socket_gid=103,
            task_id="default",
        )


def test_terminal_factory_uses_config_only_and_preserves_generic_keying(
    monkeypatch,
) -> None:
    monkeypatch.setenv("TERMINAL_ENV", "isolated_worker")
    monkeypatch.setenv(
        "TERMINAL_ISOLATED_WORKER_SOCKET",
        "/run/muncho-isolated-worker/worker.sock",
    )
    monkeypatch.setenv("TERMINAL_ISOLATED_WORKER_SERVER_UID", "101")
    monkeypatch.setenv("TERMINAL_ISOLATED_WORKER_SERVER_GID", "102")
    monkeypatch.setenv("TERMINAL_ISOLATED_WORKER_SOCKET_UID", "0")
    monkeypatch.setenv("TERMINAL_ISOLATED_WORKER_SOCKET_GID", "103")
    config = terminal_module._get_env_config()
    assert config["cwd"] == "/workspace"
    assert config["isolated_worker_server_uid"] == 101
    assert config["isolated_worker_socket_gid"] == 103
    assert terminal_module._resolve_container_task_id("conversation-42") == "conversation-42"
    with pytest.raises(ValueError, match="requires_exact_session_id"):
        terminal_module._resolve_container_task_id(None)
    with pytest.raises(ValueError, match="requires_exact_session_id"):
        terminal_module._resolve_container_task_id("default")

    captured: dict = {}

    class FakeEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(environment_module, "IsolatedWorkerEnvironment", FakeEnvironment)
    created = terminal_module._create_environment(
        env_type="isolated_worker",
        image="",
        cwd="/workspace",
        timeout=180,
        task_id="conversation-42",
        container_config={
            "isolated_worker_socket": config["isolated_worker_socket"],
            "isolated_worker_server_uid": 101,
            "isolated_worker_server_gid": 102,
            "isolated_worker_socket_uid": 0,
            "isolated_worker_socket_gid": 103,
        },
    )
    assert isinstance(created, FakeEnvironment)
    assert captured["task_id"] == "conversation-42"
    assert captured["socket_path"] == Path(
        "/run/muncho-isolated-worker/worker.sock"
    )

    monkeypatch.setenv("TERMINAL_ENV", "docker")
    terminal_module._task_env_overrides.clear()
    assert terminal_module._resolve_container_task_id("ordinary-child") == "default"


def test_isolated_worker_settings_bridge_from_config_without_lease_override(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "read_raw_config", lambda: {"terminal": {}})
    target: dict[str, str] = {}
    config_module.apply_terminal_config_to_env(
        env=target,
        override=True,
        config={
            "terminal": {
                "backend": "isolated_worker",
                "isolated_worker_socket": "/run/muncho-isolated-worker/worker.sock",
                "isolated_worker_server_uid": 101,
                "isolated_worker_server_gid": 102,
                "isolated_worker_socket_uid": 0,
                "isolated_worker_socket_gid": 103,
            }
        },
    )
    assert target == {
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_ISOLATED_WORKER_SOCKET": "/run/muncho-isolated-worker/worker.sock",
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": "101",
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": "102",
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": "0",
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": "103",
    }
    assert all("LEASE" not in key for key in target)


def test_requirements_probe_authenticates_worker_without_creating_a_lease(
    monkeypatch,
) -> None:
    import gateway.isolated_worker as worker_module

    calls: list[tuple[str, object]] = []

    class _ReadinessClient:
        def __init__(self, socket_path, **kwargs):
            calls.append(("init", (socket_path, kwargs)))

        def connect(self):
            calls.append(("connect", None))

        def close(self):
            calls.append(("close", None))

    monkeypatch.setattr(worker_module, "IsolatedWorkerClient", _ReadinessClient)
    monkeypatch.setattr(
        terminal_module,
        "_get_env_config",
        lambda: {
            "env_type": "isolated_worker",
            "isolated_worker_socket": "/run/muncho-isolated-worker/worker.sock",
            "isolated_worker_server_uid": 101,
            "isolated_worker_server_gid": 102,
            "isolated_worker_socket_uid": 0,
            "isolated_worker_socket_gid": 103,
        },
    )

    assert terminal_module.check_terminal_requirements() is True
    assert [name for name, _value in calls] == ["init", "connect", "close"]
    _socket, kwargs = calls[0][1]
    assert kwargs["lease_id"] == "lease-readiness-probe"


def test_prompt_and_host_probe_do_not_invent_a_shared_isolated_worker_lease(
    monkeypatch,
) -> None:
    import agent.prompt_builder as prompt_builder
    import tools.env_probe as env_probe

    monkeypatch.setenv("TERMINAL_ENV", "isolated_worker")
    monkeypatch.delenv("HERMES_ENVIRONMENT_HINT", raising=False)
    prompt_builder._clear_backend_probe_cache()

    assert prompt_builder._probe_remote_backend("isolated_worker") is None
    hints = prompt_builder.build_environment_hints()
    assert "Terminal backend: isolated_worker" in hints
    assert "lease-isolated Linux worker" in hints
    assert "User home directory:" not in hints

    monkeypatch.setattr(
        env_probe,
        "_python_version_of",
        lambda _binary: (_ for _ in ()).throw(
            AssertionError("host probe must not run for isolated_worker")
        ),
    )
    assert env_probe._build_probe_line() == ""

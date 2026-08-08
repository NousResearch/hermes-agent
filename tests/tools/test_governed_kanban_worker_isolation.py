"""Provider-free tests for governed Kanban worker process isolation."""

from __future__ import annotations

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_completion_broker as broker
from tools import terminal_tool
from tools import kanban_tools


ISOLATION = {
    "mode": "docker",
    "network": False,
    "toolsets": ["terminal"],
    "mount_hermes_resources": False,
    "broker_socket": "/run/hermes-kanban-broker/completion.sock",
}


def test_dispatcher_isolation_marker_replaces_profile_toolsets():
    env = {
        "TERMINAL_ENV": "local",
        "TERMINAL_DOCKER_NETWORK": "true",
        "TERMINAL_DOCKER_VOLUMES": '["/:/host"]',
    }
    toolsets = kb._apply_governed_worker_isolation(env, ISOLATION)

    assert toolsets == ["terminal"]
    assert env["HERMES_KANBAN_ISOLATED_WORKER"] == "1"
    assert env["TERMINAL_ENV"] == "docker"
    assert env["TERMINAL_DOCKER_NETWORK"] == "false"
    assert env["TERMINAL_DOCKER_VOLUMES"] == "[]"
    assert env["TERMINAL_DOCKER_FORWARD_ENV"] == "[]"
    assert env["TERMINAL_DOCKER_EXTRA_ARGS"] == "[]"
    assert env["HERMES_KANBAN_BROKER_SOCKET"] == ISOLATION["broker_socket"]


def test_dispatcher_rejects_weakened_isolation_contract():
    weakened = dict(ISOLATION, network=True)
    try:
        kb._apply_governed_worker_isolation({}, weakened)
    except RuntimeError as exc:
        assert "isolation contract is invalid" in str(exc)
    else:
        raise AssertionError("weakened isolation must fail closed")


def test_terminal_marker_overrides_unsafe_profile_and_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    monkeypatch.setenv("HERMES_KANBAN_ISOLATED_WORKER", "1")
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "untrusted-image")
    monkeypatch.setenv("TERMINAL_DOCKER_NETWORK", "true")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_RUN_AS_HOST_USER", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true")
    monkeypatch.setenv("TERMINAL_DOCKER_FORWARD_ENV", '["SECRET"]')
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", '["/:/host"]')
    monkeypatch.setenv("TERMINAL_DOCKER_ENV", '{"SECRET":"value"}')
    monkeypatch.setenv("TERMINAL_DOCKER_EXTRA_ARGS", '["--privileged"]')

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["host_cwd"] == str(tmp_path)
    assert config["cwd"] == "/workspace"
    assert config["docker_mount_cwd_to_workspace"] is True
    assert config["docker_network"] is False
    assert config["container_persistent"] is False
    assert config["docker_run_as_host_user"] is False
    assert config["docker_persist_across_processes"] is False
    assert config["docker_orphan_reaper"] is False
    assert config["docker_forward_env"] == []
    assert config["docker_volumes"] == []
    assert config["docker_env"] == {}
    assert config["docker_extra_args"] == []
    assert config["docker_mount_hermes_resources"] is False
    assert config["docker_image"] != "untrusted-image"


def test_environment_factory_forwards_resource_mount_denial(monkeypatch):
    captured = {}

    class FakeDockerEnvironment:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(terminal_tool, "_DockerEnvironment", FakeDockerEnvironment)
    monkeypatch.setattr(terminal_tool, "_maybe_reap_docker_orphans", lambda _cc: None)

    terminal_tool._create_environment(
        "docker",
        "synthetic-image",
        "/workspace",
        30,
        container_config={
            "docker_orphan_reaper": False,
            "docker_mount_hermes_resources": False,
            "docker_network": False,
        },
        host_cwd="/synthetic/workspace",
    )

    assert captured["mount_hermes_resources"] is False
    assert captured["network"] is False


def test_worker_complete_adapter_uses_broker_without_opening_db(monkeypatch):
    captured = {}

    def fake_request(socket_path, request):
        captured["socket_path"] = socket_path
        captured["request"] = request
        return {
            "version": "1.0.0",
            "request_id": request["request_id"],
            "ok": True,
            "completed": True,
            "receipt_sha256": "b" * 64,
        }

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_testbroker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_PROFILE", "profile-test")
    monkeypatch.setenv(
        "HERMES_KANBAN_BROKER_SOCKET",
        "/run/hermes-kanban-broker/completion.sock",
    )
    monkeypatch.setattr(broker, "request_completion", fake_request)
    monkeypatch.setattr(
        kanban_tools,
        "_connect",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("DB path bypassed broker")),
    )

    response = kanban_tools._handle_complete(
        {
            "task_id": "t_testbroker",
            "result": "{}",
            "summary": "synthetic",
        }
    )

    assert '"ok": true' in response.lower()
    assert str(captured["socket_path"]) == ISOLATION["broker_socket"]
    assert captured["request"]["profile"] == "profile-test"
    assert captured["request"]["run_id"] == 42

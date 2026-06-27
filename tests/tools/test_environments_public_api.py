"""Regression coverage for the public environments package API."""

from __future__ import annotations


def test_get_environment_is_publicly_importable_for_backend_probes(tmp_path):
    from tools.environments import get_environment

    env = get_environment({"env_type": "local", "cwd": str(tmp_path), "timeout": 5})

    assert env.__class__.__name__ == "LocalEnvironment"
    assert hasattr(env, "execute")


def test_get_environment_forwards_container_config(monkeypatch):
    import tools.terminal_tool as terminal_tool
    from tools.environments import get_environment

    captured = {}
    sentinel = object()

    def fake_create_environment(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(terminal_tool, "_create_environment", fake_create_environment)

    result = get_environment(
        {
            "env_type": "docker",
            "docker_image": "example/image:latest",
            "cwd": "/workspace",
            "host_cwd": "/host/project",
            "timeout": 9,
            "container_cpu": 2,
            "container_memory": 4096,
            "container_disk": 20000,
            "container_persistent": False,
            "docker_volumes": ["/host/cache:/cache"],
            "docker_forward_env": ["API_KEY"],
            "docker_env": {"MODE": "test"},
            "docker_run_as_host_user": True,
            "docker_extra_args": ["--network=none"],
            "docker_persist_across_processes": False,
            "docker_mount_cwd_to_workspace": True,
        },
        task_id="backend-probe",
    )

    assert result is sentinel
    assert captured["env_type"] == "docker"
    assert captured["image"] == "example/image:latest"
    assert captured["cwd"] == "/workspace"
    assert captured["host_cwd"] == "/host/project"
    assert captured["timeout"] == 9
    assert captured["task_id"] == "backend-probe"
    assert captured["ssh_config"] is None
    assert captured["container_config"] == {
        "container_cpu": 2,
        "container_memory": 4096,
        "container_disk": 20000,
        "container_persistent": False,
        "modal_mode": "auto",
        "docker_volumes": ["/host/cache:/cache"],
        "docker_mount_cwd_to_workspace": True,
        "docker_forward_env": ["API_KEY"],
        "docker_env": {"MODE": "test"},
        "docker_run_as_host_user": True,
        "docker_extra_args": ["--network=none"],
        "docker_persist_across_processes": False,
        "docker_orphan_reaper": True,
    }

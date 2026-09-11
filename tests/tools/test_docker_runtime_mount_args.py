from __future__ import annotations

import pytest

from tools.environments import docker as docker_env
from tools.environments.docker import (
    _mount_capable_extra_args,
    _normalize_docker_extra_args,
    _normalize_runtime_mounts,
    _runtime_mount_args,
)


def test_runtime_mount_args_use_strict_bind_mount_syntax():
    mounts = _normalize_runtime_mounts([
        {
            "source": "/host/task",
            "target": "/workspace",
            "read_only": False,
            "purpose": "workspace",
        },
        {
            "source": "/host/repo/.git",
            "target": "/host/repo/.git",
            "read_only": False,
            "purpose": "git-common-dir",
        },
    ])
    args, targets = _runtime_mount_args(mounts)
    assert targets == {"/workspace", "/host/repo/.git"}
    assert args == [
        "--mount", "type=bind,src=/host/task,dst=/workspace",
        "--mount", "type=bind,src=/host/repo/.git,dst=/host/repo/.git",
    ]


def test_runtime_mount_args_reject_duplicate_targets():
    with pytest.raises(ValueError, match="duplicate"):
        _normalize_runtime_mounts([
            {"source": "/a", "target": "/workspace"},
            {"source": "/b", "target": "/workspace"},
        ])


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--mount", "type=bind,src=/opt/data,dst=/host-data"],
        ["--mount=type=bind,src=/opt/data,dst=/host-data"],
        ["-v", "/var/run/docker.sock:/var/run/docker.sock"],
        ["-v=/var/run/docker.sock:/var/run/docker.sock"],
        ["-v/var/run/docker.sock:/var/run/docker.sock"],
        ["-itv/opt/data:/host-data"],
        ["--volume", "/opt/data:/host-data"],
        ["--volume=/opt/data:/host-data"],
        ["--volumes-from", "other-container"],
        ["--volumes-from=other-container"],
        ["--tmpfs", "/workspace:rw"],
        ["--tmpfs=/workspace:rw"],
    ],
)
def test_runtime_mounts_reject_mount_capable_extra_args_before_docker(
    monkeypatch, extra_args
):
    def must_not_reach_docker():
        pytest.fail("Docker availability/probe must not run before mount-arg rejection")

    monkeypatch.setattr(docker_env, "_ensure_docker_available", must_not_reach_docker)
    with pytest.raises(ValueError, match="mount-capable docker_extra_args"):
        docker_env.DockerEnvironment(
            image="python:3.11",
            runtime_mounts=[
                {
                    "source": "/host/task",
                    "target": "/workspace",
                    "read_only": False,
                    "purpose": "workspace",
                }
            ],
            extra_args=extra_args,
        )


def test_runtime_mounts_preserve_non_mounting_extra_args():
    benign = [
        "--hostname=kanban-worker",
        "--read-only",
        "-w",
        "/var/workspace",
        "-e",
        "ENV=value",
    ]
    normalized = _normalize_docker_extra_args(benign)
    assert normalized == benign
    assert _mount_capable_extra_args(normalized) == []


def _stub_runtime_docker_constructor(monkeypatch, tmp_path, *, remote: bool):
    """Make DockerEnvironment construction deterministic without starting Docker."""
    from tools import credential_files

    credential = tmp_path / "credential.json"
    credential.write_text("token", encoding="utf-8")
    skills = tmp_path / "skills"
    skills.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    ca = tmp_path / "egress-ca.crt"
    ca.write_text("ca", encoding="utf-8")

    monkeypatch.setattr(docker_env, "_ensure_docker_available", lambda *a, **k: None)
    monkeypatch.setattr(docker_env, "_cgroup_limits_available", lambda *a, **k: False)
    monkeypatch.setattr(docker_env, "_image_uses_init_entrypoint", lambda *a, **k: False)
    monkeypatch.setattr(docker_env, "find_docker", lambda: "docker")
    monkeypatch.setattr(docker_env.DockerEnvironment, "init_session", lambda self: None)
    monkeypatch.setattr(
        credential_files,
        "get_credential_file_mounts",
        lambda: [{"host_path": str(credential), "container_path": "/root/.auth.json"}],
    )
    monkeypatch.setattr(
        credential_files,
        "get_skills_directory_mount",
        lambda: [{"host_path": str(skills), "container_path": "/opt/hermes/skills"}],
    )
    monkeypatch.setattr(
        credential_files,
        "get_cache_directory_mounts",
        lambda: [{"host_path": str(cache), "container_path": "/opt/hermes/cache"}],
    )
    monkeypatch.setattr(
        docker_env,
        "_readonly_skill_mount_args",
        lambda: [
            "-v", f"{credential}:/root/.auth.json:ro",
            "-v", f"{skills}:/opt/hermes/skills:ro",
            "-v", f"{cache}:/opt/hermes/cache",
        ],
    )
    monkeypatch.setattr(
        docker_env,
        "_egress_proxy_args_for_docker",
        lambda: (
            ["-v", f"{ca}:/etc/ssl/certs/hermes-egress-ca.crt:ro"],
            {"HTTPS_PROXY": "http://host.docker.internal:9000"},
            ["--add-host", "host.docker.internal:host-gateway"],
        ),
    )
    monkeypatch.setattr(docker_env, "_egress_enforce_on_docker", lambda: False)

    calls = []

    class Result:
        returncode = 0
        stdout = "container-id\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return Result()

    monkeypatch.setattr(docker_env, "run_capture", fake_run)

    class RemoteSelector:
        cli_args = ("--host", "ssh://docker@example.com")

    endpoint_selector = RemoteSelector() if remote else None
    return calls, endpoint_selector, credential, skills, cache, ca


@pytest.mark.parametrize("remote", [False, True])
def test_runtime_final_docker_argv_has_only_dispatcher_host_binds(
    monkeypatch, tmp_path, remote
):
    """Default persistence + ancillary producers cannot widen a task runtime."""
    calls, selector, credential, skills, cache, ca = _stub_runtime_docker_constructor(
        monkeypatch, tmp_path, remote=remote
    )
    workspace_source = "/remote/projects/task" if remote else str(tmp_path / "task")
    if not remote:
        (tmp_path / "task").mkdir()

    profile_data = tmp_path / "profile-data"
    host_cwd = tmp_path / "host-cwd"
    profile_data.mkdir()
    host_cwd.mkdir()

    env = docker_env.DockerEnvironment(
        image="python:3.11",
        cwd="/workspace",
        task_id="kbt-runtime",
        persistent_filesystem=True,
        persist_across_processes=False,
        volumes=[f"{profile_data}:/profile-data"],
        host_cwd=str(host_cwd),
        auto_mount_cwd=True,
        endpoint_selector=selector,
        runtime_mounts=[
            {
                "source": workspace_source,
                "target": "/workspace",
                "read_only": False,
                "purpose": "workspace",
            }
        ],
    )

    run_calls = [cmd for cmd in calls if "run" in cmd]
    assert len(run_calls) == 1
    run_cmd = run_calls[0]
    joined = " ".join(run_cmd)
    expected_mount = f"type=bind,src={workspace_source},dst=/workspace"
    assert run_cmd.count("--mount") == 1
    assert expected_mount in run_cmd
    assert "-v" not in run_cmd
    assert str(profile_data) not in joined
    assert str(host_cwd) not in joined
    assert str(credential) not in joined
    assert str(skills) not in joined
    assert str(cache) not in joined
    assert str(ca) not in joined
    assert "/root:rw,exec,size=1g" in run_cmd
    assert "/home:rw,exec,size=1g" in run_cmd
    if remote:
        assert run_cmd[1:3] == ["--host", "ssh://docker@example.com"]
    assert env._all_run_args.count("--mount") == 1


def test_runtime_enforced_egress_bind_fails_before_docker_run(monkeypatch, tmp_path):
    """An enforced egress CA bind cannot bypass task-owned mount authority."""
    calls, _selector, _credential, _skills, _cache, _ca = (
        _stub_runtime_docker_constructor(monkeypatch, tmp_path, remote=False)
    )
    monkeypatch.setattr(docker_env, "_egress_enforce_on_docker", lambda: True)
    workspace = tmp_path / "task"
    workspace.mkdir()

    with pytest.raises(RuntimeError, match="egress CA host bind"):
        docker_env.DockerEnvironment(
            image="python:3.11",
            cwd="/workspace",
            task_id="kbt-runtime",
            persistent_filesystem=True,
            persist_across_processes=False,
            runtime_mounts=[
                {
                    "source": str(workspace),
                    "target": "/workspace",
                    "read_only": False,
                    "purpose": "workspace",
                }
            ],
        )

    assert not any("run" in cmd for cmd in calls)

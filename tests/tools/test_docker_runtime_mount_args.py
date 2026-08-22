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
        ["--volume", "/opt/data:/host-data"],
        ["--volume=/opt/data:/host-data"],
        ["--volumes-from", "other-container"],
        ["--volumes-from=other-container"],
    ],
)
def test_runtime_mounts_reject_mount_capable_extra_args_before_docker(
    monkeypatch, extra_args
):
    def must_not_reach_docker():
        pytest.fail("Docker availability/probe must not run before mount-arg rejection")

    monkeypatch.setattr(docker_env, "_ensure_docker_available", must_not_reach_docker)
    with pytest.raises(ValueError, match="task-scoped runtime mounts"):
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
    benign = ["--hostname=kanban-worker", "--read-only"]
    normalized = _normalize_docker_extra_args(benign)
    assert normalized == benign
    assert _mount_capable_extra_args(normalized) == []

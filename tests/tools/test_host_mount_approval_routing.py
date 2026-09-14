"""Host-mount classification and approval policy must agree."""

import pytest


def _config(backend, extra_args):
    return {
        "env_type": backend,
        "host_cwd": None,
        "docker_mount_cwd_to_workspace": False,
        "docker_volumes": [],
        "apple_container_volumes": [],
        "docker_extra_args": extra_args if backend == "docker" else [],
        "apple_container_extra_args": extra_args if backend == "apple_container" else [],
    }


def test_docker_clustered_bind_is_host_access():
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(_config("docker", ["-iv/tmp:/mnt"])) is True


@pytest.mark.parametrize("extra_args", [
    ["--volumes-from", "fixture-source:ro"],
    ["--volumes-from=fixture-source"],
    ["--volume-driver", "local", "-v", "fixture-volume:/data"],
    ["--volume-driver=local", "-v", "fixture-volume:/data"],
    ["--mount", "type=volume,source=fixture-volume,target=/data,volume-driver=local,"
     "volume-opt=type=none,volume-opt=o=bind,volume-opt=device=/tmp/fixture"],
    ["--mount=type=volume,target=/data,volume-opt=device=/tmp/fixture,volume-opt=o=bind"],
])
def test_declared_indirect_docker_mounts_are_host_access(extra_args):
    from tools.terminal_tool import _docker_has_host_access

    assert _docker_has_host_access(_config("docker", extra_args)) is True

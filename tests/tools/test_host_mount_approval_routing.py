"""Host-mount classification and approval policy must agree."""


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

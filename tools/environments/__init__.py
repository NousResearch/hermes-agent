"""Hermes execution environment backends.

Each backend provides the same interface (BaseEnvironment ABC) for running
shell commands in a specific execution context: local, Docker, SSH,
Singularity, Modal, or Daytona. (Modal additionally has direct and
Nous-managed modes, selected via terminal.modal_mode.)

The terminal_tool.py factory (_create_environment) selects the backend
based on the TERMINAL_ENV configuration.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

from tools.environments.base import BaseEnvironment

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona"})
_IMAGE_CONFIG_KEYS = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}


def _default_cwd(env_type: str) -> str:
    if env_type == "ssh":
        return "~"
    if env_type in _CONTAINER_BACKENDS:
        return "/root"
    try:
        return os.getcwd()
    except FileNotFoundError:
        return os.path.expanduser("~")


def get_environment(
    config: Mapping[str, Any] | None = None,
    *,
    task_id: str = "default",
) -> BaseEnvironment:
    """Create an execution environment from terminal-style configuration.

    This public wrapper supports lightweight callers, such as prompt backend
    probes, without requiring them to import terminal_tool private helpers
    directly. Backend construction is still delegated to the terminal tool
    factory so behavior stays aligned with normal terminal execution.
    """
    config = config or {}
    env_type = str(config.get("env_type") or "local")
    timeout = int(config.get("timeout") or 180)
    cwd = str(config.get("cwd") or _default_cwd(env_type))
    image_key = _IMAGE_CONFIG_KEYS.get(env_type)
    image = str(config.get(image_key, "") if image_key else "")

    ssh_config = None
    if env_type == "ssh":
        ssh_config = {
            "host": config.get("ssh_host", ""),
            "user": config.get("ssh_user", ""),
            "port": config.get("ssh_port", 22),
            "key": config.get("ssh_key", ""),
            "persistent": config.get("ssh_persistent", False),
        }

    container_config = None
    if env_type in _CONTAINER_BACKENDS:
        container_config = {
            "container_cpu": config.get("container_cpu", 1),
            "container_memory": config.get("container_memory", 5120),
            "container_disk": config.get("container_disk", 51200),
            "container_persistent": config.get("container_persistent", True),
            "modal_mode": config.get("modal_mode", "auto"),
            "docker_volumes": config.get("docker_volumes", []),
            "docker_mount_cwd_to_workspace": config.get(
                "docker_mount_cwd_to_workspace",
                False,
            ),
            "docker_forward_env": config.get("docker_forward_env", []),
            "docker_env": config.get("docker_env", {}),
            "docker_run_as_host_user": config.get(
                "docker_run_as_host_user",
                False,
            ),
            "docker_extra_args": config.get("docker_extra_args", []),
            "docker_persist_across_processes": config.get(
                "docker_persist_across_processes",
                True,
            ),
            "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
        }

    local_config = None
    if env_type == "local":
        local_config = {
            "persistent": config.get("local_persistent", False),
        }

    from tools.terminal_tool import _create_environment

    return _create_environment(
        env_type=env_type,
        image=image,
        cwd=cwd,
        timeout=timeout,
        ssh_config=ssh_config,
        container_config=container_config,
        local_config=local_config,
        task_id=task_id,
        host_cwd=config.get("host_cwd"),
    )


__all__ = ["BaseEnvironment", "get_environment"]

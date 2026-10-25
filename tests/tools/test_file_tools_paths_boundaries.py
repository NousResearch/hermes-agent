from pathlib import PurePosixPath

import pytest

from agent import terminal_env_registry
from agent.terminal_env_provider import TerminalEnvironmentProvider
import tools.file_tools_paths as paths
import tools.terminal_tool as terminal_tool
from tools.terminal_tool_backends import _build_plugin_env


class LocalCompatibleEnvironment:
    """A plugin may reuse a local-looking implementation for a remote sandbox."""


class ContainerProvider(TerminalEnvironmentProvider):
    is_container = True

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def create_environment(self, **kwargs):
        return LocalCompatibleEnvironment()


@pytest.mark.parametrize("backend_name", ["acmebox", "firecracker_lab"])
def test_stamped_plugin_backend_controls_path_semantics(
    backend_name, tmp_path, monkeypatch
):
    """The provider stamp is authoritative even when its class name says local."""
    host_target = tmp_path / "host-target"
    host_target.mkdir()
    container_path = tmp_path / "container-path"
    container_path.symlink_to(host_target, target_is_directory=True)

    provider = ContainerProvider(backend_name)
    previous = terminal_env_registry.snapshot_registration(backend_name)
    terminal_env_registry.register_provider(provider)
    env = _build_plugin_env(
        env_type=backend_name,
        image="",
        cwd="/workspace",
        timeout=30,
        cc={},
        task_id="default",
    )
    assert env._hermes_backend_name == backend_name
    monkeypatch.setattr(terminal_tool, "_active_environments", {"default": env})
    monkeypatch.setattr(terminal_tool, "_get_env_config", lambda: {"env_type": "local"})
    try:
        resolved = paths._resolve_path_for_task(str(container_path / "artifact.txt"))

        assert isinstance(resolved, PurePosixPath)
        assert resolved == PurePosixPath(str(container_path / "artifact.txt"))
        assert resolved != PurePosixPath(str(host_target / "artifact.txt"))
    finally:
        terminal_env_registry.restore_registration(
            backend_name, provider, previous
        )

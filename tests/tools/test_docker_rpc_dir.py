"""Tests for HERMES_RPC_DIR injection into Docker sandbox (#13142)."""

import pytest
from unittest.mock import patch


class TestCreateEnvironmentRpcDir:
    """Tests for _create_environment injecting HERMES_RPC_DIR."""

    @pytest.fixture(autouse=True)
    def _import(self):
        """Import _create_environment with all deps mocked."""
        # We need to mock the heavy imports that terminal_tool does at module level
        with patch("tools.environments.docker.subprocess.run"), \
             patch("tools.environments.docker.find_docker", return_value="/usr/bin/docker"):
            from tools.environments.docker import get_docker_rpc_dir
            from tools.terminal_tool import _create_environment, _DockerEnvironment
            self._create_environment = _create_environment
            self._DockerEnvironment = _DockerEnvironment
            self._get_docker_rpc_dir = get_docker_rpc_dir

    def test_docker_env_includes_hermes_rpc_dir(self):
        """DockerEnvironment receives HERMES_RPC_DIR in its env dict."""
        with patch.object(self._DockerEnvironment, "__init__", return_value=None) as mock_init:
            mock_init.return_value = None
            self._create_environment(
                env_type="docker",
                image="python:3.11",
                cwd="/workspace",
                timeout=60,
                container_config={"docker_env": {"FOO": "bar"}},
            )
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["env"]["HERMES_RPC_DIR"] == self._get_docker_rpc_dir("default")
            assert call_kwargs["env"]["FOO"] == "bar"

    def test_docker_env_preserves_existing_vars(self):
        """Existing docker_env vars are preserved when HERMES_RPC_DIR is added."""
        with patch.object(self._DockerEnvironment, "__init__", return_value=None) as mock_init:
            mock_init.return_value = None
            self._create_environment(
                env_type="docker",
                image="python:3.11",
                cwd="/workspace",
                timeout=60,
                container_config={"docker_env": {"MY_VAR": "hello", "NUM": "42"}},
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["env"]["MY_VAR"] == "hello"
            assert call_kwargs["env"]["NUM"] == "42"
            assert call_kwargs["env"]["HERMES_RPC_DIR"] == self._get_docker_rpc_dir("default")

    def test_docker_env_does_not_override_explicit_hermes_rpc_dir(self):
        """If HERMES_RPC_DIR is already in docker_env, it is not overridden."""
        with patch.object(self._DockerEnvironment, "__init__", return_value=None) as mock_init:
            mock_init.return_value = None
            self._create_environment(
                env_type="docker",
                image="python:3.11",
                cwd="/workspace",
                timeout=60,
                container_config={"docker_env": {"HERMES_RPC_DIR": "/custom/rpc/path"}},
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["env"]["HERMES_RPC_DIR"] == "/custom/rpc/path"

    def test_docker_env_empty_config(self):
        """When docker_env is not specified, HERMES_RPC_DIR is still injected."""
        with patch.object(self._DockerEnvironment, "__init__", return_value=None) as mock_init:
            mock_init.return_value = None
            self._create_environment(
                env_type="docker",
                image="python:3.11",
                cwd="/workspace",
                timeout=60,
                container_config={},
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["env"]["HERMES_RPC_DIR"] == self._get_docker_rpc_dir("default")

    def test_docker_env_uses_task_specific_rpc_dir(self):
        """DockerEnvironment should receive the same task-specific RPC dir as execute_code."""
        with patch.object(self._DockerEnvironment, "__init__", return_value=None) as mock_init:
            self._create_environment(
                env_type="docker",
                image="python:3.11",
                cwd="/workspace",
                timeout=60,
                task_id="task/one",
                container_config={},
            )
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["env"]["HERMES_RPC_DIR"] == self._get_docker_rpc_dir("task/one")


class TestLocalEnvironmentUnaffected:
    """Verify local environment is not touched by the fix."""

    @pytest.fixture(autouse=True)
    def _import(self):
        with patch("tools.environments.docker.subprocess.run"), \
             patch("tools.environments.docker.find_docker", return_value="/usr/bin/docker"):
            from tools.terminal_tool import _create_environment, _LocalEnvironment
            self._create_environment = _create_environment
            self._LocalEnvironment = _LocalEnvironment

    def test_local_environment_no_rpc_dir(self):
        """LocalEnvironment does not receive HERMES_RPC_DIR."""
        with patch.object(self._LocalEnvironment, "__init__", return_value=None) as mock_init:
            mock_init.return_value = None
            self._create_environment(
                env_type="local",
                image="",
                cwd="/workspace",
                timeout=60,
            )
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]
            # LocalEnvironment takes cwd and timeout, no env param
            assert "env" not in call_kwargs

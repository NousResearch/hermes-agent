"""Tests for tools.environments.docker.find_docker — Docker CLI discovery."""

import os
from unittest.mock import patch

import pytest

from tools.environments import docker as docker_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the module-level docker executable cache between tests."""
    docker_mod._docker_executable = None
    docker_mod._runtime_is_podman = None
    yield
    docker_mod._docker_executable = None
    docker_mod._runtime_is_podman = None


class TestFindDocker:
    def test_found_via_shutil_which(self):
        with patch("tools.environments.docker.shutil.which", return_value="/usr/bin/docker"):
            result = docker_mod.find_docker()
        assert result == "/usr/bin/docker"

    def test_not_in_path_falls_back_to_known_locations(self, tmp_path):
        # Create a fake docker binary at a known path
        fake_docker = tmp_path / "docker"
        fake_docker.write_text("#!/bin/sh\n")
        fake_docker.chmod(0o755)

        with patch("tools.environments.docker.shutil.which", return_value=None), \
             patch("tools.environments.docker._DOCKER_SEARCH_PATHS", [str(fake_docker)]):
            result = docker_mod.find_docker()
        assert result == str(fake_docker)

    def test_returns_none_when_not_found(self):
        with patch("tools.environments.docker.shutil.which", return_value=None), \
             patch("tools.environments.docker._DOCKER_SEARCH_PATHS", ["/nonexistent/docker"]):
            result = docker_mod.find_docker()
        assert result is None

    def test_caches_result(self):
        with patch("tools.environments.docker.shutil.which", return_value="/usr/local/bin/docker"):
            first = docker_mod.find_docker()
        # Second call should use cache, not call shutil.which again
        with patch("tools.environments.docker.shutil.which", return_value=None):
            second = docker_mod.find_docker()
        assert first == second == "/usr/local/bin/docker"

    def test_env_var_override_takes_precedence(self, tmp_path):
        """HERMES_DOCKER_BINARY overrides PATH and known-location discovery."""
        fake_binary = tmp_path / "podman"
        fake_binary.write_text("#!/bin/sh\n")
        fake_binary.chmod(0o755)

        with patch.dict(os.environ, {"HERMES_DOCKER_BINARY": str(fake_binary)}), \
             patch("tools.environments.docker.shutil.which", return_value="/usr/bin/docker"):
            result = docker_mod.find_docker()
        assert result == str(fake_binary)

    def test_env_var_override_ignored_if_not_executable(self, tmp_path):
        """Non-executable HERMES_DOCKER_BINARY falls through to normal discovery."""
        fake_binary = tmp_path / "podman"
        fake_binary.write_text("#!/bin/sh\n")
        fake_binary.chmod(0o644)  # not executable

        with patch.dict(os.environ, {"HERMES_DOCKER_BINARY": str(fake_binary)}), \
             patch("tools.environments.docker.shutil.which", return_value="/usr/bin/docker"):
            result = docker_mod.find_docker()
        assert result == "/usr/bin/docker"

    def test_env_var_override_ignored_if_nonexistent(self):
        """Non-existent HERMES_DOCKER_BINARY path falls through."""
        with patch.dict(os.environ, {"HERMES_DOCKER_BINARY": "/nonexistent/podman"}), \
             patch("tools.environments.docker.shutil.which", return_value="/usr/bin/docker"):
            result = docker_mod.find_docker()
        assert result == "/usr/bin/docker"

    def test_podman_on_path_used_when_docker_missing(self):
        """When docker is not on PATH, podman is tried next."""
        def which_side_effect(name):
            if name == "docker":
                return None
            if name == "podman":
                return "/usr/bin/podman"
            return None

        with patch("tools.environments.docker.shutil.which", side_effect=which_side_effect), \
             patch("tools.environments.docker._DOCKER_SEARCH_PATHS", []):
            result = docker_mod.find_docker()
        assert result == "/usr/bin/podman"

    def test_docker_preferred_over_podman(self):
        """When both docker and podman are on PATH, docker wins."""
        def which_side_effect(name):
            if name == "docker":
                return "/usr/bin/docker"
            if name == "podman":
                return "/usr/bin/podman"
            return None

        with patch("tools.environments.docker.shutil.which", side_effect=which_side_effect):
            result = docker_mod.find_docker()
        assert result == "/usr/bin/docker"


class TestIsPodman:
    """Tests for is_podman() and runtime_name() helpers."""

    def test_docker_is_not_podman(self):
        with patch("tools.environments.docker.shutil.which", return_value="/usr/bin/docker"):
            docker_mod.find_docker()
        assert docker_mod.is_podman() is False
        assert docker_mod.runtime_name() == "Docker"

    def test_podman_is_podman(self):
        def which_side_effect(name):
            if name == "podman":
                return "/usr/bin/podman"
            return None

        with patch("tools.environments.docker.shutil.which", side_effect=which_side_effect), \
             patch("tools.environments.docker._DOCKER_SEARCH_PATHS", []):
            docker_mod.find_docker()
        assert docker_mod.is_podman() is True
        assert docker_mod.runtime_name() == "Podman"

    def test_env_override_podman(self, tmp_path):
        fake = tmp_path / "podman"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)

        with patch.dict(os.environ, {"HERMES_DOCKER_BINARY": str(fake)}):
            docker_mod.find_docker()
        assert docker_mod.is_podman() is True

    def test_no_runtime_is_not_podman(self):
        with patch("tools.environments.docker.shutil.which", return_value=None), \
             patch("tools.environments.docker._DOCKER_SEARCH_PATHS", []):
            docker_mod.find_docker()
            assert docker_mod.is_podman() is False

    def test_is_podman_cached(self):
        """Second call uses cache."""
        with patch("tools.environments.docker.shutil.which", return_value="/usr/bin/docker"):
            docker_mod.find_docker()
            first = docker_mod.is_podman()
        # Patch find_docker to return podman — should still return cached False
        docker_mod._docker_executable = "/usr/bin/podman"
        second = docker_mod.is_podman()
        assert first is False
        assert second is False  # cached


class TestStorageOptPodman:
    """_storage_opt_supported() should return False for Podman."""

    def test_podman_skips_storage_opt(self):
        def which_side_effect(name):
            if name == "podman":
                return "/usr/bin/podman"
            return None

        with patch("tools.environments.docker.shutil.which", side_effect=which_side_effect), \
             patch("tools.environments.docker._DOCKER_SEARCH_PATHS", []):
            docker_mod.find_docker()
        # Reset storage opt cache
        docker_mod._storage_opt_ok = None
        assert docker_mod.DockerEnvironment._storage_opt_supported() is False

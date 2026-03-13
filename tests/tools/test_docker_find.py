"""Tests for tools.environments.docker.find_docker — Docker CLI discovery."""

import os
from unittest.mock import patch

import pytest

from tools.environments import docker as docker_mod


@pytest.fixture(autouse=True)
def _reset_cache():
    """Clear the module-level docker executable cache between tests."""
    docker_mod._docker_executable = None
    yield
    docker_mod._docker_executable = None


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

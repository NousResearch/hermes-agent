"""Tests for tools/container_paths.py: path translation across Docker bind mounts."""

import json
import os
from unittest.mock import patch

import pytest

import tools.container_paths as cp
from tools.terminal_scope import reset_terminal_scope, set_terminal_scope


@pytest.fixture
def skills_root(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    return root


@pytest.fixture
def no_config(monkeypatch):
    """Keep config.yaml out of the picture: only the environment or scope counts."""
    monkeypatch.setattr(cp, "_config_terminal_value", lambda key: None)


@pytest.fixture
def docker_backend(monkeypatch, skills_root, no_config):
    """Docker terminal backend whose only skill mount is ``skills_root``."""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "[]")
    with patch(
        "tools.credential_files._skill_dir_roots",
        lambda base: iter([(skills_root, f"{base}/skills")]),
    ):
        yield


@pytest.fixture
def local_backend(monkeypatch, skills_root, no_config):
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
    with patch(
        "tools.credential_files._skill_dir_roots",
        lambda base: iter([(skills_root, f"{base}/skills")]),
    ):
        yield


class TestContainerToHost:
    def test_writable_mount_resolves_existing_dir(self, tmp_path, monkeypatch, no_config):
        host = tmp_path / "ws"
        (host / "sub").mkdir(parents=True)
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{host}:/workspace"]))
        assert cp.to_host_dir("/workspace") == str(host.resolve())
        assert cp.to_host_dir("/workspace/sub") == str((host / "sub").resolve())
        assert cp.to_host_dir("/workspace/missing") is None

    def test_readonly_mount_is_not_translated(self, tmp_path, monkeypatch, no_config):
        host = tmp_path / "ro"
        host.mkdir()
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{host}:/data:ro"]))
        assert cp.container_mount_map() == []
        assert cp.to_host_dir("/data") is None

    def test_non_docker_backend_does_not_translate(self, tmp_path, monkeypatch, no_config):
        host = tmp_path / "ws"
        host.mkdir()
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{host}:/workspace"]))
        assert cp.container_mount_map() == []
        assert cp.to_host_dir("/workspace") is None

    def test_invalid_json_degrades_to_no_translation(self, monkeypatch, no_config):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", "not json")
        assert cp.container_mount_map() == []
        assert cp.to_host_dir("/workspace") is None


class TestHostToContainer:
    def test_skill_dir_maps_under_root_hermes(self, docker_backend, skills_root):
        assert cp.to_container_path(str(skills_root / "demo" / "scripts")) == (
            "/root/.hermes/skills/demo/scripts"
        )

    def test_writable_alias_wins_a_tie(self, docker_backend, tmp_path, monkeypatch):
        host = tmp_path / "ws"
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES",
            json.dumps([f"{host}:/readonly-view:ro", f"{host}:/workspace"]),
        )
        assert cp.to_container_path(str(host / "out.md")) == "/workspace/out.md"

    def test_local_backend_returns_none(self, local_backend, skills_root):
        assert cp.to_container_path(str(skills_root / "demo")) is None


class TestTerminalScope:
    def test_scoped_policy_wins_over_process_env(self, tmp_path, monkeypatch, no_config):
        """A multiplexed process translates with the active profile's mounts,
        not whatever the launch process left in os.environ."""
        env_host = tmp_path / "env-ws"
        scoped_host = tmp_path / "scoped-ws"
        env_host.mkdir()
        scoped_host.mkdir()
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps([f"{env_host}:/workspace"]))
        token = set_terminal_scope({
            "TERMINAL_ENV": "docker",
            "TERMINAL_DOCKER_VOLUMES": json.dumps([f"{scoped_host}:/workspace"]),
        })
        try:
            assert cp.to_host_dir("/workspace") == str(scoped_host.resolve())
        finally:
            reset_terminal_scope(token)
        assert cp.to_host_dir("/workspace") is None

    def test_config_fallback_does_not_write_process_env(self, tmp_path, monkeypatch):
        host = tmp_path / "ws"
        host.mkdir()
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        monkeypatch.delenv("TERMINAL_DOCKER_VOLUMES", raising=False)
        config = {"backend": "docker", "docker_volumes": json.dumps([f"{host}:/workspace"])}
        monkeypatch.setattr(cp, "_config_terminal_value", lambda key: config.get(key))
        assert cp.to_host_dir("/workspace") == str(host.resolve())
        assert "TERMINAL_ENV" not in os.environ
        assert "TERMINAL_DOCKER_VOLUMES" not in os.environ


class TestWindowsHostPaths:
    """A Windows drive path on the host side keeps its drive colon when parsed."""

    def test_drive_path_spec_is_parsed(self, monkeypatch, no_config):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES",
            json.dumps(["C:\\Users\\me\\project:/workspace", "D:/data:/data:ro"]),
        )
        assert cp._volume_specs() == [
            ("c:/Users/me/project", "/workspace", False),
            ("d:/data", "/data", True),
        ]
        assert cp.container_mount_map() == [("/workspace", "c:/Users/me/project")]

    def test_drive_path_translates_to_container(self, docker_backend, monkeypatch):
        monkeypatch.setenv(
            "TERMINAL_DOCKER_VOLUMES", json.dumps(["C:\\Users\\me\\project:/workspace"]),
        )
        assert cp.to_container_path("C:\\Users\\me\\project\\out\\report.md") == (
            "/workspace/out/report.md"
        )
        assert cp.to_container_path("c:/Users/me/project") == "/workspace"
        assert cp.to_container_path("C:\\Users\\me\\project-other\\x.md") is None

    def test_bare_drive_letter_is_not_a_host_path(self, monkeypatch, no_config):
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_VOLUMES", json.dumps(["C:/workspace"]))
        assert cp._volume_specs() == []

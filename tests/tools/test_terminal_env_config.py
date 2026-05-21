import inspect
import os

import pytest

from hermes_cli.terminal_config import default_terminal_config
from tools import terminal_tool


@pytest.fixture(autouse=True)
def clear_terminal_env_vars(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TERMINAL_"):
            monkeypatch.delenv(name, raising=False)


def test_get_env_config_without_env_uses_shared_terminal_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    terminal_defaults = default_terminal_config()

    config = terminal_tool._get_env_config()

    assert config["env_type"] == terminal_defaults["env_type"]
    assert config["cwd"] == str(tmp_path)
    assert config["timeout"] == terminal_defaults["timeout"]
    assert config["lifetime_seconds"] == terminal_defaults["lifetime_seconds"]
    assert config["docker_image"] == terminal_defaults["docker_image"]
    assert config["modal_image"] == terminal_defaults["modal_image"]
    assert config["daytona_image"] == terminal_defaults["daytona_image"]
    assert config["singularity_image"] == terminal_defaults["singularity_image"]
    assert config["vercel_runtime"] == terminal_defaults["vercel_runtime"]
    assert config["container_memory"] == terminal_defaults["container_memory"]
    assert config["container_disk"] == terminal_defaults["container_disk"]


def test_get_env_config_env_vars_override_shared_terminal_defaults(monkeypatch):
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "77")
    monkeypatch.setenv("TERMINAL_LIFETIME_SECONDS", "88")
    monkeypatch.setenv("TERMINAL_CONTAINER_MEMORY", "1024")

    config = terminal_tool._get_env_config()

    assert config["env_type"] == "docker"
    assert config["timeout"] == 77
    assert config["lifetime_seconds"] == 88
    assert config["container_memory"] == 1024


def test_terminal_diagnostic_timeout_default_does_not_use_stale_literal():
    """The operator-facing diagnostic script should not duplicate old timeout defaults."""
    source = inspect.getsource(terminal_tool)

    assert "os.getenv('TERMINAL_TIMEOUT', '60')" not in source

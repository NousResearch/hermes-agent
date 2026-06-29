"""Regression tests for Docker cold-start sandbox escape (#54354).

When ``TERMINAL_ENV`` is not set but ``terminal.backend`` is configured
in config.yaml, ``_resolve_backend_type()`` must read the backend type
directly from the config file instead of falling back to ``"local"``.
"""

import sys
from pathlib import Path
import pytest

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import tools.terminal_tool  # noqa: F401
    _tt_mod = sys.modules["tools.terminal_tool"]
except ImportError:
    pytest.skip("hermes-agent tools not importable (missing deps)", allow_module_level=True)


class TestResolveBackendType:
    """_resolve_backend_type() must not silently downgrade Docker to local."""

    def test_env_var_used_when_no_terminal_config_exists(self, monkeypatch):
        """When config has no terminal section, TERMINAL_ENV remains usable."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})

        monkeypatch.setenv("TERMINAL_ENV", "docker")
        assert _tt_mod._resolve_backend_type() == "docker"

        monkeypatch.setenv("TERMINAL_ENV", "modal")
        assert _tt_mod._resolve_backend_type() == "modal"

        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert _tt_mod._resolve_backend_type() == "local"

    def test_env_var_set_to_ssh_without_terminal_config(self, monkeypatch):
        """Explicit TERMINAL_ENV=ssh must be honoured without terminal config."""
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
        monkeypatch.setenv("TERMINAL_ENV", "ssh")
        assert _tt_mod._resolve_backend_type() == "ssh"

    def test_falls_back_to_config_when_env_not_set(self, monkeypatch):
        """When TERMINAL_ENV is not set, read terminal.backend from config.yaml."""
        monkeypatch.delenv("TERMINAL_ENV", raising=False)

        def _mock_load_config():
            return {"terminal": {"backend": "docker"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _mock_load_config,
        )
        assert _tt_mod._resolve_backend_type() == "docker"

    def test_config_backend_overrides_stale_env_var(self, monkeypatch):
        """When config has terminal.backend, config.yaml is authoritative."""
        monkeypatch.setenv("TERMINAL_ENV", "local")

        def _mock_load_config():
            return {"terminal": {"backend": "docker"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _mock_load_config,
        )
        assert _tt_mod._resolve_backend_type() == "docker"

    def test_falls_back_to_local_when_config_has_no_terminal_section(self, monkeypatch):
        """When config.yaml has no terminal section, default to local."""
        monkeypatch.delenv("TERMINAL_ENV", raising=False)

        def _mock_load_config():
            return {}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _mock_load_config,
        )
        assert _tt_mod._resolve_backend_type() == "local"

    def test_falls_back_to_local_when_config_terminal_has_no_backend(self, monkeypatch):
        """When terminal.backend is absent from config, default to local."""
        monkeypatch.delenv("TERMINAL_ENV", raising=False)

        def _mock_load_config():
            return {"terminal": {"cwd": "/home/user"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _mock_load_config,
        )
        assert _tt_mod._resolve_backend_type() == "local"

    def test_falls_back_to_local_when_config_load_fails(self, monkeypatch):
        """When config.yaml is unreadable, default to local (no crash)."""
        monkeypatch.delenv("TERMINAL_ENV", raising=False)

        def _mock_load_config():
            raise OSError("Permission denied")

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _mock_load_config,
        )
        # Must not raise — falls back to local
        assert _tt_mod._resolve_backend_type() == "local"

    def test_docker_backend_flows_through_get_env_config_from_config(self, monkeypatch):
        """End-to-end: _get_env_config() uses config.yaml during cold start."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "stale-image")

        cfg = {
            "terminal": {
                "backend": "docker",
                "docker_image": "python:3.12-slim",
                "docker_forward_env": ["OPENAI_API_KEY"],
                "docker_volumes": ["/tmp:/host-tmp:ro"],
                "docker_env": {"HERMES_TEST": "1"},
                "docker_extra_args": ["--network=none"],
                "timeout": 42,
                "container_cpu": 2.5,
                "container_memory": 1024,
                "container_disk": 2048,
            }
        }

        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: cfg)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
        config = _tt_mod._get_env_config()

        assert config["env_type"] == "docker"
        assert config["docker_image"] == "python:3.12-slim"
        assert config["docker_forward_env"] == ["OPENAI_API_KEY"]
        assert config["docker_volumes"] == ["/tmp:/host-tmp:ro"]
        assert config["docker_env"] == {"HERMES_TEST": "1"}
        assert config["docker_extra_args"] == ["--network=none"]
        assert config["timeout"] == 42
        assert config["container_cpu"] == 2.5
        assert config["container_memory"] == 1024
        assert config["container_disk"] == 2048
        assert config["cwd"] == "/root"

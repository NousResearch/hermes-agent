"""Regression tests for Docker cold-start sandbox escape (#54354).

When ``TERMINAL_ENV`` is not set but ``terminal.backend`` is configured
in config.yaml, ``_resolve_backend_type()`` must read the backend type
directly from the config file instead of falling back to ``"local"``.

When the config bridge fails for an isolated backend, ``_get_env_config()``
must fail closed instead of silently downgrading to host-local execution.
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
        """End-to-end: _get_env_config() routes to Docker when config says so."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "stale-image")

        cfg = {"terminal": {"backend": "docker", "docker_image": "python:3.12-slim"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: cfg)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)
        config = _tt_mod._get_env_config()

        # Config backend overrides stale TERMINAL_ENV=local.
        assert config["env_type"] == "docker"
        # Stale env var is replaced by the config value.
        assert config["docker_image"] == "python:3.12-slim"

    def test_config_bridge_failure_raises_when_backend_is_docker(self, monkeypatch):
        """When config bridge fails and config intends Docker, fail closed."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "stale-image")

        # Simulate a config bridge failure.
        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env",
            lambda env=None: (_ for _ in ()).throw(RuntimeError("Bridge error")),
        )
        # load_config_readonly still returns the intended Docker config
        # for the fallback check in _get_env_config.
        cfg = {"terminal": {"backend": "docker"}}
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

        with pytest.raises(RuntimeError, match="Refusing to downgrade"):
            _tt_mod._get_env_config()

    def test_config_bridge_failure_allows_local_when_backend_is_local(self, monkeypatch):
        """When config bridge fails but config intends local, it is safe to proceed."""
        monkeypatch.setenv("TERMINAL_ENV", "local")

        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env",
            lambda env=None: (_ for _ in ()).throw(RuntimeError("Bridge error")),
        )
        cfg = {"terminal": {"backend": "local"}}
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

        # Must not raise — local → local is a safe fallback.
        config = _tt_mod._get_env_config()
        assert config["env_type"] == "local"

    def test_config_bridge_failure_preserves_explicit_termin_env(self, monkeypatch):
        """When bridge fails but TERMINAL_ENV=docker was explicitly set, keep it."""
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "my-image")

        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env",
            lambda env=None: (_ for _ in ()).throw(RuntimeError("Bridge error")),
        )
        cfg = {"terminal": {"backend": "docker", "docker_image": "cfg-image"}}
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: cfg)

        # TERMINAL_ENV=docker is preserved; TERMINAL_DOCKER_IMAGE is wiped.
        # config.yaml says docker → RuntimeError.
        with pytest.raises(RuntimeError, match="Refusing to downgrade"):
            _tt_mod._get_env_config()

    def test_config_bridge_and_config_read_both_fail(self, monkeypatch):
        """When both bridge and fallback config read fail, refuse to run."""
        monkeypatch.setenv("TERMINAL_ENV", "local")

        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env",
            lambda env=None: (_ for _ in ()).throw(RuntimeError("Bridge error")),
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: (_ for _ in ()).throw(OSError("Config unreadable")),
        )

        with pytest.raises(RuntimeError, match="config.yaml is unreadable"):
            _tt_mod._get_env_config()

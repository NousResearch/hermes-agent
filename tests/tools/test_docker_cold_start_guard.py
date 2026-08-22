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

    def test_normalizes_case_and_whitespace(self, monkeypatch):
        """Backend values must be normalized (strip + lowercase) like the validator.

        Regression (CodeRabbit #61882): _probe_config_unreadable() accepts
        terminal.backend: Docker / " docker", but _resolve_backend_type()
        returned the value verbatim, so case-sensitive consumers (env_probe's
        ``in _REMOTE_BACKENDS``, credential_files, _get_env_config's
        ``in _CONTAINER_BACKENDS``) took host-local branches for a config that
        requested an isolated backend.
        """
        monkeypatch.delenv("TERMINAL_ENV", raising=False)

        def _mock_load_config():
            return {"terminal": {"backend": "Docker"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly", _mock_load_config,
        )
        assert _tt_mod._resolve_backend_type() == "docker"

        # Whitespace variant too.
        monkeypatch.delenv("TERMINAL_ENV", raising=False)
        monkeypatch.setattr(
            "hermes_cli.config.read_raw_config",
            lambda: {"terminal": {"backend": " docker "}},
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"terminal": {"backend": " docker "}},
        )
        assert _tt_mod._resolve_backend_type() == "docker"

        # Environment value is normalized the same way.
        monkeypatch.setenv("TERMINAL_ENV", "  Modal ")
        monkeypatch.setattr("hermes_cli.config.read_raw_config", lambda: {})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})
        assert _tt_mod._resolve_backend_type() == "modal"

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

    def _isolate_hermes_home(self, monkeypatch, tmp_path):
        """Point HERMES_HOME at an empty temp dir so _probe_config_unreadable()
        can never read an inherited/ambient config.yaml and divert the test onto
        the unreadable-config path (CodeRabbit #61882)."""
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)

    def test_config_bridge_failure_raises_when_backend_is_docker(
        self, monkeypatch, tmp_path,
    ):
        """When config bridge fails and config intends Docker, fail closed."""
        self._isolate_hermes_home(monkeypatch, tmp_path)
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

    def test_config_bridge_failure_allows_local_when_backend_is_local(
        self, monkeypatch, tmp_path,
    ):
        """When config bridge fails but config intends local, it is safe to proceed."""
        self._isolate_hermes_home(monkeypatch, tmp_path)
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

    def test_config_bridge_failure_preserves_explicit_termin_env(
        self, monkeypatch, tmp_path,
    ):
        """When bridge fails but TERMINAL_ENV=docker was explicitly set, keep it."""
        self._isolate_hermes_home(monkeypatch, tmp_path)
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

        # Simulate the config file being present but unparseable — the path
        # that matches real production behavior (load_config_readonly absorbs
        # parse errors via last-known-good fallback instead of raising).
        monkeypatch.setattr(
            _tt_mod,
            "_probe_config_unreadable",
            lambda: True,
        )
        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env",
            lambda env=None: (_ for _ in ()).throw(RuntimeError("Bridge error")),
        )

        with pytest.raises(RuntimeError, match="config.yaml is unreadable"):
            _tt_mod._get_env_config()

    def test_malformed_config_with_stale_local_env_fails_closed(
        self, monkeypatch, tmp_path,
    ):
        """Regression: malformed YAML + stale TERMINAL_ENV=local must refuse.

        When config.yaml exists but cannot be parsed (e.g. mid-edit broken
        YAML) and TERMINAL_ENV=local, the bridge silently absorbs the parse
        error via the last-known-good fallback.  Without the
        _probe_config_unreadable guard this would downgrade a potentially
        isolated backend to local execution.  Assert it fails closed.
        """
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text(
            "terminal:\n  backend: docker\n  docker_image: python:3.12\n"
            "  !!invalid yaml syntax here ~~~",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        # Production functions now see the malformed config.yaml at HERMES_HOME.
        # _probe_config_unreadable must detect it; _get_env_config must fail.
        assert _tt_mod._probe_config_unreadable() is True

        with pytest.raises(RuntimeError, match="config.yaml is unreadable"):
            _tt_mod._get_env_config()

    def test_malformed_config_absent_stale_docker_env_fails_closed(
        self, monkeypatch, tmp_path,
    ):
        """Malformed config + explicit TERMINAL_ENV=docker: wipe stale vars, fail."""
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        config_file = hermes_home / "config.yaml"
        config_file.write_text(
            "terminal:\n  backend: docker\n  docker_image: python:3.12\n"
            "  -- broken yaml : [",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "docker")
        monkeypatch.setenv("TERMINAL_DOCKER_IMAGE", "stale-image")

        assert _tt_mod._probe_config_unreadable() is True

        with pytest.raises(RuntimeError, match="config.yaml is unreadable"):
            _tt_mod._get_env_config()


class TestProbeConfigUnreadableShapes:
    """_probe_config_unreadable() must reject empty / non-mapping / invalid configs."""

    @pytest.mark.parametrize(
        "raw",
        [
            "",                       # empty document
            "# comment only\n",        # comment-only document
            "[]\n",                   # list root
            "- a\n- b\n",             # list root with items
            "just a string\n",        # scalar root
            "42\n",                   # numeric scalar root
        ],
    )
    def test_present_but_invalid_shape_is_unreadable(self, monkeypatch, tmp_path, raw):
        """A present config that is empty/list/scalar cannot carry a backend."""
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(raw, encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is True

    def test_non_mapping_terminal_section_is_unreadable(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text("terminal: [docker]\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is True

    def test_invalid_backend_value_is_unreadable(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "terminal:\n  backend: kubernetes\n", encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is True

    def test_valid_terminal_section_is_readable(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "terminal:\n  backend: docker\n", encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is False

    def test_terminal_section_without_backend_is_readable(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "terminal:\n  cwd: /home/user\n", encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is False

    def test_absent_config_is_readable(self, monkeypatch, tmp_path):
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is False


class TestProbeManagedConfigUnreadable:
    """_probe_config_unreadable() must also fail closed on a malformed MANAGED config.

    Regression: ``managed_scope.load_managed_config()`` is fail-open (returns {} on a
    parse error), so an admin-pinned ``terminal.backend: docker`` silently vanished
    and a stale ``TERMINAL_ENV=local`` downgraded an isolated backend to host
    execution — the managed-config counterpart of the user-config probe.
    """

    @pytest.fixture(autouse=True)
    def _managed(self, monkeypatch, tmp_path):
        self._managed_dir = tmp_path / "managed"
        self._managed_dir.mkdir()
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(self._managed_dir))
        from hermes_cli import managed_scope

        managed_scope.invalidate_managed_cache()
        yield
        monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
        managed_scope.invalidate_managed_cache()

    def test_malformed_managed_config_is_unreadable(self, monkeypatch, tmp_path):
        """A present-but-unparseable managed config.yaml must fail closed."""
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")
        (self._managed_dir / "config.yaml").write_text(
            "terminal:\n  backend: docker\n  broken [[[ yaml", encoding="utf-8",
        )

        assert _tt_mod._probe_config_unreadable() is True
        with pytest.raises(RuntimeError, match="config.yaml is unreadable"):
            _tt_mod._get_env_config()

    def test_managed_config_pins_docker_over_stale_local(self, monkeypatch, tmp_path):
        """A clean managed terminal.backend: docker must override stale TERMINAL_ENV."""
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")
        (self._managed_dir / "config.yaml").write_text(
            "terminal:\n  backend: docker\n  docker_image: python:3.12\n",
            encoding="utf-8",
        )

        assert _tt_mod._probe_config_unreadable() is False
        cfg = _tt_mod._get_env_config()
        assert cfg["env_type"] == "docker"
        assert cfg["docker_image"] == "python:3.12"

    def test_absent_managed_config_stays_readable(self, monkeypatch, tmp_path):
        """No managed file at all (or no managed scope) is the normal benign case."""
        hermes_home = tmp_path / "hermes_home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("TERMINAL_ENV", "local")

        assert _tt_mod._probe_config_unreadable() is False


class TestResolveTerminalBackend:
    """resolve_terminal_backend() must honor config on cold start and fail closed."""

    def test_cold_start_docker_config_resolves_docker(self, monkeypatch):
        """config.yaml terminal.backend: docker + stale TERMINAL_ENV=local."""
        monkeypatch.setenv("TERMINAL_ENV", "local")

        def _mock_load_config():
            return {"terminal": {"backend": "docker"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _mock_load_config)

        assert _tt_mod.resolve_terminal_backend() == "docker"

    def test_bridge_failure_returns_unknown_sentinel(self, monkeypatch):
        """On an untrusted snapshot, return non-local sentinel (fail closed)."""
        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setattr(
            "hermes_cli.config.apply_terminal_config_to_env",
            lambda env=None: (_ for _ in ()).throw(RuntimeError("Bridge error")),
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: {"terminal": {"backend": "docker"}},
        )

        assert _tt_mod.resolve_terminal_backend() == "unknown"

    def test_image_source_host_reads_fail_closed_on_cold_start_docker(self, monkeypatch):
        """image_source must deny host reads when config selects docker on cold start."""
        monkeypatch.setenv("TERMINAL_ENV", "local")

        def _mock_load_config():
            return {"terminal": {"backend": "docker"}}

        monkeypatch.setattr("hermes_cli.config.read_raw_config", _mock_load_config)
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", _mock_load_config)

        from tools.image_source import _is_local_terminal_backend

        assert _is_local_terminal_backend() is False

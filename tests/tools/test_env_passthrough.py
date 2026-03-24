"""Tests for tools.env_passthrough — skill and config env var passthrough."""

import os
import pytest
import yaml

from tools.env_passthrough import (
    clear_env_passthrough,
    get_all_passthrough,
    is_env_passthrough,
    register_env_passthrough,
    reset_config_cache,
)


@pytest.fixture(autouse=True)
def _clean_passthrough():
    """Ensure a clean passthrough state for every test."""
    clear_env_passthrough()
    reset_config_cache()
    yield
    clear_env_passthrough()
    reset_config_cache()


class TestSkillScopedPassthrough:
    def test_register_and_check(self):
        assert not is_env_passthrough("TENOR_API_KEY")
        register_env_passthrough(["TENOR_API_KEY"])
        assert is_env_passthrough("TENOR_API_KEY")

    def test_register_multiple(self):
        register_env_passthrough(["FOO_TOKEN", "BAR_SECRET"])
        assert is_env_passthrough("FOO_TOKEN")
        assert is_env_passthrough("BAR_SECRET")
        assert not is_env_passthrough("OTHER_KEY")

    def test_clear(self):
        register_env_passthrough(["TENOR_API_KEY"])
        assert is_env_passthrough("TENOR_API_KEY")
        clear_env_passthrough()
        assert not is_env_passthrough("TENOR_API_KEY")

    def test_get_all(self):
        register_env_passthrough(["A_KEY", "B_TOKEN"])
        result = get_all_passthrough()
        assert "A_KEY" in result
        assert "B_TOKEN" in result

    def test_strips_whitespace(self):
        register_env_passthrough(["  SPACED_KEY  "])
        assert is_env_passthrough("SPACED_KEY")

    def test_skips_empty(self):
        register_env_passthrough(["", "  ", "VALID_KEY"])
        assert is_env_passthrough("VALID_KEY")
        assert not is_env_passthrough("")


class TestConfigPassthrough:
    def test_reads_from_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["MY_CUSTOM_KEY", "ANOTHER_TOKEN"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        reset_config_cache()

        assert is_env_passthrough("MY_CUSTOM_KEY")
        assert is_env_passthrough("ANOTHER_TOKEN")
        assert not is_env_passthrough("UNRELATED_VAR")

    def test_empty_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": []}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        reset_config_cache()

        assert not is_env_passthrough("ANYTHING")

    def test_missing_config_key(self, tmp_path, monkeypatch):
        config = {"terminal": {"backend": "local"}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        reset_config_cache()

        assert not is_env_passthrough("ANYTHING")

    def test_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        reset_config_cache()

        assert not is_env_passthrough("ANYTHING")

    def test_union_of_skill_and_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["CONFIG_KEY"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        reset_config_cache()

        register_env_passthrough(["SKILL_KEY"])
        all_pt = get_all_passthrough()
        assert "CONFIG_KEY" in all_pt
        assert "SKILL_KEY" in all_pt


class TestExecuteCodeIntegration:
    """Verify that the passthrough is checked in execute_code's env filtering."""

    def test_secret_substring_blocked_by_default(self):
        """TENOR_API_KEY should be blocked without passthrough."""
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")

        test_env = {"PATH": "/usr/bin", "TENOR_API_KEY": "test123", "HOME": "/home/user"}
        child_env = {}
        for k, v in test_env.items():
            if is_env_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        assert "PATH" in child_env
        assert "HOME" in child_env
        assert "TENOR_API_KEY" not in child_env

    def test_passthrough_allows_secret_through(self):
        """TENOR_API_KEY should pass through when registered."""
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")

        register_env_passthrough(["TENOR_API_KEY"])

        test_env = {"PATH": "/usr/bin", "TENOR_API_KEY": "test123", "HOME": "/home/user"}
        child_env = {}
        for k, v in test_env.items():
            if is_env_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        assert "PATH" in child_env
        assert "HOME" in child_env
        assert "TENOR_API_KEY" in child_env
        assert child_env["TENOR_API_KEY"] == "test123"


class TestTerminalIntegration:
    """Verify that the passthrough is checked in terminal's env sanitizers."""

    def test_blocklisted_var_blocked_by_default(self):
        from tools.environments.local import _sanitize_subprocess_env, _HERMES_PROVIDER_ENV_BLOCKLIST

        # Pick a var we know is in the blocklist
        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        env = {blocked_var: "secret_value", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert blocked_var not in result
        assert "PATH" in result

    def test_passthrough_allows_blocklisted_var(self):
        from tools.environments.local import _sanitize_subprocess_env, _HERMES_PROVIDER_ENV_BLOCKLIST

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        register_env_passthrough([blocked_var])

        env = {blocked_var: "secret_value", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert blocked_var in result
        assert result[blocked_var] == "secret_value"

    def test_make_run_env_passthrough(self, monkeypatch):
        from tools.environments.local import _make_run_env, _HERMES_PROVIDER_ENV_BLOCKLIST

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        monkeypatch.setenv(blocked_var, "secret_value")

        # Without passthrough — blocked
        result_before = _make_run_env({})
        assert blocked_var not in result_before

        # With passthrough — allowed
        register_env_passthrough([blocked_var])
        result_after = _make_run_env({})
        assert blocked_var in result_after

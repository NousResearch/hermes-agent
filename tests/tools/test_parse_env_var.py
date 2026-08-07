"""Tests for _parse_env_var and _get_env_config env-var validation."""

import importlib
import json
from unittest.mock import patch

import pytest

import sys
import tools.terminal_tool  # noqa: F401 -- ensure module is loaded
_tt_mod = sys.modules["tools.terminal_tool"]
from tools.terminal_tool import _parse_env_var


class TestParseEnvVar:
    """Unit tests for _parse_env_var."""

    # -- valid values work normally --

    def test_valid_int(self):
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "300"}):
            assert _parse_env_var("TERMINAL_TIMEOUT", "180") == 300


    def test_get_env_config_parses_docker_forward_env_json(self):
        with patch.dict("os.environ", {
            "TERMINAL_ENV": "docker",
            "TERMINAL_DOCKER_FORWARD_ENV": '["GITHUB_TOKEN", "NPM_TOKEN"]',
        }, clear=False):
            config = _tt_mod._get_env_config()
            assert config["docker_forward_env"] == ["GITHUB_TOKEN", "NPM_TOKEN"]

    def test_get_env_config_parses_tenki_forward_env_json(self):
        with patch.dict("os.environ", {
            "TERMINAL_ENV": "tenki",
            "TERMINAL_TENKI_FORWARD_ENV": '["GITHUB_TOKEN", "GH_TOKEN"]',
        }, clear=False):
            config = _tt_mod._get_env_config()
            assert config["tenki_forward_env"] == ["GITHUB_TOKEN", "GH_TOKEN"]

    def test_get_env_config_parses_tenki_numeric_settings(self):
        with patch.dict("os.environ", {
            "TERMINAL_ENV": "tenki",
            "TERMINAL_TENKI_MAX_DURATION": "7200",
            "TERMINAL_TENKI_IDLE_TIMEOUT": "600",
            "TERMINAL_TENKI_PAUSE_RETENTION": "300",
        }, clear=False):
            config = _tt_mod._get_env_config()
            assert config["tenki_max_duration"] == 7200
            assert config["tenki_idle_timeout"] == 600
            assert config["tenki_pause_retention"] == 300

    def test_stale_invalid_tenki_value_does_not_break_local_backend(self):
        # A stale/invalid tenki value bridged from config.yaml must not abort
        # _get_env_config() for a local session that never uses tenki.
        with patch.dict("os.environ", {
            "TERMINAL_ENV": "local",
            "TERMINAL_TENKI_MAX_DURATION": "not-a-number",
        }, clear=False):
            config = _tt_mod._get_env_config()
            assert config["env_type"] == "local"
            assert config["tenki_max_duration"] == 3600

    def test_invalid_tenki_value_raises_when_tenki_backend_active(self):
        with patch.dict("os.environ", {
            "TERMINAL_ENV": "tenki",
            "TERMINAL_TENKI_MAX_DURATION": "not-a-number",
        }, clear=False):
            with pytest.raises(ValueError, match="TERMINAL_TENKI_MAX_DURATION"):
                _tt_mod._get_env_config()

    def test_create_environment_passes_tenki_forward_env(self):
        import tools.environments.tenki as tenki_module

        fake_env = object()
        with patch.object(tenki_module, "TenkiEnvironment", return_value=fake_env) as mock_tenki:
            result = _tt_mod._create_environment(
                "tenki",
                image="",
                cwd="/home/tenki",
                timeout=180,
                container_config={"tenki_forward_env": ["GITHUB_TOKEN", "GH_TOKEN"]},
            )

        assert result is fake_env
        assert mock_tenki.call_args.kwargs["forward_env"] == ["GITHUB_TOKEN", "GH_TOKEN"]

    # -- invalid int raises ValueError with env var name --


    # -- invalid JSON raises ValueError with env var name --


    def test_invalid_json_includes_type_label(self):
        with patch.dict("os.environ", {"TERMINAL_DOCKER_VOLUMES": "not json"}):
            with pytest.raises(ValueError, match="valid JSON"):
                _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")


class TestImportTimeEnvParsing:
    """Module-level env parsing should never make terminal_tool unimportable."""

    def test_invalid_foreground_timeout_falls_back_to_default(self):
        try:
            with patch.dict("os.environ", {"TERMINAL_MAX_FOREGROUND_TIMEOUT": "5m"}, clear=False):
                mod = importlib.reload(_tt_mod)
                assert mod.FOREGROUND_MAX_TIMEOUT == 600
        finally:
            importlib.reload(_tt_mod)

    def test_invalid_disk_warning_threshold_falls_back_to_default(self):
        try:
            with patch.dict("os.environ", {"TERMINAL_DISK_WARNING_GB": "huge"}, clear=False):
                mod = importlib.reload(_tt_mod)
                assert mod.DISK_USAGE_WARNING_THRESHOLD_GB == 500.0
        finally:
            importlib.reload(_tt_mod)

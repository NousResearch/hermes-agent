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


    # -- TERMINAL_TIMEOUT=0 is a "0 = infinite" misunderstanding, not a
    #    request for an instant timeout (issue #85809) --

    def test_zero_timeout_falls_back_to_default_180(self, caplog):
        """TERMINAL_TIMEOUT=0 parses as a valid int, but 0 means "time out
        instantly" for the underlying subprocess/asyncio timeout -- not
        "no timeout" as a user might reasonably assume. Every command
        must not silently fail with "timed out after 0s"."""
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "0"}, clear=False):
            config = _tt_mod._get_env_config()
            assert config["timeout"] == 180, (
                f"TERMINAL_TIMEOUT=0 must fall back to the 180s default, "
                f"got {config['timeout']}"
            )
        assert any(
            "TERMINAL_TIMEOUT" in r.message and "0" in r.message
            for r in caplog.records
        ), "a warning must be logged so the misconfiguration is diagnosable"

    def test_negative_timeout_falls_back_to_default_180(self):
        """Same guard must catch a negative value, not just exactly 0."""
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "-5"}, clear=False):
            config = _tt_mod._get_env_config()
            assert config["timeout"] == 180

    def test_positive_timeout_is_used_unchanged(self):
        """Sanity: a genuinely positive, intentional timeout override must
        pass through untouched -- the guard is a floor at 0, not a
        rewrite of every configured value."""
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "86400"}, clear=False):
            config = _tt_mod._get_env_config()
            assert config["timeout"] == 86400

    def test_unset_timeout_uses_default_180(self):
        """Sanity: the ordinary unset case still resolves to the
        documented 180s default, unaffected by the new guard."""
        with patch.dict("os.environ", {}, clear=True):
            config = _tt_mod._get_env_config()
            assert config["timeout"] == 180


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

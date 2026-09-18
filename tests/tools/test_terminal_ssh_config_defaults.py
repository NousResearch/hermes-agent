"""Tests for empty ``terminal.ssh_*`` config values.

``DEFAULT_CONFIG`` declares ssh_host/ssh_user/ssh_port/ssh_key so that
``hermes config set terminal.ssh_host ...`` validates as a known key.  Those
declared defaults are empty, and an empty value must never reach the
environment: it would override the SSH backend's own fallbacks (port 22, the
invoking user's identity), and ``TERMINAL_SSH_PORT=''`` reaches ``int('')`` in
``_parse_env_var`` -- which kills *every* terminal command, not just SSH ones.
"""

import pytest

from hermes_cli.config import _terminal_config_value_is_bridgeable
from tools.terminal_tool import _parse_env_var

SSH_KEYS = ["ssh_host", "ssh_user", "ssh_port", "ssh_key"]


class TestSshValueBridging:
    @pytest.mark.parametrize("key", SSH_KEYS)
    def test_empty_is_not_bridged(self, key):
        assert _terminal_config_value_is_bridgeable(key, "") is False

    @pytest.mark.parametrize("key", SSH_KEYS)
    def test_whitespace_is_not_bridged(self, key):
        assert _terminal_config_value_is_bridgeable(key, "   ") is False

    def test_real_values_are_bridged(self):
        assert _terminal_config_value_is_bridgeable("ssh_host", "10.0.0.1") is True
        assert _terminal_config_value_is_bridgeable("ssh_port", 22) is True

    def test_ssh_keys_are_declared_in_defaults(self):
        """Declared schema keys are what stop `config set` warning on them."""
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        terminal = DEFAULT_CONFIG["terminal"]
        for key in SSH_KEYS:
            assert key in terminal, f"terminal.{key} missing from DEFAULT_CONFIG"


class TestParseEnvVarEmptyFallback:
    def test_empty_falls_back_to_default(self, monkeypatch):
        """int('') used to explode and take every terminal command with it."""
        monkeypatch.setenv("TERMINAL_SSH_PORT", "")
        assert _parse_env_var("TERMINAL_SSH_PORT", "22") == 22

    def test_real_value_still_wins(self, monkeypatch):
        monkeypatch.setenv("TERMINAL_SSH_PORT", "2222")
        assert _parse_env_var("TERMINAL_SSH_PORT", "22") == 2222

    def test_invalid_value_still_raises(self, monkeypatch):
        """Only *empty* is forgiven -- a typo must still be reported."""
        monkeypatch.setenv("TERMINAL_SSH_PORT", "22x")
        with pytest.raises(ValueError, match="TERMINAL_SSH_PORT"):
            _parse_env_var("TERMINAL_SSH_PORT", "22")

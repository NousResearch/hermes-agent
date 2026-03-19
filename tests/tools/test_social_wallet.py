"""Tests for social tool wallet and payment features."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from tools.social_tools import (
    social_tool,
    _get_tempo_bin,
    _get_tempo_address,
    _tempo_whoami,
)


class TestGetTempoBin:
    def test_returns_none_if_not_installed(self):
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile", return_value=False):
                assert _get_tempo_bin() is None

    def test_returns_path_if_found_in_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/tempo"):
            with patch("os.path.isfile", return_value=True):
                assert _get_tempo_bin() == "/usr/local/bin/tempo"

    def test_returns_home_path_if_exists(self):
        with patch("shutil.which", return_value=None):
            with patch("os.path.isfile") as mock_isfile:
                mock_isfile.side_effect = lambda p: ".tempo/bin/tempo" in p
                result = _get_tempo_bin()
                assert result is not None
                assert "tempo" in result


class TestTempoWhoami:
    def test_returns_error_when_cli_missing(self):
        with patch("tools.social_tools._get_tempo_bin", return_value=None):
            result = _tempo_whoami()
            assert "error" in result
            assert "not found" in result["error"].lower()

    @patch("subprocess.run")
    def test_returns_wallet_info_on_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='wallet: "0x1234567890abcdef1234567890abcdef12345678"\nbalance:\n  total: "1.5"\n  symbol: USDC',
        )
        with patch("tools.social_tools._get_tempo_bin", return_value="/usr/bin/tempo"):
            result = _tempo_whoami()
            assert "wallet" in result

    @patch("subprocess.run")
    def test_returns_error_on_failure(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr="not logged in")
        with patch("tools.social_tools._get_tempo_bin", return_value="/usr/bin/tempo"):
            result = _tempo_whoami()
            assert "error" in result


class TestWalletStatusAction:
    @patch("tools.social_tools._tempo_whoami")
    def test_returns_wallet_info(self, mock_whoami, tmp_path):
        mock_whoami.return_value = {
            "wallet": {
                "wallet": "0x1234",
                "total": "1.5",
                "available": "1.5",
                "symbol": "USDC",
                "limit": "100",
                "remaining": "99",
                "network": "tempo",
            }
        }
        config = "social:\n  enabled: true\n  relay: http://localhost"
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None
            result = json.loads(social_tool(action="wallet_status"))
            assert "wallet" in result
            assert result["wallet"]["symbol"] == "USDC"

    @patch("tools.social_tools._tempo_whoami")
    def test_returns_error_when_no_wallet(self, mock_whoami, tmp_path):
        mock_whoami.return_value = {"error": "Tempo CLI not found"}
        config = "social:\n  enabled: true\n  relay: http://localhost"
        (tmp_path / "config.yaml").write_text(config)
        with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
            import tools.social_tools as mod
            mod._config_cache = None
            result = json.loads(social_tool(action="wallet_status"))
            assert "error" in result

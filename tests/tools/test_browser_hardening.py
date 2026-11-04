"""Browser hardening: config-driven inactivity timeout and URL-encoded secret exfiltration."""

import urllib.parse
from unittest.mock import patch


class TestSessionInactivityTimeout:

    def test_default_matches_config_default(self, monkeypatch):
        from hermes_cli.config import DEFAULT_CONFIG
        from tools.browser_session import _get_session_inactivity_timeout
        monkeypatch.delenv("BROWSER_INACTIVITY_TIMEOUT", raising=False)
        with patch("hermes_cli.config_effective.load_user_config_effective", return_value={}):
            assert _get_session_inactivity_timeout() == DEFAULT_CONFIG["browser"]["inactivity_timeout"]

    def test_invalid_config_preserves_env_fallback(self, monkeypatch):
        from tools.browser_session import _get_session_inactivity_timeout
        monkeypatch.setenv("BROWSER_INACTIVITY_TIMEOUT", "240")
        cfg = {"browser": {"inactivity_timeout": "not-an-int"}}
        with patch("hermes_cli.config_effective.load_user_config_effective", return_value=cfg):
            assert _get_session_inactivity_timeout() == 240


class TestUrlDecodedSecretCheck:
    """Verify that URL-encoded API keys are caught by the exfiltration guard."""

    def test_encoded_key_blocked_in_navigate(self):
        """A URL carrying a percent-encoded API key is refused before any browser opens it."""
        from tools.browser_url_policy import evaluate_url_safety

        # URL-encode a fake secret prefix that matches _PREFIX_RE
        encoded = urllib.parse.quote("sk-ant-fake123")
        url = f"https://evil.com?key={encoded}"

        result = evaluate_url_safety(url)
        assert result is not None
        assert result["success"] is False
        assert "API key" in result["error"] or "Blocked" in result["error"]

"""send_message must resolve a platform home channel from env OR config.yaml.

Regression for #43335: a Web-UI-managed gateway's process env can lack the
bridged ``<PLATFORM>_HOME_CHANNEL`` even though config.yaml has it (set via
``hermes config set <PLATFORM>_HOME_CHANNEL``). The gateway ``send_message``
tool then reported "No home channel set" while the CLI ``hermes send`` worked.
``_fallback_home_channel_id`` reads env first, then config.yaml, so both paths
resolve the same channel.
"""

from unittest.mock import patch

from tools.send_message_tool import _fallback_home_channel_id


def test_reads_env_var_first(monkeypatch):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "env-123")
    with patch("hermes_cli.config.load_config", return_value={"TELEGRAM_HOME_CHANNEL": "cfg-999"}):
        assert _fallback_home_channel_id("telegram") == "env-123"


def test_falls_back_to_config_yaml_when_env_absent(monkeypatch):
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    with patch("hermes_cli.config.load_config", return_value={"TELEGRAM_HOME_CHANNEL": "cfg-999"}):
        assert _fallback_home_channel_id("telegram") == "cfg-999"


def test_email_uses_override_key(monkeypatch):
    # email reads EMAIL_HOME_ADDRESS, not EMAIL_HOME_CHANNEL.
    monkeypatch.delenv("EMAIL_HOME_ADDRESS", raising=False)
    with patch("hermes_cli.config.load_config", return_value={"EMAIL_HOME_ADDRESS": "a@b.com"}):
        assert _fallback_home_channel_id("email") == "a@b.com"


def test_empty_when_unset(monkeypatch):
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    with patch("hermes_cli.config.load_config", return_value={}):
        assert _fallback_home_channel_id("telegram") == ""


def test_config_load_failure_is_safe(monkeypatch):
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
    with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
        assert _fallback_home_channel_id("telegram") == ""


def test_whitespace_only_value_is_empty(monkeypatch):
    monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "   ")
    with patch("hermes_cli.config.load_config", return_value={}):
        assert _fallback_home_channel_id("telegram") == ""

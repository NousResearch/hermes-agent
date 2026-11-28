"""Unattended approval refusals must not write command secrets to logs."""
import base64
import logging
from urllib.parse import quote
from unittest.mock import patch

import pytest

from tools.approval import check_all_command_guards


@pytest.mark.parametrize("secret", [
    "Bearer SYNTH_TOKEN_SUFFIX_ABC123",
    "https://synthetic-user:synthetic-pass@example.invalid/path",
    "X-Api-Key:SYNTHETIC_API_KEY_987XYZ",
    "https://example.invalid/?api_key=SYNTHETIC_API_KEY_987XYZ",
    quote("SYNTHETIC_TOKEN_ABC123", safe=""),
    base64.b64encode(b"SYNTHETIC_TOKEN_ABC123").decode(),
])
def test_refusal_does_not_log_command_secrets(monkeypatch, caplog, secret):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    for name in ("HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE", "HERMES_CRON_SESSION"):
        monkeypatch.delenv(name, raising=False)
    with patch("tools.approval_context._get_single_query_approval_mode", return_value="approve"):
        with caplog.at_level(logging.WARNING):
            result = check_all_command_guards(f"$X -rf ~/.local {secret}", "local")
    assert result["approved"] is False
    assert "Unattended approve-mode refusal" in caplog.text
    assert secret not in caplog.text
    assert "ABC123" not in caplog.text

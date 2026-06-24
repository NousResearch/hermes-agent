"""Regression tests for browser_type display redaction."""

import json
from unittest.mock import patch

from tools.browser_tool import browser_type


def test_browser_type_never_echoes_raw_typed_text(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    typed_text = "my_secret_password_123"

    with patch(
        "tools.browser_tool._run_browser_command",
        return_value={"success": True},
    ) as mock_run:
        result = json.loads(browser_type("@password", typed_text, task_id="redaction-test"))

    assert result["success"] is True
    assert result["typed"] == "[redacted typed text]"
    assert typed_text not in json.dumps(result)
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] == ["@password", typed_text]


def test_browser_type_failure_never_echoes_raw_typed_text(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    typed_text = "my_secret_password_123"

    with patch(
        "tools.browser_tool._run_browser_command",
        return_value={
            "success": False,
            "error": f"backend failed while typing {typed_text}",
            "fallback_warning": f"chrome fallback also saw {typed_text}",
        },
    ) as mock_run:
        raw_result = browser_type("@password", typed_text, task_id="redaction-test")
        result = json.loads(raw_result)

    assert result["success"] is False
    assert typed_text not in raw_result
    assert "[redacted typed text]" in raw_result
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] == ["@password", typed_text]

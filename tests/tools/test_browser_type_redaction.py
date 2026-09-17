"""Regression tests for browser_type display redaction.

Typed text is masked in every display-facing surface: values matching a secret
pattern (API keys, tokens) keep their head/tail mask, and any other value — a
human vault credential matches no pattern (#72298) — is replaced with a fixed
``[hidden text]`` placeholder. The raw value is always sent to the browser
backend regardless.
"""

import json
from unittest.mock import patch

from tools.browser_tool import browser_type


def test_browser_type_redacts_api_key_in_output(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    secret = "sk-proj-ABCD1234567890EFGH"

    with patch(
        "tools.browser_tool_session._run_browser_command",
        return_value={"success": True},
    ) as mock_run:
        result = json.loads(browser_type("@apikey", secret, task_id="redaction-test"))

    assert result["success"] is True
    assert secret not in json.dumps(result)
    assert result["typed"] == "[hidden text]"
    # Raw secret still typed into the page.
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] == ["@apikey", secret]


def test_browser_type_hides_normal_text_in_output(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    text = "hello world search query"

    with patch(
        "tools.browser_tool_session._run_browser_command",
        return_value={"success": True},
    ) as mock_run:
        result = json.loads(browser_type("@search", text, task_id="redaction-test"))

    assert result["success"] is True
    assert result["typed"] == "[hidden text]"
    assert text not in json.dumps(result)
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] == ["@search", text]


def test_browser_type_hides_plain_human_value_in_output_and_error(monkeypatch):
    # #72298: a vault-sourced human credential matches no secret pattern, yet must
    # never reach tool results, error echoes, or chat history in cleartext.
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    typed_value = "dummy-vault-pass-42"

    with patch(
        "tools.browser_tool_session._run_browser_command",
        return_value={
            "success": False,
            "error": f"backend failed while typing {typed_value}",
            "fallback_warning": f"chrome fallback also saw {typed_value}",
        },
    ) as mock_run:
        raw_result = browser_type("@login", typed_value, task_id="redaction-test")
        result = json.loads(raw_result)

    assert result["success"] is False
    assert typed_value not in raw_result
    assert "[hidden text]" in raw_result
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] == ["@login", typed_value]


def test_browser_type_failure_redacts_api_key_in_error(monkeypatch):
    monkeypatch.delenv("CAMOFOX_URL", raising=False)
    monkeypatch.delenv("BROWSER_CDP_URL", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    secret = "sk-proj-ABCD1234567890EFGH"

    with patch(
        "tools.browser_tool_session._run_browser_command",
        return_value={
            "success": False,
            "error": f"backend failed while typing {secret}",
            "fallback_warning": f"chrome fallback also saw {secret}",
        },
    ) as mock_run:
        raw_result = browser_type("@apikey", secret, task_id="redaction-test")
        result = json.loads(raw_result)

    assert result["success"] is False
    assert secret not in raw_result
    assert "[hidden text]" in raw_result
    mock_run.assert_called_once()
    assert mock_run.call_args.args[2] == ["@apikey", secret]

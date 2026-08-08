"""cronjob script/action must tolerate non-string JSON without crashing."""

from __future__ import annotations

import json

from tools.cronjob_tools import _validate_cron_script_path, cronjob


def test_validate_script_path_rejects_non_string_without_crash():
    for bad in (42, ["collect.py"], {"path": "x"}):
        err = _validate_cron_script_path(bad)
        assert err is not None, bad
        assert "must be a string" in err


def test_validate_script_path_null_and_blank_clear_field():
    assert _validate_cron_script_path(None) is None
    assert _validate_cron_script_path("") is None
    assert _validate_cron_script_path("   ") is None


def test_cronjob_non_string_action_returns_tool_error():
    result = json.loads(cronjob(action=123))
    assert result.get("success") is False
    assert "action must be a string" in result["error"]


def test_cronjob_create_with_non_string_script_returns_tool_error(monkeypatch):
    monkeypatch.setenv("HERMES_INTERACTIVE", "1")
    result = json.loads(
        cronjob(
            action="create",
            schedule="30m",
            prompt="ping health",
            script=99,
        )
    )
    assert result.get("success") is False
    assert "must be a string" in result["error"]

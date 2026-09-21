from unittest.mock import patch

import tools.approval as approval
from tools.approval_prompt import prompt_dangerous_approval
from tools.terminal_tool import TERMINAL_SCHEMA


def test_terminal_schema_and_gateway_approval_preserve_a_sanitized_purpose(monkeypatch):
    """A model-provided purpose is user-facing context, never part of the command."""
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "purpose-test")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_tirith_scan", lambda _command: {"action": "allow", "findings": []})
    captured = {}

    def wait(_session_key, _notify, data, **_kwargs):
        captured.update(data)
        return {"choice": "deny", "resolved": True}

    monkeypatch.setattr(approval, "_await_gateway_decision", wait)
    approval.register_gateway_notify("purpose-test", lambda _data: None)
    try:
        result = approval.check_all_command_guards(
            "python -c 'print(1)'", "local",
            purpose="  Report   the   current\nloop\tstatus  ",
        )
    finally:
        approval.unregister_gateway_notify("purpose-test")

    assert TERMINAL_SCHEMA["parameters"]["properties"]["purpose"]["maxLength"] == 280
    assert result["approved"] is False
    assert captured["purpose"] == "Report the current loop status"


def test_gateway_purpose_is_redacted_and_bounded(monkeypatch):
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", "purpose-test")
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.setattr(approval.approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "_tirith_scan", lambda _command: {"action": "allow", "findings": []})
    monkeypatch.setattr("agent.redact.redact_sensitive_text", lambda text, **_kwargs: text.replace("sekret", "[REDACTED]"))
    captured = {}

    def wait(_session_key, _notify, data, **_kwargs):
        captured.update(data)
        return {"choice": "deny", "resolved": True}

    monkeypatch.setattr(approval, "_await_gateway_decision", wait)
    approval.register_gateway_notify("purpose-test", lambda _data: None)
    try:
        approval.check_all_command_guards(
            "python -c 'print(1)'", "local", purpose="sekret " + "x" * 400,
        )
    finally:
        approval.unregister_gateway_notify("purpose-test")

    assert "sekret" not in captured["purpose"]
    assert len(captured["purpose"]) <= 280


def test_cli_approval_callback_receives_the_purpose_when_it_supports_it():
    captured = {}

    def callback(_command, _description, *, purpose="", **_kwargs):
        captured["purpose"] = purpose
        return "deny"

    assert prompt_dangerous_approval("rm -rf /tmp/x", "recursive delete", approval_callback=callback,
                                    purpose="Remove the temporary verification output") == "deny"
    assert captured["purpose"] == "Remove the temporary verification output"

"""Interactive CLI Execute Code must honor the approval callback."""
from uuid import uuid4

from tools import approval as A
from tools import approval_context
from tools import terminal_tool as TT


def run_guard(monkeypatch, decision):
    for key in (
        "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK",
        "HERMES_CRON_SESSION", "HERMES_SESSION_PLATFORM",
        "HERMES_SINGLE_QUERY_SESSION", "HERMES_YOLO_MODE",
    ):
        monkeypatch.delenv(key, raising=False)

    monkeypatch.setattr(A, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(A, "_is_interactive_cli", lambda: True)
    monkeypatch.setattr(A, "_is_gateway_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_single_query_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_cron_approval_context", lambda: False)
    monkeypatch.setattr(A, "_is_unattended_platform_approval_context", lambda: False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "smart")

    token = approval_context.set_current_session_key(
        f"execute-code-cli-{uuid4().hex}"
    )
    calls = []

    def callback(command, description, **kwargs):
        calls.append((command, description))
        return decision

    TT.set_approval_callback(callback)
    try:
        result = A.check_execute_code_guard(
            "import subprocess\nsubprocess.run(['echo', 'approval-probe'])",
            "local",
        )
    finally:
        TT.set_approval_callback(None)
        approval_context.reset_current_session_key(token)

    return calls, result


def test_interactive_execute_code_deny_blocks(monkeypatch):
    calls, result = run_guard(monkeypatch, "deny")
    assert calls, "approval callback was not invoked"
    assert result.get("approved") is False
    assert result.get("outcome") == "denied"


def test_interactive_execute_code_approve_once(monkeypatch):
    calls, result = run_guard(monkeypatch, "once")
    assert calls, "approval callback was not invoked"
    assert result.get("approved") is True
    assert result.get("user_approved") is True

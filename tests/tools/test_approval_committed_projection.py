"""A failed approval projection cannot replace a decision already committed by the user."""
import pytest

from tools import approval, approval_context


@pytest.mark.parametrize("choice", ["once", "deny", None])
def test_committed_choice_survives_notification_failure(tmp_path, monkeypatch, choice):
    from hermes_cli import lifecycle

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "api_server")
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
    monkeypatch.setattr("tools.terminal_tool._get_approval_callback", lambda: None)
    events, settled, notified = [], [], []
    monkeypatch.setattr(lifecycle, "invoke_hook",
                        lambda name, **data: events.append((name, data)) or [])
    session = "projection-test"
    token = approval_context.set_current_session_key(session)

    def notify(data):
        notified.append(data["request_id"])
        assert approval.register_gateway_settle(session, data["request_id"], settled.append)
        if choice is not None:
            assert approval.resolve_gateway_approval(session, choice, request_id=data["request_id"]) == 1
        raise RuntimeError("synthetic observer projection failure")

    approval.register_gateway_notify(session, notify)
    try:
        result = approval.request_tool_approval("test_operation", "explicit human choice", rule_key="projection")
        assert result["approved"] is (choice == "once")
        responses = [data["choice"] for name, data in events if name == "post_approval_response"]
        assert responses == [choice or "notify_failed"]
        assert settled == ["resolved" if choice is not None else "notify_failed"]
        assert len(notified) == 1
        assert approval.list_gateway_approvals(session) == []
        assert approval.resolve_gateway_approval(session, "once", request_id=notified[0]) == 0
    finally:
        approval.clear_session(session)
        approval.unregister_gateway_notify(session)
        approval_context.reset_current_session_key(token)

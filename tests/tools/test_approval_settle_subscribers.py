"""Settlement is per request, not an exclusive slot owned by the last surface."""
from types import SimpleNamespace

import pytest

from tools import approval
from tools import approval_gateway_wait as wait
from gateway.run_turn_runner_approval_settle import register_timeout_notice


@pytest.mark.parametrize("outcome", ["resolved", "timeout", "interrupted", "session_closed", "notify_failed"])
@pytest.mark.parametrize("raising", [False, True])
def test_all_subscribers_settle_once_in_order(monkeypatch, outcome, raising):
    session = "settle-subscribers"
    seen = []
    request_ids = []
    monkeypatch.setattr(wait._ctx, "_fire_approval_hook", lambda *a, **kw: None)
    monkeypatch.setattr(wait._ctx, "_get_approval_timeout", lambda: 0)
    monkeypatch.setattr(wait, "is_interrupted", lambda: outcome == "interrupted")

    def first(reason):
        # Callbacks are outside the queue lock and cannot resurrect a finished request.
        assert approval.list_gateway_approvals(session) == []
        assert not approval.register_gateway_settle(session, request_ids[0], seen.append)
        seen.append(("first", reason))
        if raising:
            raise RuntimeError("broken observer")

    def notify(data):
        rid = data["request_id"]
        request_ids.append(rid)
        assert approval.register_gateway_settle(session, rid, first)
        assert approval.register_gateway_settle(session, rid, lambda r: seen.append(("second", r)))
        if outcome == "resolved":
            assert approval.resolve_gateway_approval(session, "once", request_id=rid) == 1
        elif outcome == "session_closed":
            assert approval.withdraw_gateway_approval(session, rid, "client gone")
        elif outcome == "notify_failed":
            raise RuntimeError("delivery failed")

    # Zero timeout would otherwise beat an already-signalled event in the poll loop.
    if outcome in {"resolved", "session_closed"}:
        monkeypatch.setattr(wait._ctx, "_get_approval_timeout", lambda: 5)
    result = wait._await_gateway_decision(session, notify, {"command": "fixture", "pattern_key": "fixture"})
    assert seen == [("first", outcome), ("second", outcome)]
    assert approval.list_gateway_approvals(session) == []
    assert approval.resolve_gateway_approval(session, "once") == 0
    if outcome == "resolved":
        assert result["choice"] == "once"
    elif outcome == "timeout":
        assert result["resolved"] is False


def test_runner_timeout_notice_does_not_replace_adapter_hook(monkeypatch):
    session = "adapter-and-runner"
    seen = []
    monkeypatch.setattr(wait._ctx, "_fire_approval_hook", lambda *a, **kw: None)
    monkeypatch.setattr(wait._ctx, "_get_approval_timeout", lambda: 5)
    runner = SimpleNamespace(_ctx=SimpleNamespace(session_key=session))

    def notify(data):
        assert approval.register_gateway_settle(session, data["request_id"], seen.append)
        register_timeout_notice(runner, data, command="fixture", card_message_id="card")
        assert approval.resolve_gateway_approval(session, "once", request_id=data["request_id"]) == 1

    result = wait._await_gateway_decision(session, notify, {"command": "fixture", "pattern_key": "fixture"})
    assert result["choice"] == "once"
    assert seen == ["resolved"]


def test_missing_request_does_not_register():
    assert not approval.register_gateway_settle("absent-session", "absent-request", lambda reason: None)

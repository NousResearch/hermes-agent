"""Policy hooks can refuse a stored session before it becomes live."""

from types import SimpleNamespace

from hermes_cli import lifecycle
from tui_gateway import server


def test_pre_session_resume_refusal_sees_the_stored_row(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda name, **kwargs: seen.update(name=name, **kwargs) or [{"action": "block", "message": "wrong kind"}],
    )
    ctx = SimpleNamespace(
        db=SimpleNamespace(assert_resume_safe=lambda *_a, **_kw: None),
        target="stored-session",
        found={"source": "desktop", "platform": "desktop", "metadata": {"kind": "private"}},
        lazy=False,
        omit_messages=False,
        defer_history=False,
        eager_build=False,
        params={"session_id": "stored-session", "source": "tui"},
        rid="request-id",
    )

    response = server._resume_guard(ctx)

    assert response == {
        "jsonrpc": "2.0", "id": "request-id", "error": {"code": 4031, "message": "wrong kind"}}
    assert seen["name"] == "pre_session_resume"
    assert seen["session"]["metadata"] == {"kind": "private"}


def test_session_close_passes_a_host_end_reason_to_teardown(monkeypatch):
    seen = {}
    session = {"_end_msgs": []}
    monkeypatch.setattr(server, "_pop_session_by_id", lambda _sid: session)
    monkeypatch.setattr(
        server,
        "_teardown_popped_session",
        lambda got, *, end_reason: seen.update(session=got, end_reason=end_reason) or True,
    )

    response = server._methods["session.close"]("request-id", {
        "session_id": "runtime-id", "end_reason": "policy_refusal"})

    assert response["result"]["closed"] is True
    assert seen == {"session": session, "end_reason": "policy_refusal"}


def test_session_close_rejects_an_invalid_host_end_reason(monkeypatch):
    monkeypatch.setattr(server, "_pop_session_by_id", lambda _sid: (_ for _ in ()).throw(AssertionError()))

    response = server._methods["session.close"]("request-id", {
        "session_id": "runtime-id", "end_reason": "operator stop"})

    assert response == {
        "jsonrpc": "2.0", "id": "request-id",
        "error": {"code": 4004, "message": "end_reason must be a lowercase snake_case identifier"},
    }

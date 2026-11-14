"""``pre_approval_bypass``: a plugin decides who may switch approvals off. Every TUI-gateway path that
turns a bypass on asks it before anything is written (``config.set yolo`` in every scope,
``config.set approvals.mode off``, ``slash.exec /yolo``), a resumed chat's stored bypass is re-armed
only while it allows, and switching approvals back on is never asked."""

import types

import pytest

from tools import approval
from tools.approval_yolo import BYPASS_REFUSED, approval_bypass_refusal
from tui_gateway import server

SENTENCE = "Only an admin can turn off approvals here."


@pytest.fixture
def hook(monkeypatch):
    """Records each ask; refuses while ``answer["block"]`` is set."""
    calls, answer = [], {"block": True}

    def invoke_hook(name, **kwargs):
        if name != "pre_approval_bypass":
            return []
        calls.append(kwargs)
        return [{"action": "block", "message": SENTENCE}] if answer["block"] else [None]
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    yield calls, answer
    server._sessions.clear()
    approval.clear_session("row-a")


def _rpc(method, **params):
    return server.handle_request({"id": "1", "method": method, "params": params})


def test_the_first_block_refuses_with_its_sentence_and_sees_only_the_login(monkeypatch):
    seen = []

    def invoke_hook(name, **kwargs):
        seen.append(kwargs)
        return [None, {"action": "allow"}, {"action": "block", "message": f"  {SENTENCE}  "}]
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    login = {"user_id": "u-1", "provider": "oidc", "token": "never-forwarded"}
    assert approval_bypass_refusal(surface="rpc", session_key="k", identity=login) == SENTENCE
    assert seen == [{"surface": "rpc", "scope": "session", "session_key": "k", "platform": "", "user_id": "",
                     "identity": {"user_id": "u-1", "provider": "oidc"}, "session_owner": ""}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [{"action": "block"}])
    assert approval_bypass_refusal(surface="rpc") == BYPASS_REFUSED
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kwargs: [])
    assert approval_bypass_refusal(surface="rpc") is None


def test_config_set_yolo_is_refused_before_anything_changes_and_turning_it_off_is_never_asked(hook, monkeypatch):
    calls, answer = hook
    writes = []
    monkeypatch.setattr(server, "_session_db", lambda session: _Recorder(writes))
    server._sessions["sid"] = {"agent": types.SimpleNamespace(), "session_key": "row-a"}

    refused = _rpc("config.set", session_id="sid", key="yolo")
    assert refused["error"] == {"code": 4030, "message": SENTENCE}
    assert not approval.is_session_yolo_enabled("row-a") and writes == []
    assert calls[-1]["surface"] == "rpc" and calls[-1]["scope"] == "session" and calls[-1]["session_key"] == "row-a"

    process = _rpc("config.set", key="yolo", value="on")  # sessionless: the process-wide HERMES_YOLO_MODE
    assert process["error"]["code"] == 4030 and calls[-1]["scope"] == "process"
    assert "HERMES_YOLO_MODE" not in __import__("os").environ

    answer["block"] = False
    assert _rpc("config.set", session_id="sid", key="yolo")["result"]["value"] == "1"
    answer["block"] = True
    asked = len(calls)
    assert _rpc("config.set", session_id="sid", key="yolo")["result"]["value"] == "0"  # back on: never asked
    assert len(calls) == asked and writes == [True, False]


def test_approvals_mode_off_and_slash_yolo_are_refused_before_they_run(hook, monkeypatch):
    calls, _answer = hook
    dispatched = []
    monkeypatch.setattr(server, "_live_slash_command_output", lambda *args: dispatched.append(args))
    server._sessions["sid"] = {"agent": types.SimpleNamespace(), "session_key": "row-a"}

    for key in ("approvals.mode", "approval_mode"):
        answer = _rpc("config.set", key=key, value="off")
        assert answer["error"] == {"code": 4030, "message": SENTENCE} and calls[-1]["scope"] == "global"
    assert _rpc("slash.exec", session_id="sid", command="/yolo")["error"] == {"code": 4030, "message": SENTENCE}
    assert dispatched == [] and calls[-1]["session_key"] == "row-a"


def test_a_resumed_chat_re_arms_its_stored_bypass_only_while_the_hook_allows(hook, monkeypatch, tmp_path):
    from hermes_state import SessionDB
    calls, answer = hook
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    try:
        db.create_session("row-a", source="tui", user_id="member-7", model_config={"yolo_mode": True})
        resumed = _rpc("session.resume", session_id="row-a")
        assert resumed["result"]["session_key"] == "row-a"
        assert not approval.is_session_yolo_enabled("row-a")
        assert calls[-1]["surface"] == "restore" and calls[-1]["session_owner"] == "member-7"

        server._sessions.clear()
        answer["block"] = False
        _rpc("session.resume", session_id="row-a")
        assert approval.is_session_yolo_enabled("row-a")
    finally:
        db.close()


class _Recorder:
    def __init__(self, writes):
        self.writes = writes

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def set_session_yolo(self, _row, on):
        self.writes.append(on)


def test_a_shell_hook_refuses_in_either_dialect_and_context_is_no_opinion():
    from agent.shell_hooks import _parse_response
    assert _parse_response("pre_approval_bypass", '{"action": "block", "message": "no"}') == {
        "action": "block", "message": "no"}
    assert _parse_response("pre_approval_bypass", '{"decision": "block", "reason": "no"}') == {
        "action": "block", "message": "no"}
    assert _parse_response("pre_approval_bypass", '{"context": "allow it"}') is None

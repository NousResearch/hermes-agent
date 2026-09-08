"""Shared RPC contracts exercised against real registries and transcript I/O."""

import json
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def runtime(monkeypatch):
    from tui_gateway import server
    from tools import async_delegation, delegate_tool_registry

    transport = SimpleNamespace(write=lambda frame: True)
    owner = {"session_key": "parent", "history": [], "transport": transport}
    monkeypatch.setattr(server, "_sessions", {"ui-owner": owner})
    monkeypatch.setattr(delegate_tool_registry, "_active_subagents", {})
    monkeypatch.setattr(delegate_tool_registry, "_recent_subagents", {})
    monkeypatch.setattr(async_delegation, "_records", {})

    def call(method, *, via=transport, **params):
        return server.dispatch({"id": 1, "method": method,
                                "params": {"session_id": "ui-owner", **params}}, transport=via)

    return server, owner, transport, call


def test_snapshot_projects_only_this_sessions_runtime_records(runtime):
    from tools import async_delegation as bg
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import _unregister_subagent
    from tools.delegate_tool_progress import _build_child_progress_callback

    server, owner, transport, call = runtime
    release = threading.Event()
    finished = threading.Event()

    def run():
        try:
            assert release.wait(10)
            return {"results": []}
        finally:
            finished.set()

    dispatch = bg.dispatch_async_delegation_batch(
        goals=["owned task"], context="private handoff", toolsets=None, role="leaf", model="test",
        session_key="parent", origin_ui_session_id="ui-owner", runner=run)
    did = dispatch["delegation_id"]
    child = SimpleNamespace(_subagent_id="child", _delegate_depth=1, _delegation_id=did, model="test")
    _register_child(child, None, "owned task", owner_session_id="ui-owner",
                         owner_transport=transport, owner_session_record=owner)
    foreign = SimpleNamespace(_subagent_id="foreign", _delegate_depth=1, model="test")
    _register_child(foreign, None, "foreign secret", owner_session_id="other",
                         owner_transport=transport, owner_session_record={})
    progress = _build_child_progress_callback(0, "owned task",
        SimpleNamespace(tool_progress_callback=lambda *a, **kw: None), subagent_id="child")
    progress("tool.started", "read_file")
    progress("tool.completed", "read_file")
    try:
        snapshot = call("subagent.list")["result"]
        assert [s["subagent_id"] for s in snapshot["subagents"]] == ["child"]
        assert snapshot["subagents"][0]["last_tool"] == "read_file"
        assert snapshot["delegations"] == []
        assert snapshot["subagents"][0]["tool_count"] == 1
        wire = json.dumps(snapshot)
        assert "private handoff" not in wire and "foreign secret" not in wire
        assert "owner_transport" not in wire and "session_key" not in wire
        assert "error" in call("subagent.list", via=SimpleNamespace(write=lambda frame: True))
        assert "error" in call("subagent.list", session_id="missing")
        server._sessions["ui-owner"] = {**owner}
        assert call("subagent.list")["result"]["subagents"] == []
    finally:
        _unregister_subagent("child")
        _unregister_subagent("foreign")
        release.set()
        assert finished.wait(10)


def test_live_tail_and_steer_share_exact_owner_and_end_with_child(runtime):
    from run_agent import AIAgent
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import _close_subagent_steering, _unregister_subagent
    from tools.delegation_live_log import LiveTranscriptWriter

    server, owner, transport, call = runtime
    child = object.__new__(AIAgent)
    child._subagent_id = "child"
    child._delegate_depth = 1
    child.model = "test"
    child._pending_steer = None
    child._pending_steer_lock = threading.Lock()
    writer = LiveTranscriptWriter("deleg-rpc", 0, "owned task")
    child._live_transcript_path = str(writer.path)
    _register_child(child, None, "owned task", owner_session_id="ui-owner",
                         owner_transport=transport, owner_session_record=owner)
    try:
        writer.event("tool", "x" * 20000)
        writer.tool_result("read_file", "first result")
        tail = call("subagent.tail", subagent_id="child")["result"]
        assert tail["available"] and tail["truncated"] and len(tail["text"].encode()) <= 16384
        assert "first result" in tail["text"]
        writer.tool_result("read_file", "new live output")
        assert "new live output" in call("subagent.tail", subagent_id="child")["result"]["text"]
        queued = call("subagent.steer", subagent_id="child", text="change course")["result"]
        assert queued["status"] == "queued" and "delivered" not in queued
        assert _close_subagent_steering("child", child) == "change course"
        assert call("subagent.steer", subagent_id="child", text="too late")["result"]["status"] == "rejected"
        assert "error" in call("subagent.tail", subagent_id="child", via=SimpleNamespace(write=lambda frame: True))
        server._sessions["ui-owner"] = {**owner}
        assert not call("subagent.tail", subagent_id="child")["result"]["available"]
        server._sessions["ui-owner"] = owner
        _unregister_subagent("child")
        assert call("subagent.tail", subagent_id="child")["result"] == {
            "subagent_id": "child", "available": False, "text": "", "truncated": False}
    finally:
        _unregister_subagent("child")



def test_interrupt_requires_exact_live_owner_but_direct_helper_stays_legacy(runtime):
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import interrupt_subagent, _unregister_subagent

    server, owner, transport, call = runtime
    stopped = []
    child = SimpleNamespace(_subagent_id="child", _delegate_depth=1, model="test",
                            hard_interrupt=lambda message: stopped.append(message))
    _register_child(child, None, "owned", owner_session_id="ui-owner",
                    owner_transport=transport, owner_session_record=owner)
    try:
        for params in ({"session_id": ""}, {"session_id": "missing"},
                       {"via": SimpleNamespace(write=lambda frame: True)}):
            reply = call("subagent.interrupt", subagent_id="child", **params)
            assert "error" in reply or not reply["result"]["found"]
            assert stopped == []
        server._sessions["foreign"] = {**owner}
        assert not call("subagent.interrupt", session_id="foreign", subagent_id="child")["result"]["found"]
        server._sessions["ui-owner"] = {**owner}
        assert not call("subagent.interrupt", subagent_id="child")["result"]["found"]
        assert stopped == []
        server._sessions["ui-owner"] = owner
        assert call("subagent.interrupt", subagent_id="child")["result"]["found"]
        assert len(stopped) == 1
        assert interrupt_subagent("child")
        assert len(stopped) == 2
        _unregister_subagent("child")
        assert not call("subagent.interrupt", subagent_id="child")["result"]["found"]
    finally:
        _unregister_subagent("child")



def test_reattach_preserves_child_controls_including_late_registration(runtime, tmp_path):
    from tools.delegate_tool_child_run import _register_child

    server, owner, old, call = runtime
    new = type("Transport", (), {"write": lambda self, frame: True})()
    transcript = tmp_path / "child.txt"
    transcript.write_text("live child output")
    steered, stopped = [], []

    def register(sid):
        child = SimpleNamespace(_subagent_id=sid, _delegate_depth=1, model="test",
                                _live_transcript_path=str(transcript),
                                steer=lambda text: steered.append(text) or True,
                                hard_interrupt=lambda text: stopped.append(text))
        _register_child(child, None, "owned", owner_session_id="ui-owner",
                        owner_transport=old, owner_session_record=owner)

    register("before")
    owner["transport"] = server._detached_ws_transport
    owner["history_lock"] = threading.Lock()
    with server._session_resume_lock, owner["history_lock"]:
        assert server._reattach_refusal(1, "ui-owner", owner) is None
        server._rebind_live_transport("ui-owner", owner, new)
    # A dispatch captured before reload may not construct its child until afterwards.
    register("after")
    assert {row["subagent_id"] for row in call("subagent.list", via=new)["result"]["subagents"]} == {"before", "after"}
    # Closing a second authenticated viewer hands control back to the survivor.
    popup = type(new)()
    with server._session_resume_lock, owner["history_lock"]:
        server._rebind_live_transport("ui-owner", owner, popup)
    for peer in (new, popup):
        assert {r["subagent_id"] for r in call("subagent.list", via=peer)["result"]["subagents"]} == {"before", "after"}
        assert call("subagent.tail", via=peer, subagent_id="before")["result"]["text"] == "live child output"
    assert server._close_sessions_for_transport(popup) == (0, 0)
    assert server._session_transport_contains(owner, new)
    assert not server._session_transport_contains(owner, popup)
    assert {row["subagent_id"] for row in call("subagent.list", via=new)["result"]["subagents"]} == {"before", "after"}
    for sid in ("before", "after"):
        assert call("subagent.tail", via=new, subagent_id=sid)["result"]["text"] == "live child output"
        assert call("subagent.steer", via=new, subagent_id=sid, text=sid)["result"]["status"] == "queued"
        assert call("subagent.interrupt", via=new, subagent_id=sid)["result"]["found"]
    assert steered == ["before", "after"] and len(stopped) == 2
    for method in ("list", "tail", "steer", "interrupt"):
        denied = call("subagent." + method, subagent_id="before", text="old")
        assert "error" in denied or denied["result"].get("status") == "rejected"
    assert steered == ["before", "after"] and len(stopped) == 2


def test_reattach_does_not_adopt_foreign_or_retired_generations(runtime):
    from tools.delegate_tool_child_run import _register_child

    server, owner, old, call = runtime
    new = type("Transport", (), {"write": lambda self, frame: True})()
    effects = []
    for sid, session_id, record in (("foreign", "other", owner),
                                     ("retired", "ui-owner", {**owner})):
        child = SimpleNamespace(_subagent_id=sid, _delegate_depth=1, model="test",
                                steer=lambda text: effects.append(text) or True,
                                hard_interrupt=lambda text: effects.append(text))
        _register_child(child, None, "private", owner_session_id=session_id,
                        owner_transport=old, owner_session_record=record)
    with server._session_resume_lock:
        assert server._reattach_refusal(1, "ui-owner", {**owner})["error"]["code"] == 4007
        owner["_client_gone_interrupt_requested"] = True
        assert server._reattach_refusal(1, "ui-owner", owner)["error"]["code"] == 4009
        del owner["_client_gone_interrupt_requested"]
        server._rebind_live_transport("ui-owner", owner, new)
    assert call("subagent.list", via=new)["result"]["subagents"] == []
    for sid in ("foreign", "retired"):
        assert not call("subagent.tail", via=new, subagent_id=sid)["result"]["available"]
        assert call("subagent.steer", via=new, subagent_id=sid, text="deny")["result"]["status"] == "rejected"
        assert not call("subagent.interrupt", via=new, subagent_id=sid)["result"]["found"]
    assert effects == []

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
    try:
        snapshot = call("subagent.list")["result"]
        assert [s["subagent_id"] for s in snapshot["subagents"]] == ["child"]
        assert snapshot["delegations"][0]["delegation_id"] == did
        assert snapshot["delegations"][0]["subagent_ids"] == ["child"]
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

"""Real JSON-RPC routing and real compute-host pipe ownership recovery."""
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from tui_gateway import server
from tui_gateway.host_supervisor import HostSupervisor
from tools import worker_roster
from tools.delegate_tool_child_run import _register_child, _ChildRun
from tools.delegate_tool_registry import _unregister_subagent


def make_session(home, transport, key="parent"):
    from hermes_state import SessionDB
    home.mkdir(exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(key, source="desktop", model="test")
    db.close()
    return dict(profile_home=str(home), session_key=key, transport=transport, history=[])


def call(transport, sid="ui"):
    result = server.dispatch({"id": 1, "method": "subagent.snapshot", "params": {"session_id": sid}}, transport=transport)
    assert "error" not in result, result
    return result["result"]


def test_reopen_terminal_and_profile_scope(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    transport = SimpleNamespace(write=lambda _: True)
    a, b = tmp_path / "a", tmp_path / "b"
    owner = make_session(a, transport)
    other = make_session(b, transport)
    monkeypatch.setattr(server, "_sessions", {"ui": owner, "other": other})
    monkeypatch.setattr(server, "_compute_host_supervisor", None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _: False)
    child = SimpleNamespace(_subagent_id="same", _delegate_depth=1, _parent_session_id="parent", model="test")
    token = set_hermes_home_override(a)
    try:
        _register_child(child, None, "owned", owner_session_id="ui", owner_transport=transport, owner_session_record=owner)
        running = call(transport)
        assert running["workers"][0]["status"] == "running"
        from tools.delegate_tool_registry import _active_subagents, _active_subagents_lock
        with _active_subagents_lock:
            _active_subagents["same"]["status"] = "waiting"
        assert call(transport)["workers"][0]["status"] == "waiting"
        with _active_subagents_lock:
            _active_subagents["same"]["status"] = "running"
        assert call(transport, "other")["workers"] == []
        # New socket/runtime record after cold reopen; durable identity survives.
        replacement = SimpleNamespace(write=lambda _: True)
        server._sessions["ui"] = {**owner, "transport": replacement}
        assert call(replacement)["workers"][0]["run_id"] == running["workers"][0]["run_id"]
        rejected = server.dispatch({"id": 2, "method": "subagent.snapshot", "params": {"session_id": "ui"}}, transport=transport)
        assert rejected["error"]["code"] == 4001
        # The real child completion hook, no client connected and no progress callback.
        run = object.__new__(_ChildRun)
        run.subagent_id, run.child = "same", child
        run.record_terminal("completed")
        _unregister_subagent("same", agent=child)
        terminal = call(replacement)
        assert terminal["workers"][0]["status"] == "completed"
        assert terminal["workers"][0]["version"] > running["workers"][0]["version"]
        assert terminal["snapshot_seq"] > running["snapshot_seq"]
        assert terminal["coverage"] == "admitted_since_upgrade"
        from hermes_state import SessionDB
        db = SessionDB(db_path=a / "state.db")
        db.end_session("parent", "compression")
        db.create_session("tip", source="desktop", parent_session_id="parent")
        db.create_session("fork", source="desktop", parent_session_id="parent", model_config={"_branched_from": "parent"})
        db.close()
        server._sessions["ui"]["session_key"] = "tip"
        assert call(replacement)["workers"][0]["run_id"] == running["workers"][0]["run_id"]
        server._sessions["ui"]["session_key"] = "fork"
        assert call(replacement)["workers"] == []
    finally:
        _unregister_subagent("same", agent=child)
        reset_hermes_home_override(token)


# A separate interpreter uses the PRODUCTION ComputeHost frame dispatcher and
# HostSupervisor pipe/reply code. Only LLM turn execution is replaced by a
# deterministic registered child, so this needs neither provider nor live host.
HOST = '''
import json, sys
wire_stdout = sys.stdout
from types import SimpleNamespace
from tui_gateway.compute_host import ComputeHost
from tools.delegate_tool_child_run import _register_child, _ChildRun
from tools.delegate_tool_registry import _unregister_subagent
from hermes_constants import get_hermes_home
from tools import worker_roster as roster
root = SimpleNamespace(session_id="parent")
queued = roster.admit(root, [{"goal":"queued native task"}], None)[0]
child = SimpleNamespace(session_id="child-session", _subagent_id="pipe-child", _delegate_depth=1, _parent_session_id="parent", model="test")
_register_child(child, None, "pipe task", owner_session_id=None, owner_transport=None, owner_session_record=None)
host = ComputeHost(stdout=wire_stdout, heartbeat_secs=0)
print(json.dumps({"type":"hello"}), file=wire_stdout, flush=True)
for line in sys.stdin:
    frame = json.loads(line)
    if frame["type"] == "test.nest":
        nested = roster.admit(child, [{"goal":"nested native task"}], None)[0]
        wait_scope = roster.waiting_for_children(child)
        wait_scope.__enter__()
        host.emit({"type":"workers.ack", "request_id":frame["request_id"]})
    elif frame["type"] == "test.finish":
        wait_scope.__exit__(None, None, None)
        roster.finish(queued, "cancelled")
        roster.finish(nested, "completed")
        run = object.__new__(_ChildRun)
        run.subagent_id, run.child = "pipe-child", child
        run.record_terminal("completed")
        _unregister_subagent("pipe-child", agent=child)
        host.emit({"type":"workers.ack", "request_id":frame["request_id"]})
    else:
        host.handle_frame(frame)
'''


def test_compute_pipe_disconnect_restart_and_cross_profile(tmp_path, monkeypatch):
    import os
    home = tmp_path / "host"
    transport = SimpleNamespace(write=lambda _: True)
    owner = make_session(home, transport)
    other = make_session(tmp_path / "other", transport)
    monkeypatch.setattr(server, "_sessions", {"ui": owner, "other": other})
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda _: True)
    supervisor = HostSupervisor(rpc_sink=lambda _: True, registry_path=tmp_path / "host.json")
    # Use its real pipe drain and request correlation, without launching real agents.
    proc = subprocess.Popen([sys.executable, "-u", "-c", HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env={**os.environ, "HERMES_HOME": str(home)})
    supervisor._proc = proc
    reader = threading.Thread(target=supervisor._drain_stdout, args=(proc,), daemon=True)
    reader.start()
    stderr_reader = threading.Thread(target=supervisor._drain_stderr, args=(proc,), daemon=True)
    stderr_reader.start()
    monkeypatch.setattr(server, "_compute_host_supervisor", supervisor)
    try:
        assert supervisor._hello_event.wait(15)
        probe = supervisor.worker_observations(str(home.resolve()), ["parent"])
        assert probe.get("observations"), probe
        running = call(transport)
        assert {w["status"] for w in running["workers"]} == {"queued", "running"}
        assert all(w["owner_id"] != worker_roster.OWNER_ID for w in running["workers"])
        supervisor._await_reply({"type": "test.nest", "request_id": "nest"}, "nest", 10)
        nested_rows = call(transport)["workers"]
        assert sorted(w["status"] for w in nested_rows) == ["queued", "queued", "waiting"]
        assert any(w.get("parent_run_id") for w in nested_rows)
        assert call(transport, "other")["workers"] == []
        replacement = SimpleNamespace(write=lambda _: True)
        server._sessions["ui"] = {**owner, "transport": replacement}
        assert len(call(replacement)["workers"]) == 3
        supervisor._await_reply({"type": "test.finish", "request_id": "finish"}, "finish", 10)
        assert sorted(w["status"] for w in call(replacement)["workers"]) == ["cancelled", "completed", "completed"]
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        reader.join(timeout=10)
    # A new process creates a different run with the same public ID. Kill it
    # without cleanup; a persisted running row is unknown, not a false spinner.
    proc2 = subprocess.Popen([sys.executable, "-u", "-c", HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env={**os.environ, "HERMES_HOME": str(home)})
    try:
        assert json.loads(proc2.stdout.readline())["type"] == "hello"
    finally:
        proc2.kill()
        proc2.wait(timeout=10)
    stale = call(replacement)
    assert not stale["owner_available"]
    assert sorted(w["status"] for w in stale["workers"]) == ["cancelled", "completed", "completed", "unknown", "unknown"]
    assert len({w["run_id"] for w in stale["workers"]}) == 5
    assert stale["coverage"] == "admitted_since_upgrade"

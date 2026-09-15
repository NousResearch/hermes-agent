"""Connector operation RPCs and events over the real dispatch path.

- ``connectors.operation.status`` reads the live operation with the same ownership checks as
  ``connectors.list`` (4000 / 4001); ``deadline_at`` is the server's, never recomputed
- ``connection.respond`` finds the op by ``op_id`` and drives target transitions; a foreign
  transport cannot
- every target transition and the settlement emit ``connection.update`` to the session
- ``pending_connection`` on resume comes from the live registry
"""

import json
import os
import threading
import time
from contextlib import ExitStack, suppress

import pytest

from tools.connectors import live, mcp
from tools.connectors.contract import Actor, TargetState
from tools.connectors.operation import ConnectionOperation, Target
from tui_gateway import server
from tui_gateway.transport import StdioTransport


class PipeClient:
    def __init__(self, stack):
        read_fd, write_fd = os.pipe()
        self.reader = stack.enter_context(os.fdopen(read_fd, "r", encoding="utf-8"))
        self.writer = os.fdopen(write_fd, "w", encoding="utf-8")
        stack.callback(lambda: suppress(BrokenPipeError) and self.writer.close())
        self.transport = StdioTransport(lambda: self.writer, threading.Lock())
        self.frames = []
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.reader:
            self.frames.append(json.loads(line))

    def write(self, obj):
        return self.transport.write(obj)

    def close(self):
        pass

    def events(self, kind, *, expect=1):
        deadline = time.time() + 2
        while time.time() < deadline:
            found = [f["params"] for f in list(self.frames) if f.get("method") == "event" and f["params"]["type"] == kind]
            if len(found) >= expect:
                return found
            time.sleep(0.01)
        return [f["params"] for f in list(self.frames) if f.get("method") == "event" and f["params"]["type"] == kind]


SID = "op-rpc-session"


@pytest.fixture
def owned(monkeypatch):
    live.reset_for_tests()
    with ExitStack() as stack:
        owner, stranger = PipeClient(stack), PipeClient(stack)
        session = dict(transport=owner.transport, agent=None, session_key=SID, history=[],
                       history_lock=threading.Lock(), history_version=0, running=False, attached_images=[],
                       source="desktop")
        monkeypatch.setitem(server._sessions, SID, session)
        yield owner, stranger, session
    live.reset_for_tests()


def _rpc(client, method, **params):
    return server.dispatch({"jsonrpc": "2.0", "id": 7, "method": method, "params": {"session_id": SID, **params}},
                           client.transport)


def _connect_rpc(client, **params):
    """``connectors.connect`` runs on the long-handler pool and writes its reply to the transport."""
    before = len(client.frames)
    _rpc(client, "connectors.connect", **params)
    deadline = time.time() + 2
    while time.time() < deadline:
        replies = [f for f in list(client.frames)[before:] if f.get("id") == 7]
        if replies:
            return replies[-1]
        time.sleep(0.01)
    raise AssertionError("no reply")


def _open_op():
    operation = ConnectionOperation([Target("gmail", "connector", "connect"), Target("notion", "connector", "connect")],
                                    session_key=SID)
    live.open(operation)
    return operation


def test_operation_status_returns_the_live_snapshot_with_the_server_deadline(owned):
    owner, _, _ = owned
    operation = _open_op()
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/gmail")
    reply = _rpc(owner, "connectors.operation.status", op_id=operation.op_id)
    result = reply["result"]
    assert result["op_id"] == operation.op_id
    assert result["deadline_at"] == operation.deadline_at
    assert result["settled"] is False and result["settled_by"] is None
    by = {t["name"]: t for t in result["targets"]}
    assert by["gmail"]["state"] == "initiated" and by["gmail"]["connect_url"] == "https://l/gmail"
    assert by["notion"]["state"] == "pending"


def test_operation_status_is_owner_only_and_validates_params(owned):
    owner, stranger, _ = owned
    operation = _open_op()
    assert _rpc(stranger, "connectors.operation.status", op_id=operation.op_id)["error"]["code"] == 4001
    assert _rpc(owner, "connectors.operation.status")["error"]["code"] == 4000
    assert _rpc(owner, "connectors.operation.status", op_id="nope")["error"]["code"] == 4004


def test_respond_drives_the_live_operation_and_emits_update(owned):
    owner, stranger, _ = owned
    operation = _open_op()
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher)
    foreign = _rpc(stranger, "connection.respond", op_id=operation.op_id,
                   result={"targets": [{"name": "notion", "status": "skipped"}]})
    assert foreign["error"]["code"] == 4001
    assert operation.target("notion").state == TargetState.pending

    reply = _rpc(owner, "connection.respond", op_id=operation.op_id,
                 result={"targets": [{"name": "notion", "status": "skipped"}]})
    assert "result" in reply, reply
    assert operation.target("notion").state == TargetState.skipped
    assert operation.wake.is_set()
    updates = owner.events("connection.update", expect=2)
    assert updates and updates[-1]["payload"]["target"] == "notion"
    assert updates[-1]["payload"]["to"] == "skipped" and updates[-1]["payload"]["actor"] == "user"
    assert updates[-1]["payload"]["op_id"] == operation.op_id


def test_respond_applies_the_answer_under_the_session_s_profile(owned, monkeypatch, tmp_path):
    """A named-profile session's enable must flip the flag in that profile's config.yaml. The RPC
    thread carries no profile of its own, so the handler binds the session record's before the
    answer runs the backend's work."""
    import yaml
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    owner, _, session = owned
    home = tmp_path / "home"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for path in (home, profile):
        (path / "config.yaml").write_text(yaml.safe_dump({"mcp_servers": {"paper": {"command": "paper-mcp", "enabled": False}}}))
    session["profile_home"] = str(profile)

    operation = ConnectionOperation([Target("paper", "mcp", "enable")], session_key=SID)
    token = set_hermes_home_override(profile)
    try:
        live.open(operation)  # the tool thread opens it under the turn's profile
    finally:
        reset_hermes_home_override(token)
    runner = mcp.open_runner("enable")  # the real catalog backend: the write is the point
    runner.prepare(operation)
    try:
        reply = _rpc(owner, "connection.respond", op_id=operation.op_id,
                     result={"targets": [{"name": "paper", "status": "approved"}]})
    finally:
        runner.close()

    assert "result" in reply, reply
    assert yaml.safe_load((profile / "config.yaml").read_text())["mcp_servers"]["paper"]["enabled"] is True
    assert yaml.safe_load((home / "config.yaml").read_text())["mcp_servers"]["paper"]["enabled"] is False


def test_respond_cannot_claim_an_outcome_for_any_target(owned):
    """The card renders the operation; only skip, approve and Continue are its to say. The contract
    refuses any other claim (4002) before it reaches the operation."""
    owner, _, _ = owned
    operation = _open_op()
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher)
    reply = _rpc(owner, "connection.respond", op_id=operation.op_id,
                 result={"targets": [{"name": "gmail", "status": "connected"},
                                     {"name": "notion", "status": "failed"}]})
    assert reply["error"]["code"] == 4002, reply
    assert operation.target("gmail").state == TargetState.initiated
    assert operation.target("notion").state == TargetState.pending


def test_respond_continue_settles_and_emits_the_settlement_update(owned):
    owner, _, _ = owned
    operation = _open_op()
    reply = _rpc(owner, "connection.respond", op_id=operation.op_id, result={"settled_by": "continue"})
    assert "result" in reply
    assert operation.settled and operation.settled_by.value == "continue"
    settled = [u for u in owner.events("connection.update") if u["payload"].get("settled")]
    assert settled and settled[-1]["payload"]["settled_by"] == "continue"


def test_pending_connection_on_resume_comes_from_the_live_registry(owned):
    operation = _open_op()
    payload = server._pending_connection_request_payload(SID)
    assert payload["op_id"] == operation.op_id
    assert payload["deadline_at"] == operation.deadline_at
    live.close(operation)
    assert server._pending_connection_request_payload(SID) is None


def test_panel_connect_reissues_only_a_dead_link(owned, monkeypatch):
    """Try again re-mints a failed or expired target; a waiting target keeps the link it was minted
    with (the card reopens it), so the RPC refuses rather than spend a second mint."""
    owner, _, _ = owned
    operation = _open_op()
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/gmail/1")
    operation.transition("notion", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/notion/1")
    operation.transition("notion", TargetState.failed, Actor.backend_watcher, detail="vendor: nope")
    mints = []

    class Client:
        def connections(self, names, *, reinitiate=False, **_):
            mints.append((tuple(names), reinitiate))
            return {"results": [{"connector": n, "status": "initiated", "connect_url": f"https://l/{n}/2"} for n in names]}

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", Client)
    # The real dispatcher: availability gate passes, then the open operation routes to _reissue.
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **k: {"manage_connections"})

    refused = _connect_rpc(owner, connectors=["gmail"])
    assert refused["error"]["code"] == 4002 and mints == []
    assert operation.target("gmail").connect_url == "https://l/gmail/1"

    reply = _connect_rpc(owner, connectors=["notion"])
    assert "result" in reply, reply
    assert mints == [(("notion",), True)]
    assert operation.target("notion").state == TargetState.initiated
    assert operation.target("notion").connect_url == "https://l/notion/2"


@pytest.mark.parametrize("dead", [TargetState.failed, TargetState.expired])
def test_a_failed_reissue_leaves_the_row_failed_with_no_link(owned, monkeypatch, dead):
    """Try again whose mint fails must not show the row as waiting on the old dead link."""
    owner, _, _ = owned
    operation = _open_op()
    operation.transition("notion", TargetState.initiated, Actor.backend_watcher, connect_url="https://l/notion/1")
    actor = Actor.clock if dead == TargetState.expired else Actor.backend_watcher
    operation.transition("notion", dead, actor, detail="vendor: nope")

    class Client:
        def connections(self, names, *, reinitiate=False, **_):
            return {"results": [{"connector": n, "status": "failed", "status_reason": "vendor: still no"} for n in names]}

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", Client)
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **k: {"manage_connections"})

    before = len(owner.frames)
    _rpc(owner, "connectors.connect", connectors=["notion"])
    deadline = time.time() + 2
    while time.time() < deadline and not [f for f in list(owner.frames)[before:] if f.get("id") == 7]:
        time.sleep(0.01)
    target = operation.target("notion")
    assert target.state == TargetState.failed
    assert target.connect_url is None
    assert target.detail == "vendor: still no"



class FakeOAuthAttempt:
    def __init__(self, auth_url):
        self.auth_url = auth_url

    def poll(self):
        return {"status": "pending", "error": "", "tools": []}


class FakeMcpBackend:
    """The MCP work behind an authorize target; the managed gateway has no part in it."""

    def __init__(self):
        self.starts = []

    def start_oauth(self, name):
        self.starts.append(name)
        return FakeOAuthAttempt(f"https://auth.example/{name}/{len(self.starts)}")


def test_try_again_on_an_mcp_target_re_runs_its_flow_and_never_calls_the_managed_gateway(owned, monkeypatch):
    owner, _, _ = owned
    operation = ConnectionOperation([Target("linear", "mcp", "authorize")], session_key=SID)
    live.open(operation)
    backend = FakeMcpBackend()
    runner = mcp.open_runner("authorize", backend)
    runner.prepare(operation)  # the first flow: the row waits on the link it minted
    operation.transition("linear", TargetState.failed, Actor.backend_watcher, detail="the provider timed out")

    class Forbidden:
        def __init__(self):
            raise AssertionError("an MCP target must never reach the managed gateway")

    monkeypatch.setattr("tools.connectors.gateway.client.ConnectorClient", Forbidden)
    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **k: {"manage_connections"})

    reply = _connect_rpc(owner, connectors=["linear"], reconnect=True)
    assert "result" in reply, reply
    assert backend.starts == ["linear", "linear"]
    target = operation.target("linear")
    assert target.state == TargetState.initiated
    assert target.connect_url == "https://auth.example/linear/2"
    assert target.detail == ""
    runner.close()


def test_try_again_refuses_an_mcp_target_the_contract_cannot_run_again(owned, monkeypatch):
    """Only a state the transition table can leave may be run again. An MCP target has no move out
    of ``expired``, so Try again on one is refused instead of dropping a flow that goes nowhere."""
    owner, _, _ = owned
    operation = ConnectionOperation([Target("linear", "mcp", "authorize")], session_key=SID)
    live.open(operation)
    backend = FakeMcpBackend()
    runner = mcp.open_runner("authorize", backend)
    runner.prepare(operation)
    operation.target("linear").state = TargetState.expired

    monkeypatch.setattr("tools.connectors.connectors_available", lambda: True)
    monkeypatch.setattr("model_tools._select_tool_names", lambda *a, **k: {"manage_connections"})

    reply = _connect_rpc(owner, connectors=["linear"], reconnect=True)
    runner.close()

    assert reply["error"]["code"] == 4002
    assert backend.starts == ["linear"]


def test_try_again_refuses_an_operation_that_settled_during_the_call(owned):
    """Continue can settle the operation between the target read and the re-run; the result is
    already frozen, so the re-run would spend a flow nothing can report."""
    operation = ConnectionOperation([Target("linear", "mcp", "authorize")], session_key=SID)
    live.open(operation)
    backend = FakeMcpBackend()
    runner = mcp.open_runner("authorize", backend)
    runner.prepare(operation)
    operation.transition("linear", TargetState.failed, Actor.backend_watcher, detail="the provider timed out")
    operation.settled_at = time.time()

    reply = server._reissue(7, operation, {"connectors": ["linear"]})
    runner.close()

    assert reply["error"]["code"] == 4002
    assert backend.starts == ["linear"]


# ---------------------------------------------------------------------------
# seq and wake (P1-13, the deep-link return)
# ---------------------------------------------------------------------------


def test_every_connection_frame_carries_a_rising_seq(owned):
    """The renderer orders frames by ``seq``, so no two frames of one operation share one and the
    status reply is never behind the last frame it sent."""
    owner, _, _ = owned
    operation = _open_op()
    operation.transition("gmail", TargetState.initiated, Actor.backend_watcher)
    operation.transition("gmail", TargetState.connected, Actor.backend_watcher)
    seqs = [update["payload"]["seq"] for update in owner.events("connection.update", expect=2)]
    assert len(seqs) == len(set(seqs)) and seqs == sorted(seqs)
    status = _rpc(owner, "connectors.operation.status", op_id=operation.op_id)["result"]
    assert status["seq"] == seqs[-1]


def test_wake_is_owner_only_and_names_a_live_operation(owned):
    owner, stranger, _ = owned
    operation = _open_op()
    assert _rpc(stranger, "connectors.operation.wake", op_id=operation.op_id)["error"]["code"] == 4001
    assert _rpc(owner, "connectors.operation.wake")["error"]["code"] == 4000
    assert _rpc(owner, "connectors.operation.wake", op_id="nope")["error"]["code"] == 4004
    assert not operation.wake.is_set()
    assert _rpc(owner, "connectors.operation.wake", op_id=operation.op_id)["result"] == {"status": "ok"}
    assert operation.wake.is_set()


def test_a_wake_makes_the_watch_loop_read_before_its_next_tick(owned):
    """The browser came back from the vendor's done page: read the account now, not a tick from now."""
    from tools.connectors.contract import SettleReason
    from tools.connectors.run import Kind, run_operation

    owner, _, _ = owned
    read = threading.Event()
    finished = threading.Event()

    def run():
        run_operation([Target("gmail", "connector", "connect")],
                      Kind(prepare=lambda operation: None, observe=lambda operation: read.set(), note=""),
                      session_key=SID, tool_call_id=None, connection_callback=None,
                      tick_seconds=30.0, with_urls_in_result=False)
        finished.set()

    threading.Thread(target=run, daemon=True).start()
    assert read.wait(2), "the loop reads once before it sleeps"
    operation = live.current(SID)
    read.clear()

    assert _rpc(owner, "connectors.operation.wake", op_id=operation.op_id)["result"] == {"status": "ok"}
    assert read.wait(2), "the wake must not wait out the 30 s tick"

    operation.settle(SettleReason.continue_)
    assert finished.wait(2)

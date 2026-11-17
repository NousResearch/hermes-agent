"""A remote Desktop's subagent panel shows the background subagents the gateway runs for the chat.

The Desktop polls ``subagent.list`` (chips, running state), ``subagent.tail`` (a chip's live transcript),
``subagent.steer`` / ``subagent.interrupt`` (its controls) and ``subagent.roster`` (the sidebar's running
marker). On a standalone ``hermes serve`` those read this process's registry, which never holds a child
the gateway dispatched, so the panel stayed empty while the child worked. They now ask that gateway
(``owner_forward.forward_session_rpc``) when the durable ``async_delegations`` ledger says it runs one.

Real: the SessionDB and its ledger row, a live owner process (liveness and start time checked for real),
the legacy dispatcher and its session/transport authority. Replaced: the gateway identity lookup
(``live_gateway_pid_for_home``) and the dial to it (``forward_session_rpc``, tested on its own below).
"""
import asyncio
import contextlib
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB

KEY = "local-chat-1"


@pytest.fixture
def owner_process():
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    yield proc
    proc.kill()
    proc.wait()


def _start_time(pid):
    from gateway.status import get_process_start_time
    return get_process_start_time(pid)


def _running_delegation(db_path, parent, pid, started):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO async_delegations (delegation_id, origin_session, parent_session_id, state,
                   dispatched_at, updated_at, owner_pid, owner_started_at, task_json)
               VALUES ('deleg_gw', 'agent:main:local:dm:x', ?, 'running', ?, ?, ?, ?, '{"goal": "long build"}')""",
            (parent, time.time() - 60, time.time(), pid, started))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def serve(monkeypatch, tmp_path):
    """A standalone serve's legacy dispatcher with one Desktop session on stored chat ``KEY``."""
    from gateway import status
    from tools import delegate_tool_registry
    from tui_gateway import owner_forward, server

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(KEY, source="desktop")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    transport = SimpleNamespace(write=lambda frame: True)
    owner = {"session_key": KEY, "history": [], "transport": transport}
    monkeypatch.setattr(server, "_sessions", {"ui-owner": owner})
    saved = dict(delegate_tool_registry._active_subagents)
    delegate_tool_registry._active_subagents.clear()
    gateway = SimpleNamespace(pid=None, calls=[], answers={}, fail=False)
    monkeypatch.setattr(status, "live_gateway_pid_for_home", lambda _home: gateway.pid)

    def forward(home, key, method, params):
        gateway.calls.append((key, method, params))
        if gateway.fail:
            raise owner_forward.OwnerNotRunning("the gateway that owns it is stopped")
        return gateway.answers[method]
    monkeypatch.setattr(owner_forward, "forward_session_rpc", forward)

    def call(method, **params):
        return server.dispatch({"id": 1, "method": method, "params": {"session_id": "ui-owner", **params}},
                               transport=transport)

    yield SimpleNamespace(db=db, path=tmp_path / "state.db", gateway=gateway, call=call, owner=owner,
                          transport=transport)
    delegate_tool_registry._active_subagents.clear()
    delegate_tool_registry._active_subagents.update(saved)
    db.close()


def _gateway_runs_child(serve, owner_process, parent=KEY):
    _running_delegation(serve.path, parent, owner_process.pid, _start_time(owner_process.pid))
    serve.gateway.pid = owner_process.pid


GATEWAY_ROW = {"subagent_id": "sa-gw", "goal": "long build", "status": "running", "accepting_steer": True}


def test_Given_the_gateway_runs_this_chats_subagent_When_the_panel_lists_Then_it_shows_the_gateways_child(
        serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.answers["subagent.list"] = {"subagents": [GATEWAY_ROW], "delegations": []}

    # When
    result = serve.call("subagent.list")["result"]

    # Then
    assert result["subagents"] == [GATEWAY_ROW]
    assert serve.gateway.calls == [(KEY, "subagent.list", {})]


def test_Given_a_child_here_and_one_in_the_gateway_When_the_panel_lists_Then_both_show_once(
        serve, owner_process):
    # Given
    from tools.delegate_tool_child_run import _register_child
    from tools.delegate_tool_registry import _unregister_subagent
    _gateway_runs_child(serve, owner_process)
    child = SimpleNamespace(_subagent_id="sa-here", _delegate_depth=1, model="m")
    _register_child(child, None, "local", owner_session_id="ui-owner",
                    owner_transport=serve.transport, owner_session_record=serve.owner)
    serve.gateway.answers["subagent.list"] = {"subagents": [GATEWAY_ROW, {**GATEWAY_ROW, "subagent_id": "sa-here"}]}
    try:
        # When
        result = serve.call("subagent.list")["result"]
    finally:
        _unregister_subagent("sa-here")

    # Then
    assert [r["subagent_id"] for r in result["subagents"]] == ["sa-here", "sa-gw"]


def test_Given_no_gateway_subagent_When_the_panel_lists_Then_the_gateway_is_not_asked(serve, owner_process):
    # Given: a live non-gateway owner (a CLI) runs it
    _running_delegation(serve.path, KEY, owner_process.pid, _start_time(owner_process.pid))

    # When
    result = serve.call("subagent.list")["result"]

    # Then
    assert result["subagents"] == [] and serve.gateway.calls == []


def test_Given_the_gateway_stops_answering_When_the_panel_lists_Then_it_answers_from_here(serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.fail = True

    # When
    reply = serve.call("subagent.list")

    # Then
    assert reply["result"]["subagents"] == []


def test_Given_a_gateway_child_When_its_chip_opens_the_tail_Then_the_gateways_transcript_shows(
        serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.answers["subagent.tail"] = {"subagent_id": "sa-gw", "available": True, "text": "tool | sleep",
                                              "truncated": False}

    # When
    result = serve.call("subagent.tail", subagent_id="sa-gw")["result"]

    # Then
    assert result == {"subagent_id": "sa-gw", "available": True, "text": "tool | sleep", "truncated": False}
    assert serve.gateway.calls == [(KEY, "subagent.tail", {"subagent_id": "sa-gw"})]


def test_Given_a_gateway_child_When_stop_is_pressed_Then_the_gateway_interrupts_it(serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.answers["subagent.interrupt"] = {"found": True, "subagent_id": "sa-gw"}

    # When
    result = serve.call("subagent.interrupt", subagent_id="sa-gw")["result"]

    # Then
    assert result == {"found": True, "subagent_id": "sa-gw"}
    assert serve.gateway.calls == [(KEY, "subagent.interrupt", {"subagent_id": "sa-gw"})]


def test_Given_the_gateway_no_longer_has_the_child_When_stop_is_pressed_Then_it_reports_not_found(
        serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.answers["subagent.interrupt"] = {"found": False, "subagent_id": "sa-gw"}

    # When
    result = serve.call("subagent.interrupt", subagent_id="sa-gw")["result"]

    # Then
    assert result == {"found": False, "subagent_id": "sa-gw"}


def test_Given_a_gateway_child_When_it_is_steered_Then_the_gateway_queues_the_text(serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.answers["subagent.steer"] = {"status": "queued", "subagent_id": "sa-gw", "text": "faster"}

    # When
    result = serve.call("subagent.steer", subagent_id="sa-gw", text="faster")["result"]

    # Then
    assert result == {"status": "queued", "subagent_id": "sa-gw", "text": "faster"}
    assert serve.gateway.calls == [(KEY, "subagent.steer", {"subagent_id": "sa-gw", "text": "faster"})]


def test_Given_the_gateway_rejects_a_steer_When_it_is_steered_Then_it_reports_rejected(serve, owner_process):
    # Given
    _gateway_runs_child(serve, owner_process)
    serve.gateway.answers["subagent.steer"] = {"status": "rejected", "subagent_id": "sa-gw", "text": "faster"}

    # When
    result = serve.call("subagent.steer", subagent_id="sa-gw", text="faster")["result"]

    # Then
    assert result["status"] == "rejected"


def test_Given_a_connection_not_attached_to_the_chat_When_it_steers_Then_it_is_rejected_without_asking_a_gateway(
        serve, owner_process):
    # Given: another client's connection, with no authority over this chat
    from tui_gateway import server
    _gateway_runs_child(serve, owner_process)
    stranger = SimpleNamespace(write=lambda frame: True)

    # When
    reply = server.dispatch({"id": 1, "method": "subagent.steer",
                             "params": {"session_id": "ui-owner", "subagent_id": "sa-gw", "text": "faster"}},
                            transport=stranger)

    # Then
    assert reply["result"]["status"] == "rejected"
    assert serve.gateway.calls == []


def test_Given_the_gateway_runs_a_chats_subagent_When_the_sidebar_reads_the_roster_Then_the_chat_shows_running(
        serve, owner_process):
    # Given
    from tui_gateway import server
    _gateway_runs_child(serve, owner_process)

    # When
    sessions = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "subagent.roster", "params": {}})

    # Then
    assert sessions["result"]["sessions"] == [{"session_id": KEY, "subagents": 1, "background": 0}]


def test_Given_a_non_gateway_owner_When_the_sidebar_reads_the_roster_Then_the_chat_is_not_listed(
        serve, owner_process):
    # Given
    from tui_gateway import server
    _running_delegation(serve.path, KEY, owner_process.pid, _start_time(owner_process.pid))

    # When
    sessions = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "subagent.roster", "params": {}})

    # Then
    assert sessions["result"]["sessions"] == []


def test_Given_a_live_gateway_and_a_child_another_process_runs_When_the_roster_is_read_Then_it_is_not_listed(
        serve, owner_process):
    # Given: a gateway is up, but the child's owner is a different live process (a CLI)
    from tui_gateway import server
    _running_delegation(serve.path, KEY, owner_process.pid, _start_time(owner_process.pid))
    gateway = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    serve.gateway.pid = gateway.pid
    try:
        # When
        sessions = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "subagent.roster", "params": {}})
    finally:
        gateway.kill()
        gateway.wait()

    # Then
    assert sessions["result"]["sessions"] == []


def test_Given_a_gateway_child_of_a_chat_stored_elsewhere_When_the_roster_is_read_Then_it_is_not_listed(
        serve, owner_process):
    # Given: another profile's conversation id is never reported
    from tui_gateway import server
    _gateway_runs_child(serve, owner_process, parent="other-profile-chat")

    # When
    sessions = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "subagent.roster", "params": {}})

    # Then
    assert sessions["result"]["sessions"] == []


def test_Given_an_owner_When_a_session_rpc_is_forwarded_Then_it_attaches_without_the_transcript_and_calls(
        monkeypatch, tmp_path):
    # Given
    from tui_gateway import owner_forward
    seen = []

    class _Client:
        async def rpc(self, method, **params):
            seen.append((method, params))
            return {"session_id": "owner-id"} if method == "session.resume" else {"found": True}

    @contextlib.asynccontextmanager
    async def connect(home, endpoint):
        assert (home, endpoint) == (tmp_path, "ready-owner")
        yield _Client()
    monkeypatch.setattr(owner_forward, "discover_owner", lambda _home: "ready-owner")
    monkeypatch.setattr(owner_forward, "connect_owner", connect)

    # When
    answer = owner_forward.forward_session_rpc(tmp_path, KEY, "subagent.interrupt", {"subagent_id": "sa-gw"})

    # Then
    assert answer == {"found": True}
    assert seen == [("session.resume", {"session_id": KEY, "omit_messages": True}),
                    ("subagent.interrupt", {"subagent_id": "sa-gw", "session_id": "owner-id"})]


def test_Given_an_owner_that_never_answers_When_a_session_rpc_is_forwarded_Then_it_gives_up(monkeypatch, tmp_path):
    # Given
    from tui_gateway import owner_forward
    monkeypatch.setattr(owner_forward, "SESSION_RPC_TIMEOUT_S", 0.2)
    @contextlib.asynccontextmanager
    async def connect(home, endpoint):
        await asyncio.Event().wait()  # an owner that accepted the dial and never answers
        yield None
    monkeypatch.setattr(owner_forward, "discover_owner", lambda _home: "ready-owner")
    monkeypatch.setattr(owner_forward, "connect_owner", connect)

    # When / Then
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        owner_forward.forward_session_rpc(tmp_path, KEY, "subagent.list", {})
    assert time.monotonic() - started < 2

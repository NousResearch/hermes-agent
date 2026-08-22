"""Control RPCs must not block on the deferred agent build.

``approval.pending``, ``approval.received``, ``approval.respond``,
``process.kill`` and ``process.list`` need the session RECORD — specifically
``session_key`` — and never the agent. The approval trio reaches
``tools.approval``'s module-global ``_gateway_queues``, which is keyed by
``session_key`` alone; ``process.*`` matches the same key against the global
process registry. Neither structure has any relationship to the session's
``agent_ready`` event, so ``_sess``'s ``_wait_agent`` bought nothing at these
five sites and charged up to 30 seconds for it.

Four of the five are not in ``_LONG_HANDLERS``, so that charge ran inline on
the socket reader thread and stalled every RPC queued behind it — the exact
failure mode ``_LONG_HANDLERS``' own comment cites ``approval.respond`` as the
reason to protect against. And the wait is reachable by construction rather
than by accident: the desktop replays pending approvals on ``gateway.ready`` /
``session.info``, i.e. precisely while a cold resume's deferred build is still
warming, and ``_start_agent_build`` deliberately early-returns for a lazy watch
session spectating an in-flight child — leaving ``agent_ready`` unset for the
whole child run, so these RPCs did not merely stall, they timed out at 5032.

These are invariants, not timings: each handler must complete, and do its real
work, while the build event is still unset.
"""

from __future__ import annotations

import threading

import pytest

from tui_gateway import server


def building_session(tmp_path, sid: str) -> dict:
    """A session record whose deferred agent build has NOT completed."""
    session = {
        "agent": None,
        "agent_ready": threading.Event(),  # deliberately never set
        "agent_error": None,
        "attached_images": [],
        "cwd": str(tmp_path),
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "profile_home": str(tmp_path),
        "running": False,
        "session_key": sid,
        "transport": None,
    }
    server._sessions[sid] = session
    return session


@pytest.fixture
def no_build(monkeypatch):
    """Never let the real builder run — the point is the unfinished build."""
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)


@pytest.fixture
def session(tmp_path, no_build, request):
    sid = f"control-{request.node.name}"
    record = building_session(tmp_path, sid)
    yield sid, record
    server._sessions.pop(sid, None)


def call(method: str, params: dict) -> dict:
    return server._methods[method](1, params)


@pytest.fixture
def pending_approval(session):
    """One unresolved approval queued under this session's key.

    ``tools.approval`` has no public enqueue helper — the only producer is
    ``_await_gateway_decision`` on the agent thread — so this mirrors its three
    lines (build an ``_ApprovalEntry``, append it under ``_lock``) rather than
    reaching for a seam that does not exist. The queue is popped again on
    teardown because it is process-global state.
    """
    from tools import approval

    sid, record = session
    entry = approval._ApprovalEntry(
        {"command": "rm -rf /tmp/scratch", "description": "delete scratch", "request_id": "req-1"}
    )
    with approval._lock:
        approval._gateway_queues.setdefault(sid, []).append(entry)
    yield sid, record, entry
    with approval._lock:
        approval._gateway_queues.pop(sid, None)


def test_approval_pending_replays_the_queue_while_the_agent_is_building(pending_approval):
    """The replay the desktop fires on reconnect must not wait for the build."""
    sid, record, _entry = pending_approval

    response = call("approval.pending", {"session_id": sid})

    assert "error" not in response, response
    approvals = response["result"]["approvals"]
    assert [a["request_id"] for a in approvals] == ["req-1"]
    assert approvals[0]["command"] == "rm -rf /tmp/scratch"
    # The invariant that makes this a fix rather than a coincidence: the build
    # never finished, and the replay landed anyway.
    assert not record["agent_ready"].is_set()


def test_approval_received_acknowledges_while_the_agent_is_building(pending_approval):
    """Not waiting must not mean not acking — the entry is really flipped."""
    sid, record, entry = pending_approval
    assert entry.acknowledged is False

    response = call("approval.received", {"session_id": sid, "request_id": "req-1"})

    assert "error" not in response, response
    assert response["result"]["acknowledged"] is True
    assert entry.acknowledged is True
    assert not record["agent_ready"].is_set()


def test_approval_respond_resolves_while_the_agent_is_building(pending_approval):
    """The RPC _LONG_HANDLERS exists to protect must itself run unblocked."""
    sid, record, entry = pending_approval

    response = call(
        "approval.respond", {"session_id": sid, "request_id": "req-1", "choice": "deny"}
    )

    assert "error" not in response, response
    assert response["result"]["resolved"] == 1
    # The blocked agent thread is genuinely released, not just answered.
    assert entry.result == "deny"
    assert entry.event.is_set()
    assert not record["agent_ready"].is_set()


def test_approval_received_still_requires_a_request_id(session):
    """Dropping the agent wait must not drop parameter validation."""
    sid, _record = session

    response = call("approval.received", {"session_id": sid})

    assert response["error"]["code"] == 4006


@pytest.mark.parametrize(
    "method", ["approval.pending", "approval.received", "approval.respond"]
)
def test_approval_rpcs_still_reject_an_unknown_session(no_build, method):
    """_sess_building keeps _sess_nowait's 4001 — validation is not the wait."""
    response = call(method, {"session_id": "nope", "request_id": "req-1"})

    assert response["error"]["code"] == 4001


@pytest.fixture
def registered_processes(session):
    """One background process owned by this session and one owned by another.

    The registry is a process-global singleton, so both entries are removed
    again on teardown.
    """
    from tools.process_registry import ProcessSession, process_registry

    sid, record = session
    mine = ProcessSession(id="proc_mine", command="npm run dev", session_key=sid)
    theirs = ProcessSession(id="proc_theirs", command="npm run other", session_key="other-session")
    with process_registry._lock:
        process_registry._running["proc_mine"] = mine
        process_registry._running["proc_theirs"] = theirs
    yield sid, record, mine, theirs
    with process_registry._lock:
        process_registry._running.pop("proc_mine", None)
        process_registry._running.pop("proc_theirs", None)


def test_process_list_reports_this_sessions_processes_while_building(registered_processes):
    """The desktop status stack fills on a cold resume instead of sitting empty."""
    sid, record, _mine, _theirs = registered_processes

    response = call("process.list", {"session_id": sid})

    assert "error" not in response, response
    processes = response["result"]["processes"]
    # Session scoping survives: the other session's process is not disclosed.
    assert [p["session_id"] for p in processes] == ["proc_mine"]
    assert not record["agent_ready"].is_set()


def test_process_kill_still_refuses_another_sessions_process_while_building(
    registered_processes,
):
    """4044, not 5032: the handler reaches its ownership check without waiting.

    This is the discriminator for the whole change. Before the fix the wait
    fired first and every one of these calls came back 5032 "agent
    initialization timed out" — the ownership rule was never even consulted.
    """
    sid, record, _mine, _theirs = registered_processes

    response = call("process.kill", {"session_id": sid, "process_id": "proc_theirs"})

    assert response["error"]["code"] == 4044
    assert not record["agent_ready"].is_set()


def test_process_kill_still_reports_an_unknown_process_while_building(session):
    """Same discriminator on the plain not-found path."""
    sid, record = session

    response = call("process.kill", {"session_id": sid, "process_id": "proc_absent"})

    assert response["error"]["code"] == 4044
    assert not record["agent_ready"].is_set()


def test_process_kill_still_requires_a_process_id(session):
    """Dropping the agent wait must not drop parameter validation."""
    sid, _record = session

    response = call("process.kill", {"session_id": sid})

    assert response["error"]["code"] == 4012


CONVERTED_CONTROL_RPCS = (
    "approval.pending",
    "approval.received",
    "approval.respond",
    "process.list",
    "process.kill",
)


def test_converted_control_rpcs_skip_the_agent_wait_and_rollback_still_takes_it(
    session, monkeypatch
):
    """Resolver-level proof that the sweep is complete AND correctly bounded.

    The five control RPCs must reach ``_sess_building``; ``rollback.list`` is
    deliberately NOT converted, because it resolves through
    ``_with_checkpoints``, which dereferences ``session["agent"]``. Converting
    it would fault mid-build rather than merely stall — the 5020 below is that
    dereference failing against an unbuilt session, which is exactly why the
    exclusion is reasoned and not an oversight. If either half of this ever
    flips, the boundary drawn by this sweep has gone stale.
    """
    sid, _record = session
    waited: list = []
    monkeypatch.setattr(server, "_wait_agent", lambda s, rid: waited.append(rid) or None)

    for method in CONVERTED_CONTROL_RPCS:
        call(method, {"session_id": sid, "request_id": "req-1", "process_id": "proc_absent"})
    assert waited == []

    rollback = call("rollback.list", {"session_id": sid})

    assert waited == [1]
    assert rollback["error"]["code"] == 5020
    # Names the dereference that keeps rollback.* out of this sweep.
    assert "_checkpoint_mgr" in rollback["error"]["message"]

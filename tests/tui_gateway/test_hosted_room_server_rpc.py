"""Tests for the in-process hosted room session adapter."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_server_rpc import (
    HostedRoomServerRPC,
    HostedRoomSessionError,
)


def _server():
    sessions = {}
    calls = []

    def method(name, result):
        def handler(rid, params):
            calls.append((name, params))
            value = result(params) if callable(result) else result
            return {"id": rid, **value}

        return handler

    methods = {
        "session.list": method(
            "session.list",
            {"result": {"sessions": [{"id": "stored", "resolved_id": "tip", "title": "Group: room"}]}},
        ),
        "session.create": method("session.create", {"result": {"session_id": "runtime"}}),
        "session.resume": method("session.resume", {"result": {"session_id": "runtime"}}),
        "session.history": method("session.history", {"result": {"messages": [{"role": "assistant"}]}}),
        "session.interrupt": method("session.interrupt", {"result": {"interrupted": True}}),
        "approval.respond": method("approval.respond", {"result": {"resolved": 1}}),
        "prompt.submit": method("prompt.submit", {"result": {"status": "streaming"}}),
    }
    server = SimpleNamespace(
        _methods=methods,
        _sessions=sessions,
        _sessions_lock=threading.Lock(),
        _pending_approval_request_payload=lambda _session_key: None,
    )
    return server, calls


def test_routes_exact_hidden_session_and_internal_task_proof():
    server, calls = _server()
    rpc = HostedRoomServerRPC(server)
    task = TaskIdentity("room", "task", "thread", "turn")
    callback = lambda _receipt: None

    assert rpc.resolve_exact(profile="ops", title="Group: room", source="bot_room")["session_id"] == "tip"
    assert rpc.create(profile="ops", title="Group: room", source="bot_room")["session_id"] == "runtime"
    rpc.submit(
        profile="ops",
        session_id="runtime",
        prompt="Do the work",
        source="bot_room",
        task=task,
        execution_generation=2,
        on_terminal=callback,
    )

    create = next(params for method, params in calls if method == "session.create")
    submit = next(params for method, params in calls if method == "prompt.submit")
    assert create["hidden"] is True
    assert create["room_plumbing"] is True
    assert create["follow_profile_config"] is True
    assert create["close_on_disconnect"] is False
    assert submit["_hosted_task"] == {
        "room_id": "room",
        "task_id": "task",
        "thread_id": "thread",
        "turn_id": "turn",
        "execution_generation": 2,
    }
    assert submit["_hosted_terminal_callback"] is callback

    rpc.resume(profile="ops", session_id="stored", source="bot_room")
    resume = next(params for method, params in calls if method == "session.resume")
    assert resume["source"] == "bot_room"


def test_info_and_interrupt_are_exact_task_scoped():
    server, calls = _server()
    lock = threading.Lock()
    server._sessions["runtime"] = {
        "history_lock": lock,
        "running": True,
        "_hosted_room_task": {"task_id": "task-a"},
    }
    rpc = HostedRoomServerRPC(server)

    assert rpc.info(profile="ops", session_id="runtime", source="bot_room") == {
        "active": True,
        "task_id": "task-a",
    }
    rpc.interrupt(
        profile="ops",
        session_id="runtime",
        source="bot_room",
        expected_task_id="task-a",
    )
    params = next(params for method, params in calls if method == "session.interrupt")
    assert params["expected_hosted_task_id"] == "task-a"


def test_local_approval_snapshot_and_response_use_exact_request():
    server, calls = _server()
    server._pending_approval_request_payload = lambda session_key: {
        "request_id": "approval-1",
        "command": "pytest -q tests/focused",
        "choices": ["once", "deny"],
    } if session_key == "stored-session" else None
    server._sessions["runtime"] = {
        "history_lock": threading.Lock(),
        "running": True,
        "session_key": "stored-session",
        "_hosted_room_task": {"task_id": "task-a"},
    }
    rpc = HostedRoomServerRPC(server)

    info = rpc.info(profile="ops", session_id="runtime", source="bot_room")
    assert info["status"] == "waiting_for_approval"
    assert info["pending_approval"]["request_id"] == "approval-1"
    assert rpc.approve(
        session_id="runtime",
        request_id="approval-1",
        choice="once",
    ) == {"resolved": 1}
    params = next(params for method, params in calls if method == "approval.respond")
    assert params == {
        "session_id": "runtime",
        "request_id": "approval-1",
        "choice": "once",
        "all": False,
    }


def test_rpc_errors_are_typed():
    server, _calls = _server()
    server._methods["session.list"] = lambda rid, _params: {
        "id": rid,
        "error": {"code": 4007, "message": "not found"},
    }
    rpc = HostedRoomServerRPC(server)

    with pytest.raises(HostedRoomSessionError) as exc:
        rpc.resolve_exact(profile="ops", title="Group: room", source="bot_room")
    assert exc.value.code == 4007


def test_prompt_rejection_is_proven_not_admitted():
    server, _calls = _server()
    server._methods["prompt.submit"] = lambda rid, _params: {
        "id": rid,
        "error": {"code": 4121, "message": "session is already busy"},
    }
    rpc = HostedRoomServerRPC(server)

    with pytest.raises(HostedRoomSessionError) as exc:
        rpc.submit(
            profile="ops",
            session_id="runtime",
            prompt="Do the work",
            source="bot_room",
            task=TaskIdentity("room", "task", "thread", "turn"),
            execution_generation=1,
            on_terminal=lambda _receipt: None,
        )

    assert exc.value.code == 4121
    assert exc.value.not_admitted is True

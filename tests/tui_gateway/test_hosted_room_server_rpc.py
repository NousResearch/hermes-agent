"""Tests for the in-process hosted room session adapter."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from gateway.hosted_room_driver import TaskIdentity
from tui_gateway.hosted_room_driver import HostedRoomProfileUnavailableError
from tui_gateway.hosted_room_server_rpc import (
    HostedRoomServerRPC,
    HostedRoomSessionError,
)
from tui_gateway.transport import bind_transport, current_transport, reset_transport


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


def _admitted_session(
    task: TaskIdentity,
    *,
    execution_generation: int = 2,
    running: bool = True,
):
    return {
        "history_lock": threading.Lock(),
        "running": running,
        "source": "bot_room",
        "room_plumbing": True,
        "_hosted_room_task": {
            "room_id": task.room_id,
            "task_id": task.task_id,
            "thread_id": task.thread_id,
            "turn_id": task.turn_id,
            "execution_generation": execution_generation,
        },
    }


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
    lookup = next(params for method, params in calls if method == "session.list")
    submit = next(params for method, params in calls if method == "prompt.submit")
    assert lookup == {
        "profile": "ops",
        "title": "Group: room",
        "source": "bot_room",
        "include_hidden": True,
    }
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


def test_unavailable_profile_is_rejected_before_any_server_handler():
    server, calls = _server()
    rpc = HostedRoomServerRPC(
        server,
        profile_available=lambda profile: profile == "default",
    )

    with pytest.raises(HostedRoomProfileUnavailableError):
        rpc.resolve_exact(profile="deleted", title="Group: room", source="bot_room")

    assert calls == []


def test_handler_calls_use_a_private_drop_transport_and_restore_the_caller():
    server, _calls = _server()
    seen = []

    def create(rid, _params):
        seen.append(current_transport())
        return {"id": rid, "result": {"session_id": "runtime"}}

    server._methods["session.create"] = create
    caller = SimpleNamespace(write=lambda _obj: True, close=lambda: None)
    token = bind_transport(caller)
    try:
        rpc = HostedRoomServerRPC(server)
        rpc.create(profile="ops", title="Group: room", source="bot_room")
        assert current_transport() is caller
    finally:
        reset_transport(token)

    assert len(seen) == 1
    assert seen[0] is not caller
    assert seen[0].write({"private": "room text"}) is True


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


def test_interrupt_admitted_matches_full_process_local_task_proof():
    server, calls = _server()
    task = TaskIdentity("room", "task", "thread", "turn")
    server._sessions["runtime"] = _admitted_session(task)
    rpc = HostedRoomServerRPC(server)

    result = rpc.interrupt_admitted(
        task=task,
        execution_generation=2,
        source="bot_room",
    )

    assert result == {
        "status": "interrupted",
        "acknowledged": True,
        "active": True,
        "interrupted": True,
        "session_id": "runtime",
    }
    interrupt = next(params for method, params in calls if method == "session.interrupt")
    assert interrupt == {
        "session_id": "runtime",
        "expected_hosted_task_id": "task",
        "expected_hosted_execution_generation": 2,
    }
    assert not any(method in {"session.list", "session.resume"} for method, _ in calls)


def test_interrupt_admitted_acknowledges_inactive_and_absent_tasks():
    server, calls = _server()
    task = TaskIdentity("room", "task", "thread", "turn")
    server._sessions["runtime"] = _admitted_session(task, running=False)
    rpc = HostedRoomServerRPC(server)

    assert rpc.interrupt_admitted(
        task=task,
        execution_generation=2,
        source="bot_room",
    ) == {
        "status": "inactive",
        "acknowledged": True,
        "active": False,
        "interrupted": False,
        "session_id": "runtime",
    }

    server._sessions.clear()
    assert rpc.interrupt_admitted(
        task=task,
        execution_generation=2,
        source="bot_room",
    ) == {
        "status": "absent",
        "acknowledged": False,
        "active": False,
        "interrupted": False,
        "session_id": None,
    }
    assert not any(method == "session.interrupt" for method, _ in calls)


def test_concurrent_interrupt_claim_stays_pending_until_signal_is_proven():
    server, _calls = _server()
    task = TaskIdentity("room", "task", "thread", "turn")
    server._sessions["runtime"] = _admitted_session(task)
    first_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def interrupt(rid, _params):
        nonlocal call_count
        with call_lock:
            call_count += 1
            current = call_count
        if current == 1:
            first_entered.set()
            assert release_first.wait(timeout=5)
            return {"id": rid, "result": {"status": "interrupted"}}
        return {"id": rid, "result": {"status": "not_interrupted"}}

    server._methods["session.interrupt"] = interrupt
    rpc = HostedRoomServerRPC(server)
    first_result: list[dict] = []
    first = threading.Thread(
        target=lambda: first_result.append(
            dict(
                rpc.interrupt_admitted(
                    task=task,
                    execution_generation=2,
                    source="bot_room",
                )
            )
        )
    )
    first.start()
    assert first_entered.wait(timeout=5)

    try:
        overlapping = rpc.interrupt_admitted(
            task=task,
            execution_generation=2,
            source="bot_room",
        )
        assert overlapping == {
            "status": "pending",
            "acknowledged": False,
            "active": True,
            "interrupted": False,
            "session_id": "runtime",
        }
    finally:
        release_first.set()
        first.join(timeout=5)
    assert not first.is_alive()
    assert first_result == [
        {
            "status": "interrupted",
            "acknowledged": True,
            "active": True,
            "interrupted": True,
            "session_id": "runtime",
        }
    ]


def test_interrupt_admitted_fails_closed_on_ambiguous_admission():
    server, calls = _server()
    task = TaskIdentity("room", "task", "thread", "turn")
    server._sessions["runtime-a"] = _admitted_session(task)
    server._sessions["runtime-b"] = _admitted_session(task)
    rpc = HostedRoomServerRPC(server)

    with pytest.raises(HostedRoomSessionError) as exc:
        rpc.interrupt_admitted(
            task=task,
            execution_generation=2,
            source="bot_room",
        )

    assert exc.value.code == 4091
    assert not any(method == "session.interrupt" for method, _ in calls)


def test_interrupt_admitted_fails_closed_on_generation_conflict():
    server, calls = _server()
    task = TaskIdentity("room", "task", "thread", "turn")
    server._sessions["runtime"] = _admitted_session(
        task,
        execution_generation=3,
    )
    rpc = HostedRoomServerRPC(server)

    with pytest.raises(HostedRoomSessionError) as exc:
        rpc.interrupt_admitted(
            task=task,
            execution_generation=2,
            source="bot_room",
        )

    assert exc.value.code == 4092
    assert not any(method == "session.interrupt" for method, _ in calls)


def test_deleted_profile_does_not_block_process_local_admitted_interrupt():
    server, calls = _server()
    task = TaskIdentity("room", "task", "thread", "turn")
    server._sessions["runtime"] = _admitted_session(task)

    def deleted_profile(_profile):
        raise AssertionError("process-local interrupt must not inspect profiles")

    rpc = HostedRoomServerRPC(server, profile_available=deleted_profile)

    result = rpc.interrupt_admitted(
        task=task,
        execution_generation=2,
        source="bot_room",
    )

    assert result["status"] == "interrupted"
    interrupt = next(params for method, params in calls if method == "session.interrupt")
    assert "profile" not in interrupt
    assert not any(method in {"session.list", "session.resume"} for method, _ in calls)


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

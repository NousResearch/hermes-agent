"""Focused tests for model-facing delegation lifecycle control."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
import time
import uuid
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools import async_delegation as ad
from tools import delegate_tool as dt
from tools.process_registry import process_registry

def _observe_interrupt(home, delegation_id, output):
    os.environ["HERMES_HOME"] = home
    output.put(ad.interrupt_async_delegation(delegation_id, session_key="owner", reason="observer"))

@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    with dt._active_subagents_lock:
        dt._active_subagents.clear()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()
    yield
    ad._reset_for_tests()
    with dt._active_subagents_lock:
        dt._active_subagents.clear()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

def _dispatch(runner, *, session_key="owner", roots=None, interrupt_fn=None):
    return ad.dispatch_async_delegation(
        goal="controlled task",
        context=None,
        toolsets=None,
        role="leaf",
        model="test",
        session_key=session_key,
        parent_session_id="parent-owner",
        origin_ui_session_id="ui-owner",
        runner=runner,
        interrupt_fn=interrupt_fn,
        root_subagent_ids=roots,
        max_async_children=4,
    )

def _wait_terminal(delegation_id, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = ad.get_durable_delegation(delegation_id)
        if row and row["state"] not in ad._ACTIVE_STATES:
            return row
        time.sleep(0.005)
    raise AssertionError("delegation did not become terminal")

def _interrupt_twice(monkeypatch, delegation_id, while_running=None):
    barrier = threading.Barrier(2)
    real_get = ad.get_async_delegation
    synchronized = threading.local()

    def synchronized_initial_snapshot(*args, **kwargs):
        snapshot = real_get(*args, **kwargs)
        if not getattr(synchronized, "done", False):
            synchronized.done = True
            barrier.wait(timeout=2)
        return snapshot

    monkeypatch.setattr(ad, "get_async_delegation", synchronized_initial_snapshot)
    outcomes, returned = [], threading.Event()

    def request(reason):
        outcomes.append(ad.interrupt_async_delegation(
            delegation_id, session_key="owner", reason=reason
        ))
        returned.set()

    threads = [threading.Thread(target=request, args=(reason,)) for reason in ("one", "two")]
    for thread in threads:
        thread.start()
    if while_running:
        while_running(outcomes, returned)
    for thread in threads:
        thread.join(4)
        assert not thread.is_alive()
    return outcomes



def test_standalone_delegation_tool_is_not_registered():
    import tools.delegate_tool  # noqa: F401
    import tools.delegation_control  # noqa: F401
    from tools.registry import registry

    definitions = registry.get_definitions({"delegate_task", "delegation"})
    names = {item["function"]["name"] for item in definitions}
    assert "delegate_task" in names
    assert "delegation" not in names


def test_tool_schema_and_runtime_validation_are_strict():
    import tools.delegate_tool  # noqa: F401
    from tools.delegation_control import _handle_delegation_args, delegation_control
    from tools.registry import registry

    definitions = registry.get_definitions({"delegate_task"})
    control = next(item for item in definitions if item["function"]["name"] == "delegate_task")
    action_enum = control["function"]["parameters"]["properties"]["action"]["enum"]
    assert "resume" in action_enum
    assert "wait" in action_enum

    cases = [
        _handle_delegation_args({"action": "list", "unexpected": True}),
        _handle_delegation_args({"action": "status", "delegation_id": "d", "force": True}),
        _handle_delegation_args(
            {"action": "interrupt", "delegation_id": "d", "cascade": "false"}
        ),
        _handle_delegation_args(
            {"action": "wait", "delegation_id": "d", "timeout_seconds": "1"}
        ),
        _handle_delegation_args(
            {"action": "tail", "delegation_id": "d", "limit": 1.5}
        ),
        delegation_control(action="tail", delegation_id="d", limit=65),
    ]
    for output in cases:
        payload = json.loads(output)
        assert payload["status"] == "invalid_arguments"
        assert payload["error"]

    force_schema = control["function"]["parameters"]["properties"]["force"]
    assert force_schema["type"] == "boolean"


def test_resume_control_is_strict_and_forwards_live_parent(monkeypatch):
    import tools.delegate_tool  # noqa: F401
    from tools.delegation_control import _handle_delegation_args, delegation_control
    from tools.registry import registry

    definitions = registry.get_definitions({"delegate_task"})
    control = next(item for item in definitions if item["function"]["name"] == "delegate_task")
    assert "resume" in control["function"]["parameters"]["properties"]["action"]["enum"]
    for args in (
        {"action": "resume", "delegation_id": "d", "message": "next"},
        {"action": "resume", "delegation_id": "d", "subagent_id": "sa"},
        {
            "action": "resume",
            "delegation_id": "d",
            "subagent_id": "sa",
            "message": "next",
            "reason": "not allowed",
        },
    ):
        assert json.loads(_handle_delegation_args(args))["status"] == "invalid_arguments"

    repository, initial, attempt = _starting_steer_attempt(
        "deleg-resume-control", "sa-resume-control"
    )
    repository.transition_attempt(
        attempt["attempt_id"], {"starting"}, "completed"
    )
    captured = {}

    def fake_dispatch(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {
            "status": "dispatched",
            "delegation_id": args[0],
            "subagent_id": args[1],
            "run_id": "run-new",
            "attempt_id": "attempt-new",
            "attempt_number": 2,
            "child_session_id": "child-new",
            "bundle": {"history": [{"reasoning": "never expose"}]},
        }

    monkeypatch.setattr(ad, "dispatch_resumed_subagent", fake_dispatch)
    parent = object()
    payload = json.loads(
        delegation_control(
            action="resume",
            delegation_id="deleg-resume-control",
            subagent_id="sa-resume-control",
            message=" continue now ",
            session_key="owner",
            parent_agent=parent,
        )
    )
    assert payload == {
        "action": "resume",
        "status": "dispatched",
        "delegation_id": "deleg-resume-control",
        "subagent_id": "sa-resume-control",
        "run_id": "run-new",
        "attempt_id": "attempt-new",
        "attempt_number": 2,
        "child_session_id": "child-new",
    }
    assert captured["kwargs"] == {
        "session_key": "owner",
        "message": "continue now",
        "parent_agent": parent,
    }
    assert "reasoning" not in repr(payload)


def _two_terminal_runs():
    suffix = uuid.uuid4().hex
    delegation_id = f"deleg-two-runs-{suffix}"
    child_id = f"sa-two-runs-{suffix}"
    record = {
        "delegation_id": delegation_id,
        "goal": "two runs",
        "goals": ["two runs"],
        "session_key": "owner",
        "origin_ui_session_id": "ui-owner",
        "parent_session_id": "owner",
        "status": "running",
        "dispatched_at": time.time(),
        "root_subagent_ids": [child_id],
        "is_batch": False,
    }
    ad._persist_dispatch(record)
    repo = ad._repository()
    first_run = record["run_id"]
    first_attempt = record["attempt_ids"][0]
    old_metadata = {
        "child_session_id": f"child-old-{suffix}",
        "assistant_text_tail": "old tail",
        "events": [{"type": "tool.started", "name": "old-tool"}],
    }
    assert repo.transition_attempt(
        first_attempt, {"starting"}, "completed", metadata=old_metadata
    )["status"] == "updated"
    assert repo.complete_run(
        first_run,
        {"type": "async_delegation", "delegation_id": delegation_id,
         "run_id": first_run, "status": "completed"},
        {"status": "completed", "summary": "old result"},
    )["status"] == "completed"

    reserved = repo.reserve_resumed_attempt(
        child_id,
        physical_worker_id=child_id,
        owner_pid=os.getpid(),
        metadata={"child_session_id": f"child-new-{suffix}"},
    )
    assert reserved["status"] == "reserved"
    second_run = reserved["run_id"]
    second_attempt = reserved["attempt_id"]
    assert repo.transition_attempt(
        second_attempt,
        {"starting"},
        "completed",
        metadata={
            "child_session_id": f"child-new-{suffix}",
            "assistant_text_tail": "new tail",
            "events": [{"type": "tool.completed", "name": "new-tool"}],
        },
    )["status"] == "updated"
    assert repo.complete_run(
        second_run,
        {"type": "async_delegation", "delegation_id": delegation_id,
         "run_id": second_run, "status": "completed"},
        {"status": "completed", "summary": "new result"},
    )["status"] == "completed"
    return delegation_id, child_id, first_run, first_attempt, second_run, second_attempt


def test_status_and_tail_project_exact_resumed_attempts(monkeypatch):
    from tools.delegation_control import delegation_control

    monkeypatch.setattr(dt, "list_active_subagents", lambda: [])
    delegation_id, child_id, first_run, first_attempt, second_run, second_attempt = (
        _two_terminal_runs()
    )

    status = json.loads(delegation_control(
        action="status", delegation_id=delegation_id, session_key="owner"
    ))
    assert status["run_id"] == second_run
    assert status["latest_run_id"] == second_run
    assert status["active_run_id"] is None
    assert status["pending_run_count"] == 2
    assert status["subagents"][0]["attempt_id"] == second_attempt
    assert status["subagents"][0]["resume_available"] is True

    old_tail = json.loads(delegation_control(
        action="tail",
        delegation_id=delegation_id,
        attempt_id=first_attempt,
        session_key="owner",
    ))
    assert old_tail["run_id"] == first_run
    assert old_tail["subagents"][0]["attempt_id"] == first_attempt
    assert old_tail["subagents"][0]["assistant_text_tail"] == "old tail"
    assert old_tail["subagents"][0]["events"][0]["name"] == "old-tool"

    mismatch = json.loads(delegation_control(
        action="tail",
        delegation_id=delegation_id,
        subagent_id=f"other-{child_id}",
        attempt_id=first_attempt,
        session_key="owner",
    ))
    assert mismatch["status"] == "not_found"


def test_wait_is_fifo_by_default_and_exact_by_run(monkeypatch):
    monkeypatch.setattr(dt, "list_active_subagents", lambda: [])
    delegation_id, _child, first_run, _first_attempt, second_run, _second_attempt = (
        _two_terminal_runs()
    )
    first = ad.wait_for_delegation(
        delegation_id, session_key="owner", timeout_seconds=0
    )
    second = ad.wait_for_delegation(
        delegation_id, session_key="owner", timeout_seconds=0
    )
    assert first["run_id"] == first_run and first["claimed_delivery"] is True
    assert second["run_id"] == second_run and second["claimed_delivery"] is True

    exact_id, _child, old_run, _old_attempt, new_run, _new_attempt = _two_terminal_runs()
    exact = ad.wait_for_delegation(
        exact_id, session_key="owner", timeout_seconds=0, run_id=new_run
    )
    assert exact["run_id"] == new_run and exact["claimed_delivery"] is True
    assert ad._repository().inspect_delivery(exact_id, old_run)["delivery_state"] == "pending"


def test_abandon_suppresses_every_pending_run(monkeypatch):
    monkeypatch.setattr(dt, "list_active_subagents", lambda: [])
    delegation_id, _child, first_run, _first_attempt, second_run, _second_attempt = (
        _two_terminal_runs()
    )
    abandoned = ad.abandon_async_delegation(
        delegation_id, session_key="owner", reason="obsolete"
    )
    assert abandoned["status"] == "abandoned"
    assert ad._repository().inspect_delivery(
        delegation_id, first_run
    )["delivery_state"] == "suppressed"
    assert ad._repository().inspect_delivery(
        delegation_id, second_run
    )["delivery_state"] == "suppressed"


def _starting_steer_attempt(delegation_id: str, subagent_id: str):
    repository = ad._repository()
    initial = repository.register_initial_dispatch(
        {
            "delegation_id": delegation_id,
            "session_key": "owner",
            "dispatched_at": time.time(),
            "root_subagent_ids": [subagent_id],
        }
    )
    return repository, initial, initial["attempts"][0]


def test_deterministic_dispatch_steer_complete_resume_interrupt_smoke():
    suffix = uuid.uuid4().hex
    delegation_id = f"deleg-lifecycle-smoke-{suffix}"
    child_id = f"sa-lifecycle-smoke-{suffix}"
    repository, initial, first = _starting_steer_attempt(
        delegation_id, child_id
    )
    steered = repository.enqueue_steer(
        delegation_id, child_id, "owner", "finish with the latest constraint"
    )
    assert steered["status"] == "accepted"

    assert repository.transition_attempt(
        first["attempt_id"],
        {"starting"},
        "completed",
        metadata={"child_session_id": f"child-smoke-{suffix}"},
    )["status"] == "updated"
    assert repository.complete_run(
        initial["run_id"],
        {"type": "async_delegation", "delegation_id": delegation_id,
         "run_id": initial["run_id"], "status": "completed"},
        {"status": "completed", "summary": "first completion"},
    )["status"] == "completed"

    resumed = repository.reserve_resumed_attempt(
        child_id,
        physical_worker_id=child_id,
        owner_pid=os.getpid(),
        metadata={"child_session_id": f"child-smoke-{suffix}"},
    )
    assert resumed["status"] == "reserved"
    assert resumed["attempt_number"] == 2
    assert resumed["attempt_id"] != first["attempt_id"]
    assert resumed["run_id"] != initial["run_id"]
    current = repository.find_attempt(
        child_id, delegation_id=delegation_id, session_key="owner"
    )
    assert current["attempt_id"] == resumed["attempt_id"]
    assert current["logical_id"] == child_id

    interrupted = repository.request_interrupt(
        resumed["attempt_id"], "stop resumed attempt"
    )
    assert interrupted["status"] == "interrupt_requested"
    snapshot = repository.snapshot(delegation_id, session_key="owner")
    assert snapshot["children"][child_id]["attempt_id"] == resumed["attempt_id"]
    assert snapshot["children"][child_id]["status"] == "interrupt_requested"


def test_steer_queues_while_starting_then_forwards_in_order_and_acks_injection():
    from tools.delegation_control import delegation_control

    repository, initial, attempt = _starting_steer_attempt(
        "deleg-steer-starting", "sa-steer"
    )

    responses = [
        json.loads(
            delegation_control(
                action="steer",
                delegation_id="deleg-steer-starting",
                subagent_id="sa-steer",
                message=text,
                session_key="owner",
            )
        )
        for text in ("first guidance", "second guidance")
    ]
    assert [item["status"] for item in responses] == ["accepted", "accepted"]
    assert [item["steer_status"] for item in responses] == ["pending", "pending"]

    delivered = []
    agent = MagicMock()

    def accept(text, **kwargs):
        delivered.append((text, kwargs["mailbox_id"], kwargs["outcome_callback"]))
        return True

    agent.steer.side_effect = accept
    dt._register_subagent(
        {
            "subagent_id": "sa-steer",
            "delegation_attempt_id": attempt["attempt_id"],
            "delegation_run_id": initial["run_id"],
            "status": "running",
            "events": [],
            "assistant_text_tail": "",
            "agent": agent,
        }
    )

    assert [item[0] for item in delivered] == ["first guidance", "second guidance"]
    assert [repository.inspect_steer(item[1])["status"] for item in delivered] == [
        "forwarded",
        "forwarded",
    ]
    for _, _, callback in delivered:
        callback("injected")
    assert [repository.inspect_steer(item[1])["status"] for item in delivered] == [
        "injected",
        "injected",
    ]
    observed = json.loads(
        delegation_control(
            action="status",
            delegation_id="deleg-steer-starting",
            subagent_id="sa-steer",
            session_key="owner",
        )
    )["subagents"][0]["steers"]
    assert [item["status"] for item in observed] == ["injected", "injected"]
    assert all("message" not in item for item in observed)


def test_live_foreground_wait_rejects_normal_steer_with_force_hint():
    from tools.delegation_control import delegation_control

    repository, initial, attempt = _starting_steer_attempt(
        "deleg-steer-foreground", "sa-foreground"
    )

    class WaitingAgent:
        def request_durable_steer(
            self, _text, *, outcome_callback, force, **_kwargs
        ):
            assert force is False
            outcome_callback("foreground_wait")
            return {"status": "foreground_wait", "wait_kinds": ["terminal"]}

    dt._register_subagent(
        {
            "subagent_id": "sa-foreground",
            "delegation_attempt_id": attempt["attempt_id"],
            "delegation_run_id": initial["run_id"],
            "status": "running",
            "events": [],
            "assistant_text_tail": "",
            "agent": WaitingAgent(),
        }
    )

    result = json.loads(
        delegation_control(
            action="steer",
            delegation_id="deleg-steer-foreground",
            subagent_id="sa-foreground",
            message="change direction",
            force=False,
            session_key="owner",
        )
    )

    assert result["status"] == "foreground_wait"
    assert result["steer_status"] == "foreground_wait"
    assert result["wait_kinds"] == ["terminal"]
    assert "force=true" in result["hint"]
    assert repository.inspect_steer(result["mailbox_id"])["status"] == "foreground_wait"



def test_steer_rejects_foreign_terminal_and_interrupt_wins_startup_race():
    from tools.delegation_control import delegation_control

    repository, initial, attempt = _starting_steer_attempt(
        "deleg-steer-races", "sa-race"
    )
    foreign = json.loads(
        delegation_control(
            action="steer",
            delegation_id="deleg-steer-races",
            subagent_id="sa-race",
            message="foreign",
            session_key="foreign",
        )
    )
    assert foreign["status"] == "not_found"

    queued = json.loads(
        delegation_control(
            action="steer",
            delegation_id="deleg-steer-races",
            subagent_id="sa-race",
            message="must lose to interrupt",
            session_key="owner",
        )
    )
    assert ad.request_pending_subagent_interrupt(
        "deleg-steer-races", "sa-race", session_key="owner", reason="stop"
    ) == "interrupt_requested"
    assert repository.inspect_steer(queued["mailbox_id"])["status"] == (
        "superseded_by_interrupt"
    )

    agent = MagicMock()
    dt._register_subagent(
        {
            "subagent_id": "sa-race",
            "delegation_attempt_id": attempt["attempt_id"],
            "delegation_run_id": initial["run_id"],
            "status": "running",
            "events": [],
            "assistant_text_tail": "",
            "agent": agent,
        }
    )
    agent.interrupt.assert_called_once()
    agent.steer.assert_not_called()

    terminal_repo, _, terminal_attempt = _starting_steer_attempt(
        "deleg-steer-terminal", "sa-terminal"
    )
    assert terminal_repo.transition_attempt(
        terminal_attempt["attempt_id"], {"starting"}, "completed"
    )["status"] == "updated"
    terminal = json.loads(
        delegation_control(
            action="steer",
            delegation_id="deleg-steer-terminal",
            subagent_id="sa-terminal",
            message="too late",
            session_key="owner",
        )
    )
    assert terminal["status"] == "already_terminal"
    assert terminal["terminal_status"] == "completed"


@pytest.mark.parametrize(
    ("winner", "expected"),
    [
        ("interrupt", "superseded_by_interrupt"),
        ("completion", "too_late_after_completion"),
    ],
)
def test_claimed_steer_reports_honest_race_outcome(monkeypatch, winner, expected):
    from tools.delegation_control import delegation_control

    repository, initial, attempt = _starting_steer_attempt(
        f"deleg-steer-{winner}", f"sa-{winner}"
    )
    claim_started = threading.Event()
    release_claim = threading.Event()
    real_claim = type(repository).claim_steer

    def blocked_claim(self, attempt_id, mailbox_id):
        outcome = real_claim(self, attempt_id, mailbox_id)
        claim_started.set()
        assert release_claim.wait(2)
        return outcome

    monkeypatch.setattr(type(repository), "claim_steer", blocked_claim)

    class RaceAgent:
        def __init__(self):
            self.interrupted = False

        def interrupt(self, _reason):
            self.interrupted = True

        def steer(self, _text, *, outcome_callback, **_kwargs):
            if self.interrupted:
                outcome_callback("superseded_by_interrupt")
                return False
            return True

    agent = RaceAgent()
    child_id = f"sa-{winner}"
    dt._register_subagent(
        {
            "subagent_id": child_id,
            "delegation_attempt_id": attempt["attempt_id"],
            "delegation_run_id": initial["run_id"],
            "status": "running",
            "events": [],
            "assistant_text_tail": "",
            "agent": agent,
        }
    )
    responses = []

    def request_steer():
        responses.append(
            json.loads(
                delegation_control(
                    action="steer",
                    delegation_id=f"deleg-steer-{winner}",
                    subagent_id=child_id,
                    message="race guidance",
                    session_key="owner",
                )
            )
        )

    thread = threading.Thread(target=request_steer)
    thread.start()
    assert claim_started.wait(2)
    if winner == "interrupt":
        interrupted = json.loads(
            delegation_control(
                action="interrupt",
                delegation_id="deleg-steer-interrupt",
                subagent_id=child_id,
                session_key="owner",
            )
        )
        assert interrupted["status"] == "interrupt_requested"
    else:
        repository.complete_run(
            initial["run_id"],
            {
                "completed_at": time.time(),
                "status": "completed",
                "children": [
                    {
                        "subagent_id": child_id,
                        "attempt_id": attempt["attempt_id"],
                        "status": "completed",
                    }
                ],
            },
            {"status": "completed"},
        )
    release_claim.set()
    thread.join(2)

    assert not thread.is_alive()
    assert responses[0]["status"] == expected
    assert responses[0]["steer_status"] == expected


def test_compressed_parent_session_retains_control_without_root_scope_leak():
    from hermes_state import SessionDB
    from tools.delegation_control import delegation_control

    session_db = SessionDB()
    session_db.create_session("owner", source="discord")
    session_db.end_session("owner", "compression")
    session_db.create_session(
        "owner-tip", source="discord", parent_session_id="owner"
    )
    session_db.create_session(
        "owner-branch",
        source="discord",
        parent_session_id="owner",
        model_config={"_branched_from": "owner"},
    )
    dispatched = _dispatch(lambda: {"status": "completed", "summary": "ok"})
    delegation_id = dispatched["delegation_id"]
    _wait_terminal(delegation_id)

    status = json.loads(
        delegation_control(
            action="status", delegation_id=delegation_id, session_key="owner-tip"
        )
    )
    listed = json.loads(delegation_control(action="list", session_key="owner-tip"))
    branch = json.loads(
        delegation_control(
            action="status",
            delegation_id=delegation_id,
            session_key="owner-branch",
        )
    )

    assert status["delegation_id"] == delegation_id
    assert [item["delegation_id"] for item in listed["delegations"]] == [
        delegation_id
    ]
    assert branch["status"] == "not_found"


def test_list_and_status_hide_foreign_session_like_unknown():
    release = threading.Event()
    dispatched = _dispatch(
        lambda: (release.wait(2), {"status": "completed", "summary": "done"})[1],
        roots=["sa-owner"],
    )
    from tools.delegation_control import delegation_control

    try:
        own = json.loads(delegation_control(action="list", session_key="owner"))
        foreign = json.loads(delegation_control(action="list", session_key="foreign"))
        hidden = json.loads(
            delegation_control(
                action="status",
                delegation_id=dispatched["delegation_id"],
                session_key="foreign",
            )
        )
        unknown = json.loads(
            delegation_control(
                action="status", delegation_id="deleg_unknown", session_key="foreign"
            )
        )
        assert [item["delegation_id"] for item in own["delegations"]] == [
            dispatched["delegation_id"]
        ]
        assert foreign["delegations"] == []
        assert hidden == unknown | {"delegation_id": dispatched["delegation_id"]}
    finally:
        release.set()


def test_force_flag_survives_starting_mailbox_until_live_forwarding():
    from tools.delegation_control import delegation_control

    repository, initial, attempt = _starting_steer_attempt(
        "deleg-steer-force-starting", "sa-force-starting"
    )
    queued = json.loads(
        delegation_control(
            action="steer",
            delegation_id="deleg-steer-force-starting",
            subagent_id="sa-force-starting",
            message="persist force",
            force=True,
            session_key="owner",
        )
    )
    assert queued["status"] == "accepted"
    assert repository.inspect_steer(queued["mailbox_id"])["force"] == 1

    class RecordingAgent:
        def __init__(self):
            self.interrupt_event = threading.Event()
            self.calls = []

        def request_durable_steer(
            self, text, *, mailbox_id, outcome_callback, force=False
        ):
            self.calls.append((text, mailbox_id, force))
            return {"status": "accepted", "wait_kinds": []}

    agent = RecordingAgent()
    dt._register_subagent(
        {
            "subagent_id": "sa-force-starting",
            "delegation_attempt_id": attempt["attempt_id"],
            "delegation_run_id": initial["run_id"],
            "status": "running",
            "events": [],
            "assistant_text_tail": "",
            "agent": agent,
        }
    )
    try:
        assert agent.calls == [("persist force", queued["mailbox_id"], True)]
        assert repository.inspect_steer(queued["mailbox_id"])["status"] == "forwarded"
    finally:
        dt._unregister_subagent(
            "sa-force-starting", str(attempt["attempt_id"])
        )

def test_starting_child_interrupt_is_queued_then_applied_once():
    runner_started = threading.Event()
    allow_registration = threading.Event()
    allow_finish = threading.Event()
    registered = threading.Event()
    agent = MagicMock()

    def runner():
        runner_started.set()
        assert allow_registration.wait(2)
        dt._register_subagent(
            {
                "subagent_id": "sa-starting",
                "parent_id": None,
                "depth": 0,
                "goal": "controlled task",
                "status": "running",
                "started_at": time.time(),
                "agent": agent,
            }
        )
        registered.set()
        assert allow_finish.wait(2)
        dt._unregister_subagent("sa-starting")
        return {"status": "interrupted", "summary": "stopped"}

    dispatched = _dispatch(runner, roots=["sa-starting"])
    assert runner_started.wait(2)
    from tools.delegation_control import delegation_control

    try:
        payload = json.loads(
            delegation_control(
                action="interrupt",
                delegation_id=dispatched["delegation_id"],
                subagent_id="sa-starting",
                session_key="owner",
                reason="stop before startup",
            )
        )
        assert payload["status"] == "interrupt_requested"
        agent.interrupt.assert_not_called()
        allow_registration.set()
        assert registered.wait(2)
        agent.interrupt.assert_called_once_with("stop before startup")
    finally:
        allow_registration.set()
        allow_finish.set()

@pytest.mark.parametrize("supplied_attempt", ["stale", None, "", 17, "attempt_missing"])
def test_invalid_attempt_registration_and_archive_fail_closed(supplied_attempt):
    repository = ad._repository()
    initial = repository.register_initial_dispatch(
        {
            "delegation_id": "deleg-exact-attempt",
            "session_key": "owner",
            "origin_ui_session_id": "ui-owner",
            "parent_session_id": "parent-owner",
            "dispatched_at": 1.0,
            "goal": "original",
            "root_subagent_ids": ["sa-exact"],
        }
    )
    stale_attempt_id = initial["attempts"][0]["attempt_id"]
    assert repository.transition_attempt(
        stale_attempt_id, {"starting"}, "completed"
    )["status"] == "updated"
    resumed = repository.reserve_resumed_attempt(
        "sa-exact", physical_worker_id="worker-current"
    )

    invalid_record = {
        "subagent_id": "sa-exact",
        "delegation_attempt_id": (
            stale_attempt_id if supplied_attempt == "stale" else supplied_attempt
        ),
        "delegation_run_id": initial["run_id"],
        "parent_id": None,
        "goal": "invalid callback",
        "status": "running",
        "started_at": time.time(),
        "agent": MagicMock(),
    }
    dt._register_subagent(invalid_record)
    invalid_record["status"] = "error"
    dt._unregister_subagent("sa-exact")

    current = repository.snapshot("deleg-exact-attempt")["children"]["sa-exact"]
    assert current["attempt_id"] == resumed["attempt_id"]
    assert current["run_id"] == resumed["run_id"]
    assert current["status"] == "starting"
    assert current.get("goal") == "original"


def test_stale_attempt_callbacks_cannot_replace_mutate_or_remove_live_resume():
    repository = ad._repository()
    initial = repository.register_initial_dispatch(
        {
            "delegation_id": "deleg-live-resume",
            "session_key": "owner",
            "dispatched_at": 1.0,
            "root_subagent_ids": ["sa-resumed"],
        }
    )
    old_attempt = initial["attempts"][0]["attempt_id"]
    assert repository.transition_attempt(
        old_attempt, {"starting"}, "completed"
    )["status"] == "updated"
    resumed = repository.reserve_resumed_attempt(
        "sa-resumed", physical_worker_id="sa-resumed"
    )
    current_record = {
        "subagent_id": "sa-resumed",
        "delegation_attempt_id": resumed["attempt_id"],
        "delegation_run_id": resumed["run_id"],
        "status": "running",
        "events": [],
        "assistant_text_tail": "current",
        "agent": MagicMock(),
    }
    dt._register_subagent(current_record)

    stale_record = {
        "subagent_id": "sa-resumed",
        "delegation_attempt_id": old_attempt,
        "delegation_run_id": initial["run_id"],
        "status": "running",
        "agent": MagicMock(),
    }
    dt._register_subagent(stale_record)
    dt._unregister_subagent("sa-resumed", old_attempt)

    with dt._active_subagents_lock:
        assert dt._active_subagents["sa-resumed"] is current_record
        assert current_record["events"] == []
        assert current_record["assistant_text_tail"] == "current"


def test_nested_child_without_attempt_id_allocates_propagates_and_archives_exact_id():
    repository = ad._repository()
    repository.register_initial_dispatch(
        {
            "delegation_id": "deleg-nested-attempt",
            "session_key": "owner",
            "dispatched_at": 1.0,
            "root_subagent_ids": ["sa-parent"],
        }
    )
    parent = dict(
        subagent_id="sa-parent", parent_id=None, status="running", agent=MagicMock()
    )
    child = {
        "subagent_id": "sa-child",
        "parent_id": "sa-parent",
        "goal": "nested",
        "status": "running",
        "agent": MagicMock(),
    }
    dt._register_subagent(parent)
    dt._register_subagent(child)
    attempt_id = child.get("delegation_attempt_id")
    assert isinstance(attempt_id, str) and attempt_id
    child.update(status="completed", tool_count=3)
    dt._unregister_subagent("sa-child")
    archived = repository.snapshot("deleg-nested-attempt")["children"]["sa-child"]
    assert (archived["attempt_id"], archived["status"], archived["tool_count"]) == (
        attempt_id, "completed", 3
    )

def test_cross_process_interrupt_cannot_claim_owner_callback_success():
    release = threading.Event()
    callback_entered, release_callback = threading.Event(), threading.Event()
    successful_calls = []

    def failed_interrupt():
        callback_entered.set()
        assert release_callback.wait(5)
        raise RuntimeError("signal failed")

    dispatched = _dispatch(
        lambda: (release.wait(10), {"status": "completed", "summary": "done"})[1],
        roots=["sa-interrupt-retry"],
        interrupt_fn=failed_interrupt,
    )
    delegation_id = dispatched["delegation_id"]
    try:
        owner_outcome = []
        owner = threading.Thread(target=lambda: owner_outcome.append(
            ad.interrupt_async_delegation(delegation_id, session_key="owner", reason="first")
        ))
        owner.start()
        assert callback_entered.wait(2)
        ctx = multiprocessing.get_context("spawn"); output = ctx.Queue()
        observer = ctx.Process(target=_observe_interrupt, args=(os.environ["HERMES_HOME"], delegation_id, output))
        observer.start()
        observer.join(10)
        assert observer.exitcode == 0
        assert output.get(timeout=1)["status"] == "interrupt_unavailable"
        release_callback.set()
        owner.join(2)
        assert owner_outcome[0]["status"] == "interrupt_failed"
        snapshot = ad.get_durable_delegation(delegation_id)
        assert (snapshot["state"], snapshot["interrupt_requests"]) == ("running", {})

        with ad._records_lock:
            ad._records[delegation_id]["interrupt_fn"] = lambda: successful_calls.append(True)
        assert ad.interrupt_async_delegation(
            delegation_id, session_key="owner", reason="second"
        )["status"] == "interrupt_requested"
        assert successful_calls == [True]
        assert ad.take_pending_subagent_interrupt("sa-interrupt-retry") == (True, "second")
    finally:
        release_callback.set()
        release.set()

def test_concurrent_interrupt_callbacks_serialize_failure_then_success(monkeypatch):
    release = threading.Event()
    callback_rendezvous, successful_callback = threading.Barrier(2), threading.Event()
    callback_guard, counts = threading.Lock(), [0, 0, 0]  # calls, active, peak

    def interrupt_callback():
        with callback_guard:
            counts[0] += 1
            call_number = counts[0]
            counts[1] += 1
            counts[2] = max(counts[2], counts[1])
        try:
            try:
                callback_rendezvous.wait(timeout=0.5)
                if call_number == 1:
                    assert successful_callback.wait(2)
            except threading.BrokenBarrierError:
                pass
            if call_number == 1:
                raise RuntimeError("first signal failed")
            successful_callback.set()
        finally:
            with callback_guard:
                counts[1] -= 1

    dispatched = _dispatch(
        lambda: (release.wait(10), {"status": "completed", "summary": "done"})[1],
        roots=["sa-concurrent-interrupt"],
        interrupt_fn=interrupt_callback,
    )
    delegation_id = dispatched["delegation_id"]
    try:
        outcomes = _interrupt_twice(monkeypatch, delegation_id)
        statuses = sorted(item["status"] for item in outcomes)
        assert statuses == ["interrupt_failed", "interrupt_requested"]
        assert counts == [2, 0, 1]
        snapshot = ad.get_durable_delegation(delegation_id)
        assert snapshot["state"] == "interrupt_requested"
        assert list(snapshot["interrupt_requests"]) == ["sa-concurrent-interrupt"]
        pending_reason = snapshot["interrupt_requests"]["sa-concurrent-interrupt"]
        assert ad.take_pending_subagent_interrupt("sa-concurrent-interrupt") == (
            True, pending_reason
        )
        assert ad.take_pending_subagent_interrupt("sa-concurrent-interrupt") == (False, "")
    finally:
        release.set()

def test_concurrent_idempotent_interrupt_waits_for_callback_owner(monkeypatch):
    release = threading.Event()
    callback_entered, release_callback = threading.Event(), threading.Event()
    callback_calls = []

    def successful_interrupt():
        callback_calls.append(True)
        callback_entered.set()
        assert release_callback.wait(2)

    dispatched = _dispatch(
        lambda: (release.wait(10), {"status": "completed", "summary": "done"})[1],
        roots=["sa-idempotent-interrupt"],
        interrupt_fn=successful_interrupt,
    )
    delegation_id = dispatched["delegation_id"]

    def release_owner(outcomes, returned):
        assert callback_entered.wait(2)
        assert not returned.wait(0.2)
        assert outcomes == []
        release_callback.set()

    try:
        outcomes = _interrupt_twice(monkeypatch, delegation_id, release_owner)
        assert [item["status"] for item in outcomes] == ["interrupt_requested"] * 2
        assert callback_calls == [True]
        assert ad.take_pending_subagent_interrupt("sa-idempotent-interrupt")[0] is True
        assert ad.take_pending_subagent_interrupt("sa-idempotent-interrupt") == (False, "")
    finally:
        release_callback.set()
        release.set()


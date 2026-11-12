"""Opt-in ``routing: auto`` (one router call picks who answers) and Continue at the discussion cap."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_agent.gateway import hosted_room_discussion as discussion
from hermes_agent.gateway import hosted_room_driver as driver
from hermes_agent.gateway import hosted_room_router as router
from hermes_agent.gateway import hosted_rooms
from hermes_agent.gateway.hosted_room_context import set_room_context
from tests.waiting import wait_until
from hermes_agent.tui_gateway import hosted_room_router_call as router_call
from hermes_agent.tui_gateway.hosted_room_service import HostedRoomService

PROFILES = ("research", "build", "review")
ROOM_ID = "router-room"


def _service(root: Path) -> HostedRoomService:
    server = SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())
    service = HostedRoomService(server, db_path=root / "state.db")
    service.local_profiles = lambda: PROFILES
    return service


def _room(service: HostedRoomService, routing: str = "auto") -> None:
    service.create_room(room_id=ROOM_ID, name="Router test", members=[
        {"member_id": profile, "profile": profile, "handle": profile} for profile in PROFILES])
    if routing != "everyone":
        set_room_context(service.db_path, room_id=ROOM_ID, event_id="ctx", purpose="Ship releases", notes="",
                         routing=routing, lead_member_id="research" if routing == "lead" else None)


def _send(service: HostedRoomService, event_id: str, text: str, thread_id: str | None = None) -> None:
    service.send(room_id=ROOM_ID, event_id=event_id, payload={"text": text, "thread_id": thread_id or event_id})


def _wait_router(*services: HostedRoomService) -> None:
    wait_until(lambda: not any(service._router_calls.inflight((ROOM_ID, f"user:{key}"))
                               or service._router_calls.inflight((ROOM_ID, key))
                               for service in services for key in ("m1", "m2", "m3")),
               what="router call finished")


def _queued(service: HostedRoomService) -> list[str]:
    return [task["payload"]["target_member_id"]
            for task in driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")]


def _kinds(service: HostedRoomService, kind: str) -> list[dict]:
    return [event for event in service._events(ROOM_ID) if event["kind"] == kind]


def _settle_queued(service: HostedRoomService, text: str, index: int) -> None:
    binding = service.bindings()[0]
    task = driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")[0]
    lease = driver.acquire_lease(service.db_path, room_id=ROOM_ID, gateway_id=binding.gateway_id,
                                 authority_epoch=binding.authority_epoch, process_generation="router-test",
                                 ttl_seconds=30, clock=time.time)
    attempt = driver.start_task(service.db_path, task["identity"], lease, expected_cancel_generation=0, clock=time.time)
    driver.settle_task(service.db_path, attempt, settlement_id=f"s-{index}", status="settled",
                       result={"text": text}, clock=time.time)
    service.prepare_room(binding)


@pytest.fixture
def answers(monkeypatch):
    """Scripted router answers; each call records what the router was asked."""
    calls: list[dict] = []
    script: list = []

    def ask(**kwargs):
        calls.append(kwargs)
        answer = script.pop(0) if script else None
        if isinstance(answer, BaseException):
            raise answer
        return answer(**kwargs) if callable(answer) else answer
    monkeypatch.setattr(router_call, "ask_router", ask)
    return SimpleNamespace(calls=calls, script=script)


def test_auto_room_records_the_routers_pick_before_any_turn_and_only_that_member_answers(tmp_path, answers):
    service = _service(tmp_path)
    _room(service)
    answers.script.append(("build",))
    _send(service, "m1", "Can someone fix the CI build?")
    _wait_router(service)
    routed = _kinds(service, "room.routed")
    assert [event["payload"] for event in routed] == [{
        "discussion_event_id": routed[0]["payload"]["discussion_event_id"], "thread_id": "m1",
        "member_ids": ["build"], "everyone": False, "fallback": False}]
    service.prepare_room(service.bindings()[0])
    assert _queued(service) == ["build"]
    # Replay from a fresh service (a gateway restart) plans the identical task and never re-asks.
    restarted = _service(tmp_path)
    restarted.prepare_room(restarted.bindings()[0])
    assert _queued(restarted) == ["build"] and len(answers.calls) == 1


@pytest.mark.parametrize("answer", [None, RuntimeError("provider down")])
def test_router_failure_falls_back_to_everyone(tmp_path, answers, answer):
    service = _service(tmp_path)
    _room(service)
    answers.script.append(answer)
    _send(service, "m1", "Thoughts on the roadmap?")
    _wait_router(service)
    routed = _kinds(service, "room.routed")
    if isinstance(answer, BaseException):
        # A call that raised inside ask_router is a lost answer: the next preparation asks again.
        assert routed == []
        answers.script.append(None)
        service.prepare_room(service.bindings()[0])
        _wait_router(service)
        routed = _kinds(service, "room.routed")
    assert routed[0]["payload"]["everyone"] is True and routed[0]["payload"]["fallback"] is True
    service.prepare_room(service.bindings()[0])
    assert _queued(service) == ["research"]  # the everyone round, first member first


@pytest.mark.parametrize("text", ["@review please check", "@all status?", "@everyone status?"])
def test_deterministic_rules_never_call_the_router(tmp_path, answers, text):
    service = _service(tmp_path)
    _room(service)
    _send(service, "m1", text)
    _wait_router(service)
    assert answers.calls == [] and _kinds(service, "room.routed") == []
    assert _queued(service) == (["review"] if text.startswith("@review") else ["research"])


def test_a_reply_in_an_existing_thread_is_not_routed(tmp_path, answers):
    service = _service(tmp_path)
    _room(service)
    answers.script.append(("build",))
    _send(service, "m1", "Fix CI")
    _wait_router(service)
    service.prepare_room(service.bindings()[0])
    _settle_queued(service, "Done.", 0)
    _send(service, "m2", "Thanks, and the docs too", thread_id="m1")
    _wait_router(service)
    assert len(answers.calls) == 1


def test_two_workers_on_one_store_make_one_router_call(tmp_path, answers):
    """Workers sharing the room store each prepare the room before any lease; only one pays the router."""
    first, second = _service(tmp_path), _service(tmp_path)
    _room(first)
    release = threading.Event()

    def slow(**_kwargs):
        assert release.wait(10)
        return ("build",)
    answers.script.append(slow)
    _send(first, "m1", "Fix CI")
    wait_until(lambda: answers.calls, what="first worker asked the router")
    second.prepare_room(second.bindings()[0])
    release.set()
    _wait_router(first, second)
    assert len(answers.calls) == 1 and len(_kinds(first, "room.routed")) == 1
    second.prepare_room(second.bindings()[0])
    assert _queued(second) == ["build"]


def test_a_router_claim_left_by_a_dead_worker_expires(tmp_path):
    now = [1000.0]
    key, asked = (ROOM_ID, "user:m1"), []
    dead = router_call.RouterCalls(tmp_path / "state.db", clock=lambda: now[0])
    hung = threading.Event()
    assert dead.start(key, lambda: hung.wait(10) and None, lambda _picked: None)
    live = router_call.RouterCalls(tmp_path / "state.db", clock=lambda: now[0])
    done = threading.Event()
    assert not live.start(key, lambda: asked.append(1), lambda _picked: done.set())
    now[0] += router_call.CLAIM_TTL_SECONDS + 1
    assert live.start(key, lambda: asked.append(1), lambda _picked: done.set())
    assert done.wait(10) and asked == [1]
    hung.set()
    wait_until(lambda: not dead.inflight(key), what="dead worker's call returned")


def test_stop_during_the_router_call_discards_its_answer(tmp_path, answers):
    service = _service(tmp_path)
    _room(service)
    release = threading.Event()

    def slow(**_kwargs):
        assert release.wait(10)
        return ("build",)
    answers.script.append(slow)
    _send(service, "m1", "Fix CI")
    service.stop_room(ROOM_ID, cancel_id="stop-1")
    release.set()
    _wait_router(service)
    service.prepare_room(service.bindings()[0])
    assert _kinds(service, "room.routed") == [] and _queued(service) == []


def test_stop_between_the_answer_check_and_its_write_records_no_route(tmp_path, answers):
    """Stop appends without the policy lock: one landing after the answer's re-plan must not get a
    route recorded over it."""
    service = _service(tmp_path)
    _room(service)
    release = threading.Event()

    def slow(**_kwargs):
        assert release.wait(10)
        return ("build",)
    answers.script.append(slow)
    _send(service, "m1", "Fix CI")
    gateway_id, epoch = service._owned_authority(ROOM_ID)
    read, stopped = service._policy_snapshot, []

    def snapshot(room):
        result = read(room)
        if threading.current_thread().name.startswith("hosted-room-router") and not stopped:
            stopped.append(hosted_rooms.request_room_stop(
                service.db_path, room_id=ROOM_ID, cancel_id="stop-gap", expected_gateway_id=gateway_id,
                expected_epoch=epoch))
        return result
    service._policy_snapshot = snapshot
    release.set()
    _wait_router(service)
    assert stopped and _kinds(service, "room.routed") == []


@pytest.mark.parametrize("routing", ["everyone", "lead"])
def test_legacy_rooms_never_call_the_router(tmp_path, answers, routing):
    service = _service(tmp_path)
    _room(service, routing)
    _send(service, "m1", "Fix CI")
    _wait_router(service)
    assert answers.calls == [] and _kinds(service, "room.routed") == []
    assert _queued(service) == ["research"]


def test_router_prompt_carries_roles_and_recent_messages_and_answer_parsing_is_strict(tmp_path):
    prompt = router.router_prompt(
        room_name="Ops", purpose="Ship", notes="", roster=[
            {"handle": "build", "name": "Builder", "description": "CI and packaging"}],
        recent=[{"author": "user", "text": "hello " * 200}], message="Fix CI")
    assert "@build (Builder): CI and packaging" in prompt and "..." in prompt and "Fix CI" in prompt
    handles = {"build": "m-build", "review": "m-review"}
    assert router.parse_router_answer('{"members": ["@Build"]}', handles) == ("m-build",)
    assert router.parse_router_answer('{"members": "everyone"}', handles) == ()
    for bad in ('{"members": ["nobody"]}', '{"members": []}', '{"members": ["build"], "why": "x"}',
                "build", '{"assign": ["build"]}', '{"members": ["build", "build"]}'):
        assert router.parse_router_answer(bad, handles) is None, bad


# -- Continue at the cap ------------------------------------------------------------------------


def _bounded_room(tmp_path: Path) -> HostedRoomService:
    (tmp_path / "config.yaml").write_text("hosted_rooms:\n  discussion:\n    max_rounds: 3\n    max_messages: 2\n")
    service = _service(tmp_path)
    _room(service, "everyone")
    _send(service, "m1", "Discuss.")
    for index in range(2):
        _settle_queued(service, "@everyone next", index)
    return service


def test_the_cap_pauses_and_continue_grants_one_more_budget_even_after_a_restart(tmp_path):
    service = _bounded_room(tmp_path)
    bounded = _kinds(service, "room.activity")
    assert [event["payload"]["status"] for event in bounded] == ["bounded"] and _queued(service) == []
    discussion_id = bounded[0]["payload"]["discussion_event_id"]
    service = _service(tmp_path)  # a gateway restart between the pause and the click
    event = service.continue_room(ROOM_ID, discussion_event_id=discussion_id)
    assert event["kind"] == "room.continued" and event["payload"]["grant"] == 1
    assert len(_queued(service)) == 1
    for index in range(2, 4):
        _settle_queued(service, "@everyone more", index)
    activity = _kinds(service, "room.activity")
    assert [(e["payload"]["status"], e["event_id"].endswith(":c1")) for e in activity] == [
        ("bounded", False), ("bounded", True)]
    assert len(_kinds(service, "message.member")) == 4 and _queued(service) == []


@pytest.mark.parametrize("after", ["stop", "new_message", "twice"])
def test_continue_is_refused_once_the_pause_is_moot(tmp_path, after):
    service = _bounded_room(tmp_path)
    discussion_id = _kinds(service, "room.activity")[0]["payload"]["discussion_event_id"]
    if after == "stop":
        service.stop_room(ROOM_ID, cancel_id="stop")
    elif after == "new_message":
        _send(service, "m2", "never mind", thread_id="m1")
    else:
        service.continue_room(ROOM_ID, discussion_event_id=discussion_id)
    with pytest.raises(Exception, match="not paused"):
        service.continue_room(ROOM_ID, discussion_event_id=discussion_id)


def _land_in_the_continue_gap(service: HostedRoomService, monkeypatch, *appends) -> None:
    """Run ``appends`` (raw log writes, as Stop and Send make without the policy lock) right after
    continue_room's first read of the log: the deterministic interleave of a racing Stop/Send."""
    read = service._events
    pending = list(appends)

    def events(room_id):
        rows = read(room_id)
        while pending:
            pending.pop(0)()
        return rows
    monkeypatch.setattr(service, "_events", events)


def _raw_user_message(service: HostedRoomService, event_id: str, thread_id: str):
    gateway_id, epoch = service._owned_authority(ROOM_ID)
    return lambda: hosted_rooms.append_event(
        service.db_path, room_id=ROOM_ID, event_id=event_id, kind="message.user",
        actor={"kind": "user", "id": "desktop"}, payload={"text": "new ask", "thread_id": thread_id},
        authority_gateway_id=gateway_id, authority_epoch=epoch)


def test_stop_and_a_new_message_racing_continue_keep_the_new_message(tmp_path, monkeypatch):
    service = _bounded_room(tmp_path)
    discussion_id = _kinds(service, "room.activity")[0]["payload"]["discussion_event_id"]
    gateway_id, epoch = service._owned_authority(ROOM_ID)
    _land_in_the_continue_gap(
        service, monkeypatch,
        lambda: hosted_rooms.request_room_stop(service.db_path, room_id=ROOM_ID, cancel_id="stop-gap",
                                               expected_gateway_id=gateway_id, expected_epoch=epoch),
        _raw_user_message(service, "m2", "m1"))
    with pytest.raises(hosted_rooms.HostedRoomError, match="not paused"):
        service.continue_room(ROOM_ID, discussion_event_id=discussion_id)
    assert _kinds(service, "room.continued") == []
    service.prepare_room(service.bindings()[0])
    assert _queued(service) == ["research"]  # the new message's own discussion


def test_an_unrelated_event_in_the_continue_gap_still_continues(tmp_path, monkeypatch):
    service = _bounded_room(tmp_path)
    discussion_id = _kinds(service, "room.activity")[0]["payload"]["discussion_event_id"]
    _land_in_the_continue_gap(service, monkeypatch, _raw_user_message(service, "other", "other"))
    event = service.continue_room(ROOM_ID, discussion_event_id=discussion_id)
    assert event["kind"] == "room.continued" and event["seq"] == max(e["seq"] for e in service._events(ROOM_ID))


def test_a_stop_before_a_late_cap_notice_refuses_continue(tmp_path, monkeypatch):
    """The cap notice is written from a snapshot; a Stop can land just before it. That Stop still fences
    the discussion, so Continue (which could never reopen it) is refused rather than doing nothing."""
    (tmp_path / "config.yaml").write_text("hosted_rooms:\n  discussion:\n    max_rounds: 3\n    max_messages: 2\n")
    service = _service(tmp_path)
    _room(service, "everyone")
    _send(service, "m1", "Discuss.")
    _settle_queued(service, "@everyone next", 0)
    gateway_id, epoch = service._owned_authority(ROOM_ID)
    status = service._append_room_status

    def stop_first(room, decision):
        hosted_rooms.request_room_stop(service.db_path, room_id=ROOM_ID, cancel_id="late", expected_gateway_id=gateway_id,
                                       expected_epoch=epoch)
        status(room, decision)
    monkeypatch.setattr(service, "_append_room_status", stop_first)
    _settle_queued(service, "@everyone next", 1)
    kinds = [event["kind"] for event in service._events(ROOM_ID)]
    assert kinds.index("room.stop_requested") < kinds.index("room.activity")
    discussion_id = _kinds(service, "room.activity")[0]["payload"]["discussion_event_id"]
    with pytest.raises(hosted_rooms.HostedRoomError, match="not paused"):
        service.continue_room(ROOM_ID, discussion_event_id=discussion_id)


def test_continue_stops_at_the_continue_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(router, "MAX_DISCUSSION_CONTINUES", 1)
    service = _bounded_room(tmp_path)
    discussion_id = _kinds(service, "room.activity")[0]["payload"]["discussion_event_id"]
    service.continue_room(ROOM_ID, discussion_event_id=discussion_id)
    for index in range(2, 4):
        _settle_queued(service, "@everyone more", index)
    with pytest.raises(Exception, match="every Continue"):
        service.continue_room(ROOM_ID, discussion_event_id=discussion_id)


def test_status_reports_continue_limit_and_routing_modes(tmp_path):
    service = _service(tmp_path)
    _room(service, "everyone")
    status = service.status(ROOM_ID)
    assert status["max_continues"] == router.MAX_DISCUSSION_CONTINUES
    assert status["routing_modes"] == ["everyone", "lead", "auto"]


def test_planner_is_pure_over_a_routed_and_continued_log(tmp_path, answers):
    service = _service(tmp_path)
    _room(service)
    answers.script.append(("build", "review"))
    _send(service, "m1", "Fix CI and review it")
    _wait_router(service)
    events, room = service._events(ROOM_ID), service._room(ROOM_ID)
    first = discussion.plan_next_task(room, events, local_profiles=PROFILES,
                                      room_context={"routing": "auto", "purpose": "Ship releases"})
    again = discussion.plan_next_task(room, list(events), local_profiles=PROFILES,
                                      room_context={"routing": "auto", "purpose": "Ship releases"})
    assert first == again and first.task.member.member_id == "build"
    pending = discussion.plan_next_task(room, [e for e in events if e["kind"] != "room.routed"],
                                        local_profiles=PROFILES, room_context={"routing": "auto"})
    assert (pending.status, pending.reason) == ("route", "router_pending")

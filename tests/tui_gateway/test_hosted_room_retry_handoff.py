"""Retry output is new policy input, never a rewrite of an admitted turn."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from types import ModuleType

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import (
    HostedRoomPolicyCheckpoint, MAX_THREAD_TRANSCRIPT_EVENTS, MAX_TRANSCRIPT_POLICY_EVENTS,
)
from tests.tui_gateway.test_hosted_room_native_phase1 import native as native
from tests.tui_gateway.test_hosted_room_produced_media import room as native_room
from tests.tui_gateway.test_hosted_room_service import _FakeRPC, _server
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import HostedRoomService


def test_legacy_admission_does_not_gain_previously_omitted_member_files():
    # Captured from the native pre-forwarding implementation, not regenerated
    # by the new policy: this is a persisted contract compatibility fixture.
    data = json.loads((Path(__file__).parents[1] / "fixtures" / "hosted_room_member_files_legacy.json").read_text())
    task = {**data["task"], "identity": driver.TaskIdentity(**data["task"]["identity"])}
    assert any(e["kind"] == "message.member" and e["payload"].get("attachments") for e in data["events"])
    reconstructed = discussion.reconstruct_task_plan(
        data["room"], data["events"], task, local_profiles=("writer", "reviewer"))
    assert reconstructed.identity == task["identity"]
    assert reconstructed.payload == task["payload"]
    assert "attachments" not in reconstructed.payload


class RefusingRPC(_FakeRPC):
    def submit(self, **kwargs):
        raise PeerRunsHTTPError("synthetic unavailable", not_admitted=True, retryable=True)


@pytest.mark.linux_only
@pytest.mark.parametrize("handoff", [False, True])
def test_native_not_admitted_deferral_retries_with_live_owner(native, native_room, monkeypatch, handoff):
    from gateway.hosted_room_owner_probe import ALIVE, probe_owner_incarnation

    service = native_room
    binding = service.bindings()[0]
    service.send(room_id=binding.room_id, event_id="retry-proof", payload={"text": "@default check this"})
    task = driver.list_tasks(service.db_path, room_id=binding.room_id, status="queued")[0]
    session_key = service.session_key
    agent = native.make_agent("retry-proof")
    agent.session_id = session_key
    native.ready_agent(service.session_id)
    submit = native.srv._methods["prompt.submit"]
    fenced = []

    def busy_submit(rid, params):
        session = native.record(params["session_id"])
        assert session["session_key"] == session_key
        fenced.append(driver.get_task(service.db_path, task["identity"]))
        # A session becomes busy after resolution, at the actual native submit boundary.
        with session["history_lock"]:
            session["running"] = True
        try:
            response = submit(rid, params)
            assert response["error"]["code"] == 4091
            return response
        finally:
            with session["history_lock"]:
                session["running"] = False

    monkeypatch.setitem(native.srv._methods, "prompt.submit", busy_submit)
    service.runtime._process_room(binding)
    assert len(fenced) == 1
    descriptor = fenced[0]["owner_descriptor"]
    assert fenced[0]["admitted_at"] is not None
    assert descriptor["runtime_session_id"] == service.session_id
    assert descriptor["stored_session_key"] == session_key
    assert probe_owner_incarnation(descriptor) == ALIVE
    deferred = driver.get_task(service.db_path, task["identity"])
    assert deferred["status"] == "deferred"
    assert not agent.invocations
    if handoff:
        lease = service.runtime._leases[binding.room_id]
        driver.release_lease(service.db_path, lease, clock=service.runtime.clock)
        service = HostedRoomService(native.srv, db_path=service.db_path)
        monkeypatch.setattr(service, "local_profiles", lambda: ("default", "peer-profile"))
    retried = service.retry_room_task(binding.room_id, task_id=task["identity"].task_id)
    assert retried["status"] == "queued"
    assert deferred["admitted_at"] is None and deferred["owner_descriptor"] is None
    assert probe_owner_incarnation(descriptor) == ALIVE


def _service(tmp_path, monkeypatch, profiles=("writer", "reviewer")):
    server = ModuleType("test_hosted_room_retry_handoff")
    vars(server).update(vars(_server()))
    service = HostedRoomService(server, db_path=tmp_path / "state.db")
    monkeypatch.setattr(service, "local_profiles", lambda: profiles)
    monkeypatch.setattr(service, "rpc", RefusingRPC())
    service.runtime.rpc = service.rpc
    service.create_room(room_id="workshop", name="Workshop planning", members=[
        {"member_id": profile, "profile": profile, "handle": profile}
        for profile in service.local_profiles()
    ])
    return service, service.bindings()[0]


def _send(service, event_id, text="Revise 15/150CHF to 18/180CHF; share agenda.md again.",
          thread_id="workshop-thread"):
    service.send(room_id="workshop", event_id=event_id,
                 payload={"text": text, "thread_id": thread_id})


def _queued(service):
    return driver.list_tasks(service.db_path, room_id="workshop", status="queued")


def _finish(service, binding, task, text="(pass)", attachments=()):
    lease = service.runtime._ensure_lease(binding)
    attempt = driver.start_task(service.db_path, task["identity"], lease,
                                expected_cancel_generation=task["cancel_generation"], clock=service.runtime.clock)
    driver.settle_task(service.db_path, attempt, status="settled",
                       settlement_id=f"test:{task['identity'].task_id}:{attempt.execution_generation}",
                       result={"text": text, **({"attachments": list(attachments)} if attachments else {})},
                       clock=service.runtime.clock)
    service.prepare_room(binding)


def _file(service, task, content):
    uploaded = service.attachments.put(room_id="workshop", upload_id=task["identity"].task_id,
                                       kind="file", name="agenda.md", mime="text/markdown",
                                       data=content.encode())
    manifest = [{key: uploaded[key] for key in ("attachment_id", "kind", "name", "size", "mime")}]
    service.attachments.commit_message(
        room_id="workshop", event_id="dmessage:" + task["identity"].task_id.removeprefix("dtask:"),
        manifest=manifest, recipient_member_ids=["writer", "reviewer"], viewer_access=True, hold_until_event=True)
    return manifest


@pytest.mark.parametrize("persisted_overbound", [False, True])
def test_explicit_deferrals_stay_bounded_while_peer_is_pending(tmp_path, monkeypatch, persisted_overbound):
    service, binding = _service(tmp_path, monkeypatch, ("writer", "reviewer", "observer"))
    _send(service, "hold-observer", "@observer pause", thread_id="controls")
    _send(service, "initial", "Prepare the workshop agenda.")
    task = _queued(service)[0]
    frozen = deepcopy(task["payload"])
    for index in range(65):
        if index:
            task = service.retry_room_task("workshop", task_id=task["identity"].task_id)
        attempt = driver.start_task(
            service.db_path, task["identity"], service.runtime._ensure_lease(binding),
            expected_cancel_generation=task["cancel_generation"], clock=service.runtime.clock)
        # Run the real not-admitted/publication path, leaving the peer queued.
        service.runtime._execute_attempt(binding, task, attempt)
        assert driver.get_task(service.db_path, task["identity"])["status"] == "deferred"
        service.prepare_room(binding)
        assert len(_queued(service)) == 1
    events = service._events("workshop")
    deferrals = [event for event in events if event["kind"] == "turn.deferred"]
    assert len(deferrals) == 65
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_policy_events").fetchone()[0] == 2
    assert driver.get_task(service.db_path, task["identity"])["payload"] == frozen
    peer = _queued(service)[0]
    _finish(service, binding, _queued(service)[0])
    events = service._events("workshop")
    assert events[-1]["kind"] == "room.activity"
    expected = service.policy_checkpoint.snapshot(room_id="workshop", latest_seq=events[-1]["seq"])
    with sqlite3.connect(service.db_path) as conn:
        watermarks = conn.execute("SELECT * FROM hosted_room_policy_watermarks ORDER BY thread_id, member_id").fetchall()
        if persisted_overbound:
            # Persist the pre-compaction insertion shape using ONLY the real producer's
            # canonical events. The next unapplied event is the completing activity.
            source_seq = task["payload"]["source_event_seq"]
            conn.execute("""INSERT INTO hosted_room_policy_threads
                VALUES ('workshop', 'workshop-thread', 'initial', ?, 0)""", (source_seq,))
            for event in events:
                if event["seq"] == source_seq or (
                        event["payload"].get("discussion_event_id") == "initial"
                        and event["kind"] != "room.activity"):
                    conn.execute("INSERT INTO hosted_room_policy_events VALUES (?, ?, ?, ?, ?)",
                                 ("workshop", "workshop-thread", "initial", event["seq"], json.dumps(event)))
            assert conn.execute("SELECT COUNT(*) FROM hosted_room_policy_events").fetchone()[0] > 64
            conn.execute("UPDATE hosted_room_policy_cursors SET through_seq=?", (events[-1]["seq"] - 1,))
            conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=5")
    snapshot = HostedRoomPolicyCheckpoint(service.db_path).snapshot(
        room_id="workshop", latest_seq=events[-1]["seq"])
    assert snapshot == expected
    assert snapshot.through_seq == events[-1]["seq"]
    assert not snapshot.events
    assert snapshot.held_member_ids == ("observer",)
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute("SELECT * FROM hosted_room_policy_watermarks ORDER BY thread_id, member_id").fetchall() == watermarks
    assert service._events("workshop") == events
    for accepted in (task, peer):
        assert driver.get_task(service.db_path, accepted["identity"])["payload"] == accepted["payload"]
        # Exact accepted refs still reconstruct, while the legacy context-loading path
        # retains the first deferral used for the pre-publication watermark.
        for input_context in (accepted["payload"].get("input_context"), None):
            context = service.policy_checkpoint.events_for_task(
                room_id="workshop", source_event_seq=accepted["payload"]["source_event_seq"],
                input_context=input_context, task_id=accepted["identity"].task_id)
            if input_context is None:
                assert [event for event in context if event["kind"] == "turn.deferred"] == [
                    deferrals[0], deferrals[-1]]
            reconstructed = discussion.reconstruct_task_plan(
                service._room("workshop"), context, accepted, local_profiles=service.local_profiles())
            assert reconstructed.payload == accepted["payload"]


@pytest.mark.parametrize("frozen_context", [False, True], ids=["legacy", "accepted-refs"])
def test_transcript_retry_upgrade_preserves_intermediate_task_context(tmp_path, monkeypatch, frozen_context):
    service, binding = _service(tmp_path, monkeypatch, ("writer", "reviewer", "observer"))
    _send(service, "hold-observer", "@observer pause", thread_id="controls")
    holds = service.policy_checkpoint.member_holds(room_id="workshop")
    room = service._room("workshop")
    clock = service.runtime.clock
    publication_context = service._events("workshop")

    def accept(event_id):
        hosted_rooms.append_event(
            service.db_path, room_id="workshop", event_id=event_id, kind="message.user",
            actor={"kind": "user", "id": "owner"},
            authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
            payload={"text": f"@writer review {event_id}", "thread_id": "workshop-thread"})
        publication_context[:] = service._events("workshop")
        plan = discussion.plan_next_task(
            room, publication_context, local_profiles=service.local_profiles(),
            held_member_ids=("observer",), freeze_input_context=frozen_context).task
        assert plan is not None
        task = driver.admit_task(service.db_path, plan.identity, payload=plan.payload, clock=clock)
        return plan, task

    def defer(plan):
        task = driver.get_task(service.db_path, plan.identity)
        lease = service.runtime._ensure_lease(binding)
        if task["status"] == "deferred":
            task = driver.requeue_deferred_task(
                service.db_path, plan.identity, lease,
                expected_execution_generation=task["execution_generation"],
                expected_cancel_generation=task["cancel_generation"], clock=clock)
        attempt = driver.start_task(service.db_path, plan.identity, lease,
                                    expected_cancel_generation=task["cancel_generation"], clock=clock)
        driver.defer_not_admitted_task(service.db_path, attempt, reason="member_unavailable", clock=clock)
        for event in discussion.plan_publication(
                room, publication_context, plan, status="deferred", execution_generation=attempt.execution_generation,
                local_profiles=service.local_profiles()).events:
            hosted_rooms.append_event(service.db_path, **event.append_kwargs("workshop"))

    # Persist real admissions, fenced explicit retries and canonical publications while
    # the checkpoint is offline. This is a valid log/upgrade case, not automatic retries.
    first, accepted_a = accept("source-a")
    for _ in range(3):
        defer(first)
    second, accepted_b = accept("source-b")
    defer(first)
    defer(second)
    for _ in range(MAX_TRANSCRIPT_POLICY_EVENTS - 2):
        defer(first)
    events = service._events("workshop")
    receipts = [event for event in events if event["kind"] == "turn.deferred"]
    assert len(receipts) > MAX_TRANSCRIPT_POLICY_EVENTS
    decision = discussion.plan_next_task(
        room, events, local_profiles=service.local_profiles(), held_member_ids=("observer",))
    assert decision.status == "settled"
    hosted_rooms.append_event(
        service.db_path, room_id="workshop", event_id="complete-b", kind="room.activity",
        actor={"kind": "gateway", "id": binding.gateway_id},
        authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
        payload={"status": decision.status, "reason_code": decision.reason,
                 "thread_id": decision.thread_id, "discussion_event_id": decision.discussion_event_id})
    events = service._events("workshop")
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=5")
    cold = HostedRoomPolicyCheckpoint(service.db_path)
    snapshot = cold.snapshot(room_id="workshop", latest_seq=events[-1]["seq"])
    assert snapshot.through_seq == events[-1]["seq"] and not snapshot.events
    assert snapshot.holds == holds and snapshot.held_member_ids == ("observer",)
    assert service._events("workshop") == events
    with sqlite3.connect(service.db_path) as conn:
        retained = conn.execute("SELECT seq FROM hosted_room_policy_transcript WHERE kind='turn.deferred' ORDER BY seq").fetchall()
        first_receipts = [event for event in receipts if event["payload"]["task_id"] == first.identity.task_id]
        second_receipts = [event for event in receipts if event["payload"]["task_id"] == second.identity.task_id]
        assert retained == [(event["seq"],) for event in sorted(
            [first_receipts[0], first_receipts[-1], *second_receipts], key=lambda event: event["seq"])]
        watermarks = conn.execute("SELECT thread_id, member_id, seen_through_seq FROM hosted_room_policy_watermarks ORDER BY 1, 2").fetchall()
        for key, value in discussion.derive_member_watermarks(room, events, local_profiles=service.local_profiles()).items():
            assert (*key, value) in watermarks
        # Reinflate only derived v5 receipt references, with activity next. Recovery
        # must rebuild before its pre-handler walk queries the overbound transcript.
        for event in receipts:
            conn.execute("INSERT OR IGNORE INTO hosted_room_policy_transcript VALUES (?, ?, ?, ?, NULL)",
                         ("workshop", "workshop-thread", event["seq"], "turn.deferred"))
        conn.execute("INSERT INTO hosted_room_policy_threads VALUES (?, ?, ?, ?, 0)",
                     ("workshop", "workshop-thread", second.discussion_event_id, second.payload["source_event_seq"]))
        conn.execute("UPDATE hosted_room_policy_cursors SET through_seq=?", (events[-1]["seq"] - 1,))
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=5")
    cold = HostedRoomPolicyCheckpoint(service.db_path)
    assert cold.snapshot(room_id="workshop", latest_seq=events[-1]["seq"]) == snapshot
    assert service._events("workshop") == events
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute("SELECT thread_id, member_id, seen_through_seq FROM hosted_room_policy_watermarks ORDER BY 1, 2").fetchall() == watermarks
        assert conn.execute("SELECT seq FROM hosted_room_policy_transcript WHERE kind='turn.deferred' ORDER BY seq").fetchall() == retained
    for accepted in (accepted_a, accepted_b):
        assert driver.get_task(service.db_path, accepted["identity"])["payload"] == accepted["payload"]
        context = cold.events_for_task(
            room_id="workshop", source_event_seq=accepted["payload"]["source_event_seq"],
            input_context=accepted["payload"].get("input_context"), task_id=accepted["identity"].task_id)
        reconstructed = discussion.reconstruct_task_plan(
            room, context, accepted, local_profiles=service.local_profiles())
        assert reconstructed.payload == accepted["payload"]
        if not frozen_context:
            terminal_seq = next(event["seq"] for event in receipts
                                if event["payload"]["task_id"] == accepted["identity"].task_id)
            # Non-vacuous for B: A's watermark predates B, but A's latest receipt does not.
            before = [event for event in events if event["seq"] < terminal_seq]
            retained_before = [event for event in context if event["seq"] < terminal_seq]
            assert discussion.derive_member_watermarks(room, retained_before, local_profiles=service.local_profiles()) == (
                discussion.derive_member_watermarks(room, before, local_profiles=service.local_profiles()))

    # Move the display window past A's first receipt, but not past B. Before
    # compaction A's intermediate generation (between B's source and terminal)
    # supplied B's historical watermark; retaining only A's future latest cannot.
    for index in range(MAX_THREAD_TRANSCRIPT_EVENTS - 1):
        hosted_rooms.append_event(
            service.db_path, room_id="workshop", event_id=f"later-{index}", kind="message.user",
            actor={"kind": "user", "id": "owner"},
            authority_gateway_id=binding.gateway_id, authority_epoch=binding.authority_epoch,
            payload={"text": "@reviewer later request", "thread_id": "workshop-thread"})
    later = service._events("workshop")
    cold.sync(room_id="workshop", latest_seq=later[-1]["seq"])
    context = cold.events_for_task(
        room_id="workshop", source_event_seq=accepted_b["payload"]["source_event_seq"],
        input_context=accepted_b["payload"].get("input_context"), task_id=second.identity.task_id)
    assert discussion.reconstruct_task_plan(
        room, context, accepted_b, local_profiles=service.local_profiles()).payload == accepted_b["payload"]
    if not frozen_context:
        terminal_seq = second_receipts[0]["seq"]
        assert discussion.derive_member_watermarks(
            room, [event for event in context if event["seq"] < terminal_seq],
            local_profiles=service.local_profiles()) == discussion.derive_member_watermarks(
                room, [event for event in events if event["seq"] < terminal_seq],
                local_profiles=service.local_profiles())
    with sqlite3.connect(service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_policy_transcript WHERE thread_id='workshop-thread' AND kind='message.user'").fetchone()[0] == MAX_THREAD_TRANSCRIPT_EVENTS
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=5")
    cold.sync(room_id="workshop", latest_seq=later[-1]["seq"])
    assert cold.events_for_task(
        room_id="workshop", source_event_seq=accepted_b["payload"]["source_event_seq"],
        input_context=accepted_b["payload"].get("input_context"), task_id=second.identity.task_id) == context
    assert cold.member_holds(room_id="workshop") == holds
    assert service._events("workshop") == later


def test_transcript_compaction_keeps_nonadvancing_generations_visible(tmp_path, monkeypatch):
    service, _binding = _service(tmp_path, monkeypatch)
    _send(service, "initial")
    room, events, task = service._room("workshop"), service._events("workshop"), _queued(service)[0]
    plan = discussion.reconstruct_task_plan(room, events, task, local_profiles=service.local_profiles())
    for generation in (1, 3, 2):
        publication = discussion.plan_publication(
            room, events, plan, status="deferred", execution_generation=generation,
            local_profiles=service.local_profiles())
        for event in publication.events:
            hosted_rooms.append_event(service.db_path, **event.append_kwargs("workshop"))
        latest_seq = service._events("workshop")[-1]["seq"]
        if generation == 2:
            with pytest.raises(discussion.DiscussionValidationError, match="deferral generation did not advance"):
                service.policy_checkpoint.sync(room_id="workshop", latest_seq=latest_seq)
        else:
            service.policy_checkpoint.sync(room_id="workshop", latest_seq=latest_seq)


@pytest.mark.parametrize(("with_file", "gate"), [
    (False, "open"), (True, "open"), (False, "bounded"), (False, "stopped"),
    (False, "superseded"), (False, "quiet"), (False, "many_retries"),
])
def test_retry_handoff_reopens_only_for_committed_output_with_fresh_input(tmp_path, monkeypatch, with_file, gate):
    service, binding = _service(tmp_path, monkeypatch)
    _send(service, "initial", "Prepare the workshop agenda.")
    original = _queued(service)[0]
    old_file = _file(service, original, "15 participants, 150CHF") if with_file else []
    _finish(service, binding, original, "Agenda: 15 participants, 150CHF.", old_file)
    _finish(service, binding, _queued(service)[0], "Verified. How many participants?")
    # Settle the native ordinary-round continuation before retrying the followup.
    for _ in range(4):
        if not _queued(service):
            break
        _finish(service, binding, _queued(service)[0])
    assert not _queued(service)
    _send(service, "followup")
    for _ in range(3):
        service.runtime._process_room(binding)
    deferred = driver.list_tasks(service.db_path, room_id="workshop", status="deferred")
    assert len(deferred) == 2
    writer, reviewer = sorted(deferred, key=lambda task: task["payload"]["target_profile"] == "reviewer")
    frozen = {task["identity"].task_id: deepcopy(task["payload"]) for task in deferred}
    assert not _queued(service)
    if gate == "many_retries":
        for index in range(65):
            task = service.retry_room_task("workshop", task_id=writer["identity"].task_id)
            attempt = driver.start_task(service.db_path, task["identity"],
                                        service.runtime._ensure_lease(binding),
                                        expected_cancel_generation=task["cancel_generation"], clock=service.runtime.clock)
            driver.defer_not_admitted_task(service.db_path, attempt, reason="member_unavailable",
                                           clock=service.runtime.clock)
            service.prepare_room(binding)
    before = service._events("workshop")
    assert any(event["kind"] == "room.activity" and event["payload"]["reason_code"] == "silent_round"
               for event in before)

    for task in (writer, reviewer):
        service.retry_room_task("workshop", task_id=task["identity"].task_id)
    retried = [driver.get_task(service.db_path, task["identity"]) for task in (writer, reviewer)]
    assert all(task["payload"] == frozen[task["identity"].task_id] for task in retried)
    assert "18/180CHF" in retried[1]["payload"]["prompt"]
    assert "Updated agenda" not in retried[1]["payload"]["prompt"]
    if gate == "quiet":
        for task in retried:
            _finish(service, binding, task)
        assert not _queued(service)
        assert service.status("workshop")["needs_attention"] is False
        return

    # A visible write alone is not committed input: replay must wait for its receipt.
    room = service._room("workshop")
    plan = discussion.reconstruct_task_plan(room, before, writer, local_profiles=service.local_profiles())
    new_file = _file(service, writer, "18 participants, 180CHF") if with_file else []
    result = {"text": "Updated agenda: 18 participants, 180CHF. @reviewer compare both versions.",
              **({"attachments": new_file} if with_file else {})}
    publication = discussion.plan_publication(room, before, plan, status="settled", result=result,
                                              local_profiles=service.local_profiles())
    hosted_rooms.append_event(service.db_path, **publication.events[0].append_kwargs("workshop"))
    uncommitted = service._events("workshop")
    assert discussion.plan_next_task(room, uncommitted, local_profiles=service.local_profiles()).status == "idle"
    service.policy_checkpoint.snapshot(room_id="workshop", latest_seq=uncommitted[-1]["seq"])

    if gate == "bounded":
        service._append_room_status(service._room("workshop"), discussion.DiscussionDecision(
            "bounded", "max_rounds", discussion_event_id="followup", thread_id="workshop-thread"))
    elif gate == "stopped":
        hosted_rooms.request_room_stop(service.db_path, room_id="workshop", cancel_id="stop-test",
                                        expected_gateway_id=room["authority_gateway_id"], expected_epoch=1)
        service.runtime._process_room(binding)
        service.prepare_room(binding)
        assert not _queued(service)
        assert discussion.plan_next_task(room, service._events("workshop"),
                                         local_profiles=service.local_profiles()).status == "idle"
        return
    elif gate == "superseded":
        _send(service, "newer", "@reviewer a different question")

    _finish(service, binding, retried[0], result["text"], result.get("attachments", ()))
    _finish(service, binding, retried[1])  # Legitimate PASS on the old, immutable input.
    events = service._events("workshop")
    service._append_room_status(room, discussion.DiscussionDecision(
        "settled", "silent_round", discussion_event_id="followup", thread_id="workshop-thread"))
    assert service._events("workshop") == events  # Stale completion cannot erase the new output.
    decision = discussion.plan_next_task(room, events, local_profiles=service.local_profiles(), freeze_input_context=True)
    if gate in {"bounded", "superseded"}:
        if gate == "superseded":
            assert decision.discussion_event_id == "newer"
            assert all(task["payload"]["source_event_seq"] == decision.source_event_seq for task in _queued(service))
        else:
            assert decision.status == "idle"
            assert not _queued(service)
        return
    assert decision.status == "task", "silent_round must not swallow a committed retry handoff"
    fresh = _queued(service)
    assert len(fresh) == 1, "durable checkpoint must agree with full-log policy replay"
    fresh = fresh[0]
    assert decision.task is not None
    assert fresh["identity"] == decision.task.identity
    assert fresh["identity"] != reviewer["identity"]
    assert fresh["payload"]["target_profile"] == "reviewer"
    assert "Updated agenda: 18 participants, 180CHF" in fresh["payload"]["prompt"]
    if with_file:
        assert fresh["payload"]["attachments"] == new_file
        assert list(service._load_task_attachments(binding, fresh)) == [
            (new_file[0], b"18 participants, 180CHF")]
        corrupted = deepcopy(fresh)
        corrupted["payload"]["input_context"].pop("member_attachments")
        with pytest.raises(discussion.DiscussionReconstructionError):
            discussion.reconstruct_task_plan(room, events, corrupted, local_profiles=service.local_profiles())
        assert new_file[0]["attachment_id"] != old_file[0]["attachment_id"]
        assert new_file[0]["name"] == old_file[0]["name"]
    for task in deferred:
        assert driver.get_task(service.db_path, task["identity"])["payload"] == frozen[task["identity"].task_id]

    # A migrated checkpoint and repeated preparation reconstruct the exact same admission.
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=4")
    service.policy_checkpoint = HostedRoomPolicyCheckpoint(service.db_path)
    for _ in range(3):
        service.prepare_room(binding)
        assert _queued(service) == [fresh]
    _finish(service, binding, fresh)
    count = len(service._events("workshop"))
    for _ in range(3):
        service.prepare_room(binding)
    assert not _queued(service)
    assert len(service._events("workshop")) == count
    assert service.status("workshop")["blocked"] is False


@pytest.mark.parametrize("unavailable", [False, True])
def test_unavailable_is_attention_not_scheduler_block_or_quiet_pass(tmp_path, monkeypatch, unavailable):
    service, binding = _service(tmp_path, monkeypatch)
    _send(service, "first")
    if unavailable:
        for _ in range(3):
            service.runtime._process_room(binding)
    else:
        for _ in range(2):
            _finish(service, binding, _queued(service)[0])
    status = service.status("workshop")
    assert status["working"] is False
    assert status["blocked"] is False
    assert status["needs_attention"] is unavailable
    assert bool(status["pending_actions"]) is unavailable
    before = len(service._events("workshop"))
    for _ in range(3):
        service.prepare_room(binding)
    assert len(service._events("workshop")) == before
    _send(service, "healthy-thread", "@reviewer check this independently", thread_id="other")
    assert len(_queued(service)) == 1
    assert _queued(service)[0]["payload"]["target_profile"] == "reviewer"
    _finish(service, binding, _queued(service)[0], "Healthy participant completed.")
    assert not _queued(service)
    assert service.status("workshop")["blocked"] is False

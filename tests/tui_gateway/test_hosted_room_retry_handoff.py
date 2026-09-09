"""Retry output is new policy input, never a rewrite of an admitted turn."""

from copy import deepcopy
import sqlite3

import pytest

from gateway import hosted_room_discussion as discussion
from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from tests.tui_gateway.hosted_room_service_fixtures import _FakeRPC, _server
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from tui_gateway.hosted_room_service import HostedRoomService


class RefusingRPC(_FakeRPC):
    def submit(self, **kwargs):
        raise PeerRunsHTTPError("synthetic unavailable", not_admitted=True, retryable=True)


def _service(tmp_path):
    service = HostedRoomService(_server(), db_path=tmp_path / "state.db")
    service.local_profiles = lambda: ("writer", "reviewer")
    service.rpc = RefusingRPC()
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


@pytest.mark.parametrize(("with_file", "gate"), [
    (False, "open"), (True, "open"), (False, "bounded"), (False, "stopped"),
    (False, "superseded"), (False, "quiet"), (False, "many_retries"),
])
def test_retry_handoff_reopens_only_for_committed_output_with_fresh_input(tmp_path, with_file, gate):
    service, binding = _service(tmp_path)
    _send(service, "initial", "Prepare the workshop agenda.")
    original = _queued(service)[0]
    old_file = _file(service, original, "15 participants, 150CHF") if with_file else []
    _finish(service, binding, original, "Agenda: 15 participants, 150CHF.", old_file)
    _finish(service, binding, _queued(service)[0], "Verified. How many participants?")
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
            task = service.retry_room_task("workshop", task_id=writer["identity"].task_id,
                                           retry_id=f"offline:{index}")
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
        service.retry_room_task("workshop", task_id=task["identity"].task_id,
                                retry_id="retry:" + task["identity"].task_id)
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
        # The cancellation must not depend on whether a real-time backoff elapsed.
        for retry in service.runtime._unavailable_route_retries.values():
            retry["next_attempt_at"] = service.runtime.clock() + 60
        hosted_rooms.request_room_stop(service.db_path, room_id="workshop", cancel_id="stop-test",
                                        expected_gateway_id=room["authority_gateway_id"], expected_epoch=1)
        service.runtime._process_room(binding)
        service.prepare_room(binding)
        assert not _queued(service)
        assert all(driver.get_task(service.db_path, task["identity"])["status"] == "cancelled"
                   for task in retried)
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
    assert fresh["identity"] == decision.task.identity
    assert fresh["identity"] != reviewer["identity"]
    assert fresh["payload"]["target_profile"] == "reviewer"
    assert "Updated agenda: 18 participants, 180CHF" in fresh["payload"]["prompt"]
    if with_file:
        assert fresh["payload"]["attachments"] == new_file
        assert new_file[0]["attachment_id"] != old_file[0]["attachment_id"]
        assert new_file[0]["name"] == old_file[0]["name"]
    for task in deferred:
        assert driver.get_task(service.db_path, task["identity"])["payload"] == frozen[task["identity"].task_id]

    # A migrated checkpoint and repeated preparation reconstruct the exact same admission.
    with sqlite3.connect(service.db_path) as conn:
        conn.execute("UPDATE hosted_room_policy_transcript_state SET schema_version=2")
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
def test_unavailable_is_attention_not_scheduler_block_or_quiet_pass(tmp_path, unavailable):
    service, binding = _service(tmp_path)
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

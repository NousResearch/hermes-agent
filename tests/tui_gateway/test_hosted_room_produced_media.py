"""Files a member produces must reach the room as room content, not as a host path.

Driven through the real backend fixture: real session records, the real
``HostedRoomServerRPC`` bound to them, real driver settlements and the real
``HostedRoomService`` publication path. The only thing stood in for is the model itself.

One case runs the actual ``prompt.submit`` handler and takes the tagged reply from the real
hosted terminal callback, so the text this feature reads is the text that path produces. The
other cases settle the driver directly with that same shape, which is a manufactured settlement,
not callback evidence.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import room_session_title
from tui_gateway.hosted_room_service import HostedRoomService

from tests.tui_gateway.test_hosted_room_native_phase1 import (  # noqa: F401  (fixture)
    WAIT, hosted_task, native, wait_until,
)


ROOM_ID = "produced-room"
PROFILE = "default"


@pytest.fixture
def room(native, tmp_path):
    """One room whose member owns a real native session on this backend."""
    service = HostedRoomService(native.srv, db_path=tmp_path / "room.db")
    service.local_profiles = lambda: (PROFILE, "peer-profile")
    (native.home / "profiles/peer-profile").mkdir(parents=True, exist_ok=True)
    service.create_room(
        room_id=ROOM_ID, name="Produced media",
        members=[{"member_id": "default", "profile": PROFILE, "handle": "default"},
                 {"member_id": "peer", "profile": "peer-profile", "handle": "peer"}])
    service.session_id = native.create_session(room_session_title(ROOM_ID))
    # `_session_lookup_key` prefers the agent's own session_id over the record's durable key, so
    # a fake agent that keeps the harness's invented id would win that lookup and the room would
    # resolve a session nothing else knows about. Bind it to the key this session really has.
    service.session_key = native.record(service.session_id)["session_key"]
    return service


def test_the_native_terminal_callback_carries_the_produced_file_tag(native, room, tmp_path):
    """The real submit path, the real hosted terminal callback, and what it hands publication.

    The reply a member produces reaches the room as tagged text; this is where that text comes
    from, so the promotion is reading the actual contract rather than an assumed one.
    """
    produced = tmp_path / "produced.txt"
    produced.write_bytes(b"the code word lives only inside this file\n")
    agent = native.make_agent("producer")
    agent.session_id = room.session_key
    agent.reply = f"Here is the summary.\n\nMEDIA:{produced}"
    receipts: list = []
    ran: dict[str, threading.Thread] = {}
    scripted = agent.run_conversation

    def record(*args, **kwargs):
        ran["thread"] = threading.current_thread()
        return scripted(*args, **kwargs)

    agent.run_conversation = record
    response = native.submit(room.session_id, task=hosted_task(), receipts=receipts)

    assert "error" not in response, response
    # The scripted turn blocks until it is released, and the callback fires from the run thread.
    native.ready_agent(room.session_id).release_turn.set()
    wait_until(lambda: bool(receipts), "the hosted terminal callback never fired")
    ran["thread"].join(WAIT)

    assert not ran["thread"].is_alive(), "the turn thread outlived its own receipt"
    # The receipt this callback builds is `{status, text}` (prompt_turn.py), so the reply text is
    # read from that exact coordinate rather than from whichever field happens to hold it.
    receipt, = receipts

    assert receipt["status"] == "settled", receipt
    tagged = receipt["text"]

    assert f"MEDIA:{produced}" in tagged, receipt

    # The same text that callback produced, settled and published by the room.
    room.send(room_id=ROOM_ID, event_id="user-1",
              payload={"text": "@default please produce it", "thread_id": "thread-1"})
    _settle(room, tagged)
    room.prepare_room(_binding(room))
    event, = _member_events(room)
    entry, = event["payload"]["attachments"]

    assert event["payload"]["text"] == "Here is the summary."
    assert entry["name"] == "produced.txt"


def _binding(service):
    return next(binding for binding in service.bindings() if binding.room_id == ROOM_ID)


def _settle(service, reply: str) -> None:
    """Settle the queued turn with this reply.

    A manufactured settlement in the driver's own shape: it exercises publication, not the
    callback that normally produces the receipt. `test_the_native_terminal_callback_carries_the
    _produced_file_tag` covers that path.
    """
    task = driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")[0]
    binding = _binding(service)
    lease = driver.acquire_lease(
        service.db_path, room_id=ROOM_ID, gateway_id=binding.gateway_id,
        authority_epoch=binding.authority_epoch, process_generation="produced-media-test",
        ttl_seconds=300, clock=time.time)
    attempt = driver.start_task(
        service.db_path, task["identity"], lease,
        expected_cancel_generation=task["cancel_generation"], clock=time.time)
    driver.settle_task(
        service.db_path, attempt, settlement_id="result", status="settled",
        result={"text": reply}, clock=time.time)


def _member_events(service) -> list[dict]:
    events = hosted_rooms.read_events(
        service.db_path, room_id=ROOM_ID, since_seq=0, limit=hosted_rooms.MAX_LOG_LIMIT)["events"]
    return [event for event in events if event["kind"] == "message.member"]


def _run(service, reply: str) -> dict:
    service.send(room_id=ROOM_ID, event_id="user-1",
                 payload={"text": "@default please produce it", "thread_id": "thread-1"})
    _settle(service, reply)
    service.prepare_room(_binding(service))
    published = _member_events(service)
    assert len(published) == 1, published
    return published[0]


def test_a_file_the_guard_refuses_is_named_rather_than_silently_dropped(room, tmp_path):
    produced = tmp_path / "kept.txt"
    produced.write_bytes(b"this one is fine\n")
    vanished = tmp_path / "vanished.txt"

    event = _run(room, f"Two files.\n\nMEDIA:{produced}\n\nMEDIA:{vanished}")

    entry, = event["payload"]["attachments"]
    assert entry["name"] == "kept.txt"
    assert "vanished.txt" in event["payload"]["text"]
    assert "Attachment unavailable" in event["payload"]["text"]
    assert str(tmp_path) not in event["payload"]["text"]


def test_a_reply_whose_only_file_is_unusable_reports_it_and_attaches_nothing(room, tmp_path):
    event = _run(room, f"Here it is.\n\nMEDIA:{tmp_path / 'never-written.txt'}")

    assert not event["payload"].get("attachments")
    assert "never-written.txt" in event["payload"]["text"]
    assert "Attachment unavailable" in event["payload"]["text"]
    assert "MEDIA:" not in event["payload"]["text"]


@pytest.mark.parametrize(
    ("arm", "template", "attached", "expected_text"),
    [
        ("tagged", "Here is the summary.\n\nMEDIA:{path}", True, "Here is the summary."),
        # A reply that is only a tag has no words left once the tag is removed.
        ("media_only", "MEDIA:{path}", True, "produced.txt"),
        # Code fences hold examples, never deliverables.
        ("protected", "Use it like this:\n```\nMEDIA:{path}\n```", False, None),
        ("unsafe", "MEDIA:/etc/shadow", False, None),
    ],
)
def test_a_native_terminal_reply_publishes_the_file_it_produced(
    room, tmp_path, monkeypatch, arm, template, attached, expected_text,
):
    produced = tmp_path / "produced.txt"
    produced.write_bytes(b"the code word lives only inside this file\n")

    if arm == "unsafe":
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            assert path != Path("/etc/shadow"), "publication opened a rejected path"
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", guarded_open)

    event = _run(room, template.format(path=produced))

    manifest = event["payload"].get("attachments") or []
    assert bool(manifest) is attached, (arm, event["payload"])
    if not attached:
        # Native extraction preserves protected examples and refused extensionless tags.
        # The unsafe arm also proves the rejected path was never opened for delivery.
        assert "MEDIA:" in event["payload"]["text"]
        return
    entry, = manifest
    assert (entry["kind"], entry["name"], entry["mime"]) == ("file", "produced.txt", "text/plain")
    assert event["payload"]["text"] == expected_text
    assert "MEDIA:" not in event["payload"]["text"]
    stored = room.read_attachment(
        room_id=ROOM_ID, attachment_id=entry["attachment_id"], recipient_member_id=None,
        event_id=event["event_id"], viewer=True)
    assert stored.data == produced.read_bytes()


def test_a_promoted_reply_republishes_its_own_outcome_after_a_failed_append(room, tmp_path):
    """The promotion is durable the moment it is decided.

    One file is produced and one is named but gone. A publication whose event cannot be appended
    returns the committed bytes to staging; the retry must publish exactly the same text and the
    same attachment, without re-reading sources that are no longer there.
    """
    produced = tmp_path / "produced.txt"
    produced.write_bytes(b"kept bytes\n")
    missing = tmp_path / "gone.txt"
    missing.write_bytes(b"about to disappear\n")
    room.send(room_id=ROOM_ID, event_id="user-1",
              payload={"text": "@default produce both", "thread_id": "thread-1"})
    _settle(room, f"Both files.\n\nMEDIA:{produced}\n\nMEDIA:{missing}")

    task = driver.list_tasks(room.db_path, room_id=ROOM_ID, status="settled")[0]
    message_id = f"dmessage:{task['identity'].task_id.removeprefix('dtask:')}"
    missing.unlink()

    promoted, transitioned = room._promote_produced_media(
        hosted_rooms.room_state(room.db_path, room_id=ROOM_ID), task, message_id,
        task.get("result"))

    assert [entry["name"] for entry in promoted["attachments"]] == ["produced.txt"]
    assert "gone.txt" in promoted["text"] and "Attachment unavailable" in promoted["text"]
    assert "MEDIA:" not in promoted["text"]

    # The append failed: its bytes go back to staging, its decision does not.
    room.attachments.abort_message_commit(
        room_id=ROOM_ID, event_id=message_id, attachment_ids=transitioned)
    produced.unlink()

    reopened = HostedRoomService(room.server, db_path=room.db_path)
    reopened.local_profiles = room.local_profiles
    replay = reopened.attachments.find_promotion(room_id=ROOM_ID, event_id=message_id)

    assert replay is not None
    assert replay["display_text"] == promoted["text"]
    assert [entry["name"] for entry in replay["manifest"]] == ["produced.txt"]
    assert tuple(replay["unavailable"]) == ("gone.txt",)

    # The cold service is the one that publishes, from the durable outcome alone.
    reopened.prepare_room(_binding(reopened))
    event, = _member_events(reopened)

    assert event["payload"]["text"] == promoted["text"]
    entry, = event["payload"]["attachments"]
    assert entry["attachment_id"] == promoted["attachments"][0]["attachment_id"]
    stored = reopened.read_attachment(
        room_id=ROOM_ID, attachment_id=entry["attachment_id"], recipient_member_id=None,
        event_id=event["event_id"], viewer=True)
    assert stored.data == b"kept bytes\n"

    # Exactly once: a further pass publishes nothing new.
    reopened.prepare_room(_binding(reopened))
    assert len(_member_events(reopened)) == 1

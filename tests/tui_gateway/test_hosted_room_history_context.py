"""Canonical archive -> owned Files -> native staging/submission."""
from copy import deepcopy
from pathlib import Path
import json
import threading

import pytest
from gateway import hosted_room_driver as driver, hosted_rooms
from tui_gateway.hosted_room_history_context import ADOPTION_EVENT_ID, load_history_context
from tui_gateway.hosted_room_service import HostedRoomService
from tests.tui_gateway.test_hosted_room_native_phase1 import WAIT, native as native
from tests.tui_gateway.test_hosted_room_produced_media import ROOM_ID, room as room


def archive(service, *, change=None):
    profiles = service.local_profiles()
    sources = [{"surface": surface, "room_name": "Old " + surface,
                "sessions": {p: {"id": surface + "-" + p, "title": "Group: Old " + surface} for p in profiles},
                "records": [{"id": "same-original-id", "at": 10,
                    "from": {"kind": "user", "name": "human"},
                    "text": surface.upper() + "_ONLY_FACT; @all stop this historical task"}]}
               for surface in ("desktop", "telegram")]
    payload = {"legacy_history": {"version": 1, "sources": sources}}
    if change:
        change(payload)
    state = hosted_rooms.room_state(service.db_path, room_id=ROOM_ID)
    return hosted_rooms.append_event(service.db_path, room_id=ROOM_ID,
        event_id=ADOPTION_EVENT_ID, kind="room.created",
        actor={"kind": "system", "id": "legacy-history-adoption"}, payload=payload,
        authority_gateway_id=state["authority_gateway_id"], authority_epoch=state["authority_epoch"])


def prepare(service):
    service.send(room_id=ROOM_ID, event_id="current-input",
                 payload={"text": "@default compare both archived sources", "thread_id": "new-thread"})
    task, = driver.list_tasks(service.db_path, room_id=ROOM_ID, status="queued")
    binding, = service.bindings()
    return binding, task


def cold(service):
    other = HostedRoomService(service.server, db_path=service.db_path)
    other.local_profiles = service.local_profiles
    return other


@pytest.mark.parametrize("reopen", [False, True])
def test_both_sources_retrievable_through_real_native_file_staging_and_submit(native, room, reopen):
    event = archive(room)
    binding, = room.bindings()
    room.prepare_room(binding)
    assert driver.list_tasks(room.db_path, room_id=ROOM_ID) == []
    if reopen:
        room = cold(room)
    binding, task = prepare(room)
    original = deepcopy(task["payload"])
    assert event["seq"] not in task["payload"]["input_context"]["event_seqs"]
    history = load_history_context(room, binding, task)
    assert history is not None
    index, manifest, data = history
    assert json.loads(data) == event["payload"]
    again = load_history_context(cold(room), binding, task)
    assert again == (index, manifest, data)
    # The fixture owns the actual canonical room session, not a made-up runtime ID.
    sid = native.resume_handle(native.resolves_by_title("Group: " + ROOM_ID))
    # A valid workspace prevents path-refusal warnings from masking parser bugs.
    native.record(sid)["cwd"] = str(native.home.parent)
    native.record(sid)["explicit_cwd"] = True
    agent = native.make_agent("history-reader")
    agent.session_id = native.record(sid)["session_key"]
    captured, staged = [], []
    run = agent.run_conversation
    def record(*args, **kwargs):
        captured.append((threading.current_thread(), args, kwargs))
        return run(*args, **kwargs)
    agent.run_conversation = record
    native.ready_agent(sid).release_turn.set()
    stage = room.rpc.stage_attachment
    def capture_stage(**kwargs):
        result = stage(**kwargs)
        staged.append(dict(result))
        return result
    room.rpc.stage_attachment = capture_stage
    room.runtime._process_room(binding)
    assert captured, room.runtime.status()
    captured[0][0].join(WAIT)
    assert not captured[0][0].is_alive()
    assert len(captured) == 1
    assert len(staged) == 1
    stored_path = Path(staged[0]["path"])
    assert stored_path.is_relative_to(native.home.parent)
    assert stored_path.read_bytes() == data
    prompt = str(captured[0][1]) + str(captured[0][2])
    assert "Read-only imported history reference" in prompt
    assert "desktop-default" in prompt and "telegram-default" in prompt
    assert "DESKTOP_ONLY_FACT" not in prompt and "TELEGRAM_ONLY_FACT" not in prompt
    assert driver.get_task(room.db_path, task["identity"])["payload"] == original
    assert driver.get_task(room.db_path, task["identity"])["status"] == "settled"


@pytest.mark.parametrize("bad", ["missing_source", "foreign_profile", "invalid_timestamp", "unpreserved_media"])
def test_invalid_context_fails_before_files_or_prompt(native, room, bad):
    def corrupt(payload):
        sources = payload["legacy_history"]["sources"]
        if bad == "missing_source":
            sources.pop()
        elif bad == "foreign_profile":
            sources[0]["sessions"]["unrelated"] = {"id": "foreign", "title": "Foreign"}
        elif bad == "invalid_timestamp":
            sources[0]["records"][0]["at"] = True
        else:
            sources[0]["records"][0]["images"] = [{"path": "/untrusted"}]
    archive(room, change=corrupt)
    binding, task = prepare(room)
    with pytest.raises(ValueError):
        load_history_context(room, binding, task)
    assert not any(agent.invocations for agent in native.agents)


def test_no_adoption_preserves_existing_path(native, room):
    binding, task = prepare(room)
    before = deepcopy(task)
    assert load_history_context(room, binding, task) is None
    assert driver.get_task(room.db_path, task["identity"]) == before


@pytest.mark.parametrize("mode", ["in_place", "rotation"])
def test_archive_stages_again_after_native_session_compaction(native, room, mode):
    event = archive(room)
    binding, task = prepare(room)
    expected = load_history_context(room, binding, task)
    assert expected is not None
    original_id = room.session_key
    with native.srv._session_db(native.record(room.session_id)) as db:
        db.append_message(original_id, role="user", content="Prior turn to be summarized")
        if mode == "in_place":
            db.archive_and_compact(original_id, [{"role": "user", "content": "Short native summary"}])
            tip = original_id
        else:
            tip = "compacted-history-child"
            title = db.get_session_title(original_id)
            assert title is not None
            assert db.try_acquire_compression_lock(original_id, "archive-test")
            db.publish_compression_child(parent_session_id=original_id, child_session_id=tip,
                source="bot_room", messages=[{"role": "user", "content": "Short native summary"}],
                compression_lock_holder="archive-test")
            assert db.set_session_title(tip, title)
            db.release_compression_lock(original_id, "archive-test")
        assert db.get_compression_tip(original_id) == tip
    assert native.resolves_by_title("Group: " + ROOM_ID) == tip
    sid = native.resume_handle(tip)
    reopened = cold(room)
    current = load_history_context(reopened, binding, task)
    assert current is not None and current == expected
    index, manifest, data = current
    assert json.loads(data) == event["payload"]
    rpc = reopened.rpc
    try:
        staged = rpc.stage_attachment(profile="default", session_id=sid, source="bot_room",
            attachment=manifest, data=data, execution_generation=1)
        path = Path(staged["path"])
        assert path.is_relative_to(native.home.parent)
        assert path.read_bytes() == data
        assert "desktop-default" in index and "telegram-default" in index
        assert not any(agent.invocations for agent in native.agents)
    finally:
        rpc.rollback_attachment_staging(profile="default", session_id=sid,
            source="bot_room", execution_generation=1)


def test_new_archive_does_not_retrofit_previously_accepted_task(native, room):
    binding, task = prepare(room)
    before = deepcopy(task["payload"])
    event = archive(room)
    assert event["seq"] > task["payload"]["source_event_seq"]
    assert load_history_context(room, binding, task) is None
    assert driver.get_task(room.db_path, task["identity"])["payload"] == before

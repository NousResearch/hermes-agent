"""File discovery on the standalone Files source, without Bot-output schema."""

import base64
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import (
    AttachmentError,
    AttachmentNotFoundError,
    HostedRoomAttachmentStore,
)
from tui_gateway.hosted_room_service import HostedRoomService
from tests.gateway.test_hosted_room_attachment_catalog import (
    ROOM_ID,
    _create_catalog,
    _seed_events,
)


@pytest.fixture
def service(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "profiles" / "ops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    fake_server = SimpleNamespace(
        _methods={}, _sessions={}, _sessions_lock=threading.Lock()
    )
    value = HostedRoomService(fake_server, db_path=home / "state.db")
    value.create_room(
        room_id="files-room",
        name="Files source",
        members=[
            {"member_id": "default", "profile": "default", "handle": "hermes"},
            {"member_id": "ops", "profile": "ops", "handle": "ops"},
        ],
    )
    yield value
    assert value.stop(timeout=2)


def publish(service, index=1):
    item = service.put_attachment(
        room_id="files-room",
        upload_id=f"upload-{index}",
        kind="file",
        name=f"report-{index}.md",
        mime="text/markdown",
        data=b"Exact shared file\n",
    )
    event_id = f"share-{index}"
    manifest = [
        {key: item[key] for key in ("attachment_id", "kind", "name", "mime", "size")}
    ]
    service.attachments.commit_message(
        room_id="files-room",
        event_id=event_id,
        manifest=manifest,
        recipient_member_ids=["default", "ops"],
        viewer_access=True,
        hold_until_event=True,
    )
    hosted_rooms.append_event(
        service.db_path,
        room_id="files-room",
        event_id=event_id,
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "Shared", "attachments": manifest},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
        authority_epoch=1,
    )
    return item, event_id


def test_authority_claim_lookup_stays_bounded_after_index_repair(tmp_path, monkeypatch):
    db, _store = _create_catalog(tmp_path)
    _seed_events(db, total_events=2000, records={1: ["oldest.md"]})
    with sqlite3.connect(db) as conn:
        conn.execute("DROP INDEX idx_hosted_room_events_authority_claim")
    assert hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"] == 2000
    original = hosted_rooms._transaction
    steps = 0

    @contextmanager
    def transaction(*args, **kwargs):
        with original(*args, **kwargs) as conn:
            def progress():
                nonlocal steps
                steps += 1
                return 0

            conn.set_progress_handler(progress, 1)
            yield conn

    monkeypatch.setattr(hosted_rooms, "_transaction", transaction)
    assert hosted_rooms.room_state(db, room_id=ROOM_ID)["latest_seq"] == 2000
    assert 0 < steps < 1000


def test_old_database_reopen_keeps_bad_record_pages_bounded_and_reachable(tmp_path):
    db, store = _create_catalog(tmp_path)
    _seed_events(
        db, total_events=50000, records={i: [f"file-{i}.txt"] for i in range(1, 50001)}
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP INDEX idx_hosted_room_attachments_event")
        conn.execute("""UPDATE hosted_room_attachments SET created_at=(
            SELECT seq FROM hosted_room_events WHERE hosted_room_events.event_id=hosted_room_attachments.event_id
            AND hosted_room_events.room_id=hosted_room_attachments.room_id)""")
        conn.execute("UPDATE hosted_room_events SET actor_json='{}' WHERE seq>49744")
    reopened = HostedRoomAttachmentStore(db)
    first = reopened.list_published(
        room_id=ROOM_ID, authority_gateway_id="gateway-a", authority_epoch=1
    )
    assert first["items"] == [] and first["has_more"]
    second = reopened.list_published(
        room_id=ROOM_ID,
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        cursor=first["next_cursor"],
    )
    assert len(second["items"]) == 8
    assert second["latest_seq"] >= max(item["seq"] for item in second["items"])


def test_discovery_and_read_work_without_optional_safety_tables(service):
    item, event_id = publish(service)
    with sqlite3.connect(service.db_path) as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert not {"hosted_room_quarantine", "hosted_room_disband_fences"} & tables
    page = service.list_attachments(room_id="files-room", query="you")
    assert [row["attachment_id"] for row in page["items"]] == [item["attachment_id"]]
    assert (
        service.read_attachment(
            room_id="files-room",
            attachment_id=item["attachment_id"],
            event_id=event_id,
            recipient_member_id=None,
            viewer=True,
        ).data
        == b"Exact shared file\n"
    )


@pytest.mark.parametrize(
    "table", ["hosted_room_quarantine", "hosted_room_disband_fences"]
)
def test_optional_fences_are_enforced_if_installed(service, table):
    publish(service)
    with sqlite3.connect(service.db_path) as db:
        db.execute(f"CREATE TABLE {table} (room_id TEXT PRIMARY KEY)")
        db.execute(f"INSERT INTO {table} VALUES (?)", ("files-room",))
    with pytest.raises(AttachmentNotFoundError, match="unavailable"):
        service.list_attachments(room_id="files-room")


@pytest.mark.parametrize("change", ["authority", "epoch", "disband", "remove"])
def test_snapshot_rechecks_room_after_service_entry(service, monkeypatch, change):
    publish(service)
    original = service._owned_room

    def changed(room_id):
        room = original(room_id)
        statement = {
            "authority": "UPDATE hosted_rooms SET authority_gateway_id='foreign' WHERE room_id=?",
            "epoch": "UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id=?",
            "disband": "UPDATE hosted_rooms SET disbanded_at=1 WHERE room_id=?",
            "remove": "DELETE FROM hosted_rooms WHERE room_id=?",
        }[change]
        with sqlite3.connect(service.db_path) as db:
            db.execute(statement, (room_id,))
        return room

    monkeypatch.setattr(service, "_owned_room", changed)
    with pytest.raises(AttachmentNotFoundError):
        service.list_attachments(room_id="files-room")


def test_source_cursor_reopens_and_rejects_changed_authority_epoch(service):
    for index in range(9):
        publish(service, index)
    first = service.list_attachments(room_id="files-room")
    assert len(first["items"]) == 8
    publish(service, 9)
    reopened = HostedRoomService(service.server, db_path=service.db_path)
    try:
        held = reopened.list_attachments(
            room_id="files-room", cursor=first["next_cursor"]
        )
        assert [row["name"] for row in held["items"]] == ["report-0.md"]
        assert held["snapshot_seq"] == first["snapshot_seq"]
        assert held["latest_seq"] > first["latest_seq"]
        with sqlite3.connect(service.db_path) as db:
            db.execute(
                "UPDATE hosted_rooms SET authority_epoch=2 WHERE room_id='files-room'"
            )
        with pytest.raises(AttachmentError, match="cursor"):
            reopened.list_attachments(room_id="files-room", cursor=first["next_cursor"])
    finally:
        assert reopened.stop(timeout=2)


def test_slow_file_discovery_does_not_block_stop(service, monkeypatch):
    publish(service)
    entered, release = threading.Event(), threading.Event()
    original = service.attachments.list_published

    def held(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(service.attachments, "list_published", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        read = pool.submit(service.list_attachments, room_id="files-room")
        try:
            assert entered.wait(5)
            stop = pool.submit(
                service.stop_room, "files-room", cancel_id="files-source-stop"
            )
            assert stop.result(timeout=2) == 0
        finally:
            release.set()
        assert read.result(timeout=5)["items"]


def test_real_rpc_lists_and_reads_only_the_viewer_surface(service, monkeypatch):
    import tui_gateway.server as server
    from tui_gateway import methods_groups

    item, event_id = publish(service)
    # The real RPC lifecycle is active, but no Bot work is scheduled by this test.
    monkeypatch.setattr(service.runtime, "_rooms_provider", lambda: ())
    service.start()
    monkeypatch.setattr(methods_groups, "_service", service)
    method = server._methods["groups.attachment.list"]
    page = method(1, {"room_id": "files-room", "purpose": "viewer"})
    assert "error" not in page
    assert page["result"]["items"][0]["attachment_id"] == item["attachment_id"]
    assert "error" in method(2, {"room_id": "files-room", "purpose": "member"})
    assert "error" in method(
        3, {"room_id": "files-room", "purpose": "viewer", "cursor": "invalid"}
    )
    assert "groups.attachment.list" in server._LONG_HANDLERS
    reply = server._methods["groups.attachment.read"](
        4,
        {
            "room_id": "files-room",
            "event_id": event_id,
            "attachment_id": item["attachment_id"],
            "purpose": "viewer",
        },
    )
    assert "error" not in reply
    assert base64.b64decode(reply["result"]["content_base64"]) == b"Exact shared file\n"

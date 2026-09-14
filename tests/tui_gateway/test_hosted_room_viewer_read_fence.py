"""Files viewer reads retain native lifecycle and composed route fences."""

import base64
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver, hosted_rooms
from gateway.hosted_room_attachments import AttachmentError
from tui_gateway.hosted_room_service import HostedRoomService
import tui_gateway.server as server


class ViewerRPCRejected(RuntimeError):
    pass


DENIED = (
    AttachmentError,
    hosted_room_driver.RoomUnavailableError,
    hosted_rooms.HostedRoomError,
    ViewerRPCRejected,
)
DATA = b"Files-only canonical bytes"
COMPOSED_FENCE_TABLES = ("hosted_room_quarantine", "hosted_room_disband_fences")


@pytest.fixture
def published(tmp_path, monkeypatch):
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = root / "state.db"
    gateway = hosted_rooms.local_authority_gateway_id()
    room = hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Files viewer fence",
        members=[{"member_id": "ops", "profile": "ops", "handle": "ops"}],
        authority_gateway_id=gateway,
    )
    service = HostedRoomService(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()),
        db_path=db,
    )
    item = service.put_attachment(
        room_id="room-1",
        upload_id="upload-1",
        kind="file",
        name="canonical.txt",
        mime="text/plain",
        data=DATA,
    )
    manifest = [
        {key: item[key] for key in ("attachment_id", "kind", "name", "size", "mime")}
    ]
    service.attachments.commit_message(
        room_id="room-1",
        event_id="share-1",
        manifest=manifest,
        recipient_member_ids=("ops",),
        viewer_access=True,
        hold_until_event=True,
    )
    hosted_rooms.append_event(
        db,
        room_id="room-1",
        event_id="share-1",
        kind="message.user",
        actor={"kind": "user", "id": "viewer"},
        payload={"text": "Shared", "thread_id": "thread-1", "attachments": manifest},
        authority_gateway_id=gateway,
        authority_epoch=room["authority_epoch"],
    )
    monkeypatch.setattr(server, "get_hosted_room_service", lambda: service)
    return SimpleNamespace(service=service, item=item, room=room)


def _read(published, mode="service", **overrides):
    params = {
        "room_id": "room-1",
        "attachment_id": published.item["attachment_id"],
        "event_id": "share-1",
    }
    params.update(overrides)
    if mode == "service":
        return published.service.read_attachment(
            **params,
            recipient_member_id=None,
            viewer=True,
        ).data
    result = server._methods["groups.attachment.read"](
        1, {**params, "purpose": "viewer"}
    )
    if "error" in result:
        assert result["error"]["code"] == 4141
        assert "result" not in result
        raise ViewerRPCRejected(result["error"]["message"])
    return base64.b64decode(result["result"]["content_base64"], validate=True)


def _native_change(published, state):
    room = published.room
    params = {
        "room_id": room["room_id"],
        "expected_gateway_id": room["authority_gateway_id"],
        "expected_epoch": room["authority_epoch"],
    }
    if state == "disbanded":
        hosted_rooms.disband_room(published.service.db_path, **params)
    else:
        assert state in {"foreign", "epoch"}
        hosted_rooms.claim_authority(
            published.service.db_path,
            **params,
            new_gateway_id="foreign"
            if state == "foreign"
            else room["authority_gateway_id"],
            event_id="test-authority-change",
        )


def _optional_composition_fence(published, table):
    assert table in COMPOSED_FENCE_TABLES
    room = published.room
    with sqlite3.connect(published.service.db_path) as conn:
        if table == "hosted_room_quarantine":
            conn.execute(
                "INSERT INTO hosted_room_quarantine VALUES (?, ?, ?)",
                ("room-1", "test_viewer_fence", 1.0),
            )
        else:
            conn.execute(
                """INSERT INTO hosted_room_disband_fences(
                       room_id, authority_gateway_id, authority_epoch, started_at
                   ) VALUES (?, ?, ?, ?)""",
                (
                    "room-1",
                    room["authority_gateway_id"],
                    room["authority_epoch"],
                    1.0,
                ),
            )


def _metadata_change(published, state):
    assert state in {"viewer-revoked", "expired"}
    with sqlite3.connect(published.service.db_path) as conn:
        if state == "viewer-revoked":
            conn.execute(
                "UPDATE hosted_room_attachments SET viewer_access=0 WHERE room_id='room-1'"
            )
        else:
            conn.execute(
                "UPDATE hosted_room_attachments SET expires_at=1 WHERE room_id='room-1'"
            )


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("state", ["foreign", "disbanded"])
def test_native_files_lifecycle_blocks_selected_viewer(published, mode, state):
    assert _read(published, mode) == DATA
    _native_change(published, state)
    with pytest.raises(DENIED):
        _read(published, mode)


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("table", COMPOSED_FENCE_TABLES)
def test_optional_schema_composition_blocks_selected_viewer(published, mode, table):
    assert _read(published, mode) == DATA
    _optional_composition_fence(published, table)
    with pytest.raises(DENIED):
        _read(published, mode)


def _race(published, monkeypatch, mode, change):
    entered, release = threading.Event(), threading.Event()
    store = published.service.attachments
    original = store._read_blob

    def held(**kwargs):
        entered.set()
        assert release.wait(5)
        return original(**kwargs)

    monkeypatch.setattr(store, "_read_blob", held)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reading = pool.submit(_read, published, mode)
        try:
            assert entered.wait(5)
            lock = published.service._policy_lock
            assert lock.acquire(timeout=1), "viewer I/O holds the execution lock"
            lock.release()
            pool.submit(change).result(timeout=2)
        finally:
            release.set()
        with pytest.raises(DENIED):
            reading.result(timeout=5)


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("state", ["foreign", "epoch", "disbanded"])
def test_native_files_lifecycle_wins_during_blob_io(
    published, monkeypatch, mode, state
):
    _race(published, monkeypatch, mode, lambda: _native_change(published, state))


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("table", COMPOSED_FENCE_TABLES)
def test_optional_schema_composition_wins_during_blob_io(
    published, monkeypatch, mode, table
):
    _race(
        published,
        monkeypatch,
        mode,
        lambda: _optional_composition_fence(published, table),
    )


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("state", ["viewer-revoked", "expired"])
def test_attachment_revocation_wins_during_blob_io(published, monkeypatch, mode, state):
    _race(published, monkeypatch, mode, lambda: _metadata_change(published, state))


@pytest.mark.parametrize("mode", ["service", "rpc"])
def test_exact_room_event_and_legacy_share_remain_readable(published, mode):
    assert _read(published, mode) == DATA
    with pytest.raises(DENIED):
        _read(published, mode, event_id="wrong-event")
    with pytest.raises(DENIED):
        _read(published, mode, room_id="wrong-room")
    with sqlite3.connect(published.service.db_path) as conn:
        conn.execute(
            "UPDATE hosted_room_events SET authority_epoch=NULL WHERE room_id='room-1' AND event_id='share-1'"
        )
    assert _read(published, mode) == DATA


def test_files_reads_do_not_mutate_composition_fences(published):
    with sqlite3.connect(published.service.db_path) as conn:
        before = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in COMPOSED_FENCE_TABLES
        }
    assert _read(published) == DATA
    with sqlite3.connect(published.service.db_path) as conn:
        after = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in COMPOSED_FENCE_TABLES
        }
    assert before == after == {table: 0 for table in COMPOSED_FENCE_TABLES}


def test_frozen_recipient_semantics_are_not_viewer_authorization(published):
    service = published.service
    params = {
        "room_id": "room-1",
        "attachment_id": published.item["attachment_id"],
        "event_id": "share-1",
    }
    assert service.read_attachment(**params, recipient_member_id="ops").data == DATA
    with pytest.raises(AttachmentError):
        service.read_attachment(**params, recipient_member_id="wrong-member")
    _native_change(published, "foreign")
    assert service.read_attachment(**params, recipient_member_id="ops").data == DATA


@pytest.mark.parametrize("state", ["foreign", "epoch", "disbanded"])
def test_native_change_after_service_entry_is_rechecked_before_io(
    published, monkeypatch, state
):
    service = published.service
    original = service._owned_room

    def changed_after_entry(room_id):
        room = original(room_id)
        _native_change(published, state)
        return room

    monkeypatch.setattr(service, "_owned_room", changed_after_entry)
    monkeypatch.setattr(
        service.attachments,
        "_read_blob",
        lambda **_: pytest.fail("stale scope reached blob I/O"),
    )
    with pytest.raises(DENIED):
        _read(published)


def test_fresh_same_owner_epoch_can_read_an_older_retained_share(published):
    _native_change(published, "epoch")
    assert _read(published) == DATA
    assert _read(published, "rpc") == DATA


@pytest.mark.parametrize("change", ["manifest", "blob-binding", "expiry-clock"])
def test_post_byte_attachment_metadata_recheck(published, monkeypatch, change):
    """Controlled metadata-corruption/retention fixtures, not lifecycle APIs."""
    store = published.service.attachments
    replacement = store.put(
        room_id="room-1",
        upload_id="replacement",
        kind="file",
        name="replacement.txt",
        mime="text/plain",
        data=b"x" * len(DATA),
    )
    original = store._read_blob
    now = float(store.clock())
    if change == "expiry-clock":
        with sqlite3.connect(store.db_path) as conn:
            conn.execute(
                "UPDATE hosted_room_attachments SET expires_at=? WHERE attachment_id=?",
                (now + 10, published.item["attachment_id"]),
            )

    def mutate_after_bytes(**kwargs):
        data = original(**kwargs)
        if change == "expiry-clock":
            monkeypatch.setattr(store, "clock", lambda: now + 11)
        else:
            with sqlite3.connect(store.db_path) as conn:
                if change == "manifest":
                    conn.execute(
                        "UPDATE hosted_room_events SET payload_json='{}' WHERE room_id='room-1' AND event_id='share-1'"
                    )
                else:
                    conn.execute(
                        "UPDATE hosted_room_attachments SET blob_id=(SELECT blob_id FROM hosted_room_attachments WHERE attachment_id=?), sha256=? WHERE attachment_id=?",
                        (
                            replacement["attachment_id"],
                            replacement["sha256"],
                            published.item["attachment_id"],
                        ),
                    )
        return data

    monkeypatch.setattr(store, "_read_blob", mutate_after_bytes)
    with pytest.raises(AttachmentError):
        _read(published)


def test_private_transport_remains_independent_of_hosted_viewer_schema(tmp_path):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore

    store = HostedRoomAttachmentStore(tmp_path / "private.db")
    item = store.put(
        room_id="private",
        upload_id="private-upload",
        kind="file",
        name="private.txt",
        mime="text/plain",
        data=DATA,
    )
    store.commit_message(
        room_id="private",
        event_id="private-event",
        manifest=[
            {
                key: item[key]
                for key in ("attachment_id", "kind", "name", "size", "mime")
            }
        ],
        recipient_member_ids=("member",),
    )
    assert (
        store.read(
            room_id="private",
            attachment_id=item["attachment_id"],
            recipient_member_id="member",
        ).data
        == DATA
    )
    with pytest.raises(AttachmentError):
        store.read_viewer(
            room_id="private",
            attachment_id=item["attachment_id"],
            event_id="private-event",
            authority_gateway_id="gateway",
            authority_epoch=1,
        )

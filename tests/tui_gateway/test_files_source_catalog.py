"""Uncovered bounded source-catalog and Stop responsiveness witnesses.

Retained from Files donor 691bb08a5cd310fffc5f3d01653dc93f394fc080.
Exact catalog/search and viewer/RPC obligations already have accepted replacements;
only index-repair complexity and nonblocking Stop are restored here. The service
fixture is inert and never starts its runtime worker.
"""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
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
    value.local_profiles = lambda: ("default", "ops")
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
        conn.execute("DROP INDEX idx_hosted_room_events_cursor")
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

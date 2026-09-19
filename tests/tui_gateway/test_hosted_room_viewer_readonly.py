"""Actual viewer service/RPC: restored historical and current schema controls.

Service discovery alone selects the inert real fixture. Authorization, SQLite
connections, metadata checks, blob reads and RPC handlers are production code.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import sqlite3

import pytest

from tests.tui_gateway.test_hosted_room_viewer_read_fence import (
    DATA, DENIED, _read, _race, published, current_published, server,
)


def _files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def _assert_read_unchanged(root, before):
    after = _files(root)
    # SQLite mode=ro still maintains WAL reader coordination. Never use
    # immutable=1 to avoid those sidecars: that would ignore live WAL fences.
    # All durable data (including existing WAL frames) must be byte-identical;
    # a newly created WAL must be empty. No other file may appear or change.
    for key in set(before) | set(after):
        if key == "state.db-shm":
            continue  # volatile shared read marks, not application data
        if key == "state.db-wal" and key not in before:
            assert after[key] == b""
        else:
            assert after.get(key) == before.get(key), key

@pytest.mark.parametrize("layout", ["published", "current_published"])
@pytest.mark.parametrize("mode", ["service", "rpc"])
def test_viewer_reads_are_query_only(request, layout, mode, monkeypatch):
    item = request.getfixturevalue(layout)
    root = item.service.root
    before = _files(root)
    sql = []
    real_connect = sqlite3.connect

    def observe(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(sql.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", observe)
    assert _read(item, mode) == DATA
    if layout == "published":
        assert _files(root) == before  # historical DELETE-mode fixture: exact tree
    else:
        _assert_read_unchanged(root, before)
    forbidden = ("CREATE ", "ALTER ", "DROP ", "INSERT ", "UPDATE ", "DELETE ",
                 "REPLACE ", "BEGIN IMMEDIATE", "PRAGMA JOURNAL_MODE")
    assert not [s for s in sql if s.lstrip().upper().startswith(forbidden)]
    assert sql, "the real SQLite read path was not exercised"


@pytest.mark.parametrize("layout", ["published", "current_published"])
@pytest.mark.parametrize("mode", ["service", "rpc"])
def test_reserved_writer_does_not_block_viewer(request, layout, mode):
    item = request.getfixturevalue(layout)
    # A RESERVED writer permits readers in both DELETE and WAL journal modes.
    with closing(sqlite3.connect(item.service.db_path)) as writer:
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("UPDATE hosted_rooms SET name='not committed' WHERE room_id='room-1'")
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(_read, item, mode)
            try:
                assert result.result(timeout=2) == DATA
            finally:
                writer.rollback()


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("identity", ["absent", "malformed", "foreign"])
def test_viewer_never_mints_or_borrows_install_identity(published, mode, identity, monkeypatch):
    from hermes_cli import install_identity
    from hermes_constants import get_default_hermes_root
    path = get_default_hermes_root() / "install_id"
    if identity == "absent":
        path.unlink()
    else:
        path.write_text("not-an-id\n" if identity == "malformed" else "f" * 32 + "\n")
    # A cold identity cache is valid process state, not an injected authorizer.
    monkeypatch.setattr(install_identity, "_INSTALL_ID_CACHE", {"root": None, "value": None})
    before = _files(path.parent)
    with pytest.raises(DENIED):
        _read(published, mode)
    assert _files(path.parent) == before


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("state", ["missing-epoch", "missing-owner", "missing-disband", "ambiguous-room", "invalid-epoch", "empty-owner"])
def test_missing_or_ambiguous_core_authority_fails_closed(published, mode, state, monkeypatch):
    with closing(sqlite3.connect(published.service.db_path)) as conn:
        if state.startswith("missing-"):
            column = {"missing-epoch": "authority_epoch", "missing-owner": "authority_gateway_id", "missing-disband": "disbanded_at"}[state]
            conn.execute(f"ALTER TABLE hosted_rooms DROP COLUMN {column}")
        elif state == "ambiguous-room":
            conn.execute("CREATE TABLE duplicate_rooms AS SELECT * FROM hosted_rooms")
            conn.execute("INSERT INTO duplicate_rooms SELECT * FROM hosted_rooms")
            conn.execute("DROP TABLE hosted_rooms")
            conn.execute("ALTER TABLE duplicate_rooms RENAME TO hosted_rooms")
        elif state == "invalid-epoch":
            conn.execute("UPDATE hosted_rooms SET authority_epoch='invalid'")
        else:
            conn.execute("UPDATE hosted_rooms SET authority_gateway_id=''")
        conn.commit()
    before = _files(published.service.root)
    monkeypatch.setattr(published.service.attachments, "_read_blob", lambda **_: pytest.fail("invalid authority reached bytes"))
    with pytest.raises(DENIED):
        _read(published, mode)
    assert _files(published.service.root) == before


def _current_fence(item, table):
    with closing(sqlite3.connect(item.service.db_path)) as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if table == "hosted_room_quarantine":
            conn.execute("INSERT INTO hosted_room_quarantine(room_id, reason, detected_at) VALUES ('room-1', 'unsafe-lineage', 1)")
        else:
            conn.execute("INSERT INTO hosted_room_disband_fences(room_id, authority_gateway_id, authority_epoch, started_at) VALUES ('room-1', ?, ?, 1)",
                         (item.room["authority_gateway_id"], item.room["authority_epoch"]))
        conn.commit()


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("table", ["hosted_room_quarantine", "hosted_room_disband_fences"])
def test_current_mandatory_fence_denies_without_writes(current_published, mode, table):
    assert _read(current_published, mode) == DATA
    _current_fence(current_published, table)
    before = _files(current_published.service.root)
    with pytest.raises(DENIED):
        _read(current_published, mode)
    _assert_read_unchanged(current_published.service.root, before)


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("table", ["hosted_room_quarantine", "hosted_room_disband_fences"])
def test_current_mandatory_fence_wins_during_blob_io(current_published, monkeypatch, mode, table):
    _race(current_published, monkeypatch, mode, lambda: _current_fence(current_published, table))


@pytest.mark.parametrize("mode", ["service", "rpc"])
def test_frozen_recipient_does_not_grant_viewer_access(published, mode):
    with closing(sqlite3.connect(published.service.db_path)) as conn:
        conn.execute("UPDATE hosted_room_attachments SET viewer_access=0")
        conn.commit()
    assert published.service.read_attachment(room_id="room-1", attachment_id=published.item["attachment_id"],
                                            recipient_member_id="ops", event_id="share-1").data == DATA
    with pytest.raises(DENIED):
        _read(published, mode)


@pytest.mark.parametrize("purpose", ["recipient", "", "execute"])
def test_registered_rpc_rejects_nonviewer_purpose(published, purpose):
    result = server._methods["groups.attachment.read"](1, {"room_id": "room-1", "attachment_id": published.item["attachment_id"],
                                                         "event_id": "share-1", "purpose": purpose, "recipient_member_id": "ops"})
    assert result["error"]["code"] == 4141
    assert "result" not in result


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("table", ["hosted_room_events", "hosted_room_attachments"])
def test_ambiguous_published_owner_denied_before_and_after_bytes(published, monkeypatch, mode, table):
    def duplicate():
        # Explicit corruption fixture: valid historical DB has unique owners.
        with closing(sqlite3.connect(published.service.db_path)) as conn:
            conn.execute(f"CREATE TABLE duplicated AS SELECT * FROM {table}")
            conn.execute(f"INSERT INTO duplicated SELECT * FROM {table}")
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE duplicated RENAME TO {table}")
            conn.commit()
    _race(published, monkeypatch, mode, duplicate)
    monkeypatch.setattr(published.service.attachments, "_read_blob", lambda **_: pytest.fail("ambiguous owner reached bytes"))
    with pytest.raises(DENIED):
        _read(published, mode)


@pytest.mark.parametrize("mode", ["service", "rpc"])
@pytest.mark.parametrize("state", ["foreign", "epoch", "disbanded", "viewer-revoked", "expired"])
def test_current_revocation_after_actual_blob_bytes(current_published, monkeypatch, mode, state):
    from tests.tui_gateway.test_hosted_room_viewer_read_fence import _native_change, _metadata_change
    assert _read(current_published, mode) == DATA
    store = current_published.service.attachments
    original = store._read_blob
    observed = []

    def revoke_after_bytes(**kwargs):
        data = original(**kwargs)
        assert data == DATA
        observed.append(data)
        if state in {"viewer-revoked", "expired"}:
            _metadata_change(current_published, state)
        else:
            _native_change(current_published, state)
        return data

    monkeypatch.setattr(store, "_read_blob", revoke_after_bytes)
    with pytest.raises(DENIED):
        _read(current_published, mode)
    assert observed == [DATA]

"""Data-only legacy Files lifecycle hooks; Stop/runtime/network are inert."""
import sqlite3

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import DISBANDED_GRACE_SECONDS, UNCOMMITTED_TTL_SECONDS
from tests.tui_gateway.test_legacy_files_rpc import (
    DATA, forbidden, legacy_files, publish,  # noqa: F401
)


def put(f, suffix, *, room_id="room", data=DATA):
    return f.service.put_attachment(
        room_id=room_id, upload_id=suffix, kind="file", name="notes.txt", mime="text/plain", data=data)


def rows(f):
    with sqlite3.connect(f.service.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return {row["attachment_id"]: dict(row) for row in conn.execute("SELECT * FROM hosted_room_attachments")}


def blobs(f):
    with sqlite3.connect(f.service.db_path) as conn:
        return dict(conn.execute("SELECT blob_id,ref_count FROM hosted_room_attachment_blobs"))


def sentinels(f):
    # Immutable namespace witnesses, NOT fabricated custody receipts or acceptance.
    paths = [f.home / owner / "retained.bin" for owner in ("input-custody-v3", "native-media", "peer-output")]
    for path in paths:
        path.parent.mkdir()
        path.write_bytes(path.parent.name.encode() + b" private bytes")
    before = {path: path.read_bytes() for path in paths}
    with sqlite3.connect(f.service.db_path) as conn:
        for owner in ("input-custody-v3", "native-media", "peer-output"):
            conn.execute("INSERT INTO state_meta(key,value) VALUES (?,?)", ("sentinel:" + owner, owner))
    def verify():
        assert {path: path.read_bytes() for path in paths} == before
        with sqlite3.connect(f.service.db_path) as conn:
            assert dict(conn.execute("SELECT key,value FROM state_meta WHERE key LIKE 'sentinel:%'")) == {
                "sentinel:" + owner: owner for owner in ("input-custody-v3", "native-media", "peer-output")}
    return verify


def test_start_reconciles_then_prunes_real_rows_before_inert_runtime(legacy_files, monkeypatch):
    f = legacy_files
    committed = put(f, "committed")
    publish(f, committed)
    orphan = put(f, "orphan", data=b"orphan")
    publish(f, orphan, "absent", durable=False)
    mismatched = put(f, "mismatched", data=b"mismatched")
    publish(f, mismatched, "share-1", durable=False)
    partial = put(f, "partial", data=b"partial")
    live = put(f, "live", data=b"live")
    grace = put(f, "grace", data=b"grace")
    snapshot = rows(f)
    f.clock[0] += UNCOMMITTED_TTL_SECONDS + 1
    with sqlite3.connect(f.service.db_path) as conn:
        # Legacy crash residue: durable publication exists but its attachment
        # expiry was not cleared. Reconcile must rescue this before prune.
        conn.execute("UPDATE hosted_room_attachments SET expires_at=? WHERE attachment_id=?",
                     (f.clock[0] - 1, committed["attachment_id"]))
        conn.execute("UPDATE hosted_room_attachments SET expires_at=? WHERE attachment_id=?", (f.clock[0] + 100, live["attachment_id"]))
        conn.execute("UPDATE hosted_room_attachments SET state='disbanded',expires_at=? WHERE attachment_id=?", (f.clock[0] + 100, grace["attachment_id"]))
    verify = sentinels(f)
    expected = {item["attachment_id"] for item in (committed, live, grace)}
    calls = []
    def runtime_start():
        calls.append("runtime")
        current = rows(f)
        assert set(current) == expected
        assert current[committed["attachment_id"]]["expires_at"] is None
        assert current[live["attachment_id"]]["state"] == "uploaded"
        assert current[grace["attachment_id"]]["state"] == "disbanded"
        assert set(blobs(f)) == {snapshot[key]["blob_id"] for key in expected}
        for item in (orphan, partial, mismatched):
            assert not f.service.attachments._blob_path(snapshot[item["attachment_id"]]["blob_id"]).exists()
        verify()
    monkeypatch.setattr(f.service.runtime, "start", runtime_start)
    f.service.start()
    f.service.start()
    assert calls == ["runtime", "runtime"]


@pytest.mark.parametrize("failure", ["reconcile_room_events", "prune"])
def test_start_data_failure_never_reaches_runtime(legacy_files, monkeypatch, failure):
    def failed():
        raise OSError("inert data failure")
    monkeypatch.setattr(legacy_files.service.attachments, failure, failed)
    with pytest.raises(OSError, match="inert data failure"):
        legacy_files.service.start()


def disband_rpc(f):
    return f.server._methods["groups.disband"](1, {"room_id": "room", "cancel_id": "test-disband"})


@pytest.mark.parametrize("replay", ["live", "already_disbanded", "tombstone"])
def test_disband_marks_only_after_commit_and_replays_with_grace(legacy_files, monkeypatch, replay):
    f = legacy_files
    committed = put(f, "committed")
    publish(f, committed)
    staged = put(f, "staged")
    hosted_rooms.create_room(f.service.db_path, room_id="other", name="Other",
                            members=[dict(member_id="ops", profile="default", handle="ops")],
                            authority_gateway_id=f.room["authority_gateway_id"])
    other = put(f, "shared", room_id="other")
    before = rows(f)
    blob = before[other["attachment_id"]]["blob_id"]
    assert len(blobs(f)) == 1 and blobs(f)[blob] == 3
    verify = sentinels(f)
    calls = []
    original = hosted_rooms.disband_room
    def durable_disband(*args, **kwargs):
        calls.append("commit")
        return original(*args, **kwargs, now=f.clock[0])
    monkeypatch.setattr(hosted_rooms, "disband_room", durable_disband)
    if replay != "live":
        original(f.service.db_path, room_id="room", expected_gateway_id=f.room["authority_gateway_id"],
                 expected_epoch=f.room["authority_epoch"], now=f.clock[0])
        if replay == "tombstone":
            # Expire only room history through its existing data function.
            hosted_rooms.prune_disbanded_rooms(f.service.db_path, now=f.clock[0] + hosted_rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1)
    for name in ("begin_room_disband", "stop_room", "revoke_room_routes"):
        def inert(*args, _name=name, **kwargs):
            assert all(rows(f)[key]["state"] != "disbanded" for key in (committed["attachment_id"], staged["attachment_id"]))
            calls.append(_name)
        monkeypatch.setattr(f.service, name, inert)
    response = disband_rpc(f)
    assert "error" not in response, response
    assert calls == (["begin_room_disband", "stop_room", "revoke_room_routes", "commit"] if replay == "live" else ["commit"])
    current = rows(f)
    for item in (committed, staged):
        assert current[item["attachment_id"]]["state"] == "disbanded"
        assert current[item["attachment_id"]]["expires_at"] == f.clock[0] + DISBANDED_GRACE_SECONDS
    assert current[other["attachment_id"]] == before[other["attachment_id"]]
    f.clock[0] += 1
    assert "error" not in disband_rpc(f)
    assert rows(f) == current  # replay does not extend retention
    f.clock[0] += DISBANDED_GRACE_SECONDS
    # Preserve an unrelated live reference beyond the deleted room's grace.
    with sqlite3.connect(f.service.db_path) as conn:
        conn.execute("UPDATE hosted_room_attachments SET expires_at=NULL WHERE attachment_id=?", (other["attachment_id"],))
    assert "error" not in disband_rpc(f)
    assert set(rows(f)) == {other["attachment_id"]}
    assert blobs(f) == {blob: 1}
    assert f.service.attachments._blob_path(blob).read_bytes() == DATA
    verify()


@pytest.mark.parametrize("failure", ["begin_room_disband", "stop_room", "revoke_room_routes", "disband_room"])
def test_failed_or_predisband_boundary_never_marks_or_prunes(legacy_files, monkeypatch, failure):
    f = legacy_files
    item = put(f, "unchanged")
    publish(f, item)
    before, blob_before = rows(f), blobs(f)
    calls = []
    def failed(*args, **kwargs):
        raise RuntimeError("inert refusal")
    for name in ("begin_room_disband", "stop_room", "revoke_room_routes"):
        def inert(*args, _name=name, **kwargs):
            calls.append(_name)
        monkeypatch.setattr(f.service, name, failed if name == failure else inert)
    monkeypatch.setattr(hosted_rooms, "disband_room", failed if failure == "disband_room" else forbidden)
    monkeypatch.setattr(f.service.attachments, "mark_room_disbanded", forbidden)
    monkeypatch.setattr(f.service.attachments, "prune", forbidden)
    response = disband_rpc(f)
    assert response["error"]["code"] == 5114
    assert response["error"]["message"] == "inert refusal"
    assert rows(f) == before and blobs(f) == blob_before
    with sqlite3.connect(f.service.db_path) as conn:
        assert conn.execute("SELECT disbanded_at FROM hosted_rooms WHERE room_id='room'").fetchone()[0] is None

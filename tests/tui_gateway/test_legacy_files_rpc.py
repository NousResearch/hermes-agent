"""Installed legacy Files RPCs; no worker, transport, or model is started.

Viewer/store scenarios adapted from I691's viewer-read-fence and source-catalog
contracts. The fixture owns only temporary data and inert lifecycle boundaries.
"""
import base64
from dataclasses import replace
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from hermes_state import SessionDB
from tui_gateway.hosted_room_service import HostedRoomService
from tests.gateway.test_session_group_files import files  # noqa: F401


DATA = b"legacy viewer bytes"
FIELDS = ("attachment_id", "kind", "name", "size", "mime")


def forbidden(*args, **kwargs):
    pytest.fail("unleased execution/lifecycle boundary")


@pytest.fixture
def legacy_files(tmp_path, monkeypatch):
    import tui_gateway.server as server

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    with SessionDB(home / "state.db") as db:
        room = hosted_rooms.create_room(
            db.db_path, room_id="room", name="Legacy Files",
            members=[dict(member_id="ops", profile="default", handle="ops")],
            authority_gateway_id=hosted_rooms.local_authority_gateway_id())
        service = HostedRoomService(
            SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()),
            db_path=db.db_path)
        clock = [time.time()]
        service.attachments.clock = lambda: clock[0]
        monkeypatch.setattr(service.runtime, "start", forbidden)
        monkeypatch.setattr(service.runtime, "stop", forbidden)
        for name in ("begin_room_disband", "stop_room", "revoke_room_routes"):
            monkeypatch.setattr(service, name, forbidden)
        monkeypatch.setattr(server, "get_hosted_room_service", lambda: service)
        yield SimpleNamespace(service=service, server=server, room=room, db=db, clock=clock, home=home)


def rpc(f, method, **params):
    name = "groups.attachment." + method
    assert name in f.server._methods, f"missing installed RPC: {name}"
    return f.server._methods[name](7, params)


def upload(f, suffix="1", **overrides):
    result = rpc(f, "put", **{
        "room_id": "room", "upload_id": "upload-" + suffix, "kind": "file",
        "name": "report.txt", "mime": "text/plain",
        "content_base64": base64.b64encode(DATA).decode(), **overrides})
    assert "error" not in result, result
    return result["result"]["attachment"]


def publish(f, item, event="share-1", *, viewer=True, durable=True):
    manifest = [{key: item[key] for key in FIELDS}]
    f.service.attachments.commit_message(
        room_id="room", event_id=event, manifest=manifest,
        recipient_member_ids=("ops",), viewer_access=True if durable else viewer, hold_until_event=True)
    if durable:
        hosted_rooms.append_event(
            f.service.db_path, room_id="room", event_id=event, kind="message.user",
            actor={"kind": "user", "id": "viewer"},
            payload={"text": "Shared", "thread_id": "thread", "attachments": manifest},
            authority_gateway_id=f.room["authority_gateway_id"], authority_epoch=f.room["authority_epoch"])
        if not viewer:
            # A retained share with subsequently revoked viewer access, not an
            # invalid new message.user publication without viewer permission.
            with sqlite3.connect(f.service.db_path) as conn:
                conn.execute("UPDATE hosted_room_attachments SET viewer_access=0 WHERE attachment_id=?",
                             (item["attachment_id"],))


def read(f, item, **overrides):
    return rpc(f, "read", **{
        "room_id": "room", "attachment_id": item["attachment_id"],
        "event_id": "share-1", "purpose": "viewer", **overrides})


def test_local_delivery_capability_does_not_advertise_http_receiver(legacy_files, monkeypatch):
    from tui_gateway import methods_groups

    f = legacy_files
    names = {"groups.attachment." + name for name in ("put", "read", "list")}
    assert names <= f.server._methods.keys()
    assert names <= methods_groups.LONG_HANDLERS <= f.server._LONG_HANDLERS
    monkeypatch.setattr(f.server, "_room_link_run_storage_durable", lambda: True)
    monkeypatch.setattr('gateway.hosted_room_peer.gateway_room_grant_secret', lambda: b'fixture-secret' * 3)
    result = f.server._methods["groups.capabilities"](1, {})["result"]
    assert names <= set(result["methods"])
    assert {"attachment_ids", "attachment_metadata_catalog"} <= set(result["features"])
    assert "attachment_same_gateway_delivery" in result["features"]
    assert result['room_link']['catalog']['attachments'] is False
    assert result['driver'] is False


@pytest.mark.parametrize("method", ["put", "read", "list"])
def test_installed_unavailable_envelope(legacy_files, monkeypatch, method):
    monkeypatch.setattr(legacy_files.server, "get_hosted_room_service", lambda: None)
    response = rpc(legacy_files, method, purpose="viewer")
    assert response["error"]["code"] == 4123
    assert "result" not in response


def test_put_durable_commit_viewer_list_read_and_cursor_reset(legacy_files, monkeypatch):
    f = legacy_files
    first = upload(f)
    assert read(f, first)["error"]["code"] == 4141
    publish(f, first)
    second = upload(f, "2")
    publish(f, second, "share-2")
    original = f.service.attachments._read_blob
    monkeypatch.setattr(f.service.attachments, "_read_blob", forbidden)
    page = rpc(f, "list", room_id="room", purpose=" ViEwEr ", limit=1)["result"]
    assert page["items"][0]["attachment_id"] == second["attachment_id"]
    assert page["next_cursor"]
    assert "content_base64" not in page["items"][0]
    older = rpc(f, "list", room_id="room", purpose="viewer", limit=1, cursor=page["next_cursor"])["result"]
    assert older["items"][0]["attachment_id"] == first["attachment_id"]
    invalid = rpc(f, "list", room_id="room", purpose="viewer", limit=1,
                  cursor=page["next_cursor"], query="changed")
    assert invalid["error"]["code"] == 4143
    assert invalid["error"]["data"] == {
        "reason": "attachment_cursor_reset_required", "reset_required": True, "action": "return_to_latest"}
    monkeypatch.setattr(f.service.attachments, "_read_blob", original)
    reply = read(f, first, purpose=" ViEwEr ", recipient_member_id="not-a-member")["result"]
    assert base64.b64decode(reply["content_base64"], validate=True) == DATA
    assert reply["attachment"]["event_id"] == "share-1"
    assert not f.service.runtime.status()["running"]


@pytest.mark.parametrize("method,code", [("read", 4141), ("list", 4142)])
@pytest.mark.parametrize("purpose", [None, "recipient", "ops", ""])
def test_nonviewer_cannot_borrow_recipient_authority(legacy_files, method, code, purpose):
    f = legacy_files
    item = upload(f)
    publish(f, item, viewer=False)
    response = rpc(f, method, room_id="room", attachment_id=item["attachment_id"],
                   event_id="share-1", purpose=purpose, recipient_member_id="ops", viewer=True)
    assert response["error"]["code"] == code


def test_viewer_rpc_ignores_caller_recipient_authority(legacy_files):
    f = legacy_files
    item = upload(f)
    publish(f, item, viewer=False)
    assert f.service.read_attachment(room_id="room", attachment_id=item["attachment_id"],
                                     recipient_member_id="ops", event_id="share-1").data == DATA
    assert read(f, item, recipient_member_id="ops")["error"]["code"] == 4141
    assert rpc(f, "list", room_id="room", purpose="viewer", recipient_member_id="ops")["result"]["items"] == []


@pytest.mark.parametrize("override", [
    {"content_base64": "%%%"}, {"name": "../private"}, {"room_id": "missing"},
    {"kind": "pdf", "mime": "application/pdf"},
])
def test_put_reuses_validation_without_writing_bytes(legacy_files, monkeypatch, override):
    f = legacy_files
    monkeypatch.setattr("tui_gateway.hosted_room_service.shutil.which", lambda _: None)
    result = rpc(f, "put", **{
        "room_id": "room", "upload_id": "bad", "kind": "file", "name": "note.txt",
        "mime": "text/plain", "content_base64": base64.b64encode(DATA).decode(), **override})
    assert result["error"]["code"] == 4140
    with sqlite3.connect(f.service.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_attachments").fetchone()[0] == 0


def mutate(f, state):
    statements = {
        "foreign": "UPDATE hosted_rooms SET authority_gateway_id='foreign' WHERE room_id='room'",
        "epoch": "UPDATE hosted_rooms SET authority_epoch=authority_epoch+1 WHERE room_id='room'",
        "disbanded": "UPDATE hosted_rooms SET disbanded_at=1 WHERE room_id='room'",
        "expired": "UPDATE hosted_room_attachments SET expires_at=1 WHERE room_id='room'",
        "event": "UPDATE hosted_room_events SET payload_json='{}' WHERE event_id='share-1'",
        "viewer": "UPDATE hosted_room_attachments SET viewer_access=0 WHERE room_id='room'",
    }
    with sqlite3.connect(f.service.db_path) as conn:
        conn.execute(statements[state])


@pytest.mark.parametrize("state", ["foreign", "epoch", "disbanded", "expired", "event", "viewer"])
@pytest.mark.parametrize("timing", ["before", "after_bytes"])
def test_installed_read_rechecks_live_scope_and_metadata(legacy_files, monkeypatch, state, timing):
    f = legacy_files
    item = upload(f)
    publish(f, item)
    assert base64.b64decode(read(f, item)["result"]["content_base64"]) == DATA
    if timing == "before":
        if state == "epoch":
            original_room = f.service._owned_room
            def changed_scope(room_id):
                room = original_room(room_id)
                mutate(f, state)
                return room
            monkeypatch.setattr(f.service, "_owned_room", changed_scope)
        else:
            mutate(f, state)
        monkeypatch.setattr(f.service.attachments, "_read_blob", forbidden)
    else:
        original = f.service.attachments._read_blob
        def changed(**kwargs):
            data = original(**kwargs)
            mutate(f, state)
            return data
        monkeypatch.setattr(f.service.attachments, "_read_blob", changed)
    response = read(f, item, recipient_member_id="ops")
    assert response["error"]["code"] == 4141
    assert "result" not in response


@pytest.mark.asyncio
async def test_installed_legacy_names_do_not_change_canonical_routes(files):
    import tui_gateway.server as server
    from gateway.session_group_controls import dispatch_group_control
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_session_group_files import share

    assert "groups.attachment.list" in server._methods
    service, actor, gateway, db = files
    service.authority.hosted_room_service = service
    connection = SimpleNamespace(authority=service.authority, actor=actor)
    item = share(service, actor, 1)
    page = await dispatch_group_control(connection, "groups.attachment.list", {"room_id": "room"})
    assert page["items"][0]["attachment_id"] == item["attachment_id"]
    assert page["authority"] == {"gateway_id": gateway, "epoch": 1}
    result = await dispatch_group_control(connection, "groups.attachment.download", {
        "room_id": "room", "event_id": "event-1", "attachment_id": item["attachment_id"]})
    assert base64.b64decode(result["data_base64"]) == b"version 1"
    for method, params in [("groups.attachment.put", {}), ("groups.attachment.read", {}),
                           ("groups.attachment.list", {"room_id": "room", "purpose": "viewer"})]:
        with pytest.raises(RuntimeStoreError, match="invalid_params"):
            await dispatch_group_control(connection, method, params)
    for denied in (replace(actor, subject="bob"), replace(actor, profile_id="foreign"),
                   replace(actor, capabilities=frozenset())):
        with pytest.raises(RuntimeStoreError):
            await dispatch_group_control(SimpleNamespace(authority=service.authority, actor=denied),
                                         "groups.attachment.list", {"room_id": "room"})
    assert not service.runtime.status()["running"]

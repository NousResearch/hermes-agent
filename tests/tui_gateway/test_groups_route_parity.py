"""Both server routes answer the group composer alike (#145184 points 2 and 4).

* A method the backend does not serve says so; a JSON-RPC client sends no version, so neither
  route may claim the app and backend are on different versions.
* The ``tui_gateway`` route serves ``groups.attachment.upload``/``download`` and carries a member
  turn's bound files, as the gateway route does.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from tests.tui_gateway.test_hosted_room_attachment_routes import (
    _admit,
    _legacy_runtime,
    _run_until_settled_or_error,
    _server_module,
)
from tests.tui_gateway.test_hosted_room_driver_runtime import BINDING, PROFILE, ROOM_ID
import tui_gateway.server as srv
from tui_gateway import methods_groups
from tui_gateway.hosted_room_server_rpc import HostedRoomServerRPC

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _assert_not_available(error: dict, method: str) -> None:
    assert error["code"] == -32601, error
    assert error["message"] == f"unknown method: {method} — not available on this Hermes backend"
    assert "out of sync" not in error["message"]


def test_unknown_method_is_named_unavailable_on_both_routes(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection
    from hermes_state import SessionDB

    _assert_not_available(srv.handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "groups.no_such_verb", "params": {}})["error"],
        "groups.no_such_verb")

    home = tmp_path / "state"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with SessionDB(home / "state.db") as db:
        authority = SimpleNamespace(profile_id=str(home), instance_id="owner", db=db, events={}, sessions={})
        connection = AuthorityConnection(authority, object(), {"user_id": "reader"})
        reply = asyncio.run(connection.dispatch({"id": 2, "method": "groups.no_such_verb", "params": {}}))
    _assert_not_available(reply["error"], "groups.no_such_verb")


@pytest.fixture
def home(tmp_path, monkeypatch):
    class DurableRunStore:
        durable = True

    path = tmp_path / ".hermes"
    path.mkdir()
    (path / "profiles" / "ops").mkdir(parents=True)
    (path / "profiles" / "ops" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(path))
    monkeypatch.setattr(srv, "_run_idempotency_store", DurableRunStore(), raising=False)
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def _result(envelope):
    assert "error" not in envelope, envelope
    return envelope["result"]


def test_tui_route_uploads_sends_and_downloads_a_group_file(home):
    capabilities = _result(srv._methods["groups.capabilities"](1, {}))
    assert {"groups.attachment.upload", "groups.attachment.download"} <= set(capabilities["methods"])
    _result(srv._methods["groups.create"](2, {
        "room_id": "room-1", "name": "Release room",
        "members": [{"member_id": "default", "profile": "default", "handle": "hermes"},
                    {"member_id": "ops", "profile": "ops", "handle": "ops"}]}))

    data = b"exact composer bytes"
    uploaded = _result(srv._methods["groups.attachment.upload"](3, {
        "room_id": "room-1", "upload_id": "upload-1", "kind": "file", "name": "note.txt",
        "mime": "text/plain", "data_base64": base64.b64encode(data).decode("ascii")}))
    manifest = [{k: uploaded[k] for k in ("attachment_id", "kind", "name", "size", "mime")}]
    sent = _result(srv._methods["groups.send"](4, {
        "room_id": "room-1", "event_id": "send-1",
        "payload": {"text": "@ops read this", "thread_id": "thread-1", "attachments": manifest}}))

    downloaded = _result(srv._methods["groups.attachment.download"](5, {
        "room_id": "room-1", "event_id": sent["event"]["event_id"],
        "attachment_id": uploaded["attachment_id"]}))
    assert base64.b64decode(downloaded["data_base64"]) == data


@pytest.fixture
def db(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "member-home"))
    path = tmp_path / "rooms.db"
    hosted_rooms.create_room(
        path, room_id=ROOM_ID, name="Release room", members=[{"profile": PROFILE, "handle": PROFILE}],
        authority_gateway_id=BINDING.gateway_id, now=time.time())
    return path


def _bind(db: Path, uploads: list[tuple[str, str, bytes]]) -> list[dict]:
    store = HostedRoomAttachmentStore(db)
    manifest = []
    for index, (name, mime, data) in enumerate(uploads):
        saved = store.put(room_id=ROOM_ID, upload_id=f"upload-{index}", kind="image" if mime.startswith("image/")
                          else "file", name=name, mime=mime, data=data)
        manifest.append({k: saved[k] for k in ("attachment_id", "kind", "name", "size", "mime")})
    store.commit_message(room_id=ROOM_ID, event_id="event-1", manifest=manifest,
                         recipient_member_ids=["member-ops"])
    return [{**item, "event_id": "event-1"} for item in manifest]


def test_tui_member_turn_receives_its_document_and_image(db: Path):
    identity = _admit(db, attachments=_bind(db, [("note.txt", "text/plain", b"exact bytes"),
                                                 ("shot.png", "image/png", _PNG)]))
    server, submits = _server_module()
    server._profile_home = lambda _profile: None
    runtime = _legacy_runtime(db, HostedRoomServerRPC(server, db_path=db))

    task = _run_until_settled_or_error(runtime, db, identity)

    assert task["status"] == "settled", runtime.status()["last_error"]
    submitted, = submits
    document = Path(submitted["text"].split("[Shared attachment] file: ")[1].split("\n")[0])
    assert document.read_bytes() == b"exact bytes"
    image, = submitted["_hosted_images"]
    assert Path(image).read_bytes() == _PNG


def test_tui_member_turn_without_a_file_stays_text_only(db: Path):
    identity = _admit(db)
    server, submits = _server_module()
    runtime = _legacy_runtime(db, HostedRoomServerRPC(server, db_path=db))

    task = _run_until_settled_or_error(runtime, db, identity)

    assert task["status"] == "settled", runtime.status()["last_error"]
    assert submits[0]["text"] == "Read the attached note."
    assert "_hosted_images" not in submits[0]

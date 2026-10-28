"""The standalone route reaches every room method the gateway route serves (#145960).

A room handed to the gateway is fenced in the standalone store, so a room call this wire does
not forward runs against the frozen copy and is refused ("managed by another gateway"). The
forwarded set therefore follows the gateway route's own method table, never a hand list.
"""

from __future__ import annotations

import base64
import sqlite3
from types import SimpleNamespace

import pytest

import tui_gateway.server as srv
from tui_gateway import methods_groups, owner_forward

MEMBERS = [{"member_id": "default", "profile": "default", "handle": "hermes"},
           {"member_id": "ops", "profile": "ops", "handle": "ops"}]
ENDPOINT = object()


@pytest.fixture
def home(tmp_path, monkeypatch):
    path = tmp_path / ".hermes"
    (path / "profiles" / "ops").mkdir(parents=True)
    (path / "profiles" / "ops" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(path))
    methods_groups.stop_hosted_room_service(timeout=1.0)
    methods_groups.start_hosted_room_service()
    yield path
    methods_groups.stop_hosted_room_service(timeout=1.0)


def _gateway(monkeypatch, state: str) -> None:
    from hermes_cli import gateway_runtime

    def discover(home, *, timeout=2.0):
        return gateway_runtime.GatewayDiscovery(state, endpoint=ENDPOINT if state == "ready" else None)
    monkeypatch.setattr(gateway_runtime, "discover_gateway_endpoint", discover)


def _hand_room_to_gateway(monkeypatch):
    """A Group Chat created on the standalone backend, then imported (and fenced) by the gateway."""
    from gateway.hosted_rooms import default_db_path
    from gateway.hosted_rooms_gateway_import import import_standalone_rooms
    _gateway(monkeypatch, "absent")
    created = srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})
    assert "error" not in created, created
    target = default_db_path().with_name("state.db")
    with sqlite3.connect(target) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
    assert import_standalone_rooms(target) == 1
    _gateway(monkeypatch, "ready")
    return target


def test_every_gateway_room_method_reaches_the_owning_gateway(home, monkeypatch):
    from gateway.session_group_controls import GROUP_METHODS
    from tui_gateway.contracts.registry import METHODS
    _hand_room_to_gateway(monkeypatch)
    reached = []

    def forward_rpc(home, endpoint, method, params):
        reached.append(method)
        return {"methods": list(GROUP_METHODS)} if method == "groups.capabilities" else {}
    monkeypatch.setattr(owner_forward, "forward_rpc", forward_rpc)

    for method in GROUP_METHODS:
        assert method in METHODS, f"{method}: no contract on the standalone wire"
        reply = srv._methods[method](1, {"room_id": "room-desk"})
        assert "error" not in reply, (method, reply)
    capabilities = srv._methods["groups.capabilities"](2, {})["result"]

    assert reached[:len(GROUP_METHODS)] == list(GROUP_METHODS)
    assert capabilities["methods"] == list(GROUP_METHODS)  # what reaches the owner is advertised


def test_a_file_attached_to_a_handed_room_lands_in_the_gateway_copy(home, monkeypatch):
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.hosted_rooms import default_db_path
    from gateway.session_hosted_attachments import store_upload
    target = _hand_room_to_gateway(monkeypatch)
    calls = []

    def forward_rpc(home, endpoint, method, params):
        calls.append(method)
        assert method == "groups.attachment.upload"
        return store_upload(SimpleNamespace(db_path=target), params["room_id"], params)  # the owner's store
    monkeypatch.setattr(owner_forward, "forward_rpc", forward_rpc)

    data = b"exact composer bytes"
    reply = srv._methods["groups.attachment.upload"](1, {
        "room_id": "room-desk", "upload_id": "upload-1", "kind": "file", "name": "note.txt",
        "mime": "text/plain", "data_base64": base64.b64encode(data).decode("ascii")})

    assert "error" not in reply, reply  # main: "This Group Chat is managed by another gateway."
    assert calls == ["groups.attachment.upload"]
    held = HostedRoomAttachmentStore(target).find_upload(room_id="room-desk", upload_id="upload-1")
    assert held is not None and held["attachment_id"] == reply["result"]["attachment_id"]
    fenced = HostedRoomAttachmentStore(default_db_path()).find_upload(room_id="room-desk", upload_id="upload-1")
    assert fenced is None  # nothing was written to the frozen standalone copy


def test_discard_on_a_room_this_backend_drives_is_refused_locally(home, monkeypatch):
    _gateway(monkeypatch, "absent")
    srv._methods["groups.create"](1, {"room_id": "room-desk", "name": "Desk", "members": MEMBERS})

    reply = srv._methods["groups.discard"](2, {
        "room_id": "room-desk", "member_id": "ops", "task_id": "task-1", "execution_generation": 1})

    assert reply["error"]["code"] == 4127, reply

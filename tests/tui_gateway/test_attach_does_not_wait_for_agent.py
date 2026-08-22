"""Attach RPCs must not block on the deferred agent build.

``image.attach``, ``image.attach_bytes``, ``file.attach``, ``pdf.attach`` and
``clipboard.paste`` need the session RECORD (cwd, profile_home,
attached_images) — never the agent. They also run inline on the socket reader
thread (none is in ``_LONG_HANDLERS``), so any wait there stalls every RPC
queued behind them on the same socket, including the ``prompt.submit`` that
carries the image.

These are invariants, not timings: the handler must complete while the build
event is still unset, and the staged image must still reach the turn.
"""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path

import pytest

from tui_gateway import server

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4


def building_session(tmp_path, sid: str) -> dict:
    """A session record whose deferred agent build has NOT completed."""
    session = {
        "agent": None,
        "agent_ready": threading.Event(),  # deliberately never set
        "agent_error": None,
        "attached_images": [],
        "cwd": str(tmp_path),
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "image_counter": 0,
        "profile_home": str(tmp_path),
        "running": False,
        "session_key": sid,
        "transport": None,
    }
    server._sessions[sid] = session
    return session


@pytest.fixture
def no_build(monkeypatch):
    """Never let the real builder run — the point is the unfinished build."""
    monkeypatch.setattr(server, "_start_agent_build", lambda sid, session: None)


@pytest.fixture
def session(tmp_path, no_build, request):
    sid = f"attach-{request.node.name}"
    record = building_session(tmp_path, sid)
    yield sid, record
    server._sessions.pop(sid, None)


def call(method: str, params: dict) -> dict:
    return server._methods[method](1, params)


@pytest.mark.parametrize(
    ("method", "extra"),
    [
        ("image.attach_bytes", {"content_base64": base64.b64encode(PNG_BYTES).decode(), "filename": "a.png"}),
        ("file.attach", {"name": "notes.txt"}),
    ],
)
def test_attach_completes_while_agent_is_still_building(session, tmp_path, method, extra):
    """The handler returns without waiting on ``agent_ready``."""
    sid, record = session

    if method == "file.attach":
        target = tmp_path / "notes.txt"
        target.write_text("hello")
        extra = {**extra, "path": str(target)}

    response = call(method, {"session_id": sid, **extra})

    assert "error" not in response, response
    assert response["result"]["attached"] is True
    # The invariant that makes this a fix rather than a coincidence: the build
    # never finished, and the attach landed anyway.
    assert not record["agent_ready"].is_set()


def test_attached_image_is_queued_for_the_next_turn(session):
    """Not blocking must not mean not staging — the turn still gets the image."""
    sid, record = session

    response = call(
        "image.attach_bytes",
        {
            "session_id": sid,
            "content_base64": base64.b64encode(PNG_BYTES).decode(),
            "filename": "shot.png",
        },
    )

    staged = response["result"]["path"]
    assert record["attached_images"] == [staged]
    assert response["result"]["count"] == 1


def test_attach_still_rejects_an_unknown_session(no_build):
    """Dropping the agent wait must not drop session validation."""
    response = call("image.attach_bytes", {"session_id": "nope", "content_base64": "eA=="})

    assert response["error"]["code"] == 4001


def test_detach_completes_while_agent_is_still_building(session):
    """Detach is the same class as attach — record-only, so it must not wait."""
    sid, record = session
    record["attached_images"] = ["/tmp/one.png", "/tmp/two.png"]

    response = call("image.detach", {"session_id": sid, "path": "/tmp/one.png"})

    assert response["result"]["detached"] is True
    assert record["attached_images"] == ["/tmp/two.png"]
    assert not record["agent_ready"].is_set()


def test_sess_building_does_not_wait_but_sess_does(session, monkeypatch):
    """The two resolvers differ in exactly one way: the wait."""
    sid, _record = session
    waited: list[str] = []

    monkeypatch.setattr(server, "_wait_agent", lambda s, rid: waited.append(rid) or None)

    server._sess_building({"session_id": sid}, "rid-building")
    assert waited == []

    server._sess({"session_id": sid}, "rid-sess")
    assert waited == ["rid-sess"]


def relay_binding(sid, *, data: bytes, name="notes.txt", media_type="text/plain", order=0):
    return {
        "stored_session_id": sid,
        "lineage_root_id": sid,
        "name": name,
        "media_type": media_type,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "order": order,
    }


def relay_request(method: str, sid: str, data: bytes) -> dict:
    if method == "image.attach_bytes":
        return {
            "session_id": sid,
            "content_base64": "data:image/png;base64," + base64.b64encode(data).decode(),
            "filename": "shot.png",
            "relay": relay_binding(sid, data=data, name="shot.png", media_type="image/png"),
        }
    return {
        "session_id": sid,
        "data_url": "data:text/plain;base64," + base64.b64encode(data).decode(),
        "name": "notes.txt",
        "relay": relay_binding(sid, data=data),
    }


@pytest.mark.parametrize("method,data", [("file.attach", b"hello"), ("image.attach_bytes", PNG_BYTES)])
@pytest.mark.parametrize("busy_field", ["running", "_compute_host_active"])
def test_exact_relay_rejects_every_busy_state_before_mutation(session, tmp_path, method, data, busy_field):
    sid, record = session
    record[busy_field] = True

    response = call(method, relay_request(method, sid, data))

    assert response["error"]["code"] == 4009
    assert record["attached_images"] == []
    assert not (tmp_path / "attachments").exists()


@pytest.mark.parametrize("method,data", [("file.attach", b"hello"), ("image.attach_bytes", PNG_BYTES)])
@pytest.mark.parametrize("busy_field", ["running", "_compute_host_active"])
def test_exact_relay_rechecks_busy_state_immediately_before_mutation(
    session, tmp_path, monkeypatch, method, data, busy_field
):
    sid, record = session
    real_integrity = server._relay_integrity

    def become_busy(*args, **kwargs):
        result = real_integrity(*args, **kwargs)
        record[busy_field] = True
        return result

    monkeypatch.setattr(server, "_relay_integrity", become_busy)

    response = call(method, relay_request(method, sid, data))

    assert response["error"]["code"] == 4009
    assert record["attached_images"] == []
    assert not (tmp_path / "attachments").exists()


def test_exact_file_relay_verifies_and_echoes_complete_integrity_without_path(session):
    sid, _record = session
    data = b"hello"
    response = call(
        "file.attach",
        {
            "session_id": sid,
            "data_url": "data:text/plain;base64," + base64.b64encode(data).decode(),
            "name": "notes.txt",
            "relay": relay_binding(sid, data=data),
        },
    )

    assert "error" not in response, response
    assert response["result"] == {
        "attached": True,
        "bytes": 5,
        "media_type": "text/plain",
        "name": "notes.txt",
        "stored_name": "notes.txt",
        "order": 0,
        "runtime_session_id": sid,
        "sha256": hashlib.sha256(data).hexdigest(),
        "ref_text": "@file:attachments/notes.txt",
    }
    assert "path" not in response["result"]


def test_exact_image_relay_uses_the_same_closed_integrity_contract(session):
    sid, record = session
    response = call(
        "image.attach_bytes",
        {
            "session_id": sid,
            "content_base64": "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode(),
            "filename": "shot.png",
            "relay": relay_binding(
                sid,
                data=PNG_BYTES,
                name="shot.png",
                media_type="image/png",
                order=1,
            ),
        },
    )

    assert "error" not in response, response
    stored_name = Path(record["attached_images"][0]).name
    assert response["result"] == {
        "attached": True,
        "bytes": len(PNG_BYTES),
        "media_type": "image/png",
        "name": "shot.png",
        "stored_name": stored_name,
        "order": 1,
        "runtime_session_id": sid,
        "sha256": hashlib.sha256(PNG_BYTES).hexdigest(),
    }
    assert len(record["attached_images"]) == 1
    assert "path" not in response["result"]


def test_exact_file_relay_authenticates_collision_generated_target_name(session, tmp_path):
    sid, _record = session
    attachments = tmp_path / "attachments"
    attachments.mkdir()
    (attachments / "notes.txt").write_text("existing", encoding="utf-8")

    response = call("file.attach", relay_request("file.attach", sid, b"hello"))

    assert "error" not in response, response
    assert response["result"]["name"] == "notes.txt"
    assert response["result"]["stored_name"] != "notes.txt"
    assert response["result"]["ref_text"].endswith(response["result"]["stored_name"])
    staged = attachments / response["result"]["stored_name"]
    assert staged.read_bytes() == b"hello"


def test_exact_file_relay_rejects_post_stage_byte_mutation(session, monkeypatch):
    sid, _record = session
    real_stage = server._stage_session_file_attachment

    def mutate_after_stage(*args, **kwargs):
        stored_path, uploaded = real_stage(*args, **kwargs)
        stored_path.write_bytes(b"tampered")
        return stored_path, uploaded

    monkeypatch.setattr(server, "_stage_session_file_attachment", mutate_after_stage)

    response = call("file.attach", relay_request("file.attach", sid, b"hello"))

    assert response["error"]["code"] == 4017


def test_exact_image_relay_rejects_post_stage_byte_mutation(session, monkeypatch):
    sid, record = session
    real_queue = server._queue_attached_image

    def mutate_after_stage(*args, **kwargs):
        stored_path = real_queue(*args, **kwargs)
        stored_path.write_bytes(b"tampered")
        return stored_path

    monkeypatch.setattr(server, "_queue_attached_image", mutate_after_stage)

    response = call("image.attach_bytes", relay_request("image.attach_bytes", sid, PNG_BYTES))

    assert response["error"]["code"] == 4017
    assert record["attached_images"] == []


def test_exact_image_relay_binds_media_type_to_actual_image_bytes(session):
    sid, record = session
    request = relay_request("image.attach_bytes", sid, PNG_BYTES)
    request["content_base64"] = "data:image/gif;base64," + base64.b64encode(PNG_BYTES).decode()
    request["relay"]["media_type"] = "image/gif"

    response = call("image.attach_bytes", request)

    assert response["error"]["code"] == 4017
    assert record["attached_images"] == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("stored_session_id", "other"),
        ("lineage_root_id", "other"),
        ("name", "other.txt"),
        ("media_type", "application/json"),
        ("bytes", 999),
        ("sha256", "0" * 64),
        ("order", -1),
    ],
)
def test_exact_file_relay_rejects_any_integrity_or_session_mismatch(session, field, value):
    sid, record = session
    data = b"hello"
    relay = relay_binding(sid, data=data)
    relay[field] = value

    response = call(
        "file.attach",
        {
            "session_id": sid,
            "data_url": "data:text/plain;base64," + base64.b64encode(data).decode(),
            "name": "notes.txt",
            "relay": relay,
        },
    )

    assert response["error"]["code"] in {4004, 4017, 4018}
    assert not list(record.get("attached_files", []))


@pytest.mark.parametrize("name", ["../escape", "..\\escape", "C:stream", "\\\\server\\share", "NUL", "trailing."])
def test_exact_relay_rejects_cross_platform_unsafe_names_before_staging(session, name):
    sid, _record = session
    data = b"hello"
    response = call(
        "file.attach",
        {
            "session_id": sid,
            "data_url": "data:text/plain;base64," + base64.b64encode(data).decode(),
            "name": name,
            "relay": relay_binding(sid, data=data, name=name),
        },
    )

    assert response["error"]["code"] == 4004

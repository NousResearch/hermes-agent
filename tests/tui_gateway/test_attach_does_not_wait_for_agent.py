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
import threading

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

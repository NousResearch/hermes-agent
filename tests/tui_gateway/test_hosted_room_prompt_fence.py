"""Older Desktop clients cannot start a second driver for hosted rooms."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager

import pytest

from gateway import hosted_rooms
from tui_gateway.hosted_room_driver import room_session_title
import tui_gateway.server as server


def _stub_session(monkeypatch, *, title, hosted_room_id=None, session_key=None):
    session = {"id": "session-1", "title": title, "source": "bot_room"}
    if hosted_room_id is not None:
        session["hosted_room_id"] = hosted_room_id
    if session_key is not None:
        session["session_key"] = session_key
    monkeypatch.setattr(
        server,
        "_sess_nowait",
        lambda _params, _rid: (
            session,
            None,
        ),
    )


def test_direct_prompt_to_hosted_group_session_is_rejected(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    hosted_rooms.create_room(
        hosted_rooms.default_db_path(),
        room_id="room-hosted",
        name="Hosted room",
        members=[
            {"member_id": "one", "profile": "one", "handle": "one"},
            {"member_id": "two", "profile": "two", "handle": "two"},
        ],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    _stub_session(monkeypatch, title="Group: room-hosted")

    result = server._methods["prompt.submit"](
        "request-1", {"session_id": "session-1", "text": "continue"}
    )

    assert result["error"]["code"] == 4122
    assert "managed by its gateway" in result["error"]["message"]


def test_direct_prompt_uses_bound_room_id_for_bounded_session_title(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    room_id = "room-" + "x" * 120
    hosted_rooms.create_room(
        hosted_rooms.default_db_path(),
        room_id=room_id,
        name="Long hosted room",
        members=[],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    _stub_session(
        monkeypatch,
        title=room_session_title(room_id),
        hosted_room_id=room_id,
    )

    result = server._methods["prompt.submit"](
        "request-long", {"session_id": "session-1", "text": "continue"}
    )

    assert result["error"]["code"] == 4122


def test_direct_prompt_recovers_persisted_room_id_for_bounded_title(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    room_id = "room-" + "y" * 120
    hosted_rooms.create_room(
        hosted_rooms.default_db_path(),
        room_id=room_id,
        name="Persisted long room",
        members=[],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )

    class FakeSessionDB:
        def get_session_model_config_value(self, session_id, key, default=None):
            assert (session_id, key) == ("stored-room-session", "hosted_room_id")
            return room_id

    @contextmanager
    def session_db(_session):
        yield FakeSessionDB()

    monkeypatch.setattr(server, "_session_db", session_db)
    _stub_session(
        monkeypatch,
        title=room_session_title(room_id),
        session_key="stored-room-session",
    )

    result = server._methods["prompt.submit"](
        "request-persisted-long",
        {"session_id": "session-1", "text": "continue"},
    )

    assert result["error"]["code"] == 4122


def test_legacy_prompt_fence_recovers_actual_long_room_id(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    room_id = "r" * 127 + "a"
    db = hosted_rooms.default_db_path()
    hosted_rooms.create_room(
        db,
        room_id=room_id,
        name="Long hosted room",
        members=[],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    title = room_session_title(room_id)
    assert len(title) == 100
    _stub_session(monkeypatch, title=title)

    probed_room_ids = []
    probe = hosted_rooms.probe_hosted_room

    def recording_probe(db_path, *, room_id):
        probed_room_ids.append(room_id)
        return probe(db_path, room_id=room_id)

    monkeypatch.setattr(hosted_rooms, "probe_hosted_room", recording_probe)

    result = server._methods["prompt.submit"](
        "request-long", {"session_id": "session-1", "text": "continue"}
    )

    assert result["error"]["code"] == 4122
    assert probed_room_ids == [room_id]


def test_direct_prompt_to_non_hosted_group_reaches_normal_admission(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _stub_session(monkeypatch, title="Group: local-only")
    monkeypatch.setattr(
        server,
        "_ensure_active_session_slot",
        lambda _sid, _session: "normal admission reached",
    )

    result = server._methods["prompt.submit"](
        "request-2", {"session_id": "session-1", "text": "continue"}
    )

    assert result["error"] == {"code": 4090, "message": "normal admission reached"}
    assert not hosted_rooms.default_db_path().exists()


@pytest.mark.parametrize(
    "legacy_name",
    (
        "Launch room",
        "Ceo, Product Designer, Cfo",
        "Équipe",
        "Alpha/Beta",
    ),
)
def test_direct_prompt_to_legacy_named_group_reaches_normal_admission(
    tmp_path, monkeypatch, legacy_name
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _stub_session(monkeypatch, title=f"Group: {legacy_name}")
    monkeypatch.setattr(
        server,
        "_ensure_active_session_slot",
        lambda _sid, _session: "normal admission reached",
    )

    result = server._methods["prompt.submit"](
        "request-legacy", {"session_id": "session-1", "text": "continue"}
    )

    assert result["error"] == {"code": 4090, "message": "normal admission reached"}


def test_direct_prompt_is_refused_when_room_authority_cannot_be_verified(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _stub_session(monkeypatch, title="Group: room-unknown")
    monkeypatch.setattr(
        hosted_rooms,
        "probe_hosted_room",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk busy")),
    )

    result = server._methods["prompt.submit"](
        "request-3", {"session_id": "session-1", "text": "continue"}
    )

    assert result["error"]["code"] == 5122
    assert result["error"]["message"] == (
        "Could not verify this group. Try again after the gateway recovers."
    )


def test_contended_ownership_probe_fails_quickly_without_blocking_socket(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = hosted_rooms.default_db_path()
    hosted_rooms.create_room(
        db,
        room_id="room-busy",
        name="Busy room",
        members=[],
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(),
    )
    _stub_session(monkeypatch, title="Group: room-busy")

    blocker = sqlite3.connect(db)
    blocker.execute("PRAGMA journal_mode=DELETE")
    blocker.execute("BEGIN EXCLUSIVE")
    started = time.monotonic()
    try:
        result = server._methods["prompt.submit"](
            "request-busy", {"session_id": "session-1", "text": "continue"}
        )
    finally:
        blocker.rollback()
        blocker.close()

    # The probe itself has a 50ms busy timeout. Leave Windows scheduler
    # headroom under the parallel runner while still proving this path returns
    # at least 10x sooner than SQLite's normal ten-second connection timeout.
    assert time.monotonic() - started < 1.0
    assert result["error"]["code"] == 5122

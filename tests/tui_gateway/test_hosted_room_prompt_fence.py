"""Older Desktop clients cannot start a second driver for hosted rooms."""

from __future__ import annotations

import sqlite3
import time

import pytest

from gateway import hosted_rooms
import tui_gateway.server as server


def _stub_session(monkeypatch, *, title, profile_home=None):
    monkeypatch.setattr(
        server,
        "_sess_nowait",
        lambda _params, _rid: (
            {
                "id": "session-1",
                "title": title,
                "source": "bot_room",
                "profile_home": str(profile_home) if profile_home else None,
            },
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


def test_direct_prompt_to_peer_reserved_group_is_rejected_until_revoke(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    profile_home = home / "profiles" / "reviewer"
    profile_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "reviewer")
    now = time.time()
    claims = {
        "room_id": "room-peer",
        "home_install_id": "install-home",
        "authority_gateway_id": "install-home",
        "authority_epoch": 1,
        "member_id": "member-reviewer",
        "target_install_id": "install-target",
        "target_profile": "reviewer",
        "issued_at": now,
    }
    hosted_rooms.reserve_peer_room(
        hosted_rooms.default_db_path(),
        claims=claims,
        expires_at=now + 300.0,
        now=now,
    )
    _stub_session(
        monkeypatch,
        title="Group: room-peer",
        profile_home=profile_home,
    )

    rejected = server._methods["prompt.submit"](
        "request-peer",
        {"session_id": "session-1", "text": "continue"},
    )
    assert rejected["error"]["code"] == 4122
    assert "home host" in rejected["error"]["message"]

    hosted_rooms.revoke_room_grant_scope(
        hosted_rooms.default_db_path(),
        claims=claims,
        expires_at=now + 300.0,
        now=now + 150.0,
    )
    monkeypatch.setattr(
        server,
        "_ensure_active_session_slot",
        lambda _sid, _session: "normal admission reached",
    )
    admitted = server._methods["prompt.submit"](
        "request-peer-after-revoke",
        {"session_id": "session-1", "text": "continue"},
    )
    assert admitted["error"] == {
        "code": 4090,
        "message": "normal admission reached",
    }


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

    assert time.monotonic() - started < 0.5
    assert result["error"]["code"] == 5122

"""Deleting the onboarding chat: the run ends only once the delete succeeded, on every compression copy."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import onboarding_record
from session_rules.onboarding_flow import End, Facts
from hermes_cli.config import read_user_config_raw
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from tui_gateway import server

FACTS = Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=())


@pytest.fixture
def db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    db = server._get_db()
    db.sessions.create(NewSession(session_id="onb", source="desktop"))
    onboarding_record.start(db, "onb", FACTS, persistence=db)
    return db


@pytest.fixture
def ended(monkeypatch) -> list[dict]:
    told: list[dict] = []
    monkeypatch.setattr(onboarding_record, "_ended_listeners", [told.append])
    return told


def _flag(db) -> bool:
    return read_user_config_raw(Path(server._hermes_home) / "config.yaml")["onboarding"]["run"]


def _delete(db, session_id: str) -> bool:
    """Delete *session_id* through the store; True when it existed."""
    return bool(db.sessions.delete(session_id, media_root=str(Path(server._hermes_home) / "media")))


def test_a_refused_delete_leaves_the_run_going(db, ended):
    assert onboarding_record.delete_chat(db, "onb", lambda: False, persistence=db) is False

    assert onboarding_record.active_session_id(db, persistence=db) == "onb" and _flag(db) is True and ended == []


def test_deleting_the_newest_copy_ends_the_run_on_every_copy(db, ended):
    for parent, child in (("onb", "mid"), ("mid", "tip")):
        db.lineage.publish_compression_child(parent, CompressionChild(session_id=child, source="desktop"),
                                             [NewMessage(role="user", content="summary")],
                                             require_compression_lease=False)

    assert onboarding_record.delete_chat(db, "tip", lambda: _delete(db, "tip"), persistence=db)

    assert onboarding_record.active_session_id(db, persistence=db) is None and _flag(db) is False
    assert [onboarding_record.load(db, sid, persistence=db)["ended_by"] for sid in ("onb", "mid")] == [
        "deleted", "deleted"]
    assert [(e["stored_session_id"], e["outcome"]) for e in ended] == [("tip", "deleted")]


def _failing_flag_write(monkeypatch):
    from hermes_cli import onboarding_run

    def disk_full(value, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(onboarding_run, "set_run", disk_full)


def test_a_delete_whose_flag_write_fails_keeps_the_chat_for_a_retry(db, ended, monkeypatch):
    with monkeypatch.context() as failing:
        _failing_flag_write(failing)
        with pytest.raises(OSError):
            onboarding_record.delete_chat(db, "onb", lambda: _delete(db, "onb"), persistence=db)

    assert db.sessions.get("onb") is not None and ended == []
    assert onboarding_record.active_session_id(db, persistence=db) == "onb"
    assert onboarding_record.delete_chat(db, "onb", lambda: _delete(db, "onb"), persistence=db)
    assert _flag(db) is False and [e["outcome"] for e in ended] == ["deleted"]


def test_a_delete_refused_by_the_write_guard_leaves_the_flag_set(db, ended):
    from persistence.core.errors import SessionWriteGuarded

    def guarded():
        raise SessionWriteGuarded("onb")

    with pytest.raises(SessionWriteGuarded):
        onboarding_record.delete_chat(db, "onb", guarded, persistence=db)

    assert onboarding_record.active_session_id(db, persistence=db) == "onb" and _flag(db) is True and ended == []

"""Deleting the onboarding chat: the run ends only once the delete succeeded, on every compression copy."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import onboarding_record
from agent.onboarding_flow import End, Facts
from hermes_cli.config import read_user_config_raw
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
    db.create_session("onb", "desktop")
    onboarding_record.start(db, "onb", FACTS)
    return db


@pytest.fixture
def ended(monkeypatch) -> list[dict]:
    told: list[dict] = []
    monkeypatch.setattr(onboarding_record, "_ended_listeners", [told.append])
    return told


def _flag(db) -> bool:
    return read_user_config_raw(Path(server._hermes_home) / "config.yaml")["onboarding"]["run"]


def test_a_refused_delete_leaves_the_run_going(db, ended):
    assert onboarding_record.delete_chat(db, "onb", lambda: False) is False

    assert onboarding_record.active_session_id(db) == "onb" and _flag(db) is True and ended == []


def test_deleting_the_newest_copy_ends_the_run_on_every_copy(db, ended):
    for parent, child in (("onb", "mid"), ("mid", "tip")):
        db.publish_compression_child(parent_session_id=parent, child_session_id=child, source="desktop",
                                     messages=[{"role": "user", "content": "summary"}],
                                     require_compression_lease=False)

    assert onboarding_record.delete_chat(db, "tip", lambda: db.delete_session("tip"))

    assert onboarding_record.active_session_id(db) is None and _flag(db) is False
    assert [onboarding_record.load(db, sid)["ended_by"] for sid in ("onb", "mid")] == ["deleted", "deleted"]
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
            onboarding_record.delete_chat(db, "onb", lambda: db.delete_session("onb"))

    assert db.get_session("onb") is not None and onboarding_record.active_session_id(db) == "onb" and ended == []
    assert onboarding_record.delete_chat(db, "onb", lambda: db.delete_session("onb"))
    assert _flag(db) is False and [e["outcome"] for e in ended] == ["deleted"]


def test_a_delete_refused_by_the_write_guard_leaves_the_flag_set(db, ended):
    from hermes_state_errors import SessionActiveWriteGuardError

    def guarded():
        raise SessionActiveWriteGuardError("active turn lease")

    with pytest.raises(SessionActiveWriteGuardError):
        onboarding_record.delete_chat(db, "onb", guarded)

    assert onboarding_record.active_session_id(db) == "onb" and _flag(db) is True and ended == []

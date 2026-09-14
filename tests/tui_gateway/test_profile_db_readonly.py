"""Foreign-profile _profile_db handles must not write-lock another profile's live store.

A named-profile backend serving RPCs about a different profile (desktop app-global remote
mode, profile switcher) used to acquire() a WRITER on that profile's state.db per RPC and
close it in the handler's finally. Reads never need that lock, and the writer's close
participated in the deleted-WAL incident class. Read paths now open read-only, mirroring
hermes_cli.web_routers.profiles._read_profile_db; the few RPCs that genuinely write
(move-cwd, delete, set_hidden, foreign import) opt in with writer=True.
"""

from __future__ import annotations

from pathlib import Path

import tui_gateway.server as server
from hermes_state import SessionDB


def _seed_store(home: Path) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    db.create_session(session_id="seed", source="cli", model="m")
    db.close()
    return home


def _bind_foreign(monkeypatch, tmp_path: Path) -> None:
    foreign = _seed_store(tmp_path / "profiles" / "code")
    monkeypatch.setattr(server, "_profile_home", lambda name: foreign if (name or "").strip() == "code" else None)


def test_foreign_profile_db_is_read_only(monkeypatch, tmp_path):
    _bind_foreign(monkeypatch, tmp_path)
    with server._profile_db({"profile": "code"}) as db:
        assert db is not None
        assert db.read_only is True
        assert db.get_session("seed") is not None


def test_foreign_profile_db_writer_opt_in(monkeypatch, tmp_path):
    _bind_foreign(monkeypatch, tmp_path)
    with server._profile_db({"profile": "code"}, writer=True) as db:
        assert db is not None
        assert db.read_only is False
        assert db.set_session_title("seed", "renamed") is True

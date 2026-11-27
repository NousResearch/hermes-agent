"""``session.resume`` reuses a live session only within the requested profile.

The live registry is keyed by bare stored session id, and stored ids are
timestamp-based, so the same id can legitimately be live under profile A while
profile B's store also holds it. The resume fast path (and the post-build
re-check / ``_claim_or_reuse_live``) used to hand profile B's resume profile
A's runtime — the turn then ran with A's persona and wrote A's memory
(#100029). Pinned here:

* resume with profile B never reuses profile A's live session of the same id;
* the launch profile (no ``profile``) still matches live records that carry
  no ``profile_home`` — the pre-existing single-profile contract.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tui_gateway import server


class _DB:
    """Minimal profile store stand-in, also the storage handle beside it: every profile store knows ``s1``."""

    def __init__(self, db_path=None, **_kwargs):
        self.db_path = db_path
        self.sessions = SimpleNamespace(
            get=lambda target: SimpleNamespace(id="s1", cwd="", git_branch=None, git_repo_root=None,
                                               as_session_row=lambda: {"id": "s1", "cwd": ""}) if target == "s1" else None,
            get_by_title=lambda _target: None, is_kanban_owned=lambda _target: False,
            retire_undrained_queue_rows=lambda _target: 0)
        self.lineage = SimpleNamespace(resolve_resume_target=lambda target: target,
                                       ancestor_display_prefix=lambda _target: ())
        self.messages = SimpleNamespace(assert_resume_safe=lambda *_a, **_k: 0,
                                        read_resume_conversations=lambda _target: SimpleNamespace(model=(), display=()),
                                        read_conversation=lambda *_a, **_k: ())

    def close(self):
        pass


@pytest.fixture()
def homes(monkeypatch, tmp_path):
    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_profile_session_db",
                        lambda home: (_DB(db_path=home / "state.db"), True) if home else (server._get_db(), False))
    monkeypatch.setattr(server, "_persistence_beside", lambda store: store)
    monkeypatch.setattr(server, "_release_persistence", lambda _handle: None)
    monkeypatch.setattr(server, "_profile_home", lambda p: homes.get(p) if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_child_run_active", lambda *a, **k: False)
    monkeypatch.setattr(
        server, "_live_session_payload", lambda sid, session, **_k: {"session_id": sid, "message_count": 0, "messages": [], "info": {}}
    )
    known = set(server._sessions)
    yield homes
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def _resume(**params):
    return server.handle_request({"id": "1", "method": "session.resume", "params": params})


def _register_live(sid: str, profile_home) -> dict:
    record = {"session_key": "s1", "history": [], "last_active": 0.0}
    if profile_home is not None:
        record["profile_home"] = str(profile_home)
    with server._sessions_lock:
        server._sessions[sid] = record
    return record


def test_resume_with_other_profile_never_reuses_live_session(homes):
    _register_live("live-a", homes["a"])

    same = _resume(session_id="s1", profile="a", source="desktop")
    assert same["result"]["session_id"] == "live-a"

    other = _resume(session_id="s1", profile="b", source="desktop")
    new_sid = other["result"]["session_id"]
    assert new_sid != "live-a"
    assert server._sessions[new_sid]["profile_home"] == str(homes["b"])


def test_launch_profile_still_matches_records_without_profile_home(homes):
    _register_live("live-launch", None)
    _register_live("live-a", homes["a"])

    assert _resume(session_id="s1", source="desktop")["result"]["session_id"] == "live-launch"
    assert _resume(session_id="s1", profile="a", source="desktop")["result"]["session_id"] == "live-a"

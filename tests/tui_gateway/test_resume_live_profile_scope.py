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

import pytest

from tui_gateway import server


class _DB:
    """Minimal ``SessionDB`` stand-in: every profile store knows ``s1``."""

    def __init__(self, db_path=None, **_kwargs):
        self.db_path = db_path

    def close(self):
        pass

    def get_session(self, target):
        return {"id": "s1", "cwd": ""} if target == "s1" else None

    def get_session_by_title(self, _target):
        return None

    def resolve_resume_session_id(self, target):
        return target

    def reopen_session(self, _target):
        pass

    def get_resume_conversations(self, _target):
        return ([], [])

    def get_ancestor_display_prefix(self, _target):
        return []

    def get_messages_as_conversation(self, _target, **_kwargs):
        return []


@pytest.fixture()
def homes(monkeypatch, tmp_path):
    homes = {name: tmp_path / name for name in ("a", "b")}
    for home in homes.values():
        home.mkdir()
    monkeypatch.setattr("hermes_state.get_shared_session_db", _DB)
    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    monkeypatch.setattr(server, "_profile_home", lambda p: homes.get(p) if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_child_run_active", lambda _key: False)
    monkeypatch.setattr(
        server, "_live_session_payload", lambda sid, session, **_k: {"session_id": sid}
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

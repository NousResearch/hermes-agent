"""The child-run liveness registry must be profile-scoped on the resume path.

Stored session ids are timestamps that exist in several profiles' stores. A lazy
``session.resume`` trusts ``_child_run_active`` to skip database verification
for a subagent window whose first flush has not landed yet — so a bare-key
registry hit let an authenticated client on Profile B resume Profile A's
actively-running child id: the gateway accepted it without verifying B's
database ownership and mirrored A's live turn into B's transport (#120212).

Pinned here, in both directions:

* a lazy resume for a child id that is only active under ANOTHER profile is
  refused with the same "session not found" as any id missing from the
  caller's own store (fail-closed);
* a lazy resume under the OWNING profile still bypasses the database lookup
  and opens the watch window, as before.
"""

from __future__ import annotations

import pytest

from tui_gateway import server


class _EmptyDB:
    """Stand-in for ``hermes_state.SessionDB``: the caller's own store, which
    holds no row for the foreign child id."""

    def __init__(self, db_path=None, **_kwargs):
        self.db_path = db_path
        self.closed = 0

    def close(self):
        self.closed += 1

    def get_session(self, _target):
        return None

    def get_session_by_title(self, _target):
        return None

    def resolve_resume_session_id(self, target):
        return target

    def reopen_session(self, _target):
        return None

    def get_resume_conversations(self, _target):
        return ([], [])

    def get_ancestor_display_prefix(self, _target):
        return []

    def get_messages_as_conversation(self, _target, **_kwargs):
        return []


@pytest.fixture()
def two_profiles(monkeypatch, tmp_path):
    homes = {}
    for name in ("a", "b"):
        home = tmp_path / name
        home.mkdir()
        homes[name] = home
    monkeypatch.setattr(server, "_profile_home", lambda profile: homes.get(profile))
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _home: str(tmp_path))
    monkeypatch.setattr("hermes_state_registry.acquire", _EmptyDB)
    monkeypatch.setattr(server, "_find_live_session_by_key", lambda _key, *_a, **_k: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_default_session_cwd", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    known = set(server._sessions)
    yield homes
    with server._sessions_lock:
        for sid in [s for s in server._sessions if s not in known]:
            server._sessions.pop(sid, None)


def _resume(**params):
    return server.handle_request(
        {"id": "1", "method": "session.resume", "params": params}
    )


def _seed_active_run(server, profile_home, child_key):
    """Register an in-flight child run the way the delegate relay does, so the
    read side is exercised against the registry's real key form."""
    server._mirror_subagent_to_child("subagent.tool", {"child_session_id": child_key}, profile_home)


def test_lazy_resume_refuses_child_id_active_under_another_profile(two_profiles):
    _seed_active_run(server, str(two_profiles["a"]), "child-key")
    try:
        resp = _resume(session_id="child-key", profile="b", lazy=True)
    finally:
        server._active_child_runs.clear()

    assert resp["error"]["code"] == 4007


def test_lazy_resume_still_opens_watch_window_under_owning_profile(two_profiles):
    _seed_active_run(server, str(two_profiles["a"]), "child-key")
    try:
        resp = _resume(session_id="child-key", profile="a", lazy=True)
    finally:
        server._active_child_runs.clear()

    assert "error" not in resp

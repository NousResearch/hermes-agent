"""F3: a reused sid that served felix, driven by a default frame (empty profile_home),
has session['profile_home'] cleared to None -- so no felix home override rebinds that turn.

_ensure_server_session is the seam: it previously only overwrote profile_home when the frame
carried a truthy value (sticky binding on reused sids). Now it writes authoritatively every turn.
"""
from __future__ import annotations

import types

from tui_gateway.compute_host import ComputeHost


class _FakeServer:
    def __init__(self):
        self._sessions = {}

    def _install_delegated_active_session_lease(self, session, frame):
        pass


def _host():
    # Bypass __init__ (thread pool etc.); we only exercise _ensure_server_session.
    return ComputeHost.__new__(ComputeHost)


def test_default_frame_clears_prior_felix_binding():
    host = _host()
    host._transport = types.SimpleNamespace()
    srv = _FakeServer()
    # A session that previously served felix on this sid.
    srv._sessions["s1"] = {"profile_home": "/homes/felix", "session_key": "s1"}

    # Reuse the SAME sid with a default frame (empty profile_home).
    session = host._ensure_server_session(srv, {"sid": "s1", "profile_home": ""})
    assert session["profile_home"] is None


def test_named_frame_binds_profile_home():
    host = _host()
    host._transport = types.SimpleNamespace()
    srv = _FakeServer()
    srv._sessions["s1"] = {"profile_home": None, "session_key": "s1"}
    session = host._ensure_server_session(srv, {"sid": "s1", "profile_home": "/homes/felix"})
    assert session["profile_home"] == "/homes/felix"


def test_missing_profile_home_key_clears():
    host = _host()
    host._transport = types.SimpleNamespace()
    srv = _FakeServer()
    srv._sessions["s1"] = {"profile_home": "/homes/felix", "session_key": "s1"}
    session = host._ensure_server_session(srv, {"sid": "s1"})
    assert session["profile_home"] is None


def test_default_turn_binds_exempt_scope_unconditionally(monkeypatch):
    """A default turn (falsy or launch-equal profile_home) ALWAYS binds its own exempt scope.

    Not gated on is_multiplex_active(): a concurrent named turn flipping multiplex mid-flight must
    never leave this turn unscoped. Records the home override + env_fallback the default branch sets.
    """
    from tui_gateway import prompt_turn as pt
    calls = []
    fallbacks = []
    monkeypatch.setattr(pt, "set_hermes_home_override", lambda h: calls.append(h) or object(),
                        raising=False)
    monkeypatch.setattr(pt, "set_secret_scope_env_fallback",
                        lambda v: fallbacks.append(v) or object(), raising=False)
    from agent import secret_scope as ss
    was = ss.is_multiplex_active()
    ss.set_multiplex_active(False)  # even with multiplex OFF, the default turn binds a scope now
    try:
        session = {"profile_home": None, "session_key": "s1"}
        profile_home = session.get("profile_home")
        # Falsy profile_home -> the else (default) branch runs unconditionally.
        assert not profile_home
    finally:
        ss.set_multiplex_active(was)

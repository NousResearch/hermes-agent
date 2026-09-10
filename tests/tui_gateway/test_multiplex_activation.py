"""F1: the compute_host/tui_gateway turn path becomes a multiplexer once a turn serves a named
sibling profile. Flipping happens in _prepare_turn_input (bound to the real turn), NOT in the
shared _profile_home resolver -- other RPCs (groups.peer.invite, config, bot relay) call that
resolver too and must not trip fail-closed scoping.

A launch-only turn (falsy profile_home) never flips multiplex, so single-profile installs keep
their legacy os.environ secret semantics.
"""
from __future__ import annotations

import pytest

import tui_gateway.server as server
from agent import secret_scope as ss


class _Sentinel(Exception):
    """Raised from the patched _wire_callbacks to stop the turn right after the scope block."""


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    was_active = ss.is_multiplex_active()
    ss.set_multiplex_active(False)
    # Neutralize every scope installer the block calls so the test observes decisions, not effects.
    monkeypatch.setattr(server, "set_multiplex_active",
                        lambda v: _flips.append(v) or ss.set_multiplex_active(v), raising=False)
    monkeypatch.setattr(server, "set_hermes_home_override", lambda h: object(), raising=False)
    monkeypatch.setattr(server, "set_secret_scope", lambda s: object(), raising=False)
    monkeypatch.setattr(server, "build_profile_secret_scope", lambda p: object(), raising=False)
    monkeypatch.setattr(server, "set_secret_scope_env_fallback",
                        lambda v: _fallbacks.append(v) or object(), raising=False)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **k: object(), raising=False)
    import tools.terminal_scope as tscope
    monkeypatch.setattr(tscope, "install_profile_terminal_scope", lambda p: object(), raising=False)
    import tools.approval_context as approval
    monkeypatch.setattr(approval, "set_current_session_key", lambda k: object(), raising=False)
    # Stop the turn right after the scope block.
    monkeypatch.setattr(server, "_wire_callbacks", lambda sid: (_ for _ in ()).throw(_Sentinel()),
                        raising=False)
    _flips.clear()
    _fallbacks.clear()
    yield
    ss.set_multiplex_active(was_active)


_flips: list[bool] = []
_fallbacks: list[bool] = []


def _run_scope_block(profile_home):
    """Drive server._prepare_turn_input through the scope block; swallow the sentinel."""
    from tui_gateway.prompt_turn import _TurnRun
    st = _TurnRun.__new__(_TurnRun)
    from tui_gateway.prompt_turn import _TurnScopes
    st.scopes = _TurnScopes()
    session = {"session_key": "s1", "profile_home": profile_home}
    try:
        server._prepare_turn_input("s1", session, st, "hi", [])
    except _Sentinel:
        pass


def test_named_profile_turn_activates_multiplex():
    _run_scope_block("/homes/felix")
    assert True in _flips
    assert ss.is_multiplex_active() is True
    assert True not in _fallbacks  # named profile is fully fail-closed, never exempt


def test_launch_only_turn_leaves_multiplex_inactive():
    _run_scope_block("")  # falsy profile_home -> launch/default turn
    assert True not in _flips
    assert ss.is_multiplex_active() is False


def test_default_turn_binds_exempt_scope_even_when_multiplex_inactive():
    # The default branch runs unconditionally (not gated on is_multiplex_active): a concurrent
    # named turn flipping multiplex mid-flight must never leave this turn unscoped.
    ss.set_multiplex_active(False)
    _run_scope_block("")
    assert True not in _flips  # no flip
    assert _fallbacks == [True]  # but the exempt env-fallback marker IS installed


def test_profile_home_equal_to_launch_does_not_activate(monkeypatch):
    # Distinctness gate: a frame carrying the LAUNCH home must not take the fully fail-closed named
    # path (that would hard-break env-injected creds) -> falls through to the exempt default branch.
    launch = str(server._hermes_home)
    _run_scope_block(launch)
    assert True not in _flips
    assert ss.is_multiplex_active() is False
    assert _fallbacks == [True]  # exempt branch, not the fail-closed named branch

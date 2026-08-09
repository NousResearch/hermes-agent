"""Desktop/TUI first-contact profile-build onboarding via tui_gateway (#82750).

The messaging gateway stages the consent-gated profile-build offer on the
install's very first message (gateway/run_turn.py ``_hmwa_first_contact_notes``);
the TUI/Desktop surface must do the same through
``_stage_first_contact_onboarding_note`` in the prompt turn.
"""

from __future__ import annotations

import threading
import types

import pytest
import yaml

from agent.onboarding import PROFILE_BUILD_FLAG, profile_build_directive
from tui_gateway import server


def _session(agent, history=None):
    return {
        "agent": agent,
        "session_key": "session-key",
        "history": list(history or []),
        "history_lock": threading.Lock(),
    }


@pytest.fixture()
def onboarding_home(monkeypatch, tmp_path):
    """A HERMES_HOME whose config.yaml offers profile builds (the default mode)."""
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"onboarding": {"profile_build": "ask"}})
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _stage(session, agent, history_empty):
    server._stage_first_contact_onboarding_note(session, agent, history_empty)


def test_stages_profile_build_directive_on_first_contact(monkeypatch, onboarding_home):
    """Fresh install + empty history: the opt-in directive is staged on the agent
    and the offered flag is persisted before the turn runs."""
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    agent = types.SimpleNamespace()
    _stage(_session(agent), agent, history_empty=True)

    assert agent._gateway_turn_context_notes == profile_build_directive().strip()
    loaded = yaml.safe_load((onboarding_home / "config.yaml").read_text())
    assert loaded["onboarding"]["seen"][PROFILE_BUILD_FLAG] is True


def test_skips_first_contact_when_prior_sessions_exist(monkeypatch, onboarding_home):
    """An install that already holds conversations must not re-offer; nothing is
    staged and the config is untouched."""
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: True)

    agent = types.SimpleNamespace()
    _stage(_session(agent), agent, history_empty=True)

    assert getattr(agent, "_gateway_turn_context_notes", "") == ""
    loaded = yaml.safe_load((onboarding_home / "config.yaml").read_text())
    assert "seen" not in loaded.get("onboarding", {})


def test_skips_first_contact_when_history_not_empty(monkeypatch, onboarding_home):
    """A continuing conversation (history present) is not first contact."""
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    agent = types.SimpleNamespace()
    _stage(
        _session(agent, history=[{"role": "user", "content": "prior"}]),
        agent,
        history_empty=False,
    )

    assert getattr(agent, "_gateway_turn_context_notes", "") == ""


def test_skips_when_offer_already_latched(monkeypatch, onboarding_home):
    """``onboarding.seen.profile_build_offered`` set: the plain intro rides
    instead of the directive, exactly as the gateway path behaves."""
    (onboarding_home / "config.yaml").write_text(
        yaml.safe_dump(
            {"onboarding": {"profile_build": "ask", "seen": {PROFILE_BUILD_FLAG: True}}}
        )
    )
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    agent = types.SimpleNamespace()
    _stage(_session(agent), agent, history_empty=True)

    from agent.onboarding import PLAIN_INTRO_NOTE

    assert agent._gateway_turn_context_notes == PLAIN_INTRO_NOTE.strip()


def test_installs_prior_sessions_probe_counts_the_install(monkeypatch):
    """The DB probe is real: a state.db holding two rows means prior sessions,
    one row (the persisted current session) means a fresh install."""
    calls = {}

    class _DB:
        def __init__(self, n):
            self._n = n

        def session_count_ge(self, minimum):
            calls["minimum"] = minimum
            return self._n >= minimum

    class _Ctx:
        def __init__(self, db):
            self._db = db

        def __enter__(self):
            return self._db

        def __exit__(self, *exc):
            return False

    fresh = _session(types.SimpleNamespace())
    monkeypatch.setattr(server, "_session_db", lambda _s: _Ctx(_DB(1)))
    assert server._install_has_prior_sessions(fresh) is False
    assert calls["minimum"] == 2

    monkeypatch.setattr(server, "_session_db", lambda _s: _Ctx(_DB(2)))
    assert server._install_has_prior_sessions(fresh) is True

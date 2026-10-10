"""Desktop/TUI first-contact onboarding via tui_gateway (#82750).

The messaging gateway stages the offer on the
install's very first message (gateway/run_turn.py ``_hmwa_first_contact_notes``);
the TUI/Desktop surface must do the same through
``_stage_first_contact_onboarding_note`` in the prompt turn.
"""

from __future__ import annotations

import threading
import types

import pytest

from agent.onboarding import PROFILE_BUILD_FLAG, SETUP_OFFER_NOTE, profile_build_directive
from hermes_yaml import safe_dump, safe_load
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
        safe_dump({"onboarding": {"profile_build": "ask"}})
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _stage(session, agent, history_empty):
    server._stage_first_contact_onboarding_note(session, agent, history_empty, "hello")


def test_stages_setup_offer_on_first_contact(monkeypatch, onboarding_home):
    """Fresh install + empty history under guest onboarding: the /initiate-setup offer is staged
    on the agent and the offered flag is persisted before the turn runs."""
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    agent = types.SimpleNamespace()
    _stage(_session(agent), agent, history_empty=True)

    assert agent._gateway_turn_context_notes == SETUP_OFFER_NOTE.format(command="/initiate-setup")
    loaded = safe_load((onboarding_home / "config.yaml").read_text())
    assert loaded["onboarding"]["seen"][PROFILE_BUILD_FLAG] is True


def test_stages_profile_build_offer_without_guest_onboarding(monkeypatch, onboarding_home):
    """Without guest onboarding the first message keeps main's memory profile offer."""
    monkeypatch.delenv("HERMES_GUEST_ONBOARDING", raising=False)
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    agent = types.SimpleNamespace()
    _stage(_session(agent), agent, history_empty=True)

    assert agent._gateway_turn_context_notes == profile_build_directive().strip()


def test_skips_first_contact_when_prior_sessions_exist(monkeypatch, onboarding_home):
    """An install that already holds conversations must not re-offer; nothing is
    staged and the config is untouched."""
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: True)

    agent = types.SimpleNamespace()
    _stage(_session(agent), agent, history_empty=True)

    assert getattr(agent, "_gateway_turn_context_notes", "") == ""
    loaded = safe_load((onboarding_home / "config.yaml").read_text())
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
        safe_dump(
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


def test_the_task_chat_setup_hands_off_to_gets_no_note(monkeypatch, onboarding_home):
    """start_chat marks the session it opens from setup; the prompt turn passes that on."""
    monkeypatch.setenv("HERMES_GUEST_ONBOARDING", "1")
    monkeypatch.setattr(server, "_install_has_prior_sessions", lambda _s: False)

    agent = types.SimpleNamespace()
    session = {**_session(agent), "setup_handoff": True}
    _stage(session, agent, history_empty=True)

    assert getattr(agent, "_gateway_turn_context_notes", None) is None
    assert PROFILE_BUILD_FLAG not in (onboarding_home / "config.yaml").read_text()

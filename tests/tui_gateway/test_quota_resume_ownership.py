"""Ownership and lifetime of a pending quota resume: whose request it serves, which route it
runs on, who may cancel it, and what keeps its session alive until it fires."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture()
def server(monkeypatch):
    import tui_gateway.server as server_mod

    monkeypatch.setattr(server_mod, "_quota_resume_config", lambda: (True, 2))
    monkeypatch.setattr(server_mod, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server_mod, "_ensure_active_session_slot", lambda sid, s: None)
    return server_mod


def _session(**over) -> dict:
    session = {"history_lock": threading.RLock(), "running": False,
               "agent": SimpleNamespace(provider="anthropic", model="opus-4-6")}
    session.update(over)
    return session


def _plan(delay: float = -1.0) -> dict:
    return {"eligible": True, "resume_at": time.time() + delay, "provider": "anthropic"}


def _capture(server, monkeypatch) -> list:
    submitted: list = []
    monkeypatch.setattr(server, "_run_prompt_submit",
                        lambda rid, sid, session, text, **kw: submitted.append((rid, text)))
    return submitted


def test_a_new_request_gets_a_fresh_retry_budget(server, monkeypatch):
    """Attempts belong to the refused request: two resumes earlier in a long session must not
    disable the feature for every later task."""
    session = _session(_quota_resume_attempt=2)
    assert server._arm_quota_resume("s", session, _plan(600), "old task") is False
    server._quota_resume_take_turn(session, "user-turn-7")
    assert server._arm_quota_resume("s", session, _plan(600), "new task") is True


def test_resume_turn_keeps_the_budget_and_the_original_prompt(server, monkeypatch):
    submitted = _capture(server, monkeypatch)
    session = _session()
    server._arm_quota_resume("s", session, _plan(), "finish the migration")
    server._maybe_fire_quota_resume("s", session)
    rid, note = submitted[0]
    server._quota_resume_take_turn(session, rid)  # the resume's own dispatch
    session["running"] = False
    assert server._arm_quota_resume("s", session, _plan(600), note) is True
    assert session["_quota_resume_prompt"] == "finish the migration"  # not a nested note
    assert session["_quota_resume_attempt"] == 1


def test_model_switch_drops_the_armed_resume(server, monkeypatch):
    submitted = _capture(server, monkeypatch)
    session = _session()
    server._arm_quota_resume("s", session, _plan(), "finish the migration")
    session["agent"] = SimpleNamespace(provider="openrouter", model="some/other-model")
    server._maybe_fire_quota_resume("s", session)
    assert submitted == []
    assert "_quota_resume_at" not in session
    assert session["running"] is False


def test_resume_cancelled_between_due_check_and_claim_never_runs(server, monkeypatch):
    """A user turn that clears the resume while the poller waits for the turn claim wins."""
    submitted = _capture(server, monkeypatch)
    session = _session()
    server._arm_quota_resume("s", session, _plan(), "stale request")
    real_claim = server._notif_claim_turn

    def _user_turn_slips_in(sess):
        server._clear_quota_resume(sess)  # a user turn ran and finished in between
        return real_claim(sess)

    monkeypatch.setattr(server, "_notif_claim_turn", _user_turn_slips_in)
    server._maybe_fire_quota_resume("s", session)
    assert submitted == []
    assert session["running"] is False


def test_pending_resume_keeps_a_detached_session_alive(server):
    session = _session(_quota_resume_at=time.time() + 3600, _quota_resume_prompt="p")
    assert server._session_owns_live_wakeup_schedule(session) is True
    server._clear_quota_resume(session)
    assert server._session_owns_live_wakeup_schedule(session) is False

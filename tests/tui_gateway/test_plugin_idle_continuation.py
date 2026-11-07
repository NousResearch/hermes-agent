"""Host boundary of the native ``on_session_idle`` plugin hook.

A neutral in-process hook stands in for a plugin; ``_run_prompt_submit`` is the only
seam replaced, so the admission, identity and release logic under test is the real one.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture
def idle(tmp_path, monkeypatch):
    hooks, submits = [], []
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook",
                        lambda name, **kw: hooks.append((name, kw)) or [])

    def run_prompt_submit(rid, sid, session, message, **kw):
        submits.append((sid, message, kw))
        return True

    monkeypatch.setattr(server, "_run_prompt_submit", run_prompt_submit)
    session = {"agent": SimpleNamespace(session_id="owner"), "session_key": "owner", "source": "desktop",
               "profile_home": str(tmp_path), "history_lock": threading.RLock(), "running": False, "history": []}
    monkeypatch.setitem(server._sessions, "runtime", session)
    return SimpleNamespace(session=session, hooks=hooks, submits=submits)


def test_idle_hook_receives_owner_identity_and_one_submit(idle, monkeypatch):
    def hook(name, **kw):
        idle.hooks.append((name, kw))
        assert kw["submit"]("reevaluate the current task", terminal_callback=lambda result: None) is True
        # One admission per offer: a second submit in the same hook is refused.
        assert kw["submit"]("again", terminal_callback=lambda result: None) is False
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    assert server._poll_plugin_idle_once("runtime", idle.session) is True
    name, kw = idle.hooks[0]
    assert name == "on_session_idle"
    assert kw["session_id"] == "owner" and kw["source"] == "desktop"
    assert str(kw["hermes_home"]) == idle.session["profile_home"]
    assert [message for _, message, _ in idle.submits] == ["reevaluate the current task"]
    assert idle.session["running"] is True


@pytest.mark.parametrize("blocker", ["running", "_closing", "queued_prompt", "_auto_continue_scheduled"])
def test_busy_session_is_not_offered(idle, blocker):
    idle.session[blocker] = True
    assert server._poll_plugin_idle_once("runtime", idle.session) is False
    assert idle.hooks == [] and idle.submits == []


def test_submit_retained_past_the_hook_is_refused(idle, monkeypatch):
    kept = []
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook",
                        lambda name, **kw: kept.append(kw["submit"]) or [])
    assert server._poll_plugin_idle_once("runtime", idle.session) is False
    assert kept[0]("late", terminal_callback=lambda result: None) is False
    assert idle.submits == [] and idle.session["running"] is False


def test_refused_native_admission_releases_the_claim(idle, monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **kw: False)
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda name, **kw: [
        kw["submit"]("reevaluate", terminal_callback=lambda result: None)])
    assert server._poll_plugin_idle_once("runtime", idle.session) is False
    assert idle.session["running"] is False

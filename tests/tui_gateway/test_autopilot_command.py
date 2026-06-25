"""Tests for /autopilot mirroring in tui_gateway.

The TUI runs /autopilot in the persistent _SlashWorker subprocess, so its
os.environ / self.agent mutations never reach the live gateway agent. The fix
mirrors the toggle in ``_mirror_slash_side_effects`` (like /fast) and persists
state on the session dict so it survives agent rebuilds. These tests cover that
the live agent's ``autopilot_mode`` is set and surfaced in ``_session_info``.
"""

from __future__ import annotations

import importlib
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")
        yield mod
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


def _make_session(server, *, autopilot=False, goal=""):
    agent = types.SimpleNamespace(
        autopilot_mode=autopilot, model="gpt-4o", session_id="k1", service_tier=None
    )
    sid = "sid-ap"
    s = {
        "session_key": "tui-ap-1",
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    if autopilot:
        s["autopilot"] = True
    if goal:
        s["autopilot_goal"] = goal
    server._sessions[sid] = s
    return sid, s, agent


def _mirror(server, sid, s, command):
    return server._mirror_slash_side_effects(sid, s, command)


# ── _mirror_slash_side_effects: /autopilot ──────────────────────────────


def test_autopilot_on_sets_live_agent(server):
    sid, s, agent = _make_session(server)
    _mirror(server, sid, s, "/autopilot on")
    assert agent.autopilot_mode is True
    assert s["autopilot"] is True


def test_autopilot_off_clears_live_agent(server):
    sid, s, agent = _make_session(server, autopilot=True)
    _mirror(server, sid, s, "/autopilot off")
    assert agent.autopilot_mode is False
    assert s["autopilot"] is False


def test_autopilot_bare_toggles_from_session_state(server):
    sid, s, agent = _make_session(server)
    _mirror(server, sid, s, "/autopilot")
    assert agent.autopilot_mode is True
    _mirror(server, sid, s, "/autopilot")
    assert agent.autopilot_mode is False


def test_autopilot_goal_arg_enables_and_sets_goal(server):
    sid, s, agent = _make_session(server)
    _mirror(server, sid, s, "/autopilot goal ship the feature")
    assert agent.autopilot_mode is True
    assert agent._autopilot_goal == "ship the feature"
    assert s["autopilot_goal"] == "ship the feature"


def test_autopilot_bare_positional_is_not_a_goal(server):
    # Regression: a bare positional must NOT be treated as a goal (the old
    # overload made `/autopilot off now` enable with goal "off now").
    sid, s, agent = _make_session(server)
    _mirror(server, sid, s, "/autopilot ship the feature")
    assert agent.autopilot_mode is False  # unknown arg → no state change
    assert s.get("autopilot_goal", "") == ""


def test_autopilot_off_with_trailing_word_still_disables(server):
    sid, s, agent = _make_session(server, autopilot=True)
    _mirror(server, sid, s, "/autopilot off now")
    assert agent.autopilot_mode is False


def test_autopilot_goal_clear_resets_goal(server):
    sid, s, agent = _make_session(server)
    _mirror(server, sid, s, "/autopilot goal ship it")
    assert agent._autopilot_goal == "ship it"
    _mirror(server, sid, s, "/autopilot clear")
    assert agent._autopilot_goal == ""
    assert s["autopilot_goal"] == ""


def test_autopilot_status_does_not_change_state(server):
    sid, s, agent = _make_session(server, autopilot=True)
    _mirror(server, sid, s, "/autopilot status")
    assert agent.autopilot_mode is True  # unchanged


# ── persistence across agent rebuild ────────────────────────────────────


def test_apply_autopilot_to_rebuilt_agent(server):
    sid, s, _ = _make_session(server, autopilot=True, goal="finish X")
    new_agent = types.SimpleNamespace(autopilot_mode=False)
    server._apply_autopilot_to_agent(s, new_agent)
    assert new_agent.autopilot_mode is True
    assert new_agent._autopilot_goal == "finish X"


def test_apply_autopilot_noop_when_off(server):
    sid, s, _ = _make_session(server)
    new_agent = types.SimpleNamespace(autopilot_mode=True)
    server._apply_autopilot_to_agent(s, new_agent)
    assert new_agent.autopilot_mode is False  # session has no autopilot → off


# ── _session_info surfaces autopilot ────────────────────────────────────


def test_session_info_reports_autopilot(server):
    sid, s, agent = _make_session(server, autopilot=True)
    agent.autopilot_mode = True
    info = server._session_info(agent, s)
    assert info["autopilot"] is True


def test_session_info_autopilot_false_by_default(server):
    sid, s, agent = _make_session(server)
    info = server._session_info(agent, s)
    assert info["autopilot"] is False


# ── enable-kick (resume turn when enabled idle) ─────────────────────────


def test_enable_kick_starts_turn_when_idle_with_history(server, monkeypatch):
    calls = {}

    def fake_run(rid, sid, sess, text):
        calls["text"] = text
        calls["sid"] = sid

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    sid, s, agent = _make_session(server)
    s["history"] = [{"role": "user", "content": "do X"}]
    started = server._kick_autopilot_turn(sid, s)
    assert started is True
    assert calls["text"].startswith("[Autopilot]")
    assert s["running"] is True


def test_enable_kick_noop_when_no_history(server, monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: None)
    sid, s, agent = _make_session(server)
    s["history"] = []
    assert server._kick_autopilot_turn(sid, s) is False


def test_enable_kick_starts_on_goal_when_no_history_but_goal_set(server, monkeypatch):
    # Regression: `/autopilot goal <text>` on a cold TUI session (no history)
    # must START on the goal text itself, not bail.
    calls = {}

    def fake_run(rid, sid, sess, text):
        calls["text"] = text

    monkeypatch.setattr(server, "_run_prompt_submit", fake_run)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    sid, s, agent = _make_session(server)
    s["history"] = []
    s["autopilot_goal"] = "ship the feature"
    started = server._kick_autopilot_turn(sid, s)
    assert started is True
    assert calls["text"] == "ship the feature"  # drives the goal as the opening task
    assert s["running"] is True


def test_enable_kick_noop_when_running(server, monkeypatch):
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *a, **k: None)
    sid, s, agent = _make_session(server)
    s["history"] = [{"role": "user", "content": "x"}]
    s["running"] = True
    assert server._kick_autopilot_turn(sid, s) is False


def test_mirror_on_triggers_kick_when_idle(server, monkeypatch):
    fired = {"n": 0}
    monkeypatch.setattr(server, "_kick_autopilot_turn", lambda sid, s: fired.__setitem__("n", fired["n"] + 1))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    sid, s, agent = _make_session(server)
    s["history"] = [{"role": "user", "content": "x"}]
    server._mirror_slash_side_effects(sid, s, "/autopilot on")
    assert fired["n"] == 1


def test_mirror_off_does_not_kick(server, monkeypatch):
    fired = {"n": 0}
    monkeypatch.setattr(server, "_kick_autopilot_turn", lambda sid, s: fired.__setitem__("n", fired["n"] + 1))
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    sid, s, agent = _make_session(server, autopilot=True)
    server._mirror_slash_side_effects(sid, s, "/autopilot off")
    assert fired["n"] == 0

"""Behavioral contract tests for automation update actions on session.control."""

from __future__ import annotations

import importlib
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals
    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture()
def server(hermes_home, monkeypatch):
    with patch.dict(
        "sys.modules",
        {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()},
    ):
        mod = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_mtime", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()


@pytest.fixture()
def session(server):
    sid = f"sid-update-{uuid.uuid4().hex}"
    key = f"update-{uuid.uuid4().hex}"
    entry = {
        "session_key": key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
        "agent": None,
        "created_at": time.time(),
    }
    server._sessions[sid] = entry
    yield sid, key, entry
    from hermes_cli.goals import GoalManager
    from hermes_cli.heartbeat import HeartbeatManager
    from hermes_cli.loops import LoopManager
    GoalManager(key).clear()
    LoopManager(key).clear()
    HeartbeatManager(key).clear()


def _call(server, method, *, rid=91, **params):
    return server._methods[method](rid, params)


def _error(response):
    assert "error" in response
    return response["error"]


def _forbid_dispatch(server, monkeypatch):
    def forbidden(_rid, _params):
        raise AssertionError("update action must not call command.dispatch")
    monkeypatch.setitem(server._methods, "command.dispatch", forbidden)


class TestGoalUpdate:
    def test_goal_update_requires_existing_goal(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "new objective",
        }))
        assert err["code"] == 4004

    def test_goal_update_changes_objective_and_max_turns(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalManager, GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="original", status="active", turns_used=5, max_turns=20, created_at=100.0))
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "revised objective",
            "max_turns": 30,
        })
        assert "error" not in response
        goal = response["result"]["control"]["goal"]
        assert goal["title"] == "revised objective"
        assert goal["max_turns"] == 30

    def test_goal_update_preserves_runtime_state(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal, load_goal
        sid, key, _ = session
        save_goal(key, GoalState(
            goal="original", status="active", turns_used=7, max_turns=20,
            created_at=100.0, last_turn_at=200.0,
            last_verdict="continue", last_reason="still working",
        ))
        _forbid_dispatch(server, monkeypatch)
        _call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "updated",
        })
        state = load_goal(key)
        assert state.turns_used == 7
        assert state.created_at == 100.0
        assert state.last_turn_at == 200.0
        assert state.last_verdict == "continue"
        assert state.last_reason == "still working"
        assert state.status == "active"

    def test_goal_update_preserves_subgoals_when_not_provided(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal, load_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="original", status="active", subgoals=["criterion A", "criterion B"]))
        _forbid_dispatch(server, monkeypatch)
        _call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "updated",
        })
        state = load_goal(key)
        assert state.subgoals == ["criterion A", "criterion B"]

    def test_goal_update_replaces_criteria_when_provided(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal, load_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="original", status="active", subgoals=["old criterion"]))
        _forbid_dispatch(server, monkeypatch)
        _call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "updated",
            "criteria": ["new criterion 1", "new criterion 2"],
        })
        state = load_goal(key)
        assert state.subgoals == ["new criterion 1", "new criterion 2"]

    def test_goal_update_rejects_empty_prompt(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="existing", status="active"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "",
        }))
        assert err["code"] == 4004

    def test_goal_update_publishes_snapshot(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="original", status="active"))
        _forbid_dispatch(server, monkeypatch)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, sid_, payload=None: emitted.append((event, sid_, payload)))
        _call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "updated",
        })
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1
        assert updates[0][2]["control"]["goal"]["title"] == "updated"

    def test_goal_update_fails_closed_without_session_history_lock(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, load_goal, save_goal

        sid, key, entry = session
        save_goal(key, GoalState(goal="original", status="active"))
        entry.pop("history_lock")
        _forbid_dispatch(server, monkeypatch)

        response = _call(server, "session.control", session_id=sid, action="goal.update", args={
            "prompt": "updated",
        })

        assert _error(response)["code"] == 4004
        assert "lock" in response["error"]["message"].lower()
        assert load_goal(key).goal == "original"


class TestLoopUpdate:
    def test_loop_update_requires_existing_loop(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "updated prompt",
            "interval_seconds": 120,
        }))
        assert err["code"] == 4004

    def test_loop_update_changes_prompt_and_interval(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60, created_at=100.0,
        ))
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "revised prompt",
            "interval_seconds": 120,
        })
        assert "error" not in response
        loop = response["result"]["control"]["loop"]
        assert loop["prompt"] == "revised prompt"
        assert loop["interval_seconds"] >= 120

    def test_loop_update_preserves_route_and_history(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop, load_loop
        sid, key, _ = session
        save_loop(key, LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            ticks_fired=5, created_at=100.0, last_fired_at=500.0,
            route={"platform": "desktop", "chat_id": "abc"},
        ))
        _forbid_dispatch(server, monkeypatch)
        _call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        })
        state = load_loop(key)
        assert state.ticks_fired == 5
        assert state.created_at == 100.0
        assert state.last_fired_at == 500.0
        assert state.route == {"platform": "desktop", "chat_id": "abc"}
        assert state.status == "active"

    def test_loop_update_refuses_while_awaiting_response(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            awaiting_response=True,
        ))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        }))
        assert err["code"] == 4004

    def test_loop_update_refuses_run_cap_below_ticks_fired(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            ticks_fired=10,
        ))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
            "run_limit": 5,
        }))
        assert err["code"] == 4004

    def test_loop_update_allows_run_cap_at_or_above_ticks_fired(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
            ticks_fired=10,
        ))
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
            "run_limit": 10,
        })
        assert "error" not in response
        assert response["result"]["control"]["loop"]["times"] == 10

    def test_loop_update_publishes_snapshot(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(
            prompt="original", status="active", mode="interval",
            interval_seconds=60, current_delay=60,
        ))
        _forbid_dispatch(server, monkeypatch)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, sid_, payload=None: emitted.append((event, sid_, payload)))
        _call(server, "session.control", session_id=sid, action="loop.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        })
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1
        assert updates[0][2]["control"]["loop"]["prompt"] == "updated"


class TestHeartbeatUpdate:
    def test_heartbeat_update_requires_existing_heartbeat(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        }))
        assert err["code"] == 4004

    def test_heartbeat_update_changes_prompt_and_interval(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(prompt="original", interval_seconds=60, status="active", created_at=100.0))
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="heartbeat.update", args={
            "prompt": "revised",
            "interval_seconds": 300,
        })
        assert "error" not in response
        hb = response["result"]["control"]["heartbeat"]
        assert hb["prompt"] == "revised"
        assert hb["interval_seconds"] == 300

    def test_heartbeat_update_preserves_counters_and_timestamps(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat, load_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(
            prompt="original", interval_seconds=60, status="active",
            created_at=100.0, last_fired_at=500.0, fire_count=8,
        ))
        _forbid_dispatch(server, monkeypatch)
        _call(server, "session.control", session_id=sid, action="heartbeat.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        })
        state = load_heartbeat(key)
        assert state.fire_count == 8
        assert state.created_at == 100.0
        assert state.last_fired_at == 500.0
        assert state.status == "active"

    def test_heartbeat_update_does_not_falsify_last_fired_at(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat, load_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(
            prompt="original", interval_seconds=60, status="active",
            created_at=100.0, last_fired_at=500.0, fire_count=3,
        ))
        _forbid_dispatch(server, monkeypatch)
        _call(server, "session.control", session_id=sid, action="heartbeat.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        })
        state = load_heartbeat(key)
        assert state.last_fired_at == 500.0

    def test_heartbeat_update_rejects_empty_prompt(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(prompt="original", interval_seconds=60, status="active"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.update", args={
            "prompt": "",
            "interval_seconds": 120,
        }))
        assert err["code"] == 4004

    def test_heartbeat_update_publishes_snapshot(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(prompt="original", interval_seconds=60, status="active"))
        _forbid_dispatch(server, monkeypatch)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, sid_, payload=None: emitted.append((event, sid_, payload)))
        _call(server, "session.control", session_id=sid, action="heartbeat.update", args={
            "prompt": "updated",
            "interval_seconds": 120,
        })
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1
        assert updates[0][2]["control"]["heartbeat"]["prompt"] == "updated"


class TestUpdateUnknownSession:
    def test_update_returns_4004_for_missing_target(self, server):
        for action in ("goal.update", "loop.update", "heartbeat.update"):
            err = _error(_call(server, "session.control", session_id="gone", action=action, args={
                "prompt": "x", "interval_seconds": 60,
            }))
            assert err["code"] in (4001, 4004)

"""Behavioral contract tests for automation composer creation actions on session.control."""

from __future__ import annotations

import importlib
import json
import threading
import time
import uuid
from types import SimpleNamespace
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
    sid = f"sid-create-{uuid.uuid4().hex}"
    key = f"create-{uuid.uuid4().hex}"
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


def _control(server, sid):
    return _call(server, "session.control.read", session_id=sid)["result"]["control"]


def _error(response):
    assert "error" in response
    return response["error"]


def _forbid_dispatch(server, monkeypatch):
    def forbidden(_rid, _params):
        raise AssertionError("creation action must not call command.dispatch")
    monkeypatch.setitem(server._methods, "command.dispatch", forbidden)


class TestGoalCreate:
    def test_goal_create_sets_active_goal_with_prompt_and_criteria(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "Fix the login bug",
            "criteria": ["Auth flow works", "Tests pass"],
            "max_turns": 15,
        })
        assert "error" not in response
        goal = response["result"]["control"]["goal"]
        assert goal["title"] == "Fix the login bug"
        assert goal["status"] == "active"
        assert goal["max_turns"] == 15
        assert goal["subgoals"] == ["Auth flow works", "Tests pass"]
        assert goal["turns_used"] == 0

    def test_goal_create_without_criteria_succeeds(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "Refactor the module",
        })
        assert "error" not in response
        goal = response["result"]["control"]["goal"]
        assert goal["title"] == "Refactor the module"
        assert goal["subgoals"] == []

    def test_goal_create_returns_send_dispatch_with_continuation_prompt(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "Write docs",
        })
        dispatch = response["result"]["dispatch"]
        assert dispatch["type"] == "send"
        assert dispatch["message"]
        assert "Write docs" in dispatch["message"]
        assert dispatch["notice"]

    def test_goal_create_rejects_empty_prompt(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for prompt in ("", "   ", None):
            args = {"prompt": prompt} if prompt is not None else {}
            err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args=args))
            assert err["code"] == 4004

    def test_goal_create_rejects_duplicate_when_goal_active(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="existing", status="active"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "new goal",
        }))
        assert err["code"] == 4004
        assert "already" in err["message"].lower() or "existing" in err["message"].lower()

    def test_goal_create_rejects_duplicate_when_goal_done(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="completed task", status="done"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "new goal",
        }))
        assert err["code"] == 4004

    def test_goal_create_rejects_duplicate_when_goal_paused(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="paused task", status="paused"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "new goal",
        }))
        assert err["code"] == 4004

    def test_goal_create_allows_after_cleared_goal(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalState, save_goal
        sid, key, _ = session
        save_goal(key, GoalState(goal="old", status="cleared"))
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "fresh goal",
        })
        assert "error" not in response
        assert response["result"]["control"]["goal"]["title"] == "fresh goal"

    def test_goal_create_rejects_invalid_max_turns(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for mt in (-1, 0, 1.5, "abc"):
            args = {"prompt": "do stuff", "max_turns": mt}
            err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args=args))
            assert err["code"] == 4004

    def test_goal_create_rejects_huge_max_turns(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "do stuff", "max_turns": 10_000_000,
        }))
        assert err["code"] == 4004

    def test_goal_create_rejects_criteria_not_a_list(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "do stuff", "criteria": "not a list",
        }))
        assert err["code"] == 4004

    def test_goal_create_rejects_blank_criteria_items(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "do stuff", "criteria": ["valid", "   ", "also valid"],
        }))
        assert err["code"] == 4004

    def test_goal_create_publishes_session_control_update(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, sid_, payload=None: emitted.append((event, sid_, payload)))
        _call(server, "session.control", session_id=sid, action="goal.create", args={"prompt": "Fix bug"})
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1
        assert updates[0][2]["control"]["goal"]["title"] == "Fix bug"

    def test_goal_create_uses_session_profile_scope(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        from hermes_cli.goals import load_goal
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "scoped goal",
        })
        assert "error" not in response
        assert load_goal(key) is not None
        assert load_goal(key).goal == "scoped goal"


class TestLoopCreate:
    def test_loop_create_sets_active_loop_with_interval(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "Check deployment status",
            "interval_seconds": 300,
        })
        assert "error" not in response
        loop = response["result"]["control"]["loop"]
        assert loop["prompt"] == "Check deployment status"
        assert loop["status"] == "active"
        assert loop["mode"] == "interval"
        assert loop["interval_seconds"] >= 30

    def test_loop_create_leaves_initial_tick_to_scheduler(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "Poll CI",
            "interval_seconds": 300,
        })
        dispatch = response["result"]["dispatch"]
        assert dispatch["type"] == "exec"
        assert response["result"]["control"]["loop"]["ticks_fired"] == 0

    def test_loop_create_with_run_limit(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "Poll CI",
            "interval_seconds": 60,
            "run_limit": 10,
        })
        loop = response["result"]["control"]["loop"]
        assert loop["times"] == 10

    def test_loop_create_with_stop_condition(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "Watch queue",
            "interval_seconds": 120,
            "stop_condition": "queue is empty",
        })
        loop = response["result"]["control"]["loop"]
        assert loop["until"] == "queue is empty"

    def test_loop_create_rejects_empty_prompt(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for prompt in ("", "   ", None):
            args = {"prompt": prompt, "interval_seconds": 60} if prompt is not None else {"interval_seconds": 60}
            err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args=args))
            assert err["code"] == 4004

    def test_loop_create_rejects_invalid_interval(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for iv in (-1, 0, 1.5, "abc", None):
            args = {"prompt": "do stuff"}
            if iv is not None:
                args["interval_seconds"] = iv
            err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args=args))
            assert err["code"] == 4004

    def test_loop_create_rejects_huge_interval(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "do stuff", "interval_seconds": 10_000_000,
        }))
        assert err["code"] == 4004

    def test_loop_create_rejects_duplicate_when_loop_active(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(prompt="existing", status="active", mode="interval", interval_seconds=60, current_delay=60))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "new loop", "interval_seconds": 120,
        }))
        assert err["code"] == 4004
        assert "already" in err["message"].lower() or "existing" in err["message"].lower()

    def test_loop_create_rejects_duplicate_when_loop_done(self, server, session, monkeypatch):
        from hermes_cli.loops import LoopState, save_loop
        sid, key, _ = session
        save_loop(key, LoopState(prompt="finished", status="done", mode="interval", interval_seconds=60, current_delay=60))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "new loop", "interval_seconds": 120,
        }))
        assert err["code"] == 4004

    def test_loop_create_rejects_invalid_run_limit(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for rl in (-1, 0, 1.5, "abc"):
            err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args={
                "prompt": "do stuff", "interval_seconds": 60, "run_limit": rl,
            }))
            assert err["code"] == 4004

    def test_loop_create_rejects_huge_run_limit(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "do stuff", "interval_seconds": 60, "run_limit": 10_000_000,
        }))
        assert err["code"] == 4004

    def test_loop_create_publishes_session_control_update(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, sid_, payload=None: emitted.append((event, sid_, payload)))
        _call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": "Check deploy", "interval_seconds": 300,
        })
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1
        assert updates[0][2]["control"]["loop"]["prompt"] == "Check deploy"


class TestHeartbeatCreate:
    def test_heartbeat_create_sets_active_heartbeat(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "Check system health",
            "interval_seconds": 300,
        })
        assert "error" not in response
        hb = response["result"]["control"]["heartbeat"]
        assert hb["prompt"] == "Check system health"
        assert hb["status"] == "active"
        assert hb["interval_seconds"] == 300

    def test_heartbeat_create_leaves_initial_tick_to_scheduler(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "Check health",
            "interval_seconds": 300,
        })
        dispatch = response["result"]["dispatch"]
        assert dispatch["type"] == "exec"
        assert response["result"]["control"]["heartbeat"]["fire_count"] == 0

    def test_heartbeat_create_rejects_empty_prompt(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for prompt in ("", "   ", None):
            args = {"prompt": prompt, "interval_seconds": 60} if prompt is not None else {"interval_seconds": 60}
            err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.create", args=args))
            assert err["code"] == 4004

    def test_heartbeat_create_rejects_interval_below_minimum(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "Check health", "interval_seconds": 30,
        }))
        assert err["code"] == 4004

    def test_heartbeat_create_rejects_invalid_interval(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        for iv in (-1, 0, 1.5, "abc", None):
            args = {"prompt": "Check health"}
            if iv is not None:
                args["interval_seconds"] = iv
            err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.create", args=args))
            assert err["code"] == 4004

    def test_heartbeat_create_rejects_huge_interval(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "Check health", "interval_seconds": 10_000_000,
        }))
        assert err["code"] == 4004

    def test_heartbeat_create_rejects_duplicate_when_heartbeat_active(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(prompt="existing", interval_seconds=60, status="active"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "new heartbeat", "interval_seconds": 120,
        }))
        assert err["code"] == 4004
        assert "already" in err["message"].lower() or "existing" in err["message"].lower()

    def test_heartbeat_create_rejects_duplicate_when_heartbeat_paused(self, server, session, monkeypatch):
        from hermes_cli.heartbeat import HeartbeatState, save_heartbeat
        sid, key, _ = session
        save_heartbeat(key, HeartbeatState(prompt="paused", interval_seconds=60, status="paused"))
        _forbid_dispatch(server, monkeypatch)
        err = _error(_call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "new heartbeat", "interval_seconds": 120,
        }))
        assert err["code"] == 4004

    def test_heartbeat_create_publishes_session_control_update(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, sid_, payload=None: emitted.append((event, sid_, payload)))
        _call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": "Check health", "interval_seconds": 300,
        })
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1
        assert updates[0][2]["control"]["heartbeat"]["prompt"] == "Check health"


class TestCreateNoStaleSession:
    def test_goal_create_uses_session_from_request_not_global(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={
            "prompt": "Session-scoped goal",
        })
        goal = response["result"]["control"]["goal"]
        assert goal["title"] == "Session-scoped goal"
        other_key = f"other-{uuid.uuid4().hex}"
        from hermes_cli.goals import load_goal
        assert load_goal(other_key) is None

    def test_unknown_session_returns_4001_for_create(self, server):
        for action in ("goal.create", "loop.create", "heartbeat.create"):
            err = _error(_call(server, "session.control", session_id="gone", action=action, args={
                "prompt": "x", "interval_seconds": 60,
            }))
            assert err["code"] == 4001


class TestCreateInputPreserved:
    def test_goal_create_preserves_literal_prompt_text(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        literal = "Fix the <script>alert('xss')</script> & \"quotes\" issue"
        response = _call(server, "session.control", session_id=sid, action="goal.create", args={"prompt": literal})
        assert response["result"]["control"]["goal"]["title"] == literal

    def test_loop_create_preserves_literal_prompt(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        literal = "Check /status endpoint every 5m"
        response = _call(server, "session.control", session_id=sid, action="loop.create", args={
            "prompt": literal, "interval_seconds": 300,
        })
        assert response["result"]["control"]["loop"]["prompt"] == literal

    def test_heartbeat_create_preserves_literal_prompt(self, server, session, monkeypatch):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        literal = "Monitor [service] health"
        response = _call(server, "session.control", session_id=sid, action="heartbeat.create", args={
            "prompt": literal, "interval_seconds": 300,
        })
        assert response["result"]["control"]["heartbeat"]["prompt"] == literal

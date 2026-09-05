"""Behavioral contract tests for Desktop session-control JSON-RPC methods."""

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
    """Give the persisted control managers an isolated database for every test."""
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
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
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
    sid = f"sid-control-{uuid.uuid4().hex}"
    key = f"control-{uuid.uuid4().hex}"
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


def _observe_dispatch(server, monkeypatch):
    calls = []
    original = server._methods["command.dispatch"]

    def observe(rid, params):
        calls.append((rid, dict(params)))
        return original(rid, params)

    monkeypatch.setitem(server._methods, "command.dispatch", observe)
    return calls


def _forbid_dispatch(server, monkeypatch):
    def forbidden(_rid, _params):
        raise AssertionError("manager-only action must not call command.dispatch")

    monkeypatch.setitem(server._methods, "command.dispatch", forbidden)


def _save_goal(key, **overrides):
    from hermes_cli.goals import GoalState, save_goal

    fields = {
        "goal": "Finish the desktop control card",
        "status": "active",
        "turns_used": 3,
        "max_turns": 12,
        "created_at": 100.0,
        "last_turn_at": 200.0,
    }
    fields.update(overrides)
    state = GoalState(**fields)
    save_goal(key, state)
    return state


def _save_loop(key, **overrides):
    from hermes_cli.loops import LoopState, save_loop

    fields = {
        "prompt": "Check the deployment",
        "status": "active",
        "mode": "interval",
        "interval_seconds": 300,
        "current_delay": 300,
        "created_at": 100.0,
        "next_due_at": 400.0,
    }
    fields.update(overrides)
    state = LoopState(**fields)
    save_loop(key, state)
    return state


def _save_heartbeat(key, **overrides):
    from hermes_cli.heartbeat import HeartbeatState, save_heartbeat

    fields = {
        "prompt": "Check the deployment",
        "interval_seconds": 600,
        "status": "active",
        "created_at": 100.0,
        "last_fired_at": 150.0,
        "fire_count": 2,
    }
    fields.update(overrides)
    state = HeartbeatState(**fields)
    save_heartbeat(key, state)
    return state


class TestStructuredRead:
    def test_methods_are_registered_and_empty_snapshot_is_stable(self, server, session):
        sid, _, _ = session
        assert {"session.control.read", "session.control"} <= set(server._methods)
        first = _control(server, sid)
        second = _control(server, sid)
        assert first == second
        assert first == {
            "goal": None,
            "loop": None,
            "heartbeat": None,
            "revision": "",
            "updated_at": 0,
        }

    def test_goal_contract_subgoals_and_gates_are_structured_and_sanitized(self, server, session):
        from hermes_cli.goals import GoalContract, GoalGate

        sid, key, _ = session
        contract = GoalContract(outcome="Card is correct", verification="Run focused tests")
        _save_goal(
            key,
            contract=contract,
            subgoals=["Keep command routing narrow", "Document event hydration seam"],
            gates=[GoalGate(
                command="scripts/run_tests.sh tests/tui_gateway/test_session_control.py",
                timeout_seconds=90,
                max_retries=2,
                attempts=1,
                last_exit_code=1,
                last_output_tail="private output must stay private",
                last_failed_fingerprint="secret-fingerprint",
            )],
        )

        goal = _control(server, sid)["goal"]
        assert goal["title"] == "Finish the desktop control card"
        assert goal["contract"] == contract.to_dict()
        assert goal["subgoals"] == ["Keep command routing narrow", "Document event hydration seam"]
        assert goal["gates"] == [{
            "command": "scripts/run_tests.sh tests/tui_gateway/test_session_control.py",
            "timeout_seconds": 90,
            "max_retries": 2,
            "attempts": 1,
            "last_exit_code": 1,
        }]
        serialized = json.dumps(goal)
        for forbidden in (
            "last_output_tail", "last_failed_fingerprint", "private output", "secret-fingerprint",
            "route", "session_id", "credential", "api_key",
        ):
            assert forbidden not in serialized

    def test_wait_barrier_is_absolute_and_unchanged_reads_do_not_count_down(self, server, session, monkeypatch):
        sid, key, _ = session
        _save_goal(key, waiting_until=2_000.0, waiting_reason="rate limit")
        clock = SimpleNamespace(now=1_000.0)
        monkeypatch.setattr(server, "time", SimpleNamespace(time=lambda: clock.now))

        first = _control(server, sid)
        clock.now = 1_001.0
        second = _control(server, sid)

        assert first == second
        assert first["goal"]["wait_barrier"] == {
            "type": "until", "until_at": 2_000.0, "reason": "rate limit",
        }
        assert "remaining_seconds" not in first["goal"]["wait_barrier"]

    @pytest.mark.parametrize(
        ("kind", "setup"),
        [
            ("goal", lambda key: _save_goal(key)),
            ("loop", lambda key: _save_loop(key)),
            ("heartbeat", lambda key: _save_heartbeat(key)),
        ],
    )
    def test_revision_is_stable_then_changes_for_each_visible_control(self, server, session, kind, setup):
        sid, key, _ = session
        setup(key)
        before = _control(server, sid)
        assert isinstance(before["revision"], str)
        assert len(before["revision"]) == 64
        assert before == _control(server, sid)

        if kind == "goal":
            _save_goal(key, goal="A visible replacement goal")
        elif kind == "loop":
            _save_loop(key, prompt="A visibly changed loop prompt")
        else:
            _save_heartbeat(key, prompt="A visibly changed heartbeat prompt")

        assert _control(server, sid)["revision"] != before["revision"]

    def test_session_and_pid_wait_barriers_keep_absolute_targets(self, server, session):
        sid, key, _ = session
        _save_goal(key, waiting_on_session="bg-session", waiting_reason="CI")
        assert _control(server, sid)["goal"]["wait_barrier"] == {
            "type": "session", "target": "bg-session", "reason": "CI",
        }
        _save_goal(key, waiting_on_pid=4242, waiting_reason="build")
        assert _control(server, sid)["goal"]["wait_barrier"] == {
            "type": "pid", "target": 4242, "reason": "build",
        }


@pytest.mark.parametrize(
    ("action", "name", "arg", "status"),
    [
        ("goal.pause", "goal", "pause", "active"),
        ("goal.resume", "goal", "resume", "paused"),
        ("goal.clear", "goal", "clear", "active"),
        ("loop.pause", "loop", "pause", "active"),
        ("loop.resume", "loop", "resume", "paused"),
        ("loop.stop", "loop", "stop", "active"),
    ],
)
def test_dispatched_actions_use_exact_mapping_and_caller_rpc_id(
    server, session, monkeypatch, action, name, arg, status,
):
    sid, key, _ = session
    if name == "goal":
        _save_goal(key, status=status)
    else:
        _save_loop(key, status=status)
    calls = _observe_dispatch(server, monkeypatch)

    response = _call(server, "session.control", rid=743, session_id=sid, action=action)

    assert "result" in response
    assert calls == [(743, {"session_id": sid, "name": name, "arg": arg})]


class TestDispatcherBackedMutations:
    def test_goal_pause_resume_clear_mutate_real_persisted_state(self, server, session):
        sid, key, _ = session
        _save_goal(key, status="active", turns_used=8)

        paused = _call(server, "session.control", session_id=sid, action="goal.pause")
        assert paused["result"]["control"]["goal"]["status"] == "paused"

        resumed = _call(server, "session.control", session_id=sid, action="goal.resume")
        assert resumed["result"]["dispatch"]["type"] == "send"
        assert resumed["result"]["dispatch"]["message"]
        assert resumed["result"]["dispatch"]["notice"]
        assert resumed["result"]["dispatch"]["display"] == "/goal resume"
        assert resumed["result"]["control"]["goal"]["status"] == "active"

        cleared = _call(server, "session.control", session_id=sid, action="goal.clear")
        assert cleared["result"]["control"]["goal"] is None

    def test_loop_pause_resume_stop_mutate_real_persisted_state(self, server, session):
        sid, key, _ = session
        _save_loop(key)

        paused = _call(server, "session.control", session_id=sid, action="loop.pause")
        assert paused["result"]["control"]["loop"]["status"] == "paused"

        resumed = _call(server, "session.control", session_id=sid, action="loop.resume")
        assert resumed["result"]["control"]["loop"]["status"] == "active"

        stopped = _call(server, "session.control", session_id=sid, action="loop.stop")
        assert stopped["result"]["control"]["loop"] is None


class TestManagerOnlyMutations:
    def test_subgoal_add_remove_clear_are_real_one_based_mutations_without_dispatch(self, server, session, monkeypatch):
        from hermes_cli.goals import load_goal

        sid, key, _ = session
        _save_goal(key, subgoals=["First criterion"])
        _forbid_dispatch(server, monkeypatch)

        added = _call(server, "session.control", session_id=sid, action="subgoal.add", args={"text": "Second criterion"})
        assert added["result"]["dispatch"]["output"] == "✓ Added subgoal 2: Second criterion"
        assert load_goal(key).subgoals == ["First criterion", "Second criterion"]

        removed = _call(server, "session.control", session_id=sid, action="subgoal.remove", args={"index": 1})
        assert removed["result"]["dispatch"]["output"] == "✓ Removed subgoal 1: First criterion"
        assert load_goal(key).subgoals == ["Second criterion"]

        cleared = _call(server, "session.control", session_id=sid, action="subgoal.clear")
        assert cleared["result"]["dispatch"]["output"] == "✓ Cleared 1 subgoal."
        assert load_goal(key).subgoals == []

    def test_subgoal_requires_a_goal_and_valid_arguments_without_dispatch(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        assert _error(_call(server, "session.control", session_id=sid, action="subgoal.add", args={"text": "criterion"}))["code"] == 4004

        _save_goal(key, subgoals=["Only criterion"])
        for args in (
            {"text": "   "},
            {"index": "1"},
            {"index": 1.5},
            {"index": True},
            {"index": 0},
            {"index": 2},
        ):
            action = "subgoal.add" if "text" in args else "subgoal.remove"
            assert _error(_call(server, "session.control", session_id=sid, action=action, args=args))["code"] == 4004

    def test_goal_unwait_clears_the_real_barrier_without_dispatch(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalManager

        sid, key, _ = session
        _save_goal(key)
        GoalManager(key).wait_for_seconds(60, reason="backoff")
        _forbid_dispatch(server, monkeypatch)

        response = _call(server, "session.control", session_id=sid, action="goal.unwait")
        assert response["result"]["dispatch"]["output"] == "▶ Wait barrier cleared — goal loop resumes."
        assert GoalManager(key).state.waiting_until == 0.0

    def test_heartbeat_pause_resume_clear_and_no_heartbeat_messages_do_not_dispatch(self, server, session, monkeypatch):
        sid, key, _ = session
        _forbid_dispatch(server, monkeypatch)
        assert _call(server, "session.control", session_id=sid, action="heartbeat.pause")["result"]["dispatch"]["output"] == "No heartbeat set."
        assert _call(server, "session.control", session_id=sid, action="heartbeat.resume")["result"]["dispatch"]["output"] == "No heartbeat to resume."
        assert _call(server, "session.control", session_id=sid, action="heartbeat.clear")["result"]["dispatch"]["output"] == "No heartbeat set."

        _save_heartbeat(key, status="active", last_fired_at=1.0)
        paused = _call(server, "session.control", session_id=sid, action="heartbeat.pause")
        assert paused["result"]["dispatch"]["output"] == "⏸ Heartbeat paused: Check the deployment"
        assert paused["result"]["control"]["heartbeat"]["status"] == "paused"

        resumed = _call(server, "session.control", session_id=sid, action="heartbeat.resume")
        assert resumed["result"]["dispatch"]["output"] == "▶ Heartbeat resumed (every 10m): Check the deployment"
        assert resumed["result"]["control"]["heartbeat"]["last_fired_at"] > 1.0

        cleared = _call(server, "session.control", session_id=sid, action="heartbeat.clear")
        assert cleared["result"]["dispatch"]["output"] == "✓ Heartbeat cleared."
        assert cleared["result"]["control"]["heartbeat"] is None


class TestErrorsAndEvents:
    @pytest.mark.parametrize(
        ("action", "args"),
        [
            ("", {}),
            ("not.allowed", {}),
            ("goal.gate.add", {"command": "echo should-not-run"}),
            ("subgoal.add", []),
        ],
    )
    def test_invalid_actions_and_malformed_args_return_4004_without_dispatch(self, server, session, monkeypatch, action, args):
        sid, _, _ = session
        _forbid_dispatch(server, monkeypatch)
        assert _error(_call(server, "session.control", session_id=sid, action=action, args=args))["code"] == 4004

    def test_unknown_session_returns_4001(self, server):
        assert _error(_call(server, "session.control", session_id="gone", action="goal.pause"))["code"] == 4001
        assert _error(_call(server, "session.control.read", session_id="gone"))["code"] == 4001

    def test_dispatch_error_emits_no_update(self, server, session, monkeypatch):
        sid, key, _ = session
        _save_goal(key)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda *event: emitted.append(event))
        monkeypatch.setitem(server._methods, "command.dispatch", lambda rid, params: server._err(rid, 4018, "dispatch failed"))

        response = _call(server, "session.control", session_id=sid, action="goal.pause")
        assert _error(response)["code"] == 4018
        assert emitted == []

    def test_adapter_error_becomes_4004_and_emits_no_update(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalManager

        sid, key, _ = session
        _save_goal(key)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda *event: emitted.append(event))
        monkeypatch.setattr(GoalManager, "add_subgoal", lambda self, text: (_ for _ in ()).throw(RuntimeError("blocked")))

        response = _call(server, "session.control", session_id=sid, action="subgoal.add", args={"text": "criterion"})
        assert _error(response)["code"] == 4004
        assert emitted == []

    def test_success_response_snapshot_is_exactly_the_one_update_event(self, server, session, monkeypatch):
        sid, key, _ = session
        _save_goal(key)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: emitted.append((event, event_sid, payload)))

        response = _call(server, "session.control", session_id=sid, action="goal.pause")
        assert emitted == [("session.control.update", sid, {"control": response["result"]["control"]})]

    def test_dispatch_envelope_preserves_all_user_visible_fields(self, server, session, monkeypatch):
        sid, key, _ = session
        _save_goal(key, status="paused")
        expected = {
            "type": "send",
            "output": "already-rendered output",
            "notice": "Goal resumed",
            "message": "Continue toward the goal.",
            "display": "/goal resume",
        }
        monkeypatch.setitem(
            server._methods,
            "command.dispatch",
            lambda rid, params: {"jsonrpc": "2.0", "id": rid, "result": expected},
        )

        response = _call(server, "session.control", session_id=sid, action="goal.resume")
        assert response["result"]["dispatch"] == expected

    def test_emit_failure_must_not_turn_mutation_into_rpc_error(self, server, session, monkeypatch):
        """Event delivery is best-effort: a failing _emit must not swallow an
        already-applied mutation.  The persisted goal must be paused, the RPC
        response must carry the correct snapshot and dispatch envelope, and the
        emission error must be debug-logged rather than propagated."""
        from hermes_cli.goals import GoalManager

        sid, key, _ = session
        _save_goal(key, status="active")
        monkeypatch.setattr(
            server, "_emit", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("emit broken")),
        )

        response = _call(server, "session.control", session_id=sid, action="goal.pause")

        assert "result" in response, f"_emit failure leaked as RPC error: {response}"
        assert response["result"]["control"]["goal"]["status"] == "paused"
        assert response["result"]["dispatch"]["type"] == "exec"
        assert response["result"]["dispatch"]["output"]
        assert GoalManager(key).state.status == "paused"

    def test_snapshot_failure_after_mutation_does_not_emit_an_all_null_fallback(self, server, session, monkeypatch):
        from hermes_cli.goals import GoalManager

        sid, key, _ = session
        _save_goal(key)
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda *event: emitted.append(event))
        monkeypatch.setattr(server, "_snapshot_control", lambda _key: (_ for _ in ()).throw(RuntimeError("storage failed")))

        response = _call(server, "session.control", session_id=sid, action="goal.pause")
        assert _error(response)["code"] == 5031
        assert GoalManager(key).state.status == "paused"
        assert emitted == []

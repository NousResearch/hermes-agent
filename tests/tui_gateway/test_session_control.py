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


class TestUpdatePublication:
    """The Desktop card repaints from ``session.control.update``; every path that mutates control state must
    publish it exactly once, and the read after a turn must reflect the post-turn hooks."""

    def _capture(self, server, monkeypatch):
        emitted = []
        monkeypatch.setattr(server, "_emit", lambda event, event_sid, payload=None: emitted.append((event, event_sid, payload)))
        return emitted

    def test_control_action_publishes_exactly_one_update_matching_the_response(self, server, session, monkeypatch):
        sid, key, _ = session
        _save_goal(key)
        emitted = self._capture(server, monkeypatch)
        response = _call(server, "session.control", session_id=sid, action="goal.pause")
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert updates == [("session.control.update", sid, {"control": response["result"]["control"]})]

    @pytest.mark.parametrize("name,arg", [("goal", "pause"), ("loop", "pause")])
    def test_typed_slash_via_command_dispatch_publishes_the_new_state(self, server, session, monkeypatch, name, arg):
        """/goal and /loop never reach the slash worker (``_PENDING_INPUT_COMMANDS``) — the built-in path must
        publish too, or the structured card stays stale until the next turn."""
        sid, key, _ = session
        if name == "goal":
            _save_goal(key)
        else:
            from hermes_cli.loops import LoopManager

            LoopManager(key).set("poll CI", interval_seconds=300)
        emitted = self._capture(server, monkeypatch)
        response = _call(server, "command.dispatch", session_id=sid, name=name, arg=arg)
        assert "result" in response
        updates = [e for e in emitted if e[0] == "session.control.update"]
        assert len(updates) == 1 and updates[0][1] == sid
        assert updates[0][2]["control"][name]["status"] == "paused"

    def test_unknown_dispatch_publishes_nothing(self, server, session, monkeypatch):
        sid, _, _ = session
        emitted = self._capture(server, monkeypatch)
        assert "error" in _call(server, "command.dispatch", session_id=sid, name="definitely-not-a-command", arg="")
        assert [e for e in emitted if e[0] == "session.control.update"] == []

    def test_real_turn_publishes_after_the_goal_judge_ran(self, server, session, monkeypatch, tmp_path):
        """``message.complete`` precedes the post-turn goal judge, so a client refresh keyed on that event reads
        the pre-judge turn count. A real ``_run_prompt_submit`` turn (inline thread, stub agent) must publish the
        snapshot AFTER the judge with the incremented count."""
        sid, key, entry = session
        _save_goal(key, turns_used=3)
        emitted = self._capture(server, monkeypatch)

        class _InlineThread:
            def __init__(self, target=None, daemon=None, args=(), kwargs=None):
                self._t, self._a, self._k = target, args, kwargs or {}

            def start(self):
                self._t(*self._a, **self._k)

            def is_alive(self):
                return False

            def join(self, timeout=None):
                return None

        for name, value in {
            "_wire_callbacks": lambda sid_: None, "_sync_agent_model_with_config": lambda sid_, s: None,
            "_session_cwd": lambda s: str(tmp_path), "_register_session_cwd": lambda s: None,
            "_tts_stream_begin": lambda: None, "_sync_session_key_after_compress": lambda *a, **k: None,
            "_get_usage": lambda agent: {}, "_hermes_home": tmp_path,
        }.items():
            monkeypatch.setattr(server, name, value)
        monkeypatch.setattr(server.threading, "Thread", _InlineThread)
        # Deterministic judge: the model-graded verdict is not under test, the ordering is.
        from hermes_cli.goals import GoalManager

        def judge(self, raw, **kwargs):
            from hermes_cli.goals import save_goal

            self.state.turns_used += 1
            save_goal(self.session_id, self.state)
            return {"should_continue": False, "message": ""}

        monkeypatch.setattr(GoalManager, "evaluate_after_turn", judge)
        entry.update({"image_counter": 0, "slash_worker": None, "show_reasoning": False, "tool_progress_mode": "all",
                      "inflight_turn": None, "running": True,
                      "agent": SimpleNamespace(session_id=key, clear_interrupt=lambda: None,
                                               run_conversation=lambda message, **kw: {"final_response": "done"})})

        assert server._run_prompt_submit("rid", sid, entry, "work on it")

        order = [e[0] for e in emitted]
        assert "message.complete" in order and "session.control.update" in order
        assert order.index("message.complete") < order.index("session.control.update")
        assert emitted[order.index("session.control.update")][2]["control"]["goal"]["turns_used"] == 4

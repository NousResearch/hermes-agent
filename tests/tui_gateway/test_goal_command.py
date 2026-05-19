"""Tests for /goal handling in tui_gateway.

The TUI routes ``/goal`` through ``command.dispatch`` (not ``slash.exec``)
because the CLI's ``_handle_goal_command`` queues the kickoff message onto
``_pending_input``, which the slash-worker subprocess has no reader for.
Instead we handle ``/goal`` directly in the server and return a
``{"type": "send", "notice": ..., "message": ...}`` payload the TUI client
uses to render a system line and fire the kickoff prompt.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module DB cache so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


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
        mod._methods.clear()
        importlib.reload(mod)


@pytest.fixture()
def session(server):
    sid = "sid-test"
    session_key = "tui-goal-session-1"
    s = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    return sid, session_key, s


def _call(server, method, **params):
    handler = server._methods[method]
    return handler(1, params)


# ── command.dispatch /goal ────────────────────────────────────────────


def test_goal_bare_shows_status_when_none_set(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_whitespace_only_shows_status(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="   ", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_status_alias_shows_status(server, session):
    sid, _, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="status", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "No active goal" in r["result"]["output"]


def test_goal_set_returns_send_with_notice(server, session):
    sid, session_key, _ = session
    r = _call(server, "command.dispatch", name="goal", arg="build a rocket", session_id=sid)
    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == "build a rocket"
    assert "notice" in result
    assert "Goal set" in result["notice"]
    assert "20-turn budget" in result["notice"]

    # Persisted in SessionDB
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    assert mgr.state is not None
    assert mgr.state.goal == "build a rocket"
    assert mgr.state.status == "active"


def test_goal_set_with_skill_command_sends_expanded_skill_payload(server, session):
    sid, session_key, _ = session
    fake_skills = {"/hermes-agent-dev": {"name": "hermes-agent-dev", "description": "Dev workflow"}}
    fake_msg = "Loaded skill content here"

    with patch("agent.skill_commands.scan_skill_commands", return_value=fake_skills), \
         patch("agent.skill_commands.build_skill_invocation_message", return_value=fake_msg) as build:
        r = _call(
            server,
            "command.dispatch",
            name="goal",
            arg="/hermes-agent-dev fix goal handling",
            session_id=sid,
        )

    result = r["result"]
    assert result["type"] == "send"
    assert result["message"] == fake_msg
    assert "Goal set" in result["notice"]
    build.assert_called_once_with(
        "/hermes-agent-dev",
        "fix goal handling",
        task_id=session_key,
    )

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_key)
    assert mgr.state is not None
    assert mgr.state.goal == "/hermes-agent-dev fix goal handling"
    assert mgr.state.status == "active"


def test_goal_pause_after_set(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="goal", arg="write a story", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "paused" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    assert GoalManager(session_key).state.status == "paused"


def test_goal_resume_reactivates(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="goal", arg="write a story", session_id=sid)
    _call(server, "command.dispatch", name="goal", arg="pause", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="resume", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "resumed" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    assert GoalManager(session_key).state.status == "active"


def test_goal_clear_removes_active_goal(server, session):
    sid, session_key, _ = session
    _call(server, "command.dispatch", name="goal", arg="write a story", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="clear", session_id=sid)
    assert r["result"]["type"] == "exec"
    assert "cleared" in r["result"]["output"].lower()

    from hermes_cli.goals import GoalManager

    # After clear the row is marked status=cleared (kept for audit);
    # ``has_goal()`` / ``is_active()`` return False so the goal loop
    # stays off and ``status`` reports "No active goal".
    mgr = GoalManager(session_key)
    assert not mgr.has_goal()
    assert not mgr.is_active()
    assert "No active goal" in mgr.status_line()


def test_goal_stop_and_done_are_clear_aliases(server, session):
    sid, _, _ = session
    _call(server, "command.dispatch", name="goal", arg="first goal", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="stop", session_id=sid)
    assert "cleared" in r["result"]["output"].lower()

    _call(server, "command.dispatch", name="goal", arg="second goal", session_id=sid)
    r = _call(server, "command.dispatch", name="goal", arg="done", session_id=sid)
    assert "cleared" in r["result"]["output"].lower()


def test_goal_requires_session(server):
    r = _call(server, "command.dispatch", name="goal", arg="nope", session_id="unknown")
    assert "error" in r
    assert r["error"]["code"] == 4001


# ── command.dispatch /harness ─────────────────────────────────────────


def test_harness_status_dispatch_uses_control_plane(server, session, monkeypatch):
    sid, _, _ = session
    import agent.harness as harness_module

    class _ControlPlane:
        core_name = "harness-core"

        def core_status(self):
            return {
                "status": "defined",
                "case_count": 7,
                "last_run_at": None,
                "last_result": None,
            }

    class _Harness:
        @property
        def control_plane(self):
            return _ControlPlane()

    monkeypatch.setattr(harness_module, "HermesHarness", _Harness)

    r = _call(server, "command.dispatch", name="harness", arg="status", session_id=sid)

    assert r["result"]["type"] == "exec"
    assert "harness-core: defined" in r["result"]["output"]
    assert "Cases: 7" in r["result"]["output"]


def test_slash_exec_rejects_harness_routes_to_command_dispatch(server, session):
    sid, _, _ = session
    r = _call(server, "slash.exec", command="harness status", session_id=sid)
    assert "error" in r
    assert r["error"]["code"] == 4018
    assert "command.dispatch" in r["error"]["message"]


# ── slash.exec /goal routing ──────────────────────────────────────────


def test_slash_exec_rejects_goal_routes_to_command_dispatch(server, session):
    """slash.exec must reject /goal with 4018 so the TUI client falls through
    to command.dispatch. Without this, the HermesCLI slash-worker subprocess
    would set the goal but silently drop the kickoff — the queue is in-proc."""
    sid, _, _ = session
    r = _call(server, "slash.exec", command="goal status", session_id=sid)
    assert "error" in r
    assert r["error"]["code"] == 4018
    assert "command.dispatch" in r["error"]["message"]


def test_pending_input_commands_includes_goal(server):
    """Guard: _PENDING_INPUT_COMMANDS must list 'goal' — removing it would
    silently re-break the TUI."""
    assert "goal" in server._PENDING_INPUT_COMMANDS


def test_goal_continuation_emits_judging_status_and_chains_turn(server, session, monkeypatch):
    """After a turn completes, the TUI must show that /goal is still busy
    while the judge runs, then dispatch the continuation when the judge says
    to continue."""
    sid, session_key, s = session

    from hermes_cli.goals import GoalManager

    GoalManager(session_key, default_max_turns=3).set("ship the fix")

    class FakeAgent:
        model = "fake-model"

        def __init__(self):
            self.prompts: list[str] = []

        def run_conversation(self, message, conversation_history=None, stream_callback=None):
            self.prompts.append(str(message))
            text = f"response {len(self.prompts)}"
            return {
                "final_response": text,
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": text},
                ],
            }

    agent = FakeAgent()
    s["agent"] = agent
    s["running"] = True

    events: list[dict] = []
    monkeypatch.setattr(server, "write_json", lambda obj: events.append(obj) or True)
    monkeypatch.setattr(server, "_get_db", lambda: None)

    with patch(
        "hermes_cli.goals.judge_goal",
        side_effect=[
            ("continue", "more work remains", False),
            ("done", "done now", False),
        ],
    ):
        server._run_prompt_submit("rid-goal", sid, s, "ship the fix")

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if len(agent.prompts) >= 2 and not s.get("running"):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("goal continuation did not finish")

    status_events = [
        e["params"]["payload"]
        for e in events
        if e.get("method") == "event"
        and (e.get("params") or {}).get("type") == "status.update"
    ]
    assert {"kind": "goal_judging", "text": "checking goal…"} in status_events
    assert any(
        p.get("kind") == "goal" and p.get("text", "").startswith("↻ Continuing toward goal")
        for p in status_events
    )
    assert any(
        p.get("kind") == "goal" and p.get("text", "").startswith("✓ Goal achieved")
        for p in status_events
    )
    assert "Continuing toward your standing goal" in agent.prompts[1]

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
        # Reset module-level session state without re-importing. importlib.reload
        # would re-register the module's atexit hooks (ThreadPoolExecutor
        # shutdown, _shutdown_sessions); the duplicates race the stderr
        # buffer at interpreter shutdown and surface as Fatal Python error:
        # _enter_buffered_busy. Clearing the per-session dicts gives the
        # next test a clean slate; _methods is NOT cleared because it's
        # populated at module import time and re-registration only happens
        # via reload (which we don't do).
        mod._sessions.clear()
        mod._pending.clear()
        mod._answers.clear()


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


def test_goal_state_is_bound_to_lineage_root(hermes_home):
    from hermes_cli.goals import GoalManager, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("lineage-root-session", "tui")
    db.create_session("lineage-child-session", "tui", parent_session_id="lineage-root-session")
    db.create_session("lineage-grandchild-session", "tui", parent_session_id="lineage-child-session")

    GoalManager("lineage-grandchild-session").set("keep this goal on the thread")

    assert db.get_meta(_meta_key("lineage-grandchild-session")) is None
    assert db.get_meta(_meta_key("lineage-child-session")) is None
    assert db.get_meta(_meta_key("lineage-root-session")) is not None
    child_state = GoalManager("lineage-child-session").state
    root_state = GoalManager("lineage-root-session").state
    assert child_state is not None
    assert root_state is not None
    assert child_state.goal == "keep this goal on the thread"
    assert root_state.goal == "keep this goal on the thread"


def test_legacy_tip_goal_migrates_to_lineage_root(hermes_home):
    from hermes_cli.goals import GoalManager, GoalState, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("legacy-root-session", "tui")
    db.create_session("legacy-child-session", "tui", parent_session_id="legacy-root-session")
    db.set_meta(_meta_key("legacy-child-session"), GoalState(goal="legacy child goal").to_json())

    mgr = GoalManager("legacy-child-session")

    assert mgr.state is not None
    assert mgr.state.goal == "legacy child goal"
    assert db.get_meta(_meta_key("legacy-root-session")) is not None
    root_state = GoalManager("legacy-root-session").state
    assert root_state is not None
    assert root_state.goal == "legacy child goal"


def test_newer_legacy_tip_goal_beats_stale_root_goal(hermes_home):
    from hermes_cli.goals import GoalManager, GoalState, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("stale-root-session", "tui")
    db.create_session("newer-child-session", "tui", parent_session_id="stale-root-session")
    db.set_meta(
        _meta_key("stale-root-session"),
        GoalState(goal="old cleared root", status="cleared", created_at=1.0, last_turn_at=2.0).to_json(),
    )
    db.set_meta(
        _meta_key("newer-child-session"),
        GoalState(goal="new active child", status="active", created_at=3.0).to_json(),
    )

    mgr = GoalManager("newer-child-session")

    assert mgr.state is not None
    assert mgr.state.goal == "new active child"
    root_state = GoalManager("stale-root-session").state
    assert root_state is not None
    assert root_state.goal == "new active child"


def test_active_legacy_tip_goal_beats_stale_root_terminal_goal(hermes_home):
    from hermes_cli.goals import GoalManager, GoalState, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("fresh-root-session", "tui")
    db.create_session("stale-child-session", "tui", parent_session_id="fresh-root-session")
    db.set_meta(_meta_key("fresh-root-session"), GoalState(goal="fresh cleared root", status="cleared").to_json())
    db.set_meta(
        _meta_key("stale-child-session"),
        GoalState(goal="stale active child", status="active", created_at=1.0).to_json(),
    )

    mgr = GoalManager("stale-child-session")

    assert mgr.state is not None
    assert mgr.state.goal == "stale active child"
    assert mgr.state.status == "active"


def test_active_tip_goal_still_beats_newer_root_terminal_goal(hermes_home):
    from hermes_cli.goals import GoalManager, GoalState, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("fresh-root-session-2", "tui")
    db.create_session("stale-child-session-2", "tui", parent_session_id="fresh-root-session-2")
    db.set_meta(
        _meta_key("fresh-root-session-2"),
        GoalState(goal="fresh cleared root", status="cleared", created_at=3.0, last_turn_at=4.0).to_json(),
    )
    db.set_meta(
        _meta_key("stale-child-session-2"),
        GoalState(goal="stale active child", status="active", created_at=1.0).to_json(),
    )

    mgr = GoalManager("stale-child-session-2")

    assert mgr.state is not None
    assert mgr.state.goal == "stale active child"
    assert mgr.state.status == "active"


def test_legacy_parent_goal_migrates_to_lineage_root(hermes_home):
    from hermes_cli.goals import GoalManager, GoalState, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("parent-root-session", "tui")
    db.create_session("legacy-parent-session", "tui", parent_session_id="parent-root-session")
    db.create_session("parent-grandchild-session", "tui", parent_session_id="legacy-parent-session")
    db.set_meta(
        _meta_key("legacy-parent-session"),
        GoalState(goal="legacy parent goal", status="active", created_at=1.0).to_json(),
    )

    mgr = GoalManager("parent-grandchild-session")

    assert mgr.state is not None
    assert mgr.state.goal == "legacy parent goal"
    root_state = GoalManager("parent-root-session").state
    assert root_state is not None
    assert root_state.goal == "legacy parent goal"


def test_delegate_session_does_not_inherit_parent_goal(hermes_home):
    from hermes_cli.goals import GoalManager, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("delegate-parent-session", "tui")
    db.create_session(
        "delegate-child-session",
        "subagent",
        parent_session_id="delegate-parent-session",
        model_config={"_delegate_from": "delegate-parent-session"},
    )

    GoalManager("delegate-parent-session").set("parent-only goal")

    assert GoalManager("delegate-child-session").state is None
    GoalManager("delegate-child-session").set("child-only goal")
    assert db.get_meta(_meta_key("delegate-parent-session")) is not None
    assert db.get_meta(_meta_key("delegate-child-session")) is not None
    assert GoalManager("delegate-parent-session").state.goal == "parent-only goal"
    assert GoalManager("delegate-child-session").state.goal == "child-only goal"


def test_deleted_lineage_root_migrates_goal_to_orphaned_child(hermes_home):
    from hermes_cli.goals import GoalManager, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("delete-root-session", "tui")
    db.create_session("delete-child-session", "tui", parent_session_id="delete-root-session")
    GoalManager("delete-root-session").set("survive root delete")

    assert db.delete_session("delete-root-session") is True

    child = db.get_session("delete-child-session")
    assert child is not None
    assert child.get("parent_session_id") is None
    assert db.get_meta(_meta_key("delete-child-session")) is not None
    child_state = GoalManager("delete-child-session").state
    assert child_state is not None
    assert child_state.goal == "survive root delete"


def test_bulk_deleted_lineage_root_migrates_goal_to_orphaned_child(hermes_home):
    from hermes_cli.goals import GoalManager, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("bulk-delete-root-session", "tui")
    db.create_session("bulk-delete-child-session", "tui", parent_session_id="bulk-delete-root-session")
    GoalManager("bulk-delete-root-session").set("survive bulk root delete")

    assert db.delete_sessions(["bulk-delete-root-session"]) == 1

    child = db.get_session("bulk-delete-child-session")
    assert child is not None
    assert child.get("parent_session_id") is None
    assert db.get_meta(_meta_key("bulk-delete-child-session")) is not None
    child_state = GoalManager("bulk-delete-child-session").state
    assert child_state is not None
    assert child_state.goal == "survive bulk root delete"


def test_empty_deleted_lineage_root_migrates_goal_to_orphaned_child(hermes_home):
    from hermes_cli.goals import GoalManager, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("empty-delete-root-session", "tui")
    db.create_session("empty-delete-child-session", "tui", parent_session_id="empty-delete-root-session")
    GoalManager("empty-delete-root-session").set("survive empty root delete")
    db.end_session("empty-delete-root-session", "tui_shutdown")

    assert db.delete_empty_sessions() == 1

    child = db.get_session("empty-delete-child-session")
    assert child is not None
    assert child.get("parent_session_id") is None
    assert db.get_meta(_meta_key("empty-delete-child-session")) is not None
    child_state = GoalManager("empty-delete-child-session").state
    assert child_state is not None
    assert child_state.goal == "survive empty root delete"


def test_pruned_lineage_root_migrates_goal_to_orphaned_child(hermes_home):
    from hermes_cli.goals import GoalManager, _meta_key
    from hermes_state import SessionDB

    db = SessionDB()
    db.create_session("prune-root-session", "tui")
    db.create_session("prune-child-session", "tui", parent_session_id="prune-root-session")
    GoalManager("prune-root-session").set("survive prune root delete")
    db.end_session("prune-root-session", "tui_shutdown")

    assert db.prune_sessions(older_than_days=0) == 1

    child = db.get_session("prune-child-session")
    assert child is not None
    assert child.get("parent_session_id") is None
    assert db.get_meta(_meta_key("prune-child-session")) is not None
    child_state = GoalManager("prune-child-session").state
    assert child_state is not None
    assert child_state.goal == "survive prune root delete"


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

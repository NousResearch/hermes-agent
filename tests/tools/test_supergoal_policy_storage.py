"""Clarification must not depend on the availability of SQLite."""
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import goals
from hermes_cli.goal_command import dispatch_goal_command
from hermes_constants import get_hermes_home
from tools.clarify_tool import clarify_tool
from tools.registry import registry
from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS, InlineToolContext


@pytest.fixture
def storage(monkeypatch):
    path = get_hermes_home() / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")

    def read(key):
        with sqlite3.connect(path, timeout=0.01) as conn:
            row = conn.execute("SELECT value FROM state_meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else None

    def write(key, value):
        with sqlite3.connect(path, timeout=0.01) as conn:
            conn.execute("INSERT OR REPLACE INTO state_meta VALUES (?, ?)", (key, value))

    db = SimpleNamespace(get_meta=read, set_meta=write)
    monkeypatch.setattr(goals, "_get_session_db", lambda: db)
    return path, db


def invoke(route, sid, calls):
    callback = lambda *a, **k: calls.append(a) or "answer"
    args = {"question": "Choose?"}
    if route == "inline":
        agent = SimpleNamespace(session_id=sid, clarify_callback=callback)
        result = INLINE_TOOL_EXECUTORS["clarify"](agent, args, InlineToolContext(effective_task_id="other"))
    else:
        result = registry.dispatch("clarify", args, session_id=sid, callback=callback)
    return json.loads(result) if isinstance(result, str) else result


@pytest.mark.parametrize("failure", ["corrupt", "locked", "missing"])
@pytest.mark.parametrize("mode", [None, "goal", "supergoal"])
def test_real_unreadable_db_is_not_a_global_clarification_denial(storage, mode, failure):
    path, db = storage
    mgr = goals.GoalManager("session")
    if mode:
        mgr.set("finish", mode=mode)
    lock = None
    if failure == "corrupt":
        path.write_bytes(b"not sqlite")
    elif failure == "missing":
        path.unlink()
    else:
        lock = sqlite3.connect(path)
        lock.execute("BEGIN EXCLUSIVE")
    try:
        if failure != "missing":
            with pytest.raises(sqlite3.DatabaseError):
                db.get_meta("goal:session")
        for route in ("inline", "registry"):
            calls = []
            result = invoke(route, "session", calls)
            assert len(calls) == (0 if mode == "supergoal" else 1)
            assert ("error" in result) is (mode == "supergoal")
        script = '''
import json
from tools.clarify_tool import clarify_tool
calls = []
r = json.loads(clarify_tool("Choose?", session_id="session", callback=lambda *a, **k: calls.append(a) or "answer"))
print(json.dumps({"result": r, "calls": len(calls)}))
'''
        probe = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               env={**os.environ, "HERMES_HOME": str(path.parent)},
                               text=True, capture_output=True, timeout=30)
        assert probe.returncode == 0, probe.stderr
        output = json.loads(probe.stdout)
        assert output["calls"] == (0 if mode == "supergoal" else 1)
    finally:
        if lock:
            lock.close()


@pytest.mark.parametrize("action", ["pause", "clear", "replace", "done"])
def test_lifecycle_releases_policy_only_after_verified_save(storage, monkeypatch, action):
    from hermes_cli.supergoal_policy import clarification_restricted
    path, db = storage
    mgr = goals.GoalManager("session")
    mgr.set("finish", mode="supergoal")
    assert clarification_restricted("session")
    operation = {"pause": lambda: mgr.pause(), "clear": lambda: mgr.clear(),
                 "replace": lambda: mgr.set("ordinary"), "done": lambda: mgr.mark_done("verified")}[action]
    with monkeypatch.context() as patcher:
        patcher.setattr(db, "set_meta", lambda *a: None)
        with pytest.raises(RuntimeError):
            operation()
    assert clarification_restricted("session")
    mgr = goals.GoalManager("session")
    {"pause": lambda: mgr.pause(), "clear": lambda: mgr.clear(),
     "replace": lambda: mgr.set("ordinary"), "done": lambda: mgr.mark_done("verified")}[action]()
    assert not clarification_restricted("session")
    if action == "pause":
        mgr.resume()
        assert clarification_restricted("session")


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("failure", ["dropped", "filesystem"])
def test_marker_write_failure_prevents_kickoff_and_resume(storage, monkeypatch, resume, failure):
    from hermes_cli import supergoal_policy
    mgr = goals.GoalManager("session")
    if resume:
        mgr.set("finish", mode="supergoal")
        mgr.pause()
    if failure == "dropped":
        monkeypatch.setattr(supergoal_policy, "atomic_write_text", lambda *a, **k: None)
    else:
        # A real filesystem error, not a mocked DB accessor.
        marker = supergoal_policy._marker_path(mgr.session_id)
        marker.parent.mkdir(exist_ok=True)
        marker.mkdir()
    result = dispatch_goal_command(mgr, "resume" if resume else "finish", mode="supergoal", authorize_gate=lambda: None)
    assert result.error and not result.prompt
    assert not mgr.is_active()


def test_pause_release_cannot_unprotect_a_concurrent_activation(storage, monkeypatch):
    from hermes_cli import supergoal_policy
    mgr = goals.GoalManager("session")
    mgr.set("finish", mode="supergoal")
    rival = goals.GoalManager("session")
    release = supergoal_policy.release_clarification
    outcomes = []

    def concurrent_release(sid):
        outcomes.append(dispatch_goal_command(rival, "replacement", mode="supergoal", authorize_gate=lambda: None))
        release(sid)

    monkeypatch.setattr(supergoal_policy, "release_clarification", concurrent_release)
    mgr.pause()
    # A contending activation must report a technical error, not return an
    # unprotected kickoff while the prior pause is still releasing its marker.
    assert outcomes[0].error and not outcomes[0].prompt
    assert goals.load_goal("session").status == "paused"
    assert not supergoal_policy.clarification_restricted("session")
    assert not dispatch_goal_command(rival, "replacement", mode="supergoal", authorize_gate=lambda: None).error
    assert supergoal_policy.clarification_restricted("session")


def test_policy_writer_lock_excludes_a_fresh_process(storage):
    from hermes_cli.supergoal_policy import policy_write_lock
    script = '''
from hermes_cli.goals import save_goal, GoalState
try:
    save_goal("session", GoalState(goal="finish", mode="supergoal"))
except RuntimeError as exc:
    print(str(exc))
else:
    raise AssertionError("concurrent activation was allowed")
'''
    with policy_write_lock("session"):
        probe = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2],
                               env={**os.environ, "HERMES_HOME": str(get_hermes_home())},
                               text=True, capture_output=True, timeout=30)
        assert probe.returncode == 0, probe.stderr
        assert "policy update unavailable" in probe.stdout
    goals.GoalManager("session").set("finish", mode="supergoal")


def test_failed_migration_keeps_both_session_ids_restricted(storage):
    from hermes_cli.supergoal_policy import clarification_restricted
    path, db = storage
    mgr = goals.GoalManager("parent")
    mgr.set("finish", mode="supergoal")
    path.write_bytes(b"not sqlite")
    assert not goals.migrate_goal_to_session("parent", "child")
    assert clarification_restricted("parent")
    assert clarification_restricted("child")


def test_migration_does_not_report_success_when_parent_archive_read_fails(storage, monkeypatch):
    from hermes_cli.supergoal_policy import clarification_restricted
    path, db = storage
    goals.GoalManager("parent").set("finish", mode="supergoal")
    read = db.get_meta
    parent_reads = 0

    def failing_read(key):
        nonlocal parent_reads
        if key == "goal:parent":
            parent_reads += 1
            if parent_reads > 1:
                raise sqlite3.OperationalError("database is locked")
        return read(key)

    monkeypatch.setattr(db, "get_meta", failing_read)
    assert not goals.migrate_goal_to_session("parent", "child")
    assert clarification_restricted("parent")
    assert clarification_restricted("child")


def test_migration_does_not_restrict_an_existing_ordinary_child(storage):
    from hermes_cli.supergoal_policy import clarification_restricted
    goals.GoalManager("parent").set("finish", mode="supergoal")
    goals.GoalManager("child").set("ordinary")
    assert not goals.migrate_goal_to_session("parent", "child")
    assert not clarification_restricted("child")
    assert clarification_restricted("parent")


def test_stale_load_cannot_restrict_an_ordinary_replacement(storage, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from threading import Event
    from hermes_cli import supergoal_policy

    _, db = storage
    goals.save_goal("session", goals.GoalState(goal="old", mode="supergoal"))
    read = db.get_meta
    captured, resume = Event(), Event()

    def delayed_read(key):
        raw = read(key)
        if not captured.is_set():
            captured.set()
            assert resume.wait(5)
        return raw

    monkeypatch.setattr(db, "get_meta", delayed_read)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(copy_context().run, goals.load_goal, "session")
        try:
            assert captured.wait(5)
            goals.save_goal("session", goals.GoalState(goal="ordinary"))
        finally:
            resume.set()
        loaded = pending.result(timeout=5)
    assert loaded.mode == "goal"
    assert not supergoal_policy.clarification_restricted("session")
    calls = []
    assert invoke("inline", "session", calls)["user_response"] == "answer"


def test_ordinary_save_serializes_with_first_supergoal_activation(storage, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    from threading import Event, current_thread
    from hermes_cli import supergoal_policy

    _, db = storage
    entered, resume = Event(), Event()
    main_thread = current_thread()

    def delayed_db():
        if current_thread() is not main_thread:
            entered.set()
            assert resume.wait(5)
        return db

    monkeypatch.setattr(goals, "_get_session_db", delayed_db)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(copy_context().run, goals.save_goal, "session",
                              goals.GoalState(goal="ordinary"))
        try:
            assert entered.wait(5)
            try:
                goals.save_goal("session", goals.GoalState(goal="new", mode="supergoal"))
            except RuntimeError as exc:
                assert "policy update unavailable" in str(exc)
        finally:
            resume.set()
        pending.result(timeout=5)
    assert goals.GoalState.from_json(db.get_meta("goal:session")).mode == "goal"
    assert not supergoal_policy.clarification_restricted("session")


def test_legacy_active_row_is_bootstrapped_when_loaded(storage):
    from hermes_cli.supergoal_policy import clarification_restricted
    path, db = storage
    db.set_meta("goal:legacy", goals.GoalState(goal="old development row", mode="supergoal").to_json())
    assert goals.GoalManager("legacy").is_active()
    assert clarification_restricted("legacy")


@pytest.mark.linux_only
@pytest.mark.parametrize("target", ["database", "marker"])
def test_real_permission_denial_is_session_scoped(storage, target):
    from hermes_cli.supergoal_policy import _marker_path
    path, db = storage
    goals.GoalManager("protected").set("finish", mode="supergoal")
    denied = path if target == "database" else _marker_path("protected")
    previous = denied.stat().st_mode
    denied.chmod(0)
    try:
        with pytest.raises(PermissionError):
            denied.read_bytes()
        calls = []
        assert "error" in invoke("inline", "protected", calls)
        assert not calls
        assert invoke("inline", "ordinary", calls)["user_response"] == "answer"
    finally:
        denied.chmod(previous)


def test_unreadable_specific_marker_does_not_block_other_sessions(storage):
    from hermes_cli.supergoal_policy import _marker_path
    goals.GoalManager("broken").set("finish", mode="supergoal")
    _marker_path("broken").write_text("invalid")
    calls = []
    assert "error" in invoke("inline", "broken", calls)
    assert not calls
    assert invoke("inline", "ordinary", calls)["user_response"] == "answer"

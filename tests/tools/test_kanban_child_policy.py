"""HERMES-ORCH-001D: child-creation policy on the kanban_create tool.

Covers:
- restricted running worker (allow_child_creation=false) cannot create child
- graph (children / task count) unchanged on denial
- unrestricted orchestrator (no HERMES_KANBAN_TASK) can create
- allow_child_creation=true worker can create
- legacy worker without contract can create
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


FULL_SHA = "e9b8ae6be137abead6d19ed8a67c523f8c527096"


def _valid_contract(**overrides):
    base = {
        "version": 1,
        "scope": "ORCH-001D child policy only",
        "allowed_files": [
            "tools/kanban_tools.py",
            "tests/tools/test_kanban_child_policy.py",
        ],
        "forbidden_files": ["hermes_cli/main.py"],
        "base_commit": FULL_SHA,
        "required_evidence": ["pytest output", "commit SHA"],
        "required_commands": [
            "scripts/run_tests.sh tests/tools/test_kanban_child_policy.py -q"
        ],
        "allow_child_creation": False,
        "forbidden_git_actions": [
            "push",
            "merge",
            "amend",
            "reset",
            "clean",
            "restore",
            "stash",
        ],
        "notification_verified": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_ADMISSION_ENFORCE", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _task_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()["c"]


def _child_count(conn, parent_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM task_links WHERE parent_id = ?",
        (parent_id,),
    ).fetchone()["c"]


def test_restricted_worker_cannot_create_child(isolated_home, monkeypatch):
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="restricted builder",
            assignee="builder",
            contract=_valid_contract(allow_child_creation=False),
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="secret-chat-should-not-leak",
            thread_id="topic-1",
            user_id="u1",
            notifier_profile="default",
        )
        # Subscription alone may not admit if other fields fail; force claim
        # path is irrelevant — policy is contract-based on create.
        before_tasks = _task_count(conn)
        before_children = _child_count(conn, tid)
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    out = json.loads(
        kt._handle_create(
            {
                "title": "illegal child",
                "assignee": "peer",
                "parents": [tid],
            }
        )
    )
    assert out.get("error"), out
    assert "child creation denied" in out["error"].lower() or (
        kb.CHILD_CREATION_DENIED in out["error"]
    )
    assert "allow_child_creation" in out["error"]

    conn = kb.connect()
    try:
        assert _task_count(conn) == before_tasks
        assert _child_count(conn, tid) == before_children
        assert kb.child_ids(conn, tid) == []
    finally:
        conn.close()


def test_unrestricted_orchestrator_can_create(isolated_home, monkeypatch):
    from tools import kanban_tools as kt

    # Orchestrator: no HERMES_KANBAN_TASK → no contract gate applies.
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    out = json.loads(
        kt._handle_create({"title": "orch child", "assignee": "peer"})
    )
    assert "error" not in out, out
    assert out.get("task_id")


def test_allow_child_true_worker_can_create(isolated_home, monkeypatch):
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="orchestrator worker",
            assignee="orch",
            contract=_valid_contract(allow_child_creation=True),
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="123",
            thread_id="",
            user_id="u",
            notifier_profile="default",
        )
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    out = json.loads(
        kt._handle_create(
            {
                "title": "allowed child",
                "assignee": "peer",
                "parents": [tid],
            }
        )
    )
    assert "error" not in out, out
    child = out["task_id"]

    conn = kb.connect()
    try:
        assert child in kb.child_ids(conn, tid)
    finally:
        conn.close()


def test_legacy_worker_without_contract_can_create(isolated_home, monkeypatch):
    from tools import kanban_tools as kt

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="legacy worker", assignee="builder")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    out = json.loads(
        kt._handle_create({"title": "legacy child", "assignee": "peer"})
    )
    assert "error" not in out, out

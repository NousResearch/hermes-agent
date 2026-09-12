"""Regression tests for t_96e0e791 phantom profile guards."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def worker_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        task = kb.claim_task(conn, tid)
        assert task is not None
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def _profile(name: str):
    return type("P", (), {"name": name})()


def test_create_rejects_example_token_assignee(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: [
        _profile("os-reviewer"), _profile("fleet-engineer")
    ])
    for token in ("reviewer", "writer", "researcher-a", "Reviewer "):
        out = json.loads(kt._handle_create({
            "title": "child", "assignee": token, "parents": [worker_env],
        }))
        assert out.get("error"), f"{token!r} should be rejected: {out}"
        assert "not an existing profile" in out["error"]
        assert "fleet-engineer" in out["error"] and "os-reviewer" in out["error"]


def test_create_allows_example_token_when_profile_exists(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "reviewer")
    out = json.loads(kt._handle_create({
        "title": "child", "assignee": "reviewer", "parents": [worker_env],
    }))
    assert out["ok"] is True, out


def test_create_leaves_non_example_assignees_alone(monkeypatch, worker_env):
    """External seats and allowlisted built-ins have no profile dir."""
    from tools import kanban_tools as kt
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    for name in ("external-claude-dgx-lab20-3", "fable", "jarvis", "peer"):
        out = json.loads(kt._handle_create({
            "title": f"child-{name}", "assignee": name, "parents": [worker_env],
        }))
        assert out["ok"] is True, (name, out)


def _own_worker_run(monkeypatch, tid):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
        task = kb.get_task(conn, tid)
        assert task is not None
        run_id = task.current_run_id
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def test_request_review_rejects_unknown_reviewer(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    _own_worker_run(monkeypatch, worker_env)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    monkeypatch.setattr("hermes_cli.profiles.list_profiles", lambda: [_profile("os-reviewer")])
    out = json.loads(kt._handle_request_review({
        "summary": "implemented and tested", "reviewer": "reviewer",
    }))
    assert out.get("error"), out
    assert "'reviewer'" in out["error"]
    assert "not an existing profile" in out["error"]
    assert "os-reviewer" in out["error"]
    assert "warning" not in out
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        assert task is not None
        assert task.status != "review"
        assert task.assignee != "reviewer"


def test_request_review_rejects_placeholder_and_blank_reviewer(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    _own_worker_run(monkeypatch, worker_env)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    for bad in ("<existing-review-profile>", "os-reviewer>", "   "):
        out = json.loads(kt._handle_request_review({
            "summary": "implemented and tested", "reviewer": bad,
        }))
        assert out.get("error"), (bad, out)
        assert "placeholder" in out["error"] or "whitespace-only" in out["error"]


def test_create_rejects_placeholder_and_blank_assignee(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    for bad in ("<existing-profile-name>", "<fleet-engineer>", "fleet>engineer", "   "):
        out = json.loads(kt._handle_create({
            "title": "child-ph", "assignee": bad, "parents": [worker_env],
        }))
        assert out.get("error"), (bad, out)
        assert "placeholder" in out["error"] or "whitespace-only" in out["error"]


def test_request_review_keeps_known_reviewer(monkeypatch, worker_env):
    from tools import kanban_tools as kt
    _own_worker_run(monkeypatch, worker_env)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: name == "os-reviewer")
    out = json.loads(kt._handle_request_review({
        "summary": "implemented and tested", "reviewer": "os-reviewer",
    }))
    assert out["ok"] is True, out
    assert "warning" not in out
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect() as conn:
        task = kb.get_task(conn, worker_env)
        assert task is not None
        assert task.assignee == "os-reviewer"


def test_schema_descriptions_carry_no_copyable_example_tokens():
    from tools import kanban_tools as kt
    create = kt.KANBAN_CREATE_SCHEMA["parameters"]["properties"]["assignee"]
    review = kt.KANBAN_REQUEST_REVIEW_SCHEMA["parameters"]["properties"]["reviewer"]
    for desc in (create["description"], review["description"]):
        low = desc.lower()
        for token in ("'reviewer'", "'writer'", "'researcher-a'"):
            assert token not in low, (token, desc)
        assert not re.search(r"['\"`][^'\"`]*['\"`]", desc), desc
        assert "<" not in desc and ">" not in desc, desc
        assert "e.g." not in low, desc
        assert "~/.hermes/profiles" in desc
    assert "its own work" in review["description"]

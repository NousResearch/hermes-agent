"""Ghost-assignee rejection at the ``kanban_create`` tool ingress.

The devon routing failure (20.4h of ``skipped_nonspawnable`` rust) was an
agent orchestration fanning a card out onto a profile nobody had installed.
The reviewer check (#106163) already refuses non-profile reviewers at
request-review time; this extends the same contract to card creation:
a ghost assignee dies HERE — with the roster in the error — before any row
is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fresh_env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME + empty kanban DB; no worker task in scope
    (kanban_create is the orchestrator's fan-out surface)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    from pathlib import Path as _Path

    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def test_kanban_create_rejects_ghost_assignee(fresh_env):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "ghost fan-out", "assignee": "devon"}))
    assert "error" in out
    assert "devon" in out["error"]
    assert "default" in out["error"]  # installed-profiles roster makes it actionable

    # Nothing was written: the card never existed.
    with kbc.connect_closing() as conn:
        assert kb.list_tasks(conn, limit=50) == []


def test_kanban_create_accepts_live_profile(fresh_env, tmp_path):
    from tools import kanban_tools as kt

    (tmp_path / ".hermes" / "profiles" / "peer").mkdir(parents=True)
    (tmp_path / ".hermes" / "profiles" / "peer" / "config.yaml").write_text(
        "{}\n", encoding="utf-8"
    )  # identity marker
    out = json.loads(kt._handle_create({"title": "real fan-out", "assignee": "peer"}))
    assert out["ok"] is True
    assert out["task_id"]


def test_kanban_create_accepts_default(fresh_env):
    from tools import kanban_tools as kt

    out = json.loads(kt._handle_create({"title": "root lane", "assignee": "default"}))
    assert out["ok"] is True
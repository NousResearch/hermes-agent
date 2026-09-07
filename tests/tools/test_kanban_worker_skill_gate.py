"""Gate: board mutations from a dispatcher-spawned worker require the
worker-contract skill (using-superpowers) to be loaded first.

Acceptance (issue #105286):
  - worker without the skill loaded -> mutation rejected with a helpful reason
  - worker with the skill viewed via skill_view -> normal lifecycle
  - dispatcher force-load (card --skill) counts as loaded
  - kanban_show (read-only) never gated
  - non-worker contexts (orchestrator) unaffected
"""
from __future__ import annotations

import json

import pytest

from tests.tools.test_kanban_tools import worker_env  # noqa: F401  (fixture reuse)


@pytest.fixture
def clean_skill_state(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_PRELOADED_SKILLS", raising=False)
    from tools.skills_tool_dedup import _skill_view_tracker
    _skill_view_tracker.clear()


def _load_skill_via_view(task_id: str, tmp_path) -> None:
    """Simulate skill_view having served using-superpowers to this task
    (the dedup registry fingerprints the file on disk, so a real file is needed)."""
    import os
    from tools.skills_tool_dedup import _record_skill_view
    src = tmp_path / "using-superpowers.md"
    src.write_text("skill content")
    _record_skill_view(task_id, "using-superpowers", None,
                       {"name": "using-superpowers", "_source_path": str(src)})


def test_mutation_denied_without_skill(worker_env, clean_skill_state):
    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({})
    d = json.loads(out)
    assert "using-superpowers" in d.get("error", "")
    assert "skill_view" in d.get("error", "")


def test_mutation_allowed_after_skill_view(worker_env, clean_skill_state, tmp_path):
    _load_skill_via_view(worker_env, tmp_path)
    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({})
    d = json.loads(out)
    assert "using-superpowers" not in (d.get("error") or "")
    assert "heartbeat" in json.dumps(d).lower() or d.get("ok") or d.get("task_id") or "error" not in d


def test_preloaded_counts_as_loaded(worker_env, clean_skill_state, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_PRELOADED_SKILLS", "using-superpowers")
    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({})
    d = json.loads(out)
    assert "using-superpowers" not in (d.get("error") or "")


def test_show_not_gated(worker_env, clean_skill_state):
    from tools import kanban_tools as kt
    out = kt._handle_show({})
    d = json.loads(out)
    assert "task" in d  # read-only works without the skill


def test_orchestrator_unaffected(monkeypatch, tmp_path, clean_skill_state):
    monkeypatch.delenv("HEMMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="orchestrator-test", assignee="orch")
        kb.claim_task(conn, tid)
    finally:
        conn.close()

    from tools import kanban_tools as kt
    out = kt._handle_heartbeat({"task_id": tid})
    d = json.loads(out)
    assert "using-superpowers" not in (d.get("error") or "")

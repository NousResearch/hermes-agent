"""Model-facing completion metadata and durable registry handoff contracts."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def worker_board(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "metadata-contract")
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    kb.create_board("metadata-contract")
    with kbc.connect_closing(board="metadata-contract") as conn:
        tid = kb.create_task(conn, title="metadata handoff", assignee="test-worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed.status == "running"
        run_id = kb.latest_run(conn, tid).id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    return tid


def test_sanitized_handoff_metadata_explicitly_allows_caller_keys(worker_board):
    from tools import kanban_tools
    from tools.registry import invalidate_check_fn_cache, registry
    from tools.schema_sanitizer import sanitize_tool_schemas

    invalidate_check_fn_cache()
    schemas = sanitize_tool_schemas(registry.get_definitions(
        {"kanban_complete", "kanban_request_review"}, quiet=True))
    metadata = {
        schema["function"]["name"]:
        schema["function"]["parameters"]["properties"]["metadata"]
        for schema in schemas
    }
    assert set(metadata) == {"kanban_complete", "kanban_request_review"}
    for name, declaration in metadata.items():
        assert declaration["type"] == "object", name
        assert declaration.get("additionalProperties") is True, name


def test_registry_completion_persists_nested_caller_metadata(worker_board):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc
    from tools import kanban_tools
    from tools.registry import registry

    metadata = {
        "published_pr": "https://github.com/example/project/pull/42",
        "findings": [{"severity": "P2", "details": {"paths": ["tools/contract.py"]}}],
        "caller_context": {"labels": ["backend", "reviewed"], "passed": True, "count": 3},
    }
    out = json.loads(registry.dispatch("kanban_complete", {
        "summary": "Persist structured completion facts.", "metadata": metadata,
    }))
    assert out["ok"] is True, out
    assert out["task_id"] == worker_board
    with kbc.connect_closing(board="metadata-contract") as conn:
        assert kb.get_task(conn, worker_board).status == "done"
        run = kb.latest_run(conn, worker_board)
        assert run.outcome == "completed"
        assert run.summary == "Persist structured completion facts."
        assert run.metadata == {
            "published_pr": "https://github.com/example/project/pull/42",
            "findings": [{"severity": "P2", "details": {"paths": ["tools/contract.py"]}}],
            "caller_context": {"labels": ["backend", "reviewed"], "passed": True, "count": 3},
        }

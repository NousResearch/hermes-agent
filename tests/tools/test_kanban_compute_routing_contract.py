"""Kanban public-surface contracts required by compute-class routing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _isolated_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "router-orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def test_kanban_create_schema_exposes_closed_reasoning_effort_enum():
    from hermes_constants import VALID_REASONING_EFFORTS
    from tools.kanban_tools import KANBAN_CREATE_SCHEMA

    prop = KANBAN_CREATE_SCHEMA["parameters"]["properties"]["reasoning_effort"]
    assert prop["type"] == "string"
    assert prop["enum"] == ["none", *VALID_REASONING_EFFORTS]


def test_kanban_create_reasoning_effort_parity_to_db_event_show_and_spawn(
    monkeypatch, tmp_path
):
    kb = _isolated_home(monkeypatch, tmp_path)
    from tools import kanban_tools as kt

    workspace = tmp_path / "worker"
    workspace.mkdir()
    created = json.loads(
        kt._handle_create(
            {
                "title": "routed worker",
                "assignee": "bmk",
                "workspace_kind": "dir",
                "workspace_path": str(workspace),
                "model": "gpt-5.6-sol",
                "provider": "openai-codex",
                "reasoning_effort": "HIGH",
            }
        )
    )
    assert created.get("ok") is True, created

    with kb.connect() as conn:
        task = kb.get_task(conn, created["task_id"])
        events = kb.list_events(conn, created["task_id"])
    assert task.reasoning_effort == "high"
    created_event = next(event for event in events if event.kind == "created")
    assert created_event.payload["reasoning_effort"] == "high"
    assert created_event.payload["model_override"] == "gpt-5.6-sol"
    assert created_event.payload["provider_override"] == "openai-codex"

    shown = json.loads(kt._handle_show({"task_id": created["task_id"]}))
    assert shown["task"]["reasoning_effort"] == "high"
    assert shown["task"]["model_override"] == "gpt-5.6-sol"
    assert shown["task"]["provider_override"] == "openai-codex"

    captured = {}

    class FakeProc:
        pid = 4242

    def fake_popen(cmd, *args, **kwargs):
        captured["cmd"] = list(cmd)
        return FakeProc()

    monkeypatch.setattr(kb, "_resolve_hermes_argv", lambda: ["hermes"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    kb._default_spawn(task, str(workspace))

    argv = captured["cmd"]
    assert argv[argv.index("--reasoning") + 1] == "high"
    assert argv[argv.index("-m") + 1] == "gpt-5.6-sol"
    assert argv[argv.index("--provider") + 1] == "openai-codex"


def test_kanban_create_rejects_reasoning_typo_without_db_mutation(
    monkeypatch, tmp_path
):
    kb = _isolated_home(monkeypatch, tmp_path)
    from tools import kanban_tools as kt

    with kb.connect() as conn:
        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]

    result = json.loads(
        kt._handle_create(
            {
                "title": "typo must not inherit",
                "assignee": "bmk",
                "reasoning_effort": "hihg",
            }
        )
    )
    assert "error" in result
    assert "reasoning_effort" in result["error"]

    with kb.connect() as conn:
        after = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    assert after == before

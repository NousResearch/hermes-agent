from __future__ import annotations

import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.real_completion_evidence_gate


def _setup_worker(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-worker")
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="worker-test", assignee="test-worker")
        kb.claim_task(conn, tid)
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    return tid


def _evidence_file(tmp_path: Path, name: str = "tool-evidence.md") -> str:
    path = tmp_path / name
    path.write_text("tool evidence\n", encoding="utf-8")
    return str(path)


def test_kanban_complete_tool_rejects_missing_evidence(tmp_path, monkeypatch):
    tid = _setup_worker(tmp_path, monkeypatch)

    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({"summary": "done without proof"})
    payload = json.loads(out)
    err = payload.get("error", "")
    assert "kanban_complete blocked" in err
    assert "evidence path" in err
    assert "still in-flight" in err

    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
    finally:
        conn.close()


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            {"summary": "done with empty proof", "evidence_paths": [None, "", "   "]},
            id="top-level-paths-null-and-empty",
        ),
        pytest.param(
            {"summary": "done with null ref", "evidence_refs": None},
            id="top-level-refs-null",
        ),
        pytest.param(
            {"summary": "done with empty metadata", "metadata": {"evidence_refs": ""}},
            id="metadata-ref-empty-string",
        ),
        pytest.param(
            {"summary": "done with empty artifacts", "metadata": {"artifacts": []}},
            id="metadata-artifacts-empty-list",
        ),
    ],
)
def test_kanban_complete_tool_rejects_empty_and_null_evidence_fields(
    tmp_path, monkeypatch, args
):
    tid = _setup_worker(tmp_path, monkeypatch)

    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete(args)
    payload = json.loads(out)
    err = payload.get("error", "")
    assert "kanban_complete blocked" in err
    assert "evidence path" in err

    conn = kb.connect()
    try:
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "running"
        events = kb.list_events(conn, tid)
        assert any(e.kind == "completion_blocked_missing_evidence" for e in events)
    finally:
        conn.close()


def test_kanban_complete_tool_accepts_evidence_paths(tmp_path, monkeypatch):
    tid = _setup_worker(tmp_path, monkeypatch)
    evidence = _evidence_file(tmp_path)

    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "done with proof",
        "metadata": {"tests_run": 1},
        "evidence_paths": [evidence],
    })
    payload = json.loads(out)
    assert payload.get("ok") is True

    conn = kb.connect()
    try:
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.metadata is not None
        assert run.metadata["tests_run"] == 1
        assert run.metadata["evidence_refs"] == [evidence]
    finally:
        conn.close()


def test_kanban_complete_tool_accepts_artifacts_as_evidence(tmp_path, monkeypatch):
    tid = _setup_worker(tmp_path, monkeypatch)
    artifact = _evidence_file(tmp_path, "artifact.txt")

    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    out = kt._handle_complete({
        "summary": "done with artifact proof",
        "artifacts": [artifact],
    })
    payload = json.loads(out)
    assert payload.get("ok") is True

    conn = kb.connect()
    try:
        run = kb.latest_run(conn, tid)
        assert run is not None
        assert run.metadata is not None
        assert run.metadata["artifacts"] == [artifact]
        assert run.metadata["evidence_refs"] == [artifact]
        events = [e for e in kb.list_events(conn, tid) if e.kind == "completed"]
        assert events[0].payload is not None
        assert events[0].payload["artifacts"] == [artifact]
        assert events[0].payload["evidence_refs"] == [artifact]
    finally:
        conn.close()

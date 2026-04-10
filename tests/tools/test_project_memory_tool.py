"""Tests for tools/project_memory_tool.py."""

import json
from pathlib import Path

import pytest

from tools.project_memory_tool import ProjectMemoryStore, project_memory_tool


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.project_memory_tool.PROJECT_MEMORY_DIR", tmp_path)
    project_root = tmp_path / "repo"
    project_root.mkdir()
    (project_root / "app.py").write_text("print('hello')\n", encoding="utf-8")
    s = ProjectMemoryStore(project_root=str(project_root), char_limit=800, graph_edge_limit=10)
    s.load_from_disk()
    return s


class TestProjectMemoryStore:
    def test_add_note(self, store):
        result = store.add_note("Use uv for local commands", tags=["tooling"])
        assert result["success"] is True
        assert result["notes"][0]["content"] == "Use uv for local commands"

    def test_record_and_update_decision(self, store):
        created = store.record_decision(
            "Use project-scoped memory",
            rationale="Need repo-only recall",
            related_files=["app.py"],
        )
        assert created["success"] is True
        decision_id = created["decision_id"]

        updated = store.update_decision(decision_id, "Worked well in tests", status="resolved")
        assert updated["success"] is True
        assert updated["decision"]["status"] == "resolved"
        assert updated["decision"]["outcome"] == "Worked well in tests"

    def test_graph_relation(self, store):
        result = store.relate("run_agent.py", "depends_on", "project_memory_tool.py")
        assert result["success"] is True
        graph = store.view_graph(limit=10)
        assert graph["edge_count"] == 1
        assert graph["edges"][0]["predicate"] == "depends_on"

    def test_track_and_detect_drift(self, store):
        tracked = store.track_paths(["app.py"])
        assert tracked["success"] is True
        unchanged = store.drift_check()
        assert unchanged["changes"] == []

        app_path = Path(store.project_root) / "app.py"
        app_path.write_text("print('changed')\n", encoding="utf-8")
        changed = store.drift_check()
        assert changed["changes"][0]["status"] == "modified"
        assert changed["changes"][0]["path"] == "app.py"

    def test_snapshot_contains_notes_and_decisions(self, store):
        store.add_note("Use markdown briefs before coding")
        store.record_decision("Keep graph edges lightweight")
        store.load_from_disk()
        snapshot = store.format_for_system_prompt()
        assert snapshot is not None
        assert "PROJECT MEMORY" in snapshot
        assert "Use markdown briefs before coding" in snapshot
        assert "Keep graph edges lightweight" in snapshot


class TestProjectMemoryToolDispatcher:
    def test_read_defaults_to_store_overview(self, store):
        payload = json.loads(project_memory_tool(action="read", store=store))
        assert payload["success"] is True
        assert payload["namespace"] == store.namespace

    def test_add_note_via_dispatcher(self, store):
        payload = json.loads(project_memory_tool(action="add_note", content="Track architecture moves", store=store))
        assert payload["success"] is True
        assert payload["note_id"].startswith("note-")

    def test_record_decision_requires_summary(self, store):
        payload = json.loads(project_memory_tool(action="record_decision", store=store))
        assert payload["success"] is False
        assert "summary" in payload["error"]

    def test_unknown_action_returns_error(self, store):
        payload = json.loads(project_memory_tool(action="wat", store=store))
        assert payload["success"] is False

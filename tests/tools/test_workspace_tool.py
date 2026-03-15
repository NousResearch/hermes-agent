from __future__ import annotations

import json
from pathlib import Path


def _config(tmp_path: Path) -> dict:
    return {
        "workspace": {
            "enabled": True,
            "path": str(tmp_path / "workspace"),
            "auto_create": True,
            "persist_gateway_uploads": "ask",
        },
        "knowledgebase": {
            "enabled": True,
            "path": str(tmp_path / "knowledgebase"),
            "roots": [],
            "retrieval_mode": "off",
            "auto_index": True,
            "watch_for_changes": False,
            "max_injected_chunks": 6,
            "max_injected_tokens": 3200,
            "dense_top_k": 40,
            "sparse_top_k": 40,
            "fused_top_k": 30,
            "final_top_k": 8,
            "min_fused_score": 0.0,
            "injection_format": "sourced_note",
            "chunking": {
                "default_tokens": 512,
                "overlap_tokens": 80,
                "code_strategy": "structural",
                "markdown_strategy": "headings",
            },
            "embeddings": {"provider": "local", "model": "google/embeddinggemma-300m", "dimensions": 768},
            "reranker": {"enabled": False, "provider": "local", "model": "bge-reranker-v2-m3"},
            "indexing": {
                "respect_gitignore": True,
                "respect_hermesignore": True,
                "include_hidden": False,
                "max_file_mb": 10,
            },
        },
    }


class TestWorkspaceTool:
    def test_status_reports_workspace_roots(self, tmp_path, monkeypatch):
        from tools.workspace_tool import workspace_tool

        monkeypatch.setattr("tools.workspace_tool.load_config", lambda: _config(tmp_path))

        result = json.loads(workspace_tool(action="status"))

        assert result["success"] is True
        assert result["workspace_root"].endswith("workspace")
        assert result["knowledgebase_root"].endswith("knowledgebase")

    def test_index_search_and_retrieve_round_trip(self, tmp_path, monkeypatch):
        from tools.workspace_tool import workspace_tool

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "docs").mkdir(parents=True)
        (workspace / "docs" / "deploy.md").write_text("deployment checklist and rollback plan\n", encoding="utf-8")
        monkeypatch.setattr("tools.workspace_tool.load_config", lambda: cfg)

        indexed = json.loads(workspace_tool(action="index"))
        assert indexed["success"] is True
        assert indexed["file_count"] == 1
        assert indexed["chunk_count"] >= 1

        searched = json.loads(workspace_tool(action="search", query="deployment"))
        assert searched["success"] is True
        assert searched["count"] == 1
        assert searched["matches"][0]["relative_path"] == "docs/deploy.md"

        retrieved = json.loads(workspace_tool(action="retrieve", query="rollback plan"))
        assert retrieved["success"] is True
        assert retrieved["count"] >= 1
        assert retrieved["results"][0]["relative_path"] == "docs/deploy.md"

    def test_list_returns_relative_paths(self, tmp_path, monkeypatch):
        from tools.workspace_tool import workspace_tool

        cfg = _config(tmp_path)
        workspace = Path(cfg["workspace"]["path"])
        (workspace / "notes").mkdir(parents=True)
        (workspace / "notes" / "todo.txt").write_text("ship it\n", encoding="utf-8")
        monkeypatch.setattr("tools.workspace_tool.load_config", lambda: cfg)

        listed = json.loads(workspace_tool(action="list"))
        assert listed["success"] is True
        assert listed["entries"][0]["relative_path"] == "notes/todo.txt"

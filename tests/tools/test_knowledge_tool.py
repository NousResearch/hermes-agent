"""Tests for the Hermes knowledge ledger toolset.

The knowledge ledger is intentionally cold/on-demand: records live under
$HERMES_HOME/knowledge and are never injected into the system prompt by default.
"""

import json


def _json(result: str) -> dict:
    return json.loads(result)


class TestKnowledgeCapture:
    def test_non_inbox_records_require_source_attribution(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: tmp_path / "knowledge")

        result = _json(kt.knowledge_capture_tool(
            kind="decisions",
            title="Adopt MemKraft wholesale",
            content="Rejected: too much overlap with hot memory and session_search.",
        ))

        assert result["success"] is False
        assert "source" in result["error"].lower()
        assert not (tmp_path / "knowledge" / "decisions").exists()

    def test_inbox_allows_unsourced_candidate_records(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: tmp_path / "knowledge")

        result = _json(kt.knowledge_capture_tool(
            kind="inbox",
            title="Possible project convention",
            content="The project may prefer pytest -q for focused tests.",
        ))

        assert result["success"] is True
        assert result["kind"] == "inbox"
        assert result["status"] == "candidate"
        record_path = tmp_path / "knowledge" / result["path"]
        assert record_path.exists()
        text = record_path.read_text(encoding="utf-8")
        assert "status: candidate" in text
        assert "confidence: candidate" in text

    def test_unsourced_inbox_forces_candidate_status_and_confidence(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: tmp_path / "knowledge")

        result = _json(kt.knowledge_capture_tool(
            kind="inbox",
            title="Unverified extraction",
            content="Automatically extracted claim awaiting source review.",
            status="active",
            confidence="confirmed",
        ))

        assert result["success"] is True
        assert result["status"] == "candidate"
        assert result["confidence"] == "candidate"
        text = (tmp_path / "knowledge" / result["path"]).read_text(encoding="utf-8")
        assert "status: candidate" in text
        assert "confidence: candidate" in text

    def test_capture_rejects_symlinked_kind_directory(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        knowledge_dir = tmp_path / "knowledge"
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        knowledge_dir.mkdir()
        (knowledge_dir / "inbox").symlink_to(outside_dir, target_is_directory=True)
        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: knowledge_dir)

        result = _json(kt.knowledge_capture_tool(
            kind="inbox",
            title="Directory symlink guard",
            content="This must not be written outside knowledge.",
        ))

        assert result["success"] is False
        assert "symlink" in result["error"].lower() or "outside" in result["error"].lower()
        assert not any(outside_dir.iterdir())

    def test_capture_rejects_symlinked_knowledge_root(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        real_root = tmp_path / "real-knowledge"
        real_root.mkdir()
        symlink_root = tmp_path / "knowledge"
        symlink_root.symlink_to(real_root, target_is_directory=True)
        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: symlink_root)

        result = _json(kt.knowledge_capture_tool(
            kind="inbox",
            title="Root symlink guard",
            content="This must not be written through a symlinked knowledge root.",
        ))

        assert result["success"] is False
        assert "symlink" in result["error"].lower()
        assert not any(real_root.iterdir())

    def test_capture_writes_source_attributed_markdown_record(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: tmp_path / "knowledge")

        result = _json(kt.knowledge_capture_tool(
            kind="decisions",
            title="MemKraft adoption boundary",
            content="Use a cold source-attributed ledger, not a hot memory replacement.",
            sources=["conversation:discord:Agent/#1/Ping"],
            confidence="observed",
            tags=["memory", "roi"],
        ))

        assert result["success"] is True
        assert result["id"].startswith("decisions/")
        assert result["path"].startswith("decisions/")
        record_path = tmp_path / "knowledge" / result["path"]
        text = record_path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert "type: decisions" in text
        assert "confidence: observed" in text
        assert "sources:" in text
        assert "- conversation:discord:Agent/#1/Ping" in text
        assert "# MemKraft adoption boundary" in text
        assert "Use a cold source-attributed ledger" in text

    def test_capture_does_not_follow_broken_symlink_record_paths(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        knowledge_dir = tmp_path / "knowledge"
        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: knowledge_dir)
        monkeypatch.setattr(kt, "_now_iso", lambda: "2026-05-04T00:00:00Z")
        record_path = knowledge_dir / "inbox" / "20260504T000000Z-symlink-guard.md"
        outside_target = tmp_path / "outside.md"
        record_path.parent.mkdir(parents=True)
        record_path.symlink_to(outside_target)

        result = _json(kt.knowledge_capture_tool(
            kind="inbox",
            title="Symlink Guard",
            content="Write inside the knowledge ledger only.",
        ))

        assert result["success"] is True
        assert result["path"] == "inbox/20260504T000000Z-symlink-guard.md"
        assert record_path.exists()
        assert not record_path.is_symlink()
        assert not outside_target.exists()


class TestKnowledgeSearchAndGet:
    def test_search_returns_ranked_snippets_not_full_hot_memory_dump(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: tmp_path / "knowledge")
        kt.knowledge_capture_tool(
            kind="debug",
            title="Discord role rename failure",
            content="Discord returned 403 Missing Permissions because the managed bot role hierarchy blocks role rename.",
            sources=["terminal:discord-rest-probe"],
            confidence="confirmed",
            tags=["discord", "roles"],
        )
        kt.knowledge_capture_tool(
            kind="decisions",
            title="Knowledge ledger token policy",
            content="Knowledge records are cold storage and retrieved on demand only.",
            sources=["conversation:memkraft-triage"],
            confidence="observed",
        )

        result = _json(kt.knowledge_search_tool(query="role hierarchy", limit=5))

        assert result["success"] is True
        assert result["count"] >= 1
        assert result["results"][0]["title"] == "Discord role rename failure"
        assert "role hierarchy" in result["results"][0]["snippet"]
        assert "Knowledge records are cold storage" not in result["results"][0]["snippet"]

    def test_get_reads_record_by_id_and_blocks_traversal(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: tmp_path / "knowledge")
        captured = _json(kt.knowledge_capture_tool(
            kind="skill-evals",
            title="Skill patch needed",
            content="Patch skills immediately when a loaded skill is stale.",
            sources=["skill:test-driven-development"],
            confidence="observed",
        ))

        fetched = _json(kt.knowledge_get_tool(id=captured["id"]))
        assert fetched["success"] is True
        assert fetched["id"] == captured["id"]
        assert "Patch skills immediately" in fetched["content"]

        traversal = _json(kt.knowledge_get_tool(path="../memories/MEMORY.md"))
        assert traversal["success"] is False
        assert "outside" in traversal["error"].lower() or "traversal" in traversal["error"].lower()

    def test_search_skips_symlink_records_that_resolve_outside_knowledge_dir(self, tmp_path, monkeypatch):
        from tools import knowledge_tool as kt

        knowledge_dir = tmp_path / "knowledge"
        monkeypatch.setattr(kt, "get_knowledge_dir", lambda: knowledge_dir)
        outside = tmp_path / "outside.md"
        outside.write_text("# Outside\n\nsecret outside content", encoding="utf-8")
        inbox_dir = knowledge_dir / "inbox"
        inbox_dir.mkdir(parents=True)
        (inbox_dir / "evil.md").symlink_to(outside)

        result = _json(kt.knowledge_search_tool(query="secret outside", limit=5))

        assert result["success"] is True
        assert result["count"] == 0
        assert result["results"] == []


class TestKnowledgeToolsetWiring:
    def test_knowledge_toolset_is_configurable_but_not_default_core(self):
        import toolsets
        from hermes_cli import tools_config

        configurable = {name for name, _label, _desc in tools_config.CONFIGURABLE_TOOLSETS}
        assert "knowledge" in configurable
        assert "knowledge" in tools_config._DEFAULT_OFF_TOOLSETS
        assert "knowledge" in toolsets.TOOLSETS

        knowledge_tools = set(toolsets.resolve_toolset("knowledge"))
        assert knowledge_tools == {
            "knowledge_capture",
            "knowledge_search",
            "knowledge_get",
        }
        assert knowledge_tools.isdisjoint(set(toolsets._HERMES_CORE_TOOLS))

    def test_knowledge_tools_require_explicit_toolset_resolution(self):
        from model_tools import get_tool_definitions

        default_tools = {
            tool["function"]["name"]
            for tool in get_tool_definitions(quiet_mode=True)
        }
        assert "knowledge_capture" not in default_tools
        assert "knowledge_search" not in default_tools
        assert "knowledge_get" not in default_tools

        explicit_tools = {
            tool["function"]["name"]
            for tool in get_tool_definitions(enabled_toolsets=["knowledge"], quiet_mode=True)
        }
        assert {
            "knowledge_capture",
            "knowledge_search",
            "knowledge_get",
        }.issubset(explicit_tools)

    def test_capture_schema_states_cold_on_demand_contract(self):
        from tools.knowledge_tool import KNOWLEDGE_CAPTURE_SCHEMA

        description = KNOWLEDGE_CAPTURE_SCHEMA["description"]
        assert "not injected" in description.lower()
        assert "source" in description.lower()
        assert "inbox" in description.lower()

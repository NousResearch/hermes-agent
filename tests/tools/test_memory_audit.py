"""Tests for the memory audit action and nudge-prompt changes (issue #59823)."""

import json
import tempfile
from pathlib import Path

import pytest

from tools.memory_tool import MemoryStore, memory_tool, MEMORY_SCHEMA


class TestMemoryAudit:
    """Tests for MemoryStore.audit() and the memory_tool dispatch."""

    def test_audit_empty_store(self, tmp_path, monkeypatch):
        """Audit an empty store returns zero entries."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()

        result = store.audit("memory")
        assert result["success"] is True
        assert result["done"] is True
        assert result["target"] == "memory"
        assert result["entry_count"] == 0
        assert result["entries"] == []
        assert "0%" in result["usage"]

    def test_audit_with_entries(self, tmp_path, monkeypatch):
        """Audit returns all entries with indices, char counts, and previews."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()
        store.add("memory", "Entry A: user prefers Chinese")
        store.add("memory", "Entry B: project uses pytest with xdist")

        result = store.audit("memory")
        assert result["success"] is True
        assert result["entry_count"] == 2

        entries = result["entries"]
        assert entries[0]["idx"] == 0
        assert entries[0]["chars"] == len("Entry A: user prefers Chinese")
        assert "Entry A" in entries[0]["preview"]

        assert entries[1]["idx"] == 1
        assert len(entries[1]["preview"]) <= 123  # 120 chars + "..."

        # Usage should be non-zero and show percentage
        assert "%" in result["usage"]
        assert "/" in result["usage"]

    def test_audit_user_target(self, tmp_path, monkeypatch):
        """Audit works on the user target too."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()
        store.add("user", "User name: Alice")

        result = store.audit("user")
        assert result["target"] == "user"
        assert result["entry_count"] == 1
        assert "Alice" in result["entries"][0]["preview"]

    def test_audit_long_entry_truncated(self, tmp_path, monkeypatch):
        """Entries longer than 120 chars get truncated with '...'."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()
        long_text = "A" * 200
        store.add("memory", long_text)

        result = store.audit("memory")
        preview = result["entries"][0]["preview"]
        assert preview.endswith("...")
        assert len(preview) == 123  # 120 chars + "..."

    def test_audit_is_read_only(self, tmp_path, monkeypatch):
        """Calling audit does not change entries or file size."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()
        store.add("memory", "Entry 1")
        store.add("memory", "Entry 2")

        before_count = store._char_count("memory")
        before_entries = list(store.memory_entries)

        # Call audit twice
        store.audit("memory")
        result = store.audit("memory")

        after_count = store._char_count("memory")
        after_entries = list(store.memory_entries)

        assert after_count == before_count
        assert after_entries == before_entries
        assert result["entry_count"] == 2

    def test_audit_usage_percentage(self, tmp_path, monkeypatch):
        """Audit usage reflects capacity correctly."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore(memory_char_limit=500)
        store.load_from_disk()
        # Add entries to ~50%
        store.add("memory", "A" * 100)
        store.add("memory", "B" * 100)

        result = store.audit("memory")
        pct = int(result["usage"].split("%")[0])
        # 200 chars / 500 limit ≈ 40% (accounting for § delimiter: ~206/500 ≈ 41%)
        assert 35 <= pct <= 50


class TestAuditViaToolDispatch:
    """Tests for the memory_tool() dispatch with action='audit'."""

    def test_audit_via_tool(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()
        store.add("memory", "Tool test entry")

        raw = memory_tool(action="audit", target="memory", store=store)
        result = json.loads(raw)
        assert result["success"] is True
        assert result["entry_count"] == 1

    def test_audit_via_tool_user_target(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: Path(tmp_path)
        )
        store = MemoryStore()
        store.load_from_disk()
        store.add("user", "User fact")

        raw = memory_tool(action="audit", target="user", store=store)
        result = json.loads(raw)
        assert result["target"] == "user"

    def test_audit_no_store_returns_error(self):
        """Audit with no store signals unavailable."""
        raw = memory_tool(action="audit", target="memory", store=None)
        result = json.loads(raw)
        assert result["success"] is False


class TestAuditSchema:
    """Schema-level assertions for the audit action."""

    def test_audit_in_action_enum(self):
        schema = MEMORY_SCHEMA
        action_enum = schema["parameters"]["properties"]["action"]["enum"]
        assert "audit" in action_enum
        # All existing actions still present
        for act in ["add", "replace", "remove"]:
            assert act in action_enum

    def test_batch_sub_actions_unchanged(self):
        """Batch operations still only allow write actions, not audit."""
        schema = MEMORY_SCHEMA
        batch_enum = (
            schema["parameters"]["properties"]["operations"]
            ["items"]["properties"]["action"]["enum"]
        )
        # batch sub-operations are write-only — audit is top-level
        assert "audit" not in batch_enum
        assert batch_enum == ["add", "replace", "remove"]

    def test_target_unchanged(self):
        schema = MEMORY_SCHEMA
        assert schema["parameters"]["properties"]["target"]["enum"] == ["memory", "user"]

    def test_required_unchanged(self):
        """Only 'target' is required — audit doesn't need content or old_text."""
        schema = MEMORY_SCHEMA
        assert schema["parameters"]["required"] == ["target"]


class TestNudgePrompt:
    """Verify the new review prompt text includes the audit+classify workflow."""

    def test_prompt_includes_audit_step(self):
        from agent.background_review import _MEMORY_REVIEW_PROMPT
        assert "memory(action='audit')" in _MEMORY_REVIEW_PROMPT
        assert "Step 1" in _MEMORY_REVIEW_PROMPT
        assert "durable fact" in _MEMORY_REVIEW_PROMPT
        assert "task residue" in _MEMORY_REVIEW_PROMPT
        assert "Step 4 — Save" in _MEMORY_REVIEW_PROMPT

    def test_prompt_includes_cleanup_instruction(self):
        from agent.background_review import _MEMORY_REVIEW_PROMPT
        assert "Consolidate" in _MEMORY_REVIEW_PROMPT
        assert "remove task-residue" in _MEMORY_REVIEW_PROMPT

    def test_prompt_still_allows_noop(self):
        from agent.background_review import _MEMORY_REVIEW_PROMPT
        assert "Nothing to change." in _MEMORY_REVIEW_PROMPT

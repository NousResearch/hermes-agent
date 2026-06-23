"""Tests for memory tiering — core vs extended memory entries.

Core entries (prefixed with ``[core]``) are always injected into the system
prompt. Extended entries (no prefix) are on-demand via the ``search`` action.
When any entry has ``[core]``, only core entries go into the system prompt
snapshot. When no entries have ``[core]``, all entries go in (backward compat).
"""

import json
import pytest
from pathlib import Path

from tools.memory_tool import (
    MemoryStore,
    memory_tool,
    MEMORY_SCHEMA,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_memory_file(path: Path, entries: list[str]) -> None:
    """Write a MEMORY.md file with §-delimited entries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n§\n".join(entries), encoding="utf-8")


# ---------------------------------------------------------------------------
# [core] prefix parsing
# ---------------------------------------------------------------------------

class TestCorePrefix:
    """Tests for [core] prefix recognition and stripping."""

    def test_core_prefix_recognized(self, tmp_path, monkeypatch):
        """Entries starting with [core] are recognized as core."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Always needed: model config",
            "Extended: astrology data",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        assert store.memory_entries == [
            "[core] Always needed: model config",
            "Extended: astrology data",
        ]

    def test_core_prefix_stripped_in_snapshot(self, tmp_path, monkeypatch):
        """[core] prefix is stripped from the system prompt snapshot."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Always needed: model config",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "Always needed: model config" in snapshot
        assert "[core]" not in snapshot

    def test_no_core_entries_all_go_in(self, tmp_path, monkeypatch):
        """When no entries have [core], all entries go into snapshot (backward compat)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "Entry one",
            "Entry two",
            "Entry three",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "Entry one" in snapshot
        assert "Entry two" in snapshot
        assert "Entry three" in snapshot

    def test_mixed_core_and_extended_only_core_in_snapshot(self, tmp_path, monkeypatch):
        """When some entries have [core], only those go into the snapshot."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Core fact A",
            "Extended fact B",
            "[core] Core fact C",
            "Extended fact D",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "Core fact A" in snapshot
        assert "Core fact C" in snapshot
        assert "Extended fact B" not in snapshot
        assert "Extended fact D" not in snapshot

    def test_all_core_all_go_in(self, tmp_path, monkeypatch):
        """When all entries are [core], all go into snapshot."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Fact 1",
            "[core] Fact 2",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "Fact 1" in snapshot
        assert "Fact 2" in snapshot

    def test_user_profile_ignores_core_prefix(self, tmp_path, monkeypatch):
        """User profile entries always all go in — [core] has no effect on USER.md."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", ["[core] Core fact"])
        _write_memory_file(tmp_path / "USER.md", [
            "User fact A",
            "User fact B",
        ])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("user")
        assert snapshot is not None
        assert "User fact A" in snapshot
        assert "User fact B" in snapshot

    def test_core_prefix_case_sensitive(self, tmp_path, monkeypatch):
        """Only lowercase [core] is recognized. [CORE] or [Core] are not."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Real core",
            "[CORE] Not recognized as core",
            "[Core] Also not recognized",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "Real core" in snapshot
        # [CORE] and [Core] are treated as extended — they go in because
        # there IS at least one real [core], so only real [core] entries
        # are included. The uppercase variants are excluded.
        assert "Not recognized as core" not in snapshot
        assert "Also not recognized" not in snapshot


# ---------------------------------------------------------------------------
# search action
# ---------------------------------------------------------------------------

class TestMemorySearch:
    """Tests for the search action on the memory tool."""

    def test_search_finds_substring(self, tmp_path, monkeypatch):
        """search returns entries matching a case-insensitive substring."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Model: glm-5.1 primary",
            "Astrology: Cait Ba Zi Gui Yin Water",
            "Tool quirk: patch corrupts unicode",
        ])
        _write_memory_file(tmp_path / "USER.md", [
            "User prefers dark mode",
        ])
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="memory",
            query="model",
            store=store,
        ))
        assert result["success"] is True
        assert len(result["matches"]) == 1
        assert "glm-5.1" in result["matches"][0].lower()

    def test_search_case_insensitive(self, tmp_path, monkeypatch):
        """search is case-insensitive."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "GitHub: fork aj-nt/hermes-agent",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="memory",
            query="GITHUB",
            store=store,
        ))
        assert result["success"] is True
        assert len(result["matches"]) == 1

    def test_search_no_matches(self, tmp_path, monkeypatch):
        """search returns empty list when nothing matches."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "Entry one",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="memory",
            query="nonexistent",
            store=store,
        ))
        assert result["success"] is True
        assert result["matches"] == []

    def test_search_requires_query(self, tmp_path, monkeypatch):
        """search without query returns error."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="memory",
            store=store,
        ))
        assert result["success"] is False
        assert "query" in result["error"].lower()

    def test_search_user_target(self, tmp_path, monkeypatch):
        """search works on user target too."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [])
        _write_memory_file(tmp_path / "USER.md", [
            "User prefers dark mode",
            "User lives in Weston CT",
        ])
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="user",
            query="weston",
            store=store,
        ))
        assert result["success"] is True
        assert len(result["matches"]) == 1
        assert "Weston" in result["matches"][0]

    def test_search_includes_core_prefix_in_results(self, tmp_path, monkeypatch):
        """search returns entries with [core] prefix intact."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Model: glm-5.1 primary",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="memory",
            query="model",
            store=store,
        ))
        assert result["success"] is True
        assert "[core]" in result["matches"][0]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestMemoryTieringSchema:
    """Tests for schema updates supporting tiering."""

    def test_search_action_in_schema(self):
        """search is in the action enum."""
        params = MEMORY_SCHEMA["parameters"]["properties"]
        assert "search" in params["action"]["enum"]

    def test_query_param_in_schema(self):
        """query parameter exists for search action."""
        params = MEMORY_SCHEMA["parameters"]["properties"]
        assert "query" in params
        assert "search" in params["query"]["description"].lower()

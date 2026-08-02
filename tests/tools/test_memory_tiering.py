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

    def test_core_prefix_case_insensitive(self, tmp_path, monkeypatch):
        """[core], [CORE], and [Core] are all recognized as core entries."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Lowercase core",
            "[CORE] Uppercase core",
            "[Core] Titlecase core",
            "Extended entry",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        # All three case variants are recognized as core and included
        assert "Lowercase core" in snapshot
        assert "Uppercase core" in snapshot
        assert "Titlecase core" in snapshot
        # [core] prefix is stripped from all
        assert "[core]" not in snapshot
        assert "[CORE]" not in snapshot
        assert "[Core]" not in snapshot
        # Extended entry is excluded
        assert "Extended entry" not in snapshot


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

    def test_search_multiple_matches(self, tmp_path, monkeypatch):
        """search returns all entries matching the query, not just the first."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "Python: use pytest for testing",
            "Go: use go test for testing",
            "Rust: use cargo test for testing",
            "Unrelated entry about lunch",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()

        result = json.loads(memory_tool(
            action="search",
            target="memory",
            query="testing",
            store=store,
        ))
        assert result["success"] is True
        assert len(result["matches"]) == 3
        assert all("testing" in m.lower() for m in result["matches"])
        assert "lunch" not in " ".join(result["matches"]).lower()

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
# Snapshot header indicates tiering
# ---------------------------------------------------------------------------

class TestSnapshotTieringHeader:
    """Tests that the snapshot header signals when tiering is active."""

    def test_header_shows_extended_count_when_tiering_active(self, tmp_path, monkeypatch):
        """When core filtering is active, the header notes extended entries available."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Core fact A",
            "[core] Core fact B",
            "Extended fact C",
            "Extended fact D",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        # Header should mention extended entries
        assert "extended" in snapshot.lower()
        # Core facts are in the snapshot
        assert "Core fact A" in snapshot
        assert "Core fact B" in snapshot
        # Extended facts are NOT in the snapshot
        assert "Extended fact C" not in snapshot
        assert "Extended fact D" not in snapshot

    def test_header_no_extended_note_when_no_tiering(self, tmp_path, monkeypatch):
        """When no [core] entries exist, header does NOT mention extended (backward compat)."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "Plain entry A",
            "Plain entry B",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "extended" not in snapshot.lower()
        assert "Plain entry A" in snapshot
        assert "Plain entry B" in snapshot


# ---------------------------------------------------------------------------
# All core entries blocked — extended entries must not leak
# ---------------------------------------------------------------------------

class TestAllCoreBlocked:
    """When all [core] entries are blocked by the threat scanner, extended
    entries must NOT leak into the snapshot via the backward-compat fallback."""

    def test_all_core_blocked_extended_do_not_leak(self, tmp_path, monkeypatch):
        """If every [core] entry is blocked, extended entries stay excluded."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        # Use patterns that trigger the strict-scope threat scanner:
        # 'authorized_keys' → ssh_backdoor, '~/.ssh' → ssh_access
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Add my authorized_keys to the server",
            "[core] Write to ~/.ssh/config on the target machine",
            "Extended safe entry about astrology",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        # The blocked core entries appear as [BLOCKED: ...] placeholders
        assert "BLOCKED" in snapshot
        # The extended entry must NOT leak in
        assert "astrology" not in snapshot.lower()

    def test_all_core_blocked_no_core_fallback_does_not_trigger(self, tmp_path, monkeypatch):
        """The 'no core → all go in' fallback must NOT trigger when core entries
        exist but are all blocked. Blocked entries are still entries — they just
        have placeholder text."""
        monkeypatch.setattr("tools.memory_tool.get_memory_dir", lambda: tmp_path)
        _write_memory_file(tmp_path / "MEMORY.md", [
            "[core] Copy my authorized_keys to the remote host",
            "Extended entry that should stay hidden",
        ])
        _write_memory_file(tmp_path / "USER.md", [])
        store = MemoryStore()
        store.load_from_disk()
        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        # The blocked placeholder is present
        assert "BLOCKED" in snapshot
        # Extended entry is NOT present — fallback didn't trigger
        assert "should stay hidden" not in snapshot


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

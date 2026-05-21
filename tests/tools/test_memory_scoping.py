"""Tests for context_id-based memory scoping in MemoryStore.

Validates that:
- No context_id → global path (backward compatible)
- context_id → writes to contexts/{id}/, reads merge global + scoped
- Different contexts are isolated from each other
- Concurrent writes to different contexts don't block each other
"""

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.memory_tool import MemoryStore, ENTRY_DELIMITER


@pytest.fixture
def mem_dir(tmp_path):
    """Create a temporary memory directory and patch get_memory_dir to use it."""
    with patch("tools.memory_tool.get_memory_dir", return_value=tmp_path):
        yield tmp_path


class TestNoContextId:
    """Backward compatibility: context_id=None uses global paths."""

    def test_no_context_id_uses_global_path(self, mem_dir):
        store = MemoryStore()
        path = store._path_for("memory")
        assert path == mem_dir / "MEMORY.md"

        path_user = store._path_for("user")
        assert path_user == mem_dir / "USER.md"

    def test_no_context_id_backward_compatible(self, mem_dir):
        """Full round-trip: add, save, reload — same as upstream behavior."""
        store = MemoryStore()
        store.load_from_disk()

        result = store.add("memory", "global note")
        assert result["success"] is True

        # Verify file was written to global path
        assert (mem_dir / "MEMORY.md").exists()
        assert "global note" in (mem_dir / "MEMORY.md").read_text()

        # No context directory created
        assert not (mem_dir / "contexts").exists()

    def test_no_context_id_read_write_cycle(self, mem_dir):
        """Write with no context, read back — identical to upstream."""
        store = MemoryStore()
        store.load_from_disk()
        store.add("memory", "fact one")
        store.add("user", "user pref")

        # Fresh store, same dir
        store2 = MemoryStore()
        store2.load_from_disk()
        assert "fact one" in store2.memory_entries
        assert "user pref" in store2.user_entries


class TestContextId:
    """context_id routes writes to per-context subdirectory."""

    def test_context_id_creates_scoped_directory(self, mem_dir):
        store = MemoryStore(context_id="group-123")
        store.load_from_disk()

        result = store.add("memory", "scoped note")
        assert result["success"] is True

        scoped_path = mem_dir / "contexts" / "group-123" / "MEMORY.md"
        assert scoped_path.exists()
        assert "scoped note" in scoped_path.read_text()

    def test_context_id_path_for_returns_scoped(self, mem_dir):
        store = MemoryStore(context_id="chan-456")
        assert store._path_for("memory") == mem_dir / "contexts" / "chan-456" / "MEMORY.md"
        assert store._path_for("user") == mem_dir / "contexts" / "chan-456" / "USER.md"

    def test_context_id_merges_global_and_scoped(self, mem_dir):
        """Reads should include both global and context-scoped entries."""
        # Write a global entry directly
        (mem_dir / "MEMORY.md").write_text("global fact")

        # Create scoped store and add a scoped entry
        store = MemoryStore(context_id="grp-1")
        store.load_from_disk()

        # Should have loaded the global entry
        assert "global fact" in store.memory_entries

        # Add a scoped entry
        store.add("memory", "scoped fact")

        # Reload to verify merge
        store2 = MemoryStore(context_id="grp-1")
        store2.load_from_disk()
        assert "global fact" in store2.memory_entries
        assert "scoped fact" in store2.memory_entries

    def test_scoped_write_does_not_modify_global(self, mem_dir):
        """Writing with context_id should not touch global MEMORY.md."""
        (mem_dir / "MEMORY.md").write_text("original global")

        store = MemoryStore(context_id="grp-2")
        store.load_from_disk()
        store.add("memory", "scoped only")

        # Global file unchanged
        assert (mem_dir / "MEMORY.md").read_text() == "original global"

    def test_global_path_for_helper(self, mem_dir):
        store = MemoryStore(context_id="ctx-1")
        assert store._global_path_for("memory") == mem_dir / "MEMORY.md"
        assert store._global_path_for("user") == mem_dir / "USER.md"


class TestContextIsolation:
    """Different context_ids are isolated from each other."""

    def test_different_contexts_are_isolated(self, mem_dir):
        store_a = MemoryStore(context_id="group-A")
        store_a.load_from_disk()
        store_a.add("memory", "only for A")

        store_b = MemoryStore(context_id="group-B")
        store_b.load_from_disk()
        store_b.add("memory", "only for B")

        # Reload A — should not see B's entry
        store_a2 = MemoryStore(context_id="group-A")
        store_a2.load_from_disk()
        assert "only for A" in store_a2.memory_entries
        assert "only for B" not in store_a2.memory_entries

        # Reload B — should not see A's entry
        store_b2 = MemoryStore(context_id="group-B")
        store_b2.load_from_disk()
        assert "only for B" in store_b2.memory_entries
        assert "only for A" not in store_b2.memory_entries

    def test_both_see_global(self, mem_dir):
        """Both contexts should see global entries."""
        (mem_dir / "MEMORY.md").write_text("shared global")

        store_a = MemoryStore(context_id="group-A")
        store_a.load_from_disk()
        assert "shared global" in store_a.memory_entries

        store_b = MemoryStore(context_id="group-B")
        store_b.load_from_disk()
        assert "shared global" in store_b.memory_entries


class TestReplaceAndRemove:
    """Replace and remove operate on scoped files, not global."""

    def test_replace_in_scoped_context(self, mem_dir):
        store = MemoryStore(context_id="ctx-r")
        store.load_from_disk()
        store.add("memory", "old entry")
        store.replace("memory", "old entry", "new entry")

        store2 = MemoryStore(context_id="ctx-r")
        store2.load_from_disk()
        assert "new entry" in store2.memory_entries
        assert "old entry" not in store2.memory_entries

    def test_remove_from_scoped_context(self, mem_dir):
        store = MemoryStore(context_id="ctx-rm")
        store.load_from_disk()
        store.add("memory", "to remove")
        store.remove("memory", "to remove")

        store2 = MemoryStore(context_id="ctx-rm")
        store2.load_from_disk()
        assert "to remove" not in store2.memory_entries


class TestSystemPromptSnapshot:
    """System prompt snapshot merges global + scoped at load time."""

    def test_snapshot_includes_both(self, mem_dir):
        (mem_dir / "MEMORY.md").write_text("global info")

        # Write scoped entry
        scoped_dir = mem_dir / "contexts" / "snap-ctx"
        scoped_dir.mkdir(parents=True)
        (scoped_dir / "MEMORY.md").write_text("scoped info")

        store = MemoryStore(context_id="snap-ctx")
        store.load_from_disk()

        snapshot = store.format_for_system_prompt("memory")
        assert snapshot is not None
        assert "global info" in snapshot
        assert "scoped info" in snapshot


class TestConcurrentContextWrites:
    """Different contexts use different lock files, so they don't block each other."""

    def test_concurrent_context_writes_dont_block(self, mem_dir):
        results = {}
        errors = {}

        def write_to_context(ctx_id, entry):
            try:
                store = MemoryStore(context_id=ctx_id)
                store.load_from_disk()
                result = store.add("memory", entry)
                results[ctx_id] = result
            except Exception as e:
                errors[ctx_id] = e

        t1 = threading.Thread(target=write_to_context, args=("ctx-1", "entry from 1"))
        t2 = threading.Thread(target=write_to_context, args=("ctx-2", "entry from 2"))

        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert not errors, f"Errors: {errors}"
        assert results["ctx-1"]["success"] is True
        assert results["ctx-2"]["success"] is True

        # Verify isolation
        assert (mem_dir / "contexts" / "ctx-1" / "MEMORY.md").exists()
        assert (mem_dir / "contexts" / "ctx-2" / "MEMORY.md").exists()

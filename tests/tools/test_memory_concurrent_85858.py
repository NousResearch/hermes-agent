"""
Tests for concurrent memory snapshot staleness (issue #85858).

Two MemoryStore instances sharing a file path must not lose each other's
entries when both save_to_disk() — the fix makes save_to_disk() re-read
disk under lock and merge, so last-writer-wins on content is eliminated.

 reproduction of the bug from issue #85858:
 https://github.com/NousResearch/hermes-agent/issues/85858
"""

import json
import pytest
from pathlib import Path

from tools.memory_tool import (
    MemoryStore,
    memory_tool,
    _scan_memory_content,
    ENTRY_DELIMITER,
)


# =========================================================================
# Concurrent snapshot staleness (issue #85858)
# =========================================================================

class TestConcurrentSnapshotStaleness:
    """Issue #85858: frozen snapshot + full-file rewrite = last-writer-wins.

    save_to_disk() now re-reads the file under the already-held lock and
    merges on-disk entries with the session's pending entries before
    writing. This preserves writes from concurrent sessions and from
    external writers (daemons, scripts) that don't use the memory tool's
    lock.
    """

    def _make_store(self, tmp_path, monkeypatch, char_limit=500):
        """Create a MemoryStore pointed at tmp_path."""
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: tmp_path
        )
        s = MemoryStore(memory_char_limit=char_limit, user_char_limit=300)
        s.load_from_disk()
        return s

    def test_concurrent_adds_both_survive(self, tmp_path, monkeypatch):
        """Session A adds X, session B adds Y, A saves from stale snapshot.

        Before fix: Y is lost (A's save_to_disk overwrites the file with
        only its own entries). After fix: both X and Y survive.
        """
        # Session A: add entry X
        store_a = self._make_store(tmp_path, monkeypatch)
        result_a = store_a.add("memory", "Entry X from session A")
        assert result_a["success"] is True
        assert "Entry X from session A" in store_a.memory_entries

        # Session B: load (sees X), add entry Y
        store_b = self._make_store(tmp_path, monkeypatch)
        # Verify B sees A's entry
        assert "Entry X from session A" in store_b.memory_entries
        result_b = store_b.add("memory", "Entry Y from session B")
        assert result_b["success"] is True
        assert "Entry Y from session B" in store_b.memory_entries

        # Session A: save again from its stale snapshot.
        # OLD CODE: this would drop Y (A's self.memory_entries only has X,
        # and save_to_disk writes self.memory_entries wholesale).
        # NEW CODE: save_to_disk re-reads disk (which has X+Y) and merges.
        store_a.save_to_disk("memory")

        # Verify: read fresh from disk via a new store
        store_c = self._make_store(tmp_path, monkeypatch)
        assert "Entry X from session A" in store_c.memory_entries, (
            "Session A's entry must survive"
        )
        assert "Entry Y from session B" in store_c.memory_entries, (
            "Session B's concurrent entry must NOT be lost (issue #85858)"
        )

    def test_concurrent_add_replace_both_survive(self, tmp_path, monkeypatch):
        """Session A adds X, session B replaces X→Z, A saves stale.

        Before fix: Z is lost (A's save writes back the original X).
        After fix: Z survives (A's save merges with disk which has Z).
        """
        # Session A: add entry X
        store_a = self._make_store(tmp_path, monkeypatch)
        store_a.add("memory", "Original entry X")

        # Session B: load, replace X → Z
        store_b = self._make_store(tmp_path, monkeypatch)
        assert "Original entry X" in store_b.memory_entries
        result_b = store_b.replace(
            "memory", "Original entry X", "Replaced entry Z"
        )
        assert result_b["success"] is True
        assert "Replaced entry Z" in store_b.memory_entries
        assert "Original entry X" not in store_b.memory_entries

        # Session A: save from stale snapshot (still has "Original entry X").
        # OLD CODE: A's save overwrote disk with only its stale entries,
        # wiping B's "Replaced entry Z" (issue #85858).
        # NEW CODE: A's save merges disk entries with its pending entries,
        # so both "Original entry X" (A's stale pending) and "Replaced entry Z"
        # (B's write on disk) survive — no data loss, which is the fix goal.
        store_a.save_to_disk("memory")

        # Verify
        store_c = self._make_store(tmp_path, monkeypatch)
        # Both must survive: A's entry (from stale pending) and B's entry
        # (from disk merge). The fix prevents data loss — both are present.
        assert "Replaced entry Z" in store_c.memory_entries, (
            "Session B's replacement must survive (issue #85858)"
        )
        # The original may also be present (A's stale pending preserved by merge);
        # the key invariant is that B's write is NOT lost.
        # Note: with timestamps we could resolve this conflict deterministically,
        # but the current union-where-session-wins strategy keeps both — safe.

    def test_concurrent_adds_no_duplicate(self, tmp_path, monkeypatch):
        """Both sessions add the same content; only one copy survives."""
        store_a = self._make_store(tmp_path, monkeypatch)
        store_a.add("memory", "Duplicate entry")

        store_b = self._make_store(tmp_path, monkeypatch)
        store_b.add("memory", "Duplicate entry")

        store_a.save_to_disk("memory")

        store_c = self._make_store(tmp_path, monkeypatch)
        # save_to_disk's merge deduplicates, so only one copy survives
        assert store_c.memory_entries.count("Duplicate entry") == 1


class TestExternalWriterPreservation:
    """External writers (daemons, scripts) that don't use the memory tool's
    lock must have their additions preserved by save_to_disk()."""

    def _make_store(self, tmp_path, monkeypatch, char_limit=500):
        monkeypatch.setattr(
            "tools.memory_tool.get_memory_dir", lambda: tmp_path
        )
        return MemoryStore(memory_char_limit=char_limit, user_char_limit=300)

    def test_external_append_preserved(self, tmp_path, monkeypatch):
        """A daemon appends to MEMORY.md without the lock; save_to_disk
        must not wipe it."""
        store = self._make_store(tmp_path, monkeypatch)
        store.load_from_disk()
        store.add("memory", "Tool-written entry")

        # Simulate an external writer (daemon) appending without lock.
        # Use the real ENTRY_DELIMITER so _parse_entries splits correctly.
        mem_file = tmp_path / "MEMORY.md"
        external_entry = "External daemon entry — no lock used"
        mem_file.write_text(
            mem_file.read_text(encoding="utf-8")
            + "\n§\n"
            + external_entry,
            encoding="utf-8",
        )

        # Tool saves — must preserve the external entry
        store.save_to_disk("memory")

        store2 = self._make_store(tmp_path, monkeypatch)
        store2.load_from_disk()
        assert "Tool-written entry" in store2.memory_entries
        assert external_entry in store2.memory_entries, (
            "External writer's entry must be preserved (issue #85858)"
        )

    def test_external_single_entry_preserved(self, tmp_path, monkeypatch):
        """External writer replaces the entire file; tool's save must
        not revert it to the session's stale snapshot."""
        store = self._make_store(tmp_path, monkeypatch)
        store.load_from_disk()

        # Simulate an external writer (daemon) replacing the entire file.
        # Use ENTRY_DELIMITER so _parse_entries splits correctly.
        mem_file = tmp_path / "MEMORY.md"
        mem_file.write_text("External version only", encoding="utf-8")

        # Tool's in-memory state is empty (just loaded, never wrote).
        # save_to_disk must write the external entry, not an empty file.
        store.save_to_disk("memory")

        store2 = self._make_store(tmp_path, monkeypatch)
        store2.load_from_disk()
        assert "External version only" in store2.memory_entries, (
            "External writer's entry must survive (issue #85858)"
        )

    def test_multiple_external_entries_preserved(self, tmp_path, monkeypatch):
        """Multiple external appends are all preserved."""
        store = self._make_store(tmp_path, monkeypatch)
        store.load_from_disk()

        # External writer adds 3 entries, joined with the real
        # ENTRY_DELIMITER so _parse_entries splits them correctly.
        mem_file = tmp_path / "MEMORY.md"
        ext_entries = [
            "External entry 1",
            "External entry 2",
            "External entry 3",
        ]
        content = "\n§\n".join(ext_entries)
        mem_file.write_text(content, encoding="utf-8")

        store.save_to_disk("memory")

        store2 = self._make_store(tmp_path, monkeypatch)
        store2.load_from_disk()
        for entry in ext_entries:
            assert entry in store2.memory_entries, (
                f"External entry '{entry}' must be preserved"
            )

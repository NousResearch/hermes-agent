"""Tests for opt-in context_id memory partitioning in MemoryStore.

Partition model (per target ∈ {memory, user}):
  - Global files:  <memories>/MEMORY.md, USER.md  (unchanged, backward compatible)
  - Scoped files:  <memories>/contexts/{context_id}/MEMORY.md, USER.md

Guarantees under test:
  - A scoped file on disk contains ONLY scoped entries — global facts are
    never copied into it (the core isolation bug this partition model fixes).
  - Global is an OPT-IN, READ-ONLY layer: include_global defaults False. The
    default scoped read view is scoped-only.
  - Every mutation path (add / replace / remove / apply_batch) refuses to
    mutate a global entry from a scoped context, for BOTH targets.
  - Global membership is decided by set-membership against the global
    partition, per target (index-free) — different global counts in MEMORY vs
    USER never cross-guard.
  - Migration: a scoped file that wrongly contains a global entry (old buggy
    write-back) is healed on load.
  - Feature disabled (context_id=None) ⇒ behavior identical to flat upstream.

No real PII: fixtures use "u1", "123", and synthetic facts only.
"""

import sys

import pytest

from tools.memory_tool import MemoryStore, ENTRY_DELIMITER

# Patch get_memory_dir on the exact module object MemoryStore was defined in,
# not by dotted string. tests/tools/test_memory_tool_import_fallback.py deletes
# and re-imports tools.memory_tool, so a later ``sys.modules["tools.memory_tool"]``
# can be a different module object than the one holding this class's methods.
# Resolving via the class keeps the patch pointed at the right globals.
_MEM_MOD = sys.modules[MemoryStore.__module__]


@pytest.fixture
def mem_dir(tmp_path, monkeypatch):
    """Patch get_memory_dir to a temp dir; return the dir."""
    monkeypatch.setattr(_MEM_MOD, "get_memory_dir", lambda: tmp_path)
    return tmp_path


def _parse(path):
    """Parse a memory file on disk into its list of entries."""
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return []
    return [e.strip() for e in raw.split(ENTRY_DELIMITER) if e.strip()]


def _global_path(mem_dir, target):
    return mem_dir / ("USER.md" if target == "user" else "MEMORY.md")


def _scoped_path(mem_dir, context_id, target):
    return mem_dir / "contexts" / context_id / ("USER.md" if target == "user" else "MEMORY.md")


# =========================================================================
# Scoped writes never leak global entries into the scoped file
# =========================================================================

class TestScopedWriteDoesNotCopyGlobal:
    def test_scoped_add_does_not_copy_global_into_scoped_file(self, mem_dir):
        """Global MEMORY.md has 'global fact'; a scoped add writes ONLY the
        scoped fact into the scoped file — never the global one."""
        _global_path(mem_dir, "memory").write_text("global fact")

        store = MemoryStore(context_id="grp-1")
        store.load_from_disk()
        result = store.add("memory", "scoped fact")
        assert result["success"] is True

        scoped_file = _scoped_path(mem_dir, "grp-1", "memory")
        assert _parse(scoped_file) == ["scoped fact"]
        # Global file untouched.
        assert _parse(_global_path(mem_dir, "memory")) == ["global fact"]

    def test_scoped_add_of_global_fact_is_noop_not_copy(self, mem_dir):
        """Adding text that already exists globally is a no-op — it is NOT
        copied into the scoped partition (which load-time healing would strip
        anyway), even with include_global=False."""
        _global_path(mem_dir, "memory").write_text("global fact")

        store = MemoryStore(context_id="grp-1")  # include_global=False
        store.load_from_disk()
        result = store.add("memory", "global fact")
        assert result["success"] is True  # idempotent no-op

        # Scoped file must not have been created with the global entry in it.
        scoped_file = _scoped_path(mem_dir, "grp-1", "memory")
        assert _parse(scoped_file) == []
        assert _parse(_global_path(mem_dir, "memory")) == ["global fact"]

    def test_scoped_add_user_does_not_copy_global_user(self, mem_dir):
        """Same guarantee for the USER.md target."""
        _global_path(mem_dir, "user").write_text("global user pref")

        store = MemoryStore(context_id="grp-1")
        store.load_from_disk()
        result = store.add("user", "scoped user pref")
        assert result["success"] is True

        scoped_file = _scoped_path(mem_dir, "grp-1", "user")
        assert _parse(scoped_file) == ["scoped user pref"]
        assert _parse(_global_path(mem_dir, "user")) == ["global user pref"]


# =========================================================================
# Read view: global is opt-in
# =========================================================================

class TestReadViewOptInGlobal:
    def test_scoped_read_excludes_global_by_default(self, mem_dir):
        """include_global=False (default): read view is scoped-only."""
        _global_path(mem_dir, "memory").write_text("global fact")
        _scoped_path(mem_dir, "grp-1", "memory").parent.mkdir(parents=True, exist_ok=True)
        _scoped_path(mem_dir, "grp-1", "memory").write_text("scoped fact")

        store = MemoryStore(context_id="grp-1")
        store.load_from_disk()

        assert "scoped fact" in store.memory_entries
        assert "global fact" not in store.memory_entries

    def test_scoped_read_includes_global_when_opted_in(self, mem_dir):
        """include_global=True: read view = global + scoped (global first)."""
        _global_path(mem_dir, "memory").write_text("global fact")
        _scoped_path(mem_dir, "grp-1", "memory").parent.mkdir(parents=True, exist_ok=True)
        _scoped_path(mem_dir, "grp-1", "memory").write_text("scoped fact")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()

        assert "global fact" in store.memory_entries
        assert "scoped fact" in store.memory_entries
        # Global first.
        assert store.memory_entries.index("global fact") < store.memory_entries.index("scoped fact")

    def test_unscoped_read_is_global_only(self, mem_dir):
        """context_id=None: read view = global only (today's behavior)."""
        _global_path(mem_dir, "memory").write_text("global fact")
        store = MemoryStore()
        store.load_from_disk()
        assert store.memory_entries == ["global fact"]


# =========================================================================
# Mutation guards: global is read-only from a scoped context
# =========================================================================

class TestGlobalReadOnlyFromScope:
    def test_replace_global_entry_from_scope_refused(self, mem_dir):
        """Scoped store, include_global=True: replacing a global entry is
        refused (global is read-only from a scoped context)."""
        _global_path(mem_dir, "memory").write_text("the sky is blue")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        result = store.replace("memory", "sky is blue", "sky is green")

        assert result["success"] is False
        err = result["error"].lower()
        assert "global" in err
        # Nothing written to the scoped file; global unchanged.
        assert not _scoped_path(mem_dir, "grp-1", "memory").exists() or \
            "sky is green" not in _parse(_scoped_path(mem_dir, "grp-1", "memory"))
        assert _parse(_global_path(mem_dir, "memory")) == ["the sky is blue"]

    def test_remove_global_entry_from_scope_refused(self, mem_dir):
        """Removing a global entry from a scoped context is refused."""
        _global_path(mem_dir, "memory").write_text("the sky is blue")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        result = store.remove("memory", "sky is blue")

        assert result["success"] is False
        assert "global" in result["error"].lower()
        assert _parse(_global_path(mem_dir, "memory")) == ["the sky is blue"]

    def test_replace_global_user_entry_from_scope_refused(self, mem_dir):
        """Per-target: the same refusal holds for the USER.md target."""
        _global_path(mem_dir, "user").write_text("user likes tea")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        result = store.replace("user", "likes tea", "likes coffee")

        assert result["success"] is False
        assert "global" in result["error"].lower()
        assert _parse(_global_path(mem_dir, "user")) == ["user likes tea"]

    def test_replace_scoped_entry_from_scope_allowed(self, mem_dir):
        """Positive control: a scoped entry CAN be replaced from its scope."""
        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        store.add("memory", "scoped original")
        result = store.replace("memory", "scoped original", "scoped updated")

        assert result["success"] is True
        assert _parse(_scoped_path(mem_dir, "grp-1", "memory")) == ["scoped updated"]


# =========================================================================
# apply_batch is partition-aware
# =========================================================================

class TestApplyBatchPartitioning:
    def test_apply_batch_scoped_does_not_copy_global(self, mem_dir):
        """A batch of scoped ops leaves the global file untouched and writes
        no global entries into the scoped file."""
        _global_path(mem_dir, "memory").write_text("global fact")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        result = store.apply_batch("memory", [
            {"action": "add", "content": "scoped a"},
            {"action": "add", "content": "scoped b"},
        ])
        assert result["success"] is True

        assert _parse(_scoped_path(mem_dir, "grp-1", "memory")) == ["scoped a", "scoped b"]
        assert _parse(_global_path(mem_dir, "memory")) == ["global fact"]

    def test_apply_batch_refuses_global_mutation_from_scope(self, mem_dir):
        """A batch containing a replace/remove that targets a global entry is
        refused whole — nothing is written (all-or-nothing)."""
        _global_path(mem_dir, "memory").write_text("global fact")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        result = store.apply_batch("memory", [
            {"action": "add", "content": "scoped a"},
            {"action": "replace", "old_text": "global fact", "content": "hacked"},
        ])
        assert result["success"] is False
        assert "global" in result["error"].lower()

        # All-or-nothing: neither op committed.
        assert not _scoped_path(mem_dir, "grp-1", "memory").exists() or \
            _parse(_scoped_path(mem_dir, "grp-1", "memory")) == []
        assert _parse(_global_path(mem_dir, "memory")) == ["global fact"]

    def test_apply_batch_refuses_global_remove_from_scope(self, mem_dir):
        """Batch remove of a global entry from a scoped context is refused."""
        _global_path(mem_dir, "memory").write_text("global fact")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()
        result = store.apply_batch("memory", [
            {"action": "remove", "old_text": "global fact"},
        ])
        assert result["success"] is False
        assert "global" in result["error"].lower()
        assert _parse(_global_path(mem_dir, "memory")) == ["global fact"]


# =========================================================================
# Global membership is per-target (no cross-guard)
# =========================================================================

class TestGlobalMembershipPerTarget:
    def test_global_membership_is_per_target(self, mem_dir):
        """Different global contents in MEMORY vs USER don't cross-guard:
        a scoped entry in one target must remain mutable even when the OTHER
        target's global partition is larger."""
        # MEMORY global has two entries; USER global has none.
        _global_path(mem_dir, "memory").write_text(
            "mem global 1" + ENTRY_DELIMITER + "mem global 2"
        )

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()

        # A scoped USER add + replace must succeed — USER has no global entries,
        # so an index-based guard borrowing MEMORY's count would wrongly block.
        store.add("user", "scoped user entry")
        result = store.replace("user", "scoped user entry", "scoped user updated")
        assert result["success"] is True
        assert _parse(_scoped_path(mem_dir, "grp-1", "user")) == ["scoped user updated"]

        # And a scoped MEMORY entry (positioned AFTER the 2 globals in the read
        # view) is still mutable.
        store.add("memory", "scoped mem entry")
        result2 = store.replace("memory", "scoped mem entry", "scoped mem updated")
        assert result2["success"] is True
        assert "scoped mem updated" in _parse(_scoped_path(mem_dir, "grp-1", "memory"))


# =========================================================================
# Migration / self-healing of old buggy scoped files
# =========================================================================

class TestMigrationDedup:
    def test_migration_dedups_preexisting_global_in_scoped_file(self, mem_dir):
        """A scoped file that wrongly contains a global entry (from the old
        buggy write-back) is healed on load: the read view doesn't double-show
        it, and after any scoped write the scoped partition no longer holds
        the global entry."""
        _global_path(mem_dir, "memory").write_text("global fact")
        # Old bug: scoped file has BOTH the global fact and a real scoped fact.
        scoped_file = _scoped_path(mem_dir, "grp-1", "memory")
        scoped_file.parent.mkdir(parents=True, exist_ok=True)
        scoped_file.write_text("global fact" + ENTRY_DELIMITER + "real scoped fact")

        store = MemoryStore(context_id="grp-1", include_global=True)
        store.load_from_disk()

        # Read view shows the global fact exactly once, plus the scoped fact.
        assert store.memory_entries.count("global fact") == 1
        assert "real scoped fact" in store.memory_entries

        # The scoped partition no longer holds the global entry.
        store.add("memory", "another scoped fact")
        on_disk = _parse(scoped_file)
        assert "global fact" not in on_disk
        assert "real scoped fact" in on_disk
        assert "another scoped fact" in on_disk


# =========================================================================
# Feature disabled == flat upstream behavior
# =========================================================================

class TestFeatureDisabledMatchesFlat:
    def test_feature_disabled_matches_flat_behavior(self, mem_dir):
        """context_id=None behaves exactly like flat upstream: writes land in
        the global file, no contexts/ directory is ever created."""
        store = MemoryStore()
        store.load_from_disk()
        store.add("memory", "flat fact")
        store.add("user", "flat pref")

        assert _parse(_global_path(mem_dir, "memory")) == ["flat fact"]
        assert _parse(_global_path(mem_dir, "user")) == ["flat pref"]
        assert not (mem_dir / "contexts").exists()

        # Fresh store reads it straight back.
        store2 = MemoryStore()
        store2.load_from_disk()
        assert store2.memory_entries == ["flat fact"]
        assert store2.user_entries == ["flat pref"]

    def test_empty_context_id_is_none(self, mem_dir):
        """Empty-string context_id sanitizes to None (unscoped)."""
        store = MemoryStore(context_id="")
        assert store.context_id is None

    def test_context_id_path_traversal_sanitized(self, mem_dir):
        """Path traversal in context_id can't escape contexts/."""
        store = MemoryStore(context_id="../../etc/passwd")
        store.load_from_disk()
        store.add("memory", "trapped")
        # Nothing written outside the temp memory dir tree.
        assert not (mem_dir.parent.parent / "etc" / "passwd" / "MEMORY.md").exists()
        assert (mem_dir / "contexts").exists()

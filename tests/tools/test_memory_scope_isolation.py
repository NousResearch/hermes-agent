"""Tests for MemoryStore scope isolation."""

import pytest
from pathlib import Path

from tools.memory_tool import MemoryStore, builtin_memory_inventory, get_memory_dir, reset_builtin_memory


@pytest.fixture
def temp_memory_dir(tmp_path, monkeypatch):
    """Redirect memory directory to a temp directory."""
    mem_dir = tmp_path / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
    return mem_dir


class TestMemoryStoreIdentityScope:
    def test_no_scope_suffix_uses_base_dir(self, temp_memory_dir):
        store = MemoryStore()
        assert store._scope_suffix is None
        assert store._get_mem_dir() == temp_memory_dir

    def test_identity_writes_to_base_dir(self, temp_memory_dir):
        store = MemoryStore()
        store.load_from_disk()
        store.add(target="memory", content="test entry")
        assert (temp_memory_dir / "MEMORY.md").exists()


class TestMemoryStoreScopedPath:
    def test_scoped_dir(self, temp_memory_dir):
        store = MemoryStore(scope_suffix="default_abc123def456")
        expected = temp_memory_dir / "scopes" / "default_abc123def456"
        assert store._get_mem_dir() == expected

    def test_scoped_write(self, temp_memory_dir):
        suffix = "default_abc123def456"
        store = MemoryStore(scope_suffix=suffix)
        store.load_from_disk()
        store.add(target="memory", content="scoped entry")
        scoped_dir = temp_memory_dir / "scopes" / suffix
        assert (scoped_dir / "MEMORY.md").exists()
        assert not (temp_memory_dir / "MEMORY.md").exists()

    def test_different_scopes_isolated(self, temp_memory_dir):
        store_a = MemoryStore(scope_suffix="default_aaa111")
        store_a.load_from_disk()
        store_a.add(target="memory", content="memory from scope A")

        store_b = MemoryStore(scope_suffix="default_bbb222")
        store_b.load_from_disk()
        store_b.add(target="memory", content="memory from scope B")

        store_a2 = MemoryStore(scope_suffix="default_aaa111")
        store_a2.load_from_disk()
        entries = store_a2.memory_entries
        assert any("scope A" in e for e in entries)
        assert not any("scope B" in e for e in entries)

    def test_scoped_user_file(self, temp_memory_dir):
        suffix = "default_abc123"
        store = MemoryStore(scope_suffix=suffix)
        store.load_from_disk()
        store.add(target="user", content="user preference")
        scoped_file = temp_memory_dir / "scopes" / suffix / "USER.md"
        assert scoped_file.exists()
        assert not (temp_memory_dir / "USER.md").exists()

    def test_scoped_backwards_compat(self, temp_memory_dir):
        store = MemoryStore()
        store.load_from_disk()
        store.add(target="memory", content="base entry")
        store2 = MemoryStore()
        store2.load_from_disk()
        assert any("base entry" in e for e in store2.memory_entries)

    def test_scope_suffix_sanitized_against_path_traversal(self, temp_memory_dir):
        store = MemoryStore(scope_suffix="../../../etc/passwd")
        assert store._scope_suffix == "etcpasswd"
        store.load_from_disk()
        store.add(target="memory", content="safe")
        assert (temp_memory_dir / "scopes" / "etcpasswd" / "MEMORY.md").exists()


class TestMemoryScopeAdministration:
    def test_inventory_separates_identity_and_scoped_data(self, temp_memory_dir):
        identity = MemoryStore()
        identity.load_from_disk()
        identity.add(target="memory", content="identity")
        scoped = MemoryStore(scope_suffix="abc123")
        scoped.load_from_disk()
        scoped.add(target="user", content="scoped")
        inventory = builtin_memory_inventory()
        assert inventory["identity"]["memory"] > 0
        assert inventory["identity"]["user"] == 0
        assert inventory["scoped"]["user"] > 0
        assert inventory["scoped"]["namespaces"] == 1

    def test_default_reset_leaves_scoped_namespaces_untouched(self, temp_memory_dir):
        identity = MemoryStore()
        identity.load_from_disk()
        identity.add(target="memory", content="identity")
        scoped = MemoryStore(scope_suffix="abc123")
        scoped.load_from_disk()
        scoped.add(target="memory", content="scoped")
        reset_builtin_memory(target="all")
        assert not (temp_memory_dir / "MEMORY.md").exists()
        assert (temp_memory_dir / "scopes" / "abc123" / "MEMORY.md").exists()

    def test_explicit_all_scopes_reset_deletes_scoped_files(self, temp_memory_dir):
        for suffix in ("abc123", "def456"):
            store = MemoryStore(scope_suffix=suffix)
            store.load_from_disk()
            store.add(target="memory", content=suffix)
        deleted = reset_builtin_memory(target="all", include_scopes=True)
        assert len(deleted) == 2
        assert not list((temp_memory_dir / "scopes").glob("*/MEMORY.md"))
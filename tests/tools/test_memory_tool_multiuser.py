"""Per-user USER.md profile scoping (memory.per_user_profiles)."""
import json

import pytest

from tools import memory_tool
from tools.memory_tool import MemoryStore, resolve_user_scope


@pytest.fixture
def mem_home(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_tool, "get_hermes_home", lambda: tmp_path)
    (tmp_path / "memories").mkdir(parents=True)
    return tmp_path


def _write(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(memory_tool.ENTRY_DELIMITER.join(entries), encoding="utf-8")


class TestResolveUserScope:
    def test_flag_off_returns_legacy(self, mem_home):
        assert resolve_user_scope("12345678", {}) == ""
        assert resolve_user_scope("12345678", {"per_user_profiles": False}) == ""

    def test_no_user_id_uses_default_key(self, mem_home):
        cfg = {"per_user_profiles": True, "default_user_key": "Owner"}
        assert resolve_user_scope("", cfg) == "owner"
        assert resolve_user_scope(None, cfg) == "owner"

    def test_no_user_id_no_default_is_legacy(self, mem_home):
        assert resolve_user_scope("", {"per_user_profiles": True}) == ""

    def test_registry_maps_id_to_key(self, mem_home):
        reg = mem_home / "data" / "users.json"
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps(
            {"users": {"111": {"key": "alice"}, "222": {"key": "bob"}}}))
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("111", cfg) == "alice"
        assert resolve_user_scope("222", cfg) == "bob"

    def test_unlisted_id_gets_own_scope(self, mem_home):
        reg = mem_home / "data" / "users.json"
        reg.parent.mkdir(parents=True)
        reg.write_text(json.dumps({"users": {"111": {"key": "alice"}}}))
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("999", cfg) == "999"

    def test_missing_registry_falls_back_to_id(self, mem_home):
        cfg = {"per_user_profiles": True}
        assert resolve_user_scope("SomeUser", cfg) == "someuser"

    def test_custom_registry_path(self, mem_home, tmp_path):
        reg = tmp_path / "custom.json"
        reg.write_text(json.dumps({"users": {"7": {"key": "carol"}}}))
        cfg = {"per_user_profiles": True, "users_registry": str(reg)}
        assert resolve_user_scope("7", cfg) == "carol"


class TestScopedStore:
    def test_legacy_paths_without_scope(self, mem_home):
        store = MemoryStore()
        assert store._path_for("user") == mem_home / "memories" / "USER.md"
        assert store._path_for("memory") == mem_home / "memories" / "MEMORY.md"

    def test_scoped_user_path_shared_memory_path(self, mem_home):
        store = MemoryStore(user_scope="alice")
        assert store._path_for("user") == (
            mem_home / "memories" / "users" / "alice" / "USER.md")
        assert store._path_for("memory") == mem_home / "memories" / "MEMORY.md"

    def test_load_reads_scoped_profile(self, mem_home):
        _write(mem_home / "memories" / "USER.md", ["legacy profile entry"])
        _write(mem_home / "memories" / "users" / "alice" / "USER.md",
               ["alice profile entry"])
        _write(mem_home / "memories" / "MEMORY.md", ["shared fact"])

        legacy = MemoryStore()
        legacy.load_from_disk()
        assert legacy.user_entries == ["legacy profile entry"]

        alice = MemoryStore(user_scope="alice")
        alice.load_from_disk()
        assert alice.user_entries == ["alice profile entry"]
        assert alice.memory_entries == ["shared fact"]

    def test_save_creates_scoped_dir(self, mem_home):
        store = MemoryStore(user_scope="bob")
        store.load_from_disk()
        store.user_entries = ["bob likes tests"]
        store.save_to_disk("user")
        assert (mem_home / "memories" / "users" / "bob" / "USER.md").exists()
        # shared file untouched
        assert not (mem_home / "memories" / "USER.md").exists()

    def test_cross_scope_isolation(self, mem_home):
        a = MemoryStore(user_scope="alice")
        a.load_from_disk()
        a.user_entries = ["alice only"]
        a.save_to_disk("user")

        b = MemoryStore(user_scope="bob")
        b.load_from_disk()
        assert b.user_entries == []

    def test_scope_is_sanitized(self, mem_home):
        store = MemoryStore(user_scope="A/B..C!")
        assert store.user_scope == "abc"

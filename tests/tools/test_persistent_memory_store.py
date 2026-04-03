import sqlite3

import pytest

from tools.persistent_memory_store import PersistentMemoryStore


@pytest.fixture()
def store(tmp_path):
    return PersistentMemoryStore(
        db_path=tmp_path / "memory.db",
        memory_dir=tmp_path / "memories",
        memory_char_limit=500,
        user_char_limit=300,
    )


class TestPersistentMemoryStoreBasics:
    def test_initializes_database_file(self, store):
        assert store.db_path.exists()
        conn = sqlite3.connect(store.db_path)
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        conn.close()
        assert "memory_entries" in tables
        assert "memory_events" in tables

    def test_add_and_reload_roundtrip(self, tmp_path):
        store1 = PersistentMemoryStore(
            db_path=tmp_path / "memory.db",
            memory_dir=tmp_path / "memories",
            memory_char_limit=500,
            user_char_limit=300,
        )
        result = store1.add_entry("user", "User prefers terse replies", kind="preference")
        assert result["success"] is True
        entry_id = result["entry"]["id"]

        store2 = PersistentMemoryStore(
            db_path=tmp_path / "memory.db",
            memory_dir=tmp_path / "memories",
            memory_char_limit=500,
            user_char_limit=300,
        )
        entries = store2.list_entries("user")
        assert len(entries) == 1
        assert entries[0]["id"] == entry_id
        assert entries[0]["content"] == "User prefers terse replies"
        assert entries[0]["kind"] == "preference"
        assert entries[0]["status"] == "active"

    def test_exports_markdown_after_write(self, store):
        store.add_entry("memory", "Secrets belong in terminal, not chat", kind="constraint")
        exported = (store.memory_dir / "MEMORY.md").read_text(encoding="utf-8")
        assert "Secrets belong in terminal, not chat" in exported


class TestPersistentMemoryStoreSemantics:
    def test_duplicate_add_is_a_noop(self, store):
        first = store.add_entry("memory", "Prod site is moltclub.io", kind="project")
        second = store.add_entry("memory", "Prod site is moltclub.io", kind="project")

        assert first["success"] is True
        assert second["success"] is True
        assert second["message"] == "Entry already exists (no duplicate added)."
        assert len(store.list_entries("memory")) == 1

    def test_replace_marks_old_entry_superseded(self, store):
        added = store.add_entry("memory", "Timezone is PST", kind="environment")
        replaced = store.replace_entry("memory", "Timezone is PST", "Timezone is America/Los_Angeles", kind="environment")

        assert replaced["success"] is True
        active_entries = store.list_entries("memory")
        assert [e["content"] for e in active_entries] == ["Timezone is America/Los_Angeles"]

        all_entries = store.list_entries("memory", include_inactive=True)
        statuses = {e["content"]: e["status"] for e in all_entries}
        assert statuses["Timezone is PST"] == "superseded"
        assert statuses["Timezone is America/Los_Angeles"] == "active"
        assert replaced["entry"]["supersedes_id"] == added["entry"]["id"]

    def test_forget_marks_entry_forgotten(self, store):
        store.add_entry("user", "User hates walls of text", kind="preference")
        removed = store.forget_entry("user", "walls of text")

        assert removed["success"] is True
        assert store.list_entries("user") == []
        all_entries = store.list_entries("user", include_inactive=True)
        assert len(all_entries) == 1
        assert all_entries[0]["status"] == "forgotten"


class TestPersistentMemoryStoreRetrieval:
    def test_retrieve_for_prompt_prefers_user_preferences(self, store):
        store.add_entry("memory", "Project uses sqlite for local state", kind="project", importance=0.4)
        store.add_entry("user", "User prefers brutally concise replies", kind="preference", importance=0.95)
        store.add_entry("user", "User timezone is America/Los_Angeles", kind="identity", importance=0.7)

        block = store.render_prompt_block("user", char_limit=220)
        assert "brutally concise" in block
        assert "America/Los_Angeles" in block

    def test_retrieve_for_prompt_excludes_superseded_and_forgotten(self, store):
        store.add_entry("memory", "Use blue theme", kind="instruction")
        store.replace_entry("memory", "blue theme", "Use dark theme", kind="instruction")
        store.add_entry("memory", "Temporary launch banner stays", kind="project")
        store.forget_entry("memory", "launch banner")

        block = store.render_prompt_block("memory", char_limit=220)
        assert "Use dark theme" in block
        assert "Use blue theme" not in block
        assert "launch banner" not in block

    def test_render_prompt_block_respects_char_limit(self, store):
        store.add_entry("memory", "A" * 180, kind="lesson", importance=0.9)
        store.add_entry("memory", "B" * 180, kind="lesson", importance=0.8)

        block = store.render_prompt_block("memory", char_limit=220)
        assert isinstance(block, str)
        assert len(block) <= 220
        assert "MEMORY" in block

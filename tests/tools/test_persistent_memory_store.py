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
    def test_uses_live_hermes_home_when_env_changes_before_init(self, tmp_path, monkeypatch):
        dynamic_home = tmp_path / "dynamic-home"
        monkeypatch.setenv("HERMES_HOME", str(dynamic_home))

        from tools.persistent_memory_store import PersistentMemoryStore

        store = PersistentMemoryStore(memory_char_limit=500, user_char_limit=300)

        assert store.db_path == dynamic_home / "memory.db"
        assert store.memory_dir == dynamic_home / "memories"

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


class TestPersistentMemoryStoreMetadata:
    def test_add_entry_records_typed_metadata_and_provenance(self, store):
        result = store.add_entry(
            "user",
            "User prefers brutally concise replies",
            entry_type="user_preference",
            strength="hard_rule",
            source="user_explicit",
            scope="profile",
            scope_value="default",
            created_in_session_id="session-123",
            importance=0.95,
        )

        assert result["success"] is True
        entry = result["entry"]
        assert entry["entry_type"] == "user_preference"
        assert entry["kind"] == "user_preference"
        assert entry["strength"] == "hard_rule"
        assert entry["source"] == "user_explicit"
        assert entry["scope"] == "profile"
        assert entry["scope_value"] == "default"
        assert entry["created_in_session_id"] == "session-123"
        assert entry["replaced_by"] is None
        assert entry["forgotten_by"] is None

    def test_existing_db_gets_phase1_columns_on_open(self, tmp_path):
        db_path = tmp_path / "memory.db"
        memory_dir = tmp_path / "memories"
        memory_dir.mkdir()
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE memory_entries (
                id TEXT PRIMARY KEY,
                target TEXT NOT NULL,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                scope_value TEXT,
                source TEXT NOT NULL DEFAULT 'manual',
                confidence REAL NOT NULL DEFAULT 1.0,
                importance REAL NOT NULL DEFAULT 0.5,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_used_at REAL,
                use_count INTEGER NOT NULL DEFAULT 0,
                supersedes_id TEXT,
                fingerprint TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO memory_entries(id, target, kind, content, status, scope, scope_value, source, confidence, importance, created_at, updated_at, last_used_at, use_count, supersedes_id, fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-1", "memory", "lesson", "Legacy fact", "active", "global", None, "migration",
                1.0, 0.5, 1.0, 1.0, None, 0, None, "fp-1",
            ),
        )
        conn.commit()
        conn.close()

        reopened = PersistentMemoryStore(db_path=db_path, memory_dir=memory_dir, memory_char_limit=500, user_char_limit=300)
        rows = reopened.list_entries("memory", include_inactive=True)

        assert rows[0]["entry_type"] == "lesson"
        assert rows[0]["strength"] == "contextual"
        assert rows[0]["created_in_session_id"] is None
        assert rows[0]["replaced_by"] is None
        assert rows[0]["forgotten_by"] is None

    def test_replace_and_forget_record_lineage_fields(self, store):
        added = store.add_entry(
            "memory",
            "Theme is blue",
            entry_type="project_convention",
            strength="soft_pref",
            created_in_session_id="session-a",
        )
        replaced = store.replace_entry(
            "memory",
            "Theme is blue",
            "Theme is dark",
            entry_type="project_convention",
            strength="hard_rule",
            created_in_session_id="session-b",
        )
        forgotten = store.forget_entry("memory", "Theme is dark", forgotten_by="session-c")

        assert replaced["success"] is True
        assert forgotten["success"] is True

        all_entries = {e["content"]: e for e in store.list_entries("memory", include_inactive=True)}
        assert all_entries["Theme is blue"]["status"] == "superseded"
        assert all_entries["Theme is blue"]["replaced_by"] == replaced["entry"]["id"]
        assert all_entries["Theme is dark"]["status"] == "forgotten"
        assert all_entries["Theme is dark"]["supersedes_id"] == added["entry"]["id"]
        assert all_entries["Theme is dark"]["created_in_session_id"] == "session-b"
        assert all_entries["Theme is dark"]["forgotten_by"] == "session-c"


class TestPersistentMemoryStoreRetrieval:
    def test_retrieve_for_prompt_prefers_hard_rules_and_typed_preferences(self, store):
        store.add_entry(
            "user",
            "User likes emoji sometimes",
            entry_type="user_preference",
            strength="soft_pref",
            importance=0.95,
        )
        store.add_entry(
            "user",
            "Never use flattery or padding",
            entry_type="prohibition",
            strength="hard_rule",
            importance=0.60,
        )

        entries = store.retrieve_for_prompt("user")

        assert entries[0]["content"] == "Never use flattery or padding"
        assert entries[0]["_selection_reason"]
        assert "strength=hard_rule" in entries[0]["_selection_reason"]
        assert "entry_type=prohibition" in entries[0]["_selection_reason"]

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

    def test_explain_prompt_selection_reports_winners_and_reasons(self, store):
        store.add_entry(
            "user",
            "User prefers brutally concise replies",
            entry_type="user_preference",
            strength="strong_pref",
            source="user_explicit",
            importance=0.95,
        )
        store.add_entry(
            "user",
            "User timezone is America/Los_Angeles",
            entry_type="user_identity",
            strength="contextual",
            source="user_explicit",
            importance=0.4,
        )

        explained = store.explain_prompt_selection("user", char_limit=220)

        assert explained["target"] == "user"
        assert explained["selected"]
        assert explained["selected"][0]["content"] == "User prefers brutally concise replies"
        assert explained["selected"][0]["path"] == "hot_memory"
        assert "entry_type=user_preference" in explained["selected"][0]["reason"]
        assert "strength=strong_pref" in explained["selected"][0]["reason"]
        assert explained["selected"][0]["score"] >= explained["selected"][-1]["score"]

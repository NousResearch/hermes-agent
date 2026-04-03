from tools.memory_tool import MemoryStore


def test_memory_store_creates_sqlite_db_alongside_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path / "memories")
    monkeypatch.setattr("tools.persistent_memory_store.DEFAULT_DB_PATH", tmp_path / "memory.db")

    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()
    store.add("memory", "Prod site is moltclub.io")

    assert (tmp_path / "memory.db").exists()
    assert (tmp_path / "memories" / "MEMORY.md").exists()


def test_memory_store_bootstraps_existing_markdown_into_db(tmp_path, monkeypatch):
    memory_dir = tmp_path / "memories"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text("Legacy fact A\n§\nLegacy fact B", encoding="utf-8")
    (memory_dir / "USER.md").write_text("Legacy user pref", encoding="utf-8")

    monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", memory_dir)
    monkeypatch.setattr("tools.persistent_memory_store.DEFAULT_DB_PATH", tmp_path / "memory.db")

    store = MemoryStore(memory_char_limit=500, user_char_limit=300)
    store.load_from_disk()

    assert "Legacy fact A" in store.memory_entries
    assert "Legacy fact B" in store.memory_entries
    assert "Legacy user pref" in store.user_entries
    assert (tmp_path / "memory.db").exists()

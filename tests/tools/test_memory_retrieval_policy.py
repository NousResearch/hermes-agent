from tools.memory_tool import MemoryStore


def test_snapshot_uses_compact_retrieved_memory_block(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.memory_tool.MEMORY_DIR", tmp_path / "memories")
    store = MemoryStore(memory_char_limit=220, user_char_limit=160)
    store.load_from_disk()

    store.add("memory", "A" * 180)
    store.add("memory", "B" * 180)
    store.load_from_disk()

    snapshot = store.format_for_system_prompt("memory")
    assert snapshot is not None
    assert len(snapshot) <= 220
    assert "MEMORY" in snapshot

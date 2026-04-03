import json

from tools.persistent_memory_store import PersistentMemoryStore


def test_snapshot_export_includes_inactive_rows(tmp_path):
    store = PersistentMemoryStore(
        db_path=tmp_path / "memory.db",
        memory_dir=tmp_path / "memories",
        memory_char_limit=500,
        user_char_limit=300,
    )
    store.add_entry("memory", "Old fact", kind="lesson")
    store.forget_entry("memory", "Old fact")
    store.add_entry("memory", "Live fact", kind="lesson")

    snapshot = store.export_snapshot()
    statuses = {entry["content"]: entry["status"] for entry in snapshot["entries"]}
    assert statuses["Old fact"] == "forgotten"
    assert statuses["Live fact"] == "active"


def test_snapshot_import_roundtrip_preserves_tombstones_and_active_entries(tmp_path):
    source = PersistentMemoryStore(
        db_path=tmp_path / "source.db",
        memory_dir=tmp_path / "source_memories",
        memory_char_limit=500,
        user_char_limit=300,
    )
    source.add_entry("memory", "Theme is blue", kind="instruction")
    source.replace_entry("memory", "Theme is blue", "Theme is dark", kind="instruction")
    source.add_entry("user", "User timezone is America/Los_Angeles", kind="identity")
    source.forget_entry("user", "America/Los_Angeles")

    snapshot_path = tmp_path / "memory-snapshot.json"
    source.export_snapshot_to_file(snapshot_path)

    dest = PersistentMemoryStore(
        db_path=tmp_path / "dest.db",
        memory_dir=tmp_path / "dest_memories",
        memory_char_limit=500,
        user_char_limit=300,
    )
    imported = dest.import_snapshot_from_file(snapshot_path)
    assert imported["success"] is True

    active_memory = [e["content"] for e in dest.list_entries("memory")]
    assert active_memory == ["Theme is dark"]
    assert dest.list_entries("user") == []

    all_memory = {e["content"]: e["status"] for e in dest.list_entries("memory", include_inactive=True)}
    all_user = {e["content"]: e["status"] for e in dest.list_entries("user", include_inactive=True)}
    assert all_memory["Theme is blue"] == "superseded"
    assert all_memory["Theme is dark"] == "active"
    assert all_user["User timezone is America/Los_Angeles"] == "forgotten"

    assert "Theme is dark" in (dest.memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    user_md = (dest.memory_dir / "USER.md").read_text(encoding="utf-8")
    assert user_md == ""


def test_snapshot_file_is_json(tmp_path):
    store = PersistentMemoryStore(
        db_path=tmp_path / "memory.db",
        memory_dir=tmp_path / "memories",
        memory_char_limit=500,
        user_char_limit=300,
    )
    store.add_entry("memory", "Portable fact", kind="lesson")
    output = tmp_path / "snapshot.json"
    store.export_snapshot_to_file(output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["format"] == "hermes-memory-snapshot-v1"
    assert payload["entry_count"] == 1

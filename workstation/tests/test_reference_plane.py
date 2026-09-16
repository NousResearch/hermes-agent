import base64
import json

from workstation.reference_plane import ReadCache, blob_references, schema_projection
from workstation.artifacts import ArtifactStore


def test_duplicate_read_hash_authority_and_sections(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    cache = ReadCache(store, "session")
    first = cache.project({"path": "records.txt"}, "alpha\nbeta\ngamma")
    second = ReadCache(store, "session").project({"path": "records.txt"}, "alpha\nbeta\ngamma")
    assert not first["cache_hit"]
    assert second["cache_hit"]
    assert second["status"] == "unchanged"
    assert "alpha" not in json.dumps(second)
    assert cache.section(second["artifact_ref"], offset=2, limit=1) == "beta"
    changed = cache.project({"path": "records.txt"}, "ALPHA\nbeta\ngamma")
    assert not changed["cache_hit"]
    assert changed["content_hash"] != first["content_hash"]


def test_blob_content_addressing_keeps_original_recoverable(tmp_path):
    store = ArtifactStore(tmp_path)
    raw = b"pixels" * 10000
    data = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    projection = blob_references(store, "task", {"images": [data, data]})
    images = projection["images"]
    assert images[0] == images[1]
    assert store.resolve_ref(images[0]["blob_ref"]).read_bytes() == raw
    assert "base64" not in json.dumps(projection)
    assert len(json.dumps(projection)) < 1000


def test_schema_cache_fingerprint_reload_and_full(tmp_path):
    store = ArtifactStore(tmp_path)
    schema = {"name": "mcp_write", "parameters": {"type": "object", "properties": {}}}
    first = schema_projection(store, "session", schema, "v1")
    repeated = schema_projection(store, "session", schema, "v1")
    assert "schema" in first
    assert "schema" not in repeated
    assert repeated["schema_hash"] == first["schema_hash"]
    assert "schema" in schema_projection(store, "session", schema, "v2")
    assert "schema" in schema_projection(store, "session", schema, "v1", full=True)


def test_real_read_file_same_size_restored_mtime_invalidates(tmp_path, monkeypatch):
    import os
    from tools.file_tools import read_file_tool
    from workstation.task_compiler import _execution_active
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nbeta")
    original = path.stat()
    token = _execution_active.set(True)
    try:
        first = json.loads(read_file_tool(str(path), task_id="read-task"))
        second = json.loads(read_file_tool(str(path), task_id="read-task"))
        assert "content" in first
        assert second["cache_hit"]
        assert "content" not in second
        path.write_text("ALPHA\nbeta")
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        third = json.loads(read_file_tool(str(path), task_id="read-task"))
        assert "ALPHA" in third["content"]
    finally:
        _execution_active.reset(token)

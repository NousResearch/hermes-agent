"""Spawn-tree storage accepts long session IDs without exceeding directory-name limits."""

import json
from pathlib import Path

import pytest

import tui_gateway.server as server


@pytest.mark.parametrize("prefix", ["a" * 300, "𐐀" * 64])
def test_long_ids_have_stable_distinct_bounded_directories(tmp_path, monkeypatch, prefix):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = server._spawn_tree_session_dir(prefix + "one")
    second = server._spawn_tree_session_dir(prefix + "two")
    assert first.is_dir() and second.is_dir()
    assert first != second
    assert first == server._spawn_tree_session_dir(prefix + "one")
    assert len(first.name.encode("utf-8")) <= 64
    assert len(second.name.encode("utf-8")) <= 64
    assert server._spawn_tree_session_dir("abc-123_def").name == "abc-123_def"


@pytest.mark.parametrize("legacy", [False, True])
def test_long_id_save_and_list_through_rpc(tmp_path, monkeypatch, legacy):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session_id = "a" * (80 if legacy else 300)
    if legacy:
        old_dir = tmp_path / "spawn-trees" / session_id
        old_dir.mkdir(parents=True)
        old_path = old_dir / "old.json"
        old_path.write_text(json.dumps({"session_id": session_id, "subagents": [{"id": "old-child"}]}), encoding="utf-8")
        listed = server.handle_request({
            "jsonrpc": "2.0", "id": "legacy", "method": "spawn_tree.list",
            "params": {"session_id": session_id},
        })
        assert any(entry["path"] == str(old_path) for entry in listed["result"]["entries"])
    saved = server.handle_request({
        "jsonrpc": "2.0", "id": "save", "method": "spawn_tree.save",
        "params": {"session_id": session_id, "subagents": [{"id": "child"}],
                   "finished_at": 1700000000},
    })
    assert "result" in saved, saved
    path = Path(saved["result"]["path"])
    assert path.is_file()
    assert path.is_relative_to(tmp_path)
    listed = server.handle_request({
        "jsonrpc": "2.0", "id": "list", "method": "spawn_tree.list",
        "params": {"session_id": session_id},
    })
    assert "result" in listed, listed
    entries = listed["result"]["entries"]
    assert any(entry["path"] == str(path) and entry["session_id"] == session_id for entry in entries)

"""spawn_tree.list must clamp ``limit`` before slicing entries.

A negative limit is a valid Python slice (``entries[:-n]``) and would
silently drop the newest snapshots from the result. Huge limits can also
inflate memory when scanning many snapshot files.
"""

from __future__ import annotations

from pathlib import Path

from tui_gateway import server


def _seed_snapshots(tmp_path: Path, count: int) -> Path:
    session_dir = tmp_path / "spawn-trees" / "default"
    session_dir.mkdir(parents=True)
    for i in range(count):
        p = session_dir / f"snap-{i:03d}.json"
        p.write_text(
            f'{{"session_id": "default", "finished_at": {i}, "subagents": []}}',
            encoding="utf-8",
        )
    return session_dir


def test_spawn_tree_list_clamps_negative_limit(tmp_path, monkeypatch):
    _seed_snapshots(tmp_path, 5)
    monkeypatch.setattr(server, "_spawn_trees_root", lambda: tmp_path / "spawn-trees")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "spawn_tree.list",
            "params": {"session_id": "default", "limit": -2},
        }
    )
    assert "error" not in resp
    assert len(resp["result"]["entries"]) == 1


def test_spawn_tree_list_clamps_huge_limit(tmp_path, monkeypatch):
    _seed_snapshots(tmp_path, 3)
    monkeypatch.setattr(server, "_spawn_trees_root", lambda: tmp_path / "spawn-trees")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "spawn_tree.list",
            "params": {"session_id": "default", "limit": 99999},
        }
    )
    assert "error" not in resp
    assert len(resp["result"]["entries"]) == 3


def test_spawn_tree_list_invalid_limit_falls_back(tmp_path, monkeypatch):
    _seed_snapshots(tmp_path, 2)
    monkeypatch.setattr(server, "_spawn_trees_root", lambda: tmp_path / "spawn-trees")

    resp = server.handle_request(
        {
            "id": "1",
            "method": "spawn_tree.list",
            "params": {"session_id": "default", "limit": "nope"},
        }
    )
    assert "error" not in resp
    assert len(resp["result"]["entries"]) == 2

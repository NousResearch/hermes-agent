"""Regression: spawn_tree.list must clamp limit before slicing results."""

from __future__ import annotations

import json

from tui_gateway import server


def _seed_index(tmp_path, monkeypatch, n: int) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session_dir = tmp_path / "spawn-trees" / "sess-test"
    session_dir.mkdir(parents=True)
    lines = []
    for i in range(n):
        snap = session_dir / f"20260101T{i:06d}.json"
        snap.write_text("{}", encoding="utf-8")
        lines.append(
            json.dumps(
                {
                    "path": str(snap),
                    "session_id": "sess-test",
                    "started_at": float(i),
                    "finished_at": float(1000 + i),
                    "label": f"run-{i}",
                    "count": 1,
                }
            )
        )
    (session_dir / "_index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _call(**params):
    return server.handle_request(
        {
            "id": "1",
            "method": "spawn_tree.list",
            "params": {"session_id": "sess-test", **params},
        }
    )


def test_spawn_tree_list_clamps_excessive_limit(tmp_path, monkeypatch):
    _seed_index(tmp_path, monkeypatch, n=520)
    resp = _call(limit=10_000_000)
    assert "result" in resp
    assert len(resp["result"]["entries"]) == 500


def test_spawn_tree_list_clamps_negative_limit(tmp_path, monkeypatch):
    """Negative limits must not flip ``entries[:limit]`` into nearly-all rows."""
    _seed_index(tmp_path, monkeypatch, n=10)
    resp = _call(limit=-5)
    assert len(resp["result"]["entries"]) == 1


def test_spawn_tree_list_clamps_zero_limit(tmp_path, monkeypatch):
    _seed_index(tmp_path, monkeypatch, n=5)
    resp = _call(limit=0)
    assert len(resp["result"]["entries"]) == 1


def test_spawn_tree_list_default_limit(tmp_path, monkeypatch):
    _seed_index(tmp_path, monkeypatch, n=60)
    resp = _call()
    assert len(resp["result"]["entries"]) == 50


def test_spawn_tree_list_invalid_limit_falls_back_to_default(tmp_path, monkeypatch):
    _seed_index(tmp_path, monkeypatch, n=60)
    resp = _call(limit="nope")
    assert len(resp["result"]["entries"]) == 50

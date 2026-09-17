"""spawn_tree snapshot hardening: parseable non-records must not break list/load.

Covers ``tui_gateway/methods_session.py`` for #114240. The RPC envelope
helpers (``_ok``/``_err``) and tree dirs are binder-provided in production;
the tests stub them the way ``bind_module`` would.
"""
import json

import pytest

import tui_gateway.methods_session as ms


def _bind(tmp_path, monkeypatch):
    import datetime as _datetime
    import json as _json
    import time as _time
    from pathlib import Path as _Path

    # Production resolves these through bind_module onto the server's globals;
    # the tests publish the same names the way the binder would.
    monkeypatch.setattr(ms, "json", _json, raising=False)
    monkeypatch.setattr(ms, "Path", _Path, raising=False)
    monkeypatch.setattr(ms, "datetime", _datetime, raising=False)
    monkeypatch.setattr(ms, "time", _time, raising=False)
    monkeypatch.setattr(ms, "_ok", lambda rid, payload: {"ok": True, "result": payload}, raising=False)
    monkeypatch.setattr(ms, "_err", lambda rid, code, msg: {"ok": False, "code": code, "error": msg}, raising=False)
    root = tmp_path / "spawn-trees"
    monkeypatch.setattr(ms, "_spawn_trees_root", lambda: root, raising=False)
    monkeypatch.setattr(ms, "_spawn_tree_session_dir", lambda sid: root / sid, raising=False)
    return root


def _load_fn():
    return dict(ms._registry._pending)["spawn_tree.load"]


def test_legacy_entry_skips_non_record(tmp_path, monkeypatch):
    import json as _json

    # See _bind: json resolves through the production binder, not an import.
    monkeypatch.setattr(ms, "json", _json, raising=False)
    d = tmp_path / "s"
    d.mkdir()
    bad = d / "x.json"
    bad.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert ms._legacy_spawn_tree_entry(bad, d.name) is None
    assert bad.exists()
    good = d / "y.json"
    good.write_text(json.dumps({"session_id": "s", "subagents": []}), encoding="utf-8")
    entry = ms._legacy_spawn_tree_entry(good, d.name)
    assert entry is not None and entry["session_id"] == "s"


@pytest.mark.parametrize("payload", [42, "oops", [1]])
def test_load_rejects_non_record(tmp_path, monkeypatch, payload):
    root = _bind(tmp_path, monkeypatch)
    d = root / "s"
    d.mkdir(parents=True)
    p = d / "t.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    out = _load_fn()("r1", {"path": str(p)})
    assert out["ok"] is False and out["code"] == 5000
    assert p.exists()
    good = d / "g.json"
    good.write_text(json.dumps({"subagents": []}), encoding="utf-8")
    out = _load_fn()("r2", {"path": str(good)})
    assert out["ok"] is True

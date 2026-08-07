"""Clamp projects.tree / projects.project_sessions session+preview limits."""

from __future__ import annotations

from tui_gateway import methods_config as cfg
from tui_gateway import server


def test_coerce_project_session_limit_bounds():
    assert cfg._coerce_project_session_limit(-1, default=2000) == 1
    assert cfg._coerce_project_session_limit(0, default=2000) == 1
    assert cfg._coerce_project_session_limit(None, default=2000) == 2000
    assert cfg._coerce_project_session_limit("", default=5000) == 5000
    assert cfg._coerce_project_session_limit("nope", default=2000) == 2000
    assert cfg._coerce_project_session_limit(50_000, default=2000) == 10_000
    assert cfg._coerce_project_session_limit(3500, default=2000) == 3500


def test_coerce_project_preview_limit_bounds():
    assert cfg._coerce_project_preview_limit(-5, default=3) == 0
    assert cfg._coerce_project_preview_limit(0, default=3) == 0
    assert cfg._coerce_project_preview_limit(None, default=3) == 3
    assert cfg._coerce_project_preview_limit(999, default=3) == 50


def _patch_tree(monkeypatch, captured: dict):
    monkeypatch.setattr(server, "_get_db", lambda: object())

    def fake_build(db, **kwargs):
        captured.update(kwargs)
        return {"projects": [], "scoped_session_ids": []}, None

    monkeypatch.setattr(server, "_build_project_tree", fake_build)


def test_projects_tree_clamps_negative_session_limit(monkeypatch):
    captured: dict = {}
    _patch_tree(monkeypatch, captured)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "projects.tree",
            "params": {"session_limit": -1, "preview_limit": -2},
        }
    )
    assert "result" in resp
    assert captured["session_limit"] == 1
    assert captured["preview_limit"] == 0


def test_projects_tree_clamps_excessive_limits(monkeypatch):
    captured: dict = {}
    _patch_tree(monkeypatch, captured)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "projects.tree",
            "params": {"session_limit": 10_000_000, "preview_limit": 10_000},
        }
    )
    assert "result" in resp
    assert captured["session_limit"] == 10_000
    assert captured["preview_limit"] == 50


def test_projects_tree_default_limits(monkeypatch):
    captured: dict = {}
    _patch_tree(monkeypatch, captured)
    resp = server.handle_request({"id": "1", "method": "projects.tree", "params": {}})
    assert "result" in resp
    assert captured["session_limit"] == 2000
    assert captured["preview_limit"] == 3


def test_projects_project_sessions_clamps_session_limit(monkeypatch):
    captured: dict = {}
    _patch_tree(monkeypatch, captured)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "projects.project_sessions",
            "params": {"project_id": "p1", "session_limit": -1},
        }
    )
    assert "result" in resp
    assert captured["session_limit"] == 1
    assert captured["preview_limit"] == 0
    assert captured["hydrate"] is True


def test_projects_project_sessions_default_session_limit(monkeypatch):
    captured: dict = {}
    _patch_tree(monkeypatch, captured)
    resp = server.handle_request(
        {
            "id": "1",
            "method": "projects.project_sessions",
            "params": {"project_id": "p1"},
        }
    )
    assert "result" in resp
    assert captured["session_limit"] == 5000

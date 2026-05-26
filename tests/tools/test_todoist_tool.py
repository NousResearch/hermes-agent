"""Tests for Todoist read-only Sync API wrapper."""

from __future__ import annotations

import json
import urllib.error


def _urlerror_response(code: int, body: str = ""):
    class _Body:
        def read(self):
            return body.encode()
        def close(self):
            return None
    return urllib.error.HTTPError("https://api.todoist.com/api/v1/sync", code, "err", {}, _Body())


def test_todoist_missing_token_is_actionable(monkeypatch):
    from tools.todoist_tool import todoist_read_only_probe

    monkeypatch.delenv("TODOIST_API_TOKEN", raising=False)
    monkeypatch.delenv("TODOIST_TOKEN", raising=False)

    result = json.loads(todoist_read_only_probe())

    assert result["success"] is False
    assert result["missing_secret"] == "TODOIST_API_TOKEN"
    assert "Bitwarden" in result["error"]


def test_todoist_probe_uses_sync_api_and_redacts_token(monkeypatch):
    from tools.todoist_tool import todoist_read_only_probe

    captured = {}

    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "projects": [{"id": "p1", "name": "Inbox"}],
                "items": [{"id": "i1", "content": "Task", "project_id": "p1", "checked": False}],
                "labels": [{"id": "l1", "name": "Home"}],
                "sections": [],
                "sync_token": "opaque",
            }).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["data"] = req.data.decode()
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("TODOIST_API_TOKEN", "secret-token")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result_text = todoist_read_only_probe()
    result = json.loads(result_text)

    assert captured["url"] == "https://api.todoist.com/api/v1/sync"
    assert captured["timeout"] == 20
    assert "Authorization" in captured["headers"]
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert "resource_types" in captured["data"]
    assert result["success"] is True
    assert result["counts"] == {"projects": 1, "items": 1, "labels": 1, "sections": 0}
    assert result["projects"][0]["name"] == "Inbox"
    assert "secret-token" not in result_text
    assert "opaque" not in result_text


def test_todoist_probe_surfaces_http_error_without_leaking_token(monkeypatch):
    from tools.todoist_tool import todoist_read_only_probe

    def fake_urlopen(req, timeout=None):
        raise _urlerror_response(401, '{"error":"unauthorized"}')

    monkeypatch.setenv("TODOIST_API_TOKEN", "bad-token")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result_text = todoist_read_only_probe()
    result = json.loads(result_text)

    assert result["success"] is False
    assert result["status_code"] == 401
    assert "bad-token" not in result_text


def test_todoist_list_tasks_limits_and_maps_projects(monkeypatch):
    from tools.todoist_tool import todoist_list_tasks

    class _Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return json.dumps({
                "projects": [{"id": "p1", "name": "Inbox"}],
                "items": [
                    {"id": "i1", "content": "Task 1", "project_id": "p1", "checked": False, "due": {"date": "2026-05-27"}},
                    {"id": "i2", "content": "Done", "project_id": "p1", "checked": True},
                ],
                "labels": [],
                "sections": [],
                "sync_token": "opaque",
            }).encode()

    monkeypatch.setenv("TODOIST_API_TOKEN", "secret-token")
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=None: _Response())

    result = json.loads(todoist_list_tasks(limit=5, include_completed=False))

    assert result["success"] is True
    assert result["tasks_count"] == 1
    assert result["tasks"][0]["content"] == "Task 1"
    assert result["tasks"][0]["project_name"] == "Inbox"
    assert result["tasks"][0]["due"] == {"date": "2026-05-27"}

"""Tests for the Flow State local API tool module."""

import json
import urllib.error
from unittest.mock import patch

import pytest

from tools import flowstate_tool as fst


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _capturing_urlopen(seen, payload):
    def _urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["headers"] = dict(req.header_items())
        seen["body"] = None if req.data is None else json.loads(req.data.decode("utf-8"))
        seen["timeout"] = timeout
        return _Response(payload)

    return _urlopen


@pytest.fixture(autouse=True)
def flowstate_config(monkeypatch):
    monkeypatch.setattr(fst, "_FLOW_STATE_API_URL", "http://127.0.0.1:5577")
    monkeypatch.setattr(fst, "_FLOW_STATE_API_TOKEN", "token-123")


def test_list_tasks_sends_query_and_bearer_header(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fst.urllib.request,
        "urlopen",
        _capturing_urlopen(seen, {"tasks": [{"id": "t1", "title": "Plan"}]}),
    )

    result = json.loads(fst._handle_list_tasks({"status": "open", "due": "today", "limit": 5}))

    assert result["result"]["tasks"][0]["id"] == "t1"
    assert seen["url"] == "http://127.0.0.1:5577/api/tasks?status=open&due=today&limit=5"
    assert seen["method"] == "GET"
    assert seen["headers"]["Authorization"] == "Bearer token-123"


def test_create_task_omits_empty_project_id(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fst.urllib.request,
        "urlopen",
        _capturing_urlopen(seen, {"ok": True, "task": {"id": "new-id"}}),
    )

    result = json.loads(fst._handle_create_task({
        "title": "Review budget",
        "description": "Before Friday",
        "priority": "high",
        "dueDate": "2026-07-10",
        "projectId": "",
    }))

    assert result["result"]["task"]["id"] == "new-id"
    assert seen["method"] == "POST"
    assert seen["url"] == "http://127.0.0.1:5577/api/tasks"
    assert seen["body"] == {
        "title": "Review budget",
        "description": "Before Friday",
        "priority": "high",
        "dueDate": "2026-07-10",
        "projectId": None,
    }


def test_update_task_requires_a_change():
    result = json.loads(fst._handle_update_task({"id": "task-1"}))

    assert result["error"]
    assert "at least one field" in result["error"]


def test_update_task_validates_status():
    result = json.loads(fst._handle_update_task({"id": "task-1", "status": "paused"}))

    assert result["error"] == "status must be todo|done"


def test_delete_task_uses_exact_id(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fst.urllib.request,
        "urlopen",
        _capturing_urlopen(seen, {"ok": True}),
    )

    result = json.loads(fst._handle_delete_task({"id": "task/with/slash"}))

    assert result["result"]["ok"] is True
    assert seen["method"] == "DELETE"
    assert seen["url"] == "http://127.0.0.1:5577/api/tasks/task%2Fwith%2Fslash"


def test_unauthorized_error_is_actionable(monkeypatch):
    def _raise(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            {},
            None,
        )

    monkeypatch.setattr(fst.urllib.request, "urlopen", _raise)

    result = json.loads(fst._handle_list_tasks({}))

    assert "FLOW_STATE_API_TOKEN" in result["error"]


def test_unavailable_error_mentions_local_api(monkeypatch):
    def _raise(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(fst.urllib.request, "urlopen", _raise)

    result = json.loads(fst._handle_health({}))

    assert "Flow State Local Task API is unavailable" in result["error"]


def test_health_uses_existing_api_contract(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        fst.urllib.request,
        "urlopen",
        _capturing_urlopen(seen, {"ok": True}),
    )

    result = json.loads(fst._handle_health({}))

    assert result["result"] == {"ok": True}
    assert seen["url"] == "http://127.0.0.1:5577/api/health"


def test_toolset_registration_maps_all_flowstate_tools():
    from tools.registry import registry

    expected = {
        "flowstate_health",
        "flowstate_list_tasks",
        "flowstate_create_task",
        "flowstate_update_task",
        "flowstate_delete_task",
        "flowstate_get_current_timer",
    }

    for tool in expected:
        assert registry.get_toolset_for_tool(tool) == "flowstate"

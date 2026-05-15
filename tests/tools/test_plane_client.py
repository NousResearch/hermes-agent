import json
import urllib.error
from unittest.mock import patch

import pytest

import tools.plane_client as plane_client
from tools.plane_client import (
    BROWSER_USER_AGENT,
    PlaneAPIError,
    PlaneClient,
    PlaneConfigurationError,
)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def read(self):
        if self.payload is None:
            return b""
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def plane_env(monkeypatch):
    monkeypatch.setenv("PLANE_API_KEY", "plane-token")
    monkeypatch.setenv("PLANE_WORKSPACE", "ai_factory")
    monkeypatch.setenv("PLANE_PROJECT_ID", "project-123")


def test_from_env_requires_all_values(monkeypatch):
    monkeypatch.setattr(plane_client, "load_hermes_dotenv", lambda hermes_home=None: None)
    monkeypatch.delenv("PLANE_API_KEY", raising=False)
    monkeypatch.delenv("PLANE_WORKSPACE", raising=False)
    monkeypatch.delenv("PLANE_PROJECT_ID", raising=False)
    with pytest.raises(PlaneConfigurationError, match="PLANE_API_KEY"):
        PlaneClient.from_env()


def test_headers_include_browser_user_agent_and_api_key(plane_env):
    client = PlaneClient.from_env()
    headers = client.headers()
    assert headers["X-API-Key"] == "plane-token"
    assert headers["User-Agent"] == BROWSER_USER_AGENT
    assert headers["Accept"] == "application/json"


def test_request_raises_plane_api_error_with_response_body(plane_env):
    client = PlaneClient.from_env()
    err = urllib.error.HTTPError(
        url="https://api.plane.so/test",
        code=403,
        msg="Forbidden",
        hdrs=None,
        fp=None,
    )
    err.read = lambda: b'{"detail":"browser_signature_banned"}'
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(PlaneAPIError, match="403") as exc:
            client._request("GET", "/test")
    assert "browser_signature_banned" in exc.value.body


def test_get_current_user_calls_users_me_endpoint(plane_env):
    client = PlaneClient.from_env()
    calls = []

    def fake_urlopen(req, timeout=30):
        calls.append(req)
        return _FakeResponse({"id": "u1", "email": "emeric@example.com"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        user = client.get_current_user()

    assert user["email"] == "emeric@example.com"
    assert calls[0].full_url == "https://api.plane.so/api/v1/users/me/"
    assert calls[0].headers["User-agent"] == BROWSER_USER_AGENT


def test_list_work_items_follows_next_cursor_pagination(plane_env):
    client = PlaneClient.from_env()
    calls = []

    def fake_urlopen(req, timeout=30):
        calls.append(req.full_url)
        if len(calls) == 1:
            return _FakeResponse({
                "results": [{"id": "1"}],
                "next_page_results": True,
                "next_cursor": "cursor-2",
            })
        return _FakeResponse({
            "results": [{"id": "2"}],
            "next_page_results": False,
        })

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        items = client.list_work_items()

    assert [item["id"] for item in items] == ["1", "2"]
    assert "cursor=cursor-2" in calls[1]


def test_find_work_item_by_external_id_filters_client_side_after_api_lookup(plane_env):
    client = PlaneClient.from_env()
    calls = []

    def fake_paginate(path, *, params=None):
        calls.append(dict(params or {}))
        if params and params.get("external_id") == "ext-1":
            return [{"id": "wrong", "external_source": "other", "external_id": "ext-1"}]
        return [
            {"id": "match", "external_source": "nova-hermes", "external_id": "ext-1"},
            {"id": "other", "external_source": "nova-hermes", "external_id": "ext-2"},
        ]

    client.paginate = fake_paginate

    item = client.find_work_item_by_external_id(
        external_source="nova-hermes",
        external_id="ext-1",
    )

    assert item["id"] == "match"
    assert calls[0]["external_source"] == "nova-hermes"
    assert calls[0]["external_id"] == "ext-1"
    assert len(calls) == 2


def test_find_work_item_by_external_id_falls_back_when_server_rejects_external_filters(plane_env):
    client = PlaneClient.from_env()
    calls = []

    def fake_list_work_items(**kwargs):
        calls.append(kwargs)
        if kwargs.get("external_id") == "ext-1":
            raise PlaneAPIError(404, '{"error":"The requested resource does not exist."}', "https://api.plane.so/test")
        return [
            {"id": "match", "external_source": "nova-hermes", "external_id": "ext-1"},
            {"id": "other", "external_source": "nova-hermes", "external_id": "ext-2"},
        ]

    client.list_work_items = fake_list_work_items

    item = client.find_work_item_by_external_id(
        external_source="nova-hermes",
        external_id="ext-1",
    )

    assert item["id"] == "match"
    assert calls[0]["external_source"] == "nova-hermes"
    assert calls[0]["external_id"] == "ext-1"
    assert "external_source" not in calls[1]
    assert "external_id" not in calls[1]


def test_resolve_state_id_by_name_and_id(plane_env):
    client = PlaneClient.from_env()
    client._states_cache = [
        {"id": "s1", "name": "Todo"},
        {"id": "s2", "name": "Done"},
    ]
    assert client.resolve_state_id("Todo") == "s1"
    assert client.resolve_state_id("s2") == "s2"
    with pytest.raises(ValueError, match="Unknown Plane state"):
        client.resolve_state_id("Missing")


def test_resolve_label_ids_by_name_and_id(plane_env):
    client = PlaneClient.from_env()
    client._labels_cache = [
        {"id": "l1", "name": "backend"},
        {"id": "l2", "name": "research"},
    ]
    assert client.resolve_label_ids(["backend", "l2", "backend"]) == ["l1", "l2"]
    with pytest.raises(ValueError, match="Unknown Plane label"):
        client.resolve_label_ids(["missing"])


def test_create_work_item_posts_payload_to_work_items_endpoint(plane_env):
    client = PlaneClient.from_env()
    seen = {}

    def fake_urlopen(req, timeout=30):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["content_type"] = req.headers["Content-type"]
        return _FakeResponse({
            "id": "w-new",
            "sequence_id": 7,
            "name": "Nouvelle tâche",
            "external_source": "nova-hermes",
            "external_id": "ext-7",
        })

    payload = {
        "name": "Nouvelle tâche",
        "description_html": "<p>Body</p>",
        "external_source": "nova-hermes",
        "external_id": "ext-7",
    }
    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        created = client.create_work_item(payload)

    assert created["id"] == "w-new"
    assert seen["method"] == "POST"
    assert seen["url"].endswith(
        "/api/v1/workspaces/ai_factory/projects/project-123/work-items/"
    )
    assert seen["body"] == payload
    assert seen["content_type"] == "application/json"


def test_create_work_item_rejects_empty_payload(plane_env):
    client = PlaneClient.from_env()
    with pytest.raises(ValueError, match="payload is required"):
        client.create_work_item({})


def test_update_work_item_sends_patch_with_payload(plane_env):
    client = PlaneClient.from_env()
    seen = {}

    def fake_urlopen(req, timeout=30):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResponse({"id": "w1", "name": "Updated"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        updated = client.update_work_item("w1", {"name": "Updated", "priority": "high"})

    assert updated["id"] == "w1"
    assert seen["method"] == "PATCH"
    assert seen["url"].endswith(
        "/api/v1/workspaces/ai_factory/projects/project-123/work-items/w1/"
    )
    assert seen["body"] == {"name": "Updated", "priority": "high"}


def test_update_work_item_requires_id_and_payload(plane_env):
    client = PlaneClient.from_env()
    with pytest.raises(ValueError, match="work_item_id is required"):
        client.update_work_item("", {"name": "x"})
    with pytest.raises(ValueError, match="payload is required"):
        client.update_work_item("w1", {})


def test_add_comment_posts_comment_html_to_issue_comments_endpoint(plane_env):
    client = PlaneClient.from_env()
    seen = {}

    def fake_urlopen(req, timeout=30):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["content_type"] = req.headers["Content-type"]
        return _FakeResponse({"id": "c1", "comment_html": "<p>[Nova] done</p>"})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        comment = client.add_comment("w1", "<p>[Nova] done</p>")

    assert comment["id"] == "c1"
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/api/v1/workspaces/ai_factory/projects/project-123/issues/w1/comments/")
    assert seen["body"] == {"comment_html": "<p>[Nova] done</p>"}
    assert seen["content_type"] == "application/json"

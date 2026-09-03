"""Seam and route contracts for the messaging-catalog extraction (R4-C2C5)."""

from fastapi.testclient import TestClient

from hermes_cli import web_server as ws
from hermes_cli.web_routers import messaging as m


MOVED_NAMES = (
    "_build_catalog_entry",
    "_channel_managed_env_keys",
    "_MESSAGING_KEYS_PAGE_KEYS",
    "_messaging_platform_catalog",
    "_messaging_platform_payload",
    "_write_platform_enabled",
    "get_messaging_platforms",
    "update_messaging_platform",
    "test_messaging_platform",
)


def _client_with_app_state():
    prev_auth = getattr(ws.app.state, "auth_required", None)
    prev_host = getattr(ws.app.state, "bound_host", None)
    ws.app.state.auth_required = False
    ws.app.state.bound_host = None
    client = TestClient(ws.app)
    client.headers[ws._SESSION_HEADER_NAME] = ws._SESSION_TOKEN
    return client, prev_auth, prev_host


def _restore(prev_auth, prev_host):
    if prev_auth is None:
        delattr(ws.app.state, "auth_required")
    else:
        ws.app.state.auth_required = prev_auth
    if prev_host is None:
        if hasattr(ws.app.state, "bound_host"):
            delattr(ws.app.state, "bound_host")
    else:
        ws.app.state.bound_host = prev_host


def test_moved_names_are_seam_identical():
    for name in MOVED_NAMES:
        assert getattr(ws, name, None) is getattr(m, name, None), name


def test_messaging_routes_registered():
    paths = [
        route.path
        for route in ws.app.routes
        if "/api/messaging/platforms" in getattr(route, "path", "")
    ]
    assert any(path == "/api/messaging/platforms" for path in paths)
    assert "/api/messaging/platforms/{platform_id}" in paths
    assert "/api/messaging/platforms/{platform_id}/test" in paths


def test_get_messaging_platforms_returns_platforms_dict():
    client, prev_auth, prev_host = _client_with_app_state()
    try:
        response = client.get("/api/messaging/platforms")
        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, dict)
        assert "platforms" in body
        assert isinstance(body["platforms"], list)
    finally:
        _restore(prev_auth, prev_host)
        client.close()


def test_catalog_lookup_unknown_platform():
    result = m._catalog_lookup("definitely-not-a-platform")
    assert not result


def test_messaging_env_info_empty_env(monkeypatch):
    monkeypatch.setattr(m, "_discover_platform_env_vars", lambda *args, **kwargs: {})
    result = m._messaging_env_info("telegram")
    assert isinstance(result, (dict, list))


def test_channel_managed_env_keys_returns_frozenset():
    result = m._channel_managed_env_keys()
    assert isinstance(result, frozenset)

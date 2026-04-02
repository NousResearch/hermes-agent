"""Persistence tests for the Camofox browser backend.

Tests that managed persistence uses stable identity while default mode
uses random identity. The actual browser profile persistence is handled
by the Camofox server (when CAMOFOX_PROFILE_DIR is set).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.browser_camofox import (
    _drop_session,
    _get_session,
    _managed_persistence_enabled,
    camofox_close,
    camofox_navigate,
    check_camofox_available,
    cleanup_all_camofox_sessions,
    get_vnc_url,
)
from tools.browser_camofox_state import get_camofox_identity


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    return resp


def _enable_persistence():
    """Return a patch context that enables managed persistence via config."""
    config = {"browser": {"camofox": {"managed_persistence": True}}}
    return patch("tools.browser_camofox.load_config", return_value=config)


@pytest.fixture(autouse=True)
def _clear_session_state():
    import tools.browser_camofox as mod
    yield
    with mod._sessions_lock:
        mod._sessions.clear()
    mod._vnc_url = None
    mod._vnc_url_checked = False


class TestManagedPersistenceToggle:
    def test_disabled_by_default(self):
        config = {"browser": {"camofox": {"managed_persistence": False}}}
        with patch("tools.browser_camofox.load_config", return_value=config):
            assert _managed_persistence_enabled() is False

    def test_enabled_via_config_yaml(self):
        config = {"browser": {"camofox": {"managed_persistence": True}}}
        with patch("tools.browser_camofox.load_config", return_value=config):
            assert _managed_persistence_enabled() is True

    def test_disabled_when_key_missing(self):
        config = {"browser": {}}
        with patch("tools.browser_camofox.load_config", return_value=config):
            assert _managed_persistence_enabled() is False

    def test_disabled_on_config_load_error(self):
        with patch("tools.browser_camofox.load_config", side_effect=Exception("fail")):
            assert _managed_persistence_enabled() is False


class TestEphemeralMode:
    """Default behavior: random userId, no persistence."""

    def test_session_gets_random_user_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        session = _get_session("task-1")
        assert session["user_id"].startswith("hermes_")
        assert session["managed"] is False

    def test_different_tasks_get_different_user_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        s1 = _get_session("task-1")
        s2 = _get_session("task-2")
        assert s1["user_id"] != s2["user_id"]

    def test_session_reuse_within_same_task(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        s1 = _get_session("task-1")
        s2 = _get_session("task-1")
        assert s1 is s2


class TestManagedPersistenceMode:
    """With managed_persistence: stable userId derived from Hermes profile."""

    def test_session_gets_stable_user_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        with _enable_persistence():
            session = _get_session("task-1")
            expected = get_camofox_identity("task-1")
            assert session["user_id"] == expected["user_id"]
            assert session["session_key"] == expected["session_key"]
            assert session["managed"] is True

    def test_same_user_id_after_session_drop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        with _enable_persistence():
            s1 = _get_session("task-1")
            uid1 = s1["user_id"]
            _drop_session("task-1")
            s2 = _get_session("task-1")
            assert s2["user_id"] == uid1

    def test_same_user_id_across_tasks(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        with _enable_persistence():
            s1 = _get_session("task-a")
            s2 = _get_session("task-b")
            # Same profile = same userId, different session keys
            assert s1["user_id"] == s2["user_id"]
            assert s1["session_key"] != s2["session_key"]

    def test_different_profiles_get_different_user_ids(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        with _enable_persistence():
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
            s1 = _get_session("task-1")
            uid_a = s1["user_id"]
            _drop_session("task-1")

            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
            s2 = _get_session("task-1")
            assert s2["user_id"] != uid_a

    def test_navigate_uses_stable_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        requests_seen = []

        def _capture_post(url, json=None, timeout=None):
            requests_seen.append(json)
            return _mock_response(
                json_data={"tabId": "tab-1", "url": "https://example.com"}
            )

        with _enable_persistence(), \
             patch("tools.browser_camofox.requests.post", side_effect=_capture_post):
            result = json.loads(camofox_navigate("https://example.com", task_id="task-1"))

        assert result["success"] is True
        expected = get_camofox_identity("task-1")
        assert requests_seen[0]["userId"] == expected["user_id"]

    def test_navigate_reuses_identity_after_close(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        requests_seen = []

        def _capture_post(url, json=None, timeout=None):
            requests_seen.append(json)
            return _mock_response(
                json_data={"tabId": f"tab-{len(requests_seen)}", "url": "https://example.com"}
            )

        with (
            _enable_persistence(),
            patch("tools.browser_camofox.requests.post", side_effect=_capture_post),
            patch("tools.browser_camofox.requests.delete", return_value=_mock_response()),
        ):
            first = json.loads(camofox_navigate("https://example.com", task_id="task-1"))
            camofox_close("task-1")
            second = json.loads(camofox_navigate("https://example.com", task_id="task-1"))

        assert first["success"] is True
        assert second["success"] is True
        tab_requests = [req for req in requests_seen if "userId" in req]
        assert len(tab_requests) == 2
        assert tab_requests[0]["userId"] == tab_requests[1]["userId"]


class TestVncUrlDiscovery:
    """VNC URL is derived from the Camofox health endpoint."""

    def test_vnc_url_from_health_port(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://myhost:9377")
        health_resp = _mock_response(json_data={"ok": True, "vncPort": 6080})
        with patch("tools.browser_camofox.requests.get", return_value=health_resp):
            assert check_camofox_available() is True
        assert get_vnc_url() == "http://myhost:6080"

    def test_vnc_url_none_when_headless(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        health_resp = _mock_response(json_data={"ok": True})
        with patch("tools.browser_camofox.requests.get", return_value=health_resp):
            check_camofox_available()
        assert get_vnc_url() is None

    def test_vnc_url_rejects_invalid_port(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        health_resp = _mock_response(json_data={"ok": True, "vncPort": "bad"})
        with patch("tools.browser_camofox.requests.get", return_value=health_resp):
            check_camofox_available()
        assert get_vnc_url() is None

    def test_vnc_url_only_probed_once(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        health_resp = _mock_response(json_data={"ok": True, "vncPort": 6080})
        with patch("tools.browser_camofox.requests.get", return_value=health_resp) as mock_get:
            check_camofox_available()
            check_camofox_available()
        # Second call still hits /health for availability but doesn't re-parse vncPort
        assert get_vnc_url() == "http://localhost:6080"

    def test_navigate_includes_vnc_hint(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        import tools.browser_camofox as mod
        mod._vnc_url = "http://localhost:6080"
        mod._vnc_url_checked = True

        with patch("tools.browser_camofox.requests.post", return_value=_mock_response(
            json_data={"tabId": "t1", "url": "https://example.com"}
        )):
            result = json.loads(camofox_navigate("https://example.com", task_id="vnc-test"))

        assert result["vnc_url"] == "http://localhost:6080"
        assert "vnc_hint" in result

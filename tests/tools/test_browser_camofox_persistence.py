"""Persistence tests for the Camofox browser backend.

Tests that managed persistence uses stable identity while default mode
uses random identity. Camofox automatically maps each userId to a
dedicated persistent Firefox profile on the server side.
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
    camofox_soft_cleanup,
    check_camofox_available,
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


    def test_navigate_reuses_identity_after_close(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        requests_seen = []

        def _capture_post(url, json=None, timeout=None, headers=None):
            requests_seen.append((url, json))
            if url.endswith("/tabs"):
                tab_count = sum(
                    1 for request_url, _ in requests_seen if request_url.endswith("/tabs")
                )
                return _mock_response(
                    json_data={
                        "tabId": f"tab-{tab_count}",
                        "url": "https://example.com",
                    }
                )
            return _mock_response(
                json_data={
                    "url": (json or {}).get("url", ""),
                    "title": "live",
                }
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
        tab_requests = [
            body for request_url, body in requests_seen if request_url.endswith("/tabs")
        ]
        navigate_requests = [
            (request_url, body)
            for request_url, body in requests_seen
            if request_url.endswith("/navigate")
        ]
        assert len(tab_requests) == 2
        assert len(navigate_requests) == 2
        assert tab_requests[0]["userId"] == tab_requests[1]["userId"]
        assert [request_url for request_url, _ in navigate_requests] == [
            "http://localhost:9377/tabs/tab-1/navigate",
            "http://localhost:9377/tabs/tab-2/navigate",
        ]
        assert all(
            body["userId"] == tab_requests[0]["userId"]
            for _, body in navigate_requests
        )


class TestConfiguredCamofoxIdentity:
    """Externally managed Camofox sessions can provide their own identity."""

    def test_multiplex_scope_identity_wins_over_process_env_and_config(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "https://default.example")
        monkeypatch.setenv("CAMOFOX_USER_ID", "default-profile-user")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "default-profile-session")
        config = {
            "browser": {
                "camofox": {
                    "user_id": "secondary-config-user",
                    "session_key": "secondary-config-session",
                }
            }
        }
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope(
            {
                "CAMOFOX_URL": "https://secondary.example",
                "CAMOFOX_USER_ID": "secondary-scope-user",
                "CAMOFOX_SESSION_KEY": "secondary-scope-session",
            }
        )
        try:
            with (
                patch("tools.browser_camofox.load_config", return_value=config),
                patch(
                    "tools.browser_camofox.requests.post",
                    side_effect=[
                        _mock_response(json_data={"tabId": "scoped-tab"}),
                        _mock_response(
                            json_data={
                                "url": "https://example.com",
                                "title": "live",
                            }
                        ),
                    ],
                ) as mock_post,
            ):
                result = json.loads(
                    camofox_navigate("https://example.com", task_id="scoped-precedence")
                )
                create_call, navigate_call = mock_post.call_args_list
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

        assert result["success"] is True
        assert create_call.args[0] == "https://secondary.example/tabs"
        assert create_call.kwargs["json"]["userId"] == "secondary-scope-user"
        assert create_call.kwargs["json"]["listItemId"] == "secondary-scope-session"
        assert navigate_call.args[0] == (
            "https://secondary.example/tabs/scoped-tab/navigate"
        )
        assert navigate_call.kwargs["json"] == {
            "userId": "secondary-scope-user",
            "url": "https://example.com",
        }

    def test_multiplex_scope_miss_uses_profile_config_not_process_env(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_USER_ID", "default-profile-user")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "default-profile-session")
        config = {
            "browser": {
                "camofox": {
                    "user_id": "secondary-config-user",
                    "session_key": "secondary-config-session",
                }
            }
        }
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            with patch("tools.browser_camofox.load_config", return_value=config):
                session = _get_session("config-fallback")
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

        assert session["user_id"] == "secondary-config-user"
        assert session["session_key"] == "secondary-config-session"

    def test_multiplex_scope_miss_without_config_ignores_process_identity(
        self, tmp_path, monkeypatch
    ):
        from agent import secret_scope

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_USER_ID", "default-profile-user")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "default-profile-session")
        secret_scope.set_multiplex_active(True)
        token = secret_scope.set_secret_scope({})
        try:
            with patch("tools.browser_camofox.load_config", return_value={}):
                session = _get_session("fail-closed")
        finally:
            secret_scope.reset_secret_scope(token)
            secret_scope.set_multiplex_active(False)

        assert session["user_id"].startswith("hermes_")
        assert session["user_id"] != "default-profile-user"
        assert session["session_key"] == "task_fail-closed"
        assert session["managed"] is False

    def test_env_identity_overrides_default_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_USER_ID", "shared-camofox")
        monkeypatch.setenv("CAMOFOX_SESSION_KEY", "visible-tab")
        monkeypatch.setenv("CAMOFOX_ADOPT_EXISTING_TAB", "true")

        with patch("tools.browser_camofox._get", return_value={"tabs": []}) as mock_get:
            session = _get_session("task-1")

        assert session["user_id"] == "shared-camofox"
        assert session["session_key"] == "visible-tab"
        assert session["managed"] is True
        assert session["adopt_existing_tab"] is True
        mock_get.assert_called_once_with(
            "/tabs",
            params={"userId": "shared-camofox"},
            timeout=5,
        )


    def test_soft_cleanup_preserves_externally_managed_session(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")
        monkeypatch.setenv("CAMOFOX_USER_ID", "shared-camofox")

        with patch("tools.browser_camofox._get", return_value={"tabs": []}):
            _get_session("task-1")
        result = camofox_soft_cleanup("task-1")

        assert result is True
        import tools.browser_camofox as mod
        with mod._sessions_lock:
            assert "task-1" not in mod._sessions


class TestVncUrlDiscovery:
    """VNC URL is derived from the Camofox health endpoint."""

    def test_vnc_url_from_health_port(self, monkeypatch):
        monkeypatch.setenv("CAMOFOX_URL", "http://myhost:9377")
        health_resp = _mock_response(json_data={"ok": True, "vncPort": 6080})
        with patch("tools.browser_camofox.requests.get", return_value=health_resp):
            assert check_camofox_available() is True
        assert get_vnc_url() == "http://myhost:6080"


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


class TestCamofoxSoftCleanup:
    """camofox_soft_cleanup drops local state only when managed persistence is on."""

    def test_returns_true_and_drops_session_when_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        with _enable_persistence():
            _get_session("task-1")
            result = camofox_soft_cleanup("task-1")

        assert result is True
        # Session should have been dropped from in-memory store
        import tools.browser_camofox as mod
        with mod._sessions_lock:
            assert "task-1" not in mod._sessions


    def test_does_not_call_server_delete(self, tmp_path, monkeypatch):
        """Soft cleanup must never hit the Camofox /sessions DELETE endpoint."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("CAMOFOX_URL", "http://localhost:9377")

        with (
            _enable_persistence(),
            patch("tools.browser_camofox.requests.delete") as mock_delete,
        ):
            _get_session("task-1")
            camofox_soft_cleanup("task-1")

        mock_delete.assert_not_called()

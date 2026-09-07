"""Tests for Temporary Chats v2 browser isolation and owned cleanup.

Verifies:
1. Temporary sessions refuse CDP overrides, real-profile attach, and browser extension control.
2. Camofox backend creates random ephemeral identity, ignores managed persistence, and forces full session close.
3. Raw browser_cdp tool fails closed in temporary mode.
4. Ephemeral local sessions use private socket directories with 0o700 permissions and clean up completely.
5. Normal concurrent sessions are not interfered with when temporary sessions are created or cleaned.
"""

import os
import shutil
import tempfile
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agent.session_policy import mark_session_ephemeral, unmark_session_ephemeral
from tools.browser_camofox import (
    _get_session as _get_camofox_session,
    _is_task_ephemeral,
    camofox_soft_cleanup,
)
from tools.browser_cdp_tool import browser_cdp
from tools.browser_extension_router import (
    extension_controller_available,
    route_browser_tool,
)
from tools.browser_tool import _active_sessions, _session_last_activity
from tools.browser_tool_lifecycle import cleanup_browser
from tools.browser_tool_session import (
    _create_local_session,
    _create_session_for_key,
    _prepare_session_socket_dir,
)


@pytest.fixture(autouse=True)
def _isolate_test_state(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and temporary directory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir(parents=True, exist_ok=True)
    yield
    _active_sessions.clear()
    _session_last_activity.clear()


class TestBrowserCdpAndProfileIsolation:
    def test_temporary_session_refuses_cdp_override(self):
        temp_task = "temp-session-cdp-test"
        mark_session_ephemeral(temp_task)
        try:
            with patch("tools.browser_tool_cdp._get_cdp_override", return_value="ws://127.0.0.1:9222/devtools/browser/abc"):
                with pytest.raises(RuntimeError, match="CDP override cannot be used in a temporary chat"):
                    _create_session_for_key(temp_task, force_local=False)
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_temporary_session_refuses_real_profile(self):
        temp_task = "temp-session-real-profile-test"
        mark_session_ephemeral(temp_task)
        try:
            with patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
                 patch("tools.browser_tool_cloud._use_real_profile", return_value=True):
                with pytest.raises(RuntimeError, match="Real-profile browsing cannot be used in a temporary chat"):
                    _create_session_for_key(temp_task, force_local=False)
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_temporary_session_creates_isolated_local_session(self):
        temp_task = "temp-session-local-test"
        mark_session_ephemeral(temp_task)
        try:
            with patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
                 patch("tools.browser_tool_cloud._use_real_profile", return_value=False):
                session_info = _create_session_for_key(temp_task, force_local=False)
                assert session_info["features"]["local"] is True
                assert session_info["features"]["ephemeral"] is True
                assert session_info["session_name"].startswith("h_")
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_normal_session_allows_normal_local_session(self):
        normal_task = "normal-session-local-test"
        with patch("tools.browser_tool_cdp._get_cdp_override", return_value=""), \
             patch("tools.browser_tool_cloud._use_real_profile", return_value=False):
            session_info = _create_session_for_key(normal_task, force_local=False)
            assert session_info["features"]["local"] is True
            assert session_info["features"]["ephemeral"] is False


class TestCamofoxTemporaryIsolation:
    def test_is_task_ephemeral_detection(self):
        temp_task = "temp-camofox-test"
        normal_task = "normal-camofox-test"
        mark_session_ephemeral(temp_task)
        try:
            assert _is_task_ephemeral(temp_task) is True
            assert _is_task_ephemeral(f"{temp_task}_local") is True
            assert _is_task_ephemeral(normal_task) is False
            assert _is_task_ephemeral(None) is False
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_temporary_mode_refuses_camofox_identity_override(self):
        temp_task = "temp-camofox-override-test"
        mark_session_ephemeral(temp_task)
        try:
            with patch("tools.browser_camofox._camofox_identity_override", return_value={"user_id": "custom", "session_key": "k"}):
                with pytest.raises(RuntimeError, match="Persistent Camofox identity cannot be used in a temporary chat"):
                    _get_camofox_session(temp_task)
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_temporary_mode_bypasses_managed_persistence(self):
        temp_task = "temp-camofox-managed-test"
        normal_task = "normal-camofox-managed-test"
        mark_session_ephemeral(temp_task)
        try:
            managed_config = {"browser": {"camofox": {"managed_persistence": True}}}
            with patch("tools.browser_camofox.load_config", return_value=managed_config):
                temp_session = _get_camofox_session(temp_task)
                assert temp_session["user_id"].startswith("hermes_temp_")
                assert temp_session["managed"] is False
                assert temp_session["ephemeral"] is True

                normal_session = _get_camofox_session(normal_task)
                assert not normal_session["user_id"].startswith("hermes_temp_")
                assert normal_session["managed"] is True
                assert normal_session["ephemeral"] is False
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_camofox_soft_cleanup_forces_full_close_for_ephemeral(self):
        temp_task = "temp-camofox-cleanup-test"
        normal_task = "normal-camofox-cleanup-test"
        mark_session_ephemeral(temp_task)
        try:
            managed_config = {"browser": {"camofox": {"managed_persistence": True}}}
            with patch("tools.browser_camofox.load_config", return_value=managed_config):
                # Normal managed session allows soft cleanup
                assert camofox_soft_cleanup(normal_task) is True
                # Ephemeral session refuses soft cleanup (forces camofox_close)
                assert camofox_soft_cleanup(temp_task) is False
        finally:
            unmark_session_ephemeral(temp_task, force=True)


class TestBrowserCdpToolGuard:
    def test_browser_cdp_fails_in_temporary_session(self):
        temp_task = "temp-browser-cdp-tool-test"
        mark_session_ephemeral(temp_task)
        try:
            res = browser_cdp("Target.getTargets", task_id=temp_task)
            assert "browser_cdp cannot be used in a temporary chat" in res
            assert '"error"' in res or '"success": false' in res
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_browser_cdp_allowed_in_normal_session(self):
        normal_task = "normal-browser-cdp-tool-test"
        with patch("tools.browser_cdp_tool._WS_AVAILABLE", False):
            res = browser_cdp("Target.getTargets", task_id=normal_task)
            # Fails on websockets missing rather than temporary mode rejection
            assert "websockets" in res


class TestBrowserExtensionRouterGuard:
    def test_extension_controller_unavailable_in_temporary_session(self):
        temp_task = "temp-extension-test"
        mark_session_ephemeral(temp_task)
        try:
            with patch("gateway.browser_control_broker.browser_control_enabled", return_value=True), \
                 patch("tools.browser_extension_router._bound_identity", return_value=(temp_task, "user-1", "chrome-extension")):
                assert extension_controller_available("navigate") is False
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_route_browser_tool_refuses_extension_in_temporary_session(self):
        temp_task = "temp-extension-route-test"
        mark_session_ephemeral(temp_task)
        try:
            from gateway.browser_control_broker import ControllerUnavailable

            fallback = MagicMock(return_value="fallback_executed")
            broker = MagicMock()

            with pytest.raises(ControllerUnavailable, match="Browser extension attach cannot be used in a temporary chat"):
                route_browser_tool(
                    "navigate",
                    {"url": "https://example.com"},
                    fallback=fallback,
                    broker=broker,
                    enabled=True,
                    session_id=temp_task,
                    principal_id="principal-1",
                    transport_family="chrome-extension",
                )
            fallback.assert_not_called()
        finally:
            unmark_session_ephemeral(temp_task, force=True)


class TestSocketDirAndCleanup:
    def test_socket_dir_permissions_and_cleanup(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        session_name = f"h_{uuid.uuid4().hex[:10]}"
        socket_dir = _prepare_session_socket_dir(session_name)

        assert os.path.exists(socket_dir)
        # Verify 0o700 permission on POSIX systems
        if os.name != "nt":
            mode = os.stat(socket_dir).st_mode & 0o777
            assert mode == 0o700

        temp_task = "temp-cleanup-task"
        mark_session_ephemeral(temp_task)
        try:
            _active_sessions[temp_task] = {
                "session_name": session_name,
                "bb_session_id": None,
                "features": {"local": True, "ephemeral": True},
            }
            _session_last_activity[temp_task] = 100.0

            with patch("tools.browser_tool_session._run_browser_command", return_value={"success": True}):
                cleanup_browser(temp_task)

            assert temp_task not in _active_sessions
            assert temp_task not in _session_last_activity
            assert not os.path.exists(socket_dir)
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_concurrent_normal_session_noninterference(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        temp_task = "temp-concurrent-task"
        normal_task = "normal-concurrent-task"
        mark_session_ephemeral(temp_task)

        try:
            temp_sess_name = f"h_temp_{uuid.uuid4().hex[:8]}"
            normal_sess_name = f"h_norm_{uuid.uuid4().hex[:8]}"

            temp_dir = _prepare_session_socket_dir(temp_sess_name)
            norm_dir = _prepare_session_socket_dir(normal_sess_name)

            _active_sessions[temp_task] = {
                "session_name": temp_sess_name,
                "bb_session_id": None,
                "features": {"local": True, "ephemeral": True},
            }
            _active_sessions[normal_task] = {
                "session_name": normal_sess_name,
                "bb_session_id": None,
                "features": {"local": True, "ephemeral": False},
            }

            with patch("tools.browser_tool_session._run_browser_command", return_value={"success": True}):
                cleanup_browser(temp_task)

            # Temporary session cleaned up
            assert temp_task not in _active_sessions
            assert not os.path.exists(temp_dir)

            # Normal session remains untouched
            assert normal_task in _active_sessions
            assert _active_sessions[normal_task]["session_name"] == normal_sess_name
            assert os.path.exists(norm_dir)

            # Clean up normal session
            with patch("tools.browser_tool_session._run_browser_command", return_value={"success": True}):
                cleanup_browser(normal_task)
            assert not os.path.exists(norm_dir)
        finally:
            unmark_session_ephemeral(temp_task, force=True)

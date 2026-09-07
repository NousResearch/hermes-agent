"""Tests for Temporary Chats v2 browser isolation and owned cleanup.

Verifies:
1. Temporary sessions refuse CDP overrides, real-profile attach, and browser extension control.
2. Camofox backend creates random ephemeral identity, ignores managed persistence, and forces full session close.
3. Raw browser_cdp tool fails closed in temporary mode.
4. Ephemeral local sessions use private socket directories with 0o700 permissions and clean up completely.
5. Normal concurrent sessions are not interfered with when temporary sessions are created or cleaned.
"""

import os
from pathlib import Path
import shutil
import tempfile
import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agent.session_policy import is_session_ephemeral, mark_session_ephemeral, unmark_session_ephemeral
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


class TestBrowserUseCliIsolation:
    """Verify Browser-use CLI respects temporary chat isolation and fail-closed policies."""

    def test_browser_exec_refuses_local_in_ephemeral_task(self):
        from tools.browser_use_cli import browser_exec
        temp_task = "temp-bu-local-task"
        mark_session_ephemeral(temp_task)
        try:
            res = browser_exec("print('hello')", task_id=temp_task, local=True)
            res_dict = res.get("content", [{}])[0].get("text", "") if isinstance(res, dict) else str(res)
            assert "Real-profile browsing cannot be used in a temporary chat" in res_dict
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_browser_exec_refuses_cdp_override_in_ephemeral_task(self):
        from tools.browser_use_cli import browser_exec
        temp_task = "temp-bu-cdp-task"
        mark_session_ephemeral(temp_task)
        try:
            with patch("tools.browser_tool_cdp._get_cdp_override", return_value="ws://127.0.0.1:9222"):
                res = browser_exec("print('hello')", task_id=temp_task, local=False)
                res_dict = res.get("content", [{}])[0].get("text", "") if isinstance(res, dict) else str(res)
                assert "CDP override cannot be used in a temporary chat" in res_dict
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_browser_use_backend_cache_key_isolated_for_ephemeral_named_session(self):
        from tools.browser_use_cli import _backend_cache_key
        temp_task = "temp-bu-named-task"
        normal_task = "normal-bu-named-task"
        mark_session_ephemeral(temp_task)
        try:
            temp_key = _backend_cache_key(temp_task, session_name="my_session")
            norm_key = _backend_cache_key(normal_task, session_name="my_session")
            assert temp_key == f"bu-named-{temp_task}-my_session"
            assert norm_key == "bu-named-my_session"
            assert is_session_ephemeral(temp_key)
            assert not is_session_ephemeral(norm_key)
        finally:
            unmark_session_ephemeral(temp_task, force=True)

    def test_browser_use_workspace_isolated_and_cleaned(self, tmp_path, monkeypatch):
        from tools.browser_use_cli import _workspace_dir
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        temp_task = "temp-bu-workspace-task"
        mark_session_ephemeral(temp_task)
        try:
            ws_path = _workspace_dir(temp_task)
            assert ws_path is not None
            assert os.path.exists(ws_path)
            assert "hermes-temp-bu-workspace" in ws_path
            # Write a dummy file to the workspace
            test_file = Path(ws_path) / "scratch.py"
            test_file.write_text("print('test')")

            # Cleanup
            cleanup_browser(temp_task)
            assert not os.path.exists(ws_path), "Ephemeral browser-use workspace was not cleaned up"
        finally:
            unmark_session_ephemeral(temp_task, force=True)


class TestStaleRootReapSymlinkProtection:
    """Verify orphan reaper handles dead/live owners and never follows symlinks."""

    def test_reaper_refuses_symlink_traversal(self, tmp_path):
        from tools.browser_tool_lifecycle import _reap_socket_dir
        real_target_dir = tmp_path / "victim_real_dir"
        real_target_dir.mkdir()
        sensitive_file = real_target_dir / "keep_safe.txt"
        sensitive_file.write_text("DO_NOT_DELETE")

        symlink_socket_dir = tmp_path / "agent-browser-h_symlink_attack"
        symlink_socket_dir.symlink_to(real_target_dir)

        # Call _reap_socket_dir on the symlink
        result = _reap_socket_dir(str(symlink_socket_dir), "h_symlink_attack", set())
        assert result is False
        assert sensitive_file.exists(), "Symlink traversal deleted files in the target directory!"
        assert not os.path.islink(symlink_socket_dir), "Symlink was not removed safely"

    def test_reaper_cleans_stale_dead_owner_dir(self, tmp_path):
        from tools.browser_tool_lifecycle import _reap_socket_dir
        stale_dir = tmp_path / "agent-browser-h_stale_dead"
        stale_dir.mkdir()
        pid_file = stale_dir / "h_stale_dead.pid"
        pid_file.write_text("99999999")  # Non-existent PID

        result = _reap_socket_dir(str(stale_dir), "h_stale_dead", set())
        assert not stale_dir.exists(), "Stale socket dir with dead PID was not cleaned up"


class TestRealBrowserIsolationE2E:
    """Real browser E2E test against loopback HTTP server using agent-browser."""

    @pytest.fixture
    def loopback_server(self):
        import http.server
        import socketserver

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = """<!DOCTYPE html><html><head><title>Isolation Lab</title></head>
                <body><div id="content">Isolation Page</div></body></html>"""
                self.wfile.write(html.encode("utf-8"))

            def log_message(self, *args):
                pass

        server = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{port}"
        server.shutdown()
        server.server_close()

    @pytest.mark.live_system_guard_bypass
    def test_real_browser_cookie_storage_isolation_e2e(self, loopback_server, tmp_path, monkeypatch):
        """Verify real agent-browser isolates cookies and local storage between temporary and normal sessions."""
        from tools.browser_tool_session import _run_browser_command, _get_session_info
        from tools.browser_tool_install import _find_agent_browser

        if not _find_agent_browser():
            pytest.skip("agent-browser binary not available for real browser E2E")

        monkeypatch.setenv("TMPDIR", str(tmp_path))
        temp_task = f"temp-e2e-{uuid.uuid4().hex[:8]}"
        norm_task = f"norm-e2e-{uuid.uuid4().hex[:8]}"

        mark_session_ephemeral(temp_task)
        try:
            # 1. Open loopback in temporary session and set state
            open_res = _run_browser_command(temp_task, "open", [loopback_server], timeout=25)
            assert open_res.get("success"), f"Failed to open browser: {open_res}"

            temp_info = _get_session_info(temp_task)
            from tools.browser_tool import _socket_safe_tmpdir
            temp_socket_dir = os.path.join(_socket_safe_tmpdir(), f"agent-browser-{temp_info['session_name']}")
            assert os.path.exists(temp_socket_dir), "Temporary socket dir was not created"

            # Set cookie and localStorage in temporary session
            eval_res = _run_browser_command(
                temp_task, "eval",
                ["document.cookie = 'hermes_temp=sentinel_123; path=/'; localStorage.setItem('temp_key', 'temp_val');"],
                timeout=15,
            )
            assert eval_res.get("success"), f"Failed to eval in temp browser: {eval_res}"

            # Verify values present in temporary session
            read_res = _run_browser_command(
                temp_task, "eval",
                ["JSON.stringify({cookie: document.cookie, local: localStorage.getItem('temp_key')})"],
                timeout=15,
            )
            assert "sentinel_123" in str(read_res.get("data", "")), "Cookie was not set in temporary browser"

            # 2. Close temporary session
            cleanup_browser(temp_task)
            assert not os.path.exists(temp_socket_dir), "Temporary socket dir was not deleted on cleanup"

            # 3. Open subsequent fresh temporary session to same page and verify clean slate
            next_temp_task = f"temp-e2e-next-{uuid.uuid4().hex[:8]}"
            mark_session_ephemeral(next_temp_task)
            try:
                open_res2 = _run_browser_command(next_temp_task, "open", [loopback_server], timeout=25)
                assert open_res2.get("success")

                read_res2 = _run_browser_command(
                    next_temp_task, "eval",
                    ["JSON.stringify({cookie: document.cookie, local: localStorage.getItem('temp_key')})"],
                    timeout=15,
                )
                data2 = str(read_res2.get("data", ""))
                assert "sentinel_123" not in data2, f"Prior temporary cookie leaked into new temporary session: {data2}"
                assert "temp_val" not in data2, f"Prior temporary localStorage leaked into new temporary session: {data2}"
            finally:
                cleanup_browser(next_temp_task)
                unmark_session_ephemeral(next_temp_task, force=True)

            # 4. Open normal session to same page and verify clean slate
            open_norm = _run_browser_command(norm_task, "open", [loopback_server], timeout=25)
            assert open_norm.get("success")
            read_norm = _run_browser_command(
                norm_task, "eval",
                ["JSON.stringify({cookie: document.cookie, local: localStorage.getItem('temp_key')})"],
                timeout=15,
            )
            norm_data = str(read_norm.get("data", ""))
            assert "sentinel_123" not in norm_data, f"Temporary cookie leaked into normal session: {norm_data}"
            assert "temp_val" not in norm_data, f"Temporary localStorage leaked into normal session: {norm_data}"
        finally:
            cleanup_browser(norm_task)
            cleanup_browser(temp_task)
            unmark_session_ephemeral(temp_task, force=True)


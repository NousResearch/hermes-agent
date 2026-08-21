"""Behavior contracts for cua-driver 0.10 permission-mode integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_computer_use_state():
    from tools.computer_use.tool import reset_backend_for_tests

    reset_backend_for_tests()
    yield
    reset_backend_for_tests()


def test_normal_hermes_session_maps_to_standard_mode():
    from tools.computer_use import tool as computer_use

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=False,
    ):
        assert computer_use._cua_permission_mode("session-a") == "standard"


def test_any_explicit_hermes_bypass_does_not_escalate_driver_mode():
    from tools.computer_use import tool as computer_use

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,
    ):
        # Approval bypass (yolo) should NOT escalate the driver mode.
        # The configured permission_mode ceiling must be preserved.
        assert computer_use._cua_permission_mode("session-a") == "standard"


def test_gateway_session_key_yolo_does_not_escalate_driver_mode():
    """Gateway /yolo keys bypass off the gateway session_key contextvar,
    not the DB session_id the tool path passes. Mode resolution must consult
    both namespaces or /yolo is silently dead on messaging platforms.
    But the approval bypass should NOT escalate the driver mode."""
    from tools import approval
    from tools.computer_use import tool as computer_use

    gateway_key = "agent:main:telegram:private:12345"
    token = approval.set_current_session_key(gateway_key)
    try:
        approval.enable_session_yolo(gateway_key)
        # Tool dispatch passes the (different) DB session id.
        # Approval bypass should NOT escalate driver mode.
        assert computer_use._cua_permission_mode("db-sid-xyz") == "standard"
        approval.disable_session_yolo(gateway_key)
        assert computer_use._cua_permission_mode("db-sid-xyz") == "standard"
    finally:
        approval.disable_session_yolo(gateway_key)
        try:
            approval.reset_current_session_key(token)
        except Exception:
            approval.set_current_session_key("")


def test_config_mode_change_replaces_only_that_sessions_backend():
    from tools.computer_use import tool as computer_use

    created = []

    class _Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            self.stopped = False
            created.append(self)

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    # Simulate config mode change by patching _cua_configured_permission_mode
    config_mode = "standard"
    with patch(
        "tools.computer_use.cua_backend._cua_configured_permission_mode",
        side_effect=lambda: config_mode,
    ), patch(
        "tools.computer_use.cua_backend.CuaDriverBackend", _Backend
    ):
        standard = computer_use._get_backend("session-a")
        other = computer_use._get_backend("session-b")
        config_mode = "bounded"
        bounded = computer_use._get_backend("session-a")

    assert getattr(standard, "permission_mode") == "standard"
    assert getattr(standard, "stopped") is True
    assert getattr(bounded, "permission_mode") == "bounded"
    assert bounded is not standard
    assert getattr(other, "permission_mode") == "standard"
    assert getattr(other, "stopped") is False


def test_config_mode_change_is_rechecked_after_stale_backend_stops():
    from tools.computer_use import tool as computer_use

    config_mode = "standard"
    created = []

    class _Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            created.append(self)

        def start(self):
            pass

        def stop(self):
            nonlocal config_mode
            config_mode = "standard"

    with patch(
        "tools.computer_use.cua_backend._cua_configured_permission_mode",
        side_effect=lambda: config_mode,
    ), patch("tools.computer_use.cua_backend.CuaDriverBackend", _Backend):
        original = computer_use._get_backend("session-a")
        config_mode = "bounded"
        replacement = computer_use._get_backend("session-a")

    assert getattr(original, "permission_mode") == "standard"
    assert getattr(replacement, "permission_mode") == "standard"
    assert replacement is not original
    assert [backend.permission_mode for backend in created] == [
        "standard",
        "standard",
    ]


def test_release_seam_stops_backend_and_clears_session_state():
    from tools.computer_use import tool as computer_use

    backend = Mock()
    computer_use._backends["session-a"] = backend
    computer_use._backend_call_locks["session-a"] = computer_use.threading.RLock()
    computer_use._backend_permission_modes["session-a"] = "unrestricted"
    computer_use._session_auto_approve["session-a"] = True
    computer_use._always_allow["session-a"] = {("click", "background")}

    assert computer_use.release_computer_use_session("session-a") is True
    assert computer_use.release_computer_use_session("session-a") is False
    backend.stop.assert_called_once_with()
    assert "session-a" not in computer_use._backend_permission_modes
    assert "session-a" not in computer_use._session_auto_approve
    assert "session-a" not in computer_use._always_allow


def test_yolo_toggle_immediately_releases_mode_dependent_backend():
    from tools import approval

    with patch("tools.computer_use.release_computer_use_session") as release:
        approval.enable_session_yolo("session-a")
        approval.disable_session_yolo("session-a")

    assert release.call_args_list == [
        (('session-a',), {}),
        (('session-a',), {}),
    ]


def test_unrestricted_embedded_daemon_uses_private_socket_and_two_part_ack():
    from tools.computer_use import cua_backend

    process = Mock()
    process.poll.return_value = None
    process.stderr = []
    process.wait.return_value = 0
    status = SimpleNamespace(returncode=0, stdout="running", stderr="")
    stopped = SimpleNamespace(returncode=0, stdout="", stderr="")

    daemon = cua_backend._EmbeddedCuaDaemon("cua-driver", "unrestricted")
    with patch.object(
        cua_backend,
        "_resolve_mcp_invocation",
        return_value=("/opt/cua-driver", ["mcp"]),
    ), patch.object(cua_backend.subprocess, "Popen", return_value=process) as popen, patch.object(
        cua_backend.subprocess, "run", side_effect=[status, stopped]
    ):
        daemon.start()
        command = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        proxy_command, proxy_args = daemon.proxy_invocation()
        daemon.stop()

    assert command[:2] == ["/opt/cua-driver", "serve"]
    assert "--embedded" in command
    assert command[command.index("--permission-mode") + 1] == "unrestricted"
    assert "--dangerously-bypass-approvals" in command
    assert env["CUA_DRIVER_PERMISSION_MODE"] == "unrestricted"
    assert env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] == "1"
    assert proxy_command == "/opt/cua-driver"
    assert proxy_args == ["mcp", "--embedded", "--socket", daemon.socket_path]


def test_standard_backend_does_not_spawn_an_embedded_daemon():
    from tools.computer_use.cua_backend import CuaDriverBackend

    standard = CuaDriverBackend(permission_mode="standard")
    unrestricted = CuaDriverBackend(permission_mode="unrestricted")

    assert standard._embedded_daemon is None
    assert unrestricted._embedded_daemon is not None


def test_standard_existing_profile_grant_owns_private_macos_runtime():
    from tools.computer_use.cua_backend import _standard_runtime_launch_args

    args, socket_path = _standard_runtime_launch_args(
        ["mcp"],
        grant_existing_profile=True,
        platform="darwin",
        socket_path="/tmp/hermes-cua-test.sock",
    )

    assert args == [
        "mcp",
        "--grant",
        "existing-profile",
        "--socket",
        "/tmp/hermes-cua-test.sock",
    ]
    assert socket_path == "/tmp/hermes-cua-test.sock"


def test_standard_existing_profile_grant_stays_in_process_off_macos():
    from tools.computer_use.cua_backend import _standard_runtime_launch_args

    args, socket_path = _standard_runtime_launch_args(
        ["mcp"], grant_existing_profile=True, platform="linux"
    )

    assert args == ["mcp", "--grant", "existing-profile"]
    assert socket_path is None


def test_transport_reset_invalidates_native_and_browser_capabilities():
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend(permission_mode="standard")
    backend._active_pid = 10
    backend._active_window_id = 20
    backend._snapshot_tokens = {1: "old-token"}
    backend._typed_browser.state.pid = 10
    backend._typed_browser.state.window_id = 20
    backend._typed_browser.state.target_id = "old-target"
    backend._typed_browser.state.refs = {"old-ref": {"click"}}

    backend._handle_transport_reset()

    assert backend._active_pid is None
    assert backend._active_window_id is None
    assert backend._snapshot_tokens == {}
    assert backend._typed_browser.state.target_id is None
    assert backend._typed_browser.state.refs == {}


def test_yolo_with_restrictive_permission_mode_preserves_ceiling():
    """Regression test for #87837: yolo active + restrictive permission_mode
    should NOT escalate the daemon to unrestricted. The configured ceiling
    (standard | bounded) must be preserved."""
    from tools.computer_use import tool as computer_use
    from tools.computer_use import cua_backend

    # Test with standard mode
    with patch(
        "tools.computer_use.cua_backend._cua_configured_permission_mode",
        return_value="standard",
    ), patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,  # yolo active
    ):
        assert computer_use._cua_permission_mode("session-yolo-standard") == "standard"

    # Test with bounded mode
    with patch(
        "tools.computer_use.cua_backend._cua_configured_permission_mode",
        return_value="bounded",
    ), patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,  # yolo active
    ):
        assert computer_use._cua_permission_mode("session-yolo-bounded") == "bounded"

    # Verify the backend would be created with the configured mode, not unrestricted
    created_backends = []

    class _TestBackend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            created_backends.append(self)

        def start(self):
            pass

        def stop(self):
            pass

    with patch(
        "tools.computer_use.cua_backend._cua_configured_permission_mode",
        return_value="bounded",
    ), patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,  # yolo active
    ), patch("tools.computer_use.cua_backend.CuaDriverBackend", _TestBackend):
        backend = computer_use._get_backend("session-regression")
        assert backend.permission_mode == "bounded"
        assert backend.permission_mode != "unrestricted"


def test_no_escalation_warning_without_a_bypass(caplog):
    import logging

    from tools.computer_use import tool as computer_use

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=False,
    ):
        with caplog.at_level(logging.WARNING, logger=computer_use.logger.name):
            assert computer_use._cua_permission_mode("session-quiet") == "standard"

    # No escalation warning should exist since escalation no longer happens
    assert not [
        r for r in caplog.records if "escalated the cua-driver" in r.getMessage()
    ]

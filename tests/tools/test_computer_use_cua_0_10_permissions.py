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


def test_any_explicit_hermes_bypass_maps_to_unrestricted_mode():
    from tools.computer_use import tool as computer_use

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,
    ):
        assert computer_use._cua_permission_mode("session-a") == "unrestricted"


def test_gateway_session_key_yolo_maps_to_unrestricted_mode():
    """Gateway /yolo keys bypass off the gateway session_key contextvar,
    not the DB session_id the tool path passes. Mode resolution must consult
    both namespaces or /yolo is silently dead on messaging platforms."""
    from tools import approval
    from tools.computer_use import tool as computer_use

    gateway_key = "agent:main:telegram:private:12345"
    token = approval.set_current_session_key(gateway_key)
    try:
        approval.enable_session_yolo(gateway_key)
        # Tool dispatch passes the (different) DB session id.
        assert computer_use._cua_permission_mode("db-sid-xyz") == "unrestricted"
        approval.disable_session_yolo(gateway_key)
        assert computer_use._cua_permission_mode("db-sid-xyz") == "standard"
    finally:
        approval.disable_session_yolo(gateway_key)
        try:
            approval.reset_current_session_key(token)
        except Exception:
            approval.set_current_session_key("")


def test_mode_change_replaces_only_that_sessions_backend():
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

    yolo = False
    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        side_effect=lambda sid: yolo,
    ), patch(
        "tools.computer_use.cua_backend.CuaDriverBackend", _Backend
    ):
        standard = computer_use._get_backend("session-a")
        other = computer_use._get_backend("session-b")
        yolo = True
        unrestricted = computer_use._get_backend("session-a")

    assert getattr(standard, "permission_mode") == "standard"
    assert getattr(standard, "stopped") is True
    assert getattr(unrestricted, "permission_mode") == "unrestricted"
    assert unrestricted is not standard
    assert getattr(other, "permission_mode") == "standard"
    assert getattr(other, "stopped") is False


def test_mode_change_is_rechecked_after_stale_backend_stops():
    from tools.computer_use import tool as computer_use

    yolo = False
    created = []

    class _Backend:
        def __init__(self, permission_mode="standard"):
            self.permission_mode = permission_mode
            created.append(self)

        def start(self):
            pass

        def stop(self):
            nonlocal yolo
            yolo = False

    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        side_effect=lambda sid: yolo,
    ), patch("tools.computer_use.cua_backend.CuaDriverBackend", _Backend):
        original = computer_use._get_backend("session-a")
        yolo = True
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
    with patch.object(cua_backend.sys, "platform", "linux"), patch.object(
        cua_backend,
        "_resolve_mcp_invocation",
        return_value=("/opt/cua-driver", ["mcp"]),
    ), patch.object(
        # This test pins the socket/ack contract, not overlay policy. Pin the
        # policy off so the environment-dependent auto-detect (headless CI vs
        # Wayland dev box) can't add a `--help` capability-probe subprocess.run
        # call that the fixed two-entry side_effect below doesn't budget for.
        cua_backend, "_cua_no_overlay", return_value=False,
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


# ── the escalation is at least audible ──────────────────────────────────


def test_bypass_escalation_is_warned_once_per_session(caplog):
    """`-z` reads as "don't prompt me" but also drops the driver's ceiling.

    That widening is deliberate and unrestricted is reachable no other way,
    but it is easy to trigger by accident: a script takes -z for quiet output
    and loses its limits as a side effect. It must not be silent.
    """
    import logging

    from tools.computer_use import tool as computer_use

    computer_use._escalation_warned.clear()
    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,
    ):
        with caplog.at_level(logging.WARNING, logger=computer_use.logger.name):
            assert computer_use._cua_permission_mode("session-warn") == "unrestricted"
            assert computer_use._cua_permission_mode("session-warn") == "unrestricted"

    escalation = [
        r for r in caplog.records if "escalated the cua-driver" in r.getMessage()
    ]
    assert len(escalation) == 1, "warning must fire once, not on every dispatch"
    message = escalation[0].getMessage()
    assert "standard" in message
    assert "unrestricted" in message


def test_no_escalation_warning_without_a_bypass(caplog):
    import logging

    from tools.computer_use import tool as computer_use

    computer_use._escalation_warned.clear()
    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=False,
    ):
        with caplog.at_level(logging.WARNING, logger=computer_use.logger.name):
            assert computer_use._cua_permission_mode("session-quiet") == "standard"

    assert not [
        r for r in caplog.records if "escalated the cua-driver" in r.getMessage()
    ]


def test_each_session_is_warned_separately(caplog):
    import logging

    from tools.computer_use import tool as computer_use

    computer_use._escalation_warned.clear()
    with patch(
        "tools.approval.is_approval_bypass_active_for_session",
        return_value=True,
    ):
        with caplog.at_level(logging.WARNING, logger=computer_use.logger.name):
            computer_use._cua_permission_mode("session-one")
            computer_use._cua_permission_mode("session-two")

    escalation = [
        r for r in caplog.records if "escalated the cua-driver" in r.getMessage()
    ]
    assert len(escalation) == 2

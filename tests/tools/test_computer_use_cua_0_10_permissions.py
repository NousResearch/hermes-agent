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

"""Regression tests for the macOS LaunchServices private-daemon launch path.

Symptom: every unrestricted (YOLO) Hermes session re-triggered the macOS Screen
Recording prompt. The private daemon was spawned as a direct child of the Hermes
process (``cua-driver serve --embedded``), so ScreenCaptureKit attributed the
capture to Hermes' own TCC row — whose stored cdhash goes stale on every Hermes
rebuild — instead of to CuaDriver's long-lived grant.

Fix: on macOS, when a ``CuaDriver.app`` bundle is installed, launch the private
daemon through LaunchServices (``/usr/bin/open -n -W -a CuaDriver.app --args
serve ...``) so it is its own responsible process under ``com.trycua.driver``.
Isolation is unchanged: private socket, unrestricted mode, approval bypass, own
lifecycle. Non-macOS platforms and unbundled macOS installs keep the previous
direct ``serve --embedded`` spawn.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

APP = "/Applications/CuaDriver.app"
SERVE_POLICY = [
    "--no-permissions-gate",
    "--permission-mode",
    "unrestricted",
    "--dangerously-bypass-approvals",
]


def _fake_process() -> Mock:
    process = Mock()
    process.poll.return_value = None
    process.stderr = []
    process.wait.return_value = 0
    return process


def _status(returncode: int = 0):
    return SimpleNamespace(returncode=returncode, stdout="running", stderr="")


class _InlineThread:
    """Runs the target synchronously so the stderr tail is drained on start()."""

    def __init__(self, target=None, args=(), **_kwargs):
        self._target = target
        self._args = args

    def start(self):
        if self._target is not None:
            self._target(*self._args)


def _make_daemon(cua_backend, platform: str):
    with patch("tools.computer_use.cua_backend.sys.platform", platform):
        return cua_backend._EmbeddedCuaDaemon("cua-driver", "unrestricted")


def _start(cua_backend, daemon, platform, app, *, runs=None, process=None):
    """Run ``daemon.start()`` with every subprocess boundary stubbed out."""
    process = process or _fake_process()
    with patch("tools.computer_use.cua_backend.sys.platform", platform), patch.object(
        cua_backend, "resolve_cua_driver_app", return_value=app
    ), patch.object(
        cua_backend, "_resolve_mcp_invocation", return_value=("/opt/cua-driver", ["mcp"])
    ), patch(
        "tools.computer_use.cua_backend.threading.Thread", _InlineThread
    ), patch.object(
        cua_backend.subprocess, "Popen", return_value=process
    ) as popen, patch.object(
        cua_backend.subprocess, "run", side_effect=runs or [_status()]
    ) as run:
        daemon.start()
    return popen, run


# ---------------------------------------------------------------------------
# App-bundle resolution
# ---------------------------------------------------------------------------


def test_app_bundle_is_derived_from_the_installed_binarys_bundle(tmp_path):
    """The canonical install symlinks the CLI into CuaDriver.app/Contents/MacOS."""
    from tools.computer_use import cua_backend

    bundle = tmp_path / "CuaDriver.app"
    binary = bundle / "Contents" / "MacOS" / "cua-driver"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    link = tmp_path / "cua-driver"
    link.symlink_to(binary)

    with patch("tools.computer_use.cua_backend.sys.platform", "darwin"):
        assert cua_backend.resolve_cua_driver_app(str(link)) == str(bundle)


def test_app_bundle_falls_back_to_applications_directories(tmp_path):
    from tools.computer_use import cua_backend

    loose = tmp_path / "cua-driver"
    loose.write_text("#!/bin/sh\n")
    apps = tmp_path / "Applications"
    (apps / "CuaDriver.app").mkdir(parents=True)

    with patch("tools.computer_use.cua_backend.sys.platform", "darwin"), patch.object(
        cua_backend, "_CUA_DRIVER_APP_SEARCH_DIRS", (str(apps),)
    ):
        assert cua_backend.resolve_cua_driver_app(str(loose)) == str(apps / "CuaDriver.app")


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_app_bundle_resolution_is_macos_only(platform):
    from tools.computer_use import cua_backend

    with patch("tools.computer_use.cua_backend.sys.platform", platform):
        assert cua_backend.resolve_cua_driver_app("/usr/bin/cua-driver") is None


def test_app_bundle_is_none_when_nothing_is_installed(tmp_path):
    from tools.computer_use import cua_backend

    with patch("tools.computer_use.cua_backend.sys.platform", "darwin"), patch.object(
        cua_backend, "_CUA_DRIVER_APP_SEARCH_DIRS", (str(tmp_path / "nowhere"),)
    ), patch.object(cua_backend, "resolve_cua_driver_cmd", return_value=None):
        assert cua_backend.resolve_cua_driver_app() is None


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


def test_macos_daemon_launches_through_launchservices_with_private_socket():
    from tools.computer_use import cua_backend

    daemon = _make_daemon(cua_backend, "darwin")
    popen, _ = _start(cua_backend, daemon, "darwin", APP)
    argv = popen.call_args.args[0]

    assert argv[:5] == ["/usr/bin/open", "-n", "-W", "-a", APP]
    assert argv[argv.index("--args") + 1 :] == [
        "serve",
        "--socket",
        daemon.socket_path,
        *SERVE_POLICY,
    ]
    # --embedded means "inherit the host app's TCC grants", i.e. Hermes'. That
    # host attribution is the bug; LaunchServices replaces it.
    assert "--embedded" not in argv


def test_macos_launch_forwards_only_cua_policy_env_never_secrets(monkeypatch):
    """LaunchServices drops the caller's env, so policy vars ride on --env.

    argv is world-readable through ``ps``, so only the non-secret cua-driver
    policy knobs may be forwarded — never the sanitized environment wholesale.
    """
    from tools.computer_use import cua_backend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    monkeypatch.setenv("HERMES_SECRET_CANARY", "canary-must-not-leak")

    daemon = _make_daemon(cua_backend, "darwin")
    with patch.object(cua_backend, "_cua_telemetry_disabled", return_value=True):
        popen, _ = _start(cua_backend, daemon, "darwin", APP)
    argv = popen.call_args.args[0]

    forwarded = {argv[i + 1] for i, tok in enumerate(argv) if tok == "--env"}
    assert forwarded == {
        "CUA_DRIVER_RS_TELEMETRY_ENABLED=0",
        "CUA_DRIVER_PERMISSION_MODE=unrestricted",
        "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS=1",
    }
    joined = " ".join(argv)
    assert "sk-must-not-leak" not in joined
    assert "canary-must-not-leak" not in joined


def test_macos_without_an_app_bundle_keeps_the_direct_embedded_spawn():
    from tools.computer_use import cua_backend

    daemon = _make_daemon(cua_backend, "darwin")
    popen, _ = _start(cua_backend, daemon, "darwin", None)
    argv = popen.call_args.args[0]

    assert argv[:2] == ["/opt/cua-driver", "serve"]
    assert "--embedded" in argv
    assert argv[argv.index("--socket") + 1] == daemon.socket_path


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_non_macos_platforms_are_unchanged(platform):
    from tools.computer_use import cua_backend

    daemon = _make_daemon(cua_backend, platform)
    # Even with an app path in hand, non-macOS must not route through open(1);
    # resolve_cua_driver_app itself returns None off darwin.
    popen, _ = _start(cua_backend, daemon, platform, None)
    argv = popen.call_args.args[0]
    proxy_command, proxy_args = daemon.proxy_invocation()

    assert argv[:2] == ["/opt/cua-driver", "serve"]
    assert "--embedded" in argv
    assert "/usr/bin/open" not in argv
    assert proxy_command == "/opt/cua-driver"
    assert proxy_args == ["mcp", "--embedded", "--socket", daemon.socket_path]


def test_child_env_still_carries_permission_mode_and_bypass():
    from tools.computer_use import cua_backend

    daemon = _make_daemon(cua_backend, "darwin")
    popen, _ = _start(cua_backend, daemon, "darwin", APP)
    env = popen.call_args.kwargs["env"]

    assert env["CUA_DRIVER_PERMISSION_MODE"] == "unrestricted"
    assert env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] == "1"


def test_each_session_gets_its_own_socket():
    from tools.computer_use import cua_backend

    first = _make_daemon(cua_backend, "darwin")
    second = _make_daemon(cua_backend, "darwin")

    assert first.socket_path != second.socket_path


def test_readiness_polls_the_private_socket_until_the_daemon_answers():
    from tools.computer_use import cua_backend

    daemon = _make_daemon(cua_backend, "darwin")
    _, run = _start(
        cua_backend, daemon, "darwin", APP, runs=[_status(1), _status(0)]
    )

    probes = [call.args[0] for call in run.call_args_list]
    assert probes == [
        ["/opt/cua-driver", "status", "--socket", daemon.socket_path],
        ["/opt/cua-driver", "status", "--socket", daemon.socket_path],
    ]


def test_launcher_exiting_early_raises_with_the_stderr_tail():
    from tools.computer_use import cua_backend

    process = _fake_process()
    process.poll.return_value = 1
    process.stderr = ["Unable to find application named 'CuaDriver'"]
    daemon = _make_daemon(cua_backend, "darwin")

    with pytest.raises(RuntimeError, match="Unable to find application"):
        _start(cua_backend, daemon, "darwin", APP, process=process, runs=[_status(1)])


# ---------------------------------------------------------------------------
# proxy_invocation()
# ---------------------------------------------------------------------------


def test_launchservices_proxy_targets_the_socket_without_embedded():
    from tools.computer_use import cua_backend

    daemon = _make_daemon(cua_backend, "darwin")
    _start(cua_backend, daemon, "darwin", APP)
    command, args = daemon.proxy_invocation()

    assert command == "/opt/cua-driver"
    assert args == ["mcp", "--socket", daemon.socket_path]


def test_proxy_invocation_refuses_when_the_daemon_is_gone():
    from tools.computer_use import cua_backend

    process = _fake_process()
    daemon = _make_daemon(cua_backend, "darwin")
    _start(cua_backend, daemon, "darwin", APP, process=process)
    process.poll.return_value = 0

    with pytest.raises(RuntimeError):
        daemon.proxy_invocation()


# ---------------------------------------------------------------------------
# stop()
# ---------------------------------------------------------------------------


def test_stop_shuts_the_daemon_down_over_its_private_socket():
    """``stop --socket`` ends the daemon, which lets ``open -W`` exit on its own."""
    from tools.computer_use import cua_backend

    process = _fake_process()
    daemon = _make_daemon(cua_backend, "darwin")
    _start(cua_backend, daemon, "darwin", APP, process=process)

    with patch.object(cua_backend.subprocess, "run", return_value=_status()) as run:
        daemon.stop()

    assert run.call_args.args[0] == [
        "/opt/cua-driver",
        "stop",
        "--socket",
        daemon.socket_path,
    ]
    process.terminate.assert_not_called()
    process.kill.assert_not_called()


def test_stop_is_idempotent_after_the_daemon_already_exited():
    from tools.computer_use import cua_backend

    process = _fake_process()
    daemon = _make_daemon(cua_backend, "darwin")
    _start(cua_backend, daemon, "darwin", APP, process=process)
    process.poll.return_value = 0

    with patch.object(cua_backend.subprocess, "run") as run:
        daemon.stop()
        daemon.stop()

    run.assert_not_called()


# ---------------------------------------------------------------------------
# Backend wiring
# ---------------------------------------------------------------------------


def test_backend_still_gates_the_private_daemon_on_unrestricted_mode():
    from tools.computer_use.cua_backend import CuaDriverBackend

    assert CuaDriverBackend(permission_mode="standard")._embedded_daemon is None
    assert CuaDriverBackend(permission_mode="unrestricted")._embedded_daemon is not None


def test_daemon_rejects_any_mode_other_than_unrestricted():
    from tools.computer_use import cua_backend

    with pytest.raises(ValueError):
        cua_backend._EmbeddedCuaDaemon("cua-driver", "standard")

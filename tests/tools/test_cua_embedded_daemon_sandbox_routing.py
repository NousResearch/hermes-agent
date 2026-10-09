"""Regression tests for the embedded cua-driver daemon's sandbox routing.

Bug (reported live, Wian's Docker-sandboxed Bot Desktop, 2026-10-09, task t_39df9245): any
non-standard cua permission mode (``bounded``/``unrestricted`` -- the latter reachable simply via
``approvals.mode: off`` or ``--yolo``) made ``CuaDriverBackend`` spawn an ``_EmbeddedCuaDaemon``,
and that daemon's ``start()``/``proxy_invocation()``/``_socket_ready()``/``stop()`` ALWAYS ran the
``cua-driver serve``/``status``/``stop`` subprocesses directly on the HOST gateway process, even
when the Bot Desktop they were supposed to drive lived inside a Docker/ssh/apptainer sandbox. Every
``computer_use`` call for such a profile silently talked to a nonexistent host DISPLAY: discovery
calls (``list_windows``) came back empty without raising, and a full-screen capture
(``get_desktop_state``) failed with "Capture error: Connection refused (os error 111)" -- the exact
error reproduced live against the real wian sandbox during this task's investigation.

The fix threads a sandbox-routing check (``cua_backend.sandbox_serve_invocation`` /
``sandbox_cli_invocation``, the same exec-prefix + env wrapping the non-daemon MCP/CLI paths already
use) through every one-shot/long-lived subprocess ``_EmbeddedCuaDaemon`` spawns, so the daemon lands
in the SAME container the screen lives in instead of the host.
"""
from __future__ import annotations

import json

import pytest

from tools.computer_use import cua_backend as cb
from tools.computer_use import cua_backend_daemon as cb_daemon


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _fake_sandbox_invocation(recorded_key):
    """Build a fake ``sandbox_serve_invocation``/``sandbox_cli_invocation`` double that records the
    argv it was asked to wrap and returns a docker-exec-shaped invocation."""
    def _fake(argv, **kw):
        recorded_key.append((list(argv), kw))
        return (("docker", ["exec", "-i", "-u", "pn", "c0ffee", "bash", "-c",
                            f"export DISPLAY=:20; exec {' '.join(argv)}"]), {"PATH": "/usr/bin"})
    return _fake


@pytest.mark.platforms("linux")
def test_unrestricted_daemon_serve_routes_into_the_sandbox(monkeypatch):
    """start() must try sandbox_serve_invocation BEFORE resolving/spawning a host binary: a daemon
    backing a sandboxed Bot Desktop must never touch the host driver at all."""
    serve_calls: list = []
    monkeypatch.setattr(cb, "sandbox_serve_invocation", _fake_sandbox_invocation(serve_calls))

    def _host_path_must_not_run(*a, **kw):
        raise AssertionError("resolve_cua_driver_cmd must not run when the daemon is sandboxed")
    from tools.computer_use import cua_backend_driver as cb_driver
    monkeypatch.setattr(cb_driver, "resolve_cua_driver_cmd", _host_path_must_not_run)

    captured = {}

    def _fake_popen(argv, **kw):
        captured["argv"], captured["env"] = argv, kw.get("env")
        class _P:
            def poll(self):
                return None
        return _P()
    monkeypatch.setattr(cb_daemon.subprocess, "Popen", _fake_popen)

    status_calls: list = []
    monkeypatch.setattr(cb, "sandbox_cli_invocation", _fake_sandbox_invocation(status_calls))

    def _fake_run_quiet(cmd, **kw):
        return _Proc(returncode=0)
    monkeypatch.setattr(cb, "_run_quiet", _fake_run_quiet)
    monkeypatch.setattr(cb_daemon.threading, "Thread", lambda *a, **kw: type(
        "T", (), {"start": lambda self: None})())

    daemon = cb_daemon._EmbeddedCuaDaemon("", "unrestricted")
    daemon.start()

    assert serve_calls, "sandbox_serve_invocation was never consulted -- the fix regressed"
    assert daemon._sandboxed is True
    assert captured["argv"][0] == "docker"  # ran INSIDE the sandbox exec prefix, never bare on the host
    assert "DISPLAY=:20" in captured["argv"][-1]
    assert "cua-driver serve" in captured["argv"][-1]
    assert "--dangerously-bypass-approvals" in captured["argv"][-1]


@pytest.mark.platforms("linux")
def test_unrestricted_daemon_without_a_sandbox_still_uses_the_host_driver(monkeypatch):
    """Non-sandboxed desktops (bot_desktop.placement: gateway, the common case) are unaffected:
    sandbox_serve_invocation returning None falls through to the pre-existing host spawn path."""
    import tools.environments.local as local_mod
    monkeypatch.setattr(local_mod, "_HERMES_BIN_DIR", None)  # see _bypass_hermes_bin_dir_cache below
    monkeypatch.setattr(cb, "sandbox_serve_invocation", lambda argv, **kw: None)
    from tools.computer_use import cua_backend_driver as cb_driver
    monkeypatch.setattr(cb_driver, "resolve_cua_driver_cmd", lambda override=None: "/opt/cua-driver")
    monkeypatch.setattr(cb_driver, "_resolve_mcp_invocation", lambda driver_cmd, **kw: (driver_cmd, ["mcp"]))

    captured = {}

    def _fake_popen(argv, **kw):
        captured["argv"] = argv
        class _P:
            def poll(self):
                return None
        return _P()
    monkeypatch.setattr(cb_daemon.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(cb, "_run_quiet", lambda cmd, **kw: _Proc(returncode=0))
    monkeypatch.setattr(cb_daemon.threading, "Thread", lambda *a, **kw: type(
        "T", (), {"start": lambda self: None})())

    daemon = cb_daemon._EmbeddedCuaDaemon("", "unrestricted")
    daemon.start()

    assert daemon._sandboxed is False
    assert captured["argv"][:2] == ["/opt/cua-driver", "serve"]


@pytest.mark.platforms("linux")
def test_sandboxed_daemon_proxy_and_call_invocation_never_touch_host_paths(monkeypatch):
    """proxy_invocation() (long-lived MCP client) and call_invocation() (CLI fallback) must both
    route through the sandbox once start() established it, and call_invocation must never add the
    HOST-only screenshot_out_file optimization (the sandboxed driver can't see that path)."""
    monkeypatch.setattr(cb, "sandbox_serve_invocation", _fake_sandbox_invocation([]))
    monkeypatch.setattr(cb, "sandbox_cli_invocation", _fake_sandbox_invocation([]))  # status probe during start()
    monkeypatch.setattr(cb_daemon.subprocess, "Popen", lambda argv, **kw: type(
        "P", (), {"poll": lambda self: None})())
    monkeypatch.setattr(cb, "_run_quiet", lambda cmd, **kw: _Proc(returncode=0))
    monkeypatch.setattr(cb_daemon.threading, "Thread", lambda *a, **kw: type(
        "T", (), {"start": lambda self: None})())

    daemon = cb_daemon._EmbeddedCuaDaemon("", "unrestricted")
    daemon.start()
    assert daemon._sandboxed is True

    cli_calls: list = []
    monkeypatch.setattr(cb, "sandbox_cli_invocation", _fake_sandbox_invocation(cli_calls))

    proxy_cmd, proxy_args = daemon.proxy_invocation()
    assert proxy_cmd == "docker"
    assert "cua-driver mcp --embedded --socket" in " ".join(proxy_args[-1].split())  # exec script tail

    cmd, env, shot_file = daemon.call_invocation("get_window_state", {"pid": 1, "window_id": 2})
    assert cmd[0] == "docker"
    assert shot_file is None  # HOST temp path optimization must be skipped when sandboxed
    assert cli_calls and "screenshot_out_file" not in json.dumps(cli_calls[-1][0])
    assert env == {"PATH": "/usr/bin"}


def test_bounded_mode_refuses_a_sandboxed_placement(monkeypatch):
    """bounded's --capability-manifest is a HOST file path nothing mounts into the sandbox: fail
    closed with an actionable error instead of silently spawning on the host (the original bug) or
    guessing the manifest is visible in the container."""
    from tools.bot_desktop import placement, runtime as _bd_runtime
    monkeypatch.setattr(_bd_runtime, "tool_placement", lambda: placement.TERMINAL)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(b"version: 3\n")
        manifest_path = f.name

    daemon = cb_daemon._EmbeddedCuaDaemon("", "bounded", capability_manifest=manifest_path)
    with pytest.raises(RuntimeError, match="bounded is not yet supported"):
        daemon.start()

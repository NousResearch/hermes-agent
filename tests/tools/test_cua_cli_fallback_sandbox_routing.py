"""Regression tests for the cua-driver MCP->CLI fallback transport's sandbox routing.

Bug (reported live, Wian's Docker-sandboxed Bot Desktop, 2026-10-09): the MCP transport correctly
spawns ``cua-driver mcp`` INSIDE the terminal backend's sandbox via ``sandbox_mcp_invocation()``
(docker-exec + the sandbox's DISPLAY/XAUTHORITY env). But when ``list_windows`` comes back empty
over MCP, ``cua_backend_capture.py::_cli_refetch`` re-fetches via
``_CuaDriverSession._call_tool_via_cli``, and THAT function only special-cased an embedded daemon --
never the sandbox placement -- so it fell back to a bare *host* ``cua-driver call`` with the host's
(nonexistent, for a sandboxed desktop) DISPLAY: "no DISPLAY is set" / "Connection refused (os error
111)", even though the MCP transport it was falling back FROM was already running correctly inside
the very same sandbox.

The fix adds ``cua_backend.sandbox_cli_invocation()`` (the CLI-call twin of
``sandbox_mcp_invocation()``) and makes ``_call_tool_via_cli`` consult it with the same
daemon > sandbox > host precedence ``_lifecycle_coro`` already uses for the MCP transport.
"""
from __future__ import annotations

import json

from tools.computer_use import cua_backend as cb
from tools.computer_use import cua_backend_session as cb_session


class _Proc:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _session(embedded_daemon=None):
    s = object.__new__(cb_session._CuaDriverSession)
    s._embedded_daemon = embedded_daemon
    return s


def _bypass_hermes_bin_dir_cache(monkeypatch):
    """``_sanitize_subprocess_env`` -> ``_prepend_hermes_bin_dir`` resolves (and process-caches) the
    hermes install dir via ``pm.environments.payload_command_dir(<checkout root>)``, which reads
    ``<checkout root>/../manifest.json`` -- on a checkout nested directly under the real HERMES_HOME
    (this dev box's layout) that path happens to collide with the real home and trips
    ``home_io_guard``. Pre-seeding the module-level cache sidesteps the resolution entirely; these
    tests are about sandbox CLI routing, not PATH-prepending."""
    import tools.environments.local as local_mod
    monkeypatch.setattr(local_mod, "_HERMES_BIN_DIR", None)


def test_cli_fallback_routes_into_the_sandbox_when_mcp_also_does(monkeypatch):
    """The bug: _call_tool_via_cli must consult sandbox_cli_invocation (same as the MCP transport's
    own sandbox_mcp_invocation) rather than silently building a bare host-side command."""
    _bypass_hermes_bin_dir_cache(monkeypatch)
    sandbox_calls = []

    def _fake_sandbox_cli_invocation(call_argv):
        sandbox_calls.append(call_argv)
        return (("docker", ["exec", "-i", "-u", "pn", "c0ffee", "bash", "-c",
                            f"export DISPLAY=:20; exec {' '.join(call_argv)}"]), {"PATH": "/usr/bin"})

    monkeypatch.setattr(cb, "sandbox_cli_invocation", _fake_sandbox_cli_invocation)

    def _host_path_must_not_run(*a, **kw):
        raise AssertionError("resolve_cua_driver_cmd must not run when the sandbox handles the call")
    from tools.computer_use import cua_backend_driver as cb_driver
    monkeypatch.setattr(cb_driver, "resolve_cua_driver_cmd", _host_path_must_not_run)

    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Proc(stdout=json.dumps({"windows": []}))
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    session = _session(embedded_daemon=None)
    out = session._call_tool_via_cli("list_windows", {"on_screen_only": True}, timeout=5.0)

    assert out["isError"] is False
    assert sandbox_calls, "sandbox_cli_invocation was never consulted -- the fix regressed"
    assert captured["cmd"][0] == "docker"  # ran INSIDE the sandbox exec prefix, never bare on the host
    assert "DISPLAY=:20" in captured["cmd"][-1]


def test_cli_fallback_without_a_sandbox_still_uses_the_host_driver(monkeypatch):
    """Non-sandboxed desktops (bot_desktop.placement: gateway, the common case) are unaffected:
    sandbox_cli_invocation returning None falls through to the pre-existing host CLI path."""
    _bypass_hermes_bin_dir_cache(monkeypatch)
    monkeypatch.setattr(cb, "sandbox_cli_invocation", lambda call_argv: None)
    from tools.computer_use import cua_backend_driver as cb_driver
    monkeypatch.setattr(cb_driver, "resolve_cua_driver_cmd", lambda override=None: "cua-driver")

    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Proc(stdout=json.dumps({"windows": []}))
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    session = _session(embedded_daemon=None)
    out = session._call_tool_via_cli("list_windows", {}, timeout=5.0)

    assert out["isError"] is False
    assert captured["cmd"][0] == "cua-driver"
    assert captured["cmd"][1] == "call"


def test_cli_fallback_skips_screenshot_out_file_when_sandboxed(monkeypatch):
    """screenshot_out_file is a HOST filesystem path; a sandboxed driver process has its own
    filesystem namespace and could never see it, so the sandboxed branch must not add it."""
    _bypass_hermes_bin_dir_cache(monkeypatch)
    captured_argv = {}

    def _fake_sandbox_cli_invocation(call_argv):
        captured_argv["argv"] = call_argv
        return (("docker", ["exec", "-i", "c0ffee", "cua-driver", *call_argv[1:]]), {"PATH": "/usr/bin"})

    monkeypatch.setattr(cb, "sandbox_cli_invocation", _fake_sandbox_cli_invocation)

    def _fake_run(cmd, **kw):
        return _Proc(stdout=json.dumps({"tree_markdown": "root"}))
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    session = _session(embedded_daemon=None)
    session._call_tool_via_cli("get_window_state", {"pid": 1, "window_id": 2}, timeout=5.0)

    # call_argv is ["cua-driver", "call", "get_window_state", "<json args>"]
    call_args_json = captured_argv["argv"][3]
    assert "screenshot_out_file" not in call_args_json


def test_cli_fallback_prefers_embedded_daemon_over_sandbox(monkeypatch):
    """Precedence must mirror _lifecycle_coro's own MCP-transport ordering: an embedded daemon
    (bounded/unrestricted permission mode) always wins over sandbox placement."""
    _bypass_hermes_bin_dir_cache(monkeypatch)

    def _must_not_be_called(call_argv):
        raise AssertionError("sandbox_cli_invocation must not run when an embedded daemon is set")
    monkeypatch.setattr(cb, "sandbox_cli_invocation", _must_not_be_called)

    class _FakeDaemon:
        socket_path = "/tmp/cua-daemon.sock"
        def call_invocation(self, name, call_args):
            return (["cua-driver-proxy", "call", name, json.dumps(call_args), "--socket", self.socket_path],
                    {"PATH": "/usr/bin"}, None)

    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = cmd
        return _Proc(stdout=json.dumps({"windows": []}))
    import subprocess
    monkeypatch.setattr(subprocess, "run", _fake_run)

    session = _session(embedded_daemon=_FakeDaemon())
    out = session._call_tool_via_cli("list_windows", {}, timeout=5.0)

    assert out["isError"] is False
    assert captured["cmd"][0] == "cua-driver-proxy"
    assert "--socket" in captured["cmd"] and "/tmp/cua-daemon.sock" in captured["cmd"]


class _FakeDocker:
    """Stand-in for DockerEnvironment: only what streams.exec_prefix / sandbox_host read."""
    _docker_exe = "docker"
    _container_id = "c0ffee"

    def get_temp_dir(self):
        return "/tmp"


def test_sandbox_cli_invocation_rides_the_same_exec_prefix_as_mcp(monkeypatch):
    """Direct unit test of the new helper: it must wrap the CLI call argv with the identical
    exec-prefix + env machinery sandbox_mcp_invocation uses for the long-lived MCP server."""
    from tools.bot_desktop import placement, runtime
    from tools.bot_desktop import sandbox_host
    from tools.environments import streams

    monkeypatch.setattr(placement, "_setting", lambda: "terminal")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "docker")
    started: list[str] = []
    monkeypatch.setattr(runtime, "sandbox_screen_running", lambda: bool(started))
    monkeypatch.setattr(runtime, "published_env",
                        lambda: {"DISPLAY": ":20", "XAUTHORITY": "/x"} if started else {})
    monkeypatch.setattr(runtime, "start", lambda **kw: started.append("started"))
    monkeypatch.setattr(runtime, "_sandbox_env", lambda *, create: _FakeDocker())
    monkeypatch.setattr(runtime, "touch_activity", lambda: None)
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    monkeypatch.setattr(streams, "exec_prefix", lambda e, *, user=None, interactive=True:
                        ["docker", "exec", "-u", user, e._container_id])  # no -i: interactive=False

    (command, args), child_env = cb.sandbox_cli_invocation(["cua-driver", "call", "list_windows", "{}"])

    assert command == "docker"
    assert "cua-driver call list_windows" in args[-1]
    assert "export DISPLAY=:20" in args[-1]
    assert set(child_env) == {"PATH"}  # the host driver's env never reaches the sandbox driver


def test_sandbox_cli_invocation_returns_none_on_gateway_placement(monkeypatch):
    from tools.bot_desktop import placement
    monkeypatch.setattr(placement, "_setting", lambda: "gateway")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: "docker")
    assert cb.sandbox_cli_invocation(["cua-driver", "call", "list_windows", "{}"]) is None

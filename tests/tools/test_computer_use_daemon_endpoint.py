"""Endpoint/ownership contracts for #92355, through real config and backend/session imports.

Only the external driver/stdio peer is simulated; no desktop or installed daemon is touched.
"""
import concurrent.futures
import json
import os
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools.computer_use import cua_backend as cb
from tools.computer_use import cua_backend_daemon as daemon_module
from tools.computer_use import doctor
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession


@pytest.fixture
def driver_peer(tmp_path, monkeypatch):
    import subprocess

    import mcp
    import mcp.client.stdio

    command = tmp_path / "driver tools" / "cua-driver.exe"
    command.parent.mkdir()
    command.write_text("fixture only", encoding="utf-8")
    command.chmod(0o700)
    manifest_command = str(command.parent / "manifest driver.exe")
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(command))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-provider-secret")
    monkeypatch.setattr(cb, "_update_checked", True)
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **kw: None)
    state = SimpleNamespace(command=str(command), manifest_command=manifest_command,
                            args=["mcp"], spawns=[], runs=[], error=None, dead=False,
                            doctor_fallback=False, doctor_spawns=[], doctor_envs=[], calls=[])

    def endpoint(args):
        for index, arg in enumerate(args):
            if arg == "--socket":
                return args[index + 1]
            if arg.startswith("--socket="):
                return arg.partition("=")[2]
        return None

    state.endpoint = endpoint

    def run(argv, **kwargs):
        state.runs.append((list(argv), kwargs))
        verb = argv[1]
        if verb == "manifest":
            data = {"binary_version": "0.20.0", "mcp_invocation": {
                "command": manifest_command, "args": list(state.args)}, "subcommands": [
                {"name": name, "args": [{"name": flag} for flag in flags]}
                for name, flags in {
                    "mcp": ["--socket", "--grant"],
                    "serve": ["--socket", "--permission-mode", "--capability-manifest",
                              "--approve-capability-manifest", "--embedded"],
                    "stop": ["--socket"],
                }.items()]}
            text = json.dumps(data)
        elif verb == "call":
            text = ("daemon is not running" if state.dead else
                    json.dumps({"windows": [], "endpoint": endpoint(argv)}))
        elif verb in {"status", "stop"}:
            text = "running"
        elif verb == "--version":
            text = "cua-driver 0.20.0"
        elif verb == "doctor":
            text = "[ok] fixture"
        elif verb == "--help":
            text = "--no-overlay"
        else:
            raise AssertionError(f"unexpected subprocess: {argv}")
        return SimpleNamespace(returncode=0, stdout=text, stderr="")

    def popen(argv, **kwargs):
        state.runs.append((list(argv), kwargs))
        if argv[1] == "serve":
            return SimpleNamespace(stderr=[], poll=lambda: None, wait=lambda **kw: 0)
        assert argv[1] == "mcp"
        state.doctor_spawns.append(list(argv))
        state.doctor_envs.append(dict(kwargs["env"]))
        if state.doctor_fallback and len(state.doctor_spawns) % 2:
            responses = [{"result": {}}, {"result": {"isError": True, "content": [
                {"type": "text", "text": "health_report is not classified"}]}}]
            # Discovery must not be re-run for the fallback MCP connection.
            state.args = ["mcp", "--socket", str(tmp_path / "wrong doctor.sock")]
            monkeypatch.setenv("DISPLAY", "wrong-doctor-display")
        else:
            responses = [{"result": {"serverInfo": {"version": "0.20.0"}}},
                         {"result": {"structuredContent": {"accessibility": True, "screen_recording": True}}},
                         {"result": {"structuredContent": {"apps": []}}}]
        return SimpleNamespace(stdin=StringIO(), stdout=StringIO("".join(json.dumps(r) + "\n" for r in responses)),
                               stderr=StringIO(), wait=lambda **kw: 0)

    @asynccontextmanager
    async def stdio(params):
        state.spawns.append(params)
        if state.dead:
            raise RuntimeError("selected daemon is not running")
        yield params, params

    class Peer:
        def __init__(self, read, write):
            self.params = read

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=[])

        async def call_tool(self, name, args):
            state.calls.append((name, dict(args)))
            if state.error is not None:
                error, state.error = state.error, None
                raise error
            return SimpleNamespace(content=[], isError=False, structuredContent={
                "windows": [], "endpoint": endpoint(self.params.args)})

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(mcp.client.stdio, "stdio_client", stdio)
    monkeypatch.setattr(mcp, "ClientSession", Peer)
    # Avoid macOS app identity inspection: the fixture is not a signed CuaDriver.app.
    monkeypatch.setattr(daemon_module, "_embedded_daemon_spawn_command",
                        lambda command, args, **kw: [command, *args])
    return state


def _config(home, socket):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(yaml.safe_dump({"plugins": {"enabled": []}, "computer_use": {
        "daemon_socket": socket, "no_overlay": False, "grant_existing_profile": True,
        "capability_manifest": str(home / "capabilities.yaml"),
    }}), encoding="utf-8")
    (home / "capabilities.yaml").write_text("version: 3\n", encoding="utf-8")


@pytest.mark.parametrize("raw,args,error", [
    (None, ["mcp"], None),
    ("", ["mcp"], None),
    ("   ", ["mcp"], None),
    ("~/shared daemon.sock", ["mcp"], None),
    (False, ["mcp"], "must be a string"),
    (123, ["mcp"], "must be a string"),
    (["socket"], ["mcp"], "must be a string"),
    ("relative.sock", ["mcp"], "absolute path"),
    ("/bad\x00.sock", ["mcp"], "NUL"),
    ("configured", ["mcp", "--socket", "other"], "both select"),
    ("configured", ["mcp", "--socket=other"], "both select"),
    (None, ["mcp", "--socket"], "non-empty"),
    (None, ["mcp", "--socket", "--no-overlay"], "non-empty"),
    (None, ["mcp", "--socket="], "non-empty"),
    (None, ["mcp", "--socket=one", "--socket=two"], "only one"),
    pytest.param(r"\\.\pipe\shared cua", ["mcp"], None, marks=pytest.mark.windows_only),
])
def test_invalid_selectors_fail_before_connecting(tmp_path, monkeypatch, driver_peer, raw, args, error, capsys):
    raw = str(tmp_path / "configured.sock") if raw == "configured" else raw
    _config(tmp_path, raw)
    driver_peer.args = args
    session = _CuaDriverSession(_AsyncBridge())
    try:
        if error:
            with pytest.raises(ValueError, match=error):
                session.start()
            assert not driver_peer.spawns
            assert doctor.run_doctor() == 2
            assert error in capsys.readouterr().err
            assert not driver_peer.doctor_spawns
        else:
            session.start()
            expected = os.path.expanduser(raw.strip()) if raw and raw.strip() else None
            assert driver_peer.endpoint(driver_peer.spawns[-1].args) == expected
    finally:
        session.stop()
        session._bridge.stop()


@pytest.mark.parametrize("selector,mode", [
    ("config", "standard"), ("manifest-split", "standard"), ("manifest-equals", "standard"),
    ("default", "standard"), ("config", "bounded"), ("config", "unrestricted"),
])
def test_endpoint_is_stable_across_profiles_recovery_and_fallback(tmp_path, monkeypatch, driver_peer, selector, mode):
    from agent import secret_scope
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    homes = [tmp_path / "profile A", tmp_path / "profile B"]
    endpoints = [str(home / "shared daemon ; $(literal).sock") for home in homes]
    for home, socket in zip(homes, endpoints):
        _config(home, socket if selector == "config" else "")
        Path(socket).write_text("external daemon remains owned by its operator", encoding="utf-8")
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    scope = secret_scope.set_secret_scope({})
    try:
        for index in (0, 1, 0):  # real profile-local config reads A -> B -> A, no module reloads
            home, external = homes[index], endpoints[index]
            _config(home, external if selector == "config" else "")
            monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver_peer.command)
            driver_peer.args = (["mcp", "--socket", external] if selector == "manifest-split" else
                                ["mcp", f"--socket={external}"] if selector == "manifest-equals" else ["mcp"])
            if mode != "standard":
                driver_peer.args = ["mcp", f"--socket={external}"]
            monkeypatch.setenv("DISPLAY", f"profile-{index}-display")
            token = set_hermes_home_override(home)
            backend = cb.CuaDriverBackend(permission_mode=mode)
            start = len(driver_peer.runs)
            try:
                backend.start()
                expected = (backend._embedded_daemon.socket_path if backend._embedded_daemon is not None else
                            None if selector == "default" else external)
                params = driver_peer.spawns[-1]
                assert params.command == driver_peer.manifest_command
                assert driver_peer.endpoint(params.args) == expected
                assert sum(a == "--socket" or a.startswith("--socket=") for a in params.args) == (expected is not None)
                assert "--grant" not in params.args  # retired browser grant stays retired
                assert "ANTHROPIC_API_KEY" not in params.env
                assert params.env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "0"
                assert params.env["DISPLAY"] == f"profile-{index}-display"
                assert params.env["HERMES_HOME"] == str(home)
                # Editing config, PATH selection, and the manifest cannot retarget a live/recovering session.
                _config(home, str(home / "changed.sock"))
                monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", str(tmp_path / "missing-driver"))
                driver_peer.args = ["mcp", "--socket", str(home / "changed-manifest.sock")]
                monkeypatch.setenv("DISPLAY", "changed-display")
                driver_peer.error = BrokenPipeError("fixture dropped the MCP proxy")
                result = backend.call_tool("list_windows")
                assert result["structuredContent"]["endpoint"] == expected
                assert driver_peer.spawns[-1].args == params.args
                assert driver_peer.spawns[-1].env == params.env
                driver_peer.error = RuntimeError("daemon proxy congestion")
                result = backend.call_tool("list_windows")
                assert result["structuredContent"]["endpoint"] == expected
                call, options = driver_peer.runs[-1]
                assert call[:3] == [params.command, "call", "list_windows"]
                assert driver_peer.endpoint(call) == expected
                assert options["env"] == params.env
                assert "ANTHROPIC_API_KEY" not in options["env"]
                before = len(driver_peer.runs)
                driver_peer.error = RuntimeError("daemon proxy congestion")
                result = backend.call_tool("click", {"x": 1, "y": 2})
                assert result["structuredContent"]["code"] == "transport_outcome_unknown"
                assert len(driver_peer.runs) == before  # never replay a possibly-delivered action
                before_clicks = sum(name == "click" for name, _ in driver_peer.calls)
                driver_peer.error = BrokenPipeError("fixture dropped after receiving click")
                result = backend.call_tool("click", {"x": 1, "y": 2})
                assert result["structuredContent"]["code"] == "transport_outcome_unknown"
                assert sum(name == "click" for name, _ in driver_peer.calls) == before_clicks + 1
                assert len(driver_peer.runs) == before
                driver_peer.error = concurrent.futures.TimeoutError("fixture timeout")
                result = backend.call_tool("click", {"x": 1, "y": 2})
                assert result["structuredContent"]["code"] == "timeout_outcome_unknown"
                result = backend.call_tool("list_windows")
                assert result["structuredContent"]["endpoint"] == expected
                assert driver_peer.spawns[-1].args == params.args
                assert driver_peer.spawns[-1].env == params.env
                driver_peer.dead = True
                with pytest.raises(RuntimeError, match="selected.*daemon is not running"):
                    backend._session._call_tool_via_cli("list_windows", {}, 1)
                driver_peer.dead = False
            finally:
                driver_peer.dead = False
                backend.stop()
                reset_hermes_home_override(token)
            lifetime = [argv for argv, _ in driver_peer.runs[start:] if argv[1] in {"serve", "stop"}]
            assert [argv[1] for argv in lifetime] == ([] if mode == "standard" else ["serve", "stop"])
            assert all(driver_peer.endpoint(argv) == expected for argv in lifetime)
            assert Path(external).read_text(encoding="utf-8") == "external daemon remains owned by its operator"

        # An unreachable explicit endpoint fails startup without adopting or creating a daemon.
        _config(homes[0], endpoints[0])
        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver_peer.command)
        driver_peer.args = ["mcp"]
        driver_peer.dead = True
        session = _CuaDriverSession(_AsyncBridge())
        token = set_hermes_home_override(homes[0])
        before = len(driver_peer.runs)
        try:
            with pytest.raises(RuntimeError, match="selected daemon is not running"):
                session.start()
            assert driver_peer.endpoint(driver_peer.spawns[-1].args) == endpoints[0]
            assert all(argv[1] not in {"serve", "stop", "call"} for argv, _ in driver_peer.runs[before:])
        finally:
            session.stop()
            session._bridge.stop()
            driver_peer.dead = False
            reset_hermes_home_override(token)

        # Doctor's health-report fallback must keep its first selected endpoint as well.
        _config(homes[0], endpoints[0])
        monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", driver_peer.command)
        driver_peer.args = ["mcp"]
        driver_peer.doctor_fallback = True
        monkeypatch.setenv("DISPLAY", "original-doctor-display")
        token = set_hermes_home_override(homes[0])
        try:
            assert doctor.run_doctor() == 0
            assert len(driver_peer.doctor_spawns) == 2
            assert driver_peer.doctor_spawns[0] == driver_peer.doctor_spawns[1]
            assert driver_peer.doctor_envs[0]["DISPLAY"] == "original-doctor-display"
            assert driver_peer.doctor_envs[0] == driver_peer.doctor_envs[1]
            assert driver_peer.endpoint(driver_peer.doctor_spawns[0]) == endpoints[0]
        finally:
            reset_hermes_home_override(token)
    finally:
        secret_scope.reset_secret_scope(scope)
        secret_scope.set_multiplex_active(previous)

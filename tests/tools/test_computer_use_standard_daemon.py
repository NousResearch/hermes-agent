"""Standard sessions own their transport, never their user's approval decisions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.computer_use import cua_backend as cb
from tools.computer_use import cua_backend_daemon as daemon_module
from tools.computer_use import cua_backend_driver as driver


@pytest.mark.platforms("macos", "windows", "linux")
@pytest.mark.parametrize("explicit_override", [False, True])
def test_standard_host_owns_only_its_selected_driver_and_socket(tmp_path, monkeypatch, explicit_override):
    import pm

    app = tmp_path / "CuaDriver.app"
    binary = app / "Contents" / "MacOS" / "cua-driver"
    binary.parent.mkdir(parents=True)
    binary.write_text("fixture", encoding="utf-8")
    binary.chmod(0o755)
    selected = str(binary)
    monkeypatch.setenv("HERMES_CUA_DRIVER_CMD", selected if explicit_override else "")
    monkeypatch.setenv("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", "1")
    monkeypatch.setenv("CUA_DRIVER_EMBEDDED", "1")
    monkeypatch.setenv("CUA_DRIVER_RS_PERMISSIONS_GATE", "0")
    monkeypatch.setattr(pm, "ensure", lambda name: None)
    monkeypatch.setattr(pm, "ensure_import", lambda name: None)
    monkeypatch.setattr(driver, "resolve_cua_driver_cmd", lambda: selected)
    monkeypatch.setattr(cb, "cua_driver_runtime_contract_status", lambda: {"ready": True})
    monkeypatch.setattr(cb, "sandbox_mcp_invocation", lambda: None)
    monkeypatch.setattr(cb, "_computer_use_cfg", lambda: {"grant_existing_profile": True})
    monkeypatch.setattr(cb, "_cua_no_overlay", lambda: False)
    monkeypatch.setattr(daemon_module, "_validate_cua_driver_app_signature", lambda path: None)
    monkeypatch.setattr(driver, "_resolve_mcp_invocation", lambda command:
                        (str(tmp_path / "other-driver") if explicit_override else command, ["mcp"]))
    monkeypatch.setattr(cb, "cua_driver_child_env", lambda: {
        "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS": "1", "CUA_DRIVER_EMBEDDED": "1",
        "CUA_DRIVER_RS_PERMISSIONS_GATE": "0", "CUA_DRIVER_POLICY_FILE": "/reviewed/policy.rego",
        "CUA_DRIVER_MANAGED_POLICY_FILE": "/managed/policy.rego",
    })
    monkeypatch.setattr(daemon_module._EmbeddedCuaDaemon, "_sanitized_env", lambda self: self.child_env())
    commands, envs = [], []

    def spawn(command, **kwargs):
        commands.append(command)
        envs.append(kwargs["env"])
        return SimpleNamespace(stderr=(), poll=lambda: None, wait=lambda **kwargs: 0)

    def probe(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(daemon_module.subprocess, "Popen", spawn)
    monkeypatch.setattr(cb, "_run_quiet", probe)
    backend = cb.CuaDriverBackend()
    other = cb.CuaDriverBackend()
    private = backend._embedded_daemon
    assert private is not None
    assert other._embedded_daemon is not None
    assert private.socket_path != other._embedded_daemon.socket_path
    unowned = tmp_path / "unowned.sock"
    unowned.write_text("another owner", encoding="utf-8")
    other._embedded_daemon.socket_path = str(unowned)
    monkeypatch.setattr(backend._session, "start", lambda: None)
    monkeypatch.setattr(backend._session, "call_tool", lambda *args: None)
    try:
        backend.start()
        serve = commands[0]
        assert serve[serve.index("--socket") + 1] == private.socket_path
        assert serve[serve.index("--permission-mode") + 1] == "standard"
        assert not {"--embedded", "--no-permissions-gate", "--dangerously-bypass-approvals",
                    "--grant", "--approve-capability-manifest"}.intersection(serve)
        assert envs[0]["HOME"] != str(tmp_path)
        assert envs[0]["HOME"]
        if serve[0] == "/usr/bin/open":
            assert f"HOME={envs[0]['HOME']}" in serve
            assert "CUA_DRIVER_EMBEDDED=0" in serve
            assert "CUA_DRIVER_RS_PERMISSIONS_GATE=1" in serve
            assert "CUA_DRIVER_POLICY_FILE=/reviewed/policy.rego" in serve
            assert "CUA_DRIVER_MANAGED_POLICY_FILE=/managed/policy.rego" in serve
        assert envs[0]["CUA_DRIVER_PERMISSION_MODE"] == "standard"
        assert envs[0].get("CUA_DRIVER_EMBEDDED") != "1"
        assert envs[0].get("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS") != "1"
        assert envs[0].get("CUA_DRIVER_RS_PERMISSIONS_GATE") != "0"
        command, args = private.proxy_invocation()
        assert command == selected
        assert args[args.index("--socket") + 1] == private.socket_path
        assert backend._session._embedded_daemon is private
        from tools.computer_use import cua_backend_session as session_module
        fallback_commands = []
        def fallback(cmd, env, name, timeout):
            fallback_commands.append(cmd)
            return {"isError": True, "code": "permissions_pending"}
        monkeypatch.setattr(session_module, "_cli_run_json", fallback)
        result = backend._session._call_tool_via_cli("list_windows", {}, 1.0)
        assert result["isError"] is True
        assert fallback_commands == [[selected, "call", "list_windows", "{}", "--socket", private.socket_path]]
        assert private._state_dir is not None
        state_dir = private._state_dir.name
    finally:
        backend.stop()
        other.stop()
    stops = [command for command in commands if "stop" in command]
    assert stops == [[selected, "stop", "--socket", private.socket_path]]
    assert unowned.exists()
    assert not Path(state_dir).exists()


def test_sandbox_standard_session_never_starts_a_host_daemon(monkeypatch):
    import pm

    monkeypatch.setattr(cb, "sandbox_mcp_invocation", lambda: (("sandbox-exec", ["mcp"]), {}))
    monkeypatch.setattr(pm, "ensure_import", lambda name: None)
    monkeypatch.setattr(pm, "ensure", lambda *args: pytest.fail("host acquisition"))
    backend = cb.CuaDriverBackend()
    if backend._embedded_daemon is not None:
        monkeypatch.setattr(backend._embedded_daemon, "start", lambda: pytest.fail("host daemon"))
    monkeypatch.setattr(backend._session, "start", lambda: None)
    monkeypatch.setattr(backend._session, "call_tool", lambda *args: None)
    backend.start()
    assert backend._session._embedded_daemon is None
    backend.stop()

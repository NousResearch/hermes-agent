import os


def test_env_config_routes_explicit_sandbox_to_provider(monkeypatch, tmp_path):
    from tools import terminal_tool as tt

    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setenv("HERMES_SANDBOX", "openshell")
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))

    cfg = tt._get_env_config()

    assert cfg["env_type"] == "openshell"
    assert cfg["cwd"] == "/sandbox"
    assert cfg["host_cwd"] == str(tmp_path)
    assert cfg["sandbox_config"]["provider"] == "openshell"


def test_env_config_no_sandbox_forces_local(monkeypatch):
    from tools import terminal_tool as tt

    monkeypatch.setenv("TERMINAL_ENV", "openshell")
    monkeypatch.setenv("HERMES_SANDBOX", "host")

    cfg = tt._get_env_config()

    assert cfg["env_type"] == "local"


def test_create_environment_delegates_unknown_backend_to_sandbox_provider(monkeypatch, tmp_path):
    from tools import terminal_tool as tt

    seen = {}
    sentinel = object()

    class Provider:
        def create_environment(self, spec):
            seen["spec"] = spec
            return sentinel

    import plugins.sandbox as sandbox_plugins

    monkeypatch.setattr(sandbox_plugins, "load_sandbox_provider", lambda name: Provider())

    result = tt._create_environment(
        env_type="openshell",
        image="",
        cwd="/sandbox",
        timeout=11,
        task_id="task-1",
        host_cwd=str(tmp_path),
        container_config={
            "sandbox_config": {
                "mode": "remote",
                "scope": "scope-1",
                "provider": "openshell",
            }
        },
    )

    assert result is sentinel
    assert seen["spec"].provider == "openshell"
    assert seen["spec"].mode == "remote"
    assert seen["spec"].scope == "scope-1"
    assert seen["spec"].host_cwd == str(tmp_path)


def test_delegate_task_sandbox_override_resolves_host_and_openshell(monkeypatch):
    import tools.delegate_tool as dt

    monkeypatch.setattr(
        dt,
        "_load_full_config",
        lambda: {
            "sandbox": {
                "provider": "openshell",
                "mode": "mirror",
                "openshell": {
                    "from": "openclaw",
                    "remote_workspace_dir": "/sandbox",
                    "remote_agent_workspace_dir": "/agent",
                },
            }
        },
    )

    host_override = dt._resolve_child_sandbox_override(
        {"sandbox": "host"}, None, None, None, None
    )
    assert host_override["env_type"] == "local"

    sandbox_override = dt._resolve_child_sandbox_override(
        {"sandbox": "openshell", "sandbox_mode": "remote", "sandbox_scope": "child-a"},
        None,
        None,
        None,
        None,
    )
    assert sandbox_override["env_type"] == "openshell"
    assert sandbox_override["cwd"] == "/sandbox"
    assert sandbox_override["sandbox_config"]["mode"] == "remote"
    assert sandbox_override["sandbox_config"]["scope"] == "child-a"
    assert sandbox_override["sandbox_config"]["source"] == "openclaw"

    monkeypatch.setenv("HERMES_OPENSHELL_GATEWAY", "env-gw")
    monkeypatch.setenv("HERMES_OPENSHELL_GATEWAY_PORT", "18091")
    sandbox_override = dt._resolve_child_sandbox_override(
        {"sandbox": "openshell"},
        None,
        None,
        None,
        None,
    )
    assert sandbox_override["sandbox_config"]["gateway"] == "env-gw"
    assert sandbox_override["sandbox_config"]["gateway_port"] == 18091

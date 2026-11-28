"""Gateway ``plugins.manage remove`` — the Desktop Plugins hub's Uninstall door.

Drives the real ``hermes_cli.plugins_cmd`` removal core against a temp HERMES_HOME so the RPC is
proven to delete the plugin tree AND its install metadata, and to refuse anything that is not a user
install under ``<HERMES_HOME>/plugins/``.
"""

import json

import pytest

from tui_gateway import server


@pytest.fixture
def plugins_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    (home / "plugins" / "demo-plugin").mkdir(parents=True)
    (home / "plugins" / "demo-plugin" / "plugin.yaml").write_text("name: demo-plugin\nversion: 1.0.0\n", encoding="utf-8")
    (home / "plugins" / ".install-metadata.json").write_text(
        json.dumps({"demo-plugin": {"pinned_sha": "a" * 40}, "other": {"pinned_sha": "b" * 40}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _remove(name):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "remove", "name": name}})


def test_remove_deletes_user_install_and_its_metadata(plugins_home):
    resp = _remove("demo-plugin")

    assert resp["result"] == {"ok": True, "name": "demo-plugin"}
    assert not (plugins_home / "plugins" / "demo-plugin").exists()
    metadata = json.loads((plugins_home / "plugins" / ".install-metadata.json").read_text(encoding="utf-8"))
    assert "demo-plugin" not in metadata and "other" in metadata  # sibling metadata survives


def test_remove_refuses_missing_name_and_unknown_plugin(plugins_home):
    resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "remove"}})
    assert resp["error"]["code"] == 4019

    resp = _remove("../not-a-plugin")
    assert "error" in resp
    assert (plugins_home / "plugins" / "demo-plugin").exists()


@pytest.mark.parametrize("symlink", [False, True], ids=["directory", "symlink"])
def test_remove_drops_only_the_removed_plugins_tools_from_new_chats(tmp_path, monkeypatch, symlink):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.plugins import discover_plugins

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.setenv("HERMES_TUI_TOOLSETS", "remove-probe,keep-probe")
    monkeypatch.setattr(server, "_resolve_agent_model_runtime", lambda *_: (
        "test-model", {"provider": "custom", "base_url": "https://example.invalid/v1", "api_key": "test-key"},
    ))

    plugins = tmp_path / "plugins"
    install = plugins / "remove-probe-install"
    # Nest the symlink target beyond the discovery depth so only the alias loads.
    target = plugins / "sources" / "vendor" / "remove-probe" if symlink else install
    for plugin, name, tool_name in (
        (target, "remove-probe", "removed_probe_tool"),
        (plugins / "keep-probe", "keep-probe", "kept_probe_tool"),
    ):
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text(f"name: {name}\nversion: 1.0.0\n", encoding="utf-8")
        (plugin / "__init__.py").write_text(
            f'''def register(ctx):
    ctx.register_tool(
        name={tool_name!r},
        toolset={name!r},
        schema={{"name": {tool_name!r}, "description": "Test tool",
                "parameters": {{"type": "object", "properties": {{}}}}}},
        handler=lambda args, **kwargs: "{{}}",
    )
''',
            encoding="utf-8",
        )

    if symlink:
        install.symlink_to(target, target_is_directory=True)

    config = load_config()
    config["plugins"]["enabled"] = ["remove-probe", "keep-probe"]
    config["model"] = {"context_length": 128_000}
    config["tools"]["tool_search"]["enabled"] = "off"
    save_config(config)
    discover_plugins(force=True)

    def new_chat_tools():
        response = server.handle_request({"id": "create", "method": "session.create", "params": {"cwd": str(tmp_path)}})
        sid = response["result"]["session_id"]
        try:
            session = server._sessions[sid]
            server._start_agent_build(sid, session)
            assert session["agent_ready"].wait(15)
            assert session["agent"] is not None, session.get("agent_error")
            return set(session["agent"].valid_tool_names)
        finally:
            server.handle_request({"id": "close", "method": "session.close", "params": {"session_id": sid}})

    assert {"removed_probe_tool", "kept_probe_tool"} <= new_chat_tools()
    assert _remove(install.name if symlink else "remove-probe")["result"]["ok"]
    assert not install.exists()
    assert target.exists() == symlink

    tools = new_chat_tools()
    assert "removed_probe_tool" not in tools
    assert "kept_probe_tool" in tools

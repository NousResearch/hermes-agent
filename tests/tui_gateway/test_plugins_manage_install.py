"""Gateway plugins.manage install action."""

from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import server


def test_plugins_manage_install_success():
    # The real ``dashboard_install_plugin`` ok payload, key for key. Under HERMES_TEST_ISOLATION the
    # dispatcher validates the result against ``PluginsManageResult`` (extra=forbid), so a key the
    # installer emits but the contract lacks fails here instead of only in the desktop's errors.log.
    payload = {
        "ok": True,
        "plugin_name": "hello-world",
        "warnings": [],
        "python_dependencies": [],
        "missing_env": [],
        "after_install_path": None,
        "enabled": True,
    }
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value=payload,
    ) as mock_install:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "plugins.manage",
                "params": {
                    "action": "install",
                    "repo": "owner/hello-world",
                    "force": True,
                    "enable": False,
                },
            }
        )

    assert "result" in resp
    assert resp["result"]["plugin_name"] == "hello-world"
    mock_install.assert_called_once_with(
        "owner/hello-world",
        force=True,
        enable=False,
        catalog_name=None,
        ref=None,
    )


def test_plugins_manage_install_missing_identifier():
    resp = server.handle_request(
        {
            "id": "1",
            "method": "plugins.manage",
            "params": {"action": "install"},
        }
    )

    assert "error" in resp
    assert "identifier" in resp["error"]["message"]


def test_plugins_manage_install_failure():
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value={"ok": False, "error": "Git clone failed"},
    ):
        resp = server.handle_request(
            {
                "id": "1",
                "method": "plugins.manage",
                "params": {
                    "action": "install",
                    "identifier": "bad/repo",
                },
            }
        )

    assert "error" in resp
    assert "Git clone failed" in resp["error"]["message"]


def test_plugins_manage_install_catalog_name_only():
    """A catalog pick needs no identifier — the backend resolves repo + pin."""
    payload = {"ok": True, "plugin_name": "weather-plugin", "enabled": False}
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value=payload,
    ) as mock_install:
        resp = server.handle_request(
            {
                "id": "1",
                "method": "plugins.manage",
                "params": {
                    "action": "install",
                    "catalog_name": "weather-plugin",
                    "enable": False,
                },
            }
        )

    assert "result" in resp
    mock_install.assert_called_once_with(
        "",
        force=False,
        enable=False,
        catalog_name="weather-plugin",
        ref=None,
    )


def test_plugins_manage_update_requires_catalog_sidecar(tmp_path, monkeypatch):
    """Non-catalog installs are refused — their update flows stay CLI-owned."""
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_root = tmp_path / "plugins"
    (plugins_root / "plain-git-plugin").mkdir(parents=True)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_root)

    resp = server.handle_request(
        {
            "id": "1",
            "method": "plugins.manage",
            "params": {"action": "update", "name": "plain-git-plugin"},
        }
    )

    assert "error" in resp
    assert "not a catalog install" in resp["error"]["message"]


def test_plugins_manage_list_reports_desktop_half(tmp_path):
    """A unified package (plugin.yaml + desktop/plugin.js) is reported with ``has_desktop_half`` so the
    desktop app can pair its app-level copy of that half with the agent row — one package, ONE row."""
    unified = tmp_path / "media"
    (unified / "desktop").mkdir(parents=True)
    (unified / "desktop" / "plugin.js").write_text("export default {}")
    agent_only = tmp_path / "snap"
    agent_only.mkdir()
    rows = [
        ("media", "1.0", "Media", "user", unified, "media"),
        ("snap", "1.0", "Snap", "user", agent_only, "snap"),
    ]
    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=rows), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}):
        resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "list"}})

    by_name = {r["name"]: r for r in resp["result"]["plugins"]}
    assert by_name["media"]["has_desktop_half"] is True
    assert by_name["snap"]["has_desktop_half"] is False
    assert by_name["snap"]["servers"] == []


def test_plugins_manage_list_reports_declared_server_snapshot(tmp_path):
    plugin_dir = tmp_path / "example-plugin"
    plugin_dir.mkdir()
    package = SimpleNamespace(
        manifest={
            "extensions": {
                "com.nousresearch.hermes": {
                    "servers": {"example-server": {}}
                }
            }
        }
    )
    current = SimpleNamespace(state="missing_app", availability=object())
    liveness = SimpleNamespace(
        status=lambda _name: current,
        describe=lambda _decl, _availability, _state, *, app_name=None: (
            f"{app_name or 'Example App'} is not installed. Install {app_name or 'Example App'}, then try again."
        ),
    )
    rows = [("example-plugin", "1.0", "Example", "user", plugin_dir, "example-plugin")]
    import_module = server._tools_mod

    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=rows), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._is_portable_plugin_dir", return_value=True), \
         patch("hermes_cli.agent_plugins.load_agent_plugin", return_value=package), \
         patch("hermes_platform.declaration.lookup", return_value=object()), \
         patch("tools.mcp_tool_scope._resolve_server_key", return_value="example-plugin__example-server"), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}), \
         patch("tui_gateway.server._tools_mod", wraps=import_module) as modules:
        modules.side_effect = lambda name: liveness if name == "tools.mcp_liveness" else import_module(name)
        resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "list"}})

    assert resp["result"]["plugins"][0]["servers"] == [{
        "name": "example-server",
        "state": "missing_app",
        "sentence": "Example App is not installed. Install Example App, then try again.",
    }]


def test_plugins_manage_list_uses_catalog_title_for_server_status(tmp_path):
    plugin_dir = tmp_path / "example-plugin"
    plugin_dir.mkdir()
    package = SimpleNamespace(
        manifest={"extensions": {"com.nousresearch.hermes": {"servers": {"example-server": {}}}}}
    )
    current = SimpleNamespace(state="mcp_not_connected", availability=object())
    liveness = SimpleNamespace(
        status=lambda _name: current,
        describe=lambda _decl, _availability, _state, *, app_name=None: (
            f"{app_name} is running, but Hermes is not connected to its MCP server."
        ),
    )
    rows = [("example-plugin", "1.0", "Example", "user", plugin_dir, "example-plugin")]
    import_module = server._tools_mod

    with patch("hermes_cli.plugins_cmd._discover_all_plugins", return_value=rows), \
         patch("hermes_cli.plugins_cmd._get_enabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._get_disabled_set", return_value=set()), \
         patch("hermes_cli.plugins_cmd._is_portable_plugin_dir", return_value=True), \
         patch("hermes_cli.agent_plugins.load_agent_plugin", return_value=package), \
         patch("hermes_platform.declaration.lookup", return_value=object()), \
         patch("tools.mcp_tool_scope._resolve_server_key", return_value="example-plugin__example-server"), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_pins", return_value={}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_versions", return_value={}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_titles", return_value={"example-plugin": "Example App"}), \
         patch("hermes_cli.plugins_cmd_catalog.catalog_row_fields", return_value={"catalog_title": "Example App"}), \
         patch("tui_gateway.server._tools_mod", wraps=import_module) as modules:
        modules.side_effect = lambda name: liveness if name == "tools.mcp_liveness" else import_module(name)
        resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "list"}})

    server_row = resp["result"]["plugins"][0]["servers"][0]
    assert server_row["state"] == "mcp_not_connected"
    assert server_row["sentence"] == "Example App is running, but Hermes is not connected to its MCP server."

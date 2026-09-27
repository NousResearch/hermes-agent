"""Gateway plugins.manage install action."""

from unittest.mock import patch

from tui_gateway import server

def test_plugins_manage_install_missing_identifier():
    resp = server.handle_request(
        {
            "id": "1",
            "method": "plugins.manage",
            "params": {"action": "install"},
        }
    )

    assert "error" in resp

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

def test_plugins_manage_list_resolves_the_live_catalog_once_per_listing(tmp_path):
    """The server-sentence display name looks up the curated catalog title, but that lookup must
    cost ONE live-catalog resolution per listing, not one per installed plugin:
    ``load_catalog_live()`` re-fetches or re-parses the whole catalog on every call (no
    memoization), and a dead catalog host costs a request timeout per call — the exact
    per-candidate cost ``resolved_removed_entries()`` exists to eliminate."""
    import json

    import hermes_cli.plugin_catalog as plugin_catalog
    import hermes_cli.plugins_cmd as plugins_cmd
    import hermes_cli.plugins_cmd_catalog as plugins_cmd_catalog
    from hermes_cli.plugin_catalog import PluginCatalogEntry

    rows = []
    for i in range(3):
        plugin_dir = tmp_path / f"plug{i}"
        plugin_dir.mkdir()
        (plugin_dir / "plugin.json").write_text(json.dumps({
            "$schema": "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json",
            "name": f"plug{i}",
        }))
        rows.append((f"plug{i}", "1.0", "Plug", "user", plugin_dir, f"plug{i}"))

    entries = [
        PluginCatalogEntry(
            name=f"example-{i}", repo="https://example.com/repo", sha="0" * 40,
            description="", maintainer="", title=f"Plug {i}")
        for i in range(3)
    ]
    resolver_calls = []

    def counting_load():
        resolver_calls.append(1)
        return entries

    with patch.object(plugins_cmd, "_discover_all_plugins", return_value=rows), \
         patch.object(plugins_cmd, "_get_enabled_set", return_value=set()), \
         patch.object(plugins_cmd, "_get_disabled_set", return_value=set()), \
         patch.object(plugins_cmd_catalog, "catalog_pins", return_value={}), \
         patch.object(plugins_cmd_catalog, "catalog_versions", return_value={}), \
         patch.object(plugins_cmd_catalog, "catalog_install_record",
                      side_effect=lambda d: {"catalog_name": f"example-{d.name.removeprefix('plug')}"}), \
         patch.object(plugin_catalog, "load_catalog_live", side_effect=counting_load), \
         patch.object(plugins_cmd_catalog, "load_catalog_live", side_effect=counting_load):
        resp = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "list"}})

    assert "error" not in resp
    assert len(resp["result"]["plugins"]) == 3
    # ONE resolution for the whole listing (the pre-hoist code paid one per installed plugin).
    assert len(resolver_calls) == 1


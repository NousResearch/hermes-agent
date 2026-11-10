"""Row contracts for the standalone ``plugins.list`` and ``mcp.catalog`` RPCs."""

from types import SimpleNamespace

import tui_gateway.server as srv
from hermes_cli.plugins import LoadedPlugin, PluginManifest


def test_plugins_list_reports_manifest_version_and_enabled_state(monkeypatch):
    import hermes_cli.plugins as plugins_mod

    loaded = {
        "on": LoadedPlugin(manifest=PluginManifest(name="on", version="1.2.3"), enabled=True),
        "off": LoadedPlugin(manifest=PluginManifest(name="off", version="0.4.0"), enabled=False),
    }
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: SimpleNamespace(_plugins=loaded))

    reply = srv._methods["plugins.list"](1, {})

    assert reply["result"]["plugins"] == [
        {"name": "on", "version": "1.2.3", "enabled": True},
        {"name": "off", "version": "0.4.0", "enabled": False},
    ]


def test_mcp_catalog_reports_required_env_keys_and_transport_type():
    reply = srv._methods["mcp.catalog"](1, {})

    asana = next(row for row in reply["result"]["servers"] if row["name"] == "asana")
    assert asana["requires"] == ["ASANA_CLIENT_ID", "ASANA_CLIENT_SECRET"]
    assert asana["transport"] == "http"

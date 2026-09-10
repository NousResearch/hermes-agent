"""Desktop/TUI ``plugins.manage`` honours a ``ref`` pin exactly like ``hermes plugins install --ref``."""
from unittest.mock import patch

from hermes_cli.plugin_catalog import RemovedEntry
from tui_gateway import methods_tools, server


def test_plugins_manage_install_threads_ref_to_headless_install():
    sha = "a" * 40
    with patch("hermes_cli.plugins_cmd.dashboard_install_plugin", return_value={"ok": True}) as install:
        server.handle_request({"id": "1", "method": "plugins.manage",
                               "params": {"action": "install", "identifier": "org/private-plugin", "ref": sha}})
    assert install.call_args.kwargs["ref"] == sha


def test_plugins_manage_list_reports_ref_pin(monkeypatch, tmp_path):
    sha = "b" * 40
    pc = methods_tools._tools_mod("hermes_cli.plugins_cmd")
    monkeypatch.setattr(pc, "_discover_all_plugins", lambda: [("team-plugin", "1.0", "", "git", tmp_path, "team-plugin")])
    monkeypatch.setattr(pc, "_read_install_metadata", lambda: {"team-plugin": {"pinned": True, "revision": sha, "source": "x"}})
    monkeypatch.setattr(pc, "_get_enabled_set", set)
    monkeypatch.setattr(pc, "_get_disabled_set", set)
    monkeypatch.setattr(methods_tools._tools_mod("hermes_cli.plugins_cmd_catalog"), "catalog_pins", dict)
    (row,) = methods_tools._plugin_rows()
    assert row["pinned_sha"] == sha


def test_plugins_manage_list_reports_kill_list_recall(monkeypatch, tmp_path):
    # The CLI (`plugins_cmd.cmd_list`) and web dashboard (`web_server_dashboard.py`) both surface
    # `removed_annotation()` for a still-installed, catalog-recalled plugin; the desktop/TUI
    # `plugins.manage list` row builder (`_plugin_rows`) had no equivalent field at all.
    pc = methods_tools._tools_mod("hermes_cli.plugins_cmd")
    monkeypatch.setattr(
        pc, "_discover_all_plugins", lambda: [("killed-plugin", "1.0", "", "git", tmp_path, "killed-plugin")])
    monkeypatch.setattr(pc, "_read_install_metadata", dict)
    monkeypatch.setattr(pc, "_get_enabled_set", set)
    monkeypatch.setattr(pc, "_get_disabled_set", set)
    monkeypatch.setattr(methods_tools._tools_mod("hermes_cli.plugins_cmd_catalog"), "catalog_pins", dict)
    monkeypatch.setattr(
        "hermes_cli.plugin_catalog.load_removed_list",
        lambda catalog_dir=None: [RemovedEntry(name="killed-plugin", reason="malware found in a dependency")],
    )
    monkeypatch.setattr("hermes_cli.plugin_catalog.fetch_live_catalog", lambda **_: None)
    (row,) = methods_tools._plugin_rows()
    assert row["removed_reason"] == "malware found in a dependency"


def test_plugins_manage_list_omits_removed_reason_for_a_healthy_plugin(monkeypatch, tmp_path):
    pc = methods_tools._tools_mod("hermes_cli.plugins_cmd")
    monkeypatch.setattr(
        pc, "_discover_all_plugins", lambda: [("healthy-plugin", "1.0", "", "git", tmp_path, "healthy-plugin")])
    monkeypatch.setattr(pc, "_read_install_metadata", dict)
    monkeypatch.setattr(pc, "_get_enabled_set", set)
    monkeypatch.setattr(pc, "_get_disabled_set", set)
    monkeypatch.setattr(methods_tools._tools_mod("hermes_cli.plugins_cmd_catalog"), "catalog_pins", dict)
    monkeypatch.setattr("hermes_cli.plugin_catalog.load_removed_list", lambda catalog_dir=None: [])
    monkeypatch.setattr("hermes_cli.plugin_catalog.fetch_live_catalog", lambda **_: None)
    (row,) = methods_tools._plugin_rows()
    assert "removed_reason" not in row

"""Desktop/TUI ``plugins.manage`` honours a ``ref`` pin exactly like ``hermes plugins install --ref``."""
from unittest.mock import patch

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

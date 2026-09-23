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
    assert "error" in resp and "not found" in resp["error"]["message"]
    assert (plugins_home / "plugins" / "demo-plugin").exists()

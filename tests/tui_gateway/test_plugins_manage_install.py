"""Gateway plugins.manage install action."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tui_gateway import server


def test_plugins_manage_install_success():
    payload = {
        "ok": True,
        "plugin_name": "hello-world",
        "warnings": [],
        "missing_env": [],
        "after_install_path": None,
        "enabled": True,
    }
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value=payload,
    ) as mock_install:
        resp = server.handle_request({
            "id": "1",
            "method": "plugins.manage",
            "params": {
                "action": "install",
                "repo": "owner/hello-world",
                "force": True,
                "enable": False,
            },
        })

    assert "result" in resp
    assert resp["result"]["plugin_name"] == "hello-world"
    mock_install.assert_called_once_with(
        "owner/hello-world",
        force=True,
        enable=False,
        catalog_name=None,
        catalog_source="official",
    )


def test_plugins_manage_install_missing_identifier():
    resp = server.handle_request({
        "id": "1",
        "method": "plugins.manage",
        "params": {"action": "install"},
    })

    assert "error" in resp
    assert "identifier" in resp["error"]["message"]


def test_plugins_manage_install_failure():
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value={"ok": False, "error": "Git clone failed"},
    ):
        resp = server.handle_request({
            "id": "1",
            "method": "plugins.manage",
            "params": {
                "action": "install",
                "identifier": "bad/repo",
            },
        })

    assert "error" in resp
    assert "Git clone failed" in resp["error"]["message"]


def test_plugins_manage_install_catalog_name_only():
    """A catalog pick needs no identifier — the backend resolves repo + pin."""
    payload = {"ok": True, "plugin_name": "weather-plugin", "enabled": False}
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value=payload,
    ) as mock_install:
        resp = server.handle_request({
            "id": "1",
            "method": "plugins.manage",
            "params": {
                "action": "install",
                "catalog_name": "weather-plugin",
                "enable": False,
            },
        })

    assert "result" in resp
    mock_install.assert_called_once_with(
        "",
        force=False,
        enable=False,
        catalog_name="weather-plugin",
        catalog_source="official",
    )


def test_plugins_manage_install_private_marketplace_entry():
    payload = {"ok": True, "plugin_name": "private-plugin", "enabled": True}
    with patch(
        "hermes_cli.plugins_cmd.dashboard_install_plugin",
        return_value=payload,
    ) as mock_install:
        resp = server.handle_request({
            "id": "1",
            "method": "plugins.manage",
            "params": {
                "action": "install",
                "marketplace_id": "abc123",
                "marketplace_plugin_name": "private-plugin",
            },
        })

    assert "result" in resp
    mock_install.assert_called_once_with(
        "",
        force=False,
        enable=True,
        catalog_name="private-plugin",
        catalog_source="abc123",
    )


def test_plugins_manage_marketplace_actions():
    marketplace = {
        "available": True,
        "entries": [
            {
                "compatible": True,
                "description": "Private plugin",
                "display_name": "Private Plugin",
                "incompatibility_reason": "",
                "maintainer": "Team",
                "name": "private-plugin",
                "repo": "https://github.com/o/r",
                "sha": "a" * 40,
                "source_id": "abc123",
                "source_name": "Private",
                "subdir": "plugins/private-plugin",
                "tree_sha": "b" * 40,
                "version": "1.0.0",
            }
        ],
        "id": "abc123",
        "name": "Private",
        "stale": False,
        "url": "https://github.com/o/r",
    }
    with (
        patch(
            "hermes_cli.plugin_marketplaces.add_marketplace",
            return_value=marketplace,
        ) as mock_add,
        patch(
            "hermes_cli.plugin_marketplaces.list_marketplaces",
            return_value=[marketplace],
        ) as mock_list,
        patch(
            "hermes_cli.plugin_marketplaces.remove_marketplace",
            return_value=True,
        ) as mock_remove,
    ):
        added = server.handle_request({
            "id": "1",
            "method": "plugins.manage",
            "params": {"action": "marketplace_add", "url": "https://github.com/o/r"},
        })
        listed = server.handle_request({
            "id": "2",
            "method": "plugins.manage",
            "params": {"action": "marketplace_refresh"},
        })
        removed = server.handle_request({
            "id": "3",
            "method": "plugins.manage",
            "params": {"action": "marketplace_remove", "source_id": "abc123"},
        })

    public = added["result"]["marketplace"]
    assert public["id"] == "abc123"
    assert public["entries"][0]["name"] == "private-plugin"
    assert "repo" not in public["entries"][0]
    assert "sha" not in public["entries"][0]
    assert "subdir" not in public["entries"][0]
    assert listed["result"]["marketplaces"] == [public]
    assert removed["result"]["removed"] is True
    mock_add.assert_called_once_with("https://github.com/o/r")
    mock_list.assert_called_once_with(force=True)
    mock_remove.assert_called_once_with("abc123")


def test_marketplace_provenance_does_not_leak_to_same_basename_key(
    tmp_path: Path, monkeypatch
):
    import hermes_cli.plugin_catalog as catalog
    import hermes_cli.plugin_marketplaces as marketplaces
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_root = tmp_path / "plugins"
    top = plugins_root / "foo"
    nested = plugins_root / "category" / "foo"
    top.mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_root)
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_all_plugins",
        lambda: [
            ("foo", "1", "top", "user", top, "foo"),
            ("foo", "1", "nested", "user", nested, "category/foo"),
        ],
    )
    monkeypatch.setattr(
        plugins_cmd,
        "_read_install_metadata",
        lambda: {
            "foo": {
                "marketplace_id": "market",
                "marketplace_name": "Private",
                "marketplace_plugin_name": "foo",
                "installed_repo_sha": "a" * 40,
                "installed_tree_sha": "b" * 40,
            }
        },
    )
    monkeypatch.setattr(catalog, "load_catalog_live", lambda: [])
    monkeypatch.setattr(
        marketplaces,
        "list_marketplaces",
        lambda **_kwargs: [
            {
                "id": "market",
                "entries": [{"name": "foo", "sha": "a" * 40, "tree_sha": "b" * 40}],
            }
        ],
    )

    listed = server.handle_request({
        "id": "list",
        "method": "plugins.manage",
        "params": {"action": "list"},
    })
    assert listed is not None
    rows = {row["key"]: row for row in listed["result"]["plugins"]}
    assert rows["foo"]["marketplace_id"] == "market"
    assert "marketplace_id" not in rows["category/foo"]

    updated = server.handle_request({
        "id": "update",
        "method": "plugins.manage",
        "params": {"action": "update", "key": "category/foo"},
    })
    assert updated is not None
    assert "not a catalog or marketplace install" in updated["error"]["message"]


def test_private_registry_failure_keeps_official_update_pin(tmp_path, monkeypatch):
    import hermes_cli.plugin_catalog as catalog
    import hermes_cli.plugin_marketplaces as marketplaces
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_root = tmp_path / "plugins"
    target = plugins_root / "official-plugin"
    target.mkdir(parents=True)
    (target / ".hermes-catalog.json").write_text(
        json.dumps({
            "catalog_name": "official-plugin",
            "sha": "a" * 40,
            "tier": "official",
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_root)
    monkeypatch.setattr(
        plugins_cmd,
        "_discover_all_plugins",
        lambda: [
            ("official-plugin", "1", "official", "user", target, "official-plugin")
        ],
    )
    monkeypatch.setattr(plugins_cmd, "_read_install_metadata", lambda: {})
    monkeypatch.setattr(
        catalog,
        "load_catalog_live",
        lambda: [SimpleNamespace(name="official-plugin", sha="b" * 40)],
    )
    monkeypatch.setattr(
        marketplaces,
        "list_marketplaces",
        lambda: (_ for _ in ()).throw(ValueError("malformed private registry")),
    )

    listed = server.handle_request({
        "id": "list",
        "method": "plugins.manage",
        "params": {"action": "list"},
    })

    assert listed is not None
    row = listed["result"]["plugins"][0]
    assert row["catalog_sha"] == "b" * 40
    assert row["update_available"] is True


def test_plugins_manage_update_requires_catalog_sidecar(tmp_path, monkeypatch):
    """Non-catalog installs are refused — their update flows stay CLI-owned."""
    import hermes_cli.plugins_cmd as plugins_cmd

    plugins_root = tmp_path / "plugins"
    (plugins_root / "plain-git-plugin").mkdir(parents=True)
    monkeypatch.setattr(plugins_cmd, "_plugins_dir", lambda: plugins_root)

    resp = server.handle_request({
        "id": "1",
        "method": "plugins.manage",
        "params": {"action": "update", "key": "plain-git-plugin"},
    })

    assert "error" in resp
    assert "not a catalog or marketplace install" in resp["error"]["message"]

"""Gateway plugins.manage install action."""

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

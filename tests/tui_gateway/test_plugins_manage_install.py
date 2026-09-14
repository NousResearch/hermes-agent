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
        ref=None,
        allow_caution=False,
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
    assert "data" not in resp["error"]


def test_plugins_manage_install_scan_block_carries_verdict_and_findings():
    """A scan block is an error whose ``data`` carries the verdict + findings, so the desktop dialog can
    render them and offer "Install anyway" for ``caution`` instead of a dead end."""
    finding = {
        "pattern_id": "sudo_usage", "severity": "high", "category": "privilege_escalation",
        "file": "setup.sh", "line": 3, "description": "uses sudo (privilege escalation)",
    }
    blocked = {
        "ok": False, "error": "Security scan blocked plugin install: Requires confirmation",
        "scan_blocked": True, "scan_verdict": "caution", "scan_findings": [finding],
    }
    with patch("hermes_cli.plugins_cmd.dashboard_install_plugin", return_value=blocked):
        resp = server.handle_request(
            {"id": "1", "method": "plugins.manage", "params": {"action": "install", "identifier": "owner/plugin"}}
        )

    assert resp["error"]["code"] == 5026
    assert "Requires confirmation" in resp["error"]["message"]
    assert resp["error"]["data"] == {"scan_blocked": True, "scan_verdict": "caution", "scan_findings": [finding]}


def test_plugins_manage_install_allow_caution_threads_through():
    """"Install anyway" accepts the caution verdict only; it is not a ``force`` (which also replaces an
    existing install)."""
    with patch("hermes_cli.plugins_cmd.dashboard_install_plugin", return_value={"ok": True}) as install:
        server.handle_request(
            {"id": "1", "method": "plugins.manage",
             "params": {"action": "install", "identifier": "owner/plugin", "allow_caution": True}}
        )

    assert install.call_args.kwargs["allow_caution"] is True
    assert install.call_args.kwargs["force"] is False


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
        allow_caution=False,
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

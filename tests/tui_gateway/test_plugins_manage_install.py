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


def test_marketplaces_rpc_is_profile_scoped_and_does_not_install(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli import plugin_marketplaces as market

    launch = tmp_path / "launch"
    worker = tmp_path / "worker"
    launch.mkdir()
    worker.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if name == "worker" else None)
    source = {"id": market._source_id("https://example.com/team.git"),
              "name": "Team", "url": "https://example.com/team.git"}
    token = set_hermes_home_override(str(worker))
    try:
        market._write_registry([source])
        market._write_cache(source, [])
    finally:
        reset_hermes_home_override(token)

    def call(action, **kwargs):
        return server.handle_request({"id": "m", "method": "plugins.manage",
                                      "params": {"action": action, **kwargs}})

    assert call("marketplaces")["result"]["marketplaces"] == []
    rows = call("marketplaces", profile="worker")["result"]["marketplaces"]
    assert rows[0]["id"] == source["id"]
    assert rows[0]["entries"] == []
    assert not (worker / "plugins").exists()
    assert call("marketplace_remove", source_id=source["id"], profile="worker")["result"]["removed"]
    assert call("marketplaces", profile="worker")["result"]["marketplaces"] == []


def test_marketplace_update_requires_sha_bound_capability_consent(tmp_path, monkeypatch):
    from hermes_cli import plugin_marketplaces as market, plugins_cmd as pc, plugins_cmd_catalog as cat

    target = tmp_path / "plugins" / "demo"
    target.mkdir(parents=True)
    entry = {"source_id": "source", "source_name": "Team", "name": "demo",
             "repo": "https://example.com/repo.git", "subdir": "demo", "sha": "b" * 40,
             "tree_sha": "c" * 40, "compatible": True}
    monkeypatch.setattr(pc, "_plugins_dir", lambda: target.parent)
    monkeypatch.setattr(pc, "_canonical_source", lambda repo, subdir: f"{repo}#{subdir}")
    monkeypatch.setattr(pc, "_read_install_metadata", lambda: {"demo": {
        "source": f"{entry['repo']}#{entry['subdir']}",
        "revision": "a" * 40, "marketplace": {"id": "source", "plugin_name": "demo", "tree_sha": "d" * 40}}})
    monkeypatch.setattr(cat, "catalog_install_record", lambda _: None)
    monkeypatch.setattr(cat, "_local_changes", lambda _: ([], []))
    monkeypatch.setattr(cat, "_carry_user_files", lambda *a: [])
    monkeypatch.setattr(market, "get_marketplace_entry", lambda *a, **k: entry)
    monkeypatch.setattr(pc, "_resolve_git_url", lambda _: (entry["repo"], "demo"))
    monkeypatch.setattr(pc, "_clone_plugin_repo", lambda *a: entry["sha"])
    monkeypatch.setattr(pc, "_resolve_subdir_within", lambda root, subdir: root / subdir)
    monkeypatch.setattr(pc, "_read_manifest_for_install", lambda _: {"name": "demo"})
    monkeypatch.setattr(pc, "_read_manifest", lambda _: {"name": "demo"})
    monkeypatch.setattr(market, "_git", lambda *a: entry["tree_sha"])
    monkeypatch.setattr(cat, "plugin_surface", lambda manifest, path: path)
    monkeypatch.setattr(cat, "surface_delta", lambda *a: {"tools": ["new_tool"]})
    monkeypatch.setattr(cat, "surface_delta_lines", lambda _: ["Tools: new_tool"])
    installed = []
    monkeypatch.setattr(pc, "_install_plugin_core", lambda *a, **k: installed.append(k))

    def update(**extra):
        return server.handle_request({"id": "u", "method": "plugins.manage",
                                      "params": {"action": "update", "name": "demo", **extra}})

    with patch.object(pc, "_canonical_source", return_value="different source"):
        assert "error" in update()
    assert update()["result"]["consent_required"] is True
    second = update(accept_capabilities=True, ref="a" * 40)
    assert second.get("result", {}).get("consent_required") is True, second
    assert installed == []
    monkeypatch.setattr(cat, "raise_if_removed", lambda *a: None)
    monkeypatch.setattr(server, "_ensure_plugin_activation_listener", lambda: None, raising=False)
    with patch("hermes_cli.plugins_activation.activate_plugin_now", return_value={}):
        assert update(accept_capabilities=True, ref=entry["sha"])["result"]["ok"] is True
    assert installed[0]["ref"] == entry["sha"]
    assert installed[0]["scan_force"] is False
    assert installed[0]["force"] is True
    assert installed[0]["marketplace"] is entry


def test_marketplace_install_routes_selected_profile_and_identifiers(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home
    from hermes_cli import plugins_cmd

    worker = tmp_path / "worker"
    worker.mkdir()
    monkeypatch.setattr(server, "_profile_home", lambda name: worker if name == "worker" else None)
    seen = []

    def install(identifier, **kwargs):
        seen.append((get_hermes_home(), identifier, kwargs))
        return {"ok": True, "plugin_name": "demo", "enabled": False}

    monkeypatch.setattr(plugins_cmd, "dashboard_install_plugin", install)
    response = server.handle_request({"id": "install", "method": "plugins.manage", "params": {
        "action": "install", "identifier": "", "marketplace_id": "a" * 16,
        "marketplace_plugin_name": "demo", "profile": "worker", "enable": False,
    }})
    assert response["result"]["ok"] is True
    assert seen[0][0] == worker
    assert seen[0][1] == ""
    assert seen[0][2]["marketplace_id"] == "a" * 16
    assert seen[0][2]["marketplace_plugin_name"] == "demo"


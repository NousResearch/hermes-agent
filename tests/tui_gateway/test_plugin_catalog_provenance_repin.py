"""A catalog re-pin reports the repository and commit that were installed."""
import json
import shutil
import subprocess

from tui_gateway import server


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_catalog_subdir_repin_reports_new_source(tmp_path, monkeypatch):
    from hermes_cli import plugin_catalog, plugins_cmd, plugins_cmd_catalog
    from hermes_cli.plugins_manifest import SUPPORTED_MANIFEST_VERSION

    home = tmp_path / "home"
    plugins = home / "plugins"
    plugins.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    repos = []
    shas = []
    for name in ("old", "new"):
        repo = tmp_path / name
        repo.mkdir()
        git(repo, "init", "-qb", "main")
        git(repo, "config", "user.name", "Fixture")
        git(repo, "config", "user.email", "fixture@example.invalid")
        package = repo / "packages" / "sample"
        package.mkdir(parents=True)
        (package / "plugin.yaml").write_text(f"name: sample\nversion: 1.0.0\nmanifest_version: {SUPPORTED_MANIFEST_VERSION}\n")
        (package / "__init__.py").write_text(f"VERSION = {name!r}\ndef register(ctx):\n    pass\n")
        (package / "plugin.js").write_text("export default {}")
        git(repo, "add", ".")
        git(repo, "commit", "-qm", "initial")
        repos.append(repo)
        shas.append(git(repo, "rev-parse", "HEAD"))
    target = plugins / "sample"
    shutil.copytree(repos[0] / "packages" / "sample", target)
    (plugins / ".install-metadata.json").write_text(json.dumps({"sample": {
        "source": repos[0].as_uri() + "#packages/sample", "revision": shas[0], "pinned": True,
        "catalog": {"name": "sample", "repo": repos[0].as_uri(), "sha": shas[0], "pin": shas[0]},
    }}))
    entry = plugin_catalog.PluginCatalogEntry(name="sample", repo=repos[1].as_uri(), subdir="packages/sample",
                                               sha=shas[1], description="fixture", maintainer="fixture")
    monkeypatch.setattr(plugins_cmd_catalog, "get_live_catalog_entry", lambda name: entry)
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **kwargs: None)
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda catalog_dir=None: [])
    monkeypatch.setattr(plugins_cmd, "_scan_on_install_enabled", lambda: False)
    monkeypatch.setattr("hermes_cli.plugins_activation.activate_plugin_now", lambda name: {})
    def publish(staged, target, old_metadata, new_metadata, **kwargs):
        shutil.rmtree(target)
        shutil.copytree(staged, target, symlinks=True)
        (target.parent / ".install-metadata.json").write_text(json.dumps(new_metadata))
    monkeypatch.setattr("hermes_cli.plugins_transaction.publish_plugin", publish)
    response = server.handle_request({"id": "1", "method": "plugins.manage", "params": {"action": "update", "name": "sample"}})
    assert "error" not in response, response
    result = response["result"]
    record = json.loads((plugins / ".install-metadata.json").read_text())["sample"]
    assert record["revision"] == shas[1]
    assert record["source"] == repos[1].as_uri() + "#packages/sample"
    assert result["installed_repo"] == repos[1].as_uri()
    assert result["installed_subdir"] == "packages/sample"
    assert result["installed_sha"] == shas[1]
    assert "new" in (target / "__init__.py").read_text()

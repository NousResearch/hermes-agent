"""Real local Git updates report the commit actually published by the RPC."""
import json
import shutil
import subprocess

import pytest

from tui_gateway import server


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def rpc(**params):
    response = server.handle_request({"id": "1", "method": "plugins.manage", "params": params})
    assert "error" not in response, response
    return response["result"]


@pytest.mark.parametrize("batch", [False, True])
def test_real_git_update_returns_published_sha(tmp_path, monkeypatch, batch):
    from hermes_cli import plugin_catalog, plugins_cmd, plugins_updates
    from hermes_cli.plugins_manifest import SUPPORTED_MANIFEST_VERSION

    root, home = tmp_path, tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = root / "origin"
    repo.mkdir()
    git(repo, "init", "-qb", "main")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "plugin.yaml").write_text(f"name: sample\nversion: 1.0.0\nmanifest_version: {SUPPORTED_MANIFEST_VERSION}\n")
    (repo / "__init__.py").write_text("def register(ctx):\n    pass\n")
    (repo / "plugin.js").write_text("export default {}")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "initial")
    old_sha = git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(plugin_catalog, "fetch_live_catalog", lambda **kwargs: None)
    monkeypatch.setattr(plugin_catalog, "load_removed_list", lambda catalog_dir=None: [])
    monkeypatch.setattr(plugins_cmd, "_scan_on_install_enabled", lambda: False)
    monkeypatch.setattr("hermes_cli.plugins_activation.activate_plugin_now", lambda name: {})
    # PM environment builds are outside this Git/RPC integration and need network provisioning.
    def publish(staged, target, old_metadata, new_metadata, **kwargs):
        shutil.rmtree(target)
        shutil.copytree(staged, target, symlinks=True)
        (target.parent / ".install-metadata.json").write_text(json.dumps(new_metadata))
    monkeypatch.setattr("hermes_cli.plugins_transaction.publish_plugin", publish)
    plugins = home / "plugins"
    plugins.mkdir()
    target = plugins / "sample"
    subprocess.run(["git", "clone", "-q", repo.as_uri(), str(target)], check=True)
    (plugins / ".install-metadata.json").write_text(json.dumps({"sample": {"source": repo.as_uri(), "revision": old_sha}}))
    (repo / "__init__.py").write_text("VERSION = 2\ndef register(ctx):\n    pass\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "second")
    new_sha = git(repo, "rev-parse", "HEAD")
    assert old_sha != new_sha
    if batch:
        class Row:
            name = "sample"
            klass = "git"
            update_available = True
            needs_fixing = reason = min_hermes = None
            current, latest = old_sha, new_sha
        monkeypatch.setattr(plugins_updates, "run_checks", lambda _: [Row()])
        result = rpc(action="update_all")["results"][0]
        assert result["status"] == "updated", result
    else:
        result = rpc(action="update", name="sample")
        assert result["unchanged"] is False
    assert git(target, "rev-parse", "HEAD") == new_sha
    assert json.loads((home / "plugins" / ".install-metadata.json").read_text())["sample"]["revision"] == new_sha
    assert result["installed_sha"] == new_sha
    assert result["installed_repo"] == repo.as_uri()
    assert result["installed_subdir"] == ""
    assert result["has_desktop_half"] is True

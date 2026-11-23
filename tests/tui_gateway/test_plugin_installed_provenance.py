"""Installed plugin provenance is taken from installer records and the actual checkout."""
import json
import subprocess

import pytest

from tui_gateway import server


def rpc(**params):
    response = server.handle_request({"id": "1", "method": "plugins.manage", "params": params})
    assert "error" not in response, response
    return response["result"]


@pytest.mark.parametrize("catalog", [False, True])
def test_list_reports_installed_source_and_actual_revision(tmp_path, monkeypatch, catalog):
    from hermes_cli import plugins_cmd, plugins_cmd_catalog

    root = tmp_path / "plugins"
    target = root / "sample"
    target.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(target), "config", "user.name", "Test"], check=True)
    (target / "plugin.yaml").write_text("name: sample\nversion: 1.0\n")
    (target / "plugin.js").write_text("export default {}")
    subprocess.run(["git", "-C", str(target), "add", "."], check=True)
    subprocess.run(["git", "-C", str(target), "commit", "-qm", "initial"], check=True)
    sha = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    source = "https://user:secret@example.invalid/owner/repo.git#packages/sample"
    record = {"source": source, "revision": "a" * 40}
    if catalog:
        record["catalog"] = {"name": "sample", "repo": "https://wrong.invalid/repo", "sha": "b" * 40}
    (root / ".install-metadata.json").write_text(json.dumps({"sample": record}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugins_cmd, "_discover_all_plugins", lambda: [("sample", "1", "", "user", target, "sample")])
    monkeypatch.setattr(plugins_cmd_catalog, "catalog_rows_maps", lambda: ({}, {}, {}))
    row = rpc(action="list")["plugins"][0]
    assert row["installed_repo"] == "https://example.invalid/owner/repo.git"
    assert row["installed_subdir"] == "packages/sample"
    assert row["installed_sha"] == sha
    assert row["has_desktop_half"] is True


def test_subdir_record_and_unsupported_manual_install(tmp_path, monkeypatch):
    from hermes_cli.plugins_cmd import installed_source_fields

    root = tmp_path / "plugins"
    subdir = root / "sample"
    manual = root / "manual"
    subdir.mkdir(parents=True)
    manual.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sha = "1" * 40
    (root / ".install-metadata.json").write_text(json.dumps({
        "sample": {"source": "https://example.invalid/repo.git#plugins/sample", "revision": sha},
    }))
    (subdir / ".hermes-catalog.json").write_text(json.dumps({"repo": "https://forged.invalid", "sha": "2" * 40}))
    assert installed_source_fields(subdir) == {"installed_repo": "https://example.invalid/repo.git",
        "installed_subdir": "plugins/sample", "installed_sha": sha, "has_desktop_half": False}
    assert installed_source_fields(manual) == {}


@pytest.mark.parametrize("source,revision,expected", [
    ("https://alice:secret@example.invalid/repo.git#pkg", "a" * 40, "https://example.invalid/repo.git"),
    ("ssh://alice:secret@example.invalid/repo.git#pkg", "a" * 40, None),
    ("https://example.invalid/repo.git#pkg", "short", None),
    ("https://example.invalid/repo.git", "a" * 40, None),
    ("invalid", "a" * 40, None),
])
def test_source_validation(tmp_path, monkeypatch, source, revision, expected):
    from hermes_cli.plugins_cmd import installed_source_fields

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "plugins" / "sample"
    target.mkdir(parents=True)
    (target.parent / ".install-metadata.json").write_text(json.dumps({"sample": {"source": source, "revision": revision}}))
    fields = installed_source_fields(target)
    if expected is None:
        assert fields == {}
    else:
        assert fields["installed_repo"] == expected
        assert "secret" not in str(fields)


def test_update_and_batch_return_installed_provenance(tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd_update, plugins_updates
    from hermes_cli import plugins_cmd_catalog

    root = tmp_path / "plugins"
    target = root / "sample"
    target.mkdir(parents=True)
    (target / "plugin.yaml").write_text("name: sample\nversion: 1.0\n")
    (target / "plugin.js").write_text("export default {}")
    sha = "3" * 40
    (root / ".install-metadata.json").write_text(json.dumps({"sample": {
        "source": "https://example.invalid/repo.git#plugins/sample", "revision": sha}}))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugins_cmd_catalog, "catalog_install_record", lambda target: None)
    monkeypatch.setattr(plugins_cmd_update, "_pull_plugin_update", lambda *args, **kwargs: "Updated")
    monkeypatch.setattr(plugins_cmd_update, "_post_pull_housekeeping", lambda *args: None)
    monkeypatch.setattr("hermes_cli.plugins_activation.activate_plugin_now", lambda name: {})
    expected = {"installed_repo": "https://example.invalid/repo.git", "installed_subdir": "plugins/sample",
                "installed_sha": sha, "has_desktop_half": True}
    result = rpc(action="update", name="sample")
    assert {k: result[k] for k in expected} == expected

    class Row:
        name = "sample"
        klass = "git"
        update_available = True
        needs_fixing = reason = min_hermes = None
        current, latest = "1" * 40, sha

    monkeypatch.setattr(plugins_updates, "run_checks", lambda _: [Row()])
    result = rpc(action="update_all")
    row = result["results"][0]
    assert row["status"] == "updated"
    assert {k: row[k] for k in expected} == expected

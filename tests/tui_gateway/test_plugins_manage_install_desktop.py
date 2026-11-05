"""Gateway ``plugins.manage install_desktop`` / ``remove_desktop``: the browser-hosted app's install.

Drives the real RPC dispatch and a real clone of a local repository (only the clone's network leg is
pointed at it): a desktop plugin lands in the host's ``desktop-plugins/`` at the reviewed commit, a
second install is an RPC error until ``force``, and removal round-trips — all through the same
``plugins.manage`` action the dialog calls.
"""

import subprocess

from tui_gateway import server


def _manage(**params):
    return server.handle_request({"id": "1", "method": "plugins.manage", "params": params})


def test_install_desktop_round_trips_through_the_rpc(tmp_path, monkeypatch):
    from hermes_cli import plugins_cmd
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    repo = tmp_path / "gizmo"
    repo.mkdir()
    (repo / "plugin.js").write_text("export default { id: 'gizmo' }\n", encoding="utf-8")
    for args in (("init", "-q", "-b", "main"), ("add", "-A"),
                 ("-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "c")):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True,
                         text=True, encoding="utf-8").stdout.strip()
    real = plugins_cmd._clone_plugin_repo
    monkeypatch.setattr(plugins_cmd, "_clone_plugin_repo",
                        lambda dest, _url, revision, subdir=None: real(dest, repo.as_uri(), revision, subdir))
    source = {"identifier": "https://example.invalid/acme/gizmo.git", "ref": sha}

    result = _manage(action="install_desktop", **source)["result"]
    assert (result["ok"], result["plugin_name"], result["sha"]) == (True, "gizmo", sha)
    assert (home / "desktop-plugins" / "gizmo" / "plugin.js").is_file()

    refused = _manage(action="install_desktop", **source)
    assert refused["error"]["code"] == 5026 and "already exists" in refused["error"]["message"]
    assert "data" not in refused["error"]  # an ordinary refusal: the client offers Retry, not a re-review

    assert _manage(action="remove_desktop", name="gizmo")["result"]["ok"] is True
    assert not (home / "desktop-plugins" / "gizmo").exists()


def test_install_desktop_requires_a_source():
    assert _manage(action="install_desktop")["error"]["code"] == 4019

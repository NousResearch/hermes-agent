"""#123333: the pm store's pinned npm must resolve before node's bundled copy.

On Windows node ships its own npm shims in the same dir as ``node.exe``, so a
PATH that puts the node entry before the npm entry makes every ``npm ci`` /
``npm install`` from a Hermes terminal die with EBADENGINE under
engine-strict. ``_managed_runtime_path_entries`` therefore orders the
requested package's own bin dir before its dependencies' — regardless of the
order ``pm.env_for`` happened to compose.
"""

import os
import sys
import types

from tools.environments import local as local_mod
from tools.environments.local import (
    _dependents_first_path_dirs,
    _managed_runtime_path_entries,
)


def test_dependents_first_floats_package_dir_above_dependency(monkeypatch, tmp_path):
    node = tmp_path / "node-26.7.0-win32-x64"
    npm = tmp_path / "npm-12.0.2-win32-x64"
    git = tmp_path / "git-cmd"
    for d in (node, npm, git):
        d.mkdir(parents=True)

    seen_names = {}

    def fake_own(names):
        seen_names["names"] = list(names)
        # Dependents-first: the requested package's own dir, then its deps'.
        return [str(npm), str(node)]

    monkeypatch.setattr(local_mod, "_package_own_path_dirs", fake_own)

    # Bug shape from #123333: the dependency's dir precedes the package's own.
    out = _dependents_first_path_dirs(["npm"], [node, git, npm])

    assert seen_names["names"] == ["npm"]
    assert out[0] == npm
    assert out[1] == node
    assert out[2] == git
    assert len(out) == 3


def test_dependents_first_keeps_composed_order_when_already_correct(monkeypatch, tmp_path):
    node = tmp_path / "node-26.7.0-win32-x64"
    npm = tmp_path / "npm-12.0.2-win32-x64"

    monkeypatch.setattr(local_mod, "_package_own_path_dirs", lambda names: [str(npm), str(node)])

    out = _dependents_first_path_dirs(["npm"], [npm, node])
    assert [str(p) for p in out] == [str(npm), str(node)]


def test_dependents_first_falls_back_to_input_order_on_failure(monkeypatch, tmp_path):
    node = tmp_path / "node-26.7.0-win32-x64"
    npm = tmp_path / "npm-12.0.2-win32-x64"

    def boom(names):
        raise RuntimeError("store unreadable")

    monkeypatch.setattr(local_mod, "_package_own_path_dirs", boom)

    out = _dependents_first_path_dirs(["npm"], [node, npm])
    assert [str(p) for p in out] == [str(node), str(npm)]


def test_managed_runtime_entries_put_pinned_npm_first(monkeypatch, tmp_path):
    node = tmp_path / "node-26.7.0-win32-x64"
    npm = tmp_path / "npm-12.0.2-win32-x64"
    for d in (node, npm):
        d.mkdir(parents=True)

    # pm.env_for returns the bug shape: node's dir shadows the pinned npm.
    fake_pm = types.SimpleNamespace(
        env_for=lambda *names, base_env=None: {"PATH": os.pathsep.join([str(node), str(npm)])}
    )
    monkeypatch.setitem(sys.modules, "pm", fake_pm)
    monkeypatch.setattr(local_mod, "_package_own_path_dirs", lambda names: [str(npm), str(node)])
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)

    entries = _managed_runtime_path_entries()

    assert entries[0] == str(npm)
    assert entries[1] == str(node)
    assert len(entries) == 2

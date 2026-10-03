"""Terminal PATH assembly for Hermes-managed runtimes (Windows npm shadowing).

Regression tests for #123333: the node dir ships its own npm, so listed
first it shadows the pinned store npm and EBADENGINEs under engine-strict.
Store-npm dirs (npm entry point, no node.exe) must sort ahead.
"""
import os

import pytest

import pm


@pytest.mark.platforms("windows")
def test_store_npm_sorts_ahead_of_bundled_npm(tmp_path, monkeypatch):
    from tools.environments import local as local_mod

    node_dir = tmp_path / "node-26.7.0-win32-x64"
    node_dir.mkdir()
    (node_dir / "node.exe").write_bytes(b"x")
    (node_dir / "npm.cmd").write_bytes(b"x")
    npm_dir = tmp_path / "npm-12.0.2-win32-x64"
    npm_dir.mkdir()
    (npm_dir / "npm.cmd").write_bytes(b"x")
    monkeypatch.setattr(
        pm, "env_for",
        lambda *args, **kwargs: {"PATH": os.pathsep.join([str(node_dir), str(npm_dir)])},
    )
    entries = local_mod._managed_runtime_path_entries()
    assert entries.index(str(npm_dir)) < entries.index(str(node_dir))


@pytest.mark.platforms("windows")
def test_unrelated_dirs_keep_relative_order(tmp_path, monkeypatch):
    from tools.environments import local as local_mod

    first = tmp_path / "aaa"
    first.mkdir()
    second = tmp_path / "zzz"
    second.mkdir()
    monkeypatch.setattr(
        pm, "env_for",
        lambda *args, **kwargs: {"PATH": os.pathsep.join([str(first), str(second)])},
    )
    entries = local_mod._managed_runtime_path_entries()
    assert entries.index(str(first)) < entries.index(str(second))

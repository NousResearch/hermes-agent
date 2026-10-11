"""The Landlock ruleset freezes HERMES_HOME only when it holds secrets.

HERMES_HOME is always an ancestor of the protected paths and Landlock rights are hierarchical, so
freezing ancestors unconditionally would freeze a bare or first-run home too: the agent terminal
could not create files there or in any parent directory, which breaks the session-snapshot
bootstrap. Ancestors are frozen only for protected paths that hold data (a file, or a directory with
any file beneath it); read protection of real secrets is unchanged.

Cross-platform: exercises the pure set computation, not the kernel (which is Linux-only).
"""

import os
from pathlib import Path

import pytest

from tools.environments import landlock_exec as le


def _paths(home: Path):
    no_access = [str(home / ".env"), str(home / "auth.json"), str(home / "auth")]
    read_only = [str(home / "config.yaml")]
    return no_access, read_only


def test_empty_home_is_not_frozen(tmp_path):
    home = tmp_path / "data"
    home.mkdir()
    no_access, read_only = _paths(home)
    _deny, _ro, walk = le._protection_sets(no_access, read_only)
    assert le._inode(str(home)) not in walk, "an empty HERMES_HOME must stay writable"
    assert not walk


def test_home_is_frozen_once_a_secret_exists(tmp_path):
    home = tmp_path / "data"
    home.mkdir()
    (home / ".env").write_text("OPENROUTER_API_KEY=x\n")
    no_access, read_only = _paths(home)
    deny, _ro, walk = le._protection_sets(no_access, read_only)
    assert le._inode(str(home / ".env")) in deny
    assert le._inode(str(home)) in walk, "the parent of a real secret must be frozen"
    # A secret that does not exist contributes nothing on its own.
    assert le._inode(str(home / "auth.json")) is None


def test_only_config_present_freezes_home_but_keeps_config_readable(tmp_path):
    home = tmp_path / "data"
    home.mkdir()
    (home / "config.yaml").write_text("approvals: {mode: manual}\n")
    no_access, read_only = _paths(home)
    deny, ro, walk = le._protection_sets(no_access, read_only)
    assert not deny
    assert le._inode(str(home / "config.yaml")) in ro
    assert le._inode(str(home)) in walk


def test_empty_protected_dir_does_not_freeze_home(tmp_path):
    # Hermes scaffolds empty secret dirs (e.g. ``pairing/``) when it initialises a home. An empty one
    # holds no secret, so it must not freeze HERMES_HOME and every ancestor (the session-snapshot temp
    # dir and the caller's working dir included).
    home = tmp_path / "data"
    (home / "auth" / "nested").mkdir(parents=True)
    no_access, read_only = _paths(home)
    deny, _ro, walk = le._protection_sets(no_access, read_only)
    assert not walk
    assert le._inode(str(home / "auth")) in deny, "still denied whenever another secret freezes the home"


def test_protected_dir_holding_a_file_freezes_home(tmp_path):
    home = tmp_path / "data"
    (home / "auth" / "nested").mkdir(parents=True)
    (home / "auth" / "nested" / "google_oauth.json").write_text("{}")
    no_access, read_only = _paths(home)
    _deny, _ro, walk = le._protection_sets(no_access, read_only)
    assert le._inode(str(home)) in walk


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads any directory")
def test_unreadable_protected_subtree_counts_as_a_secret(tmp_path):
    home = tmp_path / "data"
    sealed = home / "auth" / "sealed"
    sealed.mkdir(parents=True)
    sealed.chmod(0)
    try:
        no_access, read_only = _paths(home)
        _deny, _ro, walk = le._protection_sets(no_access, read_only)
    finally:
        sealed.chmod(0o700)
    assert le._inode(str(home)) in walk, "a subtree that cannot be inspected must fail closed"

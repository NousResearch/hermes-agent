"""A session cwd on a dead Windows SMB share must never take an unbounded host stat.

``projects.tree`` probes every session cwd on the RPC thread. ``run_git``'s
``os.path.isdir`` fast path and the sibling ``isdir``/``realpath`` calls in
``methods_projects`` are unbounded syscalls on network paths: a cold or dead SMB
share blocks them for tens of seconds with no userspace timeout, while the
1.5s-bounded git probe answers the same share in 0.05s (#135644). The network
classification is pure string data (``ntpath.splitdrive``), so every test here
runs on any host without faking the platform.
"""

from __future__ import annotations

import os

import pytest

from tui_gateway import git_probe
from tui_gateway import server

_UNC = "//srv/share/repo"
_UNC_BS = "\\\\".join(("", "", "srv", "share", "repo"))  # \\srv\share\repo


@pytest.mark.parametrize(
    "path,may_block",
    [
        (_UNC, True),
        (_UNC_BS, True),
        ("  " + _UNC + "  ", True),  # padded, as a client may send it
        ("\\\\?\\UNC\\srv\\share\\repo", True),
        ("C:\\repo", False),
        ("c:/repo", False),
        ("/Users/dev/repo", False),
        ("/mnt/data", False),
        ("relative/repo", False),
        ("", False),
        ("//srv", True),  # bare UNC prefix (no share): splitdrive still flags it
    ],
)
def test_path_may_block_classifies_by_path_shape(path, may_block):
    assert git_probe.path_may_block(path) is may_block


def test_run_git_skips_the_isdir_guard_on_a_unc_cwd(monkeypatch):
    """The unbounded stat must never run for a UNC cwd: the bounded git spawn
    (1.5s timeout) is the probe there. Red on main: the guard stat'ed first."""
    probed = []

    def _boom(path):
        raise AssertionError(f"unbounded stat reached: {path}")

    monkeypatch.setattr(git_probe.os.path, "isdir", _boom)
    monkeypatch.setattr(
        git_probe, "bounded_git_probe",
        lambda argv, timeout: probed.append(argv) or "root")

    assert git_probe.run_git(_UNC, "rev-parse", "--show-toplevel") == "root"
    assert probed and probed[0][:3] == ["git", "-C", _UNC]


def test_run_git_keeps_the_isdir_guard_for_local_cwds(monkeypatch):
    """Deleted local worktrees dominate a long session history: the cheap stat
    guard must stay on the fast path for host-local paths."""
    statted = []
    monkeypatch.setattr(git_probe.os.path, "isdir", lambda p: statted.append(p) or False)
    monkeypatch.setattr(
        git_probe, "bounded_git_probe", lambda argv, timeout: pytest.fail("git spawned"))

    assert git_probe.run_git("/tmp/gone-worktree", "branch") == ""
    assert statted == ["/tmp/gone-worktree"]


def test_dir_exists_cached_never_stats_a_network_path(monkeypatch):
    """``build_tree`` asks per SESSION: a dead share in the history must not
    stall the RPC. Network paths answer True (always-True = eligible, the same
    contract remote-backend cwds already get) without touching the disk."""

    def _boom(path):
        raise AssertionError(f"unbounded stat reached: {path}")

    monkeypatch.setattr(os.path, "isdir", _boom)
    server._DIR_EXISTS_CACHE.clear()
    try:
        assert server._dir_exists_cached(_UNC) is True
        assert server._DIR_EXISTS_CACHE == {}  # not memoized: no stat happened
    finally:
        server._DIR_EXISTS_CACHE.clear()


def test_dir_exists_cached_still_stats_local_paths(monkeypatch):
    server._DIR_EXISTS_CACHE.clear()
    try:
        assert server._dir_exists_cached("/definitely/not/a/dir-135644") is False
        assert server._dir_exists_cached("/definitely/not/a/dir-135644") is False
    finally:
        server._DIR_EXISTS_CACHE.clear()


def test_repo_junk_checks_never_touch_the_disk_for_network_paths(monkeypatch):
    """``_is_repo_junk``/``_is_session_cwd_junk`` realpath their input; on a
    dead share that stat blocks the RPC thread just like isdir."""

    def _boom(path):
        raise AssertionError(f"unbounded realpath reached: {path}")

    monkeypatch.setattr(os.path, "realpath", _boom)
    assert server._is_repo_junk(_UNC) is False
    assert server._is_session_cwd_junk(_UNC) is False


def test_discovery_scan_skips_network_roots_without_wiping(monkeypatch):
    """A policy root on a dead share must be skipped (non-authoritative scan ->
    the cache write MERGES instead of replacing), never stat'ed."""
    recorded = []

    def _boom(path):
        raise AssertionError(f"unbounded stat reached: {path}")

    monkeypatch.setattr(os.path, "isdir", _boom)
    from hermes_cli import projects_db as pdb
    monkeypatch.setattr(pdb, "record_discovered_repos", lambda *a, **k: recorded.append(k))

    authoritative = server._scan_discovered_repos_remote(
        object(), {"roots": [_UNC, _UNC_BS], "exclude_paths": []})
    assert authoritative is False
    assert recorded == []  # no pairs found -> no cache write at all

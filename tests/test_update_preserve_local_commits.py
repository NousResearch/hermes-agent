"""Regression: `hermes update` must not silently orphan local commits.

When the checkout sits on the target branch with local-only commits AND is
behind the remote, ``merge --ff-only`` fails and the updater falls back to
``git reset --hard origin/<branch>``. The autostash only covers working-tree
changes — local COMMITS were orphaned with no backup, recoverable only via
reflog (which expires). The updater must create a ``hermes-update-backup-*``
branch at HEAD before the reset.

These drive real git repositories in tmp_path — no mocks — because the
behavior contract is about what a user can recover after the reset runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import hermes_cli.update_cmd as update_cmd


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


def _commit(cwd: Path, name: str, text: str, message: str) -> str:
    (cwd / name).write_text(text)
    _git(cwd, "add", name)
    _git(
        cwd,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return _git(cwd, "rev-parse", "HEAD").stdout.strip()


def _backup_branches(cwd: Path) -> list[str]:
    out = _git(cwd, "branch", "--list", "hermes-update-backup-*").stdout
    return [line.strip().lstrip("* ").strip() for line in out.splitlines() if line.strip()]


@pytest.fixture
def diverged_clone(tmp_path: Path) -> tuple[Path, str]:
    """A clone on main with one local-only commit AND one remote-only commit.

    This is the exact state in which ``merge --ff-only`` fails and the
    updater reaches ``git reset --hard origin/main``.
    """
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(origin))

    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(origin), str(seed))
    _commit(seed, "base.txt", "base", "base commit")
    _git(seed, "push", "origin", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))
    local_sha = _commit(clone, "local.txt", "mine", "local work")

    # The remote advances independently, so the histories diverge.
    _commit(seed, "remote.txt", "theirs", "upstream work")
    _git(seed, "push", "origin", "main")
    _git(clone, "fetch", "origin", "main")

    return clone, local_sha


def test_backup_branch_preserves_local_commits_through_the_reset(
    diverged_clone: tuple[Path, str],
) -> None:
    clone, local_sha = diverged_clone

    backup = update_cmd._preserve_local_commits_before_reset(
        ["git"], clone, "origin/main"
    )

    assert backup is not None, "local commits exist — a backup branch must be created"
    assert backup.startswith("hermes-update-backup-")
    # The backup points exactly at the pre-reset HEAD.
    assert _git(clone, "rev-parse", backup).stdout.strip() == local_sha

    # Now do what the updater does next: reset --hard to the remote tip.
    _git(clone, "reset", "--hard", "origin/main")
    assert _git(clone, "rev-parse", "HEAD").stdout.strip() != local_sha

    # The contract: the local commit is still reachable without reflog diving.
    merged = _git(clone, "branch", "--contains", local_sha).stdout
    assert backup in merged
    assert _git(clone, "log", "-1", "--format=%s", backup).stdout.strip() == "local work"


def test_no_backup_when_head_has_no_local_commits(tmp_path: Path) -> None:
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", "--initial-branch=main", str(origin))
    seed = tmp_path / "seed"
    _git(tmp_path, "clone", str(origin), str(seed))
    _commit(seed, "base.txt", "base", "base commit")
    _git(seed, "push", "origin", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", str(origin), str(clone))

    backup = update_cmd._preserve_local_commits_before_reset(
        ["git"], clone, "origin/main"
    )

    assert backup is None
    assert _backup_branches(clone) == []

"""Repo health snapshot tool.

Returns a compact structured snapshot of the current git repository state,
or a zodern:relay-style failure envelope when git is missing or the working
directory is not inside a git repo.

Snapshot fields
---------------
* ``branch`` - current branch, or ``None`` if detached HEAD
* ``clean`` - whether the working tree is completely clean
* ``dirty_count`` - number of dirty working-tree entries (modified/staged/untracked)
* ``untracked_count`` - files not tracked by git
* ``ahead`` / ``behind`` - commit-count deltas vs ``origin/<branch>`` when
  the branch tracks an upstream branch; omitted when unknown
* ``tracks_upstream`` - whether ahead/behind were computed for an upstream

Dependencies: stdlib only + ``tools.validation_helper``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from tools.validation_helper import relay_failure, relay_success

logger = logging.getLogger(__name__)


def _git_available() -> bool:
    """Return True when the ``git`` executable is on PATH."""
    return shutil.which("git") is not None


def _git(args: list[str], repo: Path) -> str | None:
    """Run ``git <args>`` in *repo* and return stdout, or ``None`` on failure."""
    try:
        completed = subprocess.run(
            ["git"] + args,
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        if completed.returncode != 0:
            logger.debug("git %s failed: %s", args, completed.stderr.strip())
            return None
        return completed.stdout
    except OSError:
        logger.debug("git executable is not runnable")
        return None
    except subprocess.TimeoutExpired:
        logger.debug("git %s timed out", args)
        return None


def _snapshot(repo: Path) -> dict[str, object] | None:
    """Build a repo-health snapshot for *repo*.

    Returns ``None`` when the path is not a git repo.
    """
    if not _git(["rev-parse", "--show-toplevel"], repo):
        return None

    branch_out = _git(["branch", "--show-current"], repo)
    branch = branch_out.strip() if branch_out is not None else None

    status_output = _git(["status", "--porcelain"], repo)
    lines = status_output.splitlines() if status_output else []

    dirty: set[str] = set()
    untracked: set[str] = set()

    for line in lines:
        if not line:
            continue
        xy = line[:2]
        path = line[3:]
        if xy.startswith("?") or xy == "!!":
            untracked.add(path)
        else:
            dirty.add(path)

    total_dirty = len(dirty | untracked)
    total_untracked = len(untracked)

    ahead: int | None = None
    behind: int | None = None
    tracks_upstream = False

    if branch:
        upstream_out = _git(
            ["rev-parse", "--abbrev-ref", f"{branch}@{{upstream}}"],
            repo,
        )
        if upstream_out and upstream_out.strip():
            tracks_upstream = True
            ahead_behind_out = _git(
                ["rev-list", "--left-right", "--count",
                 f"{branch}...{upstream_out.strip()}"],
                repo,
            )
            # NOTE: rev-list --left-right --count counts local-vs-remote as
            # `ahead\tbehind`, so the first number means commits the local
            # branch has that the upstream does not.
            if ahead_behind_out:
                parts = ahead_behind_out.strip().split("\t")
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    ahead_int, behind_int = (int(part) for part in parts)
                    ahead, behind = ahead_int, behind_int

    snapshot: dict[str, object] = {
        "branch": branch,
        "clean": total_dirty == 0,
        "dirty_count": total_dirty,
        "untracked_count": total_untracked,
        "tracks_upstream": tracks_upstream,
    }
    if ahead is not None and behind is not None:
        snapshot["ahead"] = ahead
        snapshot["behind"] = behind
    return snapshot


def check_repo_health_requirements() -> bool:
    """Availability gate.

    Returns ``False`` when:
    * ``git`` is not installed / not on PATH, or
    * the current working directory is not inside a git repository.
    """
    if not _git_available():
        return False
    return _git(["rev-parse", "--show-toplevel"], Path.cwd()) is not None


def repo_health_tool(_args: dict[str, object] | None = None) -> dict[str, object]:
    """Return a zodern:relay-style envelope around the current repo snapshot.

    Accepts an optional *args* mapping so the tool registry's uniform handler
    signature works unchanged.  All snapshot data is derived from the current
    working directory.
    """
    try:
        snapshot = _snapshot(Path.cwd())
    except Exception as exc:
        logger.debug("repo_health snapshot failed unexpectedly: %s", exc)
        return relay_failure(str(exc), code="repo_health_error")

    if snapshot is None:
        return relay_failure(
            "current directory is not inside a git repository",
            code="repo_health_error",
        )

    return relay_success(snapshot)


REPO_HEALTH_SCHEMA = {
    "name": "repo_health",
    "description": (
        "Return a compact snapshot of the git repository state: current branch, "
        "clean/dirty status, dirty/untracked file counts, and ahead/behind counts "
        "vs origin/<branch> when the branch tracks an upstream. Returns a failure "
        "when git is missing or the working directory is not inside a git repo."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}


def register(registry):  # pragma: no cover - registration side-effect
    registry.register(
        name="repo_health",
        toolset="repo",
        schema=REPO_HEALTH_SCHEMA,
        handler=repo_health_tool,
        check_fn=check_repo_health_requirements,
        emoji="🦠",
    )

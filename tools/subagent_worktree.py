"""Opt-in git worktree isolation for delegated subagents.

Inspired by Muse Code's ``--subagent-worktree-isolation`` (Meta, Aug 2026):
when isolation is on, each delegated child agent gets its own git worktree
checked out from the parent's current commit, so parallel children never
contend for the same working copy and the parent's checkout stays untouched.
This is a clean-room implementation of the documented behavior
(https://dev.meta.ai/docs/muse-code/extending#multi-agent); no Muse Code
code was referenced.

Enable in config.yaml::

    delegation:
      worktree_isolation: true   # default: false

Contract (mirrors Muse Code's documented semantics):

- **Opt-in and git-only.** In a non-git workspace the setting is ignored
  without an error and children share the parent's working directory,
  exactly as before.
- **One worktree per child**, branched from the parent repo's current
  ``HEAD`` under ``<repo>/.worktrees/subagent-<id>`` on branch
  ``hermes-subagent/<id>``.
- **The parent reviews/merges.** Children commit inside their own worktree;
  each result entry reports the worktree path, branch, commit count, and
  dirty state so the parent can review or merge each branch.
- **Clean worktrees are pruned.** A worktree with no new commits and a
  clean tree is removed automatically after the child finishes; anything
  holding work is kept and reported.

Only the local terminal backend is supported: on docker/ssh/modal/etc. the
worktree created on the host would not be visible inside the sandbox, so
isolation is skipped (with a debug log) rather than half-applied.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 30
_WORKTREES_DIRNAME = ".worktrees"
_BRANCH_NAMESPACE = "hermes-subagent"


def _run_git(args, cwd: str, timeout: int = _GIT_TIMEOUT):
    """Run a git command, capturing output. Never raises on non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def local_backend_active() -> bool:
    """True when the terminal backend is local (worktrees visible to tools)."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        backend = ((cfg.get("terminal") or {}).get("backend") or "local")
        return str(backend).strip().lower() in ("", "local")
    except Exception:
        # Legacy entry points without the shared loader default to local.
        return True


def resolve_repo_root(path: Optional[str]) -> Optional[str]:
    """Return the git toplevel for *path*, or None when not in a work tree."""
    if not path:
        return None
    try:
        candidate = os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        return None
    if not os.path.isdir(candidate):
        return None
    try:
        result = _run_git(["rev-parse", "--show-toplevel"], cwd=candidate)
    except Exception as exc:
        logger.debug("subagent worktree: rev-parse failed: %s", exc)
        return None
    if result.returncode != 0:
        return None
    root = result.stdout.strip()
    return root or None


def _ensure_gitignore_entry(repo_root: str) -> None:
    """Best-effort: keep ``.worktrees/`` out of git status."""
    gitignore = Path(repo_root) / ".gitignore"
    entry = f"{_WORKTREES_DIRNAME}/"
    try:
        existing = (
            gitignore.read_text(encoding="utf-8-sig", errors="replace")
            if gitignore.exists()
            else ""
        )
        if entry not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"{entry}\n")
    except Exception as exc:
        logger.debug("subagent worktree: could not update .gitignore: %s", exc)


def create_subagent_worktree(
    parent_cwd: Optional[str],
    subagent_id: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """Create an isolated worktree for one child agent.

    Returns metadata (``path``, ``branch``, ``repo_root``, ``base_commit``)
    on success, or ``None`` when the workspace is not a git repository or
    worktree creation fails — mirroring Muse Code, absence of git downgrades
    silently to shared-workspace behavior.
    """
    repo_root = resolve_repo_root(parent_cwd)
    if not repo_root:
        return None

    short_id = (subagent_id or uuid.uuid4().hex[:8]).replace("/", "-")
    wt_name = f"subagent-{short_id}"
    branch = f"{_BRANCH_NAMESPACE}/{wt_name}"
    wt_path = Path(repo_root) / _WORKTREES_DIRNAME / wt_name

    try:
        wt_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.warning("subagent worktree: cannot create %s: %s", wt_path.parent, exc)
        return None

    _ensure_gitignore_entry(repo_root)

    try:
        base = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
        base_commit = base.stdout.strip() if base.returncode == 0 else ""
        result = _run_git(
            ["worktree", "add", str(wt_path), "-b", branch, "HEAD"],
            cwd=repo_root,
        )
    except Exception as exc:
        logger.warning("subagent worktree: creation failed: %s", exc)
        return None
    if result.returncode != 0:
        # Common on repos with zero commits (unborn HEAD) — degrade silently.
        logger.warning(
            "subagent worktree: git worktree add failed: %s",
            result.stderr.strip(),
        )
        return None

    logger.info("subagent worktree created: %s (branch %s)", wt_path, branch)
    return {
        "path": str(wt_path),
        "branch": branch,
        "repo_root": repo_root,
        "base_commit": base_commit,
    }


def finalize_subagent_worktree(
    info: Dict[str, str], *, prune: bool = True
) -> Dict[str, Any]:
    """Inspect (and possibly prune) a child worktree after the child finishes.

    Returns a result-entry payload: path, branch, ``commits`` ahead of the
    base, ``dirty`` (uncommitted changes present), and ``pruned``. A worktree
    with zero commits and a clean tree is removed when *prune* is true;
    anything holding work is always kept for the parent to review or merge.
    """
    path = info.get("path", "")
    branch = info.get("branch", "")
    repo_root = info.get("repo_root", "")
    base_commit = info.get("base_commit", "")

    payload: Dict[str, Any] = {
        "path": path,
        "branch": branch,
        "commits": 0,
        "dirty": False,
        "pruned": False,
    }
    if not path or not os.path.isdir(path):
        payload["pruned"] = True  # nothing on disk to review
        return payload

    try:
        if base_commit:
            counted = _run_git(
                ["rev-list", "--count", f"{base_commit}..HEAD"], cwd=path
            )
            if counted.returncode == 0:
                payload["commits"] = int(counted.stdout.strip() or 0)
        status = _run_git(["status", "--porcelain"], cwd=path)
        if status.returncode == 0:
            payload["dirty"] = bool(status.stdout.strip())
    except Exception as exc:
        logger.debug("subagent worktree: finalize inspection failed: %s", exc)
        # Unknown state — keep the worktree rather than risk deleting work.
        return payload

    if prune and payload["commits"] == 0 and not payload["dirty"]:
        try:
            removed = _run_git(
                ["worktree", "remove", "--force", path], cwd=repo_root or path
            )
            if removed.returncode == 0:
                _run_git(["branch", "-D", branch], cwd=repo_root or path)
                payload["pruned"] = True
                logger.info("subagent worktree pruned (no work): %s", path)
            else:
                logger.debug(
                    "subagent worktree: prune failed: %s", removed.stderr.strip()
                )
        except Exception as exc:
            logger.debug("subagent worktree: prune failed: %s", exc)

    return payload


def build_worktree_context_note(info: Dict[str, str]) -> str:
    """Context block telling the child to work inside its isolated worktree."""
    return (
        "\n\n[WORKTREE ISOLATION] You are working in an isolated git worktree "
        f"at: {info.get('path')}\n"
        f"Your dedicated branch is: {info.get('branch')}\n"
        "All file edits and shell commands must happen inside this worktree "
        "directory (your terminal already starts there). Do NOT cd to the "
        "main repository checkout. Commit your changes to your branch when "
        "done; the parent agent will review and merge your branch. If you "
        "make no commits and leave the tree clean, the worktree is discarded "
        "automatically."
    )

"""Opt-in git worktree isolation for delegated subagents (``delegation.worktree_isolation``).

Git-only (outside a repo children share the parent's cwd); local terminal backend only (on
docker/ssh/modal the host worktree is invisible in the sandbox, so isolation is skipped). One
worktree per child under ``<repo>/.worktrees/subagent-<id>``, branch ``hermes-subagent/<id>``;
pruned only on proof (zero commits AND clean, both probes ok), else kept + ``inspection_failed``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli._subprocess_compat import harden_git_argv, noninteractive_git_env

logger = logging.getLogger(__name__)

_GIT_TIMEOUT = 30


def _run_git(args, cwd: str, timeout: int = _GIT_TIMEOUT):
    """Run git capturing output; never raises on non-zero exit.

    :func:`noninteractive_git_env` (GHSA-7x36-8jrh-v4pw): this runs unattended against whatever
    repo the parent sits in and ``worktree add`` runs hooks, so a malicious ``.git/config`` must
    not execute.
    """
    return subprocess.run(["git", *harden_git_argv(args)], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          stdin=subprocess.DEVNULL, env=noninteractive_git_env())


def local_backend_active() -> bool:
    """True when the terminal backend is local (worktrees visible to tools)."""
    try:
        from hermes_cli.config import load_config_readonly

        backend = (load_config_readonly().get("terminal") or {}).get("backend") or "local"
        return str(backend).strip().lower() in ("", "local")
    except Exception:
        # Legacy entry points without the shared loader default to local.
        return True


def resolve_repo_root(path: Optional[str]) -> Optional[str]:
    """Return the git toplevel for *path*, or None when not in a work tree."""
    candidate = os.path.abspath(os.path.expanduser(str(path))) if path else ""
    if not candidate or not os.path.isdir(candidate):
        return None
    try:
        result = _run_git(["rev-parse", "--show-toplevel"], cwd=candidate)
    except Exception as exc:
        logger.debug("subagent worktree: rev-parse failed: %s", exc)
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def _ensure_gitignore_entry(repo_root: str) -> None:
    """Best-effort: keep ``.worktrees/`` out of git status."""
    gitignore = Path(repo_root) / ".gitignore"
    try:
        existing = gitignore.read_text(encoding="utf-8-sig", errors="replace") if gitignore.exists() else ""
        if ".worktrees/" not in existing.splitlines():
            with open(gitignore, "a", encoding="utf-8") as f:
                sep = "\n" if existing and not existing.endswith("\n") else ""
                f.write(f"{sep}.worktrees/\n")
    except Exception as exc:
        logger.debug("subagent worktree: could not update .gitignore: %s", exc)


def create_subagent_worktree(parent_cwd: Optional[str], subagent_id: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Create an isolated worktree for one child; None (silent downgrade) outside git/on failure."""
    repo_root = resolve_repo_root(parent_cwd)
    if not repo_root:
        return None
    wt_name = f"subagent-{(subagent_id or uuid.uuid4().hex[:8]).replace('/', '-')}"
    branch = f"hermes-subagent/{wt_name}"
    wt_path = Path(repo_root) / ".worktrees" / wt_name
    try:
        wt_path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_gitignore_entry(repo_root)
        base = _run_git(["rev-parse", "HEAD"], cwd=repo_root)
        base_commit = base.stdout.strip() if base.returncode == 0 else ""
        result = _run_git(["worktree", "add", str(wt_path), "-b", branch, "HEAD"], cwd=repo_root)
    except Exception as exc:
        logger.warning("subagent worktree: creation failed: %s", exc)
        return None
    if result.returncode != 0:
        # Common on repos with zero commits (unborn HEAD) — degrade silently.
        logger.warning("subagent worktree: git worktree add failed: %s", result.stderr.strip())
        return None
    logger.info("subagent worktree created: %s (branch %s)", wt_path, branch)
    return {"path": str(wt_path), "branch": branch, "repo_root": repo_root, "base_commit": base_commit}


def _base_payload(info: Dict[str, str]) -> Dict[str, Any]:
    """Result-entry schema the parent expects (no creation-side internals)."""
    return {"path": info.get("path", ""), "branch": info.get("branch", ""),
            "commits": 0, "dirty": False, "pruned": False}


def mark_worktree_payload_unproven(payload: Dict[str, Any], reason: str, *,
                                   unmeasured: str = "commits/dirty") -> Dict[str, Any]:
    """Flag a worktree result payload as un-inspected, in place.

    The parent only sees this dict, so the uncertainty must travel in it or "0 commits, clean"
    reads as "the child produced nothing". *unmeasured* names only the fields actually left
    unproven (one probe can succeed while the other fails).

    See #88113.
    """
    path, branch = payload.get("path", ""), payload.get("branch", "")
    payload["inspection_failed"] = True
    payload["note"] = (f"git inspection failed ({reason}): {unmeasured} UNKNOWN — not proven "
                       f"zero/clean. The worktree and branch were preserved — inspect {path} "
                       f"(branch {branch}) before assuming no work.")
    logger.warning("subagent worktree: git inspection failed (%s) — keeping %s (branch %s) "
                   "for manual review", reason, path, branch)
    return payload


def unproven_worktree_payload(info: Dict[str, str], reason: str) -> Dict[str, Any]:
    """Complete un-inspected payload for ``delegate_tool`` when finalize raises."""
    return mark_worktree_payload_unproven(_base_payload(info), reason)


def finalize_subagent_worktree(info: Dict[str, str], *, prune: bool = True) -> Dict[str, Any]:
    """Inspect (and possibly prune) a child worktree after the child finishes.

    Prunes only when *prune*, commits==0, clean tree AND both git probes succeeded; otherwise
    keeps it with ``inspection_failed`` + ``note`` (``commits``/``dirty`` then are defaults).
    """
    path, branch = info.get("path", ""), info.get("branch", "")
    base_commit = info.get("base_commit", "")
    payload = _base_payload(info)
    if not path or not os.path.isdir(path):
        payload["pruned"] = True  # nothing on disk to review
        return payload
    # Without a base commit the count is an unproven default, and the prune
    # condition reads payload["commits"] — so it must not prune either.
    # See #88113.
    if not base_commit:
        return mark_worktree_payload_unproven(
            payload, "no base_commit recorded — commit count unmeasurable", unmeasured="commits")
    failed, unmeasured = [], []
    probes = (("commits", "rev-list", ["rev-list", "--count", f"{base_commit}..HEAD"],
               lambda s: int(s or 0)),
              ("dirty", "status", ["status", "--porcelain"], bool))
    try:
        for field, label, args, parse in probes:
            res = _run_git(args, cwd=path)
            if res.returncode == 0:
                payload[field] = parse(res.stdout.strip())
            else:
                failed.append(f"{label} exit {res.returncode}: {res.stderr.strip()[:200]}")
                unmeasured.append(field)
    except Exception as exc:
        # Timeout, OSError or non-numeric rev-list stdout: which probe raised is
        # unknowable, so neither value is trustworthy — keep the worktree.
        return mark_worktree_payload_unproven(payload, f"inspection raised: {exc}")
    if failed:
        # Destructive cleanup requires affirmative proof; defaults prove nothing.
        return mark_worktree_payload_unproven(payload, "; ".join(failed), unmeasured="/".join(unmeasured))
    if prune and payload["commits"] == 0 and not payload["dirty"]:
        cwd = info.get("repo_root", "") or path
        try:
            removed = _run_git(["worktree", "remove", "--force", path], cwd=cwd)
            if removed.returncode == 0:
                _run_git(["branch", "-D", branch], cwd=cwd)
                payload["pruned"] = True
                logger.info("subagent worktree pruned (no work): %s", path)
            else:
                logger.debug("subagent worktree: prune failed: %s", removed.stderr.strip())
        except Exception as exc:
            logger.debug("subagent worktree: prune failed: %s", exc)
    return payload


def build_worktree_context_note(info: Dict[str, str]) -> str:
    """Context block telling the child to work inside its isolated worktree."""
    return (
        "\n\n[WORKTREE ISOLATION] You are working in an isolated git worktree "
        f"at: {info.get('path')}\n"
        f"Your dedicated branch is: {info.get('branch')}\n"
        "All file edits and shell commands must happen inside this worktree directory (your "
        "terminal already starts there). Do NOT cd to the main repository checkout. Commit your "
        "changes to your branch when done; the parent agent will review and merge your branch. If "
        "you make no commits and leave the tree clean, the worktree is discarded automatically."
    )

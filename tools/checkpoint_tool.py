#!/usr/bin/env python3
"""
Checkpoint Tool — Shadow Git Snapshots

Creates and restores filesystem snapshots of a working directory using a shadow
git repository stored at ~/.hermes/checkpoints/{dir-hash}/. The shadow repo is
completely separate from any git repo the user may have — their .git/ is never
touched.

GIT_DIR + GIT_WORK_TREE environment variables redirect every git command to the
shadow repo while operating on the user's actual working tree. No git state leaks
into the project directory.

Excludes (written to shadow repo's info/exclude, not the user's .gitignore):
  node_modules/, dist/, .env, .env.*, __pycache__/, *.pyc, .DS_Store, *.log

Operations:
  take    — git add -A && git commit -m "{reason}"
  restore — git checkout {commit_hash} -- .   (HEAD not moved)
  list    — git log with short hash, ISO timestamp, reason

Design:
- Single `checkpoint` tool with action parameter: take, restore, list
- Shadow repo path is deterministic: sha256(abs_working_dir)[:16]
- check_fn gates the tool on git being present in PATH
- Behavioral guidance lives in the tool schema description
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_BASE = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "checkpoints"

DEFAULT_EXCLUDES = [
    "node_modules/",
    "dist/",
    "build/",
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
    "*.log",
    ".cache/",
    ".next/",
    ".nuxt/",
    "coverage/",
    ".pytest_cache/",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shadow_repo_path(working_dir: str) -> Path:
    """Return the shadow repo path for a given working directory.

    Uses first 16 hex chars of sha256(abs_path) so the shadow directory name
    is deterministic and collision-resistant without being too long.
    """
    abs_path = str(Path(working_dir).resolve())
    dir_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    return CHECKPOINT_BASE / dir_hash


def _git_env(shadow_repo: Path, working_dir: str) -> dict:
    """Build an environment dict that redirects git to the shadow repo.

    GIT_DIR  — points git's internal storage at the shadow directory.
    GIT_WORK_TREE — points git's file operations at the user's working dir.
    GIT_INDEX_FILE is cleared so we never accidentally share an index with the
    user's own git repo.
    """
    env = os.environ.copy()
    env["GIT_DIR"] = str(shadow_repo)
    env["GIT_WORK_TREE"] = str(Path(working_dir).resolve())
    env.pop("GIT_INDEX_FILE", None)
    env.pop("GIT_NAMESPACE", None)
    env.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
    return env


def _run_git(
    args: List[str],
    shadow_repo: Path,
    working_dir: str,
    timeout: int = 30,
) -> Tuple[bool, str, str]:
    """Run a git subcommand with the shadow repo environment.

    Returns (success: bool, stdout: str, stderr: str).
    Never raises — all errors are returned as (False, "", error_message).
    """
    env = _git_env(shadow_repo, working_dir)
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path(working_dir).resolve()),
        )
        return (
            result.returncode == 0,
            result.stdout.strip(),
            result.stderr.strip(),
        )
    except subprocess.TimeoutExpired:
        return False, "", f"git command timed out after {timeout}s: git {' '.join(args)}"
    except FileNotFoundError:
        return False, "", "git executable not found — install git to use checkpoints"
    except Exception as exc:
        return False, "", str(exc)


def _init_shadow_repo(shadow_repo: Path, working_dir: str) -> Optional[str]:
    """Initialize the shadow git repo if it doesn't already exist.

    Returns an error string on failure, None on success.

    Initialization steps:
      1. mkdir -p shadow_repo
      2. git init  (GIT_DIR points here, so git internals land in shadow_repo/)
      3. git config user.email/name  (so commits work without global git config)
      4. Write DEFAULT_EXCLUDES to shadow_repo/info/exclude
    """
    # HEAD file presence is the canonical indicator of a valid git repo
    if (shadow_repo / "HEAD").exists():
        return None

    shadow_repo.mkdir(parents=True, exist_ok=True)

    ok, _, err = _run_git(["init"], shadow_repo, working_dir)
    if not ok:
        return f"Failed to initialize shadow repo at {shadow_repo}: {err}"

    # Set a local identity so commits never fail due to missing global config
    _run_git(["config", "user.email", "hermes@local"], shadow_repo, working_dir)
    _run_git(["config", "user.name", "Hermes Agent"], shadow_repo, working_dir)

    # Write exclude patterns to info/exclude — this file is shadow-repo-local
    # and never appears in the user's working directory.
    info_dir = shadow_repo / "info"
    info_dir.mkdir(exist_ok=True)
    exclude_path = info_dir / "exclude"
    exclude_path.write_text("\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8")

    # Store the original working dir path in the shadow repo for reference
    (shadow_repo / "HERMES_WORKDIR").write_text(
        str(Path(working_dir).resolve()) + "\n", encoding="utf-8"
    )

    logger.debug("Initialized shadow checkpoint repo at %s for %s", shadow_repo, working_dir)
    return None


# ---------------------------------------------------------------------------
# CheckpointStore class
# ---------------------------------------------------------------------------


class CheckpointStore:
    """Manages shadow-git checkpoints for a single working directory.

    The shadow repo lives at ~/.hermes/checkpoints/{dir-hash}/ and is
    completely independent of any git repo the user has in their project.
    Every git call sets GIT_DIR + GIT_WORK_TREE so no state leaks.

    Methods mirror the tool actions: take, restore, list.
    """

    def __init__(self, working_dir: str):
        self.working_dir = str(Path(working_dir).resolve())
        self.shadow_repo = _shadow_repo_path(self.working_dir)

    def _ensure_initialized(self) -> Optional[str]:
        """Ensure the shadow repo exists. Returns error string or None."""
        return _init_shadow_repo(self.shadow_repo, self.working_dir)

    # ------------------------------------------------------------------
    # take
    # ------------------------------------------------------------------

    def take(self, reason: str) -> Dict[str, Any]:
        """Stage all files and commit with reason as the commit message.

        Respects info/exclude (node_modules, dist, .env, etc.).
        If nothing has changed since the last checkpoint, returns the
        existing HEAD hash without creating a new commit.
        """
        err = self._ensure_initialized()
        if err:
            return {"success": False, "error": err}

        reason = reason.strip()
        if not reason:
            return {"success": False, "error": "reason cannot be empty"}

        # Stage all files — respects shadow repo's info/exclude
        ok, _, err = _run_git(
            ["add", "-A"],
            self.shadow_repo,
            self.working_dir,
            timeout=60,
        )
        if not ok:
            return {"success": False, "error": f"git add failed: {err}"}

        # Check if anything was actually staged
        ok, status_out, _ = _run_git(
            ["status", "--porcelain"],
            self.shadow_repo,
            self.working_dir,
        )
        if ok and not status_out.strip():
            # Nothing changed — return existing HEAD rather than an error
            ok2, head, _ = _run_git(
                ["rev-parse", "--short", "HEAD"],
                self.shadow_repo,
                self.working_dir,
            )
            if ok2 and head:
                return {
                    "success": True,
                    "commit_hash": head,
                    "files_changed": 0,
                    "message": "No changes since last checkpoint — existing snapshot returned",
                    "working_dir": self.working_dir,
                }
            return {
                "success": False,
                "error": (
                    "Nothing to commit and no prior checkpoint exists. "
                    "The working directory may be empty or all files are excluded."
                ),
            }

        # Count staged files for the response
        staged_files = [l for l in status_out.splitlines() if l.strip()]

        ok, out, err = _run_git(
            ["commit", "-m", reason],
            self.shadow_repo,
            self.working_dir,
            timeout=60,
        )
        if not ok:
            return {"success": False, "error": f"git commit failed: {err}"}

        ok2, commit_hash, _ = _run_git(
            ["rev-parse", "--short", "HEAD"],
            self.shadow_repo,
            self.working_dir,
        )

        return {
            "success": True,
            "commit_hash": commit_hash if ok2 else "unknown",
            "files_changed": len(staged_files),
            "message": f"Checkpoint saved: {reason}",
            "working_dir": self.working_dir,
            "shadow_repo": str(self.shadow_repo),
        }

    # ------------------------------------------------------------------
    # restore
    # ------------------------------------------------------------------

    def restore(self, commit_hash: str) -> Dict[str, Any]:
        """Restore the working tree to match a specific checkpoint.

        Uses `git checkout {commit_hash} -- .` so the current HEAD is not
        moved. Only files that were snapshotted are restored; excluded files
        (node_modules, .env, etc.) are left untouched.
        """
        err = self._ensure_initialized()
        if err:
            return {"success": False, "error": err}

        commit_hash = commit_hash.strip()
        if not commit_hash:
            return {"success": False, "error": "commit_hash cannot be empty"}

        # Verify the commit exists before touching the working tree
        ok, full_hash, verify_err = _run_git(
            ["rev-parse", "--verify", f"{commit_hash}^{{commit}}"],
            self.shadow_repo,
            self.working_dir,
        )
        if not ok:
            return {
                "success": False,
                "error": (
                    f"Checkpoint '{commit_hash}' not found in shadow repo. "
                    f"Use action='list' to see available checkpoints. ({verify_err})"
                ),
            }

        # Retrieve commit metadata before modifying the working tree
        ok_msg, reason, _ = _run_git(
            ["log", "-1", "--pretty=format:%s", commit_hash],
            self.shadow_repo,
            self.working_dir,
        )
        ok_ts, timestamp, _ = _run_git(
            ["log", "-1", "--pretty=format:%ai", commit_hash],
            self.shadow_repo,
            self.working_dir,
        )

        # Restore working tree contents from the commit.
        # '--' separates the treeish from the pathspec; '.' means all paths.
        ok, _, restore_err = _run_git(
            ["checkout", commit_hash, "--", "."],
            self.shadow_repo,
            self.working_dir,
            timeout=120,
        )
        if not ok:
            return {"success": False, "error": f"git checkout failed: {restore_err}"}

        return {
            "success": True,
            "commit_hash": commit_hash,
            "reason": reason if ok_msg else "",
            "timestamp": timestamp if ok_ts else "",
            "message": f"Working directory restored to checkpoint {commit_hash}",
            "working_dir": self.working_dir,
            "note": (
                "Only snapshotted files were restored. "
                "Excluded paths (node_modules/, dist/, .env) were not modified."
            ),
        }

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list(self, limit: int = 20) -> Dict[str, Any]:
        """Return a list of checkpoints ordered newest-first.

        Each entry contains: commit_hash, timestamp (ISO 8601), reason.
        """
        err = self._ensure_initialized()
        if err:
            return {"success": False, "error": err}

        # Check whether any commits exist yet
        ok, _, _ = _run_git(
            ["rev-parse", "HEAD"],
            self.shadow_repo,
            self.working_dir,
        )
        if not ok:
            return {
                "success": True,
                "checkpoints": [],
                "count": 0,
                "message": "No checkpoints yet for this directory",
                "working_dir": self.working_dir,
                "shadow_repo": str(self.shadow_repo),
            }

        # \x1f (unit separator) is safe to use as a field delimiter inside
        # git's --pretty=format because it cannot appear in commit messages.
        fmt = "%h\x1f%ai\x1f%s"
        ok, out, err = _run_git(
            ["log", f"--pretty=format:{fmt}", f"-{limit}"],
            self.shadow_repo,
            self.working_dir,
        )
        if not ok:
            return {"success": False, "error": f"git log failed: {err}"}

        checkpoints = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("\x1f", 2)
            if len(parts) == 3:
                checkpoints.append({
                    "commit_hash": parts[0],
                    "timestamp": parts[1],
                    "reason": parts[2],
                })

        return {
            "success": True,
            "checkpoints": checkpoints,
            "count": len(checkpoints),
            "working_dir": self.working_dir,
            "shadow_repo": str(self.shadow_repo),
        }


# ---------------------------------------------------------------------------
# Dispatch function
# ---------------------------------------------------------------------------


def checkpoint_tool(
    action: str,
    working_dir: str = None,
    reason: str = None,
    commit_hash: str = None,
    limit: int = 20,
) -> str:
    """Single entry point for the checkpoint tool. Returns JSON string."""
    if not working_dir:
        working_dir = os.getcwd()

    try:
        working_dir = str(Path(working_dir).expanduser().resolve())
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Invalid working_dir: {exc}"})

    if not Path(working_dir).is_dir():
        return json.dumps({
            "success": False,
            "error": f"working_dir does not exist or is not a directory: {working_dir}",
        })

    store = CheckpointStore(working_dir)

    if action == "take":
        if not reason:
            return json.dumps({
                "success": False,
                "error": "reason is required for 'take' action",
            })
        result = store.take(reason)

    elif action == "restore":
        if not commit_hash:
            return json.dumps({
                "success": False,
                "error": "commit_hash is required for 'restore' action",
            })
        result = store.restore(commit_hash)

    elif action == "list":
        result = store.list(limit=max(1, min(int(limit), 100)))

    else:
        return json.dumps({
            "success": False,
            "error": f"Unknown action '{action}'. Use: take, restore, list",
        })

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def take_checkpoint(working_dir: str, reason: str) -> None:
    """Thin wrapper for call sites that only need to take a snapshot.

    Silently swallows all errors so that callers (rm, mv, overwrite) are
    never blocked or broken if git is unavailable or the snapshot fails.
    The operation proceeds regardless of whether the checkpoint succeeded.
    """
    try:
        checkpoint_tool(action="take", working_dir=working_dir, reason=reason)
    except Exception as exc:
        logger.debug("take_checkpoint silenced error: %s", exc)


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------


def check_checkpoint_requirements() -> bool:
    """Return True if git is available on PATH."""
    return shutil.which("git") is not None


# ---------------------------------------------------------------------------
# OpenAI Function-Calling Schema
# ---------------------------------------------------------------------------

CHECKPOINT_SCHEMA = {
    "name": "checkpoint",
    "description": (
        "Create and restore filesystem snapshots of a working directory using a shadow "
        "git repository stored at ~/.hermes/checkpoints/. The shadow repo is completely "
        "separate from any git repo the user has — their .git/ is never touched.\n\n"
        "WHEN TO USE:\n"
        "- Before large destructive edits (refactors, bulk deletions, rewrites)\n"
        "- Before running scripts that modify many files\n"
        "- After reaching a known-good state the user may want to return to\n"
        "- When the user says 'save this state', 'checkpoint this', or 'I want to undo later'\n\n"
        "ACTIONS:\n"
        "- take: snapshot the current working directory (git add -A && git commit)\n"
        "- restore: revert files to a prior snapshot (git checkout {hash} -- .)\n"
        "- list: show all snapshots with hashes, timestamps, and reasons\n\n"
        "EXCLUDED FROM SNAPSHOTS: node_modules/, dist/, build/, .env, .env.*, "
        "__pycache__/, *.pyc, .cache/, .next/, coverage/\n\n"
        "NOTE: restore only affects files that were snapshotted. Excluded directories "
        "(node_modules, dist, etc.) are left untouched."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["take", "restore", "list"],
                "description": "The action to perform.",
            },
            "working_dir": {
                "type": "string",
                "description": (
                    "Absolute path to the directory to snapshot or restore. "
                    "Defaults to the current working directory if omitted."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "Human-readable description of why this checkpoint was taken. "
                    "Required for 'take'. Be specific: 'before refactoring auth module', "
                    "'after completing user profile feature', 'pre-migration backup'."
                ),
            },
            "commit_hash": {
                "type": "string",
                "description": (
                    "The checkpoint hash to restore. Required for 'restore'. "
                    "Obtain hashes from the 'list' action."
                ),
            },
            "limit": {
                "type": "integer",
                "description": (
                    "Maximum number of checkpoints to return for 'list'. "
                    "Defaults to 20, maximum 100."
                ),
                "default": 20,
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="checkpoint",
    toolset="checkpoint",
    schema=CHECKPOINT_SCHEMA,
    handler=lambda args, **kw: checkpoint_tool(
        action=args.get("action", ""),
        working_dir=args.get("working_dir"),
        reason=args.get("reason"),
        commit_hash=args.get("commit_hash"),
        limit=int(args.get("limit", 20)),
    ),
    check_fn=check_checkpoint_requirements,
    requires_env=[],
    description="Filesystem snapshot and restore via shadow git repo",
)

"""
Checkpoint Manager — Transparent filesystem snapshots via shadow git repos.

Creates automatic snapshots of working directories before file-mutating
operations (write_file, patch), triggered once per conversation turn.
Provides rollback to any previous checkpoint.

This is NOT a tool — the LLM never sees it.  It's transparent infrastructure
controlled by the ``checkpoints`` config flag or ``--checkpoints`` CLI flag.

Architecture:
    ~/.hermes/checkpoints/{sha256(abs_dir)[:16]}/   — shadow git repo
        HEAD, refs/, objects/                        — standard git internals
        HERMES_WORKDIR                               — original dir path
        info/exclude                                 — default excludes

The shadow repo uses GIT_DIR + GIT_WORK_TREE so no git state leaks
into the user's project directory.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_BASE = get_hermes_home() / "checkpoints"

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
    ".venv/",
    "venv/",
    ".git/",
]

# Git subprocess timeout (seconds).
_GIT_TIMEOUT: int = max(10, min(60, int(os.getenv("HERMES_CHECKPOINT_TIMEOUT", "30"))))

# Max files to snapshot — skip huge directories to avoid slowdowns.
_MAX_FILES = 50_000

# Valid git commit hash pattern: 4–40 hex chars (short or full SHA-1/SHA-256).
_COMMIT_HASH_RE = re.compile(r'^[0-9a-fA-F]{4,64}$')


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_commit_hash(commit_hash: str) -> Optional[str]:
    """Validate a commit hash to prevent git argument injection.

    Returns an error string if invalid, None if valid.
    Values starting with '-' would be interpreted as git flags
    (e.g., '--patch', '-p') instead of revision specifiers.
    """
    if not commit_hash or not commit_hash.strip():
        return "Empty commit hash"
    if commit_hash.startswith("-"):
        return f"Invalid commit hash (must not start with '-'): {commit_hash!r}"
    if not _COMMIT_HASH_RE.match(commit_hash):
        return f"Invalid commit hash (expected 4-64 hex characters): {commit_hash!r}"
    return None


def _validate_file_path(file_path: str, working_dir: str) -> Optional[str]:
    """Validate a file path to prevent path traversal outside the working directory.

    Returns an error string if invalid, None if valid.
    """
    if not file_path or not file_path.strip():
        return "Empty file path"
    # Reject absolute paths — restore targets must be relative to the workdir
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    # Resolve and check containment within working_dir
    abs_workdir = Path(working_dir).resolve()
    resolved = (abs_workdir / file_path).resolve()
    try:
        resolved.relative_to(abs_workdir)
    except ValueError:
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


# ---------------------------------------------------------------------------
# Shadow repo helpers
# ---------------------------------------------------------------------------

def _shadow_repo_path(working_dir: str) -> Path:
    """Deterministic shadow repo path: sha256(abs_path)[:16]."""
    abs_path = str(Path(working_dir).resolve())
    dir_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]
    return CHECKPOINT_BASE / dir_hash


def _git_env(shadow_repo: Path, working_dir: str) -> dict:
    """Build env dict that redirects git to the shadow repo."""
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
    timeout: int = _GIT_TIMEOUT,
    allowed_returncodes: Optional[Set[int]] = None,
) -> tuple:
    """Run a git command against the shadow repo.  Returns (ok, stdout, stderr).

    ``allowed_returncodes`` suppresses error logging for known/expected non-zero
    exits while preserving the normal ``ok = (returncode == 0)`` contract.
    Example: ``git diff --cached --quiet`` returns 1 when changes exist.
    """
    env = _git_env(shadow_repo, working_dir)
    cmd = ["git"] + list(args)
    allowed_returncodes = allowed_returncodes or set()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=str(Path(working_dir).resolve()),
        )
        ok = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if not ok and result.returncode not in allowed_returncodes:
            logger.error(
                "Git command failed: %s (rc=%d) stderr=%s",
                " ".join(cmd), result.returncode, stderr,
            )
        return ok, stdout, stderr
    except subprocess.TimeoutExpired:
        msg = f"git timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg, exc_info=True)
        return False, "", msg
    except FileNotFoundError:
        logger.error("Git executable not found: %s", " ".join(cmd), exc_info=True)
        return False, "", "git not found"
    except Exception as exc:
        logger.error("Unexpected git error running %s: %s", " ".join(cmd), exc, exc_info=True)
        return False, "", str(exc)


def _init_shadow_repo(shadow_repo: Path, working_dir: str) -> Optional[str]:
    """Initialise shadow repo if needed.  Returns error string or None."""
    if (shadow_repo / "HEAD").exists():
        return None

    shadow_repo.mkdir(parents=True, exist_ok=True)

    ok, _, err = _run_git(["init"], shadow_repo, working_dir)
    if not ok:
        return f"Shadow repo init failed: {err}"

    _run_git(["config", "user.email", "hermes@local"], shadow_repo, working_dir)
    _run_git(["config", "user.name", "Hermes Checkpoint"], shadow_repo, working_dir)

    info_dir = shadow_repo / "info"
    info_dir.mkdir(exist_ok=True)
    (info_dir / "exclude").write_text(
        "\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8"
    )

    (shadow_repo / "HERMES_WORKDIR").write_text(
        str(Path(working_dir).resolve()) + "\n", encoding="utf-8"
    )

    logger.debug("Initialised checkpoint repo at %s for %s", shadow_repo, working_dir)
    return None


def _dir_file_count(path: str) -> int:
    """Quick file count estimate (stops early if over _MAX_FILES)."""
    count = 0
    try:
        for _ in Path(path).rglob("*"):
            count += 1
            if count > _MAX_FILES:
                return count
    except (PermissionError, OSError):
        pass
    return count


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages automatic filesystem checkpoints.

    Designed to be owned by AIAgent.  Call ``new_turn()`` at the start of
    each conversation turn and ``ensure_checkpoint(dir, reason)`` before
    any file-mutating tool call.  The manager deduplicates so at most one
    snapshot is taken per directory per turn.

    Parameters
    ----------
    enabled : bool
        Master switch (from config / CLI flag).
    max_snapshots : int
        Keep at most this many checkpoints per directory.
    """

    def __init__(self, enabled: bool = False, max_snapshots: int = 50):
        self.enabled = enabled
        self.max_snapshots = max_snapshots
        self._checkpointed_dirs: Set[str] = set()
        self._git_available: Optional[bool] = None  # lazy probe

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Reset per-turn dedup.  Call at the start of each agent iteration."""
        self._checkpointed_dirs.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        """Take a checkpoint if enabled and not already done this turn.

        Returns True if a checkpoint was taken, False otherwise.
        Never raises — all errors are silently logged.
        """
        if not self.enabled:
            return False

        # Lazy git probe
        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.debug("Checkpoints disabled: git not found")
        if not self._git_available:
            return False

        abs_dir = str(Path(working_dir).resolve())

        # Skip root, home, and other overly broad directories
        if abs_dir in ("/", str(Path.home())):
            logger.debug("Checkpoint skipped: directory too broad (%s)", abs_dir)
            return False

        # Already checkpointed this turn?
        if abs_dir in self._checkpointed_dirs:
            return False

        self._checkpointed_dirs.add(abs_dir)

        try:
            return self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)
            return False

    def list_checkpoints(self, working_dir: str) -> List[Dict]:
        """List available checkpoints for a directory.

        Returns a list of dicts with keys: hash, short_hash, timestamp, reason,
        files_changed, insertions, deletions.  Most recent first.
        """
        abs_dir = str(Path(working_dir).resolve())
        shadow = _shadow_repo_path(abs_dir)

        if not (shadow / "HEAD").exists():
            return []

        ok, stdout, _ = _run_git(
            ["log", "--format=%H|%h|%aI|%s", "-n", str(self.max_snapshots)],
            shadow, abs_dir,
        )

        if not ok or not stdout:
            return []

        results = []
        for line in stdout.splitlines():
            parts = line.split("|", 3)
            if len(parts) == 4:
                entry = {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "timestamp": parts[2],
                    "reason": parts[3],
                    "files_changed": 0,
                    "insertions": 0,
                    "deletions": 0,
                }
                # Get diffstat for this commit
                stat_ok, stat_out, _ = _run_git(
                    ["diff", "--shortstat", f"{parts[0]}~1", parts[0]],
                    shadow, abs_dir,
                    allowed_returncodes={128, 129},  # first commit has no parent
                )
                if stat_ok and stat_out:
                    self._parse_shortstat(stat_out, entry)
                results.append(entry)
        return results

    @staticmethod
    def _parse_shortstat(stat_line: str, entry: Dict) -> None:
        """Parse git --shortstat output into entry dict."""
        import re
        m = re.search(r'(\d+) file', stat_line)
        if m:
            entry["files_changed"] = int(m.group(1))
        m = re.search(r'(\d+) insertion', stat_line)
        if m:
            entry["insertions"] = int(m.group(1))
        m = re.search(r'(\d+) deletion', stat_line)
        if m:
            entry["deletions"] = int(m.group(1))

    def diff(self, working_dir: str, commit_hash: str) -> Dict:
        """Show diff between a checkpoint and the current working tree.

        Returns dict with success, diff text, and stat summary.
        """
        # Validate commit_hash to prevent git argument injection
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(Path(working_dir).resolve())
        shadow = _shadow_repo_path(abs_dir)

        if not (shadow / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        # Verify the commit exists
        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], shadow, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        # Stage current state to compare against checkpoint
        _run_git(["add", "-A"], shadow, abs_dir, timeout=_GIT_TIMEOUT * 2)

        # Get stat summary: checkpoint vs current working tree
        ok_stat, stat_out, _ = _run_git(
            ["diff", "--stat", commit_hash, "--cached"],
            shadow, abs_dir,
        )

        # Get actual diff (limited to avoid terminal flood)
        ok_diff, diff_out, _ = _run_git(
            ["diff", commit_hash, "--cached", "--no-color"],
            shadow, abs_dir,
        )

        # Unstage to avoid polluting the shadow repo index
        _run_git(["reset", "HEAD", "--quiet"], shadow, abs_dir)

        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}

        return {
            "success": True,
            "stat": stat_out if ok_stat else "",
            "diff": diff_out if ok_diff else "",
        }

    def restore(self, working_dir: str, commit_hash: str, file_path: str = None) -> Dict:
        """Restore files to a checkpoint state.

        Uses ``git checkout <hash> -- .`` (or a specific file) which restores
        tracked files without moving HEAD — safe and reversible.

        Parameters
        ----------
        file_path : str, optional
            If provided, restore only this file instead of the entire directory.

        Returns dict with success/error info.
        """
        # Validate commit_hash to prevent git argument injection
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(Path(working_dir).resolve())

        # Validate file_path to prevent path traversal outside the working dir
        if file_path:
            path_err = _validate_file_path(file_path, abs_dir)
            if path_err:
                return {"success": False, "error": path_err}

        shadow = _shadow_repo_path(abs_dir)

        if not (shadow / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        # Verify the commit exists
        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], shadow, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found", "debug": err or None}

        # Take a checkpoint of current state before restoring (so you can undo the undo)
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        # Restore — full directory or single file
        restore_target = file_path if file_path else "."
        ok, stdout, err = _run_git(
            ["checkout", commit_hash, "--", restore_target],
            shadow, abs_dir, timeout=_GIT_TIMEOUT * 2,
        )

        if not ok:
            return {"success": False, "error": f"Restore failed: {err}", "debug": err or None}

        # Get info about what was restored
        ok2, reason_out, _ = _run_git(
            ["log", "--format=%s", "-1", commit_hash], shadow, abs_dir,
        )
        reason = reason_out if ok2 else "unknown"

        result = {
            "success": True,
            "restored_to": commit_hash[:8],
            "reason": reason,
            "directory": abs_dir,
        }
        if file_path:
            result["file"] = file_path
        return result

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory for checkpointing.

        Walks up from the file's parent to find a reasonable project root
        (directory containing .git, pyproject.toml, package.json, etc.).
        Falls back to the file's parent directory.
        """
        path = Path(file_path).resolve()
        if path.is_dir():
            candidate = path
        else:
            candidate = path.parent

        # Walk up looking for project root markers
        markers = {".git", "pyproject.toml", "package.json", "Cargo.toml",
                    "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile"}
        check = candidate
        while check != check.parent:
            if any((check / m).exists() for m in markers):
                return str(check)
            check = check.parent

        # No project root found — use the file's parent
        return str(candidate)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _take(self, working_dir: str, reason: str) -> bool:
        """Take a snapshot.  Returns True on success."""
        shadow = _shadow_repo_path(working_dir)

        # Init if needed
        err = _init_shadow_repo(shadow, working_dir)
        if err:
            logger.debug("Checkpoint init failed: %s", err)
            return False

        # Quick size guard — don't try to snapshot enormous directories
        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False

        # Stage everything
        ok, _, err = _run_git(
            ["add", "-A"], shadow, working_dir, timeout=_GIT_TIMEOUT * 2,
        )
        if not ok:
            logger.debug("Checkpoint git-add failed: %s", err)
            return False

        # Check if there's anything to commit
        ok_diff, diff_out, _ = _run_git(
            ["diff", "--cached", "--quiet"],
            shadow,
            working_dir,
            allowed_returncodes={1},
        )
        if ok_diff:
            # No changes to commit
            logger.debug("Checkpoint skipped: no changes in %s", working_dir)
            return False

        # Commit
        ok, _, err = _run_git(
            ["commit", "-m", reason, "--allow-empty-message"],
            shadow, working_dir, timeout=_GIT_TIMEOUT * 2,
        )
        if not ok:
            logger.debug("Checkpoint commit failed: %s", err)
            return False

        logger.debug("Checkpoint taken in %s: %s", working_dir, reason)

        # Prune old snapshots
        self._prune(shadow, working_dir)

        return True

    def _prune(self, shadow_repo: Path, working_dir: str) -> None:
        """Keep only the last max_snapshots commits via orphan reset."""
        ok, stdout, _ = _run_git(
            ["rev-list", "--count", "HEAD"], shadow_repo, working_dir,
        )
        if not ok:
            return

        try:
            count = int(stdout)
        except ValueError:
            return

        if count <= self.max_snapshots:
            return

        # For simplicity, we don't actually prune — git's pack mechanism
        # handles this efficiently, and the objects are small.  The log
        # listing is already limited by max_snapshots.
        # Full pruning would require rebase --onto or filter-branch which
        # is fragile for a background feature.  We just limit the log view.
        logger.debug("Checkpoint repo has %d commits (limit %d)", count, self.max_snapshots)


def format_checkpoint_list(checkpoints: List[Dict], directory: str) -> str:
    """Format checkpoint list for display to user."""
    if not checkpoints:
        return f"No checkpoints found for {directory}"

    lines = [f"📸 Checkpoints for {directory}:\n"]
    for i, cp in enumerate(checkpoints, 1):
        # Parse ISO timestamp to something readable
        ts = cp["timestamp"]
        if "T" in ts:
            ts = ts.split("T")[1].split("+")[0].split("-")[0][:5]  # HH:MM
            date = cp["timestamp"].split("T")[0]
            ts = f"{date} {ts}"

        # Build change summary
        files = cp.get("files_changed", 0)
        ins = cp.get("insertions", 0)
        dele = cp.get("deletions", 0)
        if files:
            stat = f"  ({files} file{'s' if files != 1 else ''}, +{ins}/-{dele})"
        else:
            stat = ""

        lines.append(f"  {i}. {cp['short_hash']}  {ts}  {cp['reason']}{stat}")

    lines.append("\n  /rollback <N>             restore to checkpoint N")
    lines.append("  /rollback diff <N>        preview changes since checkpoint N")
    lines.append("  /rollback <N> <file>      restore a single file from checkpoint N")
    return "\n".join(lines)

"""
Checkpoint Manager — Transparent filesystem snapshots via a single shared
shadow git store.

Creates automatic snapshots of working directories before file-mutating
operations (``write_file``, ``patch``, ``terminal`` with destructive flags),
triggered once per conversation turn.  Provides rollback to any previous
checkpoint.

This is NOT a tool — the LLM never sees it.  It's transparent infrastructure
controlled by the ``checkpoints`` config flag or ``--checkpoints`` CLI flag.

This module is the facade and the single import surface for the subsystem:
``tools.checkpoint_store`` owns layout, init, migration and project metadata
(its docstring carries the storage layout); ``tools.checkpoint_restore`` owns
restore, diff, rollback planning and the agent-write ledger;
``tools.checkpoint_staging`` owns snapshot staging, index maintenance and the
post-commit budget trim; ``tools.checkpoint_maintenance`` owns ref retention,
orphan pruning and the store size cap.

Seam owners stay defined here on purpose: tests monkeypatch them on this
module, and the siblings reach them through call-time delegates.
``CHECKPOINT_BASE``/``_resolve_checkpoint_base``, ``_checkpoint_base_rel_path``,
``_staging_args``, ``_git_env`` and ``_run_git`` are the git seams; the
index-lock reclaim helpers stay for the same reason and for their log
namespace - a ``caplog`` assertion pins their record to this logger name.
"""

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import selected_git_env, windows_hide_flags
from typing import Optional

# Re-export contract: callers and tests import the checkpoint vocabulary from
# this module, so the sibling modules' surface stays reachable here.  No F401
# suppression on these imports - that rule is not selected in this repo, and a
# redundant suppression directive would itself be flagged.
#
# The siblings' ``_run_git``/``_resolve_checkpoint_base``/``_staging_args``/
# ``_checkpoint_base_rel_path`` delegates are deliberately not imported: they
# would shadow the seam owners defined below.
from tools.checkpoint_store import (
    _GIT_TIMEOUT,
    _INDEXES_DIRNAME,
    _LEGACY_PREFIX,
    _PRUNE_MARKER_NAME,
    _PROJECTS_DIRNAME,
    _REFS_PREFIX,
    _STORE_DIRNAME,
    _index_path,
    _init_store,
    _list_projects,
    _normalize_path,
    _pre_v2_shadow_repos,
    _project_hash,
    _project_meta_path,
    _ref_name,
    _register_project,
    _store_has_head,
    _store_path,
    _touch_project,
    _unlink_quiet,
    _volume_evidence,
)
from tools.checkpoint_restore import (
    RestoreMixin,
    _hash_file,
    _ledger_path,
    _load_ledger,
    _save_ledger,
)
from tools.checkpoint_staging import (
    StagingMixin,
    _dir_file_count,
    _dir_size_bytes,
    _uncaptured_note,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_BASE = get_hermes_home() / "checkpoints"
_CHECKPOINT_BASE_AT_IMPORT = CHECKPOINT_BASE


def _resolve_checkpoint_base() -> Path:
    """Active profile's checkpoint root at call time: the patched ``CHECKPOINT_BASE`` when a test
    changed it, else live profile-scoped HERMES_HOME — under the multiplexed gateway one process
    serves every profile, so the import-time constant would write every profile's code-edit
    checkpoints into the launch profile's store."""
    return CHECKPOINT_BASE if CHECKPOINT_BASE != _CHECKPOINT_BASE_AT_IMPORT else get_hermes_home() / "checkpoints"


# Minimum age of a zero-byte index lock file before we consider it stale and
# reclaim it (seconds).  Five minutes is well above any legitimate git hold
# time and well below the "failing forever" symptom window.
_STALE_INDEX_LOCK_MIN_AGE_SECONDS: int = 300


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _checkpoint_base_rel_path(working_dir: str) -> Optional[str]:
    """Path of the active checkpoint base relative to ``working_dir``, or None.

    A profile-home checkpoint (``~/.hermes/profiles/<name>``) contains the
    checkpoint base, so staging the workdir unfiltered would store the store's
    own object database inside the very snapshots it keeps — a loop no size
    cap can converge.  The answer is derived from the live
    ``_resolve_checkpoint_base()`` path, never from the literal directory name
    ``checkpoints``: a user project that happens to own a ``checkpoints/``
    directory is data and must stay in the snapshot.

    ``"."`` means the working directory *is* the base; callers staging a tree
    must treat that as "nothing to snapshot".
    """
    base = _normalize_path(str(_resolve_checkpoint_base()))
    workdir = _normalize_path(working_dir)
    try:
        return base.relative_to(workdir).as_posix()
    except ValueError:
        return None


def _staging_args(working_dir: str) -> list[str]:
    """``git add`` arguments staging ``working_dir`` without re-capturing the store.

    ``info/exclude`` is written once at store creation, so it cannot cover a
    base discovered later (an existing store) or one that moves; the exclusion
    is therefore a pathspec computed per take.  ``literal`` magic keeps a
    directory whose name holds glob characters from widening the exclusion.
    When the workdir *is* the base the exclusion covers the whole tree;
    ``_take`` refuses that case rather than committing an empty snapshot.
    """
    base_rel = _checkpoint_base_rel_path(working_dir)
    if base_rel is None:
        return ["add", "-A"]
    return ["add", "-A", "--", ".", f":(exclude,literal){base_rel}"]


# ---------------------------------------------------------------------------
# Git env
# ---------------------------------------------------------------------------

def _git_env(
    store: Path,
    working_dir: str,
    index_file: Optional[Path] = None,
) -> dict:
    """Build env dict that redirects git to the shared store.

    The shared store is internal Hermes infrastructure — it must NOT inherit
    the user's global or system git config.  User-level settings like
    ``commit.gpgsign = true``, signing hooks, or credential helpers would
    either break background snapshots or, worse, spawn interactive prompts
    (pinentry GUI windows) mid-session every time a file is written.

    Isolation strategy:
    * ``GIT_CONFIG_GLOBAL=<os.devnull>`` — ignore ``~/.gitconfig`` (git 2.32+).
    * ``GIT_CONFIG_SYSTEM=<os.devnull>`` — ignore ``/etc/gitconfig`` (git 2.32+).
    * ``GIT_CONFIG_NOSYSTEM=1`` — legacy belt-and-suspenders for older git.

    ``index_file``, if given, forces git to use a per-project index under
    ``store/indexes/<hash>`` so projects don't race on a shared index.
    """
    normalized_working_dir = _normalize_path(working_dir)
    # git child with hand-isolated config env; exact preservation — a HOME
    # rewrite would change which ~/.gitconfig the isolation vars are hiding.
    from tools.environments.local import build_subprocess_env

    env = selected_git_env(build_subprocess_env(scrub_secrets=False, inherit_profile_home=False))
    env["GIT_DIR"] = str(store)
    env["GIT_WORK_TREE"] = str(normalized_working_dir)
    env.pop("GIT_NAMESPACE", None)
    env.pop("GIT_ALTERNATE_OBJECT_DIRECTORIES", None)
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(index_file)
    else:
        env.pop("GIT_INDEX_FILE", None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return env


def _lock_has_live_holder(lock_path: Path) -> bool:
    """True if ``lock_path`` is currently held by a live process.

    Fail-safe: any probe failure (missing ``lsof``, timeout, unexpected
    exception, Windows) returns True so we never reclaim a lock we cannot
    verify is orphaned.
    """
    if os.name == "nt":
        return True
    try:
        result = subprocess.run(
            ["lsof", "-t", str(lock_path)],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return True


def _reclaim_stale_index_lock(
    lock_path: Path,
    *,
    store: Path,
    min_age_seconds: int = _STALE_INDEX_LOCK_MIN_AGE_SECONDS,
) -> bool:
    """Remove a zero-byte, unheld, old index lock if it is inside the store.

    Returns True when the lock was removed.  All guards must hold: the
    resolved path lives under ``store/"indexes"``, the file exists, size is
    zero, mtime age is at least ``min_age_seconds``, and no live process
    holds it.  Any unlink failure returns False.
    """
    try:
        resolved = lock_path.resolve()
        indexes_root = (store / _INDEXES_DIRNAME).resolve()
        if not str(resolved).startswith(str(indexes_root) + os.sep) and resolved != indexes_root:
            return False
        if not resolved.is_file():
            return False
        st = resolved.stat()
        if st.st_size != 0:
            return False
        age = time.time() - st.st_mtime
        if age < min_age_seconds:
            return False
        if _lock_has_live_holder(resolved):
            return False
        resolved.unlink()
    except OSError:
        return False
    logger.info("Reclaimed stale git index lock %s (age=%.0fs, size=0 bytes)", resolved, age)
    return True


def _run_git(
    args: list[str],
    store: Path,
    working_dir: str,
    timeout: int = _GIT_TIMEOUT,
    allowed_returncodes: Optional[set[int]] = None,
    index_file: Optional[Path] = None,
    extra_env: Optional[dict[str, str]] = None,
    _retried_lock_reclaim: bool = False,
) -> tuple[bool, str, str]:
    """Run a git command against the shared store.  Returns (ok, stdout, stderr).

    ``allowed_returncodes`` suppresses error logging for known/expected non-zero
    exits while preserving the normal ``ok = (returncode == 0)`` contract.
    Example: ``git diff --cached --quiet`` returns 1 when changes exist.
    """
    normalized_working_dir = _normalize_path(working_dir)
    if not normalized_working_dir.exists():
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg
    if not normalized_working_dir.is_dir():
        msg = f"working directory is not a directory: {normalized_working_dir}"
        logger.error("Git command skipped: %s (%s)", " ".join(["git"] + list(args)), msg)
        return False, "", msg

    env = _git_env(store, str(normalized_working_dir), index_file=index_file)
    if extra_env:
        env.update(extra_env)
    git = shutil.which("git", path=env.get("PATH", ""))
    if git is None:
        return False, "", "git is not installed or not on PATH"
    cmd = [git, *args]
    allowed_returncodes = allowed_returncodes or set()

    try:
        # NUL-delimited git output contains literal filenames, not text lines.
        text_options = {} if "-z" in args else {"text": True, "encoding": "utf-8", "errors": "replace"}
        result = subprocess.run(
            cmd,
            capture_output=True,
            **text_options,
            timeout=timeout,
            env=env,
            cwd=str(normalized_working_dir),
            stdin=subprocess.DEVNULL,
            # Checkpoints fire several bare git calls per turn from the
            # console-less desktop/gateway backend; suppress the per-call
            # conhost flash on Windows (no-op on POSIX).
            creationflags=windows_hide_flags(),
            check=False,
        )
        ok = result.returncode == 0
        stdout = os.fsdecode(result.stdout) if "-z" in args else result.stdout.strip()
        stderr = result.stderr.decode("utf-8", errors="replace").strip() if "-z" in args else result.stderr.strip()
        if not ok and result.returncode not in allowed_returncodes:
            # One-shot reclaim for orphaned zero-byte index locks.  The lock
            # path comes from git's stderr; when absent we fall back to the
            # per-project index file we set via GIT_INDEX_FILE.
            if (
                not _retried_lock_reclaim
                and "Unable to create" in stderr
                and "File exists" in stderr
            ):
                match = re.search(r"Unable to create '(.+?)': File exists", stderr)
                if match:
                    lock_path = Path(match.group(1))
                elif index_file is not None:
                    lock_path = Path(str(index_file) + ".lock")
                else:
                    lock_path = None
                if lock_path is not None and _reclaim_stale_index_lock(lock_path, store=store):
                    retry = subprocess.run(
                        cmd,
                        capture_output=True,
                        **text_options,
                        timeout=timeout,
                        env=env,
                        cwd=str(normalized_working_dir),
                        stdin=subprocess.DEVNULL,
                        creationflags=windows_hide_flags(),
                        check=False,
                    )
                    retry_ok = retry.returncode == 0
                    retry_stdout = os.fsdecode(retry.stdout) if "-z" in args else retry.stdout.strip()
                    retry_stderr = retry.stderr.decode("utf-8", errors="replace").strip() if "-z" in args else retry.stderr.strip()
                    if retry_ok or retry.returncode in allowed_returncodes:
                        return retry_ok, retry_stdout, retry_stderr
                    # Reclaim succeeded but the retry still failed — log the
                    # retry's failure and return it (do not re-reclaim).
                    logger.error(
                        "Git command failed after index-lock reclaim: %s (rc=%d) stderr=%s",
                        " ".join(cmd), retry.returncode, retry_stderr,
                    )
                    return retry_ok, retry_stdout, retry_stderr
            logger.error(
                "Git command failed: %s (rc=%d) stderr=%s",
                " ".join(cmd), result.returncode, stderr,
            )
        return ok, stdout, stderr
    except subprocess.TimeoutExpired:
        msg = f"git timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg, exc_info=True)
        return False, "", msg
    except FileNotFoundError as exc:
        missing_target = getattr(exc, "filename", None)
        if missing_target == "git":
            logger.error("Git executable not found: %s", " ".join(cmd), exc_info=True)
            return False, "", "git not found"
        msg = f"working directory not found: {normalized_working_dir}"
        logger.error("Git command failed before execution: %s (%s)", " ".join(cmd), msg, exc_info=True)
        return False, "", msg
    except Exception as exc:
        logger.error("Unexpected git error running %s: %s", " ".join(cmd), exc, exc_info=True)
        return False, "", str(exc)


def _git_out(args: list[str], store: Path, working_dir: str, rc: Optional[set[int]] = None) -> str:
    """stdout of a successful git call, else ``""``."""
    ok, out, _ = _run_git(args, store, working_dir, allowed_returncodes=rc)
    return out if ok else ""


def _ref_tip(store: Path, working_dir: str, ref: str) -> Optional[str]:
    """Commit sha at ``ref``, or None when the ref does not exist yet."""
    return _git_out(["rev-parse", "--verify", ref + "^{commit}"], store, working_dir, {128}) or None


def _list_project_refs(store: Path, working_dir: str) -> list[str]:
    out = _git_out(["for-each-ref", "--format=%(refname)", _REFS_PREFIX], store, working_dir, {128})
    return [r for r in out.splitlines() if r.strip()]


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class CheckpointManager(RestoreMixin, StagingMixin):
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
    max_total_size_mb : int
        Hard ceiling on total store size.  Oldest checkpoints per project
        are dropped when the store exceeds this after a commit.
    max_file_size_mb : int
        Skip adding any single file larger than this to a checkpoint.
        (Implemented via ``.gitignore`` excludes + a post-stage size check.)
    """

    def __init__(
        self,
        enabled: bool = False,
        max_snapshots: int = 20,
        max_total_size_mb: int = 500,
        max_file_size_mb: int = 10,
    ):
        self.enabled = enabled
        self.max_snapshots = max(1, int(max_snapshots))
        self.max_total_size_mb = max(0, int(max_total_size_mb))
        self.max_file_size_mb = max(0, int(max_file_size_mb))
        self._checkpointed_dirs: set[str] = set()

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    def new_turn(self) -> None:
        """Reset per-turn dedup.  Call at the start of each agent iteration."""
        self._checkpointed_dirs.clear()

    def unsupported_backend_reason(self, task_id: str = "default") -> Optional[str]:
        """Explain why host checkpoints are off limits for a container-backed session.

        Classifies the task's backend at call time (nothing is remembered), so /rollback is
        refused before the first mutation and follows a backend change within the session."""
        from tools.file_tools_paths import container_backend_for_task
        backend = container_backend_for_task(task_id)
        if backend is None:
            return None
        return (
            f"Checkpoints are not taken for terminal.backend={backend}: "
            "file paths belong to the container, not this host."
        )

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

        abs_dir = str(_normalize_path(working_dir))

        # Skip root, home, and other overly broad directories
        if abs_dir in {"/", str(Path.home())}:
            logger.debug("Checkpoint skipped: directory too broad (%s)", abs_dir)
            return False

        if abs_dir in self._checkpointed_dirs:
            return False

        self._checkpointed_dirs.add(abs_dir)

        try:
            from tools.checkpoint_pruning import store_lock

            with store_lock(_resolve_checkpoint_base()):
                return self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)
            return False

    def list_checkpoints(self, working_dir: str) -> list[dict]:
        """List available checkpoints for a directory (most recent first)."""
        abs_dir = str(_normalize_path(working_dir))
        store = _store_path()

        if not (store / "HEAD").exists():
            return []

        ref = _ref_name(_project_hash(abs_dir))
        ok, stdout, _ = _run_git(
            ["log", ref, "--format=%H|%h|%aI|%s", "-n", str(self.max_snapshots)],
            store, abs_dir,
            allowed_returncodes={128, 129},
        )

        if not ok or not stdout:
            return []

        results: list[dict] = []
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
                stat_ok, stat_out, _ = _run_git(
                    ["diff", "--shortstat", f"{parts[0]}~1", parts[0]],
                    store, abs_dir,
                    allowed_returncodes={128, 129},
                )
                if stat_ok and stat_out:
                    self._parse_shortstat(stat_out, entry)
                results.append(entry)
        return results

    def list_all_checkpoints(self) -> list[dict]:
        """List checkpoints across every registered project (most recent first).

        Surgical reapply of PR #10633 by @nightq (#10505) onto the v2
        single-store layout: iterate ``projects/<hash>.json`` metadata via
        ``_list_projects`` instead of the pre-v2 per-shadow-dir scan. Each
        entry carries the extra ``workdir`` key so callers can label which
        project a checkpoint belongs to.
        """
        store = _store_path()
        if not (store / "HEAD").exists():
            return []
        results: list[dict] = []
        for meta in _list_projects(store):
            workdir = meta.get("workdir") or ""
            if not workdir:
                continue
            for entry in self.list_checkpoints(workdir):
                entry["workdir"] = workdir
                results.append(entry)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return results

    @staticmethod
    def _parse_shortstat(stat_line: str, entry: dict) -> None:
        """Parse git --shortstat output into entry dict."""
        m = re.search(r'(\d+) file', stat_line)
        if m:
            entry["files_changed"] = int(m.group(1))
        m = re.search(r'(\d+) insertion', stat_line)
        if m:
            entry["insertions"] = int(m.group(1))
        m = re.search(r'(\d+) deletion', stat_line)
        if m:
            entry["deletions"] = int(m.group(1))

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory for checkpointing."""
        path = _normalize_path(file_path)
        if path.is_dir():
            candidate = path
        else:
            candidate = path.parent

        # An explicitly checkpointed root owns its ledger even when an
        # ancestor carries a project marker. Prefer the nearest owner.
        roots = [Path(meta["workdir"]) for meta in _list_projects(_store_path()) if meta.get("workdir")]
        owners = [root for root in roots if candidate.is_relative_to(root)]
        if owners:
            return str(max(owners, key=lambda root: len(root.parts)))
        markers = {".git", "pyproject.toml", "package.json", "Cargo.toml",
                    "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile"}
        home = Path.home().resolve()
        broad = {home, *home.parents}
        check = candidate
        while check != check.parent and check not in broad:
            if any((check / m).exists() for m in markers):
                return str(check)
            check = check.parent

        return str(candidate)


def format_checkpoint_list(checkpoints: list[dict], directory: str) -> str:
    """Format checkpoint list for display to user."""
    if not checkpoints:
        return f"No checkpoints found for {directory}"

    lines = [f"📸 Checkpoints for {directory}:\n"]
    for i, cp in enumerate(checkpoints, 1):
        ts = cp["timestamp"]
        if "T" in ts:
            ts = ts.split("T")[1].split("+")[0].split("-")[0][:5]
            date = cp["timestamp"].split("T")[0]
            ts = f"{date} {ts}"

        files = cp.get("files_changed", 0)
        ins = cp.get("insertions", 0)
        dele = cp.get("deletions", 0)
        if files:
            stat = f"  ({files} file{'s' if files != 1 else ''}, +{ins}/-{dele})"
        else:
            stat = ""

        # Label per-project entries when showing the cross-project view
        # (workdir key only present on list_all_checkpoints results).
        workdir = cp.get("workdir", "")
        if workdir and directory == "all directories":
            workdir_short = Path(workdir).name or workdir
            lines.append(
                f"  {i}. {cp['short_hash']}  {ts}  [{workdir_short}]  {cp['reason']}{stat}"
            )
        else:
            lines.append(f"  {i}. {cp['short_hash']}  {ts}  {cp['reason']}{stat}")

    lines.append("\n  /rollback <N>             restore to checkpoint N")
    lines.append("  /rollback diff <N>        preview changes since checkpoint N")
    lines.append("  /rollback <N> <file>      restore a single file from checkpoint N")
    return "\n".join(lines)

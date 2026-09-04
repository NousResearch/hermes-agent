"""
Checkpoint Manager — Transparent filesystem snapshots via a single shared
shadow git store.

Creates automatic snapshots of working directories before file-mutating
operations (``write_file``, ``patch``, ``terminal`` with destructive flags),
triggered once per conversation turn.  Provides rollback to any previous
checkpoint.

This is NOT a tool — the LLM never sees it.  It's transparent infrastructure
controlled by the ``checkpoints`` config flag or ``--checkpoints`` CLI flag.

Storage layout (single shared store, git objects deduplicated across projects)
-----------------------------------------------------------------------------

    ~/.hermes/checkpoints/
        store/                          — single bare-ish git repo
            HEAD, config, objects/      — standard git internals (shared)
            refs/hermes/<hash16>        — per-project branch tip
            indexes/<hash16>            — per-project git index
            projects/<hash16>.json      — {workdir, created_at, last_touch}
            info/exclude                — default excludes (shared)
        .last_prune                     — auto-prune idempotency marker
        legacy-<timestamp>/             — archived pre-v2 per-project shadow
                                          repos (auto-migrated on first init)

Why a single store?
-------------------

The pre-v2 design kept a full shadow repo per working directory.  Each one
re-stored most of the project's files under its own ``objects/`` tree, with
zero sharing across worktrees of the same project.  A single user with a
dozen worktrees of the same repo burned ~40 MB each (~500 MB total) storing
the same blobs over and over.  A single shared store lets git's content-
addressable object DB deduplicate across projects and across turns, so adding
a new worktree costs near-zero.

The shadow store uses ``GIT_DIR`` + ``GIT_WORK_TREE`` + ``GIT_INDEX_FILE``
so no git state leaks into the user's project directory.

Auto-maintenance
----------------

Shadow state accumulates over time.  ``prune_checkpoints`` deletes refs whose
recorded working directory no longer exists (orphan) or whose last touch is
older than ``retention_days`` (stale), then runs ``git gc --prune=now`` to
reclaim object storage.  A size-cap pass drops the oldest checkpoints per
project until total store size is under ``max_total_size_mb``.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from typing import Dict, List, Optional, Set, Tuple

from utils import env_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_BASE = get_hermes_home() / "checkpoints"

# Single shared store directory under CHECKPOINT_BASE.
_STORE_DIRNAME = "store"
_REFS_PREFIX = "refs/hermes"
_INDEXES_DIRNAME = "indexes"
_PROJECTS_DIRNAME = "projects"
_LEDGERS_DIRNAME = "ledgers"
_LEGACY_PREFIX = "legacy-"

# Agent-write ledger cap: newest entries retained per project.
_LEDGER_MAX_ENTRIES = 2000

DEFAULT_EXCLUDES = [
    # Dependency / build output
    "node_modules/",
    "dist/",
    "build/",
    "target/",
    "out/",
    ".next/",
    ".nuxt/",
    # Caches
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    ".cache/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    "coverage/",
    ".coverage",
    # Virtualenvs
    ".venv/",
    "venv/",
    "env/",
    # VCS
    ".git/",
    ".hg/",
    ".svn/",
    # Worktrees (Hermes convention — don't recursively snapshot siblings)
    ".worktrees/",
    # Native / compiled binaries
    "*.so",
    "*.dylib",
    "*.dll",
    "*.o",
    "*.a",
    "*.jar",
    "*.class",
    "*.exe",
    "*.obj",
    # Media / large binaries
    "*.mp4",
    "*.mov",
    "*.mkv",
    "*.webm",
    "*.zip",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.7z",
    "*.rar",
    "*.iso",
    # Secrets
    ".env",
    ".env.*",
    ".env.local",
    ".env.*.local",
    # OS junk
    ".DS_Store",
    "Thumbs.db",
    # Logs
    "*.log",
]

# Git subprocess timeout (seconds).
_GIT_TIMEOUT: int = max(10, min(60, env_int("HERMES_CHECKPOINT_TIMEOUT", 30)))

# A git lock file (``<target>.lock``) is only held for the duration of a
# single git command, bounded by ``_GIT_TIMEOUT``.  Anything older than this
# is abandoned — left behind by a git process killed mid-write (gateway
# restart, crash, OOM) — and safe to remove.
_STALE_LOCK_MAX_AGE_S = 600


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
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    abs_workdir = _normalize_path(working_dir)
    resolved = (abs_workdir / file_path).resolve()
    try:
        resolved.relative_to(abs_workdir)
    except ValueError:
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


# ---------------------------------------------------------------------------
# Path / hash helpers
# ---------------------------------------------------------------------------

def _normalize_path(path_value: str) -> Path:
    """Return a canonical absolute path for checkpoint operations."""
    return Path(path_value).expanduser().resolve()


def _project_hash(working_dir: str) -> str:
    """Deterministic per-project hash: sha256(abs_path)[:16]."""
    abs_path = str(_normalize_path(working_dir))
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def _store_path(base: Optional[Path] = None) -> Path:
    """Return the single shared shadow store path."""
    return (base or CHECKPOINT_BASE) / _STORE_DIRNAME


def _shadow_repo_path(working_dir: str) -> Path:  # pragma: no cover — kept for BC
    """Return the shared store path.

    Retained for backward-compatibility with callers / tests that imported
    this helper.  Under v2 the shadow git storage is shared across all
    projects — per-project isolation lives in refs and indexes, not in
    separate repo directories.
    """
    return _store_path()


def _index_path(store: Path, dir_hash: str) -> Path:
    return store / _INDEXES_DIRNAME / dir_hash


def _ledger_path(store: Path, dir_hash: str) -> Path:
    return store / _LEDGERS_DIRNAME / f"{dir_hash}.json"


def _hash_file(path: Path) -> Optional[str]:
    """Streaming sha256 of a file's bytes. None if unreadable/missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load_ledger(store: Path, dir_hash: str) -> Dict[str, Dict]:
    """Load the agent-write ledger: {relpath: {"sha256": ..., "ts": ...}}.

    The ledger records the content hash of every file the last successful
    ``write_file`` / ``patch`` produced, so restores can tell "Hermes wrote
    this" apart from "the user hand-edited this afterwards".
    """
    try:
        raw = _ledger_path(store, dir_hash).read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_ledger(store: Path, dir_hash: str, ledger: Dict[str, Dict]) -> None:
    """Persist the agent-write ledger, capped to the newest entries."""
    try:
        if len(ledger) > _LEDGER_MAX_ENTRIES:
            newest = sorted(
                ledger.items(),
                key=lambda kv: kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0,
                reverse=True,
            )[:_LEDGER_MAX_ENTRIES]
            ledger = dict(newest)
        path = _ledger_path(store, dir_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ledger), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug("Failed to save agent-write ledger for %s", dir_hash, exc_info=True)


def _ref_name(dir_hash: str) -> str:
    return f"{_REFS_PREFIX}/{dir_hash}"


def _project_meta_path(store: Path, dir_hash: str) -> Path:
    return store / _PROJECTS_DIRNAME / f"{dir_hash}.json"


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
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
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


def _repair_bare_repo_dirs(store: Path) -> None:
    """Recreate refs/ and branches/ dirs that ``git gc`` may have removed.

    ``git gc --prune=now`` on a bare repo with only packed refs can remove
    the empty ``refs/heads/`` directory.  Git 2.34+ requires ``refs/`` (and
    some versions require ``branches/``) to exist even when all refs are
    packed in ``packed-refs``.  Without them, ``git add -A`` returns
    ``fatal: not a git repository`` and all checkpoint operations fail
    silently.
    """
    for subdir in ("refs/heads", "branches"):
        path = store / subdir
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.debug("Repaired missing %s in checkpoint store", subdir)
            except OSError as exc:
                logger.warning(
                    "Cannot create %s in checkpoint store: %s", subdir, exc,
                )


def _repair_stale_locks(store: Path) -> int:
    """Remove abandoned git lock files from the checkpoint store.

    Git creates ``<target>.lock`` files for the duration of a single command
    and deletes them when the command finishes.  If a git process is killed
    mid-write — a gateway restart with ``KillMode=mixed``, a crash, an OOM
    kill — the lock survives.  Every later git command that needs that
    target then fails with ``fatal: Unable to create '<target>.lock': File
    exists`` (rc=128) until a human removes the file by hand, silently
    breaking checkpoints for that project.

    Only locks older than ``_STALE_LOCK_MAX_AGE_S`` are removed — a live
    concurrent git process holds its lock for seconds at most, so it is
    never disturbed.  Returns the number of locks removed.
    """
    if not store.is_dir():
        return 0
    cutoff = time.time() - _STALE_LOCK_MAX_AGE_S
    removed = 0
    for root, dirs, files in os.walk(store):
        # Skip loose-object fan-out dirs; only pack locks can appear there.
        if Path(root).name == "objects":
            dirs[:] = [d for d in dirs if d == "pack"]
        for name in files:
            if not name.endswith(".lock"):
                continue
            path = Path(root) / name
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
                    logger.warning("Removed stale checkpoint lock: %s", path)
            except OSError:
                continue
    return removed


def _run_git(
    args: List[str],
    store: Path,
    working_dir: str,
    timeout: int = _GIT_TIMEOUT,
    allowed_returncodes: Optional[Set[int]] = None,
    index_file: Optional[Path] = None,
) -> Tuple[bool, str, str]:
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
    cmd = ["git"] + list(args)
    allowed_returncodes = allowed_returncodes or set()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=timeout,
            env=env,
            cwd=str(normalized_working_dir),
            stdin=subprocess.DEVNULL,
            # Checkpoints fire several bare git calls per turn from the
            # console-less desktop/gateway backend; suppress the per-call
            # conhost flash on Windows (no-op on POSIX).
            creationflags=windows_hide_flags(),
        )
        ok = result.returncode == 0
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if (
            not ok
            and result.returncode == 128
            and "Unable to create" in stderr
            and "File exists" in stderr
            and _repair_stale_locks(store) > 0
        ):
            # A stale lock from a killed git process blocked this command.
            # Repair and retry once — the lock is gone, so the retry is
            # expected to succeed.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True, encoding='utf-8', errors='replace',
                timeout=timeout,
                env=env,
                cwd=str(normalized_working_dir),
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
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


# ---------------------------------------------------------------------------
# Store initialisation + legacy migration
# ---------------------------------------------------------------------------

def _migrate_legacy_store(base: Path) -> Optional[Path]:
    """Move pre-v2 per-project shadow repos into a ``legacy-<ts>/`` dir.

    The pre-v2 layout had one shadow git repo per working directory directly
    under ``CHECKPOINT_BASE``.  The v2 layout wants a single ``store/`` dir.
    Rather than delete the old data (users might want to recover), rename
    everything except our own v2 entries into ``legacy-<timestamp>/``.  The
    legacy dir is subject to the same retention sweep and can be manually
    cleared with ``hermes checkpoints clear-legacy``.

    Returns the legacy-archive path, or None if nothing to migrate.
    """
    if not base.exists():
        return None
    store = _store_path(base)
    legacy_root: Optional[Path] = None
    # Reserved top-level entries managed by v2.
    reserved = {_STORE_DIRNAME, _PRUNE_MARKER_NAME}
    for child in list(base.iterdir()):
        name = child.name
        if name in reserved or name.startswith(_LEGACY_PREFIX):
            continue
        # Candidate: pre-v2 shadow repo (has HEAD) OR stray dir.  Either way
        # we archive it so v2 starts clean.
        if legacy_root is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            legacy_root = base / f"{_LEGACY_PREFIX}{stamp}"
            try:
                legacy_root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Could not create legacy archive dir: %s", exc)
                return None
        dest = legacy_root / name
        try:
            shutil.move(str(child), str(dest))
        except OSError as exc:
            logger.warning("Could not archive legacy checkpoint %s: %s", child, exc)
    # If the store still hasn't been created, create it here.
    _ = store
    if legacy_root is not None:
        logger.info(
            "Migrated pre-v2 checkpoint repos to %s. "
            "Clear with `hermes checkpoints clear-legacy` when safe.",
            legacy_root,
        )
    return legacy_root


def _init_store(store: Path, working_dir: str) -> Optional[str]:
    """Initialise the shared shadow store if needed.  Returns error or None.

    Also performs one-time migration of pre-v2 per-directory shadow repos
    into ``legacy-<timestamp>/``.
    """
    base = store.parent
    # One-time legacy migration before we create the store.
    if not store.exists():
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Could not create checkpoint base: {exc}"
        # Only migrate if the base dir has pre-existing content that isn't
        # our own v2 layout.
        _migrate_legacy_store(base)

    if (store / "HEAD").exists():
        # Sweep abandoned locks from killed git processes before any
        # operation, so snapshots never hit the rc=128 lock-exists failure.
        _repair_stale_locks(store)
        return None

    store.mkdir(parents=True, exist_ok=True)
    (store / _INDEXES_DIRNAME).mkdir(exist_ok=True)
    (store / _PROJECTS_DIRNAME).mkdir(exist_ok=True)

    # ``git init --bare`` rejects GIT_WORK_TREE, so we can't use _run_git
    # here (which always sets GIT_DIR + GIT_WORK_TREE).  Use a raw
    # subprocess with just the config-isolation env vars.
    from tools.environments.local import build_subprocess_env
    init_env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    init_env["GIT_CONFIG_GLOBAL"] = os.devnull
    init_env["GIT_CONFIG_SYSTEM"] = os.devnull
    init_env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Drop any inherited GIT_* that would interfere.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_NAMESPACE",
              "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        init_env.pop(k, None)
    try:
        result = subprocess.run(
            ["git", "init", "--bare", str(store)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            env=init_env, timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            return f"Shadow store init failed: {result.stderr.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"Shadow store init failed: {exc}"

    # Per-store config (isolated by env vars above, but belt-and-suspenders).
    # Use the base dir as the working_dir for config commands — it always
    # exists since we just created the store inside it.
    cfg_wd = str(base)
    _run_git(["config", "user.email", "hermes@local"], store, cfg_wd)
    _run_git(["config", "user.name", "Hermes Checkpoint"], store, cfg_wd)
    _run_git(["config", "commit.gpgsign", "false"], store, cfg_wd)
    _run_git(["config", "tag.gpgSign", "false"], store, cfg_wd)
    _run_git(["config", "gc.auto", "0"], store, cfg_wd)

    info_dir = store / "info"
    info_dir.mkdir(exist_ok=True)
    (info_dir / "exclude").write_text(
        "\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8"
    )

    logger.debug("Initialised checkpoint store at %s", store)
    return None


def _volume_evidence(workdir: Path) -> Dict:
    """Record the identity of ``workdir``'s parent while the project is live.

    ``(st_dev, st_ino)`` of the parent directory, captured at a moment when
    the workdir itself is reachable, identifies the *directory* — not just
    the path.  A mount point resolves to the mounted filesystem's root while
    the volume is attached and to the underlying (underlay) directory after
    unmount: same path, different directory, different ``(st_dev, st_ino)``.
    Orphan pruning uses this to distinguish "the project was deleted out of
    the directory we knew" from "a different directory is now visible at
    that path because the volume is detached".

    Returns ``{}`` when the workdir is not currently reachable, when the
    filesystem does not provide a usable directory identity (a zero
    ``st_dev`` or ``st_ino`` — e.g. Windows filesystems without file IDs and
    some network shares), or when the probe fails — callers treat all of
    these as "no evidence recorded" and orphan pruning stays conservative
    for the project (never classified as orphan; retention still applies).
    """
    try:
        if not workdir.exists():
            return {}
        st = workdir.parent.stat()
        if not st.st_dev or not st.st_ino:
            return {}
        return {
            "workdir_parent_dev": st.st_dev,
            "workdir_parent_ino": st.st_ino,
        }
    except OSError:
        return {}


def _register_project(store: Path, working_dir: str) -> None:
    """Create or update ``projects/<hash>.json`` with workdir + timestamps."""
    dir_hash = _project_hash(working_dir)
    meta_path = _project_meta_path(store, dir_hash)
    now = time.time()
    meta: Dict = {"workdir": str(_normalize_path(working_dir)),
                  "created_at": now, "last_touch": now}
    evidence = _volume_evidence(_normalize_path(working_dir))
    if evidence:
        meta.update(evidence)
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                meta["created_at"] = existing.get("created_at", now)
                if not evidence:
                    # Fresh probe failed — keep the previously recorded
                    # parent identity rather than dropping it. Stale evidence
                    # only makes pruning MORE conservative (mismatch => not
                    # an orphan).
                    for key in ("workdir_parent_dev", "workdir_parent_ino"):
                        if key in existing:
                            meta[key] = existing[key]
        except (OSError, ValueError):
            pass
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write project metadata %s: %s", meta_path, exc)


def _touch_project(store: Path, working_dir: str) -> None:
    """Update last_touch for a project, preserving created_at."""
    dir_hash = _project_hash(working_dir)
    meta_path = _project_meta_path(store, dir_hash)
    if not meta_path.exists():
        _register_project(store, working_dir)
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    meta["workdir"] = str(_normalize_path(working_dir))
    meta["last_touch"] = time.time()
    meta.setdefault("created_at", meta["last_touch"])
    # Refresh the parent-directory identity while the project is observably
    # live — a remount can legitimately change it (new device, new inode).
    # On probe failure the previous evidence is kept: stale evidence can only
    # make pruning MORE conservative (mismatch => not an orphan).
    evidence = _volume_evidence(_normalize_path(working_dir))
    if evidence:
        meta.update(evidence)
    try:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not update project metadata %s: %s", meta_path, exc)


def _list_projects(store: Path) -> List[Dict]:
    """Return all registered projects under the store."""
    projects_dir = store / _PROJECTS_DIRNAME
    if not projects_dir.exists():
        return []
    out: List[Dict] = []
    for meta_path in projects_dir.glob("*.json"):
        dir_hash = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        meta["_hash"] = dir_hash
        out.append(meta)
    return out


def _pre_v2_shadow_repos(base: Path) -> List[Dict]:
    """Return pre-v2 per-project shadow repos still directly under ``base``.

    Pre-v2 layout kept one shadow git repo per working directory directly
    under ``CHECKPOINT_BASE`` (identified by a ``HEAD`` file).  This is the
    single source of truth for that scan so a preview built from it (e.g.
    ``store_status``) always matches what ``prune_checkpoints`` deletes.
    """
    out: List[Dict] = []
    if not base.exists():
        return out
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == _STORE_DIRNAME or child.name.startswith(_LEGACY_PREFIX):
            continue
        if not (child / "HEAD").exists():
            continue
        workdir: Optional[str] = None
        marker_unreadable = False
        wd_marker = child / "HERMES_WORKDIR"
        if wd_marker.exists():
            try:
                workdir = wd_marker.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                # The marker is there, we just could not read it. That is
                # not evidence the project is gone — never delete on it.
                workdir = None
                marker_unreadable = True
        out.append({
            "path": child,
            "workdir": workdir,
            "exists": bool(workdir) and Path(workdir).exists(),
            "marker_unreadable": marker_unreadable,
        })
    return out


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


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size in bytes.  Returns 0 on error."""
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


# Backwards-compatibility shim — some tests import ``_init_shadow_repo`` and
# look for ``HEAD``/``info/exclude``/``HERMES_WORKDIR``.  In v2 we also write
# those markers, but inside the shared store + under ``projects/<hash>.json``.
# The shim initialises the store and registers the project so the old
# surface keeps roughly the same shape.
def _init_shadow_repo(shadow_repo: Path, working_dir: str) -> Optional[str]:
    """Backwards-compatible initialiser.

    In v1 ``shadow_repo`` was a per-project dir; in v2 it's the shared
    ``store/`` path (or a test path that we respect).  We initialise the
    store at ``shadow_repo``, create per-project markers, and return None
    on success.
    """
    err = _init_store(shadow_repo, working_dir)
    if err:
        return err
    _register_project(shadow_repo, working_dir)
    # Compat marker for tests that look at HERMES_WORKDIR
    # (write in addition to the JSON metadata).
    try:
        (shadow_repo / "HERMES_WORKDIR").write_text(
            str(_normalize_path(working_dir)) + "\n", encoding="utf-8"
        )
    except OSError:
        pass
    return None


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

    def record_agent_write(self, file_path: str) -> None:
        """Record the content hash of a file Hermes just successfully wrote.

        Feeds the agent-write ledger used by :meth:`restore` in safe mode:
        at restore time, a file whose current content no longer matches the
        recorded hash was hand-edited by the user after Hermes last touched
        it, and is skipped instead of clobbered.

        Never raises — the ledger is best-effort bookkeeping.
        """
        if not self.enabled:
            return
        try:
            path = _normalize_path(file_path)
            digest = _hash_file(path)
            if digest is None:
                return
            working_dir = self.get_working_dir_for_path(str(path))
            store = _store_path(CHECKPOINT_BASE)
            dir_hash = _project_hash(working_dir)
            ledger = _load_ledger(store, dir_hash)
            ledger[str(path)] = {"sha256": digest, "ts": time.time()}
            _save_ledger(store, dir_hash, ledger)
        except Exception as exc:
            logger.debug("record_agent_write failed for %s: %s", file_path, exc)

    def safe_restore_plan(self, working_dir: str, commit_hash: str) -> Dict:
        """Classify files changed since ``commit_hash`` for a safe restore.

        Returns ``{"success", "restore": [rel...], "skipped": [rel...],
        "error"?}`` where ``restore`` lists files whose current content
        still matches what Hermes last wrote (per the agent-write ledger)
        and ``skipped`` lists files the user hand-edited after Hermes'
        last write or that Hermes never wrote at all.
        """
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)
        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        # Stage the current tree so the name-only diff sees new files too.
        _run_git(["add", "-A"], store, abs_dir,
                 timeout=_GIT_TIMEOUT * 2, index_file=index_file)
        ok, names_out, err = _run_git(
            ["diff", "--name-only", commit_hash, "--cached"],
            store, abs_dir, index_file=index_file,
        )
        # Reset the index back to the project ref so it doesn't drift.
        _run_git(["read-tree", _ref_name(dir_hash)], store, abs_dir,
                 index_file=index_file, allowed_returncodes={128})
        if not ok:
            return {"success": False, "error": f"Could not compute changed files: {err}"}

        ledger = _load_ledger(store, dir_hash)
        if not ledger:
            # No agent-write ledger yet (pre-existing store, or Hermes has
            # not written any files here since the ledger was introduced).
            # Signal callers to fall back to a full restore rather than
            # skipping every file.
            return {"success": True, "restore": [], "skipped": [],
                    "ledger_empty": True}
        restore: List[str] = []
        skipped: List[str] = []
        for rel in names_out.splitlines():
            rel = rel.strip()
            if not rel:
                continue
            abs_path = Path(abs_dir) / rel
            entry = ledger.get(str(abs_path))
            recorded = entry.get("sha256") if isinstance(entry, dict) else None
            if recorded is None:
                # Hermes never wrote this file (or the ledger predates it) —
                # do not touch it in safe mode.
                skipped.append(rel)
                continue
            current = _hash_file(abs_path)
            if current is None:
                # File deleted since Hermes wrote it: restoring it back is
                # safe — its last content was Hermes-authored.
                restore.append(rel)
            elif current == recorded:
                restore.append(rel)
            else:
                skipped.append(rel)
        return {"success": True, "restore": restore, "skipped": skipped}

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        """Take a checkpoint if enabled and not already done this turn.

        Returns True if a checkpoint was taken, False otherwise.
        Never raises — all errors are silently logged.
        """
        if not self.enabled:
            return False

        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.debug("Checkpoints disabled: git not found")
        if not self._git_available:
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
            return self._take(abs_dir, reason)
        except Exception as e:
            logger.debug("Checkpoint failed (non-fatal): %s", e)
            return False

    def list_checkpoints(self, working_dir: str) -> List[Dict]:
        """List available checkpoints for a directory (most recent first)."""
        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)

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

        results: List[Dict] = []
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

    def list_all_checkpoints(self) -> List[Dict]:
        """List checkpoints across every registered project (most recent first).

        Surgical reapply of PR #10633 by @nightq (#10505) onto the v2
        single-store layout: iterate ``projects/<hash>.json`` metadata via
        ``_list_projects`` instead of the pre-v2 per-shadow-dir scan. Each
        entry carries the extra ``workdir`` key so callers can label which
        project a checkpoint belongs to.
        """
        store = _store_path(CHECKPOINT_BASE)
        if not (store / "HEAD").exists():
            return []
        results: List[Dict] = []
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
    def _parse_shortstat(stat_line: str, entry: Dict) -> None:
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

    def diff(self, working_dir: str, commit_hash: str) -> Dict:
        """Show diff between a checkpoint and the current working tree."""
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))
        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        # Stage current state into the per-project index to compare.
        _run_git(["add", "-A"], store, abs_dir,
                 timeout=_GIT_TIMEOUT * 2, index_file=index_file)

        ok_stat, stat_out, _ = _run_git(
            ["diff", "--stat", commit_hash, "--cached"],
            store, abs_dir, index_file=index_file,
        )
        ok_diff, diff_out, _ = _run_git(
            ["diff", commit_hash, "--cached", "--no-color"],
            store, abs_dir, index_file=index_file,
        )

        # Reset staged tree back to the project's last checkpoint so the
        # index doesn't drift out of sync with the ref.
        ref = _ref_name(dir_hash)
        _run_git(["read-tree", ref], store, abs_dir,
                 index_file=index_file,
                 allowed_returncodes={128})

        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}

        return {
            "success": True,
            "stat": stat_out if ok_stat else "",
            "diff": diff_out if ok_diff else "",
        }

    def session_diff(self, working_dir: str) -> Dict:
        """Show the cumulative diff of everything changed in this directory.

        This powers ``/diff session``.  It answers "what has Hermes changed
        here?" by diffing the *earliest retained checkpoint* — the snapshot
        taken before the first recorded edit — against the current working
        tree.  Because checkpoints are captured just before each file-mutating
        tool call, that baseline is the pre-edit state, so the diff covers the
        first edit and everything after it.

        Note: checkpoints are a persistent per-project ref, so the earliest
        *retained* checkpoint may predate the current session (or, after
        pruning, postdate its true start).  It is an approximation of "what
        Hermes changed", not an exact per-session ledger.

        Returns the same shape as :meth:`diff` (``{"success", "stat",
        "diff"}``).  When no checkpoints exist yet — nothing has been edited —
        the call still *succeeds* with empty output and ``"empty": True`` so
        callers can show a friendly "no changes" message rather than an error.
        """
        checkpoints = self.list_checkpoints(working_dir)
        if not checkpoints:
            return {"success": True, "stat": "", "diff": "", "empty": True}

        baseline = checkpoints[-1].get("hash") or ""
        result = self.diff(working_dir, baseline)
        if result.get("success"):
            result.setdefault("baseline", baseline)
            if not result.get("stat") and not result.get("diff"):
                result["empty"] = True
        return result

    def restore(
        self,
        working_dir: str,
        commit_hash: str,
        file_path: str = None,
        safe: bool = False,
    ) -> Dict:
        """Restore files to a checkpoint state.

        With ``safe=True`` (full-directory restores only), files the user
        hand-edited after Hermes' last write — per the agent-write ledger —
        are left untouched, and only Hermes-authored changes are reverted.
        The result gains ``skipped_user_edits`` listing the preserved paths,
        ``skipped_oversize`` listing paths kept because the size cap excluded
        them from every checkpoint, and — only when a delete failed —
        ``failed_deletes`` listing paths that could not be removed.
        """
        hash_err = _validate_commit_hash(commit_hash)
        if hash_err:
            return {"success": False, "error": hash_err}

        abs_dir = str(_normalize_path(working_dir))

        if file_path:
            path_err = _validate_file_path(file_path, abs_dir)
            if path_err:
                return {"success": False, "error": path_err}

        store = _store_path(CHECKPOINT_BASE)

        if not (store / "HEAD").exists():
            return {"success": False, "error": "No checkpoints exist for this directory"}

        ok, _, err = _run_git(
            ["cat-file", "-t", commit_hash], store, abs_dir,
        )
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found",
                    "debug": err or None}

        skipped_user_edits: List[str] = []
        kept_oversize: List[str] = []
        failed_deletes: List[str] = []
        restore_paths: Optional[List[str]] = None
        if safe and not file_path:
            plan = self.safe_restore_plan(abs_dir, commit_hash)
            if not plan.get("success"):
                return {"success": False, "error": plan.get("error", "Safe-restore plan failed")}
            if plan.get("ledger_empty"):
                # No agent-write history to compare against — fall back to
                # the classic full restore rather than restoring nothing.
                restore_paths = None
            else:
                restore_paths = plan["restore"]
                skipped_user_edits = plan["skipped"]
                if not restore_paths:
                    return {
                        "success": True,
                        "restored_to": commit_hash[:8],
                        "reason": "nothing to restore (all changed files were user-edited)",
                        "directory": abs_dir,
                        "restored_files": [],
                        "skipped_user_edits": skipped_user_edits,
                        "skipped_oversize": [],
                    }

        # Take a pre-rollback snapshot so you can undo the undo.
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        dir_hash = _project_hash(abs_dir)
        index_file = _index_path(store, dir_hash)

        if restore_paths is not None:
            # Split into files present in the checkpoint (checkout) and
            # Hermes-created files absent from it (delete to restore state).
            checkout_targets: List[str] = []
            delete_targets: List[str] = []
            for rel in restore_paths:
                ok_in_commit, _, _ = _run_git(
                    ["cat-file", "-e", f"{commit_hash}:{rel}"],
                    store, abs_dir, allowed_returncodes={1, 128},
                )
                if ok_in_commit:
                    checkout_targets.append(rel)
                elif self._exceeds_size_cap(Path(abs_dir) / rel):
                    # Absent from the checkpoint because ``max_file_size_mb``
                    # kept it out (_drop_oversize_from_index), not because
                    # Hermes created it. Deleting it would not restore a prior
                    # state — no checkpoint holds one — it would destroy the
                    # only copy. The ledger records a content hash, not whether
                    # a write created or modified the file, so an oversize path
                    # cannot be proven agent-created; leaving it costs a stale
                    # file, deleting it costs the file.
                    kept_oversize.append(rel)
                else:
                    delete_targets.append(rel)
            for rel in delete_targets:
                try:
                    target = Path(abs_dir) / rel
                    if target.is_file() or target.is_symlink():
                        target.unlink()
                except OSError as exc:
                    logger.warning(
                        "Safe restore: could not remove %s: %s", rel, exc,
                    )
                    failed_deletes.append(rel)
            if not checkout_targets:
                ok, stdout, err = True, "", ""
            else:
                ok, stdout, err = _run_git(
                    ["checkout", commit_hash, "--", *checkout_targets],
                    store, abs_dir, timeout=_GIT_TIMEOUT * 2,
                    index_file=index_file,
                )
        else:
            ok, stdout, err = _run_git(
                ["checkout", commit_hash, "--", file_path if file_path else "."],
                store, abs_dir, timeout=_GIT_TIMEOUT * 2,
                index_file=index_file,
            )

        if not ok:
            return {"success": False, "error": f"Restore failed: {err}",
                    "debug": err or None}

        ok2, reason_out, _ = _run_git(
            ["log", "--format=%s", "-1", commit_hash], store, abs_dir,
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
        if restore_paths is not None:
            # Only what was actually acted on. A kept oversize path was not
            # restored (and a failed unlink left the file in place), and
            # reporting either as restored is how the data loss above stayed
            # silent: the user was told "Restored" for a file that had just
            # been unlinked.
            not_restored = set(kept_oversize) | set(failed_deletes)
            result["restored_files"] = [
                rel for rel in restore_paths if rel not in not_restored
            ]
            result["skipped_user_edits"] = skipped_user_edits
            result["skipped_oversize"] = kept_oversize
            if failed_deletes:
                result["failed_deletes"] = failed_deletes
        return result

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory for checkpointing."""
        path = _normalize_path(file_path)
        if path.is_dir():
            candidate = path
        else:
            candidate = path.parent

        markers = {".git", "pyproject.toml", "package.json", "Cargo.toml",
                    "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile"}
        check = candidate
        while check != check.parent:
            if any((check / m).exists() for m in markers):
                return str(check)
            check = check.parent

        return str(candidate)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _take(self, working_dir: str, reason: str) -> bool:
        """Take a snapshot.  Returns True on success."""
        store = _store_path(CHECKPOINT_BASE)

        err = _init_store(store, working_dir)
        if err:
            logger.debug("Checkpoint store init failed: %s", err)
            return False

        _touch_project(store, working_dir)

        # Quick size guard — don't try to snapshot enormous directories
        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False

        dir_hash = _project_hash(working_dir)
        index_file = _index_path(store, dir_hash)
        ref = _ref_name(dir_hash)

        # Seed the per-project index from the last checkpoint, if any, so the
        # diff/commit machinery sees only changes since then.  On first call,
        # clear the index so ``git add -A`` produces a clean tree.
        if index_file.exists():
            # Reset index to current ref tip to avoid accumulating stale paths.
            ok_ref, ref_commit, _ = _run_git(
                ["rev-parse", "--verify", ref + "^{commit}"],
                store, working_dir,
                allowed_returncodes={128},
            )
            if ok_ref and ref_commit:
                _run_git(
                    ["read-tree", ref_commit],
                    store, working_dir,
                    index_file=index_file,
                    allowed_returncodes={128},
                )
            else:
                try:
                    index_file.unlink()
                except OSError:
                    pass
        else:
            # First snapshot for this project.
            index_file.parent.mkdir(parents=True, exist_ok=True)

        # Stage with per-project index.  Include a per-stage file-size filter
        # via ``core.bigFileThreshold`` is not what we want — instead, we
        # rely on the exclude file for broad patterns and post-stage prune
        # any path whose size exceeds max_file_size_mb.
        ok, _, err = _run_git(
            ["add", "-A"], store, working_dir,
            timeout=_GIT_TIMEOUT * 2, index_file=index_file,
        )
        if not ok:
            logger.debug("Checkpoint git-add failed: %s", err)
            return False

        if self.max_file_size_mb > 0:
            self._drop_oversize_from_index(store, working_dir, index_file)

        # Compare against the current ref tip (not HEAD — HEAD points to a
        # branch that doesn't exist on a bare store, so ``diff --cached``
        # against HEAD would always show "new file" for every staged path).
        ok_ref, ref_commit, _ = _run_git(
            ["rev-parse", "--verify", ref + "^{commit}"],
            store, working_dir,
            allowed_returncodes={128},
        )
        has_ref = ok_ref and bool(ref_commit)

        if has_ref:
            ok_diff, _, _ = _run_git(
                ["diff-index", "--cached", "--quiet", ref_commit],
                store, working_dir,
                allowed_returncodes={1},
                index_file=index_file,
            )
            if ok_diff:
                logger.debug("Checkpoint skipped: no changes in %s", working_dir)
                return False
        else:
            # No ref yet — skip only if the index is empty.
            ok_ls, ls_out, _ = _run_git(
                ["ls-files", "--cached"],
                store, working_dir,
                index_file=index_file,
            )
            if ok_ls and not ls_out.strip():
                logger.debug("Checkpoint skipped: empty tree in %s", working_dir)
                return False

        # Write tree from per-project index.
        ok_tree, tree_sha, err = _run_git(
            ["write-tree"], store, working_dir,
            index_file=index_file,
        )
        if not ok_tree or not tree_sha:
            logger.debug("Checkpoint write-tree failed: %s", err)
            return False

        # Build commit (parent = current ref tip, if any).
        commit_args = ["commit-tree", tree_sha, "-m", reason, "--no-gpg-sign"]
        if has_ref:
            commit_args = ["commit-tree", tree_sha, "-p", ref_commit, "-m", reason, "--no-gpg-sign"]
        ok_commit, new_sha, err = _run_git(
            commit_args, store, working_dir,
            index_file=index_file,
        )
        if not ok_commit or not new_sha:
            logger.debug("Checkpoint commit-tree failed: %s", err)
            return False

        # Update the per-project ref.
        update_args = ["update-ref", ref, new_sha]
        if has_ref:
            update_args = ["update-ref", ref, new_sha, ref_commit]
        ok_update, _, err = _run_git(
            update_args, store, working_dir,
        )
        if not ok_update:
            logger.debug("Checkpoint update-ref failed: %s", err)
            return False

        logger.debug("Checkpoint taken in %s: %s (%s)", working_dir, reason, new_sha[:8])

        # Real pruning — drop old commits beyond max_snapshots.
        self._prune(store, working_dir, ref)

        # Enforce global size cap.
        self._enforce_size_cap(store)

        return True

    def _exceeds_size_cap(self, path: Path) -> bool:
        """Whether *path* is larger than ``max_file_size_mb``.

        The same test :meth:`_drop_oversize_from_index` applies when building a
        checkpoint, so "excluded from the checkpoint" and "refused deletion at
        restore" agree on one definition. A cap of 0 disables it, and an
        unstattable path is not claimed to be oversize.
        """
        cap = self.max_file_size_mb * 1024 * 1024
        if cap <= 0:
            return False
        try:
            return path.stat().st_size > cap
        except OSError:
            return False

    def _drop_oversize_from_index(
        self, store: Path, working_dir: str, index_file: Path,
    ) -> None:
        """Remove any staged file larger than ``max_file_size_mb`` from the index.

        Lets the agent keep snapshotting source code while refusing to
        swallow generated assets (datasets, model weights, logs, videos).
        """
        if self.max_file_size_mb <= 0:
            return
        ok, stdout, _ = _run_git(
            ["ls-files", "--cached", "-z"],
            store, working_dir, index_file=index_file,
        )
        if not ok or not stdout:
            return
        # ls-files -z output is NUL-separated. _run_git strips trailing
        # whitespace but that leaves NULs alone; rebuild list.
        paths = [p for p in stdout.split("\x00") if p]
        abs_workdir = _normalize_path(working_dir)
        # Same predicate safe restore consults, called rather than restated:
        # a threshold that drifted between the two would make a file both
        # absent from the checkpoint and not recognised as capped at restore,
        # which is precisely the deletion this change exists to prevent.
        oversize = [
            rel for rel in paths if self._exceeds_size_cap(abs_workdir / rel)
        ]
        if not oversize:
            return
        logger.debug(
            "Checkpoint: dropping %d oversize file(s) (>%d MB) from index",
            len(oversize), self.max_file_size_mb,
        )
        # Use --pathspec-from-file for safety with many paths.
        # Chunk into manageable batches.
        BATCH = 200
        for i in range(0, len(oversize), BATCH):
            chunk = oversize[i:i + BATCH]
            _run_git(
                ["rm", "--cached", "--quiet", "--"] + chunk,
                store, working_dir, index_file=index_file,
                allowed_returncodes={128},
            )

    def _prune(self, store: Path, working_dir: str, ref: str) -> None:
        """Keep only the last ``max_snapshots`` commits on the per-project ref.

        v1's ``_prune`` was documented as a no-op (``git``'s pack mechanism
        was supposed to handle it, but only the log view was limited — loose
        objects accumulated forever).  v2 actually rewrites the ref to drop
        commits older than ``max_snapshots`` and then runs ``git gc`` on the
        store so unreachable objects are reclaimed.
        """
        ok, stdout, _ = _run_git(
            ["rev-list", "--count", ref], store, working_dir,
            allowed_returncodes={128},
        )
        if not ok:
            return
        try:
            count = int(stdout)
        except ValueError:
            return
        if count <= self.max_snapshots:
            return

        # Collect commits oldest → newest, take last N.
        ok_list, list_out, _ = _run_git(
            ["rev-list", "--reverse", ref], store, working_dir,
        )
        if not ok_list or not list_out:
            return
        commits = list_out.splitlines()
        keep = commits[-self.max_snapshots:]

        # Rebuild a linear chain off keep[0]'s tree.
        new_parent: Optional[str] = None
        for sha in keep:
            ok_tree, tree_sha, _ = _run_git(
                ["rev-parse", f"{sha}^{{tree}}"], store, working_dir,
            )
            if not ok_tree or not tree_sha:
                return
            ok_msg, msg, _ = _run_git(
                ["log", "--format=%s", "-1", sha], store, working_dir,
            )
            commit_msg = msg if ok_msg and msg else "checkpoint"
            args = ["commit-tree", tree_sha, "-m", commit_msg, "--no-gpg-sign"]
            if new_parent is not None:
                args = ["commit-tree", tree_sha, "-p", new_parent,
                        "-m", commit_msg, "--no-gpg-sign"]
            ok_commit, new_sha, _ = _run_git(args, store, working_dir)
            if not ok_commit or not new_sha:
                return
            new_parent = new_sha

        if new_parent is None:
            return
        _run_git(["update-ref", ref, new_parent], store, working_dir)

        # Reclaim objects from the dropped commits.
        _run_git(
            ["reflog", "expire", "--expire=now", "--all"],
            store, working_dir,
        )
        _run_git(
            ["gc", "--prune=now", "--quiet"],
            store, working_dir, timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)

    def _enforce_size_cap(self, store: Path) -> None:
        """If total store size exceeds ``max_total_size_mb``, drop oldest
        checkpoints across ALL projects until under the cap.
        """
        if self.max_total_size_mb <= 0:
            return
        cap_bytes = self.max_total_size_mb * 1024 * 1024
        size = _dir_size_bytes(store)
        if size <= cap_bytes:
            return
        logger.info(
            "Checkpoint store exceeded %d MB (actual %d MB) — pruning oldest",
            self.max_total_size_mb, size // (1024 * 1024),
        )

        # Collect (commit_time, ref, sha) across all per-project refs.
        ok, stdout, _ = _run_git(
            ["for-each-ref", "--format=%(refname)", _REFS_PREFIX],
            store, str(store.parent),
            allowed_returncodes={128},
        )
        if not ok or not stdout:
            return
        refs = [r for r in stdout.splitlines() if r.strip()]

        any_dropped = False
        # Round-robin-drop oldest commit per ref until under cap.
        for _ in range(20):  # hard upper bound to avoid pathological loops
            size = _dir_size_bytes(store)
            if size <= cap_bytes:
                break
            for ref in refs:
                ok_count, count_out, _ = _run_git(
                    ["rev-list", "--count", ref], store, str(store.parent),
                    allowed_returncodes={128},
                )
                try:
                    count = int(count_out) if ok_count else 0
                except ValueError:
                    count = 0
                if count <= 1:
                    continue  # keep at least one snapshot per project
                ok_list, list_out, _ = _run_git(
                    ["rev-list", "--reverse", ref], store, str(store.parent),
                )
                if not ok_list or not list_out:
                    continue
                commits = list_out.splitlines()
                keep = commits[1:]  # drop oldest
                new_parent: Optional[str] = None
                fail = False
                for sha in keep:
                    ok_tree, tree_sha, _ = _run_git(
                        ["rev-parse", f"{sha}^{{tree}}"], store, str(store.parent),
                    )
                    if not ok_tree or not tree_sha:
                        fail = True
                        break
                    ok_msg, msg, _ = _run_git(
                        ["log", "--format=%s", "-1", sha], store, str(store.parent),
                    )
                    commit_msg = msg if ok_msg and msg else "checkpoint"
                    args = ["commit-tree", tree_sha, "-m", commit_msg, "--no-gpg-sign"]
                    if new_parent is not None:
                        args = ["commit-tree", tree_sha, "-p", new_parent,
                                "-m", commit_msg, "--no-gpg-sign"]
                    ok_commit, new_sha, _ = _run_git(args, store, str(store.parent))
                    if not ok_commit or not new_sha:
                        fail = True
                        break
                    new_parent = new_sha
                if fail or new_parent is None:
                    continue
                _run_git(["update-ref", ref, new_parent], store, str(store.parent))
                any_dropped = True
            if not any_dropped:
                break

        _run_git(
            ["reflog", "expire", "--expire=now", "--all"],
            store, str(store.parent),
        )
        _run_git(
            ["gc", "--prune=now", "--quiet"],
            store, str(store.parent), timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)


def format_checkpoint_list(checkpoints: List[Dict], directory: str) -> str:
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


# ---------------------------------------------------------------------------
# Auto-maintenance
# ---------------------------------------------------------------------------
#
# v2 rewrite.  The sweep now operates on per-project refs inside the shared
# store rather than per-project shadow repos.  Legacy-archive dirs
# (``legacy-<ts>/``) are swept with the same retention policy.

_PRUNE_MARKER_NAME = ".last_prune"


def _delete_ref(store: Path, ref: str) -> bool:
    """Delete a ref from the store.  Returns True on success."""
    ok, _, _ = _run_git(
        ["update-ref", "-d", ref], store, str(store.parent),
        allowed_returncodes={128},
    )
    return ok


def _workdir_is_observably_gone(
    workdir: str,
    parent_dev: Optional[int] = None,
    parent_ino: Optional[int] = None,
    require_parent_identity: bool = True,
) -> bool:
    """True only when we can positively observe that ``workdir`` was removed.

    ``Path.exists()`` returns False for a deleted directory AND for one whose
    storage simply is not attached right now — an unplugged external drive, a
    network share behind a downed VPN, a bind-mount absent from this
    container, an offline Windows mapped drive. Orphan pruning deletes the
    project's entire checkpoint history, so treating that ambiguity as
    "deleted" throws away the user's restore points over a transient mount
    state, unattended, at startup.

    Require corroboration, in three steps.

    First, the parent directory must be present, so the absence of the project
    inside it is something we actually observed. When the parent is missing
    too, the volume is not there and we know nothing.

    Second, the present parent must be the directory we knew — not merely a
    directory at the same path. Unmounting swaps the directory visible at a
    mount point: while the volume is attached the path resolves to the
    mounted filesystem's root; after detach it resolves to the *underlying*
    (underlay) directory, which may carry entries of its own (a ``.keep``
    placeholder, sibling mount points). Those entries were never next to the
    project and prove nothing about the volume being attached. So the
    parent's ``(st_dev, st_ino)`` must match the identity recorded in the
    project's metadata while the project was observably live
    (``parent_dev``/``parent_ino``). A mismatch means a different directory
    is visible at that path — a detached volume, not an observed deletion.
    When no identity was ever recorded (metadata written by an older
    version) and ``require_parent_identity`` is True, stay conservative and
    do not classify as orphan. Callers that have no identity channel at all
    (the frozen pre-v2 layout) pass ``require_parent_identity=False`` to
    keep the structural checks only.

    Third, the (identity-confirmed) parent must actually carry information.
    Unmounting leaves classic static mount points (``/mnt/volume/proj``, an
    fstab entry, a container bind-mount) behind as *empty* directories, so an
    empty parent is the signature of a detached volume just as much as of a
    deleted project. Prune only when the parent holds something else (we
    observed a populated directory that does not contain the project) or is
    itself a live mount point (the volume is demonstrably attached and the
    project is demonstrably not on it).

    Genuinely abandoned projects are still reclaimed by the retention/stale
    rule, which runs off ``last_touch`` rather than a filesystem probe.
    """
    if not workdir:
        return False
    path = Path(workdir)
    try:
        if path.exists():
            return False
        parent = path.parent
        # A path whose parent is itself (a filesystem root) gives us nothing
        # to corroborate against.
        if parent == path:
            return False
        if not parent.is_dir():
            return False
        if parent_dev is not None and parent_ino is not None:
            st = parent.stat()
            if (st.st_dev, st.st_ino) != (parent_dev, parent_ino):
                # A different directory is visible at the parent's path than
                # the one the project lived in — the volume is detached (its
                # underlay showing through) or was swapped. Not a deletion.
                return False
        elif require_parent_identity:
            # No recorded identity to check against — we cannot tell the
            # project's real parent from an underlay directory exposed by an
            # unmount. Unsure never deletes; retention still reclaims.
            return False
        if _dir_has_any_entry(parent):
            return True
        # Empty parent: only evidence if that directory is a mount point, i.e.
        # the volume is attached right now and simply does not hold the
        # project. An empty plain directory is an unmounted mount point as
        # readily as an emptied project root.
        return os.path.ismount(parent)
    except OSError:
        # Probe failed (permission, I/O error) — not evidence of deletion.
        return False


def _dir_has_any_entry(directory: Path) -> bool:
    """True when ``directory`` contains at least one entry.

    Stops after the first entry rather than materializing the listing; a
    project root can hold a large tree.
    """
    with os.scandir(directory) as entries:
        for _ in entries:
            return True
    return False


def prune_checkpoints(
    retention_days: int = 7,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
    max_total_size_mb: int = 0,
    orphan_allowlist: Optional[set] = None,
) -> Dict[str, int]:
    """Delete stale/orphan checkpoints and reclaim store space.

    A project entry is deleted when either:

    * ``delete_orphans=True`` and its ``workdir`` no longer exists on disk
      (the original project was deleted / moved); OR
    * its ``last_touch`` is older than ``retention_days`` days.

    ``orphan_allowlist``, when not ``None``, restricts orphan deletion to
    the given identities (v2 project ``_hash`` strings and/or pre-v2 shadow
    repo paths as ``str``). This lets a caller that showed the user a
    confirmation preview (built from ``store_status()``) bind the resulting
    deletion to exactly what was displayed — a project that only becomes
    orphaned *after* the preview (e.g. its workdir vanishes while the human
    is answering the prompt) is skipped rather than swept up under the
    earlier confirmation. Pass ``None`` (the default) to delete every
    currently-orphaned project, e.g. for ``--force`` or unattended callers
    that never show a preview.

    Additionally, if ``max_total_size_mb > 0`` and the store exceeds that
    after orphan/stale pruning, the oldest commit per remaining project is
    dropped until the store is under the cap.

    Legacy-archive dirs (``legacy-*``) older than ``retention_days`` are
    also deleted.

    Returns a dict with counts ``{"scanned", "deleted_orphan",
    "deleted_stale", "errors", "bytes_freed"}``.

    Never raises — maintenance must never block interactive startup.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    result = {
        "scanned": 0,
        "deleted_orphan": 0,
        "deleted_stale": 0,
        "errors": 0,
        "bytes_freed": 0,
    }
    if not base.exists():
        return result

    size_before = _dir_size_bytes(base)

    # --- Legacy pre-v2 per-project shadow repos (kept directly under base) ---
    # Pre-v2 layout: ``base/<hash>/HEAD`` etc.  We treat these exactly as the
    # v1 pruner did so behaviour is unchanged for anyone still on that layout
    # or sitting on a mid-migration system.
    cutoff = 0.0
    if retention_days > 0:
        cutoff = time.time() - retention_days * 86400

    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.name == _STORE_DIRNAME:
            continue
        if child.name.startswith(_LEGACY_PREFIX):
            # Legacy archive: prune by dir mtime using same retention rule.
            if retention_days <= 0:
                continue
            try:
                m = child.stat().st_mtime
            except OSError:
                continue
            if m >= cutoff:
                continue
            try:
                size = _dir_size_bytes(child)
                shutil.rmtree(child)
                result["bytes_freed"] += size
                result["deleted_stale"] += 1
            except OSError as exc:
                result["errors"] += 1
                logger.warning("Failed to delete legacy archive %s: %s", child, exc)

    # Pre-v2 per-project shadow repos.  Scanned via the same helper
    # `store_status()` uses for its orphan preview, so a confirmation prompt
    # built from that preview always matches what gets deleted here.
    for repo in _pre_v2_shadow_repos(base):
        child = repo["path"]
        result["scanned"] += 1
        reason: Optional[str] = None
        if (
            delete_orphans
            and not repo["marker_unreadable"]
            and (
                repo["workdir"] is None
                # The frozen pre-v2 layout has no metadata channel to carry a
                # recorded parent identity, so only the structural checks
                # (parent present + populated / live mount point) apply here.
                or _workdir_is_observably_gone(
                    repo["workdir"], require_parent_identity=False,
                )
            )
            and (orphan_allowlist is None or str(child) in orphan_allowlist)
        ):
            reason = "orphan"
        if reason is None and retention_days > 0:
            newest = 0.0
            try:
                for p in child.rglob("*"):
                    try:
                        mt = p.stat().st_mtime
                        newest = max(newest, mt)
                    except OSError:
                        continue
            except OSError:
                pass
            if newest > 0 and newest < cutoff:
                reason = "stale"
        if reason is None:
            continue
        try:
            size = _dir_size_bytes(child)
            shutil.rmtree(child)
            result["bytes_freed"] += size
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1
        except OSError as exc:
            result["errors"] += 1
            logger.warning("Failed to prune checkpoint repo %s: %s", child.name, exc)

    # --- v2 shared store: per-project ref pruning via metadata ---
    store = _store_path(base)
    if (store / "HEAD").exists():
        for meta in _list_projects(store):
            dir_hash = meta.get("_hash") or ""
            workdir = meta.get("workdir") or ""
            if not dir_hash:
                continue
            result["scanned"] += 1
            reason = None
            parent_dev = meta.get("workdir_parent_dev")
            parent_ino = meta.get("workdir_parent_ino")
            if not isinstance(parent_dev, int) or isinstance(parent_dev, bool):
                parent_dev = None
            if not isinstance(parent_ino, int) or isinstance(parent_ino, bool):
                parent_ino = None
            if (
                delete_orphans
                and (
                    not workdir
                    or _workdir_is_observably_gone(
                        workdir,
                        parent_dev=parent_dev,
                        parent_ino=parent_ino,
                    )
                )
                and (orphan_allowlist is None or dir_hash in orphan_allowlist)
            ):
                reason = "orphan"
            elif retention_days > 0:
                last_touch = float(meta.get("last_touch", 0) or 0)
                if last_touch > 0 and last_touch < cutoff:
                    reason = "stale"
            if reason is None:
                continue
            ref = _ref_name(dir_hash)
            _delete_ref(store, ref)
            # Drop per-project index and metadata.
            try:
                idx = _index_path(store, dir_hash)
                if idx.exists():
                    idx.unlink()
            except OSError:
                pass
            try:
                mp = _project_meta_path(store, dir_hash)
                if mp.exists():
                    mp.unlink()
            except OSError:
                pass
            if reason == "orphan":
                result["deleted_orphan"] += 1
            else:
                result["deleted_stale"] += 1

        # GC the store to reclaim unreachable objects from dropped refs.
        _run_git(
            ["reflog", "expire", "--expire=now", "--all"],
            store, str(base),
        )
        _run_git(
            ["gc", "--prune=now", "--quiet"],
            store, str(base), timeout=_GIT_TIMEOUT * 3,
        )
        _repair_bare_repo_dirs(store)

        # Size-cap pass across remaining projects.
        if max_total_size_mb > 0:
            cap_bytes = max_total_size_mb * 1024 * 1024
            for _i in range(20):
                size = _dir_size_bytes(store)
                if size <= cap_bytes:
                    break
                ok, stdout, _ = _run_git(
                    ["for-each-ref", "--format=%(refname)", _REFS_PREFIX],
                    store, str(base),
                    allowed_returncodes={128},
                )
                refs = [r for r in stdout.splitlines() if r.strip()] if ok else []
                if not refs:
                    break
                any_drop = False
                for ref in refs:
                    ok_c, count_out, _ = _run_git(
                        ["rev-list", "--count", ref], store, str(base),
                        allowed_returncodes={128},
                    )
                    try:
                        count = int(count_out) if ok_c else 0
                    except ValueError:
                        count = 0
                    if count <= 1:
                        continue
                    ok_l, lo, _ = _run_git(
                        ["rev-list", "--reverse", ref], store, str(base),
                    )
                    if not ok_l or not lo:
                        continue
                    commits = lo.splitlines()
                    keep = commits[1:]
                    new_parent: Optional[str] = None
                    fail = False
                    for sha in keep:
                        ok_t, tsha, _ = _run_git(
                            ["rev-parse", f"{sha}^{{tree}}"], store, str(base),
                        )
                        if not ok_t or not tsha:
                            fail = True
                            break
                        ok_m, m, _ = _run_git(
                            ["log", "--format=%s", "-1", sha], store, str(base),
                        )
                        msg = m if ok_m and m else "checkpoint"
                        args = ["commit-tree", tsha, "-m", msg, "--no-gpg-sign"]
                        if new_parent is not None:
                            args = ["commit-tree", tsha, "-p", new_parent,
                                    "-m", msg, "--no-gpg-sign"]
                        ok_cm, new_sha, _ = _run_git(args, store, str(base))
                        if not ok_cm or not new_sha:
                            fail = True
                            break
                        new_parent = new_sha
                    if fail or new_parent is None:
                        continue
                    _run_git(["update-ref", ref, new_parent], store, str(base))
                    any_drop = True
                if not any_drop:
                    break
            _run_git(
                ["reflog", "expire", "--expire=now", "--all"],
                store, str(base),
            )
            _run_git(
                ["gc", "--prune=now", "--quiet"],
                store, str(base), timeout=_GIT_TIMEOUT * 3,
            )
            _repair_bare_repo_dirs(store)

    size_after = _dir_size_bytes(base)
    delta = size_before - size_after
    result["bytes_freed"] = max(result["bytes_freed"], delta)

    return result


def maybe_auto_prune_checkpoints(
    retention_days: int = 7,
    min_interval_hours: int = 24,
    delete_orphans: bool = True,
    checkpoint_base: Optional[Path] = None,
    max_total_size_mb: int = 0,
) -> Dict[str, object]:
    """Idempotent wrapper around ``prune_checkpoints`` for startup hooks.

    Writes ``CHECKPOINT_BASE/.last_prune`` on completion so subsequent
    calls within ``min_interval_hours`` short-circuit.

    Returns ``{"skipped": bool, "result": prune_checkpoints-dict,
    "error": optional str}``.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict[str, object] = {"skipped": False}

    try:
        if not base.exists():
            out["result"] = {
                "scanned": 0, "deleted_orphan": 0, "deleted_stale": 0,
                "errors": 0, "bytes_freed": 0,
            }
            return out

        marker = base / _PRUNE_MARKER_NAME
        now = time.time()
        if marker.exists():
            try:
                last_ts = float(marker.read_text(encoding="utf-8").strip())
                if now - last_ts < min_interval_hours * 3600:
                    out["skipped"] = True
                    return out
            except (OSError, ValueError):
                pass  # corrupt marker — treat as no prior run

        result = prune_checkpoints(
            retention_days=retention_days,
            delete_orphans=delete_orphans,
            checkpoint_base=base,
            max_total_size_mb=max_total_size_mb,
        )
        out["result"] = result

        try:
            marker.write_text(str(now), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write checkpoint prune marker: %s", exc)

        total = result["deleted_orphan"] + result["deleted_stale"]
        if total > 0:
            logger.info(
                "checkpoint auto-maintenance: pruned %d entry(ies) "
                "(%d orphan, %d stale), reclaimed %.1f MB",
                total,
                result["deleted_orphan"],
                result["deleted_stale"],
                result["bytes_freed"] / (1024 * 1024),
            )
    except Exception as exc:
        logger.warning("checkpoint auto-maintenance failed: %s", exc)
        out["error"] = str(exc)

    return out


# ---------------------------------------------------------------------------
# Public helpers for `hermes checkpoints` CLI
# ---------------------------------------------------------------------------

def store_status(checkpoint_base: Optional[Path] = None) -> Dict:
    """Return a summary of the shadow store.

    ``{"base": path, "store_size_bytes": N, "legacy_size_bytes": N,
       "total_size_bytes": N, "project_count": N, "projects": [...],
       "pre_v2_projects": [...], "legacy_archives": [...]}``

    ``pre_v2_projects`` covers shadow repos still on the pre-v2 per-project
    layout (``base/<hash>/HEAD``) — distinct from ``legacy_archives``, which
    are already-migrated ``legacy-<ts>/`` dirs. Callers that preview an
    orphan-deletion sweep must include both ``projects`` and
    ``pre_v2_projects``, since ``prune_checkpoints`` deletes orphans from
    both layouts.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict = {
        "base": str(base),
        "store_size_bytes": 0,
        "legacy_size_bytes": 0,
        "total_size_bytes": 0,
        "project_count": 0,
        "projects": [],
        "pre_v2_projects": [],
        "legacy_archives": [],
    }
    if not base.exists():
        return out

    store = _store_path(base)
    if store.exists():
        out["store_size_bytes"] = _dir_size_bytes(store)
        if (store / "HEAD").exists():
            for meta in _list_projects(store):
                dir_hash = meta.get("_hash") or ""
                workdir = meta.get("workdir") or ""
                ref = _ref_name(dir_hash)
                ok, count_out, _ = _run_git(
                    ["rev-list", "--count", ref], store, str(base),
                    allowed_returncodes={128},
                )
                try:
                    commits = int(count_out) if ok else 0
                except ValueError:
                    commits = 0
                out["projects"].append({
                    "hash": dir_hash,
                    "workdir": workdir,
                    "exists": bool(workdir) and Path(workdir).exists(),
                    "created_at": meta.get("created_at"),
                    "last_touch": meta.get("last_touch"),
                    "commits": commits,
                })
    out["project_count"] = len(out["projects"])

    out["pre_v2_projects"] = [
        {
            "path": str(r["path"]),
            "workdir": r["workdir"],
            "exists": r["exists"],
        }
        for r in _pre_v2_shadow_repos(base)
    ]

    for child in base.iterdir():
        if child.is_dir() and child.name.startswith(_LEGACY_PREFIX):
            try:
                size = _dir_size_bytes(child)
            except OSError:
                size = 0
            out["legacy_size_bytes"] += size
            try:
                mt = child.stat().st_mtime
            except OSError:
                mt = 0
            out["legacy_archives"].append({
                "name": child.name,
                "size_bytes": size,
                "mtime": mt,
            })

    out["total_size_bytes"] = _dir_size_bytes(base)
    return out


def clear_all(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Nuke the entire checkpoint base (store + legacy).  Irreversible.

    Returns ``{"bytes_freed": N, "deleted": bool}``.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out = {"bytes_freed": 0, "deleted": False}
    if not base.exists():
        return out
    size = _dir_size_bytes(base)
    try:
        shutil.rmtree(base)
        out["bytes_freed"] = size
        out["deleted"] = True
    except OSError as exc:
        logger.warning("Could not clear checkpoint base %s: %s", base, exc)
    return out


def clear_legacy(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Delete all ``legacy-*`` archive directories.

    Returns ``{"bytes_freed": N, "deleted": count}``.
    """
    base = checkpoint_base or CHECKPOINT_BASE
    out = {"bytes_freed": 0, "deleted": 0}
    if not base.exists():
        return out
    for child in list(base.iterdir()):
        if not child.is_dir() or not child.name.startswith(_LEGACY_PREFIX):
            continue
        try:
            size = _dir_size_bytes(child)
            shutil.rmtree(child)
            out["bytes_freed"] += size
            out["deleted"] += 1
        except OSError as exc:
            logger.warning("Could not delete legacy archive %s: %s", child, exc)
    return out

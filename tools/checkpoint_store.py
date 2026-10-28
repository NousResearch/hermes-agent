"""Store layout, initialisation, migration and project metadata for checkpoints.

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

The shadow store uses ``GIT_DIR`` + ``GIT_WORK_TREE`` + ``GIT_INDEX_FILE`` so
no git state leaks into the user's project directory.

Seam-sensitive names (``_run_git``, ``_resolve_checkpoint_base``) stay defined
in ``checkpoint_manager``; the delegates at the bottom resolve them at call
time, so ``monkeypatch.setattr(checkpoint_manager, ...)`` is still observed
here.
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from hermes_cli._subprocess_compat import selected_git_env, windows_hide_flags

from utils import env_int

logger = logging.getLogger(__name__)

# Single shared store directory under CHECKPOINT_BASE.
_STORE_DIRNAME = "store"
_REFS_PREFIX = "refs/hermes"
_INDEXES_DIRNAME = "indexes"
_PROJECTS_DIRNAME = "projects"
_LEDGERS_DIRNAME = "ledgers"
_LEGACY_PREFIX = "legacy-"
_PRUNE_MARKER_NAME = ".last_prune"


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


def _normalize_path(path_value: str) -> Path:
    """Return a canonical absolute path for checkpoint operations."""
    return Path(path_value).expanduser().resolve()


def _project_hash(working_dir: str) -> str:
    """Deterministic per-project hash: sha256(abs_path)[:16]."""
    abs_path = str(_normalize_path(working_dir))
    return hashlib.sha256(abs_path.encode()).hexdigest()[:16]


def _store_path(base: Optional[Path] = None) -> Path:
    """Return the single shared shadow store path."""
    return (base or _resolve_checkpoint_base()) / _STORE_DIRNAME


def _store_has_head(store: Path) -> bool:
    return (store / "HEAD").exists()


def _index_path(store: Path, dir_hash: str) -> Path:
    return store / _INDEXES_DIRNAME / dir_hash


def _ref_name(dir_hash: str) -> str:
    return f"{_REFS_PREFIX}/{dir_hash}"


def _gitlink_paths(ls_tree_z: str) -> list[str]:
    """Paths of gitlinks (nested repositories stored as a commit reference, not
    their files) in ``git ls-tree -r -z`` output."""
    return [
        record.split("\t", 1)[1]
        for record in ls_tree_z.split("\x00")
        if record.startswith("160000 commit ") and "\t" in record
    ]


def _project_meta_path(store: Path, dir_hash: str) -> Path:
    return store / _PROJECTS_DIRNAME / f"{dir_hash}.json"


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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
        return None

    store.mkdir(parents=True, exist_ok=True)
    (store / _INDEXES_DIRNAME).mkdir(exist_ok=True)
    (store / _PROJECTS_DIRNAME).mkdir(exist_ok=True)

    # ``git init --bare`` rejects GIT_WORK_TREE, so we can't use _run_git
    # here (which always sets GIT_DIR + GIT_WORK_TREE).  Use a raw
    # subprocess with just the config-isolation env vars.
    from tools.environments.local import build_subprocess_env

    init_env = selected_git_env(build_subprocess_env(scrub_secrets=False, inherit_profile_home=False))
    init_env["GIT_CONFIG_GLOBAL"] = os.devnull
    init_env["GIT_CONFIG_SYSTEM"] = os.devnull
    init_env["GIT_CONFIG_NOSYSTEM"] = "1"
    # Drop any inherited GIT_* that would interfere.
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_NAMESPACE",
              "GIT_ALTERNATE_OBJECT_DIRECTORIES"):
        init_env.pop(k, None)
    try:
        git = shutil.which("git", path=init_env.get("PATH", ""))
        if git is None:
            return "Shadow store init failed: git is not installed or not on PATH"
        result = subprocess.run(
            [git, "init", "--bare", str(store)],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            env=init_env, timeout=_GIT_TIMEOUT,
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
            check=False,
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


def _volume_evidence(workdir: Path) -> dict:
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
    meta: dict = {"workdir": str(_normalize_path(working_dir)),
                  "created_at": now, "last_touch": now}
    evidence = _volume_evidence(_normalize_path(working_dir))
    if evidence:
        meta.update(evidence)
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8-sig"))
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
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
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


def _list_projects(store: Path) -> list[dict]:
    """Return all registered projects under the store."""
    projects_dir = store / _PROJECTS_DIRNAME
    if not projects_dir.exists():
        return []
    out: list[dict] = []
    for meta_path in projects_dir.glob("*.json"):
        dir_hash = meta_path.stem
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict):
            continue
        meta["_hash"] = dir_hash
        out.append(meta)
    return out


def _pre_v2_shadow_repos(base: Path) -> list[dict]:
    """Return pre-v2 per-project shadow repos still directly under ``base``.

    Pre-v2 layout kept one shadow git repo per working directory directly
    under ``CHECKPOINT_BASE`` (identified by a ``HEAD`` file).  This is the
    single source of truth for that scan so a preview built from it (e.g.
    ``store_status``) always matches what ``prune_checkpoints`` deletes.
    """
    out: list[dict] = []
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
                workdir = wd_marker.read_text(encoding="utf-8-sig").strip()
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


# --- facade delegates -------------------------------------------------------
# Tests monkeypatch these attributes on ``tools.checkpoint_manager``.  The
# delegates resolve them at call time (function-local import, so the module
# cycle facade -> store -> facade only closes when a call happens), which is
# what keeps ``monkeypatch.setattr(checkpoint_manager, ...)`` visible here.

def _resolve_checkpoint_base() -> Path:
    from tools import checkpoint_manager

    return checkpoint_manager._resolve_checkpoint_base()


def _run_git(*args, **kwargs):
    from tools import checkpoint_manager

    return checkpoint_manager._run_git(*args, **kwargs)

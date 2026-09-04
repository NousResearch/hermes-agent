"""Checkpoint Manager — transparent filesystem snapshots via one shared shadow git store.

Snapshots a working directory before file-mutating tool calls (once per directory per turn)
and restores any previous checkpoint.  Not a model tool; controlled by the ``checkpoints``
config / ``--checkpoints`` flag.  One store under ``~/.hermes/checkpoints/`` so git dedupes
blobs across projects (pre-v2 one-repo-per-workdir re-stored ~40 MB each): ``store/`` bare
repo with per-project ``refs/hermes/<hash16>``, ``indexes/<hash16>``, ``projects/<hash16>.json``
(workdir, timestamps, parent identity), ``ledgers/<hash16>.json`` (agent-write ledger), shared
``info/exclude``; ``.last_prune`` marker; ``legacy-<ts>/`` archived pre-v2 repos.  Git runs
with GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE so nothing leaks into the user's project.
"""

import hashlib
import itertools
import json
import logging
import os
import re
import shutil
import stat as stat_mod
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, Tuple

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from utils import env_int

logger = logging.getLogger(__name__)

CHECKPOINT_BASE = get_hermes_home() / "checkpoints"

_STORE_DIRNAME, _INDEXES_DIRNAME, _PROJECTS_DIRNAME, _LEDGERS_DIRNAME = "store", "indexes", "projects", "ledgers"
_REFS_PREFIX, _LEGACY_PREFIX, _PRUNE_MARKER_NAME = "refs/hermes", "legacy-", ".last_prune"
_LEDGER_MAX_ENTRIES = 2000  # newest agent-write entries retained per project

DEFAULT_EXCLUDES = [
    "node_modules/", "dist/", "build/", "target/", "out/", ".next/", ".nuxt/",  # dependency / build output
    "__pycache__/", "*.pyc", "*.pyo", ".cache/", ".pytest_cache/", ".mypy_cache/",  # caches
    ".ruff_cache/", "coverage/", ".coverage",
    ".venv/", "venv/", "env/",  # virtualenvs
    ".git/", ".hg/", ".svn/", ".worktrees/",  # VCS + worktrees (Hermes convention — don't snapshot siblings)
    "*.so", "*.dylib", "*.dll", "*.o", "*.a", "*.jar", "*.class", "*.exe", "*.obj",  # compiled binaries
    "*.mp4", "*.mov", "*.mkv", "*.webm", "*.zip", "*.tar", "*.tar.gz", "*.tgz",  # media / large binaries
    "*.7z", "*.rar", "*.iso",
    ".env", ".env.*", ".env.local", ".env.*.local",  # secrets
    ".DS_Store", "Thumbs.db", "*.log",  # OS junk / logs
]

_GIT_TIMEOUT: int = max(10, min(60, env_int("HERMES_CHECKPOINT_TIMEOUT", 30)))
_MAX_FILES = 50_000  # skip huge directories to avoid slowdowns
_COMMIT_HASH_RE = re.compile(r'^[0-9a-fA-F]{4,64}$')  # short or full SHA-1/SHA-256
_MB = 1024 * 1024
# Inherited GIT_* vars that would redirect the shadow store's git calls.
_GIT_LEAK_VARS = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_NAMESPACE", "GIT_ALTERNATE_OBJECT_DIRECTORIES")
# Per-store config: isolated by env vars already, but belt-and-suspenders.
_STORE_GIT_CONFIG = (("user.email", "hermes@local"), ("user.name", "Hermes Checkpoint"),
                     ("commit.gpgsign", "false"), ("tag.gpgSign", "false"), ("gc.auto", "0"))
_PROJECT_MARKERS = {".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod", "Makefile", "pom.xml", ".hg", "Gemfile"}

_SHORTSTAT_FIELDS = (("files_changed", r'(\d+) file'), ("insertions", r'(\d+) insertion'),
                     ("deletions", r'(\d+) deletion'))


def _no_store_result() -> Dict:
    return {"success": False, "error": "No checkpoints exist for this directory"}


def _empty_prune_result() -> Dict[str, int]:
    return dict.fromkeys(("scanned", "deleted_orphan", "deleted_stale", "errors", "bytes_freed"), 0)


def _validate_commit_hash(commit_hash: str) -> Optional[str]:
    """Error string if unsafe as a git revision (a leading '-' would parse as a flag), else None."""
    if not commit_hash or not commit_hash.strip():
        return "Empty commit hash"
    if commit_hash.startswith("-"):
        return f"Invalid commit hash (must not start with '-'): {commit_hash!r}"
    return None if _COMMIT_HASH_RE.match(commit_hash) else \
        f"Invalid commit hash (expected 4-64 hex characters): {commit_hash!r}"


def _validate_file_path(file_path: str, working_dir: str) -> Optional[str]:
    """Error string if ``file_path`` is absolute or escapes ``working_dir``, else None."""
    if not file_path or not file_path.strip():
        return "Empty file path"
    if os.path.isabs(file_path):
        return f"File path must be relative, got absolute path: {file_path!r}"
    abs_workdir = _normalize_path(working_dir)
    if not (abs_workdir / file_path).resolve().is_relative_to(abs_workdir):
        return f"File path escapes the working directory via traversal: {file_path!r}"
    return None


def _normalize_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _project_hash(working_dir: str) -> str:
    """Deterministic per-project hash: sha256(abs_path)[:16]."""
    return hashlib.sha256(str(_normalize_path(working_dir)).encode()).hexdigest()[:16]


def _store_path(base: Optional[Path] = None) -> Path:
    return (base or CHECKPOINT_BASE) / _STORE_DIRNAME


def _store_has_head(store: Path) -> bool:
    return (store / "HEAD").exists()


def _index_path(store: Path, dir_hash: str) -> Path:
    return store / _INDEXES_DIRNAME / dir_hash


def _ledger_path(store: Path, dir_hash: str) -> Path:
    return store / _LEDGERS_DIRNAME / f"{dir_hash}.json"


def _ref_name(dir_hash: str) -> str:
    return f"{_REFS_PREFIX}/{dir_hash}"


def _project_meta_path(store: Path, dir_hash: str) -> Path:
    return store / _PROJECTS_DIRNAME / f"{dir_hash}.json"


def _read_json_dict(path: Path) -> Optional[Dict]:
    """Parse ``path`` as a JSON object; None when missing, unreadable or not a dict."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _mtime_or_none(path: Path) -> Optional[float]:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _hash_file(path: Path) -> Optional[str]:
    """Streaming sha256 of a file's bytes. None if unreadable/missing."""
    try:
        with open(path, "rb") as fh:
            return hashlib.file_digest(fh, "sha256").hexdigest()
    except OSError:
        return None


def _load_ledger(store: Path, dir_hash: str) -> Dict[str, Dict]:
    """Agent-write ledger ``{abs_path: {"sha256", "ts"}}``: hash of every file the last
    ``write_file``/``patch`` produced, so restores can tell Hermes' writes from later user edits."""
    return _read_json_dict(_ledger_path(store, dir_hash)) or {}


def _save_ledger(store: Path, dir_hash: str, ledger: Dict[str, Dict]) -> None:
    """Persist the agent-write ledger, capped to the newest entries."""
    try:
        if len(ledger) > _LEDGER_MAX_ENTRIES:
            ts = lambda kv: kv[1].get("ts", 0) if isinstance(kv[1], dict) else 0  # noqa: E731
            ledger = dict(sorted(ledger.items(), key=ts, reverse=True)[:_LEDGER_MAX_ENTRIES])
        path = _ledger_path(store, dir_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json.tmp").write_text(json.dumps(ledger), encoding="utf-8")
        path.with_suffix(".json.tmp").replace(path)
    except OSError:
        logger.debug("Failed to save agent-write ledger for %s", dir_hash, exc_info=True)


def _isolated_git_env() -> dict:
    """Subprocess env with the user's global/system git config neutralised and inherited GIT_*
    redirects dropped: user settings (``commit.gpgsign``, hooks, credential helpers) would break
    background snapshots or spawn pinentry prompts.  GLOBAL/SYSTEM need git 2.32+; NOSYSTEM covers
    older git.  HOME is kept — rewriting it would change which ~/.gitconfig is hidden."""
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=False)
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    for key in _GIT_LEAK_VARS:
        env.pop(key, None)
    return env


def _git_env(store: Path, working_dir: str, index_file: Optional[Path] = None) -> dict:
    """Env that redirects git to the shared store (+ a per-project index if given)."""
    env = _isolated_git_env()
    env.update(GIT_DIR=str(store), GIT_WORK_TREE=str(_normalize_path(working_dir)))
    if index_file is not None:
        env["GIT_INDEX_FILE"] = str(index_file)
    return env


def _git_subprocess(cmd: List[str], env: dict, timeout: int, cwd: Optional[str] = None):
    # creationflags suppresses the per-call conhost flash on Windows (no-op on POSIX).
    return subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout,
                          env=env, cwd=cwd, stdin=subprocess.DEVNULL, creationflags=windows_hide_flags())


def _repair_bare_repo_dirs(store: Path) -> None:
    """Recreate ``refs/heads`` and ``branches`` after ``git gc``: gc on a bare repo with only
    packed refs can remove them, yet git 2.34+ requires them — without them ``git add -A``
    fails with "not a git repository" and every checkpoint operation silently fails."""
    for subdir in ("refs/heads", "branches"):
        if (store / subdir).exists():
            continue
        try:
            (store / subdir).mkdir(parents=True, exist_ok=True)
            logger.debug("Repaired missing %s in checkpoint store", subdir)
        except OSError as exc:
            logger.warning("Cannot create %s in checkpoint store: %s", subdir, exc)


def _run_git(args: List[str], store: Path, working_dir: str, timeout: int = _GIT_TIMEOUT,
             allowed_returncodes: Optional[Set[int]] = None, index_file: Optional[Path] = None) -> Tuple[bool, str, str]:
    """Run git against the shared store -> (ok, stdout, stderr).  ``allowed_returncodes`` suppresses
    error logging for expected non-zero exits (``diff --cached --quiet`` -> 1); ``ok`` stays rc == 0."""
    wd = _normalize_path(working_dir)
    cmd = ["git"] + list(args)
    if not wd.is_dir():
        msg = (f"working directory not found: {wd}" if not wd.exists()
               else f"working directory is not a directory: {wd}")
        logger.error("Git command skipped: %s (%s)", " ".join(cmd), msg)
        return False, "", msg

    try:
        result = _git_subprocess(cmd, _git_env(store, str(wd), index_file=index_file), timeout, cwd=str(wd))
    except subprocess.TimeoutExpired:
        msg = f"git timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg, exc_info=True)
        return False, "", msg
    except FileNotFoundError as exc:
        if getattr(exc, "filename", None) == "git":
            logger.error("Git executable not found: %s", " ".join(cmd), exc_info=True)
            return False, "", "git not found"
        msg = f"working directory not found: {wd}"
        logger.error("Git command failed before execution: %s (%s)", " ".join(cmd), msg, exc_info=True)
        return False, "", msg
    except Exception as exc:
        logger.error("Unexpected git error running %s: %s", " ".join(cmd), exc, exc_info=True)
        return False, "", str(exc)

    ok = result.returncode == 0
    stdout, stderr = result.stdout.strip(), result.stderr.strip()
    if not ok and result.returncode not in (allowed_returncodes or set()):
        logger.error("Git command failed: %s (rc=%d) stderr=%s",
                     " ".join(cmd), result.returncode, stderr)
    return ok, stdout, stderr


def _git_out(args: List[str], store: Path, working_dir: str, rc: Optional[Set[int]] = None) -> str:
    """stdout of a successful git call, else ``""``."""
    ok, out, _ = _run_git(args, store, working_dir, allowed_returncodes=rc)
    return out if ok else ""


def _ref_tip(store: Path, working_dir: str, ref: str) -> Optional[str]:
    """Commit sha at ``ref``, or None when the ref does not exist yet."""
    return _git_out(["rev-parse", "--verify", ref + "^{commit}"], store, working_dir, {128}) or None


def _ref_commit_count(store: Path, working_dir: str, ref: str) -> int:
    out = _git_out(["rev-list", "--count", ref], store, working_dir, {128})
    return int(out) if out.isdigit() else 0


def _ref_commits_oldest_first(store: Path, working_dir: str, ref: str) -> List[str]:
    return _git_out(["rev-list", "--reverse", ref], store, working_dir).splitlines()


def _list_project_refs(store: Path, working_dir: str) -> List[str]:
    out = _git_out(["for-each-ref", "--format=%(refname)", _REFS_PREFIX], store, working_dir, {128})
    return [r for r in out.splitlines() if r.strip()]


def _delete_ref(store: Path, ref: str) -> bool:
    ok, _, _ = _run_git(["update-ref", "-d", ref], store, str(store.parent), allowed_returncodes={128})
    return ok


def _commit_tree_args(tree_sha: str, message: str, parent: Optional[str]) -> List[str]:
    return ["commit-tree", tree_sha, *(["-p", parent] if parent is not None else []), "-m", message, "--no-gpg-sign"]


def _rebuild_linear_chain(store: Path, working_dir: str, shas: List[str]) -> Optional[str]:
    """Re-commit each sha's tree (same message) as a fresh linear chain; new tip, or None on
    any failure (caller leaves the ref untouched)."""
    new_parent: Optional[str] = None
    for sha in shas:
        tree_sha = _git_out(["rev-parse", f"{sha}^{{tree}}"], store, working_dir)
        if not tree_sha:
            return None
        msg = _git_out(["log", "--format=%s", "-1", sha], store, working_dir) or "checkpoint"
        new_parent = _git_out(_commit_tree_args(tree_sha, msg, new_parent), store, working_dir)
        if not new_parent:
            return None
    return new_parent


def _rewrite_ref_to(store: Path, working_dir: str, ref: str, commits: List[str]) -> bool:
    """Point ``ref`` at a freshly rebuilt linear chain of ``commits``; False if nothing was rewritten."""
    if not commits:
        return False
    tip = _rebuild_linear_chain(store, working_dir, commits)
    if tip is None:
        return False
    _run_git(["update-ref", ref, tip], store, working_dir)
    return True


def _gc_store(store: Path, working_dir: str) -> None:
    """Reclaim objects unreachable from the (rewritten/deleted) refs."""
    _run_git(["reflog", "expire", "--expire=now", "--all"], store, working_dir)
    _run_git(["gc", "--prune=now", "--quiet"], store, working_dir, timeout=_GIT_TIMEOUT * 3)
    _repair_bare_repo_dirs(store)


def _drop_oldest_commit(store: Path, working_dir: str, ref: str) -> bool:
    """Rewrite ``ref`` without its oldest commit; never below one snapshot."""
    if _ref_commit_count(store, working_dir, ref) <= 1:
        return False
    return _rewrite_ref_to(store, working_dir, ref, _ref_commits_oldest_first(store, working_dir, ref)[1:])


def _shrink_store_to_cap(store: Path, working_dir: str, cap_bytes: int) -> bool:
    """Round-robin-drop the oldest commit per project ref until the store fits (bounded to 20
    rounds against pathological loops).  False when there are no project refs."""
    for _ in range(20):
        if _dir_size_bytes(store) <= cap_bytes:
            break
        refs = _list_project_refs(store, working_dir)
        if not refs:
            return False
        if not any([_drop_oldest_commit(store, working_dir, ref) for ref in refs]):
            break
    return True


def _migrate_legacy_store(base: Path) -> Optional[Path]:
    """Archive pre-v2 per-project shadow repos into ``legacy-<ts>/`` (moved, not deleted —
    users may want to recover; the archive falls under retention and ``hermes checkpoints
    clear-legacy``).  Returns the archive path or None."""
    if not base.exists():
        return None
    stray = [c for c in base.iterdir() if c.name not in (_STORE_DIRNAME, _PRUNE_MARKER_NAME)
             and not c.name.startswith(_LEGACY_PREFIX)]
    if not stray:
        return None
    legacy_root = base / f"{_LEGACY_PREFIX}{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        legacy_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create legacy archive dir: %s", exc)
        return None
    for child in stray:
        try:
            shutil.move(str(child), str(legacy_root / child.name))
        except OSError as exc:
            logger.warning("Could not archive legacy checkpoint %s: %s", child, exc)
    logger.info("Migrated pre-v2 checkpoint repos to %s. "
                "Clear with `hermes checkpoints clear-legacy` when safe.", legacy_root)
    return legacy_root


def _init_store(store: Path, working_dir: str) -> Optional[str]:
    """Initialise the shared store if needed (migrating pre-v2 repos first).  Returns error or None."""
    base = store.parent
    if not store.exists():
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"Could not create checkpoint base: {exc}"
        _migrate_legacy_store(base)
    if _store_has_head(store):
        return None
    for d in (store, store / _INDEXES_DIRNAME, store / _PROJECTS_DIRNAME):
        d.mkdir(parents=True, exist_ok=True)

    # ``git init --bare`` rejects GIT_WORK_TREE, so bypass _run_git.
    try:
        result = _git_subprocess(["git", "init", "--bare", str(store)], _isolated_git_env(), _GIT_TIMEOUT)
        if result.returncode != 0:
            return f"Shadow store init failed: {result.stderr.strip()}"
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"Shadow store init failed: {exc}"
    for key, value in _STORE_GIT_CONFIG:
        _run_git(["config", key, value], store, str(base))
    (store / "info").mkdir(exist_ok=True)
    (store / "info" / "exclude").write_text("\n".join(DEFAULT_EXCLUDES) + "\n", encoding="utf-8")
    logger.debug("Initialised checkpoint store at %s", store)
    return None


def _volume_evidence(workdir: Path) -> Dict:
    """``(st_dev, st_ino)`` of ``workdir``'s parent, captured while reachable: identifies the
    *directory*, not the path (a mount point resolves to the underlay after unmount), so orphan
    pruning can tell "deleted" from "volume detached".  ``{}`` when unreachable, on error, or with
    no usable identity (zero dev/ino: Windows without file IDs, some shares) — pruning stays conservative."""
    try:
        st = workdir.parent.stat() if workdir.exists() else None
    except OSError:
        return {}
    if st is None or not st.st_dev or not st.st_ino:
        return {}
    return {"workdir_parent_dev": st.st_dev, "workdir_parent_ino": st.st_ino}


def _register_project(store: Path, working_dir: str) -> None:
    """Upsert ``projects/<hash>.json`` (workdir, last_touch, created_at; ``created_at`` survives).
    Parent identity is refreshed while the project is observably live (a remount can change it);
    on a failed probe the recorded identity is kept — stale evidence only makes pruning MORE
    conservative.  Never raises."""
    meta_path = _project_meta_path(store, _project_hash(working_dir))
    meta, now, workdir = _read_json_dict(meta_path) or {}, time.time(), _normalize_path(working_dir)
    meta.update({"workdir": str(workdir), "last_touch": now, **_volume_evidence(workdir)})
    meta.setdefault("created_at", now)
    try:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write project metadata %s: %s", meta_path, exc)


_touch_project = _register_project  # per-turn touch == re-register (same upsert)


def _list_projects(store: Path) -> List[Dict]:
    """All registered projects under the store (each tagged with ``_hash``)."""
    projects_dir = store / _PROJECTS_DIRNAME
    if not projects_dir.exists():
        return []
    metas = ((meta_path.stem, _read_json_dict(meta_path)) for meta_path in projects_dir.glob("*.json"))
    return [{**meta, "_hash": stem} for stem, meta in metas if meta is not None]


def _pre_v2_shadow_repos(base: Path) -> List[Dict]:
    """Pre-v2 per-project shadow repos (``base/<hash>/HEAD``) still under ``base``; the single
    scan so a ``store_status`` preview always matches what ``prune_checkpoints`` deletes."""
    out: List[Dict] = []
    for child in base.iterdir() if base.exists() else ():
        if (not child.is_dir() or child.name == _STORE_DIRNAME
                or child.name.startswith(_LEGACY_PREFIX) or not (child / "HEAD").exists()):
            continue
        workdir: Optional[str] = None
        marker_unreadable = False
        try:
            if (child / "HERMES_WORKDIR").exists():
                workdir = (child / "HERMES_WORKDIR").read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            marker_unreadable = True  # present but unreadable: no evidence the project is gone
        out.append({"path": child, "workdir": workdir, "marker_unreadable": marker_unreadable,
                    "exists": bool(workdir) and Path(workdir).exists()})
    return out


def _legacy_archives(base: Path) -> List[Path]:
    return [c for c in list(base.iterdir()) if c.is_dir() and c.name.startswith(_LEGACY_PREFIX)]


def _dir_file_count(path: str) -> int:
    """Quick file count estimate (stops early once over _MAX_FILES)."""
    try:
        return sum(1 for _ in itertools.islice(Path(path).rglob("*"), _MAX_FILES + 1))
    except OSError:
        return 0


def _rglob_stats(path: Path) -> Iterator[os.stat_result]:
    """stat() of everything under ``path``; unstattable entries and walk errors are skipped."""
    try:
        for p in path.rglob("*"):
            try:
                yield p.stat()
            except OSError:
                continue
    except OSError:
        return


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive size in bytes (regular files only).  0 on error."""
    return sum(st.st_size for st in _rglob_stats(path) if stat_mod.S_ISREG(st.st_mode))


def _newest_mtime(path: Path) -> float:
    """Newest mtime under ``path`` (0.0 when nothing is statable)."""
    return max((st.st_mtime for st in _rglob_stats(path)), default=0.0)


class _ProjectRefs(NamedTuple):
    """Store coordinates for one working directory (resolved at call time)."""
    abs_dir: str
    store: Path
    dir_hash: str
    index_file: Path
    ref: str


def _project_refs(working_dir: str) -> _ProjectRefs:
    abs_dir = str(_normalize_path(working_dir))
    store, dir_hash = _store_path(CHECKPOINT_BASE), _project_hash(abs_dir)
    return _ProjectRefs(abs_dir, store, dir_hash, _index_path(store, dir_hash), _ref_name(dir_hash))


def _locate(working_dir: str, commit_hash: str,
            file_path: Optional[str] = None) -> Tuple[Optional[_ProjectRefs], Optional[Dict]]:
    """Validate inputs and resolve store coordinates; ``(refs, error_result_or_None)``."""
    p = _project_refs(working_dir)
    err = _validate_commit_hash(commit_hash) or (file_path and _validate_file_path(file_path, p.abs_dir))
    if err:
        return p, {"success": False, "error": err}
    return p, None if _store_has_head(p.store) else _no_store_result()


def _stage_all(p: _ProjectRefs) -> Tuple[bool, str, str]:
    """``git add -A`` into the per-project index."""
    return _run_git(["add", "-A"], p.store, p.abs_dir, timeout=_GIT_TIMEOUT * 2, index_file=p.index_file)


def _diff_staged_tree(p: _ProjectRefs, *diff_args: List[str]) -> List[Tuple[bool, str, str]]:
    """Stage the working tree (so new files show), run each ``git diff`` variant,
    then point the index back at the ref so it doesn't drift."""
    _stage_all(p)
    results = [_run_git(args, p.store, p.abs_dir, index_file=p.index_file) for args in diff_args]
    _run_git(["read-tree", p.ref], p.store, p.abs_dir, index_file=p.index_file, allowed_returncodes={128})
    return results


def _commit_exists(p: _ProjectRefs, commit_hash: str) -> Tuple[bool, str]:
    ok, _, err = _run_git(["cat-file", "-t", commit_hash], p.store, p.abs_dir)
    return ok, err


def _restore_ok(commit_hash: str, reason: str, abs_dir: str, **extra) -> Dict:
    return {"success": True, "restored_to": commit_hash[:8], "reason": reason, "directory": abs_dir, **extra}


@dataclass
class _SafeRestoreTargets:
    checkout: List[str] = field(default_factory=list)
    kept_oversize: List[str] = field(default_factory=list)
    failed_deletes: List[str] = field(default_factory=list)


class CheckpointManager:
    """Automatic filesystem checkpoints.  Owned by AIAgent: ``new_turn()`` at the start of
    each turn, ``ensure_checkpoint(dir, reason)`` before any file-mutating tool call (at most
    one snapshot per directory per turn).  ``max_snapshots`` caps checkpoints per directory;
    ``max_total_size_mb`` is a hard store-size ceiling (oldest per project dropped after a
    commit); ``max_file_size_mb`` keeps any larger single file out of checkpoints."""

    def __init__(self, enabled: bool = False, max_snapshots: int = 20,
                 max_total_size_mb: int = 500, max_file_size_mb: int = 10):
        self.enabled = enabled
        self.max_snapshots, self.max_total_size_mb, self.max_file_size_mb = (
            max(1, int(max_snapshots)), max(0, int(max_total_size_mb)), max(0, int(max_file_size_mb)))
        self._checkpointed_dirs: Set[str] = set()
        self._git_available: Optional[bool] = None  # lazy probe

    def new_turn(self) -> None:
        """Reset per-turn dedup.  Call at the start of each agent iteration."""
        self._checkpointed_dirs.clear()

    # --- public API ---

    def record_agent_write(self, file_path: str) -> None:
        """Record the content hash of a file Hermes just wrote (agent-write ledger), so safe-mode
        :meth:`restore` can skip files the user hand-edited afterwards.  Never raises."""
        if not self.enabled:
            return
        try:
            path = _normalize_path(file_path)
            digest = _hash_file(path)
            if digest is None:
                return
            store, dir_hash = _store_path(CHECKPOINT_BASE), _project_hash(self.get_working_dir_for_path(str(path)))
            _save_ledger(store, dir_hash, {**_load_ledger(store, dir_hash), str(path): {"sha256": digest, "ts": time.time()}})
        except Exception as exc:
            logger.debug("record_agent_write failed for %s: %s", file_path, exc)

    def safe_restore_plan(self, working_dir: str, commit_hash: str) -> Dict:
        """Classify files changed since ``commit_hash``: ``restore`` = still matching what Hermes
        last wrote (or deleted since); ``skipped`` = user-edited afterwards or never written by
        Hermes.  ``ledger_empty`` => no ledger, callers fall back to a full restore."""
        p, err = _locate(working_dir, commit_hash)
        if err:
            return err

        (ok, names_out, err), = _diff_staged_tree(p, ["diff", "--name-only", commit_hash, "--cached"])
        if not ok:
            return {"success": False, "error": f"Could not compute changed files: {err}"}

        ledger = _load_ledger(p.store, p.dir_hash)
        if not ledger:
            return {"success": True, "restore": [], "skipped": [], "ledger_empty": True}
        out: Dict[str, List[str]] = {"restore": [], "skipped": []}
        for rel in filter(None, (line.strip() for line in names_out.splitlines())):
            abs_path = Path(p.abs_dir) / rel
            entry = ledger.get(str(abs_path))
            recorded = entry.get("sha256") if isinstance(entry, dict) else None
            hermes_authored = recorded is not None and _hash_file(abs_path) in (None, recorded)
            out["restore" if hermes_authored else "skipped"].append(rel)
        return {"success": True, **out}

    def ensure_checkpoint(self, working_dir: str, reason: str = "auto") -> bool:
        """Take a checkpoint if enabled and not already done this turn.  Never raises."""
        if not self.enabled:
            return False
        if self._git_available is None:
            self._git_available = shutil.which("git") is not None
            if not self._git_available:
                logger.debug("Checkpoints disabled: git not found")
        if not self._git_available:
            return False
        abs_dir = str(_normalize_path(working_dir))
        if abs_dir in {"/", str(Path.home())}:  # never snapshot root/home
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
        p = _project_refs(working_dir)
        if not _store_has_head(p.store):
            return []

        log = _git_out(["log", p.ref, "--format=%H|%h|%aI|%s", "-n", str(self.max_snapshots)],
                       p.store, p.abs_dir, {128, 129})
        results: List[Dict] = []
        for line in log.splitlines():
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            entry = {"hash": parts[0], "short_hash": parts[1], "timestamp": parts[2], "reason": parts[3],
                     "files_changed": 0, "insertions": 0, "deletions": 0}
            stat_out = _git_out(["diff", "--shortstat", f"{parts[0]}~1", parts[0]], p.store, p.abs_dir, {128, 129})
            for key, pattern in _SHORTSTAT_FIELDS:
                m = re.search(pattern, stat_out)
                if m:
                    entry[key] = int(m.group(1))
            results.append(entry)
        return results

    def list_all_checkpoints(self) -> List[Dict]:
        """Checkpoints across every registered project (most recent first), each tagged ``workdir``.

        Surgical reapply of PR #10633 by @nightq (#10505) onto the v2 single-store layout: iterate
        ``projects/<hash>.json`` metadata via ``_list_projects`` instead of the pre-v2 per-shadow-dir scan.
        Each entry carries the extra ``workdir`` key so callers can label which project a checkpoint belongs
        to.
        """
        store = _store_path(CHECKPOINT_BASE)
        if not _store_has_head(store):
            return []
        results = [{**entry, "workdir": workdir}
                   for workdir in (meta.get("workdir") or "" for meta in _list_projects(store)) if workdir
                   for entry in self.list_checkpoints(workdir)]
        return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)

    def diff(self, working_dir: str, commit_hash: str) -> Dict:
        """Show diff between a checkpoint and the current working tree."""
        p, err = _locate(working_dir, commit_hash)
        if err:
            return err
        ok, _ = _commit_exists(p, commit_hash)
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found"}

        (ok_stat, stat_out, _), (ok_diff, diff_out, _) = _diff_staged_tree(
            p, ["diff", "--stat", commit_hash, "--cached"], ["diff", commit_hash, "--cached", "--no-color"])
        if not ok_stat and not ok_diff:
            return {"success": False, "error": "Could not generate diff"}
        return {"success": True, "stat": stat_out if ok_stat else "", "diff": diff_out if ok_diff else ""}

    def session_diff(self, working_dir: str) -> Dict:
        """Cumulative diff powering ``/diff session``: earliest retained checkpoint vs working tree.
        The ref persists per project, so the baseline may predate the session or postdate it after
        pruning — an approximation.  Same shape as :meth:`diff`; no checkpoints => ``"empty": True``."""
        checkpoints = self.list_checkpoints(working_dir)
        if not checkpoints:
            return {"success": True, "stat": "", "diff": "", "empty": True}

        baseline = checkpoints[-1].get("hash") or ""
        result = self.diff(working_dir, baseline)
        if result.get("success"):
            result.setdefault("baseline", baseline)
            if not (result.get("stat") or result.get("diff")):
                result["empty"] = True
        return result

    def restore(self, working_dir: str, commit_hash: str, file_path: str = None,
                safe: bool = False) -> Dict:
        """Restore files to a checkpoint state.  ``safe=True`` (full-directory only) leaves files
        the user hand-edited after Hermes' last write untouched (agent-write ledger); the result
        then gains ``skipped_user_edits``, ``skipped_oversize`` (size cap kept them out of every
        checkpoint) and, only when a delete failed, ``failed_deletes``."""
        p, err = _locate(working_dir, commit_hash, file_path)
        if err:
            return err
        abs_dir = p.abs_dir
        ok, err = _commit_exists(p, commit_hash)
        if not ok:
            return {"success": False, "error": f"Checkpoint '{commit_hash}' not found", "debug": err or None}

        skipped_user_edits: List[str] = []
        restore_paths: Optional[List[str]] = None
        if safe and not file_path:
            plan = self.safe_restore_plan(abs_dir, commit_hash)
            if not plan.get("success"):
                return {"success": False, "error": plan.get("error", "Safe-restore plan failed")}
            if not plan.get("ledger_empty"):  # no agent-write history => classic full restore
                restore_paths, skipped_user_edits = plan["restore"], plan["skipped"]
                if not restore_paths:
                    return _restore_ok(commit_hash, "nothing to restore (all changed files were user-edited)",
                                       abs_dir, restored_files=[], skipped_user_edits=skipped_user_edits,
                                       skipped_oversize=[])

        # Take a pre-rollback snapshot so you can undo the undo.
        self._take(abs_dir, f"pre-rollback snapshot (restoring to {commit_hash[:8]})")

        targets = _SafeRestoreTargets(checkout=[file_path or "."])
        if restore_paths is not None:
            targets = self._apply_safe_restore_deletes(p, commit_hash, restore_paths)
        if targets.checkout:
            ok, _, err = _run_git(["checkout", commit_hash, "--", *targets.checkout], p.store, abs_dir,
                                  timeout=_GIT_TIMEOUT * 2, index_file=p.index_file)
            if not ok:
                return {"success": False, "error": f"Restore failed: {err}", "debug": err or None}

        reason_out = _git_out(["log", "--format=%s", "-1", commit_hash], p.store, abs_dir) or "unknown"
        result = _restore_ok(commit_hash, reason_out, abs_dir)
        if file_path:
            result["file"] = file_path
        if restore_paths is not None:
            # Report only what was actually acted on: a kept oversize path or a
            # failed unlink left the file in place and must not read as "Restored".
            not_restored = set(targets.kept_oversize) | set(targets.failed_deletes)
            result.update(restored_files=[rel for rel in restore_paths if rel not in not_restored],
                          skipped_user_edits=skipped_user_edits, skipped_oversize=targets.kept_oversize)
            if targets.failed_deletes:
                result["failed_deletes"] = targets.failed_deletes
        return result

    def _apply_safe_restore_deletes(self, p: _ProjectRefs, commit_hash: str,
                                    restore_paths: List[str]) -> _SafeRestoreTargets:
        """Split ledger-approved paths into checkout targets and delete the rest.  A path absent
        from the checkpoint is Hermes-created (delete to restore) — unless ``max_file_size_mb`` kept
        it out of every checkpoint: no prior copy exists and the ledger can't prove it agent-created
        (hashes, not create-vs-modify), so leaving it costs a stale file, deleting costs the file."""
        targets = _SafeRestoreTargets()
        for rel in restore_paths:
            ok_in_commit, _, _ = _run_git(["cat-file", "-e", f"{commit_hash}:{rel}"],
                                          p.store, p.abs_dir, allowed_returncodes={1, 128})
            target = Path(p.abs_dir) / rel
            if ok_in_commit:
                targets.checkout.append(rel)
            elif self._exceeds_size_cap(target):
                targets.kept_oversize.append(rel)
            else:
                try:
                    if target.is_file() or target.is_symlink():
                        target.unlink()
                except OSError as exc:
                    logger.warning("Safe restore: could not remove %s: %s", rel, exc)
                    targets.failed_deletes.append(rel)
        return targets

    def get_working_dir_for_path(self, file_path: str) -> str:
        """Resolve a file path to its working directory (nearest project-marker ancestor)."""
        path = _normalize_path(file_path)
        candidate = path if path.is_dir() else path.parent
        check = candidate
        while check != check.parent:
            if any((check / m).exists() for m in _PROJECT_MARKERS):
                return str(check)
            check = check.parent
        return str(candidate)

    # --- internal ---

    def _take(self, working_dir: str, reason: str) -> bool:
        """Take a snapshot.  Returns True on success."""
        p = _project_refs(working_dir)
        err = _init_store(p.store, working_dir)
        if err:
            return _step_failed("store init", err)
        _touch_project(p.store, working_dir)
        if _dir_file_count(working_dir) > _MAX_FILES:
            logger.debug("Checkpoint skipped: >%d files in %s", _MAX_FILES, working_dir)
            return False
        ref_commit = _ref_tip(p.store, working_dir, p.ref)
        _seed_project_index(p, ref_commit)

        # Broad patterns come from the exclude file; oversize paths are dropped post-stage.
        ok, _, err = _stage_all(p)
        if not ok:
            return _step_failed("git-add", err)
        if self.max_file_size_mb > 0:
            self._drop_oversize_from_index(p.store, working_dir, p.index_file)

        skip = _index_unchanged_reason(p, ref_commit)
        if skip:
            logger.debug("Checkpoint skipped: %s in %s", skip, working_dir)
            return False

        ok, tree_sha, err = _run_git(["write-tree"], p.store, working_dir, index_file=p.index_file)
        if not ok or not tree_sha:
            return _step_failed("write-tree", err)
        ok, new_sha, err = _run_git(_commit_tree_args(tree_sha, reason, ref_commit),
                                    p.store, working_dir, index_file=p.index_file)
        if not ok or not new_sha:
            return _step_failed("commit-tree", err)
        update_args = ["update-ref", p.ref, new_sha] + ([ref_commit] if ref_commit else [])
        ok, _, err = _run_git(update_args, p.store, working_dir)
        if not ok:
            return _step_failed("update-ref", err)

        logger.debug("Checkpoint taken in %s: %s (%s)", working_dir, reason, new_sha[:8])
        self._prune(p.store, working_dir, p.ref)
        self._enforce_size_cap(p.store)
        return True

    def _exceeds_size_cap(self, path: Path) -> bool:
        """Whether *path* is larger than ``max_file_size_mb`` (0 disables; unstattable => False).
        The ONE predicate for both "excluded from the checkpoint" and "refused deletion at
        restore" — a drifted threshold would delete a file with no copy."""
        cap = self.max_file_size_mb * _MB
        if cap <= 0:
            return False
        try:
            return path.stat().st_size > cap
        except OSError:
            return False

    def _drop_oversize_from_index(self, store: Path, working_dir: str, index_file: Path) -> None:
        """Unstage files larger than ``max_file_size_mb`` (datasets, weights, videos)."""
        if self.max_file_size_mb <= 0:
            return
        ok, stdout, _ = _run_git(["ls-files", "--cached", "-z"], store, working_dir, index_file=index_file)
        abs_workdir = _normalize_path(working_dir)
        # NUL-separated; _run_git's strip() leaves NULs alone.
        oversize = [rel for rel in (stdout if ok else "").split("\x00") if rel and self._exceeds_size_cap(abs_workdir / rel)]
        if not oversize:
            return
        logger.debug("Checkpoint: dropping %d oversize file(s) (>%d MB) from index",
                     len(oversize), self.max_file_size_mb)
        for i in range(0, len(oversize), 200):  # chunk: never overflow argv
            _run_git(["rm", "--cached", "--quiet", "--"] + oversize[i:i + 200],
                     store, working_dir, index_file=index_file, allowed_returncodes={128})

    def _prune(self, store: Path, working_dir: str, ref: str) -> None:
        """Rewrite the ref to its last ``max_snapshots`` commits and gc (only limiting the
        log view, as v1 did, let loose objects accumulate forever)."""
        if _ref_commit_count(store, working_dir, ref) <= self.max_snapshots:
            return
        commits = _ref_commits_oldest_first(store, working_dir, ref)
        if _rewrite_ref_to(store, working_dir, ref, commits[-self.max_snapshots:]):
            _gc_store(store, working_dir)

    def _enforce_size_cap(self, store: Path) -> None:
        """Drop oldest checkpoints across ALL projects until under ``max_total_size_mb``."""
        cap_bytes = self.max_total_size_mb * _MB
        size = _dir_size_bytes(store) if cap_bytes > 0 else 0
        if size <= cap_bytes:
            return
        logger.info("Checkpoint store exceeded %d MB (actual %d MB) — pruning oldest",
                    self.max_total_size_mb, size // _MB)
        if _shrink_store_to_cap(store, str(store.parent), cap_bytes):
            _gc_store(store, str(store.parent))


def _step_failed(step: str, err: str) -> bool:
    logger.debug("Checkpoint %s failed: %s", step, err)
    return False


def _seed_project_index(p: _ProjectRefs, ref_commit: Optional[str]) -> None:
    """Reset the per-project index to the ref tip so ``add -A`` sees only changes since.
    First snapshot: just create the indexes dir.  Existing index with no ref: discard it."""
    if not p.index_file.exists():
        p.index_file.parent.mkdir(parents=True, exist_ok=True)
    elif ref_commit:
        _run_git(["read-tree", ref_commit], p.store, p.abs_dir,
                 index_file=p.index_file, allowed_returncodes={128})
    else:
        _unlink_quiet(p.index_file)


def _index_unchanged_reason(p: _ProjectRefs, ref_commit: Optional[str]) -> Optional[str]:
    """Why a snapshot would be redundant ("no changes" / "empty tree"), else None.  Compares against
    the ref tip, not HEAD — HEAD on the bare store points at a nonexistent branch, so every staged
    path would look like a new file."""
    if ref_commit:
        ok_diff, _, _ = _run_git(["diff-index", "--cached", "--quiet", ref_commit], p.store, p.abs_dir,
                                 allowed_returncodes={1}, index_file=p.index_file)
        return "no changes" if ok_diff else None
    ok_ls, ls_out, _ = _run_git(["ls-files", "--cached"], p.store, p.abs_dir, index_file=p.index_file)
    return "empty tree" if ok_ls and not ls_out.strip() else None


def format_checkpoint_list(checkpoints: List[Dict], directory: str) -> str:
    """Format checkpoint list for display to user."""
    if not checkpoints:
        return f"No checkpoints found for {directory}"

    lines = [f"📸 Checkpoints for {directory}:\n"]
    for i, cp in enumerate(checkpoints, 1):
        ts = cp["timestamp"]
        if "T" in ts:
            ts = f"{ts.split('T')[0]} {ts.split('T')[1].split('+')[0].split('-')[0][:5]}"

        files, ins, dele = (cp.get(k, 0) for k in ("files_changed", "insertions", "deletions"))
        stat = f"  ({files} file{'s' if files != 1 else ''}, +{ins}/-{dele})" if files else ""
        workdir = cp.get("workdir", "")  # only present on list_all_checkpoints results
        tag = f"[{Path(workdir).name or workdir}]  " if workdir and directory == "all directories" else ""
        lines.append(f"  {i}. {cp['short_hash']}  {ts}  {tag}{cp['reason']}{stat}")

    lines += ["\n  /rollback <N>             restore to checkpoint N",
              "  /rollback diff <N>        preview changes since checkpoint N",
              "  /rollback <N> <file>      restore a single file from checkpoint N"]
    return "\n".join(lines)


def _workdir_is_observably_gone(workdir: str, parent_dev: Optional[int] = None, parent_ino: Optional[int] = None,
                                require_parent_identity: bool = True) -> bool:
    """True only when we can positively observe that ``workdir`` was removed.

    ``Path.exists()`` is False for a deleted dir AND for detached storage (unplugged drive,
    downed VPN share, absent bind-mount); orphan pruning deletes the whole history, so
    ambiguity never counts as "deleted".  Corroborations: (1) the parent is present;
    (2) its ``(st_dev, st_ino)`` matches the identity recorded while live — an unmount
    exposes the underlay dir, whose entries prove nothing; none recorded => conservative
    unless ``require_parent_identity=False`` (pre-v2 layout, structural checks only);
    (3) the parent is non-empty or a live mount point — unmounting leaves static mount
    points behind as *empty* dirs.  Abandoned projects still fall to ``last_touch`` retention.
    """
    if not workdir:
        return False
    path, parent = Path(workdir), Path(workdir).parent
    try:
        if path.exists() or parent == path or not parent.is_dir():
            return False
        if parent_dev is not None and parent_ino is not None:
            st = parent.stat()
            if (st.st_dev, st.st_ino) != (parent_dev, parent_ino):
                return False
        elif require_parent_identity:
            return False
        with os.scandir(parent) as entries:
            return next(entries, None) is not None or os.path.ismount(parent)
    except OSError:
        return False  # probe failed (permission, I/O error) — not evidence of deletion


def _int_or_none(value) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _sweep(entries, result: Dict[str, int], delete) -> None:
    """Shared orphan/stale sweep.  ``entries`` yields ``(item, gone, allowed, is_stale)`` where
    ``is_stale`` is a thunk (may do I/O, so only evaluated for non-orphans); "orphan" wins."""
    for item, gone, allowed, is_stale in entries:
        result["scanned"] += 1
        reason = "orphan" if gone and allowed else "stale" if is_stale() else None
        if reason is not None:
            delete(item, reason)


def _rmtree_counted(child: Path, result: Dict[str, int], key: str, fail_fmt: str, label) -> None:
    """rmtree ``child``, crediting bytes + ``result[key]``; failures count as ``errors`` when tracked."""
    try:
        size = _dir_size_bytes(child)
        shutil.rmtree(child)
        result["bytes_freed"] += size
        result[key] += 1
    except OSError as exc:
        if "errors" in result:
            result["errors"] += 1
        logger.warning(fail_fmt, label, exc)


def _prune_legacy_archives(base: Path, cutoff: float, result: Dict[str, int]) -> None:
    """Delete ``legacy-*`` archives whose mtime predates ``cutoff`` (skipped when retention is off)."""
    for child in _legacy_archives(base) if cutoff > 0 else ():
        mtime = _mtime_or_none(child)
        if mtime is not None and mtime < cutoff:
            _rmtree_counted(child, result, "deleted_stale", "Failed to delete legacy archive %s: %s", child)


def _prune_pre_v2_repos(base: Path, cutoff: float, delete_orphans: bool,
                        orphan_allowlist: Optional[set], result: Dict[str, int]) -> None:
    """Sweep pre-v2 per-project shadow repos exactly as the v1 pruner did (scan shared with
    ``store_status``; the frozen layout has no recorded parent identity, so orphan detection
    uses the structural checks only)."""
    def entries():
        for repo in _pre_v2_shadow_repos(base):
            child = repo["path"]
            gone = delete_orphans and not repo["marker_unreadable"] and (
                repo["workdir"] is None
                or _workdir_is_observably_gone(repo["workdir"], require_parent_identity=False))
            yield (child, gone, orphan_allowlist is None or str(child) in orphan_allowlist,
                   lambda c=child: cutoff > 0 and 0 < _newest_mtime(c) < cutoff)

    _sweep(entries(), result, lambda child, reason: _rmtree_counted(
        child, result, f"deleted_{reason}", "Failed to prune checkpoint repo %s: %s", child.name))


def _prune_v2_projects(store: Path, cutoff: float, delete_orphans: bool,
                       orphan_allowlist: Optional[set], result: Dict[str, int]) -> None:
    """Drop the ref, index and metadata of orphan/stale projects in the shared store."""
    def entries():
        for meta in _list_projects(store):
            dir_hash = meta.get("_hash") or ""
            workdir = meta.get("workdir") or ""
            if not dir_hash:
                continue
            gone = delete_orphans and (not workdir or _workdir_is_observably_gone(
                workdir, parent_dev=_int_or_none(meta.get("workdir_parent_dev")),
                parent_ino=_int_or_none(meta.get("workdir_parent_ino"))))
            yield (dir_hash, gone, orphan_allowlist is None or dir_hash in orphan_allowlist,
                   lambda m=meta: cutoff > 0 and 0 < float(m.get("last_touch", 0) or 0) < cutoff)

    def delete(dir_hash: str, reason: str) -> None:
        _delete_ref(store, _ref_name(dir_hash))
        _unlink_quiet(_index_path(store, dir_hash))
        _unlink_quiet(_project_meta_path(store, dir_hash))
        result[f"deleted_{reason}"] += 1

    _sweep(entries(), result, delete)


def prune_checkpoints(retention_days: int = 7, delete_orphans: bool = True, checkpoint_base: Optional[Path] = None,
                      max_total_size_mb: int = 0, orphan_allowlist: Optional[set] = None) -> Dict[str, int]:
    """Delete stale/orphan checkpoints and reclaim store space.  Never raises.  Deleted when
    ``delete_orphans`` and the workdir is observably gone, OR last touch predates ``retention_days``
    (``<= 0`` disables).  ``orphan_allowlist`` (v2 ``_hash`` strings and/or pre-v2 repo paths as
    ``str``) binds orphan deletion to exactly what a ``store_status()`` preview showed — a project
    orphaned after the preview is skipped; ``None`` deletes every current orphan (``--force``,
    unattended).  ``max_total_size_mb > 0`` drops the oldest commit per project until the store fits."""
    base = checkpoint_base or CHECKPOINT_BASE
    result = _empty_prune_result()
    if not base.exists():
        return result
    size_before = _dir_size_bytes(base)
    cutoff = time.time() - retention_days * 86400 if retention_days > 0 else 0.0
    _prune_legacy_archives(base, cutoff, result)
    _prune_pre_v2_repos(base, cutoff, delete_orphans, orphan_allowlist, result)
    store = _store_path(base)
    if _store_has_head(store):
        _prune_v2_projects(store, cutoff, delete_orphans, orphan_allowlist, result)
        _gc_store(store, str(base))
        if max_total_size_mb > 0:
            _shrink_store_to_cap(store, str(base), max_total_size_mb * _MB)
            _gc_store(store, str(base))

    result["bytes_freed"] = max(result["bytes_freed"], size_before - _dir_size_bytes(base))
    return result


def maybe_auto_prune_checkpoints(retention_days: int = 7, min_interval_hours: int = 24, delete_orphans: bool = True,
                                 checkpoint_base: Optional[Path] = None, max_total_size_mb: int = 0) -> Dict[str, object]:
    """Idempotent wrapper around ``prune_checkpoints`` for startup hooks: writes
    ``CHECKPOINT_BASE/.last_prune`` so calls within ``min_interval_hours`` short-circuit.
    Returns ``{"skipped": bool, "result": prune dict, "error": optional str}``."""
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict[str, object] = {"skipped": False}
    try:
        if not base.exists():
            out["result"] = _empty_prune_result()
            return out
        marker = base / _PRUNE_MARKER_NAME
        now = time.time()
        try:
            if marker.exists() and now - float(marker.read_text(encoding="utf-8").strip()) < min_interval_hours * 3600:
                out["skipped"] = True
                return out
        except (OSError, ValueError):
            pass  # corrupt marker — treat as no prior run
        result = out["result"] = prune_checkpoints(retention_days=retention_days, delete_orphans=delete_orphans,
                                                   checkpoint_base=base, max_total_size_mb=max_total_size_mb)
        try:
            marker.write_text(str(now), encoding="utf-8")
        except OSError as exc:
            logger.debug("Could not write checkpoint prune marker: %s", exc)

        total = result["deleted_orphan"] + result["deleted_stale"]
        if total > 0:
            logger.info("checkpoint auto-maintenance: pruned %d entry(ies) (%d orphan, %d stale), reclaimed %.1f MB",
                        total, result["deleted_orphan"], result["deleted_stale"], result["bytes_freed"] / _MB)
    except Exception as exc:
        logger.warning("checkpoint auto-maintenance failed: %s", exc)
        out["error"] = str(exc)

    return out


def store_status(checkpoint_base: Optional[Path] = None) -> Dict:
    """Summarise the shadow store: ``{"base", "store_size_bytes", "legacy_size_bytes",
    "total_size_bytes", "project_count", "projects", "pre_v2_projects", "legacy_archives"}``.
    ``pre_v2_projects`` are repos still on the pre-v2 layout, distinct from the migrated
    ``legacy_archives``; an orphan-deletion preview must include both ``projects`` and
    ``pre_v2_projects`` since ``prune_checkpoints`` sweeps both."""
    base = checkpoint_base or CHECKPOINT_BASE
    out: Dict = {"base": str(base), "store_size_bytes": 0, "legacy_size_bytes": 0, "total_size_bytes": 0,
                 "project_count": 0, "projects": [], "pre_v2_projects": [], "legacy_archives": []}
    if not base.exists():
        return out

    store = _store_path(base)
    if store.exists():
        out["store_size_bytes"] = _dir_size_bytes(store)
        if _store_has_head(store):
            out["projects"] = [{
                "hash": meta.get("_hash") or "", "workdir": meta.get("workdir") or "",
                "exists": bool(meta.get("workdir")) and Path(meta["workdir"]).exists(),
                "created_at": meta.get("created_at"), "last_touch": meta.get("last_touch"),
                "commits": _ref_commit_count(store, str(base), _ref_name(meta.get("_hash") or "")),
            } for meta in _list_projects(store)]
    out["project_count"] = len(out["projects"])
    out["pre_v2_projects"] = [{"path": str(r["path"]), "workdir": r["workdir"], "exists": r["exists"]}
                              for r in _pre_v2_shadow_repos(base)]

    out["legacy_archives"] = [{"name": c.name, "size_bytes": _dir_size_bytes(c), "mtime": _mtime_or_none(c) or 0}
                              for c in _legacy_archives(base)]
    out["legacy_size_bytes"] = sum(a["size_bytes"] for a in out["legacy_archives"])
    out["total_size_bytes"] = _dir_size_bytes(base)
    return out


def clear_all(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Nuke the entire checkpoint base (store + legacy).  Irreversible.
    Returns ``{"bytes_freed": N, "deleted": bool}``."""
    base = checkpoint_base or CHECKPOINT_BASE
    out = {"bytes_freed": 0, "deleted": False}
    if not base.exists():
        return out
    size = _dir_size_bytes(base)
    try:
        shutil.rmtree(base)
        out.update(bytes_freed=size, deleted=True)
    except OSError as exc:
        logger.warning("Could not clear checkpoint base %s: %s", base, exc)
    return out


def clear_legacy(checkpoint_base: Optional[Path] = None) -> Dict[str, int]:
    """Delete all ``legacy-*`` archive directories.  Returns ``{"bytes_freed": N, "deleted": count}``."""
    base = checkpoint_base or CHECKPOINT_BASE
    out = {"bytes_freed": 0, "deleted": 0}
    if not base.exists():
        return out
    for child in _legacy_archives(base):
        _rmtree_counted(child, out, "deleted", "Could not delete legacy archive %s: %s", child)
    return out

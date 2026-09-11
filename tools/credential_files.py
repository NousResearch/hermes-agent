"""File passthrough registry for remote terminal backends (Docker, Modal, SSH).

Sandboxes start with no host files; this module tells them which credential files
(skill ``required_credential_files`` + ``terminal.credential_files`` config), skill
dirs, and host cache dirs to mount or sync in, at creation and before each command.
"""

from __future__ import annotations

import logging
import os
import posixpath
from contextvars import ContextVar
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_dir, get_hermes_home

from agent.skill_utils import EXCLUDED_SKILL_DIRS

try:  # pragma: no cover - exercised via the fail-closed test below
    from agent.file_safety import get_read_block_error
except ImportError:  # noqa: F401 - sentinel consumed in register_credential_file
    get_read_block_error = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Session-scoped registry; ContextVar prevents cross-session bleed in the gateway.
_registered_files_var: ContextVar[Dict[str, str]] = ContextVar("_registered_files")

# Cache for config-based file list (loaded once per process; tests reset it).
_config_files: List[Dict[str, str]] | None = None
# Reused across calls so sanitized skill copies don't accumulate.
_safe_skills_tempdir: Path | None = None


def _get_registered() -> Dict[str, str]:
    val = _registered_files_var.get(None)
    if val is None:
        _registered_files_var.set(val := {})
    return val


def _mount(host_path: Path | str, container_path: str) -> Dict[str, str]:
    return {"host_path": str(host_path), "container_path": container_path}


def _contained_host_path(rel: str, hermes_home: Path, abs_msg: str, traversal_msg: str) -> Optional[Path]:
    """Resolve *rel* under HERMES_HOME, refusing absolute paths and escapes."""
    if os.path.isabs(rel):
        logger.warning(abs_msg, rel)
        return None
    host_path = hermes_home / rel
    from tools.path_security import validate_within_dir  # resolves symlinks and ``..`` before checking

    if containment_error := validate_within_dir(host_path, hermes_home):
        logger.warning(traversal_msg, rel, containment_error)
        return None
    return host_path.resolve()


def register_credential_file(relative_path: str, container_base: str = "/root/.hermes") -> bool:
    """Register a HERMES_HOME-relative credential file for mounting; True if it exists and was registered.

    Rejects absolute paths and traversal out of HERMES_HOME. Containment alone is not
    enough: HERMES_HOME holds the MASTER stores (``.env``, ``auth.json``, ``mcp-tokens/``),
    which are refused via the canonical read deny-list so the mount surface cannot hand a
    skill what the read surface denies. Fails CLOSED (logged) if the guard is unavailable or raises.
    """
    resolved = _contained_host_path(
        relative_path, get_hermes_home(),
        "credential_files: rejected absolute path %r (must be relative to HERMES_HOME)",
        "credential_files: rejected path traversal %r (%s)")
    if resolved is None:
        return False
    if not resolved.is_file():
        logger.debug("credential_files: skipping %s (not found)", resolved)
        return False
    # Master credential stores are never mountable, even though they sit inside HERMES_HOME and therefore
    # pass the containment check above. Fails CLOSED: if the canonical guard can't be consulted we refuse
    # the mount rather than risk bind-mounting auth.json into a sandbox. The import lives at module top (no
    # circular-import concern — file_safety is stdlib-only); the sentinel + logger.exception keep guard
    # failures debuggable instead of silently swallowed (#67665).
    if get_read_block_error is None:
        logger.error("credential_files: refusing %r — agent.file_safety could not be "
                     "imported, so the master-store deny-list cannot be consulted", relative_path)
        return False
    try:
        denied = get_read_block_error(str(resolved))
    except Exception:
        logger.exception("credential_files: refusing %r — read guard raised", relative_path)
        return False
    if denied:
        logger.warning("credential_files: refused %r — it is a credential store the agent "
                       "is denied from reading; a skill may mount its own service token, "
                       "not the master key files", relative_path)
        return False

    container_path = f"{container_base.rstrip('/')}/{relative_path}"
    _get_registered()[container_path] = str(resolved)
    logger.debug("credential_files: registered %s -> %s", resolved, container_path)
    return True


def register_credential_files(entries: list, container_base: str = "/root/.hermes") -> List[str]:
    """Register skill-frontmatter entries (str or dict with ``path``); return missing paths."""
    missing = []
    for entry in entries:
        if isinstance(entry, dict):
            entry = entry.get("path") or entry.get("name") or ""
        elif not isinstance(entry, str):
            continue
        rel_path = entry.strip()
        if rel_path and not register_credential_file(rel_path, container_base):
            missing.append(rel_path)
    return missing


def _load_config_files() -> List[Dict[str, str]]:
    """Load ``terminal.credential_files`` from config.yaml (cached)."""
    global _config_files
    if _config_files is not None:
        return _config_files

    result: List[Dict[str, str]] = []
    try:
        from hermes_cli.config import read_raw_config
        hermes_home = get_hermes_home()
        cred_files = cfg_get(read_raw_config(), "terminal", "credential_files")
        for item in cred_files if isinstance(cred_files, list) else []:
            rel = item.strip() if isinstance(item, str) else ""
            if not rel:
                continue
            resolved_path = _contained_host_path(
                rel, hermes_home,
                "credential_files: rejected absolute config path %r",
                "credential_files: rejected config path traversal %r (%s)")
            if resolved_path is not None and resolved_path.is_file():
                result.append(_mount(resolved_path, f"/root/.hermes/{rel}"))
    except Exception as e:
        logger.warning("Could not read terminal.credential_files from config: %s", e)

    _config_files = result
    return _config_files


def get_credential_file_mounts() -> List[Dict[str, str]]:
    """Skill-registered + config credential files as ``host_path``/``container_path`` dicts (re-checked for existence)."""
    mounts = {cp: hp for cp, hp in _get_registered().items() if Path(hp).is_file()}
    for entry in _load_config_files():
        cp, hp = entry["container_path"], entry["host_path"]
        if cp not in mounts and Path(hp).is_file():
            mounts[cp] = hp
    return [_mount(hp, cp) for cp, hp in mounts.items()]


# --- Skills directory mounts ---

def _skill_dir_roots(container_base: str) -> Iterator[Tuple[Path, str]]:
    """Yield ``(host_dir, container_root)`` for every existing skills directory.

    Local skills mount at ``<base>/skills``, external at ``<base>/external_skills/<i>``, trusted
    project-local at ``<base>/project_skills/<i>`` (own namespace so paths stay stable if external_dirs change).
    """
    base = container_base.rstrip("/")
    skills_dir = get_hermes_home() / "skills"
    if skills_dir.is_dir():
        yield skills_dir, f"{base}/skills"
    try:
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
    except ImportError:
        return
    for label, dirs in (("external_skills", get_external_skills_dirs()), ("project_skills", get_project_skills_dirs())):
        yield from ((d, f"{base}/{label}/{idx}") for idx, d in enumerate(dirs) if d.is_dir())


def _walk_skill_tree(root: Path) -> Iterator[Tuple[Path, List[Path]]]:
    """Yield ``(dir, regular_non_symlink_files)`` for every directory a sandbox should receive.

    Prunes ``EXCLUDED_SKILL_DIRS`` *before* descending so bookkeeping/dependency trees (``.hub``,
    ``.archive``, ``.curator_backups``, ``node_modules``, ``.git``, ...) the remote agent never reads
    are never even walked; sync thus agrees with discovery on what is skill content. Deliberately
    not ``is_excluded_skill_path()``: that also prunes ``references/``, ``templates/``, ``assets/``,
    ``scripts/`` — progressive-disclosure files and bundled scripts the sandbox does execute.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_SKILL_DIRS)
        base = Path(dirpath)
        yield base, [f for f in (base / n for n in filenames) if not f.is_symlink() and f.is_file()]


def get_skills_directory_mount(container_base: str = "/root/.hermes") -> list[Dict[str, str]]:
    """Directory mount entries for all skill dirs (local + external + project).

    Bind mounts follow symlinks, so a dir containing any symlink is replaced by a sanitized
    temp copy (regular files only); symlink-free dirs are returned directly, zero overhead.
    """
    return [_mount(_safe_skills_path(d), cp) for d, cp in _skill_dir_roots(container_base)]


def _safe_skills_path(skills_dir: Path) -> str:
    """Return *skills_dir* if symlink-free, else a sanitized temp copy (same exclusions as sync)."""
    global _safe_skills_tempdir

    symlinks = [p for p in skills_dir.rglob("*") if p.is_symlink()]
    if not symlinks:
        return str(skills_dir)
    for link in symlinks:
        logger.warning("credential_files: skipping symlink in skills dir: %s -> %s", link, os.readlink(link))

    import atexit
    import shutil
    import tempfile

    if _safe_skills_tempdir and _safe_skills_tempdir.is_dir():
        shutil.rmtree(_safe_skills_tempdir, ignore_errors=True)
    safe_dir = _safe_skills_tempdir = Path(tempfile.mkdtemp(prefix="hermes-skills-safe-"))

    for base, files in _walk_skill_tree(skills_dir):
        (safe_dir / base.relative_to(skills_dir)).mkdir(parents=True, exist_ok=True)
        for item in files:
            shutil.copy2(str(item), str(safe_dir / item.relative_to(skills_dir)))

    atexit.register(lambda: safe_dir.is_dir() and shutil.rmtree(safe_dir, ignore_errors=True))
    logger.info("credential_files: created symlink-safe skills copy at %s", safe_dir)
    return str(safe_dir)


def iter_skills_files(container_base: str = "/root/.hermes") -> List[Dict[str, str]]:
    """Per-file entries for all skills files (for backends that upload individually)."""
    return [_mount(item, f"{container_root}/{item.relative_to(host_dir)}")
            for host_dir, container_root in _skill_dir_roots(container_base)
            for _base, files in _walk_skill_tree(host_dir) for item in files]


# --- Cache directory mounts (documents, images, audio, videos, screenshots) ---

# (new_subpath, old_name) pairs matching hermes_constants.get_hermes_dir().
_CACHE_DIRS: list[tuple[str, str]] = [
    ("cache/documents", "document_cache"),
    ("cache/images", "image_cache"),
    ("cache/audio", "audio_cache"),
    ("cache/videos", "video_cache"),
    ("cache/screenshots", "browser_screenshots"),
    ("cache/web", "web_cache"),
    ("cache/delegation", "delegation_cache"),
    ("cache/spillover", "cache/spillover"),  # oversized tool results; host side is canonical
    # Flat top-level desktop staging dirs (tui_gateway attach RPCs; no legacy alias),
    # mounted so vision/file tools in sandboxes reach uploads and dropped files.
    # Mount it so vision can reach uploads inside sandbox containers (#69575). No legacy alias exists, so
    # both tuple slots are ``images``.
    ("images", "images"),
    # Mount it so the agent's file tools can read dropped binaries (zip/pdf/...) from inside sandbox
    # containers instead of dangling host paths (#76577).
    ("attachments", "attachments"),
]


def _cache_dir_roots(container_base: str, *, create_missing: bool) -> Iterator[Tuple[Path, str]]:
    """Yield ``(host_dir, container_root)`` per cache dir; always maps to the *new* container layout."""
    base = container_base.rstrip("/")
    for new_subpath, old_name in _CACHE_DIRS:
        host_dir = get_hermes_dir(new_subpath, old_name)
        if not host_dir.is_dir():
            if not create_missing:
                continue
            # Docker snapshots this list at container CREATION, so a dir appearing later
            # would dangle for the container's life: create it now (empty bind mount is free).
            # get_hermes_dir already picked new-vs-legacy, so this can't shadow a legacy dir.
            try:
                # Create missing staging dirs instead of skipping them: Docker snapshots this mount list at
                # container CREATION, so a dir that appears later (first desktop attachment, first clipboard
                # image) would dangle for the whole life of a persistent container (#76577). An empty
                # bind-mounted dir costs nothing; a missing mount costs the feature. get_hermes_dir()
                # already resolved new-vs-legacy layout, so creating its answer cannot shadow a populated
                # legacy dir.
                host_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue  # unwritable home (tests, RO mounts) — skip as before
        yield host_dir, f"{base}/{new_subpath}"


def get_cache_directory_mounts(container_base: str = "/root/.hermes") -> List[Dict[str, str]]:
    """Bind-mount entries for each cache directory (host layout via ``get_hermes_dir``)."""
    return [_mount(h, c) for h, c in _cache_dir_roots(container_base, create_missing=True)]


def _remap_cache_path(path: str, container_base: str, src: str, dst: str, join: Callable[[str, Path], str]) -> Optional[str]:
    """Translate *path* from the *src* side of a cache mount to its *dst* side; None if unmounted."""
    for mount in get_cache_directory_mounts(container_base=container_base):
        if Path(path).is_relative_to(mount[src]):
            return join(mount[dst], Path(path).relative_to(mount[src]))
    return None


def map_cache_path_to_container(host_path: str, container_base: str = "/root/.hermes") -> Optional[str]:
    """POSIX container path for a host path under an auto-mounted cache dir, else None."""
    return _remap_cache_path(host_path, container_base, "host_path", "container_path", lambda root, rel: posixpath.join(root, rel.as_posix()))


def from_agent_visible_cache_path(container_path: str, container_base: str = "/root/.hermes") -> str:
    """Inverse of :func:`to_agent_visible_cache_path`; unchanged unless Docker + cache dir."""
    if os.environ.get("TERMINAL_ENV", "local") != "docker":
        return container_path
    mapped = _remap_cache_path(container_path, container_base, "container_path", "host_path", lambda root, rel: str(Path(root) / rel))
    return mapped if mapped is not None else container_path


# Backends whose file-sync lands under the remote home: ``~/.hermes`` is
# expanded by the remote shell, so it resolves regardless of the actual home.
_HOME_RELATIVE_BACKENDS = frozenset({"ssh", "daytona", "vercel_sandbox"})


def to_agent_visible_cache_path(host_path: str, container_base: str = "/root/.hermes") -> str:
    """Translate a host cache path to where the active backend (TERMINAL_ENV) sees it.

    Mirrors ``_agent_cache_base_for_env`` in tools/image_generation_tool.py: docker/modal mount at
    ``/root/.hermes``; ssh/daytona/vercel_sandbox under ``~/.hermes``; plugin backends declare
    ``cache_path_base`` (None = host paths stay correct); local/singularity/unknown unchanged
    (Apptainer auto-binds the host home, so translation would dangle).

    * docker / modal — bind-mounted (docker) or per-file-synced (modal) at ``/root/.hermes`` (the
    *container_base* default). * ssh / daytona / vercel_sandbox — file-synced under the remote user's home;
    ``~/.hermes`` is shell-expanded by the remote shell, so tool commands resolve it regardless of the
    actual remote home. Previously these backends synced the bytes but still rendered the dangling host path
    (#76577 gap).
    """
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend in _HOME_RELATIVE_BACKENDS:
        container_base = "~/.hermes"
    elif backend not in ("docker", "modal"):
        try:
            from agent.terminal_env_registry import provider_flag
            plugin_base = provider_flag(backend, "cache_path_base", None)
        except Exception:
            plugin_base = None
        if not plugin_base:
            return host_path
        container_base = str(plugin_base)

    mapped = map_cache_path_to_container(host_path, container_base=container_base)
    return mapped if mapped is not None else host_path


def iter_cache_files(container_base: str = "/root/.hermes") -> List[Dict[str, str]]:
    """Per-file cache entries (Modal upload/resync); skips symlinks."""
    return [_mount(item, f"{root}/{item.relative_to(host_dir)}")
            for host_dir, root in _cache_dir_roots(container_base, create_missing=False)
            for item in host_dir.rglob("*") if not item.is_symlink() and item.is_file()]


def clear_credential_files() -> None:
    """Reset the skill-scoped registry (e.g. on session reset)."""
    _get_registered().clear()

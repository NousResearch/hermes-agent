"""File passthrough registry for remote terminal backends.

Remote backends (Docker, Modal, SSH) create sandboxes with no host files.
This module ensures that credential files, skill directories, and host-side
cache directories (documents, images, audio, screenshots) are mounted or
synced into those sandboxes so the agent can access them.

**Credentials and skills** — session-scoped registry fed by skill declarations
(``required_credential_files``) and user config (``terminal.credential_files``).

**Cache directories** — gateway-cached uploads, browser screenshots, TTS
audio, and processed images.  Mounted read-only so the remote terminal can
reference files the host side created (e.g. ``unzip`` an uploaded archive).

Remote backends call :func:`get_credential_file_mounts`,
:func:`get_skills_directory_mount` / :func:`iter_skills_files`, and
:func:`get_cache_directory_mounts` / :func:`iter_cache_files` at sandbox
creation time and before each command (for resync on Modal).
"""

from __future__ import annotations

import logging
import os
import posixpath
from contextvars import ContextVar
from pathlib import Path
from typing import Dict, List, Optional
from hermes_cli.config import cfg_get

from agent.skill_utils import EXCLUDED_SKILL_DIRS

try:  # pragma: no cover - exercised via the fail-closed test below
    from agent.file_safety import get_read_block_error
except ImportError:  # noqa: F401 - sentinel consumed in register_credential_file
    get_read_block_error = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Session-scoped list of credential files to mount.
# Backed by ContextVar to prevent cross-session data bleed in the gateway pipeline.
_registered_files_var: ContextVar[Dict[str, str]] = ContextVar("_registered_files")


def _get_registered() -> Dict[str, str]:
    """Get or create the registered credential files dict for the current context/session."""
    try:
        return _registered_files_var.get()
    except LookupError:
        val: Dict[str, str] = {}
        _registered_files_var.set(val)
        return val


# Cache for config-based file list (loaded once per process).
_config_files: List[Dict[str, str]] | None = None


def _resolve_hermes_home() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def register_credential_file(
    relative_path: str,
    container_base: str = "/root/.hermes",
) -> bool:
    """Register a credential file for mounting into remote sandboxes.

    *relative_path* is relative to ``HERMES_HOME`` (e.g. ``google_token.json``).
    Returns True if the file exists on the host and was registered.

    Security: rejects absolute paths and path traversal sequences (``..``).
    The resolved host path must remain inside HERMES_HOME so that a malicious
    skill cannot declare ``required_credential_files: ['../../.ssh/id_rsa']``
    and exfiltrate sensitive host files into a container sandbox.

    Containment alone is not sufficient, because HERMES_HOME is exactly where
    the MASTER credential stores live. A skill legitimately needs its own
    service token (``google_token.json``); it never needs ``.env`` (every
    provider key), ``auth.json`` (all provider tokens and OAuth grants),
    ``mcp-tokens/`` or the Bitwarden plaintext cache. Those are refused via
    the canonical read deny-list (``agent.file_safety.get_read_block_error``)
    — the same guard that stops the agent reading them with ``read_file``, so
    the mount surface cannot hand a skill what the read surface denies it.
    """
    hermes_home = _resolve_hermes_home()

    # Reject absolute paths — they bypass the HERMES_HOME sandbox entirely.
    if os.path.isabs(relative_path):
        logger.warning(
            "credential_files: rejected absolute path %r (must be relative to HERMES_HOME)",
            relative_path,
        )
        return False

    host_path = hermes_home / relative_path

    # Resolve symlinks and normalise ``..`` before the containment check so
    # that traversal like ``../. ssh/id_rsa`` cannot escape HERMES_HOME.
    from tools.path_security import validate_within_dir

    containment_error = validate_within_dir(host_path, hermes_home)
    if containment_error:
        logger.warning(
            "credential_files: rejected path traversal %r (%s)",
            relative_path,
            containment_error,
        )
        return False

    resolved = host_path.resolve()
    if not resolved.is_file():
        logger.debug("credential_files: skipping %s (not found)", resolved)
        return False

    # Master credential stores are never mountable, even though they sit
    # inside HERMES_HOME and therefore pass the containment check above.
    # Fails CLOSED: if the canonical guard can't be consulted we refuse the
    # mount rather than risk bind-mounting auth.json into a sandbox. The
    # import lives at module top (no circular-import concern — file_safety is
    # stdlib-only); the sentinel + logger.exception keep guard failures
    # debuggable instead of silently swallowed (#67665).
    if get_read_block_error is None:
        logger.error(
            "credential_files: refusing %r — agent.file_safety could not be "
            "imported, so the master-store deny-list cannot be consulted",
            relative_path,
        )
        return False
    try:
        denied = get_read_block_error(str(resolved))
    except Exception:
        logger.exception(
            "credential_files: refusing %r — read guard raised", relative_path
        )
        return False
    if denied:
        logger.warning(
            "credential_files: refused %r — it is a credential store the agent "
            "is denied from reading; a skill may mount its own service token, "
            "not the master key files",
            relative_path,
        )
        return False

    container_path = f"{container_base.rstrip('/')}/{relative_path}"
    _get_registered()[container_path] = str(resolved)
    logger.debug("credential_files: registered %s -> %s", resolved, container_path)
    return True


def register_credential_files(
    entries: list,
    container_base: str = "/root/.hermes",
) -> List[str]:
    """Register multiple credential files from skill frontmatter entries.

    Each entry is either a string (relative path) or a dict with a ``path``
    key.  Returns the list of relative paths that were NOT found on the host
    (i.e. missing files).
    """
    missing = []
    for entry in entries:
        if isinstance(entry, str):
            rel_path = entry.strip()
        elif isinstance(entry, dict):
            rel_path = (entry.get("path") or entry.get("name") or "").strip()
        else:
            continue
        if not rel_path:
            continue
        if not register_credential_file(rel_path, container_base):
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
        hermes_home = _resolve_hermes_home()
        cfg = read_raw_config()
        cred_files = cfg_get(cfg, "terminal", "credential_files")
        if isinstance(cred_files, list):
            from tools.path_security import validate_within_dir

            for item in cred_files:
                if isinstance(item, str) and item.strip():
                    rel = item.strip()
                    if os.path.isabs(rel):
                        logger.warning(
                            "credential_files: rejected absolute config path %r", rel,
                        )
                        continue
                    host_path = hermes_home / rel
                    containment_error = validate_within_dir(host_path, hermes_home)
                    if containment_error:
                        logger.warning(
                            "credential_files: rejected config path traversal %r (%s)",
                            rel, containment_error,
                        )
                        continue
                    resolved_path = host_path.resolve()
                    if resolved_path.is_file():
                        container_path = f"/root/.hermes/{rel}"
                        result.append({
                            "host_path": str(resolved_path),
                            "container_path": container_path,
                        })
    except Exception as e:
        logger.warning("Could not read terminal.credential_files from config: %s", e)

    _config_files = result
    return _config_files


def get_credential_file_mounts() -> List[Dict[str, str]]:
    """Return all credential files that should be mounted into remote sandboxes.

    Each item has ``host_path`` and ``container_path`` keys.
    Combines skill-registered files and user config.
    """
    mounts: Dict[str, str] = {}

    # Skill-registered files
    for container_path, host_path in _get_registered().items():
        # Re-check existence (file may have been deleted since registration)
        if Path(host_path).is_file():
            mounts[container_path] = host_path

    # Config-based files
    for entry in _load_config_files():
        cp = entry["container_path"]
        if cp not in mounts and Path(entry["host_path"]).is_file():
            mounts[cp] = entry["host_path"]

    return [
        {"host_path": hp, "container_path": cp}
        for cp, hp in mounts.items()
    ]


def get_skills_directory_mount(
    container_base: str = "/root/.hermes",
) -> list[Dict[str, str]]:
    """Return mount info for all skill directories (local + external).

    Skills may include ``scripts/``, ``templates/``, and ``references/``
    subdirectories that the agent needs to execute inside remote sandboxes.

    **Security:** Bind mounts follow symlinks, so a malicious symlink inside
    the skills tree could expose arbitrary host files to the container.  When
    symlinks are detected, this function creates a sanitized copy (regular
    files only) in a temp directory and returns that path instead.  When no
    symlinks are present (the common case), the original directory is returned
    directly with zero overhead.

    Returns a list of dicts with ``host_path`` and ``container_path`` keys.
    The local skills dir mounts at ``<container_base>/skills``, external dirs
    at ``<container_base>/external_skills/<index>``.
    """
    mounts = []
    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if skills_dir.is_dir():
        host_path = _safe_skills_path(skills_dir)
        mounts.append({
            "host_path": host_path,
            "container_path": f"{container_base.rstrip('/')}/skills",
        })

    # Mount external skill dirs
    try:
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
        for idx, ext_dir in enumerate(get_external_skills_dirs()):
            if ext_dir.is_dir():
                host_path = _safe_skills_path(ext_dir)
                mounts.append({
                    "host_path": host_path,
                    "container_path": f"{container_base.rstrip('/')}/external_skills/{idx}",
                })
        # Trusted project-local skill dirs (repo checkouts). Separate
        # namespace so container paths stay stable if external_dirs change.
        for idx, proj_dir in enumerate(get_project_skills_dirs()):
            if proj_dir.is_dir():
                host_path = _safe_skills_path(proj_dir)
                mounts.append({
                    "host_path": host_path,
                    "container_path": f"{container_base.rstrip('/')}/project_skills/{idx}",
                })
    except ImportError:
        pass

    return mounts


_safe_skills_tempdir: Path | None = None


def _safe_skills_path(skills_dir: Path) -> str:
    """Return *skills_dir* if symlink-free, else a sanitized temp copy."""
    global _safe_skills_tempdir

    symlinks = [p for p in skills_dir.rglob("*") if p.is_symlink()]
    if not symlinks:
        return str(skills_dir)

    for link in symlinks:
        logger.warning("credential_files: skipping symlink in skills dir: %s -> %s",
                       link, os.readlink(link))

    import atexit
    import shutil
    import tempfile

    # Reuse the same temp dir across calls to avoid accumulation.
    if _safe_skills_tempdir and _safe_skills_tempdir.is_dir():
        shutil.rmtree(_safe_skills_tempdir, ignore_errors=True)

    safe_dir = Path(tempfile.mkdtemp(prefix="hermes-skills-safe-"))
    _safe_skills_tempdir = safe_dir

    # Same exclusion rule as the per-file sync path (_iter_syncable_files):
    # the sanitized copy is what gets mounted, so it must not carry the
    # bookkeeping trees either. Prune before descending so a multi-GB
    # .curator_backups is never even walked.
    for dirpath, dirnames, filenames in os.walk(skills_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_SKILL_DIRS)
        base = Path(dirpath)
        (safe_dir / base.relative_to(skills_dir)).mkdir(parents=True, exist_ok=True)
        for name in filenames:
            item = base / name
            if item.is_symlink() or not item.is_file():
                continue
            shutil.copy2(str(item), str(safe_dir / item.relative_to(skills_dir)))

    def _cleanup():
        if safe_dir.is_dir():
            shutil.rmtree(safe_dir, ignore_errors=True)

    atexit.register(_cleanup)
    logger.info("credential_files: created symlink-safe skills copy at %s", safe_dir)
    return str(safe_dir)


def _iter_syncable_files(root: Path):
    """Yield ``(path, rel)`` for every regular, non-symlink file under *root*
    that a sandbox should receive.

    Prunes ``agent.skill_utils.EXCLUDED_SKILL_DIRS`` *before* descending, so
    the walk never enters local bookkeeping and dependency trees (``.hub``
    download cache, ``.archive``, ``.curator_backups``, ``node_modules``,
    ``__pycache__``, ``.git``, ...) that the remote agent never reads — the
    sync path agrees with discovery on what counts as skill content.

    This deliberately does not use ``is_excluded_skill_path()``, which also
    prunes ``references/``, ``templates/``, ``assets/`` and ``scripts/``.
    Those hold progressive-disclosure support files and bundled scripts the
    sandbox does execute, so they must keep syncing.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDED_SKILL_DIRS)
        base = Path(dirpath)
        for name in filenames:
            item = base / name
            if item.is_symlink() or not item.is_file():
                continue
            yield item, item.relative_to(root)


def iter_skills_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Yield individual (host_path, container_path) entries for skills files.

    Includes both the local skills dir and any external dirs configured via
    skills.external_dirs.  Skips symlinks and anything under
    EXCLUDED_SKILL_DIRS entirely.  Preferred for backends that upload files
    individually (Daytona, Modal) rather than mounting a directory.
    """
    result: List[Dict[str, str]] = []

    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if skills_dir.is_dir():
        container_root = f"{container_base.rstrip('/')}/skills"
        for item, rel in _iter_syncable_files(skills_dir):
            result.append({
                "host_path": str(item),
                "container_path": f"{container_root}/{rel}",
            })

    # Include external skill dirs
    try:
        from agent.skill_utils import get_external_skills_dirs, get_project_skills_dirs
        for idx, ext_dir in enumerate(get_external_skills_dirs()):
            if not ext_dir.is_dir():
                continue
            container_root = f"{container_base.rstrip('/')}/external_skills/{idx}"
            for item, rel in _iter_syncable_files(ext_dir):
                result.append({
                    "host_path": str(item),
                    "container_path": f"{container_root}/{rel}",
                })
        for idx, proj_dir in enumerate(get_project_skills_dirs()):
            if not proj_dir.is_dir():
                continue
            container_root = f"{container_base.rstrip('/')}/project_skills/{idx}"
            for item, rel in _iter_syncable_files(proj_dir):
                result.append({
                    "host_path": str(item),
                    "container_path": f"{container_root}/{rel}",
                })
    except ImportError:
        pass

    return result


# ---------------------------------------------------------------------------
# Cache directory mounts (documents, images, audio, videos, screenshots)
# ---------------------------------------------------------------------------

# The cache subdirectories that should be mirrored into remote backends.
# Each tuple is (new_subpath, old_name) matching hermes_constants.get_hermes_dir().
_CACHE_DIRS: list[tuple[str, str]] = [
    ("cache/documents", "document_cache"),
    ("cache/images", "image_cache"),
    ("cache/audio", "audio_cache"),
    ("cache/videos", "video_cache"),
    ("cache/screenshots", "browser_screenshots"),
    ("cache/web", "web_cache"),
    ("cache/delegation", "delegation_cache"),
    # Oversized tool results (tools/tool_result_storage.py). Host-side is the
    # single canonical location; mounting/syncing it lets remote backends
    # read spilled results at the translated path instead of needing a
    # separate in-sandbox copy.
    ("cache/spillover", "cache/spillover"),
    # Desktop/clipboard/PDF uploads land in the flat top-level ``images/`` dir
    # (tui_gateway attach RPCs), not under ``cache/``. Mount it so vision can
    # reach uploads inside sandbox containers (#69575). No legacy alias exists,
    # so both tuple slots are ``images``.
    ("images", "images"),
    # Desktop non-image file attachments (tui_gateway ``file.attach`` staging)
    # land in the flat top-level ``attachments/`` dir. Mount it so the agent's
    # file tools can read dropped binaries (zip/pdf/...) from inside sandbox
    # containers instead of dangling host paths (#76577).
    ("attachments", "attachments"),
]


def get_cache_directory_mounts(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return mount entries for each cache directory that exists on disk.

    Used by Docker to create bind mounts.  Each entry has ``host_path`` and
    ``container_path`` keys.  The host path is resolved via
    ``get_hermes_dir()`` for backward compatibility with old directory layouts.
    """
    from hermes_constants import get_hermes_dir

    mounts: List[Dict[str, str]] = []
    for new_subpath, old_name in _CACHE_DIRS:
        host_dir = get_hermes_dir(new_subpath, old_name)
        if not host_dir.is_dir():
            # Create missing staging dirs instead of skipping them: Docker
            # snapshots this mount list at container CREATION, so a dir that
            # appears later (first desktop attachment, first clipboard image)
            # would dangle for the whole life of a persistent container
            # (#76577). An empty bind-mounted dir costs nothing; a missing
            # mount costs the feature. get_hermes_dir() already resolved
            # new-vs-legacy layout, so creating its answer cannot shadow a
            # populated legacy dir.
            try:
                host_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue  # unwritable home (tests, RO mounts) — skip as before
        # Always map to the *new* container layout regardless of host layout.
        container_path = f"{container_base.rstrip('/')}/{new_subpath}"
        mounts.append({
            "host_path": str(host_dir),
            "container_path": container_path,
        })
    return mounts


def map_cache_path_to_container(
    host_path: str,
    container_base: str = "/root/.hermes",
) -> Optional[str]:
    """Map a host cache path to its mounted path under *container_base*.

    Returns the POSIX container path when *host_path* lives under one of the
    auto-mounted cache directories, otherwise ``None``.  Backend-agnostic: the
    caller decides which ``container_base`` applies (Docker ``/root/.hermes``,
    SSH ``<remote_home>/.hermes``, etc.) and whether translation is wanted.
    Always joins with ``posixpath`` because container/remote paths are POSIX
    regardless of the host OS.
    """
    path = Path(host_path)
    for mount in get_cache_directory_mounts(container_base=container_base):
        host_dir = Path(mount["host_path"])
        try:
            rel = path.relative_to(host_dir)
        except ValueError:
            continue
        return posixpath.join(mount["container_path"], rel.as_posix())
    return None


def from_agent_visible_cache_path(
    container_path: str,
    container_base: str = "/root/.hermes",
) -> str:
    """Translate a sandbox/container cache path back to its host path.

    Inverse of :func:`to_agent_visible_cache_path`. Returns the input unchanged
    when the active backend does not use local bind mounts, or when the path is not under any
    auto-mounted cache directory — the caller then treats a still-container
    path as "no host file" and falls back to an in-container read.
    """
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend in ("docker", "apple_container"):
        pass  # /root/.hermes default
    else:
        # Plugin-registered backends declare where synced cache files land via
        # ``cache_path_base``; None means host paths remain correct and there is
        # nothing to reverse-translate.
        plugin_base = None
        try:
            from agent.terminal_env_registry import provider_flag

            plugin_base = provider_flag(backend, "cache_path_base", None)
        except Exception:
            plugin_base = None
        if not plugin_base:
            return container_path
        container_base = str(plugin_base)

    path = Path(container_path)
    for mount in get_cache_directory_mounts(container_base=container_base):
        try:
            rel = path.relative_to(mount["container_path"])
        except ValueError:
            continue
        return str(Path(mount["host_path"]) / rel)
    return container_path


def to_agent_visible_cache_path(
    host_path: str,
    container_base: str = "/root/.hermes",
) -> str:
    """Translate a host cache path to its mounted path inside the sandbox.

    Returns the input unchanged if it is not under any auto-mounted cache
    directory, or if the active terminal backend does not require path
    translation (local).

    Per-backend base (mirrors ``_agent_cache_base_for_env`` in
    tools/image_generation_tool.py, the proven heuristics for where each
    backend's Hermes cache lands):

    * docker / modal / apple_container — bind-mounted (docker, apple_container)
      or per-file-synced (modal) at ``/root/.hermes`` (the *container_base*
      default).
    * ssh / daytona / vercel_sandbox — file-synced under the remote user's
      home; ``~/.hermes`` is shell-expanded by the remote shell, so tool
      commands resolve it regardless of the actual remote home. Previously
      these backends synced the bytes but still rendered the dangling host
      path (#76577 gap).
    * singularity — NOT translated: Apptainer auto-binds the host home, so
      the host path is directly readable and translation would dangle
      (cache dirs are not remapped into that sandbox).

    Backend is identified by TERMINAL_ENV (same env var
    tools/terminal_tool.py reads in _get_environment_config).
    """
    backend = (os.environ.get("TERMINAL_ENV") or "local").strip().lower()
    if backend in ("docker", "modal", "apple_container"):
        pass  # /root/.hermes default
    elif backend in ("ssh", "daytona", "vercel_sandbox"):
        container_base = "~/.hermes"
    else:
        # Plugin-registered backends declare where synced cache files land
        # via ``cache_path_base``; None means host paths remain correct.
        plugin_base = None
        try:
            from agent.terminal_env_registry import provider_flag

            plugin_base = provider_flag(backend, "cache_path_base", None)
        except Exception:
            plugin_base = None
        if not plugin_base:
            return host_path  # local, singularity, unknown: host path is correct
        container_base = str(plugin_base)

    mapped = map_cache_path_to_container(host_path, container_base=container_base)
    return mapped if mapped is not None else host_path


def iter_cache_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Return individual (host_path, container_path) entries for cache files.

    Used by Modal to upload files individually and resync before each command.
    Skips symlinks.  The container paths use the new ``cache/<subdir>`` layout.
    """
    from hermes_constants import get_hermes_dir

    result: List[Dict[str, str]] = []
    for new_subpath, old_name in _CACHE_DIRS:
        host_dir = get_hermes_dir(new_subpath, old_name)
        if not host_dir.is_dir():
            continue
        container_root = f"{container_base.rstrip('/')}/{new_subpath}"
        for item in host_dir.rglob("*"):
            if item.is_symlink() or not item.is_file():
                continue
            rel = item.relative_to(host_dir)
            result.append({
                "host_path": str(item),
                "container_path": f"{container_root}/{rel}",
            })
    return result


def clear_credential_files() -> None:
    """Reset the skill-scoped registry (e.g. on session reset)."""
    _get_registered().clear()


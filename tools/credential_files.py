"""Credential file passthrough registry for remote terminal backends.

Skills that declare ``required_credential_files`` in their frontmatter need
those files available inside sandboxed execution environments (Modal, Docker).
By default remote backends create bare containers with no host files.

This module provides a session-scoped registry so skill-declared credential
files (and user-configured overrides) are mounted into remote sandboxes.

Two sources feed the registry:

1. **Skill declarations** — when a skill is loaded via ``skill_view``, its
   ``required_credential_files`` entries are registered here if the files
   exist on the host.
2. **User config** — ``terminal.credential_files`` in config.yaml lets users
   explicitly list additional files to mount.

Remote backends (``tools/environments/modal.py``, ``docker.py``) call
:func:`get_credential_file_mounts` at sandbox creation time.

Each registered entry is a dict::

    {
        "host_path": "/home/user/.hermes/google_token.json",
        "container_path": "/root/.hermes/google_token.json",
    }
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Session-scoped list of credential files to mount.
# Key: container_path (deduplicated), Value: host_path
_registered_files: Dict[str, str] = {}

# Cache for config-based file list (loaded once per process).
_config_files: List[Dict[str, str]] | None = None


def _resolve_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))


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
    try:
        resolved = host_path.resolve()
        hermes_home_resolved = hermes_home.resolve()
        resolved.relative_to(hermes_home_resolved)  # raises ValueError if outside
    except ValueError:
        logger.warning(
            "credential_files: rejected path traversal %r "
            "(resolves to %s, outside HERMES_HOME %s)",
            relative_path,
            resolved,
            hermes_home_resolved,
        )
        return False

    if not resolved.is_file():
        logger.debug("credential_files: skipping %s (not found)", resolved)
        return False

    container_path = f"{container_base.rstrip('/')}/{relative_path}"
    _registered_files[container_path] = str(resolved)
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
        hermes_home = _resolve_hermes_home()
        config_path = hermes_home / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            cred_files = cfg.get("terminal", {}).get("credential_files")
            if isinstance(cred_files, list):
                for item in cred_files:
                    if isinstance(item, str) and item.strip():
                        host_path = hermes_home / item.strip()
                        if host_path.is_file():
                            container_path = f"/root/.hermes/{item.strip()}"
                            result.append({
                                "host_path": str(host_path),
                                "container_path": container_path,
                            })
    except Exception as e:
        logger.debug("Could not read terminal.credential_files from config: %s", e)

    _config_files = result
    return _config_files


def get_credential_file_mounts() -> List[Dict[str, str]]:
    """Return all credential files that should be mounted into remote sandboxes.

    Each item has ``host_path`` and ``container_path`` keys.
    Combines skill-registered files and user config.
    """
    mounts: Dict[str, str] = {}

    # Skill-registered files
    for container_path, host_path in _registered_files.items():
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
) -> Dict[str, str] | None:
    """Return mount info for a symlink-safe copy of the skills directory.

    Skills may include ``scripts/``, ``templates/``, and ``references/``
    subdirectories that the agent needs to execute inside remote sandboxes.

    **Security:** Bind mounts follow symlinks, so a malicious symlink inside
    the skills tree could expose arbitrary host files to the container.  When
    symlinks are detected, this function creates a sanitized copy (regular
    files only) in a temp directory and returns that path instead.  When no
    symlinks are present (the common case), the original directory is returned
    directly with zero overhead.

    Returns a dict with ``host_path`` and ``container_path`` keys, or None.
    """
    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if not skills_dir.is_dir():
        return None

    host_path = _safe_skills_path(skills_dir)
    return {
        "host_path": host_path,
        "container_path": f"{container_base.rstrip('/')}/skills",
    }


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

    for item in skills_dir.rglob("*"):
        if item.is_symlink():
            continue
        rel = item.relative_to(skills_dir)
        target = safe_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(target))

    def _cleanup():
        if safe_dir.is_dir():
            shutil.rmtree(safe_dir, ignore_errors=True)

    atexit.register(_cleanup)
    logger.info("credential_files: created symlink-safe skills copy at %s", safe_dir)
    return str(safe_dir)


def iter_skills_files(
    container_base: str = "/root/.hermes",
) -> List[Dict[str, str]]:
    """Yield individual (host_path, container_path) entries for skills files.

    Skips symlinks entirely.  Preferred for backends that upload files
    individually (Daytona, Modal) rather than mounting a directory.
    """
    hermes_home = _resolve_hermes_home()
    skills_dir = hermes_home / "skills"
    if not skills_dir.is_dir():
        return []

    container_root = f"{container_base.rstrip('/')}/skills"
    result: List[Dict[str, str]] = []
    for item in skills_dir.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        rel = item.relative_to(skills_dir)
        result.append({
            "host_path": str(item),
            "container_path": f"{container_root}/{rel}",
        })
    return result


def clear_credential_files() -> None:
    """Reset the skill-scoped registry (e.g. on session reset)."""
    _registered_files.clear()


def reset_config_cache() -> None:
    """Force re-read of config on next access (for testing)."""
    global _config_files
    _config_files = None

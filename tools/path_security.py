"""Shared path validation helpers for tool implementations.

Extracts the ``resolve() + relative_to()`` and ``..`` traversal check
patterns previously duplicated across skill_manager_tool, skills_tool,
skills_hub, cronjob_tools, and credential_files.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def validate_within_dir(path: Path, root: Path) -> Optional[str]:
    """Ensure *path* resolves to a location within *root*.

    Returns an error message string if validation fails, or ``None`` if the
    path is safe.  Uses ``Path.resolve()`` to follow symlinks and normalize
    ``..`` components.

    Usage::

        error = validate_within_dir(user_path, allowed_root)
        if error:
            return json.dumps({"error": error})
    """
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError) as exc:
        return f"Path escapes allowed directory: {exc}"
    return None


def has_traversal_component(path_str: str) -> bool:
    """Return True if *path_str* contains ``..`` traversal components.

    Quick check for obvious traversal attempts before doing full resolution.
    """
    parts = Path(path_str).parts
    return ".." in parts


def get_workspace_root() -> Path | None:
    """Return the configured workspace root, or ``None`` when unset.

    Environment variables take precedence so CLI/gateway bridges can make the
    value available process-wide without every tool re-reading config.yaml.
    """
    env_root = os.environ.get("HERMES_WORKSPACE_ROOT")
    if env_root is not None:
        env_root = env_root.strip()
        if not env_root:
            return None
        try:
            return Path(env_root).expanduser().resolve()
        except (OSError, RuntimeError, ValueError):
            return None

    try:
        from hermes_cli.config import load_config

        security_cfg = (load_config().get("security") or {})
        raw_root = str(security_cfg.get("workspace_root", "") or "").strip()
        if raw_root:
            return Path(raw_root).expanduser().resolve()
    except Exception:
        pass

    return None


def resolve_user_path(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    expand_user: bool = True,
) -> Path:
    """Resolve a user-supplied path against an optional base directory."""
    candidate = Path(path)
    if expand_user:
        candidate = candidate.expanduser()

    if not candidate.is_absolute():
        if base_dir is not None:
            base = Path(base_dir)
            if expand_user:
                base = base.expanduser()
            candidate = base.resolve() / candidate
        else:
            candidate = Path.cwd().resolve() / candidate

    return candidate.resolve()


def validate_workspace_path(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    label: str = "path",
    expand_user: bool = True,
) -> tuple[Path, Path | None, str | None]:
    """Resolve *path* and validate that it stays within the workspace root."""
    resolved = resolve_user_path(path, base_dir=base_dir, expand_user=expand_user)
    root = (
        resolve_user_path(workspace_root, expand_user=True)
        if workspace_root is not None
        else get_workspace_root()
    )
    if root is None:
        return resolved, None, None

    error = validate_within_dir(resolved, root)
    if error:
        return (
            resolved,
            root,
            f"Blocked: {label} resolves outside the configured workspace_root ({root}): {path}",
        )

    return resolved, root, None

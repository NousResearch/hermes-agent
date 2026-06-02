"""Hermes-managed Camofox state helpers.

Provides profile-scoped identity and state directory paths for Camofox
persistent browser profiles.  When managed persistence is enabled, Hermes
sends a deterministic userId derived from the active profile so that
Camofox can map it to the same persistent browser profile directory
across restarts.
"""

from __future__ import annotations

import secrets
import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home

CAMOFOX_STATE_DIR_NAME = "browser_auth"
CAMOFOX_STATE_SUBDIR = "camofox"
CAMOFOX_SECRET_FILE = "identity_secret"


def get_camofox_state_dir() -> Path:
    """Return the profile-scoped root directory for Camofox persistence."""
    return get_hermes_home() / CAMOFOX_STATE_DIR_NAME / CAMOFOX_STATE_SUBDIR


def _load_or_create_identity_secret() -> str:
    """Return an unguessable profile-scoped secret for managed identities."""
    import os
    state_dir = get_camofox_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    secret_path = state_dir / CAMOFOX_SECRET_FILE

    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()

    secret = secrets.token_hex(32)
    try:
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(secret)
    except FileExistsError:
        return secret_path.read_text(encoding="utf-8").strip()
    except OSError:
        # Fallback for filesystems with limited permission/flags support
        secret_path.write_text(secret, encoding="utf-8")
        try:
            secret_path.chmod(0o600)
        except OSError:
            pass
    return secret


def get_camofox_identity(task_id: Optional[str] = None) -> Dict[str, str]:
    """Return the stable Hermes-managed Camofox identity for this profile.

    The user identity is profile-scoped (same Hermes profile = same userId).
    The session key is scoped to the logical browser task so newly created
    tabs within the same profile reuse the same identity contract.
    """
    identity_secret = _load_or_create_identity_secret()
    logical_scope = task_id or "default"
    user_digest = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"camofox-user:{identity_secret}",
    ).hex[:10]
    session_digest = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"camofox-session:{identity_secret}:{logical_scope}",
    ).hex[:16]
    return {
        "user_id": f"hermes_{user_digest}",
        "session_key": f"task_{session_digest}",
    }

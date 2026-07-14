"""Hermes-managed Camofox state helpers.

Provides profile-scoped identity and state directory paths for Camofox
persistent browser profiles.  When managed persistence is enabled, Hermes
sends a deterministic userId derived from the active profile so that
Camofox can map it to the same persistent browser profile directory
across restarts.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home

CAMOFOX_STATE_DIR_NAME = "browser_auth"
CAMOFOX_STATE_SUBDIR = "camofox"


def get_camofox_state_dir() -> Path:
    """Return the profile-scoped root directory for Camofox persistence."""
    return get_hermes_home() / CAMOFOX_STATE_DIR_NAME / CAMOFOX_STATE_SUBDIR


def get_camofox_identity(task_id: Optional[str] = None, *, isolate_task: bool = False) -> Dict[str, str]:
    """Return the stable Hermes-managed Camofox identity for this profile.

    The default user identity is profile-scoped. With ``isolate_task=True`` it
    is task-scoped, giving each conversation its own Camofox browser context
    (and therefore its own visible window) while remaining stable on restart.
    The session key is always scoped to the logical browser task.
    """
    scope_root = str(get_camofox_state_dir())
    logical_scope = task_id or "default"
    user_scope = f"{scope_root}:{logical_scope}" if isolate_task else scope_root
    user_digest = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"camofox-user:{user_scope}",
    ).hex[:10]
    session_digest = uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"camofox-session:{scope_root}:{logical_scope}",
    ).hex[:16]
    return {
        "user_id": f"hermes_{user_digest}",
        "session_key": f"task_{session_digest}",
    }

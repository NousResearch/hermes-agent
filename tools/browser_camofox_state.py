"""Hermes-managed Camofox state helpers.

With managed persistence enabled, Hermes sends a deterministic userId derived from the
active profile so Camofox maps it to the same persistent browser profile across restarts.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, Optional

from hermes_constants import get_hermes_home


def get_camofox_state_dir() -> Path:
    """Return the profile-scoped root directory for Camofox persistence."""
    return get_hermes_home() / "browser_auth" / "camofox"


def get_camofox_identity(task_id: Optional[str] = None) -> Dict[str, str]:
    """Stable Hermes-managed Camofox identity: userId is profile-scoped, session key is
    scoped to the logical browser task so new tabs in the same profile reuse it."""
    scope_root = str(get_camofox_state_dir())
    user_digest = uuid.uuid5(uuid.NAMESPACE_URL, f"camofox-user:{scope_root}").hex[:10]
    session_digest = uuid.uuid5(uuid.NAMESPACE_URL, f"camofox-session:{scope_root}:{task_id or 'default'}").hex[:16]
    return {"user_id": f"hermes_{user_digest}", "session_key": f"task_{session_digest}"}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

CAMOFOX_STATE_DIR_NAME = "browser_auth"

CAMOFOX_STATE_SUBDIR = "camofox"
# ---- END PLUGIN-COMPAT ----

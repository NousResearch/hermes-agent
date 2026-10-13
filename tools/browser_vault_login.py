"""Saved-login metadata checks before automatic or manual browser OTP entry."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.vault_backends.base import LoginBackend
    from agent.vault_store import VaultItemMeta


def prepare_otp_login(handle: str) -> tuple[LoginBackend | None, VaultItemMeta | None, str | None]:
    """Resolve login metadata or return a safe refusal; never retrieve a code."""
    from agent.vault_backends import UnlockRequired, backend_for_handle
    from tools.browser_vault_tool import browser_vault_unlock

    if not handle:
        return None, None, None
    backend = backend_for_handle(handle)
    if backend is None:
        return None, None, json.dumps({"success": False, "error_type": "invalid_login",
                                     "error": "No enabled login backend owns this handle. Use browser_vault action=list."})
    if backend.needs_unlock and not backend.is_unlocked():
        unlocked = json.loads(browser_vault_unlock(backend.name))
        if not unlocked.get("success"):
            return None, None, json.dumps(unlocked)
    try:
        meta = backend.get_meta(handle)
    except UnlockRequired:
        return None, None, json.dumps({"success": False, "error_type": "unlock_required",
                                     "error": "The login backend is locked. Call browser_vault action=unlock."})
    except Exception:  # health: allow BLE001 -- plugin metadata errors may contain credentials; refuse without exposing them
        return None, None, json.dumps({"success": False, "error_type": "metadata_unavailable",
                                     "error": "Could not verify the saved login's websites. No code was requested."})
    if meta is None or meta.kind != "login":
        return None, None, json.dumps({"success": False, "error_type": "invalid_login",
                                     "error": "The handle does not identify a saved login. Use browser_vault action=list."})
    return backend, meta, None

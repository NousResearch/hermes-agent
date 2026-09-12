"""Human handoff actions for ``computer_use`` on a Bot Desktop: ``request_handoff`` asks the person to
take over the screen (login, 2FA, CAPTCHA, payment) and ``wait_for_human`` blocks until control is
back. Both are answered without touching cua-driver, so a human typing a credential is never
captured.

The notification itself (Desktop pane badge, Telegram/Discord message) is emitted by the gateway's
lease listener; this module only records intent on the lease and waits.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from tools.bot_desktop import lease as _lease

HANDOFF_ACTIONS = frozenset({"request_handoff", "wait_for_human"})
_DEFAULT_WAIT_SECONDS = 600.0
_MAX_WAIT_SECONDS = 1800.0


def handle_handoff(action: str, args: Dict[str, Any]) -> str:
    if action == "request_handoff":
        reason = str(args.get("reason") or "The agent needs you to complete a step on its screen.").strip()
        _lease.request_handoff(reason)
        return json.dumps({
            "ok": True, "action": action, "state": _lease.get().as_dict(),
            "next": "The user has been asked to open this bot's Desktop pane and take over. Call "
                    "computer_use action='wait_for_human' to block until they hand control back, then "
                    "re-capture before continuing — the screen state is whatever they left."})
    timeout = min(_MAX_WAIT_SECONDS, max(1.0, float(args.get("seconds") or _DEFAULT_WAIT_SECONDS)))
    released = _lease.wait_for_release(timeout=timeout)
    state = _lease.get().as_dict()
    if released:
        return json.dumps({"ok": True, "action": action, "state": state,
                           "next": "Control is back with you. Take a fresh capture; do not assume prior state."})
    return json.dumps({"ok": False, "action": action, "code": "still_waiting", "state": state,
                       "error": f"No hand-back within {timeout:.0f}s. Call wait_for_human again or ask the user in chat."})

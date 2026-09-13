"""Human handoff actions for ``computer_use`` on a Bot Desktop: ``request_handoff`` asks the person to
take over the screen (login, 2FA, CAPTCHA, payment) and ``wait_for_human`` blocks until control is
back. Both are answered without touching cua-driver, so a human typing a credential is never
captured.

The Desktop pane shows the request (the gateway broadcasts the lease change). Reaching the person
anywhere else is the model's job: it relays the ask in its reply, which is what lands in the chat
surface the user is actually on. This module only records intent on the lease and waits.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from tools.bot_desktop import lease as _lease

HANDOFF_ACTIONS = frozenset({"request_handoff", "wait_for_human"})
_DEFAULT_WAIT_SECONDS = 600.0
_MAX_WAIT_SECONDS = 1800.0
_DEFAULT_GRACE_SECONDS = 60.0


def handle_handoff(action: str, args: Dict[str, Any]) -> str:
    if action == "request_handoff":
        reason = str(args.get("reason") or "The agent needs you to complete a step on its screen.").strip()
        _lease.request_handoff(reason)
        return json.dumps({
            "ok": True, "action": action, "state": _lease.get().as_dict(),
            "next": "Hermes Desktop now shows 'Bot needs you' on this bot's Screen. Tell the user in your "
                    "reply what to do and that they can take over from Bots > Screen, then call "
                    "computer_use action='wait_for_human' to block until they hand control back and "
                    "re-capture before continuing — the screen state is whatever they left."})
    timeout = min(_MAX_WAIT_SECONDS, max(1.0, float(args.get("seconds") or _DEFAULT_WAIT_SECONDS)))
    grace = min(timeout, max(0.0, float(args.get("grace") or _DEFAULT_GRACE_SECONDS)))
    # Nobody has taken over yet: give them `grace` to click Take over, then return so the model can chase
    # the user in chat instead of blocking the whole timeout on a request nobody saw. Once a human holds
    # the screen, wait the full timeout for the hand-back.
    if not _lease.wait_for_takeover_or_release(timeout=grace):
        state = _lease.get().as_dict()
        return json.dumps({"ok": False, "action": action, "code": "no_takeover", "state": state,
                           "error": f"Nobody took over within {grace:.0f}s. Ask the user in chat to open Bots > Screen and "
                                    "click Take over, then call wait_for_human again."})
    released = _lease.wait_for_release(timeout=timeout)
    state = _lease.get().as_dict()
    if released:
        return json.dumps({"ok": True, "action": action, "state": state,
                           "next": "Control is back with you. Take a fresh capture; do not assume prior state."})
    return json.dumps({"ok": False, "action": action, "code": "still_waiting", "state": state,
                       "error": f"No hand-back within {timeout:.0f}s. Call wait_for_human again or ask the user in chat."})
